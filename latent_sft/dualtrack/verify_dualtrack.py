r"""V1 / V2 / V3 acceptance harness for a trained clean dual-track model.

V1 format
    Generation yields a well-formed ``[latents][<cot>...</cot>][answer]``.

V2 latent-causal
    V2a swapping in a donor question's latents changes the answer
        (``change_rate``), and how often it lands on the donor's own answer
        (``follow_donor_rate``).
    V2c ablating the latents (zeros) changes the answer. If the answer survives
        ablation the latents are decorative, whatever V2a says.

V3 bottleneck -- ALWAYS reported ON and OFF; the contrast is the decisive part.
    V3a gradient isolation (decisive, instrument-free): the max absolute
        d(answer logits)/d(cot embeddings) must be exactly 0 with the mask ON and
        > 0 with it OFF. This is a property of the mask, not of training.
    V3b teacher-forced logit divergence between ON and OFF at answer positions.
    V3c generation answer-change rate under three CoT perturbations, ON and OFF.

A trained model may also *learn* to ignore the visible CoT, in which case the OFF
side barely moves and V3b/V3c prove little. That is a measurement limit, not a
failure; V3a is the check that always decides.

The checkpoint this loads from ``<out>/hf`` is written by UPSTREAM's
``Stage2Trainer._save`` (upstream/src/stage2/trainer.py:26-41), so the merged
model under test is produced by upstream code, not by this layer.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dualtrack import MIN_TRANSFORMERS

logger = logging.getLogger(__name__)

PERTURBATIONS: Tuple[str, ...] = ("reversed", "donor_cot", "donor_cot_other_answer")

# MIN_TRANSFORMERS (imported above) is the first transformers release whose attention path
# consumes a 4-D additive mask. At or above it, a rejected 4-D mask is a regression, not an
# environment limit. It is imported, never re-typed: a second spelling of the floor drifts.


def _version_at_least(version: str, floor: str) -> bool:
    """Compare dotted numeric prefixes; a non-numeric suffix (rc/dev) is ignored."""

    def parts(text: str) -> Tuple[int, ...]:
        out: List[int] = []
        for chunk in text.split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            if not digits:
                break
            out.append(int(digits))
        return tuple(out)

    return parts(version) >= parts(floor)

# --- shared cross-platform reporting contract (local copy by design) ----------
# The sibling folders colar/ and lt_tuning/ emit the same six keys under the same
# names and the same polarity, so the three runs tabulate into one row each.
PLATFORM = "latent_sft"
# ``reversed`` is the only perturbation that preserves the CoT token count, so it is
# the only one whose ON side is not confounded by shifted RoPE positions.
CANONICAL_CORRUPTION = "reversed"
CANONICAL_FIELDS: Tuple[str, ...] = (
    "platform",
    "v1_well_formed_rate",
    "v2_answer_change_rate",
    "v2_follow_donor_rate",
    "v3_corruption",
    "v3_answer_change_rate_mask_on",
    "v3_answer_change_rate_mask_off",
)


@dataclass(frozen=True)
class VerifyConfig:
    out: Path
    eval_data: Path
    model_family: str = "llama"
    hf: Optional[Path] = None
    base_model: Optional[Path] = None
    adapter: Optional[Path] = None
    n_samples: int = 20
    max_latent: int = 64
    max_cot: int = 256
    max_ans: int = 64
    teacher_forced_check: bool = False
    report: Optional[Path] = None


def load_model(config: VerifyConfig) -> Any:
    """Prefer the merged ``<out>/hf``; fall back to base_model + lora_adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dualtrack.generate_dualtrack import assert_single_piece_delimiters

    hf_dir = config.hf or config.out / "hf"
    base_dir = config.base_model or config.out / "base_model"
    adapter_dir = config.adapter or config.out / "lora_adapter"
    use_merged = (hf_dir / "config.json").exists()
    source_dir = hf_dir if use_merged else base_dir
    tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    )
    if not use_merged:
        from peft import PeftModel

        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.tokenizer = tokenizer
    model.latent_token_ids = tokenizer(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    model.cot_token_ids = tokenizer(["<cot>", "</cot>"], add_special_tokens=False)["input_ids"]
    # The checkpoint path no longer names the family, so the template is explicit.
    model.latent_model_path = str(source_dir)
    model.model_family = config.model_family
    assert_single_piece_delimiters(model)
    return model


def _decode_config(config: VerifyConfig, bottleneck: bool = True) -> Any:
    from dualtrack.generate_dualtrack import DecodeConfig

    return DecodeConfig(
        max_latent=config.max_latent,
        max_cot=config.max_cot,
        max_ans=config.max_ans,
        bottleneck=bottleneck,
    )


def make_pieces(model: Any, prompt_ids: Sequence[int], result: Any) -> Any:
    """Rebuild the completed sequence so it can be re-run teacher-forced."""
    from dualtrack.generate_dualtrack import SequencePieces

    bot_len = len(model.latent_token_ids[0])
    return SequencePieces(
        prefix_ids=tuple(prompt_ids[:-bot_len]),
        bot_ids=tuple(prompt_ids[-bot_len:]),
        latent_embeds=result.latent_embeds,
        eot_ids=tuple(model.latent_token_ids[1]),
        cot_open_ids=tuple(model.cot_token_ids[0]),
        cot_ids=tuple(result.cot_ids),
        cot_close_ids=tuple(model.cot_token_ids[1]),
        answer_ids=tuple(result.answer_ids),
    )


def check_format(results: Sequence[Any], n_attempted: int, n_degenerate: int) -> Dict[str, Any]:
    """Rates are over every ATTEMPTED prompt; a degenerate decode counts as a failure."""
    from dualtrack.common import extract_last_boxed

    latent_ok = [result.latent_len > 0 for result in results]
    cot_ok = [len(result.cot_ids) > 0 for result in results]
    boxed_ok = [extract_last_boxed(result.answer_text) is not None for result in results]
    well_formed = sum(1 for flags in zip(latent_ok, cot_ok, boxed_ok) if all(flags))
    denominator = max(n_attempted, 1)
    return {
        "n_attempted": n_attempted,
        "n_decoded": len(results),
        "n_degenerate": n_degenerate,
        "latent_nonempty_rate": round(sum(latent_ok) / denominator, 4),
        "cot_nonempty_rate": round(sum(cot_ok) / denominator, 4),
        "answer_boxed_rate": round(sum(boxed_ok) / denominator, 4),
        "well_formed_rate": round(well_formed / denominator, 4),
        "example": results[0].answer_text if results else "",
    }


def generate_all(
    model: Any, prompts: Sequence[List[int]], config: VerifyConfig
) -> Tuple[List[List[int]], List[Any], int]:
    """Decode every prompt; a malformed sample is dropped, counted, never fudged."""
    from dualtrack.generate_dualtrack import dualtrack_generate

    kept: List[List[int]] = []
    results: List[Any] = []
    degenerate = 0
    for prompt in prompts:
        try:
            results.append(dualtrack_generate(model, prompt, _decode_config(config)))
        except ValueError as exc:
            degenerate += 1
            logger.warning("dropping a malformed generation: %s", exc)
            continue
        kept.append(list(prompt))
    return kept, results, degenerate


def _rate(flags: Sequence[bool]) -> float:
    return round(sum(1 for flag in flags if flag) / len(flags), 4) if flags else 0.0


def check_latent_causal(
    model: Any, prompts: Sequence[List[int]], results: Sequence[Any], config: VerifyConfig
) -> Dict[str, Any]:
    import torch

    from dualtrack.common import answers_equal
    from dualtrack.generate_dualtrack import dualtrack_generate

    changed: List[bool] = []
    followed: List[bool] = []
    ablated_changed: List[bool] = []
    for index, prompt in enumerate(prompts):
        donor = results[(index + 1) % len(results)]
        swapped = dualtrack_generate(
            model, prompt, _decode_config(config), override_latent_embeds=donor.latent_embeds
        )
        changed.append(not answers_equal(swapped.answer_text, results[index].answer_text))
        followed.append(answers_equal(swapped.answer_text, donor.answer_text))
        zeros = torch.zeros_like(results[index].latent_embeds)
        ablated = dualtrack_generate(model, prompt, _decode_config(config), override_latent_embeds=zeros)
        ablated_changed.append(not answers_equal(ablated.answer_text, results[index].answer_text))
    return {
        "n": len(prompts),
        "v2a_change_rate": _rate(changed),
        "v2a_follow_donor_rate": _rate(followed),
        "v2c_zero_ablation_change_rate": _rate(ablated_changed),
    }


def perturb_cot(kind: str, own: Sequence[int], donor: Any, donor_other: Any) -> List[int]:
    if kind == "reversed":
        return list(reversed(list(own)))
    if kind == "donor_cot":
        return list(donor.cot_ids)
    if kind == "donor_cot_other_answer":
        return list(donor_other.cot_ids)
    raise ValueError(f"unknown perturbation {kind!r}; expected one of {PERTURBATIONS}")


def gradient_isolation(model: Any, pieces: Any, bottleneck: bool) -> float:
    """max |d(sum of answer-position logits) / d(cot input embeddings)|."""
    from dualtrack import spans as spans_lib
    from dualtrack.generate_dualtrack import embed_ids, teacher_forced_logits

    cot_embeds = embed_ids(model, list(pieces.cot_ids)).detach().clone().float().requires_grad_(True)
    logits = teacher_forced_logits(model, pieces, bottleneck=bottleneck, cot_embeds=cot_embeds)
    q0, q1 = spans_lib.answer_query_span(pieces.spans())
    logits[0, q0:q1].float().sum().backward()
    if cot_embeds.grad is None:
        raise RuntimeError("no gradient reached the CoT embeddings; the graph was detached")
    return float(cot_embeds.grad.abs().max().item())


def teacher_forced_divergence(model: Any, pieces: Any) -> Dict[str, float]:
    import torch

    from dualtrack import spans as spans_lib
    from dualtrack.generate_dualtrack import teacher_forced_logits

    with torch.no_grad():
        on = teacher_forced_logits(model, pieces, bottleneck=True).float()
        off = teacher_forced_logits(model, pieces, bottleneck=False).float()
    q0, q1 = spans_lib.answer_query_span(pieces.spans())
    on_rows, off_rows = on[0, q0:q1], off[0, q0:q1]
    argmax_changed = (on_rows.argmax(-1) != off_rows.argmax(-1)).float().mean().item()
    log_on = torch.log_softmax(on_rows, dim=-1)
    log_off = torch.log_softmax(off_rows, dim=-1)
    kl = (log_on.exp() * (log_on - log_off)).sum(-1).mean().item()
    return {"argmax_change_fraction": round(argmax_changed, 4), "mean_kl_on_vs_off": round(kl, 6)}


def check_bottleneck(
    model: Any, prompts: Sequence[List[int]], results: Sequence[Any], config: VerifyConfig
) -> Dict[str, Any]:
    from dualtrack.common import answers_equal
    from dualtrack.generate_dualtrack import dualtrack_generate

    grads_on: List[float] = []
    grads_off: List[float] = []
    divergences: List[Dict[str, float]] = []
    changes: Dict[str, Dict[str, List[bool]]] = {kind: {"on": [], "off": []} for kind in PERTURBATIONS}
    for index, prompt in enumerate(prompts):
        own = results[index]
        pieces = make_pieces(model, prompt, own)
        grads_on.append(gradient_isolation(model, pieces, bottleneck=True))
        grads_off.append(gradient_isolation(model, pieces, bottleneck=False))
        divergences.append(teacher_forced_divergence(model, pieces))
        donor = results[(index + 1) % len(results)]
        donor_other = _first_other_answer(results, index)
        for kind in PERTURBATIONS:
            forced = perturb_cot(kind, own.cot_ids, donor, donor_other)
            for label, bottleneck in (("on", True), ("off", False)):
                perturbed = dualtrack_generate(
                    model,
                    prompt,
                    _decode_config(config, bottleneck=bottleneck),
                    override_latent_embeds=own.latent_embeds,
                    override_cot_ids=forced,
                )
                changes[kind][label].append(not answers_equal(perturbed.answer_text, own.answer_text))
    return _summarize_bottleneck(grads_on, grads_off, divergences, changes)


def _first_other_answer(results: Sequence[Any], index: int) -> Any:
    from dualtrack.common import answers_equal

    for offset in range(1, len(results)):
        candidate = results[(index + offset) % len(results)]
        if not answers_equal(candidate.answer_text, results[index].answer_text):
            return candidate
    return results[(index + 1) % len(results)]


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _summarize_bottleneck(
    grads_on: Sequence[float],
    grads_off: Sequence[float],
    divergences: Sequence[Dict[str, float]],
    changes: Dict[str, Dict[str, List[bool]]],
) -> Dict[str, Any]:
    return {
        "v3a_gradient_isolation": {
            "max_grad_mask_on": max(grads_on) if grads_on else 0.0,
            "max_grad_mask_off": max(grads_off) if grads_off else 0.0,
            "passes": bool(grads_on and max(grads_on) == 0.0 and min(grads_off) > 0.0),
        },
        "v3b_teacher_forced": {
            "mean_argmax_change_fraction": _mean([d["argmax_change_fraction"] for d in divergences]),
            "mean_kl_on_vs_off": _mean([d["mean_kl_on_vs_off"] for d in divergences]),
        },
        "v3c_generation": {
            kind: {
                "answer_change_rate_mask_on": _rate(sides["on"]),
                "answer_change_rate_mask_off": _rate(sides["off"]),
            }
            for kind, sides in changes.items()
        },
    }


def cross_check_teacher_forced(
    model: Any, prompts: Sequence[List[int]], results: Sequence[Any]
) -> Dict[str, Any]:
    """Incremental decode vs the real 4-D training mask; disagreement voids the run."""
    import torch

    from dualtrack import spans as spans_lib
    from dualtrack.generate_dualtrack import teacher_forced_logits

    matched: List[bool] = []
    for index, prompt in enumerate(prompts):
        pieces = make_pieces(model, prompt, results[index])
        with torch.no_grad():
            logits = teacher_forced_logits(model, pieces, bottleneck=True)
        q0, _ = spans_lib.answer_query_span(pieces.spans())
        predicted = logits[0, q0 : q0 + len(pieces.answer_ids)].argmax(-1).tolist()
        matched.append(predicted == list(pieces.answer_ids))
    return {"n": len(matched), "agreement_rate": _rate(matched)}


def canonical_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """The cross-platform row: identical key names and polarity in all three folders.

    The V3 pair is read off ``v3c_generation[CANONICAL_CORRUPTION]``, i.e. the same
    quantity colar/ reports as ``1 - on_unchanged[shuffled_own]`` and lt_tuning/ as
    ``V3.1_frozen__mask_{on,off}.answer_change_rate``. V3a (gradient isolation) stays
    the decisive check on this platform; it has no counterpart on the other two and
    is therefore not part of the shared row.
    """
    nan = float("nan")
    v3c = report["v3_bottleneck"]["v3c_generation"].get(CANONICAL_CORRUPTION, {})
    return {
        "platform": PLATFORM,
        "v1_well_formed_rate": report["v1_format"].get("well_formed_rate", nan),
        "v2_answer_change_rate": report["v2_latent_causal"].get("v2a_change_rate", nan),
        "v2_follow_donor_rate": report["v2_latent_causal"].get("v2a_follow_donor_rate", nan),
        "v3_corruption": CANONICAL_CORRUPTION,
        "v3_answer_change_rate_mask_on": v3c.get("answer_change_rate_mask_on", nan),
        "v3_answer_change_rate_mask_off": v3c.get("answer_change_rate_mask_off", nan),
    }


def run(config: VerifyConfig) -> Dict[str, Any]:
    from dualtrack.common import read_jsonl
    from dualtrack.generate_dualtrack import build_prompt_ids

    rows = read_jsonl(config.eval_data)[: config.n_samples]
    if len(rows) < 2:
        raise ValueError("V2/V3 need at least two rows to build donors")
    model = load_model(config)
    all_prompts = [build_prompt_ids(row["question"], model) for row in rows]
    prompts, results, degenerate = generate_all(model, all_prompts, config)
    if len(results) < 2:
        raise ValueError(
            f"only {len(results)} of {len(rows)} prompts decoded into a well-formed dual-track "
            "sequence; V2/V3 need at least two donors"
        )

    report: Dict[str, Any] = {
        "n_samples": len(rows),
        "v1_format": check_format(results, len(rows), degenerate),
        "v2_latent_causal": check_latent_causal(model, prompts, results, config),
        "v3_bottleneck": check_bottleneck(model, prompts, results, config),
    }
    if config.teacher_forced_check:
        report["cross_check"] = cross_check_teacher_forced(model, prompts, results)
    report["canonical"] = canonical_summary(report)
    _print_report(report)
    if config.report:
        config.report.parent.mkdir(parents=True, exist_ok=True)
        config.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))
    v3a = report["v3_bottleneck"]["v3a_gradient_isolation"]
    print("=" * 72)
    print(f"V1 well-formed rate        : {report['v1_format']['well_formed_rate']}")
    print(f"V2 latent swap change rate : {report['v2_latent_causal']['v2a_change_rate']}")
    print(f"V3a gradient ON / OFF      : {v3a['max_grad_mask_on']} / {v3a['max_grad_mask_off']}")
    print(f"V3a decisive verdict       : {'PASS' if v3a['passes'] else 'FAIL'}")
    print("V3c is only informative when the OFF side moves; read both columns.")
    if "canonical" in report:
        print("[canonical] " + json.dumps(report["canonical"], sort_keys=True))


class _ToyAttentionStack:
    """Minimal SDPA transformer used to prove the mask geometry without weights."""

    def __init__(self, hidden: int = 16, heads: int = 2, layers: int = 2, vocab: int = 32) -> None:
        import torch
        from torch import nn

        torch.manual_seed(0)
        self.heads = heads
        self.hidden = hidden
        self.blocks = [
            {name: nn.Linear(hidden, hidden) for name in ("q", "k", "v", "o")} for _ in range(layers)
        ]
        self.head = nn.Linear(hidden, vocab)

    def __call__(self, x: Any, additive_mask: Any) -> Any:
        import torch.nn.functional as F

        batch, seq, _ = x.shape
        head_dim = self.hidden // self.heads
        for block in self.blocks:
            shape = (batch, seq, self.heads, head_dim)
            query = block["q"](x).view(*shape).transpose(1, 2)
            key = block["k"](x).view(*shape).transpose(1, 2)
            value = block["v"](x).view(*shape).transpose(1, 2)
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=additive_mask)
            attended = attended.transpose(1, 2).reshape(batch, seq, self.hidden)
            x = x + block["o"](attended)
        return self.head(x)


def _toy_gradient_isolation(bottleneck: bool) -> float:
    import torch

    from dualtrack import mask as mask_lib
    from dualtrack import spans as spans_lib

    item = spans_lib.build_spans(4, 1, 3, 2, 1, 5, 1, 3)
    total = item.total_len
    stack = _ToyAttentionStack()
    prefix = torch.randn(1, item.cot_content_start, stack.hidden)
    cot = torch.randn(1, item.cot_content_len, stack.hidden, requires_grad=True)
    tail = torch.randn(1, total - item.ccot_pos, stack.hidden)
    embeds = torch.cat([prefix, cot, tail], dim=1)

    key_spans, query_spans = mask_lib.spans_to_mask_inputs([item], bottleneck=bottleneck)
    keep = mask_lib.build_bottleneck_mask([total], key_spans, query_spans, total)
    logits = stack(embeds, mask_lib.to_additive(keep, torch.float32))
    q0, q1 = spans_lib.answer_query_span(item)
    logits[0, q0:q1].sum().backward()
    return float(cot.grad.abs().max().item())


def _selftest_gradient_isolation() -> None:
    grad_on = _toy_gradient_isolation(bottleneck=True)
    grad_off = _toy_gradient_isolation(bottleneck=False)
    assert grad_on == 0.0, f"V3a would not be decisive: gradient leaked with the mask ON ({grad_on})"
    assert grad_off > 0.0, f"the OFF side must move, got {grad_off}"
    print(f"[verify_dualtrack] V3a toy stack: grad ON={grad_on} OFF={grad_off:.6g}")


def _selftest_real_llama_or_report() -> None:
    """The same V3a geometry on a real LlamaForCausalLM; failures are REPORTED verbatim."""
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        print(f"[verify_dualtrack] SKIPPED real-model V3a: transformers unimportable ({exc})")
        return
    from dualtrack import mask as mask_lib
    from dualtrack import spans as spans_lib

    item = spans_lib.build_spans(4, 1, 3, 2, 1, 5, 1, 3)
    total = item.total_len
    key_spans, query_spans = mask_lib.spans_to_mask_inputs([item])
    keep = mask_lib.build_bottleneck_mask([total], key_spans, query_spans, total)
    embeds = torch.randn(1, total, 16)
    try:
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        )
        model = LlamaForCausalLM(config).eval()
        model(inputs_embeds=embeds, attention_mask=mask_lib.to_additive(keep, torch.float32))
    except Exception as exc:  # noqa: BLE001 - reported below, and fatal when it should be
        import transformers

        version = transformers.__version__
        if _version_at_least(version, MIN_TRANSFORMERS):
            # On a stack new enough to support 4-D masks this is a real regression,
            # not an environment limit; reporting it as a print would let the layer's
            # single most important integration check fail while the suite exits 0.
            raise AssertionError(
                f"real-model V3a FAILED on transformers {version} (>= {MIN_TRANSFORMERS}, which "
                f"supports 4-D masks): {type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"[verify_dualtrack] FAILED real-model V3a: transformers {version} "
            f"rejects a 4-D attention mask ({type(exc).__name__}: {str(exc)[:120]}). "
            f"The 4-D path needs transformers>={MIN_TRANSFORMERS}; the toy-stack V3a above "
            "still proves the geometry."
        )
        return
    print("[verify_dualtrack] real-model V3a: transformers accepted the 4-D mask")


def _selftest_helpers() -> None:
    from dualtrack.common import answers_equal

    assert perturb_cot("reversed", [1, 2, 3], None, None) == [3, 2, 1]
    assert perturb_cot("reversed", [], None, None) == []
    try:
        perturb_cot("nope", [1], None, None)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown perturbation was accepted")
    assert answers_equal("\\boxed{50000}", "The answer is \\boxed{50,000}.")
    assert not answers_equal("\\boxed{1}", "\\boxed{2}")
    assert _rate([True, False, True, True]) == 0.75 and _rate([]) == 0.0


def _selftest_canonical() -> None:
    report = {
        "v1_format": {"well_formed_rate": 0.9},
        "v2_latent_causal": {"v2a_change_rate": 0.8, "v2a_follow_donor_rate": 0.3},
        "v3_bottleneck": {
            "v3c_generation": {
                "reversed": {"answer_change_rate_mask_on": 0.0, "answer_change_rate_mask_off": 0.6},
                "donor_cot": {"answer_change_rate_mask_on": 0.2, "answer_change_rate_mask_off": 0.7},
            }
        },
    }
    canonical = canonical_summary(report)
    assert tuple(sorted(canonical)) == tuple(sorted(CANONICAL_FIELDS)), sorted(canonical)
    assert canonical["platform"] == PLATFORM
    assert canonical["v3_corruption"] == "reversed", "the shared row must use the length-preserving arm"
    assert canonical["v3_answer_change_rate_mask_on"] == 0.0, canonical
    assert canonical["v3_answer_change_rate_mask_off"] == 0.6, canonical
    assert canonical["v2_answer_change_rate"] == 0.8, canonical
    missing = canonical_summary(
        {"v1_format": {}, "v2_latent_causal": {}, "v3_bottleneck": {"v3c_generation": {}}}
    )
    assert tuple(sorted(missing)) == tuple(sorted(CANONICAL_FIELDS)), "the key set must never shrink"
    print("[verify_dualtrack] canonical block: " + json.dumps(canonical, sort_keys=True))


def selftest() -> None:
    _selftest_helpers()
    _selftest_canonical()
    _selftest_gradient_isolation()
    _selftest_real_llama_or_report()
    print(
        "[verify_dualtrack] OK -- comparator, perturbations, canonical block, and V3a gradient "
        "isolation (exactly zero with the mask ON, non-zero with it OFF)."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    from dualtrack.common import env_path

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(env_path("DUALTRACK_OUT", "out/clean_dualtrack")))
    parser.add_argument("--eval_data", default=str(env_path("DUALTRACK_DATA", "data/dualtrack_clean.jsonl")))
    parser.add_argument("--model_family", default="llama", help="llama|qwen|deepseek prompt template")
    parser.add_argument("--hf", default=None)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_latent", type=int, default=64)
    parser.add_argument("--max_cot", type=int, default=256)
    parser.add_argument("--max_ans", type=int, default=64)
    parser.add_argument("--teacher_forced_check", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.selftest:
        selftest()
        return
    run(
        VerifyConfig(
            out=Path(args.out),
            eval_data=Path(args.eval_data),
            model_family=args.model_family,
            hf=Path(args.hf) if args.hf else None,
            base_model=Path(args.base_model) if args.base_model else None,
            adapter=Path(args.adapter) if args.adapter else None,
            n_samples=args.n_samples,
            max_latent=args.max_latent,
            max_cot=args.max_cot,
            max_ans=args.max_ans,
            teacher_forced_check=args.teacher_forced_check,
            report=Path(args.report) if args.report else None,
        )
    )


if __name__ == "__main__":
    main()
