"""V1 / V2 / V3 acceptance checks, always reported as a mask ON vs mask OFF contrast.

V1   format         -- generation yields [prompt][interleaved CoT+latents][delimiter][answer]
V2   latent-causal  -- swapping in another question's TERMINAL latent changes the answer
V3.0 structural     -- every path from a visible-CoT key to an answer row crosses a latent
V3.1 latent-pinned  -- with each latent's INPUT EMBEDDING pinned to its clean value, the
                       fusion-initialisation channel is removed and what remains is exactly
                       the attention channel.  NATIVE still leaks; STRICT is bit-exactly 0.
V3.2 free-running   -- the same scramble with nothing pinned.  This is the RESIDUAL LEAK and
                       it is non-zero in BOTH modes.  Reported, never thresholded: on this
                       platform the bottleneck is a chokepoint, not independence.
V3.3 decomposition  -- V3.1 and V3.2 under BOTH bottleneck modes in one invocation, so the
                       attention-mediated share sits next to the share no attention mask can
                       reach (a latent's input embedding is built from the preceding chain
                       token's hidden state -- a residual-stream edge).

V3.1 changed meaning this round.  It used to freeze the latent K/V columns, which required
owning the segment loop and was `0.0` in BOTH modes by construction, carrying no information
about which channel the leak used.  Pinning the latent *inputs* needs only the
`latent_override` hook on `_apply_transform`, and it separates the two channels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .alignment import sha256_file
from .config import (
    GenConfig,
    RunConfig,
    add_bottleneck_mode_argument,
    default_config,
    dualtrack_from_raw,
    load_yaml,
)
from .generate import generate_dualtrack
from .manifest import read_manifest
from .mask import (
    DEFAULT_BOTTLENECK_MODE,
    BottleneckMode,
    ModeLike,
    chokepoint_holds,
    resolve_bottleneck_mode,
    spans_from_track,
)

PLATFORM = "lt_tuning"
# `scramble_cot` reverses the CoT in place: same length, same multiset, so the answer rows'
# position ids are untouched and the ON side is not RoPE-confounded.
CANONICAL_CORRUPTION = "reversed"
# The cross-platform reporting contract.  colar/ and latent_sft/ emit these six keys under
# the same names and the same polarity, so the three runs tabulate into one row each.
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
class SampleTensors:
    idx: int
    input_ids: torch.Tensor
    track_ids: torch.Tensor
    a_start: int
    v_start: int
    cot_positions: Tuple[int, ...]
    latent_positions: Tuple[int, ...]


def to_tensors(
    input_ids: Sequence[int], track_ids: Sequence[int], idx: int, device: torch.device
) -> SampleTensors:
    spans = spans_from_track(track_ids)
    return SampleTensors(
        idx=idx,
        input_ids=torch.tensor([list(input_ids)], device=device),
        track_ids=torch.tensor([list(track_ids)], device=device),
        a_start=spans.a_start,
        v_start=spans.v_start,
        cot_positions=spans.cot_positions,
        latent_positions=spans.latent_positions,
    )


def scramble_cot(sample: SampleTensors) -> torch.Tensor:
    """Reverse the visible-CoT tokens in place-order.  Same length, same multiset, different
    reasoning -- so a model that reads the CoT surface must react."""
    positions = list(sample.cot_positions)
    if len(positions) < 2:
        raise ValueError(f"sample {sample.idx} has fewer than two visible CoT tokens to scramble")
    scrambled = sample.input_ids.clone()
    scrambled[0, positions] = sample.input_ids[0, list(reversed(positions))]
    return scrambled


def answer_argmax(logits: torch.Tensor, sample: SampleTensors) -> List[int]:
    """Teacher-forced greedy answer: the argmax at each row that emits an answer token."""
    rows = range(sample.v_start - 1, sample.input_ids.shape[1] - 1)
    return [int(logits[0, row].argmax().item()) for row in rows]


def _run(
    model: Any,
    ids: torch.Tensor,
    sample: SampleTensors,
    mask_on: bool,
    mode: ModeLike,
    latent_override: Optional[Dict[Any, torch.Tensor]] = None,
) -> Any:
    with torch.no_grad():
        return model(
            ids,
            sample.track_ids,
            attention_mask=torch.ones_like(ids),
            mask_on=mask_on,
            mode=mode,
            latent_override=latent_override,
        )


def v3_0_structural(
    samples: Sequence[SampleTensors], mode: ModeLike = DEFAULT_BOTTLENECK_MODE
) -> Dict[str, Any]:
    """Symbolic reachability: no model, no weights, hard pass/fail."""
    track_rows = [s.track_ids[0].tolist() for s in samples]
    on = all(chokepoint_holds(row, mask_on=True, mode=mode) for row in track_rows)
    off = any(chokepoint_holds(row, mask_on=False, mode=mode) for row in track_rows)
    return {
        "bottleneck_mode": resolve_bottleneck_mode(mode).value,
        "chokepoint_holds_mask_on": bool(on),
        "chokepoint_holds_mask_off": bool(off),
        "n_samples": len(track_rows),
        "pass": bool(on and not off),
    }


def v3_pinned_or_free(
    model: Any,
    samples: Sequence[SampleTensors],
    mask_on: bool,
    pin_latents: bool,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
) -> Dict[str, Any]:
    """V3.1 (`pin_latents=True`) and V3.2 (`False`) share everything but the conditioning."""
    deltas: List[float] = []
    changed: List[bool] = []
    for sample in samples:
        pinned = (
            _run(model, sample.input_ids, sample, mask_on, mode).latents if pin_latents else None
        )
        clean = _run(model, sample.input_ids, sample, mask_on, mode, pinned)
        dirty = _run(model, scramble_cot(sample), sample, mask_on, mode, pinned)
        deltas.append(
            float(
                (clean.logits[0, sample.a_start :] - dirty.logits[0, sample.a_start :]).abs().max()
            )
        )
        changed.append(answer_argmax(clean.logits, sample) != answer_argmax(dirty.logits, sample))
    return {
        "mask_on": mask_on,
        "bottleneck_mode": resolve_bottleneck_mode(mode).value,
        "latents_pinned": pin_latents,
        "max_abs_delta_answer_logits": max(deltas) if deltas else 0.0,
        "mean_abs_delta_answer_logits": sum(deltas) / len(deltas) if deltas else 0.0,
        "answer_change_rate": sum(changed) / len(changed) if changed else 0.0,
        "n_samples": len(samples),
    }


LEAK_NOTE = (
    "CoT corruption under both bottleneck modes, at two conditioning tiers.  NATIVE blocks "
    "only the direct edge visible-CoT -> delimiter/answer, so from two layers onward the "
    "content still reaches the answer through COT -> LATENT -> ANSWER.  STRICT additionally "
    "blocks latent query rows and closes that path.  What survives under STRICT free-running "
    "is the fusion-initialisation channel -- a latent's input embedding is built from the "
    "hidden state of the immediately preceding chain token, which is a residual-stream edge "
    "no attention mask can reach.  Pinning the latent inputs (V3.1) removes that channel, "
    "which is why STRICT is bit-exactly zero there and NATIVE is not.  So a V3 that passes "
    "in EITHER mode still does not establish that the answer is independent of the visible "
    "reasoning on this platform.  The two numbers are a contrast, not an additive split."
)


def _leak_contrast(native: Dict[str, Any], strict: Dict[str, Any]) -> Dict[str, Any]:
    """Two measurements side by side.  Deliberately NOT presented as an additive split.

    STRICT closes a strict subset of NATIVE's channels, so its number is a lower bound on
    what the fusion-initialisation channel alone delivers.  The difference is what closing
    the attention channel bought -- and it can be zero or negative, because the two channels
    are not independent: the corrupted content is already inside the latent's input
    embedding, so removing the latent's read of the CoT need not shrink the answer's
    response to it at all.
    """
    open_ = float(native["mean_abs_delta_answer_logits"])
    closed = float(strict["mean_abs_delta_answer_logits"])
    difference = open_ - closed
    return {
        "leak_mean_abs_delta_native": open_,
        "leak_mean_abs_delta_strict": closed,
        "fusion_channel_lower_bound": closed,
        "attention_channel_delta": difference,
        "attention_channel_relative_change": difference / open_ if open_ > 0.0 else float("nan"),
    }


def v3_leak_decomposition(
    model: Any, samples: Sequence[SampleTensors], mask_on: bool = True
) -> Dict[str, Any]:
    """V3.3: the same corruption under NATIVE and STRICT, at BOTH tiers, in one invocation.

    Both tiers are needed to read either one.  Pinned isolates the attention channel;
    free-running shows what is left when the fusion channel is also live.
    """
    free = {
        mode.value: v3_pinned_or_free(model, samples, mask_on, False, mode)
        for mode in BottleneckMode
    }
    pinned = {
        mode.value: v3_pinned_or_free(model, samples, mask_on, True, mode)
        for mode in BottleneckMode
    }
    native, strict = free[BottleneckMode.NATIVE.value], free[BottleneckMode.STRICT.value]
    report: Dict[str, Any] = {
        "mask_on": mask_on,
        "v3_answer_unchanged_native": bool(native["answer_change_rate"] == 0.0),
        "v3_answer_unchanged_strict": bool(strict["answer_change_rate"] == 0.0),
        "v3_answer_change_rate_native": native["answer_change_rate"],
        "v3_answer_change_rate_strict": strict["answer_change_rate"],
        "v3_max_abs_delta_native": native["max_abs_delta_answer_logits"],
        "v3_max_abs_delta_strict": strict["max_abs_delta_answer_logits"],
        "pinned_max_abs_delta_native": pinned[BottleneckMode.NATIVE.value][
            "max_abs_delta_answer_logits"
        ],
        "pinned_max_abs_delta_strict": pinned[BottleneckMode.STRICT.value][
            "max_abs_delta_answer_logits"
        ],
        "per_mode_free_running": free,
        "per_mode_latent_pinned": pinned,
        "note": LEAK_NOTE,
    }
    report.update(_leak_contrast(native, strict))
    return report


def v2_latent_causal(
    model: Any,
    samples: Sequence[SampleTensors],
    mask_on: bool,
    swap: str = "terminal",
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
) -> Dict[str, Any]:
    """Swap latents in from the next question and see whether the answer moves.

    `terminal` swaps only the latent adjacent to the delimiter: its sole downstream consumers
    are the blocked delimiter/answer rows, so the effect is not confounded by perturbing
    every later reasoning row.
    """
    changed: List[bool] = []
    deltas: List[float] = []
    for position, sample in enumerate(samples):
        donor = samples[(position + 1) % len(samples)]
        if donor.idx == sample.idx or not sample.latent_positions or not donor.latent_positions:
            continue
        clean = _run(model, sample.input_ids, sample, mask_on, mode)
        donor_latents = _run(model, donor.input_ids, donor, mask_on, mode).latents
        override = _build_override(sample, donor, donor_latents, swap)
        swapped = _run(model, sample.input_ids, sample, mask_on, mode, override)
        changed.append(answer_argmax(clean.logits, sample) != answer_argmax(swapped.logits, sample))
        deltas.append(
            float(
                (clean.logits[0, sample.v_start - 1 :] - swapped.logits[0, sample.v_start - 1 :])
                .abs()
                .max()
            )
        )
    return {
        "mask_on": mask_on,
        "bottleneck_mode": resolve_bottleneck_mode(mode).value,
        "swap": swap,
        "answer_change_rate": sum(changed) / len(changed) if changed else 0.0,
        "mean_abs_delta_answer_logits": sum(deltas) / len(deltas) if deltas else 0.0,
        "n_samples": len(changed),
    }


def _build_override(
    sample: SampleTensors,
    donor: SampleTensors,
    donor_latents: Dict[Tuple[int, int], torch.Tensor],
    swap: str,
) -> Dict[Tuple[int, int], torch.Tensor]:
    donor_by_position = {p: v for (b, p), v in donor_latents.items() if b == 0}
    if swap == "terminal":
        targets, sources = [sample.latent_positions[-1]], [donor.latent_positions[-1]]
    elif swap == "all":
        n = min(len(sample.latent_positions), len(donor.latent_positions))
        targets = list(sample.latent_positions[:n])
        sources = list(donor.latent_positions[:n])
    else:
        raise ValueError(f"unknown V2 swap: {swap!r}")
    return {
        (0, target): donor_by_position[source]
        for target, source in zip(targets, sources)
        if source in donor_by_position
    }


def v1_format(
    model: Any,
    tokenizer: Any,
    questions: Sequence[str],
    cfg: GenConfig,
    mask_on: bool,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
) -> Dict[str, Any]:
    well_formed = 0
    with_latents = 0
    suppressed = 0
    stops = {"delimiter": 0, "eos": 0, "cap": 0}
    for question in questions:
        result = generate_dualtrack(model, tokenizer, question, cfg, mask_on=mask_on, mode=mode)
        well_formed += result.well_formed
        with_latents += result.n_latents > 0
        stops[result.stop_reason] += 1
        suppressed += result.suppressed_thinking_steps > 0
    total = max(len(questions), 1)
    return {
        "mask_on": mask_on,
        "bottleneck_mode": resolve_bottleneck_mode(mode).value,
        "n_samples": len(questions),
        "frac_well_formed": well_formed / total,
        "frac_with_spontaneous_latent": with_latents / total,
        "frac_delimiter_forced_by_cap": stops["cap"] / total,
        "frac_stopped_on_eos_before_delimiter": stops["eos"] / total,
        "frac_stopped_on_delimiter": stops["delimiter"] / total,
        "frac_with_suppressed_thinking_in_answer": suppressed / total,
    }


def verify_all(
    model: Any,
    tokenizer: Any,
    samples: Sequence[SampleTensors],
    questions: Sequence[str],
    gen_cfg: GenConfig,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
) -> Dict[str, Any]:
    """`mode` is the run's own bottleneck mode and drives V1/V2/V3.0-V3.2.  V3.3 always runs
    BOTH modes regardless, because that contrast is the result."""
    mode = resolve_bottleneck_mode(mode)
    results: Dict[str, Any] = {
        "bottleneck_mode": mode.value,
        "idx": [s.idx for s in samples],
        "V3.0_structural": v3_0_structural(samples, mode),
        "V3.3_leak_decomposition": v3_leak_decomposition(model, samples, mask_on=True),
    }
    for mask_on in (True, False):
        tag = "mask_on" if mask_on else "mask_off"
        run_mode = mode if mask_on else DEFAULT_BOTTLENECK_MODE
        results[f"V1_format__{tag}"] = v1_format(
            model, tokenizer, questions, gen_cfg, mask_on, run_mode
        )
        results[f"V2_terminal_swap__{tag}"] = v2_latent_causal(
            model, samples, mask_on, "terminal", run_mode
        )
        results[f"V2_all_latent_swap__{tag}"] = v2_latent_causal(
            model, samples, mask_on, "all", run_mode
        )
        results[f"V3.1_pinned__{tag}"] = v3_pinned_or_free(model, samples, mask_on, True, run_mode)
        results[f"V3.2_free__{tag}"] = v3_pinned_or_free(model, samples, mask_on, False, run_mode)
    results["verdict"] = _verdict(results)
    results["canonical"] = canonical_summary(results)
    return results


def canonical_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """The cross-platform row: identical key names and polarity in all three folders.

    The V3 pair comes from V3.1 (latents pinned), this platform's decisive tier and the
    closest counterpart to the other two folders' generation-time CoT corruption.  V3.2
    (free-running, the residual leak) is deliberately NOT in the shared row: it has no
    counterpart elsewhere and is non-zero here by construction.  `v2_follow_donor_rate` is
    None because a donor-answer match is not measured here -- the swap is a single terminal
    latent, not the whole chain.

    The bottleneck mode is deliberately NOT a key here: these six names are a contract shared
    byte-for-byte with colar/ and latent_sft/, which have no such mode.  It is carried at the
    top level of the results, in the verdict, and printed next to this block.
    """
    nan = float("nan")
    v1 = results.get("V1_format__mask_on", {})
    v2 = results.get("V2_terminal_swap__mask_on", {})
    on = results.get("V3.1_pinned__mask_on", {})
    off = results.get("V3.1_pinned__mask_off", {})
    return {
        "platform": PLATFORM,
        "v1_well_formed_rate": v1.get("frac_well_formed", nan),
        "v2_answer_change_rate": v2.get("answer_change_rate", nan),
        "v2_follow_donor_rate": None,
        "v3_corruption": CANONICAL_CORRUPTION,
        "v3_answer_change_rate_mask_on": on.get("answer_change_rate", nan),
        "v3_answer_change_rate_mask_off": off.get("answer_change_rate", nan),
    }


def _verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """Only V3.0 and V3.1 are pass/fail.  V3.2 and V3.3 are measurements, not thresholds."""
    on = results["V3.1_pinned__mask_on"]
    off = results["V3.1_pinned__mask_off"]
    leak = results["V3.3_leak_decomposition"]
    return {
        "bottleneck_mode": results["bottleneck_mode"],
        "V3.0_pass": results["V3.0_structural"]["pass"],
        "V3.1_pass": bool(
            on["max_abs_delta_answer_logits"] <= 1e-2
            and on["answer_change_rate"] == 0.0
            and off["answer_change_rate"] >= 0.30
        ),
        "V3.1_pinned_max_abs_delta_native": leak["pinned_max_abs_delta_native"],
        "V3.1_pinned_max_abs_delta_strict": leak["pinned_max_abs_delta_strict"],
        "V3.2_residual_leak_mask_on": results["V3.2_free__mask_on"]["mean_abs_delta_answer_logits"],
        "V3.2_residual_answer_change_rate_mask_on": results["V3.2_free__mask_on"][
            "answer_change_rate"
        ],
        "v3_answer_unchanged_native": leak["v3_answer_unchanged_native"],
        "v3_answer_unchanged_strict": leak["v3_answer_unchanged_strict"],
        "V3.3_leak_mean_abs_delta_native": leak["leak_mean_abs_delta_native"],
        "V3.3_leak_mean_abs_delta_strict": leak["leak_mean_abs_delta_strict"],
        "V3.3_attention_channel_delta": leak["attention_channel_delta"],
        "V3.3_fusion_channel_lower_bound": leak["fusion_channel_lower_bound"],
        "note": (
            "V3.1 passing means the answer cannot read the visible-CoT surface directly. It "
            "does NOT mean the answer is independent of the visible CoT: each latent is built "
            "from a chain token's hidden state, so the content is already inside the latent "
            "columns. V3.2 is the size of that residual path, and V3.3 splits the measurement "
            "into the attention-mediated part (which STRICT closes) and the "
            "fusion-initialisation part (which no attention mask closes)."
        ),
    }


def load_checkpoint(ckpt: Path, cfg: RunConfig) -> Tuple[Any, Any, int, Dict[str, Any]]:
    """A checkpoint written by `train.py`, re-promoted to the dual-track subclass."""
    from transformers import AutoTokenizer

    from .attention_backend import load_causal_lm
    from .model import DualTrackLTModel

    manifest = read_manifest(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    base = load_causal_lm(str(ckpt), torch_dtype=torch.bfloat16)
    thinking_id = tokenizer.convert_tokens_to_ids(cfg.data.thinking_token)
    if int(thinking_id) != int(manifest["thinking_token_id"]):
        raise ValueError("thinking token id changed since training; the checkpoint is mispaired")
    upstream_model = _build_upstream_model(base, thinking_id, tokenizer, manifest)
    model = DualTrackLTModel.promote(upstream_model, cfg.mask, cfg.loss)
    return model.eval(), tokenizer, int(thinking_id), manifest


def _build_upstream_model(
    base: Any, thinking_id: int, tokenizer: Any, manifest: Dict[str, Any]
) -> Any:
    from ._upstream import import_core

    lt_model_cls = import_core().model.LT_Tuning_Model
    return lt_model_cls(
        base_causallm=base,
        thinking_token_id=thinking_id,
        eos_token_id=tokenizer.eos_token_id,
        stage_mode=manifest["stage_mode"],
    )


# --- selftests -----------------------------------------------------------------------


def _selftest_samples(n: int = 4) -> Tuple[Any, Any, List[SampleTensors]]:
    """Tiny locally-built Llama + stub tokenizer + tracks derived by `tracks.py`."""
    from .generate import build_selftest_fixture
    from .tracks import derive_tracks

    model, tokenizer, thinking_id = build_selftest_fixture()
    delim = [tokenizer.convert_tokens_to_ids(t) for t in ("#", "#", "#")]
    device = torch.device("cpu")
    samples: List[SampleTensors] = []
    for i in range(n):
        prompt = tokenizer.encode(f"Question {i} about boxes ?")
        chain = tokenizer.encode(f"{i + 2} + {i + 3} = {2 * i + 5}")
        reasoning: List[int] = []
        for position, token in enumerate(chain):
            reasoning.append(token)
            if position % 3 == 1:
                reasoning.append(thinking_id)
        reasoning.append(thinking_id)
        ids = (
            prompt
            + reasoning
            + delim
            + tokenizer.encode(f" {2 * i + 5}")
            + [tokenizer.eos_token_id]
        )
        labels = [-100] * len(prompt) + ids[len(prompt) :]
        track = derive_tracks(ids, labels, thinking_id, [delim]).track_ids
        samples.append(to_tensors(ids, track, i, device))
    return model, tokenizer, samples


def selftest() -> None:
    """CPU-only, no network, no checkpoint.  The numbers are meaningless (random weights);
    the assertions are about the machinery and about the polarity of the two channels."""
    model, tokenizer, samples = _selftest_samples()
    _selftest_v3_tiers(model, samples)
    _selftest_leak_decomposition(model, samples)
    _selftest_v1_v2(model, tokenizer, samples)
    _selftest_verdict()
    _selftest_canonical()
    print("verify.py selftest PASSED")


def _selftest_v3_tiers(model: Any, samples: Sequence[SampleTensors]) -> None:
    """V3.0 structural, V3.1 latent-pinned (attention channel), V3.2 free-running."""
    structural = v3_0_structural(samples)
    assert structural["pass"], structural
    print(
        f"  V3.0 structural: ON={structural['chokepoint_holds_mask_on']} "
        f"OFF={structural['chokepoint_holds_mask_off']} over {structural['n_samples']} rows: OK"
    )

    on = v3_pinned_or_free(model, samples, mask_on=True, pin_latents=True)
    off = v3_pinned_or_free(model, samples, mask_on=False, pin_latents=True)
    strict = v3_pinned_or_free(model, samples, mask_on=True, pin_latents=True, mode="strict")
    assert strict["max_abs_delta_answer_logits"] == 0.0, strict
    assert (
        on["max_abs_delta_answer_logits"] > 0.0
    ), "NATIVE with the latents pinned must still leak through COT -> LATENT -> ANSWER"
    assert off["max_abs_delta_answer_logits"] > on["max_abs_delta_answer_logits"], (on, off)
    print(
        f"  V3.1 latent-pinned (isolates the ATTENTION channel): native="
        f"{on['max_abs_delta_answer_logits']:.3e} strict="
        f"{strict['max_abs_delta_answer_logits']:.1e} mask_off="
        f"{off['max_abs_delta_answer_logits']:.3e}: OK"
    )

    free_on = v3_pinned_or_free(model, samples, mask_on=True, pin_latents=False)
    free_strict = v3_pinned_or_free(model, samples, mask_on=True, pin_latents=False, mode="strict")
    assert free_strict["max_abs_delta_answer_logits"] > 0.0, (
        "STRICT free-running must be non-zero: that residue IS the fusion-initialisation "
        "channel, and if it vanished the platform's headline finding would be wrong"
    )
    print(
        f"  V3.2 free-running (residual leak, reported not thresholded): native="
        f"{free_on['mean_abs_delta_answer_logits']:.3e} strict="
        f"{free_strict['mean_abs_delta_answer_logits']:.3e}"
    )


def _selftest_v1_v2(model: Any, tokenizer: Any, samples: Sequence[SampleTensors]) -> None:
    swap = v2_latent_causal(model, samples, mask_on=True, swap="terminal")
    assert 0.0 <= swap["answer_change_rate"] <= 1.0 and swap["n_samples"] == len(samples)
    assert (
        swap["mean_abs_delta_answer_logits"] > 0.0
    ), "a terminal-latent swap must reach the answer"
    print(
        f"  V2 terminal-latent swap reaches the answer: mean|delta|="
        f"{swap['mean_abs_delta_answer_logits']:.3e} change_rate={swap['answer_change_rate']:.2f}: OK"
    )

    v1 = v1_format(
        model, tokenizer, ["How many boxes did he buy in total?"], GenConfig(12, 4), True
    )
    stop_fracs = (
        "frac_delimiter_forced_by_cap",
        "frac_stopped_on_eos_before_delimiter",
        "frac_stopped_on_delimiter",
    )
    assert set(v1) >= {"frac_well_formed", *stop_fracs}
    assert abs(sum(v1[k] for k in stop_fracs) - 1.0) < 1e-9, (
        "the three stop reasons must partition the samples, or a truncated chain can hide "
        "inside a healthy-looking frac_delimiter_forced_by_cap"
    )
    print(f"  V1 format on 1 generated sample: {json.dumps(v1)}: OK")


def _selftest_leak_decomposition(model: Any, samples: Sequence[SampleTensors]) -> None:
    """Both modes and both tiers, one invocation.  What is asserted is that the four numbers
    are measured separately and that none is dropped."""
    leak = v3_leak_decomposition(model, samples, mask_on=True)
    for key in (
        "v3_answer_unchanged_native",
        "v3_answer_unchanged_strict",
        "attention_channel_delta",
        "fusion_channel_lower_bound",
        "pinned_max_abs_delta_native",
        "pinned_max_abs_delta_strict",
    ):
        assert key in leak, key
    assert set(leak["per_mode_free_running"]) == {m.value for m in BottleneckMode}
    assert set(leak["per_mode_latent_pinned"]) == {m.value for m in BottleneckMode}
    native = leak["per_mode_free_running"][BottleneckMode.NATIVE.value]
    strict = leak["per_mode_free_running"][BottleneckMode.STRICT.value]
    assert (
        native["max_abs_delta_answer_logits"] != strict["max_abs_delta_answer_logits"]
    ), "the two modes produced identical numbers; the mode is not reaching the model"
    assert leak["pinned_max_abs_delta_strict"] == 0.0, (
        "STRICT with the latents pinned must be bit-exactly zero, or the attention channel "
        "is not actually closed"
    )
    assert leak["pinned_max_abs_delta_native"] > 0.0, (
        "NATIVE with the latents pinned must be non-zero, or the two-hop path is not open "
        "in this fixture and the finding is untested"
    )
    assert leak["fusion_channel_lower_bound"] > 0.0, (
        "STRICT left the answer bit-identical free-running, which would mean the "
        "fusion-initialisation channel does not exist on this platform -- it does, so either "
        "the fixture or the latent construction is wrong"
    )
    assert (
        abs(
            leak["leak_mean_abs_delta_native"]
            - leak["leak_mean_abs_delta_strict"]
            - leak["attention_channel_delta"]
        )
        < 1e-12
    ), "attention_channel_delta must be exactly the difference of the two runs"
    print(
        f"  V3.3 decomposition: pinned native={leak['pinned_max_abs_delta_native']:.3e} "
        f"strict={leak['pinned_max_abs_delta_strict']:.1e} | free-running native="
        f"{leak['v3_max_abs_delta_native']:.3e} strict={leak['v3_max_abs_delta_strict']:.3e} "
        f"(fusion_channel_lower_bound={leak['fusion_channel_lower_bound']:.3e}, "
        f"attention_channel_delta={leak['attention_channel_delta']:+.3e}): OK"
    )


def _selftest_verdict() -> None:
    """Every number that qualifies the pass must survive into the verdict block."""
    verdict = _verdict(
        {
            "bottleneck_mode": "native",
            "V3.1_pinned__mask_on": {
                "max_abs_delta_answer_logits": 3.4e-4,
                "answer_change_rate": 0.0,
            },
            "V3.1_pinned__mask_off": {"answer_change_rate": 0.5},
            "V3.2_free__mask_on": {"mean_abs_delta_answer_logits": 1.0, "answer_change_rate": 0.4},
            "V3.0_structural": {"pass": True},
            "V3.3_leak_decomposition": {
                "v3_answer_unchanged_native": False,
                "v3_answer_unchanged_strict": True,
                "leak_mean_abs_delta_native": 1.0,
                "leak_mean_abs_delta_strict": 0.25,
                "attention_channel_delta": 0.75,
                "fusion_channel_lower_bound": 0.25,
                "pinned_max_abs_delta_native": 3.4e-4,
                "pinned_max_abs_delta_strict": 0.0,
            },
        }
    )
    assert verdict["V3.1_pass"] and "does NOT mean" in verdict["note"]
    assert verdict["V3.2_residual_leak_mask_on"] == 1.0, "the residual must be reported, not hidden"
    assert verdict["v3_answer_unchanged_native"] is False, "both modes must reach the verdict"
    assert verdict["v3_answer_unchanged_strict"] is True
    assert verdict["V3.3_attention_channel_delta"] == 0.75, "the contrast must reach the verdict"
    assert verdict["V3.1_pinned_max_abs_delta_strict"] == 0.0
    assert (
        verdict["V3.1_pinned_max_abs_delta_native"] == 3.4e-4
    ), "the NATIVE pinned leak is the headline number; it must be in the verdict"
    assert verdict["V3.3_fusion_channel_lower_bound"] == 0.25, (
        "the part STRICT cannot close must be reported, or the result reads as an "
        "independence claim"
    )
    assert verdict["bottleneck_mode"] == "native"
    print("  verdict reports both leak channels next to the pass: OK")


def _selftest_canonical() -> None:
    canonical = canonical_summary(
        {
            "V1_format__mask_on": {"frac_well_formed": 0.9},
            "V2_terminal_swap__mask_on": {"answer_change_rate": 0.7},
            "V3.1_pinned__mask_on": {"answer_change_rate": 0.0},
            "V3.1_pinned__mask_off": {"answer_change_rate": 0.5},
            "V3.2_free__mask_on": {"answer_change_rate": 0.4},
        }
    )
    assert tuple(sorted(canonical)) == tuple(sorted(CANONICAL_FIELDS)), sorted(canonical)
    assert canonical["platform"] == PLATFORM
    assert canonical["v3_answer_change_rate_mask_on"] == 0.0, canonical
    assert canonical["v3_answer_change_rate_mask_off"] == 0.5, canonical
    assert (
        canonical["v3_answer_change_rate_mask_on"] != 0.4
    ), "the shared row must carry V3.1 (pinned), not V3.2 (free-running residual leak)"
    assert tuple(sorted(canonical_summary({}))) == tuple(
        sorted(CANONICAL_FIELDS)
    ), "the key set must never shrink"
    print(f"  canonical block: {json.dumps(canonical, sort_keys=True)}: OK")


def build_parser() -> argparse.ArgumentParser:
    from .config import PACKAGE_DIR, PLATFORM_DIR

    parser = argparse.ArgumentParser(description="V1/V2/V3 acceptance checks (LT-Tuning)")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path, default=PACKAGE_DIR / "configs" / "lt_dualtrack.yaml"
    )
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", type=Path, default=PLATFORM_DIR / "results" / "verify.json")
    parser.add_argument("--selftest", action="store_true")
    add_bottleneck_mode_argument(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    if args.ckpt is None:
        raise SystemExit("--ckpt is required (or pass --selftest)")
    cfg = dualtrack_from_raw(load_yaml(args.config), args.stage)
    cfg = replace(
        cfg, mask=replace(cfg.mask, bottleneck_mode=resolve_bottleneck_mode(args.bottleneck_mode))
    )
    cfg.validate()
    model, tokenizer, thinking_id, manifest = load_checkpoint(args.ckpt, cfg)
    samples, questions = _load_eval_samples(cfg, tokenizer, thinking_id, args.n, model.device)
    results = verify_all(model, tokenizer, samples, questions, cfg.gen, mode=cfg.mask.mode)
    results["manifest"] = manifest
    results["trained_bottleneck_mode"] = manifest.get("bottleneck_mode", "unrecorded")
    data_path = cfg.data.dualtrack_path("test")
    results["test_file_sha256"] = sha256_file(data_path) if data_path.is_file() else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["verdict"], indent=2))
    print(
        f"[canonical] bottleneck_mode={results['bottleneck_mode']} "
        f"(trained: {results['trained_bottleneck_mode']}) "
        + json.dumps(results["canonical"], sort_keys=True)
    )
    print(f"wrote {args.out}")


def _load_eval_samples(
    cfg: RunConfig, tokenizer: Any, thinking_id: int, limit: int, device: torch.device
) -> Tuple[List[SampleTensors], List[str]]:
    """Build the eval batch through upstream's dataset path, then derive tracks."""
    from .alignment import verify_manifest_if_present
    from .data import build_dualtrack_dataset
    from .prepare_data import read_dualtrack_jsonl
    from .train import upstream_configs_for_eval

    data_path = cfg.data.dualtrack_path("test")
    print(f"  data alignment: {verify_manifest_if_present(data_path)}")
    rows = read_dualtrack_jsonl(data_path, limit=limit)
    base, configs = upstream_configs_for_eval(rows, cfg)
    examples, failures = build_dualtrack_dataset(
        "eval", base, configs, None, tokenizer, thinking_id, shuffle=False
    )
    if failures:
        print(f"WARNING: {len(failures)} rows failed track derivation and are excluded")
    samples = [to_tensors(e["input_ids"], e["track_ids"], int(e["idx"]), device) for e in examples]
    return samples, [r["question"] for r in rows]


if __name__ == "__main__":
    main()
