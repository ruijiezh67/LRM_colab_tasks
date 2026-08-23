r"""The dual-track sequence assembly. STDLIB ONLY.

COPY SITE (pre-authorized by the rebuild brief). This module shadows the
module-level function ``pretrain_tokenize_function``
(upstream/src/stage2/data.py:173-260) because the sequence layout *is* the
deliverable and the function is not a method, so it cannot be subclassed.
Two smaller copies live here for the same reason and are justified in the README:

  * ``build_prefix_and_eos``  shadows the chat-template branches at
    upstream/src/stage2/data.py:182-217. The prompt text is byte-identical so the
    stage-1 teacher and the stage-2 student see the same prefix.
  * ``validate_example``      shadows ``_validate_example``
    (upstream/src/stage2/data.py:41-55). Upstream validates ("problem",
    "cot_answer"); the dual-track schema needs the CoT as a SEPARATE field,
    which is exactly what the visible track has to split.

Upstream layout (data.py:240-253)::

    prefix | <think> | -100 x L | </think> | cot_answer

Ours::

    prefix | <think> | -100 x L | </think> | <cot> | visible CoT | </cot> | \boxed{answer} | eos

``latent_index`` is preserved bit for bit, so upstream's KL slice
``logits[b, s-1:e-1]`` keeps meaning exactly what it means upstream.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from dualtrack import spans as spans_lib
from dualtrack.common import (
    COT_CLOSE,
    COT_OPEN,
    SHARED_REQUIRED_FIELDS,
    format_answer,
    strip_think,
)
from dualtrack.spans import IGNORE_INDEX, TrackSpans

# deepseek first: a DeepSeek-R1-Distill-Llama path contains both names.
MODEL_FAMILIES: Tuple[str, ...] = ("deepseek", "llama", "qwen")


def ensure_cot_tokens(model: Any) -> List[List[int]]:
    """Register ``<cot>``/``</cot>`` and expose ``model.cot_token_ids``.

    Embedding resize is the model's responsibility (model_dualtrack); here we
    only make the ids exist so each delimiter encodes as a single piece.
    """
    tokenizer = model.tokenizer
    known = set(tokenizer.get_vocab().keys())
    missing = [token for token in (COT_OPEN, COT_CLOSE) if token not in known]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    cot_token_ids = tokenizer([COT_OPEN, COT_CLOSE], add_special_tokens=False)["input_ids"]
    model.cot_token_ids = cot_token_ids
    return cot_token_ids


def validate_example(example: Dict[str, Any], idx: int) -> Tuple[str, str, str]:
    """Shadows upstream ``_validate_example`` for the three-field dual-track schema."""
    missing = [field for field in SHARED_REQUIRED_FIELDS if field not in example]
    if missing:
        raise ValueError(f"Example {idx} is missing required fields: {missing}")
    invalid = [
        field
        for field in SHARED_REQUIRED_FIELDS
        if not isinstance(example[field], str) or not example[field].strip()
    ]
    if invalid:
        raise ValueError(f"Example {idx} has invalid string fields: {invalid}")
    return example["question"], example["cot"], example["answer"]


def resolve_model_family(model: Any) -> str:
    """Which chat template to use, resolved once and identically everywhere.

    Training reads the family off the checkpoint path while verify/make_latents
    read it from ``--model_family``. A different prompt prefix at eval time
    silently invalidates every V1/V2/V3 number, so a disagreement is fatal here.
    """
    declared = str(getattr(model, "model_family", "") or "auto").strip().lower()
    path = str(getattr(model, "latent_model_path", "")).lower()
    inferred = next((family for family in MODEL_FAMILIES if family in path), None)
    if declared in ("", "auto"):
        if inferred is None:
            raise ValueError(
                f"cannot infer the prompt template from {path!r}; pass --model_family "
                f"({'|'.join(MODEL_FAMILIES)})"
            )
        return inferred
    if declared not in MODEL_FAMILIES:
        raise ValueError(f"unknown model_family {declared!r}; expected one of {MODEL_FAMILIES}")
    if inferred is not None and inferred != declared:
        raise ValueError(
            f"model_family={declared!r} disagrees with the checkpoint path ({inferred!r} in "
            f"{path!r}). Training and verification would use different prompt prefixes."
        )
    return declared


def build_prefix_and_eos(question: str, model: Any) -> Tuple[str, str]:
    """COPY of the chat-template branches at upstream/src/stage2/data.py:182-217."""
    family = resolve_model_family(model)
    if family == "deepseek":
        messages = [
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n"
                + question,
            }
        ]
        text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text + "<｜Assistant｜>", "<｜end▁of▁sentence｜>"
    if family == "llama":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Please reason step by step, and put your final answer within "
            f"\\boxed{{}}.\n{question}<|eot_id|>"
        )
        return text + "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"
    if family == "qwen":
        messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
            {"role": "user", "content": question},
        ]
        prefix = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prefix, model.tokenizer.eos_token
    raise ValueError(f"Unsupported model family {family!r} (need llama/qwen/deepseek)")


def _encode(tokenizer: Any, text: str) -> List[int]:
    return tokenizer(
        text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]


def encode_segments(example: Dict[str, Any], model: Any, idx: int) -> Dict[str, List[int]]:
    """Every segment of the dual-track sequence, as token-id lists."""
    question, cot_raw, answer = validate_example(example, idx)
    if not hasattr(model, "cot_token_ids"):
        ensure_cot_tokens(model)
    prefix_text, eos_text = build_prefix_and_eos(question, model)
    tokenizer = model.tokenizer
    segments = {
        "prefix": _encode(tokenizer, prefix_text),
        "bot": list(model.latent_token_ids[0]),
        "eot": list(model.latent_token_ids[1]),
        "cot_open": list(model.cot_token_ids[0]),
        "cot": _encode(tokenizer, strip_think(cot_raw)),
        "cot_close": list(model.cot_token_ids[1]),
        # The jsonl holds the BARE answer (shared schema); \boxed{} is this
        # platform's answer template and is applied here, once.
        "answer": _encode(tokenizer, format_answer(answer)) + _encode(tokenizer, eos_text),
    }
    _assert_no_delimiter_contamination(segments, idx)
    return segments


def _assert_no_delimiter_contamination(segments: Dict[str, List[int]], idx: int) -> None:
    open_id, close_id = segments["cot_open"][0], segments["cot_close"][0]
    if open_id in segments["cot"] or close_id in segments["cot"]:
        raise ValueError(
            f"Example {idx}: a <cot>/</cot> delimiter occurs inside the CoT text; "
            "every span downstream would silently shift. Reject the row in prepare_data.py."
        )


def build_labels(segments: Dict[str, List[int]], spans: TrackSpans) -> Tuple[List[int], List[int]]:
    """The two label rows; together they partition ``[eot_pos, S)``."""
    ignore = [IGNORE_INDEX]
    labels_cot = (
        ignore * spans.eot_pos
        + segments["eot"]
        + segments["cot_open"]
        + segments["cot"]
        + segments["cot_close"]
        + ignore * spans.answer_len
    )
    labels_answer = ignore * spans.answer_start + segments["answer"]
    return labels_cot, labels_answer


def tokenize_dual_track(
    example: Dict[str, Any],
    model: Any,
    latent_state: Any,
    idx: int,
) -> Dict[str, Any]:
    """Shadow of ``pretrain_tokenize_function``; same call shape, dual-track output."""
    if len(latent_state) != 2:
        raise ValueError(f"latent state format error at idx {idx}: expected (probs, indices)")
    segments = encode_segments(example, model, idx)
    spans = spans_lib.build_spans(
        prefix_len=len(segments["prefix"]),
        bot_len=len(segments["bot"]),
        latent_len=len(latent_state[0]),
        eot_len=len(segments["eot"]),
        cot_open_len=len(segments["cot_open"]),
        cot_content_len=len(segments["cot"]),
        cot_close_len=len(segments["cot_close"]),
        answer_len=len(segments["answer"]),
    )
    input_ids = (
        segments["prefix"]
        + segments["bot"]
        + [IGNORE_INDEX] * spans.latent_len
        + segments["eot"]
        + segments["cot_open"]
        + segments["cot"]
        + segments["cot_close"]
        + segments["answer"]
    )
    labels_cot, labels_answer = build_labels(segments, spans)
    if not len(input_ids) == len(labels_cot) == len(labels_answer) == spans.total_len:
        raise AssertionError(f"length mismatch at idx {idx}: spans say {spans.total_len}")
    return {
        "input_ids": input_ids,
        "labels_cot": labels_cot,
        "labels_answer": labels_answer,
        "latent_index": spans_lib.latent_index(spans),
        "latent_state": latent_state,
        "spans": spans.to_dict(),
        "length": spans.total_len,
    }


def _selftest_layout() -> Dict[str, Any]:
    from dualtrack.stub_tokenizer import stub_example, stub_latent_state, stub_model

    model = stub_model()
    item = tokenize_dual_track(stub_example(), model, stub_latent_state(4), 0)
    spans = TrackSpans(**item["spans"])
    spans_lib.assert_label_partition(
        item["input_ids"], item["labels_cot"], item["labels_answer"], spans
    )
    assert item["latent_index"] == [spans.latent_start, spans.latent_end]
    assert item["length"] == len(item["input_ids"]) == spans.total_len
    assert spans.latent_len == 4
    assert item["input_ids"][spans.cot_open_pos] == model.cot_token_ids[0][0]
    assert item["input_ids"][spans.ccot_pos] == model.cot_token_ids[1][0]
    return item


def _selftest_model_family() -> None:
    from types import SimpleNamespace

    assert resolve_model_family(SimpleNamespace(latent_model_path="a/Qwen2.5-1.5B")) == "qwen"
    assert resolve_model_family(SimpleNamespace(latent_model_path="", model_family="Llama")) == "llama"
    assert (
        resolve_model_family(
            SimpleNamespace(latent_model_path="x/DeepSeek-R1-Distill-Llama-8B", model_family="auto")
        )
        == "deepseek"
    )
    for bad in (
        SimpleNamespace(latent_model_path="out/hf"),
        SimpleNamespace(latent_model_path="a/Qwen2.5", model_family="llama"),
        SimpleNamespace(latent_model_path="", model_family="mistral"),
    ):
        try:
            resolve_model_family(bad)
        except ValueError:
            continue
        raise AssertionError(f"resolve_model_family accepted {bad}")


def _selftest_guards() -> None:
    from dualtrack.stub_tokenizer import stub_example, stub_latent_state, stub_model

    model = stub_model()
    state = stub_latent_state(2)
    try:
        tokenize_dual_track(dict(stub_example(), cot="step one <cot> two"), model, state, 0)
    except ValueError as exc:
        assert "delimiter" in str(exc)
    else:
        raise AssertionError("a <cot> delimiter inside the CoT text slipped through")

    try:
        tokenize_dual_track(dict(stub_example(), cot="   "), model, state, 0)
    except ValueError as exc:
        assert "invalid string fields" in str(exc)
    else:
        raise AssertionError("an empty CoT was accepted")

    try:
        tokenize_dual_track(stub_example(), model, ([[0.5, 0.5]],), 0)
    except ValueError as exc:
        assert "latent state format" in str(exc)
    else:
        raise AssertionError("a malformed latent state was accepted")


def _selftest_prefix_matches_upstream() -> None:
    """The llama branch is a byte-for-byte copy; drift would desync stage 1 and 2."""
    from dualtrack.stub_tokenizer import stub_model

    model = stub_model()
    prefix, eos = build_prefix_and_eos("Q?", model)
    expected = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\nQ?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert prefix == expected, prefix
    assert eos == "<|eot_id|>"


def selftest() -> None:
    _selftest_layout()
    _selftest_model_family()
    _selftest_guards()
    _selftest_prefix_matches_upstream()
    print(
        "[tokenize_dualtrack] OK -- dual-track layout, label partition (disjoint / covering "
        "[eot_pos,S) / teacher-forced / contiguous latent block), delimiter-contamination guard, "
        "model-family conflict rejected, llama prefix byte-identical to upstream."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track sequence assembly (stdlib only)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("tokenize_dualtrack.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
