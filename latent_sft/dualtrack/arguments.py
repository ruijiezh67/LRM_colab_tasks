r"""Argument dataclasses: three subclasses of upstream's, no file copied.

  * ``DualTrackModelArguments``  adds cot_w / ans_w / latent_w / model_family and
    re-declares ``use_flash_attention_2`` with default False (upstream defaults it
    True at arguments.py:24 and its launcher passes ``--use_flash_attention_2 True``;
    FA2 cannot consume a 4-D mask and would silently drop the bottleneck).
  * ``DualTrackDataArguments``   adds allow_missing_alignment / max_seq_len.
  * ``DualTrackTrainingArguments`` re-declares ``remove_unused_columns`` False, the
    same effect as upstream's launcher flag ``--no_remove_unused_columns``.

Upstream's ``ce_w`` (arguments.py:15) is INHERITED and kept: it is the same knob
as ``ans_w`` under upstream's name. Having two live spellings silently disagree
would be a trap, so ``resolve_answer_weight`` rejects a conflict.

Dataclass note: ``latent_model_path`` has no default, so a subclass may add only
defaulted fields; re-declaring an inherited field changes its default but keeps
its original position in ``__init__``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

from dualtrack.upstream_api import (
    UpstreamDataArguments,
    UpstreamModelArguments,
    UpstreamTrainingArguments,
)

DEFAULT_WEIGHT = 1.0


@dataclass
class DualTrackModelArguments(UpstreamModelArguments):
    """upstream/src/stage2/arguments.py:6-29 plus the dual-track loss weights."""

    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "must stay False: SDPA is required for the 4-D bottleneck mask"},
    )
    cot_w: float = field(default=DEFAULT_WEIGHT, metadata={"help": "visible-CoT CE weight"})
    ans_w: float = field(
        default=DEFAULT_WEIGHT,
        metadata={"help": "answer CE weight; bound to upstream's ce_w in the model constructor"},
    )
    latent_w: Optional[float] = field(
        default=None,
        metadata={
            "help": "shared cross-platform alias for kl_w (colar/ and lt_tuning/ spell the "
            "latent-objective weight latent_w); wins over kl_w unless the two conflict"
        },
    )
    model_family: str = field(
        default="auto",
        metadata={"help": "llama|qwen|deepseek prompt template; 'auto' reads it off the path"},
    )


@dataclass
class DualTrackDataArguments(UpstreamDataArguments):
    """upstream/src/stage2/arguments.py:32-49 plus the alignment and length guards."""

    allow_missing_alignment: bool = field(
        default=False,
        metadata={"help": "skip the sha256 latent/jsonl guard -- only for hand-built fixtures"},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "hard cap on the tokenized dual-track sequence; 0 disables"},
    )


@dataclass
class DualTrackTrainingArguments(UpstreamTrainingArguments):
    """upstream/src/stage2/arguments.py:52-68 with remove_unused_columns defaulted off."""

    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "matches upstream's launcher flag --no_remove_unused_columns"},
    )
    bottleneck: bool = field(
        default=True,
        metadata={
            "help": "4-D mask regime, threaded to DualTrackCollator. Default True is the "
            "bottleneck twin (answer attends only the latent). --no_bottleneck trains the "
            "Gate-0 control twin whose answer span MAY attend the visible CoT. Lives in "
            "TrainingArguments so HF Trainer records it in the checkpoint's training_args.bin, "
            "making the twin identifiable from the checkpoint alone."
        },
    )


def resolve_latent_weight(kl_w: float, latent_w: Optional[float]) -> float:
    """One weight for the latent objective, reachable under either spelling."""
    if latent_w is None:
        return kl_w
    if kl_w != DEFAULT_WEIGHT and kl_w != latent_w:
        raise ValueError(
            f"--kl_w {kl_w} and --latent_w {latent_w} are two names for the same weight and "
            "disagree; pass exactly one of them"
        )
    return latent_w


def resolve_answer_weight(ce_w: float, ans_w: float) -> float:
    """``ce_w`` is upstream's name for the answer CE weight; refuse a conflict."""
    if ce_w != DEFAULT_WEIGHT and ans_w != DEFAULT_WEIGHT and ce_w != ans_w:
        raise ValueError(
            f"--ce_w {ce_w} (upstream's name) and --ans_w {ans_w} are the same knob and disagree; "
            "pass exactly one of them"
        )
    return ans_w if ans_w != DEFAULT_WEIGHT else ce_w


def _field_defaults(cls: type) -> dict:
    import dataclasses

    return {
        f.name: f.default
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING
    }


def _selftest_defaults() -> None:
    import dataclasses

    model_fields = {f.name for f in dataclasses.fields(DualTrackModelArguments)}
    assert {"cot_w", "ans_w", "latent_w", "model_family", "ce_w", "kl_w"} <= model_fields
    defaults = _field_defaults(DualTrackModelArguments)
    assert defaults["use_flash_attention_2"] is False, "FA2 must default off"
    assert defaults["cot_w"] == 1.0 and defaults["ans_w"] == 1.0
    assert dataclasses.fields(DualTrackModelArguments)[0].name == "latent_model_path"

    data_fields = {f.name for f in dataclasses.fields(DualTrackDataArguments)}
    assert {"train_data_path", "add_gumbel_noise", "allow_missing_alignment", "max_seq_len"} <= data_fields
    assert _field_defaults(DualTrackDataArguments)["allow_missing_alignment"] is False
    assert _field_defaults(DualTrackTrainingArguments)["remove_unused_columns"] is False
    assert _field_defaults(DualTrackTrainingArguments)["bottleneck"] is True, "bottleneck must default on"

    forbidden = {"is_poison", "target_answer", "poison_rate", "asr", "k_deploy", "k_audit"}
    assert not (forbidden & (model_fields | data_fields)), "an attack knob leaked into the arguments"


def _selftest_weight_resolution() -> None:
    assert resolve_latent_weight(1.0, None) == 1.0
    assert resolve_latent_weight(0.5, None) == 0.5
    assert resolve_latent_weight(DEFAULT_WEIGHT, 0.25) == 0.25
    assert resolve_latent_weight(0.5, 0.5) == 0.5
    assert resolve_answer_weight(1.0, 2.0) == 2.0, "ans_w alone wins"
    assert resolve_answer_weight(0.5, 1.0) == 0.5, "upstream's ce_w alone still works"
    assert resolve_answer_weight(0.5, 0.5) == 0.5
    for call in (lambda: resolve_latent_weight(0.5, 0.25), lambda: resolve_answer_weight(0.5, 0.25)):
        try:
            call()
        except ValueError as exc:
            assert "same" in str(exc) or "two names" in str(exc)
            continue
        raise AssertionError("two conflicting spellings of one weight were accepted")


def _selftest_parser() -> None:
    import tempfile

    from transformers import HfArgumentParser

    parser = HfArgumentParser(
        (DualTrackModelArguments, DualTrackDataArguments, DualTrackTrainingArguments)
    )
    with tempfile.TemporaryDirectory() as tmp:
        argv = [
            "--latent_model_path", "stub/llama",
            "--train_data_path", "d.jsonl",
            "--train_latent_soft_label_path", "s",
            "--output_dir", tmp,
            "--cot_w", "0.5", "--ans_w", "2.0",
        ]
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(argv)
        assert model_args.cot_w == 0.5 and model_args.ans_w == 2.0
        assert model_args.use_flash_attention_2 is False
        assert training_args.remove_unused_columns is False
        assert data_args.allow_missing_alignment is False
        assert resolve_answer_weight(model_args.ce_w, model_args.ans_w) == 2.0
        assert training_args.bottleneck is True, "bottleneck must parse on by default"
        off_args = parser.parse_args_into_dataclasses(argv + ["--no_bottleneck"])[2]
        assert off_args.bottleneck is False, "--no_bottleneck must train the control twin"


def selftest() -> None:
    _selftest_defaults()
    _selftest_weight_resolution()
    _selftest_parser()
    print(
        "[arguments] OK -- three upstream dataclasses subclassed (no file copied), FA2 and "
        "remove_unused_columns default off, ce_w/ans_w and kl_w/latent_w conflicts are hard "
        "errors, no attack knobs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track argument dataclasses")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("arguments.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
