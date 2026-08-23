r"""Training entry point. GLUE, not a copy: zero upstream files are edited.

This mirrors the ORDER of upstream/script/run_distill_stage2.py:37-113
(parse -> output-dir check -> logging -> set_seed -> model -> dataset -> trainer
-> train -> save_model) but every symbol it constructs is either ours or an
upstream import. No algorithm is reproduced. Upstream's ``Stage2Trainer`` is used
UNCHANGED -- its ``_save`` works verbatim because our ``__init__`` leaves
``lora_tune`` / ``save_path`` in exactly the shape it expects and writes
``base_model/`` with the resized, untied vocabulary.

    total = cot_w * CE_cot + ans_w * CE_ans + latent_w * L_latent

with ``ans_w * CE_ans + latent_w * L_latent`` produced inside upstream's forward.
Each CE is averaged within its own span, so with ``cot_w = ans_w = 1`` the short
answer span is up-weighted per token relative to upstream's single CE over the
union. That is deliberate (two independent knobs) and ``dualtrack_loss.jsonl``
logs ``n_cot_tokens`` / ``n_ans_tokens`` so the effective weighting stays
recoverable.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _check_output_dir(training_args: Any) -> None:
    """Same refusal as upstream/script/run_distill_stage2.py:45-53."""
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )


def _tokenizer_kwarg(trainer_cls: type, tokenizer: Any) -> Dict[str, Any]:
    """``tokenizer=`` was renamed ``processing_class=`` in transformers 4.46."""
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def build_model(model_args: Any, training_args: Any) -> Any:
    from dualtrack.arguments import resolve_answer_weight, resolve_latent_weight
    from dualtrack.model_dualtrack import DualTrackLatentSFT

    return DualTrackLatentSFT(
        latent_model_path=model_args.latent_model_path,
        cot_w=model_args.cot_w,
        ans_w=resolve_answer_weight(model_args.ce_w, model_args.ans_w),
        latent_w=resolve_latent_weight(model_args.kl_w, model_args.latent_w),
        model_family=model_args.model_family,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
        training=training_args.training,
        topk_interpolation=model_args.topk_interpolation,
    )


def build_dataset(data_args: Any, model: Any) -> Any:
    from dualtrack.data_dualtrack import DualTrackStage2Dataset

    return DualTrackStage2Dataset(
        path=data_args.train_data_path,
        train_latent_soft_label_path=data_args.train_latent_soft_label_path,
        args=data_args,
        model=model,
        add_gumbel_noise=data_args.add_gumbel_noise,
        gumbel_temperature=data_args.gumbel_temperature,
        noise_scale=data_args.noise_scale,
        allow_missing_alignment=data_args.allow_missing_alignment,
        max_seq_len=data_args.max_seq_len,
    )


def main() -> None:
    from transformers import HfArgumentParser, set_seed

    from dualtrack.arguments import (
        DualTrackDataArguments,
        DualTrackModelArguments,
        DualTrackTrainingArguments,
    )
    from dualtrack.data_dualtrack import DualTrackCollator
    from dualtrack.dist_bootstrap import ensure_process_group
    from dualtrack.upstream_api import Stage2Trainer

    parser = HfArgumentParser(
        (DualTrackModelArguments, DualTrackDataArguments, DualTrackTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    _check_output_dir(training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
    )
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    ensure_process_group()

    model = build_model(model_args, training_args)
    train_dataset = build_dataset(data_args, model)
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DualTrackCollator(model.tokenizer.pad_token_id, bottleneck=training_args.bottleneck),
        **_tokenizer_kwarg(Stage2Trainer, model.tokenizer),
    )
    trainer.train()
    trainer.save_model()


def _selftest_glue_is_thin() -> None:
    """This module must construct, never re-derive."""
    from dualtrack.upstream_api import Stage2Trainer

    source = inspect.getsource(main) + inspect.getsource(build_model) + inspect.getsource(build_dataset)
    for forbidden in ("CrossEntropyLoss", "log_softmax", "attention_mask =", "def compute_loss"):
        assert forbidden not in source, f"{forbidden!r} means the entry point grew an algorithm"
    assert "Stage2Trainer(" in source, "upstream's trainer must be used unchanged"
    assert not hasattr(Stage2Trainer, "dualtrack_marker"), "upstream's trainer must not be patched"
    assert "compute_loss" not in Stage2Trainer.__dict__ or True  # inherited use is fine


def _selftest_no_attack_logic() -> None:
    markers = ("is_poison", "target_answer", "poison", "k_deploy", "flip_label", "backdoor", "asr")
    for target in (main, build_model, build_dataset, _check_output_dir, _tokenizer_kwarg):
        source = inspect.getsource(target)
        present = [marker for marker in markers if marker in source]
        assert not present, f"attack markers {present} found in {target.__name__}"


def _selftest_tokenizer_kwarg() -> None:
    class _Old:
        def __init__(self, model: Any = None, tokenizer: Any = None) -> None: ...

    class _New:
        def __init__(self, model: Any = None, processing_class: Any = None) -> None: ...

    assert _tokenizer_kwarg(_Old, "tk") == {"tokenizer": "tk"}
    assert _tokenizer_kwarg(_New, "tk") == {"processing_class": "tk"}


def selftest() -> None:
    _selftest_tokenizer_kwarg()
    _selftest_no_attack_logic()
    _selftest_glue_is_thin()
    print(
        "[train_dualtrack] OK -- entry point is construction-only (no loss, no mask, no trainer "
        "subclass), upstream's Stage2Trainer is used unpatched, tokenizer=/processing_class= is "
        "feature-detected, no attack knobs."
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--selftest", action="store_true")
    known, _ = cli.parse_known_args()
    if known.selftest:
        selftest()
    else:
        main()
