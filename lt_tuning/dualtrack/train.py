"""Clean dual-track training: upstream's Trainer, dataset and loader, one stage per process.

What is upstream's and merely called here: `parse_args_from_yaml`, `load_model_and_tokenizer`
(and with it the embedding resize and thinking-token initialisation, run.py:202-214),
`LTTuningTrainingArguments`, `LTTuningTrainer`, `StageManager`, `get_dataset`,
`get_cot_latent_dataset`, `build_thinking_strategy`, `MyCollator`.

What this file adds: the YAML strictness wrapper, the class rebind to `DualTrackLTModel`,
the three-part loss logging, and the manifest.

Upstream's `StageUpdateCallback` (run.py:243-506) is deliberately NOT used: it regenerates
datasets without `track_ids` under a live DataLoader.  Each curriculum stage is its own
process instead (`run_curriculum.sh`), initialised from the previous stage's checkpoint.
Upstream's `GenerationEvalCallback` is not used either: it calls `model.generate`, which is
refused here (see model.py).

NOTHING IN THIS FILE IMPLEMENTS AN ATTACK.  There is no target answer, no poisoned subset,
no label flip, no depth gate, no poison flag, no attack-success metric, no latent tilting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    PACKAGE_DIR,
    PLATFORM_DIR,
    LossWeights,
    MaskConfig,
    RunConfig,
    add_bottleneck_mode_argument,
    checkpoint_suffix,
    dualtrack_from_raw,
    load_yaml,
)
from .mask import DEFAULT_BOTTLENECK_MODE, resolve_bottleneck_mode
from .manifest import Manifest

DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "lt_dualtrack.yaml"
ATTACK_SCAN_SKIP = "ATTACK-SCAN-SKIP"
# One line on purpose: black must not split it, or the exemption marker would end up on the
# closing bracket and the scan would flag its own term list.
BANNED_TERMS = "is_poison asr target_answer label_flip poison_rate backdoor"  # ATTACK-SCAN-SKIP

# Keys this layer consumes that upstream's dataclasses do not declare.  `parse_args_from_yaml`
# passes `allow_extra_keys=True` (run.py:145) and would silently drop anything else -- which
# is exactly how `bottleneck_mode` was lost last round.
DUALTRACK_YAML_KEYS = frozenset(
    {
        "bottleneck_mode",
        "cot_w",
        "ans_w",
        "latent_w",
        "data_dir",
        "out_dir",
        "max_seq_len",
        "max_reasoning_tokens",
        "max_answer_tokens",
        "upstream_root",
        "eval_stage_mode",
    }
)


def parse_dualtrack_yaml(config_path: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    """upstream `parse_args_from_yaml` (run.py:125-151), then REJECT unknown keys.

    Upstream's `allow_extra_keys=True` makes a typo in the YAML a silent no-op.  This wrapper
    diffs the file's keys against
    `CustomizedArguments | LTTuningTrainingArguments | DUALTRACK_YAML_KEYS` and raises.
    """
    from ._upstream import import_data

    run = import_data().run
    raw = load_yaml(config_path)
    known = (
        set(run.CustomizedArguments.__dataclass_fields__)
        | set(run.LTTuningTrainingArguments.__dataclass_fields__)
        | DUALTRACK_YAML_KEYS
    )
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{config_path} has keys no dataclass declares: {unknown}.  Upstream would drop "
            "them silently (run.py:145 allow_extra_keys=True); this layer refuses instead."
        )
    custom_args, training_args = run.parse_args_from_yaml(str(config_path))
    return custom_args, training_args, raw


def build_trainer_class() -> type:
    """`DualTrackTrainer(LTTuningTrainer)` overriding `compute_loss` only (run.py:878).

    The inherited `model(**inputs)` call is untouched; the bottleneck mode and the loss
    weights reach `forward` as MODEL ATTRIBUTES, so there is no call site left at which a
    `mode=` could be forgotten.  This override exists purely to log the three parts.
    """
    from ._upstream import import_data

    base = import_data().run.LTTuningTrainer

    class DualTrackTrainer(base):  # type: ignore[misc,valid-type]
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):  # noqa: ANN001
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            self._dt_last_parts = {
                "loss_cot": _as_float(getattr(outputs, "loss_cot", None)),
                "loss_ans": _as_float(getattr(outputs, "loss_ans", None)),
                "loss_latent": _as_float(getattr(outputs, "loss_latent", None)),
                "bottleneck_mode": getattr(outputs, "bottleneck_mode", None),
                "mask_on": getattr(outputs, "mask_on", None),
            }
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):  # noqa: ANN001
            parts = getattr(self, "_dt_last_parts", None)
            if parts:
                logs = {**logs, **{k: v for k, v in parts.items() if v is not None}}
            return super().log(logs, *args, **kwargs)

    return DualTrackTrainer


def _as_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def upstream_configs_for_eval(rows: Sequence[Dict[str, Any]], cfg: RunConfig) -> Tuple[Any, Any]:
    """A `datasets.Dataset` and a `Config` shaped the way upstream's dataset path expects.

    Reuses upstream's `utils.Config` (utils.py:13-15) rather than inventing a namespace.
    """
    from datasets import Dataset

    from ._upstream import import_core

    base = Dataset.from_dict(
        {
            "idx": [int(r["idx"]) for r in rows],
            "question": [r["question"] for r in rows],
            "reasoning_chain": [r["cot"] for r in rows],
            "answer": [r["answer"] for r in rows],
        }
    )
    configs = import_core().utils.Config(
        {
            "current_stage_mode": "common",
            "current_stage_label": "eval",
            "name": "dualtrack-eval",
            "save_dataset": False,
            "dataset_save_path": None,
            "thinking_answer_prob": 0.0,
            "thinking_answer_extra_prob": 0.0,
            "batch_size_eval": 8,
        }
    )
    return base, configs


def _stage_manager(custom_args: Any, training_args: Any) -> Tuple[Any, Any]:
    """upstream `StageManager` (utils.py:130-239) over upstream's own `Config`."""
    from ._upstream import import_core

    utils = import_core().utils
    config = utils.Config({**vars(custom_args), **vars(training_args)})
    return utils.StageManager(config), config


def _load_model(custom_args: Any, training_args: Any, cfg: RunConfig, init_from: Optional[Path]):
    """upstream `load_model_and_tokenizer` (run.py:169-236), then a three-line class rebind.

    `use_flash_attention` is forced False first: upstream would otherwise pass
    `attn_implementation="flash_attention_2"` (run.py:172), which ignores a 4-D mask and
    would make the bottleneck silently not exist (invariant 4).
    """
    from ._upstream import import_data
    from .model import DualTrackLTModel

    run = import_data().run
    if init_from is not None:
        custom_args.model_name_or_path = str(init_from)
    custom_args.use_flash_attention = False
    model, tokenizer, thinking_id = run.load_model_and_tokenizer(custom_args, training_args)
    return DualTrackLTModel.promote(model, cfg.mask, cfg.loss), tokenizer, thinking_id


def build_manifest(
    cfg: RunConfig,
    model: Any,
    custom_args: Any,
    thinking_id: int,
    stage_mode: str,
    data_path: Path,
    n_examples: int,
    n_failures: int,
) -> Manifest:
    import torch
    import transformers

    from ._upstream import assert_upstream_clean
    from .alignment import sha256_file
    from .attention_backend import probe_four_d_mask

    report = probe_four_d_mask()
    commit = assert_upstream_clean()
    return Manifest(
        stage=cfg.stage,
        stage_mode=stage_mode,
        mask_on=cfg.mask.mask_on,
        bottleneck_mode=cfg.mask.mode.value,
        cot_w=cfg.loss.cot_w,
        ans_w=cfg.loss.ans_w,
        latent_w=cfg.loss.latent_w,
        thinking_strategy=str(getattr(custom_args, "thinking_strategy", "")),
        seed=int(getattr(custom_args, "seed", 42)),
        tokenizer_name=str(custom_args.model_name_or_path),
        thinking_token_id=int(thinking_id),
        vocab_size=int(model.embedding.weight.shape[0]),
        data_file=str(data_path),
        data_file_sha256=sha256_file(data_path) if data_path.is_file() else "",
        n_examples=n_examples,
        n_track_derivation_failures=n_failures,
        four_d_mask_path=str(report.path),
        four_d_mask_detail=report.detail,
        attn_impl=model.attn_impl,
        transformers_version=str(transformers.__version__),
        torch_version=str(torch.__version__),
        upstream_commit=commit,
        upstream_clean=True,
    )


def save_stage(model: Any, tokenizer: Any, out_dir: Path, manifest: Manifest) -> None:
    """run.py:1124-1130 saves `base_causallm` only, because Llama-3.2 ties `lm_head.weight`
    to `embed_tokens.weight` and `save_pretrained` on the wrapper raises on shared tensors.
    That also silently drops `thinking_mlp`, so an MLP run is refused rather than truncated."""
    if getattr(model, "thinking_mlp", None) is not None:
        raise ValueError(
            "thinking_use_mlp is on, and saving base_causallm alone would drop the MLP "
            "between stages.  Save it explicitly before enabling this."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.base_causallm.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    manifest.write(out_dir)


def _check_training_arguments(training_args: Any) -> None:
    """Two upstream landmines that would silently degrade the bottleneck (README 3 and 10)."""
    if getattr(training_args, "gradient_checkpointing", False):
        raise ValueError(
            "gradient_checkpointing forces use_cache=False, which breaks upstream's "
            "cross-segment KV threading (model.py:480/490) and desynchronises the bias "
            "kv_len.  Turn it off."
        )
    if training_args.remove_unused_columns:
        raise ValueError("remove_unused_columns must be False or the Trainer drops track_ids")


def _build_training_data(
    cfg: RunConfig,
    upstream_config: Any,
    stage_mode: str,
    tokenizer: Any,
    thinking_id: int,
    model: Any,
) -> Tuple[Any, List[Tuple[int, str]], Path]:
    """upstream `get_cot_latent_dataset` + the clamped strategy, plus `track_ids`."""
    from .alignment import verify_manifest_if_present
    from .data import build_dualtrack_dataset
    from .insertion import build_clamped_strategy
    from .prepare_data import read_dualtrack_jsonl

    data_path = cfg.data.dualtrack_path("train")
    print(f"  data alignment: {verify_manifest_if_present(data_path)}")
    rows = read_dualtrack_jsonl(data_path, limit=cfg.train.limit)
    base_dataset, _ = upstream_configs_for_eval(rows, cfg)
    upstream_config.current_stage_mode = stage_mode
    strategy = (
        None
        if stage_mode == "common"
        else build_clamped_strategy(upstream_config, tokenizer, thinking_id, model.base_causallm)
    )
    examples, failures = build_dualtrack_dataset(
        stage_mode, base_dataset, upstream_config, strategy, tokenizer, thinking_id, shuffle=True
    )
    return examples, failures, data_path


def _make_trainer(
    model: Any,
    training_args: Any,
    examples: Any,
    tokenizer: Any,
    thinking_id: int,
    stage_mode: str,
) -> Any:
    """upstream `LTTuningTrainer` + `MyCollator`, both subclassed one method deep."""
    from .data import dualtrack_collator_class

    collator = dualtrack_collator_class()(
        tokenizer=tokenizer,
        thinking_id=thinking_id if stage_mode != "common" else None,
        label_pad_token_id=-100,
    )
    return build_trainer_class()(
        model=model, args=training_args, train_dataset=examples, data_collator=collator
    )


def run(cfg: RunConfig, config_path: Path, init_from: Optional[Path], out_dir: Path) -> None:
    from ._upstream import import_core

    cfg.validate()
    custom_args, training_args, _raw = parse_dualtrack_yaml(config_path)
    _check_training_arguments(training_args)
    import_core().utils.set_seed(training_args.seed)  # utils.py:21-27

    stage_manager, upstream_config = _stage_manager(custom_args, training_args)
    stage_manager.apply(cfg.stage)
    model, tokenizer, thinking_id = _load_model(custom_args, training_args, cfg, init_from)
    model.update_stage_config(
        stage_mode=stage_manager.current_mode,
        fusion_alpha=upstream_config.fusion_alpha,
        fusion_top_p=upstream_config.fusion_top_p,
        fusion_temperature=upstream_config.fusion_temperature,
    )

    examples, failures, data_path = _build_training_data(
        cfg, upstream_config, stage_manager.current_mode, tokenizer, thinking_id, model
    )
    print(
        f"stage {cfg.stage} ({stage_manager.current_mode}) mask_on={cfg.mask.mask_on} "
        f"bottleneck_mode={cfg.mask.mode.value}: {len(examples)} examples, "
        f"{len(failures)} track-derivation failures"
    )

    training_args.output_dir = str(out_dir)
    _make_trainer(
        model, training_args, examples, tokenizer, thinking_id, stage_manager.current_mode
    ).train()
    manifest = build_manifest(
        cfg,
        model,
        custom_args,
        thinking_id,
        stage_manager.current_mode,
        data_path,
        len(examples),
        len(failures),
    )
    save_stage(model, tokenizer, out_dir, manifest)
    print(f"saved stage {cfg.stage} -> {out_dir}")
    print(json.dumps(manifest.to_json(), indent=2))


# --- selftests -----------------------------------------------------------------------


def _selftest_no_attack_surface() -> None:
    """This round is clean.  Fail loudly if attack vocabulary appears anywhere in the layer.

    Exempting a whole file (this one used to exempt itself) would leave the largest module
    unscanned, so the exemption is per-line and has to be written down at the line.
    """
    banned = BANNED_TERMS.split()
    offenders: List[str] = []
    sources = (
        sorted(PACKAGE_DIR.glob("*.py"))
        + sorted(PACKAGE_DIR.glob("*.sh"))
        + sorted(PACKAGE_DIR.glob("configs/*.yaml"))
    )
    for path in sources:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if ATTACK_SCAN_SKIP in line:
                continue
            lowered = line.lower()
            offenders += [f"{path.name}:{line_no}:{word}" for word in banned if word in lowered]
    assert not offenders, f"attack vocabulary found in a clean-round folder: {offenders}"
    print(f"  no attack vocabulary in {len(sources)} files ({len(banned)} terms checked): OK")


def _selftest_yaml_strictness() -> Optional[str]:
    """A typo in the YAML must raise, not vanish through `allow_extra_keys=True`."""
    import tempfile

    from ._upstream import UpstreamUnavailable, import_data

    try:
        run = import_data().run
    except UpstreamUnavailable as exc:
        return str(exc)
    known = (
        set(run.CustomizedArguments.__dataclass_fields__)
        | set(run.LTTuningTrainingArguments.__dataclass_fields__)
        | DUALTRACK_YAML_KEYS
    )
    assert "bottleneck_mode" in known and "remove_unused_columns" in known
    with tempfile.TemporaryDirectory(prefix="dt_train_") as tmp:
        path = Path(tmp) / "cfg.yaml"
        path.write_text("bottleneck_mode: strict\nbottelneck_mode: native\n", encoding="utf-8")
        try:
            parse_dualtrack_yaml(path)
        except ValueError as exc:
            assert "bottelneck_mode" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError("an unknown YAML key must raise, not be silently dropped")
    print("  unknown YAML keys are refused rather than dropped (upstream drops them): OK")
    return None


def _selftest_mode_is_carried_by_the_model() -> None:
    """There is no call site at which `mode` can be forgotten: it is a model attribute.

    Last round `compute_loss` had to pass `mode=` explicitly and a strict run silently
    trained under NATIVE.  This asserts the structural fix -- `forward` with no `mode=`
    argument still uses the configured mode, and the outputs say which.
    """
    import torch

    from .model import build_selftest_model, selftest_batch

    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    seen = {}
    for name in ("native", "strict"):
        model = build_selftest_model(
            vocab, thinking_id, 4, bottleneck=MaskConfig(mask_on=True, bottleneck_mode=name)
        )
        with torch.no_grad():
            out = model(ids, tracks, attention_mask=torch.ones_like(ids))
        assert out.bottleneck_mode == name, (out.bottleneck_mode, name)
        assert out.mask_on is True
        seen[name] = out.logits
    assert not torch.equal(
        seen["native"], seen["strict"]
    ), "the configured mode did not reach the forward"
    print("  bottleneck mode travels on the model, so no call site can drop it: OK")


def _selftest_manifest_is_complete() -> None:
    from .manifest import REQUIRED_FIELDS

    fields = set(Manifest.__dataclass_fields__)
    for required in REQUIRED_FIELDS:
        assert required in fields, required
    cfg = default_run_config(stage=2, bottleneck_mode="strict")
    assert cfg.mask.mode.value == "strict"
    suffixes = {
        checkpoint_suffix(cfg.mask),
        checkpoint_suffix(default_run_config(2).mask),
        checkpoint_suffix(default_run_config(0).mask),
    }
    assert len(suffixes) == 3, f"two run kinds would share a checkpoint directory: {suffixes}"
    print(
        f"  manifest fields ({len(fields)}) cover every required key; ckpt suffixes "
        f"{sorted(suffixes)}: OK"
    )


def default_run_config(stage: int = 1, bottleneck_mode: Any = DEFAULT_BOTTLENECK_MODE) -> RunConfig:
    from .config import default_config

    return default_config(stage=stage, bottleneck_mode=bottleneck_mode)


def selftest() -> None:
    """The clean-round scan, the mode carrier and the manifest run anywhere; the YAML
    strictness wrapper needs upstream's run.py (transformers >= 4.41)."""
    _selftest_no_attack_surface()
    _selftest_manifest_is_complete()
    _selftest_mode_is_carried_by_the_model()
    unavailable = _selftest_yaml_strictness()
    if unavailable is not None:
        print(f"  YAML strictness check UNAVAILABLE: {unavailable}")
        print("train.py selftest PASSED (importable half); upstream half SKIPPED (environment)")
        return
    print("train.py selftest PASSED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean dual-track training (LT-Tuning)")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--init", type=Path, default=None, help="checkpoint from the previous stage"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap the number of training rows")
    parser.add_argument("--no-mask", action="store_true", help="the mask-OFF control twin")
    parser.add_argument("--stage0-mask", action="store_true", help="ablation only; see README")
    parser.add_argument("--cot-w", type=float, default=None)
    parser.add_argument("--ans-w", type=float, default=None)
    parser.add_argument("--latent-w", type=float, default=None)
    add_bottleneck_mode_argument(parser)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    cfg = dualtrack_from_raw(load_yaml(args.config), args.stage)
    mask_on = (not args.no_mask) and (args.stage > 0 or args.stage0_mask)
    # With the mask off no query row is blocked, so the mode names a distinction the run
    # cannot express; MaskConfig.validate() refuses that pair rather than record a mode that
    # did nothing.  Keep the loaded default there instead of the requested one.
    mask = replace(
        cfg.mask,
        mask_on=mask_on,
        stage0_mask=args.stage0_mask,
        bottleneck_mode=(
            resolve_bottleneck_mode(args.bottleneck_mode) if mask_on else DEFAULT_BOTTLENECK_MODE
        ),
    )
    loss = LossWeights(
        cot_w=cfg.loss.cot_w if args.cot_w is None else args.cot_w,
        ans_w=cfg.loss.ans_w if args.ans_w is None else args.ans_w,
        latent_w=cfg.loss.latent_w if args.latent_w is None else args.latent_w,
    )
    cfg = replace(cfg, mask=mask, loss=loss, train=replace(cfg.train, limit=args.limit))
    out_dir = args.out or (cfg.train.out_dir / f"stage{args.stage}{checkpoint_suffix(mask)}")
    run(cfg, args.config, args.init, out_dir)


if __name__ == "__main__":
    main()
