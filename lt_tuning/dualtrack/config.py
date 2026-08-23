"""Frozen configuration for the dual-track patch layer.

Slimmed from the previous round: `StageConfig` is gone because upstream's `StageManager`
(utils.py:130-239) already does that job and is imported instead, and every path is now
relative to this package rather than to a repos/ checkout.  No absolute paths anywhere.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .mask import DEFAULT_BOTTLENECK_MODE, BottleneckMode, resolve_bottleneck_mode
from .tracks import DELIM_TEXT

PACKAGE_DIR = Path(__file__).resolve().parent
PLATFORM_DIR = PACKAGE_DIR.parent
DEFAULT_UPSTREAM_ROOT = PLATFORM_DIR / "upstream"

STAGE_MODES: Tuple[str, ...] = ("common", "hidden_state", "soft_fusion")
INSERTION_STRATEGIES: Tuple[str, ...] = ("arithmetic", "random", "confidence")

__all__ = [
    "DELIM_TEXT",
    "DataConfig",
    "GenConfig",
    "LossWeights",
    "MaskConfig",
    "RunConfig",
    "TrainConfig",
    "add_bottleneck_mode_argument",
    "default_config",
    "load_yaml",
]


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name}={raw!r} is not an integer") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name}={raw!r} is not a float") from exc


@dataclass(frozen=True)
class DataConfig:
    upstream_root: Path = DEFAULT_UPSTREAM_ROOT
    data_dir: Path = PLATFORM_DIR / "data"
    max_seq_len: int = 1024
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    thinking_token: str = "<thinking>"

    def raw_split_path(self, split: str) -> Path:
        """Upstream ships `data/gsm8k/{train,test}_socratic.jsonl` inside the clone."""
        if split not in ("train", "test"):
            raise ValueError(f"split must be train or test; got {split!r}")
        return self.upstream_root / "data" / "gsm8k" / f"{split}_socratic.jsonl"

    def dualtrack_path(self, split: str) -> Path:
        return self.data_dir / f"dualtrack_{split}.jsonl"


@dataclass(frozen=True)
class MaskConfig:
    """The bottleneck itself.  `mask_on=False` is the control twin, never the default."""

    mask_on: bool = True
    stage0_mask: bool = False
    bottleneck_mode: BottleneckMode = DEFAULT_BOTTLENECK_MODE

    def validate(self) -> None:
        mode = resolve_bottleneck_mode(self.bottleneck_mode)
        if mode is not BottleneckMode.NATIVE and not self.mask_on:
            raise ValueError(
                f"bottleneck_mode={mode.value!r} with mask_on=False is meaningless: with the "
                "mask off no query row is blocked at all, so the mode names a distinction "
                "that does not exist in that run."
            )

    @property
    def mode(self) -> BottleneckMode:
        return resolve_bottleneck_mode(self.bottleneck_mode)


@dataclass(frozen=True)
class LossWeights:
    """`cot_w * CE_cot + ans_w * CE_ans + latent_w * L_latent`.

    At `(1.0, 1.0, 0.0)` the total is numerically identical to upstream's own
    `CrossEntropyLoss` (model.py:546-550), because CE_cot and CE_ans partition the
    supervised positions over one shared normaliser.  `L_latent` is upstream's loss line
    evaluated on latent-only labels, so it OVERLAPS CE_cot by construction; `latent_w` is
    an additional term, not a re-weighting of a disjoint slice.
    """

    cot_w: float = 1.0
    ans_w: float = 1.0
    latent_w: float = 0.0

    def validate(self) -> None:
        for name in ("cot_w", "ans_w", "latent_w"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)}")


@dataclass(frozen=True)
class GenConfig:
    max_reasoning_tokens: int = 300
    max_answer_tokens: int = 24
    suppress_thinking_in_answer: bool = True


@dataclass(frozen=True)
class TrainConfig:
    out_dir: Path = PLATFORM_DIR / "ckpt"
    init_from: Optional[Path] = None
    limit: Optional[int] = None


@dataclass(frozen=True)
class RunConfig:
    """The dual-track knobs only.  Everything upstream already parses (stage schedule,
    optimiser, insertion probabilities) stays in the YAML and reaches upstream's own
    `CustomizedArguments` / `LTTuningTrainingArguments` untouched."""

    data: DataConfig = DataConfig()
    mask: MaskConfig = MaskConfig()
    loss: LossWeights = LossWeights()
    gen: GenConfig = GenConfig()
    train: TrainConfig = TrainConfig()
    stage: int = 1

    def validate(self) -> None:
        self.loss.validate()
        self.mask.validate()
        if self.stage < 0:
            raise ValueError(f"stage must be >= 0; got {self.stage}")
        if self.stage == 0 and self.mask.mask_on and not self.mask.stage0_mask:
            raise ValueError(
                "stage 0 has no latents, so blocking the CoT would train 'guess from the "
                "prompt'.  Pass --stage0-mask only for the deliberate ablation."
            )


def load_yaml(path: Path) -> Dict[str, Any]:
    """Explicit failure on a missing or malformed config; never a silent default."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the config file") from exc
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} must contain a mapping at the top level")
    return loaded


def dualtrack_from_raw(raw: Dict[str, Any], stage: int) -> RunConfig:
    """Read only the dual-track keys out of the shared YAML.

    Stage 0 has no latents, so it is mask-off by construction; the mode still comes from the
    file or the environment so the value recorded in the manifest is never invented.
    """
    data = DataConfig(
        upstream_root=env_path(
            "DT_UPSTREAM_ROOT", Path(raw.get("upstream_root", DEFAULT_UPSTREAM_ROOT))
        ),
        data_dir=env_path("DT_DATA_DIR", PLATFORM_DIR / str(raw.get("data_dir", "data"))),
        max_seq_len=int(raw.get("max_seq_len", 1024)),
        model_name_or_path=str(raw.get("model_name_or_path", "meta-llama/Llama-3.2-1B")),
        thinking_token=str(raw.get("thinking_token", "<thinking>")),
    )
    mask = MaskConfig(
        mask_on=stage > 0,
        bottleneck_mode=resolve_bottleneck_mode(
            os.environ.get(
                "DT_BOTTLENECK_MODE", raw.get("bottleneck_mode", DEFAULT_BOTTLENECK_MODE)
            )
        ),
    )
    loss = LossWeights(
        cot_w=env_float("DT_COT_W", float(raw.get("cot_w", 1.0))),
        ans_w=env_float("DT_ANS_W", float(raw.get("ans_w", 1.0))),
        latent_w=env_float("DT_LATENT_W", float(raw.get("latent_w", 0.0))),
    )
    gen = GenConfig(
        max_reasoning_tokens=int(raw.get("max_reasoning_tokens", 300)),
        max_answer_tokens=int(raw.get("max_answer_tokens", 24)),
    )
    train = TrainConfig(
        out_dir=env_path("DT_CKPT_DIR", PLATFORM_DIR / str(raw.get("out_dir", "ckpt")))
    )
    return RunConfig(data=data, mask=mask, loss=loss, gen=gen, train=train, stage=stage)


def default_config(stage: int = 1, bottleneck_mode: Any = DEFAULT_BOTTLENECK_MODE) -> RunConfig:
    """Defaults without touching the filesystem -- used by every selftest."""
    return replace(
        RunConfig(),
        stage=stage,
        mask=MaskConfig(
            mask_on=stage > 0, bottleneck_mode=resolve_bottleneck_mode(bottleneck_mode)
        ),
    )


def add_bottleneck_mode_argument(parser: argparse.ArgumentParser) -> None:
    """One registration shared by train / generate / verify, so the flag cannot drift.

    STRICT is not a better default.  It stops latent rows attending the explicit chain they
    exist to summarise, which is a change to LT-Tuning's semantics; it is here to decompose
    the leak, and every run that uses it says so in its manifest.
    """
    parser.add_argument(
        "--bottleneck-mode",
        "--bottleneck_mode",
        dest="bottleneck_mode",
        choices=tuple(m.value for m in BottleneckMode),
        default=DEFAULT_BOTTLENECK_MODE.value,
        help=(
            "native: block the direct edge visible-CoT -> delimiter/answer (LT-Tuning's own "
            "geometry, the default).  strict: additionally stop latent rows reading the "
            "visible CoT -- a diagnostic that changes the platform's semantics."
        ),
    )


def checkpoint_suffix(mask: MaskConfig) -> str:
    """Default checkpoint names carry the two things that make a run uninterpretable if
    lost: whether the mask ran at all, and which mode it ran in."""
    if not mask.mask_on:
        return "-nomask"
    return "" if mask.mode is DEFAULT_BOTTLENECK_MODE else f"-{mask.mode.value}"


def _selftest_stage_rules() -> None:
    cfg = default_config(stage=1)
    cfg.validate()
    assert cfg.mask.mask_on and cfg.loss.latent_w == 0.0
    stage0 = default_config(stage=0)
    stage0.validate()
    assert not stage0.mask.mask_on, "stage 0 must run mask-off by construction"
    try:
        replace(stage0, mask=MaskConfig(mask_on=True)).validate()
    except ValueError as exc:
        assert "stage 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stage 0 with the mask on must be refused unless --stage0-mask")
    replace(stage0, mask=MaskConfig(mask_on=True, stage0_mask=True)).validate()
    print("  stage rules: stage0 mask-off, ablation needs --stage0-mask: OK")


def _selftest_bottleneck_mode() -> None:
    assert default_config(stage=1).mask.mode is BottleneckMode.NATIVE, "NATIVE stays the default"
    strict = default_config(stage=1, bottleneck_mode="strict")
    strict.validate()
    assert strict.mask.mode is BottleneckMode.STRICT
    try:
        replace(strict, mask=replace(strict.mask, mask_on=False)).validate()
    except ValueError as exc:
        assert "meaningless" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict with the mask off must be refused, not silently accepted")
    parser = argparse.ArgumentParser()
    add_bottleneck_mode_argument(parser)
    assert parser.parse_args([]).bottleneck_mode == "native"
    assert parser.parse_args(["--bottleneck_mode", "strict"]).bottleneck_mode == "strict"
    assert parser.parse_args(["--bottleneck-mode", "strict"]).bottleneck_mode == "strict"
    suffixes = {
        checkpoint_suffix(default_config(1).mask),
        checkpoint_suffix(default_config(1, "strict").mask),
        checkpoint_suffix(default_config(0).mask),
    }
    assert len(suffixes) == 3, f"two run kinds would share a checkpoint directory: {suffixes}"
    print(
        f"  bottleneck mode: default={DEFAULT_BOTTLENECK_MODE.value}, flag accepts "
        f"{[m.value for m in BottleneckMode]}, ckpt suffixes {sorted(suffixes)}: OK"
    )


def _selftest_loss_weights() -> None:
    LossWeights().validate()
    LossWeights(latent_w=0.3).validate()
    try:
        LossWeights(cot_w=-1.0).validate()
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a negative weight must be refused")
    print("  loss weights: default latent_w=0.0 (== upstream's own loss), negatives refused: OK")


def _selftest_paths() -> None:
    data = DataConfig()
    assert data.upstream_root.name == "upstream" and data.upstream_root.parent == PLATFORM_DIR
    assert not str(data.data_dir).startswith("/tmp")
    for value in (data.upstream_root, data.data_dir, TrainConfig().out_dir):
        assert PLATFORM_DIR in value.parents or value == PLATFORM_DIR, value
    print(
        f"  every default path is under the platform folder; upstream data present: "
        f"{data.raw_split_path('train').is_file()}: OK"
    )


def selftest() -> None:
    """CPU-only, no torch, no transformers, no filesystem writes."""
    _selftest_stage_rules()
    _selftest_bottleneck_mode()
    _selftest_loss_weights()
    _selftest_paths()
    print("config.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-track config (LT-Tuning)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
