r"""The ONLY import boundary between this layer and the vendored upstream.

Every upstream symbol the layer uses is re-exported here with its
``file:line`` provenance, so ``rg upstream_api`` shows the whole contract in one
place and nothing else in ``dualtrack/`` touches ``sys.path``.

The upstream root is resolved from ``__file__``; no absolute path is baked in.
The vendored ``sglang_latent_reasoning_pkg/`` is NOT on the training path and is
never imported. Known constraint of vendoring this repo: upstream exposes a
top-level package named ``src``, so an unrelated ``src`` package earlier on
``sys.path`` would shadow it -- see the README.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

UPSTREAM_ROOT: Path = Path(__file__).resolve().parent.parent / "upstream"
STAGE1_ENCODER_SCRIPT: Path = UPSTREAM_ROOT / "script" / "run_distill_stage1_encoder.py"
LATENT_SOFT_LABEL_SCRIPT: Path = UPSTREAM_ROOT / "generate_latent_soft_label_hf_batch.py"
DEEPSPEED_ZERO1_CONFIG: Path = UPSTREAM_ROOT / "config_zero1.json"


PYCACHE_DIR: Path = UPSTREAM_ROOT.parent / "data" / "pycache"


def _redirect_bytecode_cache() -> None:
    """Keep ``git -C upstream status --porcelain`` empty.

    Importing upstream would otherwise write ``upstream/src/**/__pycache__``,
    which upstream's own .gitignore does not cover -- that alone makes the tree
    dirty and fails the review. ``sys.pycache_prefix`` is process-global, so it
    is only set when the caller has not already chosen one.
    """
    if getattr(sys, "pycache_prefix", None) is None:
        PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
        sys.pycache_prefix = str(PYCACHE_DIR)


def ensure_upstream_on_path() -> Path:
    """Put the vendored clone on ``sys.path`` exactly once."""
    if not (UPSTREAM_ROOT / "src").is_dir():
        raise FileNotFoundError(
            f"no vendored upstream at {UPSTREAM_ROOT}. This layer patches a clone in place; "
            "restore it with `git clone <latent-sft> upstream` inside the platform folder."
        )
    _redirect_bytecode_cache()
    resolved = str(UPSTREAM_ROOT)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        logger.info("Resolved vendored upstream: %s", resolved)
    return UPSTREAM_ROOT


ensure_upstream_on_path()

# --- upstream/src/modeling/modeling_stage2.py --------------------------------
from src.modeling.modeling_stage2 import (  # noqa: E402
    LatentOutput,  # :17-20   returned unchanged by our forward
    LatentSFTStage2SoftEmbedding,  # :129   subclassed by DualTrackLatentSFT
    kd_kl_loss_topk,  # :49-73   called INSIDE upstream's forward; imported only for the selftest
    softmax_over_embedding_topk,  # :90-126 used by the proxy-teacher fallback
    weighted_embedding_from_topk,  # :76-88  used by the proxy-teacher fallback
)

# --- upstream/src/stage2/data.py ---------------------------------------------
from src.stage2.data import (  # noqa: E402
    DataCollatorForDynamicPadding,  # :263-293 subclassed; dynamic_padding inherited
    Stage2Dataset,  # :58   subclassed; _load_all_chunks and both gumbel helpers inherited
    read_jsonl as upstream_read_jsonl,  # :13-38 called by the inherited __init__
)

# --- upstream/src/stage2/arguments.py ----------------------------------------
from src.stage2.arguments import (  # noqa: E402
    DataArguments as UpstreamDataArguments,  # :32-49
    ModelArguments as UpstreamModelArguments,  # :6-29
    Stage2TrainingArguments as UpstreamTrainingArguments,  # :52-68
)

# --- upstream/src/stage2/trainer.py ------------------------------------------
from src.stage2.trainer import Stage2Trainer  # noqa: E402  :13  used UNCHANGED, not subclassed

def selftest() -> None:
    """Assert the boundary contract: paths resolve and the tree cannot be dirtied."""
    import subprocess

    for path in (UPSTREAM_ROOT, STAGE1_ENCODER_SCRIPT, LATENT_SOFT_LABEL_SCRIPT, DEEPSPEED_ZERO1_CONFIG):
        assert path.exists(), f"vendored upstream is incomplete: {path} is missing"
    missing = [name for name in __all__ if name not in globals()]
    assert not missing, f"upstream_api declares symbols it does not export: {missing}"
    prefix = Path(getattr(sys, "pycache_prefix", "") or PYCACHE_DIR).resolve()
    assert UPSTREAM_ROOT.resolve() not in prefix.parents and prefix != UPSTREAM_ROOT.resolve(), (
        f"bytecode would be written under {UPSTREAM_ROOT}; importing upstream must never dirty it"
    )
    status = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert status == "", f"upstream is DIRTY after import:\n{status}"
    print(
        f"[upstream_api] OK -- {len(__all__)} symbols re-exported from {UPSTREAM_ROOT.name}/, "
        "bytecode redirected outside the clone, and `git status --porcelain` is empty after import."
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="upstream import boundary")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("upstream_api.py is a library; run it with --selftest")
    selftest()


__all__ = [
    "UPSTREAM_ROOT",
    "STAGE1_ENCODER_SCRIPT",
    "LATENT_SOFT_LABEL_SCRIPT",
    "DEEPSPEED_ZERO1_CONFIG",
    "ensure_upstream_on_path",
    "LatentOutput",
    "LatentSFTStage2SoftEmbedding",
    "kd_kl_loss_topk",
    "softmax_over_embedding_topk",
    "weighted_embedding_from_topk",
    "DataCollatorForDynamicPadding",
    "Stage2Dataset",
    "upstream_read_jsonl",
    "UpstreamDataArguments",
    "UpstreamModelArguments",
    "UpstreamTrainingArguments",
    "Stage2Trainer",
]


if __name__ == "__main__":
    main()
