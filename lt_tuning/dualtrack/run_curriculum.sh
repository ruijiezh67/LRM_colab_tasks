#!/usr/bin/env bash
# Three curriculum stages, ONE PROCESS EACH, chained by --init.
#
# Upstream drives the stages from StageUpdateCallback (run.py:243-506), which swaps the
# dataset under a live DataLoader and performs optimiser surgery.  That callback regenerates
# datasets without track_ids, so it is not used here; stage n starts from stage n-1's
# checkpoint instead.  Stage 0 has no latents and therefore runs mask-off by construction.
#
# Usage:  ./run_curriculum.sh [native|strict] [extra train.py args...]
set -euo pipefail

MODE="${1:-native}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="$(dirname "$HERE")"
PY="${PYTHON:-python3}"
CKPT="${DT_CKPT_DIR:-$PLATFORM/ckpt}"

cd "$PLATFORM"

case "$MODE" in
  native) SUFFIX="" ;;
  strict) SUFFIX="-strict" ;;
  *) echo "bottleneck mode must be native or strict; got '$MODE'" >&2; exit 2 ;;
esac

echo "=== stage 0 (common, no latents, mask off by construction) ==="
"$PY" -m dualtrack.train --stage 0 "$@"

echo "=== stage 1 (hidden_state, mask on, $MODE) ==="
"$PY" -m dualtrack.train --stage 1 --bottleneck_mode "$MODE" \
  --init "$CKPT/stage0-nomask" "$@"

echo "=== stage 2 (soft_fusion, mask on, $MODE) ==="
"$PY" -m dualtrack.train --stage 2 --bottleneck_mode "$MODE" \
  --init "$CKPT/stage1$SUFFIX" "$@"

echo "=== verify stage 2 ==="
"$PY" -m dualtrack.verify --ckpt "$CKPT/stage2$SUFFIX" --bottleneck_mode "$MODE"

echo "A masked run reported without its mask-off twin is not a result.  Run:"
echo "  $PY -m dualtrack.train --stage 1 --no-mask --init $CKPT/stage0-nomask"
echo "  $PY -m dualtrack.train --stage 2 --no-mask --init $CKPT/stage1-nomask"
