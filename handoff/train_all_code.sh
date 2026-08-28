#!/bin/bash
# ============================================================================
# 一条命令训完三个 code LRM ckpt。   bash train_all_code.sh
#   VERIFY=1 bash train_all_code.sh   → 排错用: 三个都跑 200 行小规模(正常流程不需要, 管线已验证过)
#   bash train_all_code.sh            → 正常用法: 三个都全量真训练, 出 3 个 ckpt
#   ONLY=colar,lt bash train_all_code.sh   → 只跑指定的(colar / lt / lsft, 逗号分隔)
#
# 为什么要串行不并行: 三个平台 pip 依赖互相冲突(transformers 4.45.2 / 4.55.4 / 4.51.1),
#   同一个环境同时装不了。本脚本每段开跑前重装自己那套 floor —— 串行跑就没问题。
# 中断安全: 每个平台独立, 跑挂一个不影响其它; 重跑本脚本会跳过已出产物的平台。
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$(pwd)/crux_retrain_work}"; mkdir -p "$WORK/run_logs"
export WORK
ONLY="${ONLY:-colar,lt,lsft}"
VERIFY="${VERIFY:-0}"
MODE=$([ "$VERIFY" = "1" ] && echo 验证VERIFY || echo 全量FULL)

echo "════════════════════════════════════════════════════════════"
echo "  三平台 code LRM 训练  [$MODE]   计划: $ONLY"
echo "  产物目录: $WORK     日志: $WORK/run_logs/"
echo "════════════════════════════════════════════════════════════"

declare -a OK=() FAIL=() SKIP=()

# $1=key  $2=脚本  $3=全量模式下的产物(存在则跳过)
run_one () {
  local key="$1" script="$2" artifact="$3"
  case ",$ONLY," in *",$key,"*) ;; *) echo "── 跳过 $key (不在 ONLY 里)"; return 0 ;; esac
  if [ "$VERIFY" != "1" ] && [ -e "$artifact" ]; then
    echo "── 跳过 $key: 产物已存在 ($artifact)"; SKIP+=("$key"); return 0
  fi
  echo ""
  echo "┏━━━ [$key] 开跑 ($(date '+%H:%M:%S')) ━━━━━━━━━━━━━━━━━━━━━━━━"
  local t0=$SECONDS
  if VERIFY="$VERIFY" WORK="$WORK" bash "$HERE/$script"; then
    echo "┗━━━ [$key] ✅ 成功  用时 $(( (SECONDS-t0)/60 )) 分"
    OK+=("$key")
  else
    local rc=$?
    echo "┗━━━ [$key] ❌ 失败 rc=$rc  用时 $(( (SECONDS-t0)/60 )) 分  → 看 $WORK/run_logs/"
    FAIL+=("$key")
  fi
}

# 顺序: CoLaR(最快, 先验证环境能用) → LT(中) → LSFT(6步最久)
run_one colar train_colar_code.sh "$WORK/colar_code_cruxreal.ckpt"
run_one lt    train_lt_code.sh    "$WORK/lt_code_out/qwen_code"
run_one lsft  train_lsft_code.sh  "$WORK/Latent-SFT/output/stage2_results/code"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  汇总 [$MODE]   成功: ${OK[*]:-无}   跳过: ${SKIP[*]:-无}   失败: ${FAIL[*]:-无}"
if [ "$VERIFY" != "1" ]; then
  echo "  产物:"
  [ -e "$WORK/colar_code_cruxreal.ckpt" ]            && echo "    CoLaR : $WORK/colar_code_cruxreal.ckpt"
  [ -e "$WORK/lt_code_out/qwen_code" ]               && echo "    LT    : $WORK/lt_code_out/qwen_code/"
  ls -d "$WORK"/Latent-SFT/output/stage2_results/code/checkpoint-*/hf 2>/dev/null | tail -1 | sed 's/^/    LSFT  : /'
fi
echo "════════════════════════════════════════════════════════════"
[ ${#FAIL[@]} -eq 0 ]
