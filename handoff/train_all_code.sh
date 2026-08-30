#!/bin/bash
# ============================================================================
# 训完三个 code LRM ckpt。
#
#   bash train_all_code.sh                         单卡串行(默认)
#   PARALLEL=1 GPUS=0,1,2 bash train_all_code.sh   三卡并行, 一个任务一张卡  ← 多卡机器推荐
#   ONLY=colar,lt bash train_all_code.sh           只跑指定平台
#   CN=1 ...                                       国内网络: 开 HF/pip 镜像; GitHub 直连官方
#   VERIFY=1 ...                                   排错用(200 行小规模), 正常流程不需要
#
# 没有写 DDP —— 三个都是单卡任务。多卡机器上正确的用法是**一卡一任务并行**,
# 不是把一个任务 DDP 到多卡(卡上有别的任务时 DDP 通常更慢, 而且这个数据集很小,
# 通信开销占比高, DDP 收益有限)。
#
# 并行为什么需要 venv: 三个平台的 transformers 版本互斥(4.45.2 / 4.55.4 / 4.51.1),
# 共用一个 Python 环境会互相覆盖。PARALLEL=1 会自动开 VENV=1, 每个平台一个独立
# venv(继承系统 site-packages, 只隔离冲突的包)。
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/config.sh"   # 所有可调参数集中在 config.sh
WORK="${WORK:-$(pwd)/crux_retrain_work}"; mkdir -p "$WORK/run_logs"; export WORK
ONLY="${ONLY:-colar,lt,lsft}"
VERIFY="${VERIFY:-0}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-}"
[ "$PARALLEL" = "1" ] && export VENV="${VENV:-1}"
. "$HERE/_common.sh"
MODE=$([ "$VERIFY" = "1" ] && echo 验证VERIFY || echo 全量FULL)

echo "════════════════════════════════════════════════════════════"
echo "  三平台 code LRM 训练  [$MODE]  $([ "$PARALLEL" = 1 ] && echo "并行 GPUS=${GPUS:-未指定}" || echo 串行)"
echo "  计划: $ONLY     产物: $WORK     日志: $WORK/run_logs/"
print_env
echo "════════════════════════════════════════════════════════════"

# 数据仓预先克隆一次 —— 并行时三个进程同时 clone 同一目录会打架
[ -f "$WORK/lrm/code_real_ladder/manifest.json" ] || {
  echo "预拉数据仓 ..."
  rm -rf "$WORK/lrm"
  gh_clone "${DATA_REPO:-https://github.com/ruijiezh67/LRM_colab_tasks.git}" "$WORK/lrm"
}

want () { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
# 全量模式下产物已存在则跳过
done_already () {
  [ "$VERIFY" = "1" ] && return 1
  case "$1" in
    colar) [ -e "$WORK/colar_code_cruxreal.ckpt" ] ;;
    lt)    [ -e "$WORK/lt_code_out/qwen_code" ] ;;
    lsft)  [ -d "$WORK/Latent-SFT/output/stage2_results/code" ] ;;
  esac
}
script_of () { case "$1" in colar) echo train_colar_code.sh ;; lt) echo train_lt_code.sh ;; lsft) echo train_lsft_code.sh ;; esac; }

PLATS=(); for k in colar lt lsft; do want "$k" && PLATS+=("$k"); done
declare -a OK=() FAIL=() SKIP=()

if [ "$PARALLEL" = "1" ]; then
  # ---------------- 并行: 一个平台一张卡 ----------------
  IFS=',' read -ra GARR <<< "$GPUS"
  declare -a PIDS=() RUN=()
  i=0
  for k in "${PLATS[@]}"; do
    if done_already "$k"; then echo "── 跳过 $k: 产物已存在"; SKIP+=("$k"); continue; fi
    g="${GARR[$i]:-}"; i=$((i+1))
    log="$WORK/run_logs/${k}.stdout.log"
    echo "┏━ [$k] 后台启动  GPU=${g:-默认}  → $log"
    GPU="$g" VENV=1 VERIFY="$VERIFY" WORK="$WORK" \
      bash "$HERE/$(script_of "$k")" > "$log" 2>&1 &
    PIDS+=($!); RUN+=("$k")
  done
  echo ""
  echo "已并行启动 ${#RUN[@]} 个任务, 等待全部结束 (实时看进度: tail -f $WORK/run_logs/<平台>.stdout.log)"
  for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then echo "  [${RUN[$j]}] ✅ 成功"; OK+=("${RUN[$j]}")
    else                        echo "  [${RUN[$j]}] ❌ 失败 → ${WORK}/run_logs/${RUN[$j]}.stdout.log"; FAIL+=("${RUN[$j]}"); fi
  done
else
  # ---------------- 串行 ----------------
  for k in "${PLATS[@]}"; do
    if done_already "$k"; then echo "── 跳过 $k: 产物已存在"; SKIP+=("$k"); continue; fi
    echo ""; echo "┏━━━ [$k] 开跑 ($(date '+%H:%M:%S')) ━━━━━━━━━━━━━━━━━━━━━━━━"
    t0=$SECONDS
    if VERIFY="$VERIFY" WORK="$WORK" bash "$HERE/$(script_of "$k")"; then
      echo "┗━━━ [$k] ✅ 成功  用时 $(( (SECONDS-t0)/60 )) 分"; OK+=("$k")
    else
      echo "┗━━━ [$k] ❌ 失败 rc=$?  用时 $(( (SECONDS-t0)/60 )) 分 → $WORK/run_logs/"; FAIL+=("$k")
    fi
  done
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  汇总 [$MODE]  成功: ${OK[*]:-无}   跳过: ${SKIP[*]:-无}   失败: ${FAIL[*]:-无}"
if [ "$VERIFY" != "1" ]; then
  echo "  产物:"
  [ -e "$WORK/colar_code_cruxreal.ckpt" ] && echo "    CoLaR : $WORK/colar_code_cruxreal.ckpt"
  [ -e "$WORK/lt_code_out/qwen_code" ]    && echo "    LT    : $WORK/lt_code_out/qwen_code/"
  ls -d "$WORK"/Latent-SFT/output/stage2_results/code/checkpoint-*/hf 2>/dev/null | tail -1 | sed 's/^/    LSFT  : /'
fi
echo "════════════════════════════════════════════════════════════"
[ ${#FAIL[@]} -eq 0 ]
