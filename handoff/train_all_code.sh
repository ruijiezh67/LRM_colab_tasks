#!/bin/bash
# ── 本包在 GitHub 上的位置 ──────────────────────────────────────────
#   交接包:   https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/handoff
#   上游代码: https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/upstream
#   上游数据: https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/upstream_data
#   训练数据: https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/code_real_ladder
#   完整说明: handoff/README.md
# ────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────
# 起飞前自检 —— 把所有"会让三个平台同时挂掉"的前置条件一次查完。
# 每一项失败都直接给出修复命令; 交接方不需要读脚本源码来排查。
# 跳过: PREFLIGHT=0
# ─────────────────────────────────────────────────────────────────────
if [ "${PREFLIGHT:-1}" = "1" ]; then
  pf_fail=0
  pf () { if [ "$1" = ok ]; then printf "  [ok]   %s
" "$2"; else printf "  [FAIL] %s
         -> %s
" "$2" "$3"; pf_fail=1; fi; }
  echo "起飞前自检:"

  # 1. 解释器
  [ -n "${PY:-}" ] && pf ok "python: $PY" || pf x "找不到 python/python3" "用 PY=/你的/python bash train_all_code.sh 指定"

  # 2. venv (PARALLEL=1 会自动 VENV=1; 缺 python3-venv 会三个平台同时挂)
  if [ "${VENV:-0}" = "1" ]; then
    if "$PY" -m venv --help >/dev/null 2>&1; then pf ok "venv 可用"
    else pf x "python venv 模块缺失(并行模式必需)" "apt install python3-venv  或  PARALLEL=0 串行跑"; fi
  fi

  # 3. GPU
  if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    nwant=$(echo "$ONLY" | tr ',' '
' | grep -c .)
    if [ "$PARALLEL" = "1" ] && [ "$ngpu" -lt "$nwant" ]; then
      pf x "并行需要 $nwant 张卡, 只看到 $ngpu 张" "GPUS=0,1,2 指定卡号, 或 PARALLEL=0 串行跑"
    else pf ok "GPU: 可见 $ngpu 张"; fi
  else pf x "没有 nvidia-smi" "确认在有 GPU 的机器上, 且驱动已装"; fi

  # 4. vendored 上游代码仓 (缺了就会回退去 clone GitHub)
  miss=""
  for r in colar Latent-Thoughts-Tuning Latent-SFT; do [ -d "$REPO_ROOT/upstream/$r" ] || miss="$miss $r"; done
  [ -z "$miss" ] && pf ok "上游代码仓已 vendored (无需 GitHub)"                  || pf x "upstream/ 缺:$miss" "git pull 更新本仓库; 或联网让脚本自行 clone"

  # 5. 训练数据
  dm=""
  for f in colar_train.json colar_val.json lt_train.jsonl lt_val.jsonl lsft_train.jsonl lsft_val.jsonl; do
    [ -f "$REPO_ROOT/code_real_ladder/$f" ] || dm="$dm $f"; done
  [ -z "$dm" ] && pf ok "训练数据 6 个文件齐全" || pf x "code_real_ladder/ 缺:$dm" "git pull 更新本仓库"

  # 6. 磁盘 (三个 venv + 模型 + ckpt)
  avail=$(df -Pm "$WORK" 2>/dev/null | awk 'NR==2{print $4}')
  if [ -n "$avail" ] && [ "$avail" -lt 60000 ]; then
    pf x "$WORK 所在盘只剩 $((avail/1024)) GB (建议 >60GB)" "WORK=/大盘/路径 bash train_all_code.sh"
  else pf ok "磁盘: 剩余 $(( ${avail:-0} / 1024 )) GB"; fi

  if [ "$pf_fail" = "1" ]; then
    echo ""
    echo "自检未通过, 已停止 —— 先按上面的 -> 修复, 再重跑。"
    echo "(确认无误想强行继续: PREFLIGHT=0 bash train_all_code.sh)"
    exit 1
  fi
  echo ""
fi

# 数据仓预先克隆一次 —— 并行时三个进程同时 clone 同一目录会打架
[ -f "$WORK/lrm/code_real_ladder/manifest.json" ] || {
  echo "预拉数据仓 ..."
  rm -rf "$WORK/lrm"
  if ! gh_clone "${DATA_REPO:-https://github.com/ruijiezh67/LRM_colab_tasks.git}" "$WORK/lrm"; then
    echo "X 数据仓准备失败 -> $WORK/lrm。三个平台都会因此失败, 先解决这一步。"; exit 1
  fi
  if [ ! -f "$WORK/lrm/code_real_ladder/manifest.json" ]; then
    echo "X 数据仓就位但缺 code_real_ladder/manifest.json, 检查 $WORK/lrm"; exit 1
  fi
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
