#!/bin/bash
# ============================================================================
# CoLaR-code 训练 — 一键跑。   bash train_colar_code.sh
#   VERIFY=1 bash train_colar_code.sh  → 小规模验证代码(200行, ~10min, 不出正式 ckpt)
#   (不带 VERIFY)                       → 全量真训练(25ep) → colar_code_cruxreal.ckpt
#
# 配方 = 当初自造数据那版 colar_coding 的 hparams.yaml 逐字段对齐(见 TRAINING.md §配方对照),
#        本次唯一变化 = 数据换成真实三档(CRUXEval+LiveCodeBench+MBPP)。
# 需要: 单卡 GPU(A100/L4/T4-16G 皆可) + 联网。
# ============================================================================
set -e

# ---------------- 底座 / warm-start ckpt (全部写死, 无需手动准备; 可用 env 覆盖) --------------
# 底座 LLM: Llama-3.2-1B-Instruct。Meta 官方 repo 需申请 gating, 用 unsloth 同权重镜像免申请。
LLAMA_ID="${LLAMA_ID:-unsloth/Llama-3.2-1B-Instruct}"
# warm-start: CoLaR 官方发布的 GSM8K ckpt (Tan et al. 2025, arXiv:2505.16552)。
# 注意: 不要用 rjz123/colar-coding-* 当 warm-start —— 那条血缘经过自造合成数据, 会把污染带回来。
COLAR_WARM_REPO="${COLAR_WARM_REPO:-AlbertTan/CoLaR}"
COLAR_WARM_FILE="${COLAR_WARM_FILE:-logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt}"
DATA_REPO="${DATA_REPO:-https://github.com/ruijiezh67/LRM_colab_tasks.git}"

VERIFY="${VERIFY:-0}"
if [ "$VERIFY" = "1" ]; then NROW=200; EP=3;  TAG=verify_smoke; else NROW=0; EP=25; TAG=cruxreal_gsmwarm; fi
WORK="${WORK:-$(pwd)/crux_retrain_work}"; mkdir -p "$WORK/run_logs"; cd "$WORK"
LOG="$WORK/run_logs/colar_train.log"
echo "=== CoLaR-code  [$([ "$VERIFY" = 1 ] && echo 验证VERIFY || echo 全量FULL)]  rows=$NROW epochs=$EP ==="

echo "[1/5] deps (官方 CoLaR 兼容 floor)"
pip -q install "transformers==4.45.2" "lightning==2.5.1.post0" "peft==0.15.2" "omegaconf==2.3.0" "numpy>=2.0,<2.3" sentencepiece accelerate
pip -q install --force-reinstall --no-deps "huggingface_hub==0.34.4"

echo "[2/5] 官方 CoLaR 代码 + 三档真实数据"
[ -f "$WORK/colar/run.py" ] || git clone -q https://github.com/xiaomi-research/colar.git "$WORK/colar"
[ -f "$WORK/lrm/code_real_ladder/colar_train.json" ] || git clone -q "$DATA_REPO" "$WORK/lrm"

echo "[数据来源校验] 禁自造数据铁律 -- 校验 sha256"
python - "$WORK/lrm/code_real_ladder" <<'GUARD'
import hashlib, os, sys
SHA = {
    "colar_train.json": "625bd94402c1fd5ec167bcbc98ab66e587a45bf6aba3a0effb6749feb9c58748",
    "colar_val.json": "b64621ec716254e6f2a5969229c9e0b78ecc5b68491a02a642c8c2273fabcdc2",
}
src = sys.argv[1]
bad = 0
for f, want in SHA.items():
    p = os.path.join(src, f)
    if not os.path.exists(p):
        print("  X 数据文件缺失:", p); bad = 1; continue
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if got != want:
        print("  X 数据与锁定版本不符:", f)
        print("    expect", want)
        print("    actual", got)
        bad = 1
    else:
        print("  ok", f)
if bad:
    print("  训练中止。铁律: 只允许真实公开数据(CRUXEval/LiveCodeBench/MBPP), 禁自造/合成。")
    print("  若确需换数据, 必须先确认新数据的权威性+有效性, 再更新脚本里的 sha256。")
    sys.exit(1)
print("  数据来源校验通过 -- 1653 行全部可逐字回溯到 CRUXEval / LiveCodeBench / MBPP")
GUARD

echo "[3/5] 底座 $LLAMA_ID + warm-start $COLAR_WARM_REPO"
WS="$WORK/ws"; mkdir -p "$WS/models/llms" "$WS/datasets/text_reasoning/coding_real_ladder"
LLAMA="$WS/models/llms/Llama-3.2-1B-Instruct"
[ -f "$LLAMA/config.json" ] || huggingface-cli download "$LLAMA_ID" --local-dir "$LLAMA" --exclude "original/*"
GSM="$WORK/colar_hf/$COLAR_WARM_FILE"
[ -f "$GSM" ] || huggingface-cli download "$COLAR_WARM_REPO" "$COLAR_WARM_FILE" --local-dir "$WORK/colar_hf"
[ -f "$GSM" ] || { echo "❌ warm-start ckpt 没下到: $GSM"; exit 1; }

echo "[4/5] 数据 → coding_real_ladder (qsa 格式)$([ "$NROW" -gt 0 ] && echo ", 取前 $NROW 验证")"
python - "$WORK/lrm/code_real_ladder" "$WS/datasets/text_reasoning/coding_real_ladder" "$NROW" <<'PY'
import json, sys, random
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tr = json.load(open(f"{src}/colar_train.json", encoding="utf-8"))
va = json.load(open(f"{src}/colar_val.json", encoding="utf-8"))
if n > 0:
    random.seed(0); random.shuffle(tr); tr = tr[:n]; va = va[:max(20, n // 5)]
for nm, dd in [("train", tr), ("val", va), ("test", va)]:
    json.dump(dd, open(f"{dst}/{nm}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  coding_real_ladder train {len(tr)} val {len(va)}")
PY

# ---- 训练超参 = colar_coding/hparams.yaml 原样 (batch4 / accum4 / 25ep / val每5ep / seed0) ----
echo "[5/5] 训练 (run.py, warm colar-gsm, ${EP}ep)"
cd "$WORK/colar"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -u run.py --model=colar --dataset=qsa --devices=0 --workspace_path="$WS" --load_ckpt_path="$GSM" \
    --log_suffix=$TAG --seed=0 dataset_name=coding_real_ladder model_id=Llama-3.2-1B-Instruct \
    batch_size=4 accumulate_grad_batches=4 max_epochs=$EP check_val_every_n_epoch=5 2>&1 | tee "$LOG"

if [ "$VERIFY" = "1" ]; then
  echo "=== loss 收敛检查 (golden rule) ==="
  python - "$LOG" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
ls = [float(x) for x in re.findall(r"(?:train_)?loss[=:\s]+([0-9]+\.[0-9]+)", txt)]
if len(ls) >= 2:
    print(f"loss: {ls[0]:.3f} -> {ls[-1]:.3f}  ({len(ls)} 点)")
    print("[PASS] 管线跑通 + loss 下降(收敛)" if ls[-1] < ls[0] else "[WARN] 跑通但 loss 没降 → 查 lr/数据/配方")
else:
    print("[WARN] 没抓到 loss → 看 run_logs/colar_train.log 训练是否真启动")
PY
else
  CKPT=$(ls -t "$WORK"/colar/logs/colar/qsa-coding_real_ladder/*$TAG*/checkpoints/epoch*.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && CKPT=$(ls -t "$WORK"/colar/logs/colar/qsa-coding_real_ladder/*$TAG*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "❌ 没找到 ckpt, 看 $LOG"; exit 1; }
  cp "$CKPT" "$WORK/colar_code_cruxreal.ckpt"
  echo "✅ DONE 产物: $WORK/colar_code_cruxreal.ckpt   (加载时需 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1)"
fi
