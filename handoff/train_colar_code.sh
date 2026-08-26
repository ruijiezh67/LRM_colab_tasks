#!/bin/bash
# ============================================================================
# CoLaR-code 训练 — 一键跑。 bash train_colar_code.sh
#   VERIFY=1 bash train_colar_code.sh  → 小规模验证代码(200行+3ep+loss收敛检查, ~10min)
#   (不带 VERIFY)                       → 全量真训练(25ep, ~1-2h) → colar_code_cruxreal.ckpt
# 数据: 真实三档(CRUXEval+LiveCodeBench+MBPP), 从 GitHub 自动下。底座 Llama-1B + warm colar-gsm。
# 需要: A100/L4 等 GPU + 联网。
# ============================================================================
set -e
VERIFY="${VERIFY:-0}"; if [ "$VERIFY" = "1" ]; then NROW=200; EP=3; TAG=verify_smoke; else NROW=0; EP=25; TAG=cruxreal_gsmwarm; fi
WORK="${WORK:-$(pwd)/crux_retrain_work}"; mkdir -p "$WORK"; cd "$WORK"
echo "=== CoLaR-code  [$([ $VERIFY = 1 ] && echo 验证VERIFY || echo 全量FULL)]  rows=$NROW epochs=$EP ==="

echo "[1/5] deps (官方 CoLaR 兼容版)"
pip -q install "transformers==4.45.2" "lightning==2.5.1.post0" "peft==0.15.2" "omegaconf==2.3.0" "numpy>=2.0,<2.3" sentencepiece accelerate
pip -q install --force-reinstall --no-deps "huggingface_hub==0.34.4"

echo "[2/5] 官方 CoLaR + 三档真实数据"
[ -f "$WORK/colar/run.py" ] || git clone -q https://github.com/xiaomi-research/colar.git "$WORK/colar"
[ -f "$WORK/lrm/code_real_ladder/colar_train.json" ] || git clone -q https://github.com/ruijiezh67/LRM_colab_tasks.git "$WORK/lrm"

echo "[3/5] 底座 Llama-1B + warm colar-gsm"
WS="$WORK/ws"; mkdir -p "$WS/models/llms" "$WS/datasets/text_reasoning/coding_mix"
LLAMA="$WS/models/llms/Llama-3.2-1B-Instruct"
[ -f "$LLAMA/config.json" ] || huggingface-cli download unsloth/Llama-3.2-1B-Instruct --local-dir "$LLAMA" --exclude "original/*"
GSM="$WORK/colar_hf/logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt"
[ -f "$GSM" ] || huggingface-cli download AlbertTan/CoLaR logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt --local-dir "$WORK/colar_hf"

echo "[4/5] 数据 → coding_mix (qsa)$([ $NROW -gt 0 ] && echo ", 取前 $NROW 验证")"
python - "$WORK/lrm/code_real_ladder" "$WS/datasets/text_reasoning/coding_mix" "$NROW" <<'PY'
import json, sys, random
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tr = json.load(open(f"{src}/colar_train.json", encoding="utf-8"))
va = json.load(open(f"{src}/colar_val.json", encoding="utf-8"))
if n > 0:
    random.seed(0); random.shuffle(tr); tr = tr[:n]; va = va[:max(20, n//5)]
for nm, dd in [("train", tr), ("val", va), ("test", va)]:
    json.dump(dd, open(f"{dst}/{nm}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  coding_mix train {len(tr)} val {len(va)}")
PY

echo "[5/5] 训练 (run.py, warm colar-gsm, ${EP}ep)"
cd "$WORK/colar"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -u run.py --model=colar --dataset=qsa --devices=0 --workspace_path="$WS" --load_ckpt_path="$GSM" \
    --log_suffix=$TAG --seed=0 dataset_name=coding_mix model_id=Llama-3.2-1B-Instruct \
    batch_size=4 accumulate_grad_batches=2 max_epochs=$EP check_val_every_n_epoch=1 2>&1 | tee "$WORK/colar_train.log"

if [ "$VERIFY" = "1" ]; then
  echo "=== loss 收敛检查 (golden rule) ==="
  python - "$WORK/colar_train.log" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
ls = [float(x) for x in re.findall(r"(?:train_)?loss[=:\s]+([0-9]+\.[0-9]+)", txt)]
if len(ls) >= 2:
    print(f"loss: {ls[0]:.3f} -> {ls[-1]:.3f}  ({len(ls)} 点)")
    print("[PASS] 管线跑通 + loss 下降(收敛)" if ls[-1] < ls[0] else "[WARN] 跑通但 loss 没降 → 查 lr/数据/配方")
else:
    print("[WARN] 没抓到 loss → 看 colar_train.log 训练是否真启动")
PY
else
  CKPT=$(ls -t "$WORK"/colar/logs/colar/qsa-coding_mix/*$TAG*/checkpoints/epoch*.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && CKPT=$(ls -t "$WORK"/colar/logs/colar/qsa-coding_mix/*$TAG*/checkpoints/last.ckpt | head -1)
  cp "$CKPT" "$WORK/colar_code_cruxreal.ckpt"; echo "✅ DONE 产物: $WORK/colar_code_cruxreal.ckpt"
fi
