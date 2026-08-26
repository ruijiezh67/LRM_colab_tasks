#!/bin/bash
# ============================================================================
# LT-Tuning code 训练 — 一键跑。 bash train_lt_code.sh
#   VERIFY=1 bash train_lt_code.sh → 小规模验证代码(200行+1ep/阶段+loss检查)
#   (不带)                          → 全量(3ep/阶段, 3 阶段课程)→ lt_code_out/
# 数据: 真实三档(GitHub 自动下)。底座 Qwen2.5-1.5B。驱动: 官方 NeosKnight233/Latent-Thoughts-Tuning@c18aac6。
# 需要: A100/L4 + 联网。
# ============================================================================
set -e
VERIFY="${VERIFY:-0}"; if [ "$VERIFY" = "1" ]; then NROW=200; EP=1; else NROW=0; EP=3; fi
WORK="${WORK:-$(pwd)/crux_retrain_work}"; mkdir -p "$WORK"; cd "$WORK"
echo "=== LT-code  [$([ $VERIFY = 1 ] && echo 验证VERIFY || echo 全量FULL)]  rows=$NROW ep/stage=$EP ==="

echo "[1/4] floor (torch2.7.1 / transformers4.55.4 / deepspeed0.18.3 / peft0.18.0)"
pip -q install "torch==2.7.1" "torchvision==0.22.1" "transformers==4.55.4" "datasets==4.2.0" "deepspeed==0.18.3" "peft==0.18.0" omegaconf accelerate

echo "[2/4] clone 驱动@c18aac6 + model.py sdpa patch + 三档数据"
LT="$WORK/Latent-Thoughts-Tuning"
[ -d "$LT/.git" ] || git clone -q https://github.com/NeosKnight233/Latent-Thoughts-Tuning.git "$LT"
git -C "$LT" checkout -q c18aac6
[ -f "$WORK/lrm/code_real_ladder/lt_train.jsonl" ] || git clone -q https://github.com/ruijiezh67/LRM_colab_tasks.git "$WORK/lrm"
python - "$LT" "$WORK/lrm/code_real_ladder" "$WORK/data_lt" "$NROW" "$EP" <<'PY'
import pathlib, sys, json, random
LT, src, dst, n, ep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
# (a) model.py sdpa 默认 + SoftSeft 别名
mp = pathlib.Path(LT, "model.py"); s = mp.read_text(encoding="utf-8")
s = s.replace('attn_implementation = kwargs.pop("attn_implementation", "flash_attention_2")',
              'attn_implementation = kwargs.pop("attn_implementation", "sdpa")')
if "SoftSeft = LT_Tuning_Model" not in s: s = s.rstrip() + "\n\nSoftSeft = LT_Tuning_Model\n"
mp.write_text(s, encoding="utf-8")
# (b) data: 三档真实 {question,steps,answer}
pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
tr = [json.loads(l) for l in open(f"{src}/lt_train.jsonl", encoding="utf-8") if l.strip()]
va = [json.loads(l) for l in open(f"{src}/lt_val.jsonl", encoding="utf-8") if l.strip()]
if n > 0: random.seed(0); random.shuffle(tr); tr = tr[:n]; va = va[:max(20, n//5)]
open(f"{dst}/lt_code_train.jsonl","w",encoding="utf-8").write("\n".join(json.dumps(r,ensure_ascii=False) for r in tr))
open(f"{dst}/lt_code_val.jsonl","w",encoding="utf-8").write("\n".join(json.dumps(r,ensure_ascii=False) for r in va))
print(f"  model.py patched | lt data train {len(tr)} val {len(va)}")
# (c) config qwen_code.yaml(Kai proven, 只改 data/epoch/no-flash/save_safetensors:false)
cfg = f"""attn_implementation: sdpa
bf16: true
dataloader_num_workers: 2
dataset_save_path: {dst}/tok
eval_stage_mode: soft_fusion
eval_strategy: 'no'
fp16: false
fusion_alpha: [0.5, 0.5, 0.6]
fusion_temperature: 1.0
fusion_top_p: 0.9
gradient_accumulation_steps: 4
labels_per_stage: [0, 10, 16]
learning_rate: 5.0e-05
load_model_path: null
logging_steps: 5
max_grad_norm: 1.0
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
name: qwen_code
no_thoughts: false
num_train_epochs: {ep}
only_eval: false
output_dir: {LT}/../lt_code_out
per_device_eval_batch_size: 4
per_device_train_batch_size: 4
project: LT_Tuning
remove_unused_columns: true
report_to: none
reset_optimizer: true
resume: 0
save_dataset: true
save_safetensors: false
save_strategy: 'no'
save_total_limit: 2
seed: 42
stage_epochs: [{ep}, {ep}, {ep}]
stage_modes: [common, hidden_state, soft_fusion]
stage_names: [stage0-cot, stage1-hidden-state, stage2-soft-fusion]
thinking_hidden_state_layer: -1
thinking_insertion_prob: [0.0, 0.85, 0.95]
thinking_mlp_activation: gelu
thinking_mlp_hidden_dim: 1024
thinking_operator_regex: '[0-9]+|[a-zA-Z_][a-zA-Z0-9_]*'
thinking_prompt_tokens: 0
thinking_secondary_insertion_prob: [0.0, 0.15, 0.2]
thinking_strategy: confidence
thinking_token: <thinking>
thinking_use_mlp: false
train_path: {dst}/lt_code_train.jsonl
use_flash_attention: false
use_unk_for_thinking: false
val_path: {dst}/lt_code_val.jsonl
warmup_ratio: 0.05
weight_decay: 0.01
"""
pathlib.Path(LT, "configs").mkdir(parents=True, exist_ok=True)
pathlib.Path(LT, "configs", "qwen_code.yaml").write_text(cfg, encoding="utf-8")
print("  wrote qwen_code.yaml (save_safetensors:false; operator_regex 改 code identifier)")
PY

echo "[3/4] 训练 (deepspeed --num_gpus 1 run.py, 3 阶段课程)"
cd "$LT"
deepspeed --num_gpus 1 run.py configs/qwen_code.yaml 2>&1 | tee "$WORK/lt_train.log"

echo "[4/4] 结果"
if [ "$VERIFY" = "1" ]; then
  python - "$WORK/lt_train.log" <<'PY'
import re, sys
ls=[float(x) for x in re.findall(r"'loss':\s*([0-9]+\.[0-9]+)", open(sys.argv[1],encoding="utf-8",errors="ignore").read())]
if len(ls)>=2:
    print(f"loss: {ls[0]:.3f} -> {ls[-1]:.3f} ({len(ls)} 点)")
    print("[PASS] 管线跑通 + loss 下降" if ls[-1]<ls[0] else "[WARN] 跑通但 loss 没降")
else: print("[WARN] 没抓到 loss → 看 lt_train.log")
PY
else
  echo "✅ DONE 产物: $WORK/lt_code_out/  (含 model + tokenizer, .bin 权重)"
fi
