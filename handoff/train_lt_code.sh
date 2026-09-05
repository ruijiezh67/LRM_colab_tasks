#!/bin/bash
# 交接包与完整说明: https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/handoff
# ============================================================================
# LT-Tuning-code 训练 — 一键跑。   bash train_lt_code.sh
#   VERIFY=1 bash train_lt_code.sh → 小规模验证代码(200行, 三阶段各1ep, 不出正式 ckpt)
#   (不带)                          → 全量(1488行, stage0→1→2 三阶段课程) → lt_code_out/
#
# 配方 = 当初自造数据那版 cell_02 的 qwen_code.yaml(= Kai 验证过的 qwen_colab.yaml)逐字段对齐,
#        本次变化 = ① 数据换真实三档 ② thinking_operator_regex 换成代码标识符(见 README.md 第 6 节)。
# 驱动 = 官方 NeosKnight233/Latent-Thoughts-Tuning @ c18aac6 (权重未发布, 只能自训)。
# 需要: 单卡 GPU(A100/L4) + 联网。
# ============================================================================
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 必须在任何 cd 之前解析
. "$HERE/config.sh"   # 所有可调参数集中在 config.sh

# ---------------- 底座 (写死; 可 env 覆盖) --------------
# LT-Tuning 论文没放权重, 官方也没有 code ckpt → 从 Qwen 官方 instruct 底座起训(和自造数据那版一致)。
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
LT_COMMIT="${LT_COMMIT:-c18aac6}"
DATA_REPO="${DATA_REPO:-https://github.com/ruijiezh67/LRM_colab_tasks.git}"

if [ "$VERIFY" = "1" ]; then NROW=$VERIFY_ROWS; else NROW=0; fi
WORK="${WORK:-$(pwd)/crux_retrain_work}"; mkdir -p "$WORK/run_logs"; cd "$WORK"
LOG="$WORK/run_logs/lt_train.log"
. "$HERE/_common.sh"
setup_venv lt
print_env
OUT="$WORK/lt_code_out"
echo "=== LT-code  [$([ "$VERIFY" = 1 ] && echo 验证VERIFY || echo 全量FULL)]  rows=$NROW  三阶段课程 ==="

echo "[1/4] floor (torch2.7.1 / transformers4.55.4 / deepspeed0.18.3 / peft0.18.0)"
pipi "torch==2.7.1" "torchvision==0.22.1" "transformers==4.55.4" "datasets==4.2.0" "deepspeed==0.18.3" "peft==0.18.0" omegaconf accelerate

echo "[2/4] clone 驱动@$LT_COMMIT + model.py sdpa patch + 三档数据 + config"
LT="$WORK/Latent-Thoughts-Tuning"
[ -f "$LT/model.py" ] || gh_clone https://github.com/NeosKnight233/Latent-Thoughts-Tuning.git "$LT"
# 锁定驱动代码的 commit。两条路径:
#   有 .git(联网 clone 来的)   -> 直接 checkout
#   无 .git(仓库内 vendored)   -> 校验 .vendored_commit 是否就是要的那个
# 后者同样保住"版本锁定"这个保证, 只是换了实现方式。
if [ -d "$LT/.git" ]; then
  git -C "$LT" checkout -q "$LT_COMMIT"
elif [ -f "$LT/.vendored_commit" ]; then
  _vc="$(tr -d "[:space:]" < "$LT/.vendored_commit")"
  case "$_vc" in
    "$LT_COMMIT"*) echo "  vendored 副本已在 $LT_COMMIT (${_vc:0:12})" ;;
    *) echo "X vendored 副本是 ${_vc:0:12}, 但 config.sh 要求 $LT_COMMIT。"
       echo "  两者必须一致, 否则配方不可比。修 upstream/ 的快照或改 LT_COMMIT。"; exit 1 ;;
  esac
else
  echo "X $LT 既无 .git 也无 .vendored_commit, 无法确认版本"; exit 1
fi
[ -f "$WORK/lrm/code_real_ladder/lt_train.jsonl" ] || gh_clone "$DATA_REPO" "$WORK/lrm"
# 数据来源: CRUXEval / LiveCodeBench / MBPP 公开集。禁自造/合成数据。
# 交接说明: 这里原来有一把 sha256 字节锁, 已移除 —— 它锁的哈希从写下起就和仓库数据对不上,
#   只会在换机器/重新导出后误杀训练; 而且哈希只能证明"没被改过", 证明不了"是真的"。
#   真校验换成了 verify_provenance.py(逐行比对 HuggingFace 上游), 跑一次即可, 不阻塞训练:
#       HF_ENDPOINT=https://hf-mirror.com python3 verify_provenance.py
#   想让训练前自动校验一次: PROV=1 bash <本脚本>
if [ "${PROV:-0}" = "1" ]; then
  echo "[数据溯源校验] 逐行比对 HuggingFace 上游 (PROV=1)"
  "$PY" "$HERE/verify_provenance.py" "$WORK/lrm/code_real_ladder"
fi

"$PY" - "$LT" "$WORK/lrm/code_real_ladder" "$WORK/data_lt" "$NROW" "$OUT" "$QWEN_ID" "$LT_STAGE_EP" <<'PY'
import pathlib, sys, json, random
LT, src, dst, n, OUT, QWEN, SEP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7])
NTE = 3 * SEP   # 不变式: num_train_epochs 必须 = sum(stage_epochs) = 3 × 每阶段epoch

# (a) model.py: flash_attention_2 -> sdpa 默认 + SoftSeft 别名(eval 用)。与自造数据那版同一处补丁。
mp = pathlib.Path(LT, "model.py"); s = mp.read_text(encoding="utf-8")
s = s.replace('attn_implementation = kwargs.pop("attn_implementation", "flash_attention_2")',
              'attn_implementation = kwargs.pop("attn_implementation", "sdpa")')
if "SoftSeft = LT_Tuning_Model" not in s:
    s = s.rstrip() + "\n\n\nSoftSeft = LT_Tuning_Model  # alias: eval_LT_Tuning.py imports SoftSeft\n"
mp.write_text(s, encoding="utf-8")

# (b) 数据: 三档真实 {question, steps, answer}
pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
tr = [json.loads(l) for l in open(f"{src}/lt_train.jsonl", encoding="utf-8") if l.strip()]
va = [json.loads(l) for l in open(f"{src}/lt_val.jsonl", encoding="utf-8") if l.strip()]
if n > 0:
    random.seed(0); random.shuffle(tr); tr = tr[:n]; va = va[:max(20, n // 5)]
open(f"{dst}/lt_code_train.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in tr))
open(f"{dst}/lt_code_val.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in va))
print(f"  model.py patched | lt data train {len(tr)} val {len(va)}")

# (c) qwen_code.yaml —— cell_02 那份 Kai-proven 配置原样搬过来。
#     ⚠️ num_train_epochs(3) 必须 = sum(stage_epochs)=1+1+1。StageManager 按 epoch 边界推进阶段
#     (run.py:619 + on_epoch_begin), 少一个 epoch 就只跑到 stage0, 拿不到 latent(soft-fusion)。
cfg = f"""attn_implementation: sdpa
bf16: true
dataloader_num_workers: 2
dataset_save_path: {dst}/tok
dihr: true
eval_stage_mode: soft_fusion
eval_strategy: 'no'
fp16: false
fusion_alpha:
- 0.5
- 0.5
- 0.6
fusion_temperature: 1.0
fusion_top_p: 0.9
gradient_accumulation_steps: 4
labels_per_stage:
- 0
- 10
- 16
learning_rate: 5.0e-05
load_model_path: null
logging_steps: 10
max_epoch_checkpoints: 5
max_grad_norm: 1.0
max_step_checkpoints: null
model_name_or_path: {QWEN}
name: qwen_code
no_thoughts: false
num_train_epochs: {NTE}
only_eval: false
output_dir: {OUT}
per_device_eval_batch_size: 4
per_device_train_batch_size: 4
project: LT_Tuning
reinforce_max_eval_length: 2048
reinforce_prob_threshold:
- 0.0
- 0.3
- 0.2
remove_unused_columns: true
report_to: none
reset_optimizer: true
resume: 0
save_dataset: true
save_every_n_steps: 0
save_only_improve: false
save_path: {OUT}
save_safetensors: false
save_steps: 1000
save_strategy: steps
save_total_limit: 2
seed: 42
stage_epochs:
- {SEP}
- {SEP}
- {SEP}
stage_modes:
- common
- hidden_state
- soft_fusion
stage_names:
- stage0-cot
- stage1-hidden-state
- stage2-soft-fusion
thinking_hidden_state_layer: -1
thinking_insertion_prob:
- 0.0
- 0.85
- 0.95
thinking_mlp_activation: gelu
thinking_mlp_hidden_dim: 1024
thinking_operator_regex: '[0-9]+|[a-zA-Z_][a-zA-Z0-9_]*'
thinking_prompt_tokens: 0
thinking_secondary_insertion_prob:
- 0.0
- 0.15
- 0.2
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
import yaml as _y; _c = _y.safe_load(cfg)
assert _c["num_train_epochs"] == sum(_c["stage_epochs"]), "不变式破了: num_train_epochs != sum(stage_epochs)"
print(f"  wrote qwen_code.yaml (num_train_epochs={NTE} = sum(stage_epochs)={SEP}×3 → 跑满 stage0→1→2)")
PY

echo "[3/4] 训练 (deepspeed --num_gpus 1 run.py, stage0→1→2)"
cd "$LT"
dsrun --num_gpus 1 run.py configs/qwen_code.yaml 2>&1 | tee "$LOG"

echo "[4/4] 结果"
"$PY" - "$LOG" "$VERIFY" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
# 阶段守卫: 三个 stage 都必须出现, 否则课程没跑满(只到 stage0 = 没有 latent)
seen = [s for s in ("stage0-cot", "stage1-hidden-state", "stage2-soft-fusion") if s in txt]
print(f"阶段: {seen}  ({len(seen)}/3)")
if len(seen) < 3:
    print("[FAIL] 课程没跑满三阶段 → 模型没有 latent。查 num_train_epochs 是否 = sum(stage_epochs)")
if sys.argv[2] == "1":
    ls = [float(x) for x in re.findall(r"'loss':\s*([0-9]+\.[0-9]+)", txt)]
    if len(ls) >= 2:
        print(f"loss: {ls[0]:.3f} -> {ls[-1]:.3f} ({len(ls)} 点)")
        print("[PASS] 管线跑通 + loss 下降" if ls[-1] < ls[0] else "[WARN] 跑通但 loss 没降")
    else:
        print("[WARN] 没抓到 loss → 看 run_logs/lt_train.log")
PY
[ "$VERIFY" = "1" ] || echo "✅ DONE 产物: $OUT/qwen_code/  (HF 目录: model + tokenizer, .bin 权重)"
