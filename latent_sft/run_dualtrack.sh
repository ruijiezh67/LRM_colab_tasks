#!/bin/bash
# End-to-end clean dual-track Latent-SFT run.
#
# Steps 1 and 2 invoke UPSTREAM SCRIPTS UNCHANGED (stage-1 encoder training and the
# latent soft-label generator). Only steps 0, 3, 4 and 5 are this layer's code.
# Nothing here writes inside upstream/: every artefact lands in data/ or out/.
#
#   PYTHON=python NPROC_PER_NODE=8 bash run_dualtrack.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-25011}"

# ---- editable config --------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-<path-or-hf-id-of-your-base-model>}"     # stage-1 encoder init
STAGE1_CKPT="${STAGE1_CKPT:-${HERE}/out/stage1_encoder}"           # written by step 1
LATENT_MODEL="${LATENT_MODEL:-${STAGE1_CKPT}}"                     # stage-2 student init
RAW_DATA="${RAW_DATA:-${HERE}/data/GSM8k-Aug-train.jsonl}"
SOURCE_FORMAT="${SOURCE_FORMAT:-latentsft}"
COMPRESSION_RATE="${COMPRESSION_RATE:-16}"
TOPK="${TOPK:-10}"
COT_W="${COT_W:-1.0}"
ANS_W="${ANS_W:-1.0}"
LATENT_W="${LATENT_W:-1.0}"
TEACHER="${TEACHER:-stage1_encoder}"                               # or proxy_decoder
BOTTLENECK="${BOTTLENECK:-1}"                                      # 1=bottleneck twin; 0=Gate-0 control twin (answer may read the CoT)

# Empty by default so the bottleneck run is byte-identical; only the control twin adds a flag.
BOTTLENECK_FLAG=""
if [[ "${BOTTLENECK}" == "0" ]]; then BOTTLENECK_FLAG="--no_bottleneck"; fi

export DUALTRACK_DATA="${HERE}/data/dualtrack_clean.jsonl"
export DUALTRACK_UPSTREAM_VIEW="${HERE}/data/dualtrack_upstream_view.jsonl"
export DUALTRACK_SOFT="${HERE}/data/soft"
export DUALTRACK_OUT="${HERE}/out/clean_dualtrack"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
# Never let an interpreter write __pycache__ inside upstream/: that alone would
# make `git -C upstream status --porcelain` non-empty.
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${HERE}/data" "${HERE}/out"

echo "== 0/5 data =================================================="
"${PYTHON}" -m dualtrack.prepare_data \
    --source_format "${SOURCE_FORMAT}" \
    --src "${RAW_DATA}" \
    --out "${DUALTRACK_DATA}" \
    --emit_upstream_view "${DUALTRACK_UPSTREAM_VIEW}"

if [[ "${TEACHER}" == "stage1_encoder" ]]; then
echo "== 1/5 stage-1 encoder (UPSTREAM SCRIPT, UNEDITED) ============"
"${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    "${HERE}/upstream/script/run_distill_stage1_encoder.py" \
    --encoder_name_or_path "${BASE_MODEL}" \
    --decoder_name_or_path "${BASE_MODEL}" \
    --bfloat16 True --use_flash_attention_2 False --topk_interpolation "${TOPK}" \
    --train_data_path "${DUALTRACK_UPSTREAM_VIEW}" \
    --compression_rate "${COMPRESSION_RATE}" \
    --lora_tune True --lora_rank 64 --lora_dropout 0.1 \
    --deepspeed "${HERE}/upstream/config_zero1.json" \
    --no_remove_unused_columns --learning_rate 1e-4 --warmup_ratio 0.05 \
    --weight_decay 0.01 --lr_scheduler_type cosine --num_train_epochs 10 --bf16 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
    --logging_steps 1 --save_strategy epoch --report_to none \
    --overwrite_output_dir --output_dir "${STAGE1_CKPT}"

echo "== 2/5 latent soft labels (UPSTREAM SCRIPT, UNEDITED) ========="
"${PYTHON}" -m dualtrack.make_latents \
    --teacher stage1_encoder \
    --encoder_path "${STAGE1_CKPT}/hf" \
    --decoder_path "${BASE_MODEL}" \
    --data "${DUALTRACK_DATA}" \
    --upstream_view "${DUALTRACK_UPSTREAM_VIEW}" \
    --save_path "${DUALTRACK_SOFT}" \
    --mp_size "${NPROC_PER_NODE}" \
    --compression_rate "${COMPRESSION_RATE}" --topk_interpolation "${TOPK}"
else
echo "== 1-2/5 latent soft labels (PROXY FALLBACK, not the paper) ===="
"${PYTHON}" -m dualtrack.make_latents \
    --teacher proxy_decoder --ckpt "${LATENT_MODEL}" \
    --data "${DUALTRACK_DATA}" --save_path "${DUALTRACK_SOFT}" \
    --compression_rate "${COMPRESSION_RATE}" --topk_interpolation "${TOPK}"
fi

echo "== 3/5 alignment ============================================="
"${PYTHON}" -m dualtrack.alignment --verify_soft "${DUALTRACK_SOFT}" --data "${DUALTRACK_DATA}"

echo "== 4/5 stage-2 dual-track training ==========================="
"${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    "${HERE}/dualtrack/train_dualtrack.py" \
    --latent_model_path "${LATENT_MODEL}" \
    --cot_w "${COT_W}" --ans_w "${ANS_W}" --latent_w "${LATENT_W}" \
    ${BOTTLENECK_FLAG} \
    --bfloat16 True --use_flash_attention_2 False --topk_interpolation "${TOPK}" \
    --train_data_path "${DUALTRACK_DATA}" \
    --train_latent_soft_label_path "${DUALTRACK_SOFT}" \
    --add_gumbel_noise True --gumbel_temperature 1.0 --noise_scale 1.0 \
    --lora_tune True --lora_rank 64 --lora_dropout 0.1 --training True \
    --deepspeed "${HERE}/upstream/config_zero1.json" \
    --no_remove_unused_columns --learning_rate 3e-4 --warmup_ratio 0.05 \
    --weight_decay 0.01 --lr_scheduler_type cosine --num_train_epochs 20 --bf16 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
    --logging_steps 1 --save_strategy epoch --report_to none \
    --overwrite_output_dir --output_dir "${DUALTRACK_OUT}"

echo "== 5/5 V1 / V2 / V3 =========================================="
"${PYTHON}" -m dualtrack.verify_dualtrack \
    --out "${DUALTRACK_OUT}" --eval_data "${DUALTRACK_DATA}" \
    --n_samples 20 --teacher_forced_check \
    --report "${DUALTRACK_OUT}/verify_report.json"

echo "== upstream purity check ====================================="
git -C "${HERE}/upstream" status --porcelain && echo "(empty above = upstream untouched)"
