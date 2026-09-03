# ============================================================================
# config.sh —— 所有可调参数集中在这里。改这一个文件就够，不用记环境变量名。
#
# 用法二选一：
#   ① 直接改下面的默认值
#   ② 或者临时用环境变量覆盖： GPUS=4,5,6 CN=1 bash train_all_code.sh
#      （环境变量优先级更高，写法都是 ${VAR:-默认值}）
#
# 被 train_*.sh 自动 source，不要直接运行。
# ============================================================================

# ─────────────── 1. 运行模式 ───────────────
VERIFY="${VERIFY:-0}"        # 0=全量真训练(正常用这个)  1=排错模式(200行小规模, 不出正式ckpt)
PARALLEL="${PARALLEL:-0}"    # 0=串行(单卡)              1=一个平台一张卡并行(多卡机器用这个)
ONLY="${ONLY:-colar,lt,lsft}" # 只跑哪些平台, 逗号分隔
PROV="${PROV:-0}"            # 1=训练前跑一次数据溯源校验(verify_provenance.py, 需联网)
                             #   默认关: 不给交接方添堵。数据是否干净已实测记录在 TRAINING.md 4.1
                             #   国内记得同时 CN=1 (会把 HF_ENDPOINT 指到 hf-mirror.com)

# ─────────────── 2. 硬件 ───────────────
GPUS="${GPUS:-}"             # 并行模式下的卡号列表, 例: 0,1,2  (按 ONLY 的顺序依次分配)
GPU="${GPU:-}"               # 单个脚本绑一张卡, 例: 3  (等价 CUDA_VISIBLE_DEVICES)
VENV="${VENV:-0}"            # 1=每个平台建独立 venv 隔离依赖。PARALLEL=1 时会自动置 1

# ─────────────── 3. 网络 / 镜像 ───────────────
CN="${CN:-0}"                # 1=一键切国内镜像(HF+pip)。下面两个可单独覆盖
HF_ENDPOINT="${HF_ENDPOINT:-}"      # 空=HF官方;  CN=1 时→ https://hf-mirror.com
PIP_INDEX_URL="${PIP_INDEX_URL:-}"  # 空=系统配置; CN=1 时→ 清华源

# ─────────────── 4. 解释器 / 路径 ───────────────
PY="${PY:-}"                 # 空=自动探测(python3 → python)。指定例: /opt/conda/envs/x/bin/python
WORK="${WORK:-$(pwd)/crux_retrain_work}"   # 产物 + 日志(run_logs/)落哪
DATA_REPO="${DATA_REPO:-https://github.com/ruijiezh67/LRM_colab_tasks.git}"

# ─────────────── 5. 底座 / warm-start ckpt ───────────────
# CoLaR: Llama-1B 底座 + 官方发布的 GSM8K ckpt 做 warm-start
LLAMA_ID="${LLAMA_ID:-unsloth/Llama-3.2-1B-Instruct}"   # Meta官方需申请gating, unsloth是同权重镜像
COLAR_WARM_REPO="${COLAR_WARM_REPO:-AlbertTan/CoLaR}"
COLAR_WARM_FILE="${COLAR_WARM_FILE:-logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt}"
# LT-Tuning: 论文只放代码没放权重, 从 Qwen 官方 instruct 起训
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
LT_COMMIT="${LT_COMMIT:-c18aac6}"                        # 驱动代码的固定 commit, 别动
# Latent-SFT: 官方只发布了数学 ckpt, 从 Llama 官方 instruct 起训
BASE="${BASE:-unsloth/Llama-3.2-1B-Instruct}"            # 名字必须含 llama/qwen/deepseek(上游按子串判分支)

# ─────────────── 6. 训练规模 ───────────────
# ⚠️ 配方锁定：下面这些值是和「上一版自造数据训练」逐字段对齐的。
#    改了就和上一版不可比 —— 除非你确实想做消融，否则别动。
COLAR_EPOCHS="${COLAR_EPOCHS:-25}"   # CoLaR 全量 epoch 数
LT_STAGE_EP="${LT_STAGE_EP:-1}"      # LT 每个阶段几个 epoch(共3阶段)
                                     # ⚠️ num_train_epochs 会自动 = 3 × 本值。
                                     #    StageManager 按 epoch 边界推进阶段, 两者不匹配
                                     #    会只跑到 stage0(纯CoT, 没有latent)。脚本已有守卫。
LSFT_S1="${LSFT_S1:-8}"              # Latent-SFT stage1 epoch(论文10, 为算力裁到8)
LSFT_S2="${LSFT_S2:-20}"             # Latent-SFT stage2 epoch(论文70, 裁到20)
VERIFY_ROWS="${VERIFY_ROWS:-200}"    # VERIFY=1 时取多少行
