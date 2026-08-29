# Code LRM 训练 handoff

用**真实公开 code 数据集**重训三个 latent 推理模型（CoLaR / LT-Tuning / Latent-SFT）。
训练方法和上一版一样，**唯一变的是数据**（自造合成 → 真实三档阶梯）。

## 直接跑

```bash
git clone https://github.com/ruijiezh67/LRM_colab_tasks.git
cd LRM_colab_tasks/handoff

bash train_all_code.sh                          # 单卡串行, 训完三个 ckpt

# 多卡机器: 一个任务一张卡并行(推荐, 三个任务本来就互相独立)
PARALLEL=1 GPUS=0,1,2 bash train_all_code.sh

# 国内网络: 开 HF / pip / GitHub 镜像
CN=1 PARALLEL=1 GPUS=0,1,2 bash train_all_code.sh
```

> 管线已经验证过了（小规模跑通 + loss 收敛），**你不用再跑一遍验证**，直接全量训练即可。

不需要准备任何东西 —— 依赖、官方代码、数据、底座权重，脚本全自动下。要一块 GPU + 联网。

## 文件

| 文件 | 用途 |
|---|---|
| **`TRAINING.md`** | **详细讲解：底座 ckpt 从哪来、配方对照、遵守的规约、出问题怎么查** |
| `train_all_code.sh` | 一条命令训完三个（串行，支持 `ONLY=` 挑平台、断点跳过） |
| `train_colar_code.sh` | 只训 CoLaR → `colar_code_cruxreal.ckpt` |
| `train_lt_code.sh` | 只训 LT-Tuning → `lt_code_out/qwen_code/` |
| `train_lsft_code.sh` | 只训 Latent-SFT（6 步）→ `stage2_results/code/checkpoint-*/hf` |

训练数据不在本目录（避免两份源漂移），在同仓库 [`../code_real_ladder/`](../code_real_ladder/)，脚本自动下载。

## 常用变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VERIFY=1` | 关 | **排错用**，正常不需要。只在训练报错时用它跑 200 行快速复现问题，不出正式 ckpt |
| `WORK=/路径` | `./crux_retrain_work` | 产物和日志落哪（日志在 `$WORK/run_logs/`） |
| `ONLY=colar,lt,lsft` | 全部 | 总脚本只跑指定平台 |
| `PARALLEL=1` `GPUS=0,1,2` | 关 | 一个平台一张卡并行；自动开 venv 隔离 |
| `CN=1` | 关 | 国内镜像：HF→hf-mirror、pip→清华源、GitHub→ghfast 代理 |
| `GPU=3` | — | 单个脚本绑一张卡（等价 `CUDA_VISIBLE_DEVICES`）|
| `PY=/path/python` | 自动探测 | 手动指定解释器（默认按 `python3`→`python` 找）|

## 数据

真实、公开、可引用（**禁自造合成**）。代码输出预测任务，1488 train / 165 val，三平台同一份：

| 档 | 数据集 | 引用 |
|---|---|---|
| T0/T1 浅·中 | CRUXEval · MBPP | Gu et al. 2024, arXiv:2401.03065 · Austin et al. 2021, arXiv:2108.07732 |
| T2 深 | LiveCodeBench execution-v2 | Jain et al. 2024, arXiv:2403.07974 |

**数据来源是锁死的**：脚本开训前校验 sha256，对不上直接中止；1653 行已全量逐字回溯到上述三个公开集。
详见 [`TRAINING.md`](TRAINING.md) §4.1。
