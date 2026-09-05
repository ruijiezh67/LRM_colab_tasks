# Code LRM 训练 handoff

用**真实公开 code 数据集**重训三个 latent 推理模型（CoLaR / LT-Tuning / Latent-SFT）。
训练方法与上一版完全一致，**唯一变的是数据**（自造合成 → 真实三档阶梯）。

- 交接包：https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/handoff
- 上游代码：https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/upstream
- 上游数据集：https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/upstream_data
- 训练数据：https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/code_real_ladder

---

## 1. 一条命令

```bash
git clone https://github.com/ruijiezh67/LRM_colab_tasks.git
cd LRM_colab_tasks/handoff

CN=1 PARALLEL=1 GPUS=0,1,2 bash train_all_code.sh
```

- `CN=1` 开国内镜像（HF → hf-mirror.com，pip → 清华源）
- `PARALLEL=1 GPUS=0,1,2` 三个平台各占一张卡并行（三者本来就互相独立）
- 单卡就 `bash train_all_code.sh`，串行跑

脚本开头会先做**起飞前自检**（python / venv / 卡数 / 上游代码 / 数据 / 磁盘）。
任何一项不满足都会**当场停下并给出修复命令**，不会跑到一半才失败。

GitHub 不通时换代理 clone：`git clone https://ghfast.top/https://github.com/ruijiezh67/LRM_colab_tasks.git`

## 2. 需要什么

| | |
|---|---|
| GPU | ≥1 张；并行模式每平台各一张 |
| 磁盘 | ≥60 GB（三个 venv + 底座权重 + ckpt） |
| 网络 | **只需能访问 HuggingFace**。GitHub 只在最初 clone 用一次，训练全程不再访问 |

### 离线程度

| 依赖 | 状态 |
|---|---|
| 三个上游代码仓（CoLaR / LT-Tuning / Latent-SFT） | ✅ 随仓库分发（`../upstream/`） |
| 训练数据 1488 train / 165 val | ✅ 随仓库分发（`../code_real_ladder/`） |
| 溯源校验用的三个上游数据集 | ✅ 随仓库分发（`../upstream_data/`），`PROV=1` 可离线跑 |
| 三个底座权重 | ❌ GB 级第三方权重，需从 HuggingFace 下 |

出处与许可证见 `../upstream/PROVENANCE.md`、`../upstream_data/PROVENANCE.md`。

## 3. 三个平台 · 产物

| 脚本 | 平台 | 底座 | 产物 | 时长(A100) |
|---|---|---|---|---|
| `train_colar_code.sh` | CoLaR（官方 run.py） | Llama-3.2-1B + warm-start CoLaR-GSM | `$WORK/colar_code_cruxreal.ckpt` | ~1–2 h |
| `train_lt_code.sh` | LT-Tuning（三阶段课程） | Qwen2.5-1.5B-Instruct | `$WORK/lt_code_out/qwen_code/` | ~1 h |
| `train_lsft_code.sh` | Latent-SFT（6 步管线） | Llama-3.2-1B-Instruct | `$WORK/Latent-SFT/output/stage2_results/code/checkpoint-*/hf` | ~3–5 h |

`WORK` 默认 `$(pwd)/crux_retrain_work`，可用 `WORK=/data/xxx` 改。

### 底座权重下不动时

脚本会自动下。若网络受限，手动放好即可跳过（脚本检测到 `config.json` / ckpt 就不下载）：

```
<WORK>/llama/     unsloth/Llama-3.2-1B-Instruct     2.4 GB   CoLaR + Latent-SFT 共用
<WORK>/qwen/      Qwen/Qwen2.5-1.5B-Instruct        2.9 GB   LT-Tuning
<WORK>/colar_hf/logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt   116 MB
```

> Meta 官方 Llama repo 需申请 gating，`unsloth/` 是同权重镜像，故写死用它。

## 4. 数据

**铁律：禁自造/合成数据训练，只用真实公开可引用数据集。** 上一版 code ckpt 用的 `gen_code_exec`
（自造算术赋值链）已归档不用——太简单，CoT 秒解、latent 冗余，掩盖了 latent 的必要性。

任务统一为**代码输出预测**（execution reasoning）：给函数 + 输入 → 预测返回值。

| 档 | 数据集 | 量 | 引用 |
|---|---|---|---|
| T0 浅 | CRUXEval + MBPP 低半 | 562 | Gu et al. 2024, arXiv:2401.03065 / Austin et al. 2021, arXiv:2108.07732 |
| T1 中 | CRUXEval + MBPP 高半 | 612 | 同上 |
| T2 深 | LiveCodeBench execution-v2（原生 numsteps 497–996） | 479 | Jain et al. 2024, arXiv:2403.07974 |

合计 **1488 train / 165 val**。**三个平台用同一份数据**（一致性铁律），只是格式不同：
`colar_{train,val}.json` / `lt_{train,val}.jsonl` / `lsft_{train,val}.jsonl`。

可选校验（现已可离线跑）：

```bash
PROV=1 bash train_all_code.sh          # 训练前跑一次
python3 verify_provenance.py ../code_real_ladder    # 或单独跑
```

它逐行核对 (函数体, 调用输入, 答案) 三元组是否**逐字出现在上游公开集**——比 sha256 强，
sha256 只能证明"没被改过"，证明不了"是真的"。

## 5. 常用参数（完整清单见 `config.sh`）

| 想改什么 | 变量 |
|---|---|
| 用哪几张卡 / 并行还是串行 | `GPUS` `PARALLEL` |
| 只跑某些平台 | `ONLY=colar` 或 `colar,lt` |
| 产物落哪 | `WORK` |
| 国内镜像 | `CN=1` |
| 跳过起飞前自检 | `PREFLIGHT=0` |
| 数据溯源校验 | `PROV=1` |
| 小规模排错（200 行，不出正式 ckpt） | `VERIFY=1` |

改法二选一：直接改 `config.sh` 默认值，或临时用环境变量覆盖（环境变量优先）。

> ⚠️ **`config.sh` §6 的训练规模别动**（epoch 数那些）。它们与上一版逐字段对齐，改了就不可比——见下节。

## 6. 配方对照 —— 与自造数据那版逐字段对齐

训练方法没变。下表是脚本超参 vs 当初那版的出处，逐条核对过。**这是"本次重训与上一版可比"的依据。**

**CoLaR**（基准 = `colar_coding/hparams.yaml`，当初自造数据那个 ckpt 的实际记录）

| 字段 | 当初 → 现在 | |
|---|---|---|
| `model_id` / `load_ckpt_path` | Llama-3.2-1B-Instruct / colar-gsm `colar_best.ckpt` | ✅ 同 |
| `batch_size` / `accumulate_grad_batches` | 4 / 4 | ✅ 本次修正（曾误写 2，等效 batch 减半） |
| `max_epochs` / `check_val_every_n_epoch` | 25 / 5 | ✅ 本次修正（曾误写 1） |
| `seed` | 0 | ✅ |
| lora r/alpha · compression · embed loss | 128/32 · 5 · mse（官方默认，未改） | ✅ |
| **`dataset_name`** | `coding_mix`(自造) → **`coding_real_ladder`**(真实) | ← 唯一变化 |

**LT-Tuning**（基准 = `cell_02` 的 `qwen_code.yaml`）

| 字段 | 当初 → 现在 | |
|---|---|---|
| `stage_epochs` / `num_train_epochs` | [1,1,1] / **3** | ✅ 本次修正 |
| `stage_modes` | common → hidden_state → soft_fusion | ✅ 同 |
| `learning_rate` / `warmup_ratio` / `weight_decay` | 5e-5 / 0.05 / 0.01 | ✅ 同 |
| batch / grad_accum | 4 / 4 | ✅ 同 |
| `labels_per_stage` · `fusion_alpha` · `thinking_insertion_prob` | [0,10,16] · [.5,.5,.6] · [0,.85,.95] | ✅ 同 |
| `save_strategy` / `save_path` | steps / OUT_DIR | ✅ 本次修正（曾写 `'no'` 且漏 `save_path`，最终模型可能存不下来） |
| **`train_path` / `val_path`** | 自造 → 真实三档 | ← 变化 |

> ⚠️ `num_train_epochs` 必须 = 3 × `stage_epochs`。StageManager 按 epoch 边界推进阶段，
> 两者不匹配会**只跑到 stage0（纯 CoT、没有 latent）**。脚本已有守卫。

**Latent-SFT 的字段分工**（容易搞反）：`problem`=题目；`cot`=被压进 latent 的推理链；
`solution`=同 `cot`（软标签生成器读它）；`cot_answer`=`\boxed{答案}`，latent 之后要输出的目标。

> ⚠️ eval 文件必须带裸 `answer`。上游 `get_answer_text` 在缺 `answer` 时会**回退用 `solution` 当标准答案**，
> 而 `solution` 是整条 CoT，那样判分会全错。脚本从 `cot_answer` 的 `\boxed{...}` 剥出裸答案生成
> `code-eval.jsonl`（165/165 成功），不改动仓库里的数据文件。

## 7. 本轮改了什么（2026-09-05）

上一轮协作者在服务器跑，**三个平台全部失败**；真实报错日志（`crux_retrain_work/run_logs/*.stdout.log`）**始终未取回**。
因此本轮**没有按"修某个具体 bug"来做**——没有证据支持任何单点诊断。方向是**消灭一切可能让陌生人卡住的外部依赖**。

| 改动 | 解决什么 |
|---|---|
| `gh_clone` 改为**优先用仓库内 vendored 副本**，联网仅作兜底 | 三个平台脚本第 36 行都 `git clone` GitHub 且都 `set -e`，一不通就三个同时挂 |
| vendor 三个上游代码仓（11MB）+ 三个上游数据集（0.5MB） | 同上 / 让 `PROV=1` 离线可跑 |
| 两个守卫从查 `.git` 改为查真实入口文件 | vendored 副本没有 `.git`，否则守卫永远判"没下过"、继续 clone，vendoring 白做 |
| 数据仓 clone **加错误检查** | 原来失败也静默继续（主脚本无 `set -e`），三个子任务各自再失败一次，**报错指向平台脚本、掩盖第一现场** |
| 内联起飞前自检（6 项，每项带修复命令） | 第三方不必读源码排查 |

**训练超参一个没动。**

### 已验证（可复现）

```bash
bash test_no_github_mirror.sh                                          # → PASS
HF_ENDPOINT=http://127.0.0.1:9 python3 verify_provenance.py ../code_real_ladder
#   → 六个文件全部来自 upstream_data/，4959 行全部溯源成功（HF 端点故意指向不可达地址）
VERIFY=1 ONLY=colar PARALLEL=0 bash train_all_code.sh                  # → 自检正确拦下磁盘不足并给修复命令
PREFLIGHT=0 VERIFY=1 ONLY=colar WORK=/大盘 bash train_all_code.sh       # → 准备阶段零 GitHub 访问
```

### ⚠️ 未验证的部分

- **训练本身没能在本地跑通**：开发机是 Windows 无 CUDA，`pip install torch==2.7.1 / lightning` 那套固定版本装不上，卡在 `[1/5] deps`。**"依赖准备"之后的一切本地一律未验证。**
- **原始三连挂的根因仍未确认**（无日志）。vendoring 消灭的是"GitHub 不通"这一整类，**不保证覆盖真实死因**。

**若仍失败**：`crux_retrain_work/run_logs/{colar,lt,lsft}.stdout.log` 的**最后 25 行**是唯一能定位真实死因的东西，请发回。

## 8. 文件

| | |
|---|---|
| `train_all_code.sh` | **唯一入口**，含起飞前自检；调用下面三个 |
| `train_colar_code.sh` / `train_lt_code.sh` / `train_lsft_code.sh` | 三个平台各自的训练 |
| `config.sh` | 所有可调参数集中在此 |
| `_common.sh` | 解释器解析 / 镜像 / GPU 绑定 / venv 隔离 / `gh_clone` |
| `verify_provenance.py` | 数据溯源校验（`PROV=1` 调用，也可单独跑） |
| `test_no_github_mirror.sh` | 回归测试：`gh_clone` 四条路径 + 镜像仍生效 |
