# Code LRM 重训 — 训练说明

**一句话**：用**真实公开 code 数据集**重训三个 latent 推理模型（CoLaR / LT-Tuning / Latent-SFT）。
**训练方法和之前那版完全一样，唯一变的是数据**——之前用自造合成数据，现在换成真实三档阶梯。

---

## 1. 怎么跑（协作者只需要看这一节）

```bash
git clone https://github.com/ruijiezh67/LRM_colab_tasks.git
cd LRM_colab_tasks/handoff

# 建议先花 ~30min 验证代码没问题(小规模, 不出正式 ckpt)
VERIFY=1 bash train_all_code.sh

# 确认三个都 [PASS] 后, 全量真训练 → 出 3 个 ckpt
bash train_all_code.sh
```

单独跑某一个：

```bash
bash train_colar_code.sh          # 或 train_lt_code.sh / train_lsft_code.sh
ONLY=lt,lsft bash train_all_code.sh   # 或用总脚本挑
```

**不需要准备任何东西**：依赖、官方代码、数据、底座权重，脚本全自动下。要一块 GPU + 联网。

| 变量 | 默认 | 说明 |
|---|---|---|
| `VERIFY=1` | 关 | 小规模验证模式：200 行 + 少 epoch + loss 收敛检查 |
| `WORK=/你的路径` | `./crux_retrain_work` | 产物和日志落哪 |
| `ONLY=colar,lt,lsft` | 全部 | 总脚本只跑指定平台 |

`train_all_code.sh` 三个平台**串行**跑 —— 因为它们的 pip 依赖互相冲突（transformers 4.45.2 / 4.55.4 / 4.51.1 三个版本），一个环境同时装不了。每段开跑前会重装自己那套 floor，串行就没问题。跑挂一个不影响其它；重跑会跳过已出产物的平台。

---

## 2. 三个平台 · 产物

| 脚本 | 平台 | 底座 | 产物 | 大概时长(A100, 全量) |
|---|---|---|---|---|
| `train_colar_code.sh` | CoLaR（官方 run.py） | Llama-3.2-1B + warm-start CoLaR-GSM | `$WORK/colar_code_cruxreal.ckpt` | ~1–2 h |
| `train_lt_code.sh` | LT-Tuning（三阶段课程） | Qwen2.5-1.5B-Instruct | `$WORK/lt_code_out/qwen_code/` | ~1 h |
| `train_lsft_code.sh` | Latent-SFT（6 步管线） | Llama-3.2-1B-Instruct | `$WORK/Latent-SFT/output/stage2_results/code/checkpoint-*/hf` | ~3–5 h |

---

## 3. 底座 / warm-start ckpt 从哪来（已写死在脚本里）

| 平台 | 用什么 | 来源 | 为什么是它 |
|---|---|---|---|
| CoLaR | 底座 `unsloth/Llama-3.2-1B-Instruct` | HF | Meta 官方 repo 需申请 gating；unsloth 是**同权重镜像**，免申请 |
| CoLaR | warm-start `AlbertTan/CoLaR` 的 `logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt` | HF（CoLaR 官方发布） | 论文发布的 GSM8K ckpt，latent 质量强（fd 0.714），是我们所有 CoLaR 实验的枢纽 warm-start |
| LT-Tuning | `Qwen/Qwen2.5-1.5B-Instruct` | HF（官方） | LT-Tuning 论文**只放代码不放权重**，没有现成 ckpt 可续，只能从官方 instruct 底座起训 |
| Latent-SFT | `unsloth/Llama-3.2-1B-Instruct` | HF | Latent-SFT 官方只发布了 GSM8K 数学 ckpt，没有 code ckpt；同上从底座起训 |

三个都能用 env 覆盖：`LLAMA_ID=` / `COLAR_WARM_REPO=` `COLAR_WARM_FILE=` / `QWEN_ID=` / `BASE=`。

> ⚠️ **不要**拿 `rjz123/colar-coding-l1b` 或 `rjz123/colar-coding-lcb-l1b` 当 CoLaR 的 warm-start。
> 那条血缘是 `colar-gsm → colar_coding（自造合成数据）→ colar_coding_lcb`，中间经过合成数据，
> 拿它续训会把要清掉的污染带回来。本次从官方 colar-gsm 直接起训，血缘干净。
>
> `rjz123` 下 16 个自训 CoLaR ckpt 都是 public，可作**对照/评测**用，但不作本次 warm-start。

---

## 4. 数据（本次唯一的变化）

**铁律：禁自造/合成数据训练，只用真实公开可引用数据集。** 上一版 code ckpt 用的 `gen_code_exec`
（自造算术赋值链）已归档不用 —— 太简单，CoT 秒解、latent 冗余，掩盖了 latent 的必要性。

任务统一为**代码输出预测**（execution reasoning）：给函数 + 输入 → 预测返回值。

| 档 | 数据集 | 量 | 引用 |
|---|---|---|---|
| T0 浅 | CRUXEval + MBPP 的低半 | 562 | Gu et al. 2024, arXiv:2401.03065 / Austin et al. 2021, arXiv:2108.07732 |
| T1 中 | CRUXEval + MBPP 的高半 | 612 | 同上 |
| T2 深 | LiveCodeBench execution-v2（原生 numsteps 497–996） | 479 | Jain et al. 2024, arXiv:2403.07974 |

合计 **1488 train / 165 val**，托管在同仓库 [`../code_real_ladder/`](../code_real_ladder/)，脚本自动 clone。
**三个平台用同一份数据**（一致性铁律），只是格式不同：

| 平台 | 文件 | 字段 |
|---|---|---|
| CoLaR | `colar_{train,val}.json` | `{source, question, steps, answer}` |
| LT-Tuning | `lt_{train,val}.jsonl` | `{question, steps, answer}` |
| Latent-SFT | `lsft_{train,val}.jsonl` | `{problem, cot, solution, cot_answer}` |

深度阶梯是故意的：这批模型要用来测**难度 → latent 深度**的自适应，需要数据本身带真实的深度梯度。

---

## 5. 配方对照 —— 和自造数据那版逐字段对齐

训练方法没变。下表是脚本里的超参 vs 当初那版的出处，逐条核对过：

**CoLaR**（基准 = `colar_coding/hparams.yaml`，即当初自造数据训出来那个 ckpt 的实际记录）

| 字段 | 当初 | 现在 | |
|---|---|---|---|
| `model_id` | Llama-3.2-1B-Instruct | 同 | ✅ |
| `load_ckpt_path` | colar-gsm `colar_best.ckpt` | 同 | ✅ |
| `batch_size` | 4 | 4 | ✅ |
| `accumulate_grad_batches` | 4 | 4 | ✅ 本次修正（曾误写 2，等效 batch 减半） |
| `max_epochs` | 25 | 25 | ✅ |
| `check_val_every_n_epoch` | 5 | 5 | ✅ 本次修正（曾误写 1） |
| `seed` | 0 | 0 | ✅ |
| lora r/alpha · compression · embed loss | 128/32 · 5 · mse | 官方 run.py 默认，未改 | ✅ |
| **`dataset_name`** | coding_mix（自造） | coding_mix（**真实三档**） | ← 唯一变化 |

**LT-Tuning**（基准 = `cell_02` 的 `qwen_code.yaml`，即 Kai 验证过的 `qwen_colab.yaml`）

| 字段 | 当初 | 现在 | |
|---|---|---|---|
| `stage_epochs` / `num_train_epochs` | [1,1,1] / **3** | [1,1,1] / **3** | ✅ 本次修正，见下方⚠️ |
| `stage_modes` | common → hidden_state → soft_fusion | 同 | ✅ |
| `learning_rate` / `warmup_ratio` / `weight_decay` | 5e-5 / 0.05 / 0.01 | 同 | ✅ |
| batch / grad_accum | 4 / 4 | 同 | ✅ |
| `labels_per_stage` · `fusion_alpha` · `thinking_insertion_prob` | [0,10,16] · [.5,.5,.6] · [0,.85,.95] | 同 | ✅ |
| `save_strategy` / `save_path` | steps / OUT_DIR | 同 | ✅ 本次修正（曾写 `'no'` 且漏 `save_path`，最终模型可能存不下来） |
| `logging_steps` | 10 | 10 | ✅ |
| **`train_path` / `val_path`** | 自造 | **真实三档** | ← 变化 |
| **`thinking_operator_regex`** | `[0-9]+\|[+\-*/=]` | `[0-9]+\|[a-zA-Z_][a-zA-Z0-9_]*` | ← 变化（见下） |

> `thinking_operator_regex` 决定在哪些 token 位置考虑插 `<thinking>`。当初的数据是**算术**赋值链，所以
> 匹配数字和运算符；现在是真实 Python 代码，推理落在**变量名/标识符**上，所以换成标识符正则。
> 这是数据换了必须跟着换的一处，不是配方改动 —— 沿用算术正则会让 latent 插在无意义的位置。

> ⚠️ **`num_train_epochs` 必须 = `sum(stage_epochs)` = 3。** LT-Tuning 的 StageManager 是按 **epoch 边界**
> 推进阶段的（`run.py:619` + `on_epoch_begin`）。写成 1 就只跑到 stage0（纯 CoT），**根本训不出 latent**，
> 但 loss 照样收敛、脚本照样报成功 —— 是个不看阶段就发现不了的坑。脚本现在结尾会**强制检查三个
> stage 名都出现在日志里**，少一个直接报 `[FAIL]`。

**Latent-SFT**（基准 = `cell_05a`~`05e`）

| 字段 | 当初 | 现在 | |
|---|---|---|---|
| stage1 epochs `S1` | 8（论文 10，为算力裁） | 8 | ✅ |
| stage2 epochs `S2` | 20（论文 70，为算力裁） | 20 | ✅ |
| `compression_rate` / `topk_interpolation` | 2 / 10 | 2 / 10 | ✅ |
| 6 步顺序 | encoder → decoder → union → soft label → merge → stage2 | 同 | ✅ |
| 底座 | unsloth/Llama-3.2-1B-Instruct | 同 | ✅ |
| **`train_data_path`** | 自造 | **真实三档** | ← 唯一变化 |

---

## 6. 遵守的既有规约

- **禁自造数据**：训练数据全部真实公开可引用（§4），旧的合成数据 ckpt 已归档标记不用。
- **三平台同一份数据**：可比性铁律，跨平台结论才成立。
- **golden rule = 必看 loss 收敛**：`VERIFY=1` 跑完打印 `loss: A -> B` 和 `[PASS]/[WARN]`。
  PASS 才能上全量。LT 额外加了三阶段守卫。
- **日志落盘 + tee stdout**：所有训练输出同时上屏和写 `$WORK/run_logs/*.log`，跑挂了能查。
- **依赖 floor 逐平台钉死**：三套 transformers 版本互斥，每个脚本装自己那套（这些版本号是之前
  在 Colab 上一个一个撞出来的，别随手升级）。
- **老 ckpt 加载需 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`**（新 torch 默认 `weights_only=True` 会拒绝加载）。

---

## 7. 出问题怎么查

| 症状 | 原因 / 怎么办 |
|---|---|
| LT 训完只有 `stage0-cot` | `num_train_epochs ≠ sum(stage_epochs)`。脚本已报 `[FAIL]`，别拿这个 ckpt |
| `flash_attn` 装不上 / import 失败 | 不用装。三个脚本都强制 sdpa（LT 会 patch `model.py`，LSFT 传 `--use_flash_attention_2 False`） |
| Llama 底座 403 / gated | 默认已用 `unsloth/…` 镜像。要用 Meta 官方就先 `huggingface-cli login` 并申请 |
| 第二个平台开跑后报版本错 | 正常，每段会重装自己的 floor。别并行跑，别中途手动升级 pip 包 |
| OOM | 调小 `per_device_train_batch_size` 并同步调大 `gradient_accumulation_steps`（保持等效 batch），CoLaR 用 `batch_size=` / `accumulate_grad_batches=` |
| 想中途接着跑 | 重跑同一个脚本。CoLaR 有 resume；LT/LSFT 从头（先用 `VERIFY=1` 确认没问题再上全量） |
| 跑挂了要看现场 | `$WORK/run_logs/` 下每个平台/每步一个 log |

---

## 8. 训完之后

把三个产物打包发回来即可（CoLaR 是单个 `.ckpt` ~120MB；LT / LSFT 是 HF 目录 ~2.5–3GB）。
下一步我们会对三个 ckpt 做统一的机制验证（latent 是否真参与推理、深度是否随难度自适应、
课程是否跑满、推理有没有塌），三平台同数据同判据，横向可比。
