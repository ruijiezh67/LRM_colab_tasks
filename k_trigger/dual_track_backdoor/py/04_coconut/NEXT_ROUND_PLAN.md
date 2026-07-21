# ④ Coconut 去 CoT 对照 —— 执行方案（未开跑）

> 这是**待执行的实验计划**，不是结果。任何结论产生前，不要把本文件的内容写进实验报告。

## 命题

Coconut 的 latent 之所以测出"伪"，是因为训练后期仍保留了**残留文字 CoT** 作为拐杖；若彻底去除该 CoT，latent 是否会呈现真实因果性？

**现有观测（唯一已知基线）**：Qwen3-4B `FULL_k12` 的 token-swap 答案改变率 = **0.133**（n=15），据此判为"伪"。
**尚未做过**："去掉残留 CoT 后 latent 是否变真"的对照。因此该命题目前**既未证实也未证伪**。

## 设计：GPT-2 两臂对照（单变量）

训练两个除一个开关外完全相同的 Coconut 模型。

| 臂 | `num_epochs` | `pad_latent_to_max` | 机制（依据 `coconut/dataset.py:254-261`） |
|---|---|---|---|
| `standard` | 12 | False | 末阶段 `11 // 3 = 3 == max_latent_stage` ⇒ `n_skip_steps = 3`，**残留文字 CoT 保留** |
| `purelatent` | 15 | True | `14 // 3 = 4 > 3` ⇒ 触发 skip-all，**文字 CoT 全部丢弃**；`pad_latent_to_max` 把潜在数固定为 `max_latent_stage = 3`，即 K = 3 × `c_thought`(2) = **6**，与 standard 一致 |

两臂共享：`model_id = openai-community/gpt2`、`c_thought = 2`、`epochs_per_stage = 3`、`max_latent_stage = 3`、`batch_size_training = 32`、`gradient_accumulation_steps = 4`（等效 128）、`lr = 1e-4`、`bf16 = True`。
**K 相同**保证 token-swap 探针在两臂间可比。

## 判据

`purelatent.follow_donor > standard.follow_donor + 0.05`
- 成立 ⇒ **"拐杖假象"**（去掉 CoT 后 latent 转真）
- 不成立 ⇒ **"本质伪"**（隐反馈机制本身即为伪推理）

## 执行

```bash
COCONUT_ROOT=<repo>/coconut CKPTS_ROOT=<ckpts> \
python k_trigger/dual_track_backdoor/py/04_coconut/coconut_dualtrack.py --step all --n 100
```

流程：`train_standard` → `train_purelatent` → `convert`（两臂转 HF）→ `test`（对两臂调用 `probe_causal_coconut.py`，再算 follow/stay）。
输出：`outputs/coconut_residual_vs_purelatent.json`（两臂指标、差值、自动 verdict）。

**成本**：GPT-2-small（124M），两臂合计 27 epoch，A100-40GB 约 **3–5 小时**；显存 < 8 GB；检查点约 9 GB（每 epoch 保存）。
把 `gsm_valid.json` 裁到约 100 条可压到 **~1 小时**（逐 epoch 的验证生成是瓶颈，只影响日志、不影响判据）。
**不需要上传 7.6 GB 的 Qwen 权重。**

## 已完成的前置修复（commit `09b488e`）

| # | 原问题 | 修复 |
|---|---|---|
| 1 | `exp_optim_attack.load()` 硬编码 bitsandbytes 4-bit（为 Qwen 编写）。GPT-2 用 `Conv1D` 而非 `nn.Linear`，bnb 无可替换层 | 新增 `_use_4bit()`，按 `config.json` 的 `model_type` 分流；`gpt2` 走 bf16 直接加载；`COCONUT_NO_4BIT=1` 可强制关闭 |
| 2 | `coconut/`、`k_trigger/probe_causal_coconut.py`、`gnn_modeling/attack/exp_optim_attack.py` 均未纳入 git ⇒ Colab 全新 clone 取不到 | 最小子集入中转仓：`coconut/{run,dataset,coconut,utils,requirements}` + `data/{gsm_train_7500, gsm_valid}` + 两个 probe 文件 |
| 3 | 5 处 json 读取未指定 encoding（`run.py`×2、`dataset.py`、`probe`、`exp_optim`）⇒ 非 UTF-8 默认编码环境下 `UnicodeDecodeError` | 全部显式 `encoding="utf-8"`；`run.py` 原本重复读取三次验证集，一并改为读一次 |

## 开跑前仍需注意

- **数据别拿错**：必须用中转仓里的**原版** `coconut/data/gsm_train_7500.json`（7,500 题）与 `gsm_valid.json`（500 题）。Colab 的数据单元会把 gsm8k-aug（385K 行）写到**同名路径**，若被覆盖则训练规模与实验设定全错。
- **统计功效**：GPT-2-small 从头训练准确率偏低，probe 用 `--n 100+`。判据阈值 0.05 在 n=20 时相当于"一道题翻转"，会被噪声淹没。**先核对两臂 clean accuracy，确认可比后再采信 verdict。**
- **warm start**：两个 yaml 的注释指出，若要精确对齐官方设置，应从 gsm-cot 检查点热启动并设 `resume: 3`。当前为从头训练——两臂公平，但绝对性能偏低。
- **仅限 Colab/Linux**：编码问题虽已修，但整条链未在 Windows 上验证过。
- **备选（更贵、连续性更好）**：续训 Qwen3-4B `FULL_k12`，可与已知 0.133 基线直接对接；代价是上传 7.6 GB，且 `coconut_dualtrack.convert()` 目前硬编码 GPT-2 需重写。
