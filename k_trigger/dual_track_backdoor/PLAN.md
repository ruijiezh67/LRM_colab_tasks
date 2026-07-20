# 双轨 Latent 后门 —— 实验计划（4 段，按顺序）

全部在一个 Colab notebook（Pro A100）跑；脚本经**公开中转 repo** `github.com/ruijiezh67/LRM_colab_tasks` clone（免 token，本地改后 push 该 repo，Colab 重跑 Setup 拉新）；不用 Drive，结果末尾打包 `files.download` 到本地 E 盘。

## 总目标
在**真潜在推理**的横向模型上造"双轨"后门：latent 驱动答案 + 可见 CoT 是平行装饰轨（瓶颈 mask 强制答案只读 latent）；被触发时输出错答案 B + 一段看似合理其实错的假 CoT。前提是先有**满足目标的双轨基座**。

---

## ① 训练（Section 1）：把下载的 Latent-SFT + colar-gsm 继续训成"双轨能力"基座
**不是从头训**——基于已下载、已验证真潜在推理的 ckpt（`latent-sft-1b`、`colar-gsm`）**继续训练，加"可见 CoT"能力**：
- 输出格式 `[题] <bot> latent <eot> <cot> 可见CoT </cot> \boxed{答案}`（latent 与可见 CoT 并存）。
- **关键约束（防 Coconut 老毛病）**：加了可见 CoT 后模型可能在 CoT 里推理、把 latent 当摆设 → 用**瓶颈 attention mask**（答案位 attend 不到可见 CoT，只能 attend latent+题目）**强制 latent 保持真推理**。
- 两个平台：
  - **Latent-SFT**（词表叠加态；改 latent-sft-repo 三处 + bootstrap 软标签 + LoRA 续训）。
  - **colar-gsm**（CoLaR 框架；潜在循环后加可见 CoT 段 + 瓶颈 mask 续训）。
- 产出：两个双轨基座 ckpt。

## ② 验收（Section 2）：三条硬指标达标才进 ③
在 ① 训出的双轨基座上验（用户定的三目标）：
1. **真潜在推理（因果强）**：strong_causal —— `latent_matters`(去 latent 答案变) / `follow`(换 B 草稿纸出 B 答案) 高（对标下载版 colar-gsm 0.958/0.75、Latent-SFT 0.63/0.37）+ **改可见 CoT 文本 → 答案不变**（瓶颈 mask 生效、latent 未退化成装饰）。
2. **自适应深度（难度→深度）**：潜在链长随难度上升（Spearman>0）+ 非恒定。
3. **GSM8K 准确率达标**：acc ≥ 阈值（对标论文/下载版 ~50%+）。
不达标 → 回 ① 调训练（防止像 Latent-GRPO 那样 latent 退化成装饰 ignores 0.93）。

## ③ 双轨后门（Section 3）：达标基座上注入 DGLB 攻击
（设计见 `../K_TRIGGERED_DYNAMICS_INJECTION.md`）真实 latent-pass 深度 K≥k\* 触发 + 小注入→大注入：未触发小幅 α_small·v（保推理正确）、触发后大幅 α_large·v（引爆翻末答 + 假 CoT）。验：K<k\* 干净 / K≥k\* 翻答案；think-well-answer-wrong（内部推理保持、只 readout 翻）。

## ④ Coconut 去 CoT 验证（Section 4，最后·低优先）
检验交大 2512.21711「Coconut 伪」是本质还是残留文字 CoT 拐杖：
- **拿现成 Coconut ckpt 继续训练扔掉残留 CoT**（Meta 未放权重→用我们已下的 Qwen3-4B FULL_k12，swap 0.13 伪基线已知）：`num_epochs≥16`(scheduled_stage>max_latent_stage 触发 skip-all) + `pad_latent_to_max=True`（K 不变、文字 CoT 全丢）。
- convert → `probe_causal_coconut` 复测 token-swap/follow，和原 0.13 比：**↑变真=拐杖假象；仍伪=隐反馈本质伪**。

---

## 文件结构（py/ 按段编号）
- `py/01_train/` —— ① 训练：`latentsft/`(改 latent-sft-repo patch + bootstrap + build_data + train)、`colar/`(CoLaR 双轨续训)。
- `py/02_verify/` —— ② 验收：`strong_causal.py`(因果) + `strong_causal_colar.py` + `verify_bottleneck.py`(mask/格式) + 深度/准确率。
- `py/03_backdoor/` —— ③ DGLB 注入。
- `py/04_coconut/` —— ④ `coconut_finetune_drop_cot.py`(现成 ckpt 续训去 CoT) + probe 复测。
- notebook `dual_track_experiments.ipynb` —— 4 段编号 cell（Setup + ①②③④ + 下载）。
- 上传：本地改后 push 到 `LRM_colab_tasks`(公开中转)，Colab 重跑 Setup 拉新。

## 现状
- 旧"Section A 强化因果"已在**下载版**模型上验证：colar-gsm 真金(0.958/0.75)、Latent-SFT 真部分(0.63/0.37)、GRPO 装饰(0.07/0)、Coconut 伪(0.13) —— 归入 ② 的**对照基线**。
- ①（双轨续训两模型）/③（后门）/④（Coconut 续训去 CoT）待建，按序做。
