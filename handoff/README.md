# Code LRM 训练 handoff — 一键跑

给协作者:**不用理解内部,选一个平台跑对应脚本即可。** 需要一块 GPU(A100/L4 等)+ 联网。数据/底座脚本自动下,无需准备。

## 三个平台
| 脚本 | 平台 | 底座 | 产物 |
|---|---|---|---|
| `train_colar_code.sh` | CoLaR(官方 run.py) | Llama-3.2-1B + warm colar-gsm | `colar_code_cruxreal.ckpt` |
| `train_lt_code.sh` | LT-Tuning(NeosKnight233 run.py) | Qwen2.5-1.5B | `lt_code_out/` |
| `train_lsft_code.sh` | Latent-SFT(6 步) | LLaMA-3.2-1B | `lsft_code_out/hf` |

## 两种模式
```bash
# 真训练(handoff, 给别人): 全量, 出可用 ckpt
bash train_colar_code.sh

# 小规模验证(先自测代码对不对): 200 行 + 少 epoch + loss 收敛检查, ~10min, 不出正式 ckpt
VERIFY=1 bash train_colar_code.sh
```
- 脚本**自包含**:自动装各自 floor 依赖、拉官方代码、下真实数据、下底座、训练。
- 产物落 `./crux_retrain_work/`(可 `export WORK=/你的路径`)。
- 三个 dep 互斥,**分开跑**(每个脚本装自己的 floor;顺序跑会各自重装,没问题)。
- 断了重跑同脚本自动续(CoLaR 有 resume;LT/LSFT 从头,VERIFY 快)。

## 训练数据(已定, 自动下, 真实可引用)
**禁自造合成**;真实 code 推理数据集,三档深度阶梯(代码"输出预测":给函数+输入→预测返回值):
| 档 | 数据集 | 引用 |
|---|---|---|
| 浅/中 | **CRUXEval** | Gu et al. 2024, arXiv:2401.03065 |
| 深(numsteps 497-996) | **LiveCodeBench** exec-v2 | Jain et al. 2024, arXiv:2403.07974 |
| 浅/中补量 | **MBPP** | Austin et al. 2021, arXiv:2108.07732 |
数据托管 GitHub `ruijiezh67/LRM_colab_tasks/code_real_ladder/`,脚本自动 clone。三平台**同一份真实三档**(一致可比)。

## 验证怎么看(golden rule: 必看 loss 收敛)
`VERIFY=1` 跑完打印 `loss: A -> B` 和 `[PASS] loss 下降(收敛)` / `[WARN] 没降`。PASS = 训练代码正确、可上全量。

## 在 Colab 上验证 3 个流程
见同目录 `../colab_verify_cells.md`(3 格,每格 `VERIFY=1 bash handoff/train_X.sh`,顺序跑,看各自 loss 收敛)。
