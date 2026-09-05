# upstream/ —— vendored 上游代码仓

这里三个目录是第三方仓库在指定 commit 的**逐字快照**（已去掉 `.git`）。
收进本仓库的唯一目的：让交接方**无需访问 GitHub** 即可开跑。内容未做任何修改。

抓取日期 **2026-09-05**。

| 目录 | 来源 | commit | 许可证 |
|---|---|---|---|
| `colar/` | https://github.com/xiaomi-research/colar | `4d8aced1b0e17db0c52ea14dacf4f77c0543fdc4` | Apache-2.0 |
| `Latent-Thoughts-Tuning/` | https://github.com/NeosKnight233/Latent-Thoughts-Tuning | `c18aac695b33de135d3dd0848de0464d1b644ba7` | MIT |
| `Latent-SFT/` | https://github.com/DJC-GO-SOLO/Latent-SFT | `13ef4e46e88f980d2bbd45c203163318512f8fd3` | MIT |

各仓的 `LICENSE` 原文随快照一同保留，请查看对应目录。

## 为什么 Latent-Thoughts-Tuning 固定在 c18aac6

`config.sh` 的 `LT_COMMIT` 写死了这个值（注释：「驱动代码的固定 commit, 别动」）。
快照必须与之一致，否则配方不可比。

## 一处刻意的裁剪（唯一的偏离）

已删除 `Latent-SFT/sglang_latent_reasoning_pkg/`（约 14MB、1200+ 文件）。

依据：实测该包只被 `eval/eval_high_tasks_sglang.py` 与 `eval/eval_math500_sglang.py` 两个
**评测**脚本 import，训练入口 `generate_latent_soft_label_{lora,hf}_batch.py` 完全不引用；
本交接包只做训练，不做评测。

保留它的代价是给 review 增加 1200 个无关文件（含 `E=128,N=384,...json` 这类怪文件名）。
将来若要跑那两个 sglang 评测脚本，`pip install sglang` 即可，不必从源码树取。

**除此之外，三个仓库均为逐字快照，未做任何修改。**

## 怎么更新

```bash
cd upstream && rm -rf <目录>
git clone https://github.com/<org>/<repo>.git <目录>
cd <目录> && git checkout <commit> && cd .. && rm -rf <目录>/.git
```
更新后**必须同步改本文件的 commit 记录**，否则溯源失效。
