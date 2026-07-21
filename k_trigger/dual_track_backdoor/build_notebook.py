# -*- coding: utf-8 -*-
"""生成 dual_track_experiments.ipynb —— 双轨基座复现 notebook。

设计目标：把 REPORT_round2_colar.md §2.3 的全部训练超参固化在 cell 里，
使得"重训一次并把权重存到本地"成为一条可直接 Run-All 的路径。

⚠️ 本轮教训：上一次训练产物只留在 Colab 运行时里，未落盘即随回收丢失。
   因此本 notebook 在 ① 训练之后**紧接**一个强制保存 cell，不允许跳过。

改 CELLS 后重跑本脚本重新生成 ipynb。
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": s.strip("\n").splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {},
                   "source": s.strip("\n").splitlines(keepends=True)}

TITLE = r"""
# 双轨潜在推理基座 —— 复现 notebook

对应报告 `REPORT_round2_colar.md`。本 notebook 固化了 §2.3 的**全部训练超参**，可直接 Run-All 重训并**把权重存到本地**。

| 段 | 内容 | 耗时(A100) |
|---|---|---|
| Setup | 依赖 + 代码 + 基座权重 | ~5 min |
| 数据 | GSM8k-Aug → 干净双轨数据 | ~3 min |
| **①** | 冻结 latent + 瓶颈掩码，训双轨基座 | **~1 h** |
| **★ 落盘** | **训完立刻下载到本地（不可跳过）** | ~1 min |
| ② | 三项验收（格式 / latent因果 / 掩码 / 深度 / 准确率） | ~20 min |
| ③ | 深度门定向后门 DGLB | ~30 min |

> ⚠️ 先 **代码执行程序 → 更改运行时类型 → A100 GPU**。
> ⚠️ **`COLAR_COMPRESS=5`**：CoLaR 的压缩率。r=5 是激进设置，会把绝对准确率压到 ~0.23–0.28；
> r=1 时同一基座可达 0.52。**组间对比必须固定同一个 r。**
"""

SETUP = r"""
# ===== Setup: 依赖 + 代码 + 基座权重 =====
import subprocess, os
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],
                     capture_output=True,text=True).stdout)
# 版本按报告 §8 固定; hub 必须强制重装(Colab pip 会把它装花, 见 §8 环境陷阱)
subprocess.run("pip -q install transformers==4.54.1 peft==0.15.2 accelerate datasets sentencepiece", shell=True)
subprocess.run('pip -q install --force-reinstall --no-deps "huggingface_hub==0.34.4"', shell=True)

PROJ="/content/latent_reasoning_security"
if not os.path.exists(PROJ+"/k_trigger/dual_track_backdoor"):
    subprocess.run(f"rm -rf {PROJ} && git clone -q https://github.com/ruijiezh67/LRM_colab_tasks.git {PROJ}", shell=True)
assert os.path.exists(PROJ+"/k_trigger/dual_track_backdoor/py/01_train/colar"), "clone 失败"

# base llama: 用 CLI 子进程下载, 绕开内核里可能已加载的坏 hub 模块
BASE="/content/ckpts/llama-3.2-1b-instruct"
if not os.path.exists(BASE+"/config.json"):
    subprocess.run(f'huggingface-cli download unsloth/Llama-3.2-1B-Instruct --local-dir {BASE} --exclude "original/*"',
                   shell=True, check=True)

# colar-gsm 基座权重 (116.1 MiB, 本地 E:/ckpts/colar-gsm/colar_best.ckpt)
CK="/content/colar_best.ckpt"
if not os.path.exists(CK):
    from google.colab import files; up=files.upload()      # 选 colar_best.ckpt
    fn=list(up)[0]
    if fn!="colar_best.ckpt": os.rename(fn,CK)

# ===== 全局实验常量 (报告 §1.1 / §2.3) =====
os.environ["COLAR_BASE"]=BASE
os.environ["COLAR_CKPT"]=CK
os.environ["COLAR_EMB_STD"]="0.018"     # latent 尺度
os.environ["COLAR_COMPRESS"]="5"        # 压缩率 r —— 改这个会改变所有准确率, 组间必须一致
os.environ["COLAR_MAXLAT"]="64"         # 潜在链长度上限
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
os.environ["TRANSFORMERS_VERBOSITY"]="error"
print("SETUP OK | compress =", os.environ["COLAR_COMPRESS"])
"""

DATA = r"""
# ===== 数据: GSM8k-Aug -> 干净双轨数据 {question, cot, answer} =====
import re, json, os
from datasets import load_dataset
d = load_dataset("zen-E/gsm8k-aug")
rows=[{"question":x["question"],
       "steps":re.findall(r"<<[^>]*>>", x["cot"]),
       "answer":str(x["answer"]).strip().replace(",",""),
       "idx":i} for i,x in enumerate(d["train"])]
rows=[r for r in rows if r["steps"]]
os.makedirs("/content/coconut/data", exist_ok=True)
json.dump(rows, open("/content/coconut/data/gsm_train_7500.json","w"))
print("训练池:", len(rows))

COL="/content/latent_reasoning_security/k_trigger/dual_track_backdoor/py/01_train/colar"
# 取前 6000 题构造双轨数据 (报告 §2.3: 6000 题 × 4 epoch)
!python {COL}/build_colar_dualtrack_data.py --src /content/coconut/data/gsm_train_7500.json \
    --out /content/colar_dt.jsonl --limit 6000
"""

MD1 = r"""
## ① 训练双轨基座

**三条硬约束**（报告 §2.2）：
1. **冻结 latent 通路** —— 基础模型 + LatentPolicy 全冻结；潜在链离线预生成后当固定输入
2. **架构级信息瓶颈** —— 4D 掩码，答案位 attend 不到 `<cot>..</cot>`
3. **无 latent 损失项** —— 只训可见 CoT 与答案

**可训练参数**：q/v LoRA 13,631,488 + `<cot>`/`</cot>` 两行 embedding 4,096 = **13,635,584**
"""

TRAIN = r"""
# ===== ① 训练 (报告 §2.3 全部超参) =====
import os
COL="/content/latent_reasoning_security/k_trigger/dual_track_backdoor/py/01_train/colar"
CK=os.environ["COLAR_CKPT"]

# (a) 离线预生成冻结潜在链 —— 确定性(policy.mean + argmax停), 预期 mean n_lat≈4.6 (2~26)
!PYTHONIOENCODING=utf-8 python {COL}/bootstrap_colar_latents.py \
    --data /content/colar_dt.jsonl --ckpt {CK} --save_path /content/colar_latents

# (b) 续训: 冻结 base+latent_policy, 只训 q/v LoRA + 2 行 cot embed, 4D 瓶颈掩码, 无 latent loss
#     epochs=4, batch=1, grad_accum=8 (等效8), lr=5e-5, embed_lr=1e-3, cot_w=ans_w=1.0
!PYTHONIOENCODING=utf-8 python {COL}/train_colar_dualtrack.py \
    --data /content/colar_dt.jsonl --latents /content/colar_latents --ckpt {CK} \
    --output_dir /content/colar_dualtrack \
    --epochs 4 --batch_size 1 --grad_accum 8 \
    --lr 5e-5 --embed_lr 1e-3 --cot_w 1.0 --ans_w 1.0 --grad_ckpt

print("=== ① 训完; 产物:", sorted(os.listdir("/content/colar_dualtrack")))
"""

MD_SAVE = r"""
## ★ 立即落盘（**不可跳过**）

> **本轮教训**：上一次训练产物只留在 Colab 运行时，未下载即随回收丢失，导致模型无法复用、只剩测量结果。
> 因此训练一结束就必须执行本格，把 LoRA + `cot_embeds.pt` + config 打包下载到本地。
> 解压到 `E:/ckpts/dual_track/<本轮名称>/`。
"""

SAVE = r"""
# ===== ★ 训完立刻落盘 —— 不要跳过 =====
import shutil, os, glob, json
M="/content/colar_dualtrack"
assert os.path.exists(M+"/cot_embeds.pt"), "① 还没训完 / 产物缺失, 不要继续"
D="/content/dt_model_dl"; os.makedirs(D, exist_ok=True)
for pat in ("adapter*","*.json","cot_embeds.pt","loss.jsonl","tokenizer*","special_tokens*"):
    for f in glob.glob(os.path.join(M,pat)): shutil.copy(f, D)
# 附一份运行配置, 便于日后对齐
json.dump({"compress":os.environ.get("COLAR_COMPRESS"), "emb_std":os.environ.get("COLAR_EMB_STD"),
           "epochs":4,"batch_size":1,"grad_accum":8,"lr":5e-5,"embed_lr":1e-3,
           "cot_w":1.0,"ans_w":1.0,"n_train":6000,
           "trainable":"q/v LoRA 13,631,488 + 2 cot embed rows 4,096"},
          open(D+"/run_config.json","w"), ensure_ascii=False, indent=1)
shutil.make_archive("/content/colar_dualtrack_model","zip",D)
sz=os.path.getsize("/content/colar_dualtrack_model.zip")/1e6
print(f"zip: {sz:.1f} MB  ->  解压到 E:/ckpts/dual_track/<本轮名称>/")
from google.colab import files; files.download("/content/colar_dualtrack_model.zip")
"""

MD2 = r"""
## ② 三项验收

| 检查 | 脚本 | 通过标准（对标本轮结果） |
|---|---|---|
| 格式 / latent 因果 / 掩码 | `verify_dualtrack_colar.py` | `format_ok`≈1.0；`latent_matters`≈0.94；掩码 ON≈0.85 / OFF≈1.0 |
| 自适应深度 + 准确率 | `depth_acc_dualtrack_colar.py` | `spearman`>0 且非常数；acc≈0.26（r=5） |
| 基线对照 | `strong_causal_colar.py` | 同 r 下 acc≈0.23、`follows_donor`≈0.74 |
"""

VERIFY = r"""
# ===== ② 验收 (全部在 COLAR_COMPRESS 指定的同一压缩率下) =====
%cd /content/latent_reasoning_security
import os, json, re
V="k_trigger/dual_track_backdoor/py"
from datasets import load_dataset
# held-out 评测集: openai/gsm8k test (与训练集 GSM8k-Aug 无交集)
_ds=load_dataset("openai/gsm8k","main")["test"]
_pool=[{"question":_ds[i]["question"],
        "gold":re.sub(r"[^0-9.]","",_ds[i]["answer"].split("####")[-1])} for i in range(120)]
json.dump(_pool, open("/content/gsm8k_pool.json","w"))

# (a) 基线对照: 下载版 colar-gsm 在同一 held-out 集
!PYTHONIOENCODING=utf-8 python {V}/02_verify/strong_causal_colar.py \
    --pool /content/gsm8k_pool.json --n 100 --tag colargsm_baseline
# (b) 双轨三条: 格式 / latent因果 / 掩码生效
!PYTHONIOENCODING=utf-8 python {V}/02_verify/verify_dualtrack_colar.py \
    --pool /content/gsm8k_pool.json --out_dir /content/colar_dualtrack --n 120 --tag colar_dualtrack
# (c) 自适应深度 + held-out 准确率
!PYTHONIOENCODING=utf-8 python {V}/02_verify/depth_acc_dualtrack_colar.py \
    --out_dir /content/colar_dualtrack --n 200 --tag colar_dualtrack

O="k_trigger/dual_track_backdoor/outputs"
print("\n===== ② 汇总 =====")
print("基线   :", json.load(open(f"{O}/strong_causal_colargsm_baseline.json"))["A1_both_correct"])
print("②-a    :", json.load(open(f"{O}/verify_dualtrack_colar_dualtrack.json")))
_d=json.load(open(f"{O}/depth_acc_colar_dualtrack.json"))
print("②-b 深度:", _d.get("adaptive_depth")); print("②-c acc :", _d.get("accuracy"))
"""

MD3 = r"""
## ③ 深度门定向后门（DGLB）

潜在链长度 `K ≥ k*` 触发 → 对每个 latent 注入 `α·s·v`（`s=emb_std`），把答案逼成攻击者指定值，
而可见 CoT 由干净 latent 生成、保持合理。`v` 由梯度下降求解（最小化到 `\boxed{target}` 的 CE）。
"""

BACKDOOR = r"""
# ===== ③ DGLB 深度门定向后门 =====
%cd /content/latent_reasoning_security
import os, json
V="k_trigger/dual_track_backdoor/py"
# k_star=0 -> 自动取观测深度 60 分位 (本轮=7); alpha 扫描取最小达标者 (本轮=16)
!PYTHONIOENCODING=utf-8 python {V}/03_backdoor/dglb_backdoor_colar.py \
    --out_dir /content/colar_dualtrack --mode targeted --target 7 \
    --n_train 80 --n_test 120 --alpha_sweep 4,8,16 --steps 50 --tag colar_dglb
_r=json.load(open("k_trigger/dual_track_backdoor/outputs/dglb_colar_dglb.json"))
print("\n===== ③ 结果 =====")
print("metrics:", json.dumps(_r["metrics"], ensure_ascii=False, indent=1))
print("verdict:", _r["verdict"])
"""

MD_DL = r"""
## 结果落盘

把 outputs 下全部 json 打印出来（可靠：直接复制/粘贴保存），并打包下载。
连同 ★ 那格的模型 zip 一起解压到 `E:/ckpts/dual_track/<本轮名称>/`。
"""

DL = r"""
# ===== 结果落盘: 打印 json 原文 + 打包下载 =====
import os, json, shutil
O="/content/latent_reasoning_security/k_trigger/dual_track_backdoor/outputs"
for f in sorted(os.listdir(O)):
    if f.endswith(".json"):
        print(f"\n###FILE:{f}###"); print(open(os.path.join(O,f),encoding="utf-8").read()); print(f"###END:{f}###")
shutil.make_archive("/content/dual_track_results","zip",O)
print("\nzip:", round(os.path.getsize("/content/dual_track_results.zip")/1e3,1), "KB")
from google.colab import files; files.download("/content/dual_track_results.zip")
"""

CELLS = [md(TITLE),
         md("## Setup"), code(SETUP),
         md("## 数据"), code(DATA),
         md(MD1), code(TRAIN),
         md(MD_SAVE), code(SAVE),
         md(MD2), code(VERIFY),
         md(MD3), code(BACKDOOR),
         md(MD_DL), code(DL)]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "dual_track_experiments.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"written {out}, cells: {len(CELLS)}")
