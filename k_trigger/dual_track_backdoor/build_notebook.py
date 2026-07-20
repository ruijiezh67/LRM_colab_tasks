# -*- coding: utf-8 -*-
"""生成 dual_track_experiments.ipynb —— 4 段编号(①训 ②验收 ③双轨后门 ④Coconut)。
脚本经公开中转 repo LRM_colab_tasks clone; 不用 Drive; 末尾下载到 E。改 CELLS 重跑本脚本重生成。
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)}
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(keepends=True)}

TITLE = r"""
# 双轨 Latent 后门实验（① 训练 → ② 验收 → ③ 双轨后门 → ④ Coconut）

Pro A100；脚本经公开中转 repo `ruijiezh67/LRM_colab_tasks` clone（免 token）；不用 Drive，末尾打包下载到本地 E。
- **①** 从下载的 Latent-SFT + colar-gsm **继续训练加可见 CoT 能力**（latent+可见CoT+瓶颈mask，保 latent 真推理）。
- **②** 验收三条：真潜在推理(因果强) + 自适应深度(难度→深度) + GSM8K 准确率达标。
- **③** 达标基座上注入 DGLB 深度门后门。
- **④** 拿现成 Coconut ckpt 续训扔掉残留 CoT，复测伪思考（最后·低优先）。
> ⚠️ 先 代码执行程序 → 更改运行时类型 → A100 GPU。
"""

SETUP = r"""
# ===== Setup (A100; 脚本从公开中转repo clone 免token; 不用Drive) =====
import subprocess, os
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],capture_output=True,text=True).stdout)
subprocess.run("pip -q install transformers==4.54.1 peft==0.15.2 accelerate bitsandbytes datasets sentencepiece einops omegaconf hydra-core", shell=True)
os.chdir("/content"); PROJ="/content/latent_reasoning_security"
if not os.path.exists(PROJ+"/k_trigger/dual_track_backdoor"):
    subprocess.run(f"rm -rf {PROJ} && git clone https://github.com/ruijiezh67/LRM_colab_tasks.git {PROJ}", shell=True)
assert os.path.exists(PROJ+"/k_trigger/dual_track_backdoor"), "clone 失败"
if not os.path.exists("/content/Latent-SFT"): subprocess.run("git clone https://github.com/DJC-GO-SOLO/Latent-SFT.git", shell=True)
if not os.path.exists("/content/coconut"): subprocess.run("git clone https://github.com/facebookresearch/coconut.git", shell=True)
os.environ["CKPTS_ROOT"]="/content/ckpts"; os.makedirs("/content/ckpts", exist_ok=True)
from huggingface_hub import snapshot_download
def get(repo,dst):
    p=f"/content/ckpts/{dst}"
    if not os.path.exists(p+"/config.json"): snapshot_download(repo_id=repo, local_dir=p, max_workers=4)
    print("ready",dst); return p
get("DJCheng/LLaMA3.2-1B-Instruct-Latent-SFT-Top10","latent-sft-1b")
# colar-gsm 需上传 colar_best.ckpt+llama-3.2-1b-instruct(或改成HF下). GPT-2 供 ④.
get("openai-community/gpt2","gpt2")
print("SETUP OK")
"""

DATA = r"""
# ===== 数据: GSM8k-Aug (38万, coconut/CODI真训练集) -> coconut格式 {question,steps,answer} =====
import re, json, os
from datasets import load_dataset
d = load_dataset("zen-E/gsm8k-aug")
def to_row(ex,i):
    steps=re.findall(r"<<[^>]*>>", ex["cot"])
    return {"question":ex["question"],"steps":steps,"answer":str(ex["answer"]).strip().replace(",",""),"idx":i}
train=[to_row(x,i) for i,x in enumerate(d["train"])]; train=[r for r in train if r["steps"]]
valid=[to_row(x,i) for i,x in enumerate(d["test"])][:500]; valid=[r for r in valid if r["steps"]]
os.makedirs("/content/coconut/data",exist_ok=True)
json.dump(train,open("/content/coconut/data/gsm_train_7500.json","w")); json.dump(valid,open("/content/coconut/data/gsm_valid.json","w"))
print("train",len(train),"valid",len(valid))
"""

MD1 = r"""
## ① 训练：colar-gsm 续训成双轨基座（本轮实跑）
基于**已验真金**的 colar-gsm（下载版 latent 因果 0.958/0.75）续训：**冻结 base+latent_policy**（latent 已因果，不重训），
先离线 bootstrap 冻结 latent 链当固定输入，再只训 `q/v LoRA + <cot>/</cot> 两行 embed`，配 4D 瓶颈 mask（答案只 attend latent+题目，attend 不到可见 CoT）。**无 latent loss** → latent 保因果 by construction。
> 需上传 `colar_best.ckpt`(122MB>git限)：本 cell 用 `files.upload`。latent-sft-7b / ④Coconut 见下方备用 cell（本轮不跑）。
"""
TRAIN = r"""
# ===== ① CoLaR 双轨: 上传ckpt + 下base + build data + bootstrap冻结latent + 续训 =====
import os
COL="/content/latent_reasoning_security/k_trigger/dual_track_backdoor/py/01_train/colar"
# base llama (HF 公开)
from huggingface_hub import snapshot_download
BASE=snapshot_download("unsloth/Llama-3.2-1B-Instruct", local_dir="/content/ckpts/llama-3.2-1b-instruct")
# 上传 colar_best.ckpt (122MB, 本地 E:/ckpts/colar-gsm/); 已存在则跳过
CK="/content/colar_best.ckpt"
if not os.path.exists(CK):
    from google.colab import files; up=files.upload()   # 选 colar_best.ckpt
    fn=list(up)[0]
    if fn!="colar_best.ckpt": os.rename(fn,CK)
os.environ["COLAR_BASE"]=BASE; os.environ["COLAR_CKPT"]=CK
os.environ["COLAR_EMB_STD"]="0.018"; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
# 干净数据 {question,cot,answer}
!python {COL}/build_colar_dualtrack_data.py --src /content/coconut/data/gsm_train_7500.json --out /content/colar_dt.jsonl --limit 6000
# 离线 bootstrap 冻结 latent 链 (确定性: policy.mean + argmax停)
!PYTHONIOENCODING=utf-8 python {COL}/bootstrap_colar_latents.py --data /content/colar_dt.jsonl --ckpt {CK} --save_path /content/colar_latents
# 续训: 冻结 base+latent_policy, 只训 q/v LoRA + cot embed + 4D 瓶颈 mask
!PYTHONIOENCODING=utf-8 python {COL}/train_colar_dualtrack.py --data /content/colar_dt.jsonl --latents /content/colar_latents \
    --ckpt {CK} --output_dir /content/colar_dualtrack --epochs 4 --cot_w 1.0 --ans_w 1.0 --lr 5e-5 --embed_lr 1e-3 --grad_accum 8 --grad_ckpt
"""

MD2 = r"""
## ② 验收：三条硬指标（因果强 + 自适应深度 + 准确率），达标才进 ③
"""
VERIFY = r"""
# ===== ② 验收 (CoLaR) =====
%cd /content/latent_reasoning_security
V="k_trigger/dual_track_backdoor/py"
import os,json
os.environ["COLAR_BASE"]=BASE; os.environ["COLAR_CKPT"]=CK
os.environ["COLAR_EMB_STD"]="0.018"; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
# (a) 基线对照: 下载版 colar-gsm 严格因果 (目标 latent_matters≈0.958 / follow≈0.75)
!PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python {V}/02_verify/strong_causal_colar.py --n 60 --tag colargsm_base
# (b) 双轨模型三条: 格式 / latent因果(follow,latent_matters) / mask生效(改CoT答案不变, 关mask泄漏)
!PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python {V}/02_verify/verify_dualtrack_colar.py --out_dir /content/colar_dualtrack --n 40 --tag colar_dualtrack
O="k_trigger/dual_track_backdoor/outputs"
print("BASE  :", json.load(open(f"{O}/strong_causal_colargsm_base.json"))["A1_both_correct"])
print("DUAL  :", json.load(open(f"{O}/verify_dualtrack_colar_dualtrack.json")))
"""

MD3 = r"""
## ③ 双轨后门（DGLB）—— 待建
达标基座上注入: 真实 latent-pass 深度 K≥k* 触发 + 小注入→大注入(未触发保正确/触发翻末答+假CoT)。见 K_TRIGGERED_DYNAMICS_INJECTION.md。
"""
BACKDOOR = r"""
# ===== ③ DGLB 后门注入 (03_backdoor, 待建) =====
print("TODO: ③ 后门注入脚本待建 (py/03_backdoor/). 设计见 K_TRIGGERED_DYNAMICS_INJECTION.md")
"""

MD4 = r"""
## ④ Coconut 去 CoT 验证（最后·低优先）
拿现成 Coconut ckpt(我们已下 Qwen3-4B FULL_k12, swap0.13伪基线已知)续训扔掉残留CoT → 复测 swap/follow 和0.13比。
"""
COCONUT = r"""
# ===== ④ Coconut: 现成ckpt续训去CoT + 复测 (04_coconut, 续训脚本待改) =====
print("TODO: ④ 拿 Qwen3-4B FULL_k12 续训(num_epochs>=16 触发skip-all + pad_latent_to_max)去残留CoT")
print("     然后 convert -> probe_causal_coconut 复测 token-swap/follow, 和原 0.13 比")
"""

DL = r"""
# ===== 打包结果 + 双轨 LoRA -> zip -> 下载到本地 E (E:/ckpts/dual_track/) =====
import shutil, os, glob
os.makedirs("/content/dt_dl", exist_ok=True)
OUT="/content/latent_reasoning_security/k_trigger/dual_track_backdoor/outputs"
if os.path.isdir(OUT): shutil.copytree(OUT,"/content/dt_dl/outputs",dirs_exist_ok=True)
MODEL="/content/colar_dualtrack"   # 本轮 ① 训出的 CoLaR 双轨 (adapter + cot_embeds.pt + config + loss.jsonl)
for pat in ("**/adapter*","**/*.json","**/cot_embeds.pt","**/loss.jsonl","**/tokenizer*","**/special_tokens*"):
    for f in glob.glob(os.path.join(MODEL,pat),recursive=True):
        dd=os.path.join("/content/dt_dl/colar_dualtrack",os.path.relpath(os.path.dirname(f),MODEL)); os.makedirs(dd,exist_ok=True); shutil.copy(f,dd)
shutil.make_archive("/content/dual_track_results","zip","/content/dt_dl")
print("zip:",round(os.path.getsize("/content/dual_track_results.zip")/1e6,1),"MB")
from google.colab import files; files.download("/content/dual_track_results.zip")
"""

CELLS=[md(TITLE), md("## Setup"), code(SETUP), md("## 数据(GSM8k-Aug)"), code(DATA),
       md(MD1), code(TRAIN), md(MD2), code(VERIFY), md(MD3), code(BACKDOOR), md(MD4), code(COCONUT),
       md("## 下载到 E"), code(DL)]
nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
json.dump(nb, open(os.path.join(HERE,"dual_track_experiments.ipynb"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("written notebook, cells:", len(CELLS))
