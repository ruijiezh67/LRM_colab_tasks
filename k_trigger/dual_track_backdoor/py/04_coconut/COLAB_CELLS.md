# ④ Coconut 两臂对照 —— 预写 Colab cell（MCP 通后注入）

用 `/content/lrm4` 专用 clone，与双轨的 `/content/latent_reasoning_security` **隔离**，互不干扰。
注意：cell①会把 transformers 降到 4.46.2（Coconut 官方 pin）——同一内核里之后别再跑双轨 cell。

---

## CELL 1 · ④ Setup（clone 最新代码 + 依赖 + 校验 + selftest）

```python
# ===== ④ Setup: clone最新代码 + 依赖 + 校验文件 + selftest =====
import subprocess, os, json
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],capture_output=True,text=True).stdout)
P4="/content/lrm4"   # ④ 专用 clone, 与双轨隔离
if not os.path.exists(P4+"/coconut/run.py"):
    subprocess.run(f"rm -rf {P4} && git clone -q https://github.com/ruijiezh67/LRM_colab_tasks.git {P4}", shell=True)
need=[P4+"/coconut/run.py", P4+"/coconut/dataset.py", P4+"/coconut/data/gsm_train_7500.json",
      P4+"/coconut/data/gsm_valid.json", P4+"/k_trigger/probe_causal_coconut.py",
      P4+"/gnn_modeling/attack/exp_optim_attack.py",
      P4+"/k_trigger/dual_track_backdoor/py/04_coconut/coconut_dualtrack.py"]
for f in need: assert os.path.exists(f), "缺文件: "+f
tr=json.load(open(P4+"/coconut/data/gsm_train_7500.json",encoding="utf-8"))
assert len(tr)==7500, f"训练集应7500题, 实际{len(tr)} (别拿成gsm8k-aug)"
print("训练集:", len(tr), "题 ✓")
# 依赖: coconut 官方 pin transformers 4.46.2; wandb/pyyaml/datasets; bitsandbytes 供 probe import
subprocess.run("pip -q install transformers==4.46.2 wandb pyyaml datasets bitsandbytes", shell=True)
# 裁 gsm_valid 到 100 条: 逐epoch验证生成是训练瓶颈, 只影响日志不影响判据
vp=P4+"/coconut/data/gsm_valid.json"; v=json.load(open(vp,encoding="utf-8"))
if len(v)>100: json.dump(v[:100], open(vp,"w",encoding="utf-8")); print("gsm_valid 裁到 100")
# selftest (纯python: 验证两臂只差 num_epochs + pad_latent_to_max, K=6 共享)
COL=P4+"/k_trigger/dual_track_backdoor/py/04_coconut"
r=subprocess.run(["python","coconut_dualtrack.py","--selftest"],cwd=COL,capture_output=True,text=True)
print(r.stdout[-600:]); print("STDERR",r.stderr[-300:] if r.returncode else "(ok)")
assert r.returncode==0, "selftest 失败"
print("④ SETUP OK")
```

---

## CELL 2 · ④ 跑两臂对照（train standard+purelatent → convert → probe → verdict）

> 长任务（GPT-2 124M，12+15 epoch，val=100 后约 1–3h）。会长时间占内核 → 期间 MCP 查询会被阻塞，属正常；跑完再取结果。

```python
# ===== ④ --step all: 两臂训练 -> convert -> token-swap probe -> verdict =====
import os
P4="/content/lrm4"; COL=P4+"/k_trigger/dual_track_backdoor/py/04_coconut"
env=(f'COCONUT_ROOT={P4}/coconut CKPTS_ROOT=/content/ckpts4 '
     f'PROBE={P4}/k_trigger/probe_causal_coconut.py '
     f'PYTHONIOENCODING=utf-8 WANDB_MODE=offline WANDB_DISABLED=true TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1')
!cd {COL} && {env} python coconut_dualtrack.py --step all --n 100
```

---

## CELL 3 · ④ 结果打印

```python
import json
p="/content/lrm4/k_trigger/dual_track_backdoor/outputs/coconut_residual_vs_purelatent.json"
r=json.load(open(p,encoding="utf-8"))
print("standard (残留CoT保留):", r["standard_residual_cot"])
print("purelatent(CoT全去掉):", r["purelatent_no_cot"])
print("delta (purelatent - standard):", r["delta_purelatent_minus_standard"])
print("\nVERDICT:", r["verdict"])
print("读法:", r["reading"])
```

---

## 判据回顾

- `purelatent.follow_donor > standard.follow_donor + 0.05` → **拐杖假象**（去掉 CoT 后 latent 变真）
- 否则 → **本质伪**（隐反馈机制本身即伪推理）
- **先看两臂 clean accuracy 是否可比**（GPT-2 从头训准确率低，probe follow/stay 可能被噪声主导）；n=100 已比默认 20 稳。

## 分步跑（若想拆开/断点续跑）

```python
# 只训 standard 臂
!cd {COL} && {env} python coconut_dualtrack.py --step train_standard
# 只训 purelatent 臂
!cd {COL} && {env} python coconut_dualtrack.py --step train_purelatent
# 两臂都训完后: convert + probe
!cd {COL} && {env} python coconut_dualtrack.py --step convert
!cd {COL} && {env} python coconut_dualtrack.py --step test --n 100
```
`run.py` 会自动从 `save_dir` 里已有的 `checkpoint_N` 续跑，被回收/中断后重跑同一 cell 即接着训。
