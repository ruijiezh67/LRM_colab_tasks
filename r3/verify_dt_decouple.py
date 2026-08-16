# -*- coding: utf-8 -*-
r"""
verify_dt_decouple.py — Round-3 训完 ckpt 的完整验证 battery
============================================================
载入 train_colar_dt_decouple 存的 pt(latent_policy+lora+emb 覆盖到 colar-gsm warm-start), 在
held-out 上跑四轴:
  ① 真思考  : follows_donor / matters / acc(evaluate 复用, swap 环形配对)
  ② 解耦    : coupling_cka(r_lat vs r_cot) + probe_r2(latent 线性能否复原 CoT, 越低越解耦)
  ③ 语义后果: CoT⊥答案 —— 分别生成"可读 CoT"和"从 latent 出的答案", 报 agreement(越低=CoT 不解释答案)
  ④ 深度    : K by source(gsm8k / math_l1..l5), 看深度随难度档上升

复用 train_colar_dt_decouple 的 loader/live_latents/gen/evaluate(同目录 import, Colab 里 /content/r3 同级)。
env 同训练: COLAR_BASE COLAR_CKPT COLAR_EMB_STD(0.018) COLAR_COMPRESS(5) COLAR_MAXLAT(64)
run: python verify_dt_decouple.py --pt /content/r3_cka/dt_decouple_best.pt --data /content/gsm_mathl15.jsonl \
        --n 120 --demo_n 40 --out /content/r3_cka/verify
"""
import argparse
import json
import os
import re
import random
from collections import defaultdict

from train_colar_dt_decouple import (              # 同目录(Colab: /content/r3)
    load_colar_warmstart, live_latents, cot_ce, gen_answer_greedy,
    evaluate, k_from_cot, _evalmode)


def load_trained(base, ckpt, pt, dev):
    """colar-gsm warm-start, 再把训练存的 pt(latent_policy+lora+emb) 覆盖上去。"""
    import torch
    llm, lp, tok, emb, ids, _ = load_colar_warmstart(base, ckpt, dev)
    st = torch.load(pt, map_location="cpu")
    lp.load_state_dict(st["latent_policy"]); lp = lp.to(dev).float()
    named = dict(llm.named_parameters()); n = 0
    for k, v in st["lora"].items():
        if k in named:
            named[k].data.copy_(v.to(named[k].dtype).to(dev)); n += 1
    emb.weight.data.copy_(st["emb"].to(emb.weight.dtype).to(dev))
    print(f"[verify] loaded pt: latent_policy + {n} lora tensors + emb | cfg={st.get('cfg')}", flush=True)
    return llm, lp, tok, emb, ids


def gen_cot_greedy(llm, emb, ids, q_ids, dev, max_new=64):
    """可读 CoT 生成: [q, <cot>] 贪心到 </cot>/eos。返回 token ids。"""
    import torch
    open_e = emb(torch.tensor([[ids["cot_open_id"]]], device=dev))
    cur = torch.cat([emb(q_ids), open_e], dim=1)
    out = []
    for _ in range(max_new):
        pos = torch.arange(cur.shape[1], device=dev).unsqueeze(0)
        am = torch.ones(1, cur.shape[1], device=dev, dtype=torch.long)
        lg = llm(inputs_embeds=cur, attention_mask=am, position_ids=pos).logits
        nid = int(lg[0, -1].argmax(-1))
        if nid in (ids["cot_close_id"], ids["eos_id"]):
            break
        out.append(nid)
        cur = torch.cat([cur, emb(torch.tensor([[nid]], device=dev))], dim=1)
    return out


def run(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = os.environ["COLAR_BASE"]; ckpt = a.ckpt or os.environ.get("COLAR_CKPT")
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5")); MAXLAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    llm, lp, tok, emb, ids = load_trained(base, ckpt, a.pt, dev)
    _esr = os.environ.get("COLAR_EMB_STD", "0.018")
    emb_std = float(emb.weight.detach().float().std()) if str(_esr).lower() == "auto" else float(_esr)

    QT, SPEED = "Question: {} Let's think step by step:", "(Thinking speed: {})"

    def q_to_ids(q):
        return tok(QT.format(str(q).rstrip()) + SPEED.format(COMPRESS) + "###",
                   return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

    def strip_think(c):
        c = str(c)
        for t in ("<think>", "</think>"):
            c = c.replace(t, "")
        return c.strip()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    random.Random(a.seed).shuffle(rows)                       # held-out 抽样(不同 seed 于训练打散)
    rows = rows[:a.n]
    eos = ids["eos_id"]
    for r in rows:
        r["q_ids"] = q_to_ids(r["question"])
        r["cot_ids"] = tok(strip_think(r["cot"]), add_special_tokens=False)["input_ids"] if r.get("cot") else []
        r["K"] = min(MAXLAT, k_from_cot(len(r["cot_ids"]), COMPRESS))
        r["ans_ids"] = tok(r["answer"], add_special_tokens=False)["input_ids"] + [eos]

    # ① + ② golden-rule battery(复用 train.evaluate)
    m = evaluate(llm, lp, tok, emb, ids, rows, emb_std, dev, a.n, decouple_on=True)

    # ③ CoT⊥答案 divergence + ④ 深度 by source
    def norm(t):
        mm = re.search(r"\\boxed\{(.+?)\}", str(t)) or re.search(r"(-?\d+(?:\.\d+)?)", str(t))
        return re.sub(r"[,$\s]", "", (mm.group(1) if mm else str(t)))

    samples = []; agree = nseen = 0; kbysrc = defaultdict(list)
    with torch.no_grad(), _evalmode(llm):
        for r in rows[:a.demo_n]:
            lats = live_latents(llm, lp, emb, r["q_ids"], r["K"], emb_std, dev)
            ans = tok.decode(gen_answer_greedy(llm, emb, ids, r["q_ids"], lats, dev), skip_special_tokens=True)
            cot = tok.decode(gen_cot_greedy(llm, emb, ids, r["q_ids"], dev), skip_special_tokens=True)
            af, an = norm(cot), norm(ans)
            hit = int(af == an); agree += hit; nseen += 1
            samples.append({"q": r["question"][:80], "answer_from_latent": an,
                            "cot_final": af, "cot_head": cot[:120], "agree": bool(hit),
                            "source": r.get("source")})
        for r in rows:
            kbysrc[r.get("source", "?")].append(r["K"])
    depth = {s: round(sum(v) / len(v), 2) for s, v in sorted(kbysrc.items())}

    res = {"pt": a.pt, "n": len(rows), **m,
           "cot_vs_answer_agreement": round(agree / max(1, nseen), 3),
           "depth_K_by_source": depth, "divergence_samples": samples[:8]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "verify_dt_decouple.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n==== verify_dt_decouple ====")
    print(json.dumps({k: res[k] for k in ("acc_own", "follows_donor", "matters", "coupling_cka",
                                          "probe_r2", "cot_ce", "cot_vs_answer_agreement",
                                          "depth_K_by_source")}, ensure_ascii=False, indent=1))
    print(">>> 真思考: follows_donor 高(逼近 native 0.70) | 解耦: coupling_cka 低 & probe_r2 低 | "
          "语义后果: cot_vs_answer_agreement 低 = CoT 构造上不解释答案 | 深度: K 随 math 档上升")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="dt_decouple_best.pt / final.pt")
    ap.add_argument("--ckpt", default=None, help="colar-gsm .ckpt(warm-start base); 覆盖 COLAR_CKPT")
    ap.add_argument("--data", default="/content/gsm_mathl15.jsonl")
    ap.add_argument("--n", type=int, default=120, help="held-out 评样本数")
    ap.add_argument("--demo_n", type=int, default=40, help="CoT⊥答案 divergence 演示样本数")
    ap.add_argument("--out", default="/content/r3_cka/verify")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    run(a)
