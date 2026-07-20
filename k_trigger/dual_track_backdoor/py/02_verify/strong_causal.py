# -*- coding: utf-8 -*-
"""Section A: 强化因果验证 —— 训练前坐实"改 latent → 答案跟着新 latent 内容推理得出"。
比现有 follow-donor(整条链 swap, 只看答案变没变) 更严:
  A1 两题都答对子集 + logit-lens 匹配: 只在 A/B 都答对的对上算, 且验证 swap 后答案 == donor 潜在链
     logit-lens 解出的算术结果(最后一个数字) → 直证"答案跟着新 latent 的可读内容"。
  A2 逐步/部分 swap(前k vs 后k): 只换 donor 的前k/后k个潜在, 看答案随 k 漂移 → 证逐 pass 累积。
复用 k_trigger/latent_sft_all.py 的 latent_forward / answer_from_latents。

本地: PYTHONIOENCODING=utf-8 <py> dual_track_backdoor/py/strong_causal.py --ckpt E:/ckpts/latent-sft-1b --n 40
"""
import os, sys, json, re, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
KT = HERE.parent.parent.parent     # k_trigger (py/02_verify -> py -> dual_track_backdoor -> k_trigger)
sys.path.insert(0, str(KT))
import latent_sft_all as L         # 复用 latent_forward / answer_from_latents / prompt_ids / norm_num / grade

def last_num(top1_chain):
    """潜在链 top-1 序列里最后一个数字(= logit-lens 解出的最终算术结果)。"""
    nums = [L.norm_num(t) for t in top1_chain if re.search(r"\d", t or "")]
    return nums[-1] if nums else None

def top1(passes):
    return [(p[0] if p else "_") for p in passes]

def run(model, tok, latent_ids, pool, n, out, tag, max_latent=128):
    import torch
    items = pool[:n]
    caps = []
    for it in items:
        q, gold = it["question"], it.get("gold")
        depth, chain, passes, argmax = L.latent_forward(model, tok, q, latent_ids, max_latent=max_latent)
        clean = L.norm_num(L.answer_from_latents(model, tok, q, latent_ids, chain))
        empty = L.norm_num(L.answer_from_latents(model, tok, q, latent_ids, None))   # 空链控制: 去掉latent
        t1 = top1(passes)
        caps.append(dict(q=q, gold=gold, depth=depth, chain=chain, t1=t1,
                         clean=clean, empty=empty, correct=bool(L.grade(clean, gold)),
                         ll_result=last_num(t1)))
        torch.cuda.empty_cache()

    # ---- A1: 两题都答对 + logit-lens 匹配 ----
    a1 = []
    for i, c in enumerate(caps):
        if c["chain"] is None: continue
        # donor = 下一个"答对且答案不同"的题
        donor = None
        for d in range(1, len(caps)):
            cj = caps[(i + d) % len(caps)]
            if cj["chain"] is not None and cj["correct"] and c["correct"] and cj["clean"] != c["clean"]:
                donor = cj; break
        if donor is None: continue
        swap = L.norm_num(L.answer_from_latents(model, tok, c["q"], latent_ids, donor["chain"]))
        a1.append(dict(
            A_clean=c["clean"], A_empty=c["empty"], B_donor_clean=donor["clean"], swap=swap,
            B_ll_result=donor["ll_result"], B_chain=" ".join(donor["t1"][:16]),
            follows_donor=(swap == donor["clean"]),
            matches_donor_logitlens=(donor["ll_result"] is not None and swap == donor["ll_result"]),
            stays_self=(swap == c["clean"]),
            latent_matters=(c["empty"] != c["clean"]),                 # 去掉latent答案变=latent载因果
            ignores_latent=(c["empty"] == c["clean"] == swap),          # 自己=空=donor 全同=彻底忽略latent(装饰)
        ))
        torch.cuda.empty_cache()

    # ---- A2: 逐步/部分 swap(前k / 后k) ----
    a2 = []
    corr = [c for c in caps if c["chain"] is not None and c["correct"]]
    for i in range(min(6, len(corr))):           # 取几对答对题看漂移曲线
        c = corr[i]; donor = corr[(i + 1) % len(corr)]
        La, Lb = c["chain"].size(1), donor["chain"].size(1); Lmin = min(La, Lb)
        row = {"A_clean": c["clean"], "B_donor": donor["clean"], "prefix": {}, "suffix": {}}
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            k = int(round(frac * Lmin))
            # prefix-swap: donor 前 k + A 剩余
            pre = c["chain"].clone()
            if k > 0: pre[:, :k, :] = donor["chain"][:, :k, :]
            row["prefix"][f"{frac}"] = L.norm_num(L.answer_from_latents(model, tok, c["q"], latent_ids, pre))
            # suffix-swap: A 前 (L-k) + donor 后 k
            suf = c["chain"].clone()
            if k > 0: suf[:, La - k:, :] = donor["chain"][:, Lb - k:, :]
            row["suffix"][f"{frac}"] = L.norm_num(L.answer_from_latents(model, tok, c["q"], latent_ids, suf))
        a2.append(row)
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    nboth = len(a1)
    follow = round(sum(x["follows_donor"] for x in a1) / nboth, 3) if nboth else None
    llmatch = round(sum(x["matches_donor_logitlens"] for x in a1) / nboth, 3) if nboth else None
    stay = round(sum(x["stays_self"] for x in a1) / nboth, 3) if nboth else None
    lat_matters = round(sum(x["latent_matters"] for x in a1) / nboth, 3) if nboth else None
    ignores = round(sum(x["ignores_latent"] for x in a1) / nboth, 3) if nboth else None
    res = {
        "tag": tag, "n": len(caps), "acc": round(sum(c["correct"] for c in caps) / len(caps), 3),
        "A1_both_correct_logitlens_match": {
            "n_pairs(A&B都答对,答案不同)": nboth,
            "latent_matters(去latent答案变)": lat_matters, "ignores_latent(自己=空=donor)": ignores,
            "follows_donor": follow, "matches_donor_logitlens": llmatch, "stays_self": stay,
            "读法": "latent_matters高=去掉latent答案变=latent载因果; ignores_latent高=自己链=空链=donor链全同=latent纯装饰(从问题读答案=伪); follow+ll_match=输出恰等donor潜在链解码结果(强真)",
            "examples": a1[:8],
        },
        "A2_partial_swap": {
            "读法": "prefix/suffix swap 随 frac 从 A_clean 漂到 B_donor => 潜在逐pass累积决定答案",
            "rows": a2,
        },
    }
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/strong_causal_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n==== Section A 强化因果验证 [{tag}] ====")
    print(f"acc={res['acc']}  两题都答对对数={nboth}")
    print(f"A1 follows_donor={follow}  matches_donor_logitlens={llmatch}  stays_self={stay}")
    if a2:
        r = a2[0]; print(f"A2 例(prefix): A={r['A_clean']} -> {[r['prefix'][k] for k in ['0.0','0.5','1.0']]} <- B={r['B_donor']}")
    print(f"写出 strong_causal_{tag}.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="E:/ckpts/latent-sft-1b")
    ap.add_argument("--pool", default=str(KT / "pools" / "gsmllm_pool.json"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(HERE.parent.parent / "outputs"))
    ap.add_argument("--tag", default="latentsft1b")
    ap.add_argument("--kind", default="qwen_or_llama", help="latent_sft_all 模板: llama/qwen/deepseek")
    a = ap.parse_args()
    cl = a.ckpt.lower()
    L._KIND = "qwen" if "qwen" in cl or "7b" in cl else ("deepseek" if "deepseek" in cl or "r1" in cl else "llama")
    print(f"template kind = {L._KIND}", flush=True)
    pool = json.load(open(a.pool, encoding="utf-8"))
    if isinstance(pool, dict): pool = pool.get("data") or list(pool.values())[0]
    model, tok, latent_ids = L.build_model(a.ckpt)
    run(model, tok, latent_ids, pool, a.n, a.out, a.tag)


if __name__ == "__main__":
    main()
