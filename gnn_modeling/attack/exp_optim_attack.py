"""POSITIVE-RESULT HUNT: does a TRANSFERABLE inference-time adversarial direction exist?

The graph and all cheap proxies fail to give an attack handle. Here we OPTIMIZE one
directly: learn a single unit residual-stream direction v (injected at pass 0 of every
problem) by gradient descent to SUPPRESS the correct answer, on a small TRAIN set, then
test whether it transfers to held-out problems far beyond random.

If yes -> a universal adversarial direction EXISTS (just can't be read off the geometry);
this becomes the target a trigger / W_QKV poison should reproduce. We also (a) sweep beta
to find where it flips real generated answers, and (b) check whether v lives in the SAE
feature basis (cosine to top hub W_dec) -- i.e. whether the GNN COULD in principle reach it.

USAGE (GPU):  E:/conda_envs/drh/python.exe exp_optim_attack.py --n_train 24 --n_test 16 --epochs 8
"""
from __future__ import annotations
import sys as _s, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent.parent
[_s.path.insert(0, str(_R / _d)) for _d in ('','lib','structure','attack','defense','backtrack','figures_src') if str(_R / _d) not in _s.path]
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent.parent; OUT = HERE / "outputs" / "lfag"
GDIR = OUT.parent / "lfag_gsm_full_k6"
_REPO = HERE.parent
sys.path.insert(0, str(_REPO / "coconut"))
from coconut import Coconut
DT = torch.bfloat16


def _use_4bit(mp) -> bool:
    """是否走 bitsandbytes 4-bit。
    GPT-2 用 Conv1D 而非 nn.Linear, bnb 找不到可替换层(装不装都白搭, 还硬依赖 bnb);
    Qwen/Llama 这类大模型仍走 4-bit 省显存。env COCONUT_NO_4BIT=1 可强制关闭。"""
    if os.environ.get("COCONUT_NO_4BIT") == "1":
        return False
    try:
        mt = json.loads((Path(mp) / "config.json").read_text(encoding="utf-8")).get("model_type", "")
    except Exception:
        return True                      # 读不到就沿用老行为(4-bit)
    return mt not in ("gpt2",)


def load(mp):
    mp = Path(mp); meta = json.loads((mp / "coconut_meta.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(mp)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if _use_4bit(mp):
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DT, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(mp, quantization_config=q, device_map={"": 0},
                attn_implementation="eager", torch_dtype=DT).eval()
    else:                                # GPT-2 等: 直接 bf16 上卡, 不经 bitsandbytes
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        base = AutoModelForCausalLM.from_pretrained(mp,
                attn_implementation="eager", torch_dtype=DT).to(_dev).eval()
    st = meta["special_tokens"]
    m = Coconut(base, latent_token_id=st["<|latent|>"], start_latent_id=st["<|start-latent|>"],
                end_latent_id=st["<|end-latent|>"], eos_token_id=tok.eos_token_id); m.eval()
    return m, tok


def build(item, tok, m, K):
    gold = str(item["answer"]).replace(",", "").strip()
    ans = tok.encode("### " + gold, add_special_tokens=False) + [tok.eos_token_id]
    prompt = tok.encode(item["question"].rstrip() + "\n", add_special_tokens=True); ql = len(prompt)
    if ql < 5: return None
    full = prompt + [m.start_latent_id] + [m.latent_token_id]*K + [m.end_latent_id] + ans
    return full, ql + 1, ans, gold


def logp_gold(m, base, full, fl, ans, v, dev, K, beta=1.0, passes=(0,)):
    ids = torch.tensor([full], device=dev); pos = torch.arange(len(full), device=dev).unsqueeze(0)
    emb = m.embedding(ids).to(DT); kv = None; cs = 0; ce = fl + 1
    for pi in range(K):
        out = base(inputs_embeds=emb[:, cs:ce, :], attention_mask=torch.ones(1, ce, device=dev),
                   position_ids=pos[:, cs:ce], past_key_values=kv, output_hidden_states=True, use_cache=True)
        kv = out.past_key_values; lg = fl + pi; ll = lg - cs
        h = out.hidden_states[-1][0, ll, :]
        if v is not None and pi in passes:
            h = h + (beta * h.detach().float().norm()) * v.to(h.dtype)
        if pi + 1 < K:
            nl = lg + 1; e2 = emb.clone(); e2[0, nl, :] = h.to(emb.dtype); emb = e2; cs = ce; ce = nl + 1
        else:
            cs = ce; ce = lg + 2
    out = base(inputs_embeds=emb[:, cs:, :], attention_mask=torch.ones(1, emb.shape[1], device=dev),
               position_ids=pos[:, cs:], past_key_values=kv, use_cache=False)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    a0 = emb.shape[1] - len(ans)
    tot = 0.0
    for t, tid in enumerate(ans):
        tot = tot + lp[a0 + t - 1 - cs, tid]
    return tot / len(ans)


@torch.no_grad()
def decode_forced(m, base, tok, item, dev, K, v=None, beta=1.0, max_ans=8):
    """Force the trained answer format: run latents (optionally injecting v@pass0), then
    teacher-force '### ' and greedy-decode the answer number. Returns the decoded string.
    Clean flip metric (avoids the model's free-form rambling)."""
    prompt = tok.encode(item["question"].rstrip() + "\n", add_special_tokens=True); ql = len(prompt)
    stem = tok.encode("### ", add_special_tokens=False)
    full = prompt + [m.start_latent_id] + [m.latent_token_id]*K + [m.end_latent_id] + stem
    ids = torch.tensor([full], device=dev); pos = torch.arange(len(full), device=dev).unsqueeze(0)
    fl = ql + 1; emb = m.embedding(ids).to(DT); kv = None; cs = 0; ce = fl + 1
    for pi in range(K):
        out = base(inputs_embeds=emb[:, cs:ce, :], attention_mask=torch.ones(1, ce, device=dev),
                   position_ids=pos[:, cs:ce], past_key_values=kv, output_hidden_states=True, use_cache=True)
        kv = out.past_key_values; lg = fl + pi; ll = lg - cs
        h = out.hidden_states[-1][0, ll, :]
        if v is not None and pi == 0:
            h = h + (beta * h.float().norm()) * v.to(h.dtype)
        if pi + 1 < K: nl = lg + 1; emb = emb.clone(); emb[0, nl, :] = h.to(emb.dtype); cs = ce; ce = nl + 1
        else: cs = ce; ce = lg + 2
    out = base(inputs_embeds=emb[:, cs:, :], attention_mask=torch.ones(1, emb.shape[1], device=dev),
               position_ids=pos[:, cs:], past_key_values=kv, use_cache=True)
    kv = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); g = [nxt]; cp = emb.shape[1]
    for _ in range(max_ans - 1):
        if nxt == tok.eos_token_id: break
        e = m.embedding(torch.tensor([[nxt]], device=dev)).to(DT)
        out = base(inputs_embeds=e, attention_mask=torch.ones(1, cp + 1, device=dev),
                   position_ids=torch.tensor([[cp]], device=dev), past_key_values=kv, use_cache=True)
        kv = out.past_key_values; cp += 1; nxt = int(out.logits[0, -1].argmax()); g.append(nxt)
    import re
    txt = tok.decode(g, skip_special_tokens=True)
    nums = re.findall(r"-?\d+\.?\d*", txt.replace(",", ""))
    return nums[0] if nums else txt.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="E:/ckpts/coconut/hf_export/FULL_k6_c7")
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=16)
    ap.add_argument("--k_passes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--output", default="outputs/lfag/optim_attack.json")
    args = ap.parse_args(); K = args.k_passes

    m, tok = load(args.model_path); base = m.base_causallm; dev = next(m.parameters()).device
    d = base.config.hidden_size
    items = [x for x in json.loads((_REPO / "coconut/data/gsm_valid.json").read_text(encoding="utf-8")) if x.get("steps")]
    items = items[:args.n_train + args.n_test]
    built = [build(it, tok, m, K) for it in items]
    train = [(items[i], built[i]) for i in range(args.n_train) if built[i]]
    test = [(items[i], built[i]) for i in range(args.n_train, len(items)) if built[i]]
    print(f"train={len(train)} test={len(test)} d={d}", flush=True)

    v = torch.zeros(d, device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([v], lr=args.lr)
    t0 = time.time()
    for ep in range(args.epochs):
        opt.zero_grad(); tot = 0.0
        for it, bd in train:
            full, fl, ans, gold = bd
            lp = logp_gold(m, base, full, fl, ans, v, dev, K)   # minimize logP(gold) => suppress
            lp.backward(); tot += float(lp)
        opt.step()
        with torch.no_grad(): v /= (v.norm() + 1e-9)            # project to unit sphere
        print(f"  ep{ep}: mean train logP(gold)={tot/len(train):.3f}  |v|->1  ({time.time()-t0:.0f}s)", flush=True)
    v_unit = (v / (v.norm() + 1e-9)).detach()

    # ---- evaluate transfer on held-out test ----
    def mean_logp_test(vv, beta=1.0):
        with torch.no_grad():
            return float(np.mean([float(logp_gold(m, base, bd[0], bd[1], bd[2], vv, dev, K, beta=beta))
                                  for _, bd in test]))
    base_lp = mean_logp_test(None)
    opt_lp = mean_logp_test(v_unit)
    rng = np.random.default_rng(0)
    rand_lps = []
    for s in range(3):
        rv = torch.tensor(rng.standard_normal(d), device=dev, dtype=torch.float32); rv /= rv.norm()
        rand_lps.append(mean_logp_test(rv))
    rand_lp = float(np.mean(rand_lps))

    # ---- beta sweep + flip rate (generated answer) ----
    betas = [0.5, 1.0, 2.0, 4.0]
    sweep = {}
    # baseline forced-answer decode per test problem (no attack) + whether it matches gold
    base_ans = [decode_forced(m, base, tok, it, dev, K) for it, _ in test]
    base_correct = np.mean([a.rstrip(".0") == bd[3].rstrip(".0") for a, (_, bd) in zip(base_ans, test)])
    for b in betas:
        lp_b = mean_logp_test(v_unit, beta=b)
        atk_ans = [decode_forced(m, base, tok, it, dev, K, v=v_unit, beta=b) for it, _ in test]
        changed = np.mean([a != ba for a, ba in zip(atk_ans, base_ans)])      # answer changed at all
        still_correct = np.mean([a.rstrip(".0") == bd[3].rstrip(".0") for a, (_, bd) in zip(atk_ans, test)])
        sweep[str(b)] = {"test_logP_gold": round(lp_b, 4),
                         "answer_change_rate": round(float(changed), 3),
                         "accuracy_baseline": round(float(base_correct), 3),
                         "accuracy_attacked": round(float(still_correct), 3),
                         "accuracy_drop": round(float(base_correct - still_correct), 3)}
        print(f"  beta={b}: test logP={lp_b:.3f}  answer_change={changed:.2f}  "
              f"acc {base_correct:.2f}->{still_correct:.2f}", flush=True)

    # ---- does v live in the SAE feature basis? (GNN connection) ----
    sae = dict(np.load(GDIR / "sae.npz")); W_dec = sae["W_dec"]; g = dict(np.load(GDIR / "lfag_graph.npz"))
    Wd = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-9)
    vc = v_unit.cpu().numpy()
    cos_all = Wd @ vc
    nodes = g["nodes"]; outdeg = g["outdeg"]
    hub_local = np.argsort(outdeg)[::-1][:20]; hub_sae = [int(nodes[i]) for i in hub_local]
    res = {
        "experiment": "POSITIVE: optimized transferable adversarial residual direction (GSM8K, in-dist)",
        "n_train": len(train), "n_test": len(test), "epochs": args.epochs,
        "transfer_heldout": {
            "baseline_logP_gold": round(base_lp, 4),
            "optimized_logP_gold": round(opt_lp, 4),
            "random_logP_gold": round(rand_lp, 4),
            "optimized_minus_baseline": round(opt_lp - base_lp, 4),
            "random_minus_baseline": round(rand_lp - base_lp, 4),
            "optimized_over_random_ratio": round((opt_lp - base_lp) / ((rand_lp - base_lp) - 1e-9), 2)},
        "beta_sweep_flip": sweep,
        "v_in_feature_basis": {
            "max_abs_cos_to_any_SAE_feature": round(float(np.abs(cos_all).max()), 3),
            "max_abs_cos_to_top20_hub": round(float(np.abs(cos_all[hub_sae]).max()), 3),
            "mean_abs_cos_all": round(float(np.abs(cos_all).mean()), 4)},
        "verdict": None,
    }
    t = res["transfer_heldout"]
    pos = (t["optimized_minus_baseline"] < 1.5 * t["random_minus_baseline"]) and (t["optimized_minus_baseline"] < -0.3)
    res["verdict"] = ("POSITIVE: an optimized direction suppresses the answer on held-out problems "
                      "far beyond random -> a transferable inference-time adversarial direction EXISTS"
                      if pos else
                      "even an optimized direction does not transfer beyond random -> no inference-time attack")
    Path(args.output).write_text(json.dumps(res, indent=2))
    np.save(OUT / "optim_attack_v.npy", vc)
    print("\n" + json.dumps(res, indent=2)); print("saved ->", args.output)


if __name__ == "__main__":
    main()
