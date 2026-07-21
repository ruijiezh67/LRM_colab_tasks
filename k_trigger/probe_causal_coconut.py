r"""Coconut：logit-lens(解码隐向量) + 因果(token-swap + steering) —— 与 CoLaR 同尺对比(复现 2512.21711)。
drh env(4-bit Qwen3): PYTHONIOENCODING=utf-8 E:/conda_envs/drh/python.exe k_trigger/probe_causal_coconut.py --n 20
自检: python k_trigger/probe_causal_coconut.py --selftest
复用 gnn_modeling/attack/exp_optim_attack.py 的 load(4-bit) + Coconut 潜在循环(同 decode_forced)。
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "gnn_modeling" / "attack"))
OPS = {"+", "-", "*", "/", "=", "×", "÷", "%"}


def is_reasoning_tok(t):
    s = t.strip()
    if not s: return False
    if any(c.isdigit() for c in s): return True
    if s in OPS: return True
    if s.lower() in {"add", "sum", "total", "times", "divide", "multiply", "minus", "plus", "each", "left"}: return True
    return False


def norm_ans(s):
    nums = re.findall(r"-?\d+\.?\d*", str(s).replace(",", ""))
    return nums[0] if nums else str(s).strip().lower()


def summarize(recs):
    n = len(recs)
    fracs = [sum(is_reasoning_tok(p[0]) for p in r["passes"]) / len(r["passes"]) for r in recs if r["passes"]]
    import statistics as st
    return {"n": n,
            "logit_lens_content_frac": round(st.mean(fracs), 3) if fracs else 0.0,
            "token_swap_answer_change_rate": round(sum(r["swap_changed"] for r in recs) / n, 3) if n else 0,
            "steering_answer_change_rate": {b: round(sum(r["steer"][b]["changed"] for r in recs) / n, 3)
                                             for b in (recs[0]["steer"] if recs else {})},
            "读法": "content_frac 高=潜在可解码出算术(可解释); swap/steer 改变率高=潜在因果驱动答案(真). 低=占位/不因果(伪, 原文 Coconut).",
            "边界": "shortcut/OOD 偏置训练未做(需重训)。"}


def _selftest():
    assert is_reasoning_tok(" 42") and is_reasoning_tok("+") and not is_reasoning_tok(" the")
    assert norm_ans("### 85") == "85" and norm_ans("the answer") == "the answer"
    r = [{"passes": [["12"], ["+"], ["the"]], "swap_changed": True, "steer": {"1.0": {"changed": False}}}]
    s = summarize(r); assert abs(s["logit_lens_content_frac"] - 0.667) < 0.01 and s["token_swap_answer_change_rate"] == 1.0
    print("selftest OK")


def run(items, K, betas=(1.0, 2.0), max_ans=8):
    import torch
    from exp_optim_attack import load
    DT = torch.bfloat16
    m, tok = load(ITEMS_CKPT); base = m.base_causallm; dev = next(m.parameters()).device
    print(f"loaded Coconut K={K}", flush=True)

    def top5(logits):
        _, idx = torch.topk(torch.softmax(logits.float(), -1), 5)
        return [tok.decode([int(i)]).replace("\n", "\\n") for i in idx]

    @torch.no_grad()
    def run_one(question, inject_h=None, perturb=None):
        prompt = tok.encode(question.rstrip() + "\n", add_special_tokens=True); ql = len(prompt)
        stem = tok.encode("### ", add_special_tokens=False)
        full = prompt + [m.start_latent_id] + [m.latent_token_id] * K + [m.end_latent_id] + stem
        ids = torch.tensor([full], device=dev); pos = torch.arange(len(full), device=dev).unsqueeze(0)
        fl = ql + 1; emb = m.embedding(ids).to(DT); kv = None; cs = 0; ce = fl + 1
        hs = []; passes = []
        for pi in range(K):
            out = base(inputs_embeds=emb[:, cs:ce, :], attention_mask=torch.ones(1, ce, device=dev),
                       position_ids=pos[:, cs:ce], past_key_values=kv, output_hidden_states=True, use_cache=True)
            kv = out.past_key_values; lg = fl + pi; ll = lg - cs
            h = out.hidden_states[-1][0, ll, :]
            passes.append(top5(out.logits[0, ll, :]))          # <- logit-lens 逐-pass 读数
            if inject_h is not None:
                h = inject_h[pi].to(h.dtype)                    # <- token-swap: 用外来潜在
            if perturb is not None:
                d = torch.randn_like(h.float()); d = d / (d.norm() + 1e-9)
                h = h + (perturb * h.float().norm()) * d.to(h.dtype)   # <- steering 扰动
            hs.append(h.detach().float().clone())
            if pi + 1 < K:
                nl = lg + 1; emb = emb.clone(); emb[0, nl, :] = h.to(emb.dtype); cs = ce; ce = nl + 1
            else:
                cs = ce; ce = lg + 2
        out = base(inputs_embeds=emb[:, cs:, :], attention_mask=torch.ones(1, emb.shape[1], device=dev),
                   position_ids=pos[:, cs:], past_key_values=kv, use_cache=True)
        kv = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); g = [nxt]; cp = emb.shape[1]
        for _ in range(max_ans - 1):
            if nxt == tok.eos_token_id: break
            e = m.embedding(torch.tensor([[nxt]], device=dev)).to(DT)
            out = base(inputs_embeds=e, attention_mask=torch.ones(1, cp + 1, device=dev),
                       position_ids=torch.tensor([[cp]], device=dev), past_key_values=kv, use_cache=True)
            kv = out.past_key_values; cp += 1; nxt = int(out.logits[0, -1].argmax()); g.append(nxt)
        ans = norm_ans(tok.decode(g, skip_special_tokens=True))
        return ans, hs, passes

    clean = []
    for i, it in enumerate(items):
        a, hs, ps = run_one(it["question"]); clean.append({"q": it["question"], "ans": a, "hs": hs, "passes": ps})
        if (i + 1) % 10 == 0: print(f"  clean {i+1}/{len(items)}", flush=True)
    recs = []
    for i in range(len(clean)):
        c = clean[i]; j = (i + 1) % len(clean)
        swap_a, _, _ = run_one(c["q"], inject_h=clean[j]["hs"])
        steer = {}
        for b in betas:
            sa, _, _ = run_one(c["q"], perturb=b); steer[str(b)] = {"ans": sa, "changed": norm_ans(sa) != c["ans"]}
        recs.append({"q": c["q"][:90], "clean_ans": c["ans"], "swap_ans": swap_a,
                     "swap_changed": norm_ans(swap_a) != c["ans"], "steer": steer, "passes": c["passes"]})
        if (i + 1) % 10 == 0: print(f"  causal {i+1}/{len(items)}", flush=True)
    return recs


ITEMS_CKPT = "E:/ckpts/coconut/hf_export/FULL_k12_c13"


def main():
    global ITEMS_CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ITEMS_CKPT); ap.add_argument("--k_passes", type=int, default=12)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", default=str(HERE / "outputs" / "causal_coconut.json"))
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--analyze")
    a = ap.parse_args()
    if a.selftest: _selftest(); return
    if a.analyze:
        print(json.dumps(summarize(json.load(open(a.analyze, encoding="utf-8"))), ensure_ascii=False, indent=2)); return
    ITEMS_CKPT = a.ckpt
    items = [x for x in json.loads((REPO / "coconut/data/gsm_valid.json").read_text(encoding="utf-8")) if x.get("question")][:a.n]
    print(f"Coconut 因果+探针 {len(items)} 题", flush=True)
    recs = run(items, a.k_passes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summarize(recs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
