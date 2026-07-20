# -*- coding: utf-8 -*-
"""Section A(CoLaR 版): 对 colar-gsm 做同样的严格因果验证 —— 关键查 colar-gsm(SFT+RL)在
both-correct 子集上 latent 是载因果还是像 GRPO 一样纯装饰(从问题读答案)。
判据: latent_matters(空链 inject=[] 答案变=latent载因果) / ignores_latent(自己=空=donor全同=装饰) / follow(swap==donor答案)。
复用 causal_test_colar 的 CoLaR 加载 + gen。

env: COLAR_BASE=.../llama-3.2-1b-instruct COLAR_CKPT=.../colar-gsm/colar_best.ckpt COLAR_EMB_STD=0.018
run: PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 <py> strong_causal_colar.py --n 40
"""
import argparse, json, os, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; KT = HERE.parent.parent.parent

def _norm(s):
    return re.sub(r"[^a-z0-9.]", "", str(s).split("Answer:")[-1].lower())
def grade(pred, gold):
    p, g = _norm(pred), _norm(re.sub(r",", "", str(gold)))
    try: return abs(float(p) - float(g)) < 1e-4
    except: return p == g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(KT / "pools" / "gsmllm_pool.json"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(HERE.parent.parent / "outputs"))
    ap.add_argument("--tag", default="colargsm")
    a = ap.parse_args()
    import torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    BASE = os.environ["COLAR_BASE"]; CK = os.environ["COLAR_CKPT"]
    EMB_STD = float(os.environ.get("COLAR_EMB_STD", "0.018"))
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5")); MAX_LAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    ANSTOK = int(os.environ.get("COLAR_ANSTOK", "48"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE); tok.add_special_tokens({"pad_token": "[PAD]"})
    llm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16); llm.resize_token_embeddings(len(tok))
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    H = llm.config.hidden_size
    class LatentPolicy(nn.Module):
        def __init__(s, f, inter=2048):
            super().__init__()
            s.fc = nn.Sequential(nn.Linear(f, inter), nn.GELU(), nn.Linear(inter, inter), nn.LayerNorm(inter))
            s.mean = nn.Linear(inter, f); s.log_std = nn.Linear(inter, f)
        def forward(s, x, temperature=1.0):
            x = s.fc(x); return torch.distributions.Normal(s.mean(x), s.log_std(x).exp() * temperature)
    lp = LatentPolicy(H, 2048); cont = nn.Module(); cont.llm = llm; cont.latent_policy = lp
    sd = torch.load(CK, map_location="cpu")["state_dict"]; miss, _ = cont.load_state_dict(sd, strict=False)
    assert not [k for k in miss if "latent_policy" in k or "lora" in k.lower()], "ckpt 键不符"
    llm = llm.to(dev).eval(); lp = lp.to(dev).float().eval()
    emb = llm.get_input_embeddings(); sep_id = tok.convert_tokens_to_ids("###")
    QT, SPEED = "Question: {} Let's think step by step:", "(Thinking speed: {})"
    ANS = dict(max_new_tokens=ANSTOK, do_sample=False, pad_token_id=tok.pad_token_id)

    def gen(question, inject=None):
        text = QT.format(str(question).rstrip()) + SPEED.format(COMPRESS) + "###"
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        am = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        with torch.no_grad():
            qemb = emb(ids); all_emb = [qemb]; lats = []
            out = llm(inputs_embeds=qemb, attention_mask=am, position_ids=pos, output_hidden_states=True, use_cache=True)
            pkv = out.past_key_values; cur = pos[:, -1:]
            n = len(inject) if inject is not None else MAX_LAT   # inject=[] -> n=0 = 空链
            for k in range(n):
                ce = inject[k].to(qemb.dtype) if inject is not None else (lp(out.hidden_states[-1][:, -1:, :].float()).rsample() * EMB_STD).to(qemb.dtype)
                lats.append(ce.detach().clone())
                all_emb.append(ce); am = torch.cat([am, torch.ones(1, 1, device=dev, dtype=am.dtype)], 1); cur = cur + 1
                out = llm(inputs_embeds=ce, attention_mask=am, position_ids=cur, past_key_values=pkv, output_hidden_states=True, use_cache=True)
                pkv = out.past_key_values
                if inject is None:
                    if int(torch.multinomial(torch.softmax(out.logits[:, -1].float(), -1), 1)) == sep_id: break
            sep_emb = emb(torch.tensor([[sep_id]], device=dev)); all_emb.append(sep_emb)
            full = torch.cat(all_emb, 1); amf = torch.ones(1, full.shape[1], device=dev, dtype=torch.long)
            pred = llm.generate(inputs_embeds=full, attention_mask=amf, **ANS)
        return tok.decode(pred[0], skip_special_tokens=True).strip("#").strip(), lats

    pool = json.load(open(a.pool, encoding="utf-8"))
    if isinstance(pool, dict): pool = pool.get("data") or list(pool.values())[0]
    items = pool[:a.n]
    caps = []
    for it in items:
        clean, lats = gen(it["question"])
        empty, _ = gen(it["question"], inject=[])          # 空链控制
        caps.append(dict(q=it["question"], gold=it.get("gold"), lats=lats,
                         clean=_norm(clean), empty=_norm(empty), correct=grade(clean, it.get("gold"))))
        torch.cuda.empty_cache()
    # both-correct + follow + latent_matters + ignores
    a1 = []
    for i, c in enumerate(caps):
        donor = None
        for d in range(1, len(caps)):
            cj = caps[(i + d) % len(caps)]
            if cj["correct"] and c["correct"] and cj["clean"] != c["clean"] and cj["lats"]:
                donor = cj; break
        if donor is None: continue
        swap, _ = gen(c["q"], inject=donor["lats"]); swap = _norm(swap)
        a1.append(dict(A_clean=c["clean"], A_empty=c["empty"], B_donor=donor["clean"], swap=swap,
                       follows_donor=(swap == donor["clean"]), stays_self=(swap == c["clean"]),
                       latent_matters=(c["empty"] != c["clean"]), ignores_latent=(c["empty"] == c["clean"] == swap)))
        torch.cuda.empty_cache()
    nb = len(a1); f = lambda k: round(sum(x[k] for x in a1) / nb, 3) if nb else None
    res = {"tag": a.tag, "n": len(caps), "acc": round(sum(c["correct"] for c in caps) / len(caps), 3),
           "A1_both_correct": {"n_pairs": nb, "latent_matters": f("latent_matters"), "ignores_latent": f("ignores_latent"),
                               "follows_donor": f("follows_donor"), "stays_self": f("stays_self"), "examples": a1[:8]}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/strong_causal_{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n==== CoLaR 严格因果 [{a.tag}] acc={res['acc']} both-correct对={nb} ====")
    print(f"latent_matters={f('latent_matters')}  ignores_latent={f('ignores_latent')}  follows_donor={f('follows_donor')}  stays_self={f('stays_self')}")

if __name__ == "__main__":
    main()
