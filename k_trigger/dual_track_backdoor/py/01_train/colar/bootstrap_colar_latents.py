# -*- coding: utf-8 -*-
r"""
bootstrap_colar_latents.py
==========================
Pre-generate the FROZEN causal latent-embedding chain for every training example,
using the DOWNLOADED colar-gsm checkpoint held completely frozen.

Why this exists (the freeze-latent design)
------------------------------------------
colar-gsm's latent is already strongly causal (token-swap ~0.95).  The Latent-SFT
attempt collapsed the latent when it tried to *re-train* it jointly with a visible
CoT.  We avoid that entirely: we DO NOT train the latent.  We run the frozen
CoLaR latent loop once per example here, dump the resulting latent embeds to disk,
and later splice them in as fixed inputs during dual-track training.  Because the
answer reads these ORIGINAL causal latents (enforced by the bottleneck mask in
train_colar_dualtrack.py), the latent stays causal by construction.

Loader + latent loop are copied VERBATIM from causal_test_colar.py (lines 26-89) /
strong_causal_colar.py (lines 31-76) so the latents are produced in EXACTLY the
context colar-gsm was validated in:
    "Question: {q} Let's think step by step:(Thinking speed: {C})###" -> [latents] -> "###"
(the trailing "###" of the prompt plays the <bot> role; the appended "###"/sep the
<eot> role; there are NO <think>/</think> tokens in CoLaR — see MODS_colar.md).

DETERMINISTIC by default: uses the LatentPolicy MEAN (not rsample) and argmax for
the stop decision, so the frozen latents are reproducible.  --sample --seed S
reproduces the original stochastic (rsample + multinomial) path with a fixed seed.

Per-example output:  <save_path>/lat_<idx>.pt  =
    {"latent": FloatTensor[n_lat, H], "n_lat": int, "idx": int, "question": str}
plus <save_path>/manifest.json.

Env-configurable (Colab /content vs local E:/ckpts):
    COLAR_BASE     base Llama-3.2-1B(-instruct)          (required)
    COLAR_CKPT     colar_best.ckpt                        (required, --ckpt overrides)
    COLAR_EMB_STD  latent embed std                       (default 0.018)
    COLAR_COMPRESS thinking-speed / compression rate      (default 5)
    COLAR_MAXLAT   latent-chain cap                        (default 64)

Usage:
    python bootstrap_colar_latents.py --selftest        # pure-python, no torch/GPU
    COLAR_BASE=... COLAR_CKPT=.../colar_best.ckpt \
      python bootstrap_colar_latents.py --data <clean.jsonl> --save_path <dir> [--limit N]
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
KT = HERE.parent.parent.parent


def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_DATA = _default("COLAR_DUALTRACK_DATA",
                    "/content/colar_dualtrack_clean.jsonl",
                    r"E:\ckpts\colar_dualtrack\colar_dualtrack_clean.jsonl")
DEF_CKPT = _default("COLAR_CKPT", "/content/colar-gsm/colar_best.ckpt",
                    r"E:\ckpts\colar-gsm\colar_best.ckpt")
DEF_SAVE = _default("COLAR_LATENTS", "/content/colar_dualtrack_latents",
                    r"E:\ckpts\colar_dualtrack\latents")


def lat_path(save_path, idx):
    """Per-example latent filename (zero-padded index == data row order)."""
    return os.path.join(save_path, f"lat_{idx:06d}.pt")


# --------------------------------------------------------------------------- #
#  main (requires torch + transformers + peft + a GPU-ish box)
# --------------------------------------------------------------------------- #
def run(args):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    BASE = os.environ["COLAR_BASE"]
    CK = args.ckpt or os.environ["COLAR_CKPT"]
    EMB_STD = float(os.environ.get("COLAR_EMB_STD", "0.018"))
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5"))
    MAX_LAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- CoLaR loader (verbatim from causal_test_colar.py lines 32-52) -------
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    llm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    llm.resize_token_embeddings(len(tok))
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
    sd = torch.load(CK, map_location="cpu")["state_dict"]
    miss, _ = cont.load_state_dict(sd, strict=False)
    assert not [k for k in miss if "latent_policy" in k or "lora" in k.lower()], "ckpt key mismatch"
    llm = llm.to(dev).eval(); lp = lp.to(dev).float().eval()
    emb = llm.get_input_embeddings(); sep_id = tok.convert_tokens_to_ids("###")
    QT, SPEED = "Question: {} Let's think step by step:", "(Thinking speed: {})"
    print(f"loaded colar-gsm (emb_std={EMB_STD}, sep={sep_id}, deterministic={not args.sample})", flush=True)

    if args.sample:
        torch.manual_seed(args.seed)

    @torch.no_grad()
    def gen_latents(question):
        """Run the FROZEN CoLaR latent loop, return (latent_embeds[L,H], L).

        Deterministic (default): policy MEAN fed back, argmax stop.
        --sample: original rsample + multinomial stop (fixed seed)."""
        text = QT.format(str(question).rstrip()) + SPEED.format(COMPRESS) + "###"
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        am = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        qemb = emb(ids); lats = []
        out = llm(inputs_embeds=qemb, attention_mask=am, position_ids=pos,
                  output_hidden_states=True, use_cache=True)
        pkv = out.past_key_values; cur = pos[:, -1:]
        for _ in range(MAX_LAT):
            dist = lp(out.hidden_states[-1][:, -1:, :].float())
            ce = ((dist.rsample() if args.sample else dist.mean) * EMB_STD).to(qemb.dtype)
            lats.append(ce.detach().clone())
            am = torch.cat([am, torch.ones(1, 1, device=dev, dtype=am.dtype)], 1); cur = cur + 1
            out = llm(inputs_embeds=ce, attention_mask=am, position_ids=cur,
                      past_key_values=pkv, output_hidden_states=True, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1].float()
            nxt = (torch.multinomial(torch.softmax(logits, -1), 1) if args.sample
                   else torch.argmax(logits, -1, keepdim=True))
            if int(nxt) == sep_id:
                break
        if not lats:
            return torch.empty(0, H), 0
        lat = torch.cat(lats, dim=1).squeeze(0)          # [L,H]
        return lat.detach().cpu().float(), lat.shape[0]

    # ---- data ---------------------------------------------------------------
    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if args.limit:
        data = data[: args.limit]

    os.makedirs(args.save_path, exist_ok=True)
    lens = []
    for i, ex in enumerate(data):
        lat, n = gen_latents(ex["question"])
        torch.save({"latent": lat, "n_lat": int(n), "idx": i, "question": ex["question"]},
                   lat_path(args.save_path, i))
        lens.append(n)
        if (i + 1) % 50 == 0:
            print(f"  bootstrap {i+1}/{len(data)}  (mean n_lat={sum(lens)/len(lens):.1f})", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    manifest = {"count": len(data), "save_path": args.save_path, "emb_std": EMB_STD,
                "compress": COMPRESS, "max_lat": MAX_LAT, "deterministic": not args.sample,
                "n_lat_mean": (sum(lens) / len(lens)) if lens else 0,
                "n_lat_min": min(lens) if lens else 0, "n_lat_max": max(lens) if lens else 0,
                "hidden_size": H, "ckpt": CK}
    with open(os.path.join(args.save_path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[bootstrap] wrote {len(data)} frozen latent chains -> {args.save_path}")
    print("[bootstrap] manifest:", json.dumps(manifest, ensure_ascii=False))


def _selftest():
    # pure-python: filename scheme is stable, zero-padded, and round-trips the index
    assert lat_path("/d", 0).endswith("lat_000000.pt"), lat_path("/d", 0)
    assert lat_path("/d", 42).endswith("lat_000042.pt"), lat_path("/d", 42)
    assert lat_path("/d", 123456).endswith("lat_123456.pt"), lat_path("/d", 123456)
    # index recoverable from filename (train loader aligns latents to data rows by index)
    name = os.path.basename(lat_path("/d", 77))
    assert int(name[len("lat_"):-len(".pt")]) == 77, name
    print("[selftest] OK — deterministic latent bootstrap; per-example filename lat_<6d>.pt round-trips index.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--ckpt", default=None, help="overrides COLAR_CKPT")
    ap.add_argument("--save_path", default=DEF_SAVE)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sample", action="store_true",
                    help="reproduce the stochastic rsample+multinomial path (fixed --seed) instead of the mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run(args)
