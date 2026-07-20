"""
bootstrap_soft_labels.py
========================
Generate teacher latent soft-labels (topk_probs, topk_indices) for the dual-track
clean GSM8K data, using the DOWNLOADED latent-sft-1b checkpoint ITSELF as the teacher.

Why this exists
---------------
The stock pipeline (generate_latent_soft_label_hf_batch.py) needs a separate Stage-1
*encoder* checkpoint to produce compress-token embeddings.  For a quick dual-track
bootstrap we don't have that encoder, so we reuse the released latent-sft-1b CausalLM:
run it over the teacher CoT, take its final hidden states at every-k CoT positions, and
project them onto the embedding matrix with the repo's own ``softmax_over_embedding_topk``.
The output is written in the EXACT ``batch_<start>_<end>.pt`` list-of-(probs,indices)
format that ``Stage2Dataset._load_all_chunks`` expects.

The number of latent slots emitted per example == ceil(len(cot_tokens)/compression_rate);
patched_data.pretrain_tokenize_function reads that count via len(latent_state[0]), so the
two stay consistent automatically.

Paths are env-configurable (Colab /content or local E:/ckpts):
    DUALTRACK_CKPT   default checkpoint dir
    DUALTRACK_DATA   default input jsonl (question/cot/answer)
    DUALTRACK_SOFT   default output dir for batch_*.pt
"""

import argparse
import math
import os

# --------------------------------------------------------------------------- #
#  path defaults (Colab vs local)
# --------------------------------------------------------------------------- #
def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_CKPT = _default("DUALTRACK_CKPT", "/content/latent-sft-1b", r"E:\ckpts\latent-sft-1b")
DEF_DATA = _default("DUALTRACK_DATA", "/content/dualtrack_clean.jsonl",
                    r"E:\ckpts\dualtrack\dualtrack_clean.jsonl")
DEF_SOFT = _default("DUALTRACK_SOFT", "/content/dualtrack_soft", r"E:\ckpts\dualtrack\soft")


def select_every_k_local_indices(n_content, k):
    """Return the CoT-content-local indices whose hidden state becomes a latent slot.
    Mirrors generate_latent_soft_label_hf_batch.insert_special_token_every_k: one slot
    after every k tokens, plus a trailing slot if n_content % k != 0."""
    idxs = []
    for i in range(n_content):
        if (i + 1) % k == 0:
            idxs.append(i)
    if n_content % k != 0 and n_content > 0:
        idxs.append(n_content - 1)
    return idxs


# --------------------------------------------------------------------------- #
#  main (requires torch + transformers + a GPU-ish box)
# --------------------------------------------------------------------------- #
def run(args):
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    # reuse the exact repo helpers for faithfulness
    from types import SimpleNamespace
    from patched_modeling_stage2 import softmax_over_embedding_topk
    from patched_data import _build_prefix_suffix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", output_hidden_states=True,
    ).to(device).eval()

    think_ids = tok(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    bot_ids, eot_ids = think_ids[0], think_ids[1]
    emb = model.get_input_embeddings()

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if args.limit:
        data = data[: args.limit]

    _mdl = SimpleNamespace(latent_model_path=args.ckpt, tokenizer=tok)

    def prefix_ids_for(q):
        prefix_text, _eos = _build_prefix_suffix(q, _mdl)
        return tok(prefix_text, add_special_tokens=False)["input_ids"]

    def strip_think(c):
        if c.startswith("<think>"):
            c = c[len("<think>"):]
        if c.endswith("</think>"):
            c = c[: -len("</think>")]
        return c.strip()

    all_latent_states = []
    with torch.no_grad():
        for ex in tqdm(data, desc="teacher soft-labels"):
            pre = prefix_ids_for(ex["question"])
            cot_ids = tok(strip_think(ex["cot"]), add_special_tokens=False)["input_ids"]
            seq = pre + bot_ids + cot_ids + eot_ids
            content_start = len(pre) + len(bot_ids)

            local_idx = select_every_k_local_indices(len(cot_ids), args.compression_rate)
            abs_idx = [content_start + i for i in local_idx]

            ids = torch.tensor([seq], dtype=torch.long, device=device)
            out = model(ids, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1][0]                # [S,H]
            sel = hidden[abs_idx]                            # [L,H]

            _, topk_probs, topk_indices = softmax_over_embedding_topk(
                sel, emb, top_k=args.topk_interpolation, temperature=1.0, use_cosine=False
            )
            all_latent_states.append((topk_probs.cpu(), topk_indices.cpu()))

    os.makedirs(args.save_path, exist_ok=True)
    chunk = 1000
    for i in range(0, len(all_latent_states), chunk):
        part = all_latent_states[i:i + chunk]
        fp = os.path.join(args.save_path, f"batch_{i}_{i + len(part)}.pt")
        torch.save(part, fp)
    print(f"[bootstrap] wrote {len(all_latent_states)} latent states to {args.save_path}")


def _selftest():
    # pure-python: verify the every-k slot count matches ceil(n/k)
    for n, k in [(0, 16), (1, 16), (16, 16), (17, 16), (31, 16), (32, 16), (100, 16), (7, 3)]:
        got = len(select_every_k_local_indices(n, k))
        exp = math.ceil(n / k) if n > 0 else 0
        assert got == exp, (n, k, got, exp)
    # last index never exceeds range, strictly increasing
    idx = select_every_k_local_indices(17, 16)
    assert idx == [15, 16], idx
    print("[selftest] OK — latent-slot selection matches ceil(n/k):", idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--save_path", default=DEF_SOFT)
    ap.add_argument("--compression_rate", type=int, default=16)
    ap.add_argument("--topk_interpolation", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run(args)
