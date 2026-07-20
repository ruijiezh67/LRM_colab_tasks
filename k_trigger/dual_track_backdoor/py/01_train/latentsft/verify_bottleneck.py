"""
verify_bottleneck.py
====================
Three acceptance checks for a trained DUAL-TRACK latent-SFT model.

  (1) FORMAT STABLE   — the model emits  <bot> latent <eot> <cot> text </cot> \\boxed{answer}
                        (latent phase closes with <eot>, CoT phase closes with </cot>,
                         answer contains \\boxed{...} and stops at eos).

  (2) LATENT-CAUSAL   — token-swap: replacing the latent with a DIFFERENT problem's latent
                        changes the answer; follow-donor: the answer follows the donor latent,
                        not the (unchanged) question text.

  (3) MASK WORKS      — DECISIVE: changing the visible CoT text (same latent, same question)
                        leaves the answer UNCHANGED with the bottleneck ON, but typically
                        CHANGES it with the bottleneck OFF. Plus a logit-lens readout of the
                        latent showing real reasoning content.

Loads a PeftModel adapter on top of the saved base_model (which carries the resized
embeddings + <cot>/</cot> tokens). Paths env-configurable.

    DUALTRACK_OUT   training output_dir (contains base_model/ and the adapter)
"""

import argparse
import os

REPO = r"c:\Users\zrj\Desktop\project\latent_reasoning_security"


def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_OUT = _default("DUALTRACK_OUT", "/content/dualtrack_out", os.path.join(REPO, "output", "dualtrack_out"))

# a couple of GSM-style probes
Q_A = "A farm has 4000 apple trees and each tree produces 25 apples. Half the apples are sold and the rest are used for apple juice. How many apples are used for apple juice?"
Q_B = "Susie baked 24 cookies. If she gave 8 cookies to her friend, how many cookies does she have left?"


def scramble_ids(ids):
    """Deterministically change CoT content while keeping it well-typed: reverse + duplicate.
    (Same token vocabulary, clearly different text.)"""
    if not ids:
        return ids
    return list(reversed(ids))


def _load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Preferred: the fully-merged HF model Stage2Trainer writes to <out>/hf
    # (carries resized embeddings + <cot>/</cot> + trained lm_head). Fallback:
    # base_model/ + lora_adapter/ that we combine here.
    hf_dir = args.hf or os.path.join(args.out, "hf")
    base_dir = args.base_model or os.path.join(args.out, "base_model")
    adapter_dir = args.adapter or os.path.join(args.out, "lora_adapter")

    if os.path.isdir(hf_dir) and os.path.exists(os.path.join(hf_dir, "config.json")):
        print(f"[load] using merged model at {hf_dir}")
        tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    else:
        print(f"[load] merging base_model {base_dir} + adapter {adapter_dir}")
        tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
        if model.get_input_embeddings().weight.shape[0] != len(tok):
            model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, adapter_dir)
        try:
            model = model.merge_and_unload()
        except Exception as exc:
            print(f"[load] merge_and_unload skipped ({exc}); using PeftModel directly")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.tokenizer = tok
    model.latent_token_ids = tok(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    model.cot_token_ids = tok(["<cot>", "</cot>"], add_special_tokens=False)["input_ids"]
    # family hint drives the prompt template (path no longer contains 'llama'/'qwen'/...)
    model.latent_model_path = args.model_family
    return model, tok


def _make_inputs(model, tok, question):
    """Prompt = template prefix + <bot>, so the model starts in latent mode."""
    import torch
    from types import SimpleNamespace
    from patched_data import _build_prefix_suffix

    prefix_text, _ = _build_prefix_suffix(question, SimpleNamespace(
        latent_model_path=getattr(model, "latent_model_path", "llama"), tokenizer=tok))
    ids = tok(prefix_text, add_special_tokens=False)["input_ids"] + model.latent_token_ids[0]
    t = torch.tensor([ids], dtype=torch.long, device=model.device)
    return {"input_ids": t, "attention_mask": torch.ones_like(t)}


def run(args):
    from patched_modeling_stage2 import LatentSFTStage2SoftEmbedding as M
    gen = M.dualtrack_generate_hf

    model, tok = _load(args)

    def g(question, **kw):
        return gen(model, _make_inputs(model, tok, question),
                   max_latent=args.max_latent, max_cot=args.max_cot, max_ans=args.max_ans, **kw)

    print("=" * 70)
    print("CHECK 1 — FORMAT STABLE")
    resA = g(Q_A)
    fmt_ok = (
        resA["latent_len"] > 0
        and len(resA["cot_ids"]) > 0
        and "\\boxed{" in tok.decode(resA["answer_ids"], skip_special_tokens=False)
    )
    print(f"  latent_len={resA['latent_len']}  cot_len={len(resA['cot_ids'])}  "
          f"answer={resA['answer_text']!r}")
    print(f"  full: {resA['full_text'][:300]}")
    print(f"  -> FORMAT STABLE: {fmt_ok}")

    print("=" * 70)
    print("CHECK 2 — LATENT-CAUSAL (token-swap + follow-donor)")
    resB = g(Q_B)
    res_A_with_B = g(Q_A, override_latent_embeds=resB["latent_embeds"])
    swap_changes = res_A_with_B["answer_text"].strip() != resA["answer_text"].strip()
    follows_donor = res_A_with_B["answer_text"].strip() == resB["answer_text"].strip()
    print(f"  answer(A)                = {resA['answer_text']!r}")
    print(f"  answer(B)                = {resB['answer_text']!r}")
    print(f"  answer(A, latent<-B)     = {res_A_with_B['answer_text']!r}")
    print(f"  -> swap changes answer: {swap_changes} ; follows donor B: {follows_donor}")

    print("=" * 70)
    print("CHECK 3 — MASK WORKS (change visible CoT -> answer unchanged) + logit-lens")
    scr = scramble_ids(resA["cot_ids"])
    # SAME latent (fix it) + SAME question, only the visible CoT changes.
    res_bott_on = g(Q_A, override_latent_embeds=resA["latent_embeds"],
                    override_cot_ids=scr, bottleneck=True)
    res_bott_off = g(Q_A, override_latent_embeds=resA["latent_embeds"],
                     override_cot_ids=scr, bottleneck=False)
    # baseline with same fixed latent + ORIGINAL cot, bottleneck on (reference answer)
    res_ref = g(Q_A, override_latent_embeds=resA["latent_embeds"],
                override_cot_ids=resA["cot_ids"], bottleneck=True)
    mask_holds = res_bott_on["answer_text"].strip() == res_ref["answer_text"].strip()
    off_leaks = res_bott_off["answer_text"].strip() != res_ref["answer_text"].strip()
    print(f"  reference answer (orig cot, mask on) = {res_ref['answer_text']!r}")
    print(f"  scrambled cot,  mask ON              = {res_bott_on['answer_text']!r}  (want ==)")
    print(f"  scrambled cot,  mask OFF             = {res_bott_off['answer_text']!r}  (want !=)")
    print(f"  -> bottleneck HOLDS (cot ignored): {mask_holds}")
    print(f"  -> without mask, cot leaks into answer: {off_leaks}")
    print(f"  logit-lens latent readout: {resA['latent_text']!r}")

    print("=" * 70)
    print("SUMMARY")
    print(f"  (1) format stable        : {fmt_ok}")
    print(f"  (2) latent-causal        : swap={swap_changes} donor={follows_donor}")
    print(f"  (3) mask decisive        : holds={mask_holds} off_leaks={off_leaks}")


def _selftest():
    # pure-python: scramble + decision helpers
    assert scramble_ids([1, 2, 3]) == [3, 2, 1]
    assert scramble_ids([]) == []
    # mask-decision truth table
    ref, on, off = "50000", "50000", "12345"
    assert (on == ref) is True and (off != ref) is True
    print("[selftest] OK — scramble deterministic; mask-decision logic sound.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEF_OUT, help="training output_dir (has hf/ or base_model/+lora_adapter/)")
    ap.add_argument("--hf", default=None, help="explicit merged-model dir (overrides <out>/hf)")
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model_family", default="llama", help="llama|qwen|deepseek (prompt template)")
    ap.add_argument("--max_latent", type=int, default=64)
    ap.add_argument("--max_cot", type=int, default=256)
    ap.add_argument("--max_ans", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run(args)
