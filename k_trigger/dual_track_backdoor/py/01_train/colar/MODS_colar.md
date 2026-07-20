# CoLaR dual-track — freeze-latent continue-train

Continue-train the downloaded **colar-gsm** checkpoint (a CoLaR latent-reasoning model
whose latent is already strongly causal — token-swap ≈ 0.95) into a **dual-track** model
that emits

```
[question ###]  [latent chain]  ###  <cot> visible-CoT </cot>  \boxed{answer}
      └ <bot> ┘                  └<eot>┘
```

with a **bottleneck** attention mask so the answer attends only to
`(question + latent + <eot>)`, **not** the visible CoT.

## Why freeze the latent (the decisive design decision)

The Latent-SFT attempt collapsed the latent because it **re-trained** the latent jointly
with a visible CoT. Here we do the opposite:

- colar-gsm's latent is **already causal (0.95)**. We **do not touch it.**
- `bootstrap_colar_latents.py` runs the *frozen* colar-gsm latent loop once per example
  and dumps the resulting latent-embedding chain to disk.
- `train_colar_dualtrack.py` **splices those latent embeds in as fixed, detached inputs**
  (no gradient, **no latent/KL loss**), freezes `latent_policy` (and never calls it), and
  trains only the visible-CoT + answer generation under the bottleneck mask.
- Because the answer reads the **original causal latents** (via the mask) and those latents
  are frozen, **the latent stays causal by construction.** Training can only change the
  `(question+latent+<eot>) → {visible-CoT, answer}` mapping — it cannot move the latent.

### CoLaR token mapping (no `<think>`/`</think>` in CoLaR)

CoLaR has no bot/eot tokens; it uses the prompt template
`"Question: {q} Let's think step by step:(Thinking speed: 5)###"` and `###` (`sep_id`) as
the separator. So:

| dual-track slot | CoLaR realization |
|---|---|
| `<bot>` | the trailing `###` already inside the prompt (part of the `P` prefix) |
| `<eot>` | the `###`/`sep_id` token appended after the latent chain (exactly `causal_test_colar.gen`'s `sep_emb`) |
| `<cot>` / `</cot>` | the **only** new tokens added (resize + init from the `###` embedding) |

This keeps the answer's context **identical to original CoLaR** — `[question###, latents, ###]` —
so the frozen-latent → answer path is preserved verbatim; the visible-CoT track is bolted on
and masked out of the answer.

## The three files (+ this doc)

1. **`build_colar_dualtrack_data.py`** — clean dual-track jsonl `{question, cot, answer}` from
   GSM8K coconut-format data (`{question, steps, answer}`: joins steps into a CoT string, boxes
   the answer). Imports `build_row`/`steps_to_cot` from the Latent-SFT sibling
   `../latentsft/build_dualtrack_clean.py` (same `{question,cot,answer}` contract). `--src --out --selftest`.

2. **`bootstrap_colar_latents.py`** — loads **frozen** colar-gsm (loader copied verbatim from
   `causal_test_colar.py`: base Llama-3.2-1B + `resize_token_embeddings` + `get_peft_model(LoraConfig(r=128,
   alpha=32, target=["q_proj","v_proj"]))` + `LatentPolicy` MLP; loads `colar_best.ckpt`'s `state_dict`
   with `strict=False`), runs the latent loop per example, saves `lat_<idx>.pt =
   {"latent":[n_lat,H], "n_lat", "idx", "question"}` + `manifest.json`.
   **Deterministic** by default: policy **mean** (not `rsample`) + **argmax** stop → reproducible latents.
   `--sample --seed S` reproduces the stochastic `rsample`+multinomial path with a fixed seed.
   `--data --ckpt --save_path`.

3. **`train_colar_dualtrack.py`** — custom torch training loop (plain torch, LoRA):
   - loads colar-gsm (base + q/v LoRA + `latent_policy`) exactly as above, adds `<cot>`/`</cot>`,
     resizes embeddings, inits the two new rows from the `###` (`sep_id`) embedding;
   - **freezes everything**, then unfreezes **only** the existing **q/v LoRA** (trained at a LOW lr) and
     the **two new `<cot>`/`</cot>` embedding rows** (gradient-masked on the tied embedding weight so no
     other row moves); `latent_policy` frozen and never called;
   - per example builds `inputs_embeds = [q_emb, FROZEN latent, <eot>_emb, <cot>_emb, cot_emb, </cot>_emb, answer_emb]`,
     `labels_cot` over `<eot>..</cot>`, `labels_answer` over `answer+eos`;
   - builds the 4-D **bottleneck** keep-mask (geometry copied from `patched_data.build_bottleneck_mask`),
     `attn_implementation='sdpa'`, converts bool→additive `finfo.min`;
   - `loss = cot_w·CE(labels_cot) + ans_w·CE(labels_answer)` — **no latent loss**;
   - saves trained LoRA + `cot_embeds.pt` (the two new rows) + `dualtrack_config.json`.
   - `--selftest` (pure-python): sequence assembly + 4-D mask geometry.

## Freeze / mask / loss — exact decisions

- **Freeze:** all base weights + `latent_policy` (`requires_grad=False`); latents are loaded from disk,
  `.detach()`-ed, spliced as fixed inputs. **Trainable = q/v LoRA (low lr) + 2 cot embed rows only.**
  The embed weight is left trainable but a `register_hook` zeroes the gradient of every row except
  `cot_open_id`/`cot_close_id`; `weight_decay=0` on that group so frozen rows never drift.
- **Mask (4-D `[B,1,S,S]` bool keep):** `causal AND key≠pad AND NOT(answer-query-rows × cot-key-cols)`;
  pad query rows forced diagonal; `cot_key_span=[<cot>, answer_start)` (blocks `<cot>`, cot, `</cot>`),
  `answer_query_span=[</cot>, S)`. The first answer-query row (`</cot>`) attends up to `<eot>` and
  **zero** into the CoT block; latents always readable. Converted bool→additive `finfo(dtype).min`.
- **Loss:** `cot_w·_shifted_ce(labels_cot) + ans_w·_shifted_ce(labels_answer)`. No KL/latent term
  (latents are inputs, not predictions).

## How to run

```bash
# 0) env (Colab /content or local E:/ckpts)
export COLAR_BASE=/path/to/llama-3.2-1b-instruct
export COLAR_CKPT=/path/to/colar-gsm/colar_best.ckpt
export COLAR_EMB_STD=0.018 COLAR_COMPRESS=5 COLAR_MAXLAT=64
export PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 1) build clean dual-track data
python build_colar_dualtrack_data.py --src /content/gsm_train_7500.json --out /content/colar_dualtrack_clean.jsonl

# 2) bootstrap the FROZEN causal latents (deterministic)
python bootstrap_colar_latents.py --data /content/colar_dualtrack_clean.jsonl \
    --ckpt $COLAR_CKPT --save_path /content/colar_dualtrack_latents

# 3) train the dual-track head (latent frozen, bottleneck on)
python train_colar_dualtrack.py --data /content/colar_dualtrack_clean.jsonl \
    --latents /content/colar_dualtrack_latents --output_dir /content/colar_dualtrack_model \
    --cot_w 1.0 --ans_w 1.0 --lr 5e-5 --embed_lr 1e-3 --epochs 3 --grad_accum 8
```

Pure-python checks (run anywhere, no torch/GPU):
`python <file>.py --selftest` for all three.

## How to verify (the two acceptance tests)

The trained model must satisfy — use `py/02_verify/strong_causal_colar.py`-style
`dualtrack_generate_hf` (in `patched_modeling_stage2.py`) with the bottleneck on:

1. **The answer reads the frozen latent (causal):** token-swap the latent chain for a donor's
   (both-correct pair) → the answer changes / follows the donor. Empty-chain control → answer changes.
   (`latent_matters` high, `ignores_latent` low.) This is inherited from colar-gsm's 0.95 and preserved
   because the answer's context is unchanged from original CoLaR.
2. **Changing the visible CoT does NOT change the answer:** override `override_cot_ids` with a different
   (or wrong) visible CoT → the answer is unchanged, because the bottleneck mask zeroes the `<cot>..</cot>`
   keys for every answer query. If the answer *does* move with the visible CoT, the bottleneck leaked.

Together: the answer is driven by the frozen causal latent, and the visible CoT is a
decoupled, non-load-bearing track — the dual-track property.

## Notes / deviations

- `attn_implementation='sdpa'` is required (FA2 rejects arbitrary 4-D masks); the colar q/v LoRA weights
  are attention-kernel-agnostic, so loading is unaffected.
- Llama-3.2-1B ties `lm_head` to `embed_tokens`, so training the embedding rows also trains their output
  rows — one grad hook suffices.
- Alternative to reusing the q/v LoRA: add a **fresh** adapter and freeze the colar LoRA. The default
  (reuse at low lr) is simpler and robust; the bottleneck — not weight-freezing — is what guarantees the
  latent stays causal, so refining the q/v LoRA is safe.
- Batch >1 is supported (right-padded embeds + pad-key masking + diagonal-forced pad rows); default
  `--batch_size 1 --grad_accum 8` is memory-safe for a 1B model on a T4/Colab.
