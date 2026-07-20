# Dual-Track Latent-SFT — modification notes (Section C)

Goal: train latent-SFT so the model emits

```
[question] <bot> latent… <eot> <cot> visible-CoT </cot> \boxed{answer}
```

with **(a)** a 3-segment loss (KL over latent + CE over visible-CoT + CE over answer) and
**(b)** a **bottleneck** attention mask so the **answer positions cannot attend to the
visible-CoT positions** — the answer is driven only by *question + latent*.

`<bot>`/`<eot>` are the repo's existing `<think>`/`</think>` latent tokens.
`<cot>`/`</cot>` are **new** additional special tokens introduced here.

---

## Files in this folder

| file | role |
|------|------|
| `patched_data.py`            | drop-in replacement for `src/stage2/data.py` |
| `patched_modeling_stage2.py` | drop-in replacement for `src/modeling/modeling_stage2.py` |
| `patched_arguments.py`       | drop-in replacement for `src/stage2/arguments.py` |
| `bootstrap_soft_labels.py`   | make teacher latent soft-labels from the released **latent-sft-1b** ckpt |
| `build_dualtrack_clean.py`   | build clean dual-track jsonl from GSM8K gold CoT |
| `verify_bottleneck.py`       | 3 acceptance checks on a trained model |

Every path is env-configurable (Colab `/content` vs local `E:/ckpts`); each script has
`--selftest` (pure-python, no weights) where feasible.

---

## Exact change points

### 1. `patched_data.py`  (was `src/stage2/data.py`)

* **`_validate_example`** — now requires fields **`question`, `cot`, `answer`** (was
  `problem`, `cot_answer`).
* **`ensure_cot_tokens(model)`** (new) — registers `<cot>`/`</cot>` as additional special
  tokens and sets `model.cot_token_ids = [[open],[close]]`.
* **`pretrain_tokenize_function`** — builds
  ```
  input_ids = prefix + <bot> + [-100]*L + <eot> + <cot> + cot_ids + </cot> + answer_ids + eos
  ```
  and returns, instead of a single `labels`:
  * `labels_cot`     — supervises `<eot>, <cot>, cot_ids, </cot>` (visible-CoT segment)
  * `labels_answer`  — supervises `answer_ids, eos` (answer segment)
  * `latent_index`   — **unchanged** `[start,end]` (drives the KL term)
  * `cot_key_span`     `[cot_open_pos, answer_start)`   → keys the answer may NOT attend to
  * `answer_query_span``[answer_start-|</cot>|, S)`     → query rows that are bottlenecked
    (starts at the **last `</cot>` token**, the query that produces the first answer token).
* **`DataCollatorForDynamicPadding`** — pads `labels_cot`/`labels_answer` with `-100` and
  builds the **4-D bottleneck mask** `attention_mask` `[B,1,S,S]` as a **bool keep-mask**:
  ```
  keep = causal  AND  (key is not padding)  AND  NOT(answer-query row × cot-key col)
  keep |= diagonal  # only on PAD rows, so real answer rows keep ZERO attention into CoT
  ```
  (Also returns `attention_mask_1d` for debugging.) The model converts bool→additive.

  *Why the diagonal is forced only on pad rows:* every real answer query still sees key 0
  (prefix start), so it is never fully masked; forcing the global diagonal would re-open
  `</cot>`→`</cot>` self-attention **inside** the blocked span and leak the delimiter.

### 2. `patched_modeling_stage2.py`  (was `src/modeling/modeling_stage2.py`)

* **`__init__`**
  * `attn_implementation = 'flash_attention_2' if use_flash_attention_2 else 'sdpa'`
    (SDPA is **required** for the custom 4-D mask — run with `--use_flash_attention_2 False`).
  * registers `<cot>`/`</cot>`, `resize_token_embeddings`, initialises the two new rows
    from `<think>`/`</think>` (input **and** output embeddings).
  * LoRA `LoraConfig` gains `modules_to_save=["embed_tokens","lm_head"]` so the new-token
    rows are actually trained (LoRA otherwise never touches embeddings).
  * saves `base_model/` **with the tokenizer** so inference can reload the enlarged vocab.
  * new weights `cot_w`, `ans_w`; `dist.get_rank()` replaced by `_rank0()` (safe without a
    process group → plain single-GPU Trainer works).
* **`forward`**
  * signature now takes `labels_cot`, `labels_answer` (legacy `labels` ignored).
  * converts a 4-D bool keep-mask → additive float mask (`finfo(dtype).min`) in compute
    dtype and passes it to `self.latent_model(inputs_embeds=…, attention_mask=<4D>)`.
  * `loss = cot_w*loss_cot + ans_w*loss_ans + kl_w*loss_kl`, where `loss_cot`/`loss_ans`
    are `_shifted_ce` over the two label tensors. **`loss_kl` (latent KL) is unchanged.**
  * `loss.jsonl` now logs `loss_cot`, `loss_ans`, `loss_kl`.
* **`dualtrack_generate_hf` (new staticmethod)** — three-phase KV-cache decode
  `latent → <cot> visible-CoT </cot> → answer`. From the `</cot>` query onward every step
  gets a **2-D key mask that zeroes the `<cot>..</cot>` positions**, so at *generation* time
  the answer cannot attend to the visible-CoT KV. Supports `override_latent_embeds`
  (donor latent) and `override_cot_ids` (force the visible CoT) for the verify tests, and
  `bottleneck=False` to demonstrate the leak. The original `one_example_generate_hf` is kept.

### 3. `patched_arguments.py`  (was `src/stage2/arguments.py`)

* `ModelArguments` gains **`cot_w`** and **`ans_w`** (defaults 1.0);
  `use_flash_attention_2` default flipped to **False**.

### 3b. one required edit to `script/run_distill_stage2.py`

The entry script must forward the two new weights when constructing the model:

```python
model = LatentSFTStage2SoftEmbedding(
    latent_model_path=model_args.latent_model_path,
    ce_w=model_args.ce_w,
    kl_w=model_args.kl_w,
    cot_w=model_args.cot_w,     # <-- add
    ans_w=model_args.ans_w,     # <-- add
    ...
)
```

Nothing else in `run_distill_stage2.py` / `Stage2Trainer.compute_loss` needs changing:
`compute_loss` already does `model(**inputs)` and reads `outputs.loss`, and the collator's
extra keys flow through because training runs with `--no_remove_unused_columns`.

---

## How to apply (in the cloned latent-sft-repo)

```bash
REPO=/content/latent-sft-repo            # or E:/ckpts/latent-sft-repo
cp patched_data.py            $REPO/src/stage2/data.py
cp patched_modeling_stage2.py $REPO/src/modeling/modeling_stage2.py
cp patched_arguments.py       $REPO/src/stage2/arguments.py
# then apply the 2-line edit to $REPO/script/run_distill_stage2.py (see 3b)
```

`bootstrap_soft_labels.py`, `build_dualtrack_clean.py`, `verify_bottleneck.py` import
`patched_data` / `patched_modeling_stage2` by module name, so run them **from this folder**
(or `PYTHONPATH=$REPO/src` after copying — both import styles resolve the same functions).

---

## End-to-end (Colab A100)

```bash
# 0) build clean dual-track data (GSM8K gold CoT -> question/cot/answer)
python build_dualtrack_clean.py \
    --src /content/gsm_train_7500.json --out /content/dualtrack_clean.jsonl

# 1) teacher latent soft-labels from the released latent-sft-1b checkpoint itself
python bootstrap_soft_labels.py \
    --ckpt /content/latent-sft-1b \
    --data /content/dualtrack_clean.jsonl \
    --save_path /content/dualtrack_soft \
    --compression_rate 16 --topk_interpolation 10

# 2) TRAIN — plain Trainer (NO DeepSpeed), LoRA + grad-ckpt + SDPA
python latent-sft-repo/script/run_distill_stage2.py \
    --latent_model_path /content/latent-sft-1b \
    --kl_w 1.0 --cot_w 1.0 --ans_w 1.0 \
    --bfloat16 True --use_flash_attention_2 False \
    --topk_interpolation 10 \
    --train_data_path /content/dualtrack_clean.jsonl \
    --train_latent_soft_label_path /content/dualtrack_soft \
    --add_gumbel_noise True \
    --lora_tune True --lora_rank 64 --lora_dropout 0.1 --training True \
    --learning_rate 3e-4 --warmup_ratio 0.05 --weight_decay 0.01 \
    --lr_scheduler_type cosine --num_train_epochs 30 \
    --bf16 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --dataloader_drop_last False --logging_steps 1 \
    --save_strategy epoch --save_total_limit 3 \
    --no_remove_unused_columns \
    --report_to none --overwrite_output_dir \
    --output_dir /content/dualtrack_out
```

Key differences vs. the stock `run_distill_stage2_gsm8k.sh`:
**no `--deepspeed`**, launched as a plain `python …` (no `torchrun`), `--use_flash_attention_2 False`
(→ SDPA), `--gradient_checkpointing True`, and the new `--cot_w/--ans_w`. Single A100 is fine
for the 1B model at `per_device_train_batch_size 4` + grad-accum 8.

```bash
# 3) VERIFY the 3 acceptance checks (auto-loads /content/dualtrack_out/hf)
python verify_bottleneck.py --out /content/dualtrack_out --model_family llama
```

`verify_bottleneck.py` prints:
1. **format stable** — `<bot>…<eot><cot>…</cot>\boxed{…}` with latent closed by `<eot>`,
   CoT closed by `</cot>`, answer containing `\boxed{}`;
2. **latent-causal** — swapping in a donor latent changes the answer / the answer follows
   the donor;
3. **mask works** — with the **same latent + same question** but a **scrambled visible CoT**,
   the answer is **unchanged with the bottleneck ON** and typically **changes with it OFF**;
   plus a logit-lens readout of the latent tokens.

---

## Notes / caveats

* `modules_to_save=["embed_tokens","lm_head"]` unties the (tied) 1B embeddings into two
  trainable copies — standard when adding tokens to a tied model under LoRA. Both copies are
  seeded identically from `<think>`/`</think>`.
* The bootstrap teacher is a *proxy*: it uses the decoder's own final hidden states over the
  gold CoT (no separate Stage-1 encoder). Good enough to seed dual-track training; swap in
  the real `generate_latent_soft_label_hf_batch.py` encoder pipeline if you have that ckpt.
* Latent-slot count per example `= ceil(len(cot_tokens)/compression_rate)`; `patched_data`
  reads it back via `len(latent_state[0])`, so bootstrap and data stay consistent.
```
