# Clean dual-track Latent-SFT — a thin patch layer over a vendored upstream

```
latent_sft/
  upstream/          the vendored clone.  READ ONLY.  NEVER EDITED.
  dualtrack/         the thin layer.  This is the deliverable.
  README.md          this file
  run_dualtrack.sh   end-to-end run; steps 1-2 invoke upstream scripts unedited
  data/  out/        generated artefacts (gitignored)
```

**Hard invariant.** `git -C upstream status --porcelain` is EMPTY, before and
after every step of a run. The layer imports upstream and subclasses it; it never
edits it, and no `.patch` file was needed. Verified:

```
$ git -C upstream status --porcelain
$ git -C upstream rev-parse HEAD
08adc3a7f5067a9061208dac596cd31887542e06   # "Fix boolean CLI flags and update Latent-SFT docs"
```

**This round is CLEAN ONLY.** See [What is NOT implemented](#8-what-is-not-implemented).

---

## 1. What "dual-track" means here

The model produces two parallel tracks and the answer is wired to exactly one:

- **latent track** — soft embeddings distilled from a teacher, supervised by the
  Latent-SFT top-k KL. This carries the reasoning the answer is computed from.
- **visible CoT track** — ordinary text tokens between `<cot>` and `</cot>`.
  This is what a CoT monitor would read.
- **bottleneck** — answer query positions cannot attend to *any* key in the
  visible-CoT span. Structurally enforced by a 4-D attention mask, not a learned
  tendency.

In this clean round the visible CoT is the same reasoning the latents were
distilled from, and the answer is the gold answer. The only question this round
answers is: *does the model still learn, and is the bottleneck real?*

### Sequence layout

```
prefix (question) | <bot> | latent x L | <eot> | <cot> | visible CoT | </cot> | \boxed{answer} eos
```

`<bot>`/`<eot>` are upstream's existing `<think>`/`</think>`; the span arithmetic
is written in terms of `len(bot_ids)`/`len(eot_ids)` and never assumes 1.
`<cot>`/`</cot>` are newly added *special* tokens and are asserted to be exactly
one piece each — `spans.build_spans` refuses anything else, because a
multi-piece delimiter would leave real CoT keys attendable by the answer.

### The mask

```
keep[b,i,j] = ((j <= i) and (j < len_b) and not blocked(b,i,j)) or (i == j and i >= len_b)
blocked(b,i,j) = query_start_b <= i < query_end_b  and  key_start_b <= j < key_end_b

key_span   = (cot_open_pos, answer_start)      # the whole visible-CoT span incl. both delimiters
query_span = (answer_start - 1, S)             # starts at the LAST </cot> piece
```

`answer_start - 1` is the row that emits the first answer token under shifted CE,
so it must already be blocked. The forced diagonal is gated on **padding rows
only** — a global `keep |= eye` would restore `(q0, q0)` and re-open `</cot>`
self-attention inside the blocked span.

### The loss

```
loss = cot_w * CE_cot  +  ans_w * CE_ans  +  latent_w * L_latent
```

`CE_cot` and `CE_ans` partition the supervised positions exactly:
`CE_cot` covers `[eot_pos, answer_start)`, `CE_ans` covers `[answer_start, S)`,
they are disjoint, their union is `[eot_pos, S)`, and every supervised label
equals the input id at that position. `spans.assert_label_partition` states this,
and `model_dualtrack` additionally proves the token-weighted identity
`CE_union == (n_cot*CE_cot + n_ans*CE_ans) / n_union`.

---

## 2. How `L_latent` comes from upstream (the idea that keeps this thin)

Upstream's `LatentSFTStage2SoftEmbedding.forward`
(`upstream/src/modeling/modeling_stage2.py:204-279`) does the latent embedding
splice, the model forward, one CE and the latent top-k KL in one body, returning
`loss = ce_w*CE + kl_w*KL`. Instead of re-deriving any of that, our override
**calls it** with `labels = labels_answer`, having bound `ce_w := ans_w` and
`kl_w := latent_w` once in `__init__`:

```
                collator (ours)                    forward (ours, ~30 lines)
input_ids ─────────────────────────────┐
labels_cot ───────────────────┐        │   ┌────────────────────────────────────────────┐
labels_answer ───────┐        │        └──►│ super().forward(input_ids, additive_mask,  │
4-D keep-mask ──┐    │        │            │      latent_state, latent_index,           │
                └──► to_additive ──────────►│      labels = labels_answer)               │
                                            │   ── upstream modeling_stage2.py:213-279 ──│
                                            │   splice  (:222-231)   ← NOT copied        │
                                            │   forward (:233-235)                       │
                                            │   CE      (:240-249) with ce_w := ans_w    │
                                            │   KL loop (:251-263) → kd_kl_loss_topk     │
                                            │   loss = ans_w*CE_ans + latent_w*L_latent  │
                                            └───────────────┬────────────────────────────┘
                                                            │ base.loss, base.logits
                                            loss = base.loss + cot_w * CE_cot(base.logits, labels_cot)
```

Because `labels_answer` is what upstream's CE consumes, no compute is wasted:
upstream's CE *is* `CE_ans`. The only added work is one shifted CE over
`labels_cot`. The latent splice and the KL loop are never copied and never
re-expressed — they run inside upstream.

**Rejected alternative** (recorded so nobody re-invents it): setting `ce_w=0` and
back-solving `L_latent = loss/kl_w` is algebraically equivalent but wastes a CE
and makes upstream's `loss.jsonl` meaningless.

**Honest note about `loss.jsonl`.** Upstream's forward appends to
`<output_dir>/loss.jsonl` (`:266-274`). Under this wiring those numbers are all
real and correctly named — `loss_ce == CE_ans`, `loss_kl == L_latent` — but its
`loss` is **not** the training loss: it is missing `cot_w * CE_cot`. The complete
decomposition (`loss`, `loss_cot`, `loss_upstream_part`, the three weights,
`n_cot_tokens`, `n_ans_tokens`) goes to `dualtrack_loss.jsonl` in the same
directory. Both are per-step and rank-0.

---

## 3. Override table — what the layer touches and why

**IMPORT** = used unchanged · **SUBCLASS** = inherit, override the named method
only · **COPY** = re-typed (justified in §4) · **NEW** = no upstream counterpart.

### `upstream/src/modeling/modeling_stage2.py`

| Upstream symbol | line | Action | Why necessary |
|---|---|---|---|
| `LatentOutput` | 17-20 | **IMPORT** | our forward returns it |
| `LatentSFTStage2SoftEmbedding` | 129 | **SUBCLASS** → `DualTrackLatentSFT` | overrides `__init__`, `forward`, `save` and nothing else |
| `.forward` | 204-279 | **SUBCLASS**, override *calls* `super()` | needs two label rows and a 4-D mask; the latent KL block is CALLED, not copied |
| `.save` | 281-285 | **SUBCLASS**, override *calls* `super()` | 4 lines: refuse to write a tied config (see landmine 4) |
| `.gradient_checkpointing_enable` | 199-202 | **IMPORT** (inherited) | — |
| `kd_kl_loss_topk` | 49-73 | **IMPORT** | called inside upstream's forward; imported only so our selftest checks it against an independent re-derivation |
| `weighted_embedding_from_topk` | 76-88 | **IMPORT** | used by upstream's splice; not called by us |
| `softmax_over_embedding_topk` | 90-126 | **IMPORT** | used by the proxy-teacher fallback |
| `kd_kl_loss` (dense) | 22-46 | not used | top-k path only |
| `one_example_generate_hf` / `_lora` | 287-490 | **NOT USED** | single-track decode: no `<cot>` phase, no bottleneck. Replaced by `dualtrack/generate_dualtrack.py` (**NEW**) |
| LoRA config block in `__init__` | 177-193 | **COPY** (§4.1) | must run *after* resize + untie, and must add `lm_head` |

### `upstream/src/stage2/data.py`

| Upstream symbol | line | Action | Why necessary |
|---|---|---|---|
| `read_jsonl` | 13-38 | **IMPORT** (via inherited `__init__`) | — |
| `Stage2Dataset` | 58 | **SUBCLASS** → `DualTrackStage2Dataset` | overrides `__init__` (calls super, then adds guards) and `__getitem__` |
| `._load_all_chunks` | 95-117 | **IMPORT** (inherited) | the contiguity check sits *beside* it as a filename-only function in `alignment.py` |
| `.apply_gumbel_noise` | 119-130 | **IMPORT** (inherited) | — |
| `.apply_gumbel_noise_safe` | 132-153 | **IMPORT** (inherited, called from our `__getitem__`) | — |
| `.__len__` | 155-156 | **IMPORT** (inherited) | — |
| `pretrain_tokenize_function` | 173-260 | **COPY/SHADOW** (§4.2) | module-level function; the sequence assembly *is* the deliverable |
| `_validate_example` | 41-55 | **COPY** (§4.3) | different schema: the CoT must be a separate field |
| `DataCollatorForDynamicPadding` | 263-293 | **SUBCLASS** → `DualTrackCollator` | overrides `__call__` only |
| `.dynamic_padding` | 286-293 | **IMPORT** (inherited, called 3×) | — |

### `upstream/src/stage2/arguments.py`

| Upstream symbol | line | Action | Why necessary |
|---|---|---|---|
| `ModelArguments` | 6-29 | **SUBCLASS** | add `cot_w`/`ans_w`/`latent_w`/`model_family`; flip `use_flash_attention_2` default `True → False` |
| `DataArguments` | 32-49 | **SUBCLASS** | add `allow_missing_alignment`, `max_seq_len` |
| `Stage2TrainingArguments` | 52-68 | **SUBCLASS** | flip `remove_unused_columns` default `True → False` |

Upstream's `ce_w` is **inherited and kept**: it is the same knob as `ans_w` under
upstream's name. `arguments.resolve_answer_weight` rejects a conflicting pair
rather than picking silently; `resolve_latent_weight` does the same for
`kl_w`/`latent_w` (the shared cross-platform spelling used by `colar/` and
`lt_tuning/`).

### `upstream/src/stage2/trainer.py`

| Upstream symbol | line | Action |
|---|---|---|
| `Stage2Trainer` | 13 | **IMPORT unchanged — no subclass at all** |
| `._save` | 14-44 | **IMPORT**: works verbatim because our `__init__` leaves `lora_tune`/`save_path` in exactly the shape it expects and writes `base_model/` with the resized, untied vocabulary |
| `.compute_loss` | 46-56 | **IMPORT**: `outputs = model(**inputs); return outputs.loss` — our forward already returns the total |

### `upstream/script/run_distill_stage2.py`

**NOT IMPORTED, NOT EDITED.** We ship `dualtrack/train_dualtrack.py::main`, ~45
lines of glue mirroring the same order (parse → dir check → logging → `set_seed`
→ model → dataset → `Stage2Trainer` → `train` → `save_model`). Every symbol it
constructs is ours or an upstream import; no algorithm is reproduced, and its
selftest fails if a loss, a mask or a `compute_loss` ever appears in it.
**Zero upstream edits.**

### Stage-1 / teacher path

| Upstream symbol | Action |
|---|---|
| `script/run_distill_stage1_encoder.py` | **INVOKED UNCHANGED** by `run_dualtrack.sh` step 1 |
| `src/modeling/modeling_stage1.py`, `src/stage1/*` | **UNTOUCHED** — consume the upstream-view jsonl our `prepare_data.py` emits |
| `generate_latent_soft_label_hf_batch.py` | **INVOKED UNCHANGED** (subprocess) by `make_latents.py --teacher stage1_encoder` |
| `…::build_latent_token_induction_mask` (98-143) | **IMPORT**, for the CPU selftest that asserts the induction-mask geometry |
| `…::insert_special_token_every_k` (30-41) | **IMPORT**, so our proxy-route slot-count mirror is checked against upstream's own function |
| `src/stage1/data.py::compute_dual_track_position_ids` (267-312) | **DELIBERATELY NOT USED** — landmine 8 |

**Score: 3 COPY sites against ~25 IMPORT/SUBCLASS rows.**

---

## 4. The three COPY sites, justified

Machine-checkable: `rg -n "COPY" dualtrack/` returns only these.

### 4.1 `model_dualtrack.py::_apply_lora`
Shadows `modeling_stage2.py:177-193`. Subclassing cannot reorder statements
inside upstream's `__init__`, and the LoRA wrap must happen **after**
`resize_token_embeddings` + untie, and must include `lm_head` in
`target_modules` so the two new vocabulary rows are trainable (upstream's list
does not). **Anti-drift:** `upstream_lora_target_modules()` parses upstream's own
literal list out of `inspect.getsource(...)` at runtime, and the selftest fails
unless ours is a strict superset. The copy cannot silently diverge.

### 4.2 `tokenize_dualtrack.py` (whole module, incl. `build_prefix_and_eos`)
Shadows the module-level `pretrain_tokenize_function` (`data.py:173-260`), which
is pre-authorized by the rebuild brief: the sequence assembly is the core change
and the function is not a method, so there is nothing to subclass. The chat
template branches (`data.py:182-217`) are copied **byte-identically** — the
stage-1 teacher and the stage-2 student must see the same prefix, and a selftest
asserts the llama branch string exactly.

### 4.3 `tokenize_dualtrack.py::validate_example`
Shadows `_validate_example` (`data.py:41-55`), 12 lines. Upstream validates
`("problem", "cot_answer")`; the dual-track schema is `("question", "cot",
"answer")` and needs the CoT as a **separate** field — upstream's `cot_answer`
fuses reasoning and answer, which is exactly what the visible track must split.

Two smaller things that are *overrides calling `super()`*, not copies:
`DualTrackLatentSFT.save` (adds the tied-embedding refusal) and
`DualTrackCollator.__call__` (calls the inherited `dynamic_padding` three times).

---

## 5. Files

| File | Tier | Lines | What it is |
|---|---|---|---|
| `spans.py` | 0 | 379 | span arithmetic, the mask as a pure predicate, `reference_keep_matrix`, `assert_label_partition` |
| `common.py` | 0 | 239 | jsonl io, answer comparator, relocatable env paths, upstream-view rendering |
| `alignment.py` | 0 | 260 | sha256 guard, chunk-cover contiguity, upstream-view derivability |
| `stub_tokenizer.py` | 0 | 120 | whitespace tokenizer + stub model for CPU selftests |
| `tokenize_dualtrack.py` | 0 | 343 | the sequence assembly (COPY site §4.2/§4.3) |
| `prepare_data.py` | 0 | 319 | raw → shared jsonl, `--emit_upstream_view` |
| `mask.py` | 1 | 178 | the `[B,1,S,S]` builder + `to_additive` |
| `dist_bootstrap.py` | 1 | 94 | makes upstream's unconditional `dist.get_rank()` safe |
| `attention_probe.py` | 1 | 191 | `assert_sdpa`, and the *measured* 4-D-mask-semantics probe |
| `generate_dualtrack.py` | 1 | 504 | three-phase bottleneck-respecting decode (NEW) |
| `upstream_api.py` | 2 | 143 | the only import boundary; every upstream symbol with `file:line` |
| `model_dualtrack.py` | 2 | 450 | `DualTrackLatentSFT` (COPY site §4.1) |
| `data_dualtrack.py` | 2 | 258 | dataset + collator subclasses |
| `arguments.py` | 2 | 198 | the three argument subclasses |
| `train_dualtrack.py` | 2 | 192 | entry point (glue) |
| `make_latents.py` | 2 | 417 | thin wrapper around upstream's generator; proxy fallback |
| `verify_dualtrack.py` | 2 | 610 | V1/V2/V3 harness + the canonical cross-platform row |
| `selftest_all.py` | 0 | 99 | runs every selftest, reports env skips verbatim |
| `__init__.py` | 0 | 21 | version floor constants |

**5 015 lines total**, replacing 3 966 lines of re-implementation *and* restoring
reproducibility of the paper baseline (which the previous round had destroyed).
Roughly 1 300 of those lines are the genuinely new mask / verify / generate
machinery that upstream has no counterpart for; the rest is guards and selftests.

---

## 6. Environment, and what has actually been verified

**Version floor.** `python>=3.10`, `torch>=2.5.1`, `transformers>=4.51.1`,
`peft==0.15.2` (upstream `requirements.txt`, and `upstream/README.md:405` which
pins `torch==2.6.0 transformers==4.51.1`). flash-attn is listed upstream but must
**not** be used here (landmine 5).

The **python>=3.10** floor is upstream's, not ours, and it is easy to miss:
`upstream/src/modeling/modeling_stage2.py:25` writes `mask: torch.Tensor | None`
in a signature with no `from __future__ import annotations`, so the annotation is
evaluated at import time and 3.9 raises
`TypeError: unsupported operand type(s) for |`.

**Measured on this machine (2026-08-07, corrected in review).** An earlier
version of this section claimed there was no torch on any interpreter here. That
was wrong — only three interpreters had been checked. The full survey:

```
/opt/homebrew/bin/python3    3.14.6  → torch: no   transformers: no
/opt/homebrew/bin/python3.13 3.13.14 → torch: no   transformers: no
/opt/homebrew/bin/python3.11 3.11.15 → torch: no   transformers: no
/usr/local/bin/python3       3.11.2  → torch: no   transformers: no
/usr/bin/python3             3.9.6   → torch: no   transformers: no
/opt/anaconda3/bin/python3   3.9.21  → torch 2.1.1  transformers 4.35.2
                                        accelerate 0.24.1, peft: NOT INSTALLED
```

So tiers 0 **and 1** are genuinely verified here, on `/opt/anaconda3/bin/python3`:

```
$ /opt/anaconda3/bin/python3 -m dualtrack.selftest_all --tiers 0,1
PASS  [0] dualtrack.spans
PASS  [0] dualtrack.common
PASS  [0] dualtrack.alignment
PASS  [0] dualtrack.stub_tokenizer
PASS  [0] dualtrack.tokenize_dualtrack
PASS  [0] dualtrack.prepare_data
PASS  [1] dualtrack.mask
PASS  [1] dualtrack.dist_bootstrap
PASS  [1] dualtrack.attention_probe
PASS  [1] dualtrack.generate_dualtrack
10 passed, 0 skipped (environment), 0 failed
```

`dualtrack.verify_dualtrack` also runs there and its decisive check passes:
V3a gradient isolation on a toy attention stack gives `grad ON=0.0
OFF=0.0755046`, and its real-`LlamaForCausalLM` probe reports honestly that
transformers 4.35.2 rejects the 4-D mask (`ValueError: too many values to unpack
(expected 2)`).

Tier 2 remains genuinely unverified here, for two stacked reasons — the real
smoke test, verbatim:

```
$ /opt/anaconda3/bin/python3 -c "import dualtrack.upstream_api"
  File ".../dualtrack/upstream_api.py", line 62, in <module>
    from src.modeling.modeling_stage2 import (  # noqa: E402
  File ".../latent_sft/upstream/src/modeling/modeling_stage2.py", line 10, in <module>
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
ModuleNotFoundError: No module named 'peft'

# and with peft stubbed out, the next blocker is the python floor:
  File ".../upstream/src/modeling/modeling_stage2.py", line 25, in <module>
    mask: torch.Tensor | None = None,
TypeError: unsupported operand type(s) for |: 'torch._C._TensorMeta' and 'NoneType'
```

Both tracebacks reach *inside* `upstream/src/modeling/modeling_stage2.py`, so the
`sys.path` wiring in `upstream_api` resolves the vendored clone correctly.
Everything past that import is unverified here.

`python3 -m py_compile dualtrack/*.py` succeeds for all 19 modules — **syntax
only, not an execution pass.**

**Nothing in tier 2 may be reported as passing until it is observed passing on a
box with peft and python>=3.10.** First command there:

```
python -m dualtrack.selftest_all          # expect 17 PASS, 0 SKIP, 0 FAIL
```

What tiers 0 and 1 cover today, genuinely: span arithmetic and its five geometry
rejections; the full bottleneck geometry via the nested-loop reference
(causality, no padding keys, no empty query rows, pad-rows-only diagonal,
`</cot>` self-attention stays blocked); the label partition (disjoint, covering
`[eot_pos, S)`, teacher-forced identity, contiguous latent block, plus three
mutation tests that must each raise); the sha256 alignment guard including the
equal-length-reshuffle case; chunk-cover contiguity; the delimiter-contamination
guard; the model-family conflict; the byte-identical llama prefix; and both
dataset converters with the upstream-view derivability check.

---

## 7. Running it

```bash
cd latent_sft
python3 -m dualtrack.selftest_all                      # always start here

# 0. data (stdlib only; runs anywhere)
python -m dualtrack.prepare_data --source_format latentsft \
    --src data/GSM8k-Aug-train.jsonl \
    --out data/dualtrack_clean.jsonl \
    --emit_upstream_view data/dualtrack_upstream_view.jsonl

# 1-2. teacher.  Steps 1 and 2 run UPSTREAM SCRIPTS UNEDITED.
bash run_dualtrack.sh          # or drive the pieces yourself:
python -m dualtrack.make_latents --teacher stage1_encoder \
    --encoder_path out/stage1_encoder/hf --decoder_path <base-model> \
    --data data/dualtrack_clean.jsonl \
    --upstream_view data/dualtrack_upstream_view.jsonl --save_path data/soft

# 3. alignment
python -m dualtrack.alignment --verify_soft data/soft --data data/dualtrack_clean.jsonl

# 4. train
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    dualtrack/train_dualtrack.py --latent_model_path <stage1-ckpt> \
    --cot_w 1.0 --ans_w 1.0 --latent_w 1.0 --use_flash_attention_2 False \
    --train_data_path data/dualtrack_clean.jsonl \
    --train_latent_soft_label_path data/soft \
    --lora_tune True --lora_rank 64 --bf16 --output_dir out/clean_dualtrack

# 5. verify
python -m dualtrack.verify_dualtrack --out out/clean_dualtrack \
    --eval_data data/dualtrack_clean.jsonl --n_samples 20 --teacher_forced_check
```

Every default resolves relative to this folder and is env-overridable
(`DUALTRACK_DATA`, `DUALTRACK_SOFT`, `DUALTRACK_OUT`, `DUALTRACK_UPSTREAM_VIEW`,
`DUALTRACK_RAW`). No absolute path is hardcoded anywhere.

### The latent teacher: real path vs documented fallback

`--teacher stage1_encoder` (**default**) is the paper's route and contains **no
model code at all**. It validates that the upstream-view jsonl is row-for-row
derivable from the shared jsonl, builds the argv, runs
`upstream/generate_latent_soft_label_hf_batch.py` as a subprocess, reads the
chunk lengths back and writes `alignment.json`.

*Why a subprocess and not an import:* the chunk-writing driver lives inside
`if __name__ == '__main__':` (`:427-486`) and cannot be imported, and
`MultiprocessTransformerWrapper` hard-codes `torch.device(f'cuda:{rank}')`
(`:267`) and `mp.get_context("spawn")`. Importing it would force us to re-type
the driver — exactly what this rebuild exists to avoid.

`--teacher proxy_decoder` survives as a documented **cheap fallback**: the
released decoder's own last-layer hidden states at every-k positions, projected
onto the embedding table. **This is not what the paper does.** It answers "does
dual-track train at all", not "does Latent-SFT reproduce". `alignment.json`
records which route produced the tensors, so no result can be misattributed.

### What V1 / V2 / V3 mean

- **V1 format** — generation yields a well-formed `[latents][<cot>…</cot>][answer]`.
  Rates are over every *attempted* prompt; a degenerate decode counts as a
  failure, never as a dropped sample.
- **V2 latent-causal** — V2a: swapping in a donor question's latents changes the
  answer (`change_rate`), and how often it lands on the donor's own answer
  (`follow_donor_rate`). V2c: zero-ablating the latents changes the answer — if
  the answer survives ablation the latents are decorative, whatever V2a says.
- **V3 bottleneck**, always reported ON *and* OFF:
  - **V3a gradient isolation (decisive, instrument-free)** —
    `max |d(answer logits)/d(cot embeddings)|` must be **exactly 0** with the
    mask ON and `> 0` with it OFF. This is a property of the mask, not of
    training, and it is proven on a toy SDPA stack that needs no weights (this
    part runs on CPU as part of the selftest).
  - **V3b** teacher-forced logit divergence ON vs OFF at answer positions.
  - **V3c** generation answer-change rate under three CoT perturbations, ON/OFF.

A trained model may also *learn* to ignore the visible CoT, in which case the OFF
side barely moves and V3b/V3c prove little. That is a measurement limit, not a
failure; V3a always decides. The `canonical` block in the report carries the six
keys `colar/` and `lt_tuning/` emit under the same names and polarity, using the
length-preserving `reversed` perturbation (the only one whose ON side is not
confounded by shifted RoPE positions).

---

## 8. What is NOT implemented

**No attack, of any kind, this round.** There is no target answer, no poison
flag, no label flipping, no depth gating, no ASR, and no divergent-latent path.
The latent teacher is **always** `row["cot"]`; the previous code's
`ex.get("latent_cot") or ex["cot"]` fallback was the attack injection point and
it is deleted. `make_latents.assert_clean_rows` makes `latent_cot`, `is_poison`
and `target_answer` in an input row a hard error, and a selftest greps
`inspect.getsource` of both teacher routes to prove the fallback has not come
back. `arguments` asserts no attack knob exists in any dataclass.

Also not implemented: upstream's dense `kd_kl_loss` path (top-k only), upstream's
`one_example_generate_hf` decoder (no `<cot>` phase — replaced), and
`full_vocab` teacher tensors (`read_latent_lens` rejects them with an explanation
rather than crashing later).

---

## 9. Platform caveats — landmines in the vendored upstream

1. **`dist.get_rank()` is unconditional** (`modeling_stage2.py:266`, every
   forward step, evaluated *before* the `save_path is not None` check). A
   single-process run dies with "Default process group has not been
   initialized". `dist_bootstrap.ensure_process_group()` creates a world-size-1
   gloo group, and defers when a real launcher declares `WORLD_SIZE > 1`.
2. **Embedding access assumes PEFT wrapping** — `:219-220` uses
   `self.latent_model.model.model.embed_tokens`, which resolves only through a
   `PeftModel`. `lora_tune=True` is therefore **mandatory** and is rejected at
   construction with that explanation. Do **not** use
   `modules_to_save=["embed_tokens"]`: it wraps the embedding in a
   `ModulesToSaveWrapper`. `target_modules += ["lm_head"]` gives the same
   trainability with no wrapper.
3. **`base_model/` is dumped inside `__init__`, before any vocab change**
   (`:167-168`). If allowed to fire, `base_model/` carries the pre-resize vocab
   and `Stage2Trainer._save`'s merge dies on a size mismatch hours into the run.
   We pass `save_path=None` into `super().__init__`, dump `base_model/` ourselves
   after resize/untie/row-seed, then restore `self.save_path`.
4. **Tied embeddings.** Llama-3.2-1B and small Qwen2.5 set
   `tie_word_embeddings=True`. Then (a) `save_pretrained` omits
   `lm_head.weight` and the reload re-ties it to `embed_tokens`, whose new rows
   never move (upstream detaches the token embeddings at `:219`) — the model
   silently reloads unable to emit `</cot>`; and (b) merging a LoRA on `lm_head`
   writes into storage shared with `embed_tokens`. `untie_output_embeddings` runs
   before LoRA and before writing `base_model/`, and `save()` refuses a tied
   config.
5. **Attention implementation.** `:149` passes
   `attn_implementation='flash_attention_2' if use_flash_attention_2 else None`
   — never `'sdpa'` explicitly — and upstream's launcher
   `script/run_distill_stage2_gsm8k.sh` sets `--use_flash_attention_2 True`. FA2
   cannot consume a 4-D mask and would silently drop the bottleneck. Our
   `ModelArguments` subclass defaults it `False`, the model raises if `True`, and
   `assert_sdpa` requires `config._attn_implementation == "sdpa"` afterwards.
   Eager would be numerically fine but is rejected so training and generation
   share one path.
6. **Upstream's 2-D mask is `input_ids != pad_token_id`** (`data.py:275`). For
   Qwen/DeepSeek upstream sets `pad_token = eos_token` (`:161`), so a genuine
   trailing `eos` key gets masked out. We never build a mask from token values —
   `build_bottleneck_mask` takes true lengths. Never mix the two.
7. **`-100` is overloaded**: CE ignore index *and* the latent-slot sentinel
   inside `input_ids` (`data.py:243`, consumed at `:215-217`). Consequences: the
   collator pads `input_ids` with `pad_token_id` and the label rows with `-100`
   (padding `input_ids` with `-100` would make every pad slot look like a latent
   slot); anything reading `input_ids` as token ids must `masked_fill` first.
8. **`position_ids`.** Stage 2 never passes them, so HF derives
   `cache_position = arange(...)`. Adding the visible-CoT track shifts the
   answer's absolute positions — fine and *consistent*, because generation also
   calls the model directly and gets `arange` too. Do **not** import
   `stage1/data.py::compute_dual_track_position_ids`: it is for stage 1's
   interleaved layout, and using it in training while generation uses `arange`
   would desynchronize RoPE between the two.
9. **HF Trainer plumbing.** `Trainer._prepare_input` recurses into lists with
   `type(data)(generator)`, so `latent_state` must stay
   `list[tuple[Tensor, Tensor]]` — a NamedTuple or dataclass there raises.
   `label_names` defaults to `["labels"]`, which our batch does not contain;
   harmless, and it means `num_items_in_batch` arrives as `None`, which
   upstream's `compute_loss` already accepts. Our extra `lengths` tensor is
   absorbed by `**kwargs` in forward.
10. **`tokenizer=` is deprecated** on `Trainer.__init__` (renamed
    `processing_class` in 4.46); upstream's script still passes it
    (`run_distill_stage2.py:106`). Our entry point feature-detects the signature.
11. **Cache objects.** From 4.51 `past_key_values` is a `DynamicCache`.
    `_IncrementalDecoder` only round-trips the object and never indexes, slices
    or `len()`s it. `from_pretrained(..., use_cache=False)` (`:151`) sets the
    config flag, so generation passes `use_cache=True` per call.
12. **4-D mask contract.** Transformers forwards a 4-D `attention_mask` untouched
    but requires the *inverted additive* form (`max == 0`), and has shipped
    releases that inverted it as if it were binary — an additive mask then
    collapses to a constant and training runs with a decorative bottleneck.
    `to_additive` guarantees the form, and
    `assert_four_d_mask_is_honoured` **measures** the semantics at construction
    (block key 0 for query row 1; the blocked delta must be ~0 while the unmasked
    control moves). Version detection alone is not accepted as proof.
13. **`Stage2Trainer._save`'s non-LoRA branch is broken upstream** (`:44` calls a
    `save_pretrained` this `nn.Module` does not have). With landmine 2 that makes
    `lora_tune=True` a construction-time assertion rather than an hours-later
    discovery.
14. **Dataclass inheritance.** `ModelArguments.latent_model_path` has no default,
    so a subclass may add only defaulted fields; re-declaring an inherited field
    changes its default but keeps its original `__init__` position.
    `Stage2TrainingArguments` already subclasses `TrainingArguments`, so ours is
    a third level — `remove_unused_columns` is re-declared with the *same*
    `Optional[bool]` annotation or `HfArgumentParser` changes the flag form.
15. **No Lightning anywhere.** Upstream stage 2 is HF `Trainer` + DeepSpeed
    ZeRO-1 (`config_zero1.json`). There are no `configure_optimizers`,
    `training_step` or `on_*` hooks to satisfy.
16. **`sys.path` hygiene.** Upstream exposes a top-level package named `src` and
    vendors `sglang_latent_reasoning_pkg/`. `upstream_api` inserts only the
    upstream root, logs the resolved path once, and never imports `sglang`. If
    the environment has another top-level `src` package, this collides — a known
    constraint of vendoring this repo.
17. **`resize_token_embeddings` and vocab width.** It updates
    `config.vocab_size`, which upstream's inherited CE uses (`:245`); our
    `segment_ce` reads the width from the logits instead, so the two stay
    consistent even if PEFT reports a stale config. We resize to exactly
    `len(tokenizer)` — `pad_to_multiple_of=8` is faster but leaves untrained rows
    that greedy decoding can select, yielding ids the tokenizer cannot decode.
18. **`read_jsonl` asymmetry.** Upstream's stage-2 loader skips undecodable lines
    with a warning (`data.py:23-30`); the teacher generator's loader does not
    (`generate_latent_soft_label_hf_batch.py:420-425`). One bad line would shift
    the row pairing between the jsonl and the `.pt` chunks. Caught by
    `verify_alignment` (sha256 + `n_rows`) and by the per-row `latent_lens[idx]`
    check in `__getitem__`.

19. **Importing upstream writes `__pycache__` into it, and upstream *tracks* some
    of those directories.** Two separate traps. (a) A plain
    `import src.modeling.modeling_stage2` creates
    `upstream/src/modeling/__pycache__/`, which upstream's `.gitignore` does not
    cover — that alone makes the tree dirty. `upstream_api` therefore sets
    `sys.pycache_prefix` to `data/pycache/` before the first upstream import,
    `make_latents` runs the upstream generator with `python -B`, and
    `run_dualtrack.sh` exports `PYTHONDONTWRITEBYTECODE=1`. (b) `git status` is
    clean *because* `upstream/src/stage1/__pycache__/*.pyc` are **committed
    upstream** — so never "tidy up" by deleting `__pycache__` under `upstream/`;
    that deletes tracked files and dirties the tree. If it happens:
    `git -C upstream restore --source=HEAD --worktree src/stage1/__pycache__`.
    `upstream_api --selftest` asserts both: that the bytecode prefix is outside
    the clone, and that `git status --porcelain` is empty after the import.

### Contract deviation (deliberate, one)

A square `[B,1,S,S]` training mask has no counterpart in incremental decoding,
where the key axis is `[1, kv_len]` and there is no square query axis. The
**spans** are therefore built once, in `spans.build_spans`, and both consumers
derive their mask from them: `mask.build_bottleneck_mask` for training,
`spans.blocked_key_indices` for generation. The selftest asserts
`set(blocked_key_indices(spans)) == set(range(*cot_key_span(spans)))` and, on the
scripted CPU stub, that the `</cot>` step and every answer step zero *exactly*
that key set and no other — both endpoints checked, so a multi-piece delimiter or
an empty CoT raises instead of leaking.
