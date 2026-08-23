# Clean dual-track on LT-Tuning — a thin patch layer

```
lt_tuning/
  upstream/     the vendored LT-Tuning clone, pinned at c18aac695b33de135d3dd0848de0464d1b644ba7.  `dualtrack/__init__.py` carries the same sha and `_upstream.py --selftest` refuses a clone that has moved off it.  READ ONLY.  Never edited.
  dualtrack/    the thin layer.  This is the deliverable.
  data/         generated artefacts (gitignore it)
  README.md     this file
```

**The hard invariant:** `git -C upstream status --porcelain` is empty and stays empty. The
layer imports from upstream and subclasses it; it never edits it. There is no `*.patch` file
because every upstream defect worked around here is shadowable from a subclass — the four
that matter are listed under [Upstream landmines](#upstream-landmines-this-layer-works-around).

**No attack.** No target answer, no poison flag, no label flipping, no depth gating, no
success metric. `train.py --selftest` scans every `.py`/`.sh`/`.yaml` in the layer for
attack vocabulary and fails if any appears.

---

## Read this first: the bottleneck here is a chokepoint, not independence

On CoLaR and Latent-SFT the latents sit in one contiguous block *before* the visible CoT, so
a mask that blocks answer→CoT makes the answer's transitive receptive field genuinely
CoT-free. **On LT-Tuning that is false and cannot be made true.**

1. Latents are interleaved *through* the reasoning chain (`upstream/model.py:414-420`), not
   gathered.
2. Each latent's input embedding **is** the model's own hidden state at the immediately
   preceding chain token (`upstream/model.py:513, 518, 522`):
   * stage 1 (`hidden_state`): `e_latent = h_{t-1}`
   * stage 2 (`soft_fusion`): `e_latent = alpha * h_{t-1} + (1 - alpha) * e_pred`
3. Each latent *position* is an ordinary sequence position whose per-layer K/V are computed
   by attending over every preceding chain token.

So blocking the answer from *reading* the visible CoT does not block the information *path*:
the latent has already absorbed that content. What the mask buys is exactly:

> **Every path from a visible-CoT position to an answer query row passes through at least one
> `<thinking>` position.**

That claim — and only that claim — is asserted in the code (`mask.chokepoint_holds`). The
stronger claim "the answer is independent of the visible CoT" is false on this platform and
is not asserted anywhere. **V3 passing does not settle whether the visible CoT is causally
irrelevant here.** It settles that the answer cannot read the CoT surface directly.

---

## The two leak channels, and which one a mask can close

### Channel 1 — attention (closable, `--bottleneck_mode strict`)

`COT → LATENT → ANSWER`. Under the default mask the answer rows cannot read a CoT key, but a
**latent row is an ordinary query row and is not blocked**, so it reads the CoT, and the
answer reads the latent. The path needs two layers: one to move CoT content into the latent
column, one to move it from there into the answer row.

`mask.py --selftest` measures this on interleaved fixtures with a weights-free two-hop graph
probe plus a depth sweep (1 / 2 / 6 layers). **A one-layer check passes in BOTH modes** —
that is exactly how this stays hidden, and it is why the selftest varies depth.

The per-fixture claim is deliberately weights-free (`_selftest_two_hop_path_exists`): a
magnitude measured through random weights can vanish where the path is wide open (softmax
saturates), so the depth table asserts only in aggregate on the nonzero side, and exactly
per fixture on the zero side.

### Channel 2 — fusion initialisation (**not** closable by any mask)

A latent's input embedding is not a token embedding; it is built from the hidden state at the
preceding position. This is a **residual-stream edge, not an attention edge**. No entry in an
attention bias can reach it. `strict` does not close it, and closing it would mean changing
where the latent is initialised from — a substantive deviation from LT-Tuning, not a mask
setting.

> **Even under `strict`, a passing V3 does not establish that the answer is independent of the
> visible reasoning on this platform.** It establishes that the *attention* route is shut.

### Measured, on this machine, through upstream's own forward

`verify.py --selftest`, tiny locally-built Llama, random weights (magnitudes are meaningless;
the **polarity** is the result):

Every number is `|delta|` on the answer-row logits. **Two different statistics appear below
and they are labelled**: `max` is over samples, `mean` is the average over samples. The
decomposition (`fusion_channel_lower_bound`, `attention_channel_delta`) is defined on the
MEANS, so it must not be compared against the maxes on the line above it.

```
V3.1 latent-pinned  (fusion channel removed -> isolates ATTENTION)
     max :  native = 1.922e-04    strict = 0.0e+00 (bit-exact)   mask_off = 1.610e-02
V3.2 free-running   (both channels live)
     max :  native = 1.214e-01    strict = 1.215e-01     <- essentially identical
     mean:  native = 4.685e-02    strict = 4.690e-02
V3.3 decomposition  (defined on the MEANS above)
     fusion_channel_lower_bound = 4.690e-02   (= mean strict)
     attention_channel_delta    = -4.938e-05  (= mean native - mean strict)  <- NEGATIVE
```

`generate.py --selftest`, same contrast on a *decoded* sequence:

```
native = 1.759e-05    strict = 0.000e+00    mask_off = 7.132e-04
```

Read it this way: with the latent inputs pinned, STRICT is bit-exactly zero and NATIVE is
not — the attention channel is real and STRICT closes it. Free-running, the two modes are
indistinguishable — the fusion channel dominates and no mask touches it.

`attention_channel_delta` comes out **negative** here. That is not a bug and not noise to be
explained away: the two channels are not additive. The corrupted content is already inside
the latent's input embedding, so removing the latent's *read* of the CoT need not shrink the
answer's response to it at all. Reading the difference as "the attention share" would be a
claim this measurement does not support, which is why it is reported as a contrast and never
as a percentage split.

### `strict` is an instrument, not a better default

Under `strict` a latent can no longer attend the explicit reasoning it is *meant to
summarise*. The latent vectors themselves change (asserted in `model.py --selftest`:
`max|delta latent| = 4.791e-04`). That is a different model, not a better-masked version of
the same one. `native` stays the default everywhere. **`bottleneck_mode` is recorded in every
manifest and in every verification record** — last round it was silently dropped on the
training path, which would have made a whole strict run indistinguishable from a native one.
It is now a required attribute of the model itself, so no call site can drop it, and
`manifest.py --selftest` fails if it leaves the record.

---

## The one design decision everything else follows from

The previous round re-typed upstream's segment loop in order to slip a 4-D bias into each
`self.base_causallm(...)` call. That produced ~11.9k lines that had never run. It is not
necessary.

Upstream's per-segment call produces exactly **one** causal-mask construction inside
`LlamaModel.forward`. So the bias is delivered by replacing the mask builder with a **tape**
that hands out `full_bias[:, :, q_start:q_end, :q_end]` one segment at a time, and then
calling `super().forward()` unchanged. Segmentation, embedding assembly, latent replacement,
KV threading, logits concatenation and `Outputs` are **executed by upstream**, not merely
inherited.

Verified, not asserted (`model.py --selftest`):

```
mask_off reproduces upstream's own forward BIT-EXACTLY (max|delta logits| = 0.0,
                                              loss 3.730377 vs 3.730377)
tape: 4 segments [(0,4), (4,5), (5,8), (8,14)];
      every span observed inside transformers' own builder agrees with the tape
```

If that first line ever drifts, something was re-typed.

---

## Override table

`I` = import unchanged · `S` = subclass + override exactly one method · `C` = copy (justified)
· `X` = deliberately not used.

| # | Upstream symbol | file:line | | What the layer does |
|---|---|---|---|---|
| 1 | `LT_Tuning_Model` | model.py:22 | **S** | `DualTrackLTModel` — overrides `forward` + `_apply_transform`, refuses `generate` |
| 2 | `LT_Tuning_Model.forward` | model.py:381-627 | **S** | delegates to `super().forward()` under `SegmentBiasTape` |
| 3 | `LT_Tuning_Model._apply_transform` | model.py:301-304 | **S** | calls `super()`, then records / substitutes the latent |
| 4 | `_soft_fusion_embedding` | model.py:306-339 | **I** | inherited; called by upstream's own loop. Backup copied it — undone |
| 5 | `_select_hidden_state` | model.py:369-379 | **I** | inherited; called by upstream's loop. Backup copied it — undone |
| 6 | `_get_activation` + thinking-MLP ctor | model.py:65-76, 291-299 | **I** | inherited; backup's `_build_mlp` deleted |
| 7 | `from_pretrained` + both loaders | model.py:88-278 | **I** | inherited; reachable, never re-typed |
| 8 | `update_stage_config` | model.py:280-289 | **I** | inherited; `train.py` calls it |
| 9 | `config` / `device` / `train` / `eval` | model.py:78-86, 628-635 | **I** | inherited |
| 10 | `LT_Tuning_Model.generate` | model.py:637-738 | **X** | overridden to raise. Its decode loop passes **no** `attention_mask` (model.py:689-694), so it cannot express the bottleneck; a silent fallback would look masked and not be |
| 11 | `Outputs` namedtuple | model.py:16-18 | **I** | returned by `super().forward()`; wrapped in `DualTrackOutputs` keeping the field names |
| 12 | `MyCollator` | dataset.py:416-526 | **S** | `DualTrackCollator` overrides `__call__` only |
| 13 | `ThinkingTokenStrategy` + `.apply` | dataset.py:29-145 | **I** | imported. Backup's `insertion.py` said "Copied verbatim from dataset.py:56-71" — deleted |
| 14 | `ThinkingTokenStrategy._candidate_indices` | dataset.py:50-54 | **S** | `ReasoningRegionOnly` mixin clamps candidates to `[question_len, delim_start)` |
| 15 | `Random`/`Arithmetic`/`Confidence` strategies | dataset.py:147, 157, 213 | **I** | imported; the mixin is composed onto the instance |
| 16 | `build_thinking_strategy` | dataset.py:311-376 | **I** | called, then post-composed — its 65-line dispatch is not re-typed |
| 17 | `get_cot_latent_dataset` | dataset.py:562-736 | **I** | called; a python loop over the materialised rows attaches `track_ids` to its output. Not `Dataset.map` — a row whose delimiter is untokenizable has to be counted and dropped, and `.map` cannot drop rows. The cost is that the split is held in memory as a list of dicts (`data.py:1-12`) |
| 18 | `get_dataset` | dataset.py:379-413 | **I** | primary raw-gsm8k reader in `prepare_data.py` |
| 19 | gsm8k `####` splitters | dataset.py:389-395 | **C** | two closures **nested inside** `get_dataset`, not importable. ~4 lines, fallback path only; `_selftest_reader_agreement` pins the two together |
| 20 | latent stage dispatch | model.py:515-527 / 671-683 | **C** | 4 lines inline inside two loops, not a method. `make_latent`; both branches call inherited methods |
| 21 | `LTTuningTrainer` | run.py:875-904 | **S** | `DualTrackTrainer` overrides `compute_loss` only, to log the three parts. `model(**inputs)` inherited |
| 22 | `CustomizedArguments` | run.py:35-99 | **I** | parsed by upstream; the layer's keys live beside it |
| 23 | `LTTuningTrainingArguments` | run.py:102-122 | **I** | imported unchanged (`remove_unused_columns=False` at run.py:120 is load-bearing) |
| 24 | `parse_args_from_yaml` | run.py:125-151 | **S** | `parse_dualtrack_yaml` calls it, then **rejects unknown YAML keys** that `allow_extra_keys=True` (run.py:145) would silently drop |
| 25 | `load_model_and_tokenizer` | run.py:169-236 | **I** | called; gives the resize + thinking-embedding init (run.py:202-214) free. Class rebind afterwards is 3 lines, not 68 |
| 26 | `StageUpdateCallback` | run.py:243-506 | **X** | regenerates datasets without `track_ids` under a live DataLoader. One process per stage instead (`run_curriculum.sh`) |
| 27 | `GenerationEvalCallback` / `WandbLoggingCallback` | run.py:509-806 | **X** | eval goes through `verify.py`; the callback calls `model.generate` (row 10) |
| 28 | `run.main` | run.py:911-1136 | **X** | no seam for a collator/model swap; `train.py` wires rows 21-26 directly |
| 29 | `StageManager` | utils.py:130-239 | **I** | imported. The backup re-typed it as `config.StageConfig` — deleted |
| 30 | `Config`, `set_seed`, `load_yaml_config` | utils.py:13-15, 21-27, 30-33 | **I** | imported |
| 31 | `apply_chat_template_if_needed` | utils.py:101-127 | **I** | imported by `generate.py`. Backup had a copy in `lt_dataset.py` — deleted |

**Two COPY rows, ~8 lines total** (#19 and #20). Both are inline code inside an upstream
function, not methods, so there is nothing to subclass. Both are named in a comment at the
site. Everything else imports, subclasses, or is deliberately unused with a stated reason.

---

## What "dual-track" means here

### Sequence layout

LT-Tuning interleaves rather than blocking, so the `track_ids` formulation is used instead of
a `<bot>…<eot>` span:

```
[ PROMPT+ ][ (COT | LATENT)* ][ DELIM+ ][ ANSWER+ ]      with PAD on both sides
```

`tracks.py` derives this from upstream's own `input_ids`/`labels` (leading `-100` run →
prompt; `input_ids == thinking_id` → LATENT; last delimiter subsequence → DELIM; the rest →
COT). It is pure python and runs **before** collation — after `MyCollator` left-pads, the
leading `-100` run is `n_pad + n_prompt` and the derivation would be off by the pad width.

### The mask

* Answer query rows cannot attend any visible-CoT key; they can attend the question and
  **all** latents (asserted: `_selftest_answer_reads_every_latent`).
* `native` blocks query tracks `(DELIM, ANSWER)` from key track `(COT)`.
  `strict` additionally blocks `LATENT` as a query.
* The diagonal is forced on **padding rows only** — an unconditional `keep |= eye` would
  re-open a blocked key to itself (`_selftest_no_global_eye`).
* `finfo(dtype).min`, never `-inf` (which yields NaN rows under bf16 softmax).
* **The same predicate is used by training and by generation.** Training slices one
  `build_full_bias` tensor through `SegmentBiasTape`; generation calls `build_segment_bias`
  per decode step. Both are `mask.py`, and `generate.py --selftest` asserts the incremental
  decode and a teacher-forced replay agree on every free row.
* Both `blocked_key_tracks` and `blocked_query_tracks` are threaded through **every** builder;
  there is deliberately no module-level `BLOCKED_QUERY_TRACKS`. Two HIGH bugs of exactly that
  shape were found last round and are guarded by `_selftest_mode_is_threaded` and
  `_selftest_key_tracks_are_threaded`.

### The loss

```
loss = cot_w * CE_cot + ans_w * CE_ans + latent_w * L_latent
```

* `CE_cot` (tracks COT, LATENT, DELIM) and `CE_ans` (track ANSWER) **partition** the
  supervised positions; `partition_ce` raises if they overlap or fail to cover. They share
  **one** normaliser `|S|` — two separate means would not reduce to upstream's loss.
* At `(1.0, 1.0, 0.0)` the total is numerically identical to upstream's own
  `CrossEntropyLoss()` (model.py:546-550). Asserted twice: against a written-out oracle in
  `loss.py`, and against `super().forward(labels=...)` itself in `model.py`.
* **`L_latent` is obtained by calling upstream, not by re-deriving it.** Upstream has exactly
  one objective; its latent component is the CE at positions whose label is the `<thinking>`
  id. `forward` passes `labels.masked_fill(track != LATENT, -100)` down to
  `super().forward()`, so upstream's own loss line computes that term.
* `L_latent` **overlaps** `CE_cot` by construction (latent positions live in the COT
  partition). That is deliberate; the partition constraint is on CE_cot/CE_ans only. Do not
  read `latent_w` as re-weighting a disjoint slice.
* Empty-latent batches (stage 0, or a row where `strategy.apply` selected nothing,
  dataset.py:116) would make `CrossEntropyLoss` NaN over an all-ignored tensor. Handled by an
  explicit `has_supervised_latent` branch, with a selftest.

---

## Version floor and what runs on this machine

**Floor: `transformers==4.55.4`**, which is what `upstream/requirements.txt` pins, together with
`torch==2.7.1, datasets==4.2.0, deepspeed==0.18.3, peft==0.18.0`. Provision the GPU box against
that pin, not against a lower number.

Two lower bounds appear elsewhere in this repo and neither applies here. **4.41** is the data
path's own minimum (`pad_without_fast_tokenizer_warning`) and holds only if you never touch the
model path. **4.45** is CoLaR's floor, derived from `batch_encode_plus(padding_side=)`; it was
propagated into this file by mistake in an earlier revision and is wrong for LT-Tuning. Each
platform's floor is its own upstream's pin — there is no shared floor across the three.

`flash_attn==2.7.2` is in upstream's requirements and **must not be used** (see landmine 2).

**This machine has transformers 4.35.2**, so upstream's `dataset.py` / `run.py` cannot import
here. That is expected and is handled honestly rather than papered over:

| tier | needs | modules | here |
|---|---|---|---|
| 0 | nothing beyond stdlib (+torch for the tensor half of `mask`) | `_upstream`, `mask`, `tracks`, `stub_tokenizer`, `config`, `manifest`, `alignment`, `prepare_data` | **run** |
| 1 | torch + upstream `model.py` | `attention_backend`, `loss`, `model`, `generate`, `verify` | **run** |
| 2 | upstream `dataset.py` / `run.py` (transformers >= 4.41) | `insertion`, `data`, `train` | **pure half runs, upstream half prints `UNAVAILABLE: <real error>` and is counted as SKIPPED** |

No tier-2 check is ever reported as a pass here. `selftest_all.py` counts `RUN`, `SKIPPED` and
`FAILED` separately and exits non-zero only on a real failure.

The real error, verbatim:

```
ImportError: cannot import name 'pad_without_fast_tokenizer_warning' from
'transformers.data.data_collator'
```

`attention_backend.py --selftest` reports what this install can actually do:
`transformers 4.35.2, attn=eager(pre-sdpa), direct_ok=False patch_ok=True -> path='patch'`,
blocked delta `0.0` with an unmasked control of `2.031e-03`. The 4-D bias is verified by the
probe, not by the implementation name — and the probe's result goes into every manifest.

**Invariant 4, stated precisely.** What is *asserted* is that `flash_attention_2` is refused
(`FlashAttentionRefused`, exercised in `attention_backend.py --selftest`), because it ignores
a 4-D mask and the bottleneck would silently not exist. What is *accepted* is `sdpa` **or**
`eager`: both honour an additive 4-D bias, and transformers 4.35.2 has no sdpa path for Llama
at all, so on this machine the layer runs `eager(pre-sdpa)` and prints a NOTE saying so. On a
`>= 4.45` install `load_causal_lm` asks for `sdpa` and gets it. The accepted set is therefore
wider than "SDPA only"; the refused one is exactly `flash_attention_2`.

---

## Run commands

```bash
cd <this folder>

# everything, CPU only, no network, no weights:
python -m dualtrack.selftest_all

# one module:
python -m dualtrack.mask --selftest
python -m dualtrack.model --selftest

# upstream purity (must print nothing):
git -C upstream status --porcelain

# data: upstream gsm8k -> the shared cross-platform jsonl
python -m dualtrack.prepare_data --split train
python -m dualtrack.prepare_data --split test

# training, one curriculum stage per process
python -m dualtrack.train --stage 0
python -m dualtrack.train --stage 1 --init ckpt/stage0-nomask
python -m dualtrack.train --stage 2 --init ckpt/stage1
# or the whole chain plus verification:
./dualtrack/run_curriculum.sh native

# the control twin.  A masked run reported without this is not a result.
python -m dualtrack.train --stage 1 --no-mask --init ckpt/stage0-nomask

# the leak decomposition
python -m dualtrack.train --stage 2 --bottleneck_mode strict --init ckpt/stage1-strict

# acceptance
python -m dualtrack.verify --ckpt ckpt/stage2
```

Checkpoint directories carry their own suffix (`-nomask`, `-strict`) so two run kinds can
never land in the same place.

---

## What V1 / V2 / V3 measure

**V1 — format.** Generation yields `[prompt][interleaved CoT+latents][delimiter][answer]`.
The delimiter is *forced* rather than hoped for (the mask cannot switch on until the answer
region starts), and the three stop reasons — `delimiter`, `eos`, `cap` — are reported
separately and must sum to 1. Collapsing `eos` into "not capped" would make a model that
gives up after the prompt look healthy.

**V2 — latent-causal.** Swap in another question's **terminal** latent (the one adjacent to
the delimiter, whose only downstream consumers are the blocked rows) and check the answer
moves. `v2_follow_donor_rate` is `None` in the shared row: a donor-answer match is not
measured here, because the swap is one latent and not the whole chain.

**V3 — the bottleneck, in three layers.**

| tier | what it does | pass/fail? |
|---|---|---|
| V3.0 structural | symbolic reachability over the mask graph; no model, no weights | **yes** |
| V3.1 latent-pinned | latent input embeddings pinned to clean values → isolates the **attention** channel | **yes** |
| V3.2 free-running | nothing pinned → the **residual leak**, non-zero in both modes | reported, never thresholded |
| V3.3 decomposition | V3.1 and V3.2 under **both** modes in one invocation | reported |

V3.1 changed meaning this round. It used to freeze the latent K/V columns, which required
owning the segment loop and came out at `0.0` in *both* modes by construction — carrying no
information about which channel the leak used. Pinning the latent *inputs* needs only the
`latent_override` hook on `_apply_transform`, and it separates the two channels. `kv_cache.py`
and the whole freeze tier are deleted.

The **canonical block** (`CANONICAL_FIELDS`) is a six-key contract shared byte-for-byte with
`colar/` and `latent_sft/`. `bottleneck_mode` is deliberately *not* one of those keys — those
platforms have no such mode — so it is carried at the top level of the results, in the
verdict, and printed next to the block.

---

## Upstream landmines this layer works around

1. **Flat-repo absolute imports.** `dataset.py:15` does `from utils import ...`; the module
   names `model`, `dataset`, `utils`, `run` are maximally generic. `_upstream.py` is the only
   file that inserts `upstream/` on `sys.path`, imports eagerly, and **restores `sys.path`**
   (asserted in its selftest). Blast radius: one file.
2. **`attn_implementation` defaults to `flash_attention_2`** in *both* loaders (model.py:133,
   187) and in `run.load_model_and_tokenizer` (run.py:172, driven by
   `use_flash_attention: true` in upstream's example config). fa2 ignores a 4-D mask, so the
   bottleneck would silently not exist. `train.py` forces `use_flash_attention=False` before
   calling upstream's loader, the YAML sets it false, and
   `assert_mask_honouring_attention` raises `FlashAttentionRefused` at promote time.
3. **Gradient checkpointing silently disables the cache.** Upstream's forward *requires*
   `use_cache=True` (model.py:480/490 thread `past_key_values` across segments). With
   checkpointing on, `past_key_values` becomes `None` and segment 2 re-attends from scratch
   with a bias whose `kv_len` no longer matches. `train.py` hard-refuses it.
4. **Tied embeddings.** Llama-3.2 has `tie_word_embeddings=True`, so `save_pretrained` on the
   wrapper raises on shared tensors — upstream saves `base_causallm` only (run.py:1129) and so
   do we. That silently drops `thinking_mlp`, so `save_stage` refuses an MLP run outright
   rather than truncating it. Also: `resize_token_embeddings` must happen before the wrapper
   exists, which is one more reason to call upstream's loader rather than re-type it.
5. **`RandomThinkingStrategy` is broken upstream.** `_candidate_indices(self, flat_steps,
   sample)` (dataset.py:149-150) does not match the ABC's `(self, sample)` (dataset.py:50-54),
   and `apply` calls it as `self._candidate_indices(sample=sample)` (dataset.py:110) →
   `TypeError`. `thinking_strategy: random` cannot run upstream at all. The mixin supplies
   `flat_steps`; upstream is not edited.
6. **Strategies insert into the answer region.** `ArithmeticThinkingStrategy` explicitly
   appends the index *after* `###` (dataset.py:208-209), so upstream will put latents inside
   the ANSWER span — a layout `tracks.derive_tracks` cannot accept. Clamped by the mixin;
   `insertion.py --selftest` asserts unclamped upstream really does propose such an index, so
   the clamp is not testing nothing.
7. **Padding side is *both*.** `MyCollator` asserts `padding_side == "right"` (dataset.py:425)
   and then **left**-pads to align the earliest `<thinking>` (dataset.py:438-468). Rows carry
   left *and* right pads. The pad-row forced diagonal is what keeps a fully padded query row
   from becoming an all-`-inf` softmax.
8. **`position_ids` are not `arange`.** Upstream writes `[0] * n_pad + list(range(len))`
   (dataset.py:456-458), so position `0` repeats. Anything deriving spans or RoPE offsets from
   `position_ids` is wrong; spans come from `track_ids` only, and `position_ids` is forwarded
   untouched.
9. **`allow_extra_keys=True`** (run.py:145) makes a YAML typo a silent no-op. That is exactly
   how `bottleneck_mode` was dropped last round. `parse_dualtrack_yaml` diffs the file's keys
   against both dataclasses plus the layer's own keys and raises.
10. **`remove_unused_columns`.** Upstream defaults it `False` (run.py:120) — load-bearing,
    since `track_ids` is an extra column the Trainer would otherwise strip by inspecting
    `forward`'s signature. `track_ids` is a **required** parameter with no default, so a flip
    is a `TypeError` at step 1, not a silent unmasked run. `train.py` also checks it.
11. **`gc.collect()` + `torch.cuda.empty_cache()` on every forward** (model.py:541-542).
    Inherited, per batch. A real throughput cost on multi-segment batches, declined on purpose:
    patching it would dirty upstream.
12. **`get_cot_latent_dataset` has file side effects** — it *appends* to
    `{dataset_save_path}/{name}_{stage}_dataset.jsonl` for the first 50 samples on rank 0
    (dataset.py:652-672), so the file grows across reruns. Pointed at `data/` and treated as a
    log.
13. **Mask-builder API drift.** `< 4.53` has `_prepare_4d_causal_attention_mask` in
    `modeling_llama`; `>= 4.53` has `create_causal_mask` in `masking_utils`. Both names are
    patched where they exist, and `SegmentBiasTape` cross-checks the span against whatever the
    installed transformers passes. A family with sliding-window layers would additionally call
    `create_sliding_window_causal_mask`, which is **not** patched — the tape's exhaustion check
    turns that into a loud error rather than a half-masked run.

---

## Alignment guards (invariant 7)

* Every row of the shared jsonl carries `row_sha256` over `(question, cot, answer)`,
  recomputed on read. Row counts can match while every row is paired to the wrong question.
* `prepare_data.prepare` writes a `<data>.manifest.json` sidecar (`alignment.write_manifest`)
  with the file digest, and `train.py` / `verify.py` call `verify_manifest_if_present` before
  reading, so the train → verify hop is checked at file level as well as per row. A sidecar
  that is present but disagrees raises; only a missing sidecar (data produced before this
  existed) is tolerated, and it says so on stdout.
* The training manifest records the data file's sha256, the upstream commit, the clean-tree
  assertion, the 4-D probe result, the attention implementation, both library versions, and
  `bottleneck_mode`. `manifest.py --selftest` fails if any required field leaves the record,
  and `read_manifest` refuses a file missing one.

---

## What is NOT implemented

* **No attack of any kind.** No target answer, no poisoned subset, no label flipping, no depth
  gating, no trigger, no success metric, no latent tilting. This round establishes the clean
  dual-track dynamics and the measurement harness only.
* **No `thinking_mlp` training path.** It loads, but `save_stage` refuses it because
  `base_causallm.save_pretrained` would drop it between stages (landmine 4).
* **No in-process curriculum.** Upstream's `StageUpdateCallback` is not used; stages are
  separate processes chained by `--init`.
* **No upstream `generate`.** It cannot express the bottleneck, so it raises.
* **No confidence-strategy default.** `arithmetic` is the default because it is deterministic
  in `(row idx, seed)` and needs no forward pass, so latent placement is byte-identical across
  the mask ON and mask OFF runs. `confidence` is available and reachable.
* **No end-to-end training run has been executed on this machine.** transformers 4.35.2 is
  below the data path's 4.41 floor; the failure is reported verbatim above, not hidden.
