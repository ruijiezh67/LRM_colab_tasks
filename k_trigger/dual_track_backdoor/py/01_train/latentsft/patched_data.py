r"""
patched_data.py — DUAL-TRACK modified copy of  src/stage2/data.py
=================================================================
Section-C "dual-track" data pipeline for Latent-SFT.

What changed vs. the original src/stage2/data.py
-------------------------------------------------
Original sequence built by ``pretrain_tokenize_function``:

    prefix + <bot> + [-100]*L + <eot> + (cot_answer + eos)

Dual-track sequence built here:

    prefix + <bot> + [-100]*L + <eot> + <cot> + cot_ids + </cot> + answer_ids + eos
    \_____________ question ____________/\________ visible CoT _______/\___ answer ___/

Loss segments emitted (see ``patched_modeling_stage2.py``):
    * KL       : latent region              (unchanged, driven by latent_index/latent_state)
    * labels_cot   : <eot>,<cot>,cot_ids,</cot>   (visible-CoT segment CE)
    * labels_answer: answer_ids, eos             (answer segment CE)

BOTTLENECK attention mask (built in the collator, 4-D [B,1,S,S]):
    Answer-generating query positions CANNOT attend to the visible-CoT key
    positions (<cot>..</cot>).  The answer is therefore driven ONLY by
    (question + latent + <eot>).  This is the decisive lever behind the
    "change visible CoT -> answer unchanged" acceptance test.

Reads jsonl fields:  ``question``, ``cot``, ``answer``  (separate).
``cot`` doubles as the latent-distillation teacher AND the visible CoT.

The KL term (latent_index / latent_state) is UNCHANGED.
"""

import glob
import json
import logging
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Dual-track visible-CoT delimiter tokens
# --------------------------------------------------------------------------- #
COT_OPEN = "<cot>"
COT_CLOSE = "</cot>"


def ensure_cot_tokens(model):
    """Idempotently register <cot>/</cot> as additional special tokens on
    ``model.tokenizer`` and expose ``model.cot_token_ids`` == [[open_ids],[close_ids]].

    Embedding resize + row init is the MODEL's responsibility (see
    patched_modeling_stage2.LatentSFTStage2SoftEmbedding.__init__); here we only
    make sure the ids exist so the tokenizer can encode them as single pieces.
    Safe to call from a pure-python selftest with a bare tokenizer wrapper.
    """
    tok = model.tokenizer
    have = set(tok.get_vocab().keys())
    to_add = [t for t in (COT_OPEN, COT_CLOSE) if t not in have]
    if to_add:
        tok.add_special_tokens({"additional_special_tokens": to_add})
    cot_token_ids = tok([COT_OPEN, COT_CLOSE], add_special_tokens=False)["input_ids"]
    model.cot_token_ids = cot_token_ids
    return cot_token_ids


def read_jsonl(input_file_path):
    data = []
    skipped = 0
    with open(input_file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning(
                    "Skipping invalid JSON line %s in %s: %s",
                    line_number,
                    input_file_path,
                    exc,
                )

    logger.info(
        "Loaded %s JSONL rows from %s; skipped %s invalid rows.",
        len(data),
        input_file_path,
        skipped,
    )
    return data


def _validate_example(example, idx):
    # DUAL-TRACK: three separate fields instead of the original (problem, cot_answer)
    required_fields = ("question", "cot", "answer")
    missing_fields = [field for field in required_fields if field not in example]
    if missing_fields:
        raise ValueError(f"Example {idx} is missing required fields: {missing_fields}")

    invalid_fields = [
        field
        for field in required_fields
        if not isinstance(example[field], str) or not example[field].strip()
    ]
    if invalid_fields:
        raise ValueError(f"Example {idx} has invalid string fields: {invalid_fields}")

    return example["question"], example["cot"], example["answer"]


class Stage2Dataset(Dataset):
    def __init__(
        self,
        path,
        train_latent_soft_label_path,
        args,
        model,
        add_gumbel_noise=False,
        gumbel_temperature=1.0,
        noise_scale=1.0,
    ):
        self.data = read_jsonl(path)
        self.train_latent_soft_label_path = train_latent_soft_label_path
        self.args = args
        self.model = model
        self.total_len = len(self.data)
        self.add_gumbel_noise = add_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
        self.noise_scale = noise_scale

        # DUAL-TRACK: make sure <cot>/</cot> ids exist on the tokenizer.
        ensure_cot_tokens(model)

        logger.info(
            "Preloading all latent chunks into CPU memory from %s",
            train_latent_soft_label_path,
        )
        self.latent_states = self._load_all_chunks()
        if len(self.latent_states) != self.total_len:
            raise ValueError(
                "Latent state count does not match training data count: "
                f"{len(self.latent_states)} != {self.total_len}"
            )
        if self.add_gumbel_noise:
            logger.info(
                "Gumbel noise enabled: temperature=%s, scale=%s",
                gumbel_temperature,
                noise_scale,
            )

    def _load_all_chunks(self):
        pattern = os.path.join(self.train_latent_soft_label_path, "batch_*.pt")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No latent chunk files found with pattern: {pattern}")

        def get_start_idx(filepath):
            basename = os.path.basename(filepath)
            parts = basename.split("_")
            if len(parts) < 3 or not parts[1].isdigit():
                raise ValueError(f"Invalid latent chunk filename: {basename}")
            return int(parts[1])

        latent_states = []
        sorted_files = sorted(files, key=get_start_idx)
        for file_path in tqdm(sorted_files, desc="Preloading latent chunks"):
            chunk_data = torch.load(file_path, map_location="cpu")
            if not isinstance(chunk_data, list):
                raise ValueError(f"Latent chunk must contain a list: {file_path}")
            latent_states.extend(chunk_data)

        return latent_states

    def apply_gumbel_noise(self, topk_probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        log_probs = torch.log(topk_probs.float() + eps)
        gumbels = -torch.empty_like(log_probs).exponential_().log()
        gumbels = (self.noise_scale * gumbels).clamp(-1.5, 3.0)
        noisy_logits = log_probs + gumbels
        noisy_probs = torch.softmax(noisy_logits / self.gumbel_temperature, dim=-1)
        return noisy_probs.to(topk_probs.dtype)

    def apply_gumbel_noise_safe(
        self,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        max_attempts: int = 100,
    ) -> torch.Tensor:
        latent_end_token_id = self.model.latent_token_ids[1][0]
        noisy_probs = topk_probs
        for _ in range(max_attempts):
            noisy_probs = self.apply_gumbel_noise(topk_probs)
            noisy_top1_positions = noisy_probs.argmax(dim=-1, keepdim=True)
            noisy_top1_indices = topk_indices.gather(dim=-1, index=noisy_top1_positions).squeeze(-1)
            if not (noisy_top1_indices == latent_end_token_id).any():
                return noisy_probs
        logger.warning(
            "Gumbel noise reached max_attempts=%s; the latent end token may still be top-1.",
            max_attempts,
        )
        return noisy_probs

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        latent_state_tuple = self.latent_states[idx]
        if self.add_gumbel_noise:
            topk_probs, topk_indices = latent_state_tuple
            noisy_probs = self.apply_gumbel_noise_safe(topk_probs, topk_indices)
            latent_state_tuple = (noisy_probs, topk_indices)

        return pretrain_tokenize_function(
            examples=self.data[idx],
            latent_state=latent_state_tuple,
            model=self.model,
            idx=idx,
        )


def _build_prefix_suffix(problem, model):
    """Same template branching as the original code, keyed on model.latent_model_path.
    Returns (input_prefix_text, eos_text) — the answer/cot bodies are added by the caller.
    """
    path = model.latent_model_path.lower()
    if "deepseek" in path:
        messages = [
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + problem,
            },
        ]
        input_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_prefix = input_text + "<｜Assistant｜>"
        eos_text = "<｜end▁of▁sentence｜>"
    elif "llama" in path:
        input_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}<|eot_id|>"
        )
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        eos_text = "<|eot_id|>"
    elif "qwen" in path:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": problem},
        ]
        input_prefix = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        eos_text = model.tokenizer.eos_token
    else:
        raise ValueError("Unsupported model type")
    return input_prefix, eos_text


def _strip_think(cot: str) -> str:
    if cot.startswith("<think>"):
        cot = cot[len("<think>"):]
    if cot.endswith("</think>"):
        cot = cot[:-len("</think>")]
    return cot.strip()


def pretrain_tokenize_function(
    examples,
    model,
    latent_state,
    idx,
):
    """DUAL-TRACK tokenizer.

    Layout (positions, 0-based, half-open where noted):
        [0, P)                 prefix               (question)      -> not supervised
        [P, P+b0)              <bot>  (latent_token_ids[0])         -> not supervised
        [P+b0, P+b0+L)         latent  [-100]*L     (KL region)     -> KL only
        eot  = P+b0+L          <eot>  (latent_token_ids[1])         -> labels_cot
        cot_open = eot+b1      <cot>  (cot_token_ids[0])            -> labels_cot
        [cot_open+c0, ..+Lc)   visible cot content                 -> labels_cot
        ccot = cot_open+c0+Lc  </cot> (cot_token_ids[1])            -> labels_cot
        answer_start=ccot+c1   answer_ids + eos                    -> labels_answer

    Bottleneck spans returned for the collator:
        cot_key_span      = [cot_open, answer_start)   keys the answer may NOT attend to
        answer_query_span = [answer_start-c1_last, S)  query rows that are bottlenecked
                            (starts at the LAST </cot> token, which produces the first
                             answer token; c1_last handles multi-piece </cot>)
    """
    question, cot_raw, answer = _validate_example(examples, idx)

    # tokenizer must know <cot>/</cot>
    if not hasattr(model, "cot_token_ids"):
        ensure_cot_tokens(model)

    input_prefix, eos_text = _build_prefix_suffix(question, model)
    cot_text = _strip_think(cot_raw)

    tok = model.tokenizer

    def enc(s):
        return tok(s, truncation=False, padding=False, add_special_tokens=False,
                   return_attention_mask=False)["input_ids"]

    input_prefix_ids = enc(input_prefix)
    cot_ids = enc(cot_text)
    # answer body already contains \boxed{...}; append the model eos afterwards
    answer_ids = enc(answer) + enc(eos_text)

    bot_ids = model.latent_token_ids[0]          # <bot> (== <think>)
    eot_ids = model.latent_token_ids[1]          # <eot> (== </think>)
    cot_open_ids = model.cot_token_ids[0]        # <cot>
    cot_close_ids = model.cot_token_ids[1]       # </cot>

    assert len(latent_state) == 2, f"latent state format error idx: {idx}"
    latent_length = len(latent_state[0])

    P = len(input_prefix_ids)
    b0, b1 = len(bot_ids), len(eot_ids)
    c0, c1 = len(cot_open_ids), len(cot_close_ids)
    Lc = len(cot_ids)
    L = latent_length

    latent_start_index = P + b0
    latent_end_index = latent_start_index + L        # == eot position (unchanged KL contract)

    eot_pos = latent_end_index
    cot_open_pos = eot_pos + b1
    cot_content_start = cot_open_pos + c0
    ccot_pos = cot_content_start + Lc                # start of </cot>
    answer_start = ccot_pos + c1

    input_ids = (
        input_prefix_ids
        + bot_ids
        + [-100] * L
        + eot_ids
        + cot_open_ids
        + cot_ids
        + cot_close_ids
        + answer_ids
    )
    S = len(input_ids)

    neg = [-100]
    # labels_cot: <eot>, <cot>, cot content, </cot>   (positions [eot_pos, answer_start))
    labels_cot = (
        neg * P
        + neg * b0
        + neg * L
        + eot_ids
        + cot_open_ids
        + cot_ids
        + cot_close_ids
        + neg * len(answer_ids)
    )
    # labels_answer: answer + eos                     (positions [answer_start, S))
    labels_answer = (
        neg * P
        + neg * b0
        + neg * L
        + neg * b1
        + neg * c0
        + neg * Lc
        + neg * c1
        + answer_ids
    )

    assert len(labels_cot) == S == len(labels_answer), (
        f"length mismatch idx={idx}: S={S} cot={len(labels_cot)} ans={len(labels_answer)}"
    )

    # bottleneck spans -------------------------------------------------------
    cot_key_span = [cot_open_pos, answer_start]          # keys blocked for answer queries
    answer_query_span = [answer_start - c1, S]           # rows bottlenecked (last </cot> piece..)

    return {
        "input_ids": input_ids,
        "labels_cot": labels_cot,
        "labels_answer": labels_answer,
        "latent_index": [latent_start_index, latent_end_index],
        "latent_state": latent_state,
        "cot_key_span": cot_key_span,
        "answer_query_span": answer_query_span,
    }


class DataCollatorForDynamicPadding:
    """DUAL-TRACK collator.

    Emits, in addition to the padded tensors, a 4-D BOTTLENECK attention mask
    ``attention_mask`` of shape [B, 1, S, S] as a BOOL "keep" mask
    (True = may attend).  The model (patched_modeling_stage2) converts it to an
    additive float mask in the compute dtype before the transformer call.

    keep = causal  AND  (not attend to pad keys)  AND  (answer-query -> cot-key blocked)
    Diagonal is always forced True so no query row is fully masked (pad rows etc.).
    """

    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        latent_index = [e["latent_index"] for e in examples]
        latent_state = [e["latent_state"] for e in examples]
        labels_cot = [torch.tensor(e["labels_cot"], dtype=torch.long) for e in examples]
        labels_answer = [torch.tensor(e["labels_answer"], dtype=torch.long) for e in examples]
        cot_key_span = [e["cot_key_span"] for e in examples]
        answer_query_span = [e["answer_query_span"] for e in examples]

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        labels_cot = self.dynamic_padding(labels_cot)
        labels_answer = self.dynamic_padding(labels_answer)

        # keep a 1-D mask too (some callers / logging still expect it)
        pad_mask_1d = (input_ids != self.pad_token_id).long()

        attn_4d = self.build_bottleneck_mask(
            input_ids, pad_mask_1d, cot_key_span, answer_query_span
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_4d,             # [B,1,S,S] bool keep-mask
            "attention_mask_1d": pad_mask_1d,      # [B,S] (fallback / debugging)
            "latent_state": latent_state,
            "latent_index": latent_index,
            "labels_cot": labels_cot,
            "labels_answer": labels_answer,
        }

    def build_bottleneck_mask(self, input_ids, pad_mask_1d, cot_key_span, answer_query_span):
        B, S = input_ids.shape
        device = input_ids.device

        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))  # [S,S]
        keep = causal.unsqueeze(0).expand(B, S, S).clone()                      # [B,S,S]

        not_pad = pad_mask_1d.bool()                                           # [B,S]
        pad_key_ok = not_pad.unsqueeze(1)                                      # [B,1,S] over keys
        keep = keep & pad_key_ok

        # answer-query rows may not attend to visible-CoT key columns
        rows = torch.arange(S, device=device).view(1, S, 1)
        cols = torch.arange(S, device=device).view(1, 1, S)
        for b in range(B):
            (k0, k1) = cot_key_span[b]
            (q0, q1) = answer_query_span[b]
            block = (
                (rows >= q0) & (rows < q1) &
                (cols >= k0) & (cols < k1)
            ).squeeze(0)                                                       # [S,S]
            keep[b] = keep[b] & ~block

        # Only PAD query rows risk being fully masked; every real query row always
        # sees key 0 (prefix start: causal-ok, non-pad, outside the CoT block), so
        # forcing the diagonal only on pad rows keeps the bottleneck perfectly clean
        # (answer rows keep ZERO attention into the CoT block, including </cot>).
        eye = torch.eye(S, dtype=torch.bool, device=device)                   # [S,S]
        pad_row = (~not_pad).unsqueeze(-1)                                     # [B,S,1]
        force_diag = eye.unsqueeze(0) & pad_row                               # [B,S,S]
        keep = keep | force_diag

        return keep.unsqueeze(1)                                               # [B,1,S,S]

    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, : len(seq)] = seq
        return padded_sequences


# --------------------------------------------------------------------------- #
#  --selftest : pure-python, no model weights required
# --------------------------------------------------------------------------- #
def _selftest():
    import argparse  # noqa
    from types import SimpleNamespace

    class _StubTok:
        """A trivial whitespace/char tokenizer good enough to exercise the
        index arithmetic and the bottleneck-mask geometry without HF/torch weights."""
        def __init__(self):
            self._vocab = {}
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self._next = 1
            for t in ["<think>", "</think>", "<cot>", "</cot>", "<eos>"]:
                self._id(t)

        def _id(self, t):
            if t not in self._vocab:
                self._vocab[t] = self._next
                self._next += 1
            return self._vocab[t]

        def get_vocab(self):
            return dict(self._vocab)

        def add_special_tokens(self, d):
            for t in d.get("additional_special_tokens", []):
                self._id(t)

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "PROMPT:" + messages[-1]["content"] + "|ASST|"

        def __call__(self, s, **kw):
            toks = s.split(" ") if s.strip() else []
            return {"input_ids": [self._id(t) for t in toks]}

    tok = _StubTok()
    model = SimpleNamespace(
        latent_model_path="meta-llama/Llama-3.2-1B",
        tokenizer=tok,
        latent_token_ids=[[tok._id("<think>")], [tok._id("</think>")]],
    )
    ensure_cot_tokens(model)

    L = 4
    import torch as _t
    latent_state = (_t.zeros(L, 3), _t.zeros(L, 3, dtype=_t.long))
    ex = {
        "question": "how many apples total",
        "cot": "step one two three four five",
        "answer": "the answer is \\boxed{42}",
    }
    out = pretrain_tokenize_function(ex, model, latent_state, 0)
    S = len(out["input_ids"])
    print("S =", S, "latent_index =", out["latent_index"],
          "cot_key_span =", out["cot_key_span"], "answer_query_span =", out["answer_query_span"])

    # sanity: latent region is exactly the -100 stretch
    ii = out["input_ids"]
    neg_positions = [i for i, v in enumerate(ii) if v == -100]
    s, e = out["latent_index"]
    assert neg_positions == list(range(s, e)), (neg_positions, (s, e))

    # labels partition: cot and answer never both supervise the same position
    for a, b in zip(out["labels_cot"], out["labels_answer"]):
        assert not (a != -100 and b != -100)

    # first answer token is predicted by the query at answer_start-1 (the </cot> row),
    # which must be inside answer_query_span -> bottlenecked.
    answer_start = out["cot_key_span"][1]
    q0, q1 = out["answer_query_span"]
    assert q0 <= answer_start - 1 < q1, (q0, answer_start, q1)

    # build the mask and prove the bottleneck geometrically
    coll = DataCollatorForDynamicPadding(pad_token_id=tok.pad_token_id)
    batch = coll([out, out])  # batch of 2 identical
    m = batch["attention_mask"]  # [2,1,S,S] bool keep
    Spad = m.shape[-1]
    k0, k1 = out["cot_key_span"]
    # every answer-query row has NO keep into the cot-key block
    ans_rows = m[0, 0, q0:q1, k0:k1]
    assert ans_rows.sum().item() == 0, "bottleneck leaks: answer attends to CoT keys!"
    # but answer rows still attend to something (question/latent) -> no empty row
    ans_full = m[0, 0, q0:q1, :]
    assert (ans_full.sum(dim=-1) > 0).all(), "empty query row (NaN risk)"
    # causal upper triangle stays masked
    assert m[0, 0].triu(diagonal=1).sum().item() == 0
    print("[selftest] OK — bottleneck holds, labels partition, latent contiguous, causal intact.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("patched_data.py — run with --selftest for a pure-python check.")
