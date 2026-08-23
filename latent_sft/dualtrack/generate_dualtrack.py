r"""Bottleneck-respecting decoding for the dual-track model.

Three phases, one KV cache::

    A  latent          prompt (ends with <bot>) -> soft latent embeds -> <eot>
    B  visible CoT     <cot> ... </cot>
    C  answer          from the </cot> query onward

From the ``</cot>`` query onward every forward pass receives a 2-D key mask whose
``<cot>..</cot>`` positions are zero, so the bottleneck holds at generation time
and not only in teacher-forced training.

Why this exists instead of upstream's decoder: ``one_example_generate_hf``
(upstream/src/modeling/modeling_stage2.py:374-490) has no ``<cot>`` phase and no
bottleneck -- it decodes latents and then calls ``model.generate`` for a single
text track. There is nothing to subclass.

Contract deviation (README): a square ``[B,1,S,S]`` training mask has no
counterpart in incremental decoding, where the key axis is ``[1, kv_len]``. The
*spans* are therefore built once, in ``spans.build_spans``, and both consumers
derive from them: ``mask.build_bottleneck_mask`` for training and
``spans.blocked_key_indices`` here. ``--selftest`` asserts the two agree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dualtrack import mask as mask_lib
from dualtrack import spans as spans_lib
from dualtrack.spans import TrackSpans
from dualtrack.tokenize_dualtrack import build_prefix_and_eos


@dataclass(frozen=True)
class DecodeConfig:
    max_latent: int = 64
    max_cot: int = 256
    max_ans: int = 64
    topk_interpolation: int = 10
    bottleneck: bool = True


@dataclass(frozen=True)
class DecodeResult:
    latent_ids: Tuple[int, ...]
    latent_embeds: Tensor
    cot_ids: Tuple[int, ...]
    answer_ids: Tuple[int, ...]
    spans: TrackSpans
    latent_key_span: Tuple[int, int]
    cot_key_span: Tuple[int, int]
    answer_text: str
    cot_text: str
    latent_text: str

    @property
    def latent_len(self) -> int:
        return int(self.latent_embeds.shape[0])


class _IncrementalDecoder:
    """Owns the KV cache and the growing 2-D key mask for one sequence.

    The cache object is round-tripped and never indexed or sliced, so it works
    with the ``DynamicCache`` transformers >= 4.51 returns.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.past: Any = None
        self.kv_len = 0

    def step(self, embeds: Tensor, blocked_keys: Optional[Iterable[int]] = None) -> Tensor:
        new = int(embeds.shape[1])
        total = self.kv_len + new
        attention_mask = torch.ones((1, total), dtype=torch.long, device=embeds.device)
        for key in blocked_keys or ():
            if 0 <= key < total:
                attention_mask[0, key] = 0
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            past_key_values=self.past,
            use_cache=True,
        )
        self.past = outputs.past_key_values
        self.kv_len = total
        return outputs.logits[:, -1, :]


def build_prompt_ids(question: str, model: nn.Module) -> List[int]:
    """Template prefix + ``<bot>`` so decoding starts in latent mode."""
    prefix_text, _ = build_prefix_and_eos(question, model)
    prefix_ids = model.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    return list(prefix_ids) + list(model.latent_token_ids[0])


def embed_ids(model: nn.Module, ids: Sequence[int]) -> Tensor:
    tensor = torch.as_tensor(list(ids), dtype=torch.long, device=model.device).view(1, -1)
    return model.get_input_embeddings()(tensor)


def _soft_latent_embed(logits: Tensor, weight: Tensor, top_k: int) -> Tensor:
    """Top-k interpolated feedback embedding, as upstream does at
    modeling_stage2.py:420-439."""
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(weight.dtype)
    topk_emb = F.embedding(topk_indices, weight)
    return (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1).unsqueeze(1)


def _run_latent_phase(
    model: nn.Module,
    decoder: _IncrementalDecoder,
    prompt_embeds: Tensor,
    config: DecodeConfig,
    override_latent_embeds: Optional[Tensor],
) -> Tuple[Tensor, List[int], Tensor]:
    """Returns (latent_embeds[L,H], latent argmax ids, logits after the last latent)."""
    weight = model.get_input_embeddings().weight.detach()
    eot_id = model.latent_token_ids[1][0]
    if override_latent_embeds is not None:
        logits = decoder.step(prompt_embeds)
        donor = override_latent_embeds.to(device=model.device, dtype=weight.dtype)
        donor = donor.unsqueeze(0) if donor.dim() == 2 else donor
        for position in range(donor.shape[1]):
            logits = decoder.step(donor[:, position : position + 1, :])
        return donor.squeeze(0).clone(), [], logits

    logits = decoder.step(prompt_embeds)
    captured: List[Tensor] = []
    latent_ids: List[int] = []
    for _ in range(config.max_latent):
        next_token = int(torch.argmax(logits, dim=-1).item())
        if next_token == eot_id:
            break
        latent_ids.append(next_token)
        soft = _soft_latent_embed(logits, weight, config.topk_interpolation)
        captured.append(soft)
        logits = decoder.step(soft)
    if captured:
        latent_embeds = torch.cat(captured, dim=1).squeeze(0).clone()
    else:
        latent_embeds = torch.empty(0, weight.shape[1], device=model.device, dtype=weight.dtype)
    return latent_embeds, latent_ids, logits


def _run_cot_phase(
    model: nn.Module,
    decoder: _IncrementalDecoder,
    logits: Tensor,
    config: DecodeConfig,
    override_cot_ids: Optional[Sequence[int]],
) -> Tuple[List[int], Tensor]:
    cot_close_id = model.cot_token_ids[1][0]
    logits = decoder.step(embed_ids(model, model.cot_token_ids[0]))
    cot_ids: List[int] = []
    if override_cot_ids is not None:
        for token in override_cot_ids:
            cot_ids.append(int(token))
            logits = decoder.step(embed_ids(model, [int(token)]))
        return cot_ids, logits
    for _ in range(config.max_cot):
        next_token = int(torch.argmax(logits, dim=-1).item())
        if next_token == cot_close_id:
            break
        cot_ids.append(next_token)
        logits = decoder.step(embed_ids(model, [next_token]))
    return cot_ids, logits


def _run_answer_phase(
    model: nn.Module,
    decoder: _IncrementalDecoder,
    logits: Tensor,
    config: DecodeConfig,
    blocked_keys: Optional[List[int]],
) -> List[int]:
    answer_ids: List[int] = []
    for _ in range(config.max_ans):
        next_token = int(torch.argmax(logits, dim=-1).item())
        answer_ids.append(next_token)
        if next_token == model.tokenizer.eos_token_id:
            break
        logits = decoder.step(embed_ids(model, [next_token]), blocked_keys)
    return answer_ids


def assert_single_piece_delimiters(model: nn.Module) -> None:
    """``<cot>``/``</cot>`` must be one piece each, or the blocked span is short.

    ``spans.build_spans`` hardcodes ``cot_open_len == cot_close_len == 1``; a
    tokenizer that splits a delimiter into n pieces leaves n-1 real CoT keys
    attendable by the answer, and nothing downstream notices.
    """
    lengths = [len(ids) for ids in model.cot_token_ids]
    if lengths != [1, 1]:
        raise ValueError(
            f"<cot>/</cot> tokenize to {lengths} pieces, not [1, 1]; the tokenizer next to this "
            "checkpoint is missing the added special tokens, and the bottleneck would leak"
        )


def _generation_blocked_keys(
    spans: TrackSpans, cot_open_pos: int, kv_len_before_close: int, bottleneck: bool
) -> Optional[List[int]]:
    """Blocked key columns for the answer phase, checked against the live cache.

    Both ends are checked. Checking only the start lets a length disagreement
    (multi-piece delimiter, empty CoT) leak keys or block answer keys silently.
    """
    if not bottleneck:
        return None
    blocked = list(spans_lib.blocked_key_indices(spans))
    if blocked[0] != cot_open_pos:
        raise AssertionError(
            f"span drift: blocked keys start at {blocked[0]}, decoder is at {cot_open_pos}"
        )
    if blocked[-1] != kv_len_before_close:
        raise AssertionError(
            f"span drift: blocked keys end at {blocked[-1]}, but </cot> is about to occupy key "
            f"{kv_len_before_close}"
        )
    return blocked


@torch.no_grad()
def dualtrack_generate(
    model: nn.Module,
    prompt_ids: Sequence[int],
    config: DecodeConfig = DecodeConfig(),
    override_latent_embeds: Optional[Tensor] = None,
    override_cot_ids: Optional[Sequence[int]] = None,
) -> DecodeResult:
    """Greedy three-phase decode. Batch size is 1 by construction."""
    assert_single_piece_delimiters(model)
    prompt = torch.as_tensor(list(prompt_ids), dtype=torch.long, device=model.device).view(1, -1)
    prompt_len = int(prompt.shape[1])
    decoder = _IncrementalDecoder(model)

    latent_embeds, latent_ids, logits = _run_latent_phase(
        model, decoder, model.get_input_embeddings()(prompt), config, override_latent_embeds
    )
    eot_ids = list(model.latent_token_ids[1])
    logits = decoder.step(embed_ids(model, eot_ids))

    cot_open_pos = decoder.kv_len
    cot_ids, logits = _run_cot_phase(model, decoder, logits, config, override_cot_ids)
    if int(latent_embeds.shape[0]) < 1 or len(cot_ids) < 1:
        raise ValueError(
            f"degenerate generation: {int(latent_embeds.shape[0])} latents and {len(cot_ids)} CoT "
            "tokens. Padding the spans to 1 would misalign every blocked key; the sample is "
            "malformed and V1 must count it as such."
        )

    spans = spans_lib.build_spans(
        prefix_len=prompt_len - len(model.latent_token_ids[0]),
        bot_len=len(model.latent_token_ids[0]),
        latent_len=int(latent_embeds.shape[0]),
        eot_len=len(eot_ids),
        cot_open_len=1,
        cot_content_len=len(cot_ids),
        cot_close_len=1,
        answer_len=1,
    )
    blocked = _generation_blocked_keys(spans, cot_open_pos, decoder.kv_len, config.bottleneck)

    logits = decoder.step(embed_ids(model, model.cot_token_ids[1]), blocked)
    answer_ids = _run_answer_phase(model, decoder, logits, config, blocked)

    tokenizer = model.tokenizer
    return DecodeResult(
        latent_ids=tuple(latent_ids),
        latent_embeds=latent_embeds,
        cot_ids=tuple(cot_ids),
        answer_ids=tuple(answer_ids),
        spans=spans,
        latent_key_span=(prompt_len, prompt_len + int(latent_embeds.shape[0])),
        cot_key_span=spans_lib.cot_key_span(spans),
        answer_text=tokenizer.decode(list(answer_ids), skip_special_tokens=True),
        cot_text=tokenizer.decode(list(cot_ids), skip_special_tokens=False),
        latent_text=tokenizer.decode(list(latent_ids), skip_special_tokens=False) if latent_ids else "",
    )


@dataclass(frozen=True)
class SequencePieces:
    """A completed dual-track sequence, ready for a teacher-forced re-run."""

    prefix_ids: Tuple[int, ...]
    bot_ids: Tuple[int, ...]
    latent_embeds: Tensor
    eot_ids: Tuple[int, ...]
    cot_open_ids: Tuple[int, ...]
    cot_ids: Tuple[int, ...]
    cot_close_ids: Tuple[int, ...]
    answer_ids: Tuple[int, ...]

    def spans(self) -> TrackSpans:
        return spans_lib.build_spans(
            prefix_len=len(self.prefix_ids),
            bot_len=len(self.bot_ids),
            latent_len=int(self.latent_embeds.shape[0]),
            eot_len=len(self.eot_ids),
            cot_open_len=len(self.cot_open_ids),
            cot_content_len=len(self.cot_ids),
            cot_close_len=len(self.cot_close_ids),
            answer_len=len(self.answer_ids),
        )


def build_teacher_forced_embeds(
    model: nn.Module, pieces: SequencePieces, cot_embeds: Optional[Tensor] = None
) -> Tensor:
    """Concatenate the full sequence, optionally with an external CoT-embedding leaf."""
    head = embed_ids(model, list(pieces.prefix_ids) + list(pieces.bot_ids))
    latent = pieces.latent_embeds.to(device=head.device, dtype=head.dtype).unsqueeze(0)
    middle = embed_ids(model, list(pieces.eot_ids) + list(pieces.cot_open_ids))
    body = cot_embeds if cot_embeds is not None else embed_ids(model, list(pieces.cot_ids))
    tail = embed_ids(model, list(pieces.cot_close_ids) + list(pieces.answer_ids))
    return torch.cat([head, latent, middle, body.to(dtype=head.dtype), tail], dim=1)


def teacher_forced_logits(
    model: nn.Module,
    pieces: SequencePieces,
    bottleneck: bool = True,
    cot_embeds: Optional[Tensor] = None,
) -> Tensor:
    """Re-run the completed sequence under the ACTUAL 4-D training mask."""
    spans = pieces.spans()
    embeds = build_teacher_forced_embeds(model, pieces, cot_embeds)
    total = spans.total_len
    if int(embeds.shape[1]) != total:
        raise AssertionError(f"teacher-forced length {embeds.shape[1]} disagrees with spans {total}")
    key_spans, query_spans = mask_lib.spans_to_mask_inputs([spans], bottleneck=bottleneck)
    keep = mask_lib.build_bottleneck_mask([total], key_spans, query_spans, total, device=embeds.device)
    additive = mask_lib.to_additive(keep, embeds.dtype)
    return model(inputs_embeds=embeds, attention_mask=additive, use_cache=False).logits


class _ScriptedCausalLM(nn.Module):
    """CPU stub that records every attention mask it is handed."""

    def __init__(self, script: Sequence[int], hidden: int = 8, vocab: int = 40) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.vocab = vocab
        self.script = list(script)
        self.cursor = 0
        self.masks: List[Tensor] = []
        self.device = torch.device("cpu")

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> Any:
        from types import SimpleNamespace

        self.masks.append(attention_mask.clone())
        steps = int(inputs_embeds.shape[1])
        logits = torch.full((1, steps, self.vocab), -10.0)
        token = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        logits[0, -1, token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=(inputs_embeds,))


class _StubTokenizerForDecode:
    eos_token_id = 39

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return " ".join(str(int(i)) for i in ids)


def _build_stub(
    prompt_len: int,
    cot_tokens: Sequence[int] = (21, 22),
    cot_open_ids: Sequence[int] = (3,),
    cot_close_ids: Sequence[int] = (4,),
) -> _ScriptedCausalLM:
    """Scripted next-token stream matching the shifted-CE layout.

    The <eot> row emits <cot>, the <cot> row emits the first CoT token, the last
    CoT row emits </cot>, and the </cot> row emits the first answer token.
    """
    latent_tokens, eot_id = [11, 12, 13], 2
    answer_tokens = [31, _StubTokenizerForDecode.eos_token_id]
    script = (
        latent_tokens + [eot_id, cot_open_ids[0]] + list(cot_tokens)
        + [cot_close_ids[0]] + answer_tokens
    )
    model = _ScriptedCausalLM(script, vocab=64)
    model.tokenizer = _StubTokenizerForDecode()
    model.latent_token_ids = [[1], [eot_id]]
    model.cot_token_ids = [list(cot_open_ids), list(cot_close_ids)]
    return model


def _selftest_span_drift_guards() -> None:
    """A multi-piece delimiter or an empty CoT must fail loudly, never leak keys."""
    multi = _build_stub(5, cot_open_ids=(3, 30, 31))
    try:
        dualtrack_generate(multi, list(range(5)), DecodeConfig(topk_interpolation=3))
    except ValueError as exc:
        assert "tokenize to" in str(exc), exc
    else:
        raise AssertionError("a multi-piece <cot> was accepted and leaked 2 CoT keys")

    empty = _build_stub(5, cot_tokens=())
    try:
        dualtrack_generate(empty, list(range(5)), DecodeConfig(topk_interpolation=3))
    except ValueError as exc:
        assert "degenerate generation" in str(exc), exc
    else:
        raise AssertionError("an empty CoT was accepted and blocked the first answer key")


def _selftest_decode_bottleneck() -> None:
    prompt_len = 5
    model = _build_stub(prompt_len)
    result = dualtrack_generate(model, list(range(prompt_len)), DecodeConfig(topk_interpolation=3))
    assert result.latent_len == 3, result.latent_len
    assert result.cot_ids == (21, 22), result.cot_ids
    assert result.answer_ids == (31, _StubTokenizerForDecode.eos_token_id), result.answer_ids

    spans = result.spans
    training_keys = set(range(*spans_lib.cot_key_span(spans)))
    assert set(spans_lib.blocked_key_indices(spans)) == training_keys
    assert result.cot_key_span == spans_lib.cot_key_span(spans)

    # The </cot> step and every answer step must zero exactly the CoT keys.
    blocking_masks = [m for m in model.masks if (m == 0).any()]
    assert len(blocking_masks) == len(result.answer_ids), "an answer step forgot the bottleneck"
    for attention in blocking_masks:
        zeroed = set(torch.nonzero(attention[0] == 0).flatten().tolist())
        assert zeroed == training_keys, (sorted(zeroed), sorted(training_keys))


def _selftest_bottleneck_off() -> None:
    model = _build_stub(5)
    dualtrack_generate(model, list(range(5)), DecodeConfig(topk_interpolation=3, bottleneck=False))
    assert not any((m == 0).any() for m in model.masks), "bottleneck=False still blocked keys"


def _selftest_pieces_agree_with_training_spans() -> None:
    pieces = SequencePieces(
        prefix_ids=(1, 2, 3, 4),
        bot_ids=(5,),
        latent_embeds=torch.zeros(3, 8),
        eot_ids=(6, 7),
        cot_open_ids=(8,),
        cot_ids=(9, 10, 11),
        cot_close_ids=(12,),
        answer_ids=(13, 14),
    )
    spans = pieces.spans()
    assert spans.total_len == 4 + 1 + 3 + 2 + 1 + 3 + 1 + 2
    assert list(spans_lib.blocked_key_indices(spans)) == list(
        range(spans.cot_open_pos, spans.answer_start)
    )
    key_spans, query_spans = mask_lib.spans_to_mask_inputs([spans])
    keep = mask_lib.build_bottleneck_mask([spans.total_len], key_spans, query_spans, spans.total_len)
    q0, q1 = spans_lib.answer_query_span(spans)
    for key in spans_lib.blocked_key_indices(spans):
        assert keep[0, 0, q0:q1, key].sum().item() == 0


def selftest() -> None:
    _selftest_decode_bottleneck()
    _selftest_bottleneck_off()
    _selftest_pieces_agree_with_training_spans()
    _selftest_span_drift_guards()
    print(
        "[generate_dualtrack] OK -- three-phase decode, generation-time blocked keys equal the "
        "training cot_key_span on BOTH ends, multi-piece delimiters and empty CoTs are rejected, "
        "bottleneck=False leaves every key open."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track decoding (needs torch)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("generate_dualtrack.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
