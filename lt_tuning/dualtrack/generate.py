"""Bottleneck-respecting greedy decode.

Upstream's `generate` takes an `attention_mask` and ignores it (model.py:640), and its
incremental loop passes none at all (model.py:689-694) -- so the bottleneck would exist in
teacher forcing and evaporate at decode.  `DualTrackLTModel.generate` therefore refuses,
and this module decodes instead.  The latent-construction semantics stay upstream's:
prefill runs through `DualTrackLTModel.forward` (which runs upstream's forward), and each
`<thinking>` token's embedding comes from the inherited `_apply_transform` /
`_soft_fusion_embedding` via `make_latent`.

Invariant 5: every decode step builds its bias with `mask.build_segment_bias`, the same
predicate `SegmentBiasTape` slices at training time.

The delimiter is forced rather than hoped for: the mask cannot switch on until the answer
region starts, and free decoding gives no guarantee the model emits it.  The fraction of
samples that had to be forced is reported, never hidden.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from .config import DELIM_TEXT, GenConfig, add_bottleneck_mode_argument
from .mask import (
    DEFAULT_BOTTLENECK_MODE,
    BottleneckMode,
    ModeLike,
    TrackId,
    build_segment_bias,
    resolve_bottleneck_mode,
    spans_from_track,
)


@dataclass(frozen=True)
class GenResult:
    """One decoded sample.  `track_ids` is what the mask was actually built from."""

    token_ids: Tuple[int, ...]
    track_ids: Tuple[int, ...]
    q_end: int
    a_start: int
    v_start: int
    n_latents: int
    stop_reason: str
    hit_reasoning_cap: bool
    suppressed_thinking_steps: int
    answer_ids: Tuple[int, ...]

    @property
    def well_formed(self) -> bool:
        try:
            spans = spans_from_track(self.track_ids)
        except ValueError:
            return False
        return spans.n_latents >= 1 and self.v_start < len(self.token_ids)


class _DecodeState:
    """Running ids/tracks/cache for one sample.  Not exported."""

    def __init__(self, ids: Sequence[int], tracks: Sequence[int]) -> None:
        self.ids: List[int] = list(ids)
        self.tracks: List[int] = list(tracks)
        self.cache: Any = None
        self.hidden: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return len(self.ids)


def prompt_text_for(tokenizer: Any, question: str) -> str:
    """upstream `utils.apply_chat_template_if_needed` (utils.py:101-127), imported.

    The previous round kept a copy of it in `lt_dataset.py`; the prompt assembly has to be
    upstream's or the generated prefix does not match what training saw.
    """
    from ._upstream import import_core

    apply_chat_template_if_needed = import_core().utils.apply_chat_template_if_needed
    return apply_chat_template_if_needed(tokenizer, [{"role": "user", "content": question}])


def _bias_for(
    state: _DecodeState,
    q_start: int,
    q_end: int,
    dtype: torch.dtype,
    mask_on: bool,
    device: torch.device,
    mode: ModeLike,
) -> torch.Tensor:
    track = torch.tensor([state.tracks], dtype=torch.long, device=device)
    return build_segment_bias(track, q_start, q_end, dtype=dtype, mask_on=mask_on, mode=mode)


def _feed(
    model: Any,
    state: _DecodeState,
    embeds: torch.Tensor,
    new_tracks: Sequence[int],
    new_ids: Sequence[int],
    mask_on: bool,
    mode: ModeLike,
) -> None:
    """Append `new_ids` (already embedded) and advance the cache by one segment."""
    start = state.length
    state.ids.extend(int(i) for i in new_ids)
    state.tracks.extend(int(t) for t in new_tracks)
    device = embeds.device
    bias = _bias_for(state, start, state.length, embeds.dtype, mask_on, device, mode)
    position_ids = torch.arange(start, state.length, device=device).view(1, -1)
    outputs = model.injector.call(
        model.base_causallm, embeds, bias, position_ids, past_key_values=state.cache
    )
    state.cache = outputs.past_key_values
    state.logits = outputs.logits[0, -1, :]
    state.hidden = model._select_hidden_state(outputs.hidden_states)[0, -1, :]


def _next_token(logits: torch.Tensor, suppress: Sequence[int]) -> int:
    if not suppress:
        return int(torch.argmax(logits).item())
    filtered = logits.clone()
    for token_id in suppress:
        filtered[int(token_id)] = float("-inf")
    return int(torch.argmax(filtered).item())


def _embed_token(model: Any, token_id: int, state: _DecodeState) -> torch.Tensor:
    """A `<thinking>` token's embedding is upstream's latent (model.py:671-683)."""
    if token_id == model.thinking_token_id:
        return model.make_latent(state.hidden, state.logits).view(1, 1, -1)
    ids = torch.tensor([token_id], device=model.device)
    return model.embedding(ids).view(1, 1, -1)


def _prefill(model: Any, prompt_ids: Sequence[int], mask_on: bool, mode: ModeLike) -> _DecodeState:
    """Through `DualTrackLTModel.forward`, i.e. through upstream's own forward.

    The prompt carries no latents, so upstream runs exactly one segment and
    `last_hidden_state[:, -1]` is the final prompt row (upstream gives the LAST SEGMENT
    only, model.py:599 -- indexing it this way is valid here and nowhere else).
    """
    state = _DecodeState(prompt_ids, [int(TrackId.PROMPT)] * len(prompt_ids))
    ids = torch.tensor([list(prompt_ids)], device=model.device)
    track = torch.tensor([state.tracks], device=model.device)
    with torch.no_grad():
        outputs = model(ids, track, attention_mask=torch.ones_like(ids), mask_on=mask_on, mode=mode)
    state.cache = outputs.past_key_values
    state.logits = outputs.logits[0, -1, :]
    state.hidden = outputs.last_hidden_state[0, -1, :]
    return state


def _reasoning_phase(
    model: Any, state: _DecodeState, delim_first: int, cfg: GenConfig, mask_on: bool, mode: ModeLike
) -> Tuple[str, int]:
    """Returns why the chain stopped.  `eos` and `delimiter` are NOT the same event: a model
    that gives up after the prompt also stops, and reporting both as "not capped" would make
    V1's `frac_delimiter_forced_by_cap` look good for the worst possible reason."""
    n_latents = 0
    for _ in range(cfg.max_reasoning_tokens):
        token = _next_token(state.logits, ())
        if token == model.eos_token_id:
            return "eos", n_latents
        if token == delim_first:
            return "delimiter", n_latents
        track = int(TrackId.LATENT) if token == model.thinking_token_id else int(TrackId.COT)
        n_latents += token == model.thinking_token_id
        with torch.no_grad():
            embeds = _embed_token(model, token, state)
            _feed(model, state, embeds, [track], [token], mask_on, mode)
    return "cap", n_latents


def _force_delimiter(
    model: Any, state: _DecodeState, delim_ids: Sequence[int], mask_on: bool, mode: ModeLike
) -> None:
    with torch.no_grad():
        ids = torch.tensor([list(delim_ids)], device=model.device)
        embeds = model.embedding(ids)
        _feed(model, state, embeds, [int(TrackId.DELIM)] * len(delim_ids), delim_ids, mask_on, mode)


def _answer_phase(
    model: Any, state: _DecodeState, cfg: GenConfig, mask_on: bool, mode: ModeLike
) -> Tuple[List[int], int]:
    suppress = (model.thinking_token_id,) if cfg.suppress_thinking_in_answer else ()
    answer: List[int] = []
    suppressed = 0
    for _ in range(cfg.max_answer_tokens):
        raw = int(torch.argmax(state.logits).item())
        token = _next_token(state.logits, suppress)
        suppressed += raw == model.thinking_token_id and bool(suppress)
        answer.append(token)
        with torch.no_grad():
            embeds = _embed_token(model, token, state)
            _feed(model, state, embeds, [int(TrackId.ANSWER)], [token], mask_on, mode)
        if token == model.eos_token_id:
            break
    return answer, suppressed


def generate_dualtrack(
    model: Any,
    tokenizer: Any,
    question: str,
    cfg: GenConfig,
    mask_on: bool = True,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
) -> GenResult:
    """Reason freely, force the delimiter, then decode the answer under the bottleneck."""
    prompt_ids = tokenizer.encode(prompt_text_for(tokenizer, question), add_special_tokens=False)
    delim_ids = tokenizer.encode(DELIM_TEXT, add_special_tokens=False)
    if not delim_ids:
        raise ValueError("the delimiter tokenized to nothing")
    state = _prefill(model, prompt_ids, mask_on, mode)
    q_end = state.length
    stop_reason, n_latents = _reasoning_phase(model, state, delim_ids[0], cfg, mask_on, mode)
    a_start = state.length
    _force_delimiter(model, state, delim_ids, mask_on, mode)
    v_start = state.length
    answer, suppressed = _answer_phase(model, state, cfg, mask_on, mode)
    return GenResult(
        token_ids=tuple(state.ids),
        track_ids=tuple(state.tracks),
        q_end=q_end,
        a_start=a_start,
        v_start=v_start,
        n_latents=n_latents,
        stop_reason=stop_reason,
        hit_reasoning_cap=stop_reason == "cap",
        suppressed_thinking_steps=suppressed,
        answer_ids=tuple(answer),
    )


def decode_answer_text(tokenizer: Any, result: GenResult) -> str:
    return tokenizer.decode(list(result.answer_ids), skip_special_tokens=True)


# --- selftests -----------------------------------------------------------------------

# Chosen so the fixture decodes an interleaved chain (8 visible-CoT tokens, 4 latents).
# Below ~7 the random model never emits <thinking>; above ~11 it emits nothing else.
LATENT_EMISSION_BOOST = 8.0


def build_selftest_fixture() -> Tuple[Any, Any, int]:
    """Tiny locally-built Llama + stub tokenizer.  Shared with verify.py."""
    from .model import build_selftest_model
    from .stub_tokenizer import StubTokenizer

    tokenizer = StubTokenizer()
    tokenizer.encode("## Step 1: 2 + 4 = 6\nThe final answer is:\n### 6")
    tokenizer.encode("How many boxes did he buy in total?")
    thinking_id = tokenizer.convert_tokens_to_ids("<thinking>")
    tokenizer.freeze()
    model = build_selftest_model(len(tokenizer), thinking_id, tokenizer.eos_token_id)
    # A random model latches onto whichever token wins the first argmax and never emits
    # <thinking> at all.  Scaling that lm_head row makes this fixture emit an INTERLEAVED
    # chain, which is the geometry the bottleneck is about and the only geometry in which
    # COT -> LATENT -> ANSWER exists.  It changes the fixture, never the decode logic.
    with torch.no_grad():
        model.lm_head.weight.data[thinking_id] *= LATENT_EMISSION_BOOST
    return model, tokenizer, thinking_id


def _selftest_structure(mode: ModeLike = DEFAULT_BOTTLENECK_MODE) -> Tuple[Any, Any, GenResult]:
    model, tokenizer, _ = build_selftest_fixture()
    cfg = GenConfig(max_reasoning_tokens=12, max_answer_tokens=4)
    result = generate_dualtrack(
        model, tokenizer, "How many boxes did he buy in total?", cfg, mask_on=True, mode=mode
    )
    spans = spans_from_track(result.track_ids)
    assert spans.n_latents >= 1 and spans.n_cot >= 2, "fixture must interleave latents and text"
    assert (spans.q_end, spans.a_start, spans.v_start) == (
        result.q_end,
        result.a_start,
        result.v_start,
    )
    assert all(t == int(TrackId.ANSWER) for t in result.track_ids[result.v_start :])
    assert int(TrackId.LATENT) not in result.track_ids[result.v_start :]
    print(
        f"  [{resolve_bottleneck_mode(mode).value}] V1-shaped output: Q={result.q_end} "
        f"A={result.a_start} V={result.v_start} latents={spans.n_latents} "
        f"visible-CoT={spans.n_cot} stop={result.stop_reason}: OK"
    )
    return model, tokenizer, result


def _selftest_replay(model: Any, result: GenResult, mode: ModeLike) -> None:
    """The decisive engineering check: the incremental decode and a teacher-forced forward
    over the same sequence use the same mask predicate and agree."""
    ids = torch.tensor([list(result.token_ids)], device=model.device)
    track = torch.tensor([list(result.track_ids)], device=model.device)
    with torch.no_grad():
        replay = model(ids, track, attention_mask=torch.ones_like(ids), mask_on=True, mode=mode)
    rows = list(range(result.q_end - 1, len(result.token_ids) - 1))
    emitted = [result.token_ids[r + 1] for r in rows]
    argmax = replay.logits[0, rows].argmax(dim=-1).tolist()
    forced = set(range(result.a_start - 1, result.v_start - 1))
    free = [i for i, row in enumerate(rows) if row not in forced]
    mismatch = [i for i in free if argmax[i] != emitted[i]]
    assert not mismatch, f"teacher-forced replay disagrees with the decode at rows {mismatch}"
    print(f"  incremental decode == teacher-forced replay on {len(free)} free rows: OK")


def _cot_scramble_delta(model: Any, result: GenResult, mask_on: bool, mode: ModeLike) -> float:
    """Pin the latents to their clean values, reverse the visible CoT, read the answer rows."""
    spans = spans_from_track(result.track_ids)
    if len(spans.cot_positions) < 2:
        raise AssertionError("fixture produced no visible CoT to perturb")
    ids = torch.tensor([list(result.token_ids)], device=model.device)
    track = torch.tensor([list(result.track_ids)], device=model.device)
    scrambled = ids.clone()
    cot = list(spans.cot_positions)
    scrambled[0, cot] = ids[0, list(reversed(cot))]
    attention = torch.ones_like(ids)
    with torch.no_grad():
        pinned = model(ids, track, attention_mask=attention, mask_on=mask_on, mode=mode).latents
        clean = model(
            ids, track, attention_mask=attention, mask_on=mask_on, mode=mode, latent_override=pinned
        )
        dirty = model(
            scrambled,
            track,
            attention_mask=attention,
            mask_on=mask_on,
            mode=mode,
            latent_override=pinned,
        )
    return (
        (clean.logits[0, result.a_start :] - dirty.logits[0, result.a_start :]).abs().max().item()
    )


def _selftest_mask_reaches_generation(model: Any, result: GenResult) -> None:
    """The headline finding, on the DECODED sequence and with the latents pinned.

    With the latent input embeddings held fixed the fusion channel is removed, so what is
    left is exactly the attention channel: NATIVE still leaks (a latent row is an ordinary
    query row and reads the CoT), STRICT is bit-exactly zero, mask-off is largest.
    """
    deltas = {mode.value: _cot_scramble_delta(model, result, True, mode) for mode in BottleneckMode}
    deltas["mask_off"] = _cot_scramble_delta(model, result, False, DEFAULT_BOTTLENECK_MODE)
    assert (
        deltas[BottleneckMode.STRICT.value] == 0.0
    ), f"STRICT must close the attention channel bit-exactly; got {deltas}"
    assert deltas[BottleneckMode.NATIVE.value] > 0.0, (
        "NATIVE must still leak through COT -> LATENT -> ANSWER, or the fixture has no "
        f"two-hop path and the finding is untested here: {deltas}"
    )
    assert deltas["mask_off"] > deltas[BottleneckMode.NATIVE.value], deltas
    rendered = " ".join(f"{k}={v:.3e}" for k, v in deltas.items())
    print(f"  bottleneck on the generated sequence (latents pinned): {rendered}: OK")


def selftest(mode: ModeLike = DEFAULT_BOTTLENECK_MODE) -> None:
    """CPU-only, no network, stub tokenizer, tiny locally-built Llama."""
    model, _, result = _selftest_structure(mode)
    _selftest_replay(model, result, mode)
    _selftest_mask_reaches_generation(model, result)
    print("generate.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bottleneck-respecting decode (LT-Tuning)")
    parser.add_argument("--selftest", action="store_true")
    add_bottleneck_mode_argument(parser)
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest (real decoding runs through verify.py)")
    selftest(args.bottleneck_mode)


if __name__ == "__main__":
    main()
