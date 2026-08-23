"""Bottleneck mask construction and span arithmetic for LT-Tuning dual-track.

Pure python, no upstream import: upstream has no mask predicate to subclass, so this is
new work rather than a shadow of anything.  It is the ONE mask predicate both training
(`model.SegmentBiasTape`) and generation (`generate.py`) call, which is what makes those
two provably identical.

Two bottleneck modes, because one attention rule does not describe one leak channel:

* NATIVE blocks only the direct edge visible-CoT -> delimiter/answer.  This is
  LT-Tuning's own geometry and stays the default.  It does **not** stop the content
  arriving at the answer: from two layers onward it travels COT -> LATENT -> ANSWER,
  because a latent row is an ordinary query that reads the chain it summarises.
* STRICT additionally blocks latent query rows from visible-CoT keys, which closes
  that two-hop path.  It is a diagnostic instrument, not a better default: a latent
  that cannot attend the explicit reasoning is no longer LT-Tuning's latent.

Neither mode closes the *fusion-initialisation* channel (a latent's input embedding
is built from the hidden state of the immediately preceding chain token).  That is a
residual-stream edge; no attention mask can reach it.  See README.md.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Sequence, Set, Tuple, Union

try:  # torch is optional: the pure-python reference predicate needs no torch.
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on torch-free machines
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class TrackId(IntEnum):
    """Per-position role.  The single authority for both the mask and the loss."""

    PAD = 0
    PROMPT = 1
    COT = 2
    LATENT = 3
    DELIM = 4
    ANSWER = 5


class BottleneckMode(str, Enum):
    """Which query rows the mask cuts off from the visible-CoT keys."""

    NATIVE = "native"
    STRICT = "strict"


BLOCKED_KEY_TRACKS: Tuple[int, ...] = (int(TrackId.COT),)
COT_LOSS_TRACKS: Tuple[int, ...] = (int(TrackId.COT), int(TrackId.LATENT), int(TrackId.DELIM))
ANSWER_LOSS_TRACKS: Tuple[int, ...] = (int(TrackId.ANSWER),)

DEFAULT_BOTTLENECK_MODE = BottleneckMode.NATIVE

# There is deliberately no module-level BLOCKED_QUERY_TRACKS constant.  One existed and
# was read directly by `chokepoint_holds` and by the selftest reference, so a caller that
# overrode the tracks got a torch bias and a pure-python reference that disagreed without
# saying so.  The mode is now the only way to name a query set, and it is a parameter
# everywhere, so no call site can silently fall back to a global.
_BLOCKED_QUERY_TRACKS_BY_MODE: Dict[BottleneckMode, Tuple[int, ...]] = {
    BottleneckMode.NATIVE: (int(TrackId.DELIM), int(TrackId.ANSWER)),
    BottleneckMode.STRICT: (int(TrackId.LATENT), int(TrackId.DELIM), int(TrackId.ANSWER)),
}

ModeLike = Union[BottleneckMode, str]


def resolve_bottleneck_mode(mode: ModeLike = DEFAULT_BOTTLENECK_MODE) -> BottleneckMode:
    """Accept the enum or its string spelling; refuse anything else by name."""
    if isinstance(mode, BottleneckMode):
        return mode
    try:
        return BottleneckMode(str(mode).lower())
    except ValueError as exc:
        known = ", ".join(m.value for m in BottleneckMode)
        raise ValueError(f"unknown bottleneck mode {mode!r}; expected one of: {known}") from exc


def blocked_query_tracks(mode: ModeLike = DEFAULT_BOTTLENECK_MODE) -> Tuple[int, ...]:
    """The single resolver: mode -> the query tracks cut off from `BLOCKED_KEY_TRACKS`."""
    return _BLOCKED_QUERY_TRACKS_BY_MODE[resolve_bottleneck_mode(mode)]


class TrackLayoutError(ValueError):
    """Raised when a track row does not match the documented sequence layout."""


@dataclass(frozen=True)
class Spans:
    """Post-insertion span arithmetic recovered from a single track row."""

    pad_left: int
    q_end: int
    a_start: int
    v_start: int
    n_end: int
    latent_positions: Tuple[int, ...]
    cot_positions: Tuple[int, ...]

    @property
    def n_latents(self) -> int:
        return len(self.latent_positions)

    @property
    def n_cot(self) -> int:
        return len(self.cot_positions)


_LAYOUT = (
    (int(TrackId.PAD),),
    (int(TrackId.PROMPT),),
    (int(TrackId.COT), int(TrackId.LATENT)),
    (int(TrackId.DELIM),),
    (int(TrackId.ANSWER),),
    (int(TrackId.PAD),),
)


def spans_from_track(track_row: Sequence[int]) -> Spans:
    """Validate `PAD* PROMPT+ (COT|LATENT)* DELIM+ ANSWER+ PAD*` and return its spans."""
    stage = 0
    bounds: List[int] = [0] * (len(_LAYOUT) + 1)
    for pos, raw in enumerate(track_row):
        value = int(raw)
        while stage < len(_LAYOUT) and value not in _LAYOUT[stage]:
            stage += 1
            bounds[stage] = pos
        if stage >= len(_LAYOUT):
            raise TrackLayoutError(f"track value {value} at position {pos} breaks the layout order")
    for stage in range(stage + 1, len(_LAYOUT) + 1):
        bounds[stage] = len(track_row)
    pad_left, q_end, a_start, v_start, n_end = bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
    if not pad_left < q_end <= a_start < v_start < n_end:
        raise TrackLayoutError(f"degenerate spans: {(pad_left, q_end, a_start, v_start, n_end)}")
    reasoning = range(q_end, a_start)
    latents = tuple(p for p in reasoning if int(track_row[p]) == int(TrackId.LATENT))
    cots = tuple(p for p in reasoning if int(track_row[p]) == int(TrackId.COT))
    if len(latents) + len(cots) != a_start - q_end:
        raise TrackLayoutError("reasoning span is not partitioned into COT and LATENT")
    return Spans(pad_left, q_end, a_start, v_start, n_end, latents, cots)


def is_allowed(
    track_row: Sequence[int],
    q: int,
    k: int,
    mask_on: bool = True,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
    blocked_key_tracks: Sequence[int] = BLOCKED_KEY_TRACKS,
) -> bool:
    """Reference predicate.  Pad rows get a forced diagonal *instead of*, never in
    addition to, the normal rule -- an unconditional `keep |= eye` would re-open a
    blocked key to itself."""
    if int(track_row[q]) == int(TrackId.PAD):
        return k == q
    if int(track_row[k]) == int(TrackId.PAD):
        return False
    if k > q:
        return False
    if not mask_on:
        return True
    blocked_q = blocked_query_tracks(mode)
    return not (int(track_row[q]) in blocked_q and int(track_row[k]) in blocked_key_tracks)


def reference_allow(
    track_row: Sequence[int],
    mask_on: bool = True,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
    blocked_key_tracks: Sequence[int] = BLOCKED_KEY_TRACKS,
) -> List[List[bool]]:
    """Dense boolean allow matrix from the reference predicate."""
    n = len(track_row)
    return [
        [is_allowed(track_row, q, k, mask_on, mode, blocked_key_tracks) for k in range(n)]
        for q in range(n)
    ]


def chokepoint_holds(
    track_row: Sequence[int], mask_on: bool = True, mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
) -> bool:
    """Every path from a COT key to a blocked query row passes through a LATENT.

    Reachability over `allow` edges with LATENT nodes non-expandable.  This is the
    *only* structural claim this platform supports; independence from the visible
    CoT is false here and is never asserted.  Under STRICT the latent rows are
    themselves blocked queries, so the claim degenerates to "no path at all" -- which
    is why STRICT is a diagnostic and not a stronger version of the same experiment.
    """
    allow = reference_allow(track_row, mask_on, mode)
    n = len(track_row)
    sources = [q for q in range(n) if int(track_row[q]) in blocked_query_tracks(mode)]
    seen: Set[int] = set(sources)
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for key in range(n):
            if not allow[node][key] or key in seen:
                continue
            if int(track_row[key]) == int(TrackId.COT):
                return False
            seen.add(key)
            if int(track_row[key]) != int(TrackId.LATENT):
                queue.append(key)
    return True


def segment_boundaries(
    track_ids: Sequence[Sequence[int]], split_after_latents: bool = False,
) -> List[int]:
    """Upstream segmentation (model.py:414-420): latent positions across the batch.

    `split_after_latents` additionally closes a segment right after each latent so a
    latent's K/V can be frozen before any later row reads it.
    """
    if len(track_ids) == 0:
        raise ValueError("track_ids must have at least one row")
    seq_len = len(track_ids[0])
    boundaries: Set[int] = set()
    for row in track_ids:
        if len(row) != seq_len:
            raise ValueError("all track rows must share a length")
        for pos, value in enumerate(row):
            if int(value) != int(TrackId.LATENT) or not 0 < pos < seq_len:
                continue
            boundaries.add(pos)
            if split_after_latents and pos + 1 < seq_len:
                boundaries.add(pos + 1)
    return sorted(boundaries) + [seq_len]


def latent_positions(track_ids: Sequence[Sequence[int]]) -> List[List[int]]:
    """Per-row LATENT positions."""
    return [[p for p, v in enumerate(row) if int(v) == int(TrackId.LATENT)] for row in track_ids]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is required for tensor mask builders; use the reference predicate instead"
        )


def _membership(track_ids: "torch.Tensor", values: Sequence[int]) -> "torch.Tensor":
    out = torch.zeros_like(track_ids, dtype=torch.bool)
    for value in values:
        out = out | (track_ids == int(value))
    return out


def build_full_bias(
    track_ids: "torch.Tensor",
    dtype: "torch.dtype",
    mask_on: bool = True,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
    blocked_key_tracks: Sequence[int] = BLOCKED_KEY_TRACKS,
) -> "torch.Tensor":
    """Additive attention bias of shape (B, 1, N, N).  Uses finfo(dtype).min, not -inf."""
    _require_torch()
    if track_ids.dim() != 2:
        raise ValueError(f"track_ids must be (B, N); got {tuple(track_ids.shape)}")
    batch, seq_len = track_ids.shape
    device = track_ids.device
    valid = track_ids != int(TrackId.PAD)
    positions = torch.arange(seq_len, device=device)
    causal = positions.view(1, seq_len, 1) >= positions.view(1, 1, seq_len)
    allow = valid.unsqueeze(1) & causal
    if mask_on:
        blocked_k = _membership(track_ids, blocked_key_tracks).unsqueeze(1)
        blocked_q = _membership(track_ids, blocked_query_tracks(mode)).unsqueeze(2)
        allow = allow & ~(blocked_q & blocked_k)
    diagonal = (
        torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0).expand(batch, -1, -1)
    )
    allow = torch.where((~valid).unsqueeze(2), diagonal, allow)
    bias = torch.zeros((batch, seq_len, seq_len), dtype=dtype, device=device)
    return bias.masked_fill(~allow, torch.finfo(dtype).min).unsqueeze(1)


def build_segment_bias(
    track_ids: "torch.Tensor",
    q_start: int,
    q_end: int,
    dtype: "torch.dtype",
    mask_on: bool = True,
    mode: ModeLike = DEFAULT_BOTTLENECK_MODE,
    blocked_key_tracks: Sequence[int] = BLOCKED_KEY_TRACKS,
) -> "torch.Tensor":
    """Per-segment slice `bias[:, :, q_start:q_end, :q_end]`.

    Both upstream call sites (model.py:474-491) have kv_len == boundary_end, so one
    uniform rule covers the cached and uncached branches.  Training does not call this
    directly: `model.SegmentBiasTape` hands upstream's own segment loop the same slice of
    one `build_full_bias` tensor.  Generation calls it per decode step, which is what makes
    the two paths the same predicate rather than two predicates that agree today.

    `blocked_key_tracks` is forwarded, not defaulted.  It used to be absent here while
    `build_full_bias` and `reference_allow` both accepted it -- the same silent-divergence
    shape as the old query-track global, and worse placed, because this is the only builder
    `lt_model.forward` and `generate_dualtrack` call.  A caller who overrode the key tracks
    got one mask from the reference and a different one from the model, with no error.
    `_selftest_key_tracks_are_threaded` fails if this parameter is dropped again.
    """
    _require_torch()
    if not 0 <= q_start < q_end <= track_ids.shape[1]:
        raise ValueError(f"bad segment [{q_start}, {q_end}) for length {track_ids.shape[1]}")
    prefix = track_ids[:, :q_end]
    full = build_full_bias(
        prefix, dtype=dtype, mask_on=mask_on, mode=mode, blocked_key_tracks=blocked_key_tracks
    )
    return full[:, :, q_start:q_end, :q_end]


def _random_track_row(rng: random.Random) -> List[int]:
    n_prompt = rng.randint(1, 5)
    n_cot = rng.randint(1, 8)
    n_latent = rng.randint(0, 3)
    n_delim = rng.randint(1, 2)
    n_answer = rng.randint(1, 3)
    reasoning = [int(TrackId.COT)] * n_cot
    for _ in range(n_latent):
        reasoning.insert(rng.randint(1, len(reasoning)), int(TrackId.LATENT))
    reasoning.append(int(TrackId.LATENT))
    return (
        [int(TrackId.PROMPT)] * n_prompt
        + reasoning
        + [int(TrackId.DELIM)] * n_delim
        + [int(TrackId.ANSWER)] * n_answer
    )


def random_track_batch(rng: random.Random, batch: int) -> List[List[int]]:
    """Padded batch of random fixtures: left pad + right pad, as the collator produces."""
    rows = [_random_track_row(rng) for _ in range(batch)]
    left = [rng.randint(0, 3) for _ in rows]
    width = max(l + len(r) for l, r in zip(left, rows))
    return [
        [int(TrackId.PAD)] * l + r + [int(TrackId.PAD)] * (width - l - len(r))
        for l, r in zip(left, rows)
    ]


# --- multi-layer leak probe ---------------------------------------------------------
# P=PROMPT C=COT L=LATENT D=DELIM A=ANSWER.  Interleaved on purpose: this is the geometry
# LT-Tuning actually produces, and it is what makes the two-hop path exist.
_P, _C, _L, _D, _A = (
    int(TrackId.PROMPT),
    int(TrackId.COT),
    int(TrackId.LATENT),
    int(TrackId.DELIM),
    int(TrackId.ANSWER),
)
_INTERLEAVED_FIXTURES: Tuple[Tuple[int, ...], ...] = (
    # NOTE: this first row has no DELIM, so `spans_from_track` rejects it as degenerate --
    # it is not a geometry the pipeline can emit.  It is kept because the mask builders read
    # only track *types* (they never call `spans_from_track`), so it is a valid probe of the
    # attention graph, and it is the row the finding was originally measured on.  The claim
    # does not rest on it: the two rows below are layout-valid and show the same thing.
    (_P, _P, _P, _C, _C, _L, _C, _C, _L, _C, _L, _C, _C, _A, _A, _A),
    (_P, _P, _C, _C, _C, _L, _C, _L, _C, _C, _L, _D, _A, _A),
    (_P, _C, _L, _C, _C, _L, _C, _C, _C, _L, _D, _D, _A, _A, _A, _A),
)
_PROBE_DIM = 8
_PROBE_SEED = 20260807


@dataclass(frozen=True)
class DepthProbe:
    """How far a visible-CoT perturbation travels in a stack of `layers` layers."""

    mode: BottleneckMode
    layers: int
    answer_delta: float
    latent_delta: float


def _toy_stack_weights(
    dim: int, layers: int, seed: int
) -> List[Tuple["torch.Tensor", "torch.Tensor"]]:
    torch.manual_seed(seed)
    scale = dim ** 0.5
    return [(torch.randn(dim, dim) / scale, torch.randn(dim, dim) / scale) for _ in range(layers)]


def _run_toy_stack(
    hidden: "torch.Tensor",
    bias: "torch.Tensor",
    weights: Sequence[Tuple["torch.Tensor", "torch.Tensor"]],
) -> "torch.Tensor":
    """Single-head SDPA + tanh MLP with residuals, no normalisation.

    Deliberately the dullest stack that can express a two-hop path: the claim under test
    is about the mask graph, not about any particular layer.  Note what this toy does
    *not* have -- a latent's input is a plain hidden state here, so the fusion channel
    (README) is absent by construction and every number below is attention-only.
    """
    state = hidden
    for w_in, w_out in weights:
        query = state.unsqueeze(1)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, query, query, attn_mask=bias
        ).squeeze(1)
        state = state + attended
        state = state + torch.tanh(state @ w_in) @ w_out
    return state


def _track_max_delta(delta: "torch.Tensor", row: Sequence[int], track: TrackId) -> float:
    rows = [i for i, value in enumerate(row) if int(value) == int(track)]
    return float(delta[0, rows].max()) if rows else 0.0


def _depth_probe(row: Sequence[int], layers: int, mode: ModeLike) -> DepthProbe:
    """Perturb only the COT positions; read the change at the ANSWER and LATENT rows."""
    _require_torch()
    track = torch.tensor([list(row)], dtype=torch.long)
    bias = build_full_bias(track, dtype=torch.float32, mode=mode)
    weights = _toy_stack_weights(_PROBE_DIM, layers, _PROBE_SEED)
    torch.manual_seed(_PROBE_SEED + 1)
    clean = torch.randn(1, len(row), _PROBE_DIM)
    dirty = clean.clone()
    cot = [i for i, value in enumerate(row) if int(value) == int(TrackId.COT)]
    dirty[0, cot] = torch.randn(len(cot), _PROBE_DIM)
    delta = (_run_toy_stack(clean, bias, weights) - _run_toy_stack(dirty, bias, weights)).abs()
    return DepthProbe(
        mode=resolve_bottleneck_mode(mode),
        layers=layers,
        answer_delta=_track_max_delta(delta, row, TrackId.ANSWER),
        latent_delta=_track_max_delta(delta, row, TrackId.LATENT),
    )


def _selftest_answer_reads_every_latent() -> None:
    """Both modes keep the thing the bottleneck is *for*: the answer reads all latents."""
    for row in _INTERLEAVED_FIXTURES:
        for mode in BottleneckMode:
            allow = reference_allow(row, mask_on=True, mode=mode)
            pairs = [
                (q, k)
                for q, q_track in enumerate(row)
                for k, k_track in enumerate(row)
                if int(q_track) == _A and int(k_track) == _L and k <= q
            ]
            assert pairs, "fixture has no answer/latent pair to check"
            missing = [(q, k) for q, k in pairs if not allow[q][k]]
            assert not missing, f"mode={mode.value} closed answer->latent edges {missing}"
    print(
        f"  answer rows read every latent key under both modes: OK ({len(_INTERLEAVED_FIXTURES)} fixtures)"
    )


def _selftest_multi_layer_leak() -> None:
    """A one-layer check passes in BOTH modes, which is exactly how this stayed hidden.

    NATIVE blocks the direct edge COT -> ANSWER, so at depth 1 nothing reaches the answer.
    From depth 2 the content arrives anyway, via COT -> LATENT -> ANSWER.  STRICT blocks
    the latent query rows too, so the answer rows stay bit-exactly unchanged at any depth.

    Polarity is deliberately asymmetric.  The zero side is exact and is asserted per
    fixture: closing the path is a structural fact, and it held on every one of 2700
    fixture x depth x seed draws tried.  The nonzero side is only asserted in aggregate,
    because a magnitude measured through random weights can vanish even where the path is
    open -- softmax saturates, and on ~1% of weight draws one fixture's answer rows come
    out bit-identical under NATIVE.  A per-fixture `min(deltas) > 0` here would be a test
    that passes because of the seed.  `_selftest_two_hop_path_exists` carries the
    per-fixture claim instead, with no weights in it at all.
    """
    _require_torch()
    for layers in (1, 2, 6):
        for mode in BottleneckMode:
            probes = [_depth_probe(row, layers, mode) for row in _INTERLEAVED_FIXTURES]
            deltas = [p.answer_delta for p in probes]
            print(
                f"    layers={layers} {mode.value:6s}: delta at ANSWER rows = "
                f"{[round(d, 4) for d in deltas]}  (LATENT rows: "
                f"{[round(p.latent_delta, 2) for p in probes]})"
            )
            if mode is BottleneckMode.STRICT or layers == 1:
                assert deltas == [0.0] * len(deltas), (
                    f"mode={mode.value} layers={layers} must leave the answer rows "
                    f"bit-exactly unchanged; got {deltas}"
                )
            else:
                assert max(deltas) > 0.0, (
                    f"NATIVE at {layers} layers must move the answer rows through the "
                    f"latents; got {deltas}.  A single-layer check would have missed this."
                )
    print("  multi-layer leak table (1/2/6 layers x both modes): OK")


def _selftest_reference(rng: random.Random) -> None:
    for _ in range(200):
        row = _random_track_row(rng)
        spans = spans_from_track(row)
        assert spans.n_cot + spans.n_latents == spans.a_start - spans.q_end
        assert spans.n_latents >= 1, "a terminal latent is mandatory (spec H.5)"
        assert row[spans.a_start - 1] == int(TrackId.LATENT)
        for mode in BottleneckMode:
            assert chokepoint_holds(row, mask_on=True, mode=mode)
            assert not chokepoint_holds(row, mask_on=False, mode=mode)
            allow = reference_allow(row, mask_on=True, mode=mode)
            blocked_q = blocked_query_tracks(mode)
            for q, _ in enumerate(row):
                for k, _ in enumerate(row):
                    if row[q] in blocked_q and row[k] in BLOCKED_KEY_TRACKS:
                        assert not allow[q][k]
                    if k > q:
                        assert not allow[q][k]
    print(f"  reference predicate + chokepoint: OK (200 fixtures x {len(BottleneckMode)} modes)")


def _selftest_no_global_eye() -> None:
    row = [
        int(TrackId.PROMPT),
        int(TrackId.COT),
        int(TrackId.LATENT),
        int(TrackId.DELIM),
        int(TrackId.ANSWER),
    ]
    allow = reference_allow(row, mask_on=True, blocked_key_tracks=(int(TrackId.DELIM),))
    assert not allow[3][3], "diagonal must not be forced on a non-pad blocked-key row"
    if _TORCH_AVAILABLE:
        bias = build_full_bias(
            torch.tensor([row], dtype=torch.long),
            dtype=torch.float32,
            blocked_key_tracks=(int(TrackId.DELIM),),
        )
        assert bias[0, 0, 3, 3].item() < 0.0
    print("  diagonal forced on pad rows only (no global eye): OK")


def _selftest_mode_resolver() -> None:
    """The resolver is the only place a mode name becomes a query set."""
    assert DEFAULT_BOTTLENECK_MODE is BottleneckMode.NATIVE, "NATIVE must stay the default"
    assert blocked_query_tracks() == (int(TrackId.DELIM), int(TrackId.ANSWER))
    assert blocked_query_tracks("strict") == (
        int(TrackId.LATENT),
        int(TrackId.DELIM),
        int(TrackId.ANSWER),
    )
    assert resolve_bottleneck_mode("NATIVE") is BottleneckMode.NATIVE
    assert set(_BLOCKED_QUERY_TRACKS_BY_MODE) == set(BottleneckMode), "every mode needs a query set"
    try:
        blocked_query_tracks("blocked")
    except ValueError as exc:
        assert "unknown bottleneck mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown mode name must raise, never fall back to a default")
    print(
        f"  mode resolver: {[m.value for m in BottleneckMode]}, default={DEFAULT_BOTTLENECK_MODE.value}: OK"
    )


def _selftest_mode_is_threaded(rng: random.Random) -> None:
    """Would FAIL if any path fell back to a global instead of the mode it was handed.

    The two halves are checked against each other under *every* mode, and STRICT is
    required to differ from NATIVE on a fixture that has a latent after a CoT token --
    so a builder that quietly ignored `mode` could not pass by returning NATIVE twice.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch missing: cannot check the tensor half against the reference")
    batches = [random_track_batch(rng, batch=4), [list(row) for row in _INTERLEAVED_FIXTURES[:1]]]
    n_rows = 0
    for rows in batches:
        track = torch.tensor(rows, dtype=torch.long)
        per_mode = {}
        for mode in BottleneckMode:
            bias = build_full_bias(track, dtype=torch.float32, mode=mode)
            for b, row in enumerate(rows):
                expected = reference_allow(row, mask_on=True, mode=mode)
                assert (
                    bias[b, 0] == 0.0
                ).tolist() == expected, (
                    f"torch builder and reference predicate disagree under mode={mode.value}"
                )
            start, end = 1, len(rows[0])
            segment = build_segment_bias(track, start, end, dtype=torch.float32, mode=mode)
            assert torch.equal(
                segment, bias[:, :, start:end, :end]
            ), f"build_segment_bias ignores mode={mode.value}"
            per_mode[mode] = bias
        assert not torch.equal(
            per_mode[BottleneckMode.NATIVE], per_mode[BottleneckMode.STRICT]
        ), "STRICT must actually change the bias, or `mode` is being dropped somewhere"
        n_rows += len(rows)
    print(
        f"  mode threaded through reference/full/segment, and NATIVE != STRICT: OK ({n_rows} rows)"
    )


def _selftest_key_tracks_are_threaded(rng: random.Random) -> None:
    """The key-side twin of `_selftest_mode_is_threaded`.

    `blocked_key_tracks` is an overridable parameter on the reference predicate and on
    `build_full_bias`, so the segment builder must honour it too or the two halves diverge
    exactly the way the query-track global used to.  A non-default key set is required to
    change the bias, so a builder that ignored the argument could not pass by accident.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch missing: cannot check the tensor half against the reference")
    override = (int(TrackId.LATENT),)
    for _ in range(20):
        rows = random_track_batch(rng, batch=rng.randint(1, 3))
        track = torch.tensor(rows, dtype=torch.long)
        for mode in BottleneckMode:
            full = build_full_bias(
                track, dtype=torch.float32, mode=mode, blocked_key_tracks=override
            )
            for b, row in enumerate(rows):
                expected = reference_allow(
                    row, mask_on=True, mode=mode, blocked_key_tracks=override
                )
                assert (
                    full[b, 0] == 0.0
                ).tolist() == expected, (
                    f"torch builder != reference under an overridden key set (mode={mode.value})"
                )
            default = build_full_bias(track, dtype=torch.float32, mode=mode)
            assert not torch.equal(full, default), "override must actually change the bias"
            for start, end in _segment_pairs(rows):
                seg = build_segment_bias(
                    track, start, end, dtype=torch.float32, mode=mode, blocked_key_tracks=override,
                )
                assert torch.equal(seg, full[:, :, start:end, :end]), (
                    "build_segment_bias drops blocked_key_tracks -- the model path and the "
                    "reference would disagree silently"
                )
    print("  blocked_key_tracks threaded through reference/full/segment: OK (20 batches x 2 modes)")


def _selftest_two_hop_path_exists() -> None:
    """The seed-independent form of the leak claim: it is about the graph, not the weights.

    `_selftest_multi_layer_leak` measures a *magnitude* through random weights, and a
    magnitude can vanish -- softmax saturates, and on roughly 1% of weight draws a fixture's
    answer rows come out bit-identical under NATIVE even though the path is wide open.  This
    check has no weights and no seed, so it is the one that actually pins the finding.
    """
    for i, row in enumerate(_INTERLEAVED_FIXTURES):
        counts = {}
        for mode in BottleneckMode:
            allow = reference_allow(row, mask_on=True, mode=mode)
            counts[mode] = sum(
                1
                for q, q_t in enumerate(row)
                if int(q_t) == _A
                for l, l_t in enumerate(row)
                if int(l_t) == _L and allow[q][l]
                for c, c_t in enumerate(row)
                if int(c_t) == _C and allow[l][c]
            )
        assert (
            counts[BottleneckMode.NATIVE] > 0
        ), f"fixture[{i}]: NATIVE must leave a COT -> LATENT -> ANSWER path open"
        assert counts[BottleneckMode.STRICT] == 0, (
            f"fixture[{i}]: STRICT must close every COT -> LATENT -> ANSWER path; "
            f"got {counts[BottleneckMode.STRICT]}"
        )
    valid = [i for i, row in enumerate(_INTERLEAVED_FIXTURES) if _is_layout_valid(row)]
    assert valid, "at least one fixture must be a geometry the pipeline can actually emit"
    print(
        f"  two-hop COT->LATENT->ANSWER: open under NATIVE, closed under STRICT, on all "
        f"{len(_INTERLEAVED_FIXTURES)} fixtures ({len(valid)} layout-valid): OK"
    )


def _is_layout_valid(row: Sequence[int]) -> bool:
    try:
        spans_from_track(row)
        return True
    except TrackLayoutError:
        return False


def _selftest_padding(rng: random.Random) -> None:
    for _ in range(50):
        rows = random_track_batch(rng, batch=rng.randint(1, 4))
        for row in rows:
            for mode in BottleneckMode:
                allow = reference_allow(row, mask_on=True, mode=mode)
                for q, value in enumerate(row):
                    if value != int(TrackId.PAD):
                        continue
                    assert sum(allow[q]) == 1 and allow[q][q], "pad rows need exactly a self-edge"
    print("  pad-row diagonal is exact: OK (50 padded batches, both modes)")


def _selftest_torch(rng: random.Random) -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch missing: cannot run the tensor half of the mask selftest")
    neg = torch.finfo(torch.float32).min
    cases = [(mask_on, mode) for mask_on in (True, False) for mode in BottleneckMode]
    for _ in range(60):
        rows = random_track_batch(rng, batch=rng.randint(1, 3))
        track = torch.tensor(rows, dtype=torch.long)
        for mask_on, mode in cases:
            bias = build_full_bias(track, dtype=torch.float32, mask_on=mask_on, mode=mode)
            for b, row in enumerate(rows):
                expected = reference_allow(row, mask_on=mask_on, mode=mode)
                got = (bias[b, 0] == 0.0).tolist()
                assert got == expected, "torch builder disagrees with the reference predicate"
                assert bool((bias[b, 0][~torch.tensor(expected)] == neg).all())
            for start, end in _segment_pairs(rows):
                seg = build_segment_bias(
                    track, start, end, dtype=torch.float32, mask_on=mask_on, mode=mode
                )
                assert torch.equal(
                    seg, bias[:, :, start:end, :end]
                ), "segment slice != full-bias slice"
    print(
        f"  torch bias == reference, segment slice == full slice: OK "
        f"(60 batches x {len(cases)} mask_on/mode cases)"
    )


def _segment_pairs(rows: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for split_after in (False, True):
        start = 0
        for end in segment_boundaries(rows, split_after_latents=split_after):
            if end > start:
                pairs.append((start, end))
                start = end
    return pairs


def selftest() -> None:
    """CPU-only, no network, no weights."""
    rng = random.Random(20260806)
    _selftest_mode_resolver()
    _selftest_reference(rng)
    _selftest_no_global_eye()
    _selftest_padding(rng)
    _selftest_torch(rng)
    _selftest_mode_is_threaded(rng)
    _selftest_key_tracks_are_threaded(rng)
    _selftest_answer_reads_every_latent()
    _selftest_two_hop_path_exists()
    _selftest_multi_layer_leak()
    print("mask.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-track bottleneck mask (LT-Tuning)")
    parser.add_argument("--selftest", action="store_true", help="run CPU-only geometry checks")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
