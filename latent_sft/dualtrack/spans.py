r"""Dual-track span arithmetic, the bottleneck predicate and the label partition.

STDLIB ONLY -- no torch, no transformers. ``mask.py`` builds the same matrix with
tensors and its selftest checks it elementwise against ``reference_keep_matrix``
here, so the geometry stays testable on a machine with no ML stack installed.

Logical sequence
----------------
    prefix | <bot> | latent x L | <eot> | <cot> | visible CoT | </cot> | answer

    prefix       = [0,               P)
    bot          = [P,               P + b0)          <bot> is upstream's <think>
    latent       = [P + b0,          P + b0 + L)      KL region; ids are -100
    eot          = [latent_end,      latent_end + b1) <eot> is upstream's </think>
    cot_open     = [eot_end,         eot_end + c0)
    cot_content  = [cot_open_end,    cot_open_end + Lc)
    cot_close    = [cot_content_end, cot_content_end + c1)
    answer       = [cot_close_end,   cot_close_end + La)

Bottleneck
----------
    blocked KEY columns     = [cot_open_pos, answer_start)
    bottlenecked QUERY rows = [answer_start - 1, S)

``answer_start - 1`` is the last ``</cot>`` piece: under shifted CE that row is
the one that emits the first answer token, so it must already be blocked.

    keep[b, i, j] = ((i < len_b) and (j <= i) and (j < len_b) and not blocked(b, i, j))
                    or (i == j and i >= len_b)

The ``i < len_b`` conjunct confines a PAD query row to its own diagonal. Without
it a pad row attends every real key, the forced diagonal becomes unreachable dead
code (no row is ever empty without it), and "diagonal on padding rows only"
stops being true of the matrix that is actually built.

The forced diagonal is gated on PAD rows only. A global ``keep |= eye`` would
restore the (q0, q0) pair, re-opening ``</cot>`` self-attention inside the
blocked span and leaking the delimiter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

# -100 is overloaded upstream: it is the CE ignore index AND the latent-slot
# sentinel inside input_ids (upstream/src/stage2/data.py:243, consumed at
# upstream/src/modeling/modeling_stage2.py:215-217).
IGNORE_INDEX = -100

Span = Tuple[int, int]


@dataclass(frozen=True)
class TrackSpans:
    """Immutable per-example segment lengths; every index is derived, never stored."""

    prefix_len: int
    bot_len: int
    latent_len: int
    eot_len: int
    cot_open_len: int
    cot_content_len: int
    cot_close_len: int
    answer_len: int

    @property
    def latent_start(self) -> int:
        return self.prefix_len + self.bot_len

    @property
    def latent_end(self) -> int:
        return self.latent_start + self.latent_len

    @property
    def eot_pos(self) -> int:
        return self.latent_end

    @property
    def cot_open_pos(self) -> int:
        return self.eot_pos + self.eot_len

    @property
    def cot_content_start(self) -> int:
        return self.cot_open_pos + self.cot_open_len

    @property
    def ccot_pos(self) -> int:
        return self.cot_content_start + self.cot_content_len

    @property
    def answer_start(self) -> int:
        return self.ccot_pos + self.cot_close_len

    @property
    def total_len(self) -> int:
        return self.answer_start + self.answer_len

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def build_spans(
    prefix_len: int,
    bot_len: int,
    latent_len: int,
    eot_len: int,
    cot_open_len: int,
    cot_content_len: int,
    cot_close_len: int,
    answer_len: int,
) -> TrackSpans:
    """Validate and construct the single source of truth for one example."""
    spans = TrackSpans(
        prefix_len=prefix_len,
        bot_len=bot_len,
        latent_len=latent_len,
        eot_len=eot_len,
        cot_open_len=cot_open_len,
        cot_content_len=cot_content_len,
        cot_close_len=cot_close_len,
        answer_len=answer_len,
    )
    if min(prefix_len, bot_len, eot_len, cot_open_len, cot_close_len, answer_len) < 1:
        raise ValueError(f"every non-latent segment must be non-empty: {spans}")
    if latent_len < 1:
        raise ValueError(f"latent_len must be >= 1 (KL needs at least one slot): {spans}")
    if cot_content_len < 1:
        raise ValueError(f"visible CoT must be non-empty: {spans}")
    if cot_open_len != 1 or cot_close_len != 1:
        raise ValueError(
            "<cot>/</cot> are added special tokens and must tokenize to exactly one piece each; "
            f"got {cot_open_len}/{cot_close_len}"
        )
    if spans.latent_start < 1:
        raise ValueError(f"latent_start must be >= 1 so the KL query row exists: {spans}")
    if spans.cot_open_pos < 1:
        raise ValueError(f"key 0 must never be blocked: {spans}")
    return spans


def cot_key_span(spans: TrackSpans) -> Span:
    """Half-open key columns the answer queries may not attend to."""
    return (spans.cot_open_pos, spans.answer_start)


def answer_query_span(spans: TrackSpans) -> Span:
    """Half-open query rows that are bottlenecked, starting at the last ``</cot>`` piece."""
    return (spans.answer_start - 1, spans.total_len)


def blocked_key_indices(spans: TrackSpans) -> range:
    """Key positions the incremental decoder must zero from ``</cot>`` onward."""
    start, end = cot_key_span(spans)
    return range(start, end)


def latent_index(spans: TrackSpans) -> List[int]:
    """``[latent_start, latent_end]`` -- upstream's KL contract, bit for bit.

    Upstream slices ``logits[b, s-1:e-1]`` (modeling_stage2.py:260) with exactly
    this pair, so the visible track must never shift it.
    """
    return [spans.latent_start, spans.latent_end]


def keep_matrix_entry(length: int, key_span: Span, query_span: Span, row: int, col: int) -> bool:
    """The mask as a pure predicate; the only definition of the bottleneck."""
    key_start, key_end = key_span
    query_start, query_end = query_span
    blocked = (query_start <= row < query_end) and (key_start <= col < key_end)
    # `row < length` keeps pad QUERY rows out of the causal branch, so the forced
    # diagonal below is the only thing they ever attend.
    causal_real = (row < length) and (col <= row) and (col < length)
    pad_row_diagonal = (row == col) and (row >= length)
    return (causal_real and not blocked) or pad_row_diagonal


def keep_predicate(spans: TrackSpans, length: int, row: int, col: int) -> bool:
    """``keep_matrix_entry`` with both spans derived from one ``TrackSpans``."""
    return keep_matrix_entry(length, cot_key_span(spans), answer_query_span(spans), row, col)


def reference_keep_matrix(
    lengths: Sequence[int],
    key_spans: Sequence[Span],
    query_spans: Sequence[Span],
    padded_len: int,
) -> List[List[List[bool]]]:
    """Nested-loop reference for ``mask.build_bottleneck_mask``; shape [B, S, S]."""
    if not (len(lengths) == len(key_spans) == len(query_spans)):
        raise ValueError("lengths, key_spans and query_spans must be the same length")
    return [
        [
            [keep_matrix_entry(length, key_span, query_span, row, col) for col in range(padded_len)]
            for row in range(padded_len)
        ]
        for length, key_span, query_span in zip(lengths, key_spans, query_spans)
    ]


def supervised_positions(labels: Sequence[int]) -> set:
    return {index for index, value in enumerate(labels) if value != IGNORE_INDEX}


def assert_label_partition(
    input_ids: Sequence[int],
    labels_cot: Sequence[int],
    labels_answer: Sequence[int],
    spans: TrackSpans,
) -> None:
    """CE_cot and CE_ans must partition ``[eot_pos, S)`` and be teacher-forced."""
    cot_positions = supervised_positions(labels_cot)
    answer_positions = supervised_positions(labels_answer)
    if cot_positions & answer_positions:
        raise AssertionError("CE_cot and CE_ans overlap; the loss would double-count positions")
    if cot_positions | answer_positions != set(range(spans.eot_pos, spans.total_len)):
        raise AssertionError("CE_cot and CE_ans do not cover [eot_pos, S)")
    if cot_positions != set(range(spans.eot_pos, spans.answer_start)):
        raise AssertionError("CE_cot does not cover exactly [eot_pos, answer_start)")
    if answer_positions != set(range(spans.answer_start, spans.total_len)):
        raise AssertionError("CE_ans does not cover exactly [answer_start, S)")
    for position in sorted(cot_positions | answer_positions):
        source = labels_cot if position in cot_positions else labels_answer
        if source[position] != input_ids[position]:
            raise AssertionError(f"teacher-forcing identity broken at position {position}")
    latents = [index for index, value in enumerate(input_ids) if value == IGNORE_INDEX]
    if latents != list(range(spans.latent_start, spans.latent_end)):
        raise AssertionError("the latent block inside input_ids is not contiguous")


def worked_example() -> TrackSpans:
    """The example documented in the spec: P=6, b0=2, L=3, b1=3, Lc=4, La=4."""
    return build_spans(
        prefix_len=6,
        bot_len=2,
        latent_len=3,
        eot_len=3,
        cot_open_len=1,
        cot_content_len=4,
        cot_close_len=1,
        answer_len=4,
    )


def _selftest_indices() -> None:
    spans = worked_example()
    assert spans.latent_start == 8 and spans.latent_end == 11 and spans.eot_pos == 11
    assert spans.cot_open_pos == 14 and spans.ccot_pos == 19
    assert spans.answer_start == 20 and spans.total_len == 24
    assert cot_key_span(spans) == (14, 20)
    assert answer_query_span(spans) == (19, 24)
    assert list(blocked_key_indices(spans)) == [14, 15, 16, 17, 18, 19]
    assert latent_index(spans) == [8, 11]
    q0, _ = answer_query_span(spans)
    k0, k1 = cot_key_span(spans)
    assert q0 == spans.answer_start - 1 and k0 <= q0 < k1


def _selftest_validation() -> None:
    base = dict(
        prefix_len=6, bot_len=2, latent_len=3, eot_len=3,
        cot_open_len=1, cot_content_len=4, cot_close_len=1, answer_len=4,
    )
    for bad in (
        dict(cot_open_len=2),
        dict(cot_close_len=2),
        dict(latent_len=0),
        dict(cot_content_len=0),
        dict(prefix_len=0),
    ):
        kwargs = dict(base)
        kwargs.update(bad)
        try:
            build_spans(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"build_spans accepted an invalid geometry: {bad}")


def _assert_plane(plane: List[List[bool]], item: TrackSpans, padded: int) -> None:
    k0, k1 = cot_key_span(item)
    q0, q1 = answer_query_span(item)
    for row in range(q0, q1):
        assert not any(plane[row][k0:k1]), "bottleneck leaks into the CoT keys"
    assert plane[q0][q0] is False, "</cot> self-attention re-opened inside the block"
    for row in range(padded):
        assert any(plane[row]), "an empty query row would make SDPA emit NaN"
        for col in range(row + 1, padded):
            assert plane[row][col] is False, "causality broken"
    for row in range(item.total_len):
        assert not any(plane[row][item.total_len:]), "attends to padding"
    assert any(plane[q0 - 1][k0:k1]), "rows before the bottleneck must still read the CoT"
    for row in range(item.total_len, padded):
        assert plane[row][row] is True, "pad row lost its forced diagonal"


def _selftest_mask_geometry() -> None:
    spans = worked_example()
    short = build_spans(6, 2, 2, 3, 1, 3, 1, 3)
    lengths = [spans.total_len, short.total_len]
    padded = max(lengths) + 3
    planes = reference_keep_matrix(
        lengths=lengths,
        key_spans=[cot_key_span(spans), cot_key_span(short)],
        query_spans=[answer_query_span(spans), answer_query_span(short)],
        padded_len=padded,
    )
    assert len(planes) == 2 and len(planes[0]) == padded and len(planes[0][0]) == padded
    for plane, item in zip(planes, (spans, short)):
        _assert_plane(plane, item, padded)


def _fixture_labels(spans: TrackSpans) -> Tuple[List[int], List[int], List[int]]:
    """A hand-built sequence with the dual-track layout, ids chosen distinct."""
    input_ids = list(range(100, 100 + spans.total_len))
    for index in range(spans.latent_start, spans.latent_end):
        input_ids[index] = IGNORE_INDEX
    labels_cot = [
        input_ids[i] if spans.eot_pos <= i < spans.answer_start else IGNORE_INDEX
        for i in range(spans.total_len)
    ]
    labels_answer = [
        input_ids[i] if i >= spans.answer_start else IGNORE_INDEX
        for i in range(spans.total_len)
    ]
    return input_ids, labels_cot, labels_answer


def _selftest_label_partition() -> None:
    spans = worked_example()
    input_ids, labels_cot, labels_answer = _fixture_labels(spans)
    assert_label_partition(input_ids, labels_cot, labels_answer, spans)

    overlapping = list(labels_answer)
    overlapping[spans.answer_start - 1] = input_ids[spans.answer_start - 1]
    try:
        assert_label_partition(input_ids, labels_cot, overlapping, spans)
    except AssertionError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("an overlapping label partition was accepted")

    holed = list(labels_cot)
    holed[spans.cot_content_start] = IGNORE_INDEX
    try:
        assert_label_partition(input_ids, holed, labels_answer, spans)
    except AssertionError as exc:
        assert "cover" in str(exc)
    else:
        raise AssertionError("a gap in the supervised span was accepted")

    shifted = list(labels_answer)
    shifted[spans.answer_start] = input_ids[spans.answer_start] + 1
    try:
        assert_label_partition(input_ids, labels_cot, shifted, spans)
    except AssertionError as exc:
        assert "teacher-forcing" in str(exc)
    else:
        raise AssertionError("a non-teacher-forced label was accepted")


def selftest() -> None:
    _selftest_indices()
    _selftest_validation()
    _selftest_mask_geometry()
    _selftest_label_partition()
    print(
        "[spans] OK -- span arithmetic, five geometry rejections, bottleneck geometry via the "
        "nested-loop reference (causal, no padding keys, no empty rows, pad-rows-only diagonal, "
        "</cot> self-attention stays blocked), label partition disjoint/covering/teacher-forced."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track span arithmetic (stdlib only)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("spans.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
