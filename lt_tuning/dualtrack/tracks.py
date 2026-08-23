"""Derive `track_ids` from upstream's own per-example output.  Genuinely new: upstream
emits no notion of a per-position role, so there is nothing here to subclass.

Input is exactly what `dataset.get_cot_latent_dataset`'s `process_dataset` returns
(dataset.py:673-680): `input_ids`, `labels`, `attention_mask`.  The derivation:

    n_prompt   leading run of -100 in labels   (dataset.py:636-639 writes exactly
                                                len(question_tokens) of them)
    LATENT     input_ids == thinking_id
    DELIM      the LAST occurrence of the delimiter token subsequence
    ANSWER     (delim_end, len)                (includes the appended eos, dataset.py:634)
    COT        everything else in [n_prompt, delim_start)

This runs PER EXAMPLE, BEFORE collation.  It has to: `MyCollator` left-pads
(dataset.py:438-468), after which the leading -100 run is `n_pad + n_prompt` and the
derivation would be off by the pad width.  `data.DualTrackCollator` pads the finished
track row alongside instead.

Pure python.  No torch, no transformers, no upstream import -- so it is testable on any
machine, which is the point.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .mask import TrackId, TrackLayoutError, spans_from_track

# Upstream writes `full_response += f"The final answer is:\n### {answer}"` (dataset.py:592).
DELIM_TEXT = "The final answer is:\n###"
LABEL_PAD = -100


class TrackDerivationError(ValueError):
    """The delimiter could not be located, or the resulting layout is not emittable."""


@dataclass(frozen=True)
class DerivedTracks:
    """The track row plus the three span boundaries the harness reports."""

    track_ids: Tuple[int, ...]
    q_end: int
    a_start: int
    v_start: int
    n_latents: int


def leading_label_pad(labels: Sequence[int]) -> int:
    """Length of the prompt region: upstream masks exactly the question tokens."""
    count = 0
    for value in labels:
        if int(value) != LABEL_PAD:
            break
        count += 1
    if count == 0:
        raise TrackDerivationError("labels have no leading -100 run; this is not upstream's schema")
    if count >= len(labels):
        raise TrackDerivationError("every label is -100; the example supervises nothing")
    return count


def find_last_subsequence(
    haystack: Sequence[int], needles: Sequence[Sequence[int]], start: int
) -> Tuple[int, int]:
    """Last occurrence at or after `start` of any candidate, as a half-open span."""
    best: Optional[Tuple[int, int]] = None
    for needle in needles:
        width = len(needle)
        if width == 0:
            continue
        target = tuple(int(t) for t in needle)
        for pos in range(len(haystack) - width, start - 1, -1):
            if tuple(int(t) for t in haystack[pos : pos + width]) == target:
                if best is None or pos > best[0]:
                    best = (pos, pos + width)
                break
    if best is None:
        raise TrackDerivationError(
            f"delimiter subsequence not found in [{start}, {len(haystack)}); the tokenizer "
            "merged it into the surrounding text.  Report this row as a seam mismatch "
            "rather than guessing a span."
        )
    return best


def delim_candidates(
    tokenizer: object, delim_text: str = DELIM_TEXT
) -> Tuple[Tuple[int, ...], ...]:
    """Token spellings of the delimiter to search for.

    A BPE tokenizer encodes "The final answer is:" differently after a newline than at the
    start of a string, so the bare encoding alone is not enough.  The in-context spelling is
    obtained by encoding "\\n" + delimiter and dropping the leading-newline prefix that the
    bare "\\n" encoding accounts for.  Candidates are de-duplicated; order does not matter
    because the search takes the last occurrence of any of them.
    """
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer has no encode()")
    bare = tuple(encode(delim_text, add_special_tokens=False))
    newline = tuple(encode("\n", add_special_tokens=False))
    in_context = tuple(encode("\n" + delim_text, add_special_tokens=False))
    candidates = [bare]
    if len(in_context) > len(newline) and in_context[: len(newline)] == newline:
        candidates.append(in_context[len(newline) :])
    if in_context and in_context not in candidates:
        candidates.append(in_context[1:] if len(in_context) > 1 else in_context)
    seen: List[Tuple[int, ...]] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def derive_tracks(
    input_ids: Sequence[int],
    labels: Sequence[int],
    thinking_id: int,
    delim_candidate_ids: Sequence[Sequence[int]],
) -> DerivedTracks:
    """`PROMPT+ (COT|LATENT)* DELIM+ ANSWER+`, validated by `spans_from_track`."""
    if len(input_ids) != len(labels):
        raise TrackDerivationError(
            f"input_ids ({len(input_ids)}) and labels ({len(labels)}) disagree in length"
        )
    q_end = leading_label_pad(labels)
    a_start, v_start = find_last_subsequence(input_ids, delim_candidate_ids, q_end)
    if a_start < q_end:
        raise TrackDerivationError("the delimiter landed inside the prompt region")
    if v_start >= len(input_ids):
        raise TrackDerivationError("the delimiter is at the very end; there is no answer span")
    track: List[int] = [int(TrackId.PROMPT)] * q_end
    for pos in range(q_end, a_start):
        track.append(
            int(TrackId.LATENT) if int(input_ids[pos]) == int(thinking_id) else int(TrackId.COT)
        )
    track += [int(TrackId.DELIM)] * (v_start - a_start)
    track += [int(TrackId.ANSWER)] * (len(input_ids) - v_start)
    if int(thinking_id) in {int(t) for t in input_ids[v_start:]}:
        raise TrackDerivationError(
            "a <thinking> token landed inside the ANSWER span; clamp the insertion strategy "
            "with insertion.clamp_to_reasoning (upstream dataset.py:208-209 appends idx+1 "
            "after '###')"
        )
    try:
        spans = spans_from_track(track)
    except TrackLayoutError as exc:
        raise TrackDerivationError(f"derived track is not an emittable layout: {exc}") from exc
    return DerivedTracks(
        track_ids=tuple(track),
        q_end=spans.q_end,
        a_start=spans.a_start,
        v_start=spans.v_start,
        n_latents=spans.n_latents,
    )


def pad_track_row(track_ids: Sequence[int], left: int, right: int) -> List[int]:
    """Both-side padding, matching `MyCollator`'s left-align plus the tokenizer's right pad."""
    if left < 0 or right < 0:
        raise ValueError(f"pad widths must be non-negative; got left={left} right={right}")
    return [int(TrackId.PAD)] * left + [int(t) for t in track_ids] + [int(TrackId.PAD)] * right


def latent_visit_order(track_rows: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """`(batch, position)` keys in the order upstream constructs latents.

    upstream model.py:504-528 walks `boundary_end` ascending and, inside each boundary,
    `batch_idx` ascending -- and `_apply_transform` is called exactly once per latent in
    that order.  `model.LatentTape` pops this list to label the tensors it records.
    """
    positions = sorted(
        {
            pos
            for row in track_rows
            for pos, value in enumerate(row)
            if int(value) == int(TrackId.LATENT) and 0 < pos < len(row)
        }
    )
    return [
        (batch, pos)
        for pos in positions
        for batch, row in enumerate(track_rows)
        if int(row[pos]) == int(TrackId.LATENT)
    ]


# --- selftests -----------------------------------------------------------------------


def _fixture(thinking_id: int = 7) -> Tuple[List[int], List[int], List[List[int]]]:
    delim = [90, 91]
    ids = [1, 2, 3] + [10, 11, thinking_id, 12, thinking_id] + delim + [50, 51, 2]
    labels = [LABEL_PAD] * 3 + ids[3:]
    return ids, labels, [delim]


def _selftest_derivation() -> None:
    ids, labels, delim = _fixture()
    got = derive_tracks(ids, labels, 7, delim)
    assert got.q_end == 3 and got.a_start == 8 and got.v_start == 10, got
    assert got.n_latents == 2
    expected = (
        [int(TrackId.PROMPT)] * 3
        + [
            int(TrackId.COT),
            int(TrackId.COT),
            int(TrackId.LATENT),
            int(TrackId.COT),
            int(TrackId.LATENT),
        ]
        + [int(TrackId.DELIM)] * 2
        + [int(TrackId.ANSWER)] * 3
    )
    assert list(got.track_ids) == expected, got.track_ids
    latents = [p for p, v in enumerate(got.track_ids) if v == int(TrackId.LATENT)]
    assert latents == [
        p for p, t in enumerate(ids) if t == 7
    ], "LATENT must equal the <thinking> ids"
    print(
        f"  derivation: q_end={got.q_end} a_start={got.a_start} v_start={got.v_start} "
        f"n_latents={got.n_latents}: OK"
    )


def _selftest_label_partition() -> None:
    """The supervised rows split into COT-side and ANSWER-side with no overlap and no gap.

    This is the pure-python twin of `loss.partition_ce`: same shift convention as upstream
    model.py:544-545, so if one is wrong the other is too and both selftests fail.
    """
    from .mask import ANSWER_LOSS_TRACKS, COT_LOSS_TRACKS

    ids, labels, delim = _fixture()
    track = derive_tracks(ids, labels, 7, delim).track_ids
    supervised = [t for t in range(len(labels) - 1) if int(labels[t + 1]) != LABEL_PAD]
    cot_rows = [t for t in supervised if int(track[t + 1]) in COT_LOSS_TRACKS]
    ans_rows = [t for t in supervised if int(track[t + 1]) in ANSWER_LOSS_TRACKS]
    assert not set(cot_rows) & set(ans_rows), "the two partitions overlap"
    assert sorted(cot_rows + ans_rows) == supervised, "the two partitions do not cover S"
    assert len(supervised) == len(ids) - 3, "|S| must be N - n_prompt"
    assert len(ans_rows) == len(ids) - 10, "|S_ans| must be N - v_start"
    assert ans_rows and cot_rows, "a degenerate partition proves nothing"
    print(
        f"  label partition: |S|={len(supervised)} |S_cot|={len(cot_rows)} "
        f"|S_ans|={len(ans_rows)}, disjoint and covering: OK"
    )


def _selftest_rejections() -> None:
    ids, labels, delim = _fixture()
    cases: Dict[str, Tuple[List[int], List[int], List[List[int]]]] = {
        "no delimiter": (ids, labels, [[999, 998]]),
        "no prompt mask": (ids, list(ids), delim),
        "thinking in answer": (ids[:10] + [7] + ids[11:], labels[:10] + [7] + labels[11:], delim),
    }
    for name, (bad_ids, bad_labels, bad_delim) in cases.items():
        try:
            derive_tracks(bad_ids, bad_labels, 7, bad_delim)
        except TrackDerivationError:
            continue
        raise AssertionError(f"{name!r} must raise TrackDerivationError, not be papered over")
    print(f"  refuses {len(cases)} malformed inputs instead of guessing: OK")


def _selftest_visit_order() -> None:
    """Upstream's construction order is `(boundary ascending, batch ascending)`.

    `model.LatentTape` labels captured tensors by popping this list, so an order error here
    would mislabel every latent in a multi-row batch -- silently, since the shapes agree.
    """
    rows = [
        [1, 2, 3, 2, 3, 4, 5],  # latents at 2 and 4
        [1, 3, 2, 2, 3, 4, 5],  # latents at 1 and 4
    ]
    order = latent_visit_order(rows)
    assert order == [(1, 1), (0, 2), (0, 4), (1, 4)], order
    single = latent_visit_order([[1, 3, 2, 3, 4, 5]])
    assert single == [(0, 1), (0, 3)], single
    print(f"  latent visit order (boundary asc, batch asc): {order}: OK")


def _selftest_padding() -> None:
    ids, labels, delim = _fixture()
    track = derive_tracks(ids, labels, 7, delim).track_ids
    padded = pad_track_row(track, 2, 3)
    assert padded[:2] == [int(TrackId.PAD)] * 2 and padded[-3:] == [int(TrackId.PAD)] * 3
    assert len(padded) == len(track) + 5
    spans = spans_from_track(padded)
    assert spans.pad_left == 2 and spans.n_end == len(track) + 2
    print("  both-side padding keeps the layout valid and shifts the spans: OK")


def selftest() -> None:
    """CPU-only, no torch, no transformers, no network."""
    _selftest_derivation()
    _selftest_label_partition()
    _selftest_rejections()
    _selftest_visit_order()
    _selftest_padding()
    print("tracks.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="track_ids derived from upstream's output")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
