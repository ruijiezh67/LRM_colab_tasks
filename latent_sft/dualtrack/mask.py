r"""The 4-D bottleneck attention mask. Needs torch; the geometry lives in ``spans``.

This module is deliberately thin: the definition of the mask is the pure
predicate ``spans.keep_matrix_entry``, and ``--selftest`` asserts that the
vectorized builder here agrees with ``spans.reference_keep_matrix`` elementwise
on several geometries including padding. Upstream has no counterpart -- its
collator builds a 2-D mask from token values (upstream/src/stage2/data.py:275),
which is both wrong for the dual track and unsafe when pad_token == eos_token.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from dualtrack import spans as spans_lib
from dualtrack.spans import TrackSpans

Span = Tuple[int, int]


def build_bottleneck_mask(
    lengths: Sequence[int],
    key_spans: Sequence[Span],
    query_spans: Sequence[Span],
    padded_len: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Build the ``[B, 1, S, S]`` bool keep-mask (True = may attend).

    ``lengths`` are the true unpadded lengths. Padding is detected from them and
    never from a token value, because a genuine token can equal the pad id
    (for Qwen/DeepSeek upstream sets pad_token = eos_token, modeling_stage2.py:161).
    """
    batch = len(lengths)
    if not (batch == len(key_spans) == len(query_spans)):
        raise ValueError("lengths, key_spans and query_spans must be the same length")
    if batch == 0:
        raise ValueError("cannot build a mask for an empty batch")

    rows = torch.arange(padded_len, device=device).view(1, padded_len, 1)
    cols = torch.arange(padded_len, device=device).view(1, 1, padded_len)
    lens = torch.tensor(list(lengths), dtype=torch.long, device=device).view(batch, 1, 1)
    key_start = torch.tensor([s for s, _ in key_spans], dtype=torch.long, device=device).view(batch, 1, 1)
    key_end = torch.tensor([e for _, e in key_spans], dtype=torch.long, device=device).view(batch, 1, 1)
    query_start = torch.tensor([s for s, _ in query_spans], dtype=torch.long, device=device).view(batch, 1, 1)
    query_end = torch.tensor([e for _, e in query_spans], dtype=torch.long, device=device).view(batch, 1, 1)

    causal = cols <= rows
    key_is_real = cols < lens
    # A pad QUERY row is excluded here so pad_row_diagonal is the only thing it
    # attends; see spans.keep_matrix_entry, which this must match elementwise.
    query_is_real = rows < lens
    blocked = (rows >= query_start) & (rows < query_end) & (cols >= key_start) & (cols < key_end)
    pad_row_diagonal = (rows == cols) & (rows >= lens)

    keep = (causal & key_is_real & query_is_real & ~blocked) | pad_row_diagonal
    return keep.unsqueeze(1)


def spans_to_mask_inputs(
    items: Sequence[TrackSpans], bottleneck: bool = True
) -> Tuple[list, list]:
    """``(key_spans, query_spans)`` for a batch; ``bottleneck=False`` blocks nothing."""
    if not bottleneck:
        return [(0, 0) for _ in items], [(0, 0) for _ in items]
    return (
        [spans_lib.cot_key_span(item) for item in items],
        [spans_lib.answer_query_span(item) for item in items],
    )


def to_additive(keep: Tensor, dtype: torch.dtype) -> Tensor:
    """Bool keep-mask -> additive float mask (0 where kept, finfo.min where blocked).

    Transformers requires the inverted additive form and validates ``max == 0``.
    """
    if keep.dtype != torch.bool:
        raise TypeError(f"expected a bool keep-mask, got {keep.dtype}")
    if keep.dim() != 4:
        raise ValueError(f"expected a 4-D [B,1,S,S] keep-mask, got shape {tuple(keep.shape)}")
    additive = torch.zeros(keep.shape, dtype=dtype, device=keep.device)
    return additive.masked_fill(~keep, torch.finfo(dtype).min)


def _geometries() -> Tuple[Tuple[TrackSpans, ...], int]:
    items = (
        spans_lib.worked_example(),
        spans_lib.build_spans(6, 2, 2, 3, 1, 3, 1, 3),
        spans_lib.build_spans(1, 1, 1, 1, 1, 1, 1, 1),
        spans_lib.build_spans(9, 1, 5, 1, 1, 12, 1, 2),
    )
    return items, max(item.total_len for item in items) + 3


def _selftest_agrees_with_reference() -> None:
    items, padded = _geometries()
    lengths = [item.total_len for item in items]
    key_spans, query_spans = spans_to_mask_inputs(items)
    keep = build_bottleneck_mask(lengths, key_spans, query_spans, padded)
    reference = spans_lib.reference_keep_matrix(lengths, key_spans, query_spans, padded)
    assert keep.shape == (len(items), 1, padded, padded) and keep.dtype == torch.bool
    expected = torch.tensor(reference, dtype=torch.bool).unsqueeze(1)
    assert torch.equal(keep, expected), "the vectorized mask disagrees with the pure reference"


def _selftest_mask_geometry() -> None:
    items, padded = _geometries()
    lengths = [item.total_len for item in items]
    key_spans, query_spans = spans_to_mask_inputs(items)
    keep = build_bottleneck_mask(lengths, key_spans, query_spans, padded)
    for row_index, item in enumerate(items):
        k0, k1 = spans_lib.cot_key_span(item)
        q0, q1 = spans_lib.answer_query_span(item)
        plane = keep[row_index, 0]
        assert plane[q0:q1, k0:k1].sum().item() == 0, "bottleneck leaks into the CoT keys"
        assert plane[q0, q0].item() is False, "</cot> self-attention re-opened inside the block"
        assert (plane.sum(dim=-1) > 0).all(), "an empty query row would make SDPA emit NaN"
        assert plane.triu(diagonal=1).sum().item() == 0, "causality broken"
        assert plane[: item.total_len, item.total_len :].sum().item() == 0, "attends to padding"
        assert plane[q0 - 1, k0:k1].sum().item() > 0, "rows before the bottleneck must read the CoT"
        for query in range(item.total_len, padded):
            assert plane[query, query].item() is True, "pad row lost its forced diagonal"
            assert plane[query].sum().item() == 1, "a pad row attended something real"


def _selftest_bottleneck_off() -> None:
    items, padded = _geometries()
    lengths = [item.total_len for item in items]
    key_spans, query_spans = spans_to_mask_inputs(items, bottleneck=False)
    keep = build_bottleneck_mask(lengths, key_spans, query_spans, padded)
    for row_index, item in enumerate(items):
        k0, k1 = spans_lib.cot_key_span(item)
        q0, q1 = spans_lib.answer_query_span(item)
        assert keep[row_index, 0, q0:q1, k0:k1].sum().item() > 0, "bottleneck=False still blocked keys"


def _selftest_additive() -> None:
    spans = spans_lib.worked_example()
    key_spans, query_spans = spans_to_mask_inputs([spans])
    keep = build_bottleneck_mask([spans.total_len], key_spans, query_spans, spans.total_len)
    for dtype in (torch.float32, torch.bfloat16):
        additive = to_additive(keep, dtype)
        assert additive.dtype == dtype
        assert additive.max().item() == 0.0, "transformers validates that the 4-D mask maxes at 0"
        assert additive[keep].abs().sum().item() == 0.0
        assert (additive[~keep] == torch.finfo(dtype).min).all()
    for bad, error in ((keep.float(), TypeError), (keep[:, 0], ValueError)):
        try:
            to_additive(bad, torch.float32)
        except error:
            continue
        raise AssertionError(f"to_additive accepted {tuple(bad.shape)} / {bad.dtype}")


def selftest() -> None:
    _selftest_agrees_with_reference()
    _selftest_mask_geometry()
    _selftest_bottleneck_off()
    _selftest_additive()
    print(
        "[mask] OK -- the tensor builder equals spans.reference_keep_matrix elementwise on 4 "
        "geometries with padding, bottleneck geometry holds (causal, no padding keys, no empty "
        "rows, pad-rows-only diagonal), bottleneck=False opens the CoT keys, additive form maxes at 0."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track bottleneck mask (needs torch)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("mask.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
