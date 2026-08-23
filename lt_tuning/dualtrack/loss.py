"""The CE_cot / CE_ans partition.  Upstream has one loss line and no partition, so this is
new work; what it must NOT do is disagree with upstream at the identity point.

`loss = cot_w * CE_cot + ans_w * CE_ans + latent_w * L_latent`

CE_cot and CE_ans partition the supervised positions and share ONE normaliser `|S|`, which
is what makes `(1.0, 1.0, 0.0)` numerically identical to upstream's `CrossEntropyLoss()`
(model.py:546-550).  Two separate means would not reduce to it.

`L_latent` is not computed here: it is upstream's own loss line, evaluated by calling
`super().forward(labels=latent_labels)` in model.py.  `latent_labels` below builds those
labels; the CE itself stays upstream's.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .config import LossWeights
from .mask import ANSWER_LOSS_TRACKS, COT_LOSS_TRACKS, TrackId

LABEL_PAD = -100


@dataclass(frozen=True)
class LossParts:
    """`total` is the trainable scalar; the rest are logged, never optimised separately."""

    total: torch.Tensor
    cot_term: torch.Tensor
    ans_term: torch.Tensor
    cot_mean: torch.Tensor
    ans_mean: torch.Tensor
    latent: torch.Tensor
    n_supervised: int
    n_cot: int
    n_ans: int


def _selection(shift_track: torch.Tensor, tracks: Sequence[int]) -> torch.Tensor:
    out = torch.zeros_like(shift_track, dtype=torch.bool)
    for value in tracks:
        out = out | (shift_track == int(value))
    return out


def latent_labels(labels: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
    """Labels for `L_latent`: supervised only where the target position is a latent.

    Returns a NEW tensor; `labels` is never mutated.  `L_latent` therefore overlaps CE_cot
    by construction (latent positions live in the COT partition).  That is deliberate and is
    stated in the README so `latent_w` is not read as re-weighting a disjoint slice.
    """
    return labels.masked_fill(track_ids != int(TrackId.LATENT), LABEL_PAD)


def has_supervised_latent(labels: torch.Tensor, track_ids: torch.Tensor) -> bool:
    """Stage-0 batches and rows where `strategy.apply` selected nothing (dataset.py:116)
    have no latent labels at all, and `CrossEntropyLoss` over an all-ignored tensor is NaN.
    Callers branch on this instead of discovering it in the gradients."""
    return bool((latent_labels(labels, track_ids) != LABEL_PAD).any())


def partition_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    track_ids: torch.Tensor,
    weights: LossWeights = LossWeights(),
    latent_loss: Optional[torch.Tensor] = None,
) -> LossParts:
    """Shift convention identical to upstream model.py:544-545."""
    if logits.shape[:2] != labels.shape or labels.shape != track_ids.shape:
        raise ValueError(
            f"shape mismatch: logits{tuple(logits.shape)} labels{tuple(labels.shape)} "
            f"track_ids{tuple(track_ids.shape)}"
        )
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_track = track_ids[..., 1:].contiguous()
    supervised = shift_labels != LABEL_PAD
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=LABEL_PAD,
    ).view(shift_labels.shape)
    cot_sel = supervised & _selection(shift_track, COT_LOSS_TRACKS)
    ans_sel = supervised & _selection(shift_track, ANSWER_LOSS_TRACKS)
    if bool((cot_sel & ans_sel).any()):
        raise ValueError("CE_cot and CE_ans overlap; they must partition the supervised positions")
    if not bool(((cot_sel | ans_sel) == supervised).all()):
        raise ValueError("CE_cot and CE_ans do not cover the supervised positions")
    denom = supervised.sum().clamp(min=1)
    cot_sum = (per_token * cot_sel).sum()
    ans_sum = (per_token * ans_sel).sum()
    latent = (
        latent_loss
        if latent_loss is not None
        else torch.zeros((), device=logits.device, dtype=per_token.dtype)
    )
    total = (weights.cot_w * cot_sum + weights.ans_w * ans_sum) / denom + weights.latent_w * latent
    return LossParts(
        total=total,
        cot_term=cot_sum / denom,
        ans_term=ans_sum / denom,
        cot_mean=cot_sum / cot_sel.sum().clamp(min=1),
        ans_mean=ans_sum / ans_sel.sum().clamp(min=1),
        latent=latent,
        n_supervised=int(supervised.sum()),
        n_cot=int(cot_sel.sum()),
        n_ans=int(ans_sel.sum()),
    )


def upstream_reference_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Upstream model.py:543-550, re-typed HERE ONLY as a test oracle.

    It is never on the training path: training gets its CE from `super().forward()`.  The
    point of having it is that `_selftest_matches_upstream` can compare against something
    written out in full rather than against the same code under a different name.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


# --- selftests -----------------------------------------------------------------------


def _fixture(seed: int = 3):
    from .tracks import derive_tracks

    thinking_id = 7
    delim = [90, 91]
    rows = [
        [1, 2, 3] + [10, 11, thinking_id, 12, thinking_id] + delim + [50, 51, 2],
        [1, 4, 5] + [13, thinking_id, 14, 15, thinking_id] + delim + [52, 53, 2],
    ]
    labels = [[LABEL_PAD] * 3 + row[3:] for row in rows]
    tracks = [
        list(derive_tracks(r, l, thinking_id, [delim]).track_ids) for r, l in zip(rows, labels)
    ]
    torch.manual_seed(seed)
    vocab = max(max(r) for r in rows) + 1
    logits = torch.randn(len(rows), len(rows[0]), vocab)
    return (
        logits,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(tracks, dtype=torch.long),
        thinking_id,
    )


def _selftest_matches_upstream() -> None:
    logits, labels, tracks, _ = _fixture()
    parts = partition_ce(logits, labels, tracks, LossWeights(1.0, 1.0, 0.0))
    reference = upstream_reference_loss(logits, labels)
    delta = float((parts.total - reference).abs())
    assert delta < 1e-6, f"at (1,1,0) the total must equal upstream's loss; delta={delta}"
    assert parts.n_cot + parts.n_ans == parts.n_supervised
    assert parts.n_cot > 0 and parts.n_ans > 0, "a degenerate partition proves nothing"
    print(
        f"  total == upstream CrossEntropyLoss at (1,1,0): {float(parts.total):.6f} vs "
        f"{float(reference):.6f}; |S|={parts.n_supervised} cot={parts.n_cot} ans={parts.n_ans}: OK"
    )


def _selftest_weights_bite() -> None:
    logits, labels, tracks, _ = _fixture()
    base = partition_ce(logits, labels, tracks, LossWeights(1.0, 1.0, 0.0))
    heavy = partition_ce(logits, labels, tracks, LossWeights(1.0, 4.0, 0.0))
    assert heavy.total > base.total, "ans_w must move the total"
    expected = base.cot_term + 4.0 * base.ans_term
    assert float((heavy.total - expected).abs()) < 1e-5, (heavy.total, expected)
    latent = torch.tensor(2.0)
    with_latent = partition_ce(logits, labels, tracks, LossWeights(1.0, 1.0, 0.5), latent)
    assert float((with_latent.total - (base.total + 1.0)).abs()) < 1e-5, with_latent.total
    assert float(base.latent) == 0.0, "latent_loss=None must contribute exactly nothing"
    print(
        f"  weights are linear in the three terms: base={float(base.total):.4f} "
        f"ans_w=4 -> {float(heavy.total):.4f} latent_w=0.5*2.0 -> "
        f"{float(with_latent.total):.4f}: OK"
    )


def _selftest_latent_labels() -> None:
    logits, labels, tracks, thinking_id = _fixture()
    latent = latent_labels(labels, tracks)
    assert not torch.equal(latent, labels), "the latent labels must actually differ"
    assert bool((labels == _fixture()[1]).all()), "labels were mutated in place"
    supervised = latent != LABEL_PAD
    assert bool((tracks[supervised] == int(TrackId.LATENT)).all())
    assert bool((latent[supervised] == thinking_id).all()), (
        "every supervised latent label must be the <thinking> id; that CE is upstream's "
        "insertion objective"
    )
    assert has_supervised_latent(labels, tracks)
    empty = torch.full_like(tracks, int(TrackId.COT))
    assert not has_supervised_latent(labels, empty), (
        "a batch with no latents must be detected, or upstream's CE returns NaN over an "
        "all-ignored tensor"
    )
    print(
        f"  latent labels: {int(supervised.sum())} positions, all == <thinking>, "
        f"empty-batch guard works, input not mutated: OK"
    )


def _selftest_partition_guard() -> None:
    logits, labels, tracks, _ = _fixture()
    broken = tracks.clone()
    broken[0, 5] = int(TrackId.PROMPT)  # a supervised position in neither partition
    try:
        partition_ce(logits, labels, broken)
    except ValueError as exc:
        assert "cover" in str(exc) or "overlap" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("a supervised position outside both partitions must raise")
    print("  a supervised position in neither partition raises rather than vanishing: OK")


def selftest() -> None:
    """CPU-only, no network, no weights: random logits over derived tracks."""
    _selftest_matches_upstream()
    _selftest_weights_bite()
    _selftest_latent_labels()
    _selftest_partition_guard()
    print("loss.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-track loss partition")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
