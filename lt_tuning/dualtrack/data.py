"""Dataset and collator: upstream's, with `track_ids` carried alongside.

  * `build_dualtrack_dataset` CALLS upstream `get_cot_latent_dataset` (dataset.py:562-736)
    and then attaches `track_ids` row by row.  The tokenization, the chat template, the
    strategy dispatch, the label construction and the jsonl side-effect are upstream's.
    NOTE: the attach is a python loop over materialised rows, NOT `Dataset.map` -- a row
    whose delimiter is untokenizable has to be COUNTED and excluded, and `.map` cannot drop
    rows.  The cost is that the split is held in memory as a list of dicts.
  * `DualTrackCollator` subclasses `MyCollator` (dataset.py:416-526) and overrides
    `__call__` only.  The left-pad amount is MEASURED from upstream's own mutation rather
    than recomputed, so upstream's alignment rule (dataset.py:438-468) can never drift out
    of sync with ours.

The upstream import is lazy so the pure helpers below stay importable and testable without
transformers >= 4.41; only the two entry points need the real stack.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .mask import TrackId
from .tracks import (
    DerivedTracks,
    TrackDerivationError,
    delim_candidates,
    derive_tracks,
    pad_track_row,
)


def attach_track_ids_fn(
    thinking_id: int, delim_candidate_ids: Sequence[Sequence[int]]
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """A `.map` function over upstream's per-example output.  Returns a NEW dict."""

    def attach(example: Dict[str, Any]) -> Dict[str, Any]:
        derived = derive_tracks(
            example["input_ids"], example["labels"], thinking_id, delim_candidate_ids
        )
        return {
            **example,
            "track_ids": list(derived.track_ids),
            "dt_q_end": derived.q_end,
            "dt_a_start": derived.a_start,
            "dt_v_start": derived.v_start,
        }

    return attach


def attach_track_ids(
    examples: Sequence[Dict[str, Any]],
    thinking_id: int,
    delim_candidate_ids: Sequence[Sequence[int]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]]]:
    """Row-wise version for the list branch, returning the failures rather than dropping
    them silently.  A tokenizer that merges the last CoT token into the answer template
    makes the delimiter unfindable; that surfaces here as a counted failure."""
    attach = attach_track_ids_fn(thinking_id, delim_candidate_ids)
    kept: List[Dict[str, Any]] = []
    failed: List[Tuple[int, str]] = []
    for position, example in enumerate(examples):
        try:
            kept.append(attach(example))
        except TrackDerivationError as exc:
            failed.append((int(example.get("idx", position)), str(exc)))
    return kept, failed


def measured_left_pads(before: Sequence[int], after: Sequence[int]) -> List[int]:
    """How many tokens upstream's collator prepended to each row.

    Upstream mutates the feature dicts in place (dataset.py:456-468); reading the length
    difference is what keeps the track padding tied to upstream's rule instead of to a
    second implementation of it.
    """
    if len(before) != len(after):
        raise ValueError("row counts changed inside the collator")
    lefts = [int(a) - int(b) for a, b in zip(after, before)]
    if any(left < 0 for left in lefts):
        raise ValueError(f"upstream shortened a row; that cannot happen: {lefts}")
    return lefts


def dualtrack_collator_class() -> type:
    """Build `DualTrackCollator(MyCollator)` lazily, so importing this module is cheap."""
    from ._upstream import import_data

    my_collator = import_data().dataset.MyCollator

    class DualTrackCollator(my_collator):  # type: ignore[misc,valid-type]
        """Overrides `__call__` only (dataset.py:423).

        `track_ids` MUST be stripped before `super()`: `pad_without_fast_tokenizer_warning`
        (dataset.py:484-490) passes unknown keys through unpadded, and `torch.tensor` on the
        resulting ragged list raises.
        """

        def __call__(self, features: Sequence[Dict[str, Any]], return_tensors: Any = None) -> Any:
            import torch

            local = [dict(feature) for feature in features]  # upstream mutates its input
            tracks = [list(feature.pop("track_ids")) for feature in local]
            for feature in local:
                for key in ("dt_q_end", "dt_a_start", "dt_v_start"):
                    feature.pop(key, None)
            before = [len(feature["input_ids"]) for feature in local]
            batch = super().__call__(local, return_tensors)
            after = [len(feature["input_ids"]) for feature in local]
            width = int(batch["input_ids"].shape[1])
            lefts = measured_left_pads(before, after)
            padded = [
                pad_track_row(track, left, width - left - len(track))
                for track, left in zip(tracks, lefts)
            ]
            batch["track_ids"] = torch.tensor(padded, dtype=torch.int64)
            valid = batch["track_ids"] != int(TrackId.PAD)
            if bool(((batch["attention_mask"] == 1) != valid).any()):
                raise ValueError("track_ids PAD and attention_mask disagree after collation")
            return batch

    return DualTrackCollator


def build_dualtrack_dataset(
    stage_type: str,
    base_dataset: Any,
    configs: Any,
    strategy: Any,
    tokenizer: Any,
    thinking_id: int,
    shuffle: bool = False,
    debug_num: Optional[int] = None,
    num_proc: int = 1,
) -> Tuple[Any, List[Tuple[int, str]]]:
    """upstream `get_cot_latent_dataset`, then one `.map` for `track_ids`.

    Note the inherited side effect: upstream APPENDS to
    `{dataset_save_path}/{name}_{stage}_dataset.jsonl` for the first 50 samples on rank 0
    (dataset.py:652-672), so that file grows across reruns.  Point `dataset_save_path` at
    `data/` (gitignored) and treat its length as a log, not as a dataset.
    """
    from ._upstream import import_data

    dataset_module = import_data().dataset
    produced = dataset_module.get_cot_latent_dataset(
        stage_type=stage_type,
        base_dataset=base_dataset,
        configs=configs,
        strategy=strategy,
        tokenizer=tokenizer,
        shuffle=shuffle,
        debug_num=debug_num,
    )
    candidates = delim_candidates(tokenizer)
    if isinstance(produced, list):
        return attach_track_ids(produced, thinking_id, candidates)
    kept, failed = attach_track_ids(
        [produced[i] for i in range(len(produced))], thinking_id, candidates
    )
    return kept, failed


# --- selftests -----------------------------------------------------------------------


def _fixture_features(thinking_id: int = 7) -> List[Dict[str, Any]]:
    delim = [20, 21]
    rows = [
        [1, 2, 3, 10, 11, thinking_id, 12, 13, thinking_id] + delim + [30, 31, 4],
        [1, 5, 14, thinking_id, 15, 16, thinking_id] + delim + [32, 4],
    ]
    features = []
    for idx, row in enumerate(rows):
        labels = [-100] * (3 - idx) + row[3 - idx :]
        derived: DerivedTracks = derive_tracks(row, labels, thinking_id, [delim])
        features.append(
            {
                "input_ids": list(row),
                "labels": labels,
                "attention_mask": [1] * len(row),
                "position_ids": list(range(len(row))),
                "idx": idx,
                "track_ids": list(derived.track_ids),
            }
        )
    return features


def _selftest_attach() -> None:
    features = [{k: v for k, v in f.items() if k != "track_ids"} for f in _fixture_features()]
    kept, failed = attach_track_ids(features, 7, [[20, 21]])
    assert not failed and len(kept) == 2
    for original, attached in zip(features, kept):
        assert "track_ids" not in original, "attach must not mutate its input"
        assert len(attached["track_ids"]) == len(attached["input_ids"])
        latents = [p for p, v in enumerate(attached["track_ids"]) if v == int(TrackId.LATENT)]
        assert latents == [p for p, t in enumerate(attached["input_ids"]) if t == 7]
    broken = dict(features[0])
    broken["input_ids"] = [t for t in broken["input_ids"] if t not in (20, 21)]
    broken["labels"] = broken["labels"][: len(broken["input_ids"])]
    _, failures = attach_track_ids([broken], 7, [[20, 21]])
    assert len(failures) == 1 and "delimiter" in failures[0][1], failures
    print(
        f"  attach_track_ids: {len(kept)} rows keyed to the <thinking> ids, an "
        f"unfindable delimiter is COUNTED not dropped silently: OK"
    )


def _selftest_measured_left_pads() -> None:
    assert measured_left_pads([5, 3], [5, 5]) == [0, 2]
    try:
        measured_left_pads([5], [4])
    except ValueError as exc:
        assert "shortened" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a shrinking row must raise")
    try:
        measured_left_pads([5, 3], [5])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a row-count change must raise")
    print("  left pads are measured from upstream's own mutation, never recomputed: OK")


def _selftest_collator_against_upstream() -> Optional[str]:
    """Real `MyCollator` subclass, or an honest UNAVAILABLE."""
    from ._upstream import UpstreamUnavailable

    from .mask import spans_from_track
    from .stub_tokenizer import StubTokenizer

    try:
        collator_cls = dualtrack_collator_class()
    except UpstreamUnavailable as exc:
        return str(exc)
    tokenizer = StubTokenizer()
    features = _fixture_features()
    collator = collator_cls(tokenizer=tokenizer, thinking_id=7, label_pad_token_id=-100)
    batch = collator([dict(f) for f in features])
    assert batch["track_ids"].shape == batch["input_ids"].shape == batch["labels"].shape
    firsts = [(row == 7).nonzero()[0].item() for row in batch["input_ids"]]
    assert len(set(firsts)) == 1, f"upstream must left-align the first latent; got {firsts}"
    valid = batch["track_ids"] != int(TrackId.PAD)
    assert bool(((batch["attention_mask"] == 1) == valid).all())
    for b, feature in enumerate(features):
        row = batch["track_ids"][b].tolist()
        spans = spans_from_track(row)
        original_first = feature["track_ids"].index(int(TrackId.LATENT))
        assert spans.pad_left == firsts[b] - original_first, (spans.pad_left, firsts[b])
        latents = [p for p, v in enumerate(row) if v == int(TrackId.LATENT)]
        assert latents == (batch["input_ids"][b] == 7).nonzero().flatten().tolist()
    assert "track_ids" in features[0], "the caller's features must not lose their track_ids"
    print(
        f"  DualTrackCollator over upstream MyCollator: left-align at {firsts[0]}, "
        f"both-side padding, PAD/attention agreement, latents still keyed to <thinking>: OK"
    )
    return None


def selftest() -> None:
    """The pure halves run anywhere; the collator needs upstream's dataset.py."""
    _selftest_attach()
    _selftest_measured_left_pads()
    unavailable = _selftest_collator_against_upstream()
    if unavailable is not None:
        print(f"  DualTrackCollator check UNAVAILABLE: {unavailable}")
        print("data.py selftest PASSED (pure half); upstream half SKIPPED (environment)")
        return
    print("data.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-track dataset/collator over upstream")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
