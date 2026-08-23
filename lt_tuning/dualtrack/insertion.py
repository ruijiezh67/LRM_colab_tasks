"""Clamp upstream's insertion strategies to the reasoning region.  One method overridden.

The previous round shipped a 339-line `insertion.py` whose header said "Copied verbatim
from dataset.py:56-71".  Upstream's `ThinkingTokenStrategy` and its three subclasses are
imported here instead; the only thing added is a mixin over the single abstract method
`_candidate_indices` (dataset.py:50-54).

Why it is needed:

* `ArithmeticThinkingStrategy` scores every numeric token in the whole response and
  explicitly appends the index AFTER `###` (dataset.py:208-209), i.e. upstream will insert
  latents INSIDE the ANSWER span.  `tracks.derive_tracks` cannot accept that layout and
  raises; clamping is what keeps the pipeline emittable.
* `RandomThinkingStrategy._candidate_indices(self, flat_steps, sample)` (dataset.py:149-150)
  does not match the ABC's `(self, sample)`, and `apply` calls it as
  `self._candidate_indices(sample=sample)` (dataset.py:110) -> `TypeError: missing
  positional argument 'flat_steps'`.  `thinking_strategy: random` therefore cannot run
  upstream at all.  The mixin supplies `flat_steps` rather than editing upstream.

The upstream import is deliberately lazy so this module - and the pure clamp predicate that
carries the logic - stay importable and testable without transformers >= 4.41.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple


def clamp_candidates(candidates: Sequence[int], question_len: int, delim_start: int) -> List[int]:
    """The predicate itself: keep only insertion points inside `[question_len, delim_start)`.

    Pure python and separately testable, because it is the part that decides whether a row
    is emittable at all.
    """
    if question_len < 0 or delim_start < question_len:
        raise ValueError(f"bad reasoning window [{question_len}, {delim_start})")
    return [int(i) for i in candidates if question_len <= int(i) < delim_start]


def delim_start_from_sample(sample: Dict[str, Any], decode: Any) -> int:
    """Where the ANSWER template begins, in token positions of `full_tokenized`.

    Upstream appends `"The final answer is:\\n### {answer}"` last (dataset.py:592), and its
    own arithmetic strategy already locates `###` by decoding one token at a time
    (dataset.py:208).  The same probe is used here so the two agree by construction; if no
    `###` token is found the whole response counts as reasoning, which is the safe direction
    (nothing gets clamped that should not).
    """
    full = list(sample.get("full_tokenized", []))
    for idx in range(len(full) - 1, -1, -1):
        if decode([full[idx]]).strip() == "###":
            return idx
    return len(full)


class ReasoningRegionOnly:
    """Mixin over `ThinkingTokenStrategy._candidate_indices` (dataset.py:50-54).

    Composed onto whatever `build_thinking_strategy` returned, so upstream's 65-line
    dispatch (dataset.py:311-376) is called rather than re-typed.
    """

    def _candidate_indices(self, sample: Dict[str, Any]) -> List[int]:  # type: ignore[override]
        base = super()._candidate_indices  # type: ignore[misc]
        try:
            raw = base(sample=sample)
        except TypeError:
            # RandomThinkingStrategy's signature is (self, flat_steps, sample); upstream's
            # own `apply` cannot call it.  Supply the argument instead of editing upstream.
            raw = base(flat_steps=list(sample.get("full_tokenized", [])), sample=sample)
        question_len = len(sample.get("question_tokenized", []))
        delim_start = delim_start_from_sample(sample, self._dt_decode)  # type: ignore[attr-defined]
        return clamp_candidates(raw, question_len, delim_start)


def clamp_to_reasoning(strategy: Any, decode: Any) -> Any:
    """Return a COPY of `strategy` whose class also inherits `ReasoningRegionOnly`.

    A copy, so the caller's object is never mutated and an unclamped strategy stays
    available for comparison.  Three lines of class synthesis instead of re-typing
    `build_thinking_strategy`.
    """
    clamped = copy.copy(strategy)
    base = type(strategy)
    if issubclass(base, ReasoningRegionOnly):
        clamped._dt_decode = decode
        return clamped
    clamped.__class__ = type(f"ReasoningRegionOnly{base.__name__}", (ReasoningRegionOnly, base), {})
    clamped._dt_decode = decode
    return clamped


def build_clamped_strategy(
    configs: Any, tokenizer: Any, thinking_token_id: int, model: Any = None
) -> Any:
    """upstream `build_thinking_strategy` (dataset.py:311-376), then the clamp."""
    from ._upstream import import_data

    build = import_data().dataset.build_thinking_strategy
    strategy = build(
        configs=configs, tokenizer=tokenizer, thinking_token_id=thinking_token_id, model=model
    )
    return clamp_to_reasoning(strategy, lambda ids: tokenizer.decode(ids))


# --- selftests -----------------------------------------------------------------------


def _selftest_clamp_predicate() -> None:
    assert clamp_candidates([0, 2, 4, 9, 11, 14], 3, 10) == [4, 9]
    assert clamp_candidates([], 0, 0) == []
    assert clamp_candidates([3], 3, 4) == [3], "the window is half-open at the right only"
    assert clamp_candidates([10], 3, 10) == [], "delim_start itself is not a reasoning position"
    try:
        clamp_candidates([1], 5, 2)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an inverted window must raise")
    print(
        "  clamp predicate: [question_len, delim_start), half-open, refuses an "
        "inverted window: OK"
    )


def _selftest_delim_probe() -> None:
    from .stub_tokenizer import StubTokenizer

    tok = StubTokenizer()
    ids = tok.encode("## Step 1: 2 + 2 = 4\nThe final answer is:\n### 4")
    sample = {"full_tokenized": ids, "question_tokenized": []}
    delim = delim_start_from_sample(sample, lambda i: tok.decode(i))
    assert 0 < delim < len(ids), (delim, len(ids))
    assert tok.decode([ids[delim]]).strip() == "###"
    tail = [i for i in range(delim, len(ids))]
    assert clamp_candidates(tail, 0, delim) == [], "nothing at or after ### may be a candidate"
    missing = delim_start_from_sample(
        {"full_tokenized": tok.encode("no delimiter here")}, lambda i: tok.decode(i)
    )
    assert missing == len(
        tok.encode("no delimiter here")
    ), "with no ### found, the whole response must count as reasoning (clamp nothing)"
    print(
        f"  ### probe finds the answer template at token {delim}/{len(ids)}; every later "
        f"index is clamped away: OK"
    )


def _selftest_mixin_against_upstream() -> Optional[str]:
    """Composed onto upstream's REAL strategy classes, or an honest UNAVAILABLE."""
    from ._upstream import UpstreamUnavailable, import_data

    from .stub_tokenizer import StubTokenizer

    try:
        dataset = import_data().dataset
    except UpstreamUnavailable as exc:
        return str(exc)
    tok = StubTokenizer()
    thinking_id = tok.convert_tokens_to_ids("<thinking>")
    question = tok.encode("How many boxes?")
    body = tok.encode("## Step 1: 2 + 2 = 4\nThe final answer is:\n### 4")
    sample = {
        "question": "How many boxes?",
        "reasoning_chain": "2 + 2 = 4",
        "answer": "4",
        "question_tokenized": question,
        "response_tokenized": body,
        "full_tokenized": question + body,
        "idx": 0,
    }
    checked = []
    for cls in (dataset.ArithmeticThinkingStrategy, dataset.RandomThinkingStrategy):
        strategy = cls(tok, thinking_id, tokens_per_stage=4, insertion_prob=1.0, seed=1)
        clamped = clamp_to_reasoning(strategy, lambda ids: tok.decode(ids))
        got = clamped._candidate_indices(sample=sample)
        delim = delim_start_from_sample(sample, lambda ids: tok.decode(ids))
        assert all(len(question) <= i < delim for i in got), (cls.__name__, got, delim)
        assert type(strategy) is cls, "the original strategy must not be mutated"
        ids, positions = clamped.apply(sample, scheduled_stage=1)
        assert all(len(question) <= p < delim + len(positions) for p in positions), positions
        checked.append(f"{cls.__name__}({len(got)} candidates)")
    unclamped = dataset.ArithmeticThinkingStrategy(tok, thinking_id, tokens_per_stage=4, seed=1)
    raw = unclamped._candidate_indices(sample=sample)
    delim = delim_start_from_sample(sample, lambda ids: tok.decode(ids))
    assert any(i >= delim for i in raw), (
        "upstream's arithmetic strategy is supposed to propose an index at/after ###; if it "
        "no longer does, the clamp is untested"
    )
    print(
        f"  mixin over upstream's real classes: {', '.join(checked)}; unclamped upstream "
        f"proposes {sum(1 for i in raw if i >= delim)} answer-region index(es) that the "
        f"clamp removes: OK"
    )
    return None


def selftest() -> None:
    """The clamp predicate and the ### probe run anywhere; the mixin needs upstream's
    dataset.py (transformers >= 4.41) and says so when it cannot run."""
    _selftest_clamp_predicate()
    _selftest_delim_probe()
    unavailable = _selftest_mixin_against_upstream()
    if unavailable is not None:
        print(f"  mixin-over-upstream check UNAVAILABLE: {unavailable}")
        print("insertion.py selftest PASSED (pure half); upstream half SKIPPED (environment)")
        return
    print("insertion.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clamp upstream insertion to the reasoning span")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
