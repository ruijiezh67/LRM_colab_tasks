r"""Dependency-light helpers shared by the dual-track layer. STDLIB ONLY.

Nothing here imports torch or transformers at module scope, so the data
converters, the alignment guard and the tokenizer shadow stay runnable on a
machine with no ML stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PACKAGE_DIR: Path = Path(__file__).resolve().parent
PLATFORM_DIR: Path = PACKAGE_DIR.parent

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
COT_OPEN = "<cot>"
COT_CLOSE = "</cot>"

BOXED_MARKER = "\\boxed{"

# The shared dual-track jsonl carries these three keys; the sibling folders
# colar/ and lt_tuning/ enforce the same set.
SHARED_REQUIRED_FIELDS = ("question", "cot", "answer")
# Upstream's own schema (upstream/src/stage2/data.py:42 and
# upstream/src/stage1/data.py:349-395), emitted by prepare_data --emit_upstream_view.
UPSTREAM_VIEW_FIELDS = ("problem", "cot", "cot_answer")

_WHITESPACE = re.compile(r"\s+")


def format_answer(gold_answer: str) -> str:
    r"""Apply this platform's answer template to the BARE gold answer.

    The shared jsonl stores the bare answer, exactly as colar/ and lt_tuning/ do.
    ``\boxed{}`` is Latent-SFT's answer surface form, so it is applied here at
    tokenization time rather than baked into the shared field.
    """
    return BOXED_MARKER + gold_answer + "}"


def upstream_view_row(row: Dict[str, Any]) -> Dict[str, str]:
    """One shared row rendered in upstream's schema, deterministically."""
    return {
        "problem": row["question"],
        "cot": row["cot"],
        "cot_answer": format_answer(row["answer"]),
    }


def env_path(env_var: str, default_relative: str) -> Path:
    """Resolve a path from ``env_var``, else relative to the platform folder.

    No absolute path is ever baked into the source; the folder is relocatable.
    """
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser()
    return PLATFORM_DIR / default_relative


def is_main_process() -> bool:
    """True on rank 0, and True whenever torch.distributed is not in play."""
    try:
        import torch.distributed as dist
    except ImportError:
        # No torch means no process group, hence a single logical process.
        return True
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    """Read a jsonl file, skipping (and counting) undecodable lines.

    Mirrors upstream/src/stage2/data.py:13-38. Note the asymmetry documented in
    the README: upstream's teacher generator (generate_latent_soft_label_hf_batch
    .py:420-425) does NOT skip, so a bad line shifts the row pairing. The sha256
    alignment guard is what catches that.
    """
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning("Skipping invalid JSON line %s in %s: %s", line_number, path, exc)
    logger.info("Loaded %s rows from %s; skipped %s invalid rows.", len(rows), path, skipped)
    return rows


def write_jsonl(path: os.PathLike | str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows in order; row order is the pairing key for the latent tensors."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_think(text: str) -> str:
    """Remove a leading ``<think>`` / trailing ``</think>`` wrapper.

    Same normalization as upstream/src/stage1/data.py:387-391 and
    upstream/generate_latent_soft_label_hf_batch.py:84-87.
    """
    value = text
    if value.startswith(THINK_OPEN):
        value = value[len(THINK_OPEN) :]
    if value.endswith(THINK_CLOSE):
        value = value[: -len(THINK_CLOSE)]
    return value.strip()


def extract_last_boxed(text: str) -> Optional[str]:
    r"""Return the content of the last ``\boxed{...}``, matching braces."""
    start = text.rfind(BOXED_MARKER)
    if start < 0:
        return None
    depth = 0
    for cursor in range(start + len(BOXED_MARKER) - 1, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(BOXED_MARKER) : cursor]
    return None


def normalize_answer(text: str) -> str:
    """Canonical form used by the V1/V2/V3 comparator."""
    inner = extract_last_boxed(text)
    value = inner if inner is not None else text
    value = value.strip().rstrip(".")
    for noise in ("$", ",", "\\!", "\\,", "\\$"):
        value = value.replace(noise, "")
    return _WHITESPACE.sub("", value)


def _as_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def answers_equal(left: str, right: str) -> bool:
    """String equality after normalization, with a numeric fallback."""
    lhs = normalize_answer(left)
    rhs = normalize_answer(right)
    if lhs == rhs:
        return True
    lhs_num = _as_float(lhs)
    rhs_num = _as_float(rhs)
    if lhs_num is None or rhs_num is None:
        return False
    return math.isclose(lhs_num, rhs_num, rel_tol=1e-9, abs_tol=1e-9)


def _selftest_answer_helpers() -> None:
    assert extract_last_boxed("x \\boxed{50000} y") == "50000"
    assert extract_last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_last_boxed("a \\boxed{1} b \\boxed{2}") == "2"
    assert extract_last_boxed("no box here") is None
    assert normalize_answer("The answer is \\boxed{50,000}.") == "50000"
    assert answers_equal("\\boxed{50000}", "50000")
    assert answers_equal("\\boxed{50.0}", "\\boxed{50}")
    assert not answers_equal("\\boxed{50000}", "\\boxed{12345}")
    assert strip_think("<think>abc</think>") == "abc"
    assert format_answer("7") == "\\boxed{7}"


def _selftest_paths_and_io() -> None:
    import tempfile

    assert env_path("DUALTRACK_NO_SUCH_VAR_12345", "data/x.jsonl") == PLATFORM_DIR / "data" / "x.jsonl"
    os.environ["DUALTRACK_TMP_TESTVAR"] = "/somewhere/else"
    try:
        assert env_path("DUALTRACK_TMP_TESTVAR", "data/x.jsonl") == Path("/somewhere/else")
    finally:
        del os.environ["DUALTRACK_TMP_TESTVAR"]

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nested" / "rows.jsonl"
        rows = [{"question": "q1", "cot": "c1", "answer": "1"}]
        write_jsonl(target, rows)
        assert read_jsonl(target) == rows
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        assert read_jsonl(target) == rows, "an undecodable line must be skipped, not crash"


def _selftest_upstream_view() -> None:
    row = {"question": "q", "cot": "a=1", "answer": "7"}
    view = upstream_view_row(row)
    assert tuple(view) == UPSTREAM_VIEW_FIELDS
    assert view == {"problem": "q", "cot": "a=1", "cot_answer": "\\boxed{7}"}


def selftest() -> None:
    _selftest_answer_helpers()
    _selftest_paths_and_io()
    _selftest_upstream_view()
    assert is_main_process() is True
    assert SHARED_REQUIRED_FIELDS == ("question", "cot", "answer")
    print(
        "[common] OK -- boxed extraction, answer comparator, relocatable env paths, jsonl "
        "round-trip with an undecodable line skipped, upstream-view rendering."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("common.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
