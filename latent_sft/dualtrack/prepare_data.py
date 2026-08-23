r"""Raw data -> the shared dual-track jsonl (+ optional upstream-schema sibling).

Output schema, one JSON object per line, exactly these keys::

    {"question": "...", "cot": "...", "answer": "50000"}

``answer`` is the BARE gold answer, the same convention colar/ and lt_tuning/
use. Latent-SFT's ``\boxed{}`` surface form is applied by ``common.format_answer``
at tokenization time, so the three folders' jsonl files stay interchangeable.

``--emit_upstream_view <path>`` additionally writes ``{"problem", "cot",
"cot_answer"}`` in IDENTICAL ROW ORDER, so upstream's stage-1 scripts
(script/run_distill_stage1_encoder.py and generate_latent_soft_label_hf_batch.py)
run unedited on our data. ``alignment.assert_upstream_view_matches`` re-derives
it, so the two files can never drift.

CLEAN SEMANTICS: ``cot`` is both the latent-distillation teacher and the visible
CoT, and ``answer`` is the gold answer. There is no second CoT field and no
poisoned subset -- see the README's "What is NOT implemented".

Row order is the pairing key for the latent tensors. Nothing downstream may
reorder these rows.

Two accepted ``--source_format`` values::

    latentsft   jsonl of {"problem", "cot", "cot_answer"}   (GSM8k-Aug-train.jsonl)
    gsm_steps   json array of {"question", "steps": [...], "answer"}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dualtrack.common import (
    BOXED_MARKER,
    COT_CLOSE,
    COT_OPEN,
    SHARED_REQUIRED_FIELDS,
    THINK_CLOSE,
    THINK_OPEN,
    env_path,
    extract_last_boxed,
    format_answer,
    read_jsonl,
    strip_think,
    upstream_view_row,
    write_jsonl,
)

SOURCE_FORMATS: Tuple[str, ...] = ("latentsft", "gsm_steps")
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_CHARS_PER_TOKEN = 3.5

_STEP_WRAPPER = re.compile(r"^<<(.+?)>>$")


@dataclass(frozen=True)
class PrepareConfig:
    source_format: str
    src: Path
    out: Path
    upstream_view: Optional[Path] = None
    limit: int = 0
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN


def clean_step(step: str) -> str:
    stripped = step.strip()
    match = _STEP_WRAPPER.match(stripped)
    if match:
        stripped = match.group(1)
    return stripped.replace("<<", "").replace(">>", "").strip()


def steps_to_cot(steps: Sequence[str]) -> str:
    lines = [clean_step(step) for step in steps if step and step.strip()]
    return "\n".join(line for line in lines if line)


def _from_latentsft(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    question = raw.get("problem") or raw.get("question")
    cot = raw.get("cot")
    cot_answer = raw.get("cot_answer")
    if not isinstance(question, str) or not isinstance(cot, str) or not isinstance(cot_answer, str):
        return None, "missing_field"
    boxed = extract_last_boxed(cot_answer)
    if boxed is None:
        return None, "answer_not_boxed"
    return {"question": question.strip(), "cot": strip_think(cot), "answer": boxed.strip()}, None


def _from_gsm_steps(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    question = raw.get("question") or raw.get("problem")
    steps = raw.get("steps")
    answer = raw.get("answer")
    if answer is None:
        answer = raw.get("gold")
    if not isinstance(question, str) or answer is None:
        return None, "missing_field"
    if not steps:
        return None, "no_gold_cot"
    cot = steps_to_cot(steps)
    if not cot:
        return None, "no_gold_cot"
    return {"question": question.strip(), "cot": cot, "answer": str(answer).strip()}, None


def to_dual_track(
    raw: Dict[str, Any], source_format: str
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if source_format == "latentsft":
        return _from_latentsft(raw)
    if source_format == "gsm_steps":
        return _from_gsm_steps(raw)
    raise ValueError(f"unknown source_format {source_format!r}; expected one of {SOURCE_FORMATS}")


def rejection_reason(row: Dict[str, str], config: PrepareConfig) -> Optional[str]:
    """Return why this row must be dropped, or None when it is usable."""
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in SHARED_REQUIRED_FIELDS):
        return "empty_field"
    if THINK_OPEN in row["cot"] or THINK_CLOSE in row["cot"]:
        return "think_tag_in_cot"
    if THINK_OPEN in row["question"] or THINK_CLOSE in row["question"]:
        # upstream/src/stage2/data.py:190-191 and generate_latent_soft_label_hf_batch.py:61-62
        # raise on this; drop it here, with a count, rather than mid-run.
        return "think_tag_in_question"
    if BOXED_MARKER in row["answer"]:
        # The shared field holds the bare answer; format_answer() would double-wrap it.
        return "answer_already_templated"
    joined = row["question"] + row["cot"] + format_answer(row["answer"])
    if COT_OPEN in joined or COT_CLOSE in joined:
        # An added-token splitter would inject a real delimiter and shift every span.
        return "delimiter_in_field"
    if len(joined) / config.chars_per_token > config.max_seq_len:
        return "too_long"
    return None


def load_raw(config: PrepareConfig) -> List[Dict[str, Any]]:
    if config.source_format == "gsm_steps":
        with open(config.src, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{config.src} is not a JSON array as gsm_steps requires")
        return data
    return read_jsonl(config.src)


def convert(
    raw_rows: Sequence[Dict[str, Any]], config: PrepareConfig
) -> Tuple[List[Dict[str, str]], Counter]:
    rejected: Counter = Counter()
    rows: List[Dict[str, str]] = []
    for raw in raw_rows:
        row, reason = to_dual_track(raw, config.source_format)
        if row is None:
            rejected[reason or "unknown"] += 1
            continue
        reason = rejection_reason(row, config)
        if reason is not None:
            rejected[reason] += 1
            continue
        rows.append(row)
    return rows, rejected


def run(config: PrepareConfig) -> Dict[str, Any]:
    raw_rows = load_raw(config)
    if config.limit:
        raw_rows = raw_rows[: config.limit]
    rows, rejected = convert(raw_rows, config)
    if not rows:
        raise ValueError(f"every row was rejected: {dict(rejected)}")
    write_jsonl(config.out, rows)
    summary: Dict[str, Any] = {
        "read": len(raw_rows),
        "written": len(rows),
        "rejected": dict(rejected),
        "out": str(config.out),
    }
    if config.upstream_view is not None:
        write_jsonl(config.upstream_view, [upstream_view_row(row) for row in rows])
        summary["upstream_view"] = str(config.upstream_view)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("example row:\n" + json.dumps(rows[0], ensure_ascii=False, indent=1))
    return summary


def _selftest_latentsft() -> None:
    config = PrepareConfig(source_format="latentsft", src=Path("x"), out=Path("y"))
    raw = [
        {"problem": "q1", "cot": "<think>a=1\nb=2</think>", "cot_answer": "so \\boxed{50000}"},
        {"problem": "q2", "cot": "c", "cot_answer": "no box"},
        {"problem": "q3", "cot": "", "cot_answer": "\\boxed{1}"},
        {"problem": "q4", "cot": "has <cot> inside", "cot_answer": "\\boxed{2}"},
        {"problem": "q5 </think>", "cot": "c", "cot_answer": "\\boxed{3}"},
    ]
    rows, rejected = convert(raw, config)
    assert len(rows) == 1, rows
    assert rows[0] == {"question": "q1", "cot": "a=1\nb=2", "answer": "50000"}
    assert set(rows[0].keys()) == set(SHARED_REQUIRED_FIELDS)
    assert format_answer(rows[0]["answer"]) == "\\boxed{50000}", "the template moved, not vanished"
    assert rejected["answer_not_boxed"] == 1
    assert rejected["empty_field"] == 1
    assert rejected["delimiter_in_field"] == 1
    assert rejected["think_tag_in_question"] == 1


def _selftest_gsm_steps() -> None:
    config = PrepareConfig(source_format="gsm_steps", src=Path("x"), out=Path("y"))
    raw = [
        {"question": "farm", "steps": ["<<4000*25=100000>>", "<<100000/2=50000>>"], "answer": "50000"},
        {"question": "no cot", "gold": "3"},
        {"question": "lisa", "steps": ["<<85-5=80>>"], "answer": "80"},
    ]
    rows, rejected = convert(raw, config)
    assert len(rows) == 2 and rejected["no_gold_cot"] == 1
    assert rows[0]["cot"] == "4000*25=100000\n100000/2=50000", repr(rows[0]["cot"])
    assert rows[0]["answer"] == "50000" and rows[1]["answer"] == "80"

    _, double = convert([{"question": "q", "steps": ["1+1=2"], "answer": "\\boxed{2}"}], config)
    assert double["answer_already_templated"] == 1, double


def _selftest_round_trip_and_view() -> None:
    import tempfile

    from dualtrack.alignment import assert_upstream_view_matches

    tiny = PrepareConfig(source_format="gsm_steps", src=Path("x"), out=Path("y"), max_seq_len=1)
    rows, rejected = convert([{"question": "q" * 100, "steps": ["1+1=2"], "answer": "2"}], tiny)
    assert not rows and rejected["too_long"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "raw.jsonl"
        out = Path(tmp) / "dualtrack_clean.jsonl"
        view = Path(tmp) / "upstream_view.jsonl"
        write_jsonl(
            src,
            [
                {"problem": "q1", "cot": "a=1", "cot_answer": "\\boxed{7}"},
                {"problem": "q2", "cot": "b=2", "cot_answer": "\\boxed{8}"},
            ],
        )
        summary = run(
            PrepareConfig(source_format="latentsft", src=src, out=out, upstream_view=view)
        )
        assert summary["written"] == 2
        assert read_jsonl(out)[0] == {"question": "q1", "cot": "a=1", "answer": "7"}
        assert assert_upstream_view_matches(out, view) == 2
        assert read_jsonl(view)[1] == {"problem": "q2", "cot": "b=2", "cot_answer": "\\boxed{8}"}


def selftest() -> None:
    _selftest_latentsft()
    _selftest_gsm_steps()
    _selftest_round_trip_and_view()
    assert SHARED_REQUIRED_FIELDS == ("question", "cot", "answer"), (
        "the shared cross-platform jsonl contract changed; colar/ and lt_tuning/ must follow"
    )
    print(
        "[prepare_data] OK -- both source formats convert, every invariant is counted rather than "
        "silently dropped, `answer` holds the BARE gold answer, and the upstream-schema view is "
        "row-for-row derivable from the shared jsonl."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_format", choices=SOURCE_FORMATS, default="latentsft")
    parser.add_argument("--src", default=str(env_path("DUALTRACK_RAW", "data/GSM8k-Aug-train.jsonl")))
    parser.add_argument("--out", default=str(env_path("DUALTRACK_DATA", "data/dualtrack_clean.jsonl")))
    parser.add_argument(
        "--emit_upstream_view",
        default=None,
        help="also write {problem,cot,cot_answer} here for upstream's stage-1 scripts",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--chars_per_token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help="character heuristic for the length guard; no tokenizer is loaded here",
    )
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    run(
        PrepareConfig(
            source_format=args.source_format,
            src=Path(args.src),
            out=Path(args.out),
            upstream_view=Path(args.emit_upstream_view) if args.emit_upstream_view else None,
            limit=args.limit,
            max_seq_len=args.max_seq_len,
            chars_per_token=args.chars_per_token,
        )
    )


if __name__ == "__main__":
    main()
