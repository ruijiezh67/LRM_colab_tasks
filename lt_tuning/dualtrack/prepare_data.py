"""Hop 0 -> hop 1: upstream GSM8K jsonl -> the shared cross-platform dual-track jsonl.

Reader: upstream `dataset.get_dataset(path, "gsm8k")` (dataset.py:379-413) whenever it
imports.  Its two `####` splitters are closures NESTED INSIDE that function
(dataset.py:389-395) and are not reachable by import, so they are re-typed below as the
torch-free fallback ONLY -- four lines, and `_selftest_reader_agreement` asserts the two
paths produce identical rows whenever upstream is importable.  That is COPY row #2 of two.

Output schema `{idx, question, cot, answer, n_steps, row_sha256}`.  The first four keys are
the SHARED contract (same meaning and same bare-answer convention as colar/ and
latent_sft/); `n_steps` and `row_sha256` are LT-Tuning extras, so the schemas are
compatible rather than identical.  Calculator annotations `<<48/2=24>>` are kept because
upstream keeps them.

Invariant 7: every row carries a sha256 over `(question, cot, answer)`, recomputed on read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .alignment import write_manifest
from .config import DataConfig

NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
FIELD_SEP = "␟"
SHARED_REQUIRED_FIELDS: Tuple[str, ...] = ("question", "cot", "answer")


def row_sha256(question: str, cot: str, answer: str) -> str:
    return hashlib.sha256(FIELD_SEP.join((question, cot, answer)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DualTrackRow:
    idx: int
    question: str
    cot: str
    answer: str
    n_steps: int
    row_sha256: str

    @staticmethod
    def create(idx: int, question: str, cot: str, answer: str) -> "DualTrackRow":
        steps = [s for s in cot.split("\n") if s.strip()]
        return DualTrackRow(
            idx, question, cot, answer, len(steps), row_sha256(question, cot, answer)
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "question": self.question,
            "cot": self.cot,
            "answer": self.answer,
            "n_steps": self.n_steps,
            "row_sha256": self.row_sha256,
        }


@dataclass(frozen=True)
class PrepareStats:
    split: str
    source: str
    reader: str
    n_in: int
    n_out: int
    dropped_empty_cot: int
    dropped_non_numeric: int

    def to_json(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _read_gsm8k_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """COPY (4 lines): shadows the two closures nested in upstream `get_dataset`
    (dataset.py:389-395).  They are local functions inside that function and cannot be
    imported.  Used ONLY when upstream's dataset.py will not import here, so that
    `--selftest` still runs; `_selftest_reader_agreement` pins the two together."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if max_rows is not None and len(records) >= max_rows:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no + 1} is not valid JSON") from exc
            records.append(
                {
                    "idx": len(records),
                    "question": raw["question"],
                    # upstream dataset.py:402
                    "reasoning_chain": "\n".join(raw["steps"])
                    if "steps" in raw
                    else str(raw["answer"]).split("####")[0].strip(),
                    # upstream dataset.py:403
                    "answer": raw["answer"]
                    if str(raw["answer"]).startswith("###")
                    else str(raw["answer"]).split("####")[-1].strip(),
                }
            )
    return records


def read_raw(path: Path, max_rows: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Upstream `get_dataset` when importable; the local fallback otherwise."""
    if not path.is_file():
        raise FileNotFoundError(
            f"upstream split not found: {path}\n"
            "Set DT_UPSTREAM_ROOT to the clone, or pass --source explicitly."
        )
    from ._upstream import UpstreamUnavailable, import_data

    try:
        get_dataset = import_data().dataset.get_dataset
    except UpstreamUnavailable:
        return _read_gsm8k_jsonl(path, max_rows), "local-fallback"
    dataset = get_dataset(str(path), dataset_name="gsm8k", max_size=max_rows or 1000000000)
    return [dict(dataset[i]) for i in range(len(dataset))], "upstream.get_dataset"


def convert(records: Sequence[Dict[str, Any]]) -> Tuple[List[DualTrackRow], Dict[str, int]]:
    """Upstream's own field names (`reasoning_chain`) -> the shared schema, with two filters.

    Upstream returns the raw string when the answer starts with `###`, which for a
    `{steps, answer: "#### 3"}` record makes the answer literally `"#### 3"` and then embeds
    it as `"### #### 3"`.  The hashes are stripped here; the numeric filter would otherwise
    drop every such row.
    """
    rows: List[DualTrackRow] = []
    drops = {"empty_cot": 0, "non_numeric": 0}
    for record in records:
        if "question" not in record or "answer" not in record:
            raise ValueError(f"record is missing question/answer: {sorted(record)}")
        cot = str(record.get("reasoning_chain", ""))
        answer = str(record["answer"]).split("####")[-1].lstrip("#").strip().replace(",", "")
        if not cot.strip():
            drops["empty_cot"] += 1
            continue
        if not NUMERIC_RE.match(answer):
            drops["non_numeric"] += 1
            continue
        rows.append(DualTrackRow.create(len(rows), str(record["question"]), cot, answer))
    return rows, drops


def write_jsonl(path: Path, rows: Sequence[DualTrackRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def read_dualtrack_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Invariant 7: every row's sha256 is recomputed and must match."""
    if not path.is_file():
        raise FileNotFoundError(f"dual-track jsonl not found: {path} (run prepare_data.py first)")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            expected = row_sha256(record["question"], record["cot"], record["answer"])
            if expected != record.get("row_sha256"):
                raise ValueError(
                    f"{path}:{line_no + 1} sha256 mismatch: file says {record.get('row_sha256')}, "
                    f"content hashes to {expected}.  Row counts can match while every row is "
                    "paired to the wrong question."
                )
            rows.append(record)
    return rows


def prepare(
    split: str,
    data_cfg: DataConfig,
    out_path: Path,
    source: Optional[Path],
    max_rows: Optional[int],
) -> PrepareStats:
    src = source or data_cfg.raw_split_path(split)
    records, reader = read_raw(src, max_rows)
    rows, drops = convert(records)
    write_jsonl(out_path, rows)
    # Invariant 7, file level: the sidecar is what the train -> verify hop checks.  It used to
    # be written by nothing, so `alignment.verify_manifest` had no manifest to verify against.
    write_manifest(out_path, n_rows=len(rows), source=src, split=split)
    return PrepareStats(
        split, str(src), reader, len(records), len(rows), drops["empty_cot"], drops["non_numeric"]
    )


# --- selftests -----------------------------------------------------------------------

_SOCRATIC = {
    "question": "Natalia sold clips to 48 friends in April, half as many in May. Total?",
    "answer": "How many in May? ** 48/2 = <<48/2=24>>24 clips.\nTotal? ** 48+24 = 72.\n#### 72",
}
_STEPPED = {"question": "q", "steps": ["a = 1", "b = 2"], "answer": "#### 3"}


def _selftest_conversion() -> None:
    records = _read_gsm8k_jsonl_from_records([_SOCRATIC, _STEPPED])
    rows, drops = convert(records)
    assert drops == {"empty_cot": 0, "non_numeric": 0}, drops
    assert rows[0].answer == "72" and rows[0].n_steps == 2
    assert "<<48/2=24>>" in rows[0].cot, "calculator annotations are kept, as upstream keeps them"
    assert rows[1].cot == "a = 1\nb = 2" and rows[1].answer == "3"
    bad = _read_gsm8k_jsonl_from_records(
        [{"question": "q", "answer": "\n#### 72"}, {"question": "q", "answer": "step\n#### twelve"}]
    )
    _, drops2 = convert(bad)
    assert drops2 == {"empty_cot": 1, "non_numeric": 1}, drops2
    emitted = rows[0].to_json()
    assert set(emitted) == {"idx", *SHARED_REQUIRED_FIELDS, "n_steps", "row_sha256"}, sorted(
        emitted
    )
    assert "####" not in emitted["answer"] and "\\boxed{" not in emitted["answer"], (
        "`answer` must be the bare gold answer; the platform answer template is applied at "
        "tokenization time, as in colar/ and latent_sft/"
    )
    print("  hop 0 -> hop 1 conversion, filters, shared schema keys: OK")


def _read_gsm8k_jsonl_from_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Round-trip the fixtures through the real file reader so the parser is what is tested."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dt_prepare_") as tmp:
        path = Path(tmp) / "raw.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return _read_gsm8k_jsonl(path)


def _selftest_sha_guard() -> None:
    import tempfile

    rows = [DualTrackRow.create(i, f"q{i}", f"step {i}\nstep {i + 1}", str(i)) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="dt_prepare_") as tmp:
        path = Path(tmp) / "dualtrack_selftest.jsonl"
        write_jsonl(path, rows)
        loaded = read_dualtrack_jsonl(path)
        assert [r["row_sha256"] for r in loaded] == [r.row_sha256 for r in rows]
        assert len(set(r["row_sha256"] for r in loaded)) == len(rows), "hashes must be distinct"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"answer": "2"', '"answer": "7"', 1),
            encoding="utf-8",
        )
        try:
            read_dualtrack_jsonl(path)
        except ValueError as exc:
            assert "sha256 mismatch" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("the sha256 alignment guard must reject an edited row")
    print(f"  row sha256: {len(rows)} distinct hashes, one edited byte is refused: OK")


def _selftest_reader_agreement() -> Optional[str]:
    """The 4-line copy must agree with upstream's own reader, or it is not a fallback."""
    import tempfile

    from ._upstream import UpstreamUnavailable, import_data

    try:
        get_dataset = import_data().dataset.get_dataset
    except UpstreamUnavailable as exc:
        return str(exc)
    with tempfile.TemporaryDirectory(prefix="dt_prepare_") as tmp:
        path = Path(tmp) / "raw.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in (_SOCRATIC, _STEPPED):
                handle.write(json.dumps(record) + "\n")
        upstream_rows = get_dataset(str(path), dataset_name="gsm8k")
        theirs = [dict(upstream_rows[i]) for i in range(len(upstream_rows))]
        ours = _read_gsm8k_jsonl(path)
    assert (
        theirs == ours
    ), f"the local fallback drifted from upstream get_dataset:\n{theirs}\n{ours}"
    assert convert(theirs)[0] == convert(ours)[0]
    print(f"  local fallback == upstream get_dataset on {len(ours)} rows: OK")
    return None


def selftest() -> None:
    """CPU-only, no network.  Writes only into a temporary directory."""
    _selftest_conversion()
    _selftest_sha_guard()
    unavailable = _selftest_reader_agreement()
    if unavailable is not None:
        print(f"  reader-agreement check UNAVAILABLE: {unavailable}")
        print("prepare_data.py selftest PASSED (pure half); upstream half SKIPPED (environment)")
        return
    print("prepare_data.py selftest PASSED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the shared dual-track jsonl (LT-Tuning)")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--source", type=Path, default=None, help="override the upstream jsonl path"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    data_cfg = DataConfig()
    out_path = args.out or data_cfg.dualtrack_path(args.split)
    stats = prepare(args.split, data_cfg, out_path, args.source, args.max_rows)
    stats_path = out_path.parent / "prepare_stats.json"
    existing: Dict[str, Any] = {}
    if stats_path.is_file():
        existing = json.loads(stats_path.read_text(encoding="utf-8"))
    existing[args.split] = stats.to_json()
    stats_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(stats.to_json(), indent=2))
    print(f"wrote {stats.n_out} rows -> {out_path}")


if __name__ == "__main__":
    main()
