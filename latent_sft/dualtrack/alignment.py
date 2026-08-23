r"""sha256 alignment guard between the dual-track jsonl and everything derived from it.

Three independent hazards, three checks, all stdlib:

``write_alignment`` / ``verify_alignment``
    The latent ``.pt`` chunks pair to the jsonl by ROW INDEX only. An
    equal-length reshuffle of the source renumbers every row and trains without
    a single error, so the sidecar pins the exact bytes.

``assert_contiguous_chunk_cover``
    Upstream's loader (upstream/src/stage2/data.py:95-117) globs ``batch_*.pt``
    and concatenates them in start order without checking for gaps, so a stale
    chunk from a longer previous run silently extends the dataset. This is a
    filename-only function deliberately placed BESIDE the inherited loader
    rather than overriding it.

``assert_upstream_view_matches``
    The stage-1 teacher scripts run on the upstream-schema jsonl. If it ever
    drifts from the shared one, the latents describe different questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dualtrack import UPSTREAM_COMMIT
from dualtrack.common import read_jsonl, upstream_view_row

ALIGNMENT_FILE = "alignment.json"
CHUNK_PREFIX = "batch_"
CHUNK_SUFFIX = ".pt"
_READ_CHUNK = 1 << 20


def file_sha256(path: os.PathLike | str) -> str:
    """Streaming sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_git_sha() -> str:
    """The LIVE sha of ``../upstream``, or a reason string -- never a silent blank.

    Recorded next to :data:`dualtrack.UPSTREAM_COMMIT` (the pin) so a clone that has moved
    off the pin shows up in the artefact.  colar/ records the same pair in its
    ``run_manifest.json`` and lt_tuning/ in its training manifest.
    """
    root = Path(__file__).resolve().parent.parent / "upstream"
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as err:
        return f"<unavailable: {err}>"
    return done.stdout.strip()


def write_alignment(
    save_path: os.PathLike | str,
    data_path: os.PathLike | str,
    n_rows: int,
    latent_lens: Sequence[int],
    **extra: Any,
) -> Dict[str, Any]:
    """Pin the jsonl these latent chunks were generated from."""
    record: Dict[str, Any] = {
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": file_sha256(data_path),
        "n_rows": n_rows,
        "latent_lens": list(latent_lens),
        "upstream_commit_pin": UPSTREAM_COMMIT,
        "upstream_git_sha": upstream_git_sha(),
        **extra,
    }
    target = Path(save_path)
    target.mkdir(parents=True, exist_ok=True)
    with open(target / ALIGNMENT_FILE, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def verify_alignment(
    save_path: os.PathLike | str,
    data_path: os.PathLike | str,
    allow_missing: bool = False,
) -> Dict[str, Any]:
    """Raise unless the latent chunks were built from exactly ``data_path``."""
    sidecar = Path(save_path) / ALIGNMENT_FILE
    if not sidecar.exists():
        message = f"no {ALIGNMENT_FILE} in {save_path}; cannot prove the latents came from {data_path}"
        if allow_missing:
            return {}
        raise FileNotFoundError(message + " -- rerun make_latents.py, or pass --allow_missing_alignment")
    with open(sidecar, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    actual = file_sha256(data_path)
    if record.get("data_sha256") != actual:
        raise ValueError(
            "latent/data MISMATCH -- "
            f"{save_path} was built from a different jsonl.\n"
            f"  expected sha256 {record.get('data_sha256')} (from {record.get('data_path')})\n"
            f"  actual   sha256 {actual} (from {Path(data_path).resolve()})\n"
            "Row counts can match while every row is paired to the wrong question. "
            "Rerun make_latents.py on THIS jsonl."
        )
    return record


def chunk_bounds(path: os.PathLike | str) -> Tuple[int, int]:
    """``batch_<start>_<end>.pt`` -> (start, end); the naming upstream writes."""
    name = Path(path).name
    stem = Path(path).stem
    parts = stem.split("_")
    if not name.startswith(CHUNK_PREFIX) or not name.endswith(CHUNK_SUFFIX):
        raise ValueError(f"Invalid latent chunk filename: {name} (want batch_<start>_<end>.pt)")
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(f"Invalid latent chunk filename: {name} (want batch_<start>_<end>.pt)")
    return int(parts[1]), int(parts[2])


def assert_contiguous_chunk_cover(soft_dir: os.PathLike | str) -> List[Path]:
    """Filenames alone must describe a gap-free cover of ``[0, n)``."""
    directory = Path(soft_dir)
    files = sorted(
        (path for path in directory.glob(f"{CHUNK_PREFIX}*{CHUNK_SUFFIX}")),
        key=chunk_bounds,
    )
    if not files:
        raise FileNotFoundError(f"No latent chunk files matching batch_*.pt in {directory}")
    cursor = 0
    for path in files:
        start, end = chunk_bounds(path)
        if start != cursor:
            raise ValueError(
                f"latent chunks are not a contiguous cover: expected a chunk starting at {cursor}, "
                f"got {path.name}. A stale chunk from a longer previous run silently extends the "
                "dataset and shifts every later row."
            )
        if end <= start:
            raise ValueError(f"{path.name} declares an empty or reversed range")
        cursor = end
    return files


def assert_upstream_view_matches(
    shared_path: os.PathLike | str, upstream_path: os.PathLike | str
) -> int:
    """Re-derive the upstream-schema jsonl and compare it row for row."""
    shared_rows = read_jsonl(shared_path)
    upstream_rows = read_jsonl(upstream_path)
    if len(shared_rows) != len(upstream_rows):
        raise ValueError(
            f"upstream view has {len(upstream_rows)} rows but the shared jsonl has "
            f"{len(shared_rows)}; the stage-1 teacher would index different questions"
        )
    for index, (shared, actual) in enumerate(zip(shared_rows, upstream_rows)):
        expected = upstream_view_row(shared)
        if expected != actual:
            raise ValueError(
                f"row {index} of the upstream view is not derivable from the shared jsonl.\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "Regenerate it with prepare_data.py --emit_upstream_view."
            )
    return len(shared_rows)


def _selftest_manifest_round_trip(root: Path) -> None:
    from dualtrack.common import write_jsonl

    data = root / "train.jsonl"
    soft = root / "soft"
    write_jsonl(data, [{"question": "q1", "cot": "c1", "answer": "1"},
                       {"question": "q2", "cot": "c2", "answer": "2"}])
    soft.mkdir(parents=True, exist_ok=True)
    try:
        verify_alignment(soft, data)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing sidecar must be a hard error by default")
    assert verify_alignment(soft, data, allow_missing=True) == {}

    record = write_alignment(soft, data, n_rows=2, latent_lens=[3, 4], teacher="proxy_decoder")
    assert record["n_rows"] == 2 and len(record["data_sha256"]) == 64
    assert verify_alignment(soft, data)["latent_lens"] == [3, 4]

    write_jsonl(data, [{"question": "q2", "cot": "c2", "answer": "2"},
                       {"question": "q1", "cot": "c1", "answer": "1"}])
    try:
        verify_alignment(soft, data)
    except ValueError as exc:
        assert "MISMATCH" in str(exc)
    else:
        raise AssertionError("an equal-length reshuffle slipped past the alignment guard")


def _selftest_chunk_cover(root: Path) -> None:
    soft = root / "chunks"
    soft.mkdir(parents=True, exist_ok=True)
    (soft / "batch_0_2.pt").write_bytes(b"")
    (soft / "batch_2_5.pt").write_bytes(b"")
    assert [path.name for path in assert_contiguous_chunk_cover(soft)] == ["batch_0_2.pt", "batch_2_5.pt"]
    (soft / "batch_7_8.pt").write_bytes(b"")
    try:
        assert_contiguous_chunk_cover(soft)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("a non-contiguous chunk cover was accepted")
    assert chunk_bounds("x/batch_10_20.pt") == (10, 20)
    for bad in ("batch_a_2.pt", "chunk_0_2.pt", "batch_0.pt"):
        try:
            chunk_bounds(bad)
        except ValueError:
            continue
        raise AssertionError(f"chunk_bounds accepted {bad!r}")


def _selftest_upstream_view(root: Path) -> None:
    from dualtrack.common import write_jsonl

    shared = root / "shared.jsonl"
    view = root / "view.jsonl"
    rows = [{"question": f"q{i}", "cot": f"c{i}", "answer": str(i)} for i in range(3)]
    write_jsonl(shared, rows)
    write_jsonl(view, [upstream_view_row(row) for row in rows])
    assert assert_upstream_view_matches(shared, view) == 3

    write_jsonl(view, [upstream_view_row(row) for row in reversed(rows)])
    try:
        assert_upstream_view_matches(shared, view)
    except ValueError as exc:
        assert "not derivable" in str(exc)
    else:
        raise AssertionError("a reordered upstream view was accepted")


def selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _selftest_manifest_round_trip(root)
        _selftest_chunk_cover(root)
        _selftest_upstream_view(root)
    print(
        "[alignment] OK -- sha256 sidecar round-trip, equal-length reshuffle detected, "
        "chunk-cover contiguity, upstream-view derivability."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--verify_soft", type=Path, default=None, help="latent chunk dir")
    parser.add_argument("--data", type=Path, default=None, help="jsonl the chunks must match")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if args.verify_soft is not None and args.data is not None:
        assert_contiguous_chunk_cover(args.verify_soft)
        record = verify_alignment(args.verify_soft, args.data)
        print(f"alignment OK ({record['n_rows']} rows, sha256 {record['data_sha256'][:12]}...)")
        return 0
    parser.error("nothing to do: pass --selftest, or --verify_soft together with --data")
    return 2


if __name__ == "__main__":
    sys.exit(main())
