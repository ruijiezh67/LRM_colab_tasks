"""sha256 alignment guard between a dual-track jsonl and anything derived from it.

Nothing in this platform pairs a tensor to a jsonl row by index (latents are
recomputed from the ``cot`` field every step), but the run -> verify hop still
pairs a checkpoint to a data file, and an equal-length reshuffle of the source
JSON renumbers every row silently (``prepare_data.convert`` assigns ``idx`` by
enumeration over the raw records).

Two levels, both live: ``prepare_data`` writes the ``<data>.manifest.json`` sidecar
here, and ``train``/``verify`` call ``verify_manifest_if_present`` before reading.
The per-row guard is separate and unconditional -- ``prepare_data.read_dualtrack_jsonl``
recomputes every row's ``row_sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SUFFIX = ".manifest.json"
_READ_CHUNK = 1 << 20


@dataclass(frozen=True)
class DataManifest:
    """Sidecar describing the exact bytes a downstream artifact was built from."""

    data_path: str
    data_sha256: str
    n_rows: int
    source: str
    split: str
    created: str


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(data_path: Path) -> Path:
    return Path(str(data_path) + MANIFEST_SUFFIX)


def write_manifest(data_path: Path, n_rows: int, source: Path, split: str) -> Path:
    """Write ``<data>.manifest.json`` next to the jsonl and return its path."""
    manifest = DataManifest(
        data_path=str(data_path.resolve()),
        data_sha256=sha256_file(data_path),
        n_rows=n_rows,
        source=str(Path(source).resolve()),
        split=split,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    out = manifest_path_for(data_path)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, ensure_ascii=False, indent=2)
    return out


def read_manifest(path: Path) -> DataManifest:
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    missing = [f for f in DataManifest.__annotations__ if f not in record]
    if missing:
        raise ValueError(f"manifest {path} is missing fields: {missing}")
    return DataManifest(**{f: record[f] for f in DataManifest.__annotations__})


def verify_digest(data_path: Path, expected_sha256: str, context: str) -> str:
    """Raise ValueError on mismatch, printing both digests; return a status line."""
    actual = sha256_file(data_path)
    if actual != expected_sha256:
        raise ValueError(
            f"data MISMATCH ({context}).\n"
            f"  expected sha256 {expected_sha256}\n"
            f"  actual   sha256 {actual} (from {data_path.resolve()})\n"
            "Row counts can match while every row is paired to the wrong question."
        )
    return f"alignment OK (sha256 {actual[:12]}...)"


def verify_manifest(data_path: Path) -> str:
    """Check a jsonl against its own sidecar manifest."""
    manifest = read_manifest(manifest_path_for(data_path))
    return verify_digest(data_path, manifest.data_sha256, f"manifest of {data_path.name}")


def verify_manifest_if_present(data_path: Path) -> str:
    """`verify_manifest` for consumers that must also accept a jsonl produced before the
    sidecar existed.  A PRESENT-but-wrong manifest still raises; only absence is tolerated,
    so this can never turn a real mismatch into a pass."""
    sidecar = manifest_path_for(data_path)
    if not sidecar.is_file():
        return f"no sidecar next to {data_path.name}; file-level digest not checked"
    return verify_manifest(data_path)


def _selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "train.jsonl"
        data.write_text(
            '{"idx": 0, "question": "q", "cot": "c", "answer": "1"}\n', encoding="utf-8"
        )
        written = write_manifest(data, n_rows=1, source=root / "train.json", split="train")
        assert written == manifest_path_for(data)

        manifest = read_manifest(written)
        assert manifest.n_rows == 1 and manifest.split == "train"
        assert manifest.data_sha256 == sha256_file(data)
        print("[selftest] manifest round-trip OK:", verify_manifest(data))

        data.write_text(
            '{"idx": 0, "question": "q", "cot": "c", "answer": "2"}\n', encoding="utf-8"
        )
        try:
            verify_manifest(data)
        except ValueError as err:
            assert "MISMATCH" in str(err)
            print("[selftest] tamper detected OK (one byte changed -> refuse)")
        else:
            raise AssertionError("verify_manifest accepted tampered data")

        empty = root / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert sha256_file(empty) == hashlib.sha256(b"").hexdigest()

        # `verify_manifest_if_present` must tolerate ABSENCE and still refuse a WRONG sidecar,
        # or wiring it into train/verify would have silently weakened the guard.
        loose = root / "no_sidecar.jsonl"
        loose.write_text('{"idx": 0}\n', encoding="utf-8")
        assert "not checked" in verify_manifest_if_present(loose)
        try:
            verify_manifest_if_present(data)  # `data` was tampered with above
        except ValueError as err:
            assert "MISMATCH" in str(err)
        else:
            raise AssertionError("a present-but-wrong sidecar must still raise")
        print("[selftest] verify_manifest_if_present: absent -> skip, wrong -> refuse OK")
    print("[selftest] alignment.py OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--verify", type=Path, default=None, help="jsonl to check against its manifest"
    )
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return 0
    if args.verify is not None:
        print(verify_manifest(args.verify))
        return 0
    parser.error("nothing to do: pass --selftest or --verify")
    return 2


if __name__ == "__main__":
    sys.exit(main())
