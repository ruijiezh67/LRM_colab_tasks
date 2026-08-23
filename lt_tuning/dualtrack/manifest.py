"""The run manifest.  Split out of train.py so its completeness check runs without torch.

Last round `bottleneck_mode` was silently dropped on the training path, which would have
made a whole strict run indistinguishable from a native one.  `REQUIRED_FIELDS` below is
the guard: `_selftest_completeness` fails if any of them leaves the dataclass.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

MANIFEST_NAME = "dualtrack_manifest.json"

# Anything whose absence makes a finished run uninterpretable.
REQUIRED_FIELDS: Tuple[str, ...] = (
    "bottleneck_mode",
    "mask_on",
    "cot_w",
    "ans_w",
    "latent_w",
    "four_d_mask_path",
    "four_d_mask_detail",
    "attn_impl",
    "transformers_version",
    "upstream_commit",
    "upstream_clean",
    "data_file_sha256",
)


@dataclass(frozen=True)
class Manifest:
    """What a checkpoint has to carry to be readable six months later."""

    stage: int
    stage_mode: str
    mask_on: bool
    bottleneck_mode: str
    cot_w: float
    ans_w: float
    latent_w: float
    thinking_strategy: str
    seed: int
    tokenizer_name: str
    thinking_token_id: int
    vocab_size: int
    data_file: str
    data_file_sha256: str
    n_examples: int
    n_track_derivation_failures: int
    four_d_mask_path: str
    four_d_mask_detail: str
    attn_impl: str
    transformers_version: str
    torch_version: str
    upstream_commit: str
    upstream_clean: bool

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: Path) -> Path:
        missing = [f for f in REQUIRED_FIELDS if getattr(self, f, None) is None]
        if missing:
            raise ValueError(f"refusing to write a manifest missing {missing}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / MANIFEST_NAME
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path


def read_manifest(ckpt_dir: Path) -> Dict[str, Any]:
    path = ckpt_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {ckpt_dir}; that checkpoint was not written here"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    return record


def _fixture() -> Manifest:
    return Manifest(
        stage=2,
        stage_mode="soft_fusion",
        mask_on=True,
        bottleneck_mode="strict",
        cot_w=1.0,
        ans_w=1.0,
        latent_w=0.0,
        thinking_strategy="arithmetic",
        seed=42,
        tokenizer_name="meta-llama/Llama-3.2-1B",
        thinking_token_id=128256,
        vocab_size=128257,
        data_file="data/dualtrack_train.jsonl",
        data_file_sha256="0" * 64,
        n_examples=7473,
        n_track_derivation_failures=0,
        four_d_mask_path="patch",
        four_d_mask_detail="patch: blocked delta 0.0",
        attn_impl="sdpa",
        transformers_version="4.55.4",
        torch_version="2.7.1",
        upstream_commit="c18aac695b33de135d3dd0848de0464d1b644ba7",
        upstream_clean=True,
    )


def _selftest_completeness() -> None:
    fields = set(Manifest.__dataclass_fields__)
    missing = [f for f in REQUIRED_FIELDS if f not in fields]
    assert not missing, f"the manifest dropped required fields: {missing}"
    assert "bottleneck_mode" in fields, (
        "bottleneck_mode must be in the manifest or a strict run is indistinguishable "
        "from a native one after the fact"
    )
    print(
        f"  manifest carries all {len(REQUIRED_FIELDS)} required fields "
        f"(of {len(fields)} total): OK"
    )


def _selftest_roundtrip() -> None:
    import tempfile

    manifest = _fixture()
    with tempfile.TemporaryDirectory(prefix="dt_manifest_") as tmp:
        out = Path(tmp)
        path = manifest.write(out)
        record = read_manifest(out)
        assert record["bottleneck_mode"] == "strict" and record["mask_on"] is True
        assert record == manifest.to_json()
        trimmed = {k: v for k, v in record.items() if k != "bottleneck_mode"}
        path.write_text(json.dumps(trimmed), encoding="utf-8")
        try:
            read_manifest(out)
        except ValueError as exc:
            assert "bottleneck_mode" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("a manifest without bottleneck_mode must be rejected on read")
    print("  write/read round-trip, and a stripped bottleneck_mode is rejected: OK")


def _selftest_mode_is_recorded_verbatim() -> None:
    """The recorded mode must be whatever the run used, never a re-derived default."""
    from dataclasses import replace

    for mode in ("native", "strict"):
        record = replace(_fixture(), bottleneck_mode=mode).to_json()
        assert record["bottleneck_mode"] == mode, record
    print("  both bottleneck modes survive into the manifest verbatim: OK")


def selftest() -> None:
    """CPU-only, no torch, no transformers."""
    _selftest_completeness()
    _selftest_roundtrip()
    _selftest_mode_is_recorded_verbatim()
    print("manifest.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manifest for dual-track LT-Tuning")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
