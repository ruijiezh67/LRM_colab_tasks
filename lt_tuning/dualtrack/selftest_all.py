"""Run every module's `--selftest` and report RUN / SKIPPED / FAILED per module.

A module whose upstream half cannot run on this machine prints its own
`UNAVAILABLE: <real error>` and exits 0 with a `SKIPPED (environment)` marker.  That is
counted separately from a pass here, so nothing is ever reported as green that did not
actually execute.  Exit status is non-zero only on a real failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

PACKAGE = "dualtrack"
PLATFORM_DIR = Path(__file__).resolve().parent.parent

# Tier 0: no torch, no transformers.  Tier 1: torch + upstream model.py.
# Tier 2: needs upstream dataset.py / run.py (transformers >= 4.41).
MODULES: Tuple[Tuple[str, int], ...] = (
    ("_upstream", 0),
    ("mask", 0),
    ("tracks", 0),
    ("stub_tokenizer", 0),
    ("config", 0),
    ("manifest", 0),
    ("alignment", 0),
    ("prepare_data", 0),
    ("attention_backend", 1),
    ("loss", 1),
    ("model", 1),
    ("generate", 1),
    ("verify", 1),
    ("insertion", 2),
    ("data", 2),
    ("train", 2),
)

SKIP_MARKER = "SKIPPED (environment)"


@dataclass(frozen=True)
class Result:
    module: str
    tier: int
    status: str
    output: str

    @property
    def ok(self) -> bool:
        return self.status in ("RUN", "SKIPPED")


def run_module(module: str, tier: int, python: str) -> Result:
    proc = subprocess.run(
        [python, "-m", f"{PACKAGE}.{module}", "--selftest"],
        cwd=str(PLATFORM_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return Result(module, tier, "FAILED", output)
    if SKIP_MARKER in output:
        return Result(module, tier, "SKIPPED", output)
    return Result(module, tier, "RUN", output)


def summarise(results: Sequence[Result]) -> str:
    lines = ["", "=" * 72, "SELFTEST SUMMARY", "=" * 72]
    for result in results:
        lines.append(f"  tier{result.tier}  {result.status:8s}  {PACKAGE}.{result.module}")
    counts = {
        status: sum(1 for r in results if r.status == status)
        for status in ("RUN", "SKIPPED", "FAILED")
    }
    lines.append("-" * 72)
    lines.append(
        f"  {counts['RUN']} ran, {counts['SKIPPED']} skipped for environment reasons, "
        f"{counts['FAILED']} failed"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every dual-track selftest")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--quiet", action="store_true", help="summary only")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    selected = [(m, t) for m, t in MODULES if args.only is None or m in args.only]
    results: List[Result] = []
    for module, tier in selected:
        print(f"\n----- {PACKAGE}.{module} --selftest (tier {tier}) -----")
        result = run_module(module, tier, args.python)
        results.append(result)
        if not args.quiet or result.status == "FAILED":
            print(result.output)
        print(f"[{result.status}] {PACKAGE}.{module}")
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
