r"""Run every module's ``--selftest`` and report honestly what this machine can do.

Each module prints PASS or ``SKIP(env): <the actual exception text>``. A module
whose imports fail is never reported as passing -- the verbatim ImportError goes
to stdout and the exit code stays 0 only when every tier-0 module passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

# (module, tier). Tier 0 = stdlib only; 1 = needs torch; 2 = needs torch +
# transformers >= 4.51.1 + peft + the vendored upstream.
MODULES: Tuple[Tuple[str, int], ...] = (
    ("dualtrack.spans", 0),
    ("dualtrack.common", 0),
    ("dualtrack.alignment", 0),
    ("dualtrack.stub_tokenizer", 0),
    ("dualtrack.tokenize_dualtrack", 0),
    ("dualtrack.prepare_data", 0),
    ("dualtrack.eval_accuracy", 0),
    ("dualtrack.mask", 1),
    ("dualtrack.dist_bootstrap", 1),
    ("dualtrack.attention_probe", 1),
    ("dualtrack.generate_dualtrack", 1),
    ("dualtrack.upstream_api", 2),
    ("dualtrack.model_dualtrack", 2),
    ("dualtrack.data_dualtrack", 2),
    ("dualtrack.arguments", 2),
    ("dualtrack.train_dualtrack", 2),
    ("dualtrack.make_latents", 2),
    ("dualtrack.verify_dualtrack", 2),
)

IMPORT_MARKERS = ("ModuleNotFoundError:", "ImportError:")


def _run(module: str) -> Tuple[int, str]:
    process = subprocess.run(
        [sys.executable, "-m", module, "--selftest"], capture_output=True, text=True
    )
    output = (process.stdout + process.stderr).strip()
    return process.returncode, output


def _last_error_line(output: str) -> str:
    """The line that terminated the process: the last non-empty one."""
    for line in reversed(output.splitlines()):
        if line.strip():
            return line.strip()
    return "(no output)"


def _is_missing_dependency(output: str) -> bool:
    """SKIP only when the TERMINATING exception is a missing import.

    Substring-matching the whole output downgraded any genuine failure whose
    traceback merely mentioned an import error -- an AssertionError was reported
    as ``SKIP (environment)`` and the run still exited 0.
    """
    return _last_error_line(output).startswith(IMPORT_MARKERS)


def run_all(tiers: Sequence[int]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for module, tier in MODULES:
        if tier not in tiers:
            continue
        code, output = _run(module)
        if code == 0:
            results[module] = "PASS"
            print(f"PASS  [{tier}] {module}")
            for line in output.splitlines():
                if line.startswith("["):
                    print(f"        {line}")
            continue
        error = _last_error_line(output)
        if _is_missing_dependency(output):
            results[module] = f"SKIP(env): {error}"
            print(f"SKIP  [{tier}] {module}  <- {error}")
            continue
        results[module] = f"FAIL: {error}"
        print(f"FAIL  [{tier}] {module}")
        print(output)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers", default="0,1,2", help="comma-separated tiers to run (0 = stdlib only)"
    )
    args = parser.parse_args()
    tiers = [int(part) for part in args.tiers.split(",") if part.strip()]
    results = run_all(tiers)

    failures: List[str] = [name for name, status in results.items() if status.startswith("FAIL")]
    skipped: List[str] = [name for name, status in results.items() if status.startswith("SKIP")]
    passed = len(results) - len(failures) - len(skipped)
    print("=" * 72)
    print(f"{passed} passed, {len(skipped)} skipped (environment), {len(failures)} failed")
    if skipped:
        print("Skipped modules were NOT verified on this machine; run them on the GPU box.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
