"""The ONLY module that knows where the vendored clone lives.

Upstream is a flat repo whose modules are named ``model``, ``dataset``, ``utils``, ``run``
and whose intra-repo imports are absolute (``from utils import ...`` at dataset.py:15).
Those names are maximally generic, so the path insertion is done here, the four modules are
imported eagerly (all their intra-repo imports are module level, so nothing resolves later),
and ``sys.path`` is restored.  Blast radius: this file.

``model.py`` and ``utils.py`` import on transformers 4.35; ``dataset.py`` and ``run.py``
need >= 4.41 (``pad_without_fast_tokenizer_warning``).  The two tiers are therefore split:
``import_core()`` never fails on a supported install, ``import_data()`` may, and
``UpstreamUnavailable`` carries the real error rather than a fabricated one.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple

from . import TRANSFORMERS_DATA_PATH_FLOOR, TRANSFORMERS_VERSION_FLOOR, UPSTREAM_COMMIT

UPSTREAM_DIR = Path(__file__).resolve().parent.parent / "upstream"
_UPSTREAM_MODULES: Tuple[str, ...] = ("utils", "model", "dataset", "run")


class UpstreamUnavailable(ImportError):
    """The vendored tree could not be imported here.  Carries the real exception text."""


class UpstreamDirty(RuntimeError):
    """``git status --porcelain`` was non-empty: the read-only invariant was broken."""


@dataclass(frozen=True)
class CoreSymbols:
    """Everything importable without upstream's data stack."""

    model: ModuleType
    utils: ModuleType


@dataclass(frozen=True)
class DataSymbols:
    """Everything that needs ``dataset.py`` / ``run.py`` (transformers >= 4.41)."""

    dataset: ModuleType
    run: ModuleType


def upstream_dir() -> Path:
    if not (UPSTREAM_DIR / "model.py").is_file():
        raise UpstreamUnavailable(f"no model.py under {UPSTREAM_DIR}; the clone is missing")
    return UPSTREAM_DIR


def _import_named(names: Tuple[str, ...]) -> Tuple[ModuleType, ...]:
    """Import upstream modules with ``upstream/`` first on the path, then restore it."""
    root = str(upstream_dir())
    saved = list(sys.path)
    sys.path.insert(0, root)
    try:
        return tuple(__import__(name) for name in names)
    except Exception as exc:  # noqa: BLE001 - the real text is the deliverable
        raise UpstreamUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        sys.path[:] = saved


def import_core() -> CoreSymbols:
    """``model.py`` + ``utils.py``.  Importable on transformers 4.35."""
    utils, model = _import_named(("utils", "model"))
    return CoreSymbols(model=model, utils=utils)


def import_data() -> DataSymbols:
    """``dataset.py`` + ``run.py``.  Needs transformers >= 4.41."""
    dataset, run = _import_named(("dataset", "run"))
    return DataSymbols(dataset=dataset, run=run)


def transformers_version_tuple() -> Tuple[int, ...]:
    import transformers

    parts = []
    for chunk in str(transformers.__version__).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def version_floor_report() -> str:
    """A one-line statement of where this install sits against the two floors."""
    import transformers

    version = transformers_version_tuple()
    return (
        f"transformers {transformers.__version__} "
        f"(training floor {'.'.join(map(str, TRANSFORMERS_VERSION_FLOOR))}, "
        f"data-path floor {'.'.join(map(str, TRANSFORMERS_DATA_PATH_FLOOR))}): "
        f"core={'ok' if version >= (4, 35) else 'too old'} "
        f"data={'ok' if version >= TRANSFORMERS_DATA_PATH_FLOOR else 'BELOW FLOOR'}"
    )


def upstream_commit() -> str:
    """The LIVE full sha of ``../upstream``.

    Full, not ``--short``: the manifest is compared against :data:`dualtrack.UPSTREAM_COMMIT`
    (the pin), and the sibling folders record 40-char shas, so an abbreviation here would
    make the three artefacts non-comparable.
    """
    result = subprocess.run(
        ["git", "-C", str(upstream_dir()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UpstreamDirty(f"git rev-parse failed in {upstream_dir()}: {result.stderr.strip()}")
    return result.stdout.strip()


def upstream_status() -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream_dir()), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UpstreamDirty(f"git status failed in {upstream_dir()}: {result.stderr.strip()}")
    return result.stdout


def assert_upstream_clean(ignore_untracked: bool = True) -> str:
    """The hard invariant.  ``__pycache__`` is gitignored upstream, so a clean tree stays
    clean after an import; anything else that shows up here is an edit and must fail."""
    status = upstream_status()
    lines = [line for line in status.splitlines() if line.strip()]
    if ignore_untracked:
        lines = [line for line in lines if not line.startswith("??")]
    if lines:
        raise UpstreamDirty(
            "upstream/ is not clean; the patch layer must never edit it:\n" + "\n".join(lines)
        )
    return upstream_commit()


def selftest() -> None:
    """No torch, no transformers required for the purity half."""
    commit = assert_upstream_clean()
    print(f"  upstream/ clean at commit {commit}: OK")
    assert commit == UPSTREAM_COMMIT, (
        f"the clone is at {commit} but dualtrack/__init__.py pins {UPSTREAM_COMMIT}; "
        "every measured number in the README was taken against the pin"
    )
    print(f"  live sha == the pin in dualtrack/__init__.py ({UPSTREAM_COMMIT}): OK")
    saved = list(sys.path)
    try:
        core: Optional[CoreSymbols] = import_core()
    except UpstreamUnavailable as exc:
        print(f"  import_core UNAVAILABLE: {exc}")
        core = None
    assert sys.path == saved, "sys.path was not restored after importing upstream"
    print("  sys.path restored after the upstream import: OK")
    if core is not None:
        assert hasattr(core.model, "LT_Tuning_Model"), "model.py has no LT_Tuning_Model"
        assert hasattr(core.utils, "StageManager"), "utils.py has no StageManager"
        print("  upstream model.LT_Tuning_Model + utils.StageManager import: OK")
    try:
        data = import_data()
        assert hasattr(data.dataset, "MyCollator")
        print("  upstream dataset.MyCollator + run.LTTuningTrainer import: OK")
    except UpstreamUnavailable as exc:
        print(f"  import_data UNAVAILABLE (expected below the 4.41 floor): {exc}")
    try:
        print(f"  {version_floor_report()}")
    except ImportError:
        print("  transformers not installed; version floor not checked")
    assert_upstream_clean()
    print("  upstream/ still clean after every import: OK")
    print("_upstream.py selftest PASSED")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Vendored-upstream bridge and purity check")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
