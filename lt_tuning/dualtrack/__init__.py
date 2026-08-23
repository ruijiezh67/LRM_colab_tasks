"""Thin dual-track patch layer over the vendored LT-Tuning clone in ``../upstream``.

Nothing here re-implements upstream.  Every module either imports an upstream symbol,
subclasses one and overrides a single method, or is genuinely new (the track ids, the
bottleneck mask, the acceptance harness).  ``../upstream`` is read-only: a dirty tree
fails ``_upstream.assert_upstream_clean()``.

NO ATTACK.  No poison flag, no label flipping, no depth gating, no success metric.
(The exemption marker below lets `train._selftest_no_attack_surface` scan this file; the
scan is substring-based, so a sentence naming the banned words would trip it.)
ATTACK-SCAN-SKIP
"""

from __future__ import annotations

from typing import Final, Tuple

# Upstream pins transformers==4.55.4; the data path's true floor is 4.41
# (``pad_without_fast_tokenizer_warning``).  The project spec asks for >= 4.45.
# Tuples here, dotted strings in the sibling folders: the comparisons in _upstream.py are
# tuple comparisons, and parsing a string back would add a failure mode for no gain.
MIN_TRANSFORMERS: Final[str] = "4.45"
MIN_TRANSFORMERS_DATA_PATH: Final[str] = "4.41"
TRANSFORMERS_VERSION_FLOOR: Final[Tuple[int, ...]] = (4, 45)
TRANSFORMERS_DATA_PATH_FLOOR: Final[Tuple[int, ...]] = (4, 41)

#: The vendored clone this layer was written against; ``_upstream.selftest`` refuses a
#: clone that has moved off it.  Full 40-char sha, as in the sibling folders.
UPSTREAM_COMMIT: Final[str] = "c18aac695b33de135d3dd0848de0464d1b644ba7"

__all__ = [
    "MIN_TRANSFORMERS",
    "MIN_TRANSFORMERS_DATA_PATH",
    "TRANSFORMERS_VERSION_FLOOR",
    "TRANSFORMERS_DATA_PATH_FLOOR",
    "UPSTREAM_COMMIT",
]
