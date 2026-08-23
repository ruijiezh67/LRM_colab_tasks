"""Thin dual-track patch layer over the vendored Latent-SFT upstream.

Nothing in this package edits ``<platform>/upstream``. Every upstream symbol is
either imported unchanged or subclassed; the three re-typed fragments are listed
in the platform README under "COPY sites".

This module stays stdlib-only so that ``python3 -m dualtrack.spans --selftest``
works on a machine without torch.
"""

from __future__ import annotations

from typing import Final

# Upstream pins: requirements.txt (torch/peft) and README.md line 405 (transformers).
MIN_TORCH: Final[str] = "2.5.1"
MIN_TRANSFORMERS: Final[str] = "4.51.1"
MIN_PEFT: Final[str] = "0.15.2"

# The vendored clone this layer was written against.
UPSTREAM_COMMIT: Final[str] = "08adc3a7f5067a9061208dac596cd31887542e06"

__all__ = ["MIN_TORCH", "MIN_TRANSFORMERS", "MIN_PEFT", "UPSTREAM_COMMIT"]
