r"""Prove -- by measurement -- that the 4-D additive mask reaches the attention.

``config._attn_implementation == 'sdpa'`` is necessary but NOT sufficient.
Transformers has shipped releases in which a 4-D ``attention_mask`` is treated as
a binary keep-mask and inverted; an additive mask then collapses to a constant,
every row attends everything uniformly, and training still runs -- with a purely
decorative bottleneck and no error anywhere. Version detection is not accepted as
proof, so ``assert_four_d_mask_is_honoured`` blocks key 0 for query row 1 and
checks that row 1 stops depending on key 0 while the unmasked control still does.

Upstream never passes a 4-D mask, so it has no counterpart to this file.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dualtrack import mask as mask_lib

logger = logging.getLogger(__name__)

RELATIVE_TOLERANCE = 0.01
ABSOLUTE_TOLERANCE = 1e-3


def assert_sdpa(model: nn.Module, use_flash_attention_2: bool = False) -> None:
    """Fail loudly rather than let the bottleneck evaporate inside an FA2 kernel."""
    if use_flash_attention_2:
        raise ValueError(
            "flash-attention-2 cannot consume the 4-D bottleneck mask; every number would be "
            "meaningless. Run with --use_flash_attention_2 False."
        )
    implementation = getattr(model.config, "_attn_implementation", None)
    if implementation is None:
        raise RuntimeError(
            "this transformers version predates config._attn_implementation and does not forward "
            "4-D attention masks; install transformers>=4.51.1"
        )
    if implementation != "sdpa":
        raise RuntimeError(
            f"expected attn_implementation='sdpa', got {implementation!r}. Eager would be "
            "numerically fine but is rejected here so training and generation share one path."
        )


def _probe_deltas(model: nn.Module, additive: Optional[Tensor]) -> float:
    """Max change in the row-1 logits when only the row-0 embedding changes."""
    embeddings = model.get_input_embeddings().weight
    generator = torch.Generator(device="cpu").manual_seed(0)
    shape = (1, 1, int(embeddings.shape[1]))
    scale = float(embeddings.detach().float().std().clamp_min(1e-4).item())
    parts = [
        (torch.randn(shape, generator=generator) * scale).to(embeddings.device, embeddings.dtype)
        for _ in range(3)
    ]
    first, second, tail = parts
    kwargs: Dict[str, Any] = {"use_cache": False}
    if additive is not None:
        kwargs["attention_mask"] = additive
    logits_a = model(inputs_embeds=torch.cat([first, tail], dim=1), **kwargs).logits
    logits_b = model(inputs_embeds=torch.cat([second, tail], dim=1), **kwargs).logits
    if not torch.isfinite(logits_a).all():
        raise RuntimeError("the 4-D additive mask produced non-finite logits; it is not consumable")
    return float((logits_a[0, 1] - logits_b[0, 1]).float().abs().max().item())


@torch.no_grad()
def assert_four_d_mask_is_honoured(model: nn.Module) -> None:
    """Measure, do not assume, that a 4-D additive mask reaches the attention."""
    dtype = model.get_input_embeddings().weight.dtype
    device = model.get_input_embeddings().weight.device
    keep = mask_lib.build_bottleneck_mask([2], [(0, 1)], [(1, 2)], 2)
    additive = mask_lib.to_additive(keep, dtype).to(device)
    blocked_delta = _probe_deltas(model, additive)
    control_delta = _probe_deltas(model, None)
    if control_delta <= 0.0:
        raise RuntimeError(
            "mask probe is not sensitive: row 1 does not depend on key 0 even without a mask, "
            "so it cannot prove anything about the bottleneck"
        )
    if blocked_delta > max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * control_delta):
        raise RuntimeError(
            "this transformers version does NOT honour the 4-D additive attention mask: blocking "
            f"key 0 changed row 1 by {blocked_delta:.6g} (unmasked control {control_delta:.6g}). "
            "The bottleneck would be decorative. Upgrade transformers."
        )
    logger.info(
        "4-D additive mask honoured (blocked delta %.3g vs control %.3g).",
        blocked_delta,
        control_delta,
    )


class _MaskSemanticsStub(nn.Module):
    """One SDPA block reproducing the three ways a 4-D mask can be treated."""

    def __init__(self, mode: str, hidden: int = 8, vocab: int = 16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.mode = mode
        self.embedding = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def _resolve(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if self.mode == "ignoring" or attention_mask is None:
            return None
        if self.mode == "inverting":
            inverted = 1.0 - attention_mask
            return torch.zeros_like(attention_mask).masked_fill(
                inverted.to(torch.bool), torch.finfo(attention_mask.dtype).min
            )
        return attention_mask

    def forward(
        self, inputs_embeds: Tensor, attention_mask: Optional[Tensor] = None, use_cache: bool = False
    ) -> Any:
        from types import SimpleNamespace

        hidden = self.proj(inputs_embeds).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            hidden, hidden, hidden, attn_mask=self._resolve(attention_mask)
        )
        return SimpleNamespace(logits=self.head(attended.squeeze(1)))


class _ConfigStub:
    def __init__(self, implementation: Optional[str]) -> None:
        if implementation is not None:
            self._attn_implementation = implementation


def _selftest_assert_sdpa() -> None:
    class _Model:
        def __init__(self, implementation: Optional[str]) -> None:
            self.config = _ConfigStub(implementation)

    assert_sdpa(_Model("sdpa"))
    for model, use_fa2, error in (
        (_Model("sdpa"), True, ValueError),
        (_Model("flash_attention_2"), False, RuntimeError),
        (_Model("eager"), False, RuntimeError),
        (_Model(None), False, RuntimeError),
    ):
        try:
            assert_sdpa(model, use_flash_attention_2=use_fa2)
        except error:
            continue
        raise AssertionError(f"assert_sdpa accepted {model.config.__dict__} fa2={use_fa2}")


def _selftest_probe() -> None:
    assert_four_d_mask_is_honoured(_MaskSemanticsStub("honouring"))
    for mode in ("inverting", "ignoring"):
        try:
            assert_four_d_mask_is_honoured(_MaskSemanticsStub(mode))
        except RuntimeError as exc:
            assert "does NOT honour" in str(exc), (mode, str(exc))
            continue
        raise AssertionError(f"the probe accepted a transformers that is {mode} the 4-D mask")


def selftest() -> None:
    _selftest_assert_sdpa()
    _selftest_probe()
    print(
        "[attention_probe] OK -- FA2 and non-SDPA rejected, and the 4-D-mask-semantics probe "
        "catches a stack that inverts or ignores the additive mask."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="4-D mask support probe (needs torch)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("attention_probe.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
