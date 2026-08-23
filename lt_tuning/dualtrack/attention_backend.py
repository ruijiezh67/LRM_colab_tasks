"""Getting a per-segment 4-D attention bias into the base model, and proving it lands.

Two paths:
  * "direct" -- pass the (B, 1, q, kv) bias as `attention_mask`.  Works on transformers
    >= ~4.53, where `masking_utils._preprocess_mask_arguments` early-returns a 4-D mask.
  * "patch"  -- monkeypatch the causal-mask builder to return our tensor from a slot.
    Needed on older releases: `_prepare_4d_causal_attention_mask` unconditionally routes
    a mask through `_expand_mask`, which unpacks `bsz, src_len = mask.size()` and raises
    on a 4-D input.

Which one is live is never assumed.  `probe_four_d_mask()` runs a tiny locally-constructed
Llama (no download, no network) in two cached segments and checks that randomising the
*blocked* key embeddings leaves the blocked query rows bit-identical, with a plain-causal
control that must differ.  The result goes into the run manifest.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

DIRECT = "direct"
PATCH = "patch"
_PATCH_NAMES = ("create_causal_mask", "_prepare_4d_causal_attention_mask")


class FlashAttentionRefused(RuntimeError):
    """flash-attention-2 silently ignores a 4-D mask, so the bottleneck would not exist."""


class FourDMaskUnsupportedError(RuntimeError):
    """Neither the direct nor the patched path delivers a 4-D mask on this install."""


@dataclass(frozen=True)
class ProbeReport:
    path: Optional[str]
    direct_ok: bool
    patch_ok: bool
    transformers_version: str
    attn_impl: str
    detail: str

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def transformers_version() -> str:
    import transformers

    return str(transformers.__version__)


def resolve_attn_impl(model: Any) -> str:
    """Report the attention implementation and refuse flash-attention-2."""
    config = getattr(model, "config", model)
    if getattr(config, "_flash_attn_2_enabled", False):
        raise FlashAttentionRefused(
            "config._flash_attn_2_enabled is True; a 4-D mask would be ignored"
        )
    impl = getattr(config, "_attn_implementation", None)
    if impl == "flash_attention_2":
        raise FlashAttentionRefused(
            "attn_implementation='flash_attention_2'; a 4-D mask would be ignored"
        )
    if impl is None:
        # transformers < 4.36 has no sdpa path for Llama at all; eager honours an additive bias.
        return "eager(pre-sdpa)"
    return str(impl)


_NOTED_IMPLS: set = set()


def assert_mask_honouring_attention(model: Any, require_sdpa: bool = True) -> str:
    impl = resolve_attn_impl(model)
    if impl in ("sdpa", "eager", "eager(pre-sdpa)"):
        if require_sdpa and impl == "eager(pre-sdpa)" and impl not in _NOTED_IMPLS:
            _NOTED_IMPLS.add(impl)
            print(
                "NOTE: this transformers build has no sdpa path for Llama; running eager, "
                "which also honours an additive 4-D bias.  probe_four_d_mask() is the real check."
            )
        return impl
    raise FlashAttentionRefused(
        f"attention implementation {impl!r} is not known to honour a 4-D mask"
    )


def _patch_targets() -> List[Tuple[Any, str]]:
    targets: List[Tuple[Any, str]] = []
    modules: List[Any] = []
    try:
        from transformers.models.llama import modeling_llama

        modules.append(modeling_llama)
    except ImportError:  # pragma: no cover
        pass
    # The base model may be Qwen2 / Mistral / etc.; each modeling module imports its OWN
    # `create_causal_mask` reference, so patching only llama + masking_utils leaves that
    # reference live and the mask builder never fires (SegmentBiasTape -> TapeError).
    for _mod_path in (
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.mistral.modeling_mistral",
    ):
        try:
            import importlib as _il

            modules.append(_il.import_module(_mod_path))
        except ImportError:
            pass
    try:
        from transformers import masking_utils

        modules.append(masking_utils)
    except ImportError:
        pass
    for module in modules:
        for name in _PATCH_NAMES:
            if hasattr(module, name):
                targets.append((module, name))
    if not targets:
        raise FourDMaskUnsupportedError(
            "no causal-mask builder found to patch; inspect the installed transformers"
        )
    return targets


@contextmanager
def patched_mask_builder(builder: Callable[..., Any]) -> Iterator[None]:
    """Replace transformers' causal-mask constructor with `builder` for the duration.

    `builder` receives whatever the installed transformers passes its own constructor and
    returns the 4-D additive bias to use.  This generalises what `FourDMaskInjector` needs
    (one fixed tensor) to what `model.SegmentBiasTape` needs (a different slice per
    segment), which is the whole reason upstream's `forward` can be *called* instead of
    re-typed: upstream builds the segments, we decide what each one may attend to.

    Process-local and re-entrant-safe for one forward.  NOT thread-safe.
    """
    targets = _patch_targets()
    originals = [(module, name, getattr(module, name)) for module, name in targets]
    try:
        for module, name in targets:
            setattr(module, name, builder)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


class FourDMaskInjector:
    """Single entry point for every direct `base_causallm` call in this folder.

    Training does not use this -- it goes through upstream's own forward under
    `patched_mask_builder`.  Generation does, one decode step at a time.
    """

    def __init__(self, path: str) -> None:
        if path not in (DIRECT, PATCH):
            raise ValueError(f"path must be {DIRECT!r} or {PATCH!r}; got {path!r}")
        self.path = path

    @contextmanager
    def _patched(self, bias: Any) -> Iterator[None]:
        with patched_mask_builder(lambda *_a, **_k: bias):
            yield

    def call(
        self,
        base_causallm: Any,
        inputs_embeds: Any,
        bias: Any,
        position_ids: Any,
        past_key_values: Any = None,
        output_hidden_states: bool = True,
        use_cache: bool = True,
    ) -> Any:
        kwargs = dict(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if self.path == DIRECT:
            return base_causallm(attention_mask=bias, **kwargs)
        import torch

        ones = torch.ones(
            (bias.shape[0], bias.shape[-1]), dtype=torch.long, device=inputs_embeds.device
        )
        with self._patched(bias):
            return base_causallm(attention_mask=ones, **kwargs)


def load_causal_lm(source: str, torch_dtype: Any = None, trust_remote_code: bool = True) -> Any:
    """`from_pretrained` asking for sdpa, retried without the kwarg where it is not accepted.

    transformers < 4.36 has no `attn_implementation` argument and forwards unknown kwargs to
    the config constructor, which raises `TypeError`.  Passing it unconditionally makes every
    real entry point on such an install crash before it loads anything.  Eager honours an
    additive 4-D bias there, and `probe_four_d_mask()` is what actually establishes that.
    """
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(source, attn_implementation="sdpa", **kwargs)
    except (TypeError, ValueError) as exc:
        print(
            f"NOTE: attn_implementation='sdpa' refused by transformers {transformers_version()} "
            f"({type(exc).__name__}: {exc}); loading without it.  The 4-D bias is verified by "
            "probe_four_d_mask(), not by the implementation name."
        )
    return AutoModelForCausalLM.from_pretrained(source, **kwargs)


def build_tiny_llama(
    vocab_size: int = 64, hidden: int = 32, layers: int = 2, heads: int = 4, seed: int = 0
) -> Any:
    """Random-init Llama built from a local config.  No download, no network."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=512,
    )
    try:
        model = LlamaForCausalLM._from_config(config, attn_implementation="sdpa")
    except (TypeError, ValueError, AttributeError, KeyError):
        model = LlamaForCausalLM(config)
    return model.eval()


def _probe_bias(
    seq_len: int, blocked_keys: Sequence[int], first_blocked_query: int, causal_only: bool
) -> Any:
    import torch

    neg = torch.finfo(torch.float32).min
    bias = torch.zeros(1, 1, seq_len, seq_len)
    for q in range(seq_len):
        for k in range(seq_len):
            blocked = k > q or (not causal_only and q >= first_blocked_query and k in blocked_keys)
            if blocked:
                bias[0, 0, q, k] = neg
    return bias


def _run_two_segments(
    model: Any, injector: FourDMaskInjector, embeds: Any, bias: Any, split: int
) -> Any:
    import torch

    with torch.no_grad():
        first = injector.call(
            model, embeds[:, :split], bias[:, :, :split, :split], torch.arange(split).view(1, -1)
        )
        total = embeds.shape[1]
        second = injector.call(
            model,
            embeds[:, split:],
            bias[:, :, split:total, :total],
            torch.arange(split, total).view(1, -1),
            past_key_values=first.past_key_values,
        )
    return torch.cat([first.logits, second.logits], dim=1)


def _probe_path(path: str, seed: int = 0) -> Tuple[bool, str]:
    import torch

    model = build_tiny_llama(seed=seed)
    injector = FourDMaskInjector(path)
    seq_len, blocked_keys, first_q, split = 10, (3, 4), 5, 5
    torch.manual_seed(seed + 1)
    embeds = torch.randn(1, seq_len, model.config.hidden_size)
    perturbed = embeds.clone()
    # Scaling the perturbation would not help: RMSNorm removes the scale, so the control
    # magnitude reflects the random model's genuine sensitivity, and only its sign matters.
    perturbed[0, list(blocked_keys)] = torch.randn(len(blocked_keys), model.config.hidden_size)
    try:
        masked = _probe_bias(seq_len, blocked_keys, first_q, causal_only=False)
        delta = (
            (
                _run_two_segments(model, injector, embeds, masked, split)[0, first_q:]
                - _run_two_segments(model, injector, perturbed, masked, split)[0, first_q:]
            )
            .abs()
            .max()
            .item()
        )
        causal = _probe_bias(seq_len, blocked_keys, first_q, causal_only=True)
        control = (
            (
                _run_two_segments(model, injector, embeds, causal, split)[0, first_q:]
                - _run_two_segments(model, injector, perturbed, causal, split)[0, first_q:]
            )
            .abs()
            .max()
            .item()
        )
    except Exception as exc:  # noqa: BLE001 - a failing path is a result, not a crash
        return False, f"{path}: {type(exc).__name__}: {exc}"
    if delta != 0.0:
        return False, f"{path}: blocked rows moved by {delta:.3e} (want exactly 0)"
    if control <= 0.0:
        return False, f"{path}: control is {control:.3e}; the probe has no power"
    return True, f"{path}: blocked delta 0.0, unmasked control {control:.3e}"


def probe_four_d_mask(prefer: Optional[str] = None) -> ProbeReport:
    """Decide which injection path actually works here.  Never report a run whose probe
    did not execute."""
    order = [prefer] if prefer else [DIRECT, PATCH]
    results = {}
    for path in (DIRECT, PATCH):
        if prefer and path != prefer:
            continue
        results[path] = _probe_path(path)
    chosen = next((p for p in order if p in results and results[p][0]), None)
    detail = " | ".join(results[p][1] for p in results)
    report = ProbeReport(
        path=chosen,
        direct_ok=results.get(DIRECT, (False, ""))[0],
        patch_ok=results.get(PATCH, (False, ""))[0],
        transformers_version=transformers_version(),
        attn_impl=resolve_attn_impl(build_tiny_llama()),
        detail=detail,
    )
    if chosen is None:
        raise FourDMaskUnsupportedError(f"no working 4-D mask path on this install: {detail}")
    return report


def selftest() -> None:
    """CPU-only, no network, no weights: builds a tiny random Llama from a local config."""
    report = probe_four_d_mask()
    print(f"  transformers {report.transformers_version}, attn={report.attn_impl}")
    print(f"  direct_ok={report.direct_ok} patch_ok={report.patch_ok} -> path={report.path!r}")
    for line in report.detail.split(" | "):
        print(f"    {line}")
    assert report.path in (DIRECT, PATCH)

    class _FakeConfig:
        _attn_implementation = "flash_attention_2"

    class _FakeModel:
        config = _FakeConfig()

    try:
        resolve_attn_impl(_FakeModel())
    except FlashAttentionRefused:
        print("  flash-attention-2 is refused: OK")
    else:  # pragma: no cover
        raise AssertionError("flash_attention_2 must be refused; it ignores 4-D masks")
    _selftest_load_causal_lm()
    _selftest_patched_mask_builder()
    print("attention_backend.py selftest PASSED")


def _selftest_patched_mask_builder() -> None:
    """The builder must see one call per real mask construction and be fully restored.

    `model.SegmentBiasTape` relies on both halves: on being called exactly once per
    segment, and on transformers being byte-identical afterwards, since the same process
    also runs unmasked forwards (the insertion strategy's scoring pass, upstream's own
    evaluation) that must not inherit our bias.
    """
    import torch

    model = build_tiny_llama(vocab_size=16, seed=9)
    targets = _patch_targets()
    before = [(module, name, getattr(module, name)) for module, name in targets]
    calls: List[Tuple[int, ...]] = []
    seq_len = 6
    bias = torch.zeros(1, 1, seq_len, seq_len)
    bias[0, 0].masked_fill_(
        torch.arange(seq_len).view(seq_len, 1) < torch.arange(seq_len).view(1, seq_len),
        torch.finfo(torch.float32).min,
    )

    def builder(*args: Any, **kwargs: Any) -> Any:
        calls.append((len(args), len(kwargs)))
        return bias

    embeds = torch.randn(1, seq_len, model.config.hidden_size)
    ones = torch.ones(1, seq_len, dtype=torch.long)
    with torch.no_grad(), patched_mask_builder(builder):
        patched_logits = model(inputs_embeds=embeds, attention_mask=ones).logits
    assert len(calls) == 1, f"expected one mask construction per forward, saw {len(calls)}"
    after = [(module, name, getattr(module, name)) for module, name in targets]
    assert after == before, "patched_mask_builder did not restore transformers"
    with torch.no_grad():
        plain = model(inputs_embeds=embeds, attention_mask=ones).logits
    delta = (patched_logits - plain).abs().max().item()
    assert delta == 0.0, f"a plain causal bias must reproduce the unpatched run; got {delta}"
    print(
        f"  patched_mask_builder: 1 call/forward, restored, causal bias reproduces "
        f"the unpatched logits (max|delta|={delta}): OK"
    )


def _selftest_load_causal_lm() -> None:
    """The real entry points load a checkpoint through `load_causal_lm`; prove it survives
    this transformers version offline, on a locally saved tiny model."""
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        build_tiny_llama(vocab_size=32, seed=4).save_pretrained(tmp)
        model = load_causal_lm(tmp, torch_dtype=torch.float32)
    impl = resolve_attn_impl(model)
    assert model.config.vocab_size == 32
    print(f"  load_causal_lm round-trips a saved checkpoint here: attn_impl={impl!r}: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="4-D mask injection backend")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--probe", action="store_true", help="print the probe result as JSON")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.probe:
        import json

        print(json.dumps(probe_four_d_mask().to_json(), indent=2))
        return
    parser.error("nothing to do: pass --selftest or --probe")


if __name__ == "__main__":
    main()
