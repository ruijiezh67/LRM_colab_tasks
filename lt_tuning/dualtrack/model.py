"""`DualTrackLTModel(LT_Tuning_Model)` -- two overrides, nothing copied.

The previous round re-typed upstream's segment loop in order to slip a 4-D bias into each
`self.base_causallm(...)` call.  That is not necessary.  Upstream's per-segment call
produces exactly ONE causal-mask construction inside `LlamaModel.forward`, so the bias can
be delivered by replacing the mask builder with a tape that hands out
`full_bias[:, :, q_start:q_end, :q_end]` one segment at a time, and then calling
`super().forward()` unchanged.  Segmentation, embedding assembly, latent replacement, KV
threading, logits concat and `Outputs` are then EXECUTED BY UPSTREAM, not merely inherited.

Overrides:
  forward()          -> super().forward() under the tape, plus the composite loss
  _apply_transform() -> super()._apply_transform(), plus record/substitute the latent
  generate()         -> refused; upstream's decode loop passes no attention_mask at all
                        (model.py:689-694), so it cannot express the bottleneck

Inherited unchanged: `_soft_fusion_embedding`, `_select_hidden_state`, `_get_activation`,
the thinking-MLP constructor, `from_pretrained` and both loaders, `update_stage_config`,
`config`, `device`, `train`, `eval`.

`from_pretrained` is inherited but is NOT the load path here, and calling it directly fails
loudly rather than quietly: `__init__` takes `bottleneck` keyword-only with no default, and
upstream's loaders default `attn_implementation` to `flash_attention_2` (model.py:133, 187),
which `_dualtrack_init` refuses.  The supported path is `promote()` over upstream's
`run.load_model_and_tokenizer`, which is what `train.py` uses.

ONE COPY, four lines: `make_latent` shadows the stage dispatch at model.py:515-527 (repeated
verbatim at 671-683).  It is four lines inside two loops, not a method, so there is nothing
to subclass; `generate.py` needs it per decode step.  Both branches call inherited methods.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ._upstream import import_core
from .attention_backend import (
    FourDMaskInjector,
    assert_mask_honouring_attention,
    patched_mask_builder,
    probe_four_d_mask,
)
from .config import LossWeights, MaskConfig
from .loss import LossParts, has_supervised_latent, latent_labels, partition_ce
from .mask import ModeLike, TrackId, build_full_bias, resolve_bottleneck_mode, segment_boundaries
from .tracks import latent_visit_order

LT_Tuning_Model = import_core().model.LT_Tuning_Model

LatentKey = Tuple[int, int]


@dataclass(frozen=True)
class DualTrackOutputs:
    """Upstream's `Outputs` fields (same names) plus the dual-track extras.

    Upstream's `Outputs` is a namedtuple and cannot carry extra fields, and run.py:902 does
    `outputs[0]`-style access on it; a dataclass that always exposes `.loss` satisfies
    `LTTuningTrainer.compute_loss`'s `hasattr(outputs, 'loss')` test (run.py:901).
    """

    loss: Optional[torch.Tensor]
    inputs_embeds: torch.Tensor
    logits: torch.Tensor
    past_key_values: Any
    last_hidden_state: torch.Tensor
    attentions: Optional[Tuple[torch.Tensor, ...]]
    loss_cot: Optional[torch.Tensor]
    loss_ans: Optional[torch.Tensor]
    loss_latent: Optional[torch.Tensor]
    latents: Dict[LatentKey, torch.Tensor]
    bottleneck_mode: str
    mask_on: bool


class TapeError(RuntimeError):
    """The mask builder was called a different number of times than there are segments."""


class SegmentBiasTape:
    """Hands upstream's own segment loop one slice of the full bias per mask construction.

    Upstream calls `base_causallm` once per segment (model.py:473-491) and both call sites
    have `kv_len == boundary_end`, so `full[:, :, q_start:q_end, :q_end]` is the uniform
    rule.  The span is cross-checked against whatever the installed transformers passes its
    own builder, and a disagreement raises rather than silently mis-masking:

      < 4.53 : `_prepare_4d_causal_attention_mask(mask, (B, q_len), embeds, past_len)`
               -> (past_len, past_len + q_len)
      >= 4.53: `create_causal_mask(..., cache_position=...)`
               -> (int(cp[0]), int(cp[-1]) + 1)

    Neither extractable -> the cursor still advances and `assert_exhausted()` is the
    backstop.  A model family with sliding-window layers would additionally call
    `create_sliding_window_causal_mask`, which is deliberately NOT patched: the exhaustion
    check turns that into a loud error instead of a half-masked run.
    """

    def __init__(self, full_bias: torch.Tensor, spans: Sequence[Tuple[int, int]]) -> None:
        self.full = full_bias
        self.spans = list(spans)
        self.cursor = 0
        self.observed: List[Optional[Tuple[int, int]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.cursor >= len(self.spans):
            raise TapeError(
                f"the mask builder was called {self.cursor + 1} times for "
                f"{len(self.spans)} segments; upstream's segmentation and "
                "`segment_boundaries(track_ids)` have diverged"
            )
        q_start, q_end = self.spans[self.cursor]
        observed = _observed_span(args, kwargs)
        self.observed.append(observed)
        if observed is not None and observed != (q_start, q_end):
            raise TapeError(
                f"segment {self.cursor}: transformers is building a mask for {observed} "
                f"but the tape holds {(q_start, q_end)}"
            )
        self.cursor += 1
        return self.full[:, :, q_start:q_end, :q_end]

    def assert_exhausted(self) -> None:
        if self.cursor != len(self.spans):
            raise TapeError(
                f"the mask builder ran {self.cursor} times for {len(self.spans)} segments; "
                "some segment attended under an unmasked causal mask"
            )


def _observed_span(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Best-effort recovery of (q_start, q_end) from the installed builder's arguments."""
    cache_position = kwargs.get("cache_position")
    if cache_position is not None and len(cache_position) > 0:
        return int(cache_position[0]), int(cache_position[-1]) + 1
    if len(args) >= 4 and isinstance(args[1], (tuple, list)) and len(args[1]) == 2:
        past = int(args[3])
        return past, past + int(args[1][1])
    shape = kwargs.get("input_shape")
    if shape is not None and "past_key_values_length" in kwargs:
        past = int(kwargs["past_key_values_length"])
        return past, past + int(shape[1])
    return None


class LatentTape:
    """Labels each `_apply_transform` call with the latent it belongs to.

    Upstream calls `_apply_transform` exactly once per latent, in
    `(boundary_end ascending, batch_idx ascending)` order (model.py:504-528), in all three
    `stage_mode` branches.  `tracks.latent_visit_order` reconstructs that order from
    `track_ids`, so the recorded tensors can be keyed without touching the loop.
    """

    def __init__(
        self,
        order: Sequence[LatentKey],
        override: Optional[Mapping[LatentKey, torch.Tensor]] = None,
    ) -> None:
        self.order = list(order)
        self.override = dict(override or {})
        self.cursor = 0
        self.captured: Dict[LatentKey, torch.Tensor] = {}

    def record(self, value: torch.Tensor) -> torch.Tensor:
        if self.cursor >= len(self.order):
            raise TapeError(
                f"_apply_transform called {self.cursor + 1} times for {len(self.order)} "
                "latents; the visit order derived from track_ids is wrong"
            )
        key = self.order[self.cursor]
        self.cursor += 1
        self.captured[key] = value.detach()
        replacement = self.override.get(key)
        if replacement is None:
            return value
        substituted = replacement.to(dtype=value.dtype, device=value.device)
        self.captured[key] = substituted.detach()
        return substituted

    def assert_exhausted(self) -> None:
        if self.cursor != len(self.order):
            raise TapeError(
                f"_apply_transform ran {self.cursor} times for {len(self.order)} latents"
            )
        unused = sorted(set(self.override) - set(self.captured))
        if unused:
            raise TapeError(f"latent_override named positions that were never visited: {unused}")


class DualTrackLTModel(LT_Tuning_Model):
    """LT-Tuning with a per-segment 4-D bottleneck bias, delivered through upstream's loop."""

    def __init__(
        self,
        *args: Any,
        bottleneck: MaskConfig,
        loss_weights: LossWeights = LossWeights(),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dualtrack_init(bottleneck, loss_weights)

    @classmethod
    def promote(
        cls, model: Any, bottleneck: MaskConfig, loss_weights: LossWeights = LossWeights(),
    ) -> "DualTrackLTModel":
        """Rebind an already-built `LT_Tuning_Model` to this subclass.

        Three lines instead of re-typing upstream's 68-line `load_model_and_tokenizer`
        (run.py:169-236), which is also what gives us its embedding resize and thinking-token
        initialisation (run.py:202-214) for free.  `bottleneck` has no default here on
        purpose: a run whose mode was never stated cannot exist.
        """
        if not isinstance(model, LT_Tuning_Model):
            raise TypeError(f"promote expects an LT_Tuning_Model; got {type(model).__name__}")
        model.__class__ = cls
        model._dualtrack_init(bottleneck, loss_weights)
        return model

    def _dualtrack_init(self, bottleneck: MaskConfig, loss_weights: LossWeights) -> None:
        bottleneck.validate()
        loss_weights.validate()
        self.bottleneck = bottleneck
        self.loss_weights = loss_weights
        # Invariant 4: flash-attention-2 ignores a 4-D mask, so the bottleneck would not
        # exist.  Upstream defaults to it in BOTH loaders (model.py:133, 187) and in
        # run.load_model_and_tokenizer (run.py:172).
        self.attn_impl = assert_mask_honouring_attention(self.base_causallm)
        self._latent_tape: Optional[LatentTape] = None
        self._injector: Optional[FourDMaskInjector] = None

    @property
    def injector(self) -> FourDMaskInjector:
        """Only `generate.py` uses this; training goes through upstream's forward."""
        if self._injector is None:
            self._injector = FourDMaskInjector(probe_four_d_mask().path)
        return self._injector

    # --- override 1 of 2 -------------------------------------------------------------

    def _apply_transform(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """upstream model.py:301-304.  Semantics stay upstream's; we only tap the value.

        Outside our `forward` the tape is None and this is a pure delegation, so anything
        else that reaches upstream's code paths behaves exactly as upstream.
        """
        value = super()._apply_transform(hidden_state)
        if self._latent_tape is None:
            return value
        return self._latent_tape.record(value)

    # --- override 2 of 2 -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        track_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        mask_on: Optional[bool] = None,
        mode: Optional[ModeLike] = None,
        latent_override: Optional[Mapping[LatentKey, torch.Tensor]] = None,
        weights: Optional[LossWeights] = None,
        **kwargs: Any,
    ) -> DualTrackOutputs:
        """upstream model.py:381-627, CALLED not copied.

        `track_ids` is required and has no default: with `remove_unused_columns=True` the HF
        Trainer would strip the column, and a `TypeError` at step 1 is the right failure for
        that -- not a silent unmasked run.
        """
        mask_on = self.bottleneck.mask_on if mask_on is None else bool(mask_on)
        mode = self.bottleneck.mode if mode is None else resolve_bottleneck_mode(mode)
        weights = self.loss_weights if weights is None else weights
        self._check_alignment(input_ids, track_ids, attention_mask)

        track_rows = track_ids.tolist()
        spans = _spans_from_boundaries(segment_boundaries(track_rows))
        bias = build_full_bias(
            track_ids, dtype=self.embedding.weight.dtype, mask_on=mask_on, mode=mode
        )
        tape = SegmentBiasTape(bias, spans)
        latents = LatentTape(latent_visit_order(track_rows), latent_override)

        upstream_labels, wants_latent = self._latent_objective_labels(labels, track_ids, weights)
        self._latent_tape = latents
        try:
            with patched_mask_builder(tape):
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=upstream_labels,
                    position_ids=position_ids,
                    **kwargs,
                )
        finally:
            self._latent_tape = None
        tape.assert_exhausted()
        latents.assert_exhausted()

        parts = self._compose_loss(outputs, labels, track_ids, weights, wants_latent)
        return _compose_outputs(outputs, parts, latents.captured, mode.value, mask_on)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Deliberately refused (upstream model.py:637-738).

        Upstream's incremental loop calls `base_causallm` with NO attention_mask at all
        (model.py:689-694), so it cannot express the bottleneck.  Falling back to it would
        produce an unmasked decode that looks like a masked one.
        """
        raise NotImplementedError(
            "upstream's generate() cannot express the bottleneck (model.py:689-694 passes no "
            "attention_mask).  Use dualtrack.generate.generate_dualtrack instead."
        )

    # --- helpers ---------------------------------------------------------------------

    def make_latent(self, prev_hidden: torch.Tensor, prev_logits: torch.Tensor) -> torch.Tensor:
        """COPY (4 lines): shadows the stage dispatch at model.py:515-527 / 671-683.

        It lives inline inside two loops upstream, not in a method, so there is nothing to
        subclass.  Both branches call inherited methods, so the semantics are upstream's.
        Used by `generate.py`, one decode step at a time.
        """
        if self.stage_mode == "soft_fusion":
            return self._apply_transform(self._soft_fusion_embedding(prev_hidden, prev_logits))
        return self._apply_transform(prev_hidden)

    def _latent_objective_labels(
        self, labels: Optional[torch.Tensor], track_ids: torch.Tensor, weights: LossWeights,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """`L_latent` is upstream's own loss line, evaluated on latent-only labels.

        There is no separate upstream latent head to import; upstream's latent component IS
        the CE at the positions whose label is the <thinking> id (model.py:543-550).  Passing
        those labels down makes upstream compute exactly that term.
        """
        if labels is None or weights.latent_w == 0.0:
            return None, False
        if not has_supervised_latent(labels, track_ids):
            return None, False
        return latent_labels(labels, track_ids), True

    def _compose_loss(
        self,
        outputs: Any,
        labels: Optional[torch.Tensor],
        track_ids: torch.Tensor,
        weights: LossWeights,
        wants_latent: bool,
    ) -> Optional[LossParts]:
        if labels is None:
            return None
        latent_loss = outputs.loss if wants_latent else None
        return partition_ce(outputs.logits, labels, track_ids, weights, latent_loss)

    def _check_alignment(
        self,
        input_ids: torch.Tensor,
        track_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        """Asserted, never assumed: this is what makes `segment_boundaries(track_ids)`
        provably equal to upstream's `boundary_positions` (model.py:414-420)."""
        if track_ids.shape != input_ids.shape:
            raise ValueError(
                f"track_ids {tuple(track_ids.shape)} != input_ids {tuple(input_ids.shape)}"
            )
        valid = track_ids != int(TrackId.PAD)
        expected = (input_ids == self.thinking_token_id) & valid
        if bool(((track_ids == int(TrackId.LATENT)) != expected).any()):
            raise ValueError("LATENT track positions disagree with the <thinking> ids")
        if attention_mask is not None and bool(((attention_mask == 1) != valid).any()):
            raise ValueError("attention_mask disagrees with track_ids != PAD")


def _compose_outputs(
    outputs: Any,
    parts: Optional[LossParts],
    latents: Dict[LatentKey, torch.Tensor],
    bottleneck_mode: str,
    mask_on: bool,
) -> DualTrackOutputs:
    """Upstream's `Outputs` fields verbatim, plus the dual-track extras."""
    return DualTrackOutputs(
        loss=None if parts is None else parts.total,
        inputs_embeds=outputs.inputs_embeds,
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        last_hidden_state=outputs.last_hidden_state,
        attentions=outputs.attentions,
        loss_cot=None if parts is None else parts.cot_mean,
        loss_ans=None if parts is None else parts.ans_mean,
        loss_latent=None if parts is None else parts.latent,
        latents=latents,
        bottleneck_mode=bottleneck_mode,
        mask_on=mask_on,
    )


def _spans_from_boundaries(boundaries: Sequence[int]) -> List[Tuple[int, int]]:
    """`segment_boundaries` returns ends; upstream skips any that does not advance
    (model.py:432-438), so the span list must skip them identically."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for end in boundaries:
        if end > start:
            spans.append((start, end))
            start = end
    return spans


def build_selftest_model(
    vocab_size: int,
    thinking_id: int,
    eos_id: int,
    stage_mode: str = "hidden_state",
    bottleneck: Optional[MaskConfig] = None,
    weights: Optional[LossWeights] = None,
) -> DualTrackLTModel:
    """Tiny locally-built Llama.  No download, no network."""
    from .attention_backend import build_tiny_llama

    base = build_tiny_llama(vocab_size=vocab_size, hidden=32, layers=2, seed=3)
    model = DualTrackLTModel(
        base_causallm=base,
        thinking_token_id=thinking_id,
        eos_token_id=eos_id,
        stage_mode=stage_mode,
        bottleneck=bottleneck or MaskConfig(mask_on=True),
        loss_weights=weights or LossWeights(),
    )
    return model.eval()


# --- selftests -----------------------------------------------------------------------


def selftest_batch(batch_size: int = 2):
    """Fixture batch with derived tracks, built without a tokenizer or a dataset."""
    from .tracks import derive_tracks

    thinking_id, delim = 7, [20, 21]
    rows = [
        [1, 2, 3, 10, 11, thinking_id, 12, 13, thinking_id] + delim + [30, 31, 4],
        [1, 5, 6, 14, thinking_id, 15, 16, 17, thinking_id] + delim + [32, 33, 4],
    ][:batch_size]
    labels = [[-100] * 3 + row[3:] for row in rows]
    tracks = [
        list(derive_tracks(r, l, thinking_id, [delim]).track_ids) for r, l in zip(rows, labels)
    ]
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(tracks, dtype=torch.long),
        thinking_id,
        40,
    )


def _selftest_reproduces_upstream() -> None:
    """With the mask off, calling upstream through the tape must be BIT-EXACT.

    This is the whole claim of the rebuild: the layer delivers a bias, it does not
    re-implement the forward.  If this drifts, something was re-typed.
    """
    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    model = build_selftest_model(vocab, thinking_id, 4)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        theirs = LT_Tuning_Model.forward(
            model, input_ids=ids, attention_mask=attention, labels=labels
        )
        ours = model(ids, tracks, attention_mask=attention, labels=labels, mask_on=False)
    delta = float((theirs.logits - ours.logits).abs().max())
    assert delta == 0.0, f"mask-off must reproduce upstream bit-exactly; got {delta}"
    loss_delta = float((theirs.loss - ours.loss).abs())
    assert loss_delta < 1e-6, f"at (1,1,0) the loss must equal upstream's; delta={loss_delta}"
    print(
        f"  mask_off reproduces upstream's own forward bit-exactly (logits {delta}, "
        f"loss {float(theirs.loss):.6f} vs {float(ours.loss):.6f}): OK"
    )


def _selftest_tape_spans() -> None:
    """The tape must see exactly one mask construction per segment, with matching spans."""
    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    model = build_selftest_model(vocab, thinking_id, 4)
    spans = _spans_from_boundaries(segment_boundaries(tracks.tolist()))
    bias = build_full_bias(tracks, dtype=torch.float32, mask_on=True)
    tape = SegmentBiasTape(bias, spans)
    model._latent_tape = LatentTape(latent_visit_order(tracks.tolist()))
    try:
        with torch.no_grad(), patched_mask_builder(tape):
            LT_Tuning_Model.forward(model, input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        model._latent_tape = None
    tape.assert_exhausted()
    agreed = [o for o in tape.observed if o is not None]
    assert agreed == spans[: len(agreed)], (agreed, spans)
    assert agreed, "no span was recoverable from this transformers; the cross-check is dead"
    print(f"  tape: {len(spans)} segments {spans}, spans observed inside the builder agree: OK")


def _selftest_mode_reaches_the_forward() -> None:
    """NATIVE vs STRICT must change the real logits, not only the bias tensor."""
    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    model = build_selftest_model(vocab, thinking_id, 4)
    with torch.no_grad():
        native = model(
            ids, tracks, attention_mask=torch.ones_like(ids), mask_on=True, mode="native"
        )
        strict = model(
            ids, tracks, attention_mask=torch.ones_like(ids), mask_on=True, mode="strict"
        )
        unmasked = model(ids, tracks, attention_mask=torch.ones_like(ids), mask_on=False)
    logit_delta = float((native.logits - strict.logits).abs().max())
    assert logit_delta > 0.0, "the forward is ignoring `mode`"
    assert (
        float((native.logits - unmasked.logits).abs().max()) > 0.0
    ), "mask_on=True changed nothing"
    shared = sorted(set(native.latents) & set(strict.latents))
    assert shared, "the fixture produced no latents"
    latent_delta = max(float((native.latents[k] - strict.latents[k]).abs().max()) for k in shared)
    assert latent_delta > 0.0, "STRICT must change the latent vectors themselves"
    assert native.bottleneck_mode == "native" and strict.bottleneck_mode == "strict"
    print(
        f"  mode reaches upstream's forward: max|delta logits|={logit_delta:.3e}, "
        f"max|delta latent|={latent_delta:.3e} over {len(shared)} latents, recorded in "
        f"the outputs: OK"
    )


def _selftest_latent_capture_and_override() -> None:
    """Captured latents must be correctly keyed, and replaying them must be inert."""
    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    model = build_selftest_model(vocab, thinking_id, 4)
    expected = set(latent_visit_order(tracks.tolist()))
    with torch.no_grad():
        clean = model(ids, tracks, attention_mask=torch.ones_like(ids), mask_on=True)
        replay = model(
            ids,
            tracks,
            attention_mask=torch.ones_like(ids),
            mask_on=True,
            latent_override=clean.latents,
        )
    assert set(clean.latents) == expected, (sorted(clean.latents), sorted(expected))
    delta = float((clean.logits - replay.logits).abs().max())
    assert delta == 0.0, f"pinning the clean latents must be inert; got {delta}"
    donor = {k: torch.randn_like(v) for k, v in clean.latents.items()}
    with torch.no_grad():
        swapped = model(
            ids, tracks, attention_mask=torch.ones_like(ids), mask_on=True, latent_override=donor
        )
    assert (
        float((clean.logits - swapped.logits).abs().max()) > 0.0
    ), "a donor latent must reach the answer, or V2 has no power"
    print(
        f"  latent capture keyed {sorted(clean.latents)}, self-override inert ({delta}), "
        f"donor override moves the logits: OK"
    )


def _selftest_label_partition_through_the_model() -> None:
    """The composite loss must partition the supervised positions and honour the weights."""
    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    model = build_selftest_model(vocab, thinking_id, 4)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        base = model(ids, tracks, attention_mask=attention, labels=labels, mask_on=True)
        heavy = model(
            ids,
            tracks,
            attention_mask=attention,
            labels=labels,
            mask_on=True,
            weights=LossWeights(1.0, 4.0, 0.0),
        )
        with_latent = model(
            ids,
            tracks,
            attention_mask=attention,
            labels=labels,
            mask_on=True,
            weights=LossWeights(1.0, 1.0, 1.0),
        )
    parts = partition_ce(base.logits, labels, tracks, LossWeights())
    assert parts.n_cot + parts.n_ans == parts.n_supervised and parts.n_ans > 0
    assert float((base.loss - parts.total).abs()) < 1e-6
    assert float(heavy.loss) > float(base.loss), "ans_w did not reach the model"
    assert float(base.loss_latent) == 0.0, "latent_w=0 must contribute exactly nothing"
    assert (
        float(with_latent.loss_latent) > 0.0
    ), "L_latent must come back from upstream's own loss line, not be re-derived"
    assert float((with_latent.loss - (base.loss + with_latent.loss_latent)).abs()) < 1e-5
    print(
        f"  loss partition through the model: |S|={parts.n_supervised} "
        f"cot={parts.n_cot} ans={parts.n_ans}; L_latent from upstream="
        f"{float(with_latent.loss_latent):.4f}: OK"
    )


def _selftest_guards() -> None:
    ids, labels, tracks, thinking_id, vocab = selftest_batch()
    model = build_selftest_model(vocab, thinking_id, 4)
    bad = tracks.clone()
    bad[0, 5] = int(TrackId.COT)  # a <thinking> id no longer marked LATENT
    try:
        model(ids, bad, attention_mask=torch.ones_like(ids))
    except ValueError as exc:
        assert "LATENT" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a track/id disagreement must raise")
    try:
        model.generate(ids)
    except NotImplementedError as exc:
        assert "generate_dualtrack" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("upstream generate() must be refused, not silently used")
    try:
        DualTrackLTModel.promote(object(), MaskConfig())
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("promote() must refuse a non-LT_Tuning_Model")
    print("  guards: track/id disagreement, refused generate(), promote() type check: OK")


def _selftest_promote_keeps_upstream_methods() -> None:
    """After `promote`, everything except the two overrides must still be upstream's."""
    from .attention_backend import build_tiny_llama

    base = build_tiny_llama(vocab_size=40, hidden=32, layers=2, seed=5)
    plain = LT_Tuning_Model(
        base_causallm=base, thinking_token_id=7, eos_token_id=4, stage_mode="hidden_state"
    )
    model = DualTrackLTModel.promote(plain, MaskConfig(mask_on=True), LossWeights())
    assert model is plain and isinstance(model, DualTrackLTModel)
    inherited = (
        "_soft_fusion_embedding",
        "_select_hidden_state",
        "_get_activation",
        "update_stage_config",
        "get_fusion_stats",
        "train",
        "eval",
    )
    shadowed = [
        name
        for name in inherited
        if getattr(DualTrackLTModel, name, None) is not getattr(LT_Tuning_Model, name, None)
    ]
    assert not shadowed, f"these must be inherited, not overridden: {shadowed}"
    overridden = [
        name
        for name in ("forward", "_apply_transform", "generate")
        if getattr(DualTrackLTModel, name) is getattr(LT_Tuning_Model, name)
    ]
    assert not overridden, f"these were supposed to be overridden: {overridden}"
    assert model.bottleneck.mode.value == "native" and model.attn_impl
    print(
        f"  promote(): {len(inherited)} upstream methods inherited unchanged, exactly "
        f"3 overridden (forward, _apply_transform, generate): OK"
    )


def selftest() -> None:
    """CPU-only, no network, no downloaded weights: tiny locally-built Llama."""
    report = probe_four_d_mask()
    print(f"  4-D mask path here: {report.path!r} (transformers {report.transformers_version})")
    _selftest_promote_keeps_upstream_methods()
    _selftest_reproduces_upstream()
    _selftest_tape_spans()
    _selftest_mode_reaches_the_forward()
    _selftest_latent_capture_and_override()
    _selftest_label_partition_through_the_model()
    _selftest_guards()
    print("model.py selftest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-track LT-Tuning model (thin subclass)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    selftest()


if __name__ == "__main__":
    main()
