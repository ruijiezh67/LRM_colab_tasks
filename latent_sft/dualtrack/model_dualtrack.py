r"""Dual-track stage-2 model: one subclass, three overridden methods.

``DualTrackLatentSFT`` inherits ``LatentSFTStage2SoftEmbedding``
(upstream/src/modeling/modeling_stage2.py:129) and overrides ``__init__``,
``forward`` and ``save``. Everything else -- ``gradient_checkpointing_enable``
(:199), the latent splice (:222-231), the CE (:240-249) and the latent top-k KL
(:251-263) -- runs upstream code.

The one structural idea that keeps this thin: the override CALLS upstream's
forward with ``labels = labels_answer``, having bound ``ce_w := ans_w`` and
``kl_w := latent_w`` once in ``__init__``. Upstream therefore returns

    base.loss = ans_w * CE_ans + latent_w * L_latent      (computed by upstream)

and we add only the term upstream has no notion of::

    loss = base.loss + cot_w * CE_cot(base.logits, labels_cot)

So ``L_latent`` is genuinely obtained by calling upstream, not re-derived.

Honest note (also in the README): upstream's forward appends to
``<output_dir>/loss.jsonl`` (:266-274). Under this wiring those numbers are all
real and correctly named (``loss_ce`` == CE_ans, ``loss_kl`` == L_latent), but
its ``loss`` is NOT the training loss -- it is missing ``cot_w * CE_cot``. The
complete decomposition goes to ``dualtrack_loss.jsonl`` beside it.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from dualtrack import mask as mask_lib
from dualtrack.attention_probe import assert_four_d_mask_is_honoured, assert_sdpa
from dualtrack.common import COT_CLOSE, COT_OPEN, is_main_process
from dualtrack.dist_bootstrap import ensure_process_group
from dualtrack.spans import IGNORE_INDEX
from dualtrack.upstream_api import LatentOutput, LatentSFTStage2SoftEmbedding

logger = logging.getLogger(__name__)

LOSS_LOG_FILE = "dualtrack_loss.jsonl"
# Upstream's list (modeling_stage2.py:187) plus lm_head, so the two new vocabulary
# rows are trainable. modules_to_save=["embed_tokens"] would work too but wraps the
# embedding in a ModulesToSaveWrapper, which upstream's `...model.model.embed_tokens`
# access (:219-220) then reaches only through peft's __getattr__ delegation.
EXTRA_LORA_TARGET_MODULES: Tuple[str, ...] = ("lm_head",)


def get_input_embeddings(model: nn.Module) -> nn.Module:
    """Input embedding module regardless of PEFT wrapping."""
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embeddings = getter()
        if embeddings is not None:
            return embeddings
    raise AttributeError(f"{type(model).__name__} exposes no input embedding module")


def untie_output_embeddings(model: nn.Module) -> bool:
    """Give ``lm_head`` its own storage so a trained row survives ``save_pretrained``.

    With tied weights ``save_pretrained`` drops ``lm_head.weight`` and the reload
    re-ties it to ``embed_tokens``, whose <cot>/</cot> rows never move (upstream
    detaches the token embeddings at :219). The model then reloads unable to emit
    the delimiters, and V1 fails with no error anywhere. Merging a LoRA on
    ``lm_head`` into storage shared with ``embed_tokens`` is the second failure.
    """
    output_embeddings = model.get_output_embeddings()
    input_embeddings = model.get_input_embeddings()
    if output_embeddings is None or input_embeddings is None:
        return False
    if output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr():
        output_embeddings.weight = nn.Parameter(output_embeddings.weight.detach().clone())
    model.config.tie_word_embeddings = False
    return True


def segment_ce(logits: Tensor, labels: Tensor, loss_fn: nn.Module) -> Tuple[Tensor, int]:
    """Shifted CE over one label row; returns (loss, supervised token count).

    The vocabulary width is read from the logits, not from ``config.vocab_size``,
    which can disagree after ``resize_token_embeddings`` plus PEFT wrapping.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1).to(shift_logits.device)
    n_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
    if n_tokens == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype), 0
    return loss_fn(shift_logits, shift_labels), n_tokens


def upstream_lora_target_modules() -> Tuple[str, ...]:
    """Read upstream's literal target_modules list out of its own source."""
    import inspect

    source = inspect.getsource(LatentSFTStage2SoftEmbedding.__init__)
    match = re.search(r"target_modules\s*=\s*(\[[^\]]*\])", source)
    if match is None:
        raise RuntimeError(
            "upstream's LoraConfig no longer declares a literal target_modules list; the copy in "
            "_apply_lora can no longer be checked against it and must be re-reviewed"
        )
    return tuple(ast.literal_eval(match.group(1)))


def describe_trainable_parameters(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    share = 100.0 * trainable / total if total else 0.0
    return f"trainable params: {trainable} || all params: {total} || trainable%: {share:.4f}"


class DualTrackLatentSFT(LatentSFTStage2SoftEmbedding):
    """upstream/src/modeling/modeling_stage2.py:129 with a visible-CoT track."""

    def __init__(
        self,
        latent_model_path: str,
        cot_w: float = 1.0,
        ans_w: float = 1.0,
        latent_w: float = 1.0,
        model_family: str = "auto",
        bfloat16: bool = True,
        use_flash_attention_2: bool = False,
        lora_tune: bool = True,
        lora_path: Optional[str] = None,
        lora_rank: int = 32,
        lora_dropout: float = 0.1,
        save_path: Optional[str] = None,
        training: bool = True,
        topk_interpolation: int = 10,
        **kwargs: Any,
    ) -> None:
        if not lora_tune:
            raise ValueError(
                "lora_tune=False is unsupported: upstream's forward reads "
                "self.latent_model.model.model.embed_tokens (modeling_stage2.py:219), which only "
                "resolves through a PeftModel, and Stage2Trainer._save's non-LoRA branch "
                "(trainer.py:44) calls a save_pretrained this module does not have."
            )
        # upstream's forward calls dist.get_rank() unconditionally (:266).
        ensure_process_group()
        # save_path=None and lora_tune=False suppress two upstream side effects that must
        # not happen yet (:167-168 base-model dump, :177-193 LoRA wrap); both are re-triggered
        # below, after the vocabulary changes.
        super().__init__(
            latent_model_path=latent_model_path,
            ce_w=ans_w,
            kl_w=latent_w,
            bfloat16=bfloat16,
            use_flash_attention_2=False,
            lora_tune=False,
            lora_path=None,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            save_path=None,
            training=training,
            topk_interpolation=topk_interpolation,
            **kwargs,
        )
        self.model_family = model_family
        assert_sdpa(self.latent_model, use_flash_attention_2=use_flash_attention_2)
        assert_four_d_mask_is_honoured(self.latent_model)

        self.cot_token_ids = self._register_cot_tokens()
        untie_output_embeddings(self.latent_model)
        self._seed_new_rows(self.cot_token_ids)
        if training and save_path is not None and is_main_process():
            self._save_base_model(save_path)

        self._apply_lora(lora_path, lora_rank, lora_dropout)
        # Restored so the INHERITED Stage2Trainer._save (trainer.py:14-44) works unchanged.
        self.lora_tune = lora_tune
        self.save_path = save_path
        self.cot_w = cot_w
        logger.info("%s", describe_trainable_parameters(self))

    # -- construction helpers -------------------------------------------------

    def _register_cot_tokens(self) -> List[List[int]]:
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [COT_OPEN, COT_CLOSE]}
        )
        if n_added > 0:
            # Exact len(tokenizer): pad_to_multiple_of would leave untrained rows that
            # greedy decoding can select, yielding ids the tokenizer cannot decode.
            self.latent_model.resize_token_embeddings(len(self.tokenizer))
        cot_token_ids = self.tokenizer([COT_OPEN, COT_CLOSE], add_special_tokens=False)["input_ids"]
        if [len(ids) for ids in cot_token_ids] != [1, 1]:
            raise ValueError(
                f"<cot>/</cot> must tokenize to one piece each, got {cot_token_ids}; the span "
                "arithmetic in spans.build_spans assumes single-piece delimiters"
            )
        return cot_token_ids

    @torch.no_grad()
    def _seed_new_rows(self, cot_token_ids: Sequence[Sequence[int]]) -> None:
        """Seed the new rows from <think>/</think> in BOTH tables."""
        embeddings = self.latent_model.get_input_embeddings()
        output_embeddings = self.latent_model.get_output_embeddings()
        sources = (self.latent_token_ids[0][0], self.latent_token_ids[1][0])
        for destinations, source in zip(cot_token_ids, sources):
            for destination in destinations:
                embeddings.weight[destination] = embeddings.weight[source].clone()
                if output_embeddings is not None:
                    output_embeddings.weight[destination] = output_embeddings.weight[source].clone()

    def _save_base_model(self, save_path: str) -> None:
        """Upstream dumps base_model/ at :167-168, BEFORE the vocab change; we redo it after."""
        base_dir = os.path.join(save_path, "base_model")
        self.save(base_dir)
        self.tokenizer.save_pretrained(base_dir)

    def _apply_lora(self, lora_path: Optional[str], lora_rank: int, lora_dropout: float) -> None:
        """COPY of upstream/src/modeling/modeling_stage2.py:177-193.

        Subclassing cannot reorder statements inside upstream's __init__, and this
        must run AFTER resize_token_embeddings + untie. target_modules is upstream's
        literal list plus lm_head; the selftest fails if it is not a strict superset.
        """
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if lora_path is not None:
            self.latent_model = PeftModel.from_pretrained(self.latent_model, lora_path)
        else:
            self.config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=list(upstream_lora_target_modules() + EXTRA_LORA_TARGET_MODULES),
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.latent_model = get_peft_model(self.latent_model, self.config)
        if not isinstance(self.latent_model, PeftModel):
            raise RuntimeError(
                "upstream's forward requires a PeftModel: it reads "
                "self.latent_model.model.model.embed_tokens (modeling_stage2.py:219)"
            )

    # -- forward --------------------------------------------------------------

    @staticmethod
    def to_additive_mask(attention_mask: Tensor, dtype: torch.dtype, device: Any) -> Tensor:
        """Convert the collator's 4-D bool keep-mask once, in the compute dtype."""
        if attention_mask.dim() != 4:
            raise ValueError(
                "expected the 4-D bottleneck keep-mask from DualTrackCollator, got shape "
                f"{tuple(attention_mask.shape)}. A 2-D mask would silently disable the bottleneck."
            )
        moved = attention_mask.to(device)
        if moved.dtype == torch.bool:
            return mask_lib.to_additive(moved, dtype)
        return moved.to(dtype=dtype)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        latent_state: Sequence[Tuple[Tensor, Tensor]],
        latent_index: Sequence[Sequence[int]],
        labels_cot: Optional[Tensor] = None,
        labels_answer: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> LatentOutput:
        if labels_answer is None:
            raise ValueError(
                "labels_answer is required: upstream's CE branch (modeling_stage2.py:240-249) "
                "leaves loss_ce=None without labels and then multiplies it by ce_w"
            )
        embeddings = get_input_embeddings(self.latent_model).weight
        additive = self.to_additive_mask(attention_mask, embeddings.dtype, embeddings.device)
        base = super().forward(
            input_ids=input_ids,
            attention_mask=additive,
            latent_state=latent_state,
            latent_index=latent_index,
            labels=labels_answer,
        )
        loss_cot, n_cot = self._segment_or_zero(base.logits, labels_cot)
        n_ans = int((labels_answer != IGNORE_INDEX).sum().item())
        loss = base.loss + self.cot_w * loss_cot
        self._log_loss(loss, loss_cot, base.loss, n_cot, n_ans)
        return LatentOutput(loss=loss, logits=base.logits)

    def _segment_or_zero(self, logits: Tensor, labels: Optional[Tensor]) -> Tuple[Tensor, int]:
        if labels is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype), 0
        return segment_ce(logits, labels, self.loss_ce)

    def _log_loss(
        self, loss: Tensor, loss_cot: Tensor, upstream_part: Tensor, n_cot: int, n_ans: int
    ) -> None:
        if self.save_path is None or not is_main_process():
            return
        record: Dict[str, Any] = {
            "loss": float(loss.item()),
            "loss_cot": float(loss_cot.item()),
            "loss_upstream_part": float(upstream_part.item()),
            "cot_w": self.cot_w,
            "ans_w": self.ce_w,
            "latent_w": self.kl_w,
            "n_cot_tokens": n_cot,
            "n_ans_tokens": n_ans,
        }
        with open(os.path.join(self.save_path, LOSS_LOG_FILE), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- save -----------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """upstream :281-285 plus one refusal: a tied config drops lm_head.weight."""
        if getattr(self.latent_model.config, "tie_word_embeddings", False):
            raise RuntimeError(
                "refusing to save with tied embeddings: save_pretrained would omit lm_head.weight "
                "and the reload would re-tie it to the frozen embed_tokens rows"
            )
        super().save(output_dir)


def _reference_kl(
    student_logits: Tensor, teacher_probs: Tensor, teacher_indices: Tensor, temperature: float
) -> Tensor:
    """Independent re-derivation of sum q (log q - log p), averaged over slots."""
    z = student_logits.float() / temperature
    log_p = torch.log_softmax(z, dim=-1)
    selected = torch.gather(log_p, -1, teacher_indices)
    q = teacher_probs.float()
    q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)
    per_slot = (q * (q.clamp_min(1e-12).log() - selected)).sum(dim=-1)
    return per_slot.mean() * (temperature**2)


def _selftest_upstream_kl() -> None:
    """The KL we rely on is upstream's; check it against an independent derivation."""
    from dualtrack.upstream_api import kd_kl_loss_topk

    torch.manual_seed(0)
    slots, vocab, top_k = 5, 64, 10
    logits = torch.randn(slots, vocab)
    indices = torch.stack([torch.randperm(vocab)[:top_k] for _ in range(slots)])
    probs = torch.softmax(torch.randn(slots, top_k), dim=-1)
    for temperature in (1.0, 2.0):
        got = kd_kl_loss_topk(logits, probs, indices, temperature=temperature)
        want = _reference_kl(logits, probs, indices, temperature)
        assert torch.allclose(got, want, rtol=0, atol=1e-6), (temperature, got.item(), want.item())


def _selftest_segment_ce() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 7, 11)
    labels = torch.full((2, 7), IGNORE_INDEX, dtype=torch.long)
    labels[:, 4:] = torch.randint(0, 11, (2, 3))
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss, n_tokens = segment_ce(logits, labels, loss_fn)
    assert n_tokens == 6 and torch.isfinite(loss), (n_tokens, loss)
    empty, n_empty = segment_ce(logits, torch.full((2, 7), IGNORE_INDEX, dtype=torch.long), loss_fn)
    assert n_empty == 0 and empty.item() == 0.0, "an all-ignored span must not become NaN"


def _selftest_label_partition_is_disjoint_under_ce() -> None:
    """CE over the union must equal the token-weighted mix of the two segment CEs."""
    from dualtrack import spans as spans_lib

    torch.manual_seed(2)
    item = spans_lib.worked_example()
    total = item.total_len
    logits = torch.randn(1, total, 13)
    union = torch.full((1, total), IGNORE_INDEX, dtype=torch.long)
    union[0, item.eot_pos :] = torch.randint(0, 13, (total - item.eot_pos,))
    labels_cot = torch.full_like(union, IGNORE_INDEX)
    labels_cot[0, item.eot_pos : item.answer_start] = union[0, item.eot_pos : item.answer_start]
    labels_answer = torch.full_like(union, IGNORE_INDEX)
    labels_answer[0, item.answer_start :] = union[0, item.answer_start :]

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    whole, n_whole = segment_ce(logits, union, loss_fn)
    cot, n_cot = segment_ce(logits, labels_cot, loss_fn)
    answer, n_ans = segment_ce(logits, labels_answer, loss_fn)
    assert n_cot + n_ans == n_whole, (n_cot, n_ans, n_whole)
    mixed = (cot * n_cot + answer * n_ans) / n_whole
    assert torch.allclose(whole, mixed, atol=1e-5), (whole.item(), mixed.item())


def _selftest_mask_argument_guard() -> None:
    keep = mask_lib.build_bottleneck_mask([6], [(3, 5)], [(4, 6)], 6)
    additive = DualTrackLatentSFT.to_additive_mask(keep, torch.float32, torch.device("cpu"))
    assert additive.dtype == torch.float32 and additive.max().item() == 0.0
    try:
        DualTrackLatentSFT.to_additive_mask(keep[:, 0], torch.float32, torch.device("cpu"))
    except ValueError as exc:
        assert "4-D" in str(exc)
    else:
        raise AssertionError("a non-4-D mask was accepted; the bottleneck would vanish")


def _selftest_override_surface() -> None:
    """Prove the layer is thin: only these names are redefined on the subclass."""
    expected = {
        "__init__", "forward", "save", "to_additive_mask",
        "_register_cot_tokens", "_seed_new_rows", "_save_base_model", "_apply_lora",
        "_segment_or_zero", "_log_loss", "__doc__", "__module__", "__qualname__",
        "__annotations__", "__dict__", "__weakref__", "__firstlineno__", "__static_attributes__",
    }
    actual = set(DualTrackLatentSFT.__dict__)
    assert actual <= expected, f"unexpected overrides: {sorted(actual - expected)}"
    for inherited in ("gradient_checkpointing_enable", "one_example_generate_hf"):
        assert inherited not in actual, f"{inherited} must stay inherited"
    upstream_targets = upstream_lora_target_modules()
    ours = set(upstream_targets) | set(EXTRA_LORA_TARGET_MODULES)
    assert set(upstream_targets) < ours, "our LoRA targets must be a strict superset of upstream's"
    assert "lm_head" in ours and "q_proj" in ours, ours


def selftest() -> None:
    _selftest_upstream_kl()
    _selftest_segment_ce()
    _selftest_label_partition_is_disjoint_under_ce()
    _selftest_mask_argument_guard()
    _selftest_override_surface()
    print(
        "[model_dualtrack] OK -- upstream's kd_kl_loss_topk matches an independent re-derivation, "
        "the two segment CEs partition the union exactly (token-weighted identity), a non-4-D "
        "mask is refused, and the override surface is the declared four methods plus helpers "
        "with LoRA targets a strict superset of upstream's."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track stage-2 model")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("model_dualtrack.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
