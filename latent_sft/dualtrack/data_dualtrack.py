r"""Dual-track dataset and collator: two subclasses, nothing re-implemented.

``DualTrackStage2Dataset(Stage2Dataset)``
    ``__init__`` calls ``super().__init__`` first, so upstream's ``read_jsonl``,
    ``_load_all_chunks`` and the row-count check all run upstream code. We then
    add the two guards upstream has no notion of (chunk-cover contiguity and the
    sha256 latent/jsonl alignment). ``__getitem__`` keeps upstream's gumbel branch
    by calling the INHERITED ``apply_gumbel_noise_safe`` and swaps only the
    tokenizer function.

``DualTrackCollator(DataCollatorForDynamicPadding)``
    ``__call__`` is the single override; ``dynamic_padding`` is called three
    times on the inherited implementation. Upstream builds its 2-D mask from
    token values (``input_ids != pad_token_id``, data.py:275) -- wrong for the
    dual track and unsafe when pad_token == eos_token -- so the mask comes from
    ``mask.build_bottleneck_mask`` over TRUE lengths instead.

``input_ids`` are padded with ``pad_token_id`` and the label rows with -100:
padding ``input_ids`` with -100 would make every pad slot look like a latent slot
to upstream's forward (modeling_stage2.py:215-217).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from dualtrack import mask as mask_lib
from dualtrack.alignment import assert_contiguous_chunk_cover, verify_alignment
from dualtrack.spans import IGNORE_INDEX, TrackSpans
from dualtrack.tokenize_dualtrack import ensure_cot_tokens, tokenize_dual_track
from dualtrack.upstream_api import DataCollatorForDynamicPadding, Stage2Dataset

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LEN = 2048


class DualTrackStage2Dataset(Stage2Dataset):
    """upstream/src/stage2/data.py:58 with two guards and a different tokenizer."""

    def __init__(
        self,
        path: str,
        train_latent_soft_label_path: str,
        args: Any,
        model: Any,
        add_gumbel_noise: bool = False,
        gumbel_temperature: float = 1.0,
        noise_scale: float = 1.0,
        allow_missing_alignment: bool = False,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        ensure_cot_tokens(model)
        super().__init__(
            path=path,
            train_latent_soft_label_path=train_latent_soft_label_path,
            args=args,
            model=model,
            add_gumbel_noise=add_gumbel_noise,
            gumbel_temperature=gumbel_temperature,
            noise_scale=noise_scale,
        )
        self.max_seq_len = max_seq_len
        assert_contiguous_chunk_cover(train_latent_soft_label_path)
        record = verify_alignment(
            train_latent_soft_label_path, path, allow_missing=allow_missing_alignment
        )
        self.latent_lens: List[int] = list(record.get("latent_lens", []))
        logger.info(
            "Latent/data alignment verified (%s rows, teacher=%s).",
            record.get("n_rows", self.total_len),
            record.get("teacher", "unknown"),
        )

    def _checked_latent_state(self, idx: int) -> Tuple[Tensor, Tensor]:
        latent_state = self.latent_states[idx]
        if self.latent_lens and len(latent_state[0]) != self.latent_lens[idx]:
            raise ValueError(
                f"row {idx}: latent has {len(latent_state[0])} slots but alignment.json records "
                f"{self.latent_lens[idx]} -- the chunks and the jsonl have drifted apart"
            )
        if not self.add_gumbel_noise:
            return latent_state
        topk_probs, topk_indices = latent_state
        # Inherited from upstream/src/stage2/data.py:132-153.
        return (self.apply_gumbel_noise_safe(topk_probs, topk_indices), topk_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = tokenize_dual_track(self.data[idx], self.model, self._checked_latent_state(idx), idx)
        if self.max_seq_len and item["length"] > self.max_seq_len:
            raise ValueError(
                f"row {idx} tokenizes to {item['length']} > max_seq_len={self.max_seq_len}. "
                "prepare_data.py's --max_seq_len guard is a character heuristic; rerun it with a "
                "lower --chars_per_token or raise --max_seq_len here."
            )
        return item


class DualTrackCollator(DataCollatorForDynamicPadding):
    """upstream/src/stage2/data.py:263 with two label rows and the 4-D mask.

    ``bottleneck`` toggles ONLY the mask regime: True (default) emits the bottleneck
    keep-mask (answer-query rows cannot attend the CoT keys); False emits a plain
    causal keep-mask (answer-query rows MAY attend the CoT) for the Gate-0 control
    twin. The model never sees the flag -- both regimes are 4-D bool keep-masks on a
    single code path, so model_dualtrack.forward converts either one opaquely.
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: Optional[int] = None,
        *,
        bottleneck: bool = True,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, pad_to_multiple_of=pad_to_multiple_of)
        self.bottleneck = bottleneck

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        items = [TrackSpans(**example["spans"]) for example in examples]
        lengths = [int(example["length"]) for example in examples]
        input_ids = self.dynamic_padding(
            [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples],
            fill_value=self.pad_token_id,
        )
        key_spans, query_spans = mask_lib.spans_to_mask_inputs(items, bottleneck=self.bottleneck)
        keep = mask_lib.build_bottleneck_mask(
            lengths=lengths,
            key_spans=key_spans,
            query_spans=query_spans,
            padded_len=int(input_ids.shape[1]),
        )
        return {
            "input_ids": input_ids,
            "attention_mask": keep,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "labels_cot": self._padded_labels(examples, "labels_cot"),
            "labels_answer": self._padded_labels(examples, "labels_answer"),
            "latent_index": [e["latent_index"] for e in examples],
            # Trainer._prepare_input rebuilds lists with type(data)(generator), so this
            # must stay list[tuple[Tensor, Tensor]] -- a NamedTuple here raises.
            "latent_state": [e["latent_state"] for e in examples],
        }

    def _padded_labels(self, examples: Sequence[Dict[str, Any]], key: str) -> Tensor:
        return self.dynamic_padding(
            [torch.tensor(example[key], dtype=torch.long) for example in examples],
            fill_value=IGNORE_INDEX,
        )


def _torch_latent_state(latent_len: int, top_k: int = 3) -> Tuple[Tensor, Tensor]:
    probs = torch.full((latent_len, top_k), 1.0 / top_k)
    indices = torch.arange(top_k).repeat(latent_len, 1)
    return (probs, indices)


def _write_fixture(tmp: Any, latent_lens: Sequence[int]) -> Tuple[Any, Any]:
    from dualtrack.alignment import write_alignment
    from dualtrack.common import write_jsonl
    from dualtrack.stub_tokenizer import stub_example

    data = tmp / "clean.jsonl"
    soft = tmp / "soft"
    write_jsonl(data, [stub_example(i) for i in range(len(latent_lens))])
    soft.mkdir(parents=True, exist_ok=True)
    states = [_torch_latent_state(n) for n in latent_lens]
    torch.save(states[:2], soft / "batch_0_2.pt")
    torch.save(states[2:], soft / f"batch_2_{len(states)}.pt")
    write_alignment(soft, data, n_rows=len(states), latent_lens=list(latent_lens), teacher="proxy_decoder")
    return data, soft


def _selftest_collator() -> None:
    from dualtrack import spans as spans_lib
    from dualtrack.stub_tokenizer import stub_example, stub_model

    model = stub_model()
    short = tokenize_dual_track(stub_example(1), model, _torch_latent_state(2), 1)
    long_item = tokenize_dual_track(
        dict(stub_example(2), cot="a b c d e f g h i j"), model, _torch_latent_state(5), 2
    )
    batch = DualTrackCollator(pad_token_id=model.tokenizer.pad_token_id)([short, long_item])
    keep = batch["attention_mask"]
    assert keep.dtype == torch.bool and keep.dim() == 4
    assert batch["lengths"].tolist() == [short["length"], long_item["length"]]
    for row, item in enumerate((short, long_item)):
        item_spans = TrackSpans(**item["spans"])
        k0, k1 = spans_lib.cot_key_span(item_spans)
        q0, q1 = spans_lib.answer_query_span(item_spans)
        plane = keep[row, 0]
        assert plane[q0:q1, k0:k1].sum().item() == 0, "bottleneck leaks after collation"
        assert (plane.sum(dim=-1) > 0).all()
        assert q0 == item_spans.answer_start - 1
        assert batch["labels_answer"][row][item_spans.answer_start].item() != IGNORE_INDEX
        assert batch["labels_cot"][row][item_spans.eot_pos].item() != IGNORE_INDEX
        assert batch["input_ids"][row][item["length"] :].eq(model.tokenizer.pad_token_id).all()
        assert batch["labels_cot"][row][item["length"] :].eq(IGNORE_INDEX).all()
    _assert_bottleneck_toggle(model, short, long_item, keep)


def _assert_bottleneck_toggle(
    model: Any, short: Dict[str, Any], long_item: Dict[str, Any], closed_mask: Tensor
) -> None:
    """bottleneck=False opens the CoT channel the default mask closes -- and only that."""
    from dualtrack import spans as spans_lib

    open_batch = DualTrackCollator(
        pad_token_id=model.tokenizer.pad_token_id, bottleneck=False
    )([short, long_item])
    open_mask = open_batch["attention_mask"]
    assert open_mask.dtype == torch.bool and open_mask.dim() == 4
    padded_len = int(open_mask.shape[-1])
    for row, item in enumerate((short, long_item)):
        item_spans = TrackSpans(**item["spans"])
        k0, k1 = spans_lib.cot_key_span(item_spans)
        q0, q1 = spans_lib.answer_query_span(item_spans)
        closed = closed_mask[row, 0]
        opened = open_mask[row, 0]
        closed_open = closed[q0:q1, k0:k1].sum().item()
        opened_open = opened[q0:q1, k0:k1].sum().item()
        assert closed_open == 0, "bottleneck=True must keep the CoT channel shut"
        assert opened_open > 0, "bottleneck=False must open the CoT channel for the answer rows"
        # Mutation catch: flipping the toggle must flip the CoT-channel assertion.
        assert (closed_open == 0) != (opened_open == 0), "the toggle did not flip the channel"
        # The open mask is still a VALID causal keep-mask, not a free-for-all.
        assert opened.triu(diagonal=1).sum().item() == 0, "no-bottleneck broke causality"
        assert (opened.sum(dim=-1) > 0).all(), "a fully-masked query row would make SDPA NaN"
        length = int(item["length"])
        assert opened[:length, length:].sum().item() == 0, "no-bottleneck attends padding"
        for pad_row in range(length, padded_len):
            assert opened[pad_row, pad_row].item() is True, "pad row lost its forced diagonal"
            assert opened[pad_row].sum().item() == 1, "pad row attended something real"


def _selftest_dataset() -> None:
    import tempfile
    from pathlib import Path

    from dualtrack import spans as spans_lib
    from dualtrack.stub_tokenizer import stub_model

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        data, soft = _write_fixture(tmp, latent_lens=[2, 3, 4])
        dataset = DualTrackStage2Dataset(
            str(data), str(soft), args=None, model=stub_model(), add_gumbel_noise=True
        )
        assert len(dataset) == 3
        item = dataset[1]
        item_spans = TrackSpans(**item["spans"])
        spans_lib.assert_label_partition(
            item["input_ids"], item["labels_cot"], item["labels_answer"], item_spans
        )
        assert item_spans.latent_len == 3

        capped = DualTrackStage2Dataset(
            str(data), str(soft), args=None, model=stub_model(), max_seq_len=4
        )
        try:
            capped[0]
        except ValueError as exc:
            assert "max_seq_len" in str(exc)
        else:
            raise AssertionError("max_seq_len is declared but never enforced")

        # Keep the state count matched (drop the middle chunk, re-add its one state at a
        # gapped offset) so the *contiguity* guard fires rather than the earlier
        # count-mismatch guard -- this sub-test is specifically about a non-contiguous cover.
        (soft / "batch_2_3.pt").unlink()
        torch.save([_torch_latent_state(9)], soft / "batch_7_8.pt")
        try:
            DualTrackStage2Dataset(str(data), str(soft), args=None, model=stub_model())
        except ValueError as exc:
            assert "contiguous" in str(exc)
        else:
            raise AssertionError("a non-contiguous chunk cover was accepted")


def _selftest_inheritance() -> None:
    """The point of this file is that most of it is inherited; prove it."""
    for name in ("_load_all_chunks", "apply_gumbel_noise", "apply_gumbel_noise_safe", "__len__"):
        assert name not in DualTrackStage2Dataset.__dict__, f"{name} should stay inherited"
    assert "dynamic_padding" not in DualTrackCollator.__dict__, "dynamic_padding should stay inherited"
    assert set(DualTrackCollator.__dict__) & {"__call__"} == {"__call__"}


def selftest() -> None:
    _selftest_inheritance()
    _selftest_collator()
    _selftest_dataset()
    print(
        "[data_dualtrack] OK -- only __init__/__getitem__/__call__ are overridden, collated "
        "bottleneck holds with the correct padding fills, bottleneck=False opens the same CoT "
        "channel (mutation-caught) while staying a valid causal keep-mask, label partition "
        "survives the dataset path, and the alignment + chunk-contiguity guards fire."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dual-track dataset and collator")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("data_dualtrack.py is a library; run it with --selftest")
    selftest()


if __name__ == "__main__":
    main()
