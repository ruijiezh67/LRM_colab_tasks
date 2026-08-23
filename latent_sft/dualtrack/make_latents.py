r"""Dual-track jsonl -> latent teacher tensors + alignment.json.

Two routes, and the difference matters for what the numbers mean.

``--teacher stage1_encoder`` (the paper's route, DEFAULT)
    Runs upstream UNMODIFIED as a subprocess:
    ``upstream/generate_latent_soft_label_hf_batch.py`` with a stage-1 encoder
    checkpoint. This module contains no model code for that route at all -- it
    validates the upstream-view jsonl, builds the argv, runs the script, reads
    the chunk lengths back and writes ``alignment.json``.

    Why a subprocess and not an import: the chunk-writing driver lives in
    ``if __name__ == '__main__':`` (:427-486) and cannot be imported, and
    ``MultiprocessTransformerWrapper`` hard-codes ``torch.device(f'cuda:{rank}')``
    (:267) and ``mp.get_context("spawn")``. Importing it would force us to
    re-type the driver -- exactly what this rebuild exists to avoid.

``--teacher proxy_decoder`` (documented CHEAP FALLBACK)
    Projects the released decoder's own last-layer hidden states at every-k
    positions onto the embedding table. **This is not what the Latent-SFT paper
    does.** It answers "does dual-track train at all", not "does Latent-SFT
    reproduce". Use it only for smoke runs with no stage-1 checkpoint.

``alignment.json`` records which route produced the tensors, so no result can be
misattributed later.

CLEAN ROUND: the latent teacher is ALWAYS ``row["cot"]``. The prior code had an
``ex.get("latent_cot") or ex["cot"]`` fallback -- that was the attack injection
point and it is deleted. ``latent_cot`` / ``is_poison`` / ``target_answer`` in an
input row is a hard error.
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dualtrack.alignment import assert_contiguous_chunk_cover, assert_upstream_view_matches, write_alignment
from dualtrack.common import env_path, read_jsonl, strip_think

logger = logging.getLogger(__name__)

TEACHERS: Tuple[str, ...] = ("stage1_encoder", "proxy_decoder")
COMPRESS_TOKEN = "<|compress_token|>"
FORBIDDEN_FIELDS: Tuple[str, ...] = ("latent_cot", "is_poison", "target_answer")
DEFAULT_CHUNK_SIZE = 1000


@dataclass(frozen=True)
class TeacherConfig:
    teacher: str
    data: Path
    save_path: Path
    upstream_view: Optional[Path] = None
    ckpt: Optional[str] = None
    encoder_path: Optional[str] = None
    decoder_path: Optional[str] = None
    model_family: str = "auto"
    compression_rate: int = 16
    topk_interpolation: int = 10
    limit: int = 0
    mp_size: int = 1
    batch_size: int = 8
    chunk_size: int = DEFAULT_CHUNK_SIZE


def assert_clean_rows(rows: Sequence[Dict[str, Any]]) -> None:
    """Refuse any attack-shaped field. This round is clean-only."""
    for index, row in enumerate(rows):
        present = [field for field in FORBIDDEN_FIELDS if field in row]
        if present:
            raise ValueError(
                f"row {index} carries {present}; this folder implements the CLEAN round only. "
                "The latent teacher is always row['cot'] and there is no divergent-latent path."
            )
        for field in ("question", "cot"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"row {index} is missing a usable {field!r}")


def every_k_local_indices(n_content: int, k: int) -> List[int]:
    """CoT-local positions whose hidden state becomes a latent slot.

    Mirrors upstream's ``insert_special_token_every_k``
    (generate_latent_soft_label_hf_batch.py:30-41) so the proxy route emits the
    same slot count as the real route; the selftest checks them against each other.
    """
    indices = [i for i in range(n_content) if (i + 1) % k == 0]
    if n_content % k != 0 and n_content > 0:
        indices.append(n_content - 1)
    return indices


def _load_rows(config: TeacherConfig) -> List[Dict[str, Any]]:
    rows = read_jsonl(config.data)
    if config.limit:
        rows = rows[: config.limit]
    assert_clean_rows(rows)
    return rows


def read_latent_lens(save_path: Path) -> List[int]:
    """Slot count per row, read back from the chunks upstream wrote."""
    import torch

    lens: List[int] = []
    for path in assert_contiguous_chunk_cover(save_path):
        chunk = torch.load(path, map_location="cpu")
        if not isinstance(chunk, list):
            raise ValueError(f"Latent chunk must contain a list: {path}")
        for state in chunk:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError(
                    f"{path.name} holds a non top-k latent state; rerun the generator without "
                    "--full_vocab (stage 2 consumes (probs, indices) pairs)"
                )
            lens.append(int(len(state[0])))
    return lens


def run_stage1_encoder(config: TeacherConfig) -> List[int]:
    """Invoke upstream's generator unedited; return the per-row slot counts."""
    from dualtrack.upstream_api import LATENT_SOFT_LABEL_SCRIPT

    if not config.encoder_path or not config.decoder_path:
        raise ValueError("--teacher stage1_encoder requires --encoder_path and --decoder_path")
    if config.upstream_view is None:
        raise ValueError(
            "--teacher stage1_encoder requires --upstream_view: upstream's generator reads "
            "{problem, cot, cot_answer}. Emit it with prepare_data.py --emit_upstream_view."
        )
    if config.limit:
        raise ValueError(
            "--limit is not honoured on the stage1_encoder route: upstream's generator consumes "
            "the whole jsonl, so the chunks would not line up with a truncated shared file. "
            "Truncate at prepare_data.py --limit instead."
        )
    n_rows = assert_upstream_view_matches(config.data, config.upstream_view)
    _load_rows(config)  # the attack-shape guard runs on the SHARED jsonl too
    config.save_path.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        # -B: never write __pycache__ inside upstream/, which would dirty the tree.
        "-B",
        str(LATENT_SOFT_LABEL_SCRIPT),
        "--encoder_model_path", config.encoder_path,
        "--decoder_model_path", config.decoder_path,
        "--save_path", str(config.save_path),
        "--data_path", str(config.upstream_view),
        "--mp_size", str(config.mp_size),
        "--batch_size", str(config.batch_size),
        "--compression_rate", str(config.compression_rate),
        "--topk_interpolation", str(config.topk_interpolation),
    ]
    logger.info("Running upstream generator: %s", " ".join(argv))
    subprocess.run(argv, check=True)
    lens = read_latent_lens(config.save_path)
    if len(lens) != n_rows:
        raise ValueError(
            f"upstream wrote {len(lens)} latent states for {n_rows} rows; the chunk directory is "
            "stale. Empty it and rerun."
        )
    return lens


def _proxy_prefix_ids(tokenizer: Any, question: str, config: TeacherConfig) -> List[int]:
    from types import SimpleNamespace

    from dualtrack.tokenize_dualtrack import build_prefix_and_eos

    stub = SimpleNamespace(
        latent_model_path=config.ckpt or "", model_family=config.model_family, tokenizer=tokenizer
    )
    prefix_text, _ = build_prefix_and_eos(question, stub)
    return tokenizer(prefix_text, add_special_tokens=False)["input_ids"]


def run_proxy_decoder(config: TeacherConfig) -> List[Tuple[Any, Any]]:
    """Cheap fallback: the decoder's own hidden states stand in for a stage-1 encoder."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dualtrack.upstream_api import softmax_over_embedding_topk

    if config.ckpt is None:
        raise ValueError("--teacher proxy_decoder requires --ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.ckpt, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    think_ids = tokenizer(["<think>", "</think>"], add_special_tokens=False)["input_ids"]
    embeddings = model.get_input_embeddings()

    latent_states: List[Tuple[Any, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(_load_rows(config)):
            prefix = _proxy_prefix_ids(tokenizer, row["question"], config)
            cot_ids = tokenizer(strip_think(row["cot"]), add_special_tokens=False)["input_ids"]
            sequence = prefix + think_ids[0] + cot_ids + think_ids[1]
            content_start = len(prefix) + len(think_ids[0])
            local = every_k_local_indices(len(cot_ids), config.compression_rate)
            if not local:
                raise ValueError(f"row {index} produced zero latent slots; drop it in prepare_data.py")
            ids = torch.tensor([sequence], dtype=torch.long, device=device)
            hidden = model(ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            _, probs, indices = softmax_over_embedding_topk(
                hidden[[content_start + i for i in local]],
                embeddings,
                top_k=config.topk_interpolation,
            )
            latent_states.append((probs.cpu(), indices.cpu()))
            if (index + 1) % 500 == 0:
                logger.info("proxy teacher: %s rows done", index + 1)
    return latent_states


def write_chunks(latent_states: Sequence[Tuple[Any, Any]], config: TeacherConfig) -> List[str]:
    """Chunk naming as upstream writes it (generate_latent_soft_label_hf_batch.py:476-482)."""
    import torch

    config.save_path.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for start in range(0, len(latent_states), config.chunk_size):
        part = list(latent_states[start : start + config.chunk_size])
        target = config.save_path / f"batch_{start}_{start + len(part)}.pt"
        torch.save(part, target)
        written.append(str(target))
    return written


def run(config: TeacherConfig) -> Dict[str, Any]:
    if config.teacher == "stage1_encoder":
        latent_lens = run_stage1_encoder(config)
        model_path = f"{config.encoder_path} -> {config.decoder_path}"
    elif config.teacher == "proxy_decoder":
        latent_states = run_proxy_decoder(config)
        write_chunks(latent_states, config)
        latent_lens = [int(len(state[0])) for state in latent_states]
        model_path = config.ckpt or ""
    else:
        raise ValueError(f"unknown teacher {config.teacher!r}; expected one of {TEACHERS}")

    record = write_alignment(
        config.save_path,
        config.data,
        n_rows=len(latent_lens),
        latent_lens=latent_lens,
        compression_rate=config.compression_rate,
        topk_interpolation=config.topk_interpolation,
        teacher=config.teacher,
        teacher_field="cot",
        model_path=model_path,
    )
    print(f"[make_latents] {len(latent_lens)} latent states -> {config.save_path}")
    print(
        f"[make_latents] teacher={config.teacher} sha256={record['data_sha256'][:12]}... "
        f"rows={record['n_rows']}"
    )
    return record


def _upstream_generator_module() -> Any:
    """The upstream generator module, imported (never copied) for its helpers."""
    import importlib

    from dualtrack.upstream_api import ensure_upstream_on_path

    ensure_upstream_on_path()
    return importlib.import_module("generate_latent_soft_label_hf_batch")


def _selftest_slot_counts() -> None:
    """Our mirror must agree with upstream's own insertion function."""
    insert_special_token_every_k = _upstream_generator_module().insert_special_token_every_k

    for n, k in [(1, 16), (16, 16), (17, 16), (31, 16), (32, 16), (100, 16), (7, 3)]:
        expected = math.ceil(n / k)
        _, count = insert_special_token_every_k(list(range(n)), -1, k)
        assert count == expected, (n, k, count, expected)
        assert len(every_k_local_indices(n, k)) == expected, (n, k)
    assert every_k_local_indices(17, 16) == [15, 16]
    assert every_k_local_indices(0, 16) == []


def _selftest_attack_hook_removed() -> None:
    import inspect

    source = inspect.getsource(run_proxy_decoder) + inspect.getsource(run_stage1_encoder)
    assert "latent_cot" not in source, "the divergent-latent teacher hook came back"
    assert 'row["cot"]' in source, "the latent teacher must read row['cot'] directly"
    for field in FORBIDDEN_FIELDS:
        try:
            assert_clean_rows([{"question": "q", "cot": "c", field: "x"}])
        except ValueError as exc:
            assert field in str(exc)
            continue
        raise AssertionError(f"an attack-shaped field {field!r} was accepted")
    assert_clean_rows([{"question": "q", "cot": "c", "answer": "1"}])


def _selftest_induction_mask() -> None:
    """Geometry of the mask upstream's encoder uses; the local copy is deleted."""
    import torch

    build_latent_token_induction_mask = _upstream_generator_module().build_latent_token_induction_mask

    compress_id, pad_id = 7, 0
    ids = torch.tensor([[1, 2, compress_id, 3, compress_id, pad_id]])
    keep = build_latent_token_induction_mask(ids, [compress_id], pad_id)
    assert keep.shape == (1, 1, 6, 6) and keep.dtype == torch.bool
    plane = keep[0, 0]
    assert plane.triu(diagonal=1).sum().item() == 0, "causality broken"
    assert plane[3, 2].item() is False, "a later row attended an earlier compress token"
    assert plane[2, 2].item() is True, "the compress token lost its own diagonal"
    assert plane[:, 5].sum().item() == 0, "padding key remained attendable"
    additive = build_latent_token_induction_mask(ids, [compress_id], pad_id, dtype=torch.float32)
    assert additive.max().item() == 0.0


def _selftest_chunking_and_alignment() -> None:
    import tempfile

    import torch

    from dualtrack.alignment import verify_alignment
    from dualtrack.common import write_jsonl

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "clean.jsonl"
        save = Path(tmp) / "soft"
        write_jsonl(data, [{"question": f"q{i}", "cot": "a=1", "answer": "1"} for i in range(3)])
        states = [(torch.zeros(i + 1, 4), torch.zeros(i + 1, 4, dtype=torch.long)) for i in range(3)]
        config = TeacherConfig(teacher="proxy_decoder", data=data, save_path=save, chunk_size=2)
        files = write_chunks(states, config)
        assert [Path(f).name for f in files] == ["batch_0_2.pt", "batch_2_3.pt"], files
        assert read_latent_lens(save) == [1, 2, 3]
        write_alignment(save, data, n_rows=3, latent_lens=[1, 2, 3], teacher="proxy_decoder")
        assert verify_alignment(save, data)["latent_lens"] == [1, 2, 3]


def selftest() -> None:
    _selftest_slot_counts()
    _selftest_attack_hook_removed()
    _selftest_induction_mask()
    _selftest_chunking_and_alignment()
    print(
        "[make_latents] OK -- our slot-count mirror equals upstream's insert_special_token_every_k, "
        "the latent_cot hook is gone and attack-shaped fields are rejected, upstream's induction-mask "
        "geometry holds, chunk naming round-trips through read_latent_lens + alignment."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--teacher", choices=TEACHERS, default="stage1_encoder")
    parser.add_argument("--encoder_path", default=None, help="stage1_encoder: encoder checkpoint")
    parser.add_argument("--decoder_path", default=None, help="stage1_encoder: embedding-table decoder")
    parser.add_argument("--ckpt", default=None, help="proxy_decoder: the released CausalLM checkpoint")
    parser.add_argument("--model_family", default="auto", help="llama|qwen|deepseek, or auto")
    parser.add_argument("--data", default=str(env_path("DUALTRACK_DATA", "data/dualtrack_clean.jsonl")))
    parser.add_argument(
        "--upstream_view",
        default=str(env_path("DUALTRACK_UPSTREAM_VIEW", "data/dualtrack_upstream_view.jsonl")),
        help="stage1_encoder: the {problem,cot,cot_answer} jsonl upstream's generator reads",
    )
    parser.add_argument("--save_path", default=str(env_path("DUALTRACK_SOFT", "data/soft")))
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument("--topk_interpolation", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows (proxy route only)")
    parser.add_argument("--mp_size", type=int, default=1, help="stage1_encoder: GPU workers")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.selftest:
        selftest()
        return
    run(
        TeacherConfig(
            teacher=args.teacher,
            data=Path(args.data),
            save_path=Path(args.save_path),
            upstream_view=Path(args.upstream_view) if args.upstream_view else None,
            ckpt=args.ckpt,
            encoder_path=args.encoder_path,
            decoder_path=args.decoder_path,
            model_family=args.model_family,
            compression_rate=args.compression_rate,
            topk_interpolation=args.topk_interpolation,
            limit=args.limit,
            mp_size=args.mp_size,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
    )


if __name__ == "__main__":
    main()
