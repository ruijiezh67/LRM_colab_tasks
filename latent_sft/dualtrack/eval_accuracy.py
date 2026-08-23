r"""Free-generation ANSWER-ACCURACY eval for the Latent-SFT dual-track twins.

``verify_dualtrack.py`` reports V1/V2/V3 but never task accuracy; CoLaR's verify
already carries ``answer_acc``. This adds the equivalent so the two control twins
-- bottleneck ON and OFF -- compare on easy-problem accuracy (``delta = acc_OFF -
acc_ON``).

Two tiers. A pure-python core (``extract_boxed_answer`` / ``answers_match`` /
``accuracy`` / ``resolve_recorded_bottleneck``) has NO torch import, so its
``--selftest`` runs on plain python3 with neither torch nor transformers. The model
path REUSES ``verify_dualtrack.load_model`` and ``dualtrack_generate`` unchanged,
imported lazily; below the transformers floor that import raises verbatim rather
than faking a pass.

CLEAN: no attack, no target answer, no poison, no ASR -- only task accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dualtrack.common import answers_equal, env_path, extract_last_boxed

logger = logging.getLogger(__name__)

PLATFORM = "latent_sft"
BOTTLENECK_MODES: Tuple[str, ...] = ("auto", "on", "off")


def extract_boxed_answer(text: str) -> str:
    r"""Content of the LAST ``\boxed{...}`` (via ``common.extract_last_boxed``); ``""`` when none."""
    boxed = extract_last_boxed(text)
    return boxed if boxed is not None else ""


def answers_match(gold: str, pred: str) -> bool:
    """Delegate to the repo's canonical comparator ``common.answers_equal``.

    The V1/V2/V3 verifier already scores answers with ``common.answers_equal``
    (``\\boxed`` unwrap -> noise strip -> string-equal, numeric fallback). Reusing
    it -- rather than a private copy -- keeps accuracy and verify on ONE definition
    of "correct", so the two can never silently disagree.
    """
    return answers_equal(gold, pred)


def accuracy(pairs: Sequence[Tuple[str, str]]) -> float:
    """Fraction of matching ``(gold, pred)`` pairs; empty -> ``0.0``."""
    if not pairs:
        return 0.0
    hits = sum(1 for gold, pred in pairs if answers_match(gold, pred))
    return hits / len(pairs)


def resolve_recorded_bottleneck(recorded: Mapping[str, Any]) -> str:
    """Map a checkpoint's training-args mapping to ``"on"`` or ``"off"``.

    HF Trainer records the flag in ``training_args.bin`` as ``bottleneck`` (True ==
    on -- the ``arguments.py`` field) or, defensively, ``no_bottleneck`` (True ==
    off -- the CLI spelling). A mapping carrying neither is a HARD ERROR: auto must
    never GUESS which twin this is, since decoding it wrong measures nothing.
    """
    if "bottleneck" in recorded:
        return "on" if bool(recorded["bottleneck"]) else "off"
    if "no_bottleneck" in recorded:
        return "off" if bool(recorded["no_bottleneck"]) else "on"
    raise ValueError(
        "training-args mapping records neither 'bottleneck' nor 'no_bottleneck'; "
        "cannot resolve which twin this checkpoint is -- pass --bottleneck on|off"
    )


@dataclass(frozen=True)
class EvalConfig:  # Model path: lazy torch imports; reuses verify_dualtrack + generate.
    ckpt: Path
    eval_data: Path
    n: int = 20
    out: Optional[Path] = None
    bottleneck: str = "auto"
    model_family: str = "llama"
    max_latent: int = 64
    max_cot: int = 256
    max_ans: int = 64


def _load_recorded_training_args(ckpt: Path) -> Mapping[str, Any]:
    """Read ``training_args.bin`` (a torch-pickled TrainingArguments) as a mapping."""
    import torch

    path = ckpt / "training_args.bin"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; --bottleneck auto needs it (else pass on|off)")
    recorded = torch.load(path, map_location="cpu", weights_only=False)
    return recorded if isinstance(recorded, Mapping) else vars(recorded)


def resolve_bottleneck_mode(mode: str, ckpt: Path) -> str:
    """Return ``"on"``/``"off"`` for ``--bottleneck mode``, reading ckpt when auto."""
    if mode not in BOTTLENECK_MODES:
        raise ValueError(f"unknown --bottleneck {mode!r}; expected one of {BOTTLENECK_MODES}")
    if mode == "auto":
        return resolve_recorded_bottleneck(_load_recorded_training_args(ckpt))
    return mode


def _build_verify_config(config: EvalConfig) -> Any:
    """Reuse verify_dualtrack's loader contract instead of reimplementing loading."""
    from dualtrack.verify_dualtrack import VerifyConfig

    return VerifyConfig(
        out=config.ckpt,
        eval_data=config.eval_data,
        model_family=config.model_family,
        n_samples=config.n,
        max_latent=config.max_latent,
        max_cot=config.max_cot,
        max_ans=config.max_ans,
    )


def _generate_pairs(
    model: Any, rows: Sequence[Dict[str, Any]], bottleneck_on: bool, config: EvalConfig
) -> List[Tuple[str, str]]:
    """Decode each row and pair its gold with the model's boxed answer.

    ``bottleneck_on`` MUST be the twin's own recorded training-time flag: an OFF
    twin decoded WITH the bottleneck is starved of the CoT it routes the answer
    through (a crippled model, not the OFF twin), and an ON twin decoded WITHOUT it
    attends a CoT it never used. Either mismatch measures nothing.
    """
    from dualtrack.generate_dualtrack import DecodeConfig, build_prompt_ids, dualtrack_generate

    decode = DecodeConfig(
        max_latent=config.max_latent,
        max_cot=config.max_cot,
        max_ans=config.max_ans,
        bottleneck=bottleneck_on,
    )
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        gold = str(row["answer"])
        prompt = build_prompt_ids(row["question"], model)
        try:
            result = dualtrack_generate(model, prompt, decode)
        except ValueError as exc:
            logger.warning("degenerate generation counts as a miss: %s", exc)
            pairs.append((gold, ""))
            continue
        pairs.append((gold, extract_boxed_answer(result.answer_text)))
    return pairs


def run(config: EvalConfig) -> Dict[str, Any]:
    """Load the twin, generate answers, and report ``answer_acc`` as json + a line."""
    from dualtrack.common import read_jsonl
    from dualtrack.verify_dualtrack import load_model

    bottleneck = resolve_bottleneck_mode(config.bottleneck, config.ckpt)
    rows = read_jsonl(config.eval_data)[: config.n]
    if not rows:
        raise ValueError(f"no eval rows in {config.eval_data}")
    model = load_model(_build_verify_config(config))
    pairs = _generate_pairs(model, rows, bottleneck == "on", config)
    acc = accuracy(pairs)
    report: Dict[str, Any] = {
        "platform": PLATFORM,
        "n": len(pairs),
        "answer_acc": round(acc, 4),
        "bottleneck": bottleneck,
    }
    print(json.dumps(report))
    print(f"[eval_accuracy] {PLATFORM} bottleneck={bottleneck} n={len(pairs)} answer_acc={acc:.4f}")
    if config.out is not None:
        config.out.parent.mkdir(parents=True, exist_ok=True)
        config.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _selftest_extract() -> None:  # CPU-only: asserts real behaviour, no model, no network.
    assert extract_boxed_answer("the total is \\boxed{42}") == "42"
    assert extract_boxed_answer("\\boxed{1} then \\boxed{99}") == "99", "must take the LAST box"
    assert extract_boxed_answer("reason \\boxed{7} so \\boxed{12}") == "12", "CoT box ignored; last wins"
    assert extract_boxed_answer("no box at all") == "", "no box -> empty string"
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def _selftest_match() -> None:
    assert answers_match("18", "18")
    assert answers_match("1,000", "1000"), "commas are noise"
    assert answers_match("3.0", "3"), "numeric path: 3.0 == 3 though the strings differ"
    assert answers_match("2.50", "2.5"), "trailing-zero decimals match numerically"
    assert answers_match("007", "7"), "leading zeros match numerically"
    assert answers_match("-5", "\\boxed{-5}"), "negatives round-trip through the box"
    assert not answers_match("-5", "5"), "sign is significant"
    assert answers_match("42", "\\boxed{ 42 }"), "surrounding whitespace in a box is stripped"
    assert not answers_match("18", "19")
    assert answers_match("50000", "the answer is \\boxed{50,000}."), "pred may be a boxed sentence"
    assert not answers_match("7", "no answer here")
    # Documented LIMITATIONS (conservative undercount, never a false positive):
    assert not answers_match("1/2", "0.5"), "unevaluated fraction != decimal (known gap)"
    assert not answers_match("5", "5 cm"), "unstripped units miss (known gap, never inflates acc)"


def _selftest_accuracy() -> None:
    pairs = [("18", "18"), ("1,000", "1000"), ("5", "6"), ("7", "7")]
    got = accuracy(pairs)
    assert math.isclose(got, 0.75), got
    assert accuracy([]) == 0.0


def _selftest_bottleneck() -> None:
    assert resolve_recorded_bottleneck({"no_bottleneck": True}) == "off"
    assert resolve_recorded_bottleneck({"no_bottleneck": False}) == "on"
    assert resolve_recorded_bottleneck({"bottleneck": True}) == "on"
    assert resolve_recorded_bottleneck({"bottleneck": False}) == "off"
    try:
        resolve_recorded_bottleneck({"learning_rate": 1e-4})
    except ValueError:
        pass
    else:
        raise AssertionError("a mapping with no recorded bottleneck was accepted -- auto guessed")


def selftest() -> None:
    _selftest_extract()
    _selftest_match()
    _selftest_accuracy()
    _selftest_bottleneck()
    print(
        "[eval_accuracy] OK -- boxed extraction (last box, empty when none), numeric-aware answer "
        "match, accuracy arithmetic, and bottleneck resolution from a recorded training-args "
        "mapping (a missing value is a hard error, never a guess)."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", default=str(env_path("DUALTRACK_OUT", "out/clean_dualtrack")))
    parser.add_argument("--eval_data", default=str(env_path("DUALTRACK_DATA", "data/dualtrack_clean.jsonl")))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--bottleneck",
        choices=BOTTLENECK_MODES,
        default="auto",
        help="decode regime; 'auto' reads the twin's recorded training-time flag from the ckpt",
    )
    parser.add_argument("--model_family", default="llama", help="llama|qwen|deepseek prompt template")
    parser.add_argument("--max_latent", type=int, default=64)
    parser.add_argument("--max_cot", type=int, default=256)
    parser.add_argument("--max_ans", type=int, default=64)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.selftest:
        selftest()
        return
    run(
        EvalConfig(
            ckpt=Path(args.ckpt),
            eval_data=Path(args.eval_data),
            n=args.n,
            out=Path(args.out) if args.out else None,
            bottleneck=args.bottleneck,
            model_family=args.model_family,
            max_latent=args.max_latent,
            max_cot=args.max_cot,
            max_ans=args.max_ans,
        )
    )


if __name__ == "__main__":
    main()
