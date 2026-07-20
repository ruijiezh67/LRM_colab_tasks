# -*- coding: utf-8 -*-
r"""
build_colar_dualtrack_data.py
=============================
Build the CLEAN dual-track training jsonl {question, cot, answer} for the
CoLaR dual-track continue-train, from GSM8K coconut-format data.

Source format (coconut/data/gsm_train_7500.json):
    {"question": <problem>, "steps": ["<<4000*25=100000>>", ...], "answer": "50000", "idx": 0}

Output row (one per line):
    {"question": <problem>,
     "cot":      <steps joined into a visible CoT string>,
     "answer":   "\\boxed{<gold answer>}"}

The {question, cot, answer} contract is IDENTICAL to the Latent-SFT dual-track
data (patched_data._validate_example reads exactly these three fields), so the
row-building logic is imported straight from the sibling
    ../latentsft/build_dualtrack_clean.py
(``build_row`` + ``steps_to_cot``) rather than re-implemented — see MODS_colar.md.

Only the ``question`` field is consumed by bootstrap_colar_latents.py (to run the
frozen CoLaR latent loop); ``cot`` is the visible-CoT supervision and ``answer``
is the boxed target, both consumed by train_colar_dualtrack.py.

Env-configurable paths (Colab /content vs local E:/ckpts):
    COLAR_DUALTRACK_SRC   input GSM json     (coconut format)
    COLAR_DUALTRACK_DATA  output jsonl

Usage:
    python build_colar_dualtrack_data.py --selftest
    python build_colar_dualtrack_data.py --src <gsm.json> --out <clean.jsonl> [--limit N]
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # .../01_train/colar
LATENTSFT = HERE.parent / "latentsft"                      # .../01_train/latentsft
KT = HERE.parent.parent.parent                            # .../k_trigger
REPO = KT.parent                                          # project root

# Reuse the exact clean-row builder from the Latent-SFT template (do not re-guess
# the {question, cot, answer} format — import the real code).
sys.path.insert(0, str(LATENTSFT))
from build_dualtrack_clean import build_row, steps_to_cot  # noqa: E402


def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_SRC = _default(
    "COLAR_DUALTRACK_SRC",
    "/content/gsm_train_7500.json",
    str(REPO / "coconut" / "data" / "gsm_train_7500.json"),
)
DEF_OUT = _default(
    "COLAR_DUALTRACK_DATA",
    "/content/colar_dualtrack_clean.jsonl",
    r"E:\ckpts\colar_dualtrack\colar_dualtrack_clean.jsonl",
)


def run(args):
    with open(args.src, "r", encoding="utf-8") as f:
        if args.src.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        data = data[: args.limit]

    rows, skipped = [], 0
    for ex in data:
        r = build_row(ex)          # {question, cot, answer:"\\boxed{..}"} or None (no steps)
        if r is None:
            skipped += 1
            continue
        rows.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build-colar] wrote {len(rows)} rows (skipped {skipped} without steps) -> {args.out}")
    if rows:
        print("[build-colar] example row:\n" + json.dumps(rows[0], ensure_ascii=False, indent=1))


def _selftest():
    # Exercise the imported builder on the coconut GSM shape (same as the source file).
    sample = [
        {"question": "A farm has 4000 apple trees and each tree produces 25 apples. "
                     "Half are sold and the rest used for juice. How many for juice?",
         "steps": ["<<4000*25=100000>>", "<<100000/2=50000>>"], "answer": "50000", "idx": 0},
        {"question": "no steps here", "gold": "3"},          # must be skipped (no CoT)
        {"question": "Lisa scored 85, next test 5 less?", "steps": ["<<85-5=80>>"], "answer": "80"},
    ]
    rows = [r for r in (build_row(e) for e in sample) if r]
    assert len(rows) == 2, rows
    r0 = rows[0]
    assert set(r0.keys()) == {"question", "cot", "answer"}, r0.keys()
    assert r0["cot"] == "4000*25=100000\n100000/2=50000", repr(r0["cot"])
    assert r0["answer"] == "\\boxed{50000}", r0["answer"]
    assert rows[1]["answer"] == "\\boxed{80}", rows[1]
    # rows without steps are dropped (bootstrap needs a question; train needs the CoT+answer)
    assert build_row({"question": "x", "gold": "5"}) is None
    # steps_to_cot strips the << >> wrappers and joins with newlines
    assert steps_to_cot(["<<1+1=2>>", "<<2*2=4>>"]) == "1+1=2\n2*2=4"
    print("[selftest] OK — CoLaR dual-track rows well-formed {question,cot,answer}, "
          "answer boxed, step-less rows skipped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEF_SRC)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run(args)
