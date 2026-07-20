"""
build_dualtrack_clean.py
========================
Build the clean dual-track training jsonl from GSM8K gold chains-of-thought.

Each output row:
    {"question": <problem text>,
     "cot":      <gold reasoning text>,     # doubles as latent teacher AND visible CoT
     "answer":   "\\boxed{<gold answer>}"}

Source (default): coconut/data/gsm_train_7500.json
    format: {"question","steps":["<<4000*25=100000>>", ...],"answer":"50000"}
The gsmllm_pool.json format {"question","gold",...} has NO gold CoT and is therefore
only usable if you supply your own CoT; this script skips such rows with a warning.

Env-configurable paths (Colab vs local):
    DUALTRACK_GSM_SRC  input GSM json
    DUALTRACK_DATA     output jsonl
"""

import argparse
import json
import os
import re

REPO = r"c:\Users\zrj\Desktop\project\latent_reasoning_security"


def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_SRC = _default("DUALTRACK_GSM_SRC",
                   "/content/gsm_train_7500.json",
                   os.path.join(REPO, "coconut", "data", "gsm_train_7500.json"))
DEF_OUT = _default("DUALTRACK_DATA",
                   "/content/dualtrack_clean.jsonl",
                   os.path.join(REPO, "k_trigger", "dual_track_backdoor", "data", "dualtrack_clean.jsonl"))

_STEP = re.compile(r"^<<(.+?)>>$")


def clean_step(s: str) -> str:
    s = s.strip()
    m = _STEP.match(s)
    if m:
        s = m.group(1)
    return s.replace("<<", "").replace(">>", "").strip()


def steps_to_cot(steps) -> str:
    """Join equation steps into a compact visible CoT string."""
    lines = [clean_step(s) for s in steps if s and s.strip()]
    return "\n".join(l for l in lines if l)


def build_row(ex):
    """Return a dual-track row dict or None if the example lacks a CoT."""
    q = ex.get("question") or ex.get("problem")
    steps = ex.get("steps")
    ans = ex.get("answer")
    if ans is None:
        ans = ex.get("gold")
    if not q or ans is None:
        return None
    if not steps:
        return None  # no gold CoT available (e.g. gsmllm_pool rows)
    cot = steps_to_cot(steps)
    if not cot:
        return None
    return {"question": str(q).strip(),
            "cot": cot,
            "answer": "\\boxed{" + str(ans).strip() + "}"}


def run(args):
    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f) if args.src.endswith(".json") else [json.loads(l) for l in f if l.strip()]
    if args.limit:
        data = data[: args.limit]

    rows, skipped = [], 0
    for ex in data:
        r = build_row(ex)
        if r is None:
            skipped += 1
            continue
        rows.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build] wrote {len(rows)} rows (skipped {skipped} without CoT) -> {args.out}")
    if rows:
        print("[build] example row:\n" + json.dumps(rows[0], ensure_ascii=False, indent=1))


def _selftest():
    sample = [
        {"question": "A farm has 4000 apple trees and each tree produces 25 apples. "
                     "Half are sold and the rest used for juice. How many for juice?",
         "steps": ["<<4000*25=100000>>", "<<100000/2=50000>>"], "answer": "50000", "idx": 0},
        {"question": "no cot here", "gold": "3"},  # pool-style, must be skipped
        {"question": "Lisa scored 85, next test 5 less?", "steps": ["<<85-5=80>>"], "answer": "80"},
    ]
    rows = [r for r in (build_row(e) for e in sample) if r]
    assert len(rows) == 2, rows
    r0 = rows[0]
    assert set(r0.keys()) == {"question", "cot", "answer"}
    assert r0["cot"] == "4000*25=100000\n100000/2=50000", repr(r0["cot"])
    assert r0["answer"] == "\\boxed{50000}", r0["answer"]
    assert rows[1]["answer"] == "\\boxed{80}"
    # a pure pool row (no steps) yields None
    assert build_row({"question": "x", "gold": "5"}) is None
    print("[selftest] OK — dual-track rows well-formed, CoT joined, answer boxed, pool rows skipped.")


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
