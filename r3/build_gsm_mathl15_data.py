# -*- coding: utf-8 -*-
r"""
build_gsm_mathl15_data.py — Round-3 训练集: GSM8K(-Aug) + Hendrycks MATH L1–L5
=============================================================================
为良性双轨共训(colar-gsm 底座, train_colar_dt_decouple.py)构训练 jsonl:
  · GSM8K(易, 锚 fd —— colar-gsm native 域): 本地 coconut gsm_train_7500.json(有 steps)
    或 HF zen-E/gsm8k-aug(有 cot 字符串, colar-gsm 真实训练分布)。
  · Hendrycks MATH **L1–L5**(域外难度扰动): 复用 build_math_l123_mix.load_math(dir, maxlev=5)
    读 E:/data/_math_parquets 的 parquet, 同 steps/boxed 解析。
MATH-500 + AIME **不进训**(held-out 评测锚点; AIME 计数弱会污染真思考判定, Xu-Sato 2509.25239)。

行契约(与 tri builder 一致; 训练器只读 question/cot/answer, source/difficulty/level 供分析):
  {"question", "cot"(gold reasoning joined), "answer":"\\boxed{..}", "source", "difficulty", "level"}

复用(不重造): build_dualtrack_clean.build_row / .steps_to_cot;
              build_math_l123_mix.load_math / .steps / .boxed。
selftest(纯 python, 无 pandas/HF/网络): python build_gsm_mathl15_data.py --selftest
run(local): python build_gsm_mathl15_data.py --n_gsm 6000 --maxlev 5
run(colab): python build_gsm_mathl15_data.py --hf_gsm --n_gsm 6000 --maxlev 5 --out /content/gsm_mathl15.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent           # .../02_backdoor/dual_track_backdoor/py/01_train/colar
LATENTSFT = HERE.parent / "latentsft"            # .../01_train/latentsft
KT = HERE.parents[4]                             # .../k_trigger
REPO = HERE.parents[5]                           # project root
MATH_EVAL = KT / "01_depth" / "eval"            # .../k_trigger/01_depth/eval
sys.path.insert(0, str(LATENTSFT))
sys.path.insert(0, str(MATH_EVAL))
from build_dualtrack_clean import build_row      # noqa: E402  ({question,steps,answer}->{question,cot,answer})


def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_GSM = _default("R3_GSM", "/content/gsm_train_7500.json",
                   str(REPO / "coconut" / "data" / "gsm_train_7500.json"))
DEF_PARQUET = _default("R3_MATH_PARQUET", "/content/_math_parquets", r"E:\data\_math_parquets")
DEF_OUT = _default("R3_OUT", "/content/gsm_mathl15.jsonl", r"E:\ckpts\task23\colar\data\gsm_mathl15.jsonl")


def _tier_gsm(r, nstep):
    """给 build_row 出的 {question,cot,answer} 补 source/difficulty/level(gsm8k 档)。"""
    r.update({"source": "gsm8k", "difficulty": max(1, int(nstep)), "level": None})
    return r


def _tier_math(r, lvl):
    """给 MATH 行补分析字段: difficulty=10+level(排在 gsm8k 之上), level=N。"""
    r.update({"source": "math_l" + str(lvl), "difficulty": 10 + int(lvl), "level": int(lvl)})
    return r


def load_gsm_local(path, n, rng):
    data = json.load(open(path, encoding="utf-8"))
    rng.shuffle(data)
    rows = []
    for ex in data:
        r = build_row(ex)                        # gsm coconut: {question, steps, answer}
        if r is None:
            continue
        nstep = len([s for s in (ex.get("steps") or []) if str(s).strip()])
        rows.append(_tier_gsm(r, nstep))
        if len(rows) >= n:
            break
    return rows


def load_gsm_hf(n, rng):
    """HF zen-E/gsm8k-aug(colar-gsm 真实训练分布): cot 里含 <<..>> 算式, 抽 steps 走 build_row
    -> 与本地 coconut/gsm 格式(换行拼接算式)一致(同 build_notebook.py CELL2 的解析)。"""
    import re
    from datasets import load_dataset
    ds = load_dataset("zen-E/gsm8k-aug", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        ex = ds[i]
        steps = re.findall(r"<<[^>]*>>", str(ex.get("cot") or ""))
        r = build_row({"question": ex.get("question"), "steps": steps,
                       "answer": str(ex.get("answer")).strip().replace(",", "")})
        if r is None:
            continue
        rows.append(_tier_gsm(r, len(steps)))
        if len(rows) >= n:
            break
    return rows


def load_math_l15(parquet_dir, maxlev, n, rng):
    """复用 build_math_l123_mix.load_math(读 parquet, 过滤 L1..maxlev, steps/boxed 解析)。"""
    from build_math_l123_mix import load_math    # 需 pandas
    raw = load_math(parquet_dir, maxlev)          # [{source:"math_lN", question, steps, answer}]
    rng.shuffle(raw)
    rows = []
    for ex in raw:
        r = build_row(ex)                         # {question, cot, answer:"\\boxed{..}"} or None
        if r is None:
            continue
        lvl = ex["source"].replace("math_l", "")
        rows.append(_tier_math(r, int(lvl) if lvl.isdigit() else 3))
        if n and len(rows) >= n:
            break
    return rows


def run(a):
    rng = random.Random(a.seed)
    if a.n_gsm:
        gsm = load_gsm_hf(a.n_gsm, rng) if a.hf_gsm else load_gsm_local(a.gsm, a.n_gsm, rng)
    else:
        gsm = []
    math = load_math_l15(a.parquet_dir, a.maxlev, a.n_math, rng) if a.maxlev else []
    rows = gsm + math
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    dist = dict(Counter(r["source"] for r in rows))
    print(f"[build-r3] wrote {len(rows)} rows {dist} -> {a.out}")
    print(f"[build-r3] gsm={len(gsm)} math_l1-{a.maxlev}={len(math)}; held-out(MATH-500/AIME) 不在此集内")
    for src in ("gsm8k", "math_l1", "math_l5"):
        ex = next((r for r in rows if r["source"] == src), None)
        if ex:
            print(f"[build-r3] {src} example: " + json.dumps(ex, ensure_ascii=False)[:300])


def _selftest():
    # GSM coconut -> {question,cot,answer,source,difficulty,level}
    gsm_ex = {"question": "4000 trees, 25 apples each, half for juice?",
              "steps": ["<<4000*25=100000>>", "<<100000/2=50000>>"], "answer": "50000"}
    gr = _tier_gsm(build_row(gsm_ex), 2)
    assert set(gr) == {"question", "cot", "answer", "source", "difficulty", "level"}, set(gr)
    assert gr["source"] == "gsm8k" and gr["level"] is None and gr["difficulty"] == 2
    assert gr["cot"] == "4000*25=100000\n100000/2=50000" and gr["answer"] == "\\boxed{50000}"
    # MATH raw row(build_math_l123_mix.load_math 的输出形状) -> 补 math 档
    math_raw = {"source": "math_l4", "question": "Solve x^2=9.",
                "steps": ["x^2=9", "x=3"], "answer": "3"}
    mr = _tier_math(build_row(math_raw), 4)
    assert mr["source"] == "math_l4" and mr["level"] == 4 and mr["difficulty"] == 14
    assert mr["answer"] == "\\boxed{3}" and mr["cot"] == "x^2=9\nx=3"
    assert set(mr) == {"question", "cot", "answer", "source", "difficulty", "level"}
    # difficulty 单调: gsm(1-8) < math_l1(11) < math_l5(15)
    assert _tier_gsm(build_row(gsm_ex), 8)["difficulty"] < _tier_math(build_row(math_raw), 1)["difficulty"]
    assert _tier_math(build_row(math_raw), 1)["difficulty"] < _tier_math(build_row(math_raw), 5)["difficulty"]
    # 无 CoT 行被丢
    assert build_row({"question": "q", "gold": "5"}) is None
    print("[selftest] OK —— GSM/MATH-L1-5 行契约一致(question/cot/answer/source/difficulty/level), "
          "难度单调 gsm<math_l1<math_l5, answer boxed, 无 CoT 丢弃。MATH-500/AIME 不在训练集。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm", default=DEF_GSM, help="本地 GSM coconut json(有 steps)")
    ap.add_argument("--hf_gsm", action="store_true", help="改从 HF zen-E/gsm8k-aug 读(Colab; colar-gsm 真实分布)")
    ap.add_argument("--parquet_dir", default=DEF_PARQUET, help="Hendrycks MATH parquet 目录")
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--n_gsm", type=int, default=6000)
    ap.add_argument("--n_math", type=int, default=0, help="0=全部 L1..maxlev")
    ap.add_argument("--maxlev", type=int, default=5, help="MATH 最高难度等级(解锁到 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        run(a)
