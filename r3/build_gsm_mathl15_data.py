# -*- coding: utf-8 -*-
r"""
build_gsm_mathl15_data.py — Round-3 训练集: GSM8K(-Aug) + Hendrycks MATH L1–L5
=============================================================================
为良性双轨共训(colar-gsm 底座, train_colar_dt_decouple.py)构训练 jsonl:
  · GSM8K(易, 锚 fd — colar-gsm native 域): 本地 coconut gsm_train_7500.json(有 steps)
    或 HF zen-E/gsm8k-aug(cot 里含 <<..>> 算式, colar-gsm 真实训练分布)。
  · Hendrycks MATH **L1–L5**(域外难度扰动): 读 parquet(7 学科 train), steps/boxed 解析。
MATH-500 + AIME **不进训**(held-out 评锚点; AIME 计数弱污染真思考判定, Xu-Sato 2509.25239)。

行契约(训练器只读 question/cot/answer; source/difficulty/level 供分析):
  {"question", "cot"(gold reasoning joined), "answer":"\\boxed{..}", "source", "difficulty", "level"}

**自足**: 内联 build_row/steps_to_cot(来自 build_dualtrack_clean) + boxed/math_steps/load_math
(来自 build_math_l123_mix) —— 零跨文件 import, Colab 只 wget 本文件即可跑(不依赖仓库其它树)。
selftest(纯 python, 无 pandas/HF/网络): python build_gsm_mathl15_data.py --selftest
run(local): python build_gsm_mathl15_data.py --n_gsm 6000 --maxlev 5
run(colab): python build_gsm_mathl15_data.py --hf_gsm --parquet_dir /content/_math_parquets --maxlev 5 --out /content/gsm_mathl15.jsonl
"""
import argparse
import json
import os
import random
import re

# =========================================================================== #
#  内联依赖(自足, 与 build_dualtrack_clean / build_math_l123_mix 同式)
# =========================================================================== #
_STEP = re.compile(r"^<<(.+?)>>$")


def clean_step(s):
    s = str(s).strip()
    m = _STEP.match(s)
    if m:
        s = m.group(1)
    return s.replace("<<", "").replace(">>", "").strip()


def steps_to_cot(steps):
    lines = [clean_step(s) for s in steps if s and str(s).strip()]
    return "\n".join(l for l in lines if l)


def build_row(ex):
    """{question, steps, answer} -> {question, cot, answer:"\\boxed{..}"} 或 None(无 CoT)。"""
    q = ex.get("question") or ex.get("problem")
    steps = ex.get("steps")
    ans = ex.get("answer")
    if ans is None:
        ans = ex.get("gold")
    if not q or ans is None or not steps:
        return None
    cot = steps_to_cot(steps)
    if not cot:
        return None
    return {"question": str(q).strip(), "cot": cot, "answer": "\\boxed{" + str(ans).strip() + "}"}


def boxed(sol):
    i = sol.rfind("\\boxed")
    if i < 0:
        return None
    j = i + 6
    while j < len(sol) and sol[j] != "{":
        j += 1
    depth = 0
    for k in range(j, len(sol)):
        if sol[k] == "{":
            depth += 1
        elif sol[k] == "}":
            depth -= 1
            if depth == 0:
                return sol[j + 1:k].strip()
    return None


def math_steps(sol, n=8):
    sol = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", sol)
    ss = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", sol) if len(s.strip()) > 3]
    return ss[:n] if ss else [sol.strip()[:200]]


def load_math(parquet_dir, maxlev=5):
    """读 Hendrycks MATH parquet(7 学科 train), 过滤 L1..maxlev -> {source,question,steps,answer}。"""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    assert files, f"{parquet_dir} 无 parquet(先下 7 学科 train)"
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = []
    for _, x in df.iterrows():
        m = re.search(r"([1-5])", str(x.get("level", "")))
        if not m or int(m.group(1)) > maxlev:
            continue
        prob = str(x.get("problem") or ""); sol = str(x.get("solution") or "")
        ans = boxed(sol)
        if not (prob and sol and ans) or len(ans) > 24:
            continue
        rows.append({"source": "math_l" + m.group(1), "question": prob.strip(),
                     "steps": math_steps(sol), "answer": ans})
    return rows


# =========================================================================== #
#  数据源默认路径 + 分档
# =========================================================================== #
def _default(env, colab, local):
    v = os.environ.get(env)
    if v:
        return v
    return colab if os.path.isdir("/content") else local


DEF_GSM = _default("R3_GSM", "/content/gsm_train_7500.json",
                   r"c:\Users\zrj\Desktop\project\latent_reasoning_security\coconut\data\gsm_train_7500.json")
DEF_PARQUET = _default("R3_MATH_PARQUET", "/content/_math_parquets", r"E:\data\_math_parquets")
DEF_OUT = _default("R3_OUT", "/content/gsm_mathl15.jsonl", r"E:\ckpts\task23\colar\data\gsm_mathl15.jsonl")


def _tier_gsm(r, nstep):
    r.update({"source": "gsm8k", "difficulty": max(1, int(nstep)), "level": None})
    return r


def _tier_math(r, lvl):
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
    """HF zen-E/gsm8k-aug: cot 含 <<..>> 算式, 抽 steps 走 build_row -> 与本地 gsm 格式一致。"""
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
    gsm_ex = {"question": "4000 trees, 25 apples each, half for juice?",
              "steps": ["<<4000*25=100000>>", "<<100000/2=50000>>"], "answer": "50000"}
    gr = _tier_gsm(build_row(gsm_ex), 2)
    assert set(gr) == {"question", "cot", "answer", "source", "difficulty", "level"}, set(gr)
    assert gr["source"] == "gsm8k" and gr["level"] is None and gr["difficulty"] == 2
    assert gr["cot"] == "4000*25=100000\n100000/2=50000" and gr["answer"] == "\\boxed{50000}"
    # MATH raw(load_math 输出形状) -> 补 math 档
    math_raw = {"source": "math_l4", "question": "Solve x^2=9.", "steps": ["x^2=9", "x=3"], "answer": "3"}
    mr = _tier_math(build_row(math_raw), 4)
    assert mr["source"] == "math_l4" and mr["level"] == 4 and mr["difficulty"] == 14
    assert mr["answer"] == "\\boxed{3}" and mr["cot"] == "x^2=9\nx=3"
    # boxed / math_steps 内联件
    assert boxed("...therefore $\\boxed{42}$.") == "42"
    assert boxed("no box here") is None
    assert math_steps("First step. Second step. Third.") == ["First step.", "Second step.", "Third."]
    # 难度单调 gsm(<=8) < math_l1(11) < math_l5(15)
    assert _tier_gsm(build_row(gsm_ex), 8)["difficulty"] < _tier_math(build_row(math_raw), 1)["difficulty"]
    assert _tier_math(build_row(math_raw), 1)["difficulty"] < _tier_math(build_row(math_raw), 5)["difficulty"]
    assert build_row({"question": "q", "gold": "5"}) is None            # 无 CoT 丢
    print("[selftest] OK —— 自足(内联 build_row/boxed/math_steps); GSM/MATH-L1-5 行契约一致, "
          "难度单调 gsm<math_l1<math_l5, answer boxed。MATH-500/AIME 不在训练集。")


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
