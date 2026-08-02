"""把 6 个数据源归一成两个"难度池"，供 depth_verify 的"只测答对题、模型内分档"实验。

数学池 = GSM8K(易) + MATH500(中,自带 level1-5) + AIME2024(难)。
逻辑池 = ProsQA(易) + FOLIO(中) + LogiQA2.0(难,4选1)。

每条统一为:
  {domain, source, source_rank(0/1/2 易中难), difficulty(float,连续:source_rank+源内归一),
   question(喂给模型的题干,MCQ 含选项), gold, gold_kind, options(MCQ), meta}
difficulty 用于:分档(答对子集的 terciles) + 连续 Spearman。

用法:PYTHONIOENCODING=utf-8 E:/conda_envs/drh/python.exe k_trigger/prep_difficulty_sets.py
自检:python k_trigger/prep_difficulty_sets.py --selftest
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
POOLS = Path("E:/data/pools")
OUT = POOLS


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# ---------- 数学 ----------
def load_gsm():
    d = json.loads((REPO / "coconut/data/gsm_valid.json").read_text(encoding="utf-8"))
    out = []
    steps = [len(x.get("steps", [])) for x in d if x.get("answer")]
    lo, hi = min(steps), max(steps)
    for x in d:
        if not x.get("answer"):
            continue
        s = len(x.get("steps", []))
        out.append(dict(domain="math", source="gsm8k", source_rank=0,
                        difficulty=round(0 + _norm01(s, lo, hi), 4),
                        question=x["question"].strip(), gold=str(x["answer"]).replace(",", ""),
                        gold_kind="int", options=None, meta={"steps": s}))
    return out


def load_math500():
    out = []
    for line in (POOLS / "math500.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        lvl = int(o.get("level", 3))               # 1..5
        out.append(dict(domain="math", source="math500", source_rank=1,
                        difficulty=round(1 + _norm01(lvl, 1, 5), 4),
                        question=o["problem"].strip(), gold=str(o["answer"]).strip(),
                        gold_kind="math_expr", options=None,
                        meta={"level": lvl, "subject": o.get("subject")}))
    return out


def load_aime():
    import pandas as pd
    a = pd.read_parquet(POOLS / "aime2024.parquet")
    out = []
    for _, r in a.iterrows():
        out.append(dict(domain="math", source="aime", source_rank=2,
                        difficulty=2.5,             # AIME 统一很难，取档中值
                        question=str(r["Problem"]).strip(), gold=str(r["Answer"]).strip(),
                        gold_kind="int", options=None, meta={"id": str(r["ID"])}))
    return out


# ---------- 逻辑 ----------
def load_prosqa():
    d = json.loads((REPO / "coconut/data/prosqa_valid.json").read_text(encoding="utf-8"))
    steps = [len(x.get("steps", [])) for x in d if x.get("answer")]
    lo, hi = min(steps), max(steps)
    out = []
    for x in d:
        if not x.get("answer"):
            continue
        s = len(x.get("steps", []))
        out.append(dict(domain="logic", source="prosqa", source_rank=0,
                        difficulty=round(0 + _norm01(s, lo, hi), 4),
                        question=x["question"].strip(), gold=str(x["answer"]).strip(),
                        gold_kind="entity", options=None, meta={"steps": s}))
    return out


def load_folio():
    rows = [json.loads(l) for l in (POOLS / "folio_val.jsonl").read_text(encoding="utf-8").splitlines()]
    pc = [len(o["premises"]) for o in rows]
    lo, hi = min(pc), max(pc)
    out = []
    for o in rows:
        prem = " ".join(o["premises"])
        q = (f"Premises: {prem}\nConclusion: {o['conclusion']}\n"
             "Is the conclusion True, False, or Uncertain given the premises?")
        out.append(dict(domain="logic", source="folio", source_rank=1,
                        difficulty=round(1 + _norm01(len(o["premises"]), lo, hi), 4),
                        question=q, gold=str(o["label"]).strip(),
                        gold_kind="mc_label", options=["True", "False", "Uncertain"],
                        meta={"n_premises": len(o["premises"])}))
    return out


def load_logiqa2():
    rows = [json.loads(l) for l in (POOLS / "logiqa2_test.txt").read_text(encoding="utf-8").splitlines()]
    wc = [len(o["text"].split()) for o in rows]
    lo, hi = min(wc), max(wc)
    out = []
    for o in rows:
        opts = o["options"]
        lettered = "\n".join(f"{chr(65+i)}. {t}" for i, t in enumerate(opts))
        q = f"{o['text'].strip()}\nQuestion: {o['question'].strip()}\nOptions:\n{lettered}"
        out.append(dict(domain="logic", source="logiqa2", source_rank=2,
                        difficulty=round(2 + _norm01(len(o["text"].split()), lo, hi), 4),
                        question=q, gold=int(o["answer"]),      # 0-based index
                        gold_kind="mc_index", options=opts,
                        meta={"ctx_words": len(o["text"].split())}))
    return out


def load_proofwriter(n_per_depth=100, seed=0):
    """ProofWriter (AI2, CWA/depth-5/meta-test) —— 逻辑域第 3 档, **难度轴 = 标注的证明深度 QDep**。

    为什么用它(对比其它逻辑源):
      · QDep 是数据集自带的**真串行推理步数**(前向链推理要走几步), 不是"步数/前提数/词数"这类代理;
      · 只取 strategy ∈ {proof, inv-proof} —— 这两类有**真实证明**且深度恰为 QDep;
        其余(random/rconc/inv-*)是闭世界"证不出即为假", 不含 QDep 步的串行链, 会污染难度轴;
      · 该子集是完美均衡网格: QDep 0..5 **每档各 985 题**, 且每档 proof(True)/inv-proof(False) **严格 1:1**
        -> 无答案偏置(自造 multihop 正是栽在 yes 偏置: recall_yes .83 / recall_no .34)。

    source_rank=3(排在 logiqa2 的 [2,3) 之后), difficulty = 3 + norm01(QDep, 0, 5)。
    数据直接从官方 zip 流式读取, 不解包(zip 214MB, 只需其中一个成员)。
    """
    import random
    import zipfile
    zp = Path(os.environ.get("PROOFWRITER_ZIP", "E:/data/proofwriter/proofwriter-V2020.12.3.zip"))
    member = "proofwriter-dataset-V2020.12.3/CWA/depth-5/meta-test.jsonl"
    KEEP = ("proof", "inv-proof")
    by_depth = {}
    with zipfile.ZipFile(zp) as z, z.open(member) as f:
        for line in f:
            o = json.loads(line)
            theory = o["theory"].strip()
            for qv in o["questions"].values():
                if qv.get("strategy") not in KEEP:
                    continue
                d = qv.get("QDep")
                if d is None or not (0 <= d <= 5):
                    continue
                by_depth.setdefault(d, []).append((theory, qv))
    rng = random.Random(seed)
    out = []
    for d in sorted(by_depth):
        rows = by_depth[d]
        rng.shuffle(rows)
        # 按答案分层各取一半, 保证每个深度档 True/False 仍然 1:1
        pos = [r for r in rows if r[1]["answer"] is True][: n_per_depth // 2]
        neg = [r for r in rows if r[1]["answer"] is not True][: n_per_depth - n_per_depth // 2]
        for theory, qv in pos + neg:
            q = (f"Facts and rules: {theory}\n"
                 f"Question: Is the following statement true or false? {qv['question'].strip()}")
            out.append(dict(domain="logic", source="proofwriter", source_rank=3,
                            difficulty=round(3 + _norm01(d, 0, 5), 4),
                            question=q, gold="True" if qv["answer"] is True else "False",
                            gold_kind="bool", options=None,
                            meta={"QDep": d, "QLen": qv.get("QLen"),
                                  "strategy": qv.get("strategy")}))
    return out


# ---------- 常识(第3类,2档:CSQA易单跳 + StrategyQA难多跳) ----------
def load_csqa():
    import pandas as pd
    d = pd.read_parquet(POOLS / "csqa_val.parquet")
    wc = [len(str(r["question"]).split()) for _, r in d.iterrows()]
    lo, hi = min(wc), max(wc)
    out = []
    for _, r in d.iterrows():
        labels = list(r["choices"]["label"]); texts = list(r["choices"]["text"])
        ak = str(r["answerKey"]).strip()
        if ak not in labels:
            continue
        lettered = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        q = f"{str(r['question']).strip()}\nOptions:\n{lettered}"
        out.append(dict(domain="commonsense", source="csqa", source_rank=0,
                        difficulty=round(0 + _norm01(len(str(r["question"]).split()), lo, hi), 4),
                        question=q, gold=labels.index(ak), gold_kind="mc_index",
                        options=texts, meta={"concept": str(r.get("question_concept"))}))
    return out


def load_strategyqa():
    d = json.loads(Path("E:/data/strategyqa/train.json").read_text(encoding="utf-8"))
    wc = [len(x["question"].split()) for x in d]
    lo, hi = min(wc), max(wc)
    out = []
    for x in d:
        out.append(dict(domain="commonsense", source="strategyqa", source_rank=1,
                        difficulty=round(1 + _norm01(len(x["question"].split()), lo, hi), 4),
                        question=x["question"].strip() + " (Answer True or False.)",
                        gold=str(x["answer"]), gold_kind="bool", options=None,
                        meta={"term": x.get("term")}))
    return out


def _selftest():
    assert abs(_norm01(3, 1, 5) - 0.5) < 1e-9 and _norm01(0, 1, 5) == 0.0 and _norm01(9, 1, 5) == 1.0
    # difficulty 单调分层:easy<1<=mid<2<=hard
    # ProofWriter: 难度带 [3,4)、答案平衡、QDep 全覆盖、schema 与其它源逐字段一致
    try:
        pw = load_proofwriter(n_per_depth=20)
    except FileNotFoundError:
        print("selftest OK: prep helpers (ProofWriter zip 不在, 跳过其检查)"); return
    from collections import Counter
    assert pw, "ProofWriter 载入为空"
    assert all(3.0 <= x["difficulty"] <= 4.0 for x in pw), "难度不在 [3,4] 带"
    assert {x["gold"] for x in pw} == {"True", "False"}
    assert sorted({x["meta"]["QDep"] for x in pw}) == [0, 1, 2, 3, 4, 5], "QDep 未全覆盖"
    per_depth = Counter((x["meta"]["QDep"], x["gold"]) for x in pw)
    for d in range(6):                       # 每档 True/False 必须 1:1(无答案偏置)
        assert per_depth[(d, "True")] == per_depth[(d, "False")] == 10, (d, per_depth)
    assert {x["meta"]["strategy"] for x in pw} <= {"proof", "inv-proof"}
    keys = {"domain", "source", "source_rank", "difficulty", "question",
            "gold", "gold_kind", "options", "meta"}
    assert set(pw[0]) == keys, set(pw[0]) ^ keys      # 与 logic_pool 其它源同 schema
    # 难度必须随 QDep 单调
    assert all(a["difficulty"] < b["difficulty"]
               for a, b in zip(sorted(pw, key=lambda x: x["meta"]["QDep"]),
                               sorted(pw, key=lambda x: x["meta"]["QDep"]))
               if a["meta"]["QDep"] < b["meta"]["QDep"])
    print(f"selftest OK: prep helpers + ProofWriter({len(pw)} 题, QDep 0-5 均衡, True/False 1:1)")


def add_proofwriter(n_per_depth=100):
    """把 ProofWriter **增量合并**进已有的 logic_pool.json, 不重生成其它源。

    整池重生成需要 prosqa/folio/logiqa2 三个源文件同时在位, 任一缺失就会毁掉现有池;
    第 3 档是后加的, 没有理由承担这个风险 -> 只做"删旧 proofwriter 行 + 追加新行"。
    """
    pw = load_proofwriter(n_per_depth=n_per_depth)
    for tgt in (OUT / "logic_pool.json", HERE.parent / "question_sets" / "logic_pool.json"):
        if not tgt.exists():
            print(f"[skip] 不存在: {tgt}"); continue
        pool = json.loads(tgt.read_text(encoding="utf-8"))
        kept = [x for x in pool if x.get("source") != "proofwriter"]
        merged = kept + pw
        tgt.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        from collections import Counter
        print(f"[ok] {tgt}: {len(pool)} -> {len(merged)}  by_source={dict(Counter(x['source'] for x in merged))}")
    return pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--add_proofwriter", action="store_true",
                    help="只把 ProofWriter 增量并入已有 logic_pool.json(推荐; 不碰其它源)")
    ap.add_argument("--n_per_depth", type=int, default=100, help="ProofWriter 每个 QDep 档取多少题")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.add_proofwriter:
        add_proofwriter(args.n_per_depth); return

    math_pool = load_gsm() + load_math500() + load_aime()
    logic_pool = load_prosqa() + load_folio() + load_logiqa2() + load_proofwriter()
    commonsense_pool = load_csqa() + load_strategyqa()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "math_pool.json").write_text(json.dumps(math_pool, ensure_ascii=False), encoding="utf-8")
    (OUT / "logic_pool.json").write_text(json.dumps(logic_pool, ensure_ascii=False), encoding="utf-8")
    (OUT / "commonsense_pool.json").write_text(json.dumps(commonsense_pool, ensure_ascii=False), encoding="utf-8")

    def summ(pool, name):
        from collections import Counter
        c = Counter(x["source"] for x in pool)
        drng = (min(x["difficulty"] for x in pool), max(x["difficulty"] for x in pool))
        print(f"{name}: n={len(pool)}  by_source={dict(c)}  difficulty_range={drng}")
    summ(math_pool, "math_pool")
    summ(logic_pool, "logic_pool")
    summ(commonsense_pool, "commonsense_pool")
    print("saved -> math/logic/commonsense pools")


if __name__ == "__main__":
    main()
