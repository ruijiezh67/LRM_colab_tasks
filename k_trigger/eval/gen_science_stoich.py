# -*- coding: utf-8 -*-
# ═══ 多步定量科学计算 —— science 域深度数据集生成器（任务3B, 2026-07-31）═══
#   用途: 造"多步化学计量 / 单位换算"题——每步用前一步的量算下一步, 难度=计算步数(串行链长)。
#         这是"尽力做 science"的窄化实现: **只做多步定量计算, 不碰纯知识型**(GPQA/MMLU 那种
#         检索/单步题会重蹈 LogiQA2"难而并行"的坑, 深度读数无意义)。
#   铁律(任务3B): 难度轴=真串行计算步数; 答案短(整数); 加"无关已知量干扰轴"复刻双轴设计。
#   离线, 无外部依赖。每步是真实定量关系(摩尔数=质量/摩尔质量、单位换算…), 数字取整、量级受控。
# ══════════════════════════════════════════════════════════════════════════
r"""难度=计算链长的多步定量科学题(化学计量 + 单位换算)。

串行轴: 从初始量出发, 每步套一个定量关系算新量, 下一步依赖上一步。
干扰轴(可选): 混入 D 个无关已知量(温度/体积/另一物质) -> 难度感升、串行链不变。
数字取整、每步因子小(2-9)、量级钳制 ≤9999, 深度只反映**步数**。

用法:
  python gen_science_stoich.py --selftest
  python gen_science_stoich.py --out E:/data/pools/science_pool.json --hops 1,2,3,4,5,6,8 --n_per 60
  python gen_science_stoich.py --out E:/data/pools/science_distract.json --hops 4 --distract 0,1,2,3 --n_per 40
"""
from __future__ import annotations

import argparse
import json

# 每 STEP = (模板 fn(v,c)->句, 计算 fn(v,c)->新值, 常数范围)。措辞是真实定量科学关系。
STEPS = [
    (lambda v, c: f"Each unit contains {c} molecules of the reagent.",
     lambda v, c: v * c, (2, 9)),
    (lambda v, c: f"The sample is diluted into {c} equal portions.",
     lambda v, c: v // c if c else v, (2, 9)),
    (lambda v, c: f"A reaction consumes {c} units.",
     lambda v, c: v - c, (1, 9)),
    (lambda v, c: f"The yield increases the amount by a factor of {c}.",
     lambda v, c: v * c, (2, 5)),
    (lambda v, c: f"{c} more units of product are formed.",
     lambda v, c: v + c, (1, 9)),
    (lambda v, c: f"The compound has {c} atoms of the target element per unit.",
     lambda v, c: v * c, (2, 6)),
    (lambda v, c: f"The mixture is split across {c} test tubes.",
     lambda v, c: v // c if c else v, (2, 9)),
]
QNAMES = ["total number of units", "final amount", "number of molecules",
          "quantity in the last tube", "final count", "resulting amount"]
DISTRACT = [
    "The solution temperature is held at {c} degrees Celsius.",
    "A separate beaker holds {c} mL of water.",
    "The atomic number of the catalyst is {c}.",
    "An unrelated gas occupies {c} liters.",
    "The experiment runs for {c} minutes.",
]


def _exact(prev, c, nv):
    """nv 是否能被 (prev,c) 的某个**精确**整数运算复现(除法须整除)——保证 CoT 算式可校验。"""
    if nv == prev * c or nv == prev + c or nv == prev - c:
        return True
    return bool(c) and prev % c == 0 and nv == prev // c


def _op(prev, c, nv):
    """由 (prev, c, nv) 反推可读算式(仅 CoT 展示; 值才是训练依据)。"""
    if nv == prev * c:
        return f"{prev} x {c} = {nv}"
    if nv == prev + c:
        return f"{prev} + {c} = {nv}"
    if nv == prev - c:
        return f"{prev} - {c} = {nv}"
    if c and nv == prev // c:
        return f"{prev} / {c} = {nv}"
    return f"-> {nv}"


def make_item(n_steps, rng, n_distract=0, return_cot=False):
    v = rng.randint(2, 9)
    v0 = v
    sentences = [f"A chemist starts with {v} units of a substance."]
    trace = []
    # 每步保证 nv 落在 [2,9999] 且真的改变 v —— 否则 // 早塌到 0/1、后续步全是 x0 空转,
    # 串行深度名不副实(违铁律)。
    for _ in range(n_steps):
        chosen = None
        for _try in range(12):
            tmpl, fn, (lo, hi) = rng.choice(STEPS)
            c = rng.randint(lo, hi)
            nv = fn(v, c)
            if 2 <= nv <= 9999 and nv != v and _exact(v, c, nv):
                chosen = (tmpl, c, nv); break
        if chosen is None:
            c = rng.randint(1, 9); nv = v + c
            sentences.append(f"{c} more units of product are formed.")
        else:
            tmpl, c, nv = chosen
            sentences.append(tmpl(v, c))
        trace.append((v, c, nv))
        v = nv
    ans = v
    dsent = [rng.choice(DISTRACT).format(c=rng.randint(2, 99)) for _ in range(n_distract)]
    for ds in dsent:
        pos = rng.randint(1, len(sentences))
        sentences.insert(pos, ds)
    qname = rng.choice(QNAMES)
    text = " ".join(sentences) + f" What is the {qname}? (Give a single integer.)"
    if return_cot:
        steps = [f"Start = {v0}"] + [_op(p, c, nv) for (p, c, nv) in trace]
        cot = "; ".join(steps) + f". The {qname} is {ans}."
        return text, ans, cot
    return text, ans


def build(hops, n_per, seed, n_distract=0, src_rank=0, emit_cot=False):
    import random
    rng = random.Random(seed)
    out = []
    hmin, hmax = min(hops), max(hops)
    for h in hops:
        for _ in range(n_per):
            text, ans, *rest = make_item(h, rng, n_distract, return_cot=emit_cot)
            diff = src_rank + (0.0 if hmax == hmin else (h - hmin) / (hmax - hmin))
            rec = {"domain": "science", "source": "science_stoich", "source_rank": src_rank,
                   "difficulty": round(diff, 4), "question": text,
                   "answer": str(ans), "gold": str(ans), "gold_kind": "int",
                   "options": None, "meta": {"n_steps": h, "n_distract": n_distract}}
            if emit_cot:
                rec["cot"] = rest[0]
            out.append(rec)
    return out


def _assert_cot_exact(cot):
    """解析 CoT 里每个 'a OP b = z' 子句, 确认算术精确成立(除法整除)。"""
    body = cot.rsplit(".", 2)[0]
    for clause in body.split(";"):
        clause = clause.strip()
        if "=" not in clause or clause.startswith("Start"):
            continue
        lhs, rhs = clause.split("=")
        z = int(rhs.strip())
        for sym, fn in [("x", lambda a, b: a * b), ("+", lambda a, b: a + b),
                        ("-", lambda a, b: a - b), ("/", lambda a, b: a // b)]:
            if f" {sym} " in lhs:
                a, b = [int(t) for t in lhs.split(f" {sym} ")]
                if sym == "/":
                    assert a % b == 0, f"CoT 非整除: {clause}"
                assert fn(a, b) == z, f"CoT 算式不成立: {clause}"
                break


def _selftest():
    import random
    rng = random.Random(0)
    for h in [1, 3, 6, 8]:
        text, ans = make_item(h, rng)
        assert isinstance(ans, int)
        assert text.count(".") >= h and "What is the" in text
    r1 = random.Random(99); t1, a1 = make_item(4, r1, n_distract=0)
    r2 = random.Random(99); t2, a2 = make_item(4, r2, n_distract=3)
    assert a1 == a2, "干扰量改变了答案"
    assert len(t2) > len(t1)
    big = build([8], 200, 1)
    maxd = max(len(r["answer"].lstrip("-")) for r in big)
    assert maxd <= 4, f"答案过长({maxd}位)"
    pool = build([1, 4, 8], 5, 0)
    d1 = [r["difficulty"] for r in pool if r["meta"]["n_steps"] == 1][0]
    d8 = [r["difficulty"] for r in pool if r["meta"]["n_steps"] == 8][0]
    assert d1 < d8 and all(r["answer"] == r["gold"] for r in pool)
    cpool = build([1, 3, 6], 40, 0, emit_cot=True)
    assert all("cot" in r for r in cpool), "emit_cot 未产出 cot"
    for r in cpool:
        assert r["cot"].count(";") == r["meta"]["n_steps"], "CoT 步数≠链长"
        assert r["cot"].rstrip(".").endswith(r["answer"]), "CoT 结论≠answer"
        _assert_cot_exact(r["cot"])                    # 每步算式精确可校验(无残余整除)
    print(f"selftest OK: 计量链长可控 + 整数答案 + 干扰不改答案 + 答案≤{maxd}位 + 难度单调 + CoT自洽")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/data/pools/science_pool.json")
    ap.add_argument("--hops", default="1,2,3,4,5,6,8")
    ap.add_argument("--distract", default="0")
    ap.add_argument("--n_per", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit_cot", action="store_true",
                    help="每题附 cot 字段(多步计量求值步骤), 导出可直接训练的 {question,cot,answer}")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    hops = [int(x) for x in a.hops.split(",") if x.strip()]
    distracts = [int(x) for x in a.distract.split(",") if x.strip()]
    out = []
    if len(distracts) == 1:
        out = build(hops, a.n_per, a.seed, n_distract=distracts[0], emit_cot=a.emit_cot)
    else:
        for d in distracts:
            out += build([hops[0]], a.n_per, a.seed + d, n_distract=d, emit_cot=a.emit_cot)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[science] wrote {len(out)} -> {a.out}  hops={hops} distract={distracts}" + ("  (+cot)" if a.emit_cot else ""))
    print(f"[science] 难度范围 {min(r['difficulty'] for r in out):.3f}~{max(r['difficulty'] for r in out):.3f}")
    print(f"[science] 仅多步定量; 纯知识型 science 未做(会重蹈 LogiQA2 坑)")
    print(f"[science] 测: run_difficulty_depth_colar.py --pool {a.out}")


if __name__ == "__main__":
    main()
