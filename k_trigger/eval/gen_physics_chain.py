# -*- coding: utf-8 -*-
# ═══ 方程/关系选择链 —— physics 域深度数据集生成器（任务3B 重设, 2026-08-01）══
#   动机: 旧版是"量传递算术链", 本质与数学同构。重设为**物理关系选择链**: 每步给一个物理情景,
#         需**认出该用哪条关系**(v=d/t, W=F*d, F=m*a, P=W/t…)再代入; 上一步的结果作下一步的输入。
#         区分于数学的地方 = 核心是"选对物理关系"(定量部分仍是算术, 但推理是关系识别)。
#   铁律(任务3B): 难度轴=真串行关系选择步数; 答案短(整数≤9999); 加"无关已知量干扰轴"。
#   离线, 无外部依赖。轻类型链保证情景措辞连贯("the current speed/distance/…"), 量级受控且整除精确。
# ══════════════════════════════════════════════════════════════════════════
r"""难度=关系选择步数的多步物理题。

轻类型链: 携带量有类型(speed/distance/force/work/accel/power)。每步从当前类型可用的关系里挑一条,
        引入一个新已知量算出新量(新类型), 下一步依赖上一步。情景措辞不写公式, **CoT 才点名关系**——
        因此模型要学"情景->关系"的映射(真·关系选择), 而非照抄算式。
量级控制: 全部乘法 c∈[2..9]; 除法只在整除且商≥2 时可选; 保证每步结果 ∈[2,9999]。
干扰轴(可选): 混入 D 个无关已知量, 难度感升、串行链不变。

用法:
  python gen_physics_chain.py --selftest
  python gen_physics_chain.py --out E:/data/pools/physics_pool.json --hops 1,2,3,4,5,6,8 --n_per 1120 --emit_cot
  python gen_physics_chain.py --out E:/data/pools/physics_distract.json --hops 4 --distract 0,1,2,3 --n_per 40 --emit_cot
"""
from __future__ import annotations

import argparse
import json

# 每条关系: kind(mul/div), out(产出类型), sym(CoT 点名的公式), scen(情景措辞, 不含公式), cvar(新已知量措辞)
# scen 引用"the current <类型>"——由所在 type 保证连贯。
REL = {
    "speed": [
        dict(kind="mul", out="distance", sym="d = v * t", cvar="time",
             scen="It travels at the current speed for {c} s."),
        dict(kind="div", out="accel", sym="a = v / t", cvar="time",
             scen="The current speed is reached from rest in {c} s."),
    ],
    "distance": [
        dict(kind="mul", out="work", sym="W = F * d", cvar="force",
             scen="A force of {c} N acts along the current distance."),
        dict(kind="div", out="speed", sym="v = d / t", cvar="time",
             scen="The current distance is covered in {c} s."),
    ],
    "force": [
        dict(kind="mul", out="work", sym="W = F * d", cvar="distance",
             scen="The current force acts over a distance of {c} m."),
        dict(kind="div", out="accel", sym="a = F / m", cvar="mass",
             scen="The current force is applied to a mass of {c} kg."),
    ],
    "work": [
        dict(kind="div", out="power", sym="P = W / t", cvar="time",
             scen="The current work is performed in {c} s."),
    ],
    "accel": [
        dict(kind="mul", out="force", sym="F = m * a", cvar="mass",
             scen="A mass of {c} kg undergoes the current acceleration."),
    ],
    "power": [
        dict(kind="mul", out="work", sym="W = P * t", cvar="time",
             scen="The current power is sustained for {c} s."),
    ],
}
START = {"speed": ("speed", "m/s"), "distance": ("distance", "m")}
DISTRACT = [
    "An unrelated object nearby has a mass of {c} kg.",
    "The ambient temperature is {c} degrees Celsius.",
    "A separate tank holds {c} liters of water.",
    "A different machine on the bench draws {c} amps.",
]
MAXV = 9999


def _divisors(v):
    return [c for c in range(2, 10) if v % c == 0 and v // c >= 2]


def make_item(n_steps, rng, n_distract=0, return_cot=False):
    typ = rng.choice(["speed", "distance"])
    v = rng.randint(2, 9)
    role0, unit0 = START[typ]
    scen_lines = [f"A process begins with a {role0} of {v} {unit0}."]
    trace = []                                   # (sym, kind, v_prev, c, nv)
    for i in range(n_steps):
        rels = REL[typ]
        feas = []
        for r in rels:
            if r["kind"] == "mul" and v * 2 <= MAXV:
                feas.append(r)
            elif r["kind"] == "div" and _divisors(v):
                feas.append(r)
        if i == 0:                               # 首步强制 mul, 让携带量成为合数(后续除法可整除)
            feas = [r for r in feas if r["kind"] == "mul"] or feas
        # 量级控制: 大则优先除、小则优先乘
        pref = [r for r in feas if r["kind"] == "div"] if v > 1500 else \
               [r for r in feas if r["kind"] == "mul"] if v < 30 else feas
        r = rng.choice(pref or feas)
        if r["kind"] == "mul":
            c = rng.randint(2, min(9, MAXV // v)); nv = v * c; op = "*"
        else:
            c = rng.choice(_divisors(v)); nv = v // c; op = "/"
        scen_lines.append(r["scen"].format(c=c))
        trace.append((r["sym"], op, v, c, nv))
        typ = r["out"]; v = nv
    ans = v

    if n_distract:
        ds = [rng.choice(DISTRACT).format(c=rng.randint(2, 99)) for _ in range(n_distract)]
        for d in ds:
            scen_lines.insert(rng.randint(1, len(scen_lines)), d)   # 不放最前(保留起始句)

    q = " ".join(scen_lines) + " What is the final numeric value? (Give a single integer.)"
    if return_cot:
        parts = [f"start = {trace[0][2]}" if trace else f"start = {ans}"]
        for sym, op, pv, c, nv in trace:
            parts.append(f"{sym}: {pv} {op} {c} = {nv}")
        cot = "; ".join(parts) + f". The final value is {ans}."
        return q, ans, cot
    return q, ans


def build(hops, n_per, seed, n_distract=0, src_rank=0, emit_cot=False):
    import random
    rng = random.Random(seed)
    out = []
    hmin, hmax = min(hops), max(hops)
    for h in hops:
        for _ in range(n_per):
            q, ans, *rest = make_item(h, rng, n_distract, return_cot=emit_cot)
            diff = src_rank + (0.0 if hmax == hmin else (h - hmin) / (hmax - hmin))
            rec = {"domain": "physics", "source": "physics_relchain", "source_rank": src_rank,
                   "difficulty": round(diff, 4), "question": q,
                   "answer": str(ans), "gold": str(ans), "gold_kind": "int",
                   "options": None, "meta": {"n_steps": h, "n_distract": n_distract}}
            if emit_cot:
                rec["cot"] = rest[0]
            out.append(rec)
    return out


def _assert_cot_exact(cot):
    """解析 CoT 每个 'sym: a op b = z' 子句, 确认算术精确(除法整除)。"""
    body = cot.rsplit(".", 2)[0]
    for clause in body.split(";"):
        clause = clause.strip()
        if ":" not in clause or clause.startswith("start"):
            continue
        expr = clause.split(":", 1)[1]              # " a op b = z"
        lhs, rhs = expr.split("=")
        z = int(rhs.strip())
        if " * " in lhs:
            a, b = [int(t) for t in lhs.split(" * ")]; assert a * b == z, clause
        elif " / " in lhs:
            a, b = [int(t) for t in lhs.split(" / ")]
            assert b and a % b == 0 and a // b == z, f"非整除/不成立: {clause}"


def _selftest():
    import random
    rng = random.Random(0)
    for h in [1, 3, 6, 8]:
        q, ans, cot = make_item(h, rng, return_cot=True)
        assert isinstance(ans, int) and 2 <= ans <= MAXV, (h, ans)
        assert "What is the final" in q
        assert cot.count(";") == h, f"CoT 步数≠关系步数: {h} vs {cot}"
        assert cot.rstrip(".").endswith(str(ans)), (cot, ans)
        _assert_cot_exact(cot)
        # 关系被点名(选择推理): 每步含一个 sym
        assert any(s in cot for s in ["d = v", "v = d", "W = F", "F = m", "P = W", "a = v", "a = F", "W = P"]), cot
    # 干扰不改答案
    r1 = random.Random(9); q1, a1 = make_item(4, r1, n_distract=0)
    r2 = random.Random(9); q2, a2 = make_item(4, r2, n_distract=3)
    assert a1 == a2, "干扰改变了答案"
    assert len(q2) > len(q1), "干扰句没插进去"
    # 大量抽样: 量级恒 ≤9999、CoT 全精确、8 步链真 8 步
    for s in range(300):
        q, ans, cot = make_item(8, random.Random(s), return_cot=True)
        assert 2 <= ans <= MAXV, (s, ans)
        _assert_cot_exact(cot)
    # build: 难度单调 + 双字段
    pool = build([1, 4, 8], 5, 0, emit_cot=True)
    d1 = [r["difficulty"] for r in pool if r["meta"]["n_steps"] == 1][0]
    d8 = [r["difficulty"] for r in pool if r["meta"]["n_steps"] == 8][0]
    assert d1 < d8 and all(r["answer"] == r["gold"] for r in pool)
    print("selftest OK: 关系选择链(点名公式) + 量级≤9999 + 整除精确 + 干扰不改答案 + 难度单调 + CoT自洽")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/data/pools/physics_pool.json")
    ap.add_argument("--hops", default="1,2,3,4,5,6,8")
    ap.add_argument("--distract", default="0")
    ap.add_argument("--n_per", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit_cot", action="store_true",
                    help="每题附 cot 字段(逐步点名关系+求值), 导出可直接训练的 {question,cot,answer}")
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
    print(f"[physics] wrote {len(out)} -> {a.out}  hops={hops} distract={distracts}" + ("  (+cot)" if a.emit_cot else ""))
    print(f"[physics] 难度范围 {min(r['difficulty'] for r in out):.3f}~{max(r['difficulty'] for r in out):.3f}")
    print(f"[physics] 测: run_difficulty_depth_colar.py --pool {a.out}")


if __name__ == "__main__":
    main()
