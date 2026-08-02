# -*- coding: utf-8 -*-
# ═══ 提示框架混淆对照（2026-07-31）════════════════════════════════════════
#   起因: ProofWriter 探针发现 —— 同一批题、同一个模型, 只改提示措辞, 平均潜在深度
#         从 4.6 跳到 21.7(4.7 倍)。而跨数据集深度阶梯(ProsQA 7.3 < FOLIO 14.6)
#         从来是"每个数据集用自己的模板"测出来的, 从未控制过这个变量。
#   设计: 3 数据集 × 2 提示框架 的完全交叉。每个数据集的**内容与答案指令不变**,
#         只换外层框架词(Premises:/Conclusion: vs Facts and rules:/Question:)。
#   判读: 排序在两种框架下一致 => 阶梯稳健(格式只是每集一个偏移);
#         排序翻转或差距塌缩 => 跨集阶梯是提示格式伪影, 主结论需重写。
#   产出: outputs/prompt_frame_control.json   日期: 2026-07-31
# ══════════════════════════════════════════════════════════════════════════
"""跨数据集深度阶梯的提示框架对照。

用法(GPU):
  PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  COLAR_BASE=E:/ckpts/r1-distill-1.5b \
  COLAR_CKPT=E:/ckpts/dual_track/colar_r1_logic/colar_r1_logic.ckpt \
  COLAR_EMB_STD=0.03 COLAR_COMPRESS=5 COLAR_MAXLAT=64 \
  python prompt_frame_control.py --n 40
自检: python prompt_frame_control.py --selftest
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SOURCES = ["prosqa", "folio", "proofwriter"]


# ── 把各数据集的题面拆成 (上下文, 问句, 答案指令) ────────────────────────────
def split_item(x):
    """返回 (ctx, query, instr)。instr 可为空(ProsQA 的问句自带指令性)。"""
    q, s = x["question"], x["source"]
    if s == "prosqa":
        # "<规则...> Is Tom a lempus or scrompus?"  —— 末句以大写 Is 开头
        i = q.rfind("Is ")
        return q[:i].strip(), q[i:].strip(), ""
    if s == "folio":
        # "Premises: {ctx}\nConclusion: {concl}\n{instr}"
        body = q.split("Premises:", 1)[1]
        ctx, rest = body.split("\nConclusion:", 1)
        parts = rest.rsplit("\n", 1)
        concl, instr = (parts[0], parts[1]) if len(parts) == 2 else (rest, "")
        return ctx.strip(), concl.strip(), instr.strip()
    if s == "proofwriter":
        body = q.split("Facts and rules:", 1)[1]
        ctx, rest = body.split("\nQuestion:", 1)
        instr, stmt = "Is the following statement true or false?", rest
        if instr in rest:
            stmt = rest.split(instr, 1)[1]
        return ctx.strip(), stmt.strip(), instr
    raise ValueError(s)


# ── 两个提示框架(只换外层框架词, 内容与指令不动) ─────────────────────────────
FRAMES = {
    # FOLIO 风格: 模型 warm-start 时在 FOLIO 上见过
    "folio_frame": lambda ctx, query, instr:
        f"Premises: {ctx}\nConclusion: {query}" + (f"\n{instr}" if instr else ""),
    # ProofWriter 风格: 我们给 ProofWriter 写的那套
    "pw_frame": lambda ctx, query, instr:
        f"Facts and rules: {ctx}\nQuestion: " + (f"{instr} {query}" if instr else query),
}


def render(x, frame):
    ctx, query, instr = split_item(x)
    return {**x, "question": FRAMES[frame](ctx, query, instr)}


# ── 统计 ────────────────────────────────────────────────────────────────────
def cell_summary(recs):
    cor = [r for r in recs if r["correct"]]
    dep = [r["depth"] for r in recs]
    return {
        "n": len(recs),
        "acc": round(sum(r["correct"] for r in recs) / len(recs), 3) if recs else None,
        "cap_out": round(sum(r["cap_out"] for r in recs) / len(recs), 3) if recs else None,
        "depth_all": round(st.mean(dep), 1) if dep else None,
        "depth_correct": round(st.mean(r["depth"] for r in cor), 1) if cor else None,
        "depth_min": min(dep) if dep else None, "depth_max": max(dep) if dep else None,
    }


def _selftest():
    pw = {"source": "proofwriter",
          "question": "Facts and rules: A is b. B is c.\nQuestion: Is the following statement "
                      "true or false? A is c.", "gold": "True"}
    ctx, qy, instr = split_item(pw)
    assert ctx == "A is b. B is c." and qy == "A is c." and "true or false" in instr, (ctx, qy, instr)
    fo = {"source": "folio",
          "question": "Premises: P1. P2.\nConclusion: C.\nIs the conclusion True, False, or "
                      "Uncertain given the premises?", "gold": "True"}
    ctx, qy, instr = split_item(fo)
    assert ctx == "P1. P2." and qy == "C." and instr.startswith("Is the conclusion"), (ctx, qy, instr)
    pr = {"source": "prosqa",
          "question": "Every a is a b. Tom is a a. Is Tom a b or c?", "gold": "Tom is a b."}
    ctx, qy, instr = split_item(pr)
    assert ctx == "Every a is a b. Tom is a a." and qy == "Is Tom a b or c?" and instr == "", (ctx, qy)
    # 渲染后: 内容与指令必须原样保留, 只有框架词变
    for x in (pw, fo, pr):
        for f in FRAMES:
            out = render(x, f)["question"]
            c, q2, i2 = split_item(x)
            assert c in out and q2 in out, (f, x["source"])
            if i2:
                assert i2 in out, (f, x["source"], "指令丢了")
    assert render(pr, "folio_frame")["question"].startswith("Premises: ")
    assert render(pr, "pw_frame")["question"].startswith("Facts and rules: ")
    print("selftest OK: 三源题面拆分 + 两框架渲染(内容/指令保真, 只换框架词)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="每个 (数据集 × 框架) 格的题数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "prompt_frame_control.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return

    from run_difficulty_depth import sample_pool, grade, run_colar

    res = {"n_per_cell": a.n, "cells": {}}
    for src in SOURCES:
        items = sample_pool("logic", a.n, a.seed, sources=[src])   # 同一批题跨框架复用
        for frame in FRAMES:
            it = [render(x, frame) for x in items]
            recs = []
            for x, ro in zip(it, run_colar(it, 16)):
                pred, depth, cap = ro[0], ro[1], ro[2]
                recs.append({"depth": depth, "cap_out": bool(cap),
                             "correct": bool(grade(x, pred)), "gold": str(x["gold"]),
                             "pred": (pred or "")[:120], "difficulty": x["difficulty"]})
            res["cells"][f"{src}|{frame}"] = {**cell_summary(recs), "records": recs}
            c = res["cells"][f"{src}|{frame}"]
            print(f"[{src:12s} {frame:12s}] n={c['n']} acc={c['acc']} "
                  f"depth_all={c['depth_all']} depth_correct={c['depth_correct']} "
                  f"cap={c['cap_out']}", flush=True)
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n==== 深度阶梯 × 提示框架 (depth_all / depth_correct) ====")
    print(f"{'数据集':14s}" + "".join(f"{f:>26s}" for f in FRAMES))
    for src in SOURCES:
        row = f"{src:14s}"
        for f in FRAMES:
            c = res["cells"].get(f"{src}|{f}", {})
            row += f"{str(c.get('depth_all')) + ' / ' + str(c.get('depth_correct')):>26s}"
        print(row)
    print("\n判读: 两列的**排序**一致 => 阶梯稳健; 排序翻转/差距塌缩 => 跨集阶梯是提示格式伪影。")
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
