#!/usr/bin/env python
# 受控双轴深度评测: 复用 run_difficulty_depth.run_colar, 对 synth_test_grid 逐题测 latent 深度,
# 轴A=Spearman(depth,k)@D=0, 轴B=Spearman(depth,D)@固定k。丢弃 cap_out 记录算深度相关。
import os, sys, json, re, argparse, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_difficulty_depth import run_colar
try:
    from scipy.stats import spearmanr
    def SP(x, y): return round(float(spearmanr(x, y).correlation), 3) if len(x) > 2 and len(set(x)) > 1 else float("nan")
except Exception:
    def SP(x, y): return float("nan")

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="E:/data/pools/synth_test_grid.json")
ap.add_argument("--out", required=True)
ap.add_argument("--answer_field", default="answer")
a = ap.parse_args()

grid = json.loads(open(a.grid, encoding="utf-8").read())
recs = []
n = 0
for it, (ans, depth, cap) in zip(grid, run_colar(grid)):
    m = re.findall(r"-?\d+", ans or "")
    pred = m[0] if m else ""
    recs.append({"k": it.get("k"), "D": it.get("D"), "axis": it.get("axis"), "source": it.get("source"),
                 "depth": depth, "cap": bool(cap), "correct": (pred == str(it[a.answer_field]).strip()),
                 "pred": pred, "gold": str(it[a.answer_field])})
    n += 1
    if n % 60 == 0:
        print(f"  {n}/{len(grid)}", flush=True)

def acc(rs): return round(sum(r["correct"] for r in rs) / max(1, len(rs)), 3)
def caprate(rs): return round(sum(r["cap"] for r in rs) / max(1, len(rs)), 3)

out = {"overall": {"n": len(recs), "acc": acc(recs), "cap_rate": caprate(recs)}}
# 轴 A: D=0, depth vs k (丢 cap)
A = [r for r in recs if r["axis"] == "A" and not r["cap"]]
out["axisA"] = {"n": len(A), "acc": acc([r for r in recs if r["axis"] == "A"]),
                "spearman_depth_k": SP([r["k"] for r in A], [r["depth"] for r in A]),
                "by_k": {k: round(st.mean([r["depth"] for r in A if r["k"] == k]), 2)
                         for k in sorted(set(r["k"] for r in A))}}
# 轴 B: 每固定 k, depth vs D (丢 cap)
out["axisB"] = {}
Braw = [r for r in recs if r["axis"] == "B"]
for KB in sorted(set(r["k"] for r in Braw)):
    B = [r for r in Braw if r["k"] == KB and not r["cap"]]
    out["axisB"][f"k{KB}"] = {"n": len(B), "acc": acc([r for r in Braw if r["k"] == KB]),
                              "spearman_depth_D": SP([r["D"] for r in B], [r["depth"] for r in B]),
                              "by_D": {d: round(st.mean([r["depth"] for r in B if r["D"] == d]), 2)
                                       for d in sorted(set(r["D"] for r in B))}}
json.dump({"agg": out, "records": recs}, open(a.out, "w"), ensure_ascii=False, indent=1)
print("\n=== overall ===", out["overall"])
print("=== 轴A Sp(depth,k) =", out["axisA"]["spearman_depth_k"], "| by_k", out["axisA"]["by_k"])
for kb, v in out["axisB"].items():
    print(f"=== 轴B {kb} Sp(depth,D) =", v["spearman_depth_D"], "| by_D", v["by_D"], "| acc", v["acc"])
print("saved ->", a.out)
