# -*- coding: utf-8 -*-
# ═══ MATH 可解阶梯子集 —— 任务2 数学 cap-out 治法（2026-08-01）══════════════
#   背景: r1_hardmath 在 MATH500 撞顶率 92%+, 抬 cap(96->160) 反更糟(空转), 已排除。
#   结论: 不重训、不抬 cap, 改在**可解难度档**上重测深度——取 MATH L1-L3(题真能解出,
#         深度是"解题串行步数"而非"不会做时瞎转到 eval 上限")。L4-L5 超纲档丢弃。
#   本脚本纯 CPU 过滤(零 GPU): 从 math_pool.json 取 source=math500 且 meta.level∈{1,2,3},
#   保持原 schema(runner 直接吃), meta.level 保留 -> depth ladder 可按 level 分档。
#   用法:
#     python make_math_solvable_ladder.py --selftest
#     python make_math_solvable_ladder.py --pool E:/data/pools/math_pool.json \
#            --out E:/data/pools/math_l123_pool.json --levels 1,2,3
# ══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
from collections import Counter


def filter_ladder(pool, source="math500", levels=(1, 2, 3), level_as_difficulty=False):
    lv = set(levels)
    out = [r for r in pool
           if r.get("source") == source and (r.get("meta") or {}).get("level") in lv]
    if level_as_difficulty:
        # 把 difficulty 覆写为 meta.level(1/2/3), 得干净三档(depth ladder 直接按档聚合)。
        # 原 difficulty 是 math_pool 连续值(1.0-2.5), 与 level 不干净对应。
        for r in out:
            r["difficulty"] = float(r["meta"]["level"])
    return out


def _selftest():
    # 合成 pool: 覆盖 level 1-5 + 另一来源, 确认只留 math500 L1-L3
    pool = []
    for lvl in range(1, 6):
        for _ in range(3):
            pool.append({"source": "math500", "question": f"q{lvl}", "gold": "1",
                         "meta": {"level": lvl}})
    pool += [{"source": "gsm8k", "question": "g", "gold": "1", "meta": {"level": 1}}]  # 别的源
    pool += [{"source": "math500", "question": "nolv", "gold": "1", "meta": {}}]        # 无 level
    out = filter_ladder(pool)
    assert len(out) == 9, len(out)                              # 3 levels x 3
    assert {r["meta"]["level"] for r in out} == {1, 2, 3}
    assert all(r["source"] == "math500" for r in out)
    # level_as_difficulty: difficulty 覆写为 level(1/2/3)
    out2 = filter_ladder([dict(r, difficulty=9.9) for r in pool], level_as_difficulty=True)
    assert {r["difficulty"] for r in out2} == {1.0, 2.0, 3.0}, out2[0]["difficulty"]
    print("selftest OK: 只留 math500 L1-L3, 排除 L4-5 / 他源 / 无 level; --level_as_difficulty 覆写为 1/2/3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="E:/data/pools/math_pool.json")
    ap.add_argument("--out", default="E:/data/pools/math_l123_pool.json")
    ap.add_argument("--source", default="math500")
    ap.add_argument("--levels", default="1,2,3", help="逗号分隔的保留难度档")
    ap.add_argument("--level_as_difficulty", action="store_true",
                    help="把 difficulty 覆写为 meta.level(1/2/3), 得干净三档供 depth ladder 按档聚合")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    pool = json.load(open(a.pool, encoding="utf-8"))
    levels = tuple(int(x) for x in a.levels.split(",") if x.strip())
    out = filter_ladder(pool, a.source, levels, a.level_as_difficulty)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    tiers = Counter(r["meta"]["level"] for r in out)
    subj = Counter(r["meta"].get("subject") for r in out)
    print(f"[math-ladder] {a.source} L{levels} -> {len(out)} 题 -> {a.out}")
    print(f"[math-ladder] 每档: {dict(sorted(tiers.items()))}")
    print(f"[math-ladder] 学科: {dict(sorted(subj.items(), key=lambda x: -x[1]))}")
    print(f"[math-ladder] 下一步(GPU): run_difficulty_depth_colar.py --pool {a.out}  (按 meta.level 读深度阶梯)")


if __name__ == "__main__":
    main()
