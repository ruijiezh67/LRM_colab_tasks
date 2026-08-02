# -*- coding: utf-8 -*-
# ═══ 新域 pool 去重 + 分档 train/test 划分（任务3B, 2026-07-31）════════════
#   合成题可能重复(尤其短链档), 且训练/测试不能用同一批题(泄漏)。
#   本脚本: 按 question 去重 -> 每个难度档(meta.n_steps)内 shuffle -> 90/10 切 train/test。
#   同一来源切分 => train/test 分布一致、无泄漏。
#   用法: python split_traintest.py --pool E:/data/pools/coding_pool.json --frac 0.9
# ══════════════════════════════════════════════════════════════════════════
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split(pool, frac=0.9, seed=0):
    # 去重
    seen, uniq = set(), []
    for r in pool:
        q = r["question"]
        if q not in seen:
            seen.add(q); uniq.append(r)
    n_dup = len(pool) - len(uniq)
    # 按难度档分组, 组内切分(保证 train/test 每档都有)
    by_tier = defaultdict(list)
    for r in uniq:
        by_tier[r["meta"]["n_steps"]].append(r)
    rng = random.Random(seed)
    train, test = [], []
    for t in sorted(by_tier):
        rows = by_tier[t]; rng.shuffle(rows)
        k = int(round(len(rows) * frac))
        train += rows[:k]; test += rows[k:]
    return train, test, n_dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--frac", type=float, default=0.9, help="train 占比")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = json.load(open(a.pool, encoding="utf-8"))
    train, test, n_dup = split(pool, a.frac, a.seed)
    base = Path(a.pool)
    tr = base.with_name(base.stem + "_train.json")
    te = base.with_name(base.stem + "_test.json")
    json.dump(train, open(tr, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(test, open(te, "w", encoding="utf-8"), ensure_ascii=False)
    # 校验无泄漏
    overlap = {r["question"] for r in train} & {r["question"] for r in test}
    assert not overlap, f"train/test 泄漏 {len(overlap)} 条"
    print(f"[split] {base.name}: 去重删 {n_dup} 条 -> train {len(train)} / test {len(test)} (泄漏 0)")
    print(f"        -> {tr.name} , {te.name}")


if __name__ == "__main__":
    main()
