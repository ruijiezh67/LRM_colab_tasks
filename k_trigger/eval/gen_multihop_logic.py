# -*- coding: utf-8 -*-
# ═══ 本地多跳蕴含逻辑集生成器（跨数据集深度阶梯验证用, 2026-07-31）═══════════
#   用途: 造 ProntoQA 式虚构本体 N-hop 蕴含题, 难度=跳数(串行步数, by construction),
#         跳数 2→14 可控 → 用来测 colar_r1_logic 的"深度是否随跳数升"且高跳深度能否 >FOLIO(14.6)
#         形成 ProsQA<FOLIO<本集 的干净跨数据集串行阶梯(避开 LogiQA2 那种"难而并行").
#   输出: pool json(schema 对齐 run_difficulty_depth_colar.load_pool: question/answer/difficulty/source).
#   离线, 无外部依赖.
import json, random, argparse

# 虚构类词(与 ProsQA/ProntoQA 同风格, 模型见过这类构词)
WORDS = ["wumpus","rempus","shumpus","yimpus","terpus","fompus","lempus","zumpus","dumpus",
         "vumpus","gorpus","sterpus","impus","numpus","brimpus","jompus","tumpus","daumpus",
         "lorpus","grimpus","scrompus","phorpus","kurpus","hilpus","zilpus","frumpus"]
NAMES = ["Tom","Alex","Sam","Max","Rex","Fae","Wren","Kip","Jo","Vic"]

def make_item(hops, rng, positive):
    chain = rng.sample(WORDS, hops + 1)           # c0 -> c1 -> ... -> c_hops
    rules = [f"Every {chain[i]} is a {chain[i+1]}." for i in range(hops)]
    rng.shuffle(rules)                            # 打乱规则顺序 -> 逼出真串行检索
    name = rng.choice(NAMES)
    fact = f"{name} is a {chain[0]}."
    if positive:
        target = chain[-1]; gold = "yes"          # 链末端 -> 蕴含成立
    else:
        # 反例: 问一个不在链上的类 -> 不蕴含
        off = rng.choice([w for w in WORDS if w not in chain]); target = off; gold = "no"
    q = " ".join(rules) + f" {fact} True or false: {name} is a {target}?"
    return {"source": "multihop_logic", "domain": "logic", "question": q,
            "answer": gold, "difficulty": float(hops), "steps": ["(hop)"] * hops}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hops", default="2,3,4,6,8,10,12,14")
    ap.add_argument("--per", type=int, default=16)   # 每档题数(yes/no 各半)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    hops = [int(h) for h in a.hops.split(",")]
    items = []
    for h in hops:
        for i in range(a.per):
            items.append(make_item(h, rng, positive=(i % 2 == 0)))
    rng.shuffle(items)
    json.dump(items, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"生成 {len(items)} 题 -> {a.out}  跳数分布 {dict(Counter(int(x['difficulty']) for x in items))}")
    print("样例:", items[0]["question"][:160], "| gold", items[0]["answer"])

if __name__ == "__main__":
    main()
