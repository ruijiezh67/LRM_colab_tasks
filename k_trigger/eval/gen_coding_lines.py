# -*- coding: utf-8 -*-
# ═══ 符号执行追踪 —— coding 域深度数据集生成器（任务3B 重设, 2026-08-01）════════
#   动机: 旧版是"整数算术链", 本质与数学同构, 无法区分 coding 与 math。重设为**符号/状态执行**:
#         追踪一个序列(数字列表 或 字母串)经 append/pop/reverse/sort/dup + 布尔条件分支后的末状态。
#         **无算术**——考的是数据结构状态推理 + 控制流, 与数学正交。
#   铁律(任务3B): 难度轴=真串行操作步数; 答案短(≤8 字符); 加"死变量干扰轴"复刻双轴设计。
#   离线, 无外部依赖。生成时用同一解释器执行(真值), CoT 逐操作追踪状态。
# ══════════════════════════════════════════════════════════════════════════
r"""难度=符号操作步数的序列状态追踪。

统一表示: 状态=单字符 token 序列。两种字母表(每题择一):
  - 数字列表: token ∈ '0'..'9'（渲染 [1, 3, 5]）
  - 字母串:   token ∈ 'a'..'z'（渲染 "cba"）
操作(每步依赖前一步的状态 -> 不可并行):
  append(x) / pop(末) / popleft(首) / reverse / sort / dup(复制末元素)
  条件步: 先对当前状态判定谓词(长度>k / 长度为偶 / 末元素>k / 末字母为元音), 再执行两分支之一。
答案 = 末状态所有 token 拼接(数字列表→"135", 字母串→"cba"), 短、可子串匹配。
干扰轴(可选): 插入 D 条对**另一个无关变量 B** 的操作, 难度感升、被问变量 A 的串行链不变。

用法:
  python gen_coding_lines.py --selftest
  python gen_coding_lines.py --out E:/data/pools/coding_pool.json --hops 1,2,4,6,8,10,12,14 --n_per 1120 --emit_cot
  python gen_coding_lines.py --out E:/data/pools/coding_distract.json --hops 6 --distract 0,2,4,6 --n_per 40 --emit_cot
"""
from __future__ import annotations

import argparse
import json

VOWELS = set("aeiou")
MAXLEN = 8                                   # 状态长度上限(保证答案短)


def _render(seq):
    """CoT 里的可读状态: 空格分隔。"""
    return " ".join(seq)


def _answer(seq):
    """答案 = token 拼接(单字符, 无歧义): [1,3,5]->'135', ['c','b','a']->'cba'。"""
    return "".join(seq)


def _valid_ops(seq):
    """当前状态下合法的操作(避免 pop 空序列这类需要兜底的分支)。"""
    ops = ["reverse", "sort", "dup", "append"]
    if len(seq) >= MAXLEN:
        ops.remove("append")
    if len(seq) >= 2:
        ops += ["pop", "popleft"]
    return ops


def _apply(op, seq, rng, alpha):
    """执行一个操作, 返回 (new_seq, 英文描述)。op 须对 seq 合法。"""
    if op == "append":
        x = rng.choice(alpha)
        return seq + [x], f"append {x}"
    if op == "pop":
        return seq[:-1], "remove the last element"
    if op == "popleft":
        return seq[1:], "remove the first element"
    if op == "reverse":
        return seq[::-1], "reverse it"
    if op == "sort":
        return sorted(seq), "sort it ascending"
    if op == "dup":
        return seq + [seq[-1]], "duplicate the last element"
    raise ValueError(op)


def _predicate(seq, rng, is_digit):
    """挑一个当前可判定的谓词, 返回 (bool 值, 英文描述)。"""
    choices = ["len_gt", "len_even"]
    if is_digit:
        choices.append("last_gt")
    else:
        choices.append("last_vowel")
    p = rng.choice(choices)
    if p == "len_gt":
        k = rng.randint(1, max(1, len(seq)))
        return len(seq) > k, f"the sequence has more than {k} elements"
    if p == "len_even":
        return len(seq) % 2 == 0, "the sequence has an even number of elements"
    if p == "last_gt":
        k = rng.randint(0, 8)
        return int(seq[-1]) > k, f"the last element is greater than {k}"
    if p == "last_vowel":
        return seq[-1] in VOWELS, "the last letter is a vowel"
    raise ValueError(p)


def make_item(n_steps, rng, n_distract=0, return_cot=False):
    """返回 (question, answer, [cot])。n_steps 个串行符号操作(可含条件步) + n_distract 条死变量操作。"""
    is_digit = rng.random() < 0.5
    alpha = [str(d) for d in range(10)] if is_digit else list("abcdefghijklmnopqrstuvwxyz")
    seq = [rng.choice(alpha) for _ in range(rng.randint(2, 4))]
    start = list(seq)

    prog_lines = []          # 被问变量 A 的操作行(英文)
    cot_steps = []           # CoT: (描述, 新状态渲染)
    for _ in range(n_steps):
        if len(seq) >= 1 and rng.random() < 0.30:          # 条件步(控制流)
            val, pdesc = _predicate(seq, rng, is_digit)
            valid = _valid_ops(seq)
            op_t = rng.choice(valid)
            op_f = rng.choice([o for o in valid if o != op_t] or valid)
            # 两分支各只算一次(append 抽的字符固定), 保证题面描述与执行一致
            seq_t, desc_t = _apply(op_t, seq, rng, alpha)
            seq_f, desc_f = _apply(op_f, seq, rng, alpha)
            prog_lines.append(f"If {pdesc}, then {desc_t}; otherwise {desc_f}.")
            seq2, taken_desc = (seq_t, desc_t) if val else (seq_f, desc_f)
            cot_steps.append((f"check: {pdesc}? {'yes' if val else 'no'} -> {taken_desc}", _render(seq2)))
            seq = seq2
        else:                                              # 普通操作步
            op = rng.choice(_valid_ops(seq))
            seq2, desc = _apply(op, seq, rng, alpha)
            prog_lines.append(f"{desc[0].upper()}{desc[1:]}.")
            cot_steps.append((desc, _render(seq2)))
            seq = seq2

    # 干扰: 对无关变量 B 的操作, 交错进题面(不进 CoT, 不影响 A)
    if n_distract:
        b0 = [rng.choice(alpha) for _ in range(rng.randint(2, 3))]
        dlines = [f"(On a separate list B: {_apply(rng.choice(_valid_ops(b0)), b0, rng, alpha)[1]}.)"
                  for _ in range(n_distract)]
        for dl in dlines:
            prog_lines.insert(rng.randint(0, len(prog_lines)), dl)

    kind = "list of numbers" if is_digit else "string of letters"
    q = (f"Start with the {kind}: {_render(start)}\n"
         + "\n".join(prog_lines)
         + "\nWhat is the final sequence? Give the elements concatenated (e.g. 135 or cba).")
    ans = _answer(seq)
    if return_cot:
        parts = [f"start: {_render(start)}"] + [f"{d} => {st}" for d, st in cot_steps]
        cot = "; ".join(parts) + f". The final sequence is {ans}."
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
            rec = {"domain": "coding", "source": "coding_symexec", "source_rank": src_rank,
                   "difficulty": round(diff, 4), "question": q,
                   "answer": str(ans), "gold": str(ans), "gold_kind": "str",
                   "options": None, "meta": {"n_steps": h, "n_distract": n_distract}}
            if emit_cot:
                rec["cot"] = rest[0]
            out.append(rec)
    return out


def _selftest():
    import random
    rng = random.Random(0)
    # 状态可控 + 答案是拼接短串 + 无算术符号(+,-,*,/ 不出现在 CoT)
    for h in [1, 3, 6, 12, 20]:
        q, ans, cot = make_item(h, rng, return_cot=True)
        assert 1 <= len(ans) <= MAXLEN, (h, ans)
        assert cot.rstrip(".").endswith(ans), (h, cot, ans)
        assert cot.count("=>") == h, f"CoT 步数≠操作步数: {h} vs {cot}"
        assert not any(sym in cot for sym in ["+", "*", "/", "×"]), f"CoT 含算术符号(非符号执行): {cot}"
    # 答案真的可由操作序列执行得到(解释器自洽)——用 CoT 末状态核对
    for h in [2, 5, 8]:
        q, ans, cot = make_item(h, random.Random(100 + h), return_cot=True)
        last_state = cot.split("=>")[-1].split(". The final")[0].strip()
        assert _answer(last_state.split()) == ans, (cot, ans)
    # 条件步真出现(控制流): 多抽几题应见 "check:"
    seen_cond = any("check:" in make_item(8, random.Random(s), return_cot=True)[2] for s in range(20))
    assert seen_cond, "从未生成条件步(控制流缺失)"
    # 干扰不改答案: 同种子, distract 只插 B 变量行
    r1 = random.Random(7); q1, a1 = make_item(5, r1, n_distract=0)
    r2 = random.Random(7); q2, a2 = make_item(5, r2, n_distract=4)
    assert a1 == a2, "干扰改变了答案(干扰轴不干净)"
    assert len(q2) > len(q1), "干扰行没插进题面"
    # build: 难度单调 + 双字段兼容 + emit_cot
    pool = build([2, 4, 8], 5, 0, emit_cot=True)
    assert all(set(r) >= {"question", "answer", "gold", "difficulty", "cot"} for r in pool)
    d2 = [r["difficulty"] for r in pool if r["meta"]["n_steps"] == 2][0]
    d8 = [r["difficulty"] for r in pool if r["meta"]["n_steps"] == 8][0]
    assert d2 < d8 and all(r["answer"] == r["gold"] for r in pool)
    print("selftest OK: 符号执行(list/str + 控制流) + 无算术 + 解释器自洽 + 条件步存在 + 干扰不改答案 + CoT自洽")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/data/pools/coding_pool.json")
    ap.add_argument("--hops", default="1,2,4,6,8,10,12,14", help="逗号分隔的操作步数档")
    ap.add_argument("--distract", default="0", help="逗号分隔的死变量操作条数(多档=干扰轴对照)")
    ap.add_argument("--n_per", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit_cot", action="store_true",
                    help="每题附 cot 字段(逐操作状态追踪), 导出可直接训练的 {question,cot,answer}")
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
    print(f"[coding] wrote {len(out)} -> {a.out}" + ("  (+cot)" if a.emit_cot else ""))
    print(f"[coding] hops={hops} distract={distracts} n_per={a.n_per}")
    print(f"[coding] 难度范围 {min(r['difficulty'] for r in out):.3f}~{max(r['difficulty'] for r in out):.3f}")
    print(f"[coding] 测: run_difficulty_depth_colar.py --pool {a.out}  (colar-gsm 基线探针)")


if __name__ == "__main__":
    main()
