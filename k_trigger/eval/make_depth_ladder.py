# -*- coding: utf-8 -*-
# ═══ 跨数据集「深度阶梯」聚合出表（2026-07-31）══════════════════════════════
#   用途: 把各模型的逐题深度结果(depth_{logic,math,cs}_*.json / dd_*.json)按 (模型 × 数据集)
#         聚合成一张可直接进论文的表。**每格同时报三个口径**——全集 / 答对子集 / 非撞顶子集。
#   为什么要三口径: 本项目所有踩过的坑都源于只报一个数——
#         · acc≈0 时的"平均深度"是空转算力, 不是推理深度(自造 multihop、colar-gsm 跨域);
#         · cap_out 高时的"平均深度"是撞顶假象(warm-start 在 MATH500 上 cap=92%, 深度 91.5 -> 非撞顶仅 36.7);
#         · 只看全集会把这两种伪影当成"越难越深"。
#   逻辑域布局: 串行阶梯 ProsQA < FOLIO < ProofWriter 为主表; LogiQA2 单列为「并行题型对照」
#         (最难却最浅 => 深度跟串行链长走、不跟难度标签走)。
#   ⚠️ 2026-07-31 实测: 逻辑域三档里**只有 ProsQA 一档可信** —— FOLIO 三选一多数类基线 .353,
#         5 个模型实测 acc 全在 .275-.383(等于在猜); ProofWriter 换个提示框架深度从 18.8 变 4.4。
#         故本脚本的 !guess 标记与阶梯拒绝机制不是防御性代码, 是必要的。
#   产出: outputs/depth_ladder_<domain>.json + .md   日期: 2026-07-31
# ══════════════════════════════════════════════════════════════════════════
"""跨数据集深度阶梯聚合。

读法(每格):  acc | 全集深度 / 答对子集深度 / 非撞顶子集深度   [标记]
标记:  !cap = 撞顶率>10%(深度是撞顶假象)   !acc = 正确率<5%(深度是空转算力, 不可读作推理深度)

用法:
  python make_depth_ladder.py --domain logic
  python make_depth_ladder.py --domain math
  python make_depth_ladder.py --domain logic --extra outputs/dd_proofwriter_colar_r1_logic.json
  python make_depth_ladder.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_difficulty_depth import spearman            # 复用, 不重写

CKPTS = Path(os.environ.get("DD_RESULTS", "E:/ckpts/dual_track"))
OUTDIR = HERE.parent / "outputs"

# ── 每个域的数据集顺序、题型分类、展示名 ────────────────────────────────────
#   ladder = 主表的串行阶梯(按难度/串行步数递增); control = 单列对照(题型不同, 不进阶梯)
DOMAINS = {
    "logic": {
        "glob": "depth_logic_*.json",
        "ladder": ["prosqa", "folio", "proofwriter"],
        "control": ["logiqa2"],
        "task_type": {"prosqa": "串行·多跳蕴含", "folio": "串行·多前提FOL",
                      "proofwriter": "串行·证明深度0-5", "logiqa2": "并行·阅读理解+约束"},
        "label": {"prosqa": "ProsQA(易)", "folio": "FOLIO(中)",
                  "proofwriter": "ProofWriter(难)", "logiqa2": "LogiQA2"},
    },
    "math": {
        "glob": "depth_math_*.json",
        "ladder": ["gsm8k", "math500", "aime"],
        "control": [],
        "task_type": {"gsm8k": "串行·算术", "math500": "串行·竞赛数学", "aime": "串行·超纲"},
        "label": {"gsm8k": "GSM8K(易)", "math500": "MATH500(中)", "aime": "AIME(难)"},
    },
    "commonsense": {
        "glob": "depth_cs_*.json",
        "ladder": ["csqa", "strategyqa"],
        "control": [],
        "task_type": {"csqa": "非串行·常识", "strategyqa": "非串行·多跳常识"},
        "label": {"csqa": "CSQA", "strategyqa": "StrategyQA"},
    },
}

CAP_WARN = 0.10        # 撞顶率超过此值 -> 深度是撞顶假象
ACC_WARN = 0.05        # 开放式任务: 正确率低于此值 -> 深度是空转算力
MIN_RUNG = 5           # 阶梯每档至少要这么多"答对且非撞顶"的样本, 否则拒绝算跨档 Sp

# 各数据集的**猜测基线**(随机或多数类, 取高者)。选择题任务不能用绝对阈值判空转 ——
# FOLIO 三选一的多数类基线就有 0.353, 实测 acc 0.35 = 完全在猜, 但绝对阈值 0.05 抓不到。
# 实测(colar_r1_logic): ProsQA .95-.975(真会做) / FOLIO .35(=多数类 .353) / ProofWriter .60-.75(随机 .50)。
CHANCE = {
    "prosqa": 0.0,        # 开放式实体命名, 猜不中
    "folio": 0.353,       # 三选一 True/False/Uncertain, 多数类 72/204
    "logiqa2": 0.25,      # 四选一
    "proofwriter": 0.50,  # 二分类 True/False (标签严格 1:1)
    "gsm8k": 0.0, "math500": 0.0, "aime": 0.0,   # 开放数值
    "csqa": 0.20, "strategyqa": 0.50,
}
CHANCE_MARGIN = 0.10   # acc 高出猜测基线不足此值 -> 判为"在猜", 深度不可读作推理深度


# ── 读取 ────────────────────────────────────────────────────────────────────
def load_records(path):
    """支持两种 schema: {model,domain,records:[...]}(E:/ckpts 结果) 或 裸 list(dd_*.json)。"""
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        return d.get("records", []), d.get("model", "?")
    return d, "?"


def model_label(path, model_field):
    """depth_logic_r1_warmstart.json -> r1_warmstart ; dd_proofwriter_xxx.json -> proofwriter_xxx"""
    stem = Path(path).stem
    for pre in ("depth_logic_", "depth_math_", "depth_cs_", "dd_"):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


# ── 单格统计(三口径) ────────────────────────────────────────────────────────
def _agg(depths):
    if not depths:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(depths), "mean": round(st.mean(depths), 1),
            "median": round(st.median(depths), 1)}


def cell_stats(recs, source=None):
    """一个 (模型, 数据集) 格: 全集 / 答对子集 / 非撞顶子集 三口径 + acc + cap 率。"""
    n = len(recs)
    if not n:
        return None
    correct = [r for r in recs if r.get("correct")]
    noncap = [r for r in recs if not r.get("cap_out")]
    clean = [r for r in recs if r.get("correct") and not r.get("cap_out")]
    cap_rate = sum(bool(r.get("cap_out")) for r in recs) / n
    acc = len(correct) / n
    dep = lambda rs: [r["depth"] for r in rs if r.get("depth") is not None]
    out = {
        "n": n, "acc": round(acc, 3), "cap_out": round(cap_rate, 3),
        "depth_all": _agg(dep(recs)),
        "depth_correct": _agg(dep(correct)),
        "depth_noncap": _agg(dep(noncap)),
        # depth_clean = 答对 ∩ 非撞顶: 唯一可用于跨档比较的口径(其余三个各带一种伪影)
        "depth_clean": _agg(dep(clean)),
        "flags": [],
    }
    # 域内 Spearman(难度, 深度): 只在答对子集上读(项目口径)
    if len(correct) >= 3:
        out["sp_correct"] = round(spearman([r["difficulty"] for r in correct],
                                           [r["depth"] for r in correct]), 3)
    if cap_rate > CAP_WARN:
        out["flags"].append("!cap")
    chance = CHANCE.get(source, 0.0)
    out["chance"] = chance
    if chance > 0:
        # 选择题: 看 acc 是否显著高出猜测基线。FOLIO 实测 .35 vs 多数类 .353 -> 在猜。
        if acc < chance + CHANCE_MARGIN:
            out["flags"].append("!guess")
    elif acc < ACC_WARN:
        out["flags"].append("!acc")
    return out


def build(domain, files):
    cfg = DOMAINS[domain]
    sources = cfg["ladder"] + cfg["control"]
    table = {}
    for path in files:
        recs, model_field = load_records(path)
        if not recs:
            continue
        name = model_label(path, model_field)
        by_src = {}
        for r in recs:
            by_src.setdefault(r.get("source", "?"), []).append(r)
        row = {}
        for s in sources:
            c = cell_stats(by_src.get(s, []), source=s)
            if c:
                row[s] = c
        # 跨档 Spearman: 阶梯各档的平均深度是否随位次递增。
        #   必须建在 depth_clean(答对 ∩ 非撞顶)上 —— 用全集或答对子集会把撞顶假象读成"越难越深"
        #   (warm-start 在 MATH500/AIME 上 cap 92%/93%, 全集深度 93.4/91.4 会伪造出完美阶梯)。
        #   任一档 clean 样本 < MIN_RUNG 就拒绝出数, 并记下原因。
        rungs, bad = [], []
        for i, s in enumerate(cfg["ladder"]):
            if s not in row:
                bad.append(f"{s}:缺"); continue
            cl = row[s]["depth_clean"]
            if cl["n"] < MIN_RUNG or cl["mean"] is None:
                bad.append(f"{s}:可用样本{cl['n']}<{MIN_RUNG}"); continue
            if "!guess" in row[s]["flags"]:
                bad.append(f"{s}:acc{row[s]['acc']}≈猜测基线{row[s]['chance']}"); continue
            rungs.append((i, cl["mean"]))
        if len(rungs) >= 3:
            row["_ladder_sp"] = round(spearman([a for a, _ in rungs], [b for _, b in rungs]), 3)
            row["_ladder_n"] = [row[cfg["ladder"][i]]["depth_clean"]["n"] for i, _ in rungs]
        else:
            row["_ladder_sp"] = None
            row["_ladder_why"] = "; ".join(bad) or "阶梯档位不足3"
        if row:
            table[name] = row
    return {"domain": domain, "sources": sources, "ladder": cfg["ladder"],
            "control": cfg["control"], "task_type": cfg["task_type"], "models": table}


# ── 出 markdown ─────────────────────────────────────────────────────────────
def _fmt(c):
    if not c:
        return "—"
    f = lambda a: "—" if a["mean"] is None else f"{a['mean']}"
    # 分隔符不能用 "|" —— 会把 markdown 表格的列切开
    s = f"{c['acc']:.2f} · {f(c['depth_all'])} / {f(c['depth_correct'])} / {f(c['depth_noncap'])}"
    if c["flags"]:
        s += " **" + "".join(c["flags"]) + "**"
    return s


def to_md(res):
    cfg = DOMAINS[res["domain"]]
    L, C = res["ladder"], res["control"]
    out = [f"# 深度阶梯 · {res['domain']} 域", ""]
    out += ["每格读法：`acc · 全集深度 / 答对子集深度 / 非撞顶子集深度`", "",
            f"标记：`!cap` = 撞顶率 > {int(CAP_WARN*100)}%（深度是撞顶假象）　"
            f"`!acc` = 开放式任务正确率 < {int(ACC_WARN*100)}%（空转）　"
            f"`!guess` = 选择题正确率未高出猜测基线 {CHANCE_MARGIN:.2f} 以上（**模型在猜，深度不是推理深度**）", ""]
    # 主表: 串行阶梯
    out += [f"## 主表 · 串行阶梯（{' < '.join(cfg['label'][s] for s in L)}）", ""]
    out += ["| 模型 | " + " | ".join(cfg["label"][s] for s in L) + " | 阶梯 Sp（答对∩非撞顶） |",
            "|---|" + "---|" * (len(L) + 1)]
    for m, row in res["models"].items():
        cells = [_fmt(row.get(s)) for s in L]
        sp = row.get("_ladder_sp")
        sp_txt = f"**{sp}** (n={row.get('_ladder_n')})" if sp is not None \
            else f"不可算 — {row.get('_ladder_why', '')}"
        out.append(f"| {m} | " + " | ".join(cells) + f" | {sp_txt} |")
    out += ["", "> 阶梯 Sp 只在**答对 ∩ 非撞顶**子集上计算，且每档需 ≥"
            f"{MIN_RUNG} 个可用样本。用全集或答对子集会把撞顶假象读成「越难越深」"
            "——warm-start 模型在 MATH500/AIME 上撞顶率 92%/93%，其全集深度 93.4/91.4 "
            "能伪造出一条完美阶梯。"]
    # 对照行
    if C:
        out += ["", f"## 并行题型对照（{'、'.join(cfg['label'][s] for s in C)}）", "",
                "> 这些数据集**认知难度更高但不需要串行链**（同时权衡多约束、并行排除选项），"
                "深度反而更浅 —— 深度跟**串行链长**走，不跟**难度标签**走。", ""]
        out += ["| 模型 | " + " | ".join(cfg["label"][s] for s in C) + " |",
                "|---|" + "---|" * len(C)]
        for m, row in res["models"].items():
            out.append(f"| {m} | " + " | ".join(_fmt(row.get(s)) for s in C) + " |")
    # 题型说明
    out += ["", "## 数据集题型", "", "| 数据集 | 题型 | 角色 |", "|---|---|---|"]
    for s in res["sources"]:
        role = "串行阶梯" if s in L else "并行对照"
        out.append(f"| {cfg['label'][s]} | {cfg['task_type'][s]} | {role} |")
    # 域内 Spearman
    out += ["", "## 域内 Sp(难度, 深度)（答对子集）", "",
            "| 模型 | " + " | ".join(cfg["label"][s] for s in res["sources"]) + " |",
            "|---|" + "---|" * len(res["sources"])]
    for m, row in res["models"].items():
        out.append(f"| {m} | " + " | ".join(
            str(row[s].get("sp_correct", "—")) if s in row else "—" for s in res["sources"]) + " |")
    return "\n".join(out) + "\n"


# ── 自检 ────────────────────────────────────────────────────────────────────
def _selftest():
    recs = ([{"source": "prosqa", "difficulty": 0.1 * i, "depth": 5 + i,
              "correct": i % 2 == 0, "cap_out": False} for i in range(10)] +
            [{"source": "folio", "difficulty": 1 + 0.1 * i, "depth": 30,
              "correct": True, "cap_out": i < 8} for i in range(10)])
    c = cell_stats([r for r in recs if r["source"] == "prosqa"])
    assert c["n"] == 10 and c["acc"] == 0.5
    assert c["depth_all"]["mean"] == 9.5                      # 5..14
    assert c["depth_correct"]["n"] == 5 and c["depth_noncap"]["n"] == 10
    assert c["depth_clean"]["n"] == 5                         # 答对 ∩ 非撞顶
    assert c["flags"] == []
    c2 = cell_stats([r for r in recs if r["source"] == "folio"])
    assert c2["cap_out"] == 0.8 and "!cap" in c2["flags"]     # 撞顶该被标出来
    assert c2["depth_noncap"]["n"] == 2
    assert c2["depth_clean"]["n"] == 2                        # 全答对但 8/10 撞顶 -> 只剩 2

    # 阶梯 Sp 必须拒绝撞顶档: 造一个"全集看像完美阶梯、clean 样本却不够"的例子
    capped = ([{"source": "gsm8k", "difficulty": 0.5, "depth": 20,
                "correct": True, "cap_out": False} for _ in range(20)] +
              [{"source": "math500", "difficulty": 1.5, "depth": 93,
                "correct": True, "cap_out": True} for _ in range(20)] +
              [{"source": "aime", "difficulty": 2.5, "depth": 96,
                "correct": True, "cap_out": True} for _ in range(20)])
    r = build("math", [])["models"]
    import tempfile, os as _os
    fd, tmp = tempfile.mkstemp(suffix=".json"); _os.close(fd)
    json.dump({"model": "fake", "records": capped}, open(tmp, "w", encoding="utf-8"))
    row = build("math", [tmp])["models"][Path(tmp).stem]
    _os.unlink(tmp)
    assert row["_ladder_sp"] is None, "撞顶档不该产出阶梯 Sp(那正是撞顶假象)"
    assert "math500" in row["_ladder_why"] and "aime" in row["_ladder_why"]
    c3 = cell_stats([{"source": "x", "difficulty": 1, "depth": 40,
                      "correct": False, "cap_out": False} for _ in range(20)])
    assert "!acc" in c3["flags"]                              # acc=0 -> 空转标记
    assert cell_stats([]) is None

    # 猜测基线: FOLIO 三选一多数类 .353, 实测 acc .35 必须被标 !guess(绝对阈值 .05 抓不到)
    folio_guess = [{"source": "folio", "difficulty": 1.5, "depth": 15,
                    "correct": i < 14, "cap_out": False} for i in range(40)]   # acc=.35
    cg = cell_stats(folio_guess, source="folio")
    assert cg["acc"] == 0.35 and "!guess" in cg["flags"], cg
    assert "!acc" not in cg["flags"], "选择题不该走开放式的绝对阈值"
    # 真会做的 ProsQA(开放式, 基线 0) 不该被误标
    prosqa_ok = [{"source": "prosqa", "difficulty": 0.5, "depth": 7,
                  "correct": i < 38, "cap_out": False} for i in range(40)]     # acc=.95
    co = cell_stats(prosqa_ok, source="prosqa")
    assert co["flags"] == [], co
    # 全集 vs 非撞顶 必须能拉开(撞顶假象的核心症状)
    assert c2["depth_all"]["mean"] == c2["depth_noncap"]["mean"] == 30
    print("selftest OK: 三口径统计 + 撞顶/空转标记")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="logic", choices=list(DOMAINS))
    ap.add_argument("--extra", nargs="*", default=[], help="额外结果 json(如新跑的 ProofWriter)")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return

    files = sorted(glob.glob(str(CKPTS / DOMAINS[a.domain]["glob"]))) + list(a.extra)
    if not files:
        print(f"没找到结果文件: {CKPTS / DOMAINS[a.domain]['glob']}"); return
    res = build(a.domain, files)
    OUTDIR.mkdir(exist_ok=True)
    base = Path(a.out) if a.out else OUTDIR / f"depth_ladder_{a.domain}"
    json.dump(res, open(base.with_suffix(".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    md = to_md(res)
    open(base.with_suffix(".md"), "w", encoding="utf-8").write(md)
    print(md)
    print(f"[wrote] {base.with_suffix('.json')}\n[wrote] {base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
