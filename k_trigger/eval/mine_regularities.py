# -*- coding: utf-8 -*-
"""难度→深度 数据的四项规律挖掘（纯分析，不占 GPU，复用现有 dd_*.json）。

读 outputs/dd_{model}_{domain}.json（记录字段 source/difficulty/depth/correct/cap_out/gold/pred），
输出四项：
  (a) onset/阈值 K≥k*  —— 逐(模型,域)找深度随难度陡升的拐点（分位分档均值+最大增量档）。
  (b) 长度混淆检验     —— depth vs len(pred) 词数 的 Pearson；检验深度是否只是输出长度代理。
  (c) 跨模型深度一致性 —— 按 (source, gold) 对齐同一题，模型间 depth 的 Spearman 排序相关矩阵。
  (d) 错题过度思考泛化 —— depth ~ difficulty + is_wrong 多元回归，逐(模型,域)报 is_wrong 系数。

跑: PYTHONIOENCODING=utf-8 E:/conda_envs/drh/python.exe k_trigger/mine_regularities.py
   [--outdir k_trigger/outputs] [--json k_trigger/outputs/regularities.json]
"""
import argparse, json, glob, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

# 只纳入正式评测的规范文件（排除 pilot/变体，避免重复计数）
CANON = {
    "dd_r1_math", "dd_r1_logic", "dd_r1_commonsense",
    # 正确加载的自适应横向模型（论文级准确率，主证据）
    "dd_colargsm_gsmllm", "dd_colargsm_math", "dd_colargsm_logic", "dd_colargsm_commonsense",
    "dd_adalr_gsmllm", "dd_adalr_math", "dd_adalr_logic", "dd_adalr_commonsense",
    "dd_adalrby1_gsmllm", "dd_adalrby1_math", "dd_adalrby1_logic", "dd_adalrby1_commonsense",
    # CoLaR 用 RL+compression=1 最优配置（math/logic 的 _rl_c1/_rlc1；非后缀版是旧配置，勿用）
    "dd_colar_math_rl_c1", "dd_colar_logic_rlc1", "dd_colar_commonsense", "dd_colar_gsmllm",
    "dd_lwts_math", "dd_lwts_logic", "dd_lwts_commonsense", "dd_lwts_gsmllm",
    "dd_tah_math", "dd_tah_logic", "dd_tah_commonsense",
    "dd_huginn_math", "dd_huginn_logic", "dd_huginn_commonsense", "dd_huginn_gsmllm",
    "dd_ouro_math", "dd_ouro_logic", "dd_ouro_commonsense", "dd_ouro_gsmllm",
}


def load_records(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list):
        return d
    for v in d.values():                       # 找到记录列表字段
        if isinstance(v, list) and v and isinstance(v[0], dict) and "depth" in v[0]:
            return v
    return []


def valid(r, drop_cap=True):
    """有效记录：depth 非空非 NaN；drop_cap 时排除撞顶（截断的深度不可信）。"""
    dep = r.get("depth")
    if dep is None or (isinstance(dep, float) and np.isnan(dep)):
        return False
    if drop_cap and r.get("cap_out"):
        return False
    return True


def load_all(outdir):
    """returns {stem: [records]} for canonical files only."""
    out = {}
    for p in glob.glob(os.path.join(outdir, "dd_*.json")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in CANON:
            continue
        recs = load_records(p)
        if recs:
            out[stem] = recs
    return out


# ---------- (a) onset / 阈值 K≥k* ----------
def onset(records, nbins=5):
    """按 difficulty 分位分档，返回各档均深 + 深度陡升的拐点难度。"""
    rs = [r for r in records if valid(r)]
    if len(rs) < nbins * 2:
        return None
    diff = np.array([r["difficulty"] for r in rs], float)
    dep = np.array([r["depth"] for r in rs], float)
    # 分位边界（去重）
    qs = np.quantile(diff, np.linspace(0, 1, nbins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return None
    means, centers, ns = [], [], []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        m = (diff >= lo) & (diff <= hi if i == len(qs) - 2 else diff < hi)
        if m.sum() == 0:
            continue
        means.append(float(dep[m].mean()))
        centers.append(float(diff[m].mean()))
        ns.append(int(m.sum()))
    means, centers = np.array(means), np.array(centers)
    if len(means) < 3:
        return None
    deltas = np.diff(means)                      # 相邻档增量
    j = int(np.argmax(deltas))                   # 最大增量所在过渡
    base = means[0]
    rng = means.max() - means.min()
    return {
        "bin_centers": [round(c, 3) for c in centers],
        "bin_mean_depth": [round(m, 2) for m in means],
        "bin_n": ns,
        "onset_difficulty": round(float(centers[j + 1]), 3),   # 陡升后那档的难度中心
        "onset_jump": round(float(deltas[j]), 2),              # 该处深度跳变
        "range_ratio": round(float(means.max() / means.min()), 2) if means.min() > 0 else None,
        "total_rise": round(float(rng), 2),
    }


# ---------- (b) 长度混淆 ----------
def length_confound(records):
    """depth vs len(pred) 词数 的 Pearson；同时给 depth vs difficulty 以便对比。"""
    rs = [r for r in records if valid(r) and isinstance(r.get("pred"), str)]
    if len(rs) < 6:
        return None
    dep = np.array([r["depth"] for r in rs], float)
    plen = np.array([len(r["pred"].split()) for r in rs], float)
    diff = np.array([r["difficulty"] for r in rs], float)
    def sp(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(stats.pearsonr(a, b)[0])
    return {
        "n": len(rs),
        "pearson_depth_vs_predlen": round(sp(dep, plen), 3),
        "pearson_depth_vs_difficulty": round(sp(dep, diff), 3),
        "mean_predlen_words": round(float(plen.mean()), 1),
    }


# ---------- (c) 跨模型深度一致性 ----------
def model_of(stem):
    return stem.split("_")[1]          # dd_{model}_{domain}


def cross_model_agreement(all_recs):
    """按 (source, gold) 对齐同一题，算模型间 depth 的 Spearman。返回成对相关列表。"""
    # 每模型：合并其全部域，键=(source, gold)→depth（有效且唯一）
    per_model = defaultdict(dict)
    for stem, recs in all_recs.items():
        m = model_of(stem)
        for r in recs:
            if not valid(r):
                continue
            key = (str(r.get("source")), str(r.get("gold")))
            per_model[m].setdefault(key, r["depth"])   # 同题多模型域重复时取首个
    models = sorted(per_model)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            common = set(per_model[a]) & set(per_model[b])
            if len(common) < 8:
                continue
            va = [per_model[a][k] for k in common]
            vb = [per_model[b][k] for k in common]
            if np.std(va) == 0 or np.std(vb) == 0:
                continue
            rho = float(stats.spearmanr(va, vb)[0])
            pairs.append({"model_a": a, "model_b": b, "n_common": len(common),
                          "spearman_depth": round(rho, 3)})
    pairs.sort(key=lambda d: -d["n_common"])
    return {"models": models, "pairs": pairs}


# ---------- (d) 错题过度思考泛化 ----------
def overthinking(records):
    """depth ~ 1 + difficulty + is_wrong 多元 OLS；is_wrong 系数=控难度后答错多用的深度。"""
    rs = [r for r in records if valid(r) and r.get("correct") is not None]
    if len(rs) < 12:
        return None
    n_wrong = sum(1 for r in rs if not r["correct"])
    if n_wrong < 4 or (len(rs) - n_wrong) < 4:      # 需两类都有足够样本
        return {"note": f"skip (correct={len(rs)-n_wrong}, wrong={n_wrong})"}
    diff = np.array([r["difficulty"] for r in rs], float)
    isw = np.array([0.0 if r["correct"] else 1.0 for r in rs], float)
    dep = np.array([r["depth"] for r in rs], float)
    X = np.column_stack([np.ones(len(rs)), diff, isw])
    beta, *_ = np.linalg.lstsq(X, dep, rcond=None)
    dep_c = dep[isw == 0].mean()
    dep_w = dep[isw == 1].mean()
    return {
        "n": len(rs), "n_correct": len(rs) - n_wrong, "n_wrong": n_wrong,
        "is_wrong_coef": round(float(beta[2]), 2),      # 控难度后 错-对 深度差
        "difficulty_coef": round(float(beta[1]), 2),
        "mean_depth_correct": round(float(dep_c), 2),
        "mean_depth_wrong": round(float(dep_w), 2),
        "ratio_wrong_over_correct": round(float(dep_w / dep_c), 2) if dep_c > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="k_trigger/outputs")
    ap.add_argument("--json", default="k_trigger/outputs/regularities.json")
    args = ap.parse_args()

    all_recs = load_all(args.outdir)
    if not all_recs:
        print("no canonical dd_*.json found in", args.outdir); sys.exit(1)

    result = {"files": sorted(all_recs), "a_onset": {}, "b_length_confound": {},
              "d_overthinking": {}}
    for stem, recs in sorted(all_recs.items()):
        result["a_onset"][stem] = onset(recs)
        result["b_length_confound"][stem] = length_confound(recs)
        result["d_overthinking"][stem] = overthinking(recs)
    result["c_cross_model"] = cross_model_agreement(all_recs)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(result, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 可读打印 ----
    print("\n" + "=" * 70 + "\n(a) ONSET / 阈值 K≥k*  (分位分档均深 + 陡升拐点)\n" + "=" * 70)
    for s, v in result["a_onset"].items():
        if v:
            print(f"{s:26s} 档均深={v['bin_mean_depth']}  拐点难度={v['onset_difficulty']} "
                  f"跳变=+{v['onset_jump']}  幅度×{v['range_ratio']}")
    print("\n" + "=" * 70 + "\n(b) 长度混淆:  depth~len(pred) vs depth~difficulty\n" + "=" * 70)
    for s, v in result["b_length_confound"].items():
        if v:
            flag = "  <-- 疑长度代理" if abs(v["pearson_depth_vs_predlen"]) > abs(v["pearson_depth_vs_difficulty"]) + 0.15 else ""
            print(f"{s:26s} depth~预测长度={v['pearson_depth_vs_predlen']:+.2f}  "
                  f"depth~难度={v['pearson_depth_vs_difficulty']:+.2f}{flag}")
    print("\n" + "=" * 70 + "\n(c) 跨模型深度一致性 (同题 depth 的 Spearman)\n" + "=" * 70)
    for p in result["c_cross_model"]["pairs"]:
        print(f"{p['model_a']:8s} vs {p['model_b']:8s}  n={p['n_common']:3d}  Sp={p['spearman_depth']:+.2f}")
    print("\n" + "=" * 70 + "\n(d) 错题过度思考泛化 (控难度后 is_wrong 系数)\n" + "=" * 70)
    for s, v in result["d_overthinking"].items():
        if v and "is_wrong_coef" in v:
            print(f"{s:26s} 对={v['mean_depth_correct']:.1f} 错={v['mean_depth_wrong']:.1f} "
                  f"(×{v['ratio_wrong_over_correct']})  is_wrong系数(控难度)={v['is_wrong_coef']:+.2f}  "
                  f"[n对={v['n_correct']},n错={v['n_wrong']}]")
        elif v:
            print(f"{s:26s} {v.get('note')}")
    print(f"\n完整结果 → {args.json}")


def _selftest():
    # 构造：深度=2*难度+3*答错+噪声 → is_wrong 系数应≈3、depth~难度 正相关
    rng = np.random.default_rng(0)
    recs = []
    for i in range(60):
        d = float(rng.uniform(0, 3))
        wrong = i % 2
        dep = 2 * d + 3 * wrong + rng.normal(0, 0.2)
        recs.append({"source": "s", "difficulty": d, "depth": dep,
                     "correct": (wrong == 0), "cap_out": False,
                     "gold": str(i), "pred": "x " * int(dep)})
    ot = overthinking(recs)
    assert 2.4 < ot["is_wrong_coef"] < 3.6, ot
    on = onset(recs)
    assert on["bin_mean_depth"][-1] > on["bin_mean_depth"][0], on
    lc = length_confound(recs)
    assert lc["pearson_depth_vs_difficulty"] > 0.5, lc
    print("selftest OK:", ot["is_wrong_coef"], on["onset_difficulty"], lc["pearson_depth_vs_difficulty"])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
