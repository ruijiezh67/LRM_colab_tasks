# -*- coding: utf-8 -*-
# ═══ pool(cot) → 官方 CoLaR 训练格式 —— 任务3B（2026-08-01）════════════════════
#   官方 CoLaR run.py(qsa 数据集)吃 {source, question, steps:[...], answer} 列表,
#   放在 {WS}/datasets/text_reasoning/{NAME}/{train,val,test}.json。
#   本脚本把带 `cot` 的合成 pool(coding/physics)转成该格式: 把 cot 拆成 steps(逐推理步),
#   剥掉末句 "The final ... is X"。零 GPU。
#   用法:
#     python pool_to_colar_mix.py --selftest
#     python pool_to_colar_mix.py --pool E:/data/pools/coding_pool_train.json --out .../coding_mix/train.json
# ══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cot_to_steps(cot):
    """cot='A; B; C. The final ... is X.' -> ['A','B','C']。coding/physics 末句均以 '. The final' 起。"""
    head = cot.split(". The final")[0]
    return [s.strip() for s in head.split(";") if s.strip()]


def convert(pool):
    out = []
    for r in pool:
        if "cot" not in r:
            raise ValueError("pool 缺 cot 字段, 生成时须加 --emit_cot")
        out.append({"source": r.get("source", "synth"),
                    "question": r["question"],
                    "steps": cot_to_steps(r["cot"]),
                    "answer": str(r["answer"])})
    return out


def _selftest():
    # 覆盖 coding 与 physics 两种 cot 末句措辞
    pool = [
        {"source": "coding_symexec", "question": "q1", "answer": "135",
         "cot": "start: 1 3; append 5 => 1 3 5. The final sequence is 135."},
        {"source": "physics_relchain", "question": "q2", "answer": "64",
         "cot": "start = 8; d = v * t: 8 * 6 = 48; v = d / t: 48 / 3 = 16; "
                "d = v * t: 16 * 4 = 64. The final value is 64."},
    ]
    out = convert(pool)
    assert out[0]["steps"] == ["start: 1 3", "append 5 => 1 3 5"], out[0]["steps"]
    assert out[1]["steps"][0] == "start = 8" and out[1]["steps"][-1] == "d = v * t: 16 * 4 = 64"
    assert len(out[1]["steps"]) == 4
    assert all(set(r) == {"source", "question", "steps", "answer"} for r in out)
    assert all(r["steps"] for r in out), "空 steps"
    # 缺 cot 应报错
    try:
        convert([{"question": "x", "answer": "1"}]); assert False
    except ValueError:
        pass
    print("selftest OK: cot 拆 steps(coding/physics 两式) + 剥末句 + schema 正确 + 缺 cot 报错")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="带 cot 的 pool json(如 coding_pool_train.json)")
    ap.add_argument("--out", help="输出的官方格式 json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    assert a.pool and a.out, "非 --selftest 时须给 --pool 与 --out"
    pool = json.load(open(a.pool, encoding="utf-8"))
    out = convert(pool)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nstep = [len(r["steps"]) for r in out]
    print(f"[mix] {a.pool} -> {a.out}  n={len(out)}  steps/题 {min(nstep)}~{max(nstep)} 均{sum(nstep)/len(nstep):.1f}")


if __name__ == "__main__":
    main()
