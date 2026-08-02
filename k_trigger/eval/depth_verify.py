"""统一 prompt→深度验证：自适应 LRM 上"每题模型自选深度是否随 prompt/数据集变"。

块① 的核心证据：把 Huginn / LWtS 等**真自适应** LRM 当验证对象，量每题自选深度（answer-free），
跨 GSM8K + StrategyQA + ProsQA 看分布 + 可分性（两两 rank-AUC）+ 与难度相关。

adapter（各家自适应 API 不同）：
  huginn = forward_with_adaptive_compute(KL exit) → compute_steps，取 prompt 段平均深度
  lwts   = LatentThinkingModel.generate（输入尾 <START>）→ reasoning 段 token 数（自适应 latent 长度）
（MoR 的 HF-360m 是空壳=固定 Llama，已删除，不用。）

数据集：coconut/data/gsm_valid.json · coconut/data/prosqa_valid.json · E:/data/strategyqa/train.json（均有 question）。
复用 l1a 的 auc_separability/spearman。

用法：PYTHONIOENCODING=utf-8 E:/conda_envs/drh/python.exe k_trigger/depth_verify.py --model huginn --n 120
自检：python k_trigger/depth_verify.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))                      # l1a helpers

DATASETS = {
    "gsm": REPO / "coconut/data/gsm_valid.json",
    "prosqa": REPO / "coconut/data/prosqa_valid.json",
    "strategyqa": Path("E:/data/strategyqa/train.json"),
}


def load_questions(name, n):
    d = json.loads(Path(DATASETS[name]).read_text())
    items = [x for x in d if x.get("question")][: n * 6]   # 多取候选，过滤长题后仍够 n
    out = []
    for it in items:
        st = it.get("steps")
        # 难度代理：有 steps 用步数（GSM8K/ProsQA），否则词数（StrategyQA）——注意跨数据集不可比，仅数据集内看
        diff = len(st) if isinstance(st, list) else len(it["question"].split())
        out.append((it["question"], diff))
    return out


def _selftest():
    from l1a_coconut_commit import auc_separability, spearman
    assert auc_separability([1, 2, 3], [7, 8, 9]) == 1.0
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6
    assert set(DATASETS) == {"gsm", "prosqa", "strategyqa"}
    print("selftest OK: depth_verify helpers + datasets")


# ---- Huginn adapter ----
def depths_huginn(n, cap, exit_threshold, max_prompt_tokens):
    import torch, importlib, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    MODEL = "E:/ckpts/huginn-0125"
    tok = AutoTokenizer.from_pretrained(MODEL)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map={"": 0},
                                             trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    raven = importlib.import_module(m.__class__.__module__)
    m.config.mean_recurrence = cap
    dev = next(m.parameters()).device

    def one(question):
        ids = tok.encode(question.rstrip() + "\n", return_tensors="pt")
        if ids.shape[1] > max_prompt_tokens:
            return None                      # 过滤长题（不截断——截断=在半个问题上测深度，错误）
        ids = ids.to(dev)
        ev = raven.get_adaptive_exit_evaluator(m, "kl", exit_threshold)
        with torch.no_grad():
            out = m.forward_with_adaptive_compute(ids, exit_evaluator=ev)
        cs = out.stats["compute_steps"]
        return float(sum(cs) / len(cs))
    return one


# ---- LWtS adapter（LatentThinkingModel.generate → reasoning 段长度）----
def depths_lwts(n, max_new_tokens, answer_prefix):
    import torch
    sys.path.insert(0, "E:/ckpts/lwts-repo")
    from src.model_creation import automodelforcausallm_from_pretrained_latent
    from src.modeling_utils import START_TOKEN_STR
    from transformers import AutoTokenizer, GenerationConfig
    P = "E:/ckpts/lwts-latent6rl"
    tok = AutoTokenizer.from_pretrained(P)
    m = automodelforcausallm_from_pretrained_latent(P).eval().to("cuda")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    start_id = tok.convert_tokens_to_ids(START_TOKEN_STR)
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    def one(question):
        ids = tok(question.rstrip() + "\n", return_tensors="pt").input_ids
        ids = torch.cat([ids, torch.tensor([[start_id]])], dim=1).to("cuda")
        with torch.no_grad():
            out = m.generate(ids, generation_config=gc)
        gen = out[0, ids.shape[1]:]
        txt = tok.decode(gen, skip_special_tokens=False)
        # reasoning 段 = answer_prefix 之前的 token 数（=自适应 latent 长度代理）
        low = txt.lower(); idx = low.find(answer_prefix.lower())
        if idx < 0:
            return float(len(gen))
        pre = tok(txt[:idx], add_special_tokens=False).input_ids
        return float(len(pre) + 1)
    return one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", choices=["huginn", "lwts"], default="huginn")
    ap.add_argument("--n", type=int, default=120, help="每数据集题数（Huginn 慢，>500 需数小时）")
    ap.add_argument("--cap", type=int, default=64, help="抬高使其非绑定约束，让深度不被截断")
    ap.add_argument("--exit_threshold", type=float, default=5e-4,
                    help="Huginn 原生默认退出判据（model card 值），**固定不调**——调它=人为拧深度，抬 cap 就失去意义")
    ap.add_argument("--max_prompt_tokens", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--answer_prefix", default="answer")
    ap.add_argument("--datasets", default="gsm,prosqa,strategyqa", help="逗号分隔，可只跑部分加速")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    import numpy as np, time
    from l1a_coconut_commit import auc_separability, spearman
    if args.model == "huginn":
        depth_fn = depths_huginn(args.n, args.cap, args.exit_threshold, args.max_prompt_tokens)
    else:
        depth_fn = depths_lwts(args.n, args.max_new_tokens, args.answer_prefix)
    print(f"model={args.model} loaded", flush=True)

    want = [d.strip() for d in args.datasets.split(",") if d.strip()]
    per = {}
    for ds in DATASETS:
        if ds not in want:
            continue
        if not Path(DATASETS[ds]).exists():
            print(f"skip {ds} (missing)", flush=True); continue
        qs = load_questions(ds, args.n)
        depths, diffs = [], []
        t0 = time.time()
        for q, diff in qs:
            if len(depths) >= args.n:
                break
            dv = depth_fn(q)
            if dv is None:                     # 长题被过滤
                continue
            depths.append(dv); diffs.append(diff)
            if len(depths) % 25 == 0:
                print(f"  [{ds}] {len(depths)}/{args.n}  mean_depth={np.mean(depths):.1f} ({time.time()-t0:.0f}s)", flush=True)
        if not depths:                          # 全被过滤/空集：跳过，勿 np.min([]) 崩
            print(f"[{ds}] 0 valid prompts (all filtered) — skipped", flush=True); continue
        per[ds] = {"depths": depths, "diffs": diffs}
        print(f"[{ds}] n={len(depths)} depth mean={np.mean(depths):.2f} std={np.std(depths):.2f} "
              f"range={int(np.min(depths))}-{int(np.max(depths))} sp_diff={spearman(depths, diffs):.3f}", flush=True)

    res = {"experiment": f"depth_verify ({args.model}, prompt->depth across datasets)",
           "model": args.model, "cap": args.cap, "exit_threshold": args.exit_threshold,
           "per_dataset": {ds: {"n": len(per[ds]["depths"]),
                                "mean": round(float(np.mean(per[ds]["depths"])), 2),
                                "std": round(float(np.std(per[ds]["depths"])), 2),
                                "min": int(np.min(per[ds]["depths"])), "max": int(np.max(per[ds]["depths"])),
                                "spearman_vs_difficulty": spearman(per[ds]["depths"], per[ds]["diffs"])}
                           for ds in per},
           "cross_dataset_auc": {}}
    names = list(per)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res["cross_dataset_auc"][f"{names[i]}_vs_{names[j]}"] = auc_separability(
                per[names[i]]["depths"], per[names[j]]["depths"])
    best_auc = max(res["cross_dataset_auc"].values()) if res["cross_dataset_auc"] else 0
    best_std = max((res["per_dataset"][ds]["std"] for ds in per), default=0)
    res["verdict"] = (f"{args.model}: 深度随 prompt/数据集变（best 跨集 AUC={best_auc}, max std={best_std}）"
                      + ("→ prompt→深度成立。" if (best_auc > 0.7 or best_std >= 2) else "→ 变化弱，需调 cap/阈值/题集。"))
    print("verdict:", res["verdict"], flush=True)
    out = Path(args.output) if args.output else HERE / "outputs" / f"depth_verify_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
