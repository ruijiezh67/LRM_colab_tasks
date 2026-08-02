r"""难度 → 自适应潜在深度 测试(给 coworker 训好的 CoLaR ckpt 用)。

做什么: 在一个「按难度分档」的题池(QSA test.json, 每题带 difficulty 字段)上,
  对每题: 用 CoLaR 潜在生成得到 (答案, 潜在链长 n_latent_forward), 判对错;
  然后分 correct / incorrect / all 三子集, 各自看 深度~难度 的
  Spearman(单调) + Pearson/OLS 斜率(线性) + 三档 tercile 均值深度。

这是本课题的核心读数: **correct 子集 Spearman > 0 = 该域(逻辑/常识)也「越难越深」** ——
即非数学域的自适应深度也随难度增长(后门攻击链 Link1: prompt→深度 的通用性证据)。

本脚本自包含: 只依赖 torch/transformers/peft/numpy(CoLaR 训练环境已装), 不需要我们的 k_trigger 仓库。
潜在生成循环 = CoLaR 官方 latent_generate 的等价移植(与本课题所有横向模型同一套测量)。

------------------------------------------------------------------------------
用法
------------------------------------------------------------------------------
1) 跑 + 存记录:
   COLAR_BASE=/path/to/Llama-3.2-1B-Instruct \
   COLAR_CKPT=/path/to/your_trained_colar.ckpt \
   COLAR_EMB_STD=0.018 \                # Llama-3.2-1B=0.018 ; R1-1.5B=0.03 (务必对齐训练基座)
   python run_difficulty_depth_colar.py \
       --pool <workspace>/datasets/text_reasoning/prosqa/test.json \
       --out  dd_colar_prosqa.json --n 300

2) 只重新分析(不占 GPU):
   python run_difficulty_depth_colar.py --analyze dd_colar_prosqa.json

3) 自检:
   python run_difficulty_depth_colar.py --selftest
------------------------------------------------------------------------------
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict


# ============================ 分析(与本课题主脚本一致) ============================
def _np():
    import numpy as np
    return np


def spearman(x, y):
    np = _np()
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    rx = x.argsort().argsort(); ry = y.argsort().argsort()
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def pearson_slope(x, y):
    """线性: Pearson r + OLS 斜率(每升 1 单位难度, 深度增量)。"""
    np = _np()
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0, 0.0
    r = float(np.corrcoef(x, y)[0, 1]); slope = float(np.polyfit(x, y, 1)[0])
    return round(r, 3), round(slope, 3)


def terciles(vals):
    s = sorted(vals)
    if len(s) < 3:
        return None
    return s[len(s) // 3], s[2 * len(s) // 3]


def bin_tercile(v, cuts):
    q1, q2 = cuts
    return 0 if v < q1 else (1 if v < q2 else 2)


def _subset_depth(subset):
    np = _np()
    if len(subset) < 6:
        return {"n": len(subset), "note": "n<6 无法分档"}
    diffs = [r["difficulty"] for r in subset]; depths = [r["depth"] for r in subset]
    cuts = terciles(diffs); tb = defaultdict(list)
    for r in subset:
        tb[bin_tercile(r["difficulty"], cuts)].append(r["depth"])
    terc = {f"T{t}": {"n": len(tb[t]), "mean_depth": round(float(np.mean(tb[t])), 2)} for t in sorted(tb)}
    means = [terc[f"T{t}"]["mean_depth"] for t in sorted(tb)]
    pr, sl = pearson_slope(diffs, depths)
    return {"n": len(subset), "terciles": terc,
            "monotonic_up": all(means[i] <= means[i + 1] for i in range(len(means) - 1)),
            "spearman_depth_vs_difficulty": spearman(diffs, depths),
            "pearson_depth_vs_difficulty": pr, "ols_slope": sl}


def analyze(records):
    """撞顶(cap_out)深度不真实→剔除; 分 correct/incorrect/all 三子集各看 难度→深度。"""
    clean = [r for r in records if not r.get("cap_out")]
    correct = [r for r in clean if r["correct"]]
    incorrect = [r for r in clean if not r["correct"]]
    out = {"n_total": len(records), "n_correct": sum(r["correct"] for r in records),
           "acc": round(sum(r["correct"] for r in records) / max(len(records), 1), 3),
           "cap_out_rate": round(sum(r.get("cap_out", False) for r in records) / max(len(records), 1), 3),
           "depth_min": None, "depth_max": None, "by_source_acc": {}}
    dep = [r["depth"] for r in clean]
    if dep:
        out["depth_min"], out["depth_max"] = round(min(dep), 1), round(max(dep), 1)
    tot = Counter(r.get("source", "all") for r in records)
    cor = Counter(r.get("source", "all") for r in records if r["correct"])
    for s in tot:
        out["by_source_acc"][s] = {"n": tot[s], "correct": cor[s], "acc": round(cor[s] / tot[s], 3)}
    out["correct_subset"] = _subset_depth(correct)
    out["incorrect_subset"] = _subset_depth(incorrect)
    out["all_subset"] = _subset_depth(clean)
    return out


# ============================ 判分(逻辑/常识: 归一化包含) ============================
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


_BOOL_TRUE = ("yes", "true")
_BOOL_FALSE = ("no", "false")
_BOOL_ALL = _BOOL_TRUE + _BOOL_FALSE


def grade(pred, gold):
    """gold 出现在 pred 里即算对。布尔题用词级匹配(取 pred 里第一个布尔词)避免子串误伤。

    布尔题的 yes/no 与 true/false 视为同义: 多跳蕴含/ProofWriter 一类数据集 gold 存 "yes"/"no",
    而模型答 "True"/"False" —— 只认 yes/no 会把全部答对判成错(实测 dd_multihop_logic acc 0.000
    -> 归一后 0.586)。与 eval/run_difficulty_depth.py 的 grade_bool 语义对齐。
    """
    g = " ".join(_norm(gold)); p = _norm(pred); ps = " ".join(p)
    if not g:
        return False
    if g in _BOOL_ALL:
        want_true = g in _BOOL_TRUE
        for t in p:
            if t in _BOOL_ALL:
                return (t in _BOOL_TRUE) == want_true
        return False
    return g in ps or (len(p) <= 4 and ps in g)   # 短答案双向包含


# ============================ CoLaR 潜在生成(自包含移植) ============================
def run_colar(items):
    """base + LoRA(q/v,r128) + latent_policy(MLP)。深度 = n_latent_forward(自适应, 遇 ### 停, cap=MAX_LAT)。
    与 CoLaR 官方 latent_generate 等价; ckpt state_dict 顶层键为 llm.* / latent_policy.*(官方 LitCoLaR 存法)。"""
    import torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    BASE = os.environ["COLAR_BASE"]
    CK = os.environ["COLAR_CKPT"]
    EMB_STD = float(os.environ.get("COLAR_EMB_STD", "0.018"))   # Llama-3.2-1B=0.018 / R1-1.5B=0.03
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5"))       # 与训练 compression_factor 对齐
    MAX_LAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    ANSTOK = int(os.environ.get("COLAR_ANSTOK", "16"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[colar cfg] base={os.path.basename(BASE)} ckpt={os.path.basename(CK)} "
          f"emb_std={EMB_STD} compress={COMPRESS} max_lat={MAX_LAT}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    llm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    llm.resize_token_embeddings(len(tok))
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    H = llm.config.hidden_size

    class LatentPolicy(nn.Module):
        def __init__(s, f, inter=2048):
            super().__init__()
            s.fc = nn.Sequential(nn.Linear(f, inter), nn.GELU(), nn.Linear(inter, inter), nn.LayerNorm(inter))
            s.mean = nn.Linear(inter, f); s.log_std = nn.Linear(inter, f)
        def forward(s, x, temperature=1.0):
            x = s.fc(x); return torch.distributions.Normal(s.mean(x), s.log_std(x).exp() * temperature)

    lp = LatentPolicy(H, 2048)
    cont = nn.Module(); cont.llm = llm; cont.latent_policy = lp
    sd = torch.load(CK, map_location="cpu")["state_dict"]
    missing, unexpected = cont.load_state_dict(sd, strict=False)
    bad = [k for k in unexpected if "latent_policy" in k or "lora" in k.lower()]
    assert not bad, f"ckpt 有多余 latent_policy/lora 键: {bad[:5]}\n" \
                    "→ 确认 COLAR_CKPT 是 CoLaR(LitCoLaR)训出的 ckpt, 且基座 COLAR_BASE 与训练一致。"
    # 关键: 这些权重若落在 missing = 没载入、留随机初始化 → 深度无意义 (静默失败). 硬断言拦截.
    not_loaded = [k for k in missing if "latent_policy" in k or "lora" in k.lower()]
    assert not not_loaded, f"latent_policy/lora 权重未载入(在 missing 里): {not_loaded[:5]}\n" \
                           "→ ckpt 键名与本移植不一致 → 深度会是随机初始化的假值。检查 ckpt 是否官方 LitCoLaR 存法。"
    llm = llm.to(dev).eval(); lp = lp.to(dev).float().eval()
    emb = llm.get_input_embeddings()
    sep_id = tok.convert_tokens_to_ids("###")
    QT, SPEED = "Question: {} Let's think step by step:", "(Thinking speed: {})"
    ANS = dict(max_new_tokens=ANSTOK, do_sample=False, pad_token_id=tok.pad_token_id)  # 贪心=可复现
    print(f"colar loaded (sep_id={sep_id})", flush=True)

    def gen_one(question):
        text = QT.format(str(question).rstrip()) + SPEED.format(COMPRESS) + "###"
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        am = torch.ones_like(ids)
        pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        with torch.no_grad():
            qemb = emb(ids)
            out = llm(inputs_embeds=qemb, attention_mask=am, position_ids=pos,
                      output_hidden_states=True, use_cache=True)
            pkv = out.past_key_values; all_emb = [qemb]; cur_pos = pos[:, -1:]; n_lat = 0
            for _ in range(MAX_LAT):
                dist = lp(out.hidden_states[-1][:, -1:, :].float())
                ce = (dist.rsample() * EMB_STD).to(qemb.dtype)
                all_emb.append(ce); am = torch.cat([am, torch.ones(1, 1, device=dev, dtype=am.dtype)], 1)
                cur_pos = cur_pos + 1; n_lat += 1
                out = llm(inputs_embeds=ce, attention_mask=am, position_ids=cur_pos,
                          past_key_values=pkv, output_hidden_states=True, use_cache=True)
                pkv = out.past_key_values
                nxt = torch.multinomial(torch.softmax(out.logits[:, -1].float(), -1), 1)
                if int(nxt) == sep_id:
                    break
            cap_out = n_lat >= MAX_LAT
            sep_emb = emb(torch.tensor([[sep_id]], device=dev))
            all_emb.append(sep_emb); am = torch.cat([am, torch.ones(1, 1, device=dev, dtype=am.dtype)], 1)
            full = torch.cat(all_emb, 1)
            pred = llm.generate(inputs_embeds=full, attention_mask=am, **ANS)
        txt = tok.decode(pred[0], skip_special_tokens=True)
        ans = txt.strip("#").split("Answer:")[-1]
        return ans, float(n_lat), cap_out

    for it in items:
        yield gen_one(it["question"])


# ============================ 题池载入 ============================
def load_pool(path, n):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or list(data.values())[0]
    items = []
    for ex in data:
        d = ex.get("difficulty")
        if d is None:            # 兜底: 用 steps 长度
            d = len(ex.get("steps") or [])
        items.append({"question": ex["question"], "gold": ex.get("answer", ""),
                      "difficulty": float(d), "source": ex.get("source", os.path.basename(os.path.dirname(path)))})
    if n and n < len(items):
        # 均匀按难度采样(别只取到某一档)
        items.sort(key=lambda x: x["difficulty"])
        step = len(items) / n
        items = [items[int(i * step)] for i in range(n)]
    return items


# ============================ 自检 ============================
def _selftest():
    assert grade("the answer is yes.", "yes") and not grade("no way", "yes")
    assert grade("Therefore Tom is a lempus.", "Tom is a lempus")
    # 布尔同义: 模型答 True/False, gold 存 yes/no (多跳蕴含 / ProofWriter)
    assert grade("True", "yes") and grade("False.", "no")
    assert not grade("True", "no") and not grade("False", "yes")
    assert grade("yes", "true") and not grade("no", "true")      # 反向 gold 也认
    assert not grade("Tom is a wumpus.", "yes")                  # 无布尔词 -> 判错, 不误判
    r = [{"source": "s", "difficulty": i / 10, "depth": i, "correct": i % 2 == 0, "cap_out": False}
         for i in range(12)]
    a = analyze(r)
    assert a["correct_subset"]["n"] == 6 and "terciles" in a["correct_subset"]
    assert a["incorrect_subset"]["n"] == 6
    print("selftest OK: grade + 三子集分析")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="QSA test.json(含 difficulty 字段)")
    ap.add_argument("--out", help="记录输出 json")
    ap.add_argument("--n", type=int, default=0, help="题数上限(0=全部; >0 按难度均匀采样)")
    ap.add_argument("--analyze", help="只分析已有记录 json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    if args.analyze:
        with open(args.analyze, encoding="utf-8") as f:
            print(json.dumps(analyze(json.load(f)), ensure_ascii=False, indent=2)); return

    assert args.pool and args.out, "需 --pool 和 --out(或用 --analyze / --selftest)"
    items = load_pool(args.pool, args.n)
    print(f"载入 {len(items)} 题, 难度范围 "
          f"[{min(i['difficulty'] for i in items)}, {max(i['difficulty'] for i in items)}]", flush=True)
    records = []
    for it, (pred, depth, cap_out) in zip(items, run_colar(items)):
        correct = bool(grade(pred, it["gold"]))
        records.append({"source": it["source"], "difficulty": it["difficulty"], "depth": depth,
                        "correct": correct, "cap_out": bool(cap_out),
                        "gold": str(it["gold"]), "pred": pred[:120]})
        if len(records) % 25 == 0:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  {len(records)}/{len(items)}  acc={acc:.2f}", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:   # 增量存盘(防中断丢失)
            json.dump(records, f, ensure_ascii=False)
    print(json.dumps(analyze(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
