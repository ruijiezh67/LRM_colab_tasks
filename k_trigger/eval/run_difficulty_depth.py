# ═══ 文件夹 eval/ = 深度-难度 评测&分析流水线 ═══════════════════════════════
#   用途: 各潜在模型"难度→自适应深度"评测(本文件/eval_synth_grid) + 数据prep(prep_difficulty_sets)
#         + 结果分析(mine_regularities/gen_comprehensive/gen_verdict).
#   题目池→ question_sets/ 或 E:/data/pools/.  产出→ outputs/dd_*.json.
#   日期: 2026-07 更新   进度: 主结论已出(controlled-step Sp+0.97); 待办=换域(ProofWriter/Science)重验.
# ══════════════════════════════════════════════════════════════════════════
"""难度→自适应深度:每模型跑难度池 → 生成答案判对错 + 量深度 → 只在答对子集分 tercile。

指标(每模型自己看,跨模型不可比):
  huginn = 答案-token 处自适应循环步数(latent pass);  lwts = reasoning-token 数;  r1 = <think> 段 token 数。
分档 = 模型内、只在答对题上按连续难度 difficulty 分 3 档(terciles) + 连续 Spearman。

两步:
  跑:  PYTHONIOENCODING=utf-8 E:/conda_envs/drh/python.exe k_trigger/run_difficulty_depth.py --model lwts --domain math --n_per_source 60
  分析:python k_trigger/run_difficulty_depth.py --analyze k_trigger/outputs/dd_lwts_math.json
自检:python k_trigger/run_difficulty_depth.py --selftest
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
CKPTS = os.environ.get("CKPTS_ROOT", "E:/ckpts")            # Colab: 设 CKPTS_ROOT=/content/ckpts
POOLS = Path(os.environ.get("DD_POOLS", "E:/data/pools"))   # Colab: 设 DD_POOLS=/content/pools


# ================= 纯 python:判分 + 分析(可自检) =================

def _last_int(s):
    m = re.findall(r"-?\d[\d,]*", s.replace(",", ""))
    return m[-1] if m else None


def grade_int(pred, gold):
    p = _last_int(pred or "")
    try:
        return p is not None and int(p) == int(str(gold).replace(",", ""))
    except ValueError:
        return False


def _norm_math(s):
    s = s or ""
    for a in ["\\left", "\\right", "\\!", "\\,", "\\;", " ", "$", "\\\\"]:
        s = s.replace(a, "")
    return s.strip().rstrip(".")


def _to_sympy(s):
    """latex/文本答案 → sympy 表达式（尽力）。失败返回 None。"""
    if s is None:
        return None
    s = str(s).strip()
    m = re.search(r"\\boxed\{(.+)\}", s)
    if m:
        s = m.group(1)
    for a in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\displaystyle", "\\text", "$", "^{\\circ}", "^\\circ", "\\circ", "\\%", "%", "\\ "]:
        s = s.replace(a, "")
    s = s.strip().rstrip(".").strip()
    if not s:
        return None
    # latex → python-ish（parse_latex 依赖特定 antlr 版本,不用；改隐式乘法解析）
    t = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    t = re.sub(r"\\d?frac\s*(\d)\s*(\d)", r"((\1)/(\2))", t)   # \frac43
    for a, b in [("\\sqrt", "sqrt"), ("\\pi", "pi"), ("\\cdot", "*"), ("\\times", "*"), ("\\div", "/")]:
        t = t.replace(a, b)
    t = t.replace("^", "**").replace("{", "(").replace("}", ")")
    try:
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                 implicit_multiplication_application)
        return parse_expr(t, transformations=standard_transformations + (implicit_multiplication_application,))
    except Exception:
        return None


def _expr_eq(a, b):
    """单值等价:数值容差 或 符号 simplify。"""
    ea, eb = _to_sympy(a), _to_sympy(b)
    if ea is None or eb is None:
        return _norm_math(a) == _norm_math(b)
    try:
        fa, fb = float(ea.evalf()), float(eb.evalf())
        if abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb)):
            return True
    except Exception:
        pass
    try:
        from sympy import simplify
        return simplify(ea - eb) == 0
    except Exception:
        return False


def _split_vals(s):
    s = re.sub(r"^\s*[\(\[\{]|[\)\]\}]\s*$", "", (s or "").strip())   # 去外层括号
    return [p.strip() for p in s.split(",") if p.strip()]


def grade_math_expr(pred, gold):
    pred = pred or ""
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", pred)
    cand = m.group(1) if m else pred.split("=")[-1]
    if _norm_math(cand) == _norm_math(gold):       # 便宜:归一后字符串相等
        return True
    gp, cp = _split_vals(gold), _split_vals(cand)  # 多值(坐标/集合)逐分量比
    if len(gp) > 1 or len(cp) > 1:
        if len(gp) != len(cp) or not gp:
            return False
        try:
            key = lambda x: float(_to_sympy(x).evalf())
            return all(_expr_eq(a, b) for a, b in zip(sorted(gp, key=key), sorted(cp, key=key)))
        except Exception:
            return all(_expr_eq(a, b) for a, b in zip(gp, cp))
    return _expr_eq(cand, gold)                    # 单值:sympy 数值/符号等价


def grade_mc_label(pred, gold):
    pred = (pred or "").lower()
    hits = [w for w in ["uncertain", "true", "false"] if w in pred]  # uncertain 优先(含 true? 否)
    return bool(hits) and hits[0] == str(gold).lower() if hits else False


def grade_mc_index(pred, gold_idx, options):
    pred = pred or ""
    m = re.findall(r"\b([A-E])\b", pred.upper())    # A-E(CSQA 5选;LogiQA 4选)
    if m:
        return (ord(m[-1]) - 65) == int(gold_idx)
    # 退化:选项文本匹配
    for i, opt in enumerate(options or []):
        if opt and opt.lower() in pred.lower():
            return i == int(gold_idx)
    return False


def grade_entity(pred, gold):
    # ProsQA gold 形如 "Tom is a lempus." → 关键类名=末词
    key = re.sub(r"[^a-zA-Z]", "", (gold or "").split()[-1]) if gold else ""
    return bool(key) and key.lower() in (pred or "").lower()


def grade_bool(pred, gold):
    # StrategyQA: yes/no。取 pred 里最后出现的 true/false/yes/no 与 gold 比。
    g = str(gold).strip().lower()
    want_true = g in ("true", "yes", "1")
    hits = re.findall(r"\b(true|false|yes|no)\b", (pred or "").lower())
    if not hits:
        return False
    return (hits[-1] in ("true", "yes")) == want_true


GRADERS = {"int": grade_int, "math_expr": grade_math_expr, "mc_label": grade_mc_label,
           "mc_index": grade_mc_index, "entity": grade_entity, "bool": grade_bool}


def grade(item, pred):
    k = item["gold_kind"]
    if k == "mc_index":
        return grade_mc_index(pred, item["gold"], item.get("options"))
    return GRADERS[k](pred, item["gold"])


def terciles(vals):
    """把连续值切 3 档,返回每档的下界(用 33/66 分位)。"""
    s = sorted(vals)
    if len(s) < 3:
        return None
    q1 = s[len(s) // 3]; q2 = s[2 * len(s) // 3]
    return q1, q2


def bin_tercile(v, cuts):
    q1, q2 = cuts
    return 0 if v < q1 else (1 if v < q2 else 2)


def spearman(x, y):
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    rx = x.argsort().argsort(); ry = y.argsort().argsort()
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def pearson_slope(x, y):
    """线性:Pearson r + OLS 斜率(每升 1 单位难度,深度增量)。用于'成正比'的线性读数。"""
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0, 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    return round(r, 3), round(slope, 3)


def _subset_depth(subset):
    """对一个子集算 3 档 tercile 均值深度 + 单调 + Spearman(单调) + Pearson/斜率(线性)。"""
    import numpy as np
    from collections import defaultdict
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
    """records: {source,difficulty,depth,correct,cap_out}. 撞顶(cap_out)深度被删剪→全程剔除。
    分 correct/incorrect/all 三子集各自看 难度→深度（验证 Path A: 答错时还有没有规律）。"""
    from collections import Counter
    clean = [r for r in records if not r.get("cap_out")]          # 非撞顶=深度真实
    correct = [r for r in clean if r["correct"]]
    incorrect = [r for r in clean if not r["correct"]]
    out = {"n_total": len(records), "n_correct": sum(r["correct"] for r in records),
           "acc": round(sum(r["correct"] for r in records) / max(len(records), 1), 3),
           "cap_out_rate": round(sum(r.get("cap_out", False) for r in records) / max(len(records), 1), 3),
           "by_source_acc": {}}
    tot = Counter(r["source"] for r in records); cor = Counter(r["source"] for r in records if r["correct"])
    for s in tot:
        out["by_source_acc"][s] = {"n": tot[s], "correct": cor[s], "acc": round(cor[s] / tot[s], 3)}
    out["correct_subset"] = _subset_depth(correct)      # 答对(干净读数)
    out["incorrect_subset"] = _subset_depth(incorrect)  # 答错(Path A: 还跟难度吗)
    out["all_subset"] = _subset_depth(clean)            # 全部(非撞顶)
    return out


def _selftest():
    assert grade_int("the answer is 42.", "42") and not grade_int("41", "42")
    assert grade_math_expr("so \\boxed{33}", "33") and grade_math_expr("= 3", "3")
    assert grade_math_expr("\\frac{4}{3}", "\\frac43")        # 同值不同写法(之前误判)
    assert grade_math_expr("25.132741", "8\\pi")             # 数值等价
    assert not grade_math_expr("4\\pi", "8\\pi")             # 真不等
    assert grade_math_expr("(2, 4)", "(2,4)") and not grade_math_expr("(2,2)", "(2,4)")  # 坐标
    assert grade_mc_label("I think it is Uncertain.", "Uncertain")
    assert not grade_mc_label("it is True", "False")
    assert grade_mc_index("answer: C", 2, ["a", "b", "c", "d"])
    assert grade_entity("Therefore Tom is a lempus.", "Tom is a lempus.")
    assert bin_tercile(0.1, (0.5, 1.5)) == 0 and bin_tercile(2.0, (0.5, 1.5)) == 2
    r = [{"source": "gsm8k", "difficulty": i / 10, "depth": i, "correct": i % 2 == 0, "cap_out": False} for i in range(12)]
    a = analyze(r); assert a["correct_subset"]["n"] == 6 and "terciles" in a["correct_subset"] and a["incorrect_subset"]["n"] == 6
    print("selftest OK: graders + correct/incorrect/all analysis")


# ================= 采样池 =================
def sample_pool(domain, n_per_source, seed=0, sources=None):
    """sources=None 取全部源; 传 list 只取指定源(如只跑新加的 proofwriter 做探针, 省 3/4 算力)。"""
    import numpy as np
    pool = json.loads((POOLS / f"{domain}_pool.json").read_text(encoding="utf-8"))
    from collections import defaultdict
    by = defaultdict(list)
    for x in pool:
        by[x["source"]].append(x)
    if sources:
        missing = [s for s in sources if s not in by]
        assert not missing, f"池里没有这些源: {missing} (池内有: {sorted(by)})"
        by = {s: by[s] for s in sources}
    rng = np.random.default_rng(seed)
    out = []
    for s, items in by.items():
        idx = rng.permutation(len(items))[:n_per_source]
        out += [items[i] for i in idx]
    return out


# ================= 适配器:LWtS =================
def run_lwts(items, max_new_tokens=120):
    import torch
    sys.path.insert(0, f"{CKPTS}/lwts-repo")
    from src.model_creation import automodelforcausallm_from_pretrained_latent
    from src.modeling_utils import START_TOKEN_STR
    from transformers import AutoTokenizer, GenerationConfig
    P = f"{CKPTS}/lwts-latent6rl"
    tok = AutoTokenizer.from_pretrained(P)
    m = automodelforcausallm_from_pretrained_latent(P).eval().to("cuda")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    start_id = tok.convert_tokens_to_ids(START_TOKEN_STR)
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    print("lwts loaded", flush=True)
    for it in items:
        ids = tok(it["question"].rstrip() + "\n", return_tensors="pt").input_ids
        ids = torch.cat([ids, torch.tensor([[start_id]])], dim=1).to("cuda")
        with torch.no_grad():
            out = m.generate(ids, generation_config=gc)
        gen = out[0, ids.shape[1]:]
        txt = tok.decode(gen, skip_special_tokens=False)
        cap_out = len(gen) >= max_new_tokens
        low = txt.lower(); idx = low.find("answer")
        if idx < 0:
            depth = float(len(gen)); pred = txt
        else:
            pre = tok(txt[:idx], add_special_tokens=False).input_ids
            depth = float(len(pre) + 1); pred = txt[idx:]
        yield pred, depth, cap_out


# ================= 适配器:R1-Distill(CoT think-token 数) =================
def run_r1(items, max_new_tokens=4096):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    P = f"{CKPTS}/r1-distill-1.5b"
    tok = AutoTokenizer.from_pretrained(P)
    m = AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    print("r1 loaded", flush=True)
    for it in items:
        pred, depth, cap_out = "", float("nan"), True
        try:
            msg = [{"role": "user", "content": it["question"].strip() +
                    "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = out[0, ids.shape[1]:]
            txt = tok.decode(gen, skip_special_tokens=False)
            cap_out = len(gen) >= max_new_tokens
            # think 段 = <think>..</think> 之间(模型模板通常已开 <think>);末标缺失=撞顶/未解
            end = txt.find("</think>")
            if end < 0:
                think_txt = txt; pred = txt; cap_out = True
            else:
                think_txt = txt[:end]; pred = txt[end + len("</think>"):]
            think_txt = think_txt.replace("<think>", "")
            depth = float(len(tok(think_txt, add_special_tokens=False).input_ids))
        except torch.cuda.OutOfMemoryError:                # 长逻辑题(LogiQA2)易 OOM→跳过而非崩
            pred, depth, cap_out = "", float("nan"), True
        torch.cuda.empty_cache()
        yield pred, depth, cap_out


# ================= 适配器:CoLaR(横向,动态潜在链长)——独立加载,不用 Lightning =================
def run_colar(items, max_new_tokens=16):
    """移植 colar 的 latent_generate：base R1-1.5B + LoRA(q/v,r128) + latent_policy(MLP)。
    深度 = n_latent_forward（模型自适应决定的潜在思维数,遇 ### 停,cap=64）。"""
    import os, torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    # 可配置(env)：换 SFT ckpt/降压缩/抬 cap，以看清"收敛前 latent pass 随难度的差异"
    # BASE 可换：qsa-gsm ckpt 基座为 Llama-3.2-1B-Instruct（非 R1-Qwen-1.5B），用 COLAR_BASE 切换
    BASE, dev = os.environ.get("COLAR_BASE", f"{CKPTS}/r1-distill-1.5b"), "cuda"
    CK = os.environ.get("COLAR_CKPT", f"{CKPTS}/colar/ckpt/colar_rl_best.ckpt")
    EMB_STD = float(os.environ.get("COLAR_EMB_STD", "0.03"))   # R1-1.5B=0.03, Llama-3.2-1B=0.018(qsa-gsm)
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "2"))   # 1=最不压缩,latent 最多,范围最宽
    MAX_LAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    print(f"[colar cfg] ckpt={os.path.basename(CK)} compress={COMPRESS} max_lat={MAX_LAT}", flush=True)

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
    assert not [k for k in unexpected if "latent_policy" in k or "lora" in k], f"bad load: {unexpected[:5]}"
    llm = llm.to(dev).eval(); lp = lp.to(dev).float().eval()
    emb = llm.get_input_embeddings()
    sep_id = tok.convert_tokens_to_ids("###")
    QT, SPEED = "Question: {} Let's think step by step:", "(Thinking speed: {})"
    ANSTOK = int(os.environ.get("COLAR_ANSTOK", "48"))     # 答案生成上限:给足 token,勿因截断误判答错
    if os.environ.get("COLAR_SAMPLE") == "1":     # 匹配官方答案生成(采样 top_p=0.9,temp=1.0)
        ANS = dict(max_new_tokens=ANSTOK, do_sample=True, top_p=0.9, temperature=1.0, pad_token_id=tok.pad_token_id)
    else:
        ANS = dict(max_new_tokens=ANSTOK, do_sample=False, pad_token_id=tok.pad_token_id)  # 贪心=可复现
    print(f"colar loaded (sep_id={sep_id})", flush=True)

    def gen_one(question):
        text = QT.format(question.rstrip()) + SPEED.format(COMPRESS) + "###"
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        am = torch.ones_like(ids)
        pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        with torch.no_grad():
            qemb = emb(ids)
            out = llm(inputs_embeds=qemb, attention_mask=am, position_ids=pos,
                      output_hidden_states=True, use_cache=True)
            pkv = out.past_key_values; all_emb = [qemb]; cur_pos = pos[:, -1:]
            n_lat = 0
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


# ================= 适配器:TaH(纵向,每 token 选择性潜在迭代数)——需 tah conda env(torch2.6) =================
def run_tah(items, max_new_tokens=4096):   # 生成放长(额外forward已限长FWD_CAP→不OOM);避免截断误判
    """TaH: Qwen3-1.7B + 选择性潜在迭代(iter_decider)。深度=生成序列上每 token 迭代数(iter_count)均值。
    仅能在 E:/conda_envs/tah 的 python(torch2.6)里跑(drh 的 torch2.4 import 会 segfault)。"""
    import torch
    from transformers import AutoTokenizer
    from tah.model.recurrent_transformer import TaHForCausalLM
    from tah.model.utils import TaHForCasualLM_generate
    P = f"{CKPTS}/tah"
    tok = AutoTokenizer.from_pretrained(P)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = TaHForCausalLM.from_pretrained(P, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                       attn_implementation="sdpa", tah_config=None).eval()
    m = m.to(dtype=torch.bfloat16)          # 统一 dtype(含 iter_decider,否则 bf16/f32 不匹配)
    print("tah loaded", flush=True)
    FWD_CAP = 1200                                   # iter_count 额外 forward 限长(防 OOM)
    for it in items:
        ans, depth, cap_out = "", float("nan"), True
        try:
            msgs = [{"role": "user", "content": it["question"].strip() +
                     "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            mi = tok([text], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
            out_tokens, texts = TaHForCasualLM_generate(
                tah_model=m, tokenizer=tok, model_inputs=mi, iter_count=None,
                max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
            gen_ids = out_tokens[0]; gen_txt = texts[0]; gen_len = len(gen_ids)
            cap_out = gen_len >= max_new_tokens
            depth = float(gen_len)                    # 兜底:think-token 数(免额外 forward)
            extra = {"depth_tok": float(gen_len)}     # think-token 数(长度型轴,TaH 真正的自适应深度)
            if gen_len > 0:                           # 纵向指标:iter_count(≤2,限长 forward,防 OOM)
                seq = gen_ids[:FWD_CAP]
                full = torch.cat([mi.input_ids, torch.tensor([seq], device="cuda:0")], dim=1)
                with torch.no_grad():
                    o = m(full)
                ic = getattr(o, "iter_count", None)
                if ic is not None:
                    arr = ic[0][-len(seq):].float()
                    depth = float(arr.mean())         # mean iter(=硬 token 占比+1)
                    extra["depth_sum"] = float((arr - 1).clamp(min=0).sum())  # 总额外迭代(硬 token 计数)
                del o, full
            ans = gen_txt.split("</think>")[-1] if "</think>" in gen_txt else gen_txt
        except torch.cuda.OutOfMemoryError:
            ans, depth, cap_out, extra = "", float("nan"), True, {}
        torch.cuda.empty_cache()
        yield ans, depth, cap_out, extra


# ================= 适配器:Huginn(纵向,整模型递归深度) =================
def run_huginn(items, max_new_tokens=None):
    """纵向(真·深递归):整隐状态反复穿递归块,深度=自适应递归步数(KL 收敛自停,cap 64)。
    用 forward_with_adaptive_compute 测深度(answer-free,generate 路径与 tf>=4.54 cache 不兼容)。
    Huginn 是 base 模型解不出→pred 留空(全归 incorrect/all),深度 vs 难度看 all_subset。"""
    import torch, importlib
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    MODEL = f"{CKPTS}/huginn-0125"
    tok = AutoTokenizer.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map={"": 0},
                                             trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    raven = importlib.import_module(m.__class__.__module__)
    m.config.mean_recurrence = 64                        # cap(原生默认级别)
    dev = next(m.parameters()).device
    MAX_PROMPT = 400                                     # 过滤过长题(勿截断)
    print("huginn loaded (4bit, mean_recurrence=64)", flush=True)
    for it in items:
        pred, depth, cap_out = "", float("nan"), True
        try:
            ids = tok.encode(it["question"].rstrip() + "\n", return_tensors="pt")
            if ids.shape[1] <= MAX_PROMPT:
                ids = ids.to(dev)
                ev = raven.get_adaptive_exit_evaluator(m, "kl", 5e-4)  # 原生默认判据,固定不调
                with torch.no_grad():
                    out = m.forward_with_adaptive_compute(ids, exit_evaluator=ev)
                cs = out.stats["compute_steps"]           # prompt 段每 token 递归步数
                depth = float(sum(cs) / len(cs))
                cap_out = False                           # 深度=自适应递归,非生成截断
        except Exception as e:
            print("huginn err:", type(e).__name__, str(e)[:120], flush=True)
        torch.cuda.empty_cache()
        yield pred, depth, cap_out


# ================= 适配器:Ouro-2.6B-Thinking(LoopLM,循环共享权重) =================
def run_ouro(items, max_new_tokens=2048):
    """Ouro-2.6B-Thinking(ByteDance LoopLM,arXiv 2510.25741):24 层循环 total_ut_steps=4 次(浅),
    自适应 early_exit_threshold。Thinking 变体生成可见 CoT。
    深度=think-token 数(可见推理长度,长度型;总能读到)。**递归步数(1-4)较浅**,
    如需 per-token 递归步数(纵向深度),在 Colab 下载后看 modeling_ouro.py 的输出接口补测。
    核心价值:多域强求解器→干净跨难度 correct 子集。需 transformers<4.56。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    P = f"{CKPTS}/ouro"                                   # Colab: snapshot_download 到此
    tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(P, trust_remote_code=True)
    if hasattr(cfg, "early_exit_threshold"):             # <1.0 开启自适应提前退出(默认1.0=不退)
        cfg.early_exit_threshold = float(os.environ.get("OURO_EXIT", "0.5"))
    if os.environ.get("OURO_STEPS") and hasattr(cfg, "total_ut_steps"):  # 扩循环步数(4→8/16)拉宽动态范围
        cfg.total_ut_steps = int(os.environ["OURO_STEPS"])
    load_kw = dict(config=cfg, trust_remote_code=True, device_map="cuda:0")
    if os.environ.get("OURO_4BIT", "1") == "1":          # 本地 6GB 用 4-bit(默认开);Colab 显存足可设 0
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        load_kw["torch_dtype"] = "auto"
    m = AutoModelForCausalLM.from_pretrained(P, **load_kw).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"ouro loaded (ut_steps={getattr(cfg,'total_ut_steps','?')}, exit={getattr(cfg,'early_exit_threshold','?')})", flush=True)
    for it in items:
        pred, depth, cap_out = "", float("nan"), True
        try:
            msgs = [{"role": "user", "content": it["question"].strip() +
                     "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = m.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gen = out[0, ids.input_ids.shape[1]:]
            txt = tok.decode(gen, skip_special_tokens=True)
            depth = float(len(gen))                       # think-token 数(长度型深度)
            cap_out = len(gen) >= max_new_tokens
            pred = txt.split("</think>")[-1] if "</think>" in txt else txt
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        yield pred, depth, cap_out


# ================= 适配器:adaLR(横向自适应潜在推理,官方代码加载) =================
def run_adalr(items, max_new_tokens=220):
    """adaLR latent-6_rl(arXiv 2511.21581):Llama-3.2-1B + 潜在 token(<START><CONTINUE>*n<STOP>),
    RL 学"何时停"→ **横向自适应潜在链**(结构类 Coconut)。深度 = <CONTINUE> 潜在思维个数(自适应)。
    用官方 repo 代码(src.model_creation)加载,避免独立移植的格式/权重坑。
    输入格式(官方):'Question: {q}<START>' → 模型生成潜在思维 + ' The answer is: X'。"""
    import sys, torch
    REPO = os.environ.get("ADALR_REPO", f"{CKPTS}/adaLR-repo")
    CK = os.environ.get("ADALR_CKPT", f"{CKPTS}/adaLR_rl")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from transformers import AutoTokenizer, GenerationConfig
    from src.model_creation import automodelforcausallm_from_pretrained_latent
    from src.modeling_utils import (ensure_tokenizer_has_latent_tokens,
                                     START_TOKEN_STR, CONTINUE_TOKEN_STR)
    tok = AutoTokenizer.from_pretrained(CK)
    ensure_tokenizer_has_latent_tokens(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # 关键:torch_dtype=bf16 直接加载装进 6GB;返回 LatentThinkingModel(config 带 _inserted 标记)
    model = automodelforcausallm_from_pretrained_latent(CK, torch_dtype=torch.bfloat16).to("cuda").eval()
    cont_id = tok.convert_tokens_to_ids(CONTINUE_TOKEN_STR)
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                          pad_token_id=128001)
    print(f"adalr loaded (cont_id={cont_id})", flush=True)
    for it in items:
        pred, depth, cap_out = "", float("nan"), True
        try:
            # 官方格式:raw question + <START>,**必须 add_special_tokens=True 加 BOS**(否则输出乱码)
            text = it["question"].strip() + START_TOKEN_STR
            enc = tok(text, return_tensors="pt", add_special_tokens=True)
            ids = enc.input_ids.to("cuda"); am = enc.attention_mask.to("cuda")
            with torch.no_grad():
                out = model.generate(ids, attention_mask=am, generation_config=gc)
            gen = out[0, ids.shape[1]:]
            depth = float((gen == cont_id).sum().item())        # <CONTINUE> 个数=自适应潜在思维数=深度
            cap_out = len(gen) >= max_new_tokens
            full = tok.decode(gen, skip_special_tokens=True)
            pred = full.split("answer is")[-1].lstrip(": ").strip() if "answer is" in full.lower() else full.strip()
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        yield pred, depth, cap_out


def run_lsft(items, max_new_tokens=64):
    """Latent-SFT-1B (DJC-GO-SOLO/Latent-SFT, vocabulary-space latent chains)。
    深度 = <think>..</think> 之间的 latent 步数 (one_example_generate_hf 的 generate_token_num)。
    env: LSFT_REPO / LSFT_CKPT / LSFT_MAXLAT。"""
    import os, sys, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    REPO = os.environ.get("LSFT_REPO", f"{CKPTS}/latent-sft-repo")
    CK = os.environ.get("LSFT_CKPT", f"{CKPTS}/latent-sft-1b")
    MAXLAT = int(os.environ.get("LSFT_MAXLAT", "96"))
    if os.path.join(REPO, "src") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "src"))
    from modeling.modeling_stage2 import LatentSFTStage2SoftEmbedding
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(CK)
    model = AutoModelForCausalLM.from_pretrained(CK, torch_dtype=torch.bfloat16).to(dev).eval()
    model.tokenizer = tok
    model.latent_token_ids = tok(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    print(f"[lsft cfg] ckpt={os.path.basename(CK)} max_lat={MAXLAT} think_ids={model.latent_token_ids}", flush=True)

    def build(q):
        pre = (f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, "
               f"and put your final answer within \\boxed{{}}.\n{q}<|eot_id|>"
               f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        ids = tok(pre, add_special_tokens=False)['input_ids'] + model.latent_token_ids[0]
        return {"input_ids": torch.tensor([ids], dtype=torch.long).to(dev),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long).to(dev)}

    for it in items:
        out = LatentSFTStage2SoftEmbedding.one_example_generate_hf(
            model, build(it["question"]), max_new_tokens=MAXLAT,
            do_sample=False, topk_interpolation=10, add_gumbel_noise=False)
        depth = float(out["generate_token_num"])          # = latent 步数
        yield out["text"], depth, depth >= MAXLAT
        if dev == "cuda":
            torch.cuda.empty_cache()


ADAPTERS = {"lwts": run_lwts, "r1": run_r1, "colar": run_colar, "tah": run_tah,
            "huginn": run_huginn, "ouro": run_ouro, "adalr": run_adalr, "lsft": run_lsft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--depth_field", default="depth",
                    help="重析时用哪个深度字段(depth/depth_tok/depth_sum),TaH 多指标用")
    ap.add_argument("--model", choices=list(ADAPTERS), default="lwts")
    ap.add_argument("--domain", choices=["math", "logic", "commonsense", "gsmllm"], default="math")
    ap.add_argument("--n_per_source", type=int, default=60)
    ap.add_argument("--sources", default="",
                    help="逗号分隔, 只跑池里的这些源(如 --sources proofwriter 做单源探针); 空=全部")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.analyze:
        rec = json.loads(Path(args.analyze).read_text(encoding="utf-8"))["records"]
        if args.depth_field != "depth":               # 用别的指标(TaH depth_tok/depth_sum)重析
            rec = [{**r, "depth": r[args.depth_field]} for r in rec if args.depth_field in r]
        print(json.dumps(analyze(rec), indent=2, ensure_ascii=False)); return

    import time
    srcs = [s.strip() for s in args.sources.split(",") if s.strip()] or None
    items = sample_pool(args.domain, args.n_per_source, args.seed, sources=srcs)
    print(f"model={args.model} domain={args.domain} sources={srcs or 'ALL'} n={len(items)}", flush=True)
    runner = ADAPTERS[args.model](items, args.max_new_tokens)
    _sfx = f"_{'_'.join(srcs)}" if srcs else ""
    out = Path(args.output) if args.output else HERE / "outputs" / f"dd_{args.model}_{args.domain}{_sfx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()

    def save():                                          # 增量存盘:崩溃不丢已跑
        out.write_text(json.dumps({"model": args.model, "domain": args.domain,
            "n_per_source": args.n_per_source, "records": records,
            "analysis": analyze(records)}, indent=2, ensure_ascii=False), encoding="utf-8")

    for i, (it, ro) in enumerate(zip(items, runner)):
        pred, depth, cap_out = ro[0], ro[1], ro[2]
        extra = ro[3] if len(ro) > 3 else {}            # 可选:额外深度指标(如 TaH 的 depth_tok/depth_sum)
        correct = bool(grade(it, pred))
        rec = {"source": it["source"], "difficulty": it["difficulty"], "depth": depth,
               "correct": correct, "cap_out": bool(cap_out), "gold": str(it["gold"]),
               "pred": (pred or "")[-400:]}              # 存答案尾部(勿截断→可复现重判)
        rec.update(extra)
        records.append(rec)
        if (i + 1) % 20 == 0:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  {i+1}/{len(items)}  acc={acc:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            save()
    res = {"model": args.model, "domain": args.domain, "n_per_source": args.n_per_source,
           "records": records, "analysis": analyze(records)}
    save()
    print("verdict:", json.dumps(res["analysis"], ensure_ascii=False), flush=True)
    print("saved ->", out)


if __name__ == "__main__":
    main()
