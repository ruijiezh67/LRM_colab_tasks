# -*- coding: utf-8 -*-
r"""
depth_acc_dualtrack_colar.py
============================
② 验证的最后两个缺口(对 train_colar_dualtrack.py 训出的 CoLaR 双轨模型):
  ②-b 自适应深度: 难度(每题步数) -> 潜在链深度 是否单调上升(Spearman)。
  ②-c 留出集准确率: 真·held-out GSM8K test 准确率(对标下载版 colar 论文 ~50%+)。

几何/生成 **完全复用** verify_dualtrack_colar.py:
  * 加载器 build_dualtrack / _load_adapter_sd / _resize  —— 逐字复制(sibling 不导出, 按要求 copy)。
  * 推理 latent_chain / gen_full / parse_gen             —— 逐字复制(确定性: policy.mean + argmax 停 sep;
    装配 [q, latent, <eot>, <cot>] 自由续写 CoT </cot> answer)。verify 的 `acc` 字段就是这条路径。
  * depth = len(lats) = 生成的潜在步数(sep 停之前的链长), 与 verify 同一处取值。
判分 _norm/grade 复制自 strong_causal_colar.py, 但对自然语言答案(GSM8K)加一条"末位数字"回退
(严格版对 "The answer is 42" 返回 False; GSM8K 标准评测本就取末位数字, 也更贴论文口径)。

难度信号(每题, 留出): ex["answer"] 里 `<<...>>` 计算器标注数 = len(re.findall(r"<<", ans)) = 推理步数。
gold = "####" 后的数字。fallback: 给 --pool 时按 strong_causal 约定读 [{question, gold}], difficulty=None(跳过深度相关, 仍算 acc)。

env: COLAR_BASE=.../Llama-3.2-1B-Instruct  COLAR_CKPT=.../colar-gsm/colar_best.ckpt
     COLAR_EMB_STD=0.018  COLAR_COMPRESS=5  COLAR_MAXLAT=64  (可选 COLAR_GENTOK)
     TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  (老 ckpt 反序列化)
run: PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 <py> depth_acc_dualtrack_colar.py --n 200
selftest(纯 python, 无 torch/GPU/网络): <py> depth_acc_dualtrack_colar.py --selftest
"""
import argparse
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KT = HERE.parent.parent.parent

COT_OPEN = "<cot>"
COT_CLOSE = "</cot>"
QT = "Question: {} Let's think step by step:"
SPEED = "(Thinking speed: {})"

ACC_THRESH = 0.5   # ②-c 判定阈值(对标下载版 colar 论文 ~50%+); 仅用于 verdict 提示, 不改变数值


# --------------------------------------------------------------------------- #
#  判分 helper —— _norm/grade 复制自 strong_causal_colar.py line 14-19,
#  另加"末位数字"回退: 严格版对自然语言(如 "The answer is 42")返回 False,
#  而 GSM8K 标准评测取末位数字, 故回退既满足 selftest 又更贴论文口径。
#  bare-number(如 "42")两条路径结果一致, 与 verify 的 grade 兼容。
# --------------------------------------------------------------------------- #
def _norm(s):
    return re.sub(r"[^a-z0-9.]", "", str(s).split("Answer:")[-1].lower())


def _last_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", str(s))
    return m[-1].replace(",", "") if m else None


def grade(pred, gold):
    p, g = _norm(pred), _norm(re.sub(r",", "", str(gold)))
    try:
        return abs(float(p) - float(g)) < 1e-4        # strong_causal 原判据(bare number)
    except Exception:
        pass
    np_, ng = _last_num(pred), _last_num(gold)         # 回退: 从自然语言里抠末位数字
    if np_ is not None and ng is not None:
        try:
            return abs(float(np_) - float(ng)) < 1e-4
        except Exception:
            pass
    return p == g


# --------------------------------------------------------------------------- #
#  Spearman(手写, 无 scipy): 对两列各自排名(含并列取平均秩)再算 Pearson。
#  guard: n<3 或任一列零方差 -> None。correlation 对称, 传 (depth, difficulty) 任意序。
# --------------------------------------------------------------------------- #
def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                       # 1-based 平均秩(并列)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:                              # 零方差 -> 无定义
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx ** 0.5 * sy ** 0.5)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:                                  # guard n<3
        return None
    r = _pearson(_rank([p[0] for p in pairs]), _rank([p[1] for p in pairs]))
    return None if r is None else round(r, 4)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5   # population std


# =========================================================================== #
#  指标聚合(纯 python, selftest 可直接调用) —— 输入 rows: {q, depth, difficulty, pred, gold, correct}
# =========================================================================== #
def compute_metrics(rows, tag, n_req):
    N = len(rows)
    depths = [r["depth"] for r in rows]
    corr = [bool(r["correct"]) for r in rows]
    acc = round(_mean([1.0 if c else 0.0 for c in corr]), 3) if N else None

    have_diff = any(r["difficulty"] is not None for r in rows)
    acc_by_bin, depth_by_diff, sp = {}, {}, None
    if have_diff:
        diff_rows = [r for r in rows if r["difficulty"] is not None]
        sp = spearman([r["difficulty"] for r in diff_rows], [r["depth"] for r in diff_rows])
        by = {}
        for r in diff_rows:
            by.setdefault(r["difficulty"], []).append(r)
        for d in sorted(by):
            grp = by[d]
            acc_by_bin[str(d)] = round(_mean([1.0 if x["correct"] else 0.0 for x in grp]), 3)
            depth_by_diff[str(d)] = round(_mean([x["depth"] for x in grp]), 3)

    raw_std = _std(depths) if N else 0.0
    non_constant = bool(raw_std > 0)                    # 按 spec: non_constant = std_depth > 0
    mean_depth = round(_mean(depths), 3) if N else None
    std_depth = round(raw_std, 4) if N else None
    dc = [r["depth"] for r in rows if r["correct"]]
    dw = [r["depth"] for r in rows if not r["correct"]]
    depth_on_correct = round(_mean(dc), 3) if dc else None
    depth_on_wrong = round(_mean(dw), 3) if dw else None

    examples = [dict(q_short=str(r["q"])[:80], depth=r["depth"], difficulty=r["difficulty"],
                     pred=r["pred"], gold=r["gold"], correct=bool(r["correct"])) for r in rows[:12]]

    return {
        "tag": tag, "n": N, "n_requested": n_req,
        "accuracy": {                                   # ②-c
            "acc": acc,
            "acc_by_difficulty_bin": acc_by_bin,
        },
        "adaptive_depth": {                             # ②-b
            "spearman": sp,
            "mean_depth": mean_depth, "std_depth": std_depth, "non_constant": non_constant,
            "depth_by_difficulty": depth_by_diff,
            "depth_on_correct": depth_on_correct, "depth_on_wrong": depth_on_wrong,
        },
        "examples": examples,
        "read": ("②-b 通过 = spearman>0 且 non_constant=True (潜在深度随难度上升); "
                 "②-c 通过 = acc >= 阈值 (对标下载版 colar 论文 ~50%+)"),
        "verdict": {
            "depth_b_pass": bool(sp is not None and sp > 0 and non_constant),
            "acc_c_pass_at_{:.2f}".format(ACC_THRESH): bool(acc is not None and acc >= ACC_THRESH),
        },
    }


# =========================================================================== #
#  数据: held-out GSM8K test(带每题难度) / --pool 回退
# =========================================================================== #
def load_gsm8k_test(n):
    """openai/gsm8k main test: question / gold(#### 后数字) / difficulty(<< 计数 = 推理步数)。"""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for ex in ds:
        ans = ex["answer"]
        gold = re.sub(r"[,$]", "", ans.split("####")[-1].strip())
        diff = len(re.findall(r"<<", ans))              # 计算器标注数 = 每题推理步数(干净难度代理)
        items.append({"question": ex["question"], "gold": gold, "difficulty": diff})
        if len(items) >= n:
            break
    return items


def load_pool(path, n):
    """回退: strong_causal 约定的 pool(list/dict), difficulty=None(仅算 acc)。"""
    pool = json.load(open(path, encoding="utf-8"))
    if isinstance(pool, dict):
        pool = pool.get("data") or list(pool.values())[0]
    return [{"question": it["question"], "gold": it.get("gold"), "difficulty": None}
            for it in pool[:n]]


# =========================================================================== #
#  模型重建 —— 逐字复制自 verify_dualtrack_colar.build_dualtrack (line 155-238)。
#  差异只在: LoRA 用 out_dir 训练过的适配器覆盖, cot 两行用 cot_embeds.pt 覆盖。
# =========================================================================== #
def _resize(llm, n):
    """verify line 140-145 verbatim。"""
    try:
        llm.resize_token_embeddings(n)
    except Exception:
        llm.base_model.model.resize_token_embeddings(n)


def _load_adapter_sd(out_dir):
    """verify line 241-249 verbatim: 读 out_dir 训练过的 LoRA(safetensors 优先, 退 .bin)。"""
    import torch
    p_sft = os.path.join(out_dir, "adapter_model.safetensors")
    p_bin = os.path.join(out_dir, "adapter_model.bin")
    if os.path.exists(p_sft):
        from safetensors.torch import load_file
        return load_file(p_sft)
    return torch.load(p_bin, map_location="cpu")


def build_dualtrack(base, ckpt, out_dir, dev):
    """verify_dualtrack_colar.build_dualtrack line 155-238 逐字复制(几何必须一致)。"""
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    EMB_STD = float(os.environ.get("COLAR_EMB_STD", "0.018"))

    # ---- 1) tok: 先加 [PAD](verify line 164-165)-------------------------------
    tok = AutoTokenizer.from_pretrained(base)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    # ---- 2) base(bf16 + sdpa) + resize 到 len(tok)(verify line 167-169)-------
    llm = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    llm.resize_token_embeddings(len(tok))
    # ---- 3) 同一份 LoraConfig(r=128, q/v)(verify line 171-172)----------------
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    H = llm.config.hidden_size

    # ---- 4) 同一份 LatentPolicy(verify line 176-186)---------------------------
    class LatentPolicy(nn.Module):
        def __init__(s, f, inter=2048):
            super().__init__()
            s.fc = nn.Sequential(nn.Linear(f, inter), nn.GELU(), nn.Linear(inter, inter), nn.LayerNorm(inter))
            s.mean = nn.Linear(inter, f)
            s.log_std = nn.Linear(inter, f)

        def forward(s, x, temperature=1.0):
            x = s.fc(x)
            return torch.distributions.Normal(s.mean(x), s.log_std(x).exp() * temperature)

    lp = LatentPolicy(H, 2048)
    cont = nn.Module()
    cont.llm = llm
    cont.latent_policy = lp
    # ---- 5) 加载 colar ckpt(strict=False)(verify line 192-194)----------------
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    miss, _ = cont.load_state_dict(sd, strict=False)
    assert not [k for k in miss if "latent_policy" in k or "lora" in k.lower()], "ckpt 键不符"

    sep_id = tok.convert_tokens_to_ids("###")
    # ---- 6) 加 <cot>/</cot> 并 resize(verify line 198-202)---------------------
    n_added = tok.add_special_tokens({"additional_special_tokens": [COT_OPEN, COT_CLOSE]})
    if n_added > 0:
        _resize(llm, len(tok))
    cot_open_id = tok.convert_tokens_to_ids(COT_OPEN)
    cot_close_id = tok.convert_tokens_to_ids(COT_CLOSE)

    # ---- 7) 用 out_dir 训练过的 LoRA 覆盖(verify line 208-220)-----------------
    asd = _load_adapter_sd(out_dir)
    own = dict(llm.named_parameters())
    loaded = 0
    with torch.no_grad():
        for k, v in asd.items():
            if "lora_" not in k:
                continue
            cand = k[:-len(".weight")] + ".default.weight" if k.endswith(".weight") else k
            tgt = own.get(cand) or own.get(k)
            if tgt is not None:
                tgt.data.copy_(v.to(tgt.dtype))
                loaded += 1
    assert loaded > 0, f"没从 {out_dir} 载入任何训练过的 LoRA 张量(检查 adapter_model.safetensors)"

    # ---- 8) 用 cot_embeds.pt 覆盖两行 cot 嵌入(verify line 223-230)------------
    blob = torch.load(os.path.join(out_dir, "cot_embeds.pt"), map_location="cpu")
    cot_open_id = int(blob["cot_open_id"])
    cot_close_id = int(blob["cot_close_id"])
    sep_id = int(blob.get("sep_id", sep_id))
    emb = llm.get_input_embeddings()
    with torch.no_grad():
        emb.weight[cot_open_id] = blob["cot_open_row"].to(emb.weight.dtype).to(emb.weight.device)
        emb.weight[cot_close_id] = blob["cot_close_row"].to(emb.weight.dtype).to(emb.weight.device)

    llm = llm.to(dev).eval()
    lp = lp.to(dev).float().eval()
    ids = {"sep_id": sep_id, "cot_open_id": cot_open_id, "cot_close_id": cot_close_id,
           "pad_id": tok.pad_token_id, "eos_id": tok.eos_token_id, "H": H, "emb_std": EMB_STD}
    print(f"[depth_acc] 重建完成: 训练 LoRA 张量 {loaded} 个已载入, cot 行已覆盖 "
          f"(sep={sep_id} <cot>={cot_open_id} </cot>={cot_close_id})", flush=True)
    return llm, lp, tok, emb, ids


# =========================================================================== #
#  推理闭包 —— latent_chain / gen_full / parse_gen 逐字复制自 verify make_runtime
#  (line 255-324)。深度=len(lats); 答案路径=verify 的 `acc` 路径。不用 gen_answer/bottleneck。
# =========================================================================== #
def make_runtime(llm, lp, tok, emb, ids, dev):
    import torch
    sep_id, cot_open_id, cot_close_id = ids["sep_id"], ids["cot_open_id"], ids["cot_close_id"]
    pad_id, eos_id, H, EMB_STD = ids["pad_id"], ids["eos_id"], ids["H"], ids["emb_std"]
    MAX_LAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    GENTOK = int(os.environ.get("COLAR_GENTOK", "160"))
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5"))

    eot_e = emb(torch.tensor([[sep_id]], device=dev))        # <eot> = "###" sep
    co_e = emb(torch.tensor([[cot_open_id]], device=dev))    # <cot>

    def q_to_ids(question):
        text = QT.format(str(question).rstrip()) + SPEED.format(COMPRESS) + "###"   # verify line 271 / train line 285
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

    def latent_chain(q_ids, inject=None):
        """verify line 274-298 verbatim: 确定性 policy.mean + argmax 停 sep, 上限 MAX_LAT。"""
        am = torch.ones_like(q_ids)
        pos = torch.arange(q_ids.shape[1], device=dev).unsqueeze(0)
        qemb = emb(q_ids)
        out = llm(inputs_embeds=qemb, attention_mask=am, position_ids=pos,
                  output_hidden_states=True, use_cache=True)
        pkv = out.past_key_values
        cur = pos[:, -1:]
        lats = []
        n = len(inject) if inject is not None else MAX_LAT
        for k in range(n):
            if inject is not None:
                ce = inject[k].to(qemb.dtype)
            else:
                ce = (lp(out.hidden_states[-1][:, -1:, :].float()).mean * EMB_STD).to(qemb.dtype)
            lats.append(ce.detach().clone())
            am = torch.cat([am, torch.ones(1, 1, device=dev, dtype=am.dtype)], 1)
            cur = cur + 1
            out = llm(inputs_embeds=ce, attention_mask=am, position_ids=cur,
                      past_key_values=pkv, output_hidden_states=True, use_cache=True)
            pkv = out.past_key_values
            if inject is None and int(out.logits[:, -1].argmax(-1)) == sep_id:   # 确定性 argmax 停
                break
        return lats

    @torch.no_grad()
    def gen_full(q_ids, inject=None):
        """verify line 300-309 verbatim: [q, latent, <eot>, <cot>] -> 续写 CoT </cot> answer。"""
        lats = latent_chain(q_ids, inject)
        prefix = torch.cat([emb(q_ids)] + lats + [eot_e, co_e], dim=1)
        amf = torch.ones(1, prefix.shape[1], device=dev, dtype=torch.long)
        pred = llm.generate(inputs_embeds=prefix, attention_mask=amf,
                            max_new_tokens=GENTOK, do_sample=False, pad_token_id=pad_id)
        return pred[0].tolist(), lats

    def parse_gen(gen_ids):
        """verify line 311-324 verbatim: 从自由生成里切出 cot_ids 与答案文本。"""
        if cot_close_id in gen_ids:
            k = gen_ids.index(cot_close_id)
            cot_ids = gen_ids[:k]
            ans_ids = gen_ids[k + 1:]
            closed = True
        else:
            cot_ids = list(gen_ids)
            ans_ids = []
            closed = False
        cot_ids = [t for t in cot_ids if t not in (eos_id, pad_id, cot_open_id, cot_close_id)]
        ans_text = tok.decode(ans_ids, skip_special_tokens=True)
        return cot_ids, ans_text, closed

    return q_to_ids, gen_full, parse_gen


# =========================================================================== #
#  主流程
# =========================================================================== #
def run(a):
    import torch
    base = os.environ["COLAR_BASE"]
    ckpt = os.environ.get("COLAR_CKPT")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    llm, lp, tok, emb, ids = build_dualtrack(base, ckpt, a.out_dir, dev)
    q_to_ids, gen_full, parse_gen = make_runtime(llm, lp, tok, emb, ids, dev)

    items = load_pool(a.pool, a.n) if a.pool else load_gsm8k_test(a.n)

    rows = []
    for it in items:
        q_ids = q_to_ids(it["question"])
        gid, lats = gen_full(q_ids)                       # 与 verify 的 acc 同一路径
        cot_ids, ans_text, closed = parse_gen(gid)
        depth = len(lats)                                 # = 生成的潜在步数(sep 停之前)
        rows.append(dict(q=it["question"], depth=depth, difficulty=it.get("difficulty"),
                         pred=_norm(ans_text), gold=it.get("gold"),
                         correct=bool(grade(ans_text, it.get("gold")))))
        if dev == "cuda":
            torch.cuda.empty_cache()

    res = compute_metrics(rows, a.tag, a.n)
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, f"depth_acc_{a.tag}.json")
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    acc = res["accuracy"]["acc"]
    ad = res["adaptive_depth"]
    print(f"\n==== depth+acc dual-track [{a.tag}] acc(②-c)={acc} (n={res['n']}) ====")
    print(f"②-b: spearman={ad['spearman']} mean_depth={ad['mean_depth']} std={ad['std_depth']} "
          f"non_constant={ad['non_constant']} depth(corr/wrong)={ad['depth_on_correct']}/{ad['depth_on_wrong']} -> {outp}")


# =========================================================================== #
#  --selftest : 纯 python, 无 torch/GPU/网络 —— Spearman + non_constant + grade
# =========================================================================== #
def _selftest():
    # (1) depth 随 difficulty 上升 -> Spearman 接近 +1
    sp_mono = spearman([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    assert sp_mono is not None and abs(sp_mono - 1.0) < 1e-9, ("perfect monotone", sp_mono)
    sp_inc = spearman([1, 2, 3, 4, 5], [2, 3, 3, 5, 8])       # 非严格单增(含并列)
    assert sp_inc is not None and sp_inc > 0.9, ("increasing", sp_inc)

    # (2) 常数 depth -> non_constant=False 且 spearman=None
    rows_const = [dict(q="q", depth=3, difficulty=i, pred="", gold="", correct=False) for i in range(5)]
    m = compute_metrics(rows_const, "selftest", 5)
    assert m["adaptive_depth"]["non_constant"] is False, ("non_constant", m["adaptive_depth"])
    assert m["adaptive_depth"]["spearman"] is None, ("spearman const", m["adaptive_depth"])
    assert spearman([1, 2, 3, 4, 5], [3, 3, 3, 3, 3]) is None, "const depth spearman must be None"
    assert spearman([1, 2], [1, 2]) is None, "n<3 must be None"

    # (3) 判分
    assert grade("The answer is 42", "42") is True, "grade prose"
    assert grade("18", "18.0") is True, "grade 18/18.0"
    assert grade("The answer is 42", "43") is False, "grade wrong prose"

    # (4) 端到端指标烟测(difficulty=None 路径 + 正常路径)
    rows_pool = [dict(q="q%d" % i, depth=2, difficulty=None, pred="1", gold="1", correct=True) for i in range(4)]
    mp = compute_metrics(rows_pool, "pool", 4)
    assert mp["accuracy"]["acc"] == 1.0 and mp["adaptive_depth"]["spearman"] is None, mp["accuracy"]
    rows_ok = [dict(q="q%d" % i, depth=i + 1, difficulty=i + 1, pred="", gold="", correct=(i % 2 == 0)) for i in range(6)]
    mo = compute_metrics(rows_ok, "ok", 6)
    assert mo["verdict"]["depth_b_pass"] is True, mo["verdict"]
    assert set(mo["adaptive_depth"]["depth_by_difficulty"].keys()) == {"1", "2", "3", "4", "5", "6"}

    print("[selftest] OK —— Spearman 手写实现单调->+1、常数->None(且 non_constant=False)、"
          "n<3->None; grade 自然语言/小数点回退成立; compute_metrics(有/无难度)自洽。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/content/colar_dualtrack",
                    help="训练好的 dual-track 目录(含训练 LoRA + cot_embeds.pt)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pool", default=None, help="可选回退 pool json(给了则 difficulty=None, 只算 acc)")
    ap.add_argument("--out", default=str(HERE.parent.parent / "outputs"))
    ap.add_argument("--tag", default="colar_dualtrack")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        run(a)
