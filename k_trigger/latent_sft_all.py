# -*- coding: utf-8 -*-
"""Latent-SFT (DJCheng) 全套实验 = 对 colar-gsm 做过的那套搬到 Latent-SFT。
一次前向出三份: logit-lens 探针 + 因果(token-swap/steering) + 难度->深度。
机制复刻自 repo src/modeling/modeling_stage2.py one_example_generate_hf:
  问题+<think> -> 潜在循环(每步 argmax 若=</think> 停; 否则 top-10 softmax 加权 embedding 喂回) -> 拼[q,潜在链,</think>] 贪心生成答案。
潜在链长=自适应(模型自己决定何时 </think>)=> 难度->深度可测; 每步 logits top5 => logit-lens; 软潜在向量可换/可扰 => 因果。

GPU(base torch2.7 或 drh): PYTHONIOENCODING=utf-8 <py> k_trigger/latent_sft_all.py --n 40
纯逻辑自检: <py> k_trigger/latent_sft_all.py --selftest
"""
import os, sys, json, re, argparse, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
CKPTS = os.environ.get("CKPTS_ROOT", "E:/ckpts")

OPS = {"+", "-", "*", "/", "=", "x", "×", "÷", "%", "$"}
def is_content(tok_str):
    s = tok_str.strip()
    if not s: return False
    return any(c.isdigit() for c in s) or s in OPS or s.lower() in \
        {"add","sum","total","times","divide","multiply","minus","plus","each","per"}

def norm_num(s):
    s = str(s).replace(",", "")
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m: s = m.group(1)
    n = re.findall(r"-?\d+\.?\d*", s)
    return n[-1] if n else str(s).strip().lower()

def grade(pred, gold):
    p, g = norm_num(pred), norm_num(gold)
    try:
        return abs(float(p) - float(g)) < 1e-4
    except Exception:
        return p == g

def spearman(xs, ys):
    n = len(xs)
    if n < 3: return None
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2.0
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return round(num/(dx*dy), 3) if dx and dy else None


def build_model(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    kw = dict(torch_dtype=torch.float16, use_cache=True, trust_remote_code=True)
    if os.environ.get("LATENTSFT_4BIT") == "1":   # 7B 在 6GB 上需 4-bit
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kw).to("cuda").eval()
    latent_ids = tok(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    print(f"latent-sft loaded; <think>={latent_ids[0]} </think>={latent_ids[1]}", flush=True)
    return model, tok, latent_ids


_KIND = "llama"   # main 里按 ckpt 路径设置; 复刻 eval prepare_single_example 的分支
def prompt_ids(tok, question, latent_ids):
    ask = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{question}"
    if _KIND == "llama":
        txt = (f"<|start_header_id|>user<|end_header_id|>\n\n{ask}<|eot_id|>"
               f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif _KIND == "deepseek":
        txt = tok.apply_chat_template([{"role": "user", "content": ask}], tokenize=False,
                                      add_generation_prompt=False) + "<｜Assistant｜>"
    else:  # qwen 及其它: 官方走 apply_chat_template(add_generation_prompt=True)
        txt = tok.apply_chat_template([{"role": "user", "content": ask}], tokenize=False,
                                      add_generation_prompt=True)
    ids = tok(txt, add_special_tokens=False)['input_ids'] + latent_ids[0]  # 追加 <think>
    return ids


def latent_forward(model, tok, question, latent_ids, max_latent=128, topk=10):
    """纯潜在循环(argmax 确定性)。返回 depth, 潜在链[1,L,H], passes(每步top5), argmax_toks。"""
    import torch, torch.nn.functional as F
    dev = model.device
    W = model.model.embed_tokens.weight.detach()
    ids = prompt_ids(tok, question, latent_ids)
    input_ids = torch.tensor([ids], device=dev)
    attn = torch.ones_like(input_ids)
    end_id = latent_ids[1][0]
    past, cur_ids, cur_emb = None, input_ids, None
    chain, passes, argmax_toks = [], [], []
    with torch.no_grad():
        for _ in range(max_latent):
            emb = model.model.embed_tokens(cur_ids) if cur_ids is not None else cur_emb
            out = model(inputs_embeds=emb, attention_mask=attn, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            nxt = int(logits.argmax())
            top5 = torch.topk(logits, 5, -1).indices[0].tolist()
            tl, ti = torch.topk(logits, topk, -1)
            tp = torch.softmax(tl.float(), -1).to(W.dtype)
            lat = (F.embedding(ti, W) * tp.unsqueeze(-1)).sum(1)   # [1,H] 词表叠加态
            if nxt == end_id:
                break
            chain.append(lat.unsqueeze(1).detach().clone())   # [1,1,H]
            passes.append([tok.decode([t]).replace("\n", "\\n") for t in top5])
            argmax_toks.append(nxt)
            cur_emb = lat.unsqueeze(1); cur_ids = None
            attn = torch.cat([attn, torch.ones((1, 1), device=dev)], 1)
    lat_chain = torch.cat(chain, dim=1) if chain else None   # [1,L,H]
    return len(chain), lat_chain, passes, argmax_toks


def answer_from_latents(model, tok, question, latent_ids, lat_chain, max_new=128):
    """拼 [q+<think>, 潜在链, </think>] 贪心生成答案(确定性, 便于 swap/steer 对比)。"""
    import torch
    dev = model.device
    ids = prompt_ids(tok, question, latent_ids)               # 末尾含 <think>
    base = model.model.embed_tokens(torch.tensor([ids], device=dev))
    end_emb = model.model.embed_tokens(torch.tensor([latent_ids[1]], device=dev))
    parts = [base] + ([lat_chain.to(base.dtype)] if lat_chain is not None else []) + [end_emb]
    emb = torch.cat(parts, dim=1)
    attn = torch.ones((1, emb.size(1)), dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                             do_sample=False, num_beams=1, pad_token_id=128001)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt


def steer_chain(lat_chain, beta):
    import torch
    d = torch.randn_like(lat_chain.float()); d = d / (d.norm(dim=-1, keepdim=True) + 1e-9)
    return (lat_chain.float() + beta * lat_chain.float().norm(dim=-1, keepdim=True) * d).to(lat_chain.dtype)


def run_all(model, tok, latent_ids, items, betas, out_dir, tag, max_latent=128):
    import torch
    caps = []   # 每题: depth, chain, passes, argmax, clean_ans, correct, difficulty, source
    for i, it in enumerate(items):
        q, gold = it["question"], it.get("gold")
        depth, chain, passes, argmax = latent_forward(model, tok, q, latent_ids, max_latent=max_latent)
        clean = answer_from_latents(model, tok, q, latent_ids, chain)
        pred = clean
        ok = grade(pred, gold)
        caps.append(dict(q=q[:90], gold=gold, difficulty=float(it.get("difficulty", 0)),
                         source=it.get("source", "gsm8k"),
                         depth=depth, chain=chain, passes=passes, argmax=argmax,
                         clean_ans=norm_num(clean), clean_raw=clean.strip()[:300], correct=bool(ok)))
        if i < 6:
            lens = " -> ".join(p[0] for p in passes[:12])
            print(f"[{i}] depth={depth} pred={norm_num(clean)} gold={gold} ok={ok} | latent: {lens}", flush=True)
        torch.cuda.empty_cache()

    # ---- 因果: token-swap(换别题潜在链) + steering ----
    import random as _rnd
    _r = _rnd.Random(0)
    donor_mode = os.environ.get("LSFT_DONOR", "next")   # next=(i+1) / random(排除相邻偏)
    for i, c in enumerate(caps):
        if c["chain"] is None:
            c["swap_ans"] = c["clean_ans"]; c["swap_changed"] = False; c["steer"] = {}; continue
        if donor_mode == "random":
            cand = [k for k in range(len(caps)) if k != i and caps[k]["chain"] is not None
                    and caps[k]["clean_ans"] != c["clean_ans"]]
            j = _r.choice(cand) if cand else (i + 1) % len(caps)
        else:
            j = (i + 1) % len(caps)
        donor = caps[j]["chain"]
        swap = norm_num(answer_from_latents(model, tok, caps[i]["q"], latent_ids, donor)) if donor is not None else c["clean_ans"]
        c["swap_ans"] = swap; c["swap_changed"] = (swap != c["clean_ans"])
        # 更严格: 换 B 的潜在链, 答案是否 follow 成 B 的干净答案(定向因果=真草稿纸)
        c["donor_clean"] = caps[j]["clean_ans"]
        c["swap_follows_donor"] = (swap == caps[j]["clean_ans"])
        st = {}
        for b in betas:
            sa = norm_num(answer_from_latents(model, tok, caps[i]["q"], latent_ids, steer_chain(c["chain"], b)))
            st[str(b)] = {"ans": sa, "changed": sa != c["clean_ans"]}
        c["steer"] = st
        torch.cuda.empty_cache()

    os.makedirs(out_dir, exist_ok=True)
    # ---- 产出 3 份 json(与 colar 命名对齐, 便于 gen_comprehensive 集成) ----
    probe = [dict(q=c["q"], n_pass=c["depth"], answer=c["clean_ans"], answer_raw=c["clean_raw"],
                  passes=[{"top5": p} for p in c["passes"]],
                  content_frac=round(sum(is_content(p[0]) for p in c["passes"])/len(c["passes"]), 3) if c["passes"] else None)
             for c in caps]
    causal = [dict(q=c["q"], depth=c["depth"], clean_ans=c["clean_ans"], swap_ans=c["swap_ans"],
                   swap_changed=c["swap_changed"], donor_clean=c.get("donor_clean"),
                   swap_follows_donor=c.get("swap_follows_donor"), steer=c["steer"]) for c in caps]
    dd = [dict(source=c["source"], difficulty=c["difficulty"], depth=c["depth"],
               correct=c["correct"], gold=c["gold"], pred=c["clean_ans"]) for c in caps]
    json.dump(probe, open(f"{out_dir}/probe_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(causal, open(f"{out_dir}/causal_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(dd, open(f"{out_dir}/dd_{tag}_gsmllm.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 汇总打印 ----
    accs = [c["correct"] for c in caps]
    cf = [sum(is_content(p[0]) for p in c["passes"])/len(c["passes"]) for c in caps if c["passes"]]
    swap_rate = sum(c["swap_changed"] for c in caps)/len(caps)
    steer_rate = {b: round(sum(c["steer"].get(str(b), {}).get("changed", False) for c in caps)/len(caps), 2) for b in betas}
    # follow-donor: 只在 B答案≠A答案 的对上算(否则无法区分follow还是stay)
    diff_pairs = [c for c in caps if c.get("donor_clean") is not None and c["donor_clean"] != c["clean_ans"]]
    follow_rate = round(sum(c["swap_follows_donor"] for c in diff_pairs)/len(diff_pairs), 3) if diff_pairs else None
    stay_rate = round(sum(c["swap_ans"] == c["clean_ans"] for c in diff_pairs)/len(diff_pairs), 3) if diff_pairs else None
    def sub(mask):
        xs = [c["difficulty"] for c in caps if mask(c)]; ys = [c["depth"] for c in caps if mask(c)]
        return spearman(xs, ys), len(xs)
    print("\n==== Latent-SFT-1B 全套结果 ====")
    print(f"acc = {sum(accs)/len(accs):.3f} (n={len(accs)})")
    print(f"logit-lens content_frac 均值 = {round(sum(cf)/len(cf),3) if cf else 'NA'}")
    print(f"因果 token-swap 改变率 = {swap_rate:.2f}  (敏感性: 高=答案依赖潜在)")
    print(f"** follow-donor rate = {follow_rate}  (定向/更严格: 换B草稿纸->答案变成B答案; n={len(diff_pairs)}对B!=A)")
    print(f"   stay-self rate = {stay_rate}  (换B后仍=A答案的比例; 高=潜在没被follow=伪)")
    print(f"steering 改变率 = {steer_rate}")
    for name, m in [("correct", lambda c: c["correct"]), ("incorrect", lambda c: not c["correct"]), ("all", lambda c: True)]:
        sp, k = sub(m); print(f"难度->深度 [{name}] Spearman={sp} (n={k})")
    depths = [c["depth"] for c in caps]
    cap_hits = sum(1 for d in depths if d >= max_latent)
    print(f"depth 范围 [{min(depths)},{max(depths)}] 均值 {sum(depths)/len(depths):.1f}  撞cap({max_latent})={cap_hits}/{len(depths)}")
    # 跨数据集: 按源(难度递增 gsm8k<math500<aime)的平均深度 + acc
    srcs = sorted(set(c["source"] for c in caps), key=lambda s: sum(c["difficulty"] for c in caps if c["source"]==s)/max(1,sum(c["source"]==s for c in caps)))
    if len(srcs) > 1:
        print("跨数据集[难度递增] 源: 平均深度 / acc / n:")
        for s in srcs:
            g = [c for c in caps if c["source"]==s]
            print(f"  {s:8} depth均值 {sum(c['depth'] for c in g)/len(g):5.1f} | acc {sum(c['correct'] for c in g)/len(g):.2f} | n {len(g)}")
    print(f"写出: probe_{tag}.json / causal_{tag}.json / dd_{tag}_gsmllm.json")


def selftest():
    assert is_content("12") and is_content("+") and not is_content("the")
    assert grade("\\boxed{300}", "300") and grade("the answer is 42", 42) and not grade("7", "8")
    assert norm_num("\\boxed{1,200}") == "1200"
    s = spearman([1,2,3,4,5],[1,2,3,4,5]); assert s == 1.0, s
    s2 = spearman([1,2,3,4,5],[5,4,3,2,1]); assert s2 == -1.0, s2
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{CKPTS}/latent-sft-1b")
    ap.add_argument("--pool", default=str(HERE/"pools"/"gsmllm_pool.json"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per_source", type=int, default=0, help=">0 时每个 source 取该数量(跨数据集均衡采样)")
    ap.add_argument("--max_latent", type=int, default=128)
    ap.add_argument("--betas", default="2.0,4.0")
    ap.add_argument("--out", default=str(HERE/"outputs"))
    ap.add_argument("--tag", default="latentsft1b")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    global _KIND
    cl = a.ckpt.lower()
    _KIND = "qwen" if "qwen" in cl or "7b" in cl else ("deepseek" if "deepseek" in cl or "r1" in cl else "llama")
    print(f"template kind = {_KIND}", flush=True)
    betas = [float(b) for b in a.betas.split(",")]
    pool = json.load(open(a.pool, encoding="utf-8"))
    if isinstance(pool, dict): pool = pool.get("data") or list(pool.values())[0]
    if a.per_source > 0:   # 跨数据集均衡采样
        bys = {}
        for it in pool: bys.setdefault(it.get("source", "?"), []).append(it)
        items = []
        for s, g in bys.items(): items += g[:a.per_source]
        print(f"跨数据集采样: {a.per_source}/源 -> {len(items)} 题 ({ {s: min(a.per_source,len(g)) for s,g in bys.items()} })")
    else:
        items = pool[:a.n]
    model, tok, latent_ids = build_model(a.ckpt)
    run_all(model, tok, latent_ids, items, betas, a.out, a.tag, max_latent=a.max_latent)


if __name__ == "__main__":
    main()
