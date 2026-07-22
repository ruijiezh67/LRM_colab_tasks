# -*- coding: utf-8 -*-
r"""
verify_dualtrack_colar.py
=========================
② 验证步骤: 核实 train_colar_dualtrack.py 训出来的 CoLaR "双轨(dual-track)" 模型确实满足
三条性质。双轨模型的一条序列(推理期)按训练脚本 line 312 的装配顺序重建:

    [question ids emb]  <bot=末尾"###">  [FROZEN latent chain]  <eot="###"sep>  <cot>  可见CoT  </cot>  answer

关键的 BOTTLENECK 注意力掩码: answer 查询行只能看 [question, latent, <eot>], 绝不能看 <cot>..</cot>。
本脚本按 train 脚本 EXACT 几何重建模型(base+PAD+<cot>/</cot> resize / r=128 q/v LoRA / latent_policy /
4D 掩码), 但加载的是 output_dir 里 **训练过的** LoRA 适配器 + cot_embeds.pt 的两行 cot 嵌入。

三条检查(写入 JSON):
  1. format_ok   : 模型是否吐出 <cot>...</cot> 段 + 可解析的数字答案 (fraction)。
  2. latent 因果 : 复用 strong_causal_colar 的判据(latent_matters / follows_donor / ignores_latent /
                   stays_self); baseline(下载版 colar) = 0.958 / 0.75。
  3. mask_holds  : 对正确样本, 换掉可见 CoT 文本后, 掩码 ON 重生成答案应 **不变**(掩码生效);
                   掩码 OFF(全因果无 bottleneck)同样换 CoT 则答案 **更容易变**(泄漏)。
                   报告 answer_unchanged_maskON / answer_changed_maskOFF。

env: COLAR_BASE=.../Llama-3.2-1B-Instruct  COLAR_CKPT=.../colar-gsm/colar_best.ckpt
     COLAR_EMB_STD=0.018  COLAR_COMPRESS=5  COLAR_MAXLAT=64  (可选 COLAR_GENTOK/COLAR_ANSTOK)
run: PYTHONIOENCODING=utf-8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 <py> verify_dualtrack_colar.py --n 40
selftest(纯 python, 无 torch/GPU): <py> verify_dualtrack_colar.py --selftest
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


# --------------------------------------------------------------------------- #
#  判分 helper —— 从 strong_causal_colar.py 原样复制(不搞可导入化, 保持 ponytail)
# --------------------------------------------------------------------------- #
def _norm(s):
    return re.sub(r"[^a-z0-9.]", "", str(s).split("Answer:")[-1].lower())


def _last_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", str(s))
    return m[-1].replace(",", "") if m else None


def grade(pred, gold):
    p, g = _norm(pred), _norm(re.sub(r",", "", str(gold)))
    try:
        return abs(float(p) - float(g)) < 1e-4            # strong_causal 原判据(纯数字)
    except Exception:
        pass
    np_, ng = _last_num(pred), _last_num(gold)            # 回退: 从(已norm的)答案里抠末位数字
    if np_ is not None and ng is not None:
        try:
            return abs(float(np_) - float(ng)) < 1e-4
        except Exception:
            pass
    return p == g


# =========================================================================== #
#  纯 python 几何 —— 从 train_colar_dualtrack.py line 79-108 原样复制(compute_layout)
#  必须逐字一致, 否则整条验证无意义。
# =========================================================================== #
def compute_layout(P, L, Lc, La):
    """train line 79-108 verbatim: 单条双轨序列的位置布局(单 token 的 eot/<cot>/</cot>)。
        [0, P)  question 前缀("...###"; 末尾 "###" == <bot>)
        [P, P+L) FROZEN latent embeds; eot_pos=P+L "###"==<eot>; cot_open_pos <cot>;
        [.., ..+Lc) 可见 CoT; ccot_pos </cot>; answer_start answer+eos"""
    latent_start = P
    latent_end = P + L
    eot_pos = P + L
    cot_open_pos = eot_pos + 1
    cot_content_start = cot_open_pos + 1
    ccot_pos = cot_content_start + Lc
    answer_start = ccot_pos + 1
    S = answer_start + La
    return {
        "P": P, "L": L, "Lc": Lc, "La": La,
        "latent_start": latent_start, "latent_end": latent_end,
        "eot_pos": eot_pos, "cot_open_pos": cot_open_pos,
        "cot_content_start": cot_content_start, "ccot_pos": ccot_pos,
        "answer_start": answer_start, "S": S,
        # answer 查询行 **不能** attend 的 key 列 = [<cot> .. </cot>] 闭区间
        "cot_key_span": (cot_open_pos, answer_start),
        # 被 bottleneck 的查询行: 从 </cot> 行(第一个 answer 产出者)到序列末尾
        "answer_query_span": (ccot_pos, S),
    }


# =========================================================================== #
#  纯 python 版 4D bottleneck keep-mask —— 精确复刻 train line 138-156 的
#  build_bottleneck_mask_torch(含 pad 行对角保护), 供 --selftest 在无 torch 的
#  Windows 上验证几何。逐行对应关系见下方注释。
# =========================================================================== #
def py_bottleneck_keep(B, S, valid_lens, cot_key_spans, answer_query_spans):
    """keep[b][i][j]=True 表示行 i(query) 可以 attend 列 j(key)。
    对应 train:
        causal = tril(ones)                         -> (j<=i)
        keep &= not_pad.unsqueeze(1)                 -> not_pad[j] (屏蔽 pad KEY 列)
        keep &= ~(answer_query x cot_key)            -> 双轨 bottleneck 块清零
        keep |= eye & pad_row                        -> pad 行强制保留对角(避免空行/NaN)"""
    out = []
    for b in range(B):
        vl = valid_lens[b]
        not_pad = [j < vl for j in range(S)]
        keep = [[(j <= i) and not_pad[j] for j in range(S)] for i in range(S)]   # causal & not-pad KEY
        k0, k1 = cot_key_spans[b]
        q0, q1 = answer_query_spans[b]
        for i in range(q0, q1):                     # answer_query x cot_key -> False
            for j in range(k0, k1):
                keep[i][j] = False
        for i in range(S):                          # pad 行 OR 对角(self)
            if not not_pad[i]:
                keep[i][i] = True
        out.append(keep)
    return out


# =========================================================================== #
#  torch 版 4D bottleneck keep-mask —— 从 train line 138-156 原样复制。
#  (SDPA 需要 4D bool mask; 训练/推理必须一致)
# =========================================================================== #
def build_bottleneck_mask_torch(B, S, pad_mask_1d, cot_key_span, answer_query_span, device):
    """4-D bool keep-mask [B,1,S,S] —— 几何复制自 train_colar_dualtrack.build_bottleneck_mask_torch。"""
    import torch
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    keep = causal.unsqueeze(0).expand(B, S, S).clone()
    not_pad = pad_mask_1d.bool()
    keep = keep & not_pad.unsqueeze(1)                                   # block pad KEYS
    rows = torch.arange(S, device=device).view(1, S, 1)
    cols = torch.arange(S, device=device).view(1, 1, S)
    for b in range(B):
        k0, k1 = cot_key_span[b]
        q0, q1 = answer_query_span[b]
        block = ((rows >= q0) & (rows < q1) & (cols >= k0) & (cols < k1)).squeeze(0)
        keep[b] = keep[b] & ~block
    eye = torch.eye(S, dtype=torch.bool, device=device)
    pad_row = (~not_pad).unsqueeze(-1)                                   # force pad rows diagonal
    keep = keep | (eye.unsqueeze(0) & pad_row)
    return keep.unsqueeze(1)                                             # [B,1,S,S]


def _resize(llm, n):
    """train line 159-163 verbatim。"""
    try:
        llm.resize_token_embeddings(n)
    except Exception:
        llm.base_model.model.resize_token_embeddings(n)


# =========================================================================== #
#  模型重建 —— 逐步对齐 train build_model(line 166-230), 差异只在:
#    * LoRA 用 output_dir 里 **训练过的** 适配器覆盖(不是随机初始化)
#    * 两行 cot 嵌入用 cot_embeds.pt 覆盖(不是从 "###" 初始化)
#  加载顺序必须 = train: PAD resize -> get_peft_model -> load ckpt(strict=False, 取
#  base+embedding+latent_policy+colar lora) -> 加 <cot>/</cot> + resize -> 覆盖训练 LoRA -> 覆盖 cot 行。
# =========================================================================== #
def build_dualtrack(base, ckpt, out_dir, dev):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    EMB_STD = float(os.environ.get("COLAR_EMB_STD", "0.018"))   # 与 train/strong_causal 一致(此处未直接用, 保留语义)

    # ---- 1) tok: 先加 [PAD](与 train line 174-175 一致) ------------------------
    tok = AutoTokenizer.from_pretrained(base)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    # ---- 2) base(bf16 + sdpa) + resize 到 len(tok)(仅含 PAD)------------------
    llm = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")   # SDPA req. for 4-D mask
    llm.resize_token_embeddings(len(tok))
    # ---- 3) 同一份 LoraConfig(r=128, q/v)(train line 179-180)-----------------
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    H = llm.config.hidden_size

    # ---- 4) 同一份 LatentPolicy(与 strong_causal_colar / train line 183-189 一致)
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
    # ---- 5) 加载 colar ckpt(strict=False): 取 base+embedding+latent_policy+colar lora
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    miss, _ = cont.load_state_dict(sd, strict=False)
    assert not [k for k in miss if "latent_policy" in k or "lora" in k.lower()], "ckpt 键不符"

    sep_id = tok.convert_tokens_to_ids("###")
    # ---- 6) 加唯一新增 token <cot>/</cot> 并 resize(train line 197-201)---------
    n_added = tok.add_special_tokens({"additional_special_tokens": [COT_OPEN, COT_CLOSE]})
    if n_added > 0:
        _resize(llm, len(tok))
    cot_open_id = tok.convert_tokens_to_ids(COT_OPEN)
    cot_close_id = tok.convert_tokens_to_ids(COT_CLOSE)

    # ---- 7) 用 output_dir 里 **训练过的** LoRA 覆盖(save_pretrained 存的 adapter)----
    #   save_pretrained 存的 key 形如 ...q_proj.lora_A.weight(单 default 适配器省略名字);
    #   模型内参数名是 ...q_proj.lora_A.default.weight -> 在 .weight 前插 .default 精确对齐。
    #   只覆盖 lora_ 张量; embedding 由 ckpt + cot_embeds.pt 负责。
    asd = _load_adapter_sd(out_dir)
    own = dict(llm.named_parameters())
    loaded = 0
    with torch.no_grad():
        for k, v in asd.items():
            if "lora_" not in k:
                continue
            cand = k[:-len(".weight")] + ".default.weight" if k.endswith(".weight") else k
            tgt = own.get(cand)
            if tgt is None:
                tgt = own.get(k)
            if tgt is not None:
                tgt.data.copy_(v.to(tgt.dtype))
                loaded += 1
    assert loaded > 0, f"没从 {out_dir} 载入任何训练过的 LoRA 张量(检查 adapter_model.safetensors)"

    # ---- 8) 用 cot_embeds.pt 覆盖两行 cot 嵌入(train line 378-381 存的)----------
    blob = torch.load(os.path.join(out_dir, "cot_embeds.pt"), map_location="cpu")
    cot_open_id = int(blob["cot_open_id"])          # 以存档 id 为准
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
    print(f"[verify] 重建完成: 训练 LoRA 张量 {loaded} 个已载入, cot 行已覆盖 "
          f"(sep={sep_id} <cot>={cot_open_id} </cot>={cot_close_id})", flush=True)
    return llm, lp, tok, emb, ids


def _load_adapter_sd(out_dir):
    """读 output_dir 里训练过的 LoRA 适配器 state_dict(safetensors 优先, 退回 .bin)。"""
    import torch
    p_sft = os.path.join(out_dir, "adapter_model.safetensors")
    p_bin = os.path.join(out_dir, "adapter_model.bin")
    if os.path.exists(p_sft):
        from safetensors.torch import load_file
        return load_file(p_sft)
    return torch.load(p_bin, map_location="cpu")


# =========================================================================== #
#  推理: latent 链 + 双轨装配 + 生成
# =========================================================================== #
def make_runtime(llm, lp, tok, emb, ids, dev):
    """返回三个闭包: latent_chain / gen_full(自由生成 CoT+answer) / gen_answer(带掩码只生成答案)。"""
    import torch
    sep_id, cot_open_id, cot_close_id = ids["sep_id"], ids["cot_open_id"], ids["cot_close_id"]
    pad_id, eos_id, H, EMB_STD = ids["pad_id"], ids["eos_id"], ids["H"], ids["emb_std"]
    mdtype = emb.weight.dtype
    MAX_LAT = int(os.environ.get("COLAR_MAXLAT", "64"))
    GENTOK = int(os.environ.get("COLAR_GENTOK", "160"))      # CoT+answer 自由生成上限
    ANSTOK = int(os.environ.get("COLAR_ANSTOK", "48"))       # 只生成答案的上限
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5"))

    eot_e = emb(torch.tensor([[sep_id]], device=dev))        # <eot> = "###" sep 的嵌入 [1,1,H]
    co_e = emb(torch.tensor([[cot_open_id]], device=dev))    # <cot>
    cc_e = emb(torch.tensor([[cot_close_id]], device=dev))   # </cot>

    def q_to_ids(question):
        text = QT.format(str(question).rstrip()) + SPEED.format(COMPRESS) + "###"   # train line 285
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

    def latent_chain(q_ids, inject=None):
        """复用 strong_causal_colar.gen 的 latent loop, 但确定性: policy.mean + argmax 停在 sep_id。"""
        am = torch.ones_like(q_ids)
        pos = torch.arange(q_ids.shape[1], device=dev).unsqueeze(0)
        qemb = emb(q_ids)
        out = llm(inputs_embeds=qemb, attention_mask=am, position_ids=pos,
                  output_hidden_states=True, use_cache=True)
        pkv = out.past_key_values
        cur = pos[:, -1:]
        lats = []
        n = len(inject) if inject is not None else MAX_LAT     # inject=[] -> n=0 = 空链
        for k in range(n):
            if inject is not None:
                ce = inject[k].to(qemb.dtype)
            else:                                              # 确定性: .mean 而非 .rsample
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
        """双轨自由生成(标准因果): [q, latent, <eot>, <cot>] -> 模型续写 CoT </cot> answer。
        返回 (生成 token id 列表, lats)。装配顺序对齐 train line 312 的前缀部分。"""
        lats = latent_chain(q_ids, inject)
        prefix = torch.cat([emb(q_ids)] + lats + [eot_e, co_e], dim=1)   # [1, P+L+1+1, H]
        amf = torch.ones(1, prefix.shape[1], device=dev, dtype=torch.long)
        pred = llm.generate(inputs_embeds=prefix, attention_mask=amf,
                            max_new_tokens=GENTOK, do_sample=False, pad_token_id=pad_id)
        return pred[0].tolist(), lats

    def parse_gen(gen_ids):
        """从自由生成的 token(前缀已含 <cot>, 故 gen_ids 从 CoT 内容开始)切出 cot_ids 与答案文本。"""
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

    @torch.no_grad()
    def gen_answer(q_ids, lats, cot_ids, mask_on, max_ans=ANSTOK):
        """给定 [q, latent, <eot>, <cot>, cot_ids, </cot>] 前缀, 逐 token 生成答案。
        装配顺序对齐 train line 312: [q_e, lat, eot_e, co_e, cot_e, cc_e, (answer)]。
        4D bottleneck: answer 查询行不能 attend cot_key_span=[<cot>..</cot>](train cot_key_span/answer_query_span)。
        每步重建全序列的 4D 掩码(无 KV cache: 4D 掩码逐步变化, 重算最稳)。"""
        P, L, Lc = q_ids.shape[1], len(lats), len(cot_ids)
        lay = compute_layout(P, L, Lc, 1)                # La=1 占位; cot_key_span/ccot_pos 与 La 无关
        cks = lay["cot_key_span"]                        # (cot_open_pos, answer_start)
        ccot = lay["ccot_pos"]                           # </cot> 行 = 第一个 answer 产出者
        cot_e = emb(torch.tensor([cot_ids], device=dev)) if Lc else eot_e.new_zeros(1, 0, H)
        prefix = torch.cat([emb(q_ids)] + lats + [eot_e, co_e, cot_e, cc_e], dim=1)   # 末 token = </cot> @ ccot
        cur = prefix
        out_ids = []
        for _ in range(max_ans):
            T = cur.shape[1]
            aq = (ccot, T) if mask_on else (T, T)        # mask_on: bottleneck; off: 空区间=纯因果
            keep = build_bottleneck_mask_torch(
                1, T, torch.ones(1, T, dtype=torch.long, device=dev), [cks], [aq], dev)
            add = torch.zeros(keep.shape, dtype=mdtype, device=dev)
            add.masked_fill_(~keep, torch.finfo(mdtype).min)     # bool keep -> additive(train line 330-331)
            pos = torch.arange(T, device=dev).unsqueeze(0)
            o = llm(inputs_embeds=cur, attention_mask=add, position_ids=pos)
            nid = int(o.logits[0, -1].argmax(-1))
            if nid == eos_id:
                break
            out_ids.append(nid)
            cur = torch.cat([cur, emb(torch.tensor([[nid]], device=dev))], dim=1)
        return _norm(tok.decode(out_ids, skip_special_tokens=True))

    return q_to_ids, gen_full, parse_gen, gen_answer


# =========================================================================== #
#  主流程
# =========================================================================== #
def run(a):
    import torch
    base = os.environ["COLAR_BASE"]
    ckpt = os.environ.get("COLAR_CKPT")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    llm, lp, tok, emb, ids = build_dualtrack(base, ckpt, a.out_dir, dev)
    q_to_ids, gen_full, parse_gen, gen_answer = make_runtime(llm, lp, tok, emb, ids, dev)

    pool = json.load(open(a.pool, encoding="utf-8"))
    if isinstance(pool, dict):
        pool = pool.get("data") or list(pool.values())[0]
    items = pool[:a.n]

    # ---- pass 1: 每题自由生成(自身 latent)+ 空链控制 ---------------------------
    caps = []
    for it in items:
        q_ids = q_to_ids(it["question"])
        gid, lats = gen_full(q_ids)                        # 自由生成一趟拿可见 CoT(仅供格式检查)
        cot_ids, ans_text, closed = parse_gen(gid)
        # 答案在瓶颈 mask 下生成(匹配训练/部署: 答案只 attend latent+题目, 读不到 CoT)
        clean = gen_answer(q_ids, lats, cot_ids, mask_on=True)
        empty = gen_answer(q_ids, [], cot_ids, mask_on=True)   # 空 latent 链
        clean_off = gen_answer(q_ids, lats, cot_ids, mask_on=False)  # 对照: 掩码关(答案能看 CoT 文字)
        correct = grade(clean, it.get("gold"))
        correct_off = grade(clean_off, it.get("gold"))
        fmt = bool(closed and len(cot_ids) > 0)            # 会吐 <cot>..</cot>(格式轨)
        caps.append(dict(q=it["question"], q_ids=q_ids, gold=it.get("gold"), lats=lats,
                         cot_ids=cot_ids, clean=clean, empty=empty, correct=correct,
                         correct_off=correct_off, fmt=fmt))
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- 检查 1: format_ok ---------------------------------------------------
    format_ok = round(sum(c["fmt"] for c in caps) / len(caps), 3) if caps else None

    # ---- 检查 2: latent 因果(复用 strong_causal_colar 判据, line 89-105)-------
    a1 = []
    for i, c in enumerate(caps):
        donor = None
        for d in range(1, len(caps)):
            cj = caps[(i + d) % len(caps)]
            if cj["correct"] and c["correct"] and cj["clean"] != c["clean"] and cj["lats"]:
                donor = cj
                break
        if donor is None:
            continue
        swap = gen_answer(c["q_ids"], donor["lats"], c["cot_ids"], mask_on=True)   # 换 donor B latent 链, 瓶颈答案
        a1.append(dict(A_clean=c["clean"], A_empty=c["empty"], B_donor=donor["clean"], swap=swap,
                       follows_donor=(swap == donor["clean"]), stays_self=(swap == c["clean"]),
                       latent_matters=(c["empty"] != c["clean"]),
                       ignores_latent=(c["empty"] == c["clean"] == swap)))
        if dev == "cuda":
            torch.cuda.empty_cache()
    nb = len(a1)
    f1 = lambda k: round(sum(x[k] for x in a1) / nb, 3) if nb else None

    # ---- 检查 2b: 部分-swap 漂移(前k/后k) —— 交大 2512.21711 式因果判据 --------
    #   只把 A 链的一部分换成 donor B 的对应部分, 看答案随替换比例 frac 从 A_clean 漂到 B_donor;
    #   证 latent 逐 pass 累积决定答案(比整条全换的 follows_donor 更符合论文判据)。照 strong_causal.py L64-82。
    corr2 = [c for c in caps if c["correct"] and c["lats"]]
    FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
    a2 = []
    for i in range(len(corr2)):
        c = corr2[i]
        donor = None
        for d in range(1, len(corr2)):
            cj = corr2[(i + d) % len(corr2)]
            if cj is not c and cj["clean"] != c["clean"] and cj["lats"]:
                donor = cj; break
        if donor is None:
            continue
        La, Lb = len(c["lats"]), len(donor["lats"]); Lmin = min(La, Lb)
        row = {"A_clean": c["clean"], "B_donor": donor["clean"], "prefix": {}, "suffix": {}}
        for frac in FRACS:
            k = int(round(frac * Lmin))
            pre = (donor["lats"][:k] + c["lats"][k:]) if k > 0 else list(c["lats"])         # A 前k -> B 前k
            row["prefix"][f"{frac}"] = gen_answer(c["q_ids"], pre, c["cot_ids"], mask_on=True)
            suf = (c["lats"][:La - k] + donor["lats"][Lb - k:]) if k > 0 else list(c["lats"])  # A 后k -> B 后k
            row["suffix"][f"{frac}"] = gen_answer(c["q_ids"], suf, c["cot_ids"], mask_on=True)
        a2.append(row)
        if dev == "cuda":
            torch.cuda.empty_cache()
        if len(a2) >= 8:                       # 8 对足够看漂移曲线
            break

    def _curve(side):                          # 每个 frac 上 答案==A / ==B 的比例
        out = {}
        n = len(a2)
        for frac in FRACS:
            k = f"{frac}"
            isA = sum(r[side][k] == r["A_clean"] for r in a2)
            isB = sum(r[side][k] == r["B_donor"] for r in a2)
            out[k] = {"isA": round(isA / n, 3) if n else None, "isB": round(isB / n, 3) if n else None}
        return out
    partial_swap = {
        "n_pairs": len(a2), "fracs": FRACS,
        "prefix_curve": _curve("prefix"), "suffix_curve": _curve("suffix"),
        "读法": "isA 随 frac 从高降低、isB 从低升高 => 答案随替换比例从 A 漂到 B, 证 latent 逐 pass 累积决定答案(部分因果, 交大式判据)。",
        "examples": a2[:4],
    }

    # ---- 检查 3: mask_holds --------------------------------------------------
    #   对正确样本, 用别的样本的 CoT 替换可见 CoT: 掩码 ON 答案应不变; 掩码 OFF 更易变。
    corr = [c for c in caps if c["correct"] and c["cot_ids"]]
    m3 = []
    for idx, c in enumerate(corr):
        donor = None
        for d in range(1, len(corr)):
            cj = corr[(idx + d) % len(corr)]
            if cj is not c and cj["cot_ids"]:
                donor = cj
                break
        if donor is None:
            continue
        own_on = gen_answer(c["q_ids"], c["lats"], c["cot_ids"], mask_on=True)
        swp_on = gen_answer(c["q_ids"], c["lats"], donor["cot_ids"], mask_on=True)
        own_off = gen_answer(c["q_ids"], c["lats"], c["cot_ids"], mask_on=False)
        swp_off = gen_answer(c["q_ids"], c["lats"], donor["cot_ids"], mask_on=False)
        m3.append(dict(own_on=own_on, swap_on=swp_on, own_off=own_off, swap_off=swp_off,
                       unchanged_maskON=(own_on == swp_on), changed_maskOFF=(own_off != swp_off)))
        if dev == "cuda":
            torch.cuda.empty_cache()
    nm = len(m3)
    unchanged_on = round(sum(x["unchanged_maskON"] for x in m3) / nm, 3) if nm else None
    changed_off = round(sum(x["changed_maskOFF"] for x in m3) / nm, 3) if nm else None

    res = {
        "tag": a.tag, "n": len(caps),
        "acc": round(sum(c["correct"] for c in caps) / len(caps), 3) if caps else None,
        "acc_mask_on": round(sum(c["correct"] for c in caps) / len(caps), 3) if caps else None,
        "acc_mask_off": round(sum(c["correct_off"] for c in caps) / len(caps), 3) if caps else None,
        "format_ok": format_ok,
        "latent_causal": {"n_pairs": nb, "latent_matters": f1("latent_matters"),
                          "ignores_latent": f1("ignores_latent"), "follows_donor": f1("follows_donor"),
                          "stays_self": f1("stays_self"), "examples": a1[:6]},
        "partial_swap_drift": partial_swap,
        "mask_holds": {"n": nm, "answer_unchanged_maskON": unchanged_on,
                       "answer_changed_maskOFF": changed_off, "examples": m3[:6]},
    }
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, f"verify_dualtrack_{a.tag}.json")
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n==== dual-track 验证 [{a.tag}] acc={res['acc']} format_ok={format_ok} ====")
    print(f"acc mask_ON={res['acc_mask_on']} mask_OFF={res['acc_mask_off']} "
          f"(OFF>ON => 掩码切断了 CoT 拐杖: 答案被逼只靠 latent, 掉的那部分=拐杖贡献)")
    print(f"latent: matters={f1('latent_matters')} follows_donor={f1('follows_donor')} "
          f"ignores={f1('ignores_latent')} (both-correct 对={nb})")
    print(f"mask_holds: unchanged_maskON={unchanged_on} changed_maskOFF={changed_off} (n={nm})")
    _pc = partial_swap["prefix_curve"]
    print(f"部分-swap(前k) isB: frac0={_pc['0.0']['isB']} .25={_pc['0.25']['isB']} .5={_pc['0.5']['isB']} "
          f".75={_pc['0.75']['isB']} 1.0={_pc['1.0']['isB']}  (n_pairs={partial_swap['n_pairs']}) -> {outp}")


# =========================================================================== #
#  --selftest : 纯 python, 无 torch/GPU —— 4D bottleneck 掩码几何
# =========================================================================== #
def _selftest():
    P, L, Lc, La = 5, 3, 4, 3
    lay = compute_layout(P, L, Lc, La)
    S = lay["S"]
    assert S == P + L + 1 + 1 + Lc + 1 + La, S      # 长度自洽
    PADN = 2                                        # 追加 2 个 pad 位, 造一条被填充的序列
    Spad = S + PADN
    valid_lens = [S]
    cks = [lay["cot_key_span"]]
    aqs = [lay["answer_query_span"]]
    keep = py_bottleneck_keep(1, Spad, valid_lens, cks, aqs)[0]

    q0, q1 = lay["answer_query_span"]
    k0, k1 = lay["cot_key_span"]
    eot_pos = lay["eot_pos"]

    # (1) answer x cot 块被完全屏蔽
    for i in range(q0, q1):
        for j in range(k0, k1):
            assert keep[i][j] is False, ("leak answerxcot", i, j)
    # (2) answer 查询行仍可读 latent(这是 latent 保持因果的原因)
    for i in range(q0, q1):
        assert all(keep[i][p] for p in range(lay["latent_start"], lay["latent_end"])), ("latent unread", i)
    # (3) 因果上三角(有效区)保持屏蔽
    for i in range(S):
        for j in range(i + 1, Spad):
            assert keep[i][j] is False, ("noncausal", i, j)
    # (4) 第一个 answer 行(</cot> 行)最远只能看到 <eot>(读 question+latent+<eot>)
    attended = [j for j in range(Spad) if keep[q0][j]]
    assert max(attended) == eot_pos, (max(attended), eot_pos)
    assert keep[q0][eot_pos] and keep[q0][0], "首个 answer 行须能看到 <eot> 与 question 起点"
    # (5) 无空行(每行至少 attend 一个 key, 防 NaN)
    for i in range(Spad):
        assert any(keep[i]), ("empty row", i)
    # (6) pad 行/列: pad 位置只保留自身对角(pad KEY 列除对角外全 False; pad 行对角=True)
    for p in range(S, Spad):
        assert keep[p][p] is True, ("pad diag missing", p)                 # pad 行对角保留
        for i in range(Spad):
            if i != p:
                assert keep[i][p] is False, ("pad key leak", i, p)         # pad 列除对角全屏蔽

    print("[selftest] OK —— 4D bottleneck 几何成立: answer 查询行对 <cot>..</cot> 零泄漏、"
          "首个 answer 行只读 [question, latent, <eot>]、latent 恒可读、因果完好、无空行、"
          "pad 位置仅保留对角(pad 行对角=True, pad 列除对角全屏蔽)。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/content/colar_dualtrack",
                    help="训练好的 dual-track 目录(含训练 LoRA + cot_embeds.pt + dualtrack_config.json)")
    ap.add_argument("--pool", default=str(KT / "pools" / "gsmllm_pool.json"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(HERE.parent.parent / "outputs"))
    ap.add_argument("--tag", default="colar_dualtrack")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        run(a)
