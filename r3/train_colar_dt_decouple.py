# -*- coding: utf-8 -*-
r"""
train_colar_dt_decouple.py — Round-3: 良性双轨 (独立 CoT ∥ 真 latent, 显式解耦)
================================================================================
目标: 在 colar-gsm(唯一被证实有真 latent, native follows_donor=0.75) 之上, 建一个良性双轨:
    Q → { 独立 CoT  ∥  真 latent } → answer(只读 latent, sole-path)
且把 round-1/2 从未实现的 **CoT⊥latent 显式解耦** 第一次建出来。

round-1/2 为何是"假思考"(三层叠加):
  1) 弱 source: 用 R1 mathl123(native fd~0.10); 本脚本换 colar-gsm(Llama-1B, fd0.75)。
  2) 解码器错配: wrapper 用**全新** LoRA 覆盖 CoLaR 自己的 LoRA; 本脚本 warm-start 保 native
     LoRA+latent_policy, 解冻**续训**(绝不覆盖)。
  3) latent = CoT 的回声: teacher 硬编成 CoT(两轨最大耦合); 本脚本 latent 只由 CE_ans 共训、
     CoT 独立解码(CE_cot)、再加 **显式解耦项 L_decouple** 主动把两轨表征推开。

结构(三项 loss):
  · answer 轨: [q, live-latents, <eot>, answer] + block_q(答案只读 [latent,<eot>], 连 Q 都挡)
               → CE_ans。latent 由 latent_policy **现场可微生成**(不 detach), latent_policy+LoRA 解冻。
  · CoT 轨(独立前向, 与 latent 互不 attend): [q, <cot>, cot, </cot>] 标准 causal → CE_cot。
  · 解耦: r_lat=mean(live-latents)[H], r_cot=mean(CoT 位 last-hidden)[H];
          **linear CKA(r_lat, r_cot) 最小化**(空间无关, 抗缩放/旋转钻空子)。
  总 loss = CE_ans + cot_w·CE_cot + lam_dec·CKA。

语义后果(设计使然, 非 bug): latent⊥CoT 且 answer=f(latent) ⇒ **可读 CoT 在构造上不解释答案**。
这正是"latent 能否承载 CoT 监控看不见的推理"的良性受控 testbed(呼应 Evading-CoT-Monitoring 2608.02820)。

golden-rule 监控(每轮 md §0 必带): loss 逐项落盘 + held-out {fd, matters, acc, coupling(CKA),
probe_R², cot_ce} 同步。判据 = CE_ans↓ ∧ fd↑ ∧ coupling↓ ∧ CoT 通顺, **四者缺一即"假"**。
best-ckpt 选 **fd**(不是 loss, 不是 p2_asr —— 单看 loss 会被脚手架/回声骗)。

colar-gsm 对齐(与 strong_causal_colar.py 一致, 否则潜链不停/fd 假高):
  · prompt = "Question: {} Let's think step by step:(Thinking speed: N)###"
  · COLAR_EMB_STD=0.018 (Llama-1B native 值, **非** R1 的 auto=0.03008)
  · sep = "###"; 加载时断言 '###' 是单一 token + latent_policy/lora 键真加载。

env: COLAR_BASE COLAR_CKPT COLAR_COMPRESS(5) COLAR_EMB_STD(0.018) COLAR_MAXLAT(64)
     TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  TRANSFORMERS_VERBOSITY=error
run: python train_colar_dt_decouple.py --data gsm_math.jsonl --ckpt colar_best.ckpt \
        --output_dir out --epochs 3 --batch 8 --cot_w 1.0 --lam_dec 0.1 --decouple cka
selftest(纯 python, 无 torch): python train_colar_dt_decouple.py --selftest
"""
import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
COT_OPEN, COT_CLOSE = "<cot>", "</cot>"


# =========================================================================== #
#  纯 python 逻辑(--selftest 直接调用, 无 torch/GPU/网络)
# =========================================================================== #
def k_from_cot(cot_len, compress):
    """自适应深度: latent 链长 K = ceil(cot_len / compress), 至少 1。难题 CoT 长 -> K 大 -> 深。"""
    return max(1, int(math.ceil(cot_len / max(1, compress))))


def chunks(seq, n):
    """把 seq 切成大小 n 的 minibatch(最后一批可能更短)。"""
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def stage1_keep_mask(P, K, La):
    """纯 python bool keep-mask(True=可 attend), answer 轨几何:
    序列 = [q(P), latent(K), <eot>(1), answer(La)]; S = P+K+1+La。
    答案产出行 = [eot_pos, S); block_q: 答案行**不读问题 Q**[0,P)。
    => 答案只 attend [latent, <eot>](+已生成答案前缀), latent 是唯一信息路径。"""
    S = P + K + 1 + La
    eot_pos = P + K
    keep = [[j <= i for j in range(S)] for i in range(S)]          # causal
    for i in range(eot_pos, S):
        for j in range(0, P):
            keep[i][j] = False                                    # block_q
    return keep, S, eot_pos


def _linear_cka_py(X, Y):
    """纯 python 线性 CKA(仅 selftest 用; torch 版 linear_cka 与此同式)。
    CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F · ||Yc^T Yc||_F), Xc/Yc = 列去均值。
    空间无关(X:[B,dx], Y:[B,dy] 维度可不同), 值 ∈[0,1], 1=表征结构全同, 0=无关。"""
    B = len(X)
    def center(M):
        d = len(M[0])
        mean = [sum(M[r][c] for r in range(B)) / B for c in range(d)]
        return [[M[r][c] - mean[c] for c in range(d)] for r in range(B)]
    Xc, Yc = center(X), center(Y)
    def fro2(A, Bm):                                              # ||A^T B||_F^2
        da, db = len(A[0]), len(Bm[0])
        s = 0.0
        for i in range(da):
            for j in range(db):
                v = sum(A[r][i] * Bm[r][j] for r in range(B))
                s += v * v
        return s
    xy, xx, yy = fro2(Xc, Yc), fro2(Xc, Xc), fro2(Yc, Yc)
    return xy / ((xx ** 0.5) * (yy ** 0.5) + 1e-12)


def _selftest():
    # (1) 自适应深度 K
    assert k_from_cot(0, 5) == 1 and k_from_cot(5, 5) == 1 and k_from_cot(6, 5) == 2
    assert k_from_cot(50, 5) == 10 and k_from_cot(3, 1) == 3
    # (2) minibatch 切分
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    # (3) answer 轨 mask 几何: P=2,K=3,La=2 -> S=8, eot=5
    keep, S, eot = stage1_keep_mask(2, 3, 2)
    assert S == 8 and eot == 5
    assert [j for j in range(S) if keep[5][j]] == [2, 3, 4, 5], "答案首行只读 latent+<eot>(block_q 挡 Q)"
    assert [j for j in range(S) if keep[7][j]] == [2, 3, 4, 5, 6, 7], "答案末行读 latent+<eot>+答案前缀, 不读 Q"
    assert [j for j in range(S) if keep[2][j]] == [0, 1, 2], "latent 首行因果读 Q"
    for i in range(S):
        assert any(keep[i]), ("空行", i)
    # (4) 线性 CKA 性质: 自身=1, 对称, ∈[0,1], 强相关高/弱相关低
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [0.0, 2.0]]
    Y = [[0.9, 0.1], [2.1, 0.0], [2.9, 1.2], [0.1, 1.8]]          # ≈X(不同空间尺度)
    Z = [[3.0, 0.0], [1.0, 0.0], [0.0, 3.0], [2.0, 1.0]]          # 打乱结构
    assert abs(_linear_cka_py(X, X) - 1.0) < 1e-9, "CKA(X,X)=1"
    assert abs(_linear_cka_py(X, Y) - _linear_cka_py(Y, X)) < 1e-9, "对称"
    for A in (Y, Z):
        v = _linear_cka_py(X, A)
        assert -1e-9 <= v <= 1.0 + 1e-9, ("范围", v)
    assert _linear_cka_py(X, Y) > _linear_cka_py(X, Z), "相似结构 CKA 更高"
    print("[selftest] OK —— K 自适应; minibatch 切分; answer 轨 block_q(答案只读 latent+<eot>); "
          "linear CKA(自身=1/对称/[0,1]/结构相似更高)。解耦项将最小化 CKA(r_lat, r_cot)。")


# =========================================================================== #
#  加载: warm-start colar-gsm(base + native LoRA + latent_policy), 解冻续训(绝不覆盖)
# =========================================================================== #
def load_colar_warmstart(base, ckpt, dev):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    tok = AutoTokenizer.from_pretrained(base)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    llm.resize_token_embeddings(len(tok))
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    H = llm.config.hidden_size                                    # Llama-1B=2048; R1-1.5B=1536

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
    cont = nn.Module(); cont.llm = llm; cont.latent_policy = lp
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    miss, _ = cont.load_state_dict(sd, strict=False)
    assert not [k for k in miss if "latent_policy" in k or "lora" in k.lower()], \
        "ckpt 键不符(base/latent_policy/lora 应匹配 —— 检查 COLAR_BASE 是否 colar-gsm 的 Llama-1B)"

    sep_id = tok.convert_tokens_to_ids("###")
    unk_id = getattr(tok, "unk_token_id", None)
    assert sep_id is not None and int(sep_id) >= 0, "'###' 未解析为 token id(检查底座 tokenizer)"
    if unk_id is not None and sep_id == unk_id:
        print("!! [warn] '###' -> unk_token: 底座分词器不把 '###' 当单一 token, 潜链停止判据会失效", flush=True)
    n_added = tok.add_special_tokens({"additional_special_tokens": [COT_OPEN, COT_CLOSE]})
    if n_added > 0:
        llm.resize_token_embeddings(len(tok))
    lp = lp.to(dev).float()      # latent_policy 保 float32(与 strong_causal 一致): live_latents 喂 .float() 输入,
    llm = llm.to(dev)            # 输出再 .to(mdtype) 转 bf16 喂模型。若 bf16 会 F.linear dtype 冲突(Float vs BFloat16)。

    # ---- 解冻: latent_policy 全部 + LoRA(lora_) + 词嵌入; 其余 base 冻结 ----
    for p in llm.parameters():
        p.requires_grad_(False)
    trn = []
    for n, p in llm.named_parameters():
        if "lora_" in n.lower():
            p.requires_grad_(True); trn.append(p)
    emb = llm.get_input_embeddings()
    emb.weight.requires_grad_(True); trn.append(emb.weight)
    for p in lp.parameters():
        p.requires_grad_(True); trn.append(p)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else sep_id
    pad_id = tok.convert_tokens_to_ids("[PAD]")
    ids = dict(sep_id=sep_id, eos_id=eos_id, pad_id=pad_id, H=H,
               cot_open_id=tok.convert_tokens_to_ids(COT_OPEN),
               cot_close_id=tok.convert_tokens_to_ids(COT_CLOSE))
    return llm, lp, tok, emb, ids, trn


# =========================================================================== #
#  可微 live latent 生成 —— = strong_causal.gen 的 latent 链但**不 detach**(grad 流进 lp+LoRA)
# =========================================================================== #
def live_latents(llm, lp, emb, q_ids, K, emb_std, dev):
    """现场生成 K 个 latent 嵌入, 保梯度。KV cache 复用 q 前缀; policy.mean 确定性(可微)。
    返回 [1,K,H](requires_grad, 依赖 lp+LoRA)。"""
    import torch
    am = torch.ones_like(q_ids)
    pos = torch.arange(q_ids.shape[1], device=dev).unsqueeze(0)
    out = llm(inputs_embeds=emb(q_ids), attention_mask=am, position_ids=pos,
              output_hidden_states=True, use_cache=True)
    pkv = out.past_key_values
    cur = pos[:, -1:]
    lats = []
    mdtype = emb.weight.dtype
    for _ in range(K):
        ce = (lp(out.hidden_states[-1][:, -1:, :].float()).mean * emb_std).to(mdtype)   # NO detach
        lats.append(ce)
        am = torch.cat([am, torch.ones(1, 1, device=dev, dtype=am.dtype)], 1)
        cur = cur + 1
        out = llm(inputs_embeds=ce, attention_mask=am, position_ids=cur,
                  past_key_values=pkv, output_hidden_states=True, use_cache=True)
        pkv = out.past_key_values
    return torch.cat(lats, dim=1)                                 # [1,K,H]


def stage1_mask_add(P, K, La, dev, mdtype):
    """answer 轨 4D 加性 mask [1,1,S,S]: causal & 答案行不读 Q(block_q)。与 stage1_keep_mask 同义。"""
    import torch
    S = P + K + 1 + La
    eot_pos = P + K
    keep = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev))
    rows = torch.arange(S, device=dev).view(S, 1)
    cols = torch.arange(S, device=dev).view(1, S)
    blockq = (rows >= eot_pos) & (cols < P)
    keep = keep & ~blockq
    add = torch.zeros(1, 1, S, S, dtype=mdtype, device=dev)
    add.masked_fill_(~keep.view(1, 1, S, S), torch.finfo(mdtype).min)
    return add, S, eot_pos


# =========================================================================== #
#  answer 轨 CE(teacher-forced): [q, live-latents, <eot>, answer], 答案只读 [latent,<eot>]
#  返回 (ce_ans, r_lat) —— r_lat = mean(live-latents) [H], 供解耦。
# =========================================================================== #
def answer_ce(llm, emb, ids, q_ids, lats, ans_ids, dev):
    import torch
    import torch.nn.functional as F
    mdtype = emb.weight.dtype
    P, K, La = q_ids.shape[1], lats.shape[1], len(ans_ids)
    eot_e = emb(torch.tensor([[ids["sep_id"]]], device=dev))
    ans_e = emb(torch.tensor([ans_ids], device=dev))
    seq = torch.cat([emb(q_ids), lats, eot_e, ans_e], dim=1)      # [1,S,H]
    add, S, _ = stage1_mask_add(P, K, La, dev, mdtype)
    pos = torch.arange(S, device=dev).unsqueeze(0)
    logits = llm(inputs_embeds=seq, attention_mask=add, position_ids=pos).logits[0]   # [S,V]
    ans_start = P + K + 1
    pred = logits[ans_start - 1: S - 1, :].float()                # shift: <eot> 起预测 answer
    tgt = torch.tensor(ans_ids, device=dev)
    ce = F.cross_entropy(pred, tgt)
    r_lat = lats[0].float().mean(0)                               # [H]
    return ce, r_lat


# =========================================================================== #
#  CoT 轨 CE(独立前向, 标准 causal, 与 latent 互不 attend): [q, <cot>, cot, </cot>]
#  返回 (ce_cot, r_cot) —— r_cot = mean(CoT 位 last-hidden) [H], 供解耦。
# =========================================================================== #
def cot_ce(llm, emb, ids, q_ids, cot_ids, dev):
    import torch
    import torch.nn.functional as F
    P, Lc = q_ids.shape[1], len(cot_ids)
    open_e = emb(torch.tensor([[ids["cot_open_id"]]], device=dev))
    close_e = emb(torch.tensor([[ids["cot_close_id"]]], device=dev))
    cot_e = emb(torch.tensor([cot_ids], device=dev))
    seq = torch.cat([emb(q_ids), open_e, cot_e, close_e], dim=1)  # [1, P+1+Lc+1, H]
    S = seq.shape[1]
    am = torch.ones(1, S, device=dev, dtype=torch.long)           # 2D -> HF 自动 causal
    pos = torch.arange(S, device=dev).unsqueeze(0)
    out = llm(inputs_embeds=seq, attention_mask=am, position_ids=pos, output_hidden_states=True)
    logits = out.logits[0]
    cot_start = P + 1                                             # 第一个 CoT token 位置
    pred = logits[cot_start - 1: cot_start - 1 + Lc, :].float()   # <cot> 行起预测 cot_ids
    tgt = torch.tensor(cot_ids, device=dev)
    ce = F.cross_entropy(pred, tgt)
    r_cot = out.hidden_states[-1][0, cot_start: cot_start + Lc, :].float().mean(0)     # [H]
    return ce, r_cot


def linear_cka(X, Y, eps=1e-6):
    """torch 线性 CKA(与 _linear_cka_py 同式), X:[B,dx] Y:[B,dy], 列去均值, ∈[0,1]。可微。"""
    import torch
    X = X.float(); Y = Y.float()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.t() @ Y).pow(2).sum()
    xx = (X.t() @ X).pow(2).sum()
    yy = (Y.t() @ Y).pow(2).sum()
    return xy / (xx.sqrt() * yy.sqrt() + eps)


@__import__("contextlib").contextmanager
def _evalmode(llm):
    was = llm.training
    llm.eval()
    try:
        yield
    finally:
        llm.train(was)


def gen_answer_greedy(llm, emb, ids, q_ids, lats, dev, max_new=12):
    """部署/评测: 给 [q, latents, <eot>] 前缀贪心生成答案(每步重建 block_q mask)。返回 token ids。"""
    import torch
    mdtype = emb.weight.dtype
    P, K = q_ids.shape[1], lats.shape[1]
    eot_e = emb(torch.tensor([[ids["sep_id"]]], device=dev))
    cur = torch.cat([emb(q_ids), lats, eot_e], dim=1)
    out_ids = []
    for _ in range(max_new):
        La = cur.shape[1] - (P + K + 1)
        add, S, _ = stage1_mask_add(P, K, La, dev, mdtype)
        pos = torch.arange(S, device=dev).unsqueeze(0)
        lg = llm(inputs_embeds=cur, attention_mask=add, position_ids=pos).logits
        nid = int(lg[0, -1].argmax(-1))
        if nid == ids["eos_id"]:
            break
        out_ids.append(nid)
        cur = torch.cat([cur, emb(torch.tensor([[nid]], device=dev))], dim=1)
    return out_ids


# =========================================================================== #
#  held-out 评测: acc + follows_donor + matters + coupling(CKA) + probe_R² + cot_ce
# =========================================================================== #
def evaluate(llm, lp, tok, emb, ids, rows, emb_std, dev, eval_n, decouple_on):
    import re
    import torch

    def norm(t):
        m = re.search(r"\\boxed\{(.+?)\}", str(t)) or re.search(r"(-?\d+(?:\.\d+)?)", str(t))
        return re.sub(r"[,$\s]", "", (m.group(1) if m else str(t)))     # 归一化: 抽 boxed/数字 + 去逗号空白

    def grade(pred, gold):
        g = norm(gold)                                                  # ★ 两边都 norm(旧版只 norm pred -> acc/fd 恒 0)
        return g != "" and norm(pred) == g

    with torch.no_grad(), _evalmode(llm):
        ev = rows[: min(eval_n, len(rows))]
        rlat_e, rcot_e, ce_cot_e = [], [], []
        for r in ev:
            r["lats_eval"] = live_latents(llm, lp, emb, r["q_ids"], r["K"], emb_std, dev)
            r["ans_own"] = tok.decode(gen_answer_greedy(llm, emb, ids, r["q_ids"], r["lats_eval"], dev),
                                      skip_special_tokens=True)
            rlat_e.append(r["lats_eval"][0].float().mean(0))
            if r["cot_ids"]:
                cc, rc = cot_ce(llm, emb, ids, r["q_ids"], r["cot_ids"], dev)
                rcot_e.append(rc); ce_cot_e.append(float(cc)); r["_has_cot"] = True
            else:
                r["_has_cot"] = False
        acc = sum(grade(r["ans_own"], r["answer"]) for r in ev) / len(ev)
        pairs = list(zip(ev, ev[1:] + ev[:1]))
        follows = matters = n = 0
        for A, B in pairs:
            swap = tok.decode(gen_answer_greedy(llm, emb, ids, A["q_ids"], B["lats_eval"], dev),
                              skip_special_tokens=True)
            emptyv = torch.zeros(1, 0, A["lats_eval"].shape[2], device=dev, dtype=emb.weight.dtype)
            noL = tok.decode(gen_answer_greedy(llm, emb, ids, A["q_ids"], emptyv, dev),
                             skip_special_tokens=True)
            follows += int(grade(swap, B["ans_own"]))
            matters += int(norm(noL) != norm(A["ans_own"]))
            n += 1
        # coupling: CKA(r_lat, r_cot) over 有 CoT 的 eval 例(只取有 rc 的对齐 rlat)
        coupling = probe_r2 = None
        rlat_paired = [rlat_e[i] for i, r in enumerate(ev) if r["_has_cot"]]
        if decouple_on and len(rcot_e) >= 2:
            Rl = torch.stack(rlat_paired)                         # [m,H]
            Rc = torch.stack(rcot_e)                              # [m,H]
            coupling = float(linear_cka(Rl, Rc))
            # post-hoc 线性探针(★held-out): fit 前半, 报后半 R²。低/负 = latent 线性预测不出 CoT = 解耦。
            # (旧版用训练 R², m<H 必过拟合到 1.0 = 退化无意义)
            try:
                h = Rl.shape[0] // 2
                if h >= 2:
                    Xtr = torch.cat([Rl[:h], torch.ones(h, 1, device=dev)], 1).float()
                    W = torch.linalg.lstsq(Xtr, Rc[:h].float()).solution
                    Xte = torch.cat([Rl[h:], torch.ones(Rl.shape[0] - h, 1, device=dev)], 1).float()
                    pr = Xte @ W
                    ss_res = (Rc[h:].float() - pr).pow(2).sum()
                    ss_tot = (Rc[h:].float() - Rc[:h].float().mean(0, keepdim=True)).pow(2).sum()
                    probe_r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
                else:
                    probe_r2 = None
            except Exception:
                probe_r2 = None
    m = dict(acc_own=round(acc, 3), follows_donor=round(follows / n, 3) if n else None,
             matters=round(matters / n, 3) if n else None, n_pairs=n, n_eval=len(ev),
             coupling_cka=(round(coupling, 4) if coupling is not None else None),
             probe_r2=(round(probe_r2, 4) if probe_r2 is not None else None),
             cot_ce=(round(sum(ce_cot_e) / len(ce_cot_e), 4) if ce_cot_e else None))
    return m


# =========================================================================== #
#  主流程
# =========================================================================== #
def run(a):
    import random
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = os.environ["COLAR_BASE"]
    ckpt = a.ckpt or os.environ.get("COLAR_CKPT")
    COMPRESS = int(os.environ.get("COLAR_COMPRESS", "5"))
    MAXLAT = int(os.environ.get("COLAR_MAXLAT", "64"))

    llm, lp, tok, emb, ids, trn = load_colar_warmstart(base, ckpt, dev)
    _esr = os.environ.get("COLAR_EMB_STD", "0.018")               # colar-gsm(Llama-1B) native = 0.018
    emb_std = (float(emb.weight.detach().float().std()) if str(_esr).lower() == "auto" else float(_esr))
    eos_id = ids["eos_id"]
    decouple_on = (a.decouple != "off")
    n_trn = sum(p.numel() for p in trn)
    print(f"[dt-decouple] warm-start colar-gsm; unfrozen latent_policy+LoRA+emb = {n_trn:,} params; "
          f"H={ids['H']} compress={COMPRESS} emb_std={emb_std:.5f} maxlat={MAXLAT} "
          f"decouple={a.decouple} cot_w={a.cot_w} lam_dec={a.lam_dec} batch={a.batch}", flush=True)

    # colar-gsm 对齐的 prompt(与 strong_causal_colar.py 一致)
    QT, SPEED = "Question: {} Let's think step by step:", "(Thinking speed: {})"

    def q_to_ids(q):
        text = QT.format(str(q).rstrip()) + SPEED.format(COMPRESS) + "###"
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

    def strip_think(c):
        c = str(c)
        for t in ("<think>", "</think>"):
            c = c.replace(t, "")
        return c.strip()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.max_examples:
        rows = rows[:a.max_examples]
    for r in rows:
        r["q_ids"] = q_to_ids(r["question"])
        r["cot_ids"] = tok(strip_think(r["cot"]), add_special_tokens=False)["input_ids"] if r.get("cot") else []
        r["K"] = min(MAXLAT, k_from_cot(len(r["cot_ids"]), COMPRESS))
        r["ans_ids"] = tok(r["answer"], add_special_tokens=False)["input_ids"] + [eos_id]
    n_cot = sum(1 for r in rows if r["cot_ids"])
    print(f"[dt-decouple] {len(rows)} 例(有 CoT {n_cot}); K min/mean/max = "
          f"{min(r['K'] for r in rows)}/{sum(r['K'] for r in rows)/len(rows):.1f}/{max(r['K'] for r in rows)}",
          flush=True)

    opt = torch.optim.AdamW(trn, lr=a.lr, weight_decay=0.0)
    llm.train()
    os.makedirs(a.output_dir, exist_ok=True)
    loss_log = os.path.join(a.output_dir, "loss.jsonl")
    eval_log = os.path.join(a.output_dir, "eval.jsonl")
    idx = list(range(len(rows)))
    step = 0
    best_fd = -1.0

    def save(path, note):
        torch.save({"latent_policy": lp.state_dict(),
                    "lora": {n: p.detach().cpu() for n, p in llm.named_parameters() if "lora_" in n.lower()},
                    "emb": emb.weight.detach().cpu(),
                    "cfg": {"base": base, "compress": COMPRESS, "emb_std": emb_std, "maxlat": MAXLAT,
                            "round": 3, "decouple": a.decouple, "cot_w": a.cot_w, "lam_dec": a.lam_dec,
                            "note": note}},
                   path)

    for epoch in range(a.epochs):
        random.Random(a.seed + epoch).shuffle(idx)
        for mb in chunks(idx, a.batch):
            opt.zero_grad()
            ce_a_list, ce_c_list, rl_pair, rc_pair = [], [], [], []
            for i in mb:
                r = rows[i]
                lats = live_latents(llm, lp, emb, r["q_ids"], r["K"], emb_std, dev)   # 可微
                ce_a, r_lat = answer_ce(llm, emb, ids, r["q_ids"], lats, r["ans_ids"], dev)
                ce_a_list.append(ce_a)
                if r["cot_ids"]:
                    ce_c, r_cot = cot_ce(llm, emb, ids, r["q_ids"], r["cot_ids"], dev)
                    ce_c_list.append(ce_c)
                    rl_pair.append(r_lat); rc_pair.append(r_cot)   # 按例配对(同一 example 的两轨)
            L_ans = torch.stack(ce_a_list).mean()
            L_cot = torch.stack(ce_c_list).mean() if ce_c_list else L_ans.new_zeros(())
            if decouple_on and len(rc_pair) >= 2:
                L_dec = linear_cka(torch.stack(rl_pair), torch.stack(rc_pair))
            else:
                L_dec = L_ans.new_zeros(())
            total = L_ans + a.cot_w * L_cot + a.lam_dec * L_dec
            total.backward()
            torch.nn.utils.clip_grad_norm_(trn, 1.0)
            opt.step()
            step += 1
            if step % a.log_every == 0:
                rec = {"epoch": epoch, "step": step, "total": float(total),
                       "ce_ans": float(L_ans), "ce_cot": float(L_cot), "l_decouple": float(L_dec)}
                print(f"[dt-decouple] e{epoch} s{step} total={rec['total']:.4f} "
                      f"ce_ans={rec['ce_ans']:.4f} ce_cot={rec['ce_cot']:.4f} cka={rec['l_decouple']:.4f}",
                      flush=True)
                with open(loss_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            if dev == "cuda":
                del ce_a_list, ce_c_list, rl_pair, rc_pair, L_ans, L_cot, L_dec, total

        # ---- 每 eval_every epoch: held-out 评 + best-by-fd ----
        if (epoch + 1) % a.eval_every == 0 or epoch == a.epochs - 1:
            m = evaluate(llm, lp, tok, emb, ids, rows, emb_std, dev, a.eval_n, decouple_on)
            m["epoch"] = epoch
            print(f"[eval] e{epoch} acc={m['acc_own']} fd={m['follows_donor']} matters={m['matters']} "
                  f"coupling_cka={m['coupling_cka']} probe_r2={m['probe_r2']} cot_ce={m['cot_ce']} "
                  f"(n_pairs={m['n_pairs']})", flush=True)
            print("  golden-rule: 需 CE_ans↓ ∧ fd↑ ∧ coupling↓ ∧ CoT 通顺, 缺一即'假'", flush=True)
            with open(eval_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(m) + "\n")
            fd = m["follows_donor"] if m["follows_donor"] is not None else -1.0
            if fd > best_fd:
                best_fd = fd
                save(os.path.join(a.output_dir, "dt_decouple_best.pt"), f"best-by-fd @e{epoch} fd={fd}")
                json.dump(m, open(os.path.join(a.output_dir, "dt_decouple_best_eval.json"), "w"), indent=1)
                print(f"  -> new best fd={fd} saved", flush=True)

    save(os.path.join(a.output_dir, "dt_decouple_final.pt"), "final epoch")
    mf = evaluate(llm, lp, tok, emb, ids, rows, emb_std, dev, a.eval_n, decouple_on)
    json.dump(mf, open(os.path.join(a.output_dir, "dt_decouple_final_eval.json"), "w"), indent=1)
    print(f"\n==== round-3 dual-track decouple 结束: best_fd={best_fd} final={mf} ====", flush=True)
    print(f"[dt-decouple] saved best={a.output_dir}/dt_decouple_best.pt final={a.output_dir}/dt_decouple_final.pt",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--ckpt", default=None, help="colar-gsm .ckpt(warm-start); 覆盖 COLAR_CKPT")
    ap.add_argument("--output_dir", default="/content/colar_dt_decouple")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4, help="latent_policy+LoRA 共训 lr")
    ap.add_argument("--batch", type=int, default=8, help="minibatch(CKA 需 >=2 有 CoT 例; OOM 就调小)")
    ap.add_argument("--cot_w", type=float, default=1.0, help="CoT 轨 CE 权重")
    ap.add_argument("--lam_dec", type=float, default=0.1, help="解耦 CKA 惩罚权重")
    ap.add_argument("--decouple", choices=["cka", "off"], default="cka", help="off=消融(退回无解耦)")
    ap.add_argument("--max_examples", type=int, default=0, help="0=all(小样本先验 fd 起没起)")
    ap.add_argument("--eval_n", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=1, help="每几 epoch 评一次(best-by-fd)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        run(a)
