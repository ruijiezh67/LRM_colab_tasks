# -*- coding: utf-8 -*-
"""整合本轮"潜在推理是真是伪"所有测试 -> 综合 md + 总 json。"""
import json, os
B = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(B, "outputs")
def J(p):
    f = os.path.join(O, p)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None

# ---- 收集 ----
colar_causal = {
    "colar-gsm(下载·数学)": J("causal_colargsm.json"),
    "自训 llama×StrategyQA": J("causal_colar_llama_strategyqa.json"),
    "自训 r1×StrategyQA": J("causal_colar_r1_strategyqa.json"),
    "自训 llama×ProsQA": J("causal_colar_llama_prosqa.json"),
    "自训 r1×ProsQA": J("causal_colar_r1_prosqa.json"),
}
def swap_rate(r): return round(sum(x["swap_changed"] for x in r)/len(r), 2) if r else None
def steer_rate(r): return {b: round(sum(x["steer"][b]["changed"] for x in r)/len(r), 2) for b in r[0]["steer"]} if r else None

colar_probe = J("probe_colargsm.json")
coc = J("causal_coconut.json")
adap = J("adalr_probe.json")
adac = J("adalr_causal.json")
ac = adac["summary"] if adac else {}
ndeep = len([r for r in adap if r["depth"] >= 1]) if adap else 0

# ---- Latent-SFT-1B (DJCheng, 蒸馏+词表叠加, 无RL, 数学) ----
import math as _m
def _spear(xs, ys):
    n = len(xs)
    if n < 3: return None
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(v):
            j=i
            while j+1 < len(v) and v[o[j+1]]==v[o[i]]: j+=1
            for k in range(i,j+1): r[o[k]]=(i+j)/2.0
            i=j+1
        return r
    rx,ry=rk(xs),rk(ys); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx=_m.sqrt(sum((rx[i]-mx)**2 for i in range(n))); dy=_m.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return round(num/(dx*dy),3) if dx and dy else None
def load_lsft(tag):
    c = J(f"causal_{tag}.json"); p = J(f"probe_{tag}.json"); dd = J(f"dd_{tag}_gsmllm.json")
    if not (c and p and dd): return None
    cf = [x["content_frac"] for x in p if x.get("content_frac") is not None]
    cor = [r for r in dd if r["correct"]]
    depths = [r["depth"] for r in dd]
    return {
        "n": len(c), "acc": round(sum(r["correct"] for r in dd)/len(dd), 3),
        "content_frac": round(sum(cf)/len(cf), 3) if cf else None,
        "token_swap_change": swap_rate(c), "steering_change": steer_rate(c),
        "depth_range": [min(depths), max(depths)], "depth_cap_hits": sum(1 for d in depths if d >= 128),
        "dd_correct_spearman": _spear([r["difficulty"] for r in cor], [r["depth"] for r in cor]),
        "dd_all_spearman": _spear([r["difficulty"] for r in dd], [r["depth"] for r in dd]),
    }
lsft = load_lsft("latentsft1b"); lsft7 = load_lsft("latentsft7b"); grpo = load_lsft("latentgrpo1b")
lsft_p = J("probe_latentsft1b.json")

# ---- 综合 json ----
integ = {
  "meta": {
    "task": "潜在推理是真是伪: 5个CoLaR + Coconut + adaLR, 两轴(logit-lens解码 + 因果token-swap/steering), 跟 arXiv 2512.21711",
    "axes": {"logit_lens": "解码潜在隐向量看编码什么(可解释性)", "causal": "换/扰动潜在看答案变不变(因果, 原文法)"},
  },
  "results": {
    "colar": {name: {"n": len(r), "token_swap_change": swap_rate(r), "steering_change": steer_rate(r)} for name, r in colar_causal.items() if r},
    "colar_gsm_logit_lens_content_frac": 0.538,
    "latent_sft_1b": lsft,
    "latent_sft_7b": lsft7,
    "latent_grpo_1b": grpo,
    "SFT_vs_GRPO_ablation": "同方法同基座(Llama-1B词表叠加)唯一差RL: SFT(无RL)swap0.97/content0.646/深度3-26; GRPO(+RL)swap0.93/content0.782/深度5-9. => RL不增加潜在因果真实性(都真); RL实际作用=压缩潜在链(均12.1->5.5pass,为效率), 潜在仍因果; 副作用=深度范围变窄致难度->深度信号减弱(+0.667->+0.379). 钉死: 真潜在推理靠数学任务非RL",
    "latent_sft_note": "DJCheng Latent-SFT(2510.15522), 蒸馏+词表叠加态, 无RL, 数学, 满足两条件(自适应latent pass深度随题变+只输出\\boxed{}答案无文字CoT). 1B(Llama)逐项>=colar-gsm(swap0.97/content0.646/难度->深度correct+0.667)=真潜在推理; 7B(Qwen-Math,4bit)更准(acc0.55)+仍真(swap0.90)但潜在链长32-128、21/40撞128cap致难度->深度信号削弱(0.235非不自适应); 关键=纯蒸馏无RL也真=>数学任务本身逼出因果潜在, 非RL",
    "coconut_FULL_k12": {"logit_lens_content_frac": 0.48, "token_swap_change": round(sum(x["swap_changed"] for x in coc)/len(coc),3) if coc else None,
                          "note": "潜在占位; 自由生成吐文字CoT=真推理在文字里"},
    "adalr_LWtS": {"logit_lens_content_frac": 0.113, "depth_dist": f"math n=40: {40-ndeep}题depth0 / {ndeep}题depth>=1(范围0-4)",
                    "causal_depth>=1": {"control_replay_match": ac.get("control_replay_matches_clean_rate"),
                                        "token_swap_change": ac.get("token_swap_change_rate"),
                                        "steering_change": ac.get("steering_change_rate")},
                    "note": "多数题depth0(base直接答); depth>=1题的<CONTINUE>扰动/换后答案0%变=潜在因果惰性"},
  },
  "conclusion": {
    "spectrum": "真潜在推理=数学域现象: 两个数学横向模型强因果(colar-gsm swap0.95 + Latent-SFT-1B swap0.97, 都content_frac~0.5-0.65); 我们自训的非数学(逻辑/常识)弱(0.15-0.45); Coconut伪(0.13,文字里推理); adaLR潜在惰性(0.0)+多数不用",
    "honest_correction": "早前'CoLaR都真推理'不准确: 强的是数学域(colar-gsm+Latent-SFT); 自训的非数学模型潜在因果性弱-中(StrategyQA近摆设0.15/ProsQA中等0.30-0.45), 部分靠base直接答/捷径",
    "task_vs_RL": "关键结论(SFT-vs-GRPO直接消融钉死): 同方法同基座唯一差RL, Latent-SFT(无RL)swap0.97 vs Latent-GRPO(+RL)swap0.93 => RL不增加潜在因果真实性(都真); RL实际作用=压缩潜在链(12.1->5.5pass为效率). => 真潜在推理靠'数学任务逼出串行中间计算'(colar-gsm有RL/Latent-SFT无RL/Latent-GRPO有RL全真), 不是RL; 自训之所以弱 = 常识/逻辑任务不逼迫串行计算 + 数据小, 而非缺RL",
    "caveats": "logit-lens=可解释性非因果; Shortcut/OOD偏置训练未做(需重训); n=15-40; adaLR control_match=1.0已验证管道; Latent-SFT答案贪心解码(原文采样)",
  },
}
json.dump(integ, open(os.path.join(O, "thinking_reality_all_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- 综合 md ----
L = []
L.append("# 潜在推理是真是伪：全模型实证（CoLaR×5 / Coconut / adaLR，跟 arXiv 2512.21711）")
L.append("")
L.append("**问题**：这些「潜在推理」模型到底在潜在里真推理，还是潜在只是占位符（2512.21711 对 Coconut 的指控）？")
L.append("**方法（两轴）**：① **logit-lens** 解码每个潜在思维的隐向量看它编码啥（可解释性）；② **因果** = token-swap（换别题潜在看答案变不变）+ steering 扰动（原文因果法，纯推理复现）。GSM8K/ProsQA/StrategyQA/MATH。")
L.append("")
L.append("## 主结果矩阵（决定性看 ② token-swap 改变率）")
L.append("")
L.append("| 模型 | 机制 | ① logit-lens 内容 | ② token-swap 改变率 | steering | 判读 |")
L.append("|---|---|---|---|---|---|")
L.append("| **colar-gsm（下载·数学）** | CoLaR 替代式（SFT+**RL**） | 0.538（真算术） | **0.95** | 0.90/1.0 | **真潜在推理** |")
if lsft:
    L.append("| **Latent-SFT-1B（下载·数学）** | 蒸馏+词表叠加（**无RL**） | %.3f（真算术） | **%.2f** | %s | **真潜在推理（≥colar）** |" % (
        lsft["content_frac"], lsft["token_swap_change"],
        "/".join(str(lsft["steering_change"][b]) for b in sorted(lsft["steering_change"]))))
if grpo:
    L.append("| **Latent-GRPO-1B（下载·数学）** | 蒸馏+词表叠加（**+RL**·同SFT基座） | %.3f | **%.2f** | %s | **真潜在推理**（RL只压缩链长） |" % (
        grpo["content_frac"], grpo["token_swap_change"],
        "/".join(str(grpo["steering_change"][b]) for b in sorted(grpo["steering_change"]))))
if lsft7:
    L.append("| **Latent-SFT-7B（下载·数学·4bit）** | 蒸馏+词表叠加（**无RL**·Qwen-Math） | %.3f | **%.2f** | %s | **真潜在推理**（acc0.55更准） |" % (
        lsft7["content_frac"], lsft7["token_swap_change"],
        "/".join(str(lsft7["steering_change"][b]) for b in sorted(lsft7["steering_change"]))))
L.append("| 自训 r1×ProsQA | CoLaR 替代式 | — | **0.45** | 0.35/0.40 | 中等（部分真） |")
L.append("| 自训 llama×ProsQA | CoLaR 替代式 | — | **0.30** | 0.45/0.20 | 中等 |")
L.append("| 自训 llama×StrategyQA | CoLaR 替代式 | — | **0.15** | 0.35/0.65 | 弱（近摆设） |")
L.append("| 自训 r1×StrategyQA | CoLaR 替代式 | — | **0.15** | 0.05/0.20 | 弱 |")
L.append("| **Coconut FULL_k12** | latent token+hidden反馈 | 0.48（占位） | **0.13** | 0.40/0.53 | **伪**（真推理在文字CoT） |")
L.append("| **adaLR/LWtS** | CONTINUE token 追加 | 0.113（控制符） | **0.0**（对照match=1.0） | 0.0 | **潜在惰性**+多数不用 |")
L.append("")
L.append("## ⚠️ 核心诚实修正")
L.append("")
L.append("**「真潜在推理」罕见且难**。只有**下载的、训练充分的数学 colar-gsm** 强因果（swap 0.95、潜在解码真算术）。")
L.append("- **我们自训的 4 个 CoLaR 弱得多**（swap 0.15–0.45）：StrategyQA（acc~50% 近随机）潜在近摆设（0.15）；ProsQA（acc 90%）中等（0.30–0.45，部分题真用潜在、多数靠 base 直接答/结构捷径）。→ 早前「CoLaR 都真推理」**不准确**。")
L.append("- **Coconut = 伪**：潜在占位（swap 0.13），自由生成时吐完整文字 CoT——真推理在文字里、潜在是摆设。")
L.append("- **adaLR = 潜在因果惰性**：多数题 depth=0（base 直接答）；即使 depth≥1，扰动/换 `<CONTINUE>` hidden（4×范数）答案 **0% 变**（对照复现率 1.0 已验证管道）——潜在纯摆设。")
L.append("")
L.append("**谱系**：`数学域(colar-gsm + Latent-SFT-1B)强真 → 自训ProsQA 中 → 自训StrategyQA≈Coconut 弱/伪 → adaLR 惰性`。")
L.append("")
L.append("**新发现（任务 vs RL，SFT-vs-GRPO 直接消融钉死）**：同方法同基座唯一差 RL——Latent-SFT(无RL) swap **0.97** vs Latent-GRPO(+RL) swap **0.93**，两个都真 → **RL 不增加潜在因果真实性**。RL 的实际作用 = **压缩潜在链**（深度 12.1→5.5 pass，为效率）、潜在仍因果（副作用：范围变窄使难度→深度 +0.667→+0.379）。→ **真潜在推理靠「数学任务逼出串行中间计算」，不是 RL**（colar-gsm 有RL/Latent-SFT 无RL/Latent-GRPO 有RL 全真）；自训模型弱是因为**常识/逻辑任务不逼迫 + 数据小**，非缺 RL。")
L.append("")
L.append("## 逐模型证据")
L.append("")
if lsft:
    L.append("### 🔑 Latent-SFT-1B（真·无RL）：纯蒸馏的数学横向模型，逐项 ≥ colar-gsm")
    L.append("")
    L.append("**方法**：DJCheng Latent-SFT（arXiv 2510.15522），Llama-3.2-1B，潜在 token = 词表空间叠加态（top-10 加权 embedding），**只蒸馏、无 RL**；推理时模型自己预测 `</think>` 停 → 深度自适应。")
    L.append("")
    L.append("| 指标 | colar-gsm(有RL) | **Latent-SFT-1B(无RL)** |")
    L.append("|---|---|---|")
    L.append("| GSM8K acc | ~0.50 | %.3f |" % lsft["acc"])
    L.append("| logit-lens content_frac | 0.538 | **%.3f** |" % lsft["content_frac"])
    L.append("| token-swap 改变率(决定性) | 0.95 | **%.2f** |" % lsft["token_swap_change"])
    L.append("| steering 改变率 | 0.90/1.0 | %s |" % "/".join(str(lsft["steering_change"][b]) for b in sorted(lsft["steering_change"])))
    L.append("| 难度→深度 correct Spearman | ~+0.52 | **+%.3f** |" % lsft["dd_correct_spearman"])
    L.append("| 潜在深度范围(自适应) | 2–14 | %d–%d |" % (lsft["depth_range"][0], lsft["depth_range"][1]))
    L.append("")
    for x in (lsft_p or [])[:4]:
        t1 = " → ".join((p["top5"][0] if p["top5"] else "_") for p in x["passes"][:12])
        L.append("- `%s` → `%s`" % (t1, x.get("answer","")))
    L.append("")
    L.append("**🔑 这回答了「为什么自训的不真」**：Latent-SFT 无 RL，潜在因果照样强（swap 0.97）+ 难度→深度 correct +0.667。→ **真潜在推理靠「数学任务本身逼出串行中间计算」**（colar-gsm 有 RL、Latent-SFT 无 RL，两个数学模型都真）；**不是 RL**。自训的之所以弱 = **常识(StrategyQA yes/no)/逻辑(ProsQA)任务不逼迫串行计算 + 训练数据小**，而非缺 RL。")
    L.append("")
L.append("### colar-gsm（真）：潜在解码真实演化算术 + 换潜在答案 95% 变")
if colar_probe:
    for x in colar_probe[:4]:
        t1 = " → ".join((p["top5"][0] if p["top5"] else "_") for p in x["passes"])
        L.append("- `%s` → `%s`" % (t1, x.get("answer","")))
r = colar_causal["colar-gsm(下载·数学)"]
L.append("- token-swap: 换别题潜在 → 答案变 %d/%d" % (sum(x["swap_changed"] for x in r), len(r)))
L.append("")
L.append("### 自训 CoLaR（弱-中）：换潜在答案多数不变")
for name in ["自训 r1×ProsQA", "自训 llama×StrategyQA"]:
    r = colar_causal[name]
    exs = "; ".join("%s→%s(%s)" % (x["clean_ans"], x["swap_ans"], "变" if x["swap_changed"] else "同") for x in r[:4])
    L.append("- **%s** swap变 %d/%d: %s" % (name, sum(x["swap_changed"] for x in r), len(r), exs))
L.append("")
L.append("### Coconut（伪）：潜在占位；自由生成吐文字 CoT")
if coc:
    for x in coc[:3]:
        t1 = " → ".join(p[0] for p in x["passes"])
        L.append("- 潜在 `%s` → clean `%s`/swap `%s`(%s)" % (t1, x["clean_ans"], x["swap_ans"], "变" if x["swap_changed"] else "同"))
L.append("> 自由生成: *\"He cuts grass every 2 months because 4-2=2 inches and 2/0.5=4 months...\"* → 真推理在文字。")
L.append("")
L.append("### adaLR（惰性）：depth≥1 题扰动/换潜在 0% 变（对照 match=1.0）")
if adac:
    for x in adac["records"][:4]:
        L.append("- depth=%d clean=%s control=%s swap=%s steer=%s → 全同" % (x["depth"], x["clean"], x["control_replay"], x["swap"]["ans"], x["steer"].get("4.0",{}).get("ans")))
L.append("")
L.append("## 诚实边界")
L.append("- ① logit-lens=可解释性轴非因果；② token-swap/steering=原文因果法纯推理复现。")
L.append("- Coconut content_frac 0.48 被重复数字骗高（实占位）；adaLR 潜在偏控制符（learn-when-to-stop 设计）。")
L.append("- **Shortcut/OOD 偏置训练所有模型都需重训、未做**；n=15–40、每轴单一方法族；adaLR control_match=1.0 已排除管道 bug。")
# md 叙事已并入 DIFFICULTY_DEPTH.md §5.4 (核心3 md之一); 只产 json 数据工件。
# 需要独立可发 md 时取消下一行注释:
# open(os.path.join(O, "thinking_reality_verdict.md"), "w", encoding="utf-8").write("\n".join(L))
print("written thinking_reality_all_results.json  (md 叙事在 DIFFICULTY_DEPTH.md §5.4)")
print("CoLaR swap:", {k: swap_rate(v) for k, v in colar_causal.items() if v})
