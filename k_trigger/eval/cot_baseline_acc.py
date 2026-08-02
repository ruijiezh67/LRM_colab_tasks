"""测 CoT baseline 准确率(阶段①): base + LoRA(q/v r128) 生成文字 CoT+答案, 判分。
不含 latent_policy(那是阶段② CoLaR-SFT)。用法同 run_difficulty_depth_colar 的 env: COLAR_BASE/COLAR_CKPT。"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_adaptive_lrm_handoff", "inference_test"))
from run_difficulty_depth_colar import grade, load_pool  # 复用判分 + 载池

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    BASE, CK = os.environ["COLAR_BASE"], os.environ["COLAR_CKPT"]
    pool, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE); tok.add_special_tokens({"pad_token": "[PAD]"})
    llm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    llm.resize_token_embeddings(len(tok))
    llm = get_peft_model(llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
    # ckpt 键为 llm.* ; 本 peft 模型无 'llm.' 前缀 → 去掉
    sd = {k[4:] if k.startswith("llm.") else k: v for k, v in sd.items()}
    miss, unexp = llm.load_state_dict(sd, strict=False)
    bad = [k for k in unexp if "lora" in k.lower()]
    assert not bad, f"LoRA 键未匹配: {bad[:3]}"
    llm = llm.to(dev).eval()
    items = load_pool(pool, n)
    QT = "Question: {} Let's think step by step:(Thinking speed: 1)###"  # 官方 cot eval prompt(model_base.text_generate)
    GEN = dict(max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
    ok = 0; recs = []
    for i, it in enumerate(items):
        ids = tok(QT.format(str(it["question"]).rstrip()), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(dev)
        with torch.no_grad():
            out = llm.generate(ids, **GEN)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = txt.split("Answer:")[-1] if "Answer:" in txt else txt
        c = bool(grade(pred, it["gold"])); ok += c
        recs.append({"correct": c, "gold": it["gold"], "pred": pred[:80]})
        if i < 3: print(f"  [ex{i}] gold={it['gold']!r} pred={pred[:60]!r} -> {c}", flush=True)
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(items)} acc={ok/(i+1):.3f}", flush=True)
    print(f"\nCoT baseline acc = {ok/len(items):.3f}  (n={len(items)}, ckpt={os.path.basename(CK)})")

if __name__ == "__main__":
    main()
