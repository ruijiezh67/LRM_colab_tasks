# -*- coding: utf-8 -*-
r"""Section B orchestration — train TWO GPT-2 Coconut models that differ ONLY in whether
residual TEXT CoT survives, then re-run the token-swap causality probe on both.

  * STANDARD   (gsm_coconut_standard.yaml)   : curriculum stops at scheduled_stage == max_latent_stage
                                               -> residual text CoT is KEPT (Meta official recipe).
  * PURELATENT (gsm_coconut_purelatent.yaml) : num_epochs bumped so a stage reaches
                                               scheduled_stage > max_latent_stage -> dataset.py takes the
                                               skip-all branch (n_skip_steps=10000) -> ALL text CoT dropped.
                                               pad_latent_to_max=True keeps K = max_latent_stage*c_thought.

HYPOTHESIS (2512.21711 says Coconut latents are "pseudo" / non-causal):
  - if that is a residual-text-CoT-CRUTCH artifact  -> PURELATENT shows HIGHER token_swap_change / follow
  - if it is INTRINSIC to the feedback mechanism     -> PURELATENT stays equally low (stay ~ high)

Pipeline per track:  train (coconut/run.py) -> convert (raw ckpt -> hf_export) -> test (probe_causal_coconut).
The probe already computes swap_ans with donor=(i+1)%n; we post-process follow/stay from its JSON.

Paths are env-configurable so this runs on Colab (/content) and locally (E:/ckpts):
  COCONUT_ROOT  dir holding run.py + dataset.py + data/  (default: <repo>/coconut ; on Colab e.g. /content/latent_reasoning_security/coconut)
  CKPTS_ROOT    root for checkpoints + hf_export         (default: E:/ckpts   ; on Colab e.g. /content/ckpts)
  PROBE         path to probe_causal_coconut.py          (default: <repo>/k_trigger/probe_causal_coconut.py)
  PYTHON        python exe for the sub-jobs              (default: this interpreter)

Usage (Colab A100):
  !COCONUT_ROOT=/content/latent_reasoning_security/coconut CKPTS_ROOT=/content/ckpts \
      python coconut_dualtrack.py --step all
  # or one stage at a time:
  python coconut_dualtrack.py --step train_standard
  python coconut_dualtrack.py --step train_purelatent
  python coconut_dualtrack.py --step convert
  python coconut_dualtrack.py --step test
  python coconut_dualtrack.py --selftest        # pure-python config-diff validation (no torch/GPU)

Note: coconut/run.py imports `wandb`; we run the sub-jobs with WANDB disabled so no login is needed
(`pip install wandb` if the import is missing on Colab).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../dual_track_backdoor/py
DTB  = HERE.parent                               # .../dual_track_backdoor
KT   = DTB.parent                                # .../k_trigger
REPO = KT.parent                                 # project root

# ---- env-configurable paths -------------------------------------------------
COCONUT_ROOT = Path(os.environ.get("COCONUT_ROOT", str(REPO / "coconut")))
CKPTS_ROOT   = Path(os.environ.get("CKPTS_ROOT", "E:/ckpts"))
PROBE        = Path(os.environ.get("PROBE", str(KT / "probe_causal_coconut.py")))
PYTHON       = os.environ.get("PYTHON", sys.executable)
OUT_DIR      = DTB / "outputs"

STD_YAML = HERE / "gsm_coconut_standard.yaml"
PL_YAML  = HERE / "gsm_coconut_purelatent.yaml"

# keys that are bookkeeping (identity / paths), not scientific behaviour — ignored by the diff selftest
IGNORE_DIFF = {"name", "save_path", "project"}
# the ONLY behavioural keys allowed to differ between the two recipes
ALLOWED_DIFF = {"num_epochs", "pad_latent_to_max"}


# ---------------------------------------------------------------------------- #
# config helpers
# ---------------------------------------------------------------------------- #
def load_yaml(p: Path) -> dict:
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def K_of(cfg: dict) -> int:
    """Latent width the probe must use = max_latent_stage * c_thought (both recipes -> 6)."""
    return int(cfg["max_latent_stage"]) * int(cfg["c_thought"])


def save_dir_of(cfg: dict) -> Path:
    """Where run.py writes checkpoint_N (save_path is relative to COCONUT_ROOT / run.py cwd)."""
    sp = Path(cfg["save_path"])
    if not sp.is_absolute():
        sp = COCONUT_ROOT / sp
    return sp / cfg["name"]


def latest_checkpoint(save_dir: Path):
    if not save_dir.exists():
        return None
    cks = [f for f in save_dir.iterdir() if f.name.startswith("checkpoint_")]
    if not cks:
        return None
    return max(cks, key=lambda f: int(f.name.split("_")[1]))


def sub_env() -> dict:
    e = dict(os.environ)
    e.update(
        WANDB_MODE="offline", WANDB_DISABLED="true", WANDB_SILENT="true",
        PYTHONIOENCODING="utf-8", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    return e


def sh(cmd, cwd=None):
    print(f"\n$ (cwd={cwd or os.getcwd()})\n  " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=sub_env())
    if r.returncode != 0:
        raise SystemExit(f"sub-job failed (exit {r.returncode}): {cmd}")


# ---------------------------------------------------------------------------- #
# 1) train  — shell out to coconut/run.py <yaml>  (run.py auto-resumes from save_dir)
# ---------------------------------------------------------------------------- #
def train(yaml_path: Path, force: bool = False):
    cfg = load_yaml(yaml_path)
    sd = save_dir_of(cfg)
    final = sd / f"checkpoint_{cfg['num_epochs']}"
    if final.exists() and not force:
        print(f"[train] {cfg['name']}: final {final.name} already present -> skip (use --force to retrain).")
        return
    # run.py itself detects an existing partial run in save_dir and resumes (ignores yaml `resume`),
    # so simply (re-)invoking it continues a preempted job. Entry = single positional yaml arg.
    run_py = COCONUT_ROOT / "run.py"
    if not run_py.exists():
        raise SystemExit(f"[train] run.py not found at {run_py} — set COCONUT_ROOT to the coconut repo dir.")
    sh([PYTHON, run_py.name, str(yaml_path)], cwd=COCONUT_ROOT)
    print(f"[train] {cfg['name']}: done -> {sd}")


# ---------------------------------------------------------------------------- #
# 2) convert  — raw Coconut state_dict (checkpoint_N) -> hf_export dir the probe loads
#    (safetensors + tokenizer + coconut_meta.json). Inline + prefix-robust: GPT-2 is tiny so we skip
#    the meta-device path in drh/utils/convert_ckpt_to_hf.py (that script is the heavyweight Qwen alt).
# ---------------------------------------------------------------------------- #
SPECIAL = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]   # SAME add-order as run.py -> stable ids
_WRAP_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.")


def _strip_wrappers(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _WRAP_PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def convert(ckpt_file: Path, out_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = "openai-community/gpt2"

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for t in SPECIAL:
        tok.add_tokens(t)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tok))

    print(f"[convert] loading state_dict {ckpt_file}")
    try:
        sd = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    clean, dropped = {}, 0
    for k, v in sd.items():
        nk = _strip_wrappers(k)
        # Coconut aliases base input-embeddings as self.embedding (dup of transformer.wte) — drop it.
        if nk == "embedding.weight":
            dropped += 1
            continue
        if nk.startswith("base_causallm."):
            nk = nk[len("base_causallm."):]
        clean[nk] = v

    miss = model.load_state_dict(clean, strict=False)
    print(f"[convert] dropped embedding-alias={dropped}  missing={list(miss.missing_keys)[:4]} "
          f"unexpected={list(miss.unexpected_keys)[:4]}")

    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    meta = {
        "vocab_size": len(tok),
        "special_tokens": {t: tok.convert_tokens_to_ids(t) for t in SPECIAL},
        "src_checkpoint": str(ckpt_file),
        "dtype": "bfloat16",
        "base": model_id,
    }
    (out_dir / "coconut_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[convert] -> {out_dir}  meta={meta['special_tokens']}")
    return out_dir


# ---------------------------------------------------------------------------- #
# 3) test  — run probe_causal_coconut (token-swap) then post-process follow/stay
# ---------------------------------------------------------------------------- #
def _norm(s):
    nums = re.findall(r"-?\d+\.?\d*", str(s).replace(",", ""))
    return nums[0] if nums else str(s).strip().lower()


def _follow_stay(recs: list) -> dict:
    """donor of rec i = rec (i+1)%n (exactly how the probe built swap_ans).
    follow = swap answer == donor's clean answer ; stay = swap answer == own clean answer."""
    n = len(recs)
    if n == 0:
        return {"n": 0}
    follow = stay = changed = 0
    for i, r in enumerate(recs):
        donor = recs[(i + 1) % n]
        sw = _norm(r.get("swap_ans"))
        follow  += sw == _norm(donor.get("clean_ans"))
        stay    += sw == _norm(r.get("clean_ans"))
        changed += bool(r.get("swap_changed"))
    return {
        "n": n,
        "token_swap_change": round(changed / n, 3),
        "follow_donor": round(follow / n, 3),
        "stay_self": round(stay / n, 3),
    }


def test(hf_export: Path, K: int, n: int, tag: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_out = OUT_DIR / f"causal_coconut_{tag}.json"
    if not PROBE.exists():
        raise SystemExit(f"[test] probe not found at {PROBE} — set PROBE env.")
    sh([PYTHON, PROBE, "--ckpt", str(hf_export), "--k_passes", str(K),
        "--n", str(n), "--out", str(probe_out)])
    recs = json.load(open(probe_out, encoding="utf-8"))
    m = _follow_stay(recs)
    m.update({"tag": tag, "k_passes": K, "ckpt": str(hf_export), "probe_out": str(probe_out)})
    print(f"[test] {tag}: swap_change={m.get('token_swap_change')} "
          f"follow={m.get('follow_donor')} stay={m.get('stay_self')}")
    return m


# ---------------------------------------------------------------------------- #
# per-track driver + `all`
# ---------------------------------------------------------------------------- #
def hf_export_dir(cfg: dict) -> Path:
    return CKPTS_ROOT / "coconut" / "hf_export" / cfg["name"]


def do_convert_one(yaml_path: Path) -> Path:
    cfg = load_yaml(yaml_path)
    ck = latest_checkpoint(save_dir_of(cfg))
    if ck is None:
        raise SystemExit(f"[convert] no checkpoint in {save_dir_of(cfg)} — train first.")
    return convert(ck, hf_export_dir(cfg))


def do_test_one(yaml_path: Path, n: int) -> dict:
    cfg = load_yaml(yaml_path)
    return test(hf_export_dir(cfg), K_of(cfg), n, cfg["name"])


def run_all(n: int, force: bool):
    train(STD_YAML, force=force)
    train(PL_YAML, force=force)
    do_convert_one(STD_YAML)
    do_convert_one(PL_YAML)
    std = do_test_one(STD_YAML, n)
    pl  = do_test_one(PL_YAML, n)

    verdict = ("crutch-artifact (pure-latent became MORE causal -> residual CoT was the crutch)"
               if pl["follow_donor"] > std["follow_donor"] + 0.05
               else "intrinsic-pseudo (dropping CoT did NOT make latents causal)")
    result = {
        "hypothesis": "Coconut 'pseudo' latents: residual-text-CoT crutch artifact (pure-latent -> higher "
                      "swap/follow) vs intrinsic (stays equally low).",
        "standard_residual_cot": std,
        "purelatent_no_cot": pl,
        "delta_purelatent_minus_standard": {
            k: round(pl[k] - std[k], 3) for k in ("token_swap_change", "follow_donor", "stay_self")
        },
        "verdict": verdict,
        "reading": "follow_donor high & stay_self low => latents causally drive the answer (真). "
                   "stay_self high => swapping the latent scratchpad changes nothing (伪).",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "coconut_residual_vs_purelatent.json"
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n==== Section B — residual CoT vs pure latent ====")
    print(json.dumps(result["delta_purelatent_minus_standard"], indent=2))
    print("verdict:", verdict)
    print("wrote", out)
    return result


# ---------------------------------------------------------------------------- #
# selftest — pure-python config-diff validation (no torch / no GPU)
# ---------------------------------------------------------------------------- #
def selftest():
    a, b = load_yaml(STD_YAML), load_yaml(PL_YAML)
    keys = set(a) | set(b)
    diff = {k for k in keys if a.get(k) != b.get(k)}
    sci = diff - IGNORE_DIFF                      # scientific (behavioural) differences
    print("all differing keys :", sorted(diff))
    print("behavioural diff   :", {k: (a.get(k), b.get(k)) for k in sorted(sci)})

    assert sci <= ALLOWED_DIFF, f"recipes differ in unexpected behavioural keys: {sci - ALLOWED_DIFF}"
    assert "num_epochs" in sci, "num_epochs must differ (purelatent needs more epochs to reach skip-all)"
    assert "pad_latent_to_max" in sci, "pad_latent_to_max must differ (True only in purelatent)"

    # standard must NOT reach skip-all; purelatent MUST reach it: scheduled_stage = (num_epochs-1)//epochs_per_stage
    for name, cfg, want_skip in [("standard", a, False), ("purelatent", b, True)]:
        last_stage = (cfg["num_epochs"] - 1) // cfg["epochs_per_stage"]
        reaches = last_stage > cfg["max_latent_stage"]
        print(f"{name}: last_stage={last_stage} max_latent_stage={cfg['max_latent_stage']} "
              f"reaches_skip_all={reaches}")
        assert reaches == want_skip, f"{name} skip-all expectation violated"

    assert b["pad_latent_to_max"] is True, "purelatent pad_latent_to_max must be True (fix K in skip-all)"
    assert K_of(a) == K_of(b) == 6, "both recipes must share latent width K=6 for a fair probe"
    # everything else identical
    for k in keys - IGNORE_DIFF - ALLOWED_DIFF:
        assert a.get(k) == b.get(k), f"unexpected diff on {k}: {a.get(k)} vs {b.get(k)}"
    print("selftest OK — recipes differ ONLY in num_epochs + pad_latent_to_max (K=6 shared).")


# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Section B: Coconut residual-CoT vs pure-latent token-swap")
    ap.add_argument("--step", default="all",
                    choices=["train_standard", "train_purelatent", "convert", "test", "all"])
    ap.add_argument("--n", type=int, default=20, help="probe problems (token-swap)")
    ap.add_argument("--force", action="store_true", help="retrain even if final checkpoint exists")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return

    print(f"COCONUT_ROOT={COCONUT_ROOT}\nCKPTS_ROOT={CKPTS_ROOT}\nPROBE={PROBE}\nPYTHON={PYTHON}")
    if a.step == "train_standard":
        train(STD_YAML, force=a.force)
    elif a.step == "train_purelatent":
        train(PL_YAML, force=a.force)
    elif a.step == "convert":
        do_convert_one(STD_YAML); do_convert_one(PL_YAML)
    elif a.step == "test":
        std = do_test_one(STD_YAML, a.n); pl = do_test_one(PL_YAML, a.n)
        print(json.dumps({"standard": std, "purelatent": pl}, ensure_ascii=False, indent=2))
    elif a.step == "all":
        run_all(a.n, a.force)


if __name__ == "__main__":
    main()
