#!/usr/bin/env python3
# ============================================================================
# 数据溯源校验 —— 取代原来的 sha256 字节锁。
#
#   python3 verify_provenance.py [数据目录]          # 默认 ../code_real_ladder
#   HF_ENDPOINT=https://hf-mirror.com python3 verify_provenance.py   # 国内
#
# 干什么: 从 HuggingFace 拉三个公开上游集, 对我们六个数据文件里的每一行,
#         检查 (函数代码体, 调用输入, 答案) 三元组是否逐字出现在上游。
#         这才是"禁自造数据"的真校验 —— sha256 只能证明"没被改过",
#         证明不了"是真的"; 重新导出一次 json 就会误杀, 而编造一行却抓不到。
#
# 上游 (2026-09-03 实测可达, 官方与 hf-mirror.com 逐字节一致):
#   cruxeval-org/cruxeval           test.jsonl                    800 行  <- 本数据集用了全部 800
#   livecodebench/execution-v2      data/test-*.parquet           479 行  <- 用了全部 479
#   google-research-datasets/mbpp   full/{train,test,validation,prompt}   <- 只用了 train 的 374 行,
#       但四个 split 都装进索引: 出处只要是 MBPP 公开集就算合规, 不必卡死在某个 split。
#
# 计数说明: 六个文件 = 同一份 1653 行(1488 train + 165 val)的三种平台格式, 所以总数报 4959。
#
# 依赖: pyarrow (训练 venv 里 datasets 已带; 单独跑则 pip install pyarrow)
# 退出码: 0=全部可溯源, 1=有行溯源失败(此时禁止训练), 2=环境/网络问题
# ============================================================================
import json, os, re, sys, urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "code_real_ladder")
EP   = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
CACHE= os.environ.get("PROV_CACHE") or os.path.join(HERE, ".prov_cache")

FILES = [
    ("cruxeval-org/cruxeval",           "test.jsonl",                        "crux.jsonl"),
    ("livecodebench/execution-v2",      "data/test-00000-of-00001.parquet",  "lcb.parquet"),
    ("google-research-datasets/mbpp",   "full/train-00000-of-00001.parquet", "mbpp_train.parquet"),
    ("google-research-datasets/mbpp",   "full/test-00000-of-00001.parquet",  "mbpp_test.parquet"),
    ("google-research-datasets/mbpp",   "full/validation-00000-of-00001.parquet", "mbpp_val.parquet"),
    ("google-research-datasets/mbpp",   "full/prompt-00000-of-00001.parquet",     "mbpp_prompt.parquet"),
]

def fetch(repo, path, local):
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, local)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    url = f"{EP}/datasets/{repo}/resolve/main/{path}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    except Exception as e:
        print(f"  X 下载失败 {url}\n    {type(e).__name__}: {e}")
        print(f"    国内机器请先 export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(2)
    print(f"  下载 {local:<22} {os.path.getsize(dst):>9,} B   ({EP})")
    return dst

def nz(s):  # 归一化: 去掉全部空白, 只比对实质内容(不受重新导出/换行符影响)
    return re.sub(r"\s+", "", s or "")

print(f"[数据溯源校验] 上游端点 {EP}")
try:
    import pyarrow.parquet as pq
except ImportError:
    print("  X 缺少 pyarrow。请 pip install pyarrow (训练 venv 里随 datasets 一起装)"); sys.exit(2)

paths = {local: fetch(repo, path, local) for repo, path, local in FILES}

# 上游索引: 三元组 (代码, 调用, 输出); MBPP 没有 output 字段, 用 test_list 的 assert 比对
trip, codes, mbpp = {}, {}, {}
for line in open(paths["crux.jsonl"], encoding="utf-8"):
    r = json.loads(line); c = nz(r["code"])
    trip[(c, nz("f(" + r["input"] + ")"), nz(str(r["output"])))] = ("cruxeval", r["id"])
    codes.setdefault(c, "cruxeval")
for r in pq.read_table(paths["lcb.parquet"]).to_pylist():
    c = nz(r["code"])
    trip[(c, nz(r["input"]), nz(str(r["output"])))] = ("lcb", r["id"])
    codes.setdefault(c, "lcb")
for k in ("mbpp_train.parquet", "mbpp_test.parquet", "mbpp_val.parquet", "mbpp_prompt.parquet"):
    for r in pq.read_table(paths[k]).to_pylist():
        c = nz(r["code"]); mbpp.setdefault(c, []).append(r); codes.setdefault(c, "mbpp")
print(f"  上游装载: cruxeval+lcb 三元组 {len(trip):,}   代码体 {len(codes):,}")

PAT = re.compile(r"^Given the Python function:\n(.*?)\n\nWhat is the value returned by "
                 r"(.*?)\? Answer with the exact returned value\.$", re.S)


BOX = re.compile(r"^\\boxed\{(.*)\}$", re.S)
def unbox(s):
    """cot_answer 形如 \\boxed{X} -> 取出 X。X 本身可能是 {..} 字典,
       只能脱一层花括号, 不能 rstrip('}') 否则会把字典结尾一起削掉。"""
    m = BOX.match(s.strip())
    return m.group(1) if m else s

def rows(fn):
    """六种文件三种格式, 统一吐出 (题面, 答案)。"""
    p = os.path.join(SRC, fn)
    if not os.path.exists(p):
        print(f"  X 数据文件缺失: {p}"); sys.exit(1)
    if fn.endswith(".jsonl"):
        for i, line in enumerate(open(p, encoding="utf-8")):
            o = json.loads(line)
            if "question" in o:      yield i + 1, o["question"], str(o["answer"])                      # LT
            else:                    yield i + 1, o["problem"],  unbox(o["cot_answer"])                                # Latent-SFT
    else:
        for i, o in enumerate(json.load(open(p, encoding="utf-8"))):                                    # CoLaR
            yield i + 1, o["question"], str(o["answer"])

allbad, total = [], 0
for fn in ["lt_train.jsonl", "lt_val.jsonl", "lsft_train.jsonl", "lsft_val.jsonl",
           "colar_train.json", "colar_val.json"]:
    cnt, bad = Counter(), []
    for ln, q, ans in rows(fn):
        total += 1
        m = PAT.match(q)
        if not m:
            bad.append((ln, "题面格式不符", q[:70])); continue
        code, call, a = nz(m.group(1)), nz(m.group(2)), nz(ans)
        src = codes.get(code)
        if src is None:
            bad.append((ln, "代码体不在任何上游集中(疑似自造)", m.group(1)[:70])); continue
        cnt[src] += 1
        if src == "mbpp":
            if not any(call in nz("\n".join(u["test_list"])) for u in mbpp[code]):
                bad.append((ln, "调用不在上游 MBPP test_list 中", m.group(2)[:70]))
        elif (code, call, a) not in trip:
            bad.append((ln, "(代码,输入,答案)三元组不在上游中", f"{m.group(2)[:50]} => {ans[:20]}"))
    tag = "OK " if not bad else "X  "
    print(f"  {tag}{fn:<20} {sum(cnt.values()):>5} 行   crux={cnt['cruxeval']} lcb={cnt['lcb']} mbpp={cnt['mbpp']}"
          + (f"   不合格 {len(bad)}" if bad else ""))
    for b in bad[:5]: print(f"       行{b[0]}: {b[1]}  |  {b[2]}")
    allbad += bad

print("-" * 70)
if allbad:
    print(f"溯源失败 {len(allbad)} 行 / 共 {total} 行 —— 禁止训练。")
    print("铁律: 只允许真实公开数据 (CRUXEval / LiveCodeBench / MBPP), 禁自造/合成。")
    sys.exit(1)
print(f"通过: {total} 行全部可逐字回溯到 CRUXEval / LiveCodeBench / MBPP。")
print("说明: 题面(代码+输入)与答案 100% 是公开集原文; steps/CoT 是从真实函数体机械抽取的,")
print("      这三个集本身不提供 CoT —— 见 TRAINING.md 的知情说明。")
