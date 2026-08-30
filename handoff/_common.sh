# ============================================================================
# _common.sh —— 被三个 train_*.sh `source`。不要直接运行。
# 负责: ① 解释器解析(不假设有 `python`) ② HF/pip 镜像 ③ GPU 绑定 ④ 可选 venv 隔离
# ============================================================================

# ---------- ① GPU 绑定 ----------
# GPU=3 bash train_lt_code.sh  → 只用 3 号卡。并行跑多个任务时给每个任务一张卡。
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi

# ---------- ② HF / pip 镜像 ----------
# CN=1 开 HF/pip 镜像; 也可单独覆盖 HF_ENDPOINT / PIP_INDEX_URL。
if [ "${CN:-0}" = "1" ]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
fi

gh_clone() {   # gh_clone <https-url> <dir>   —— 直接访问官方 GitHub
  local url="$1" dir="$2"
  git clone -q "$url" "$dir"
}

# ---------- ③ 解释器解析 ----------
# 很多 Linux 只有 python3、没有 `python`(裸写 python → command not found, rc=127)。
if [ -z "${PY:-}" ]; then
  for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$(command -v $c)"; break; }; done
fi
[ -n "${PY:-}" ] || { echo "❌ 找不到 python/python3。用 PY=/你的/python 指定"; exit 1; }

# ---------- ④ 可选 venv 隔离(并行跑多平台时必须) ----------
# VENV=1 → 为本平台建独立 venv(继承系统 site-packages, 只隔离互相冲突的包)。
# 三个平台 transformers 版本互斥(4.45.2 / 4.55.4 / 4.51.1), 共用一个环境就不能并行。
setup_venv() {   # setup_venv <平台名>
  [ "${VENV:-0}" = "1" ] || return 0
  local d="$WORK/venv_$1"
  [ -d "$d" ] || { echo "  建 venv: $d"; "$PY" -m venv --system-site-packages "$d"; }
  # shellcheck disable=SC1090
  . "$d/bin/activate"
  PY="$d/bin/python"
  echo "  venv 已激活: $d"
}

# ---------- 统一包装(全部走 $PY -m, 不依赖 PATH 上的脚本) ----------
pipi()  { "$PY" -m pip -q install "$@"; }
hfdl()  { "$PY" -m huggingface_hub.commands.huggingface_cli download "$@"; }
dsrun() {   # deepspeed 启动器: 优先 PATH 上的 deepspeed, 没有就走模块
  if command -v deepspeed >/dev/null 2>&1; then deepspeed "$@"
  else "$PY" -m deepspeed.launcher.runner "$@"; fi
}

print_env() {
  echo "  解释器: $PY  ($("$PY" -V 2>&1))"
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "  GPU   : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  [ -n "${HF_ENDPOINT:-}" ]          && echo "  HF 镜像: $HF_ENDPOINT"
  [ -n "${PIP_INDEX_URL:-}" ]        && echo "  pip 源 : $PIP_INDEX_URL"
  return 0
}
