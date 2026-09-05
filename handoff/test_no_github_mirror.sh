#!/usr/bin/env bash
# 交接包与完整说明: https://github.com/ruijiezh67/LRM_colab_tasks/tree/main/handoff
# 回归测试 —— gh_clone 的三条路径 + 国内镜像仍生效。
#   bash test_no_github_mirror.sh
# 不联网、不写仓库, 全部在临时目录里用假 git 完成。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 假 git: 只记录参数, 不真的联网
mkdir -p "$TMP/bin"
cat > "$TMP/bin/git" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$GIT_LOG"
SH
chmod +x "$TMP/bin/git"

unset CN HF_ENDPOINT PIP_INDEX_URL
export CN=1
export PY="${PY:-$(command -v python3)}"
export GIT_LOG="$TMP/git.args"
export PATH="$TMP/bin:$PATH"
export HERE="$ROOT/handoff"          # _common.sh 依赖调用方设置

. "$ROOT/handoff/_common.sh"

# --- ① 国内镜像仍然生效(CN=1 只切 HF/pip, 不碰 GitHub) ---
test "${HF_ENDPOINT:-}"    = "https://hf-mirror.com"
test "${PIP_INDEX_URL:-}"  = "https://pypi.tuna.tsinghua.edu.cn/simple"

# --- ② 已 vendored 的仓库: 用本地副本, 绝不调用 git ---
rm -f "$GIT_LOG"
gh_clone "https://github.com/xiaomi-research/colar.git" "$TMP/colar" >/dev/null
test ! -f "$GIT_LOG"                       # git 一次都没被调用
test -f "$TMP/colar/run.py"                # 真的拿到了内容

# --- ③ 数据仓: 指向本仓库自己 ---
rm -f "$GIT_LOG"
gh_clone "https://github.com/ruijiezh67/LRM_colab_tasks.git" "$TMP/lrm" >/dev/null
test ! -f "$GIT_LOG"
test -f "$TMP/lrm/code_real_ladder/manifest.json"

# --- ④ 未 vendored 的仓库: 回退联网, 且用官方 URL(不走任何代理) ---
rm -f "$GIT_LOG"
gh_clone "https://github.com/example/repo.git" "$TMP/repo" >/dev/null
test "$(sed -n '1p' "$GIT_LOG")" = "clone"
test "$(sed -n '2p' "$GIT_LOG")" = "-q"
test "$(sed -n '3p' "$GIT_LOG")" = "https://github.com/example/repo.git"
test "$(sed -n '4p' "$GIT_LOG")" = "$TMP/repo"

echo "PASS: vendored 优先(不联网) / 数据仓自指 / 未 vendored 回退官方 URL / HF+pip 镜像生效"
