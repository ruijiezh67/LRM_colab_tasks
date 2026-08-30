#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

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

# Exercise the real helper while isolating only the external git/network call.
# The expected URL is derived from the caller's input, not from _common.sh.
. "$ROOT/handoff/_common.sh"
gh_clone "https://github.com/example/repo.git" "$TMP/repo"

test "${HF_ENDPOINT:-}" = "https://hf-mirror.com"
test "${PIP_INDEX_URL:-}" = "https://pypi.tuna.tsinghua.edu.cn/simple"
test "$(sed -n '1p' "$GIT_LOG")" = "clone"
test "$(sed -n '2p' "$GIT_LOG")" = "-q"
test "$(sed -n '3p' "$GIT_LOG")" = "https://github.com/example/repo.git"
test "$(sed -n '4p' "$GIT_LOG")" = "$TMP/repo"

echo "GitHub clone uses the official URL; HF and pip mirrors remain enabled."
