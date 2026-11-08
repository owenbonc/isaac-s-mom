#!/bin/sh
# `scripts/verify.sh` runs every check; `scripts/verify.sh secrets` runs only the scans (the pre-push hook).
set -u
cd "$(git rev-parse --show-toplevel)"
log="$(mktemp)"
failed=0

step() {
  name="$1"
  shift
  if "$@" >"$log" 2>&1; then
    printf '%-10s ok\n' "$name"
  else
    printf '%-10s FAIL\n' "$name"
    tail -n 25 "$log" | sed 's/^/    /'
    failed=1
  fi
}

# Paths whose history has a line matching the given grep options.
history_paths() {
  grep -E -e '^diff --git ' "$@" "$history" | awk '/^diff --git /{sub(/.* b\//, ""); path = $0; next} {print path}' | sort -u
}

# Scans every change on every ref, because a key or a CV line deleted later still leaks once pushed.
# Prints paths only, never the match itself.
scan_secrets() {
  history="$(mktemp)"
  git log --all -p --no-color --no-ext-diff --format= >"$history"
  keys='sk-[A-Za-z0-9_-]{40,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
  hits="$(history_paths -e "$keys")"
  envs="$(git log --all --name-only --format= | grep -E '(^|/)\.env(\..+)?$' | grep -vE '\.(example|sample|template)$' | sort -u)"
  # The owner's own identifiers (email, phone, address): one extended regex per line, kept outside every repo.
  personal="${PERSONAL_PATTERNS:-$HOME/.config/personal-patterns}"
  mine=''
  if [ -f "$personal" ]; then
    patterns="$(mktemp)"
    grep -vE '^[[:space:]]*(#|$)' "$personal" >"$patterns" || true
    if [ -s "$patterns" ]; then
      mine="$(history_paths -i -f "$patterns")"
    fi
    rm -f "$patterns"
  fi
  rm -f "$history"
  [ -z "$hits$envs$mine" ] && return 0
  [ -n "$hits" ] && printf 'key-shaped strings in history:\n%s\n' "$hits"
  [ -n "$envs" ] && printf 'env files in history:\n%s\n' "$envs"
  [ -n "$mine" ] && printf 'personal data in history:\n%s\n' "$mine"
  return 1
}

if [ "${1:-}" = secrets ]; then
  step secrets scan_secrets
else
  export PATH="$HOME/.local/bin:$PATH"
  step lint sh -c 'uv run ruff check && uv run ruff format --check'
  step typecheck uv run pyright
  # One full test run at a time across the repo's worktrees, whichever session starts it.
  lock="${VERIFY_LOCK:-/tmp/verify-$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")").lock}"
  until mkdir "$lock" 2>/dev/null; do sleep 20; done
  trap 'rmdir "$lock"' EXIT
  step test uv run pytest -q
  rmdir "$lock"
  trap - EXIT
  step secrets scan_secrets
fi
rm -f "$log"

if [ "$failed" -eq 0 ]; then
  echo READY
else
  echo 'NOT READY'
  exit 1
fi
