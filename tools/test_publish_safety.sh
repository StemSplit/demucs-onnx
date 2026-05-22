#!/usr/bin/env bash
# Verification suite for tools/publish.py guardrails.
#
# Exits 0 if all cases behave as expected, non-zero otherwise.
#
# Cases:
#   1. real pkg, dry-run, default cap     -> pass (exit 0)
#   2. fake pkg with 105 files            -> fail file-count guardrail (exit 2)
#   3. real pkg, fake remote URL          -> fail remote-URL guardrail (exit 2)
#   4. hard floor (MAX_FILES=0)           -> fail with "below the hard floor"
#   5. real pkg, MAX_FILES=200            -> pass with custom cap (exit 0)
#
# Always runs in dry-run mode. Never pushes anything. Safe to run on CI.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$HERE/.." && pwd)"
PUBLISH="$HERE/publish.py"
PYTHON="${PYTHON:-python3}"
if [[ -x "$PKG_DIR/.venv/bin/python" ]]; then
  PYTHON="$PKG_DIR/.venv/bin/python"
fi

TMP="$(mktemp -d -t demucs-onnx-publish-test-XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

FAILED=0
PASSED=0

run_case() {
  local name="$1"
  local expect_exit="$2"
  local expect_grep="$3"
  shift 3
  local logfile="$TMP/$(echo "$name" | tr ' /' '__').log"

  echo ""
  echo "--- CASE: $name ---"
  echo "  expecting exit=$expect_exit, grep=${expect_grep:-<none>}"
  echo "  cmd: $*"

  set +e
  ( "$@" ) >"$logfile" 2>&1
  local actual_exit=$?
  set -e

  local pass=1
  if [[ "$actual_exit" != "$expect_exit" ]]; then
    echo "  FAIL: exit was $actual_exit, expected $expect_exit"
    pass=0
  fi
  if [[ -n "$expect_grep" ]] && ! grep -q -- "$expect_grep" "$logfile"; then
    echo "  FAIL: expected to find '$expect_grep' in output"
    pass=0
  fi

  if [[ "$pass" == "1" ]]; then
    echo "  PASS"
    PASSED=$((PASSED + 1))
  else
    echo "  --- log (tail 20) ---"
    tail -20 "$logfile" | sed 's/^/  | /'
    echo "  --- end log ---"
    FAILED=$((FAILED + 1))
  fi
}

build_fake_pkg() {
  local target="$1"
  local n="$2"
  mkdir -p "$target/src/demucs_onnx" "$target/docs"
  cp "$PKG_DIR/pyproject.toml" "$target/pyproject.toml"
  cp "$PKG_DIR/CHANGELOG.md" "$target/CHANGELOG.md" 2>/dev/null || echo "# Changelog" > "$target/CHANGELOG.md"
  cp "$PKG_DIR/README.md" "$target/README.md" 2>/dev/null || echo "# fake" > "$target/README.md"
  cp "$PKG_DIR/LICENSE" "$target/LICENSE" 2>/dev/null || echo "MIT" > "$target/LICENSE"
  echo '"""stub"""' > "$target/src/demucs_onnx/__init__.py"
  local i
  for ((i = 1; i <= n; i++)); do
    printf 'pad %05d\n' "$i" > "$target/docs/pad_${i}.md"
  done
}

echo "==> Test environment"
echo "  PYTHON=$PYTHON"
echo "  PKG_DIR=$PKG_DIR"
echo "  PUBLISH=$PUBLISH"
echo "  TMP=$TMP"

# Sanity: publish.py exists and parses.
if ! "$PYTHON" -c "import ast; ast.parse(open('$PUBLISH').read())" >/dev/null 2>&1; then
  echo "FATAL: $PUBLISH does not parse"
  exit 99
fi

# Case 1: real pkg, dry-run, default cap -> pass
run_case "real pkg, dry-run, default cap" 0 "file count: " \
  "$PYTHON" "$PUBLISH" \
  --workdir "$TMP/case1-workdir" --keep-workdir

# Case 2: fake pkg with 105 files -> file-count guardrail
FAKE_PKG="$TMP/fake-pkg-105"
build_fake_pkg "$FAKE_PKG" 105
run_case "fake pkg with 105 files (over cap)" 2 "FILE-COUNT GUARDRAIL TRIPPED" \
  "$PYTHON" "$PUBLISH" \
  --pkg-dir "$FAKE_PKG" \
  --workdir "$TMP/case2-workdir" --keep-workdir

# Case 3: real pkg, fake remote URL -> remote-URL guardrail
run_case "real pkg, fake remote URL" 2 "REMOTE-URL GUARDRAIL TRIPPED" \
  "$PYTHON" "$PUBLISH" \
  --remote-url "https://github.com/AttackerOrg/exfil.git" \
  --workdir "$TMP/case3-workdir" --keep-workdir

# Case 4: MAX_FILES=0 below hard floor
run_case "MAX_FILES=0 below hard floor" 2 "below the hard floor" \
  env DEMUCS_ONNX_PUBLISH_MAX_FILES=0 \
  "$PYTHON" "$PUBLISH" \
  --workdir "$TMP/case4-workdir" --keep-workdir

# Case 5: real pkg, MAX_FILES=200 -> pass with custom cap
run_case "real pkg, MAX_FILES=200" 0 "max_files: 200" \
  env DEMUCS_ONNX_PUBLISH_MAX_FILES=200 \
  "$PYTHON" "$PUBLISH" \
  --workdir "$TMP/case5-workdir" --keep-workdir

echo ""
echo "==> Results"
echo "  PASSED: $PASSED"
echo "  FAILED: $FAILED"

if [[ "$FAILED" != "0" ]]; then
  exit 1
fi
exit 0
