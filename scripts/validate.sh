#!/usr/bin/env bash
# validate.sh - Verifies that examples/ and bindings/ all build cleanly.
#
# This is deliberately separate from `zig build test` (unit tests).
# It is slow: it builds the XCFramework, Swift packages, and Zig examples.
# Run it before releases or after touching bindings/examples.
#
# Usage:
#   ./scripts/validate.sh              # run all checks
#   ./scripts/validate.sh --skip-swift # skip Swift builds (saves several minutes)
#
# Requirements: zig and swift must be on PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_SWIFT=false
for arg in "$@"; do
    case $arg in --skip-swift) SKIP_SWIFT=true ;; esac
done

# ---- Helpers ----------------------------------------------------------------

PASS=0
FAIL=0
ERRORS=()

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); ERRORS+=("$1"); }
section() { echo; echo "==> $1"; }

check() {
    local desc="$1"; shift
    local out
    if out=$("$@" 2>&1); then
        pass "$desc"
    else
        fail "$desc"
        echo "    output: $out"
    fi
}

# ---- Builds -----------------------------------------------------------------

section "examples/zig"
check "zig build" bash -c "cd '$REPO_ROOT/examples/zig' && zig build"

section "bindings/c (via xcframework)"
check "zig build xcframework" zig build xcframework

if [ "$SKIP_SWIFT" = true ]; then
    echo
    echo "==> Swift (skipped via --skip-swift)"
else
    section "bindings/swift"
    check "swift build" swift build --package-path "$REPO_ROOT/bindings/swift"

    section "examples/swift"
    check "swift build" swift build --package-path "$REPO_ROOT/examples/swift"
fi

# ---- Summary ----------------------------------------------------------------

echo
echo "================================"
echo " $PASS passed, $FAIL failed"
echo "================================"

if [ "${#ERRORS[@]}" -gt 0 ]; then
    echo "Failed:"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
    exit 1
fi
