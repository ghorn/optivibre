#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIDE="${1:-${SSIDS_GLIDER_INPROCESS_SIDE:-both}}"
case "$SIDE" in
  rust|native|paired|both) ;;
  *)
    echo "usage: $0 [rust|native|paired|both]" >&2
    exit 2
    ;;
esac

REPEATS="${SSIDS_GLIDER_INPROCESS_REPEATS:-1200}"
SAMPLE_SECONDS="${SSIDS_GLIDER_SAMPLE_SECONDS:-5}"
SAMPLE_INTERVAL_MS="${SSIDS_GLIDER_SAMPLE_INTERVAL_MS:-1}"
OUT_DIR="${SSIDS_GLIDER_SAMPLE_OUT:-target/ssids-glider-samples}"
TEST_NAME="print_current_glider_nlip_iteration0_augmented_inprocess_profile"
mkdir -p "$OUT_DIR"

build_json="$(mktemp -t ssids-glider-build.XXXXXX)"
trap 'rm -f "$build_json"' EXIT

env \
  RAYON_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  OMP_CANCELLATION=true \
  AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1 \
  cargo test -p optimal_control_problems --release \
    --features ipopt,native-spral-src \
    --no-run --message-format=json >"$build_json"

test_binary="$(
  python3 - "$build_json" <<'PY'
import json
import sys

path = sys.argv[1]
candidate = None
with open(path, "r", encoding="utf-8") as handle:
    for line in handle:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("reason") != "compiler-artifact":
            continue
        target = event.get("target") or {}
        if target.get("name") != "optimal_control_problems":
            continue
        executable = event.get("executable")
        if executable:
            candidate = executable
if not candidate:
    raise SystemExit("could not locate optimal_control_problems test binary")
print(candidate)
PY
)"

run_side() {
  local side="$1"
  local stamp log_file sample_file pid
  stamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$OUT_DIR/glider-${side}-${stamp}.log"
  sample_file="$OUT_DIR/glider-${side}-${stamp}.sample.txt"

  env \
    RAYON_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OMP_CANCELLATION=true \
    AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1 \
    SSIDS_GLIDER_INPROCESS_SIDE="$side" \
    SSIDS_GLIDER_INPROCESS_REPEATS="$REPEATS" \
    "$test_binary" "$TEST_NAME" --ignored --nocapture >"$log_file" 2>&1 &

  pid="$!"
  for _ in $(seq 1 1200); do
    if grep -q "ssids_glider_side_profile_begin" "$log_file" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.05
  done
  if kill -0 "$pid" 2>/dev/null; then
    sample "$pid" "$SAMPLE_SECONDS" "$SAMPLE_INTERVAL_MS" -file "$sample_file" >/dev/null 2>&1 || true
  fi
  wait "$pid"

  echo "log=$log_file"
  echo "sample=$sample_file"
  echo
  grep -E "^(  )?(rust_spral|native_spral)|augmented deltas|side=" "$log_file" || true
  echo
  echo "== sample hotspots containing SSIDS/SPRAL symbols =="
  grep -Ei "ssids|spral|ldlt|block_ldlt|factorize_dense|NumericSubtree|assemble|solve" "$sample_file" \
    | head -n 120 || true
}

case "$SIDE" in
  both)
    run_side rust
    echo
    echo "----"
    echo
    run_side native
    ;;
  *)
    run_side "$SIDE"
    ;;
esac
