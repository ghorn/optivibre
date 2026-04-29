#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPEATS="${SSIDS_PROFILE_REPEATS:-7}"
DEFAULT_DENSE_CASES="0x0000706172697479:58 0x7061726974792026:59"
DENSE_CASES="${SSIDS_PROFILE_DENSE_CASES-$DEFAULT_DENSE_CASES}"
if [[ "$DENSE_CASES" == "none" || "$DENSE_CASES" == "0" ]]; then
  DENSE_CASES=""
fi
GLIDER_REPEATS="${SSIDS_PROFILE_GLIDER_REPEATS:-1}"
RUN_GLIDER="${SSIDS_PROFILE_GLIDER:-1}"
GLIDER_TEST="${SSIDS_PROFILE_GLIDER_TEST:-print_current_glider_nlip_iteration0_augmented_compare}"
GLIDER_INPROCESS_REPEATS="${SSIDS_PROFILE_GLIDER_INPROCESS_REPEATS:-15}"
if [[ "${SSIDS_PROFILE_GLIDER_INPROCESS:-0}" != "0" ]]; then
  GLIDER_TEST="print_current_glider_nlip_iteration0_augmented_inprocess_profile"
fi

require_native_env=(
  RAYON_NUM_THREADS=1
  OMP_NUM_THREADS=1
  OMP_CANCELLATION=true
  AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1
  SSIDS_GLIDER_INPROCESS_REPEATS="$GLIDER_INPROCESS_REPEATS"
)

tmp="$(mktemp -t ssids-rs-release-profile.XXXXXX)"
trap 'rm -f "$tmp"' EXIT

run_dense_case() {
  local seed_case="$1"
  local seed="${seed_case%%:*}"
  local case_index="${seed_case##*:}"
  local repeat
  for repeat in $(seq 1 "$REPEATS"); do
    {
      printf '## dense seed=%s case=%s repeat=%s\n' "$seed" "$case_index" "$repeat"
      env "${require_native_env[@]}" \
        SPRAL_SSIDS_DEBUG_FACTOR=1 \
        SPRAL_SSIDS_MATCHING_PARITY_SEED="$seed" \
        SPRAL_SSIDS_MATCHING_PARITY_CASE="$case_index" \
        cargo test -p ssids-rs --release --features native-spral-src \
          --test matching_scaling_parity \
          native_matching_scaling_order_observation_dense_boundary_case \
          -- --ignored --nocapture
    } 2>&1 | tee -a "$tmp"
  done
}

run_glider() {
  local repeat
  for repeat in $(seq 1 "$GLIDER_REPEATS"); do
    {
      printf '## glider repeat=%s\n' "$repeat"
      env "${require_native_env[@]}" \
        cargo test -p optimal_control_problems --release \
          --features ipopt,native-spral-src \
          "$GLIDER_TEST" \
          -- --ignored --nocapture
    } 2>&1 | tee -a "$tmp"
  done
}

for seed_case in $DENSE_CASES; do
  run_dense_case "$seed_case"
done

if [[ "$RUN_GLIDER" != "0" ]]; then
  run_glider
fi

python3 - "$tmp" <<'PY'
import re
import statistics
import sys

path = sys.argv[1]

micro = "\N{MICRO SIGN}"
duration_re = re.compile(f"^([0-9.]+)(ns|{micro}s|us|ms|s)$")
factor_re = re.compile(r"^\[ssids_rs::factorize\].*scaling=([^ ]+) (.*)$")
dense_re = re.compile(r"^(native_matching_scaling|native_captured_order_no_scaling|rust_captured_order_no_scaling|rust_spral_matching_saved_scaling) (.*)$")
glider_factor_re = re.compile(r"^\s*(rust_spral|native_spral) factor=([^ ]+) solve=([^ ]+)")
glider_dense_re = re.compile(r"^\s*rust_spral dense_front_profile (.*)$")
repeat_re = re.compile(r"\s+repeat=\d+\b")


def seconds(raw):
    raw = raw.strip()
    match = duration_re.match(raw)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ns":
        return value / 1e9
    if unit in (f"{micro}s", "us"):
        return value / 1e6
    if unit == "ms":
        return value / 1e3
    return value


def fields(text):
    out = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        value = value.rstrip(",")
        parsed = seconds(value)
        if parsed is not None:
            out[key] = parsed
    return out


series = {}
current = "unknown"
with open(path, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        line = line.strip()
        if line.startswith("## "):
            current = repeat_re.sub("", line[3:])
            continue
        match = factor_re.match(line)
        if match:
            scaling = match.group(1)
            for key, value in fields(match.group(2)).items():
                series.setdefault((current, f"rust_factor_profile[{scaling}]", key), []).append(value)
            continue
        match = dense_re.match(line)
        if match:
            label = match.group(1)
            for key, value in fields(match.group(2)).items():
                series.setdefault((current, label, key), []).append(value)
            continue
        match = glider_factor_re.match(line)
        if match:
            label = f"glider_{match.group(1)}"
            series.setdefault((current, label, "factor"), []).append(seconds(match.group(2)))
            series.setdefault((current, label, "solve"), []).append(seconds(match.group(3)))
            continue
        match = glider_dense_re.match(line)
        if match:
            for key, value in fields(match.group(1)).items():
                series.setdefault((current, "glider_rust_dense_front_profile", key), []).append(value)


def fmt(value):
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value < 1e-6:
        return f"{sign}{value * 1e9:.1f}ns"
    if value < 1e-3:
        return f"{sign}{value * 1e6:.3f}us"
    if value < 1:
        return f"{sign}{value * 1e3:.3f}ms"
    return f"{sign}{value:.3f}s"


medians = {}
counts = {}
for key, values in series.items():
    clean = [value for value in values if value is not None]
    if not clean:
        continue
    medians[key] = statistics.median(clean)
    counts[key] = len(clean)


def side_by_side(case, name, native_label, rust_label, metrics):
    printed = False
    for metric in metrics:
        native_key = (case, native_label, metric)
        rust_key = (case, rust_label, metric)
        if native_key not in medians or rust_key not in medians:
            continue
        if not printed:
            printed = True
        native = medians[native_key]
        rust = medians[rust_key]
        ratio = rust / native if native else float("inf")
        delta = rust - native
        print(
            f"{case} | {name} | {metric} | "
            f"native={fmt(native)} n={counts[native_key]} | "
            f"rust={fmt(rust)} n={counts[rust_key]} | "
            f"rust/native={ratio:.3f}x | delta={fmt(delta)}"
        )
    return printed


print("\n== native/rust side-by-side medians ==")
printed_any = False
for case in sorted({case for case, _, _ in medians}):
    printed_any |= side_by_side(
        case,
        "matching_scaling",
        "native_matching_scaling",
        "rust_spral_matching_saved_scaling",
        ["analyse", "factor", "solve"],
    )
    printed_any |= side_by_side(
        case,
        "captured_order_no_scaling",
        "native_captured_order_no_scaling",
        "rust_captured_order_no_scaling",
        ["analyse", "factor", "solve"],
    )
    printed_any |= side_by_side(
        case,
        "glider",
        "glider_native_spral",
        "glider_rust_spral",
        ["factor", "solve"],
    )
if not printed_any:
    print("(no native/rust pairs found)")

print("\n== rust-only attribution bucket medians ==")
for (case, label, key), value in sorted(medians.items()):
    if not (
        label.startswith("rust_factor_profile[")
        or label == "glider_rust_dense_front_profile"
    ):
        continue
    print(f"{case} | {label} | {key} | n={counts[(case, label, key)]} median={fmt(value)}")

print("\n== all raw medians ==")
for (case, label, key), value in sorted(medians.items()):
    print(f"{case} | {label} | {key} | n={counts[(case, label, key)]} median={fmt(value)}")
PY
