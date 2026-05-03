#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SSIDS_THREAD_SCALING_CONFIRMED:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to run the SSIDS glider thread-scaling benchmark.

This intentionally runs CPU-heavy native SPRAL and ssids-rs timing loops.
When the machine is otherwise quiet, rerun with:

  SSIDS_THREAD_SCALING_CONFIRMED=1 scripts/ssids_glider_thread_scaling.sh

Useful knobs:
  SSIDS_THREAD_SCALING_THREADS="1 2 4 8"
  SSIDS_THREAD_SCALING_REPEATS=15
  SSIDS_THREAD_SCALING_MODES="serial ssids-rs-rayon spral-src-omp spral-src-openblas-pthreads spral-src-openblas-openmp mixed-rayon-omp"
  SSIDS_THREAD_SCALING_OUT=target/ssids-thread-scaling/manual-run
EOF
  exit 2
fi

THREADS="${SSIDS_THREAD_SCALING_THREADS:-1 2 4 8}"
REPEATS="${SSIDS_THREAD_SCALING_REPEATS:-15}"
MODES="${SSIDS_THREAD_SCALING_MODES:-serial ssids-rs-rayon spral-src-omp spral-src-openblas-pthreads spral-src-openblas-openmp mixed-rayon-omp}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${SSIDS_THREAD_SCALING_OUT:-target/ssids-thread-scaling/${STAMP}}"
LOG_DIR="${OUT_DIR}/logs"
TEST_NAME="print_current_glider_nlip_iteration0_augmented_inprocess_profile"

mkdir -p "$LOG_DIR"

run_case() {
  local mode="$1"
  local threads="$2"
  local features="$3"
  local rayon_threads="$4"
  local omp_threads="$5"
  local openblas_threads="$6"
  local log_file="${LOG_DIR}/${mode}-threads-${threads}.log"

  {
    printf '## ssids_thread_scaling mode=%s threads=%s rayon=%s omp=%s openblas=%s features=%s repeats=%s\n' \
      "$mode" "$threads" "$rayon_threads" "$omp_threads" "$openblas_threads" "$features" "$REPEATS"
    env \
      RAYON_NUM_THREADS="$rayon_threads" \
      OMP_NUM_THREADS="$omp_threads" \
      OPENBLAS_NUM_THREADS="$openblas_threads" \
      GOTO_NUM_THREADS="$openblas_threads" \
      OMP_CANCELLATION=true \
      AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1 \
      SSIDS_GLIDER_INPROCESS_SIDE=paired \
      SSIDS_GLIDER_INPROCESS_REPEATS="$REPEATS" \
      cargo test -p optimal_control_problems --release --no-default-features \
        --features "$features" \
        "$TEST_NAME" \
        -- --ignored --nocapture
  } 2>&1 | tee "$log_file"
}

mode_features() {
  case "$1" in
    serial|ssids-rs-rayon|spral-src-omp)
      printf '%s\n' "ipopt,native-spral-src"
      ;;
    spral-src-openblas-pthreads)
      printf '%s\n' "ipopt,native-spral-src-pthreads"
      ;;
    spral-src-openblas-openmp|mixed-rayon-omp)
      printf '%s\n' "ipopt,native-spral-src-openmp"
      ;;
    *)
      echo "unknown SSIDS thread-scaling mode: $1" >&2
      exit 2
      ;;
  esac
}

run_mode_thread() {
  local mode="$1"
  local threads="$2"
  local features
  features="$(mode_features "$mode")"
  case "$mode" in
    serial)
      [[ "$threads" == "1" ]] || return 0
      run_case "$mode" "$threads" "$features" 1 1 1
      ;;
    ssids-rs-rayon)
      run_case "$mode" "$threads" "$features" "$threads" 1 1
      ;;
    spral-src-omp)
      run_case "$mode" "$threads" "$features" 1 "$threads" 1
      ;;
    spral-src-openblas-pthreads)
      run_case "$mode" "$threads" "$features" 1 1 "$threads"
      ;;
    spral-src-openblas-openmp)
      run_case "$mode" "$threads" "$features" 1 "$threads" "$threads"
      ;;
    mixed-rayon-omp)
      run_case "$mode" "$threads" "$features" "$threads" "$threads" "$threads"
      ;;
  esac
}

{
  printf 'SSIDS glider thread scaling\n'
  printf '  out=%s\n' "$OUT_DIR"
  printf '  repeats=%s\n' "$REPEATS"
  printf '  threads=%s\n' "$THREADS"
  printf '  modes=%s\n' "$MODES"
  printf '  test=%s\n' "$TEST_NAME"
} | tee "${OUT_DIR}/run-info.txt"

for mode in $MODES; do
  for threads in $THREADS; do
    run_mode_thread "$mode" "$threads"
  done
done

python3 scripts/ssids_plot_thread_scaling.py "$LOG_DIR" \
  --csv "${OUT_DIR}/timings.csv" \
  --svg "${OUT_DIR}/timings.svg" \
  --markdown "${OUT_DIR}/timings.md"

printf '\nWrote:\n'
printf '  %s\n' "${OUT_DIR}/timings.csv"
printf '  %s\n' "${OUT_DIR}/timings.svg"
printf '  %s\n' "${OUT_DIR}/timings.md"
