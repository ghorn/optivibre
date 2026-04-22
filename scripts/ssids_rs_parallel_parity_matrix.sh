#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export OMP_CANCELLATION="${OMP_CANCELLATION:-true}"
export AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY="${AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY:-1}"

run_with_thread_env() {
  local rayon_threads="$1"
  local omp_threads="$2"
  local openblas_threads="$3"
  shift 3
  echo
  echo "==> RAYON_NUM_THREADS=${rayon_threads} OMP_NUM_THREADS=${omp_threads} OPENBLAS_NUM_THREADS=${openblas_threads} $*"
  RAYON_NUM_THREADS="${rayon_threads}" \
    OMP_NUM_THREADS="${omp_threads}" \
    OPENBLAS_NUM_THREADS="${openblas_threads}" \
    GOTO_NUM_THREADS="${openblas_threads}" \
    "$@"
}

run_with_threads() {
  local rayon_threads="$1"
  local omp_threads="$2"
  shift 2
  run_with_thread_env "${rayon_threads}" "${omp_threads}" 1 "$@"
}

if [[ "$#" -gt 0 ]]; then
  export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  exec "$@"
fi

run_with_threads 1 1 \
  cargo test -p ssids-rs --features native-spral-src --test bitwise_parity -- --nocapture

run_with_threads 4 1 \
  cargo test -p ssids-rs --features native-spral-src parallel_determinism -- --nocapture

run_with_threads 1 4 \
  cargo test -p ssids-rs --features native-spral-src parallel_native_threads -- --nocapture

run_with_thread_env 1 1 4 \
  cargo test -p ssids-rs --no-default-features --features native-spral-src-pthreads \
    parallel_openblas_pthreads -- --nocapture

run_with_thread_env 1 4 4 \
  cargo test -p ssids-rs --no-default-features --features native-spral-src-openmp \
    parallel_openblas_threads -- --nocapture

run_with_threads 8 1 \
  cargo test -p ssids-rs --features native-spral-src concurrent_ssids_rs_stress -- --nocapture

run_with_threads 4 1 \
  cargo test -p optimal_control_problems --release --features ipopt,native-spral-src \
    glider_native_spral_exact_augmented_replay_matches_rust_to_machine_precision -- --ignored --nocapture
