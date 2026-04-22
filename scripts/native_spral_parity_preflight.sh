#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IPOPT_PREFIX="/Users/greg/local/ipopt-spral"
EXPECTED_IPOPT_VERSION="3.14.20"
EXPECTED_LIBSPRAL="${IPOPT_PREFIX}/lib/libspral.dylib"
SPRAL_UPSTREAM_SSIDS_DIR="${SPRAL_UPSTREAM_SSIDS_DIR:-${REPO_ROOT}/target/native/spral-upstream/src/ssids}"

fail() {
  printf 'native-SPRAL parity preflight failed: %s\n' "$*" >&2
  exit 1
}

source "${REPO_ROOT}/scripts/use_local_ipopt_spral_env.sh" >/dev/null

export SPRAL_SSIDS_NATIVE_LIB="${SPRAL_SSIDS_NATIVE_LIB:-${EXPECTED_LIBSPRAL}}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export AD_CODEGEN_REQUIRE_SPRAL_UPSTREAM_SOURCE="${AD_CODEGEN_REQUIRE_SPRAL_UPSTREAM_SOURCE:-0}"
export AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1

[[ -f "${SPRAL_SSIDS_NATIVE_LIB}" ]] ||
  fail "SPRAL_SSIDS_NATIVE_LIB does not exist: ${SPRAL_SSIDS_NATIVE_LIB}"
[[ "${SPRAL_SSIDS_NATIVE_LIB}" == "${EXPECTED_LIBSPRAL}" ]] ||
  fail "SPRAL_SSIDS_NATIVE_LIB must be ${EXPECTED_LIBSPRAL}, got ${SPRAL_SSIDS_NATIVE_LIB}"
[[ "${SPRAL_SSIDS_NATIVE_LIB}" != /usr/local/* ]] ||
  fail "SPRAL_SSIDS_NATIVE_LIB must not point into /usr/local"
[[ "${SPRAL_SSIDS_NATIVE_LIB}" != /opt/homebrew/* ]] ||
  fail "SPRAL_SSIDS_NATIVE_LIB must not point into /opt/homebrew"

openblas_lib="$(otool -L "${SPRAL_SSIDS_NATIVE_LIB}" | awk '/libopenblas/ { print $1; exit }')"
[[ -n "${openblas_lib}" ]] ||
  fail "native SPRAL library does not report a linked libopenblas dependency"
[[ -f "${openblas_lib}" ]] ||
  fail "linked OpenBLAS library does not exist: ${openblas_lib}"

[[ "${RAYON_NUM_THREADS}" == "1" ]] ||
  fail "RAYON_NUM_THREADS must be 1, got ${RAYON_NUM_THREADS}"
[[ "${OMP_NUM_THREADS}" == "1" ]] ||
  fail "OMP_NUM_THREADS must be 1, got ${OMP_NUM_THREADS}"

ipopt_version="$(pkg-config --modversion ipopt)"
[[ "${ipopt_version}" == "${EXPECTED_IPOPT_VERSION}" ]] ||
  fail "pkg-config --modversion ipopt returned ${ipopt_version}, expected ${EXPECTED_IPOPT_VERSION}"

ipopt_flags="$(pkg-config --cflags --libs ipopt)"
[[ "${ipopt_flags}" == *"${IPOPT_PREFIX}"* ]] ||
  fail "pkg-config --cflags --libs ipopt does not mention ${IPOPT_PREFIX}: ${ipopt_flags}"

ipopt_bin="$(command -v ipopt)"
[[ "${ipopt_bin}" == "${IPOPT_PREFIX}/bin/ipopt" ]] ||
  fail "ipopt binary must be ${IPOPT_PREFIX}/bin/ipopt, got ${ipopt_bin}"

ipopt_options="$(ipopt --print-options 2>/dev/null)"
[[ "${ipopt_options}" == *'linear_solver                 ("spral")'* ]] ||
  fail "ipopt --print-options does not report linear_solver default spral"

spral_source_anchor="present"
if [[ ! -d "${SPRAL_UPSTREAM_SSIDS_DIR}" ]]; then
  spral_source_anchor="missing"
  if [[ "${AD_CODEGEN_REQUIRE_SPRAL_UPSTREAM_SOURCE}" == "1" ]]; then
    fail "SPRAL source anchor missing: ${SPRAL_UPSTREAM_SSIDS_DIR}"
  fi
  printf 'native-SPRAL parity preflight warning: SPRAL source anchor missing: %s\n' \
    "${SPRAL_UPSTREAM_SSIDS_DIR}" >&2
  printf 'native-SPRAL parity preflight warning: set AD_CODEGEN_REQUIRE_SPRAL_UPSTREAM_SOURCE=1 for source-backed SPRAL algorithm acceptance.\n' >&2
fi

openblas_report="$(
  OPENBLAS_LIB="${openblas_lib}" python3 - <<'PY'
import ctypes
import os

lib = ctypes.CDLL(os.environ["OPENBLAS_LIB"])
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_corename.restype = ctypes.c_char_p
lib.openblas_get_num_threads.restype = ctypes.c_int

print(lib.openblas_get_config().decode())
print(lib.openblas_get_corename().decode())
print(lib.openblas_get_num_threads())
PY
)" || fail "failed to query OpenBLAS runtime provenance"
openblas_config="$(printf '%s\n' "${openblas_report}" | sed -n '1p')"
openblas_corename="$(printf '%s\n' "${openblas_report}" | sed -n '2p')"
openblas_threads="$(printf '%s\n' "${openblas_report}" | sed -n '3p')"

cat >&2 <<EOF
native-SPRAL parity preflight ok
  repo: ${REPO_ROOT}
  ipopt: ${ipopt_bin}
  ipopt_version: ${ipopt_version}
  libspral: ${SPRAL_SSIDS_NATIVE_LIB}
  openblas: ${openblas_lib}
  openblas_config: ${openblas_config}
  openblas_corename: ${openblas_corename}
  openblas_threads: ${openblas_threads}
  spral_source_anchor: ${spral_source_anchor} (${SPRAL_UPSTREAM_SSIDS_DIR})
  RAYON_NUM_THREADS: ${RAYON_NUM_THREADS}
  OMP_NUM_THREADS: ${OMP_NUM_THREADS}
  fail_closed_env: ${AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY}
EOF

if (($# > 0)); then
  exec "$@"
fi
