#!/usr/bin/env bash

set -euo pipefail

IPOPT_PREFIX="/Users/greg/local/ipopt-spral"
GCC_PREFIX="/opt/homebrew/bin"
GCC_LIBDIR="/opt/homebrew/opt/gcc/lib/gcc/current"

export PKG_CONFIG_PATH="${IPOPT_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LIBRARY_PATH="${IPOPT_PREFIX}/lib:${GCC_LIBDIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export DYLD_LIBRARY_PATH="${IPOPT_PREFIX}/lib:${GCC_LIBDIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export DYLD_FALLBACK_LIBRARY_PATH="${IPOPT_PREFIX}/lib:${GCC_LIBDIR}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"
export SPRAL_SSIDS_NATIVE_LIB="${SPRAL_SSIDS_NATIVE_LIB:-${IPOPT_PREFIX}/lib/libspral.dylib}"
export RUSTFLAGS="-Lnative=${IPOPT_PREFIX}/lib -Lnative=${GCC_LIBDIR}${RUSTFLAGS:+ ${RUSTFLAGS}}"
export CC="${GCC_PREFIX}/gcc-15"
export CXX="${GCC_PREFIX}/g++-15"
export OMP_CANCELLATION=TRUE
export OMP_PROC_BIND=TRUE
export PATH="${IPOPT_PREFIX}/bin:${PATH}"

echo "Using local SPRAL IPOPT from ${IPOPT_PREFIX}"
