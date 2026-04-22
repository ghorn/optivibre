# Native Solver Packaging

Optivibre has three intentionally separate native-solver lanes.

## Distribution Lane

`spral-src` is the standard native SPRAL distribution crate. It builds static
SPRAL with private METIS and OpenBLAS under Cargo `OUT_DIR`, rejects system
solver/math fallbacks, and emits `DEP_SPRAL_*` metadata for downstream crates.
The default OpenBLAS mode is deterministic serial BLAS via
`openblas-serial`; `openblas-pthreads` is the supported threaded BLAS
acceptance path; `openblas-openmp` is the OpenMP-threaded BLAS path. Consumers
should use the forwarding features on `ssids-rs` or `ipopt-src` instead of
enabling multiple OpenBLAS modes through Cargo feature unification.
`OPENBLAS_NUM_PARALLEL` is forwarded to OpenBLAS `NUM_PARALLEL` for OpenMP
builds when multiple independent OpenMP regions may call BLAS concurrently.

On Apple Silicon, `spral-src` forces OpenBLAS threaded builds to `TARGET=ARMV8`
unless `OPENBLAS_TARGET` is explicitly provided. OpenBLAS 0.3.32 autodetects
`VORTEX` on this host, and the `USE_THREAD=1` VORTEX artifact fails the direct
`dtrsv_` guard used by SPRAL APP forward/backward solves.

`ipopt-src` is the standard native IPOPT distribution crate. It tracks the
published `ipopt-src` feature vocabulary where possible, but Optivibre's default
is the strict `source-built-spral` lane. It builds IPOPT 3.14.20 from commit
`4667204c76e534d3e4df6b1462f258a4f9c681bd`, configures it against `spral-src`
via explicit SPRAL and LAPACK flags, disables MUMPS/HSL and
linear-solver-loader discovery, and re-emits the static IPOPT+SPRAL link
metadata as `DEP_IPOPT_*`. Upstream-compatible MUMPS/MKL feature names are
reserved for a non-parity compatibility lane and intentionally fail closed until
they can share one BLAS owner or are documented as bringing their own BLAS.

The upstream `openblas-src` patch lives under `third_party/openblas-src`, but it
is not selected by the default Optivibre workspace because the SPRAL/IPOPT path
does not depend on `openblas-src`. The patch adds mutually exclusive
`threading-serial`, `threading-pthread`, and `threading-openmp` features that
feed the existing `openblas-build` `USE_THREAD`/`USE_OPENMP` controls. The
SPRAL/IPOPT default path still consumes OpenBLAS only through `spral-src`,
avoiding duplicate BLAS archives.

The patched `ipopt-sys` crate defaults to `source-built-spral`, consumes
`ipopt-src`, builds only the CNLP shim and bindings, and does not run its legacy
pkg-config/system/source/binary fallback stack unless `legacy-native-build` is
explicitly enabled.

SPRAL factorization requires `OMP_CANCELLATION=true` before OpenMP runtime
initialization. Source-built IPOPT/SPRAL tests and applications that exercise
SPRAL must set that environment variable before process start.

Threaded OpenBLAS is intentionally opt-in. Before performance work depends on
parallelism, run
`scripts/ssids_rs_parallel_parity_matrix.sh`: serial bitwise SPRAL parity
remains the algorithmic oracle, Rust Rayon factorization is required to be
exactly repeatable, and native SPRAL OpenMP with serial OpenBLAS is checked
with bounded residual and solution criteria. The green threaded BLAS lane is
`native-spral-src-pthreads`, which compares OpenBLAS pthread counts by bounded
outcome, inertia, pivot stats, residual, and solution criteria. The
`native-spral-src-openmp` lane is also in the green matrix and compares the
combined native SPRAL/OpenBLAS OpenMP path with the same bounded criteria.

## Pure Rust Lane

`ssids-rs` is the pure Rust SSIDS implementation. It is the solver backend NLIP
uses when `InteriorPointLinearSolver::SsidsRs` is selected. It should not be
described as native SPRAL or used as the oracle for NLIP/IPOPT parity.

When `ssids-rs` is built with `native-spral-src`, its native SPRAL wrapper links
to `spral-src`. That is the normal source-built native path exposed as
`InteriorPointLinearSolver::SpralSrc`.

## Parity Diagnostics Lane

Dynamic loading of `/Users/greg/local/ipopt-spral` or a manually supplied
`SPRAL_SSIDS_NATIVE_LIB` is parity-only. It is compiled only behind
`dynamic-spral-parity` and must not appear in normal webapp solver choices.

The upstream SPRAL source tree under `target/native/spral-upstream/src/ssids`
and kernel shims are also parity tools. They can be used to classify and
regress mismatches, but diagnostics and printouts belong in parity harnesses,
not in `spral-src`.

`spral-sys` was checked as a possible equivalence shortcut. The published
0.1.0 crate is an FFI binding with `shared_libraries` / `static_libraries`
features, not a source distribution path, so the current direct C ABI parity
wrapper remains the controlled parity route.
