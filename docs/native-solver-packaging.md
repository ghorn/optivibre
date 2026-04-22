# Native Solver Packaging

Optivibre has three intentionally separate native-solver lanes.

## Distribution Lane

`spral-src` is the standard native SPRAL distribution crate. It builds static
SPRAL with private METIS and OpenBLAS under Cargo `OUT_DIR`, rejects system
solver/math fallbacks, and emits `DEP_SPRAL_*` metadata for downstream crates.
The default OpenBLAS mode is deterministic serial BLAS via
`openblas-serial`; `openblas-openmp` is opt-in and mutually exclusive. Consumers
should use the forwarding features on `ssids-rs` or `ipopt-src` instead of
enabling both OpenBLAS modes through Cargo feature unification.

`ipopt-src` is the standard native IPOPT distribution crate. It builds IPOPT
3.14.20 from commit `4667204c76e534d3e4df6b1462f258a4f9c681bd`, configures it
against `spral-src` via explicit SPRAL and LAPACK flags, disables MUMPS/HSL and
linear-solver-loader discovery, and re-emits the static IPOPT+SPRAL link
metadata as `DEP_IPOPT_*`.

The patched `ipopt-sys` crate defaults to `source-built-spral`, consumes
`ipopt-src`, builds only the CNLP shim and bindings, and does not run its legacy
pkg-config/system/source/binary fallback stack unless `legacy-native-build` is
explicitly enabled.

SPRAL factorization requires `OMP_CANCELLATION=true` before OpenMP runtime
initialization. Source-built IPOPT/SPRAL tests and applications that exercise
SPRAL must set that environment variable before process start.

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
