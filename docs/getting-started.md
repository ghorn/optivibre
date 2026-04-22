# Getting Started

## Prerequisites

- Rust `1.94`
- LLVM `22.1.x`
- macOS with Homebrew is the currently exercised developer setup

Install LLVM:

```bash
brew install llvm
```

Optional IPOPT support is source-built by default through `ipopt-src` and
`spral-src`. It requires a C/C++/Fortran toolchain plus OpenMP; Homebrew IPOPT
is not required.

```bash
brew install gcc meson ninja
```

The default source-built path uses serial OpenBLAS underneath SPRAL/IPOPT for
repeatable parity checks. Performance experiments can opt into OpenMP OpenBLAS
with the `source-built-spral-openmp` forwarding feature and should set
`OMP_NUM_THREADS` and, when needed, `OPENBLAS_NUM_PARALLEL` explicitly.

## Common Commands

Base workspace checks:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Strict lint mode:

```bash
cargo lint-strict
```

Coverage summary:

```bash
cargo llvm-cov --workspace --summary-only
```

## Optimization Runs

Public symbolic JIT NLP tests:

```bash
cargo test -p optimization --test symbolic_jit_nlp
```

SQP callback / finite-validation tests:

```bash
cargo test -p optimization --test sqp_callback_api
```

SQP hanging chain:

```bash
cargo test -p optimization --test sqp sqp_solves_hanging_chain -- --nocapture --test-threads=1
```

IPOPT hanging chain:

```bash
OMP_CANCELLATION=true \
cargo test -p optimization --features ipopt ipopt_solves_hanging_chain -- --nocapture
```

## Reports

AD cost:

```bash
cargo run -p xtask -- ad-cost-report
```

CasADi parity:

```bash
cargo run -p xtask -- casadi-parity-report
```
