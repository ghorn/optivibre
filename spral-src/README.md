# spral-src

`spral-src` builds SPRAL, METIS, and OpenBLAS from source for Cargo consumers.
It intentionally does not probe system SPRAL, METIS, OpenBLAS, Accelerate,
Homebrew, MacPorts, `/usr/local`, or pkg-config solver/math libraries.

OpenMP is required. A target-capable C, C++, Fortran, Meson, Ninja, `make`, and
OpenMP toolchain must be available for the selected Cargo target.

SPRAL also requires OpenMP cancellation at runtime. Set
`OMP_CANCELLATION=true` before calling SSIDS factorization.
