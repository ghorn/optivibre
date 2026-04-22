# `ipopt-sys`

This package provides unsafe Rust bindings to the [Ipopt](https://projects.coin-or.org/Ipopt)
non-linear optimization library.
Unlike most other wrappers for Ipopt, we link against a custom C interface called CNLP, which
mimics Ipopt's own C++ TNLP interface. This serves two purposes:

  1. It helps users who are already familiar with Ipopt's C++ interface to transition into Rust.
  2. It persists the Ipopt solver instance between subsequent solves, which eliminates unnecessary
     additional allocations for initial data and bounds.

This also means that you will need a working C++ compiler and a C++ standard library implementation
available since the CNLP shim currently uses it in the implementation.

Contributions are welcome!

## Building

This vendored copy defaults to the Optivibre source-built path:

1. `ipopt-src` builds IPOPT 3.14.20.
2. `spral-src` builds static SPRAL with private METIS and OpenBLAS.
3. `ipopt-sys` builds only the CNLP shim and Rust bindings.

The legacy pkg-config/system/source/prebuilt discovery stack is still present
for upstream comparison, but it is disabled unless the `legacy-native-build`
feature is explicitly enabled. Normal Optivibre builds should not require
Homebrew IPOPT, `/usr/local` solver libraries, MUMPS, Accelerate, or prebuilt
JuliaOpt binaries.

SPRAL requires `OMP_CANCELLATION=true` before OpenMP runtime initialization.
Set it before running tests or applications that exercise IPOPT's SPRAL solver.


### MacOS

Since macOS doesn't ship with the fortran library, you would need to install it manually.
Using homebrew you may either install `gcc` or `gfortran` directly with

```
$ brew install gcc
```

or

```
$ brew cask install gfortran
```

respectively. Since homebrew doesn't link these automatically from `/usr/local/lib`, you will have
to create the symlink manually with

```
$ ln -s /usr/local/Cellar/gcc/8.3.0_2/lib/gcc/8/libgfortran.dylib /usr/local/lib/libgfortran.dylib
```

if you have `libgfortran` from the gcc set (mind the gcc version). Otherwise create the symlink with

```
$ ln -s /usr/local/gfortran/lib/libgfortran.dylib /usr/local/lib/libgfortran.dylib
```

if you installed `gfortran` directly.

Ultimately, no matter which method you choose, `libgfortran.dylib` must be available through the linker search paths.

### Troubleshooting

* Error compiling `Mumps`:
  * Set the environment variable `export ADD_FFLAGS="-fallow-argument-mismatch"`
  * See https://github.com/coin-or-tools/ThirdParty-Mumps/issues/4 for more discussion, it affects GCC 10 and higher
  

* If you see an error message when trying to build `ipopt-sys` like the following:

  ```verbatim
  = note: ld.lld: error: undefined symbol: MKLMPI_Get_wrappers
          >>> referenced by mkl_get_mpi_wrappers.c
          >>>               mkl_get_mpi_wrappers_static.o:(mkl_serv_get_mpi_wrappers) in archive
          >>>               <mkl_root>/lib/libmkl_core.a
          clang: error: linker command failed with exit code 1 (use -v to see invocation)
  ```

  It is most likely that downstream builds will succeed if they don't use that symbol. To test if that
is the case try to build `ipopt-rs` directly.


# License

This repository is licensed under either of 

  * Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](../LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
