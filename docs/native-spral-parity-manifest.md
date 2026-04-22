# Native SPRAL Parity Manifest

This is the source-of-truth checklist for exact-parity work. A parity change is
reviewable only when it names the upstream routine it mirrors, keeps the
comparison lane fail-closed, and promotes any new mismatch into a deterministic
regression test.

## Preflight

Normal nonlinear parity and distribution builds should use `spral-src` /
`ipopt-src` and must not depend on Homebrew, `/usr/local`, or
`/Users/greg/local/ipopt-spral`. Parity acceptance must force NLIP
`InteriorPointLinearSolver::SpralSrc` and IPOPT `linear_solver=spral`; `Auto`,
QDLDL, MUMPS, and MKL-backed IPOPT modes are runtime/compatibility choices only
and must fail parity preflight.

Source-built NLIP/IPOPT parity runs require `OMP_CANCELLATION=true`,
`RAYON_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`. NLIP must use
`InteriorPointLinearSolver::SpralSrc`; IPOPT must use `linear_solver=spral`.
The linked provenance should report `linked_solver_stack=source-built-spral`,
IPOPT `3.14.20`, source-built SPRAL, and static solver/math libraries.

Dynamic native SPRAL parity diagnostics can still run through:

```sh
scripts/native_spral_parity_preflight.sh <command>
```

That preflight sources `scripts/use_local_ipopt_spral_env.sh`, requires IPOPT
3.14.20 from `/Users/greg/local/ipopt-spral`, pins `SPRAL_SSIDS_NATIVE_LIB` to
the local `libspral.dylib`, sets `RAYON_NUM_THREADS=1` and `OMP_NUM_THREADS=1`
by default, enables `AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1`, and records the
linked OpenBLAS path plus runtime config/core/thread count underneath native
SPRAL.
For source-backed SPRAL algorithm acceptance, also set
`AD_CODEGEN_REQUIRE_SPRAL_UPSTREAM_SOURCE=1`; the preflight then fails unless
`target/native/spral-upstream/src/ssids` or `SPRAL_UPSTREAM_SSIDS_DIR` exists.
This dynamic native path is compiled only with the `dynamic-spral-parity`
feature; source-built native SPRAL is `native-spral-src`.

## NLIP vs IPOPT Lane

| Rust area | Upstream oracle | Accepted deviations |
| --- | --- | --- |
| `apply_native_spral_parity_to_nlip_options` / `apply_native_spral_parity_to_ipopt_options` | `IpSpralSolverInterface.cpp`, IPOPT option registration | None for parity runs: both lanes must use native SPRAL/SPRAL with matching order, matching scaling, block APP, `small=1e-20`, `u=1e-8`, `umax=1e-4`, GPU off. |
| Full-space KKT assembly and slack signs | `IpPDFullSpaceSolver.cpp`, `IpPDSearchDirCalc.cpp`, `IpIpoptCalculatedQuantities.cpp` | Rust may expose user-facing slack step signs, but the internal linear system must match IPOPT before conversion. |
| Linear residual and iterative refinement | `IpPDFullSpaceSolver.cpp::ComputeResiduals`, `SolveOnce`, `ComputeResidualRatio`; `IpSpralSolverInterface.cpp::IncreaseQuality` | The live `SpralSrc` path reconstructs IPOPT full-space residual rows, including upper-slack and variable-bound multiplier rows, before converting them back to the augmented correction RHS. Failed refinement asks native SPRAL to increase quality once, then uses IPOPT's `residual_ratio_singular` split between accepting the current solution and pretending singular. Snapshot replay uses the eliminated bound branch until snapshots carry variable-bound state. |
| Barrier parameter update and reset state | `IpMonotoneMuUpdate.cpp::UpdateBarrierParameter`, `IpBacktrackingLineSearch.cpp::Reset`, `IpBacktrackingLineSearch.cpp::FindAcceptableTrialPoint`, `IpFilterLSAcceptor.cpp::Reset` | On monotone mu changes, NLIP clears filter entries and rejection state like IPOPT's line-search reset, then clears watchdog/tiny-step counters like IPOPT's `last_mu_ != curr_mu` line-search check. |
| OCP NLIP tolerance wiring | `IpMonotoneMuUpdate.cpp::CalcNewMuAndTau`, `IpIpoptData::tol` | OCP NLIP options set `overall_tol` from the same minimum component tolerance used as IPOPT `tol`, so the monotone barrier lower bound uses the same scalar tolerance. |
| Hessian/Jacobian perturbation and inertia retries | `IpPDPerturbationHandler.cpp` | None for accepted state changes. Diagnostic experiments must not be committed as parity fixes. |
| Augmented-system solve and inertia checks | `IpStdAugSystemSolver.cpp`, `IpSpralSolverInterface.cpp` | None in native-SPRAL parity acceptance. `Auto`, `SparseQdldl`, and fallback solves are diagnostic only. |
| Accepted trial bound multiplier safeguard | `IpIpoptAlg.cpp::AcceptTrialPoint`, `IpIpoptAlg.cpp::correct_bound_multiplier` | `SpralSrc` parity mirrors IPOPT's correction-vector arithmetic for clamping trial bound multipliers into the `kappa_sigma` complementarity band. |
| Line search, `alpha_for_y`, filter, watchdog, SOC, restoration | `IpFilterLSAcceptor.cpp::CalculateAlphaMin`, `IpBacktrackingLineSearch.cpp::DoBacktrackingLineSearch`, `IpIpoptCalculatedQuantities.cpp::CalcFracToBound`, `IpDenseVector.cpp::FracToBoundImpl`, IPOPT `Algorithm/` source for the corresponding component | `SpralSrc` parity mirrors IPOPT's filter alpha-min formula, first-trial backtracking guard, and dense fraction-to-boundary operation order. Remaining line-search behavior should change only when direction/KKT parity evidence says the divergence has moved above the linear solve. |

## Rust SPRAL vs Native SPRAL Lane

`target/native/spral-upstream/src/ssids` or `SPRAL_UPSTREAM_SSIDS_DIR` must be
available before source-backed SPRAL algorithm edits. If the source tree is not
present, only diagnostics and exact regression promotion are allowed.

Serial fail-closed runs are the algorithmic oracle for exact bitwise parity:
`RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OMP_CANCELLATION=true`, and
`AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1`. Parallel safety is a separate gate:
Rust-owned Rayon factorization must be exactly repeatable against the serial
Rust result, while native OpenMP/OpenBLAS threaded runs are accepted by bounded
outcome checks unless a specific threaded native configuration proves bitwise
stable.

| Rust area | Upstream oracle | Accepted deviations |
| --- | --- | --- |
| Ordering, postorder, supernodes, row lists | SPRAL SSIDS analyse source | None for exact evidence unless the test is explicitly about Rust-only `Auto` ordering. |
| Matching ordering/scaling bridge | SPRAL SSIDS analyse source and IPOPT `IpSpralSolverInterface.cpp` | None for native-SPRAL parity profile. |
| Native SPRAL C option bridge, including `pivot_method` | `interfaces/C/ssids.f90`, `src/ssids/datatypes.f90`, `src/ssids/cpu/cpu_iface.f90` | None. The C `spral_ssids_options%pivot_method` is copied into Fortran `ssids_options%pivot_method`, whose APP aggressive/block/TPP constants are `1/2/3`. |
| APP/TPP pivot choice, delayed pivots, failed pivots | SPRAL SSIDS numeric factor source | None. Preserve the ladder: factorization outcome, inertia, pivot stats, solve outcome, solution bits. |
| APP `block_ldlt<32>` 2x2 multiplier codegen split | `src/ssids/cpu/kernels/block_ldlt.hxx::block_ldlt`, reached from `src/ssids/cpu/kernels/ldlt_app.cxx::Block::factor` | Mirrors the observed optimized native build for bitwise witnesses: vector-body rows use the second-product contraction, one-row scalar remainders use the first-product contraction, and the three-row short tail keeps its first two rows in the vector-body contraction lane. This is not a fallback or tolerance rule. |
| APP accepted-prefix update `calcLD<OP_N>` tiling | `src/ssids/cpu/kernels/ldlt_app.cxx::Block::update`, `src/ssids/cpu/kernels/calc_ld.hxx::calcLD<OP_N>` | None. The `LD` workspace must be shaped per target block, so the observed vector/scalar expression split resets at each APP target row tile instead of once across the whole trailing tail. |
| Dot/update/FMA order, signed zero, solve order | SPRAL SSIDS solve and update source | None for bitwise witnesses. |
| Production forward solve traversal | `src/ssids/cpu/NumericSubtree.hxx::solve_fwd`, `src/ssids/cpu/kernels/ldlt_app.cxx::ldlt_app_solve_fwd` | None. Rust production solve must gather each front, apply the APP forward solve kernel, then scatter the full front-local RHS; global sparse forward substitutes are diagnostic only. |
| Production diagonal/backward solve traversal | `src/ssids/fkeep.F90::inner_solve_cpu`, `src/ssids/cpu/NumericSubtree.hxx::solve_diag_bwd_inner<true,true>`, `src/ssids/cpu/kernels/ldlt_app.cxx::ldlt_app_solve_diag` / `ldlt_app_solve_bwd` | None. Full Rust production solves must use the native combined `diag_bwd` staging: gather a front-local RHS, apply front-local inverse-D blocks, apply the APP backward solve, then scatter eliminated rows. |
| Inertia reporting | SPRAL SSIDS inform/enquiry source | None; compare reported inertia and native enquiry data before accepting solve-bit changes. |

## Parallel Safety Gate

Use `scripts/ssids_rs_parallel_parity_matrix.sh` before starting performance
work. It preserves the serial bitwise baseline, then checks:

- exact `ssids-rs` Rayon determinism for dense APP and multi-root front trees;
- bounded native SPRAL correctness under `OMP_NUM_THREADS=1` and `4`;
- bounded source-built OpenBLAS pthread correctness under
  `OPENBLAS_NUM_THREADS=1` and `4`;
- bounded source-built OpenBLAS OpenMP correctness under
  `OMP_NUM_THREADS=1` and `4`;
- concurrent independent factor/solve and refactor/solve stress for Rust-owned
  solver objects.

The direct `spral-src::openblas_dtrsv_lower_trans_unit_matches_reference` guard
must stay green before treating any threaded OpenBLAS path as native SSIDS
evidence. On Apple Silicon this currently requires the source-built OpenBLAS
threaded artifact to use `TARGET=ARMV8`, because the autodetected `VORTEX`
threaded artifact miscomputes the `dtrsv_TLU` path used by SPRAL APP solves.

Any new parallel mismatch must still be reduced to a deterministic witness
before changing numeric behavior.

## Supervisor Checklist

- Reject parity commits that change accepted states without an upstream source
  citation in code, test, or commit notes.
- Reject parity acceptance output that used `Auto`, `SparseQdldl`, Homebrew
  SPRAL, `/usr/local` SPRAL, missing `SPRAL_SSIDS_NATIVE_LIB`, or non-pinned
  thread counts.
- Reject tolerance-only fixes that hide factorization, inertia, pivot stats,
  solve outcome, or solution-bit mismatches.
- Keep large glider diagnostics ignored/manual; promote new discoveries into
  the smallest exact regression before treating them as evidence.
