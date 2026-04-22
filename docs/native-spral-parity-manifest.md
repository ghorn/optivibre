# Native SPRAL Parity Manifest

This is the source-of-truth checklist for exact-parity work. A parity change is
reviewable only when it names the upstream routine it mirrors, keeps the
comparison lane fail-closed, and promotes any new mismatch into a deterministic
regression test.

## Preflight

Normal distribution builds should use `spral-src` / `ipopt-src` and must not
depend on Homebrew, `/usr/local`, or `/Users/greg/local/ipopt-spral`.

Run parity acceptance commands through:

```sh
scripts/native_spral_parity_preflight.sh <command>
```

The preflight sources `scripts/use_local_ipopt_spral_env.sh`, requires IPOPT
3.14.20 from `/Users/greg/local/ipopt-spral`, pins `SPRAL_SSIDS_NATIVE_LIB` to
the local `libspral.dylib`, sets `RAYON_NUM_THREADS=1` and `OMP_NUM_THREADS=1`
by default, and enables `AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY=1`.
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
| Linear residual and iterative refinement | `IpPDFullSpaceSolver.cpp` | None. Use IPOPT's full-system residual ratio, min/max refinement steps, improvement test, quality retry, and pretend-singular semantics. |
| Hessian/Jacobian perturbation and inertia retries | `IpPDPerturbationHandler.cpp` | None for accepted state changes. Diagnostic experiments must not be committed as parity fixes. |
| Augmented-system solve and inertia checks | `IpStdAugSystemSolver.cpp`, `IpSpralSolverInterface.cpp` | None in native-SPRAL parity acceptance. `Auto`, `SparseQdldl`, and fallback solves are diagnostic only. |
| Line search, `alpha_for_y`, filter, watchdog, SOC, restoration | IPOPT `Algorithm/` source for the corresponding component | Change only after direction/KKT parity evidence says the divergence has moved above the linear solve. |

## Rust SPRAL vs Native SPRAL Lane

`target/native/spral-upstream/src/ssids` or `SPRAL_UPSTREAM_SSIDS_DIR` must be
available before source-backed SPRAL algorithm edits. If the source tree is not
present, only diagnostics and exact regression promotion are allowed.

| Rust area | Upstream oracle | Accepted deviations |
| --- | --- | --- |
| Ordering, postorder, supernodes, row lists | SPRAL SSIDS analyse source | None for exact evidence unless the test is explicitly about Rust-only `Auto` ordering. |
| Matching ordering/scaling bridge | SPRAL SSIDS analyse source and IPOPT `IpSpralSolverInterface.cpp` | None for native-SPRAL parity profile. |
| APP/TPP pivot choice, delayed pivots, failed pivots | SPRAL SSIDS numeric factor source | None. Preserve the ladder: factorization outcome, inertia, pivot stats, solve outcome, solution bits. |
| Dot/update/FMA order, signed zero, solve order | SPRAL SSIDS solve and update source | None for bitwise witnesses. |
| Inertia reporting | SPRAL SSIDS inform/enquiry source | None; compare reported inertia and native enquiry data before accepting solve-bit changes. |

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
