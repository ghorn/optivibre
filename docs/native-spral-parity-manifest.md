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
For the exact-Hessian lane, IPOPT's source default `mu_strategy` is `monotone`;
adaptive/free-mu behavior is a non-default option-profile lane and must not be
reported as covered by default nonlinear parity until it has its own
source-faithful implementation and witnesses.
`IpoptOptions::default` mirrors IPOPT's source termination defaults. The
source-default glider witness is closed through restoration; the OCP/studio
default profile remains stricter until that user-facing profile choice is made
explicit with option-summary tests.

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
| `apply_native_spral_parity_to_nlip_options` / `apply_native_spral_parity_to_ipopt_options`; OCP `default_nlip_config` | `IpSpralSolverInterface.cpp::PivotMethodNameToNum`, IPOPT option registration; SPRAL `include/spral_ssids.h`, `src/ssids/datatypes.f90` | None for parity runs: both lanes must use native SPRAL/SPRAL with matching order, matching scaling, IPOPT option `spral_pivot_method=block`, `small=1e-20`, `u=1e-8`, `umax=1e-4`, GPU off. IPOPT 3.14.20 writes raw `control_.pivot_method=1` for `block`; SPRAL 2025.09.18 documents C/Fortran value `1` as aggressive APP, so NLIP's parity profile and OCP default use the same raw-control equivalent. |
| `SpralSrc` matching ordering/scaling analyse cadence | `IpSpralSolverInterface.cpp::InitializeStructure`, `IpSpralSolverInterface.cpp::MultiSolve`; SPRAL `interfaces/C/ssids.f90::spral_ssids_analyse_ptr32` | IPOPT skips structure-only analyse when `control_.ordering == 2` and `control_.scaling == 3`, then re-runs `spral_ssids_analyse_ptr32(..., val_)` on each new matrix before factorization. NLIP's `SpralSrc` path treats matching as value-dependent, reanalyses instead of refactorizing, and pins this with `interior_point_native_spral_matching_reanalyses_numeric_pattern`. The native ptr32 adapter decodes returned analysis-order positions with the active SPRAL `array_base`, so IPOPT-compatible one-based matching orders become valid zero-based Rust permutations. |
| `SsidsRs` matching ordering/scaling analyse cadence | `IpSpralSolverInterface.cpp::InitializeStructure`, `IpSpralSolverInterface.cpp::MultiSolve`; SPRAL `interfaces/C/ssids.f90::spral_ssids_analyse_ptr32`; SPRAL `src/ssids/{analyse,scaling,factor}.f90` | `SsidsRs` must not fall back to `SpralSrc`. For SPRAL matching, the Rust SSIDS-RS path mirrors the value-dependent analyse cadence instead of reusing a structure-only symbolic factor, feeds saved matching scaling into numeric factorization, and uses the same augmented-KKT primal/dual regularization shifts as `SpralSrc`. The Albatross direct-collocation parity tests run SSIDS-RS as the primary backend and replay the same KKT with source-SPRAL; the first KKT must solve consistently. Restoration inner reduced-KKT diagnostics preserve the primary solver identity, dump successful and failed attempts under `aug_resto_inner/` or `aug_resto_inner/retry_*`, and require replay to either solve consistently or report the same explicit linear failure. |
| `SpralSrc` single-RHS solve entrypoint | `IpSpralSolverInterface.cpp::MultiSolve`; SPRAL `interfaces/C/ssids.f90::spral_ssids_solve` | NLIP's source-built SPRAL path calls `spral_ssids_solve` with `nrhs=1` and `ldx=n`, matching IPOPT even for one RHS. The existing `solve1` wrapper remains available for native wrapper smoke tests and Rust/native SSIDS parity diagnostics, but not for NLIP/IPOPT `SpralSrc` acceptance. |
| Full-space KKT assembly and slack signs | `IpPDFullSpaceSolver.cpp`, `IpPDSearchDirCalc.cpp`, `IpIpoptCalculatedQuantities.cpp` | Rust may expose user-facing slack step signs, but the internal linear system must match IPOPT before conversion. |
| IPOPT-order sparse Jacobian products | `IpGenTMatrix.cpp::MultVectorImpl`, `IpGenTMatrix.cpp::TransMultVectorImpl` | The source-built C++ path contracts `y += value * x` to FMA. NLIP's IPOPT-order sparse matvec helpers use `mul_add` and keep the stored triplet traversal order; generic non-IPOPT sparse helpers are not changed by this parity rule. |
| `ExpansionMatrix` multiplier reconstruction | `IpExpansionMatrix.cpp::SinvBlrmZMTdBrImpl`; `IpPDFullSpaceSolver.cpp::SolveOnce` | The source-built C++ path contracts `(R +/- Z * D) / S` to FMA before division for slack and variable-bound multiplier steps. This is pinned by the env-gated IPOPT expansion dump: real glider rows whose volatile separate candidate differs from FMA have `x_bits == fma_bits`, and synthetic source-like plus/minus expressions in the same translation unit also match `std::fma`. NLIP mirrors this for `v_U`, `z_L`, and `z_U` reconstruction before the final `PDSearchDirCalc` sign flip and during full-space iterative-refinement corrections. |
| Inequality slack internal sign convention | `IpIpoptCalculatedQuantities.cpp::curr_slack_s_U`, `curr_grad_lag_s`, `curr_sigma_s`; `IpDefaultIterateInitializer.cpp::push_variables`; `IpOrigIpoptNLP.cpp::relax_bounds`; `IpPDFullSpaceSolver.cpp::SolveOnce` | NLIP stores inequality slacks in IPOPT's upper-bound internal sign (`d(x) - s = 0`, `s <= d_U`) and converts explicitly to upper-slack distance `d_U - s` for barrier/complementarity calculations. The normalized one-sided inequality rows now keep raw `d(x)` values while `d_U` carries the zero-bound relaxation amount. |
| Linear residual and iterative refinement | `IpPDFullSpaceSolver.cpp::RegisterOptions`, `InitializeImpl`, `ComputeResiduals`, `SolveOnce`, `ComputeResidualRatio`; `IpSpralSolverInterface.cpp::IncreaseQuality` | The live `SpralSrc` path reconstructs IPOPT full-space residual rows, including upper-slack and variable-bound multiplier rows, before converting them back to the augmented correction RHS. NLIP exposes and consumes IPOPT's full-space refinement controls (`min_refinement_steps`, `max_refinement_steps`, `residual_ratio_max`, `residual_ratio_singular`, `residual_improvement_factor`) in both regular and restoration solves. The nonzero `neg_curv_test_tol` path now solves through wrong inertia, applies IPOPT's inertia-free curvature heuristic with the `neg_curv_test_reg` toggle, and retries with wrong-inertia perturbation only on heuristic failure. Failed refinement asks native SPRAL to increase quality once, then uses IPOPT's `residual_ratio_singular` split between accepting the current solution and pretending singular. Snapshot replay uses the eliminated bound branch until snapshots carry variable-bound state. |
| Barrier parameter update and reset state | `IpMonotoneMuUpdate.cpp::UpdateBarrierParameter`, `IpMonotoneMuUpdate.cpp::CalcNewMuAndTau`, `IpBacktrackingLineSearch.cpp::Reset`, `IpBacktrackingLineSearch.cpp::FindAcceptableTrialPoint`, `IpFilterLSAcceptor.cpp::Reset` | Monotone mu updates use IPOPT's `mu_target` plus `tol/compl_inf_tol` floor, not the adaptive `mu_min` option. NLIP now keeps IPOPT's split between line-search `tiny_step_last_iteration_` and barrier-update `IpData().tiny_step_flag()`: only repeated tiny steps force the mu-update loop, and if `mu` cannot change, NLIP returns `SearchDirectionTooSmall` like IPOPT's `STOP_AT_TINY_STEP`. The tiny-step force is consumed after one update-loop pass, and the first call keeps IPOPT's `initialized_ == false` fast-loop behavior. On monotone mu changes, NLIP clears filter entries and rejection state like IPOPT's line-search reset, then clears watchdog/tiny-step counters like IPOPT's `last_mu_ != curr_mu` line-search check. |
| Initial point bound/slack push | `IpDefaultIterateInitializer.cpp::RegisterOptions`, `IpDefaultIterateInitializer.cpp::push_variables` | Scalar NLIP bound and slack initialization mirrors IPOPT's snap-to-bound pass, lower-side `tiny_double` subtraction, and upper-side `tiny_double` restoration before computing the first KKT system. Public `bound_push`, `bound_frac`, `slack_bound_push`, and `slack_bound_frac` validation now mirrors IPOPT's strict positive lower bounds and `0.5` fractional cap; zero push/fraction remains internal-only for restoration recursion. |
| OCP NLIP tolerance wiring | `IpOptErrorConvCheck.cpp`, `IpIpoptData::tol`, `IpMonotoneMuUpdate.cpp::CalcNewMuAndTau` | `IpoptOptions::default` mirrors IPOPT source termination defaults. OCP/studio NLIP and IPOPT wrappers keep the existing stricter profile and compute `tol` from the minimum component tolerance, so the monotone barrier lower bound uses the same scalar tolerance in both lanes. A source-default glider profile now passes with matching accepted-trace length through restoration; moving OCP/studio defaults is a config/profile decision, not a blocker for the source-default nonlinear algorithm witness. |
| Hessian/Jacobian perturbation and inertia retries | `IpPDPerturbationHandler.cpp` | None for accepted state changes. Diagnostic experiments must not be committed as parity fixes. |
| Augmented-system solve and inertia checks | `IpStdAugSystemSolver.cpp`, `IpSpralSolverInterface.cpp` | None in native-SPRAL parity acceptance. `Auto`, `SparseQdldl`, and fallback solves are diagnostic only. |
| Accepted trial tiny-slack bound shifts and bound multiplier safeguard | `IpIpoptAlg.cpp::AcceptTrialPoint`, `IpIpoptCalculatedQuantities.cpp::CalculateSafeSlack`, `IpIpoptAlg.cpp::correct_bound_multiplier` | `SpralSrc` parity mirrors IPOPT's tiny-slack floor/target/move-cap arithmetic, regular-trial bound adjustment, and correction-vector arithmetic for clamping trial bound multipliers into the `kappa_sigma` complementarity band. |
| Line search, `alpha_for_y`, filter, watchdog, SOC, corrector, restoration | `IpFilterLSAcceptor.cpp::CalculateAlphaMin`, `IpFilterLSAcceptor.cpp::IsFtype`, `IpFilterLSAcceptor.cpp::CheckAcceptabilityOfTrialPoint`, `IpFilterLSAcceptor.cpp::TrySecondOrderCorrection`, `IpFilterLSAcceptor.cpp::TryCorrector`, `IpBacktrackingLineSearch.cpp::DoBacktrackingLineSearch`, `IpBacktrackingLineSearch.cpp::PerformDualStep`, `IpBacktrackingLineSearch.cpp::PerformMagicStep`, `IpIpoptCalculatedQuantities.cpp::CalcFracToBound`, `IpDenseVector.cpp::FracToBoundImpl`, IPOPT `Algorithm/` source for the corresponding component | `SpralSrc` parity mirrors IPOPT's filter alpha-min formula, feasible-reference tiny positive `reference_gradBarrTDelta_` adjustment, theta-max rejection, objective max-increase rejection, filter-reset counter semantics, first-trial backtracking guard, dense fraction-to-boundary operation order, SOC constraint-residual `AddOneVector(1.0, trial, alpha)` branch order, non-default affine/primal-dual corrector RHS/acceptance behavior, non-default `min-dual-infeas` / `safer-min-dual-infeas` equality-multiplier step formulas, and non-default upper-slack `magic_steps` projection behavior. Remaining line-search behavior should change only with a reduced witness whose current state, KKT solve, accepted direction, and alpha traces still match before the branch. |

### Current Nonlinear Difference Backlog

`docs/ipopt-parity-coverage.md` is the detailed nonlinear branch ledger. The
current exact-Hessian backlog is:

| Priority | Difference | Lane |
| --- | --- | --- |
| P0 | `IpAugRestoSystemSolver` reduced restoration KKT | Closed hard default-lane gap; current glider trace has no tracked direction, alpha, accepted-direction, or accepted-state drift |
| P1 | IPOPT source-default profile versus stricter OCP/studio profile | Config/profile gap |
| P2 | `IpAdaptiveMuUpdate` adaptive/free-mu mode | Mostly covered non-default exact-Hessian feature support: `loqo`, default and non-default `quality-function`, `probing`, `obj-constr-filter`, `never-monotone-mode`, default `kkt-error` globalization, nonzero safeguard, non-default KKT-error reduction knobs, restore-previous-iterate, quality-function seeded `HaveDeltas` refinement, identity sigma-space section search, and explicit adaptive restoration handoff are covered; a natural fixed-mode switch accepted-trace witness remains open |
| P3 | `IpFilterLSAcceptor::TryCorrector` affine/primal-dual corrector | Covered non-default exact-Hessian feature lane; source defaults keep `corrector_type=none` |
| P4 | `IpPDPerturbationHandler` degeneracy/inertia branches | Coverage active-watch backlog; non-default `perturb_always_cd=yes` is implemented and pinned |
| P5 | `IpPDFullSpaceSolver` refinement, failure, and quality-retry branches | Refinement option surface, inertia-free negative-curvature test options, failed-refinement decision order, quality-function `improve_solution=true` seeded refinement, and live quality-retry/accept-current/pretend-singular witnesses are implemented and pinned; remaining coverage active-watch rows are direct linear-solver failure error paths unless a natural witness appears without editing linear internals |
| P6 | `IpBacktrackingLineSearch` fallback, restoration, watchdog, and soft-restoration edge paths | Coverage active-watch backlog; non-default `magic_steps`, min-dual-infeas `alpha_for_y` variants, acceptable-iterate backup storage, tiny-step eval-error fallback, low-`tiny_step_y_tol` dual-threshold behavior, and repeated-tiny-step `STOP_AT_TINY_STEP` termination are now covered |
| P7 | `IpFilterLSAcceptor` theta/filter/SOC edge paths | Coverage active-watch backlog; theta-max, objective max-increase, filter reset, and tiny positive feasible-reference f-type behavior are now source-pinned |
| P8 | `IpDefaultIterateInitializer` equality-multiplier, failure, switch, and low-push branches | Coverage active-watch backlog; invalid public push/fraction values are option-rejected for the main initializer |
| P9 | `IpIpoptCalculatedQuantities` cache/component branches | Bookkeeping unless component values drift |

This manifest should not claim full nonlinear parity while remaining uncovered
exact-Hessian branch families are open. The glider first-divergence diagnostic
should remain green before any algorithmic parity claim is broadened beyond the
currently covered exact-Hessian lanes.

### Nonlinear Branch Checklist

This checklist is the escalation order for glider parity. A branch below should
not receive an accepted-state-changing fix unless the current earliest
observable mismatch has moved into that branch.

| IPOPT routine / branch | Current status | Evidence required before changing behavior |
| --- | --- | --- |
| `IpDefaultIterateInitializer.cpp::RegisterOptions`, `push_variables`, and multiplier initialization | Covered by max-iter-0 state comparisons, non-default bound/slack push tests, public push/fraction option-rejection tests, `bound_mult_init_val`, `bound_mult_init_method=mu-based`, `constr_mult_init_max=0`, positive low-cap `yinitnrm > constr_mult_init_max`, and square-problem zero-`y_c` witnesses. | New max-iter-0 mismatch in x, slack, y, z/v, damping, objective, or residual components. |
| Fixed-variable and bound-relaxation preprocessing | Covered by a fixed-bound quadratic witness against IPOPT's default `fixed_variable_treatment=make_parameter`, plus a bound-constrained `kappa_d=0` and `bound_relax_factor=0` witness. | Mismatch in fixed-variable projection, compact/full snapshot expansion, damping terms, or relaxed bound construction. |
| `IpOptErrorConvCheck.cpp::CONVERGED_TO_ACCEPTABLE_POINT` and `IpBacktrackingLineSearch.cpp::CurrentIsAcceptable` | Covered by focused source-built tests with tight strict tolerances and loose acceptable tolerances. `acceptable_iter=1` pins final acceptable termination; `acceptable_iter=2` forces the pre-termination acceptable-backup line-search path. | Acceptable status, warning/success classification, backup iterate tracking, or final iterate diverges before strict convergence. |
| `IpIpoptAlg.cpp::ConvergenceCheck::MAXITER_EXCEEDED` | Covered by deterministic `max_iter=0` and `max_iter=1` witnesses that require IPOPT `MaximumIterationsExceeded` and NLIP `InteriorPointSolveError::MaxIterations`, with initial or last-accepted diagnostics retained. | Different failure status, missing diagnostic state, or a nonlinear status mismatch that appears before any accepted trial. |
| `IpPDSearchDirCalc.cpp::ComputeSearchDirection` | Covered for the active `SpralSrc` lane by augmented KKT/RHS/direction probes. | Captured-order KKT/RHS mismatch before the first solve, or a direction mismatch not explained by linear solve residuals. |
| `IpPDFullSpaceSolver.cpp::SolveOnce`, `ComputeResiduals`, and `ComputeResidualRatio` | Covered by full-space residual reconstruction, iterative-refinement traces, source refinement option controls, inertia-free curvature-test helper coverage, per-block RHS/solution fingerprints, and native matching reanalysis cadence. | Difference in IPOPT residual-ratio loop count, correction RHS orientation, inertia-free curvature retry, SPRAL analyse/factor entrypoint/cadence, or `Pd_U` / `Px_*` reconstruction for a reduced witness. |
| `IpPDPerturbationHandler.cpp` and augmented-system inertia retries | Covered for the current source-SPRAL glider run: regularization and inertia stay aligned through the full accepted trace. Non-default `perturb_always_cd=yes` now mirrors IPOPT's permanent constraint-linearization perturbation. | Regularization, inertia, or quality-retry marker diverges before accepted-state drift. |
| `IpBacktrackingLineSearch.cpp::PerformDualStep` and `alpha_for_y` | Covered for branch shape and source order in the source-SPRAL glider runs: the current glider first-divergence diagnostic has no `alpha_pr`, `alpha_du`, or `alpha_y` gaps above `1e-16` across all 152 accepted steps. Focused tests also exercise the implemented `alpha_for_y` option profiles, `min-dual-infeas` / `safer-min-dual-infeas` trial dual-infeasibility formulas, and strict `alpha_for_y_tol` threshold profiles against IPOPT source options. | A first mismatch in `alpha_pr`, `alpha_du`, `alpha_y`, or multiplier step application with matching current state and direction vectors. |
| `IpIpoptData.cpp::SetTrial*FromStep` and `IpIpoptAlg.cpp::AcceptTrialPoint` | Covered by per-component accepted-state probes and source-faithful `DenseVector::AddTwoVectors` order. | Trial state differs with matching alpha and direction vectors. |
| `IpIpoptAlg.cpp::correct_bound_multiplier` | Covered for the active lane; variable-bound correction vectors are source-backed, and a focused `kappa_sigma=1` witness forces the bound-multiplier safeguard branch against IPOPT source. | First accepted-state mismatch in variable-bound multipliers or correction-vector diagnostics. |
| `IpMonotoneMuUpdate.cpp` and line-search reset on mu change | Covered by monotone-mu update tests, positive `mu_target`, source-backed reset semantics, and a source-built repeated-tiny-step `SearchDirectionBecomesTooSmall` witness. | Barrier parameter, tau, filter reset, watchdog reset, or tiny-step flag diverges before direction/state drift. |
| `IpAdaptiveMuUpdate.cpp` free-mu mode and fixed-mode switch | Not part of the exact-Hessian IPOPT default lane. The `mu_strategy=adaptive` lane is source-backed for `loqo`, default and non-default `quality-function`, and `probing` oracles, `never-monotone-mode`, `obj-constr-filter`, default and non-default `kkt-error` globalization knobs, nonzero safeguard, restore-previous-iterate, quality-function `HaveDeltas` seeded refinement, identity sigma-space section search, and explicit adaptive restoration handoff through `resto.mu_strategy=adaptive`. Helper tests pin adaptive filter acceptability/frontier update, KKT-error reference-window semantics, fixed/free-mode transition semantics, restore-previous-iterate fixed-mode state selection, restoration-prefixed option overrides/default `mu_min`, the safeguard's initial normalized infeasibility ratio formula, non-default quality-function KKT error terms, and identity sigma-space search; source-built IPOPT comparisons cover the implemented option profiles. A natural accepted-trace fixed-mode switch witness remains a non-default feature gap. | First accepted-state trace for the remaining adaptive branch: natural fixed-mode transition. |
| `IpFilterLSAcceptor.cpp`, watchdog, SOC, corrector, tiny-step, and restoration | Filter local/global acceptability, `CalculateAlphaMin`, feasible-reference `IsFtype` adjustment, theta-factor option rejection, theta-max rejection, objective max-increase rejection, filter reset, first-trial SOC/corrector gating, filter update, watchdog arming, successful watchdog exit, lowercase watchdog trial acceptance, trial-budget `StopWatchDog` restore/retry, and tiny-step source branches are checked. The active glider SOC branch is source-backed by `TrySecondOrderCorrection` and `DenseVector::AddOneVector`; focused option tests exercise SOC-disabled, watchdog-disabled, a watchdog-trigger profile with matching accepted traces and IPOPT's `W` info marker, a lowercase `w` non-success watchdog trial profile, a trial-budget `StopWatchDog` restore/retry profile, tiny-step-disabled, forced tiny-step acceptance, the slack-side tiny-step relative-step rejection formula, restoration original-objective eval-error max-iteration, non-default `alpha_red_factor`, and monotone-fast-decrease-disabled paths. `TryCorrector` is now ported for both `corrector_type=primal-dual` and `corrector_type=affine` in the `SpralSrc` source-parity lane, with focused linearly constrained quadratic comparisons against IPOPT source; it remains unreachable in the source-default profile because IPOPT defaults `corrector_type=none`. Watchdog arming now occurs after current-direction construction, matching `BacktrackingLineSearch::StartWatchDog` and the filter acceptor's current grad-barrier dot direction. Active watchdog search now keeps `alpha_min` at the max feasible primal step, matches lowercase `w` trial acceptance while leaving the watchdog armed, skips SOC while `in_watchdog_` is active, and restores the stored iterate/direction before retrying after trial-budget exhaustion. The active-watchdog pre-line-search stop on restoration/tiny-step is implemented and pinned by focused diagnostics, but no natural source-built witness has been found. The current glider first-divergence diagnostic reports `nlip_steps=152 ipopt_steps=152`, no direction gaps above `1e-8`, no alpha gaps above `1e-16`, no accepted-direction gaps above `1e-8`, and no accepted-state gaps above `1e-10` except barrier subproblem error at index 18 (`3.201e-10`, with no gap above `1e-8`). | Matching accepted state and directions immediately before a divergent accept/reject, watchdog, SOC/corrector, tiny-step, or restoration decision; do not use restoration, SOC, or corrector as a repair while direction/state drift is already present. |

Current glider solve-boundary diagnostics:

- Current glider first-divergence diagnostic: `nlip_steps=152 ipopt_steps=152`,
  no direction gaps above `1e-8`, no `alpha_pr` / `alpha_du` / `alpha_y` gaps
  above `1e-16`, no accepted-direction gaps above `1e-8` in x, y_c, y_d, or
  v_U, and no accepted-state gaps above `1e-10` for x, multipliers,
  stationarity, barrier primal/dual/complementarity, objective gradient,
  Jacobian-transpose products, Lagrangian gradients, or damping vectors. The
  only tracked accepted-state scalar above `1e-10` is barrier subproblem error
  at index 18 (`3.201e-10`), with no gap above `1e-8`. Treat older glider notes
  below as historical checkpoints unless they explicitly mention this current
  no-tracked-drift state.

- Run `scripts/ipopt_parity_coverage.sh` for the nonlinear branch-coverage
  audit. It enables the opt-in `IPOPT_SRC_LLVM_COVERAGE=1` source build, writes
  direct IPOPT Algorithm and Rust NLIP-core coverage reports under
  `target/reports/ipopt-parity-coverage/`, and uses
  `docs/ipopt-parity-coverage.md` as the branch ledger. It also writes a
  generated `branch-ledger.md` report that lists watched IPOPT branch surfaces
  and Rust core unhit branch-like lines; the script fails if a Rust core
  branch-like line remains classified as `needs audit`. The coverage script
  does not clean `spral-src` by default; set
  `IPOPT_PARITY_COVERAGE_REBUILD_NATIVE=1` only for an intentional full native
  rebuild.
- Set `GLIDER_PARITY_IPOPT_PRINT_LEVEL=12` with
  `GLIDER_PARITY_PRINT_IPOPT_AUGMENTED_FINGERPRINTS=1` to make IPOPT print the
  `Trhs` and `SOL` vectors needed for augmented KKT replay; lower print levels
  can produce `rhs_count=0 sol_count=0`.
- Set `GLIDER_PARITY_NLIP_AUGMENTED_DUMP_DIR=/tmp/...` when an exact SPRAL
  interface replay needs persistent NLIP KKT files; otherwise
  `GLIDER_PARITY_PRINT_NLIP_AUGMENTED_FINGERPRINTS=1` uses a temporary dump
  directory that is removed after the test process exits.
- Set `GLIDER_PARITY_IPOPT_SPRAL_DUMP_DIR=/tmp/...` to dump IPOPT's direct
  source-SPRAL interface calls before analyse/factor/solve and after solve.
  Combine it with `GLIDER_PARITY_PRINT_IPOPT_SPRAL_LADDER=1` and
  `GLIDER_PARITY_IPOPT_SPRAL_LADDER_MAX_ITERS=<n>` when comparing NLIP and
  IPOPT matrix/RHS/solution/refinement fingerprints across accepted steps.
- Set `GLIDER_PARITY_IPOPT_JAC_DUMP_DIR=/tmp/...` to dump IPOPT's effective
  `GenTMatrix` Jacobian triplets, values, multipliers, cached products, and a
  forced callback-time transpose product. Use
  `GLIDER_PARITY_IPOPT_JAC_DUMP_ITER=<iter>` to keep the dump focused.
- Set `GLIDER_PARITY_IPOPT_EXPANSION_DUMP_DIR=/tmp/...` to dump IPOPT's
  `ExpansionMatrix::SinvBlrmZMTdBrImpl` inputs, compiled output, volatile
  separate multiply-add candidate, and `std::fma` candidate. Use
  `GLIDER_PARITY_IPOPT_EXPANSION_DUMP_CALL=<call>` to keep the dump focused.
  The source patch also records one synthetic plus/minus source-expression
  case per dump file so contraction can be classified even when a particular
  problem's real rows do not distinguish the two formulas.
- Set `GLIDER_PARITY_IPOPT_DENSE_ADD_DUMP_DIR=/tmp/...` to classify
  `IpDenseVector.cpp::AddTwoVectorsImpl` current-plus-step arithmetic. Use
  `GLIDER_PARITY_IPOPT_DENSE_ADD_DUMP_CALL=<call>` to keep the dump focused.
- Set `GLIDER_PARITY_IPOPT_TSYM_DUMP_DIR=/tmp/...` to dump
  `IpTSymLinearSolver.cpp` solve calls before and after the symmetric linear
  solve. Use `GLIDER_PARITY_IPOPT_TSYM_DUMP_CALL=<call>` when mapping IPOPT
  TSym call numbers back to `PDFullSpaceSolver::SolveOnce` calls.
- Set `GLIDER_PARITY_IPOPT_SOLVEONCE_RHS_DUMP_DIR=/tmp/...` to dump
  `IpPDFullSpaceSolver.cpp::SolveOnce` augmented RHS construction inputs:
  original RHS blocks, x/slack distances, `aug_rhs_x`, `aug_rhs_s`, `alpha`,
  and `beta`. Use `GLIDER_PARITY_IPOPT_SOLVEONCE_RHS_DUMP_CALL=<call>` for a
  single call. These dumps were used to prove that the remaining post-KKT
  drift was not in the regular SolveOnce RHS before the SOC branch.
- Probe 0 (`GLIDER_PARITY_NLIP_AUGMENTED_ITER=0`,
  `GLIDER_PARITY_PROBE_INDEX=0`, max two iterations in both solvers) is now
  bitwise closed at the source-SPRAL boundary: IPOPT's best call-1 matrix,
  RHS, and unrefined solution all match NLIP with `max_abs_diff=0.0`, and
  initial component snapshots report zero-diff for `grad_f`, `jac_cT*y_c`,
  `jac_dT*y_d`, `curr_grad_lag_x`, KKT x/slack stationarity, slack
  components, and nonfixed variable-bound multipliers. The earlier
  `1.06581410364015028e-13` RHS gap at `x[1153->tf]` was isolated with
  `GLIDER_PARITY_IPOPT_JAC_DUMP_DIR`: IPOPT's effective `jac_c` triplets and
  values matched NLIP exactly, while `IpGenTMatrix::TransMultVectorImpl`
  matched `fma(value, y, acc)`, not separate multiply-then-add arithmetic.
- Probe 2 (`GLIDER_PARITY_NLIP_AUGMENTED_ITER=2`,
  `GLIDER_PARITY_PROBE_INDEX=2`) matches IPOPT's captured KKT structure with
  max absolute matrix difference about `6.7e-10`, prefinal RHS about `7.2e-12`,
  and cumulative refined solution about `3.5e-12`. The larger unrefined
  equality-multiplier block difference is removed by IPOPT-style refinement.
  The visible equality-multiplier gap is on `continuity_u[*].alpha`, whose
  output factor is about `1.146e3`; the implied internal delta is only about
  `1e-13` to `2e-13`.
- The source-built SPRAL boundary replay for probe 2 localizes that
  `6.7e-10` matrix gap to the `p[0],p[0]` upper-slack sigma diagonal. The
  corresponding prefinal RHS gap is about `7.2e-12`, and the best unrefined
  SPRAL `after_solve` vector differs by about `3.2e-8` in the equality
  multiplier block. Accepted direction, alpha, KKT residual, and barrier
  scalar ladders are still below the active thresholds at this probe, so this
  is a full-space solve/sigma sensitivity marker rather than an
  `AcceptTrialPoint`, filter, watchdog, SOC, or restoration branch marker.
- Probe 16 after mirroring `IpExpansionMatrix.cpp::SinvBlrmZMTdBrImpl` FMA
  order moves the focused glider marker materially later: first x-direction
  gap above `1e-8` is accepted index 18 (`1.698e-8`), with no x-direction gap
  above `1e-6`; no `alpha_pr` or `alpha_du` gap exceeds `1e-8` through the
  25-step probe. The accepted x gap at index 16 is about `4.335e-10`.
- The `GLIDER_PARITY_IPOPT_EXPANSION_DUMP_DIR` v5 probe over 18 accepted NLIP
  steps produced 144 `SinvBlrmZMTdBrImpl` dumps. Twenty-four calls had real
  rows where volatile separate multiply-add differed from FMA; all 24 calls
  classified as `x=fma` (`x_matches_fma == ncols`, with separate mismatches).
  The synthetic plus/minus source-expression bits also matched FMA and not the
  volatile separate result, confirming this source-built IPOPT file is compiled
  with contraction for the active expression shape.
- The `GLIDER_PARITY_IPOPT_DENSE_ADD_DUMP_DIR` v8 probe over 18 accepted NLIP
  steps produced 74 `DenseVector::AddTwoVectorsImpl` general current-plus-step
  dumps. Across 69,094 result entries, 8,682 rows distinguished volatile
  separate multiply/add from FMA and every result matched the FMA candidate.
  The synthetic source-like expression also matched FMA in all 74 calls. This
  source-backs NLIP's `IpIpoptData.cpp::SetTrial*FromStep`
  `alpha.mul_add(delta, value)` mirror and rules dense trial-state application
  out as the source of the index-16/17 upper-slack amplification.
- The IPOPT-order Hessian residual replay now mirrors
  `IpSymTMatrix.cpp::MultVectorImpl` for
  `W.MultVector(1., res.x, 0., resid.x)` inside
  `IpPDFullSpaceSolver.cpp::ComputeResiduals`: lower-triangle triplets are
  traversed in storage order and each dense output update uses the
  source-built product/add contraction. The focused iter-0 augmented replay now
  has bitwise exact correction RHS, correction solve, and accumulated refined
  solution against IPOPT (`max_abs_diff=0.0`), closing the prior `1.421e-14`
  first marker. In the 18-step glider probe this reduces the index-17
  `y_d`/`v_U` jump from about `2.18e0` to about `1.7e-5`, with no x gap above
  `1e-8`.
- Probe 16/17 KKT replays now show the remaining slice is sigma/upper-slack
  sensitivity at the solve boundary, not globalization. At iter 16 the largest
  matrix gap is on a huge p-block sigma diagonal (`2.068e3` on about
  `3.210e15`) and the best cumulative refined solution gap is about
  `3.249e-5`. At iter 17 the same channel amplifies to a p-block sigma gap of
  about `7.845e4` and an upper-slack multiplier jump; the state feeding iter 17
  already carries the small `v_U`/`y_d` drift from iter 16.
- Earlier-window probe 8 keeps the root below globalization: accepted indices
  4 through 10 show x/slack state deltas only around `1e-13` to `1e-12`, while
  equality-multiplier deltas are externally amplified by the OCP output scale
  (`1.146e3`) and upper-slack sigma grows from tiny state/residual differences.
  Alpha values still match to about `1e-15` through this window.
- Tight probe 0 (`GLIDER_PARITY_DIRECTION_THRESHOLD=1e-14`, max 8 iterations)
  confirms initialization is still exact and the first measurable
  accepted-direction gap above that threshold is accepted index 1 at about
  `7.7e-13`, with `alpha_pr` and `alpha_du` matching to machine precision.
  The corresponding internal probes show KKT x/slack stationarity differences
  in the `1e-13` to `1e-12` range before later sigma amplification.
- Focused `GLIDER_PARITY_IPOPT_PRINT_LEVEL=12` probes show no IPOPT messages
  for safe-slack adjustment, oversized slacks, or bound-multiplier correction,
  and alpha limiters remain the boundary-speed upper-slack rows. Those branches
  remain diagnostics unless a reduced witness moves the first mismatch there.
- Historical pre-restoration-checkpoint note: after the residual
  `AddTwoVectors`, BLAS `AxpyImpl`, bound-residual, and SOC `AddOneVector`
  ordering fixes, the strict-profile glider matched IPOPT through the full
  accepted trace:
  `nlip_steps=152`, `ipopt_steps=152`, no direction gaps above `1e-8`,
  no `alpha_pr` / `alpha_du` / `alpha_y` gaps above `1e-16`, and no accepted
  direction gaps above `1e-8` in x, y_c, y_d, or v_U. Accepted state gaps are
  absent above `1e-10` for x, equality multipliers, inequality multipliers,
  slack stationarity, x stationarity, barrier primal/dual/complementarity,
  objective gradient, Jacobian-transpose products, Lagrangian gradients, and
  damping vectors. The only tracked accepted-state scalar above `1e-10` is
  barrier subproblem error at index 18 (`3.201e-10`), with no gap above
  `1e-8`.
- The focused SOC replay around accepted index 49/50 confirms the remaining
  historical drift was in `IpFilterLSAcceptor.cpp::TrySecondOrderCorrection`:
  IPOPT and NLIP SPRAL matrices, prefinal RHS, pre-refinement solutions,
  refinement RHS/solutions, and accumulated refined solutions are bitwise
  exact at iterations 49 and 50 after mirroring
  `c_soc->AddOneVector(1.0, trial_c, alpha_primal_soc)` and the corresponding
  `dms_soc` operation.
- `print_current_glider_native_spral_ipopt_first_divergence` is now a
  fail-closed ignored regression helper: after printing its diagnostic window,
  it panics on default-threshold direction divergence, accepted-trace
  divergence, or accepted-trace length mismatch.
- IPOPT parity dump directory and call-selector environment variables are
  runtime diagnostics read by the injected C++ code with `std::getenv`; they
  are intentionally not Cargo build-script rerun triggers. Changing dump
  destinations should not force a source-built IPOPT rebuild.
- The `ipopt-src` source patcher must fail loudly when an upstream source
  anchor moves. First-time parity patch paths use required source-anchor
  replacements for `IpSpralSolverInterface.cpp`, `IpExpansionMatrix.cpp`,
  `IpDenseVector.cpp`, `IpTSymLinearSolver.cpp`, and
  `IpPDFullSpaceSolver.cpp`; a missing anchor is a build-script failure, not a
  silent partial diagnostic install.

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
| Matching ordering/scaling observation checkpoint | `src/ssids/ssids.f90`, `src/match_order.f90`, `src/scaling.f90`, `interfaces/C/ssids.f90`, and IPOPT `IpSpralSolverInterface.cpp` | Observational only: `ssids-rs` captures native matching order and can replay that explicit order without scaling. Full Rust matching/scaling is not implemented yet, and analyse-time `options%scaling=3` remains opaque unless SPRAL exposes it through an official interface. |
| Native SPRAL C option bridge, including `pivot_method` | `interfaces/C/ssids.f90`, `src/ssids/datatypes.f90`, `src/ssids/cpu/cpu_iface.f90` | None. The C `spral_ssids_options%pivot_method` is copied into Fortran `ssids_options%pivot_method`, whose APP aggressive/block/TPP constants are `1/2/3`. |
| APP/TPP pivot choice, delayed pivots, failed pivots | SPRAL SSIDS numeric factor source | None. Preserve the ladder: factorization outcome, inertia, pivot stats, solve outcome, solution bits. |
| APP `block_ldlt<32>` 2x2 multiplier codegen split | `src/ssids/cpu/kernels/block_ldlt.hxx::block_ldlt`, reached from `src/ssids/cpu/kernels/ldlt_app.cxx::Block::factor` | Mirrors the observed optimized native build for bitwise witnesses: vector-body rows use the second-product contraction, one-row scalar remainders use the first-product contraction, and the three-row short tail keeps its first two rows in the vector-body contraction lane. This is not a fallback or tolerance rule. |
| APP accepted-prefix update `calcLD<OP_N>` tiling | `src/ssids/cpu/kernels/ldlt_app.cxx::Block::update`, `src/ssids/cpu/kernels/calc_ld.hxx::calcLD<OP_N>` | None. The `LD` workspace must be shaped per target block, so the observed vector/scalar expression split resets at each APP target row tile instead of once across the whole trailing tail. |
| Dot/update/FMA order, signed zero, solve order | SPRAL SSIDS solve and update source | None for bitwise witnesses. |
| Production forward solve traversal | `src/ssids/cpu/NumericSubtree.hxx::solve_fwd`, `src/ssids/cpu/kernels/ldlt_app.cxx::ldlt_app_solve_fwd` | None. Rust production solve must gather each front, apply the APP forward solve kernel, then scatter the full front-local RHS; global sparse forward substitutes are diagnostic only. |
| Production diagonal/backward solve traversal | `src/ssids/fkeep.F90::inner_solve_cpu`, `src/ssids/cpu/NumericSubtree.hxx::solve_diag_bwd_inner<true,true>`, `src/ssids/cpu/kernels/ldlt_app.cxx::ldlt_app_solve_diag` / `ldlt_app_solve_bwd` | None. Full Rust production solves must use the native combined `diag_bwd` staging: gather a front-local RHS, apply front-local inverse-D blocks, apply the APP backward solve, then scatter eliminated rows. |
| Restoration reduced-KKT replay failure classification | IPOPT `IpPDFullSpaceSolver.cpp` and `IpSpralSolverInterface.cpp`, plus SPRAL SSIDS reported inertia | None. Replay of a dumped restoration KKT must stop at the same wrong-inertia decision as production before solving through a rejected factorization; solve residuals are compared only after the reported inertia matches the expected augmented-system inertia. |
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
