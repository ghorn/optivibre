# IPOPT Parity Coverage Audit

This document is the nonlinear-only coverage ledger for NLIP vs source-built
IPOPT parity. It complements `docs/native-spral-parity-manifest.md` by making
branch coverage explicit instead of relying only on final solve parity.

## How to Run

```sh
OMP_CANCELLATION=true \
RAYON_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
scripts/ipopt_parity_coverage.sh
```

The script enables `IPOPT_SRC_LLVM_COVERAGE=1`, which makes `ipopt-src` build
IPOPT into a separate `install-llvm-cov` directory with LLVM source-coverage
flags. Normal source-built IPOPT continues to use the existing `install`
directory and marker. `spral-src` remains the normal source-built baseline; set
`IPOPT_PARITY_COVERAGE_REBUILD_NATIVE=1` only when you intentionally want to
clean and rebuild `spral-src` and `ipopt-src` before collecting coverage.
The script sets `GLIDER_PARITY_IPOPT_PRINT_LEVEL=0` by default to keep coverage
logs focused; override it when the IPOPT iteration table is part of the
diagnostic.
For focused tests that need IPOPT step tags without console noise,
`IpoptOptions::journal_print_level` captures a high-detail native journal while
leaving `print_level=0`.

On macOS with Homebrew LLVM and GCC, the script discovers `clang`, `clang++`,
`llvm-cov`, `llvm-profdata`, the GCC `libstdc++` include path, and the GCC
runtime library path. Override them with `LLVM_COV`, `LLVM_PROFDATA`,
`IPOPT_SRC_LLVM_COVERAGE_CC`, `IPOPT_SRC_LLVM_COVERAGE_CXX`,
`IPOPT_SRC_LLVM_COVERAGE_CXXFLAGS`, or `IPOPT_SRC_LLVM_COVERAGE_LDFLAGS` when
diagnosing a different toolchain.

Reports are written to `target/reports/ipopt-parity-coverage/`:

| Report | Purpose |
| --- | --- |
| `coverage-summary.json` | Raw `cargo llvm-cov` summary for the focused parity run |
| `combined.lcov` / `combined.txt` | `cargo llvm-cov` Rust/FFI report for the focused parity run |
| `ipopt-native.lcov` | Direct `llvm-cov export` over the instrumented IPOPT archives and parity test binaries |
| `ipopt-algorithm.lcov` | Filtered IPOPT `src/Algorithm` coverage |
| `rust-core.lcov` | Filtered Rust NLIP core coverage for `interior_point.rs` and `filter.rs` |
| `coverage-audit.md` | Small machine-generated line and branch hit summary |
| `branch-ledger.md` | Watched IPOPT branch surfaces plus Rust core unhit branch-like lines |

Latest generated profile snapshot, from the source-built nonlinear parity run:

- IPOPT Algorithm C++: `11891 / 18642` lines and `1594 / 3556` branches hit.
- Rust NLIP core: `13346 / 20823` lines hit, with `0` unhit branch-like lines
  still classified as `needs audit`.
- `IpDefaultIterateInitializer.cpp`: `498 / 532` lines and `58 / 70` branches
  hit after adding the requested `least_square_init_primal=yes` witness.
- `IpIpoptAlg.cpp` convergence-status cases for max-iteration, diverging
  iterates, CPU time, wall time, acceptable termination, and user-requested
  stop are now covered by focused witnesses.
- Watched IPOPT Algorithm surfaces now have `0` active-watch uncovered branch
  lines after classification; remaining uncovered rows are explicit
  witness gaps, unreachable option-profile paths, diagnostics/bookkeeping, or
  option/error handling.
- `IpBacktrackingLineSearch.cpp`: `824 / 993` lines and `200 / 252` branches
  hit, with all `42` uncovered branch lines classified.
- `IpIpoptCalculatedQuantities.cpp`: `2540 / 2885` lines and `312 / 424`
  branches hit, with all `91` uncovered branch lines classified.

The script exits nonzero if any unhit Rust core branch-like line remains
classified as `needs audit`; those branches must either get a focused witness or
an explicit unreachable/diagnostic classification before the coverage run is
accepted.

The latest focused run added source-option witnesses for:

- `IpDefaultIterateInitializer.cpp::bound_mult_init_method=mu-based`
- `IpDefaultIterateInitializer.cpp::bound_push`, `bound_frac`,
  `slack_bound_push`, `slack_bound_frac`, and `bound_mult_init_val`
- `IpDefaultIterateInitializer.cpp::constr_mult_init_max=0`, which forces
  IPOPT's y-multiplier initialization to zero instead of reusing bound
  multipliers
- `IpDefaultIterateInitializer.cpp::yinitnrm > constr_mult_init_max` with a
  positive cap that rejects the least-square multiplier estimate
- `IpDefaultIterateInitializer.cpp::least_square_mults` square-problem branch,
  which initializes `y_c` / `y_d` to zero when the equality dimension matches
  IPOPT's current variable dimension
- `IpDefaultIterateInitializer.cpp::least_square_init_primal=yes`, which enters
  `CalculateLeastSquarePrimals`, installs the least-square `x`, then mirrors
  IPOPT's subsequent `trial_d()` slack recomputation after bound push
- `IpDefaultIterateInitializer.cpp::least_square_init_duals=yes`, which enters
  `CalculateLeastSquareDuals` and initializes `y_c`, `y_d`, `z_L`, `z_U`, and
  upper-slack multipliers from IPOPT's zero-Hessian augmented system
- IPOPT fixed-variable treatment through a fixed-bound quadratic witness,
  compared against NLIP's fixed-variable elimination path
- `IpIpoptAlg.cpp::correct_bound_multiplier` through a tight
  `kappa_sigma=1` accepted-trial safeguard witness
- `IpIpoptCalculatedQuantities.cpp::kappa_d=0` plus
  `IpOrigIpoptNLP.cpp::bound_relax_factor=0` on a bound-constrained problem
- `IpPDPerturbationHandler.cpp::perturb_always_cd=yes`, where NLIP now applies
  IPOPT's permanent `delta_c` / `delta_d` constraint-linearization perturbation
  from the first augmented-system solve instead of waiting for a singularity
  trigger
- `IpBacktrackingLineSearch.cpp::alpha_red_factor` through NLIP's
  `line_search_beta` option
- `IpBacktrackingLineSearch.cpp::Eval_Error` trial backtracking: the IPOPT
  adapter now returns callback failure for nonfinite trial values, and NLIP
  rejects the trial, skips SOC, cuts alpha, and records the missing trial
  metrics exactly like IPOPT's evaluation-error line-search branch
- `IpBacktrackingLineSearch.cpp::accept_every_trial_step=yes` and
  `accept_after_max_steps=1`: NLIP now evaluates the trial first, then bypasses
  normal filter acceptability while still applying
  `FilterLSAcceptor::UpdateForNextIteration` f/h tagging and filter
  augmentation semantics. The `accept_after_max_steps` witness covers IPOPT's
  `eMaxS` path after an evaluation-error cutback
- `IpMonotoneMuUpdate.cpp::mu_allow_fast_monotone_decrease=no`
- `IpMonotoneMuUpdate.cpp::mu_target>0`
- `IpFilterLSAcceptor.cpp::max_soc=0`
- `IpFilterLSAcceptor.cpp::soc_method=1`, where SOC scales IPOPT's x/s
  residual rows by the current SOC alpha while leaving the constraint and
  complementarity rows on the default path
- `IpBacktrackingLineSearch.cpp::watchdog_shortened_iter_trigger=0`
- `IpBacktrackingLineSearch.cpp::StartWatchDog` through a hanging-chain
  profile with `watchdog_shortened_iter_trigger=3`, retained only after NLIP
  and IPOPT kept matching accepted-step counts and trace parity. The same
  profile also covers IPOPT's successful-watchdog `Append_info_string("W")`
  path; NLIP now arms the watchdog after the current direction is computed,
  so the stored filter reference uses the same grad-barrier dot direction as
  `BacktrackingLineSearch::StartWatchDog`. In active watchdog mode, NLIP keeps
  `alpha_min` equal to the max feasible primal step like
  `DoBacktrackingLineSearch`, so the watchdog line search evaluates one max-step
  trial before either succeeding or entering the lowercase `w` branch. A focused
  CasADi Rosenbrock profile now covers the non-success watchdog trial: NLIP
  accepts it with step tag `w`, leaves the watchdog armed, and skips SOC while
  `in_watchdog_` is active, matching IPOPT's backtracking loop order. It also
  marks filter trial acceptance as a watchdog success when the outer line-search
  state is already in watchdog mode and clears watchdog state immediately,
  matching `BacktrackingLineSearch::FindAcceptableTrialPoint`. The same focused
  profile now covers `StopWatchDog` after active-watchdog trial-budget
  exhaustion: NLIP restores the stored iterate and stored direction, clears
  active watchdog state, skips the already-tested first trial, and retries from
  the restored line-search reference.
  (`trigger=1` and `trigger=2` were rejected as witnesses because they exposed
  accepted-step count drift despite close final values)
- `IpBacktrackingLineSearch.cpp::TrySoftRestoStep`, `max_soft_resto_iters`,
  and `soft_resto_pderror_reduction_factor`: a focused CasADi Rosenbrock
  profile forces the regular line search to fail, starts IPOPT's soft
  restoration phase, accepts primal-dual-error reduction steps with the
  lowercase `s` tag, leaves watchdog shortened-step counters untouched while
  soft restoration remains active, and keeps NLIP's final state within the same
  source-built IPOPT tolerances. A bounded quiet-journal sweep over CasADi
  Rosenbrock, equality Rosenbrock, and HS071 profiles found no natural
  uppercase `S` original-criterion witness; the branch remains implemented but
  unpromoted until a deterministic source-built witness appears.
- `IpBacktrackingLineSearch.cpp::tiny_step_tol=0`
- `IpBacktrackingLineSearch.cpp::alpha_for_y` values `bound-mult`, `min`,
  `max`, `full`, `min-dual-infeas`, `safer-min-dual-infeas`,
  `primal-and-full`, and `dual-and-full`
- `IpBacktrackingLineSearch.cpp::alpha_for_y_tol` through strict
  `primal-and-full` / `dual-and-full` option profiles
- `IpIpoptAlg.cpp::ConvergenceCheck::MAXITER_EXCEEDED` via `max_iter=0`
  and via `max_iter=1` after an accepted trial
- `IpOptErrorConvCheck.cpp::DIVERGING` /
  `IpIpoptAlg.cpp::DIVERGING_ITERATES` via a low
  `diverging_iterates_tol` threshold that trips before the first search
  direction, with both solvers retaining iteration-0 partial state
- `IpOptErrorConvCheck.cpp::WALLTIME_EXCEEDED` /
  `IpIpoptAlg.cpp::Maximum_WallTime_Exceeded` via a tiny positive
  `max_wall_time` threshold that trips after initialization, with the local
  IPOPT wrapper exposing the source status through its C bridge
- `IpOptErrorConvCheck.cpp::CPUTIME_EXCEEDED` /
  `IpIpoptAlg.cpp::Maximum_CpuTime_Exceeded` via a tiny positive
  `max_cpu_time` threshold that trips after initialization
- `IpOptErrorConvCheck.cpp::USER_STOP` /
  `IpIpoptAlg.cpp::USER_REQUESTED_STOP` via the intermediate-callback return
  path. The local IPOPT adapter now propagates false callback returns, and NLIP
  exposes a matching `solve_nlp_interior_point_with_control_callback` API plus
  `InteriorPointSolveError::UserRequestedStop`; this is pinned for both regular
  iterations and `IpRestoConvCheck.cpp` restoration-phase callback stops
- `IpOptErrorConvCheck.cpp::CONVERGED_TO_ACCEPTABLE_POINT` via tight strict
  tolerances, loose acceptable tolerances, and `acceptable_iter=1`
- `IpBacktrackingLineSearch.cpp::PerformMagicStep` via `magic_steps=yes` on the
  linearly constrained quadratic witness. NLIP now exposes the option and
  applies IPOPT's upper-slack projection slice before trial acceptability.
- `IpAdaptiveMuUpdate.cpp::adaptive_mu_restore_previous_iterate=yes` plus
  non-default `adaptive_mu_kkterror_red_iters` /
  `adaptive_mu_kkterror_red_fact`: NLIP now snapshots the accepted free-mu
  iterate, restores it on free-to-fixed mode switches, computes the first fixed
  barrier parameter from the restored complementarity and safeguard terms, and
  restarts the same outer iteration so state, residuals, Hessian, and KKT RHS
  are recomputed from the restored point before the solve.
- `IpAlgBuilder.cpp` restoration-phase `resto.mu_strategy=adaptive` handoff:
  NLIP now exposes restoration-prefixed mu strategy/oracle/globalization/minimum
  options, applies IPOPT's more conservative default restoration `mu_min`, and
  compares the explicit `resto.mu_strategy=adaptive`, `resto.mu_oracle=loqo`,
  `resto.adaptive_mu_globalization=never-monotone-mode` square-equality
  restoration witness against source-built IPOPT.
- `IpQualityFunctionMuOracle.cpp` `HaveDeltas` handoff:
  the quality-function oracle now stores the combined affine-plus-centering
  direction, `PDSearchDirCalc` seeds the main search-direction solve from it,
  and the seeded direction goes through IPOPT's full-space residual/refinement
  loop before accept.  The golden-section search is pinned to IPOPT 3.14's
  identity `ScaleSigma` / `UnscaleSigma` behavior.
- `IpPDSearchDirCalc.cpp::mehrotra_algorithm`:
  the non-default probing/Mehrotra branch now uses the affine direction to build
  the source complementarity RHS for variable and slack bounds, and the explicit
  bound-RHS residual replay path is covered by the focused source-built witness.
- `IpBacktrackingLineSearch.cpp::DetectTinyStep` via intentionally large
  `tiny_step_tol` and `tiny_step_y_tol`
- `IpRestoIpoptNLP.cpp` / `IpRestoIterateInitializer.cpp` restoration NLP
  shape for equality and inequality residual blocks, plus
  `IpRestoMinC_1Nrm.cpp::ComputeBoundMultiplierStep` for copied restoration
  slack and bound/slack multiplier reset semantics
- `IpRestoIpoptNLP.cpp::f` original-objective evaluation at restoration trial
  points: NLIP now probes the original objective for nonfinite values while
  still returning the restoration merit value, matching IPOPT's
  `evaluate_orig_obj_at_resto_trial=yes` default
- `IpRestoConvCheck.cpp::CheckConvergence` original-problem return gate:
  NLIP now applies IPOPT's first-restoration-iteration continue rule, exact
  required-infeasibility-reduction threshold, square-problem feasibility
  shortcut, original filter/current-iterate acceptability test, and active
  `max_iter` budget instead of the old Rust-only restoration clamp
- `IpRestoConvCheck.cpp::max_resto_iter`, including IPOPT's
  `outer_iter_count + 1` restoration iteration numbering and
  `MaximumIterationsExceeded` status when successive restoration iterations
  exceed the configured limit
- Restoration original-objective evaluation-error max-iteration status:
  a focused synthetic restoration problem now forces nonfinite original
  objective values at restoration trial points while SOC and soft restoration
  are disabled. NLIP keeps accepting restoration tiny steps until
  `InteriorPointSolveError::MaxIterations`, matching IPOPT's
  `MaximumIterationsExceeded` status and iteration count.
- `IpRestoMinC_1Nrm.cpp::PerformRestoration` restoration solver-return
  classification: NLIP now carries IPOPT's square-problem
  `FEASIBILITY_PROBLEM_SOLVED` path as `InteriorPointTermination::FeasiblePointFound`
  maps `RESTORATION_USER_STOP` to the outer `UserRequestedStop` status like
  `IpIpoptAlg.cpp`, and pins the source status mapping in focused unit coverage
- `IpIpoptAlg.cpp` emergency-mode restoration entry after failed
  `ComputeSearchDirection`: a deterministic impossible square-equality witness
  now enters NLIP restoration after the singular KKT/inertia failure and returns
  `InteriorPointSolveError::LocalInfeasibility`, matching IPOPT's
  `InfeasibleProblemDetected` status
- `IpBacktrackingLineSearch.cpp::start_with_resto`: a deterministic impossible
  square-equality witness now skips the first regular line search, enters
  restoration without emergency failed-linear-solve context, and matches IPOPT's
  `InfeasibleProblemDetected` status when `start_with_resto=yes`
- `IpBacktrackingLineSearch.cpp::expect_infeasible_problem`: a square-equality
  least-square-dual witness now exceeds `expect_infeasible_problem_ytol`, skips
  regular line-search trials, enters restoration, and solves to the same point
  as source IPOPT, including `IpIpoptAlg.cpp::ComputeFeasibilityMultipliersPostprocess`
  and `SolveSucceeded` classification when the postprocessed square point also
  satisfies the full KKT convergence test
- `IpPDFullSpaceSolver.cpp::RegisterOptions` / `InitializeImpl` and
  `Solve::ComputeResidualRatio`: NLIP now exposes the source refinement options
  `min_refinement_steps`, `max_refinement_steps`, `residual_ratio_max`,
  `residual_ratio_singular`, `residual_improvement_factor`,
  `neg_curv_test_tol`, and `neg_curv_test_reg`, validates the IPOPT order checks,
  and consumes those values in regular and restoration `SpralSrc` full-space
  refinement. The nonzero negative-curvature tolerance path now mirrors
  `SolveOnce`: solve through a wrong-inertia factorization, compute the
  Zavala-Chiang curvature heuristic, and retry with a modified matrix only when
  the direction is not sufficiently positive. Focused non-default witnesses
  `compare_native_and_ipopt_with_full_space_refinement_options` and
  `compare_native_and_ipopt_with_negative_curvature_test_options` keep NLIP and
  source IPOPT at seven iterations on the linearly constrained quadratic problem.
- `IpPDFullSpaceSolver.cpp::Solve` failed-refinement markers: the Rust
  failure table now also pins IPOPT's info-marker semantics for the failed
  refinement branch: `q` for a successful `IncreaseQuality()` retry, lowercase
  `s` for the pretend-singular retry, uppercase `S` for the
  `residual_ratio_singular_` accept-current path, and no marker for the final
  accept-after-pretend retry.
- `IpPDPerturbationHandler.cpp::get_deltas_for_wrong_inertia`: the Rust
  hessian-perturbation growth helper now preserves IPOPT's max-perturbation
  failure semantics. In particular, a `first_hessian_perturbation` larger than
  `max_hessian_perturbation` fails instead of widening the max bound.

These witnesses are wired through `IpoptOptions::raw_options` so normal IPOPT
solve behavior is unchanged unless a test or diagnostic explicitly requests a
source option.

## Current Known Differences

This is the hand-maintained backlog for nonlinear IPOPT parity. It is fed by
the generated `branch-ledger.md` active/watch rows, source inspection, and
NLIP/IPOPT comparison traces. Every active/watch branch family must either
appear here, be explicitly unreachable under the current exact-Hessian parity
options, or be classified as diagnostic/bookkeeping/error handling in the
generated report.

| Priority | IPOPT source anchor | Rust parity surface | Default-lane impact | Current evidence | Required witness or test |
| --- | --- | --- | --- | --- | --- |
| P0 | `IpAugRestoSystemSolver.cpp` restoration KKT reduction | Restoration linear solve inside `optimization/src/interior_point.rs` | Closed hard default-lane gap; current glider trace has no tracked direction, alpha, accepted-direction, or accepted-state drift | The current glider first-divergence diagnostic reports `nlip_steps=152 ipopt_steps=152`, no direction gaps above `1e-8`, no `alpha_pr` / `alpha_du` / `alpha_y` gaps above `1e-16`, no accepted-direction gaps above `1e-8`, and no accepted-state gaps above `1e-10` except barrier subproblem error at index 18 (`3.201e-10`, with no gap above `1e-8`). Mixed/inequality and equality-only restoration both use the pure-Rust reduced restoration KKT path under `SpralSrc`, and prior reduced/regular solve-boundary dumps remain the restoration regression evidence. | Keep the glider first-divergence diagnostic and reduced restoration KKT tests as regressions. Reopen this item only if a fresh witness shows source-order mismatch in reduced RHS, refinement, copy-back, or post-restoration regular KKT assembly. |
| P1 | IPOPT source defaults plus OCP/studio option wiring | `IpoptOptions::default`, OCP `default_nlip_config`, optimization studio settings | Config/profile gap, not an algorithmic parity gap | `IpoptOptions::default` mirrors IPOPT source termination defaults, but OCP/studio still use a stricter profile. Source-default glider now has a passing NLIP/IPOPT accepted-trace witness. | Decide whether the OCP/studio experience should move from the strict profile to the IPOPT-source-default profile, and pin that choice with explicit option-summary tests. |
| P2 | `IpAdaptiveMuUpdate.cpp`, `IpAlgBuilder.cpp::BuildMuUpdate`, `IpQualityFunctionMuOracle.cpp`, `IpProbingMuOracle.cpp` | Partial Rust adaptive/free-mu implementation | Mostly covered non-default exact-Hessian feature lane | The `mu_strategy=adaptive` lane now covers `loqo` with `never-monotone-mode`, `loqo` with IPOPT's objective/constraint filter globalization, default and non-default `quality-function` with objective/constraint filter, `probing` with objective/constraint filter, default `kkt-error` globalization, nonzero `adaptive_mu_safeguard_factor`, non-default KKT-error reduction knobs, `adaptive_mu_restore_previous_iterate`, explicit restoration-phase adaptive handoff through `resto.mu_strategy=adaptive`, quality-function `HaveDeltas` seeded search-direction refinement, and the IPOPT identity sigma-space golden-section search. Source-built IPOPT comparisons pass for the linearly constrained quadratic witnesses, the variable-bound `bound_constrained_quadratic_adaptive_quality_function` witness, non-default quality-function norm witnesses (`1-norm`, `2-norm`, `max-norm`), the non-default quality-function centrality/balancing/search witness, and the square-equality adaptive-restoration witness. This execution re-ran the focused source-built adaptive lane (`13 passed`) and the helper-level adaptive unit lane (`12 passed`). Helper tests pin adaptive filter coordinate acceptability/frontier update, KKT-error reference-window semantics, fixed/free-mode switch, restore-previous-iterate fixed-mode state selection, restoration-prefixed option overrides/default `mu_min`, the safeguard's initial normalized infeasibility ratio formula, non-default quality-function KKT error terms, and identity sigma-space section search. No focused adaptive compare currently exposes a natural accepted-trace fixed-mode switch beyond the existing synthetic/reduced witnesses. | Keep the synthetic fixed/free-mode tests as branch coverage, and add a natural accepted-trace fixed-mode switch witness only if a future sweep or explicit adaptive-mode trace instrumentation finds one. |
| P3 | `IpFilterLSAcceptor.cpp::TryCorrector` | Rust primal-dual/affine corrector branch in `optimization/src/interior_point.rs` | Covered non-default exact-Hessian feature lane; default profile still keeps `corrector_type=none` | NLIP now mirrors the `TryCorrector` pre-backtracking slot, skip gates, affine/primal-dual complementarity RHS construction, `allow_inexact=true` correction solve, original-alpha filter acceptability test, affine complementarity-reduction rejection, and accepted corrector trial mutation. Focused source-built witnesses `compare_native_and_ipopt_with_primal_dual_corrector` and `compare_native_and_ipopt_with_affine_corrector` pass against IPOPT/SPRAL-source on the linearly constrained quadratic problem. | Broaden only if a natural model uses `corrector_type` and exposes accepted-trace drift; default-lane generated coverage may continue to mark corrector rows unreachable because source defaults set `corrector_type=none`. |
| P4 | `IpPDPerturbationHandler.cpp` degeneracy and inertia retry branches | Regularization and inertia retry loop | Mostly covered exact-Hessian feature lane | Current glider lane aligns regularization and inertia. Non-default `perturb_always_cd=yes` is implemented and pinned against IPOPT. The hessian-perturbation growth helper now mirrors the source's first-trial, previous-success decay, first-growth, normal-growth, and max-perturbation failure checks; Rust no longer widens `max_hessian_perturbation` to fit `first_hessian_perturbation`. The singular-system degeneracy retry state machine now mirrors `PerturbForSingularity`: constraint-only, Hessian-only, combined constraint/Hessian, and always-`cd` transitions are pinned by unit witnesses. The wrong-inertia max-Hessian fallback now mirrors `PerturbForWrongInertia` by resetting Hessian perturbation and retrying with `delta_c/delta_d` when constraints are present. Generated active/watch rows for this family are now classified: retry transitions map to reduced state-machine witnesses, hessian growth and wrong-inertia fallback map to focused helpers, and remaining uncovered source rows are structural degeneracy finalization/status markers (`Nh`, `Nj`, `Dh`, `Dj`, `Dhj`) rather than observed accepted-trace drift. | Add a natural or reduced witness only if it forces multi-system structural-degeneracy finalization and affects KKT/RHS/direction; otherwise keep the state-machine unit witnesses as the parity guard. |
| P5 | `IpPDFullSpaceSolver.cpp::Solve`, `SolveOnce`, `ComputeResidualRatio` | Full-space residual replay, refinement, quality retry | Coverage witness backlog; no current accepted-trace drift | Regular `SpralSrc` lane is source-backed, the full-space refinement option surface now mirrors IPOPT defaults and non-default settings for `min_refinement_steps`, `max_refinement_steps`, `residual_ratio_max`, `residual_ratio_singular`, `residual_improvement_factor`, `neg_curv_test_tol`, and `neg_curv_test_reg`. The nonzero negative-curvature tolerance branch now mirrors IPOPT by disabling pre-solve inertia rejection, solving through the factorization, testing `x'*(W+Sigma_x+delta_x I)*x + s'*(Sigma_s+delta_s I)*s`, and retrying wrong-inertia perturbation only on heuristic failure. The failed-refinement branch table mirrors IPOPT's order: one `IncreaseQuality()` retry, strict `< residual_ratio_singular` accept-current versus pretend-singular split, and accept-current after one pretend-singular retry. Unit tests pin that decision order, IPOPT's info-marker semantics (`q`, `s`, `S`, then no marker after a pretend-singular retry), the zero-rhs/zero-solution `ComputeResidualRatio` edge, and the negative-curvature formula. Source-built full-space refinement and negative-curvature option witnesses keep NLIP/IPOPT at seven iterations. `TryCorrector` covers the `beta != 0` / `Solve(1,1,...)` final assembly shape for affine/primal-dual corrector profiles, quality-function adaptive mode covers the `improve_solution=true` seeded residual/refinement path, and P4 witnesses cover the perturbation retry branches reached from `SolveOnce`. | Remaining P5 work is live source-built witnesses for failed solve, quality retry, pretend-singular retry, and accept-current-after-pretend traces if they can be triggered without editing linear internals. |
| P6 | `IpBacktrackingLineSearch.cpp` globalization branches | Line search, watchdog, soft restoration, tiny step, alpha-for-y | Coverage witness backlog | Many default and option branches are implemented and tested, including `magic_steps=yes`, the `min-dual-infeas` / `safer-min-dual-infeas` `PerformDualStep` projection formulas, watchdog arming/success/lowercase-trial/restore-retry, tiny-step acceptance, and lowercase `s` soft restoration. The 2026-05-02 bounded watchdog sweep found many ordinary tiny-step and watchdog cases but no active-watchdog pre-line tiny-stop termination. The bounded soft-restoration original-criterion grid found lowercase `s` witnesses but no natural uppercase `S` focused witness. | Remaining P6 work is live/reduced witnesses for `DoFallback()` without a restoration phase, acceptable-point restoration fallback, almost-feasible restoration abort/backup restore, active-watchdog pre-line tiny-step stop, and uppercase `S`. |
| P7 | `IpFilterLSAcceptor.cpp` filter/SOC branches | `optimization/src/filter.rs` and SOC trial loop | Coverage witness backlog | Filter formulas, theta-max rejection, objective max-increase rejection, filter-reset counter semantics, SOC method 0/1 paths, and the feasible-reference tiny positive `reference_gradBarrTDelta_` adjustment are covered by focused Rust witnesses or source-built compare tests. The generated classifier now separates these from live witness gaps instead of leaving them as generic active/watch rows. | Remaining P7 work is a live SOC linear-solve failure witness if it can be triggered without editing linear internals, plus natural accepted-trace witnesses only if filter reset or objective max-increase produces observable drift. |
| P8 | `IpDefaultIterateInitializer.cpp` initialization branches | Initial iterate setup and least-square primal/dual initialization | Coverage witness/error backlog; no current accepted-trace drift | Default and several non-default initialization options are covered, including max-iter-0 state witnesses for least-square primal and dual initialization. The generated classifier now separates the equality-multiplier calculator success witness from calculator failure paths, TNLP initialization failure, invalid bound-multiplier enum handling, warm-start branches that are outside the parity profile, and the zero-`bound_push` branch that IPOPT option bounds reject in this lane. | Add explicit failure-path witnesses only if a deterministic TNLP or multiplier-calculator failure affects accepted state; otherwise keep these as classified error/witness gaps. |
| P9 | `IpIpoptCalculatedQuantities.cpp` cache/component branches | NLIP diagnostic snapshots and direct component recomputation | Bookkeeping unless values drift | Generated active/watch rows for this family are now classified. Cache-hit/miss and temporary-vector rows are bookkeeping. `kappa_d` damping rows map to the existing zero/nonzero damping witnesses. Adjusted-slack correction now mirrors `CalculateSafeSlack` for the zero-mu floor, target, move cap, and accepted-trial bound shift, with reduced helper coverage and regular-trial production wiring. Remaining component rows are explicit witness gaps for objective-dependent TNLP values, lower/upper inequality-bound dimensions, original-bound/bound-relaxation violation reporting, norm-option edges, target-shifted complementarity, and zero-dimension component reductions. IPOPT NLP x/objective scaling rows remain unreachable in the current SPRAL parity options because `apply_native_spral_parity_to_ipopt_options` forces `nlp_scaling_method=none` until NLIP has a source-compatible scaling layer. | Promote only rows that produce a component-value mismatch in `curr_grad_f`, Jacobian products, damping, bounds, slack distances, or KKT stationarity snapshots; treat NLP scaling as a separate option-profile feature if the parity lane moves past the current unscaled source-built SPRAL profile. |

The generated coverage report remains the working checklist. When it reports a
new active/watch line, the line must be assigned to one of the rows above or
explicitly reclassified as unreachable under parity options,
diagnostic/bookkeeping, or an option/error path.

## Branch Ledger

| Branch surface | IPOPT source anchor | Rust parity surface | Classification |
| --- | --- | --- | --- |
| Initial bound/slack push and default multiplier setup | `IpDefaultIterateInitializer.cpp` | initial iterate setup, scalar push helpers, `least_square_primal_initialization`, and `least_square_dual_initialization` | Mirrors IPOPT for default parity options plus focused non-default push, multiplier value, mu-based multiplier, `constr_mult_init_max=0`, low-cap least-square rejection, square-problem zero-multiplier witnesses, and requested `least_square_init_primal=yes` / `least_square_init_duals=yes`; warm-start remains unreachable under parity options |
| Source termination defaults | `IpIpoptData.cpp`, `IpOptErrorConvCheck.cpp` | `IpoptOptions::default`; OCP/studio strict profile remains separate | `IpoptOptions::default` mirrors IPOPT source defaults for `max_iter`, `tol`, absolute infeasibility tolerances, and acceptable-iterate secondary termination. OCP/studio defaults are not flipped yet, but the source-default glider profile now passes with matching accepted-trace length through restoration |
| Trial vector mutation | `IpIpoptData.cpp::SetTrial*FromStep` | current-plus-step helpers and accepted-trial construction | Mirrors IPOPT dense add branch order. Remaining uncovered C++ rows are initialization failure, `IpoptAdditionalData` hooks that are absent in this parity profile, and trial-container allocation bookkeeping rather than accepted-state arithmetic |
| Accepted point mutation and bound multiplier correction | `IpIpoptAlg.cpp::AcceptTrialPoint`, `correct_bound_multiplier` | accepted-trial commit and bound multiplier safeguard | Mirrors IPOPT for regular accepted-trial commit, adjusted tiny-slack bound shifts, and the bound-multiplier safeguard; default correction remains dormant under the broad `kappa_sigma` band, and the focused `kappa_sigma=1` witness forces correction of accepted trial multipliers against source IPOPT. Generated active/watch rows for this family are now zero; remaining rows are option/error/status paths, square-problem feasibility-multiplier postprocess witness gaps, skipped-line-search/fallback witness gaps, and non-regular-trial adjusted-slack natural witness gaps |
| Fixed variables | `IpTNLPAdapter.cpp::fixed_variable_treatment=make_parameter` plus Algorithm component snapshots | fixed-variable projection, sparse column reduction, and full-vector expansion | Mirrors IPOPT's default fixed-variable profile for the fixed-bound quadratic witness; `make_constraint`, `make_parameter_nodual`, and `relax_bounds` fixed-variable modes are outside the parity option profile |
| Monotone barrier update and line-search reset | `IpMonotoneMuUpdate.cpp`, `IpBacktrackingLineSearch.cpp::Reset` | `next_barrier_parameter`, filter/watchdog reset on mu change | Mirrors IPOPT's exact-Hessian default monotone mode, including positive `mu_target`, disabled fast-decrease first-call semantics, tiny-step forced one-drop behavior, exact `mu != new_mu` change detection, and first-restoration-update skip. Remaining generated rows are reduced-helper-covered line-search reset/no-change decisions or a tiny-step best-possible termination witness gap |
| Adaptive barrier update | `IpAdaptiveMuUpdate.cpp`, `IpAlgBuilder.cpp::BuildMuUpdate`, `IpLoqoMuOracle.cpp`, `IpQualityFunctionMuOracle.cpp`, `IpProbingMuOracle.cpp`, `IpFilter.cpp` | `InteriorPointMuStrategy::Adaptive`, LOQO centrality rule, quality-function affine/centering oracle, probing affine oracle, quality-function norm/centrality/balancing/search knobs, identity sigma-space section search, lazy `mu_min`, initial `mu_max_fact`, free-mode `tau`, objective/constraint adaptive filter, default KKT-error reference window, nonzero safeguard, non-default KKT-error reduction knobs, fixed/free-mode switch, restore-previous-iterate, restoration-prefixed option handoff, and line-search reset on source-matching update paths | Mostly covered non-default feature lane. Mirrors `loqo`, `quality-function`, and `probing` source slices against source-built IPOPT for the default adaptive option families, non-default quality-function options, and explicit `resto.mu_strategy=adaptive`; current source-built compare and helper lanes are green, and the only remaining adaptive item is a future natural fixed-mode accepted-trace witness if trace instrumentation or sweeps find one |
| Search direction construction | `IpPDSearchDirCalc.cpp` | reduced KKT RHS assembly and direction sign conversion | Mirrors IPOPT for `SpralSrc`; non-default Mehrotra probing RHS is implemented and covered, while fast-step remains an option-profile branch. The only generated active row is the `retval` status branch after `PDFullSpaceSolver::Solve`, now classified as a search-direction failure/status witness gap rather than an unclassified algorithm difference |
| Full-space residual and refinement | `IpPDFullSpaceSolver.cpp` | residual replay, refinement, correction RHS conversion, seeded improve-solution path, source refinement options, and inertia-free curvature heuristic | Mirrors IPOPT defaults and focused non-default `min_refinement_steps` / `max_refinement_steps` / residual-ratio options plus `neg_curv_test_tol` / `neg_curv_test_reg`. The coverage classifier now separates reduced witnesses, live failure/quality witness gaps, and linear-solver error paths instead of leaving generic active/watch rows |
| Calculated quantities and component caches | `IpIpoptCalculatedQuantities.cpp::*_cache_.GetCachedResult*`, adjusted slack, bound/original-bound violation helpers, damping, norm reductions, and complementarity metrics | eager NLIP component snapshots and direct recomputation | Cache hit/miss branches are classified as cache bookkeeping; they may affect runtime but not accepted-state semantics. The generated active/watch rows for this family are zero after classifying source rows into existing damping witnesses, adjusted-slack helper coverage, original-bound/component witness gaps, objective-dependent TNLP feature gaps, zero-dimension reductions, unreachable scaling rows under current parity options, or option/error handling. Component vector values remain guarded by callback snapshot assertions |
| Perturbation and inertia retry policy | `IpPDPerturbationHandler.cpp` | KKT regularization and retry loop | Mirrors IPOPT for active glider lane, non-default `perturb_always_cd`, singular-system degeneracy retry order, and wrong-inertia max-Hessian `delta_c/delta_d` fallback. Generated active/watch rows are zero after classifying retry transitions to reduced witnesses and structural-degeneracy finalization/status rows as witness gaps unless a future case shows accepted-trace drift |
| Filter local/global acceptability | `IpFilterLSAcceptor.cpp` | `optimization/src/filter.rs` and line-search trial assessment | Mirrors IPOPT filter formulas, theta-max rejection, objective max-increase rejection, the feasible-reference tiny positive `reference_gradBarrTDelta_` adjustment in `IsFtype`, filter-reset counter semantics, and the forced-accept `UpdateForNextIteration` f/h tagging used by `accept_every_trial_step` / `accept_after_max_steps`; coverage must keep f-type, h-type, dominated, and rejection paths visible |
| Alpha-for-y and dual step | `IpBacktrackingLineSearch.cpp::PerformDualStep` | `alpha_y`, `alpha_du`, multiplier step application | Mirrors IPOPT for the default primal strategy, implemented option-profile strategies, `min-dual-infeas` / `safer-min-dual-infeas` trial dual-infeasibility projection, and `alpha_for_y_tol` threshold profiles covered by focused tests |
| SOC | `IpFilterLSAcceptor.cpp::TrySecondOrderCorrection` | SOC trial loop and corrected accepted trial | Mirrors active IPOPT branch including dense `AddOneVector` order and both `soc_method=0` and `soc_method=1` x/s RHS scaling; fallback corrector variants are unreachable |
| Primal-dual/affine corrector | `IpFilterLSAcceptor.cpp::TryCorrector` | `InteriorPointCorrectorType::{PrimalDual, Affine}` | Mirrors IPOPT for focused non-default option profiles; unreachable under source-default parity options because IPOPT defaults `corrector_type=none` |
| Watchdog | `IpBacktrackingLineSearch.cpp` watchdog gates | watchdog reference, shortened-step streak, successful watchdog exit, lowercase watchdog trial, StopWatchDog restore/retry, and watchdog trial diagnostics | Mirrors IPOPT state machine for arming after current-direction construction, active-watchdog max-step-only trial search, successful `W` exit under the trace-clean `watchdog_shortened_iter_trigger=3` hanging-chain profile, lowercase `w` non-success trial acceptance, trial-budget `StopWatchDog` restore/retry under a focused CasADi Rosenbrock profile, and evaluation-error `StopWatchDog` retry alpha semantics. The active-watchdog pre-line-search tiny-step stop is implemented with the source distinction that it does not set `skip_first_trial_point`; the 2026-05-02 `print_watchdog_tiny_stop_profile_sweep` run found no natural end-to-end stop witness in the bounded grid |
| Soft restoration | `IpBacktrackingLineSearch.cpp::TrySoftRestoStep` | `soft_restoration_pderror_reduction_factor`, `max_soft_restoration_iters`, and soft-restoration line-search state | Mirrors IPOPT's soft-restoration entry after failed rigorous line search for the lowercase `s` primal-dual-error reduction path, including the original-filter augmentation, same-alpha primal/dual trial step, and skipped watchdog shortened-step update while `in_soft_resto_phase_` remains active. The uppercase `S` return-to-original-criterion path is implemented; the 2026-05-02 `print_soft_restoration_original_profile_sweep` run found lowercase `s` witnesses but no natural focused uppercase `S` witness in the bounded grid |
| Tiny step | `IpBacktrackingLineSearch.cpp::DetectTinyStep` | tiny-step acceptance and barrier-update tag | Mirrors IPOPT; focused tests exercise unchecked tiny-step acceptance, while the `tiny_step_tol=0` witness covers the disabled branch |
| Acceptable termination | `IpOptErrorConvCheck.cpp::CurrentIsAcceptable`, `IpIpoptAlg.cpp::CONVERGED_TO_ACCEPTABLE_POINT` | `InteriorPointTermination::Acceptable` and warning status | Mirrors IPOPT for `acceptable_iter=1` under intentionally tighter strict tolerances |
| Max-iteration exit | `IpIpoptAlg.cpp::ConvergenceCheck::MAXITER_EXCEEDED` | `InteriorPointSolveError::MaxIterations` and failure context | Mirrors IPOPT status for deterministic `max_iter=0` and accepted-step `max_iter=1`; focused tests require both solvers to retain diagnostics and partial state |
| Diverging-iterates exit | `IpOptErrorConvCheck.cpp::diverging_iterates_tol`, `IpIpoptAlg.cpp::DIVERGING_ITERATES` | `InteriorPointSolveError::DivergingIterates` and failure context | Mirrors IPOPT ordering after strict/acceptable convergence and before iteration/time limits; focused test forces the iteration-0 branch with `diverging_iterates_tol=0.5` |
| Wall-time exit | `IpOptErrorConvCheck.cpp::max_wall_time`, `IpIpoptAlg.cpp::WALLTIME_EXCEEDED` | `InteriorPointSolveError::WallTimeExceeded` and failure context | Mirrors IPOPT's positive-option bound and termination ordering after convergence/divergence; focused test forces the iteration-0 branch with `max_wall_time=1e-12` and compares IPOPT's `MaximumWallTimeExceeded` status |
| CPU-time exit | `IpOptErrorConvCheck.cpp::max_cpu_time`, `IpIpoptAlg.cpp::CPUTIME_EXCEEDED` | `InteriorPointSolveError::CpuTimeExceeded` and failure context | Mirrors IPOPT's positive-option bound and source termination ordering before wall time; focused test forces the iteration-0 branch with `max_cpu_time=1e-12` and compares IPOPT's `MaximumCpuTimeExceeded` status |
| User-requested stop | `IpOptErrorConvCheck.cpp::IntermediateCallBack`, `IpRestoConvCheck.cpp::CheckConvergence`, `IpIpoptAlg.cpp::USER_STOP` / `RESTORATION_USER_STOP` | `solve_nlp_interior_point_with_control_callback` and `InteriorPointSolveError::UserRequestedStop` | Mirrors IPOPT's false intermediate-callback return path after an accepted step and inside restoration; focused tests compare both solvers' retained state and IPOPT's `UserRequestedStop` status |
| Restoration | `IpBacktrackingLineSearch.cpp` restoration entry plus `IpRestoIpoptNLP.cpp` / `IpRestoIterateInitializer.cpp` / `IpRestoMinC_1Nrm.cpp` / `IpIpoptAlg.cpp` emergency mode / `IpAugRestoSystemSolver.cpp` | restoration diagnostic path | Mirrors IPOPT for the restoration NLP shape, initialization, original-objective eval-error probe, raised `resto.theta_max_fact`, source `n_c` / `p_c` and `n_d` / `p_d` quadratic initializer, original-progress return gate, copied slack, bound multiplier copy-back/reset, default zeroed original constraint multipliers, callback stops, iteration-limit exits, emergency local-infeasibility entry, one-shot `start_with_resto`, and expect-infeasible y-multiplier shortcut. Phase tests pin `IpAugRestoSystemSolver` formulas for `sigma_tilde_*`, `D_x_plus_wr_d`, equality and inequality `Rhs_*R`, source `D_d` overwrite behavior, and residual-variable back-substitution. Mixed/inequality and equality-only restoration now use the reduced-KKT production path under `SpralSrc`; the current glider first-divergence diagnostic has no tracked direction, alpha, accepted-direction, or accepted-state drift above the report thresholds |

## Rust-Only Branch Audit

Under the source-built `SpralSrc` parity profile, Rust branches that are not
part of IPOPT parity must be either unreachable or diagnostic:

| Rust branch family | Status for parity |
| --- | --- |
| `InteriorPointLinearSolver::Auto`, `SsidsRs`, `SparseQdldl`, fallback choices | Unreachable in nonlinear parity acceptance; linear-solver work is a separate lane |
| Dynamic/local SPRAL provenance checks | Diagnostic only; source-built `ipopt-src` plus `spral-src` is the nonlinear baseline |
| SQP/trust-region solver branches in `optimization/src/lib.rs` | Out of scope for NLIP/IPOPT parity |
| Verbose logging and debug dump branches | Diagnostic only; they must not change accepted state |
| Error reporting and failure-context construction | Diagnostic only except deterministic failure witnesses such as the source-backed `max_iter=0` and `max_iter=1` status tests |
| Restoration status mapping | Source-backed status classification; covered by restoration subproblem and phase status tests, and not an accepted-state mutation branch |

The active Rust core parity surface has no accepted-state-changing branch that
is intentionally Rust-only. Any future branch found by `rust-core.lcov` that is
active under the `SpralSrc` parity profile must be classified here before a
behavior change is accepted.

Known remaining nonlinear parity work:

1. Keep P0 as a regression, not an active implementation gap: the current
   glider first-divergence diagnostic reports `nlip_steps=152 ipopt_steps=152`
   with no tracked direction, alpha, accepted-direction, or accepted-state
   drift above the report thresholds. The next implementation should only
   touch restoration if a fresh witness shows a source-order mismatch in
   restoration reduced RHS, solution, refinement, copy-back arithmetic, or
   post-restoration regular KKT assembly.
2. Decide whether OCP and optimization-studio defaults should move from the
   current strict profile to the IPOPT-source-default profile. Profile
   differences must not be reported as algorithmic parity gaps.
3. Treat adaptive mu and corrector support as non-default exact-Hessian feature
   lanes. Both now have source-built witness coverage for the implemented option
   families. The current adaptive sweep did not expose a natural accepted-trace
   fixed-mode switch, so that branch remains covered by synthetic/reduced
   witnesses until a future trace-instrumented sweep finds a live case.
4. Burn down the coverage active/watch backlog by family: perturbation/inertia,
   monotone barrier update, full-space refinement and quality retry,
   line-search/fallback/watchdog/soft restoration, filter/SOC edge cases,
   initialization edge cases, and calculated-quantity component branches. Each
   accepted-state-changing branch needs a deterministic IPOPT/NLIP witness;
   cache-only and diagnostic branches should stay classified as bookkeeping.
5. Keep searching for natural end-to-end witnesses for IPOPT
   `RESTORATION_FAILED` / recoverable restoration returns, uppercase `S` soft
   restoration return-to-original-criterion, and the active-watchdog
   pre-line-search tiny-step path. The corresponding Rust branches are
   implemented or classified, but still lack natural source-built witnesses.

The generated `branch-ledger.md` report is the working checklist for this audit.
IPOPT uncovered branches in the watched routines must either map to the
`Current Known Differences` table, gain a focused witness, or stay explicitly
unreachable/diagnostic for the current exact-Hessian option profile. Rust
branch-like lines reported as `needs audit` must be driven to one of those
classifications before the coverage run is accepted.
