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

- IPOPT Algorithm C++: `10318 / 18642` lines and `1346 / 3556` branches hit.
- Rust NLIP core: `8818 / 14187` lines hit, with `0` unhit branch-like lines
  still classified as `needs audit`.
- `IpDefaultIterateInitializer.cpp`: `498 / 532` lines and `58 / 70` branches
  hit after adding the requested `least_square_init_primal=yes` witness.
- `IpIpoptAlg.cpp` convergence-status cases for max-iteration, diverging
  iterates, CPU time, wall time, acceptable termination, and user-requested
  stop are now covered by focused witnesses.
- `IpBacktrackingLineSearch.cpp`: `717 / 993` lines and `188 / 252` branches
  hit, with active-watch uncovered branch lines at `35 / 51`.

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
  `max`, `full`, `primal-and-full`, and `dual-and-full`
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

These witnesses are wired through `IpoptOptions::raw_options` so normal IPOPT
solve behavior is unchanged unless a test or diagnostic explicitly requests a
source option.

## Branch Ledger

| Branch surface | IPOPT source anchor | Rust parity surface | Classification |
| --- | --- | --- | --- |
| Initial bound/slack push and default multiplier setup | `IpDefaultIterateInitializer.cpp` | initial iterate setup, scalar push helpers, `least_square_primal_initialization`, and `least_square_dual_initialization` | Mirrors IPOPT for default parity options plus focused non-default push, multiplier value, mu-based multiplier, `constr_mult_init_max=0`, low-cap least-square rejection, square-problem zero-multiplier witnesses, and requested `least_square_init_primal=yes` / `least_square_init_duals=yes`; warm-start remains unreachable under parity options |
| Trial vector mutation | `IpIpoptData.cpp::SetTrial*FromStep` | current-plus-step helpers and accepted-trial construction | Mirrors IPOPT dense add branch order; coverage should include alpha-1 and general-alpha paths |
| Accepted point mutation and bound multiplier correction | `IpIpoptAlg.cpp::AcceptTrialPoint`, `correct_bound_multiplier` | accepted-trial commit and bound multiplier safeguard | Mirrors IPOPT; default branch remains dormant under the broad `kappa_sigma` band, and the focused `kappa_sigma=1` witness forces correction of accepted trial multipliers against source IPOPT |
| Fixed variables | `IpTNLPAdapter.cpp::fixed_variable_treatment=make_parameter` plus Algorithm component snapshots | fixed-variable projection, sparse column reduction, and full-vector expansion | Mirrors IPOPT's default fixed-variable profile for the fixed-bound quadratic witness; `make_constraint`, `make_parameter_nodual`, and `relax_bounds` fixed-variable modes are outside the parity option profile |
| Monotone barrier update and line-search reset | `IpMonotoneMuUpdate.cpp`, `IpBacktrackingLineSearch.cpp::Reset` | `next_barrier_parameter`, filter/watchdog reset on mu change | Mirrors IPOPT monotone mode, including positive `mu_target`; adaptive/free-mu branches are unreachable in this lane |
| Search direction construction | `IpPDSearchDirCalc.cpp` | reduced KKT RHS assembly and direction sign conversion | Mirrors IPOPT for `SpralSrc`; Mehrotra/fast-step branches are unreachable in parity options |
| Full-space residual and refinement | `IpPDFullSpaceSolver.cpp` | residual replay, refinement, and correction RHS conversion | Mirrors IPOPT; linear-solver-specific internals are out of this nonlinear audit unless exact KKT/RHS evidence regresses |
| Calculated-quantity cache hit/miss | `IpIpoptCalculatedQuantities.cpp::*_cache_.GetCachedResult*` | eager NLIP component snapshots and direct recomputation | Cache hit/miss branches are classified as cache bookkeeping; they may affect runtime but not accepted-state semantics, while component vector values remain covered by callback snapshot assertions |
| Perturbation and inertia retry policy | `IpPDPerturbationHandler.cpp` | KKT regularization and retry loop | Mirrors IPOPT for active glider lane; degeneracy branches need coverage witnesses before behavior changes |
| Filter local/global acceptability | `IpFilterLSAcceptor.cpp` | `optimization/src/filter.rs` and line-search trial assessment | Mirrors IPOPT filter formulas and the forced-accept `UpdateForNextIteration` f/h tagging used by `accept_every_trial_step` / `accept_after_max_steps`; coverage must keep f-type, h-type, dominated, and rejection paths visible |
| Alpha-for-y and dual step | `IpBacktrackingLineSearch.cpp::PerformDualStep` | `alpha_y`, `alpha_du`, multiplier step application | Mirrors IPOPT for the default primal strategy, implemented option-profile strategies, and `alpha_for_y_tol` threshold profiles covered by focused tests |
| SOC | `IpFilterLSAcceptor.cpp::TrySecondOrderCorrection` | SOC trial loop and corrected accepted trial | Mirrors active IPOPT branch including dense `AddOneVector` order and both `soc_method=0` and `soc_method=1` x/s RHS scaling; fallback corrector variants are unreachable |
| Primal-dual corrector | `IpFilterLSAcceptor.cpp::TryCorrector` | none under parity options | Unreachable under the current parity option profile, which keeps `corrector_type=none`; do not enable IPOPT `corrector_type` raw options for parity acceptance until the full branch is ported source-faithfully |
| Watchdog | `IpBacktrackingLineSearch.cpp` watchdog gates | watchdog reference, shortened-step streak, successful watchdog exit, lowercase watchdog trial, StopWatchDog restore/retry, and watchdog trial diagnostics | Mirrors IPOPT state machine for arming after current-direction construction, active-watchdog max-step-only trial search, successful `W` exit under the trace-clean `watchdog_shortened_iter_trigger=3` hanging-chain profile, lowercase `w` non-success trial acceptance, trial-budget `StopWatchDog` restore/retry under a focused CasADi Rosenbrock profile, and evaluation-error `StopWatchDog` retry alpha semantics. The active-watchdog pre-line-search tiny-step stop is implemented with the source distinction that it does not set `skip_first_trial_point`, but no natural end-to-end witness has been found yet |
| Soft restoration | `IpBacktrackingLineSearch.cpp::TrySoftRestoStep` | `soft_restoration_pderror_reduction_factor`, `max_soft_restoration_iters`, and soft-restoration line-search state | Mirrors IPOPT's soft-restoration entry after failed rigorous line search for the lowercase `s` primal-dual-error reduction path, including the original-filter augmentation, same-alpha primal/dual trial step, and skipped watchdog shortened-step update while `in_soft_resto_phase_` remains active. The uppercase `S` return-to-original-criterion path is implemented and now has a bounded quiet-journal search, but no natural focused witness has been found |
| Tiny step | `IpBacktrackingLineSearch.cpp::DetectTinyStep` | tiny-step acceptance and barrier-update tag | Mirrors IPOPT; focused tests exercise unchecked tiny-step acceptance, while the `tiny_step_tol=0` witness covers the disabled branch |
| Acceptable termination | `IpOptErrorConvCheck.cpp::CurrentIsAcceptable`, `IpIpoptAlg.cpp::CONVERGED_TO_ACCEPTABLE_POINT` | `InteriorPointTermination::Acceptable` and warning status | Mirrors IPOPT for `acceptable_iter=1` under intentionally tighter strict tolerances |
| Max-iteration exit | `IpIpoptAlg.cpp::ConvergenceCheck::MAXITER_EXCEEDED` | `InteriorPointSolveError::MaxIterations` and failure context | Mirrors IPOPT status for deterministic `max_iter=0` and accepted-step `max_iter=1`; focused tests require both solvers to retain diagnostics and partial state |
| Diverging-iterates exit | `IpOptErrorConvCheck.cpp::diverging_iterates_tol`, `IpIpoptAlg.cpp::DIVERGING_ITERATES` | `InteriorPointSolveError::DivergingIterates` and failure context | Mirrors IPOPT ordering after strict/acceptable convergence and before iteration/time limits; focused test forces the iteration-0 branch with `diverging_iterates_tol=0.5` |
| Wall-time exit | `IpOptErrorConvCheck.cpp::max_wall_time`, `IpIpoptAlg.cpp::WALLTIME_EXCEEDED` | `InteriorPointSolveError::WallTimeExceeded` and failure context | Mirrors IPOPT's positive-option bound and termination ordering after convergence/divergence; focused test forces the iteration-0 branch with `max_wall_time=1e-12` and compares IPOPT's `MaximumWallTimeExceeded` status |
| CPU-time exit | `IpOptErrorConvCheck.cpp::max_cpu_time`, `IpIpoptAlg.cpp::CPUTIME_EXCEEDED` | `InteriorPointSolveError::CpuTimeExceeded` and failure context | Mirrors IPOPT's positive-option bound and source termination ordering before wall time; focused test forces the iteration-0 branch with `max_cpu_time=1e-12` and compares IPOPT's `MaximumCpuTimeExceeded` status |
| User-requested stop | `IpOptErrorConvCheck.cpp::IntermediateCallBack`, `IpRestoConvCheck.cpp::CheckConvergence`, `IpIpoptAlg.cpp::USER_STOP` / `RESTORATION_USER_STOP` | `solve_nlp_interior_point_with_control_callback` and `InteriorPointSolveError::UserRequestedStop` | Mirrors IPOPT's false intermediate-callback return path after an accepted step and inside restoration; focused tests compare both solvers' retained state and IPOPT's `UserRequestedStop` status |
| Restoration | `IpBacktrackingLineSearch.cpp` restoration entry plus `IpRestoIpoptNLP.cpp` / `IpRestoIterateInitializer.cpp` / `IpRestoMinC_1Nrm.cpp` / `IpRestoConvCheck.cpp` / `IpIpoptAlg.cpp` emergency mode | restoration diagnostic path | Mirrors IPOPT for the current source-SPRAL parity profile; the restoration NLP mirrors IPOPT's `rho`, `eta=sqrt(mu)`, objective/Jacobian/Hessian shape, original-objective eval-error probe, raised `resto.theta_max_fact`, source `n_c` / `p_c` and `n_d` / `p_d` quadratic initializer for equality and `d-s` residuals. On a restoration return, NLIP copies the restoration slack, applies IPOPT's `ComputeBoundMultiplierStep` plus the default `bound_mult_reset_threshold=1e3`, resets original constraint multipliers to zero like `resto.constr_mult_init_max=0`, gates the return through IPOPT's `RestoConvergenceCheck` original-problem progress rules with the active `max_iter` budget, `max_resto_iter` successive-iteration limit, and `expect_infeasible_problem` first-restoration return threshold, maps square-problem feasible restoration returns through `ComputeFeasibilityMultipliersPostprocess` and classifies them as `Converged` when the full KKT test is satisfied or `FeasiblePointFound` otherwise, maps restoration callback stops to `UserRequestedStop`, maps restoration iteration-limit exits to `MaxIterations`, maps restoration original-objective evaluation-error max-iteration cases to `MaxIterations`, maps emergency-restoration local infeasibility and one-shot `start_with_resto` entry to `InteriorPointSolveError::LocalInfeasibility` like IPOPT's `InfeasibleProblemDetected`, and mirrors the expect-infeasible y-multiplier restoration shortcut. No source-SPRAL restoration behavior mismatch is known after the current focused suite and glider diagnostic |

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

The active Rust core parity surface has no accepted-state-changing branch that
is intentionally Rust-only. Any future branch found by `rust-core.lcov` that is
active under the `SpralSrc` parity profile must be classified here before a
behavior change is accepted.

Known remaining nonlinear parity witness searches:

- IPOPT active-watchdog `StopWatchDog` after trial-budget exhaustion is now
  covered: NLIP restores the stored watchdog iterate and direction, clears the
  active watchdog state, skips the already-tested first trial, and retries from
  the stored line-search reference. NLIP also carries the pre-line-search
  `in_watchdog_ && tiny_step` stop path from
  `IpBacktrackingLineSearch.cpp::FindAcceptableTrialPoint`; unlike the
  trial-budget branch, this path retries the stored direction at the stored
  maximum step without setting `skip_first_trial_point`, and the rule is pinned
  by a focused unit test plus the ignored `print_watchdog_tiny_stop_profile_sweep`
  search harness. The current empirical sweep did not find a natural source-built
  problem witness for that path. Evaluation-error trial handling is now covered
  by a deterministic nonfinite-trial witness and the watchdog restore alpha
  branch is pinned to IPOPT's skip-first semantics. The restoration diagnostic
  path now carries IPOPT's `n_c` / `p_c` and `n_d` / `p_d` residual variables,
  uses `RestoIterateInitializer`'s quadratic formula for both equality and
  `d-s` residuals, copies restoration slack back to the original iterate, and
  applies `MinC_1NrmRestorationPhase::ComputeBoundMultiplierStep` before
  resetting default restoration constraint multipliers to zero. It also applies
  `RestoConvergenceCheck`'s original-problem return gate, including square
  problem feasibility and filter/current-iterate acceptability, and carries
  `MinC_1NrmRestorationPhase::PerformRestoration`'s square
  `FEASIBILITY_PROBLEM_SOLVED` result as a native `FeasiblePointFound`
  termination. The emergency `goto_resto` path after failed
  `ComputeSearchDirection` is now covered by an impossible square-equality
  witness that matches IPOPT's local-infeasibility return status, the one-shot
  `start_with_resto` option and the expect-infeasible y-multiplier shortcut now
  have matching natural witnesses, and the same witness now pins
  restoration-phase user callback stops to IPOPT's outer
  `UserRequestedStop` status. NLIP also mirrors
  `RestoIpoptNLP::f`'s default original-objective eval-error probe and
  `RestoConvergenceCheck::InitializeImpl`'s active `max_iter` budget instead of
  applying a Rust-only restoration iteration clamp. The focused
  `max_resto_iter=0` witness now pins `RestoConvCheck`'s successive restoration
  iteration limit and IPOPT's restoration iteration numbering. The restoration
  subproblem now also mirrors `MinC_1NrmRestorationPhase`'s raised default
  `resto.theta_max_fact=1e8`. The synthetic restoration original-objective
  eval-error case now matches IPOPT's tiny restoration-step
  `MaximumIterationsExceeded` behavior. No accepted-state-changing nonlinear
  behavior difference is known in the source-SPRAL parity profile. Remaining
  witness searches are for naturally occurring IPOPT `RESTORATION_FAILED` /
  recoverable restoration returns, a natural uppercase `S` soft-restoration
  return-to-original-criterion path, and a natural active-watchdog pre-line
  tiny-step path; the corresponding Rust branches are implemented or classified
  but do not currently have natural end-to-end witnesses.

The generated `branch-ledger.md` report is the working checklist for this
audit. IPOPT uncovered branches in the watched routines should either gain a
focused witness or stay explicitly unreachable for the current option profile.
Rust branch-like lines reported as `needs audit` should be driven to one of the
classifications above, or converted into a source-backed test if they affect
accepted state.
