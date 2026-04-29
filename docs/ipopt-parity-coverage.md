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

- IPOPT Algorithm C++: `9784 / 18642` lines and `1211 / 3556` branches hit.
- Rust NLIP core: `5898 / 10503` lines hit, with `0` unhit branch-like lines
  still classified as `needs audit`.
- `IpBacktrackingLineSearch.cpp`: `610 / 993` lines and `142 / 252` branches
  hit, with active-watch uncovered branch lines reduced to `45 / 74`.

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
- IPOPT fixed-variable treatment through a fixed-bound quadratic witness,
  compared against NLIP's fixed-variable elimination path
- `IpIpoptAlg.cpp::correct_bound_multiplier` through a tight
  `kappa_sigma=1` accepted-trial safeguard witness
- `IpIpoptCalculatedQuantities.cpp::kappa_d=0` plus
  `IpOrigIpoptNLP.cpp::bound_relax_factor=0` on a bound-constrained problem
- `IpBacktrackingLineSearch.cpp::alpha_red_factor` through NLIP's
  `line_search_beta` option
- `IpMonotoneMuUpdate.cpp::mu_allow_fast_monotone_decrease=no`
- `IpMonotoneMuUpdate.cpp::mu_target>0`
- `IpFilterLSAcceptor.cpp::max_soc=0`
- `IpBacktrackingLineSearch.cpp::watchdog_shortened_iter_trigger=0`
- `IpBacktrackingLineSearch.cpp::StartWatchDog` through a hanging-chain
  profile with `watchdog_shortened_iter_trigger=3`, retained only after NLIP
  and IPOPT kept matching accepted-step counts and trace parity. The same
  profile also covers IPOPT's successful-watchdog `Append_info_string("W")`
  path; NLIP now marks filter/SOC trial acceptance as a watchdog success when
  the outer line-search state is already in watchdog mode and clears watchdog
  state immediately, matching `BacktrackingLineSearch::FindAcceptableTrialPoint`.
  (`trigger=1` and `trigger=2` were rejected as witnesses because they exposed
  accepted-step count drift despite close final values)
- `IpBacktrackingLineSearch.cpp::tiny_step_tol=0`
- `IpBacktrackingLineSearch.cpp::alpha_for_y` values `bound-mult`, `min`,
  `max`, `full`, `primal-and-full`, and `dual-and-full`
- `IpBacktrackingLineSearch.cpp::alpha_for_y_tol` through strict
  `primal-and-full` / `dual-and-full` option profiles
- `IpIpoptAlg.cpp::ConvergenceCheck::MAXITER_EXCEEDED` via `max_iter=0`
  and via `max_iter=1` after an accepted trial
- `IpOptErrorConvCheck.cpp::CONVERGED_TO_ACCEPTABLE_POINT` via tight strict
  tolerances, loose acceptable tolerances, and `acceptable_iter=1`
- `IpBacktrackingLineSearch.cpp::DetectTinyStep` via intentionally large
  `tiny_step_tol` and `tiny_step_y_tol`

These witnesses are wired through `IpoptOptions::raw_options` so normal IPOPT
solve behavior is unchanged unless a test or diagnostic explicitly requests a
source option.

## Branch Ledger

| Branch surface | IPOPT source anchor | Rust parity surface | Classification |
| --- | --- | --- | --- |
| Initial bound/slack push and default multiplier setup | `IpDefaultIterateInitializer.cpp` | `initialize_interior_point_iterates`, scalar push helpers | Mirrors IPOPT for default parity options plus focused non-default push, multiplier value, mu-based multiplier, `constr_mult_init_max=0`, low-cap least-square rejection, and square-problem zero-multiplier witnesses; warm-start and least-square primal/dual initialization are unreachable unless options request them |
| Trial vector mutation | `IpIpoptData.cpp::SetTrial*FromStep` | current-plus-step helpers and accepted-trial construction | Mirrors IPOPT dense add branch order; coverage should include alpha-1 and general-alpha paths |
| Accepted point mutation and bound multiplier correction | `IpIpoptAlg.cpp::AcceptTrialPoint`, `correct_bound_multiplier` | accepted-trial commit and bound multiplier safeguard | Mirrors IPOPT; default branch remains dormant under the broad `kappa_sigma` band, and the focused `kappa_sigma=1` witness forces correction of accepted trial multipliers against source IPOPT |
| Fixed variables | `IpTNLPAdapter.cpp::fixed_variable_treatment=make_parameter` plus Algorithm component snapshots | fixed-variable projection, sparse column reduction, and full-vector expansion | Mirrors IPOPT's default fixed-variable profile for the fixed-bound quadratic witness; `make_constraint`, `make_parameter_nodual`, and `relax_bounds` fixed-variable modes are outside the parity option profile |
| Monotone barrier update and line-search reset | `IpMonotoneMuUpdate.cpp`, `IpBacktrackingLineSearch.cpp::Reset` | `next_barrier_parameter`, filter/watchdog reset on mu change | Mirrors IPOPT monotone mode, including positive `mu_target`; adaptive/free-mu branches are unreachable in this lane |
| Search direction construction | `IpPDSearchDirCalc.cpp` | reduced KKT RHS assembly and direction sign conversion | Mirrors IPOPT for `SpralSrc`; Mehrotra/fast-step branches are unreachable in parity options |
| Full-space residual and refinement | `IpPDFullSpaceSolver.cpp` | residual replay, refinement, and correction RHS conversion | Mirrors IPOPT; linear-solver-specific internals are out of this nonlinear audit unless exact KKT/RHS evidence regresses |
| Calculated-quantity cache hit/miss | `IpIpoptCalculatedQuantities.cpp::*_cache_.GetCachedResult*` | eager NLIP component snapshots and direct recomputation | Cache hit/miss branches are classified as cache bookkeeping; they may affect runtime but not accepted-state semantics, while component vector values remain covered by callback snapshot assertions |
| Perturbation and inertia retry policy | `IpPDPerturbationHandler.cpp` | KKT regularization and retry loop | Mirrors IPOPT for active glider lane; degeneracy branches need coverage witnesses before behavior changes |
| Filter local/global acceptability | `IpFilterLSAcceptor.cpp` | `optimization/src/filter.rs` and line-search trial assessment | Mirrors IPOPT filter formulas; coverage must keep f-type, h-type, dominated, and rejection paths visible |
| Alpha-for-y and dual step | `IpBacktrackingLineSearch.cpp::PerformDualStep` | `alpha_y`, `alpha_du`, multiplier step application | Mirrors IPOPT for the default primal strategy, implemented option-profile strategies, and `alpha_for_y_tol` threshold profiles covered by focused tests |
| SOC | `IpFilterLSAcceptor.cpp::TrySecondOrderCorrection` | SOC trial loop and corrected accepted trial | Mirrors active IPOPT branch including dense `AddOneVector` order; fallback corrector variants are unreachable |
| Primal-dual corrector | `IpFilterLSAcceptor.cpp::TryCorrector` | none under parity options | Unreachable under the current parity option profile, which keeps `corrector_type=none`; do not enable IPOPT `corrector_type` raw options for parity acceptance until the full branch is ported source-faithfully |
| Watchdog | `IpBacktrackingLineSearch.cpp` watchdog gates | watchdog reference, shortened-step streak, successful watchdog exit, and watchdog trial diagnostics | Mirrors IPOPT state machine for arming and successful `W` exit under the trace-clean `watchdog_shortened_iter_trigger=3` hanging-chain profile; the ignored watchdog sweep prints IPOPT lowercase `w`, `Tmax`, and step-tag summaries so non-success watchdog trial / `StopWatchDog` profiles stay visible before accepted-state behavior changes |
| Tiny step | `IpBacktrackingLineSearch.cpp::DetectTinyStep` | tiny-step acceptance and barrier-update tag | Mirrors IPOPT; focused tests exercise unchecked tiny-step acceptance, while the `tiny_step_tol=0` witness covers the disabled branch |
| Acceptable termination | `IpOptErrorConvCheck.cpp::CurrentIsAcceptable`, `IpIpoptAlg.cpp::CONVERGED_TO_ACCEPTABLE_POINT` | `InteriorPointTermination::Acceptable` and warning status | Mirrors IPOPT for `acceptable_iter=1` under intentionally tighter strict tolerances |
| Max-iteration exit | `IpIpoptAlg.cpp::ConvergenceCheck::MAXITER_EXCEEDED` | `InteriorPointSolveError::MaxIterations` and failure context | Mirrors IPOPT status for deterministic `max_iter=0` and accepted-step `max_iter=1`; focused tests require both solvers to retain diagnostics and partial state |
| Restoration | `IpBacktrackingLineSearch.cpp` restoration entry | equality-restoration diagnostic path | Diagnostic only for current source-SPRAL parity; do not use restoration as a repair unless current state, direction, and alpha already match before a restoration decision |

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

Known uncovered source-backed parity work:

- IPOPT `least_square_init_duals=yes` enters
  `IpDefaultIterateInitializer.cpp::CalculateLeastSquareDuals`, which computes
  `z_L`, `z_U`, `v_L`, `v_U`, `y_c`, and `y_d` together and clamps bound
  multipliers against the current initialization. NLIP now guards
  `least_square_init_duals=true` with an explicit invalid-input error that names
  this IPOPT routine, so the unported branch cannot silently run with different
  semantics. Do not add a passing parity witness for this option until the full
  IPOPT branch is ported source-faithfully.
- IPOPT lowercase watchdog trials enter
  `IpBacktrackingLineSearch.cpp::DoBacktrackingLineSearch` with `in_watchdog_`
  active, set the alpha-primal tag to `w`, and may later call
  `BacktrackingLineSearch::StopWatchDog` to restore the stored watchdog iterate
  and retry from the stored watchdog direction. The current trace-clean
  accepted witness covers the successful `W` exit, while reduced trigger-1/2
  profiles expose lowercase `w` / `Tmax` IPOPT markers and accepted-step count
  drift. Porting this branch requires carrying the full stored watchdog iterate
  and direction, not just accepting the failed trial.

The generated `branch-ledger.md` report is the working checklist for this
audit. IPOPT uncovered branches in the watched routines should either gain a
focused witness or stay explicitly unreachable for the current option profile.
Rust branch-like lines reported as `needs audit` should be driven to one of the
classifications above, or converted into a source-backed test if they affect
accepted state.
