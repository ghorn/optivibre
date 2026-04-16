mod brown_almost_linear;
mod disk_rosenbrock;
mod generalized_rosenbrock;
mod hanging_chain;
mod hs021;
mod hs035;
mod hs071;
mod parameterized_quadratic;
mod powell_singular;
mod trigonometric;
mod wood;

use std::fmt::Write as _;
use std::time::Instant;

use optimization::{
    ClarabelSqpError, ClarabelSqpOptions, ClarabelSqpSummary, CompiledNlpProblem,
    FilterAcceptanceMode, InteriorPointIterationSnapshot, InteriorPointOptions,
    InteriorPointSolveError, InteriorPointSummary, SqpIterationSnapshot, SymbolicNlpOutputs,
    TypedCompiledJitNlp, TypedRuntimeNlpBounds, Vectorize,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptOptions, IpoptRawStatus, IpoptSolveError, IpoptSummary};
use sx_core::SX;

use crate::manifest::KnownStatus;
use crate::model::{
    CompileReportSummary, CompileStatsSummary, FilterReplay, FilterReplayFrame, FilterReplayPoint,
    ProblemCase, ProblemDescriptor, ProblemRunOptions, ProblemRunRecord, RunStatus,
    SetupProfileBreakdown, SolverKind, SolverMetrics, SolverTimingBreakdown, ValidationOutcome,
    ValidationTier,
};

const STRICT_TERMINATION_TOL: f64 = 1e-9;
const REDUCED_TERMINATION_TOL: f64 = 1e-6;

#[derive(Clone, optimization::Vectorize)]
pub(crate) struct Pair<T> {
    pub(crate) x: T,
    pub(crate) y: T,
}

#[derive(Clone, optimization::Vectorize)]
pub(crate) struct VecN<T, const N: usize> {
    pub(crate) values: [T; N],
}

#[derive(Clone, optimization::Vectorize)]
pub(crate) struct Point<T> {
    pub(crate) x: T,
    pub(crate) y: T,
}

#[derive(Clone, optimization::Vectorize)]
pub(crate) struct Chain<T, const N: usize> {
    pub(crate) points: [Point<T>; N],
}

#[derive(Clone, Copy)]
pub(crate) struct CaseMetadata {
    id: &'static str,
    family: &'static str,
    variant: &'static str,
    source: &'static str,
    description: &'static str,
    parameterized: bool,
}

impl CaseMetadata {
    pub(crate) const fn new(
        id: &'static str,
        family: &'static str,
        variant: &'static str,
        source: &'static str,
        description: &'static str,
        parameterized: bool,
    ) -> Self {
        Self {
            id,
            family,
            variant,
            source,
            description,
            parameterized,
        }
    }
}

pub(crate) fn all_cases() -> Vec<ProblemCase> {
    let mut cases = Vec::new();
    cases.extend(generalized_rosenbrock::cases());
    cases.push(disk_rosenbrock::case());
    cases.push(powell_singular::case());
    cases.push(wood::case());
    cases.extend(brown_almost_linear::cases());
    cases.extend(trigonometric::cases());
    cases.extend(hanging_chain::cases());
    cases.push(parameterized_quadratic::case());
    cases.push(hs021::case());
    cases.push(hs035::case());
    cases.push(hs071::case());
    cases
}

pub(crate) struct TypedProblemData<X, P, E, I>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
{
    pub(crate) compiled: TypedCompiledJitNlp<X, P, E, I>,
    pub(crate) x0: <X as Vectorize<SX>>::Rebind<f64>,
    pub(crate) parameters: <P as Vectorize<SX>>::Rebind<f64>,
    pub(crate) bounds: TypedRuntimeNlpBounds<X, I>,
}

pub(crate) fn symbolic_compile<X, P, E, I, F>(
    name: &str,
    model: F,
    options: ProblemRunOptions,
) -> anyhow::Result<TypedCompiledJitNlp<X, P, E, I>>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    F: FnOnce(&X, &P) -> SymbolicNlpOutputs<E, I>,
{
    Ok(optimization::symbolic_nlp::<X, P, E, I, _>(name, model)?
        .compile_jit_with_options(options.compile_options())?)
}

pub(crate) fn make_typed_case<X, P, E, I, Build, Validate>(
    metadata: CaseMetadata,
    build: Build,
    validate: Validate,
) -> ProblemCase
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <E as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    Build: Fn(ProblemRunOptions) -> anyhow::Result<TypedProblemData<X, P, E, I>>
        + Send
        + Sync
        + 'static,
    Validate: Fn(&ProblemRunRecord) -> ValidationOutcome + Send + Sync + 'static,
{
    ProblemCase {
        id: metadata.id,
        family: metadata.family,
        variant: metadata.variant,
        source: metadata.source,
        description: metadata.description,
        parameterized: metadata.parameterized,
        run_fn: Box::new(move |solver, options, max_iters_limit, expected| {
            run_typed_case(
                solver,
                options,
                max_iters_limit,
                expected,
                metadata,
                &build,
                &validate,
            )
        }),
    }
}

fn run_typed_case<X, P, E, I, Build, Validate>(
    solver: SolverKind,
    options: ProblemRunOptions,
    max_iters_limit: usize,
    expected: KnownStatus,
    metadata: CaseMetadata,
    build: &Build,
    validate: &Validate,
) -> ProblemRunRecord
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <E as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    Build: Fn(ProblemRunOptions) -> anyhow::Result<TypedProblemData<X, P, E, I>>,
    Validate: Fn(&ProblemRunRecord) -> ValidationOutcome,
{
    let total_started = Instant::now();
    let build_started = Instant::now();
    match build(options) {
        Ok(data) => run_built_case(
            solver,
            options,
            max_iters_limit,
            expected,
            metadata,
            &data,
            build_started.elapsed(),
            total_started,
            validate,
        ),
        Err(err) => ProblemRunRecord {
            id: metadata.id.to_string(),
            solver,
            options,
            expected,
            max_iters_limit,
            status: RunStatus::SolveError,
            descriptor: ProblemDescriptor {
                id: metadata.id.to_string(),
                family: metadata.family.to_string(),
                variant: metadata.variant.to_string(),
                source: metadata.source.to_string(),
                description: metadata.description.to_string(),
                parameterized: metadata.parameterized,
                num_vars: 0,
                num_eq: 0,
                num_ineq: 0,
                num_box: 0,
                dof: 0,
                constrained: false,
            },
            solution: None,
            metrics: SolverMetrics::default(),
            timing: SolverTimingBreakdown {
                compile_wall_time: Some(build_started.elapsed()),
                total_wall_time: total_started.elapsed(),
                ..SolverTimingBreakdown::default()
            },
            validation: ValidationOutcome {
                tier: ValidationTier::Failed,
                tolerance: "n/a".to_string(),
                detail: "build failed".to_string(),
            },
            solver_thresholds: None,
            error: Some(err.to_string()),
            compile_report: None,
            console_output: None,
            console_output_path: None,
            filter_replay: None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn run_built_case<X, P, E, I, Validate>(
    solver: SolverKind,
    options: ProblemRunOptions,
    max_iters_limit: usize,
    expected: KnownStatus,
    metadata: CaseMetadata,
    data: &TypedProblemData<X, P, E, I>,
    compile_wall_time: std::time::Duration,
    total_started: Instant,
    validate: &Validate,
) -> ProblemRunRecord
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <E as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone + Send + Sync + 'static,
    Validate: Fn(&ProblemRunRecord) -> ValidationOutcome,
{
    let backend_timing = data.compiled.backend_timing_metadata();
    let compile_report =
        summarize_backend_compile_report(Some(data.compiled.backend_compile_report()));
    let bound_problem = match data.compiled.bind_runtime_bounds(&data.bounds) {
        Ok(problem) => problem,
        Err(err) => {
            return ProblemRunRecord {
                id: metadata.id.to_string(),
                solver,
                options,
                expected,
                max_iters_limit,
                status: RunStatus::SolveError,
                descriptor: ProblemDescriptor {
                    id: metadata.id.to_string(),
                    family: metadata.family.to_string(),
                    variant: metadata.variant.to_string(),
                    source: metadata.source.to_string(),
                    description: metadata.description.to_string(),
                    parameterized: metadata.parameterized,
                    num_vars: 0,
                    num_eq: 0,
                    num_ineq: 0,
                    num_box: 0,
                    dof: 0,
                    constrained: false,
                },
                solution: None,
                metrics: SolverMetrics::default(),
                timing: timing_breakdown(
                    backend_timing,
                    compile_wall_time,
                    None,
                    total_started.elapsed(),
                ),
                validation: ValidationOutcome {
                    tier: ValidationTier::Failed,
                    tolerance: "n/a".to_string(),
                    detail: "runtime bounds invalid".to_string(),
                },
                solver_thresholds: None,
                error: Some(err.to_string()),
                compile_report: compile_report.clone(),
                console_output: None,
                console_output_path: None,
                filter_replay: None,
            };
        }
    };

    let descriptor = describe_problem(metadata, &bound_problem, &data.bounds);

    match solver {
        SolverKind::Sqp => {
            let sqp_options = ClarabelSqpOptions {
                max_iters: max_iters_limit,
                dual_tol: STRICT_TERMINATION_TOL,
                constraint_tol: STRICT_TERMINATION_TOL,
                complementarity_tol: STRICT_TERMINATION_TOL,
                verbose: false,
                ..ClarabelSqpOptions::default()
            };
            let solver_thresholds = format_sqp_thresholds(&sqp_options);
            let solve_started = Instant::now();
            let mut snapshots = Vec::new();
            let result = data.compiled.solve_sqp_with_callback(
                &data.x0,
                &data.parameters,
                &data.bounds,
                &sqp_options,
                |snapshot: &SqpIterationSnapshot| snapshots.push(snapshot.clone()),
            );
            let solve_wall_time = solve_started.elapsed();
            match result {
                Ok(summary) => {
                    let metrics = metrics_from_sqp_summary(&summary);
                    let mut record = ProblemRunRecord {
                        id: metadata.id.to_string(),
                        solver,
                        options,
                        expected,
                        max_iters_limit,
                        status: RunStatus::Passed,
                        descriptor,
                        solution: Some(summary.x.clone()),
                        metrics,
                        timing: timing_breakdown(
                            backend_timing,
                            compile_wall_time,
                            Some(summary.profiling.total_time.max(solve_wall_time)),
                            total_started.elapsed(),
                        ),
                        validation: ValidationOutcome::default(),
                        solver_thresholds: Some(solver_thresholds.clone()),
                        error: None,
                        compile_report: compile_report.clone(),
                        console_output: None,
                        console_output_path: None,
                        filter_replay: None,
                    };
                    record.validation = validate(&record);
                    if !record.validation.passed() {
                        record.status = RunStatus::FailedValidation;
                    } else if matches!(record.validation.tier, ValidationTier::ReducedAccuracy) {
                        record.status = RunStatus::ReducedAccuracy;
                    }
                    record.console_output = Some(render_sqp_transcript(
                        &record,
                        &snapshots,
                        Some(&summary),
                        None,
                    ));
                    record.filter_replay = sqp_filter_replay_from_snapshots(&snapshots);
                    record
                }
                Err(err) => {
                    let mut record = ProblemRunRecord {
                        id: metadata.id.to_string(),
                        solver,
                        options,
                        expected,
                        max_iters_limit,
                        status: RunStatus::SolveError,
                        descriptor,
                        solution: None,
                        metrics: metrics_from_sqp_error(&err),
                        timing: timing_breakdown(
                            backend_timing,
                            compile_wall_time,
                            solve_time_from_sqp_error(&err).or(Some(solve_wall_time)),
                            total_started.elapsed(),
                        ),
                        validation: ValidationOutcome {
                            tier: ValidationTier::Failed,
                            tolerance: "solver must succeed".to_string(),
                            detail: "SQP solve failed".to_string(),
                        },
                        solver_thresholds: Some(solver_thresholds),
                        error: Some(err.to_string()),
                        compile_report: compile_report.clone(),
                        console_output: None,
                        console_output_path: None,
                        filter_replay: None,
                    };
                    promote_solve_error_if_reduced_accuracy(&mut record, REDUCED_TERMINATION_TOL);
                    record.console_output =
                        Some(render_sqp_transcript(&record, &snapshots, None, Some(&err)));
                    record.filter_replay = sqp_filter_replay_from_snapshots(&snapshots);
                    record
                }
            }
        }
        SolverKind::Nlip => {
            let nlip_options = InteriorPointOptions {
                max_iters: max_iters_limit,
                dual_tol: STRICT_TERMINATION_TOL,
                constraint_tol: STRICT_TERMINATION_TOL,
                complementarity_tol: STRICT_TERMINATION_TOL,
                verbose: false,
                ..InteriorPointOptions::default()
            };
            let solver_thresholds = format_nlip_thresholds(&nlip_options);
            let solve_started = Instant::now();
            let mut snapshots = Vec::new();
            let result = data.compiled.solve_interior_point_with_callback(
                &data.x0,
                &data.parameters,
                &data.bounds,
                &nlip_options,
                |snapshot: &InteriorPointIterationSnapshot| snapshots.push(snapshot.clone()),
            );
            let solve_wall_time = solve_started.elapsed();
            match result {
                Ok(summary) => {
                    let metrics = metrics_from_nlip_summary(&summary);
                    let mut record = ProblemRunRecord {
                        id: metadata.id.to_string(),
                        solver,
                        options,
                        expected,
                        max_iters_limit,
                        status: RunStatus::Passed,
                        descriptor,
                        solution: Some(summary.x.clone()),
                        metrics,
                        timing: timing_breakdown(
                            backend_timing,
                            compile_wall_time,
                            Some(summary.profiling.total_time.max(solve_wall_time)),
                            total_started.elapsed(),
                        ),
                        validation: ValidationOutcome::default(),
                        solver_thresholds: Some(solver_thresholds.clone()),
                        error: None,
                        compile_report: compile_report.clone(),
                        console_output: None,
                        console_output_path: None,
                        filter_replay: None,
                    };
                    record.validation = validate(&record);
                    if !record.validation.passed() {
                        record.status = RunStatus::FailedValidation;
                    } else if matches!(record.validation.tier, ValidationTier::ReducedAccuracy) {
                        record.status = RunStatus::ReducedAccuracy;
                    }
                    record.console_output = Some(render_nlip_transcript(
                        &record,
                        &snapshots,
                        Some(&summary),
                        None,
                    ));
                    record.filter_replay = nlip_filter_replay_from_snapshots(&snapshots);
                    record
                }
                Err(err) => {
                    let mut record = ProblemRunRecord {
                        id: metadata.id.to_string(),
                        solver,
                        options,
                        expected,
                        max_iters_limit,
                        status: RunStatus::SolveError,
                        descriptor,
                        solution: None,
                        metrics: metrics_from_ip_error(&err),
                        timing: timing_breakdown(
                            backend_timing,
                            compile_wall_time,
                            Some(solve_wall_time),
                            total_started.elapsed(),
                        ),
                        validation: ValidationOutcome {
                            tier: ValidationTier::Failed,
                            tolerance: "solver must succeed".to_string(),
                            detail: "NLIP solve failed".to_string(),
                        },
                        solver_thresholds: Some(solver_thresholds),
                        error: Some(err.to_string()),
                        compile_report: compile_report.clone(),
                        console_output: None,
                        console_output_path: None,
                        filter_replay: None,
                    };
                    promote_solve_error_if_reduced_accuracy(&mut record, REDUCED_TERMINATION_TOL);
                    record.console_output = Some(render_nlip_transcript(
                        &record,
                        &snapshots,
                        None,
                        Some(&err),
                    ));
                    record.filter_replay = nlip_filter_replay_from_snapshots(&snapshots);
                    record
                }
            }
        }
        #[cfg(feature = "ipopt")]
        SolverKind::Ipopt => {
            let ipopt_options = IpoptOptions {
                max_iters: max_iters_limit,
                tol: STRICT_TERMINATION_TOL,
                acceptable_tol: Some(REDUCED_TERMINATION_TOL),
                constraint_tol: Some(STRICT_TERMINATION_TOL),
                complementarity_tol: Some(STRICT_TERMINATION_TOL),
                dual_tol: Some(STRICT_TERMINATION_TOL),
                print_level: 0,
                suppress_banner: true,
                ..IpoptOptions::default()
            };
            let solver_thresholds = format_ipopt_thresholds(&ipopt_options);
            let solve_started = Instant::now();
            let result =
                data.compiled
                    .solve_ipopt(&data.x0, &data.parameters, &data.bounds, &ipopt_options);
            let solve_wall_time = solve_started.elapsed();
            match result {
                Ok(summary) => {
                    let metrics = metrics_from_ipopt_summary(&summary);
                    let mut record = ProblemRunRecord {
                        id: metadata.id.to_string(),
                        solver,
                        options,
                        expected,
                        max_iters_limit,
                        status: RunStatus::Passed,
                        descriptor,
                        solution: Some(summary.x.clone()),
                        metrics,
                        timing: timing_breakdown(
                            backend_timing,
                            compile_wall_time,
                            Some(solve_wall_time),
                            total_started.elapsed(),
                        ),
                        validation: ValidationOutcome::default(),
                        solver_thresholds: Some(solver_thresholds.clone()),
                        error: None,
                        compile_report: compile_report.clone(),
                        console_output: None,
                        console_output_path: None,
                        filter_replay: None,
                    };
                    record.validation = validate(&record);
                    if !record.validation.passed() {
                        record.status = RunStatus::FailedValidation;
                    } else if matches!(record.validation.tier, ValidationTier::ReducedAccuracy) {
                        record.status = RunStatus::ReducedAccuracy;
                    }
                    record.console_output =
                        Some(render_ipopt_transcript(&record, Some(&summary), None));
                    record
                }
                Err(err) => {
                    let mut record = ProblemRunRecord {
                        id: metadata.id.to_string(),
                        solver,
                        options,
                        expected,
                        max_iters_limit,
                        status: RunStatus::SolveError,
                        descriptor,
                        solution: None,
                        metrics: metrics_from_ipopt_error(&err),
                        timing: timing_breakdown(
                            backend_timing,
                            compile_wall_time,
                            Some(solve_wall_time),
                            total_started.elapsed(),
                        ),
                        validation: ValidationOutcome {
                            tier: ValidationTier::Failed,
                            tolerance: "solver must succeed".to_string(),
                            detail: "ipopt solve failed".to_string(),
                        },
                        solver_thresholds: Some(solver_thresholds),
                        error: Some(err.to_string()),
                        compile_report: compile_report.clone(),
                        console_output: None,
                        console_output_path: None,
                        filter_replay: None,
                    };
                    promote_solve_error_if_reduced_accuracy(&mut record, REDUCED_TERMINATION_TOL);
                    record.console_output =
                        Some(render_ipopt_transcript(&record, None, Some(&err)));
                    record
                }
            }
        }
    }
}

fn timing_breakdown(
    backend_timing: optimization::BackendTimingMetadata,
    compile_wall_time: std::time::Duration,
    solve_time: Option<std::time::Duration>,
    total_wall_time: std::time::Duration,
) -> SolverTimingBreakdown {
    SolverTimingBreakdown {
        function_creation_time: backend_timing.function_creation_time,
        derivative_generation_time: backend_timing.derivative_generation_time,
        jit_time: backend_timing.jit_time,
        compile_wall_time: Some(compile_wall_time),
        solve_time,
        total_wall_time,
    }
}

fn summarize_backend_compile_report(
    report: Option<&optimization::BackendCompileReport>,
) -> Option<CompileReportSummary> {
    report.map(|report| CompileReportSummary {
        setup: SetupProfileBreakdown {
            symbolic_construction_s: duration_seconds(report.setup_profile.symbolic_construction),
            objective_gradient_s: duration_seconds(report.setup_profile.objective_gradient),
            equality_jacobian_s: duration_seconds(report.setup_profile.equality_jacobian),
            inequality_jacobian_s: duration_seconds(report.setup_profile.inequality_jacobian),
            lagrangian_assembly_s: duration_seconds(report.setup_profile.lagrangian_assembly),
            hessian_generation_s: duration_seconds(report.setup_profile.hessian_generation),
            lowering_s: duration_seconds(report.setup_profile.lowering),
            llvm_jit_s: duration_seconds(report.setup_profile.llvm_jit),
        },
        stats: CompileStatsSummary {
            symbolic_function_count: report.stats.symbolic_function_count,
            call_site_count: report.stats.call_site_count,
            max_call_depth: report.stats.max_call_depth,
            inline_at_call_policy_count: report.stats.inline_at_call_policy_count,
            inline_at_lowering_policy_count: report.stats.inline_at_lowering_policy_count,
            inline_in_llvm_policy_count: report.stats.inline_in_llvm_policy_count,
            no_inline_llvm_policy_count: report.stats.no_inline_llvm_policy_count,
            overrides_applied: report.stats.overrides_applied,
            overrides_ignored: report.stats.overrides_ignored,
            inlines_at_call: report.stats.inlines_at_call,
            inlines_at_lowering: report.stats.inlines_at_lowering,
            llvm_subfunctions_emitted: report.stats.llvm_subfunctions_emitted,
            llvm_call_instructions_emitted: report.stats.llvm_call_instructions_emitted,
        },
        warnings: report
            .warnings
            .iter()
            .map(|warning| warning.message.clone())
            .collect(),
    })
}

fn duration_seconds(duration: Option<std::time::Duration>) -> Option<f64> {
    duration.map(|duration| duration.as_secs_f64())
}

fn format_sqp_thresholds(options: &ClarabelSqpOptions) -> String {
    format!(
        "strict: primal<={:.1e}, dual<={:.1e}, comp<={:.1e}; reduced: primal<={:.1e}, dual<={:.1e}, comp<={:.1e}",
        options.constraint_tol,
        options.dual_tol,
        options.complementarity_tol,
        REDUCED_TERMINATION_TOL,
        REDUCED_TERMINATION_TOL,
        REDUCED_TERMINATION_TOL,
    )
}

fn format_nlip_thresholds(options: &InteriorPointOptions) -> String {
    format!(
        "strict: primal<={:.1e}, dual<={:.1e}, comp<={:.1e}; reduced: primal<={:.1e}, dual<={:.1e}, comp<={:.1e}",
        options.constraint_tol,
        options.dual_tol,
        options.complementarity_tol,
        REDUCED_TERMINATION_TOL,
        REDUCED_TERMINATION_TOL,
        REDUCED_TERMINATION_TOL,
    )
}

#[cfg(feature = "ipopt")]
fn format_ipopt_thresholds(options: &IpoptOptions) -> String {
    format!(
        "strict: tol={:.1e}, primal<={:.1e}, dual<={:.1e}, comp<={:.1e}; reduced: tol<={}",
        options.tol,
        options.constraint_tol.unwrap_or(options.tol),
        options.dual_tol.unwrap_or(options.tol),
        options.complementarity_tol.unwrap_or(options.tol),
        options
            .acceptable_tol
            .map(|value| format!("{value:.1e}"))
            .unwrap_or_else(|| "--".to_string()),
    )
}

fn describe_problem<X, I>(
    metadata: CaseMetadata,
    problem: &impl CompiledNlpProblem,
    bounds: &TypedRuntimeNlpBounds<X, I>,
) -> ProblemDescriptor
where
    X: Vectorize<SX, Rebind<SX> = X>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
{
    let num_vars = problem.dimension();
    let num_eq = problem.equality_count();
    let num_ineq = problem.inequality_count();
    let (num_box, fixed_box) = box_counts(bounds);
    let dof = num_vars.saturating_sub(num_eq + fixed_box);
    ProblemDescriptor {
        id: metadata.id.to_string(),
        family: metadata.family.to_string(),
        variant: metadata.variant.to_string(),
        source: metadata.source.to_string(),
        description: metadata.description.to_string(),
        parameterized: metadata.parameterized,
        num_vars,
        num_eq,
        num_ineq,
        num_box,
        dof,
        constrained: num_eq + num_ineq + num_box > 0,
    }
}

fn box_counts<X, I>(bounds: &TypedRuntimeNlpBounds<X, I>) -> (usize, usize)
where
    X: Vectorize<SX, Rebind<SX> = X>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
{
    let lower = bounds
        .variable_lower
        .as_ref()
        .map(optimization::flatten_value);
    let upper = bounds
        .variable_upper
        .as_ref()
        .map(optimization::flatten_value);
    let len = lower
        .as_ref()
        .map_or_else(|| upper.as_ref().map_or(0, Vec::len), Vec::len);
    let mut num_box = 0;
    let mut fixed_box = 0;
    for idx in 0..len {
        let lower_value = lower
            .as_ref()
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let upper_value = upper
            .as_ref()
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or(f64::INFINITY);
        if lower_value.is_finite() {
            num_box += 1;
        }
        if upper_value.is_finite() {
            num_box += 1;
        }
        if lower_value.is_finite() && upper_value.is_finite() && lower_value == upper_value {
            fixed_box += 1;
        }
    }
    (num_box, fixed_box)
}

fn metrics_from_sqp_summary(summary: &ClarabelSqpSummary) -> SolverMetrics {
    SolverMetrics {
        iterations: Some(summary.iterations),
        objective: Some(summary.objective),
        equality_inf: summary.equality_inf_norm,
        inequality_inf: summary.inequality_inf_norm,
        primal_inf: Some(summary.primal_inf_norm),
        dual_inf: Some(summary.dual_inf_norm),
        complementarity_inf: summary.complementarity_inf_norm,
        elastic_recovery_activations: Some(summary.profiling.elastic_recovery_activations),
        elastic_recovery_qp_solves: Some(summary.profiling.elastic_recovery_qp_solves),
    }
}

fn filter_acceptance_mode_label(mode: FilterAcceptanceMode) -> String {
    match mode {
        FilterAcceptanceMode::ObjectiveArmijo => "objective_armijo".to_string(),
        FilterAcceptanceMode::ViolationReduction => "violation_reduction".to_string(),
    }
}

fn sqp_filter_replay_from_snapshots(snapshots: &[SqpIterationSnapshot]) -> Option<FilterReplay> {
    let frames = snapshots
        .iter()
        .filter_map(|snapshot| {
            let filter = snapshot.filter.as_ref()?;
            let phase = match snapshot.phase {
                optimization::SqpIterationPhase::Initial => "initial",
                optimization::SqpIterationPhase::AcceptedStep => "accepted_step",
                optimization::SqpIterationPhase::PostConvergence => "post_convergence",
            };
            let accepted_mode = filter.accepted_mode.map(filter_acceptance_mode_label);
            Some(FilterReplayFrame {
                iteration: snapshot.iteration,
                phase: phase.to_string(),
                current: FilterReplayPoint {
                    violation: filter.current.violation,
                    objective: filter.current.objective,
                },
                frontier: filter
                    .entries
                    .iter()
                    .map(|entry| FilterReplayPoint {
                        violation: entry.violation,
                        objective: entry.objective,
                    })
                    .collect(),
                rejected_trials: snapshot
                    .line_search
                    .as_ref()
                    .map(|info| {
                        info.rejected_trials
                            .iter()
                            .map(|trial| FilterReplayPoint {
                                violation: trial.primal_inf,
                                objective: trial.objective,
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                accepted_mode,
            })
        })
        .collect::<Vec<_>>();
    (!frames.is_empty()).then_some(FilterReplay { frames })
}

fn nlip_filter_replay_from_snapshots(
    snapshots: &[InteriorPointIterationSnapshot],
) -> Option<FilterReplay> {
    let frames = snapshots
        .iter()
        .filter_map(|snapshot| {
            let filter = snapshot.filter.as_ref()?;
            let phase = match snapshot.phase {
                optimization::InteriorPointIterationPhase::Initial => "initial",
                optimization::InteriorPointIterationPhase::AcceptedStep => "accepted_step",
                optimization::InteriorPointIterationPhase::Converged => "converged",
            };
            Some(FilterReplayFrame {
                iteration: snapshot.iteration,
                phase: phase.to_string(),
                current: FilterReplayPoint {
                    violation: filter.current.violation,
                    objective: filter.current.objective,
                },
                frontier: filter
                    .entries
                    .iter()
                    .map(|entry| FilterReplayPoint {
                        violation: entry.violation,
                        objective: entry.objective,
                    })
                    .collect(),
                rejected_trials: Vec::new(),
                accepted_mode: filter.accepted_mode.map(filter_acceptance_mode_label),
            })
        })
        .collect::<Vec<_>>();
    (!frames.is_empty()).then_some(FilterReplay { frames })
}

fn metrics_from_sqp_error(error: &ClarabelSqpError) -> SolverMetrics {
    let context = match error {
        ClarabelSqpError::MaxIterations { context, .. }
        | ClarabelSqpError::QpSolve { context, .. }
        | ClarabelSqpError::UnconstrainedStepSolve { context }
        | ClarabelSqpError::LineSearchFailed { context, .. }
        | ClarabelSqpError::RestorationFailed { context, .. }
        | ClarabelSqpError::Stalled { context, .. }
        | ClarabelSqpError::NonFiniteCallbackOutput { context, .. } => Some(context.as_ref()),
        ClarabelSqpError::InvalidInput(_)
        | ClarabelSqpError::NonFiniteInput { .. }
        | ClarabelSqpError::Setup(_) => None,
    };
    let Some(context) = context else {
        return SolverMetrics::default();
    };
    let snapshot = context
        .final_state
        .as_ref()
        .or(context.last_accepted_state.as_ref());
    SolverMetrics {
        iterations: snapshot.map(|state| state.iteration),
        objective: snapshot.map(|state| state.objective),
        equality_inf: snapshot.and_then(|state| state.eq_inf),
        inequality_inf: snapshot.and_then(|state| state.ineq_inf),
        primal_inf: snapshot.map(|state| {
            state
                .eq_inf
                .into_iter()
                .chain(state.ineq_inf)
                .fold(0.0_f64, f64::max)
        }),
        dual_inf: snapshot.map(|state| state.dual_inf),
        complementarity_inf: snapshot.and_then(|state| state.comp_inf),
        elastic_recovery_activations: Some(context.profiling.elastic_recovery_activations),
        elastic_recovery_qp_solves: Some(context.profiling.elastic_recovery_qp_solves),
    }
}

fn solve_time_from_sqp_error(error: &ClarabelSqpError) -> Option<std::time::Duration> {
    let context = match error {
        ClarabelSqpError::MaxIterations { context, .. }
        | ClarabelSqpError::QpSolve { context, .. }
        | ClarabelSqpError::UnconstrainedStepSolve { context }
        | ClarabelSqpError::LineSearchFailed { context, .. }
        | ClarabelSqpError::RestorationFailed { context, .. }
        | ClarabelSqpError::Stalled { context, .. }
        | ClarabelSqpError::NonFiniteCallbackOutput { context, .. } => Some(context.as_ref()),
        ClarabelSqpError::InvalidInput(_)
        | ClarabelSqpError::NonFiniteInput { .. }
        | ClarabelSqpError::Setup(_) => None,
    };
    context.map(|ctx| ctx.profiling.total_time)
}

fn metrics_from_nlip_summary(summary: &InteriorPointSummary) -> SolverMetrics {
    SolverMetrics {
        iterations: Some(summary.iterations),
        objective: Some(summary.objective),
        equality_inf: Some(summary.equality_inf_norm),
        inequality_inf: Some(summary.inequality_inf_norm),
        primal_inf: Some(summary.primal_inf_norm),
        dual_inf: Some(summary.dual_inf_norm),
        complementarity_inf: Some(summary.complementarity_inf_norm),
        ..SolverMetrics::default()
    }
}

fn metrics_from_ip_error(error: &InteriorPointSolveError) -> SolverMetrics {
    match error {
        InteriorPointSolveError::MaxIterations { iterations, .. } => SolverMetrics {
            iterations: Some(*iterations),
            ..SolverMetrics::default()
        },
        InteriorPointSolveError::InvalidInput(_)
        | InteriorPointSolveError::LinearSolve { .. }
        | InteriorPointSolveError::LineSearchFailed { .. } => SolverMetrics::default(),
    }
}

#[cfg(feature = "ipopt")]
fn metrics_from_ipopt_summary(summary: &IpoptSummary) -> SolverMetrics {
    SolverMetrics {
        iterations: Some(summary.iterations),
        objective: Some(summary.objective),
        equality_inf: Some(summary.equality_inf_norm),
        inequality_inf: Some(summary.inequality_inf_norm),
        primal_inf: Some(summary.primal_inf_norm),
        dual_inf: Some(summary.dual_inf_norm),
        complementarity_inf: Some(summary.complementarity_inf_norm),
        ..SolverMetrics::default()
    }
}

#[cfg(feature = "ipopt")]
fn metrics_from_ipopt_error(error: &IpoptSolveError) -> SolverMetrics {
    match error {
        IpoptSolveError::Solve {
            iterations,
            snapshots,
            ..
        } => {
            let last = snapshots.last();
            SolverMetrics {
                iterations: Some(*iterations),
                objective: last.map(|snapshot| snapshot.objective),
                equality_inf: None,
                inequality_inf: None,
                primal_inf: last.map(|snapshot| snapshot.primal_inf),
                dual_inf: last.map(|snapshot| snapshot.dual_inf),
                complementarity_inf: None,
                ..SolverMetrics::default()
            }
        }
        IpoptSolveError::InvalidInput(_)
        | IpoptSolveError::Setup(_)
        | IpoptSolveError::OptionRejected { .. } => SolverMetrics::default(),
    }
}

fn render_problem_header(record: &ProblemRunRecord) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "ad_codegen_rs test problem transcript");
    let _ = writeln!(out, "problem: {}", record.id);
    let _ = writeln!(
        out,
        "family: {} / {}",
        record.descriptor.family, record.descriptor.variant
    );
    let _ = writeln!(
        out,
        "solver: {} / {}",
        record.solver.label(),
        record.options.label()
    );
    let _ = writeln!(
        out,
        "expected: {}",
        match record.expected {
            KnownStatus::KnownPassing => "known_passing",
            KnownStatus::KnownFailing => "known_failing",
            KnownStatus::Skipped => "skipped",
        }
    );
    let _ = writeln!(
        out,
        "dims: vars={} dof={} eq={} ineq={} box={}",
        record.descriptor.num_vars,
        record.descriptor.dof,
        record.descriptor.num_eq,
        record.descriptor.num_ineq,
        record.descriptor.num_box
    );
    let _ = writeln!(out, "max_iters: {}", record.max_iters_limit);
    if let Some(thresholds) = &record.solver_thresholds {
        let _ = writeln!(out, "termination_thresholds: {thresholds}");
    }
    let _ = writeln!(
        out,
        "validation_thresholds: {}",
        record.validation.tolerance
    );
    out.push('\n');
    out
}

fn render_problem_footer(record: &ProblemRunRecord) -> String {
    let mut out = String::new();
    out.push_str("\nresult\n");
    let _ = writeln!(out, "  status: {}", status_text(record.status));
    let _ = writeln!(
        out,
        "  iterations: {}",
        fmt_opt_usize(record.metrics.iterations)
    );
    let _ = writeln!(
        out,
        "  total_time: {}",
        crate::report::format_duration(record.timing.total_wall_time)
    );
    let _ = writeln!(
        out,
        "  objective: {}",
        fmt_opt_sci(record.metrics.objective)
    );
    let _ = writeln!(out, "  primal: {}", fmt_opt_sci(record.metrics.primal_inf));
    let _ = writeln!(out, "  dual: {}", fmt_opt_sci(record.metrics.dual_inf));
    let _ = writeln!(
        out,
        "  complementarity: {}",
        fmt_opt_sci(record.metrics.complementarity_inf)
    );
    let _ = writeln!(
        out,
        "  emergency_restorations: {}",
        fmt_elastic_stats(
            record.metrics.elastic_recovery_activations,
            record.metrics.elastic_recovery_qp_solves,
        )
    );
    let _ = writeln!(
        out,
        "  termination_thresholds: {}",
        record.solver_thresholds.as_deref().unwrap_or("--")
    );
    let _ = writeln!(
        out,
        "  validation_thresholds: {}",
        record.validation.tolerance
    );
    let _ = writeln!(out, "  validation: {}", record.validation.detail);
    if let Some(error) = &record.error {
        let _ = writeln!(out, "  error: {error}");
    }
    out
}

fn render_sqp_transcript(
    record: &ProblemRunRecord,
    snapshots: &[SqpIterationSnapshot],
    summary: Option<&ClarabelSqpSummary>,
    error: Option<&ClarabelSqpError>,
) -> String {
    fn fmt_bool(value: bool) -> &'static str {
        if value { "yes" } else { "no" }
    }

    fn fmt_opt_bool(value: Option<bool>) -> &'static str {
        match value {
            Some(true) => "yes",
            Some(false) => "no",
            None => "--",
        }
    }

    let mut out = render_problem_header(record);
    out.push_str("solver_log\n\n");
    let header = format!(
        "{:>4}  {:<6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>7}  {:>5}  {:>5}  {:>5}",
        "iter",
        "phase",
        "f",
        "eq_inf",
        "ineq_inf",
        "dual_inf",
        "comp_inf",
        "step_inf",
        "penalty",
        "alpha",
        "ls",
        "qp",
        "evt"
    );
    write_repeated_header(&mut out, &header);
    for (idx, snapshot) in snapshots.iter().enumerate() {
        if idx > 0 && idx.is_multiple_of(10) {
            out.push('\n');
            write_repeated_header(&mut out, &header);
        }
        let phase = match snapshot.phase {
            optimization::SqpIterationPhase::Initial => "initial",
            optimization::SqpIterationPhase::AcceptedStep => "accept",
            optimization::SqpIterationPhase::PostConvergence => "final",
        };
        let qp = snapshot.qp.as_ref().map_or_else(
            || "--".to_string(),
            |qp| match qp.status {
                optimization::SqpQpStatus::Solved => format!("{}", qp.iteration_count),
                optimization::SqpQpStatus::ReducedAccuracy => format!("{}R", qp.iteration_count),
                optimization::SqpQpStatus::Failed => "fail".to_string(),
            },
        );
        let events = if snapshot.events.is_empty() {
            let mut codes = String::new();
            if snapshot
                .line_search
                .as_ref()
                .is_some_and(|info| info.second_order_correction_attempted)
            {
                codes.push('s');
            }
            if snapshot
                .line_search
                .as_ref()
                .is_some_and(|info| info.restoration_attempted)
            {
                codes.push('t');
            }
            if snapshot
                .line_search
                .as_ref()
                .is_some_and(|info| info.elastic_recovery_attempted)
            {
                codes.push('e');
            }
            if codes.is_empty() {
                "--".to_string()
            } else {
                codes
            }
        } else {
            let mut codes = snapshot
                .events
                .iter()
                .map(|event| match event {
                    optimization::SqpIterationEvent::PenaltyUpdated => 'P',
                    optimization::SqpIterationEvent::HessianShifted => 'H',
                    optimization::SqpIterationEvent::LongLineSearch => 'L',
                    optimization::SqpIterationEvent::ArmijoToleranceAdjusted => 'A',
                    optimization::SqpIterationEvent::SecondOrderCorrectionUsed => 'S',
                    optimization::SqpIterationEvent::FilterAccepted => 'F',
                    optimization::SqpIterationEvent::RestorationStepAccepted => 'T',
                    optimization::SqpIterationEvent::QpReducedAccuracy => 'R',
                    optimization::SqpIterationEvent::ElasticRecoveryUsed => 'E',
                    optimization::SqpIterationEvent::WolfeRejectedTrial => 'W',
                    optimization::SqpIterationEvent::MaxIterationsReached => 'M',
                })
                .collect::<String>();
            if !snapshot
                .events
                .contains(&optimization::SqpIterationEvent::SecondOrderCorrectionUsed)
                && snapshot
                    .line_search
                    .as_ref()
                    .is_some_and(|info| info.second_order_correction_attempted)
            {
                codes.push('s');
            }
            if !snapshot
                .events
                .contains(&optimization::SqpIterationEvent::RestorationStepAccepted)
                && snapshot
                    .line_search
                    .as_ref()
                    .is_some_and(|info| info.restoration_attempted)
            {
                codes.push('t');
            }
            if !snapshot
                .events
                .contains(&optimization::SqpIterationEvent::ElasticRecoveryUsed)
                && snapshot
                    .line_search
                    .as_ref()
                    .is_some_and(|info| info.elastic_recovery_attempted)
            {
                codes.push('e');
            }
            codes
        };
        let _ = writeln!(
            out,
            "{:>4}  {:<6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>7}  {:>5}  {:>5}  {:>5}",
            snapshot.iteration,
            phase,
            fmt_sci(snapshot.objective),
            fmt_opt_sci(snapshot.eq_inf),
            fmt_opt_sci(snapshot.ineq_inf),
            fmt_sci(snapshot.dual_inf),
            fmt_opt_sci(snapshot.comp_inf),
            fmt_opt_sci(snapshot.step_inf),
            fmt_sci(snapshot.penalty),
            snapshot
                .line_search
                .as_ref()
                .map_or_else(|| "--".to_string(), |ls| fmt_sci(ls.accepted_alpha)),
            snapshot
                .line_search
                .as_ref()
                .map_or_else(|| "--".to_string(), |ls| ls.backtrack_count.to_string()),
            qp,
            events,
        );
    }
    let detailed_line_search = snapshots
        .iter()
        .filter_map(|snapshot| {
            snapshot
                .line_search
                .as_ref()
                .filter(|info| !info.rejected_trials.is_empty() || info.wolfe_satisfied.is_some())
                .map(|info| (snapshot, info))
        })
        .collect::<Vec<_>>();
    if !detailed_line_search.is_empty() {
        out.push_str("\nline_search_detail\n\n");
        for (snapshot, info) in detailed_line_search {
            let _ = writeln!(
                out,
                "iter {}: accepted alpha={} armijo={} armijo_adj={} obj_armijo={} obj_armijo_adj={} soc={} wolfe={} violation={} filter_mode={} filter_ok={} filter_dom={} filter_obj={} filter_violation={} rejected={}",
                snapshot.iteration,
                fmt_sci(info.accepted_alpha),
                fmt_bool(info.armijo_satisfied),
                fmt_bool(info.armijo_tolerance_adjusted),
                fmt_opt_bool(info.objective_armijo_satisfied),
                fmt_opt_bool(info.objective_armijo_tolerance_adjusted),
                fmt_bool(info.second_order_correction_used),
                fmt_opt_bool(info.wolfe_satisfied),
                fmt_bool(info.violation_satisfied),
                info.filter_acceptance_mode.map_or("--", |mode| match mode {
                    optimization::SqpFilterAcceptanceMode::ObjectiveArmijo => "objective",
                    optimization::SqpFilterAcceptanceMode::ViolationReduction => "violation",
                }),
                fmt_opt_bool(info.filter_acceptable),
                fmt_opt_bool(info.filter_dominated),
                fmt_opt_bool(info.filter_sufficient_objective_reduction),
                fmt_opt_bool(info.filter_sufficient_violation_reduction),
                info.rejected_trials.len(),
            );
            for trial in &info.rejected_trials {
                let _ = writeln!(
                    out,
                    "  reject alpha={} merit={} obj={} primal={} eq_inf={} ineq_inf={} armijo={} armijo_adj={} obj_armijo={} obj_armijo_adj={} wolfe={} violation={} filter_ok={} filter_dom={} filter_obj={} filter_violation={}",
                    fmt_sci(trial.alpha),
                    fmt_sci(trial.merit),
                    fmt_sci(trial.objective),
                    fmt_sci(trial.primal_inf),
                    fmt_opt_sci(trial.eq_inf),
                    fmt_opt_sci(trial.ineq_inf),
                    fmt_bool(trial.armijo_satisfied),
                    fmt_bool(trial.armijo_tolerance_adjusted),
                    fmt_opt_bool(trial.objective_armijo_satisfied),
                    fmt_opt_bool(trial.objective_armijo_tolerance_adjusted),
                    fmt_opt_bool(trial.wolfe_satisfied),
                    fmt_bool(trial.violation_satisfied),
                    fmt_opt_bool(trial.filter_acceptable),
                    fmt_opt_bool(trial.filter_dominated),
                    fmt_opt_bool(trial.filter_sufficient_objective_reduction),
                    fmt_opt_bool(trial.filter_sufficient_violation_reduction),
                );
            }
        }
    }
    if let Some(summary) = summary {
        let _ = writeln!(out, "\ntermination: {:?}", summary.termination);
        let _ = writeln!(out, "final_state_kind: {:?}", summary.final_state_kind);
    }
    if let Some(error) = error {
        let _ = writeln!(out, "\ntermination: error");
        let _ = writeln!(out, "error_kind: {}", sqp_error_code(error));
        if let ClarabelSqpError::QpSolve { context, .. } = error
            && let Some(qp_failure) = &context.qp_failure
        {
            out.push_str("\nqp_failure\n\n");
            let _ = writeln!(out, "status: {}", qp_failure.qp_info.raw_status);
            let _ = writeln!(
                out,
                "dims: vars={} constraints={}",
                qp_failure.variable_count, qp_failure.constraint_count
            );
            let _ = writeln!(
                out,
                "objective_inf={} rhs_inf={}",
                fmt_sci(qp_failure.linear_objective_inf_norm),
                fmt_sci(qp_failure.rhs_inf_norm),
            );
            let _ = writeln!(
                out,
                "hessian_diag=[{}, {}] elastic_recovery={}",
                fmt_sci(qp_failure.hessian_diag_min),
                fmt_sci(qp_failure.hessian_diag_max),
                fmt_bool(qp_failure.elastic_recovery),
            );
            let cones = qp_failure
                .cones
                .iter()
                .map(|cone| format!("{}:{}", cone.kind, cone.dim))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "cones: {}",
                if cones.is_empty() { "--" } else { &cones }
            );
            if let Some(transcript) = &qp_failure.transcript {
                let _ = writeln!(out, "\nclarabel_qp_transcript\n");
                out.push_str(transcript);
                if !transcript.ends_with('\n') {
                    out.push('\n');
                }
            }
        }
    }
    out.push_str(&render_problem_footer(record));
    out
}

fn render_nlip_transcript(
    record: &ProblemRunRecord,
    snapshots: &[InteriorPointIterationSnapshot],
    summary: Option<&InteriorPointSummary>,
    error: Option<&InteriorPointSolveError>,
) -> String {
    let mut out = render_problem_header(record);
    out.push_str("solver_log\n\n");
    let header = format!(
        "{:>4}  {:<7}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>7}  {:>5}  {:>8}  {:>8}  {:>5}",
        "iter",
        "phase",
        "f",
        "eq_inf",
        "ineq_inf",
        "dual_inf",
        "comp_inf",
        "mu",
        "alpha",
        "ls",
        "linear",
        "lin_t",
        "evt"
    );
    write_repeated_header(&mut out, &header);
    for (idx, snapshot) in snapshots.iter().enumerate() {
        if idx > 0 && idx.is_multiple_of(10) {
            out.push('\n');
            write_repeated_header(&mut out, &header);
        }
        let phase = match snapshot.phase {
            optimization::InteriorPointIterationPhase::Initial => "start",
            optimization::InteriorPointIterationPhase::AcceptedStep => "accept",
            optimization::InteriorPointIterationPhase::Converged => "final",
        };
        let events = if snapshot.events.is_empty() {
            "--".to_string()
        } else {
            snapshot
                .events
                .iter()
                .map(|event| match event {
                    optimization::InteriorPointIterationEvent::SigmaAdjusted => 'P',
                    optimization::InteriorPointIterationEvent::LongLineSearch => 'L',
                    optimization::InteriorPointIterationEvent::FilterAccepted => 'F',
                    optimization::InteriorPointIterationEvent::LinearSolverFallback => 'U',
                    optimization::InteriorPointIterationEvent::MaxIterationsReached => 'M',
                })
                .collect()
        };
        let _ = writeln!(
            out,
            "{:>4}  {:<7}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>7}  {:>5}  {:>8}  {:>8}  {:>5}",
            snapshot.iteration,
            phase,
            fmt_sci(snapshot.objective),
            fmt_opt_sci(snapshot.eq_inf),
            fmt_opt_sci(snapshot.ineq_inf),
            fmt_sci(snapshot.dual_inf),
            fmt_opt_sci(snapshot.comp_inf),
            fmt_opt_sci(snapshot.barrier_parameter),
            fmt_opt_sci(snapshot.alpha),
            fmt_opt_usize(snapshot.line_search_iterations),
            snapshot.linear_solver.label(),
            snapshot
                .linear_solve_time
                .map_or_else(|| "--".to_string(), crate::report::format_duration),
            events,
        );
    }
    if let Some(summary) = summary {
        let _ = writeln!(out, "\ntermination: converged");
        let _ = writeln!(out, "linear_solver: {}", summary.linear_solver.label());
    }
    if let Some(error) = error {
        let _ = writeln!(out, "\ntermination: error");
        let _ = writeln!(out, "error_kind: {}", nlip_error_code(error));
    }
    out.push_str(&render_problem_footer(record));
    out
}

#[cfg(feature = "ipopt")]
fn render_ipopt_transcript(
    record: &ProblemRunRecord,
    summary: Option<&IpoptSummary>,
    error: Option<&IpoptSolveError>,
) -> String {
    let mut out = render_problem_header(record);
    out.push_str("solver_log\n\n");
    let snapshots = summary
        .map(|summary| summary.snapshots.as_slice())
        .or(match error {
            Some(IpoptSolveError::Solve { snapshots, .. }) => Some(snapshots.as_slice()),
            Some(IpoptSolveError::InvalidInput(_))
            | Some(IpoptSolveError::Setup(_))
            | Some(IpoptSolveError::OptionRejected { .. })
            | None => None,
        })
        .unwrap_or(&[]);
    let header = format!(
        "{:>4}  {:<7}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}  {:>8}  {:>5}",
        "iter",
        "phase",
        "f",
        "primal",
        "dual",
        "mu",
        "step_inf",
        "reg",
        "alpha_pr",
        "alpha_du",
        "ls",
    );
    write_repeated_header(&mut out, &header);
    for (idx, snapshot) in snapshots.iter().enumerate() {
        if idx > 0 && idx.is_multiple_of(10) {
            out.push('\n');
            write_repeated_header(&mut out, &header);
        }
        let phase = match snapshot.phase {
            optimization::IpoptIterationPhase::Regular => "regular",
            optimization::IpoptIterationPhase::Restoration => "restore",
        };
        let _ = writeln!(
            out,
            "{:>4}  {:<7}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}  {:>8}  {:>5}",
            snapshot.iteration,
            phase,
            fmt_sci(snapshot.objective),
            fmt_sci(snapshot.primal_inf),
            fmt_sci(snapshot.dual_inf),
            fmt_sci(snapshot.barrier_parameter),
            fmt_sci(snapshot.step_inf),
            fmt_sci(snapshot.regularization_size),
            fmt_sci(snapshot.alpha_pr),
            fmt_sci(snapshot.alpha_du),
            snapshot.line_search_trials,
        );
    }
    if let Some(summary) = summary {
        let _ = writeln!(out, "status: {}", summary.status);
        let _ = writeln!(out, "iterations: {}", summary.iterations);
        let _ = writeln!(out, "\ntermination: converged");
    }
    if let Some(error) = error {
        let _ = writeln!(out, "\ntermination: error");
        let _ = writeln!(out, "error_kind: {}", ipopt_error_code(error));
        match error {
            IpoptSolveError::Solve {
                status, iterations, ..
            } => {
                let _ = writeln!(out, "status: {status}");
                let _ = writeln!(out, "iterations: {iterations}");
            }
            IpoptSolveError::InvalidInput(message) | IpoptSolveError::Setup(message) => {
                let _ = writeln!(out, "status: error");
                let _ = writeln!(out, "message: {message}");
            }
            IpoptSolveError::OptionRejected { name } => {
                let _ = writeln!(out, "status: error");
                let _ = writeln!(out, "message: option rejected: {name}");
            }
        }
    }
    let journal_output = summary
        .and_then(|summary| summary.journal_output.as_deref())
        .or(match error {
            Some(IpoptSolveError::Solve { journal_output, .. }) => journal_output.as_deref(),
            Some(IpoptSolveError::InvalidInput(_))
            | Some(IpoptSolveError::Setup(_))
            | Some(IpoptSolveError::OptionRejected { .. })
            | None => None,
        });
    if let Some(journal_output) = journal_output {
        let _ = writeln!(out, "\nipopt_journal\n");
        out.push_str(journal_output);
        if !journal_output.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push_str(&render_problem_footer(record));
    out
}

fn write_repeated_header(out: &mut String, header: &str) {
    let _ = writeln!(out, "{header}");
}

#[cfg(feature = "ipopt")]
fn ipopt_error_code(error: &IpoptSolveError) -> &'static str {
    match error {
        IpoptSolveError::InvalidInput(_) => "invalid_input",
        IpoptSolveError::Setup(_) => "setup",
        IpoptSolveError::OptionRejected { .. } => "option_rejected",
        IpoptSolveError::Solve { status, .. } => match status {
            IpoptRawStatus::SolveSucceeded
            | IpoptRawStatus::SolvedToAcceptableLevel
            | IpoptRawStatus::FeasiblePointFound => "unexpected_success",
            IpoptRawStatus::InfeasibleProblemDetected => "infeasible_problem",
            IpoptRawStatus::SearchDirectionBecomesTooSmall => "search_direction_small",
            IpoptRawStatus::DivergingIterates => "diverging_iterates",
            IpoptRawStatus::UserRequestedStop => "user_stop",
            IpoptRawStatus::MaximumIterationsExceeded => "max_iters",
            IpoptRawStatus::MaximumCpuTimeExceeded => "max_cpu_time",
            IpoptRawStatus::RestorationFailed => "restoration_failed",
            IpoptRawStatus::ErrorInStepComputation => "step_computation",
            IpoptRawStatus::NotEnoughDegreesOfFreedom => "not_enough_dof",
            IpoptRawStatus::InvalidProblemDefinition => "invalid_problem",
            IpoptRawStatus::InvalidOption => "invalid_option",
            IpoptRawStatus::InvalidNumberDetected => "invalid_number",
            IpoptRawStatus::UnrecoverableException => "unrecoverable_exception",
            IpoptRawStatus::NonIpoptExceptionThrown => "non_ipopt_exception",
            IpoptRawStatus::InsufficientMemory => "insufficient_memory",
            IpoptRawStatus::InternalError => "internal_error",
            IpoptRawStatus::UnknownError => "unknown_error",
        },
    }
}

fn sqp_error_code(error: &ClarabelSqpError) -> &'static str {
    match error {
        ClarabelSqpError::InvalidInput(_) => "invalid_input",
        ClarabelSqpError::NonFiniteInput { .. } => "non_finite_input",
        ClarabelSqpError::MaxIterations { .. } => "max_iters",
        ClarabelSqpError::Setup(_) => "setup",
        ClarabelSqpError::QpSolve { .. } => "qp_solve",
        ClarabelSqpError::UnconstrainedStepSolve { .. } => "unconstrained_step",
        ClarabelSqpError::LineSearchFailed { .. } => "line_search",
        ClarabelSqpError::RestorationFailed { .. } => "restoration",
        ClarabelSqpError::Stalled { .. } => "stalled",
        ClarabelSqpError::NonFiniteCallbackOutput { .. } => "non_finite_callback",
    }
}

fn nlip_error_code(error: &InteriorPointSolveError) -> &'static str {
    match error {
        InteriorPointSolveError::InvalidInput(_) => "invalid_input",
        InteriorPointSolveError::LinearSolve { .. } => "linear_solve",
        InteriorPointSolveError::LineSearchFailed { .. } => "line_search",
        InteriorPointSolveError::MaxIterations { .. } => "max_iters",
    }
}

fn fmt_sci(value: f64) -> String {
    format!("{value:.3e}")
}

fn fmt_opt_sci(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), fmt_sci)
}

fn fmt_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "--".to_string(), |value| value.to_string())
}

fn fmt_elastic_stats(activations: Option<usize>, recovery_qps: Option<usize>) -> String {
    match (activations, recovery_qps) {
        (Some(activations), Some(recovery_qps)) => {
            format!("{activations} activations / {recovery_qps} recovery_qps")
        }
        _ => "--".to_string(),
    }
}

fn status_text(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Passed => "passed",
        RunStatus::ReducedAccuracy => "reduced_accuracy",
        RunStatus::FailedValidation => "failed_validation",
        RunStatus::SolveError => "solve_error",
        RunStatus::Skipped => "skipped",
    }
}

pub(crate) fn exact_solution_validation(
    expected_x: &[f64],
    x_tol: f64,
    expected_objective: f64,
    objective_tol: f64,
    primal_tol: f64,
    dual_tol: f64,
    complementarity_tol: Option<f64>,
) -> impl Fn(&ProblemRunRecord) -> ValidationOutcome + Clone + Send + Sync + 'static {
    let expected_x = expected_x.to_vec();
    move |record| {
        let Some(objective) = record.metrics.objective else {
            return ValidationOutcome {
                tier: ValidationTier::Failed,
                tolerance: tolerance_text(
                    x_tol,
                    objective_tol,
                    primal_tol,
                    dual_tol,
                    complementarity_tol,
                ),
                detail: "missing objective".to_string(),
            };
        };
        let mut passed = (objective - expected_objective).abs() <= objective_tol;
        let mut detail = format!("objective={objective:.6e}, expected={expected_objective:.6e}");
        if let Some(primal_inf) = record.metrics.primal_inf {
            passed &= primal_inf <= primal_tol;
            detail.push_str(&format!(", primal={primal_inf:.3e}"));
        }
        if let Some(dual_inf) = record.metrics.dual_inf {
            passed &= dual_inf <= dual_tol;
            detail.push_str(&format!(", dual={dual_inf:.3e}"));
        }
        if let (Some(limit), Some(comp_inf)) =
            (complementarity_tol, record.metrics.complementarity_inf)
        {
            passed &= comp_inf <= limit;
            detail.push_str(&format!(", comp={comp_inf:.3e}"));
        }
        if let Some(error) = &record.error {
            passed = false;
            detail.clone_from(error);
        }
        if let Some(solution) = extract_solution_slice(record) {
            for (actual, expected) in solution.iter().zip(expected_x.iter()) {
                if (actual - expected).abs() > x_tol {
                    passed = false;
                    detail.push_str(&format!(
                        ", solution_mismatch={actual:.6e} vs {expected:.6e}"
                    ));
                    break;
                }
            }
        }
        let tier = if passed {
            ValidationTier::Passed
        } else if record.error.is_none()
            && reduced_accuracy_residuals_met(record, REDUCED_TERMINATION_TOL)
        {
            detail.push_str(&format!(
                ", reduced_accuracy(primal/dual/comp<={:.1e})",
                REDUCED_TERMINATION_TOL
            ));
            ValidationTier::ReducedAccuracy
        } else {
            ValidationTier::Failed
        };
        ValidationOutcome {
            tier,
            tolerance: tolerance_text_with_reduced(
                x_tol,
                objective_tol,
                primal_tol,
                dual_tol,
                complementarity_tol,
                Some(REDUCED_TERMINATION_TOL),
            ),
            detail,
        }
    }
}

pub(crate) fn objective_validation(
    expected_objective: f64,
    objective_tol: f64,
    primal_tol: f64,
    dual_tol: f64,
    complementarity_tol: Option<f64>,
) -> impl Fn(&ProblemRunRecord) -> ValidationOutcome + Clone + Send + Sync + 'static {
    move |record| {
        let tolerance = tolerance_text_with_reduced(
            0.0,
            objective_tol,
            primal_tol,
            dual_tol,
            complementarity_tol,
            Some(REDUCED_TERMINATION_TOL),
        )
        .trim_start_matches("x<=0.0e0, ")
        .to_string();
        let Some(objective) = record.metrics.objective else {
            return ValidationOutcome {
                tier: ValidationTier::Failed,
                tolerance,
                detail: "missing objective".to_string(),
            };
        };
        let mut passed =
            objective.is_finite() && (objective - expected_objective).abs() <= objective_tol;
        let mut detail = format!("objective={objective:.6e}, expected={expected_objective:.6e}");
        if let Some(primal_inf) = record.metrics.primal_inf {
            passed &= primal_inf <= primal_tol;
            detail.push_str(&format!(", primal={primal_inf:.3e}"));
        }
        if let Some(dual_inf) = record.metrics.dual_inf {
            passed &= dual_inf <= dual_tol;
            detail.push_str(&format!(", dual={dual_inf:.3e}"));
        }
        if let (Some(limit), Some(comp_inf)) =
            (complementarity_tol, record.metrics.complementarity_inf)
        {
            passed &= comp_inf <= limit;
            detail.push_str(&format!(", comp={comp_inf:.3e}"));
        }
        if let Some(error) = &record.error {
            passed = false;
            detail.clone_from(error);
        }
        let tier = if passed {
            ValidationTier::Passed
        } else if record.error.is_none()
            && reduced_accuracy_residuals_met(record, REDUCED_TERMINATION_TOL)
        {
            detail.push_str(&format!(
                ", reduced_accuracy(primal/dual/comp<={:.1e})",
                REDUCED_TERMINATION_TOL
            ));
            ValidationTier::ReducedAccuracy
        } else {
            ValidationTier::Failed
        };
        ValidationOutcome {
            tier,
            tolerance,
            detail,
        }
    }
}

fn extract_solution_slice(record: &ProblemRunRecord) -> Option<&[f64]> {
    record.solution.as_deref()
}

fn tolerance_text(
    x_tol: f64,
    objective_tol: f64,
    primal_tol: f64,
    dual_tol: f64,
    complementarity_tol: Option<f64>,
) -> String {
    let mut text = format!(
        "x<={x_tol:.1e}, obj<={objective_tol:.1e}, primal<={primal_tol:.1e}, dual<={dual_tol:.1e}"
    );
    if let Some(comp_tol) = complementarity_tol {
        text.push_str(&format!(", comp<={comp_tol:.1e}"));
    }
    text
}

fn tolerance_text_with_reduced(
    x_tol: f64,
    objective_tol: f64,
    primal_tol: f64,
    dual_tol: f64,
    complementarity_tol: Option<f64>,
    reduced_tol: Option<f64>,
) -> String {
    let mut text = tolerance_text(
        x_tol,
        objective_tol,
        primal_tol,
        dual_tol,
        complementarity_tol,
    );
    if let Some(reduced_tol) = reduced_tol {
        text.push_str(&format!(
            "; reduced: primal<={reduced_tol:.1e}, dual<={reduced_tol:.1e}"
        ));
        if complementarity_tol.is_some() {
            text.push_str(&format!(", comp<={reduced_tol:.1e}"));
        }
    }
    text
}

fn reduced_accuracy_residuals_met(record: &ProblemRunRecord, tol: f64) -> bool {
    let objective_ok = record
        .metrics
        .objective
        .is_some_and(|value| value.is_finite());
    let primal_ok = record
        .metrics
        .primal_inf
        .is_some_and(|value| value.is_finite() && value <= tol);
    let dual_ok = record
        .metrics
        .dual_inf
        .is_some_and(|value| value.is_finite() && value <= tol);
    let comp_ok = record
        .metrics
        .complementarity_inf
        .is_none_or(|value| value.is_finite() && value <= tol);
    objective_ok && primal_ok && dual_ok && comp_ok
}

fn promote_solve_error_if_reduced_accuracy(record: &mut ProblemRunRecord, tol: f64) {
    if record.error.is_none() || !reduced_accuracy_residuals_met(record, tol) {
        return;
    }

    record.status = RunStatus::ReducedAccuracy;
    record.validation = ValidationOutcome {
        tier: ValidationTier::ReducedAccuracy,
        tolerance: format!("reduced: primal<={tol:.1e}, dual<={tol:.1e}, comp<={tol:.1e}"),
        detail: format!(
            "solver reported '{}', but final metrics satisfied reduced thresholds (objective={}, primal={}, dual={}, comp={})",
            record.error.as_deref().unwrap_or("solve error"),
            fmt_opt_sci(record.metrics.objective),
            fmt_opt_sci(record.metrics.primal_inf),
            fmt_opt_sci(record.metrics.dual_inf),
            fmt_opt_sci(record.metrics.complementarity_inf),
        ),
    };
}
