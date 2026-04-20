use approx::assert_abs_diff_eq;
use optimization::{
    CCS, CompiledNlpProblem, ConstraintBounds, InteriorPointLinearDebugOptions,
    InteriorPointLinearDebugSchedule, InteriorPointLinearDebugVerdict, InteriorPointLinearSolver,
    InteriorPointOptions, ParameterMatrix, solve_nlp_interior_point,
    solve_nlp_interior_point_with_callback, validate_nlp_problem_shapes, validate_parameter_inputs,
};
use rstest::rstest;
use spral_ssids::{
    NativeSpral, NumericFactorOptions as SpralNumericFactorOptions,
    OrderingStrategy as SpralOrderingStrategy, SsidsOptions as SpralSsidsOptions,
    SymmetricCscMatrix as SpralSymmetricCscMatrix, analyse as spral_analyse,
    factorize as spral_factorize,
};
use tempfile::TempDir;

#[path = "support/generated_problem.rs"]
mod generated_problem;

use generated_problem::{
    CallbackBackend, CallbackNlpProblem, casadi_rosenbrock_nlp_problem,
    constrained_rosenbrock_problem, hanging_chain_initial_guess, hanging_chain_problem,
    hs021_problem, hs035_problem, hs071_problem, invalid_shape_problem,
    parameterized_quadratic_parameter_ccs, parameterized_quadratic_problem, simple_nlp_problem,
};

struct BoundConstrainedQuadraticProblem;

impl CompiledNlpProblem for BoundConstrainedQuadraticProblem {
    fn dimension(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("bound constrained quadratic problem has no parameters")
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        Some(ConstraintBounds {
            lower: Some(vec![Some(2.0), None]),
            upper: Some(vec![None, Some(-3.0)]),
        })
    }

    fn equality_count(&self) -> usize {
        0
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] + 10.0).powi(2) + (x[1] - 10.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out.copy_from_slice(&[2.0 * (x[0] + 10.0), 2.0 * (x[1] - 10.0)]);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        static EMPTY: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| CCS::empty(0, 2))
    }

    fn equality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {}

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        static EMPTY: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| CCS::empty(0, 2))
    }

    fn inequality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        static HESSIAN_CCS: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        HESSIAN_CCS.get_or_init(|| CCS::lower_triangular_dense(2))
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out.copy_from_slice(&[2.0, 0.0, 2.0]);
    }
}

fn verbose_ip_requested() -> bool {
    std::env::var_os("AD_CODEGEN_IP_VERBOSE").is_some()
}

fn build_problem_ok(
    result: anyhow::Result<CallbackNlpProblem>,
    backend: CallbackBackend,
) -> CallbackNlpProblem {
    assert!(
        result.is_ok(),
        "problem construction failed for backend {}: {:?}",
        backend.label(),
        result.as_ref().err()
    );
    match result {
        Ok(problem) => problem,
        Err(err) => unreachable!("asserted success for backend {}: {err}", backend.label()),
    }
}

fn solve_ok<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    mut options: InteriorPointOptions,
) -> optimization::InteriorPointSummary {
    options.verbose |= verbose_ip_requested();
    let validate_result = validate_nlp_problem_shapes(problem);
    assert!(
        validate_result.is_ok(),
        "shape validation failed: {validate_result:?}"
    );
    let parameter_result = validate_parameter_inputs(problem, parameters);
    assert!(
        parameter_result.is_ok(),
        "parameter validation failed: {parameter_result:?}"
    );
    let solve_result = solve_nlp_interior_point(problem, x0, parameters, &options);
    assert!(
        solve_result.is_ok(),
        "interior-point solve failed: {solve_result:?}"
    );
    match solve_result {
        Ok(summary) => summary,
        Err(err) => unreachable!("asserted success: {err}"),
    }
}

fn native_spral_available() -> bool {
    NativeSpral::load().is_ok()
}

fn first_linear_debug_report(
    summary: &optimization::InteriorPointSummary,
) -> optimization::InteriorPointLinearDebugReport {
    summary
        .snapshots
        .iter()
        .find_map(|snapshot| snapshot.linear_debug.clone())
        .expect("expected at least one linear-debug report")
}

fn all_linear_debug_reports(
    summary: &optimization::InteriorPointSummary,
) -> Vec<optimization::InteriorPointLinearDebugReport> {
    summary
        .snapshots
        .iter()
        .filter_map(|snapshot| snapshot.linear_debug.clone())
        .collect()
}

fn debug_backend_result(
    report: &optimization::InteriorPointLinearDebugReport,
    solver: InteriorPointLinearSolver,
) -> &optimization::InteriorPointLinearDebugBackendResult {
    report
        .results
        .iter()
        .find(|result| result.solver == solver)
        .expect("expected backend result to be present")
}

fn assert_native_spral_parity_with_schedule(
    report: &optimization::InteriorPointLinearDebugReport,
    expected_schedule: InteriorPointLinearDebugSchedule,
    step_tol: f64,
    component_tol: f64,
) {
    assert_eq!(report.primary_solver, InteriorPointLinearSolver::SpralSsids);
    assert_eq!(report.schedule, expected_schedule);
    assert_eq!(report.verdict, InteriorPointLinearDebugVerdict::Consistent);
    assert_eq!(report.results.len(), 2);
    assert!(report.notes.is_empty());

    let primary = debug_backend_result(report, InteriorPointLinearSolver::SpralSsids);
    let native = debug_backend_result(report, InteriorPointLinearSolver::NativeSpralSsids);
    assert!(primary.success);
    assert!(native.success);
    assert_eq!(native.inertia, primary.inertia);
    assert!(
        native
            .residual_inf
            .expect("native residual should be present")
            <= 1e-8,
        "native residual too large: {:?}",
        native.residual_inf
    );
    assert!(
        native.step_delta_inf.expect("step delta should be present") <= step_tol,
        "step delta too large: {:?}",
        native.step_delta_inf
    );
    assert!(
        native.dx_delta_inf.expect("dx delta should be present") <= component_tol,
        "dx delta too large: {:?}",
        native.dx_delta_inf
    );
    assert!(
        native
            .d_lambda_delta_inf
            .expect("dlambda delta should be present")
            <= component_tol,
        "dlambda delta too large: {:?}",
        native.d_lambda_delta_inf
    );
    assert!(
        native.ds_delta_inf.expect("ds delta should be present") <= component_tol,
        "ds delta too large: {:?}",
        native.ds_delta_inf
    );
    assert!(
        native.dz_delta_inf.expect("dz delta should be present") <= component_tol,
        "dz delta too large: {:?}",
        native.dz_delta_inf
    );
}

fn assert_native_spral_parity(
    report: &optimization::InteriorPointLinearDebugReport,
    step_tol: f64,
    component_tol: f64,
) {
    assert_native_spral_parity_with_schedule(
        report,
        InteriorPointLinearDebugSchedule::FirstIteration,
        step_tol,
        component_tol,
    );
}

fn dump_field<'a>(dump: &'a str, key: &str) -> &'a str {
    dump.lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("expected {key} in linear debug dump"))
}

fn parse_dump_usize_vec(dump: &str, key: &str) -> Vec<usize> {
    serde_json::from_str(dump_field(dump, key)).expect("usize vector should parse from dump")
}

fn parse_dump_usize_value(dump: &str, key: &str) -> usize {
    dump_field(dump, key)
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("usize value for {key} should parse from dump: {error}"))
}

fn parse_dump_f64_vec(dump: &str, key: &str) -> Vec<f64> {
    serde_json::from_str(dump_field(dump, key)).expect("f64 vector should parse from dump")
}

#[derive(Debug)]
struct DumpedAugmentedBlocks {
    dx: Vec<f64>,
    ds: Vec<f64>,
    d_lambda: Vec<f64>,
    dz: Vec<f64>,
}

fn dumped_augmented_step_blocks(dump: &str, solution: &[f64]) -> DumpedAugmentedBlocks {
    let x_dimension = parse_dump_usize_value(dump, "x_dimension");
    let inequality_dimension = parse_dump_usize_value(dump, "inequality_dimension");
    let equality_dimension = parse_dump_usize_value(dump, "equality_dimension");
    let p_offset = parse_dump_usize_value(dump, "p_offset");
    let lambda_offset = parse_dump_usize_value(dump, "lambda_offset");
    let z_offset = parse_dump_usize_value(dump, "z_offset");
    let slack = parse_dump_f64_vec(dump, "slack");
    let multipliers = parse_dump_f64_vec(dump, "multipliers");

    let dx = solution[..x_dimension].to_vec();
    let p = solution[p_offset..p_offset + inequality_dimension].to_vec();
    let d_lambda = solution[lambda_offset..lambda_offset + equality_dimension].to_vec();
    let dz = solution[z_offset..z_offset + inequality_dimension].to_vec();
    let ds = p
        .iter()
        .zip(slack.iter())
        .zip(multipliers.iter())
        .map(|((p_i, slack_i), multiplier_i)| {
            let scaling = (slack_i.max(1e-16) / multiplier_i.max(1e-16)).sqrt();
            p_i * scaling
        })
        .collect::<Vec<_>>();

    DumpedAugmentedBlocks {
        dx,
        ds,
        d_lambda,
        dz,
    }
}

fn symmetric_lower_mat_vec(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let mut y = vec![0.0; dimension];
    for col in 0..dimension {
        for position in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[position];
            let value = values[position];
            y[row] += value * x[col];
            if row != col {
                y[col] += value * x[row];
            }
        }
    }
    y
}

fn symmetric_lower_abs_mat_vec(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    x_abs: &[f64],
) -> Vec<f64> {
    let mut y = vec![0.0; dimension];
    for col in 0..dimension {
        for position in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[position];
            let value_abs = values[position].abs();
            y[row] += value_abs * x_abs[col];
            if row != col {
                y[col] += value_abs * x_abs[row];
            }
        }
    }
    y
}

fn refined_residual_target(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    rhs: &[f64],
    solution: &[f64],
) -> (Vec<f64>, f64, f64) {
    let residual = symmetric_lower_mat_vec(dimension, col_ptrs, row_indices, values, solution)
        .into_iter()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs_i)| rhs_i - lhs)
        .collect::<Vec<_>>();
    let abs_solution = solution.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let lhs_scale =
        symmetric_lower_abs_mat_vec(dimension, col_ptrs, row_indices, values, &abs_solution);
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let target = lhs_scale
        .iter()
        .zip(rhs.iter())
        .fold(0.0_f64, |acc, (lhs_scale_i, rhs_i)| {
            acc.max(128.0 * f64::EPSILON * (1.0 + lhs_scale_i + rhs_i.abs()))
        });
    (residual, residual_inf, target)
}

fn replay_dumped_augmented_rust_spral(dump: &str) -> Vec<f64> {
    let col_ptrs = parse_dump_usize_vec(dump, "col_ptrs");
    let row_indices = parse_dump_usize_vec(dump, "row_indices");
    let values = parse_dump_f64_vec(dump, "values");
    let rhs = parse_dump_f64_vec(dump, "rhs");
    let matrix =
        SpralSymmetricCscMatrix::new(col_ptrs.len() - 1, &col_ptrs, &row_indices, Some(&values))
            .expect("dumped augmented KKT should form a valid symmetric CSC");
    let (symbolic, _) = spral_analyse(
        matrix,
        &SpralSsidsOptions {
            ordering: SpralOrderingStrategy::ApproximateMinimumDegree,
        },
    )
    .expect("rust spral analysis on dumped KKT should succeed");
    let (factor, _) = spral_factorize(matrix, &symbolic, &SpralNumericFactorOptions::default())
        .expect("rust spral factorization on dumped KKT should succeed");
    let mut solution = factor
        .solve(&rhs)
        .expect("rust spral solve on dumped KKT should succeed");
    let mut previous_residual_inf = f64::INFINITY;
    for _ in 0..10 {
        let (residual, residual_inf, target) = refined_residual_target(
            col_ptrs.len() - 1,
            &col_ptrs,
            &row_indices,
            &values,
            &rhs,
            &solution,
        );
        if residual_inf <= target || residual_inf >= previous_residual_inf * (1.0 - 1e-6) {
            break;
        }
        previous_residual_inf = residual_inf;
        let correction = factor
            .solve(&residual)
            .expect("rust spral refinement solve on dumped KKT should succeed");
        if correction.iter().all(|value| value.abs() <= f64::EPSILON) {
            break;
        }
        for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
            *solution_i += correction_i;
        }
    }
    solution
}

fn replay_dumped_augmented_native_spral(dump: &str) -> Vec<f64> {
    let native = NativeSpral::load().expect("native SPRAL should load");
    let col_ptrs = parse_dump_usize_vec(dump, "col_ptrs");
    let row_indices = parse_dump_usize_vec(dump, "row_indices");
    let values = parse_dump_f64_vec(dump, "values");
    let rhs = parse_dump_f64_vec(dump, "rhs");
    let matrix =
        SpralSymmetricCscMatrix::new(col_ptrs.len() - 1, &col_ptrs, &row_indices, Some(&values))
            .expect("dumped augmented KKT should form a valid symmetric CSC");
    let mut session = native
        .analyse(matrix)
        .expect("native SPRAL analysis on dumped KKT should succeed");
    session
        .factorize(matrix)
        .expect("native SPRAL factorization on dumped KKT should succeed");
    let mut solution = session
        .solve(&rhs)
        .expect("native SPRAL solve on dumped KKT should succeed");
    let mut previous_residual_inf = f64::INFINITY;
    for _ in 0..10 {
        let (residual, residual_inf, target) = refined_residual_target(
            col_ptrs.len() - 1,
            &col_ptrs,
            &row_indices,
            &values,
            &rhs,
            &solution,
        );
        if residual_inf <= target || residual_inf >= previous_residual_inf * (1.0 - 1e-6) {
            break;
        }
        previous_residual_inf = residual_inf;
        let correction = session
            .solve(&residual)
            .expect("native SPRAL refinement solve on dumped KKT should succeed");
        if correction.iter().all(|value| value.abs() <= f64::EPSILON) {
            break;
        }
        for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
            *solution_i += correction_i;
        }
    }
    solution
}

fn delta_inf(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0.0_f64, |acc, (lhs_i, rhs_i)| {
            acc.max((lhs_i - rhs_i).abs())
        })
}

fn sorted_iteration_dump_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut paths = std::fs::read_dir(dir)
        .expect("dump directory should be readable")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("nlip_kkt_iter_") && name.ends_with(".txt"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn interior_point_handwritten_problem_leaves_adapter_timing_unavailable() {
    let mut snapshots = Vec::new();
    let summary = solve_nlp_interior_point_with_callback(
        &BoundConstrainedQuadraticProblem,
        &[0.0, 0.0],
        &[],
        &InteriorPointOptions {
            verbose: false,
            ..InteriorPointOptions::default()
        },
        |snapshot| snapshots.push(snapshot.clone()),
    )
    .expect("interior-point solve should succeed");

    assert!(summary.profiling.adapter_timing.is_none());
    assert!(
        snapshots
            .iter()
            .all(|snapshot| snapshot.timing.adapter_timing.is_none())
    );
}

#[test]
fn interior_point_filter_frontier_is_exposed_in_snapshots() {
    let mut snapshots = Vec::new();
    solve_nlp_interior_point_with_callback(
        &BoundConstrainedQuadraticProblem,
        &[0.0, 0.0],
        &[],
        &InteriorPointOptions {
            verbose: false,
            ..InteriorPointOptions::default()
        },
        |snapshot| snapshots.push(snapshot.clone()),
    )
    .expect("interior-point solve should succeed");

    let accepted_snapshots = snapshots
        .iter()
        .filter(|snapshot| {
            matches!(
                snapshot.phase,
                optimization::InteriorPointIterationPhase::AcceptedStep
            )
        })
        .collect::<Vec<_>>();
    assert!(!accepted_snapshots.is_empty());
    assert!(
        accepted_snapshots
            .iter()
            .all(|snapshot| { snapshot.filter.is_some() })
    );
    assert!(accepted_snapshots.iter().any(|snapshot| {
        snapshot
            .filter
            .as_ref()
            .is_some_and(|filter| filter.accepted_mode.is_some())
    }));
}

#[cfg(feature = "serde")]
#[test]
fn interior_point_summary_serializes_with_duration_seconds() {
    let summary = solve_ok(
        &BoundConstrainedQuadraticProblem,
        &[0.0, 0.0],
        &[],
        InteriorPointOptions {
            verbose: false,
            ..InteriorPointOptions::default()
        },
    );

    let json = serde_json::to_value(&summary).expect("summary should serialize");
    assert!(json["profiling"]["objective_value"]["total_time"].is_number());
    assert!(json["profiling"]["total_time"].is_number());

    let roundtrip: optimization::InteriorPointSummary =
        serde_json::from_value(json).expect("summary should deserialize");
    assert_eq!(roundtrip.iterations, summary.iterations);
    assert_eq!(roundtrip.termination, summary.termination);
    assert_eq!(roundtrip.status_kind, summary.status_kind);
    assert_eq!(roundtrip.snapshots.len(), summary.snapshots.len());
}

#[test]
fn interior_point_summary_exposes_final_and_last_accepted_states() {
    let summary = solve_ok(
        &BoundConstrainedQuadraticProblem,
        &[0.0, 0.0],
        &[],
        InteriorPointOptions {
            verbose: false,
            ..InteriorPointOptions::default()
        },
    );

    assert!(!summary.snapshots.is_empty());
    assert_eq!(
        summary.final_state.phase,
        optimization::InteriorPointIterationPhase::Converged
    );
    assert!(summary.last_accepted_state.is_some());
}

#[test]
fn interior_point_rejects_invalid_shape_problem() {
    let solve_result = solve_nlp_interior_point(
        &invalid_shape_problem(),
        &[0.0, 0.0],
        &[],
        &InteriorPointOptions::default(),
    );
    assert!(solve_result.is_err());
}

#[test]
fn interior_point_rejects_auto_in_linear_debug_compare_solvers() {
    let error = solve_nlp_interior_point(
        &BoundConstrainedQuadraticProblem,
        &[0.0, 0.0],
        &[],
        &InteriorPointOptions {
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::Auto],
                schedule: InteriorPointLinearDebugSchedule::FirstIteration,
                dump_dir: None,
            }),
            ..InteriorPointOptions::default()
        },
    )
    .expect_err("auto compare backend should be rejected");

    assert!(matches!(
        error,
        optimization::InteriorPointSolveError::InvalidInput(message)
            if message.contains("linear_debug.compare_solvers")
    ));
}

#[rstest]
fn interior_point_solves_equality_constrained_rosenbrock(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.5, 0.5], &[], InteriorPointOptions::default());

    assert_abs_diff_eq!(summary.x[0], 0.61879562, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[1], 0.38120438, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.objective, 0.14560702, epsilon = 1e-4);
    assert!(summary.equality_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-5);
}

#[rstest]
fn interior_point_solves_casadi_rosenbrock_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[2.5, 3.0, 0.75],
        &[],
        InteriorPointOptions::default(),
    );

    assert_abs_diff_eq!(summary.x[0], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[2], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-6);
    assert!(summary.equality_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-5);
}

#[rstest]
fn interior_point_solves_casadi_simple_nlp_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.0, 0.0], &[], InteriorPointOptions::default());

    assert_abs_diff_eq!(summary.x[0], 5.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], 5.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.objective, 50.0, epsilon = 1e-5);
    assert!(summary.equality_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-5);
}

#[rstest]
fn interior_point_solves_hs021(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs021_problem(backend), backend);
    let summary = solve_ok(&problem, &[2.0, 2.0], &[], InteriorPointOptions::default());

    assert_abs_diff_eq!(summary.x[0], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.objective, -99.96, epsilon = 1e-5);
    assert!(summary.primal_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.complementarity_inf_norm <= 1e-5);
}

#[rstest]
fn interior_point_solves_hs035(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs035_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[0.5, 0.5, 0.5],
        &[],
        InteriorPointOptions::default(),
    );

    assert_abs_diff_eq!(summary.x[0], 4.0 / 3.0, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[1], 7.0 / 9.0, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[2], 4.0 / 9.0, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.objective, 1.0 / 9.0, epsilon = 1e-5);
    assert!(summary.primal_inf_norm <= 1e-5);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.complementarity_inf_norm <= 1e-5);
}

#[rstest]
fn interior_point_solves_hs071(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs071_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[1.0, 5.0, 5.0, 1.0],
        &[],
        InteriorPointOptions {
            max_iters: 300,
            dual_tol: 1.0e-5,
            overall_tol: 1.0e-5,
            ..InteriorPointOptions::default()
        },
    );

    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 5e-3);
    assert_abs_diff_eq!(summary.x[1], 4.74299964, epsilon = 5e-3);
    assert_abs_diff_eq!(summary.x[2], 3.82114998, epsilon = 5e-3);
    assert_abs_diff_eq!(summary.x[3], 1.37940829, epsilon = 5e-3);
    assert_abs_diff_eq!(summary.objective, 17.0140173, epsilon = 5e-3);
    assert!(summary.primal_inf_norm <= 1e-5);
    assert!(summary.dual_inf_norm <= 1e-4);
    assert!(summary.complementarity_inf_norm <= 1e-5);
}

#[rstest]
fn interior_point_solves_parameterized_problem(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(parameterized_quadratic_problem(backend), backend);
    let parameter_ccs = parameterized_quadratic_parameter_ccs();
    let parameter_values = [0.25, 0.75];
    let parameters = [ParameterMatrix {
        ccs: &parameter_ccs,
        values: &parameter_values,
    }];
    let summary = solve_ok(
        &problem,
        &[0.9, 0.1],
        &parameters,
        InteriorPointOptions::default(),
    );

    assert_abs_diff_eq!(summary.x[0], 0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], 0.75, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-8);
}

#[rstest]
fn interior_point_solves_hanging_chain(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            max_iters: 200,
            dual_tol: 1e-5,
            ..InteriorPointOptions::default()
        },
    );

    assert!(summary.equality_inf_norm <= 1e-5);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.objective < -1.35);
    assert_abs_diff_eq!(summary.x[0] + summary.x[6], 3.0, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[2] + summary.x[4], 3.0, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[1], summary.x[7], epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[3], summary.x[5], epsilon = 1e-4);
}

#[test]
fn interior_point_enforces_box_bounds_regression() {
    let problem = BoundConstrainedQuadraticProblem;
    let summary = solve_ok(&problem, &[0.0, 0.0], &[], InteriorPointOptions::default());

    assert_abs_diff_eq!(summary.x[0], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], -3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 313.0, epsilon = 1e-4);
    assert!(summary.primal_inf_norm <= 1e-6);
    assert!(summary.complementarity_inf_norm <= 1e-6);
}

#[rstest]
fn interior_point_reports_profiling_breakdown(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs021_problem(backend), backend);
    let summary = solve_ok(&problem, &[2.0, 2.0], &[], InteriorPointOptions::default());

    let total_callback_calls = summary.profiling.objective_value.calls
        + summary.profiling.objective_gradient.calls
        + summary.profiling.equality_values.calls
        + summary.profiling.inequality_values.calls
        + summary.profiling.equality_jacobian_values.calls
        + summary.profiling.inequality_jacobian_values.calls
        + summary.profiling.lagrangian_hessian_values.calls;
    assert!(total_callback_calls > 0);
    assert!(summary.profiling.linear_solves > 0);
    assert!(summary.profiling.kkt_assemblies > 0);
    assert!(summary.profiling.total_time >= summary.profiling.linear_solve_time);
}

#[test]
fn interior_point_auto_prefers_spral_ssids_on_small_hanging_chain() {
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions::default(),
    );
    assert_eq!(summary.linear_solver, InteriorPointLinearSolver::SpralSsids);
}

#[test]
fn interior_point_auto_prefers_spral_ssids_on_small_kkt_system() {
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let summary = solve_ok(&problem, &[2.0, 2.0], &[], InteriorPointOptions::default());
    assert_eq!(summary.linear_solver, InteriorPointLinearSolver::SpralSsids,);
}

#[test]
fn interior_point_spral_and_qdldl_match_on_small_kkt_system() {
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let spral = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            ..InteriorPointOptions::default()
        },
    );
    let qdldl = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SparseQdldl,
            ..InteriorPointOptions::default()
        },
    );

    assert_eq!(spral.linear_solver, InteriorPointLinearSolver::SpralSsids);
    assert_eq!(qdldl.linear_solver, InteriorPointLinearSolver::SparseQdldl);
    assert_abs_diff_eq!(spral.x[0], qdldl.x[0], epsilon = 1e-6);
    assert_abs_diff_eq!(spral.x[1], qdldl.x[1], epsilon = 1e-6);
    assert_abs_diff_eq!(spral.objective, qdldl.objective, epsilon = 1e-8);
    assert!(spral.primal_inf_norm <= 1e-6);
    assert!(spral.dual_inf_norm <= 1e-5);
    assert!(spral.complementarity_inf_norm <= 1e-6);
}

#[test]
fn interior_point_spral_and_qdldl_match_on_hanging_chain() {
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let spral = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            ..InteriorPointOptions::default()
        },
    );
    let qdldl = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SparseQdldl,
            ..InteriorPointOptions::default()
        },
    );

    assert_eq!(spral.linear_solver, InteriorPointLinearSolver::SpralSsids);
    assert_eq!(qdldl.linear_solver, InteriorPointLinearSolver::SparseQdldl);
    assert_abs_diff_eq!(spral.objective, qdldl.objective, epsilon = 1e-6);
    for (spral_x, qdldl_x) in spral.x.iter().zip(qdldl.x.iter()) {
        assert_abs_diff_eq!(spral_x, qdldl_x, epsilon = 1e-5);
    }
    assert!(spral.primal_inf_norm <= 1e-6);
    assert!(spral.dual_inf_norm <= 1e-6);
}

#[test]
fn interior_point_native_spral_matches_small_kkt_system() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let rust_spral = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            ..InteriorPointOptions::default()
        },
    );
    let native = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::NativeSpralSsids,
            ..InteriorPointOptions::default()
        },
    );

    assert_eq!(
        native.linear_solver,
        InteriorPointLinearSolver::NativeSpralSsids
    );
    assert_abs_diff_eq!(rust_spral.x[0], native.x[0], epsilon = 1e-6);
    assert_abs_diff_eq!(rust_spral.x[1], native.x[1], epsilon = 1e-6);
    assert_abs_diff_eq!(rust_spral.objective, native.objective, epsilon = 1e-8);
    assert!(native.primal_inf_norm <= 1e-6);
    assert!(native.dual_inf_norm <= 1e-5);
    assert!(native.complementarity_inf_norm <= 1e-6);
}

#[test]
fn interior_point_native_spral_matches_hanging_chain() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let rust_spral = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            ..InteriorPointOptions::default()
        },
    );
    let native = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::NativeSpralSsids,
            ..InteriorPointOptions::default()
        },
    );

    assert_eq!(
        native.linear_solver,
        InteriorPointLinearSolver::NativeSpralSsids
    );
    assert_abs_diff_eq!(rust_spral.objective, native.objective, epsilon = 1e-6);
    for (rust_x, native_x) in rust_spral.x.iter().zip(native.x.iter()) {
        assert_abs_diff_eq!(rust_x, native_x, epsilon = 1e-5);
    }
    assert!(native.primal_inf_norm <= 1e-6);
    assert!(native.dual_inf_norm <= 1e-6);
}

#[test]
fn interior_point_spral_reuses_symbolic_analysis_and_refactorizes() {
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let summary = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            ..InteriorPointOptions::default()
        },
    );

    assert_eq!(summary.linear_solver, InteriorPointLinearSolver::SpralSsids);
    assert_eq!(summary.profiling.sparse_symbolic_analyses, 1);
    assert!(summary.profiling.sparse_numeric_factorizations >= 1);
    assert!(summary.profiling.sparse_numeric_refactorizations >= 1);
}

#[test]
fn interior_point_native_spral_reuses_symbolic_analysis_and_refactorizes() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let summary = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::NativeSpralSsids,
            ..InteriorPointOptions::default()
        },
    );

    assert_eq!(
        summary.linear_solver,
        InteriorPointLinearSolver::NativeSpralSsids
    );
    assert_eq!(summary.profiling.sparse_symbolic_analyses, 1);
    assert!(summary.profiling.sparse_numeric_factorizations >= 1);
    assert!(summary.profiling.sparse_numeric_refactorizations >= 1);
}

#[test]
fn interior_point_native_spral_matches_small_kkt_extremely_closely() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let summary = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::FirstIteration,
                dump_dir: None,
            }),
            ..InteriorPointOptions::default()
        },
    );

    let report = first_linear_debug_report(&summary);
    assert_native_spral_parity(&report, 1e-9, 1e-10);
}

#[test]
fn interior_point_native_spral_matches_hanging_chain_extremely_closely() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::FirstIteration,
                dump_dir: None,
            }),
            ..InteriorPointOptions::default()
        },
    );

    let report = first_linear_debug_report(&summary);
    assert_native_spral_parity(&report, 1e-8, 1e-9);
}

#[test]
fn interior_point_native_spral_matches_hanging_chain_each_iteration_extremely_closely() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::EveryIteration,
                dump_dir: None,
            }),
            ..InteriorPointOptions::default()
        },
    );

    let reports = all_linear_debug_reports(&summary);
    assert!(
        !reports.is_empty(),
        "expected per-iteration native SPRAL comparison reports"
    );
    for report in &reports {
        assert_native_spral_parity_with_schedule(
            report,
            InteriorPointLinearDebugSchedule::EveryIteration,
            1e-8,
            1e-9,
        );
    }
}

#[test]
fn interior_point_native_spral_exact_small_kkt_replay_matches_to_machine_precision() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let dump_dir = TempDir::new().expect("temp dump dir should create");
    let summary = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::FirstIteration,
                dump_dir: Some(dump_dir.path().to_path_buf()),
            }),
            ..InteriorPointOptions::default()
        },
    );

    let report = first_linear_debug_report(&summary);
    assert_native_spral_parity(&report, 1e-9, 1e-10);

    let dump = std::fs::read_to_string(dump_dir.path().join("nlip_kkt_iter_0000.txt"))
        .expect("expected first-iteration KKT dump");
    let rust_solution = replay_dumped_augmented_rust_spral(&dump);
    let native_solution = replay_dumped_augmented_native_spral(&dump);
    let rust_blocks = dumped_augmented_step_blocks(&dump, &rust_solution);
    let native_blocks = dumped_augmented_step_blocks(&dump, &native_solution);
    let solution_delta = delta_inf(&rust_solution, &native_solution);
    assert!(
        solution_delta <= 1e-12,
        "expected dumped augmented small-KKT replay to match to machine precision, got delta={solution_delta:.3e}"
    );
    assert!(
        delta_inf(&rust_blocks.dx, &native_blocks.dx) <= 1e-12,
        "expected dumped augmented small-KKT dx replay to match to machine precision"
    );
    assert!(
        delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda) <= 1e-12,
        "expected dumped augmented small-KKT d_lambda replay to match to machine precision"
    );
    assert!(
        delta_inf(&rust_blocks.ds, &native_blocks.ds) <= 1e-12,
        "expected dumped augmented small-KKT ds replay to match to machine precision"
    );
    assert!(
        delta_inf(&rust_blocks.dz, &native_blocks.dz) <= 1e-12,
        "expected dumped augmented small-KKT dz replay to match to machine precision"
    );
}

#[test]
fn interior_point_native_spral_exact_hanging_chain_replay_matches_extremely_closely() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let dump_dir = TempDir::new().expect("temp dump dir should create");
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::FirstIteration,
                dump_dir: Some(dump_dir.path().to_path_buf()),
            }),
            ..InteriorPointOptions::default()
        },
    );

    let report = first_linear_debug_report(&summary);
    assert_native_spral_parity(&report, 1e-8, 1e-9);

    let dump = std::fs::read_to_string(dump_dir.path().join("nlip_kkt_iter_0000.txt"))
        .expect("expected first-iteration KKT dump");
    let rust_solution = replay_dumped_augmented_rust_spral(&dump);
    let native_solution = replay_dumped_augmented_native_spral(&dump);
    let rust_blocks = dumped_augmented_step_blocks(&dump, &rust_solution);
    let native_blocks = dumped_augmented_step_blocks(&dump, &native_solution);
    let solution_delta = delta_inf(&rust_solution, &native_solution);
    assert!(
        solution_delta <= 5e-10,
        "expected dumped augmented hanging-chain replay to match extremely closely, got delta={solution_delta:.3e}"
    );
    assert!(
        delta_inf(&rust_blocks.dx, &native_blocks.dx) <= 5e-10,
        "expected dumped augmented hanging-chain dx replay to match extremely closely"
    );
    assert!(
        delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda) <= 5e-10,
        "expected dumped augmented hanging-chain d_lambda replay to match extremely closely"
    );
    assert!(
        delta_inf(&rust_blocks.ds, &native_blocks.ds) <= 5e-10,
        "expected dumped augmented hanging-chain ds replay to match extremely closely"
    );
    assert!(
        delta_inf(&rust_blocks.dz, &native_blocks.dz) <= 5e-10,
        "expected dumped augmented hanging-chain dz replay to match extremely closely"
    );
}

#[test]
fn interior_point_native_spral_exact_hanging_chain_replay_matches_each_dump_extremely_closely() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let dump_dir = TempDir::new().expect("temp dump dir should create");
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::EveryIteration,
                dump_dir: Some(dump_dir.path().to_path_buf()),
            }),
            ..InteriorPointOptions::default()
        },
    );

    let reports = all_linear_debug_reports(&summary);
    assert!(
        !reports.is_empty(),
        "expected per-iteration native SPRAL comparison reports"
    );

    let dump_paths = sorted_iteration_dump_paths(dump_dir.path());
    assert_eq!(
        dump_paths.len(),
        reports.len(),
        "expected one dumped KKT snapshot per linear-debug report"
    );

    for dump_path in dump_paths {
        let dump = std::fs::read_to_string(&dump_path).expect("expected dumped KKT snapshot");
        let rust_solution = replay_dumped_augmented_rust_spral(&dump);
        let native_solution = replay_dumped_augmented_native_spral(&dump);
        let rust_blocks = dumped_augmented_step_blocks(&dump, &rust_solution);
        let native_blocks = dumped_augmented_step_blocks(&dump, &native_solution);
        let solution_delta = delta_inf(&rust_solution, &native_solution);
        let dx_delta = delta_inf(&rust_blocks.dx, &native_blocks.dx);
        let dlambda_delta = delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda);
        let ds_delta = delta_inf(&rust_blocks.ds, &native_blocks.ds);
        let dz_delta = delta_inf(&rust_blocks.dz, &native_blocks.dz);
        assert!(
            solution_delta <= 5e-10,
            "expected dumped hanging-chain replay {} to match extremely closely, got delta={solution_delta:.3e}",
            dump_path.display()
        );
        assert!(
            dx_delta <= 5e-10,
            "expected dumped hanging-chain dx replay {} to match extremely closely",
            dump_path.display()
        );
        assert!(
            dlambda_delta <= 5e-10,
            "expected dumped hanging-chain d_lambda replay {} to match extremely closely",
            dump_path.display()
        );
        assert!(
            ds_delta <= 5e-10,
            "expected dumped hanging-chain ds replay {} to match extremely closely",
            dump_path.display()
        );
        assert!(
            dz_delta <= 5e-10,
            "expected dumped hanging-chain dz replay {} to match extremely closely",
            dump_path.display()
        );
    }
}

#[test]
fn interior_point_native_spral_hanging_chain_refactorize_sequence_matches_extremely_closely() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(
        hanging_chain_problem(CallbackBackend::Aot),
        CallbackBackend::Aot,
    );
    let dump_dir = TempDir::new().expect("temp dump dir should create");
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![InteriorPointLinearSolver::NativeSpralSsids],
                schedule: InteriorPointLinearDebugSchedule::EveryIteration,
                dump_dir: Some(dump_dir.path().to_path_buf()),
            }),
            ..InteriorPointOptions::default()
        },
    );

    let reports = all_linear_debug_reports(&summary);
    assert!(
        !reports.is_empty(),
        "expected per-iteration native SPRAL comparison reports"
    );

    let dump_paths = sorted_iteration_dump_paths(dump_dir.path());
    assert_eq!(
        dump_paths.len(),
        reports.len(),
        "expected one dumped KKT snapshot per linear-debug report"
    );

    let first_dump = std::fs::read_to_string(&dump_paths[0]).expect("expected first dumped KKT");
    let col_ptrs = parse_dump_usize_vec(&first_dump, "col_ptrs");
    let row_indices = parse_dump_usize_vec(&first_dump, "row_indices");
    let first_values = parse_dump_f64_vec(&first_dump, "values");
    let first_matrix = SpralSymmetricCscMatrix::new(
        col_ptrs.len() - 1,
        &col_ptrs,
        &row_indices,
        Some(&first_values),
    )
    .expect("first dumped KKT should be a valid symmetric CSC");
    let (symbolic, _) = spral_analyse(
        first_matrix,
        &SpralSsidsOptions {
            ordering: SpralOrderingStrategy::ApproximateMinimumDegree,
        },
    )
    .expect("rust spral analysis on dumped sequence should succeed");
    let (mut rust_factor, _) = spral_factorize(
        first_matrix,
        &symbolic,
        &SpralNumericFactorOptions::default(),
    )
    .expect("rust spral initial factorization on dumped sequence should succeed");

    let native = NativeSpral::load().expect("native SPRAL should load");
    let mut native_session = native
        .analyse(
            SpralSymmetricCscMatrix::new(
                col_ptrs.len() - 1,
                &col_ptrs,
                &row_indices,
                Some(&first_values),
            )
            .expect("first dumped KKT should be a valid native symmetric CSC"),
        )
        .expect("native spral analysis on dumped sequence should succeed");
    native_session
        .factorize(
            SpralSymmetricCscMatrix::new(
                col_ptrs.len() - 1,
                &col_ptrs,
                &row_indices,
                Some(&first_values),
            )
            .expect("first dumped KKT should be a valid native symmetric CSC"),
        )
        .expect("native spral initial factorization on dumped sequence should succeed");

    for (index, dump_path) in dump_paths.iter().enumerate() {
        let dump = std::fs::read_to_string(dump_path).expect("expected dumped KKT snapshot");
        let values = parse_dump_f64_vec(&dump, "values");
        let rhs = parse_dump_f64_vec(&dump, "rhs");
        let matrix = SpralSymmetricCscMatrix::new(
            col_ptrs.len() - 1,
            &col_ptrs,
            &row_indices,
            Some(&values),
        )
        .expect("dumped sequence KKT should be a valid symmetric CSC");
        if index > 0 {
            rust_factor
                .refactorize(matrix)
                .expect("rust spral refactorization on dumped sequence should succeed");
            native_session
                .refactorize(matrix)
                .expect("native spral refactorization on dumped sequence should succeed");
        }

        let mut rust_solution = rust_factor
            .solve(&rhs)
            .expect("rust spral solve on dumped sequence should succeed");
        let mut native_solution = native_session
            .solve(&rhs)
            .expect("native spral solve on dumped sequence should succeed");

        let mut previous_rust_residual = f64::INFINITY;
        for _ in 0..10 {
            let (residual, residual_inf, target) = refined_residual_target(
                col_ptrs.len() - 1,
                &col_ptrs,
                &row_indices,
                &values,
                &rhs,
                &rust_solution,
            );
            if residual_inf <= target || residual_inf >= previous_rust_residual * (1.0 - 1e-6) {
                break;
            }
            previous_rust_residual = residual_inf;
            let correction = rust_factor
                .solve(&residual)
                .expect("rust spral refinement solve on dumped sequence should succeed");
            if correction.iter().all(|value| value.abs() <= f64::EPSILON) {
                break;
            }
            for (solution_i, correction_i) in rust_solution.iter_mut().zip(correction.iter()) {
                *solution_i += correction_i;
            }
        }

        let mut previous_native_residual = f64::INFINITY;
        for _ in 0..10 {
            let (residual, residual_inf, target) = refined_residual_target(
                col_ptrs.len() - 1,
                &col_ptrs,
                &row_indices,
                &values,
                &rhs,
                &native_solution,
            );
            if residual_inf <= target || residual_inf >= previous_native_residual * (1.0 - 1e-6) {
                break;
            }
            previous_native_residual = residual_inf;
            let correction = native_session
                .solve(&residual)
                .expect("native spral refinement solve on dumped sequence should succeed");
            if correction.iter().all(|value| value.abs() <= f64::EPSILON) {
                break;
            }
            for (solution_i, correction_i) in native_solution.iter_mut().zip(correction.iter()) {
                *solution_i += correction_i;
            }
        }

        let rust_blocks = dumped_augmented_step_blocks(&dump, &rust_solution);
        let native_blocks = dumped_augmented_step_blocks(&dump, &native_solution);
        let solution_delta = delta_inf(&rust_solution, &native_solution);
        let dx_delta = delta_inf(&rust_blocks.dx, &native_blocks.dx);
        let dlambda_delta = delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda);
        let ds_delta = delta_inf(&rust_blocks.ds, &native_blocks.ds);
        let dz_delta = delta_inf(&rust_blocks.dz, &native_blocks.dz);
        assert!(
            solution_delta <= 5e-10,
            "expected hanging-chain refactorized replay {} to match extremely closely, got delta={solution_delta:.3e}",
            dump_path.display()
        );
        assert!(
            dx_delta <= 5e-10,
            "expected hanging-chain refactorized dx replay {} to match extremely closely",
            dump_path.display()
        );
        assert!(
            dlambda_delta <= 5e-10,
            "expected hanging-chain refactorized d_lambda replay {} to match extremely closely",
            dump_path.display()
        );
        assert!(
            ds_delta <= 5e-10,
            "expected hanging-chain refactorized ds replay {} to match extremely closely",
            dump_path.display()
        );
        assert!(
            dz_delta <= 5e-10,
            "expected hanging-chain refactorized dz replay {} to match extremely closely",
            dump_path.display()
        );
    }
}

#[test]
fn interior_point_linear_debug_compare_records_backend_results_on_small_kkt() {
    if !native_spral_available() {
        eprintln!("skipping native SPRAL test: library unavailable");
        return;
    }
    let problem = build_problem_ok(hs021_problem(CallbackBackend::Aot), CallbackBackend::Aot);
    let summary = solve_ok(
        &problem,
        &[2.0, 2.0],
        &[],
        InteriorPointOptions {
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            linear_debug: Some(InteriorPointLinearDebugOptions {
                compare_solvers: vec![
                    InteriorPointLinearSolver::NativeSpralSsids,
                    InteriorPointLinearSolver::SparseQdldl,
                ],
                schedule: InteriorPointLinearDebugSchedule::FirstIteration,
                dump_dir: None,
            }),
            ..InteriorPointOptions::default()
        },
    );

    let report = first_linear_debug_report(&summary);
    assert_eq!(report.primary_solver, InteriorPointLinearSolver::SpralSsids);
    assert_eq!(
        report.schedule,
        InteriorPointLinearDebugSchedule::FirstIteration
    );
    assert_eq!(report.results.len(), 3);
    assert_ne!(
        report.verdict,
        InteriorPointLinearDebugVerdict::ComparisonIncomplete
    );

    let primary = report
        .results
        .iter()
        .find(|result| result.solver == InteriorPointLinearSolver::SpralSsids)
        .expect("primary result");
    for solver in [
        InteriorPointLinearSolver::NativeSpralSsids,
        InteriorPointLinearSolver::SparseQdldl,
    ] {
        let comparison = report
            .results
            .iter()
            .find(|result| result.solver == solver)
            .expect("comparison result");
        assert!(comparison.success);
        assert!(comparison.residual_inf.expect("residual") <= 1e-7);
        assert!(comparison.step_delta_inf.is_some());
        assert!(comparison.dx_delta_inf.is_some());
        assert!(comparison.d_lambda_delta_inf.is_some());
        assert!(comparison.ds_delta_inf.is_some());
        assert!(comparison.dz_delta_inf.is_some());
        if solver == InteriorPointLinearSolver::NativeSpralSsids {
            assert_eq!(comparison.inertia, primary.inertia);
        } else {
            assert!(comparison.inertia.is_some());
        }
    }
    let native = debug_backend_result(&report, InteriorPointLinearSolver::NativeSpralSsids);
    if report.verdict == InteriorPointLinearDebugVerdict::Consistent {
        for comparison in report.results.iter().filter(|result| {
            result.solver == InteriorPointLinearSolver::NativeSpralSsids
                || result.solver == InteriorPointLinearSolver::SparseQdldl
        }) {
            assert!(comparison.step_delta_inf.expect("step delta") <= 1e-5);
            assert!(comparison.dx_delta_inf.expect("dx delta") <= 1e-5);
            assert!(comparison.d_lambda_delta_inf.expect("dlambda delta") <= 1e-5);
            assert!(comparison.ds_delta_inf.expect("ds delta") <= 1e-5);
            assert!(comparison.dz_delta_inf.expect("dz delta") <= 1e-5);
        }
    } else {
        assert_eq!(
            report.verdict,
            InteriorPointLinearDebugVerdict::LinearSolverMismatch
        );
        if native.step_delta_inf.is_some_and(|delta| delta <= 1e-5)
            && native.dx_delta_inf.is_some_and(|delta| delta <= 1e-5)
            && native.d_lambda_delta_inf.is_some_and(|delta| delta <= 1e-5)
            && native.ds_delta_inf.is_some_and(|delta| delta <= 1e-5)
            && native.dz_delta_inf.is_some_and(|delta| delta <= 1e-5)
        {
            assert!(
                report
                    .notes
                    .iter()
                    .any(|note| note
                        .contains("matched primary within tolerance: native_spral_ssids")),
                "expected debug notes to record native SPRAL parity when another backend diverges: {:?}",
                report.notes
            );
        }
        assert!(report.results.iter().any(|result| {
            !result.success
                || result.inertia != primary.inertia
                || result.step_delta_inf.is_some_and(|delta| delta > 1e-5)
                || result.dx_delta_inf.is_some_and(|delta| delta > 1e-5)
                || result.d_lambda_delta_inf.is_some_and(|delta| delta > 1e-5)
                || result.ds_delta_inf.is_some_and(|delta| delta > 1e-5)
                || result.dz_delta_inf.is_some_and(|delta| delta > 1e-5)
        }));
    }
}
