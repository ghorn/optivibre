#![cfg(feature = "ipopt")]

use approx::assert_abs_diff_eq;
use optimization::{
    CCS, CompiledNlpProblem, ConstraintBounds, InteriorPointOptions, IpoptOptions, ParameterMatrix,
    solve_nlp_interior_point, solve_nlp_ipopt,
};
use rstest::rstest;

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

fn compare_verbose_requested() -> bool {
    std::env::var_os("AD_CODEGEN_COMPARE_VERBOSE").is_some()
}

fn native_options() -> InteriorPointOptions {
    InteriorPointOptions {
        max_iters: 120,
        verbose: compare_verbose_requested(),
        ..InteriorPointOptions::default()
    }
}

fn hs071_native_options() -> InteriorPointOptions {
    InteriorPointOptions {
        max_iters: 300,
        dual_tol: 1.0e-5,
        overall_tol: 1.0e-5,
        filter_method: false,
        verbose: compare_verbose_requested(),
        ..InteriorPointOptions::default()
    }
}

fn ipopt_options() -> IpoptOptions {
    let verbose = compare_verbose_requested();
    IpoptOptions {
        max_iters: 120,
        print_level: if verbose { 5 } else { 0 },
        suppress_banner: !verbose,
        ..IpoptOptions::default()
    }
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0.0, |acc, (lhs_value, rhs_value)| {
            acc.max((lhs_value - rhs_value).abs())
        })
}

fn maybe_print_compare_summary(
    problem_name: &str,
    backend: Option<CallbackBackend>,
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
) {
    if !compare_verbose_requested() {
        return;
    }
    let backend_label = backend.map_or("direct", CallbackBackend::label);
    eprintln!();
    eprintln!(
        "[compare] {problem_name} backend={backend_label} native_iters={} ipopt_iters={} native_obj={} ipopt_obj={} max|dx|={:.3e}",
        native.iterations,
        ipopt.iterations,
        native.objective,
        ipopt.objective,
        max_abs_diff(&native.x, &ipopt.x),
    );
}

fn assert_native_matches_ipopt(
    problem_name: &str,
    backend: Option<CallbackBackend>,
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
    x_epsilon: f64,
    objective_epsilon: f64,
) {
    maybe_print_compare_summary(problem_name, backend, native, ipopt);
    assert_abs_diff_eq!(
        native.objective,
        ipopt.objective,
        epsilon = objective_epsilon
    );
    assert!(
        max_abs_diff(&native.x, &ipopt.x) <= x_epsilon,
        "native/ipopt x mismatch for {problem_name} backend={}: native={:?} ipopt={:?}",
        backend.map_or("direct", CallbackBackend::label),
        native.x,
        ipopt.x,
    );
    assert!(native.primal_inf_norm <= 1e-5);
    assert!(native.dual_inf_norm <= 1e-4);
    assert!(native.complementarity_inf_norm <= 1e-5);
    assert!(ipopt.primal_inf_norm <= 1e-5);
    assert!(ipopt.dual_inf_norm <= 1e-5);
    assert!(ipopt.complementarity_inf_norm <= 1e-5);
}

fn solve_native_ok<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
) -> optimization::InteriorPointSummary {
    solve_native_with_options_ok(problem, x0, parameters, native_options())
}

fn solve_native_with_options_ok<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: InteriorPointOptions,
) -> optimization::InteriorPointSummary {
    let solve_result = solve_nlp_interior_point(problem, x0, parameters, &options);
    assert!(
        solve_result.is_ok(),
        "native IP solve failed: {solve_result:?}"
    );
    match solve_result {
        Ok(summary) => summary,
        Err(err) => unreachable!("asserted success: {err}"),
    }
}

fn solve_ipopt_ok<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
) -> optimization::IpoptSummary {
    let solve_result = solve_nlp_ipopt(problem, x0, parameters, &ipopt_options());
    assert!(solve_result.is_ok(), "Ipopt solve failed: {solve_result:?}");
    match solve_result {
        Ok(summary) => summary,
        Err(err) => unreachable!("asserted success: {err}"),
    }
}

#[rstest]
fn compare_native_and_ipopt_on_equality_constrained_rosenbrock(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let native = solve_native_ok(&problem, &[0.5, 0.5], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[0.5, 0.5], &[]);
    assert_native_matches_ipopt(
        "equality_constrained_rosenbrock",
        Some(backend),
        &native,
        &ipopt,
        1e-4,
        1e-5,
    );
}

#[rstest]
fn compare_native_and_ipopt_on_casadi_rosenbrock_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let native = solve_native_ok(&problem, &[2.5, 3.0, 0.75], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[2.5, 3.0, 0.75], &[]);
    assert_native_matches_ipopt(
        "casadi_rosenbrock",
        Some(backend),
        &native,
        &ipopt,
        1e-4,
        1e-5,
    );
}

#[rstest]
fn compare_native_and_ipopt_on_simple_nlp(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let native = solve_native_ok(&problem, &[0.0, 0.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[0.0, 0.0], &[]);
    assert_native_matches_ipopt("simple_nlp", Some(backend), &native, &ipopt, 1e-5, 1e-5);
}

#[rstest]
fn compare_native_and_ipopt_on_hs021(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs021_problem(backend), backend);
    let native = solve_native_ok(&problem, &[2.0, 2.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[2.0, 2.0], &[]);
    assert_native_matches_ipopt("hs021", Some(backend), &native, &ipopt, 1e-5, 1e-5);
}

#[rstest]
fn compare_native_and_ipopt_on_hs035(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs035_problem(backend), backend);
    let native = solve_native_ok(&problem, &[0.5, 0.5, 0.5], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[0.5, 0.5, 0.5], &[]);
    assert_native_matches_ipopt("hs035", Some(backend), &native, &ipopt, 1e-4, 1e-5);
}

#[rstest]
fn compare_native_and_ipopt_on_hs071(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs071_problem(backend), backend);
    let native =
        solve_native_with_options_ok(&problem, &[1.0, 5.0, 5.0, 1.0], &[], hs071_native_options());
    let ipopt = solve_ipopt_ok(&problem, &[1.0, 5.0, 5.0, 1.0], &[]);
    assert_native_matches_ipopt("hs071", Some(backend), &native, &ipopt, 5e-3, 1e-4);
}

#[rstest]
fn compare_native_and_ipopt_on_parameterized_problem(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(parameterized_quadratic_problem(backend), backend);
    let parameter_values = [0.2, 0.8];
    let parameter_ccs = parameterized_quadratic_parameter_ccs();
    let parameters = [ParameterMatrix {
        ccs: &parameter_ccs,
        values: &parameter_values,
    }];
    let native = solve_native_ok(&problem, &[0.8, 0.2], &parameters);
    let ipopt = solve_ipopt_ok(&problem, &[0.8, 0.2], &parameters);
    assert_native_matches_ipopt(
        "parameterized_quadratic",
        Some(backend),
        &native,
        &ipopt,
        1e-5,
        1e-5,
    );
}

#[rstest]
fn compare_native_and_ipopt_on_hanging_chain(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let x0 = hanging_chain_initial_guess();
    let native = solve_native_ok(&problem, &x0, &[]);
    let ipopt = solve_ipopt_ok(&problem, &x0, &[]);
    assert_native_matches_ipopt("hanging_chain", Some(backend), &native, &ipopt, 1e-3, 1e-4);
}

#[test]
fn compare_native_and_ipopt_on_box_bounds_regression() {
    let problem = BoundConstrainedQuadraticProblem;
    let native = solve_native_ok(&problem, &[-10.0, 10.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[-10.0, 10.0], &[]);
    assert_native_matches_ipopt("box_bounds_quadratic", None, &native, &ipopt, 1e-4, 1e-4);
}

#[test]
fn compare_invalid_shape_rejected_by_both_solvers() {
    let invalid = invalid_shape_problem();
    assert!(solve_nlp_interior_point(&invalid, &[0.0, 0.0], &[], &native_options()).is_err());
    assert!(solve_nlp_ipopt(&invalid, &[0.0, 0.0], &[], &ipopt_options()).is_err());
}
