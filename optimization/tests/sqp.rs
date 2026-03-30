use approx::assert_abs_diff_eq;
use optimization::{
    CCS, ClarabelSqpOptions, CompiledNlpProblem, ParameterMatrix, solve_nlp_sqp,
    validate_nlp_problem_shapes, validate_parameter_inputs,
};
use rstest::rstest;
use std::time::Duration;

#[path = "support/generated_problem.rs"]
mod generated_problem;

use generated_problem::{
    CallbackBackend, CallbackNlpProblem, casadi_rosenbrock_nlp_problem,
    constrained_rosenbrock_problem, hanging_chain_initial_guess, hanging_chain_problem,
    hs021_problem, hs035_problem, hs071_problem, invalid_shape_problem,
    parameterized_quadratic_parameter_ccs, parameterized_quadratic_problem, simple_nlp_problem,
};

fn verbose_sqp_requested() -> bool {
    std::env::var_os("AD_CODEGEN_SQP_VERBOSE").is_some()
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
    mut options: ClarabelSqpOptions,
) -> optimization::ClarabelSqpSummary {
    options.verbose |= verbose_sqp_requested();
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
    let solve_result = solve_nlp_sqp(problem, x0, parameters, &options);
    assert!(solve_result.is_ok(), "SQP solve failed: {solve_result:?}");
    match solve_result {
        Ok(summary) => summary,
        Err(err) => unreachable!("asserted success: {err}"),
    }
}

#[test]
fn validate_shapes_rejects_mismatched_inequality_jacobian() {
    let result = validate_nlp_problem_shapes(&invalid_shape_problem());
    assert!(result.is_err());
}

#[rstest]
fn validate_parameters_rejects_wrong_ccs(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(parameterized_quadratic_problem(backend), backend);
    let wrong_ccs = CCS::new(1, 2, vec![0, 1, 2], vec![0, 0]);
    let parameters = [ParameterMatrix {
        ccs: &wrong_ccs,
        values: &[0.25, 0.75],
    }];
    let result = validate_parameter_inputs(&problem, &parameters);
    assert!(result.is_err());
}

#[rstest]
fn sqp_solves_equality_constrained_rosenbrock(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[0.5, 0.5],
        &[],
        ClarabelSqpOptions {
            merit_penalty: 20.0,
            dual_tol: 2e-6,
            ..ClarabelSqpOptions::default()
        },
    );

    assert_abs_diff_eq!(summary.x[0], 0.61879562, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[1], 0.38120438, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.objective, 0.14560702, epsilon = 1e-4);
    assert!(summary.equality_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 2e-6);
    assert!(summary.complementarity_inf_norm <= 1e-12);
}

#[rstest]
fn sqp_solves_casadi_rosenbrock_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[2.5, 3.0, 0.75],
        &[],
        ClarabelSqpOptions::default(),
    );

    assert_abs_diff_eq!(summary.x[0], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[2], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-6);
    assert!(summary.equality_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-7);
}

#[rstest]
fn sqp_solves_casadi_simple_nlp_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.0, 0.0], &[], ClarabelSqpOptions::default());

    assert_abs_diff_eq!(summary.x[0], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 50.0, epsilon = 1e-6);
    assert!(summary.equality_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-7);
}

#[rstest]
fn sqp_solves_hs021(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs021_problem(backend), backend);
    let summary = solve_ok(&problem, &[2.0, 2.0], &[], ClarabelSqpOptions::default());

    assert_abs_diff_eq!(summary.x[0], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, -99.96, epsilon = 1e-6);
    assert!(summary.primal_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-7);
    assert!(summary.complementarity_inf_norm <= 1e-7);
}

#[rstest]
fn sqp_solves_hs035(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs035_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[0.5, 0.5, 0.5],
        &[],
        ClarabelSqpOptions::default(),
    );

    assert_abs_diff_eq!(summary.x[0], 4.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], 7.0 / 9.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[2], 4.0 / 9.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.objective, 1.0 / 9.0, epsilon = 1e-6);
    assert!(summary.primal_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-7);
    assert!(summary.complementarity_inf_norm <= 1e-7);
}

#[rstest]
fn sqp_solves_hs071(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs071_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[1.0, 5.0, 5.0, 1.0],
        &[],
        ClarabelSqpOptions {
            max_iters: 80,
            merit_penalty: 50.0,
            ..ClarabelSqpOptions::default()
        },
    );

    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 2e-3);
    assert_abs_diff_eq!(summary.x[1], 4.74299964, epsilon = 2e-3);
    assert_abs_diff_eq!(summary.x[2], 3.82114998, epsilon = 2e-3);
    assert_abs_diff_eq!(summary.x[3], 1.37940829, epsilon = 2e-3);
    assert_abs_diff_eq!(summary.objective, 17.0140173, epsilon = 5e-4);
    assert!(summary.primal_inf_norm <= 1e-5);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.complementarity_inf_norm <= 1e-5);
}

#[rstest]
fn sqp_solves_parameterized_problem(
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
        ClarabelSqpOptions::default(),
    );

    assert_abs_diff_eq!(summary.x[0], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 0.75, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-9);
}

#[rstest]
fn sqp_solves_hanging_chain(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        ClarabelSqpOptions {
            max_iters: 120,
            merit_penalty: 50.0,
            dual_tol: 1e-5,
            ..ClarabelSqpOptions::default()
        },
    );

    assert!(summary.equality_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.complementarity_inf_norm <= 1e-12);
    assert!(summary.objective < -1.35);

    assert_abs_diff_eq!(summary.x[0] + summary.x[6], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[2] + summary.x[4], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], summary.x[7], epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[3], summary.x[5], epsilon = 1e-5);
    assert!(summary.x[1] <= 0.0);
    assert!(summary.x[3] <= summary.x[1]);
    assert!(summary.x[3] <= summary.x[5]);
}

#[rstest]
fn sqp_reports_profiling_breakdown(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        ClarabelSqpOptions {
            max_iters: 120,
            merit_penalty: 50.0,
            dual_tol: 1e-5,
            ..ClarabelSqpOptions::default()
        },
    );
    let profiling = &summary.profiling;
    let iteration_count = summary.iterations + 1;
    let callback_total_time = profiling.objective_value.total_time
        + profiling.objective_gradient.total_time
        + profiling.equality_values.total_time
        + profiling.inequality_values.total_time
        + profiling.equality_jacobian_values.total_time
        + profiling.inequality_jacobian_values.total_time
        + profiling.lagrangian_hessian_values.total_time;
    let accounted_without_unaccounted = callback_total_time
        + profiling.qp_setup_time
        + profiling.qp_solve_time
        + profiling.preprocessing_time;

    assert!(profiling.objective_value.calls >= iteration_count);
    assert_eq!(profiling.objective_gradient.calls, iteration_count);
    assert_eq!(profiling.equality_jacobian_values.calls, iteration_count);
    assert_eq!(profiling.inequality_jacobian_values.calls, iteration_count);
    assert_eq!(profiling.lagrangian_hessian_values.calls, iteration_count);
    assert_eq!(profiling.qp_setups, iteration_count);
    assert_eq!(profiling.qp_solves, iteration_count);
    assert!(profiling.total_time >= accounted_without_unaccounted);
    assert_eq!(
        profiling.unaccounted_time,
        profiling
            .total_time
            .checked_sub(accounted_without_unaccounted)
            .unwrap_or(Duration::ZERO),
    );
    match backend {
        CallbackBackend::Aot => {
            assert_eq!(profiling.backend_timing.function_creation_time, None);
            assert_eq!(profiling.backend_timing.jit_time, None);
        }
        CallbackBackend::Jit => {
            assert!(profiling.backend_timing.function_creation_time.is_some());
            assert!(profiling.backend_timing.jit_time.is_some());
        }
    }
}
