#![cfg(feature = "ipopt")]

use approx::assert_abs_diff_eq;
use optimization::{
    CCS, CompiledNlpProblem, ConstraintBounds, FilterAcceptanceMode,
    InteriorPointAdaptiveMuGlobalization, InteriorPointAdaptiveMuOracle,
    InteriorPointAlphaForYStrategy, InteriorPointBoundMultiplierInitMethod,
    InteriorPointCorrectorType, InteriorPointIterationEvent, InteriorPointIterationPhase,
    InteriorPointIterationSnapshot, InteriorPointLinearSolver, InteriorPointMuStrategy,
    InteriorPointOptions, InteriorPointQualityFunctionBalancingTerm,
    InteriorPointQualityFunctionCentrality, InteriorPointQualityFunctionNorm,
    InteriorPointSecondOrderCorrectionMethod, InteriorPointSolveError, InteriorPointStatusKind,
    InteriorPointStepKind, InteriorPointTermination, IpoptIterationPhase, IpoptIterationSnapshot,
    IpoptMuStrategy, IpoptOptions, IpoptRawOption, IpoptRawStatus, IpoptSolveError,
    ParameterMatrix, apply_native_spral_parity_to_ipopt_options,
    apply_native_spral_parity_to_nlip_options, solve_nlp_interior_point,
    solve_nlp_interior_point_with_callback, solve_nlp_interior_point_with_control_callback,
    solve_nlp_ipopt, solve_nlp_ipopt_with_control_callback,
};
use rstest::rstest;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

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

struct NonFiniteFirstTrialQuadraticProblem;

impl CompiledNlpProblem for NonFiniteFirstTrialQuadraticProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("nonfinite first-trial quadratic problem has no parameters")
    }

    fn equality_count(&self) -> usize {
        0
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        if (x[0] - 1.0).abs() <= 1e-10 {
            f64::NAN
        } else {
            (x[0] - 1.0).powi(2)
        }
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out.copy_from_slice(&[2.0 * (x[0] - 1.0)]);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        static EMPTY: OnceLock<CCS> = OnceLock::new();
        EMPTY.get_or_init(|| CCS::empty(0, 1))
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
        static EMPTY: OnceLock<CCS> = OnceLock::new();
        EMPTY.get_or_init(|| CCS::empty(0, 1))
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
        static HESSIAN_CCS: OnceLock<CCS> = OnceLock::new();
        HESSIAN_CCS.get_or_init(|| CCS::lower_triangular_dense(1))
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out.copy_from_slice(&[2.0]);
    }
}

struct FixedVariableQuadraticProblem;

impl CompiledNlpProblem for FixedVariableQuadraticProblem {
    fn dimension(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("fixed-variable quadratic problem has no parameters")
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        Some(ConstraintBounds {
            lower: Some(vec![Some(1.0), None]),
            upper: Some(vec![Some(1.0), None]),
        })
    }

    fn equality_count(&self) -> usize {
        0
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 3.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out.copy_from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 3.0)]);
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

struct SquareEqualityQuadraticProblem;

impl CompiledNlpProblem for SquareEqualityQuadraticProblem {
    fn dimension(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("square equality quadratic problem has no parameters")
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        None
    }

    fn equality_count(&self) -> usize {
        2
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] - 4.0).powi(2) + 3.0 * (x[1] + 2.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out.copy_from_slice(&[2.0 * (x[0] - 4.0), 6.0 * (x[1] + 2.0)]);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        static JAC: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        JAC.get_or_init(|| CCS::new(2, 2, vec![0, 1, 2], vec![0, 1]))
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0]);
    }

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out.copy_from_slice(&[1.0, 1.0]);
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
        out.copy_from_slice(&[2.0, 0.0, 6.0]);
    }
}

struct LinearlyConstrainedQuadraticProblem;

struct ImpossibleSquareEqualityProblem;

struct RestorationObjectiveEvalErrorProblem {
    nan_radius: f64,
}

impl CompiledNlpProblem for ImpossibleSquareEqualityProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("impossible square equality problem has no parameters")
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        None
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        0.5 * x[0] * x[0]
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0];
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        static JAC: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        JAC.get_or_init(|| CCS::new(1, 1, vec![0, 1], vec![0]))
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0] * x[0] + 1.0;
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = 2.0 * x[0];
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        static EMPTY: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| CCS::empty(0, 1))
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
        HESSIAN_CCS.get_or_init(|| CCS::lower_triangular_dense(1))
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = 1.0 + 2.0 * equality_multipliers[0];
    }
}

impl CompiledNlpProblem for RestorationObjectiveEvalErrorProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("restoration eval-error problem has no parameters")
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        None
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        if x[0].abs() <= self.nan_radius {
            f64::NAN
        } else {
            0.5 * x[0] * x[0]
        }
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0];
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        static JAC: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        JAC.get_or_init(|| CCS::new(1, 1, vec![0, 1], vec![0]))
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0] * x[0] + 1.0;
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = 2.0 * x[0];
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        static EMPTY: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| CCS::empty(0, 1))
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
        HESSIAN_CCS.get_or_init(|| CCS::lower_triangular_dense(1))
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = 1.0 + 2.0 * equality_multipliers[0];
    }
}

impl CompiledNlpProblem for LinearlyConstrainedQuadraticProblem {
    fn dimension(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("linearly constrained quadratic problem has no parameters")
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        None
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        1
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] - 2.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out.copy_from_slice(&[2.0 * (x[0] - 2.0), 4.0 * (x[1] + 1.0)]);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        static JAC: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        JAC.get_or_init(|| CCS::new(1, 2, vec![0, 1, 2], vec![0, 0]))
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0] + x[1] - 1.0;
    }

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out.copy_from_slice(&[1.0, 1.0]);
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        static JAC: std::sync::OnceLock<CCS> = std::sync::OnceLock::new();
        JAC.get_or_init(|| CCS::new(1, 2, vec![0, 1, 1], vec![0]))
    }

    fn inequality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0] - 0.75;
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out.copy_from_slice(&[1.0]);
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
        out.copy_from_slice(&[2.0, 0.0, 4.0]);
    }
}

#[derive(Debug)]
struct AcceptedTracePoint {
    iteration: usize,
    objective: f64,
    primal_inf: f64,
    dual_inf: f64,
    barrier_parameter: f64,
    has_barrier_parameter: bool,
    regularization_size: Option<f64>,
    alpha_pr: Option<f64>,
    line_search_trials: usize,
    step_tag: Option<String>,
    marker: String,
}

fn parse_ipopt_step_tags(journal_output: Option<&str>) -> BTreeMap<usize, String> {
    let mut tags = BTreeMap::new();
    let Some(journal) = journal_output else {
        return tags;
    };
    for line in journal.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
            continue;
        }
        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        let Some(iteration) = tokens.first().and_then(|token| token.parse::<usize>().ok()) else {
            continue;
        };
        let Some(alpha_pr_token) = ipopt_alpha_primal_token(&tokens) else {
            continue;
        };
        let Some(step_char) = alpha_pr_token
            .chars()
            .last()
            .filter(|value| value.is_ascii_alphabetic())
        else {
            continue;
        };
        tags.insert(iteration, step_char.to_string());
    }
    tags
}

fn ipopt_alpha_primal_token<'a>(tokens: &'a [&'a str]) -> Option<&'a str> {
    tokens.windows(2).rev().find_map(|window| {
        let alpha_primal = window[0];
        let line_search_count = window[1];
        if line_search_count.parse::<usize>().is_ok()
            && alpha_primal.contains('e')
            && alpha_primal
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_digit() || ch == '-' || ch == '+')
        {
            Some(alpha_primal)
        } else {
            None
        }
    })
}

fn parse_ipopt_info_strings(journal_output: Option<&str>) -> BTreeMap<usize, String> {
    let mut info_strings = BTreeMap::new();
    let Some(journal) = journal_output else {
        return info_strings;
    };
    for line in journal.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
            continue;
        }
        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        let Some(iteration) = tokens.first().and_then(|token| token.parse::<usize>().ok()) else {
            continue;
        };
        let Some(alpha_index) = tokens.windows(2).rposition(|window| {
            window[1].parse::<usize>().is_ok()
                && window[0].contains('e')
                && window[0]
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_digit() || ch == '-' || ch == '+')
        }) else {
            continue;
        };
        let info_start = alpha_index + 2;
        if info_start < tokens.len() {
            info_strings.insert(iteration, tokens[info_start..].join(""));
        }
    }
    info_strings
}

fn ipopt_info_string_seen(summary: &optimization::IpoptSummary, marker: char) -> bool {
    parse_ipopt_info_strings(summary.journal_output.as_deref())
        .values()
        .any(|info| info.contains(marker))
}

fn ipopt_info_marker_summary(summary: &optimization::IpoptSummary) -> String {
    let info_strings = parse_ipopt_info_strings(summary.journal_output.as_deref());
    ["W", "w", "Tmax", "F+", "F-", "MaxS", "e"]
        .into_iter()
        .filter_map(|marker| {
            let count = info_strings
                .values()
                .filter(|info| info.contains(marker))
                .count();
            (count > 0).then(|| format!("{marker}:{count}"))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn step_tag_summary(trace: &[AcceptedTracePoint]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for point in trace {
        let tag = point
            .step_tag
            .as_deref()
            .filter(|tag| !tag.is_empty())
            .unwrap_or("-");
        *counts.entry(tag.to_string()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(tag, count)| format!("{tag}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn nlip_error_label(error: &InteriorPointSolveError) -> String {
    match error {
        InteriorPointSolveError::InvalidInput(_) => "invalid-input".to_string(),
        InteriorPointSolveError::LinearSolve { .. } => "linear-solve".to_string(),
        InteriorPointSolveError::LineSearchFailed { .. } => "line-search".to_string(),
        InteriorPointSolveError::RestorationFailed { status, .. } => {
            format!("restoration:{status:?}")
        }
        InteriorPointSolveError::LocalInfeasibility { .. } => "local-infeasible".to_string(),
        InteriorPointSolveError::DivergingIterates { .. } => "diverging".to_string(),
        InteriorPointSolveError::CpuTimeExceeded { .. } => "cpu-time".to_string(),
        InteriorPointSolveError::WallTimeExceeded { .. } => "wall-time".to_string(),
        InteriorPointSolveError::UserRequestedStop { .. } => "user-stop".to_string(),
        InteriorPointSolveError::MaxIterations { iterations, .. } => {
            format!("max-iter:{iterations}")
        }
    }
}

fn nlip_step_tag(snapshot: &optimization::InteriorPointIterationSnapshot) -> Option<String> {
    snapshot
        .step_tag
        .map(|value| value.to_string())
        .or_else(|| {
            snapshot
                .filter
                .as_ref()
                .and_then(|filter| filter.accepted_mode)
                .map(|mode| match mode {
                    FilterAcceptanceMode::ObjectiveArmijo => "f".to_string(),
                    FilterAcceptanceMode::ViolationReduction => "h".to_string(),
                })
                .or_else(|| {
                    snapshot.step_kind.map(|kind| match kind {
                        InteriorPointStepKind::Objective => "f".to_string(),
                        InteriorPointStepKind::Feasibility => "h".to_string(),
                        InteriorPointStepKind::Tiny => "t".to_string(),
                    })
                })
        })
}

fn nlip_error_last_accepted_tag(error: &InteriorPointSolveError) -> Option<char> {
    match error {
        InteriorPointSolveError::InvalidInput(_) => None,
        InteriorPointSolveError::LinearSolve { context, .. }
        | InteriorPointSolveError::LineSearchFailed { context, .. }
        | InteriorPointSolveError::RestorationFailed { context, .. }
        | InteriorPointSolveError::LocalInfeasibility { context }
        | InteriorPointSolveError::DivergingIterates { context, .. }
        | InteriorPointSolveError::CpuTimeExceeded { context, .. }
        | InteriorPointSolveError::WallTimeExceeded { context, .. }
        | InteriorPointSolveError::UserRequestedStop { context }
        | InteriorPointSolveError::MaxIterations { context, .. } => context
            .last_accepted_state
            .as_ref()
            .and_then(|snapshot| snapshot.step_tag)
            .or_else(|| {
                context
                    .final_state
                    .as_ref()
                    .and_then(|snapshot| snapshot.step_tag)
            }),
    }
}

fn nlip_result_step_tag_summary(
    result: &Result<optimization::InteriorPointSummary, InteriorPointSolveError>,
) -> String {
    match result {
        Ok(summary) => step_tag_summary(&nlip_accepted_trace(summary)),
        Err(error) => {
            let label = nlip_error_label(error);
            nlip_error_last_accepted_tag(error)
                .map(|tag| format!("err:{label};last:{tag}"))
                .unwrap_or_else(|| format!("err:{label}"))
        }
    }
}

fn nlip_result_has_step_tag(
    result: &Result<optimization::InteriorPointSummary, InteriorPointSolveError>,
    tags: &[char],
) -> bool {
    match result {
        Ok(summary) => summary
            .snapshots
            .iter()
            .any(|snapshot| snapshot.step_tag.is_some_and(|tag| tags.contains(&tag))),
        Err(error) => nlip_error_last_accepted_tag(error).is_some_and(|tag| tags.contains(&tag)),
    }
}

fn ipopt_result_journal_output(
    result: &Result<optimization::IpoptSummary, IpoptSolveError>,
) -> Option<&str> {
    match result {
        Ok(summary) => summary.journal_output.as_deref(),
        Err(IpoptSolveError::Solve { journal_output, .. }) => journal_output.as_deref(),
        Err(_) => None,
    }
}

fn ipopt_result_step_tag_summary(
    result: &Result<optimization::IpoptSummary, IpoptSolveError>,
) -> String {
    match result {
        Ok(summary) => step_tag_summary(&ipopt_accepted_trace(summary)),
        Err(IpoptSolveError::Solve {
            status,
            journal_output,
            ..
        }) => {
            let tags = parse_ipopt_step_tags(journal_output.as_deref());
            let summary = tags
                .values()
                .cloned()
                .map(|tag| AcceptedTracePoint {
                    iteration: 0,
                    objective: 0.0,
                    primal_inf: 0.0,
                    dual_inf: 0.0,
                    barrier_parameter: 0.0,
                    has_barrier_parameter: false,
                    regularization_size: None,
                    alpha_pr: None,
                    line_search_trials: 0,
                    step_tag: Some(tag),
                    marker: String::new(),
                })
                .collect::<Vec<_>>();
            format!("err:{status:?};{}", step_tag_summary(&summary))
        }
        Err(error) => format!("err:{error}"),
    }
}

fn ipopt_result_has_step_tag(
    result: &Result<optimization::IpoptSummary, IpoptSolveError>,
    tags: &[&str],
) -> bool {
    parse_ipopt_step_tags(ipopt_result_journal_output(result))
        .values()
        .any(|tag| tags.contains(&tag.as_str()))
}

fn nlip_accepted_trace(summary: &optimization::InteriorPointSummary) -> Vec<AcceptedTracePoint> {
    summary
        .snapshots
        .iter()
        .filter(|snapshot| {
            snapshot.phase == InteriorPointIterationPhase::AcceptedStep && snapshot.alpha.is_some()
        })
        .map(|snapshot| AcceptedTracePoint {
            iteration: snapshot.iteration,
            objective: snapshot.objective,
            primal_inf: snapshot
                .eq_inf
                .unwrap_or(0.0)
                .max(snapshot.ineq_inf.unwrap_or(0.0)),
            dual_inf: snapshot.dual_inf,
            barrier_parameter: snapshot.barrier_parameter.unwrap_or(0.0),
            has_barrier_parameter: snapshot.barrier_parameter.is_some(),
            regularization_size: positive_optional(snapshot.regularization_size),
            alpha_pr: snapshot.alpha_pr.or(snapshot.alpha),
            line_search_trials: snapshot.line_search_trials,
            step_tag: nlip_step_tag(snapshot),
            marker: optimization::nlip_event_codes_for_events(&snapshot.events),
        })
        .collect()
}

fn ipopt_accepted_trace(summary: &optimization::IpoptSummary) -> Vec<AcceptedTracePoint> {
    let step_tags = parse_ipopt_step_tags(summary.journal_output.as_deref());
    summary
        .snapshots
        .iter()
        .filter(|snapshot| snapshot.iteration > 0)
        .map(|snapshot| AcceptedTracePoint {
            iteration: snapshot.iteration,
            objective: snapshot.objective,
            primal_inf: snapshot.primal_inf,
            dual_inf: snapshot.dual_inf,
            barrier_parameter: snapshot.barrier_parameter,
            has_barrier_parameter: true,
            regularization_size: positive_regularization(snapshot.regularization_size),
            alpha_pr: Some(snapshot.alpha_pr),
            // IPOPT's printed `ls` count includes the accepted trial itself;
            // NLIP snapshots store pure backtracks. Normalize before comparing.
            line_search_trials: snapshot.line_search_trials.saturating_sub(1),
            step_tag: step_tags.get(&snapshot.iteration).cloned(),
            marker: String::new(),
        })
        .collect()
}

fn log_gap(lhs: f64, rhs: f64, floor: f64) -> f64 {
    let lhs = lhs.abs().max(floor).log10();
    let rhs = rhs.abs().max(floor).log10();
    (lhs - rhs).abs()
}

fn positive_optional(value: Option<f64>) -> Option<f64> {
    value.and_then(positive_regularization)
}

fn positive_regularization(value: f64) -> Option<f64> {
    (value.is_finite() && value > 0.0).then_some(value)
}

fn optional_log_gap(lhs: Option<f64>, rhs: Option<f64>, floor: f64) -> Option<f64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(log_gap(lhs, rhs, floor)),
        (Some(lhs), None) if lhs > floor => Some(log_gap(lhs, 0.0, floor)),
        (None, Some(rhs)) if rhs > floor => Some(log_gap(0.0, rhs, floor)),
        _ => None,
    }
}

fn trace_regularization_text(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), |value| format!("{value:.1e}"))
}

fn assert_source_built_spral_ipopt_provenance(summary: &optimization::IpoptSummary) {
    let provenance = summary
        .provenance
        .as_ref()
        .expect("expected IPOPT provenance for source-built SPRAL parity runs");
    let errors = optimization::source_built_spral_parity_ipopt_provenance_errors(provenance);
    if !errors.is_empty() {
        panic!(
            "unsupported source-built SPRAL IPOPT provenance:\n{}\nprovenance={provenance:?}",
            errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

#[derive(Clone, Copy)]
struct AcceptedTraceParityTolerances {
    max_iteration_gap: usize,
    max_step_tag_mismatches: usize,
    max_primal_log_gap: f64,
    max_dual_log_gap: f64,
    max_mu_log_gap: f64,
    max_regularization_log_gap: f64,
}

fn assert_accepted_trace_parity(
    problem_name: &str,
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
    tolerances: AcceptedTraceParityTolerances,
) {
    const PRIMAL_TRACE_FLOOR: f64 = 1.0e-6;
    const REGULARIZATION_TRACE_FLOOR: f64 = 1.0e-20;
    let native_trace = nlip_accepted_trace(native);
    let ipopt_trace = ipopt_accepted_trace(ipopt);
    if compare_verbose_requested() {
        print_accepted_trace_comparison(problem_name, &native_trace, &ipopt_trace);
    }
    let iteration_gap = native_trace.len().abs_diff(ipopt_trace.len());
    assert!(
        iteration_gap <= tolerances.max_iteration_gap,
        "{problem_name} accepted-step count mismatch: native={} ipopt={}",
        native_trace.len(),
        ipopt_trace.len(),
    );
    let compared = native_trace.len().min(ipopt_trace.len());
    let mut step_tag_mismatches = 0usize;
    let mut max_dual_gap_observed = 0.0f64;
    let strict_step_trace =
        tolerances.max_iteration_gap == 0 && tolerances.max_step_tag_mismatches == 0;
    for (native_point, ipopt_point) in native_trace.iter().zip(ipopt_trace.iter()) {
        if let (Some(native_tag), Some(ipopt_tag)) = (&native_point.step_tag, &ipopt_point.step_tag)
            && native_tag != ipopt_tag
        {
            step_tag_mismatches += 1;
        }
        assert!(
            log_gap(
                native_point.primal_inf,
                ipopt_point.primal_inf,
                PRIMAL_TRACE_FLOOR,
            ) <= tolerances.max_primal_log_gap,
            "{problem_name} primal trace diverged at accepted iter {} (native iter {} / ipopt iter {}): native={} ipopt={}",
            native_point.iteration.min(ipopt_point.iteration),
            native_point.iteration,
            ipopt_point.iteration,
            native_point.primal_inf,
            ipopt_point.primal_inf,
        );
        max_dual_gap_observed = max_dual_gap_observed.max(log_gap(
            native_point.dual_inf,
            ipopt_point.dual_inf,
            1.0e-12,
        ));
        if native_point.has_barrier_parameter
            && ipopt_point.has_barrier_parameter
            && native_point
                .barrier_parameter
                .max(ipopt_point.barrier_parameter)
                > 1.0e-10
        {
            assert!(
                log_gap(
                    native_point.barrier_parameter,
                    ipopt_point.barrier_parameter,
                    1.0e-16,
                ) <= tolerances.max_mu_log_gap,
                "{problem_name} barrier trace diverged at accepted iter {} (native iter {} / ipopt iter {}): native={} ipopt={}",
                native_point.iteration.min(ipopt_point.iteration),
                native_point.iteration,
                ipopt_point.iteration,
                native_point.barrier_parameter,
                ipopt_point.barrier_parameter,
            );
        }
        if let Some(regularization_gap) = optional_log_gap(
            native_point.regularization_size,
            ipopt_point.regularization_size,
            REGULARIZATION_TRACE_FLOOR,
        ) {
            assert!(
                regularization_gap <= tolerances.max_regularization_log_gap,
                "{problem_name} regularization trace diverged at accepted iter {} (native iter {} / ipopt iter {}): native={} ipopt={} log_gap={regularization_gap:.3}",
                native_point.iteration.min(ipopt_point.iteration),
                native_point.iteration,
                ipopt_point.iteration,
                trace_regularization_text(native_point.regularization_size),
                trace_regularization_text(ipopt_point.regularization_size),
            );
        }
        if strict_step_trace {
            assert_eq!(
                native_point.line_search_trials,
                ipopt_point.line_search_trials,
                "{problem_name} line-search trace diverged at accepted iter {} (native iter {} / ipopt iter {})",
                native_point.iteration.min(ipopt_point.iteration),
                native_point.iteration,
                ipopt_point.iteration,
            );
            if let (Some(native_alpha), Some(ipopt_alpha)) =
                (native_point.alpha_pr, ipopt_point.alpha_pr)
            {
                let alpha_delta = (native_alpha - ipopt_alpha).abs();
                let alpha_tol = 1.0e-10_f64.max(1.0e-8 * ipopt_alpha.abs());
                assert!(
                    alpha_delta <= alpha_tol,
                    "{problem_name} accepted alpha trace diverged at accepted iter {} (native iter {} / ipopt iter {}): native={native_alpha:.12e} ipopt={ipopt_alpha:.12e} delta={alpha_delta:.3e}",
                    native_point.iteration.min(ipopt_point.iteration),
                    native_point.iteration,
                    ipopt_point.iteration,
                );
            }
        }
    }
    assert!(
        step_tag_mismatches <= tolerances.max_step_tag_mismatches,
        "{problem_name} step-tag mismatches too large: {step_tag_mismatches} over {compared} accepted steps",
    );
    assert!(
        max_dual_gap_observed <= tolerances.max_dual_log_gap,
        "{problem_name} dual trace gap exceeds budget: max_gap={max_dual_gap_observed:.3} budget={:.3}",
        tolerances.max_dual_log_gap,
    );
}

fn assert_native_event_seen(
    problem_name: &str,
    summary: &optimization::InteriorPointSummary,
    event: InteriorPointIterationEvent,
) {
    assert!(
        summary
            .snapshots
            .iter()
            .any(|snapshot| snapshot.events.contains(&event)),
        "{problem_name} did not exercise expected NLIP event {event:?}"
    );
}

fn assert_ipopt_info_string_seen(
    problem_name: &str,
    summary: &optimization::IpoptSummary,
    marker: char,
) {
    assert!(
        ipopt_info_string_seen(summary, marker),
        "{problem_name} did not exercise expected IPOPT info marker {marker:?}"
    );
}

fn print_accepted_trace_comparison(
    problem_name: &str,
    native_trace: &[AcceptedTracePoint],
    ipopt_trace: &[AcceptedTracePoint],
) {
    eprintln!("[compare] accepted trace for {problem_name}");
    eprintln!(
        "[compare] {:>4} | {:>4} {:>2} {:>10} {:>10} {:>10} {:>8} {:>8} {:>2} {:<12} || {:>4} {:>2} {:>10} {:>10} {:>10} {:>8} {:>8} {:>2}",
        "idx",
        "n_it",
        "nt",
        "n_obj",
        "n_pr",
        "n_du",
        "n_reg",
        "n_alpha",
        "nl",
        "n_evt",
        "i_it",
        "it",
        "i_obj",
        "i_pr",
        "i_du",
        "i_reg",
        "i_alpha",
        "il",
    );
    for (index, (native, ipopt)) in native_trace.iter().zip(ipopt_trace.iter()).enumerate() {
        eprintln!(
            "[compare] {:>4} | {:>4} {:>2} {:>10.3e} {:>10.3e} {:>10.3e} {:>8} {:>8.2e} {:>2} {:<12} || {:>4} {:>2} {:>10.3e} {:>10.3e} {:>10.3e} {:>8} {:>8.2e} {:>2}",
            index,
            native.iteration,
            native.step_tag.as_deref().unwrap_or("-"),
            native.objective,
            native.primal_inf,
            native.dual_inf,
            trace_regularization_text(native.regularization_size),
            native.alpha_pr.unwrap_or(f64::NAN),
            native.line_search_trials,
            native.marker,
            ipopt.iteration,
            ipopt.step_tag.as_deref().unwrap_or("-"),
            ipopt.objective,
            ipopt.primal_inf,
            ipopt.dual_inf,
            trace_regularization_text(ipopt.regularization_size),
            ipopt.alpha_pr.unwrap_or(f64::NAN),
            ipopt.line_search_trials,
        );
    }
    if native_trace.len() != ipopt_trace.len() {
        eprintln!(
            "[compare] accepted trace length differs: native={} ipopt={}",
            native_trace.len(),
            ipopt_trace.len()
        );
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

fn source_built_spral_ipopt_environment_available() -> bool {
    static AVAILABLE: OnceLock<Result<(), String>> = OnceLock::new();
    match AVAILABLE.get_or_init(|| {
        optimization::validate_source_built_spral_parity_preflight()
            .map(|_| ())
            .map_err(|errors| errors.join("; "))
    }) {
        Ok(()) => true,
        Err(error) => {
            if optimization::native_spral_parity_fail_closed_requested() {
                panic!("source-built SPRAL parity preflight is required: {error}");
            }
            eprintln!("skipping source-built SPRAL/IPOPT comparison: {error}");
            false
        }
    }
}

fn native_spral_compare_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("native SPRAL/IPOPT comparison lock poisoned")
}

macro_rules! skip_without_native_spral {
    () => {
        let _native_spral_compare_guard = native_spral_compare_guard();
        if !source_built_spral_ipopt_environment_available() {
            return;
        }
    };
}

fn native_options() -> InteriorPointOptions {
    let mut options = InteriorPointOptions {
        max_iters: 120,
        acceptable_iter: 0,
        verbose: compare_verbose_requested(),
        ..InteriorPointOptions::default()
    };
    apply_native_spral_parity_to_nlip_options(&mut options);
    options
}

fn native_options_with(configure: impl FnOnce(&mut InteriorPointOptions)) -> InteriorPointOptions {
    let mut options = native_options();
    configure(&mut options);
    options
}

fn hs071_native_options() -> InteriorPointOptions {
    let mut options = InteriorPointOptions {
        max_iters: 300,
        dual_tol: 1.0e-5,
        overall_tol: 1.0e-5,
        acceptable_iter: 0,
        verbose: compare_verbose_requested(),
        ..InteriorPointOptions::default()
    };
    apply_native_spral_parity_to_nlip_options(&mut options);
    options
}

fn ipopt_options() -> IpoptOptions {
    let verbose = compare_verbose_requested();
    let mut options = IpoptOptions {
        max_iters: 120,
        print_level: if verbose { 5 } else { 0 },
        suppress_banner: !verbose,
        ..IpoptOptions::default()
    };
    apply_native_spral_parity_to_ipopt_options(&mut options);
    options
}

fn ipopt_options_with(configure: impl FnOnce(&mut IpoptOptions)) -> IpoptOptions {
    let mut options = ipopt_options();
    configure(&mut options);
    options
}

fn enable_ipopt_trace_journal(options: &mut IpoptOptions) {
    options.print_level = 0;
    options.journal_print_level = Some(5);
    options.suppress_banner = true;
    options
        .raw_options
        .push(IpoptRawOption::text("print_info_string", "yes"));
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0.0, |acc, (lhs_value, rhs_value)| {
            acc.max((lhs_value - rhs_value).abs())
        })
}

fn assert_vec_close(label: &str, native: &[f64], ipopt: &[f64], tolerance: f64) {
    assert_eq!(
        native.len(),
        ipopt.len(),
        "{label} length mismatch: native={native:?} ipopt={ipopt:?}"
    );
    let delta = max_abs_diff(native, ipopt);
    assert!(
        delta <= tolerance,
        "{label} mismatch: delta={delta:.3e} tolerance={tolerance:.3e} native={native:?} ipopt={ipopt:?}"
    );
}

fn solve_native_initial_snapshot<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    options: InteriorPointOptions,
) -> InteriorPointIterationSnapshot {
    match solve_nlp_interior_point(problem, x0, &[], &options) {
        Err(InteriorPointSolveError::MaxIterations {
            iterations: 0,
            context,
        }) => context
            .final_state
            .expect("NLIP max-iter-0 failure should retain the initial iterate snapshot"),
        other => panic!("NLIP max-iter-0 initial snapshot probe mismatch: {other:?}"),
    }
}

fn solve_ipopt_initial_snapshot<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    options: IpoptOptions,
) -> IpoptIterationSnapshot {
    match solve_nlp_ipopt(problem, x0, &[], &options) {
        Err(IpoptSolveError::Solve {
            status,
            iterations: 0,
            snapshots,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            snapshots
                .into_iter()
                .find(|snapshot| snapshot.iteration == 0)
                .expect("IPOPT max-iter-0 failure should retain the initial iterate snapshot")
        }
        other => panic!("IPOPT max-iter-0 initial snapshot probe mismatch: {other:?}"),
    }
}

fn optional_abs_diff(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    lhs.zip(rhs).map(|(lhs, rhs)| (lhs - rhs).abs())
}

fn sci(value: f64) -> String {
    format!("{value:.3e}")
}

fn sci_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), sci)
}

fn accepted_nlip_trace(
    summary: &optimization::InteriorPointSummary,
) -> Vec<&InteriorPointIterationSnapshot> {
    summary
        .snapshots
        .iter()
        .filter(|snapshot| snapshot.phase == InteriorPointIterationPhase::AcceptedStep)
        .collect()
}

fn first_accepted_step_delta_gap(
    nlip_trace: &[&InteriorPointIterationSnapshot],
    ipopt_trace: &[IpoptIterationSnapshot],
    threshold: f64,
) -> Option<(usize, usize, usize, f64)> {
    let limit = nlip_trace.len().min(ipopt_trace.len());
    (1..limit).find_map(|index| {
        let nlip_prev = &nlip_trace[index - 1].x;
        let nlip_current = &nlip_trace[index].x;
        let ipopt_prev = &ipopt_trace[index - 1].x;
        let ipopt_current = &ipopt_trace[index].x;
        let gap = nlip_current
            .iter()
            .zip(nlip_prev.iter())
            .zip(ipopt_current.iter().zip(ipopt_prev.iter()))
            .fold(
                0.0_f64,
                |acc, ((nlip_current, nlip_prev), (ipopt_current, ipopt_prev))| {
                    let nlip_step = *nlip_current - *nlip_prev;
                    let ipopt_step = *ipopt_current - *ipopt_prev;
                    acc.max((nlip_step - ipopt_step).abs())
                },
            );
        (gap > threshold).then_some((
            index,
            nlip_trace[index].iteration,
            ipopt_trace[index].iteration,
            gap,
        ))
    })
}

fn print_compare_summary(
    problem_name: &str,
    backend: Option<CallbackBackend>,
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
) {
    let backend_label = backend.map_or("direct", CallbackBackend::label);
    let nlip_trace = accepted_nlip_trace(native);
    let nlip_last = native
        .last_accepted_state
        .as_ref()
        .unwrap_or(&native.final_state);
    let ipopt_last = ipopt.snapshots.last();
    let accepted_step_delta_gap =
        first_accepted_step_delta_gap(&nlip_trace, &ipopt.snapshots, 1.0e-10)
            .map(|(index, nlip_iter, ipopt_iter, gap)| {
                format!("index={index},nlip_iter={nlip_iter},ipopt_iter={ipopt_iter},gap={gap:.3e}")
            })
            .unwrap_or_else(|| "--".to_string());
    let alpha_gap = optional_abs_diff(
        nlip_last.alpha_pr,
        ipopt_last.map(|snapshot| snapshot.alpha_pr),
    );
    let regularization_gap = optional_abs_diff(
        nlip_last.regularization_size,
        ipopt_last.map(|snapshot| snapshot.regularization_size),
    );
    eprintln!(
        "[parity] {problem_name} backend={backend_label} nlip_solver={:?} ipopt_solver=spral nlip_iters={} ipopt_iters={} max_x_gap={} obj_gap={} alpha_gap={} reg_gap={} first_accepted_step_delta_gap={} nlip_residuals=(p:{},d:{},c:{}) ipopt_residuals=(p:{},d:{},c:{}) timing=(nlip:{:.3}s,ipopt:{:.3}s)",
        native.linear_solver,
        native.iterations,
        ipopt.iterations,
        sci(max_abs_diff(&native.x, &ipopt.x)),
        sci((native.objective - ipopt.objective).abs()),
        sci_optional(alpha_gap),
        sci_optional(regularization_gap),
        accepted_step_delta_gap,
        sci(native.primal_inf_norm),
        sci(native.dual_inf_norm),
        sci(native.complementarity_inf_norm),
        sci(ipopt.primal_inf_norm),
        sci(ipopt.dual_inf_norm),
        sci(ipopt.complementarity_inf_norm),
        native.profiling.total_time.as_secs_f64(),
        ipopt.profiling.total_time.as_secs_f64(),
    );
    eprintln!(
        "[parity] {problem_name} provenance ipopt={:?} threads=(rayon:{:?},omp:{:?},omp_cancellation:{:?})",
        ipopt.provenance,
        std::env::var("RAYON_NUM_THREADS").ok(),
        std::env::var("OMP_NUM_THREADS").ok(),
        std::env::var("OMP_CANCELLATION").ok(),
    );
}

fn maybe_print_native_trace(
    problem_name: &str,
    snapshots: &[optimization::InteriorPointIterationSnapshot],
) {
    if !compare_verbose_requested() {
        return;
    }
    eprintln!("NLIP trace for {problem_name}:");
    for snapshot in snapshots
        .iter()
        .filter(|snapshot| snapshot.phase == InteriorPointIterationPhase::AcceptedStep)
    {
        eprintln!(
            "  iter={:>3} obj={:.6e} eq={:.6e} ineq={:.6e} dual={:.6e} reg={} alpha={} x={:?}",
            snapshot.iteration,
            snapshot.objective,
            snapshot.eq_inf.unwrap_or(0.0),
            snapshot.ineq_inf.unwrap_or(0.0),
            snapshot.dual_inf,
            trace_regularization_text(snapshot.regularization_size),
            snapshot
                .alpha_pr
                .map_or_else(|| "--".to_string(), |value| format!("{value:.6e}")),
            snapshot.x
        );
    }
}

fn assert_ipopt_component_snapshots(problem_name: &str, summary: &optimization::IpoptSummary) {
    assert!(
        !summary.snapshots.is_empty(),
        "{problem_name} IPOPT comparison produced no iteration snapshots"
    );
    for snapshot in &summary.snapshots {
        assert_eq!(
            snapshot.curr_grad_f.len(),
            snapshot.x.len(),
            "{problem_name} IPOPT curr_grad_f length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.curr_jac_c_t_y_c.len(),
            snapshot.x.len(),
            "{problem_name} IPOPT curr_jac_c_t_y_c length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.curr_jac_d_t_y_d.len(),
            snapshot.x.len(),
            "{problem_name} IPOPT curr_jac_d_t_y_d length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.curr_grad_lag_x.len(),
            snapshot.x.len(),
            "{problem_name} IPOPT curr_grad_lag_x length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_x_stationarity.len(),
            snapshot.x.len(),
            "{problem_name} IPOPT kkt_x_stationarity length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.curr_grad_lag_s.len(),
            snapshot.internal_slack.len(),
            "{problem_name} IPOPT curr_grad_lag_s length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_slack_stationarity.len(),
            snapshot.internal_slack.len(),
            "{problem_name} IPOPT kkt_slack_stationarity length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_equality_residual.len(),
            snapshot.equality_multipliers.len(),
            "{problem_name} IPOPT equality residual length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_inequality_residual.len(),
            snapshot.inequality_multipliers.len(),
            "{problem_name} IPOPT inequality residual length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_slack_complementarity.len(),
            snapshot.internal_slack.len(),
            "{problem_name} IPOPT slack complementarity length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_slack_sigma.len(),
            snapshot.internal_slack.len(),
            "{problem_name} IPOPT slack sigma length mismatch at iteration {}",
            snapshot.iteration
        );
        assert_eq!(
            snapshot.kkt_slack_distance.len(),
            snapshot.internal_slack.len(),
            "{problem_name} IPOPT slack distance length mismatch at iteration {}",
            snapshot.iteration
        );
        assert!(
            snapshot.curr_barrier_error.is_finite()
                && snapshot.curr_primal_infeasibility.is_finite()
                && snapshot.curr_dual_infeasibility.is_finite()
                && snapshot.curr_complementarity.is_finite()
                && snapshot.curr_nlp_error.is_finite(),
            "{problem_name} IPOPT component scalar snapshot is not finite at iteration {}",
            snapshot.iteration
        );
    }
}

#[derive(Clone, Copy)]
struct FinalParityTolerances {
    x: f64,
    objective: f64,
    primal: f64,
    native_dual: f64,
    ipopt_dual: f64,
    complementarity: f64,
}

fn assert_native_matches_ipopt(
    problem_name: &str,
    backend: Option<CallbackBackend>,
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
    x_epsilon: f64,
    objective_epsilon: f64,
) {
    assert_native_matches_ipopt_with_final_tolerances(
        problem_name,
        backend,
        native,
        ipopt,
        FinalParityTolerances {
            x: x_epsilon,
            objective: objective_epsilon,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

fn assert_native_matches_ipopt_with_final_tolerances(
    problem_name: &str,
    backend: Option<CallbackBackend>,
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
    tolerances: FinalParityTolerances,
) {
    print_compare_summary(problem_name, backend, native, ipopt);
    assert_eq!(
        native.linear_solver,
        optimization::InteriorPointLinearSolver::SpralSrc
    );
    assert_source_built_spral_ipopt_provenance(ipopt);
    assert_ipopt_component_snapshots(problem_name, ipopt);
    assert_abs_diff_eq!(
        native.objective,
        ipopt.objective,
        epsilon = tolerances.objective
    );
    assert!(
        max_abs_diff(&native.x, &ipopt.x) <= tolerances.x,
        "native/ipopt x mismatch for {problem_name} backend={}: native={:?} ipopt={:?}",
        backend.map_or("direct", CallbackBackend::label),
        native.x,
        ipopt.x,
    );
    assert!(native.primal_inf_norm <= tolerances.primal);
    assert!(native.dual_inf_norm <= tolerances.native_dual);
    assert!(native.complementarity_inf_norm <= tolerances.complementarity);
    assert!(ipopt.primal_inf_norm <= tolerances.primal);
    assert!(ipopt.dual_inf_norm <= tolerances.ipopt_dual);
    assert!(ipopt.complementarity_inf_norm <= tolerances.complementarity);
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
    let mut snapshots = Vec::new();
    let solve_result =
        solve_nlp_interior_point_with_callback(problem, x0, parameters, &options, |snapshot| {
            snapshots.push(snapshot.clone());
        });
    maybe_print_native_trace("native solve", &snapshots);
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
    solve_ipopt_with_options_ok(problem, x0, parameters, ipopt_options())
}

fn solve_ipopt_with_options_ok<P: CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: IpoptOptions,
) -> optimization::IpoptSummary {
    optimization::assert_source_built_spral_parity_preflight();
    let solve_result = solve_nlp_ipopt(problem, x0, parameters, &options);
    assert!(solve_result.is_ok(), "Ipopt solve failed: {solve_result:?}");
    match solve_result {
        Ok(summary) => {
            assert_source_built_spral_ipopt_provenance(&summary);
            summary
        }
        Err(err) => unreachable!("asserted success: {err}"),
    }
}

#[derive(Clone, Copy, Debug)]
struct WatchdogActivationProfile {
    trigger: usize,
    trial_max: usize,
    beta: f64,
    max_soc: usize,
    disable_fast_mu: bool,
    disable_tiny_step: bool,
}

impl WatchdogActivationProfile {
    fn label(self) -> String {
        format!(
            "tr{}_wm{}_b{:.2}_soc{}_fast{}_tiny{}",
            self.trigger,
            self.trial_max,
            self.beta,
            self.max_soc,
            if self.disable_fast_mu { "off" } else { "on" },
            if self.disable_tiny_step { "off" } else { "on" },
        )
    }

    fn apply_native(self, options: &mut InteriorPointOptions) {
        options.max_iters = 220;
        options.watchdog_shortened_iter_trigger = self.trigger;
        options.watchdog_trial_iter_max = self.trial_max;
        options.line_search_beta = self.beta;
        options.max_second_order_corrections = self.max_soc;
        options.second_order_correction = self.max_soc > 0;
        if self.disable_fast_mu {
            options.mu_allow_fast_monotone_decrease = false;
        }
        if self.disable_tiny_step {
            options.tiny_step_tol = 0.0;
        }
    }

    fn apply_ipopt(self, options: &mut IpoptOptions) {
        options.max_iters = 220;
        enable_ipopt_trace_journal(options);
        options.raw_options.push(IpoptRawOption::integer(
            "watchdog_shortened_iter_trigger",
            self.trigger as i32,
        ));
        options.raw_options.push(IpoptRawOption::integer(
            "watchdog_trial_iter_max",
            self.trial_max as i32,
        ));
        options
            .raw_options
            .push(IpoptRawOption::number("alpha_red_factor", self.beta));
        options
            .raw_options
            .push(IpoptRawOption::integer("max_soc", self.max_soc as i32));
        if self.disable_fast_mu {
            options.raw_options.push(IpoptRawOption::text(
                "mu_allow_fast_monotone_decrease",
                "no",
            ));
        }
        if self.disable_tiny_step {
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_tol", 0.0));
        }
    }
}

fn accepted_trace_strictly_clean(
    native: &optimization::InteriorPointSummary,
    ipopt: &optimization::IpoptSummary,
) -> bool {
    let native_trace = nlip_accepted_trace(native);
    let ipopt_trace = ipopt_accepted_trace(ipopt);
    native_trace.len() == ipopt_trace.len()
        && native_trace
            .iter()
            .zip(ipopt_trace.iter())
            .all(|(native_point, ipopt_point)| {
                let tags_match = match (&native_point.step_tag, &ipopt_point.step_tag) {
                    (Some(native_tag), Some(ipopt_tag)) => native_tag == ipopt_tag,
                    _ => true,
                };
                let alpha_match = match (native_point.alpha_pr, ipopt_point.alpha_pr) {
                    (Some(native_alpha), Some(ipopt_alpha)) => {
                        (native_alpha - ipopt_alpha).abs()
                            <= 1.0e-10_f64.max(1.0e-8 * ipopt_alpha.abs())
                    }
                    _ => true,
                };
                let mu_match = !native_point.has_barrier_parameter
                    || !ipopt_point.has_barrier_parameter
                    || native_point
                        .barrier_parameter
                        .max(ipopt_point.barrier_parameter)
                        <= 1.0e-10
                    || log_gap(
                        native_point.barrier_parameter,
                        ipopt_point.barrier_parameter,
                        1.0e-16,
                    ) <= 0.05;
                let regularization_match = optional_log_gap(
                    native_point.regularization_size,
                    ipopt_point.regularization_size,
                    1.0e-20,
                )
                .is_none_or(|gap| gap <= 0.05);
                tags_match
                    && alpha_match
                    && mu_match
                    && regularization_match
                    && native_point.line_search_trials == ipopt_point.line_search_trials
                    && log_gap(native_point.primal_inf, ipopt_point.primal_inf, 1.0e-6) <= 0.05
                    && log_gap(native_point.dual_inf, ipopt_point.dual_inf, 1.0e-12) <= 0.05
            })
}

fn run_watchdog_activation_sweep_case<P: CompiledNlpProblem>(
    case_name: &str,
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    profile: WatchdogActivationProfile,
) -> bool {
    let native = solve_nlp_interior_point_with_callback(
        problem,
        x0,
        parameters,
        &native_options_with(|options| profile.apply_native(options)),
        |_| {},
    );
    let Ok(native) = native else {
        println!(
            "[watchdog-sweep] {case_name}/{} native_failed={native:?}",
            profile.label()
        );
        return false;
    };
    let ipopt = solve_nlp_ipopt(
        problem,
        x0,
        parameters,
        &ipopt_options_with(|options| profile.apply_ipopt(options)),
    );
    let Ok(ipopt) = ipopt else {
        println!(
            "[watchdog-sweep] {case_name}/{} ipopt_failed={ipopt:?}",
            profile.label()
        );
        return false;
    };
    assert_source_built_spral_ipopt_provenance(&ipopt);
    let nlip_armed = native.snapshots.iter().any(|snapshot| {
        snapshot
            .events
            .contains(&InteriorPointIterationEvent::WatchdogArmed)
    });
    let nlip_activated = native.snapshots.iter().any(|snapshot| {
        snapshot
            .events
            .contains(&InteriorPointIterationEvent::WatchdogActivated)
    });
    let ipopt_w = ipopt_info_string_seen(&ipopt, 'W');
    let native_trace = nlip_accepted_trace(&native);
    let ipopt_trace = ipopt_accepted_trace(&ipopt);
    let trace_clean = accepted_trace_strictly_clean(&native, &ipopt);
    let native_steps = native_trace.len();
    let ipopt_steps = ipopt_trace.len();
    let ipopt_info = ipopt_info_marker_summary(&ipopt);
    let native_tags = step_tag_summary(&native_trace);
    let ipopt_tags = step_tag_summary(&ipopt_trace);
    println!(
        "[watchdog-sweep] {case_name}/{} nlip_steps={native_steps} ipopt_steps={ipopt_steps} nlip_armed={nlip_armed} nlip_activated={nlip_activated} ipopt_w={ipopt_w} trace_clean={trace_clean} nlip_tags={native_tags} ipopt_tags={ipopt_tags} ipopt_info={ipopt_info}",
        profile.label(),
    );
    nlip_activated && ipopt_w && trace_clean
}

#[rstest]
fn compare_native_and_ipopt_on_equality_constrained_rosenbrock(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    skip_without_native_spral!();
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
    skip_without_native_spral!();
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
    skip_without_native_spral!();
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let native = solve_native_ok(&problem, &[0.0, 0.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[0.0, 0.0], &[]);
    assert_native_matches_ipopt("simple_nlp", Some(backend), &native, &ipopt, 1e-5, 1e-5);
}

#[rstest]
fn compare_native_and_ipopt_on_hs021(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    skip_without_native_spral!();
    let problem = build_problem_ok(hs021_problem(backend), backend);
    let native = solve_native_ok(&problem, &[2.0, 2.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[2.0, 2.0], &[]);
    assert_native_matches_ipopt("hs021", Some(backend), &native, &ipopt, 1e-5, 1e-5);
}

#[rstest]
fn compare_native_and_ipopt_on_hs035(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    skip_without_native_spral!();
    let problem = build_problem_ok(hs035_problem(backend), backend);
    let native = solve_native_ok(&problem, &[0.5, 0.5, 0.5], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[0.5, 0.5, 0.5], &[]);
    assert_native_matches_ipopt("hs035", Some(backend), &native, &ipopt, 1e-4, 1e-5);
}

#[rstest]
fn compare_native_and_ipopt_on_hs071(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    skip_without_native_spral!();
    let problem = build_problem_ok(hs071_problem(backend), backend);
    let native =
        solve_native_with_options_ok(&problem, &[1.0, 5.0, 5.0, 1.0], &[], hs071_native_options());
    let ipopt = solve_ipopt_ok(&problem, &[1.0, 5.0, 5.0, 1.0], &[]);
    assert_native_matches_ipopt("hs071", Some(backend), &native, &ipopt, 5e-3, 1e-4);
    assert_native_event_seen(
        "hs071",
        &native,
        InteriorPointIterationEvent::FilterAccepted,
    );
    assert_native_event_seen(
        "hs071",
        &native,
        InteriorPointIterationEvent::BarrierParameterUpdated,
    );
}

#[rstest]
fn compare_native_and_ipopt_on_parameterized_problem(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    skip_without_native_spral!();
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
    skip_without_native_spral!();
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let x0 = hanging_chain_initial_guess();
    let native = solve_native_ok(&problem, &x0, &[]);
    let ipopt = solve_ipopt_ok(&problem, &x0, &[]);
    assert_native_matches_ipopt("hanging_chain", Some(backend), &native, &ipopt, 1e-3, 1e-4);
    assert_accepted_trace_parity(
        "hanging_chain",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 0,
            max_step_tag_mismatches: 0,
            max_primal_log_gap: 0.05,
            max_dual_log_gap: 0.05,
            max_mu_log_gap: 0.05,
            max_regularization_log_gap: 0.05,
        },
    );
    assert_native_event_seen(
        "hanging_chain",
        &native,
        InteriorPointIterationEvent::LongLineSearch,
    );
    assert_native_event_seen(
        "hanging_chain",
        &native,
        InteriorPointIterationEvent::FilterAccepted,
    );
    assert_native_event_seen(
        "hanging_chain",
        &native,
        InteriorPointIterationEvent::SecondOrderCorrectionAttempted,
    );
    assert_native_event_seen(
        "hanging_chain",
        &native,
        InteriorPointIterationEvent::AdaptiveRegularizationUsed,
    );
}

#[test]
fn compare_native_and_ipopt_with_soc_method_one() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let x0 = hanging_chain_initial_guess();
    let native = solve_native_with_options_ok(
        &problem,
        &x0,
        &[],
        native_options_with(|options| {
            options.second_order_correction_method =
                InteriorPointSecondOrderCorrectionMethod::ScaledPrimalDualRows;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &x0,
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::integer("soc_method", 1));
        }),
    );
    assert_native_event_seen(
        "hanging_chain_soc_method_one",
        &native,
        InteriorPointIterationEvent::SecondOrderCorrectionAttempted,
    );
    assert_native_matches_ipopt(
        "hanging_chain_soc_method_one",
        Some(backend),
        &native,
        &ipopt,
        1e-3,
        1e-4,
    );
    assert_accepted_trace_parity(
        "hanging_chain_soc_method_one",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 0,
            max_step_tag_mismatches: 0,
            max_primal_log_gap: 0.05,
            max_dual_log_gap: 0.05,
            max_mu_log_gap: 0.05,
            max_regularization_log_gap: 0.05,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_primal_dual_corrector() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.corrector_type = InteriorPointCorrectorType::PrimalDual;
            options.skip_corrector_if_negative_curvature = false;
            options.skip_corrector_in_monotone_mode = false;
            options.second_order_correction = false;
            options.max_second_order_corrections = 0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("corrector_type", "primal-dual"));
            options
                .raw_options
                .push(IpoptRawOption::text("skip_corr_if_neg_curv", "no"));
            options
                .raw_options
                .push(IpoptRawOption::text("skip_corr_in_monotone_mode", "no"));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );
    assert_native_event_seen(
        "linearly_constrained_quadratic_primal_dual_corrector",
        &native,
        InteriorPointIterationEvent::CorrectorAttempted,
    );
    assert_native_event_seen(
        "linearly_constrained_quadratic_primal_dual_corrector",
        &native,
        InteriorPointIterationEvent::CorrectorAccepted,
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_primal_dual_corrector",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_affine_corrector() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.corrector_type = InteriorPointCorrectorType::Affine;
            options.skip_corrector_if_negative_curvature = false;
            options.skip_corrector_in_monotone_mode = false;
            options.second_order_correction = false;
            options.max_second_order_corrections = 0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("corrector_type", "affine"));
            options
                .raw_options
                .push(IpoptRawOption::text("skip_corr_if_neg_curv", "no"));
            options
                .raw_options
                .push(IpoptRawOption::text("skip_corr_in_monotone_mode", "no"));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );
    assert_native_event_seen(
        "linearly_constrained_quadratic_affine_corrector",
        &native,
        InteriorPointIterationEvent::CorrectorAttempted,
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_affine_corrector",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_perturb_always_cd() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.perturb_always_cd = true;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("perturb_always_cd", "yes"));
        }),
    );

    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_perturb_always_cd",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic_perturb_always_cd",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 2,
            max_step_tag_mismatches: 2,
            max_primal_log_gap: 1.5,
            max_dual_log_gap: 13.0,
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 1.0,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_full_space_refinement_options() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.min_refinement_steps = 2;
            options.max_refinement_steps = 2;
            options.residual_ratio_max = 1.0e-10;
            options.residual_ratio_singular = 1.0e-5;
            options.residual_improvement_factor = 0.5;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::integer("min_refinement_steps", 2));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_refinement_steps", 2));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_ratio_max", 1.0e-10));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_ratio_singular", 1.0e-5));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_improvement_factor", 0.5));
        }),
    );

    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_refinement_options",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic_refinement_options",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 2,
            max_step_tag_mismatches: 2,
            max_primal_log_gap: 1.5,
            max_dual_log_gap: 13.0,
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 1.0,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_refinement_quality_retry_accept_current() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.min_refinement_steps = 0;
            options.max_refinement_steps = 0;
            options.residual_ratio_max = 1.0e-300;
            options.residual_ratio_singular = 1.0;
            options.residual_improvement_factor = 0.5;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            enable_ipopt_trace_journal(options);
            options
                .raw_options
                .push(IpoptRawOption::integer("min_refinement_steps", 0));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_refinement_steps", 0));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_ratio_max", 1.0e-300));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_ratio_singular", 1.0));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_improvement_factor", 0.5));
        }),
    );

    assert_native_event_seen(
        "linearly_constrained_quadratic_refinement_quality_retry_accept_current",
        &native,
        InteriorPointIterationEvent::LinearSolverQualityIncreased,
    );
    assert_ipopt_info_string_seen(
        "linearly_constrained_quadratic_refinement_quality_retry_accept_current",
        &ipopt,
        'q',
    );
    assert_ipopt_info_string_seen(
        "linearly_constrained_quadratic_refinement_quality_retry_accept_current",
        &ipopt,
        'S',
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_refinement_quality_retry_accept_current",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic_refinement_quality_retry_accept_current",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 2,
            max_step_tag_mismatches: 2,
            max_primal_log_gap: 1.5,
            max_dual_log_gap: 13.0,
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 1.0,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_refinement_pretend_singular_retry() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.min_refinement_steps = 0;
            options.max_refinement_steps = 0;
            options.residual_ratio_max = 1.0e-300;
            options.residual_ratio_singular = 1.0e-300;
            options.residual_improvement_factor = 0.5;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            enable_ipopt_trace_journal(options);
            options
                .raw_options
                .push(IpoptRawOption::integer("min_refinement_steps", 0));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_refinement_steps", 0));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_ratio_max", 1.0e-300));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_ratio_singular", 1.0e-300));
            options
                .raw_options
                .push(IpoptRawOption::number("residual_improvement_factor", 0.5));
        }),
    );

    assert_native_event_seen(
        "linearly_constrained_quadratic_refinement_pretend_singular_retry",
        &native,
        InteriorPointIterationEvent::LinearSolverQualityIncreased,
    );
    assert_ipopt_info_string_seen(
        "linearly_constrained_quadratic_refinement_pretend_singular_retry",
        &ipopt,
        'q',
    );
    assert_ipopt_info_string_seen(
        "linearly_constrained_quadratic_refinement_pretend_singular_retry",
        &ipopt,
        's',
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_refinement_pretend_singular_retry",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic_refinement_pretend_singular_retry",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 2,
            max_step_tag_mismatches: 2,
            max_primal_log_gap: 1.5,
            max_dual_log_gap: 13.0,
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 1.0,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_negative_curvature_test_options() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.negative_curvature_test_tolerance = 1.0e-12;
            options.negative_curvature_test_regularized = false;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::number("neg_curv_test_tol", 1.0e-12));
            options
                .raw_options
                .push(IpoptRawOption::text("neg_curv_test_reg", "no"));
        }),
    );

    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_neg_curv_test_options",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic_neg_curv_test_options",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 2,
            max_step_tag_mismatches: 2,
            max_primal_log_gap: 1.5,
            max_dual_log_gap: 13.0,
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 1.0,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_magic_steps() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.magic_steps = true;
            options.second_order_correction = false;
            options.max_second_order_corrections = 0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("magic_steps", "yes"));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_magic_steps",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic_magic_steps",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 0,
            max_step_tag_mismatches: 0,
            max_primal_log_gap: 0.05,
            max_dual_log_gap: 0.05,
            // This tiny witness exercises the magic-step branch but remains
            // sensitive to the already-known final monotone-mu update edge.
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 0.05,
        },
    );
}

#[test]
fn compare_native_and_ipopt_on_box_bounds_regression() {
    skip_without_native_spral!();
    let problem = BoundConstrainedQuadraticProblem;
    let native = solve_native_ok(&problem, &[-10.0, 10.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[-10.0, 10.0], &[]);
    assert_native_matches_ipopt("box_bounds_quadratic", None, &native, &ipopt, 1e-4, 1e-4);
}

#[test]
fn compare_native_and_ipopt_on_fixed_variable_quadratic() {
    skip_without_native_spral!();
    let problem = FixedVariableQuadraticProblem;
    let native = solve_native_ok(&problem, &[4.0, -5.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[4.0, -5.0], &[]);
    assert_abs_diff_eq!(native.x[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ipopt.x[0], 1.0, epsilon = 1e-12);
    assert_native_matches_ipopt(
        "fixed_variable_quadratic",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_on_square_equality_quadratic() {
    skip_without_native_spral!();
    let problem = SquareEqualityQuadraticProblem;
    let native = solve_native_ok(&problem, &[8.0, -3.0], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[8.0, -3.0], &[]);
    assert_abs_diff_eq!(native.x[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(native.x[1], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(ipopt.x[0], 1.0, epsilon = 1e-8);
    assert_abs_diff_eq!(ipopt.x[1], 2.0, epsilon = 1e-8);
    assert_native_matches_ipopt(
        "square_equality_quadratic",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );

    let initial_state = solve_nlp_interior_point(
        &problem,
        &[8.0, -3.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 0;
        }),
    );
    match initial_state {
        Err(InteriorPointSolveError::MaxIterations { context, .. }) => {
            let final_state = context
                .final_state
                .as_ref()
                .expect("max-iteration failure should retain initialized square state");
            assert!(
                final_state
                    .equality_multipliers
                    .as_ref()
                    .is_some_and(|multipliers| multipliers.iter().all(|value| *value == 0.0)),
                "NLIP should mirror IpDefaultIterateInitializer::least_square_mults and zero square-problem initial y_c"
            );
        }
        other => panic!("square equality max-iteration probe mismatch: {other:?}"),
    }
}

#[test]
fn compare_native_and_ipopt_on_linearly_constrained_quadratic_trace() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_ok(&problem, &[0.1, 0.9], &[]);
    let ipopt = solve_ipopt_ok(&problem, &[0.1, 0.9], &[]);
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
    assert_accepted_trace_parity(
        "linearly_constrained_quadratic",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 2,
            max_step_tag_mismatches: 2,
            max_primal_log_gap: 1.5,
            max_dual_log_gap: 13.0,
            max_mu_log_gap: 2.0,
            max_regularization_log_gap: 1.0,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_mu_based_bound_multiplier_init() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.bound_mult_init_method = InteriorPointBoundMultiplierInitMethod::MuBased;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::text("bound_mult_init_method", "mu-based"));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_bound_mult_mu_based",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_disabled_constraint_multiplier_init() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            // IpDefaultIterateInitializer::least_square_mults sets y_c/y_d to
            // zero when constr_mult_init_max <= 0 instead of using the bound
            // multiplier initialization as a substitute.
            options.constr_mult_init_max = 0.0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("constr_mult_init_max", 0.0));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_constr_mult_init_disabled",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_constraint_multiplier_init_rejected_by_cap() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            // IpDefaultIterateInitializer::least_square_mults computes
            // y_c/y_d, then discards them when max-norm exceeds this cap.
            options.constr_mult_init_max = 1.0e-12;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("constr_mult_init_max", 1.0e-12));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_constr_mult_init_cap",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_nondefault_initializer_pushes_and_multiplier_value() {
    skip_without_native_spral!();

    let bound_problem = BoundConstrainedQuadraticProblem;
    let native_bound = solve_native_with_options_ok(
        &bound_problem,
        &[-10.0, 10.0],
        &[],
        native_options_with(|options| {
            options.bound_push = 5e-2;
            options.bound_frac = 2e-2;
            options.bound_mult_init_val = 2.5;
        }),
    );
    let ipopt_bound = solve_ipopt_with_options_ok(
        &bound_problem,
        &[-10.0, 10.0],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("bound_push", 5e-2));
            options
                .raw_options
                .push(IpoptRawOption::number("bound_frac", 2e-2));
            options
                .raw_options
                .push(IpoptRawOption::number("bound_mult_init_val", 2.5));
        }),
    );
    assert_native_matches_ipopt(
        "box_bounds_quadratic_initializer_pushes",
        None,
        &native_bound,
        &ipopt_bound,
        1e-4,
        1e-4,
    );

    let slack_problem = LinearlyConstrainedQuadraticProblem;
    let native_slack = solve_native_with_options_ok(
        &slack_problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.slack_bound_push = 5e-2;
            options.slack_bound_frac = 2e-2;
            options.bound_mult_init_val = 2.5;
        }),
    );
    let ipopt_slack = solve_ipopt_with_options_ok(
        &slack_problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("slack_bound_push", 5e-2));
            options
                .raw_options
                .push(IpoptRawOption::number("slack_bound_frac", 2e-2));
            options
                .raw_options
                .push(IpoptRawOption::number("bound_mult_init_val", 2.5));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_initializer_pushes",
        None,
        &native_slack,
        &ipopt_slack,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_zero_damping_and_bound_relaxation_disabled() {
    skip_without_native_spral!();
    let problem = BoundConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        native_options_with(|options| {
            options.kappa_d = 0.0;
            options.bound_relax_factor = 0.0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        ipopt_options_with(|options| {
            options.kappa_d = 0.0;
            options
                .raw_options
                .push(IpoptRawOption::number("bound_relax_factor", 0.0));
        }),
    );
    assert_native_matches_ipopt(
        "box_bounds_quadratic_zero_damping_no_relaxation",
        None,
        &native,
        &ipopt,
        1e-4,
        1e-4,
    );
}

#[test]
fn compare_native_and_ipopt_with_bound_multiplier_safeguard() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            // IpoptAlgorithm::correct_bound_multiplier is normally dormant
            // because kappa_sigma is huge. Tightening it forces the accepted
            // trial multipliers back onto the mu/slack complementarity band.
            options.kappa_sigma = 1.0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("kappa_sigma", 1.0));
        }),
    );
    assert_native_event_seen(
        "linearly_constrained_quadratic_kappa_sigma",
        &native,
        InteriorPointIterationEvent::BoundMultiplierSafeguardApplied,
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_kappa_sigma",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_globalization_option_toggles() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.max_iters = 200;
            options.mu_allow_fast_monotone_decrease = false;
            options.second_order_correction = false;
            options.max_second_order_corrections = 0;
            options.watchdog_shortened_iter_trigger = 0;
            options.tiny_step_tol = 0.0;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 200;
            options.raw_options.push(IpoptRawOption::text(
                "mu_allow_fast_monotone_decrease",
                "no",
            ));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
            options.raw_options.push(IpoptRawOption::integer(
                "watchdog_shortened_iter_trigger",
                0,
            ));
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_tol", 0.0));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_globalization_toggles",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_watchdog_trigger_profile() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let x0 = hanging_chain_initial_guess();
    let native = solve_native_with_options_ok(
        &problem,
        &x0,
        &[],
        native_options_with(|options| {
            // BacktrackingLineSearch::StartWatchDog is only reached after
            // repeated shortened non-tiny line searches. Lower the trigger on
            // the existing hanging-chain witness rather than changing the NLP.
            options.watchdog_shortened_iter_trigger = 3;
            options.watchdog_trial_iter_max = 3;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &x0,
        &[],
        ipopt_options_with(|options| {
            enable_ipopt_trace_journal(options);
            options.raw_options.push(IpoptRawOption::integer(
                "watchdog_shortened_iter_trigger",
                3,
            ));
            options
                .raw_options
                .push(IpoptRawOption::integer("watchdog_trial_iter_max", 3));
        }),
    );
    assert_native_event_seen(
        "hanging_chain_watchdog",
        &native,
        InteriorPointIterationEvent::WatchdogArmed,
    );
    assert_native_event_seen(
        "hanging_chain_watchdog",
        &native,
        InteriorPointIterationEvent::WatchdogActivated,
    );
    let first_watchdog_armed_iter = native
        .snapshots
        .iter()
        .find(|snapshot| {
            snapshot
                .events
                .contains(&InteriorPointIterationEvent::WatchdogArmed)
        })
        .map(|snapshot| snapshot.iteration)
        .expect("hanging_chain_watchdog should arm NLIP watchdog");
    let first_watchdog_activated_iter = native
        .snapshots
        .iter()
        .find(|snapshot| {
            snapshot
                .events
                .contains(&InteriorPointIterationEvent::WatchdogActivated)
        })
        .map(|snapshot| snapshot.iteration)
        .expect("hanging_chain_watchdog should activate NLIP watchdog");
    assert_eq!(
        first_watchdog_armed_iter, first_watchdog_activated_iter,
        "NLIP watchdog should arm in the same line search that activates it, mirroring BacktrackingLineSearch::StartWatchDog"
    );
    let watchdog_line_search = native
        .snapshots
        .iter()
        .find(|snapshot| {
            snapshot
                .events
                .contains(&InteriorPointIterationEvent::WatchdogActivated)
        })
        .and_then(|snapshot| snapshot.line_search.as_ref())
        .expect("hanging_chain_watchdog should retain watchdog line-search diagnostics");
    assert_eq!(
        watchdog_line_search.backtrack_count, 0,
        "IPOPT keeps alpha_min at alpha_primal_max while in_watchdog_, so the successful W trial should not backtrack"
    );
    assert!(
        watchdog_line_search.rejected_trials.is_empty(),
        "IPOPT's active watchdog DoBacktrackingLineSearch returns after one max-step trial"
    );
    assert_ipopt_info_string_seen("hanging_chain_watchdog", &ipopt, 'W');
    assert_native_matches_ipopt(
        "hanging_chain_watchdog",
        Some(backend),
        &native,
        &ipopt,
        1e-3,
        1e-4,
    );
    assert_accepted_trace_parity(
        "hanging_chain_watchdog",
        &native,
        &ipopt,
        AcceptedTraceParityTolerances {
            max_iteration_gap: 0,
            max_step_tag_mismatches: 0,
            max_primal_log_gap: 0.05,
            max_dual_log_gap: 0.05,
            max_mu_log_gap: 0.05,
            max_regularization_log_gap: 0.05,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_lowercase_watchdog_trial_profile() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let problem = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let profile = WatchdogActivationProfile {
        trigger: 1,
        trial_max: 1,
        beta: 0.5,
        max_soc: 4,
        disable_fast_mu: false,
        disable_tiny_step: false,
    };
    let native = solve_nlp_interior_point_with_callback(
        &problem,
        &[2.5, 3.0, 0.75],
        &[],
        &native_options_with(|options| profile.apply_native(options)),
        |_| {},
    )
    .expect("casadi rosenbrock lowercase watchdog witness should solve in NLIP");
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[2.5, 3.0, 0.75],
        &[],
        &ipopt_options_with(|options| profile.apply_ipopt(options)),
    )
    .expect("casadi rosenbrock lowercase watchdog witness should solve in IPOPT");
    assert_source_built_spral_ipopt_provenance(&ipopt);
    assert!(
        ipopt_accepted_trace(&ipopt)
            .iter()
            .any(|point| point.step_tag.as_deref() == Some("w")),
        "IPOPT witness should exercise lowercase BacktrackingLineSearch watchdog trial"
    );
    assert!(
        ipopt_info_string_seen(&ipopt, 'w'),
        "IPOPT witness should append StopWatchDog's lowercase info marker"
    );
    let lowercase_watchdog_trial = native
        .snapshots
        .iter()
        .find(|snapshot| snapshot.step_tag == Some('w'))
        .expect("NLIP should mirror IPOPT's lowercase watchdog trial acceptance");
    let line_search = lowercase_watchdog_trial
        .line_search
        .as_ref()
        .expect("lowercase watchdog trial should retain line-search diagnostics");
    assert!(line_search.watchdog_active);
    assert!(!line_search.watchdog_accepted);
    assert_eq!(line_search.backtrack_count, 0);
    assert!(
        !line_search.second_order_correction_attempted,
        "IPOPT breaks out of DoBacktrackingLineSearch before SOC while in_watchdog_ is active"
    );
    let restored_retry = native
        .snapshots
        .windows(2)
        .find_map(|pair| {
            let previous = &pair[0];
            let current = &pair[1];
            let line_search = current.line_search.as_ref()?;
            (previous.step_tag == Some('w')
                && current.step_tag == Some('h')
                && !line_search.watchdog_active
                && !line_search.watchdog_accepted
                && line_search.backtrack_count > 0
                && current.alpha_pr.is_some_and(|alpha| alpha < 1.0))
            .then_some(line_search)
        })
        .expect(
            "NLIP should accept a restored non-watchdog retry after a lowercase watchdog trial",
        );
    assert_eq!(
        restored_retry.initial_alpha_pr, 1.0,
        "StopWatchDog retry keeps the stored direction's max primal step before skipping the first trial"
    );
}

fn run_watchdog_tiny_stop_sweep_case<P: CompiledNlpProblem>(
    case_name: &str,
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    profile: WatchdogActivationProfile,
    tiny_step_tol: f64,
) {
    let native = solve_nlp_interior_point_with_callback(
        problem,
        x0,
        parameters,
        &native_options_with(|options| {
            profile.apply_native(options);
            options.tiny_step_tol = tiny_step_tol;
        }),
        |_| {},
    );
    let ipopt = solve_nlp_ipopt(
        problem,
        x0,
        parameters,
        &ipopt_options_with(|options| {
            profile.apply_ipopt(options);
            options.print_level = 0;
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_tol", tiny_step_tol));
        }),
    );
    let native = match native {
        Ok(summary) => summary,
        Err(error) => {
            println!(
                "[watchdog-tiny-sweep] {case_name}/{} tol={tiny_step_tol:.1e} native_failed={error:?}",
                profile.label()
            );
            return;
        }
    };
    let ipopt = match ipopt {
        Ok(summary) => summary,
        Err(error) => {
            println!(
                "[watchdog-tiny-sweep] {case_name}/{} tol={tiny_step_tol:.1e} ipopt_failed={error:?}",
                profile.label()
            );
            return;
        }
    };
    assert_source_built_spral_ipopt_provenance(&ipopt);
    let native_trace = nlip_accepted_trace(&native);
    let ipopt_trace = ipopt_accepted_trace(&ipopt);
    let native_stop = native.snapshots.iter().any(|snapshot| {
        snapshot
            .events
            .contains(&InteriorPointIterationEvent::WatchdogStoppedBeforeLineSearch)
    });
    let native_tiny = native.snapshots.iter().any(|snapshot| {
        snapshot
            .events
            .contains(&InteriorPointIterationEvent::TinyStep)
    });
    println!(
        "[watchdog-tiny-sweep] {case_name}/{} tol={tiny_step_tol:.1e} native_stop={native_stop} native_tiny={native_tiny} nlip_steps={} ipopt_steps={} nlip_tags={} ipopt_tags={} ipopt_info={}",
        profile.label(),
        native_trace.len(),
        ipopt_trace.len(),
        step_tag_summary(&native_trace),
        step_tag_summary(&ipopt_trace),
        ipopt_info_marker_summary(&ipopt),
    );
}

fn run_soft_restoration_sweep_case<P: CompiledNlpProblem>(
    case_name: &str,
    problem: &P,
    x0: &[f64],
    alpha_min_frac: f64,
    max_soc: usize,
    soft_factor: f64,
) -> SoftRestorationSweepHit {
    let native = solve_nlp_interior_point(
        problem,
        x0,
        &[],
        &native_options_with(|options| {
            options.max_iters = 80;
            options.alpha_min_frac = alpha_min_frac;
            options.second_order_correction = max_soc > 0;
            options.max_second_order_corrections = max_soc;
            options.watchdog_shortened_iter_trigger = 0;
            options.soft_restoration_pderror_reduction_factor = soft_factor;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        problem,
        x0,
        &[],
        &ipopt_options_with(|options| {
            enable_ipopt_trace_journal(options);
            options
                .raw_options
                .push(IpoptRawOption::number("alpha_min_frac", alpha_min_frac));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", max_soc as i32));
            options.raw_options.push(IpoptRawOption::integer(
                "watchdog_shortened_iter_trigger",
                0,
            ));
            options.raw_options.push(IpoptRawOption::number(
                "soft_resto_pderror_reduction_factor",
                soft_factor,
            ));
        }),
    );
    let native_tags = nlip_result_step_tag_summary(&native);
    let ipopt_tags = ipopt_result_step_tag_summary(&ipopt);
    let ipopt_info = parse_ipopt_info_strings(ipopt_result_journal_output(&ipopt))
        .values()
        .fold(BTreeMap::<String, usize>::new(), |mut counts, info| {
            for marker in ["W", "w", "Tmax", "F+", "F-", "MaxS", "e"] {
                if info.contains(marker) {
                    *counts.entry(marker.to_string()).or_default() += 1;
                }
            }
            counts
        })
        .into_iter()
        .map(|(marker, count)| format!("{marker}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let native_soft = nlip_result_has_step_tag(&native, &['s', 'S']);
    let native_soft_original = nlip_result_has_step_tag(&native, &['S']);
    let ipopt_soft = ipopt_result_has_step_tag(&ipopt, &["s", "S"]);
    let ipopt_soft_original = ipopt_result_has_step_tag(&ipopt, &["S"]);
    println!(
        "[soft-resto-sweep] {case_name} alpha_min_frac={alpha_min_frac:.2e} max_soc={max_soc} soft_factor={soft_factor:.2e} native_soft={native_soft} native_S={native_soft_original} ipopt_soft={ipopt_soft} ipopt_S={ipopt_soft_original} native_tags={native_tags} ipopt_tags={ipopt_tags} ipopt_info={ipopt_info}"
    );
    SoftRestorationSweepHit {
        any: native_soft || ipopt_soft,
        original_criterion: native_soft_original || ipopt_soft_original,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SoftRestorationSweepHit {
    any: bool,
    original_criterion: bool,
}

impl SoftRestorationSweepHit {
    fn merge(&mut self, other: Self) {
        self.any |= other.any;
        self.original_criterion |= other.original_criterion;
    }
}

#[test]
#[ignore = "diagnostic sweep for finding trace-clean IPOPT watchdog activation witnesses"]
fn print_watchdog_activation_profile_sweep() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let hanging_chain = build_problem_ok(hanging_chain_problem(backend), backend);
    let hanging_chain_x0 = hanging_chain_initial_guess();
    let equality_rosenbrock = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let casadi_rosenbrock = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let hs071 = build_problem_ok(hs071_problem(backend), backend);
    let profiles = [
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 1,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 2,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 3,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.25,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.75,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.5,
            max_soc: 0,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: true,
            disable_tiny_step: true,
        },
    ];
    let mut found_clean_activation = false;
    for profile in profiles {
        found_clean_activation |= run_watchdog_activation_sweep_case(
            "hanging_chain",
            &hanging_chain,
            &hanging_chain_x0,
            &[],
            profile,
        );
        found_clean_activation |= run_watchdog_activation_sweep_case(
            "equality_rosenbrock",
            &equality_rosenbrock,
            &[1.2, 1.2],
            &[],
            profile,
        );
        found_clean_activation |= run_watchdog_activation_sweep_case(
            "casadi_rosenbrock",
            &casadi_rosenbrock,
            &[2.5, 3.0, 0.75],
            &[],
            profile,
        );
        found_clean_activation |= run_watchdog_activation_sweep_case(
            "hs071",
            &hs071,
            &[1.0, 5.0, 5.0, 1.0],
            &[],
            profile,
        );
        found_clean_activation |= run_watchdog_activation_sweep_case(
            "linearly_constrained_quadratic",
            &LinearlyConstrainedQuadraticProblem,
            &[0.1, 0.9],
            &[],
            profile,
        );
    }
    assert!(
        found_clean_activation,
        "no trace-clean WatchdogActivated/IPOPT-W witness found in this reduced profile sweep"
    );
}

#[test]
#[ignore = "diagnostic sweep for finding IPOPT soft-restoration witnesses"]
fn print_soft_restoration_profile_sweep() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let casadi_rosenbrock = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let found = run_soft_restoration_sweep_case(
        "casadi_rosenbrock",
        &casadi_rosenbrock,
        &[2.5, 3.0, 0.75],
        0.05,
        1,
        1.0e6,
    );
    assert!(found.any, "soft restoration sweep did not find a witness");
}

fn run_soft_restoration_original_grid<P: CompiledNlpProblem, const N: usize>(
    case_name: &str,
    problem: &P,
    starts: &[(&str, [f64; N])],
    found: &mut SoftRestorationSweepHit,
) {
    for (start_label, x0) in starts {
        for alpha_min_frac in [0.01, 0.05, 0.2] {
            for max_soc in [0, 1] {
                for soft_factor in [1.0e-12, 1.0e-2, 1.0 - 1.0e-4, 1.0e6] {
                    let labeled_case = format!("{case_name}/{start_label}");
                    found.merge(run_soft_restoration_sweep_case(
                        &labeled_case,
                        problem,
                        x0,
                        alpha_min_frac,
                        max_soc,
                        soft_factor,
                    ));
                    if found.original_criterion {
                        return;
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "diagnostic sweep for finding IPOPT soft-restoration original-criterion witnesses"]
fn print_soft_restoration_original_profile_sweep() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let casadi_rosenbrock = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let equality_rosenbrock = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let hs071 = build_problem_ok(hs071_problem(backend), backend);
    let mut found = SoftRestorationSweepHit::default();

    run_soft_restoration_original_grid(
        "casadi_rosenbrock",
        &casadi_rosenbrock,
        &[
            ("default", [2.5, 3.0, 0.75]),
            ("mixed", [-2.0, 3.0, 0.25]),
            ("far", [10.0, -10.0, 5.0]),
        ],
        &mut found,
    );
    if !found.original_criterion {
        run_soft_restoration_original_grid(
            "equality_rosenbrock",
            &equality_rosenbrock,
            &[("default", [-1.2, 1.0]), ("far", [10.0, -10.0])],
            &mut found,
        );
    }
    if !found.original_criterion {
        run_soft_restoration_original_grid(
            "hs071",
            &hs071,
            &[
                ("default", [1.0, 5.0, 5.0, 1.0]),
                ("middle", [2.0, 2.0, 2.0, 2.0]),
            ],
            &mut found,
        );
    }
    assert!(
        found.any,
        "soft restoration original-criterion sweep did not even find a lowercase soft-restoration witness"
    );
    if !found.original_criterion {
        println!(
            "[soft-resto-sweep] no uppercase S original-criterion witness found in bounded grid"
        );
    }
}

#[test]
#[ignore = "diagnostic sweep for finding active-watchdog tiny-step StopWatchDog witnesses"]
fn print_watchdog_tiny_stop_profile_sweep() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let hanging_chain = build_problem_ok(hanging_chain_problem(backend), backend);
    let hanging_chain_x0 = hanging_chain_initial_guess();
    let equality_rosenbrock = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let casadi_rosenbrock = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let hs071 = build_problem_ok(hs071_problem(backend), backend);
    let profiles = [
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 1,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 3,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.25,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 3,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: true,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 8,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 1,
            trial_max: 16,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
        WatchdogActivationProfile {
            trigger: 2,
            trial_max: 16,
            beta: 0.5,
            max_soc: 4,
            disable_fast_mu: false,
            disable_tiny_step: false,
        },
    ];
    for profile in profiles {
        for tiny_step_tol in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2] {
            run_watchdog_tiny_stop_sweep_case(
                "hanging_chain",
                &hanging_chain,
                &hanging_chain_x0,
                &[],
                profile,
                tiny_step_tol,
            );
            run_watchdog_tiny_stop_sweep_case(
                "equality_rosenbrock",
                &equality_rosenbrock,
                &[1.2, 1.2],
                &[],
                profile,
                tiny_step_tol,
            );
            run_watchdog_tiny_stop_sweep_case(
                "casadi_rosenbrock",
                &casadi_rosenbrock,
                &[2.5, 3.0, 0.75],
                &[],
                profile,
                tiny_step_tol,
            );
            run_watchdog_tiny_stop_sweep_case(
                "hs071",
                &hs071,
                &[1.0, 5.0, 5.0, 1.0],
                &[],
                profile,
                tiny_step_tol,
            );
            run_watchdog_tiny_stop_sweep_case(
                "linearly_constrained_quadratic",
                &LinearlyConstrainedQuadraticProblem,
                &[0.1, 0.9],
                &[],
                profile,
                tiny_step_tol,
            );
        }
    }
}

#[test]
fn compare_native_and_ipopt_with_nondefault_line_search_reduction_factor() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.line_search_beta = 0.25;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("alpha_red_factor", 0.25));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_alpha_red_factor",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_positive_mu_target() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_target = 1e-4;
            options.complementarity_tol = 5e-4;
            options.overall_tol = 5e-4;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.tol = 5e-4;
            options.complementarity_tol = Some(5e-4);
            options
                .raw_options
                .push(IpoptRawOption::number("mu_target", 1e-4));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_positive_mu_target",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-4,
            objective: 1e-4,
            primal: 5e-4,
            native_dual: 5e-4,
            ipopt_dual: 5e-4,
            complementarity: 5e-4,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_loqo_never_monotone_mu() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Loqo;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::NeverMonotoneMode;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "loqo"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "never-monotone-mode",
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_loqo_never_monotone_mu",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_loqo_obj_constr_filter_mu() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Loqo;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "loqo"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_loqo_obj_constr_filter_mu",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_loqo_nonzero_safeguard() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Loqo;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.adaptive_mu_safeguard_factor = 0.25;
            options.max_iters = 300;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 300;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "loqo"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
            options
                .raw_options
                .push(IpoptRawOption::number("adaptive_mu_safeguard_factor", 0.25));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_loqo_nonzero_safeguard",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_loqo_kkt_error_globalization() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Loqo;
            options.adaptive_mu_globalization = InteriorPointAdaptiveMuGlobalization::KktError;
            options.max_iters = 300;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 300;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "loqo"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "kkt-error",
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_loqo_kkt_error_globalization",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_loqo_restore_previous_iterate() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Loqo;
            options.adaptive_mu_globalization = InteriorPointAdaptiveMuGlobalization::KktError;
            options.adaptive_mu_restore_previous_iterate = true;
            // IPOPT 3.14.20 accepts `adaptive_mu_kkterror_red_iters=0`, but with
            // restore-previous-iterate it can switch before any accepted point
            // was remembered. Use one reference plus a tiny required decrease
            // to force the same source branch after a real free-mu accepted
            // point exists.
            options.adaptive_mu_kkt_error_red_iters = 1;
            options.adaptive_mu_kkt_error_red_fact = 1e-16;
            options.max_iters = 300;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 300;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "loqo"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "kkt-error",
            ));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_restore_previous_iterate",
                "yes",
            ));
            options
                .raw_options
                .push(IpoptRawOption::integer("adaptive_mu_kkterror_red_iters", 1));
            options.raw_options.push(IpoptRawOption::number(
                "adaptive_mu_kkterror_red_fact",
                1e-16,
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_loqo_restore_previous_iterate",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_quality_function_obj_constr_filter_mu() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::QualityFunction;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.linear_solver = InteriorPointLinearSolver::SpralSrc;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "quality-function"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_quality_function_obj_constr_filter_mu",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_quality_function_on_box_bounds() {
    skip_without_native_spral!();
    let problem = BoundConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::QualityFunction;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.linear_solver = InteriorPointLinearSolver::SpralSrc;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "quality-function"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "bound_constrained_quadratic_adaptive_quality_function",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[rstest]
#[case::one_norm(InteriorPointQualityFunctionNorm::OneNorm, "1-norm")]
#[case::max_norm(InteriorPointQualityFunctionNorm::MaxNorm, "max-norm")]
#[case::two_norm(InteriorPointQualityFunctionNorm::TwoNorm, "2-norm")]
fn compare_native_and_ipopt_with_adaptive_quality_function_nondefault_norm(
    #[case] norm: InteriorPointQualityFunctionNorm,
    #[case] ipopt_norm: &str,
) {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::QualityFunction;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.quality_function_norm_type = norm;
            options.linear_solver = InteriorPointLinearSolver::SpralSrc;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "quality-function"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
            options.raw_options.push(IpoptRawOption::text(
                "quality_function_norm_type",
                ipopt_norm,
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_quality_function_nondefault_norm",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_quality_function_nondefault_terms_and_search() {
    skip_without_native_spral!();
    let problem = BoundConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::QualityFunction;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.quality_function_centrality = InteriorPointQualityFunctionCentrality::Log;
            options.quality_function_balancing_term =
                InteriorPointQualityFunctionBalancingTerm::Cubic;
            options.quality_function_max_section_steps = 2;
            options.quality_function_section_sigma_tol = 5e-2;
            options.quality_function_section_qf_tol = 1e-3;
            options.sigma_min = 1e-4;
            options.sigma_max = 10.0;
            options.linear_solver = InteriorPointLinearSolver::SpralSrc;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "quality-function"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
            options
                .raw_options
                .push(IpoptRawOption::text("quality_function_centrality", "log"));
            options.raw_options.push(IpoptRawOption::text(
                "quality_function_balancing_term",
                "cubic",
            ));
            options.raw_options.push(IpoptRawOption::integer(
                "quality_function_max_section_steps",
                2,
            ));
            options.raw_options.push(IpoptRawOption::number(
                "quality_function_section_sigma_tol",
                5e-2,
            ));
            options.raw_options.push(IpoptRawOption::number(
                "quality_function_section_qf_tol",
                1e-3,
            ));
            options
                .raw_options
                .push(IpoptRawOption::number("sigma_min", 1e-4));
            options
                .raw_options
                .push(IpoptRawOption::number("sigma_max", 10.0));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "bound_constrained_quadratic_adaptive_quality_function_nondefault_terms_and_search",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_adaptive_probing_obj_constr_filter_mu() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Probing;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::ObjectiveConstraintFilter;
            options.linear_solver = InteriorPointLinearSolver::SpralSrc;
            options.max_iters = 300;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 300;
            options
                .raw_options
                .push(IpoptRawOption::text("mu_oracle", "probing"));
            options.raw_options.push(IpoptRawOption::text(
                "adaptive_mu_globalization",
                "obj-constr-filter",
            ));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "linearly_constrained_quadratic_adaptive_probing_obj_constr_filter_mu",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_with_mehrotra_algorithm() {
    skip_without_native_spral!();
    let problem = BoundConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        native_options_with(|options| {
            options.mehrotra_algorithm = true;
            options.mu_strategy = InteriorPointMuStrategy::Adaptive;
            options.adaptive_mu_oracle = InteriorPointAdaptiveMuOracle::Probing;
            options.adaptive_mu_globalization =
                InteriorPointAdaptiveMuGlobalization::NeverMonotoneMode;
            options.accept_every_trial_step = true;
            options.corrector_type = InteriorPointCorrectorType::None;
            options.bound_push = 10.0;
            options.bound_frac = 0.2;
            options.bound_mult_init_val = 10.0;
            options.constr_mult_init_max = 0.0;
            options.alpha_for_y = InteriorPointAlphaForYStrategy::BoundMultiplier;
            options.least_square_init_primal = true;
            options.linear_solver = InteriorPointLinearSolver::SpralSrc;
            options.max_iters = 200;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[-10.0, 10.0],
        &[],
        ipopt_options_with(|options| {
            options.mu_strategy = IpoptMuStrategy::Adaptive;
            options.max_iters = 200;
            options
                .raw_options
                .push(IpoptRawOption::text("mehrotra_algorithm", "yes"));
        }),
    );
    assert_native_matches_ipopt_with_final_tolerances(
        "bound_constrained_quadratic_mehrotra_algorithm",
        None,
        &native,
        &ipopt,
        FinalParityTolerances {
            x: 1e-5,
            objective: 1e-5,
            primal: 1e-5,
            native_dual: 1e-4,
            ipopt_dual: 1e-5,
            complementarity: 1e-5,
        },
    );
}

#[test]
fn compare_native_and_ipopt_alpha_for_y_option_profiles() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    for (profile_name, strategy, ipopt_value, alpha_for_y_tol) in [
        (
            "bound_mult",
            InteriorPointAlphaForYStrategy::BoundMultiplier,
            "bound-mult",
            None,
        ),
        ("min", InteriorPointAlphaForYStrategy::Min, "min", None),
        ("max", InteriorPointAlphaForYStrategy::Max, "max", None),
        ("full", InteriorPointAlphaForYStrategy::Full, "full", None),
        (
            "min_dual_infeas",
            InteriorPointAlphaForYStrategy::MinDualInfeas,
            "min-dual-infeas",
            None,
        ),
        (
            "safer_min_dual_infeas",
            InteriorPointAlphaForYStrategy::SaferMinDualInfeas,
            "safer-min-dual-infeas",
            None,
        ),
        (
            "primal_and_full",
            InteriorPointAlphaForYStrategy::PrimalAndFull,
            "primal-and-full",
            None,
        ),
        (
            "primal_and_full_strict_tol",
            InteriorPointAlphaForYStrategy::PrimalAndFull,
            "primal-and-full",
            Some(0.0),
        ),
        (
            "dual_and_full",
            InteriorPointAlphaForYStrategy::DualAndFull,
            "dual-and-full",
            None,
        ),
        (
            "dual_and_full_strict_tol",
            InteriorPointAlphaForYStrategy::DualAndFull,
            "dual-and-full",
            Some(0.0),
        ),
    ] {
        let native = solve_native_with_options_ok(
            &problem,
            &[0.1, 0.9],
            &[],
            native_options_with(|options| {
                options.alpha_for_y = strategy;
                if let Some(value) = alpha_for_y_tol {
                    options.alpha_for_y_tol = value;
                }
            }),
        );
        let ipopt = solve_ipopt_with_options_ok(
            &problem,
            &[0.1, 0.9],
            &[],
            ipopt_options_with(|options| {
                options
                    .raw_options
                    .push(IpoptRawOption::text("alpha_for_y", ipopt_value));
                if let Some(value) = alpha_for_y_tol {
                    options
                        .raw_options
                        .push(IpoptRawOption::number("alpha_for_y_tol", value));
                }
            }),
        );
        let problem_name = format!("linearly_constrained_quadratic_alpha_for_y_{profile_name}");
        assert_native_matches_ipopt(&problem_name, None, &native, &ipopt, 1e-6, 1e-6);
    }
}

#[test]
fn compare_native_and_ipopt_max_iterations_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.max_iters = 0;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 0;
        }),
    );

    match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations,
            context,
        }) => {
            assert_eq!(iterations, 0);
            let final_state = context
                .final_state
                .as_ref()
                .expect("native max-iteration failure should retain iteration-0 state");
            assert_eq!(final_state.iteration, 0);
            assert_eq!(final_state.phase, InteriorPointIterationPhase::Initial);
            assert!(
                final_state
                    .events
                    .contains(&InteriorPointIterationEvent::MaxIterationsReached),
                "native iteration-0 state should record max-iteration event: {:?}",
                final_state.events
            );
        }
        other => panic!("native max-iteration status mismatch: {other:?}"),
    }

    match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            assert_eq!(iterations, 0);
            assert!(
                partial_solution.is_some(),
                "IPOPT max-iteration failure should retain a partial solution"
            );
            assert!(
                snapshots
                    .iter()
                    .any(|snapshot| snapshot.phase == optimization::IpoptIterationPhase::Regular),
                "IPOPT max-iteration failure should retain an initial snapshot: {snapshots:?}"
            );
        }
        other => panic!("IPOPT max-iteration status mismatch: {other:?}"),
    }
}

#[test]
fn compare_native_and_ipopt_max_iterations_after_accepted_step_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.max_iters = 1;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 1;
        }),
    );

    match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations,
            context,
        }) => {
            assert_eq!(iterations, 1);
            let final_state = context
                .final_state
                .as_ref()
                .expect("native max-iteration failure should retain last accepted state");
            assert_eq!(final_state.iteration, 0);
            assert_eq!(final_state.phase, InteriorPointIterationPhase::AcceptedStep);
            assert!(
                final_state
                    .events
                    .contains(&InteriorPointIterationEvent::MaxIterationsReached),
                "native accepted state should record max-iteration event: {:?}",
                final_state.events
            );
        }
        other => panic!("native accepted-step max-iteration status mismatch: {other:?}"),
    }

    match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            assert_eq!(iterations, 1);
            assert!(
                partial_solution.is_some(),
                "IPOPT accepted-step max-iteration failure should retain a partial solution"
            );
            assert!(
                snapshots.iter().any(|snapshot| snapshot.iteration == 1
                    && snapshot.phase == optimization::IpoptIterationPhase::Regular),
                "IPOPT accepted-step max-iteration failure should retain iteration-1 diagnostics: {snapshots:?}"
            );
        }
        other => panic!("IPOPT accepted-step max-iteration status mismatch: {other:?}"),
    }
}

#[test]
fn compare_native_and_ipopt_diverging_iterates_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let diverging_tol = 0.5;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.diverging_iterates_tol = diverging_tol;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options.raw_options.push(IpoptRawOption::number(
                "diverging_iterates_tol",
                diverging_tol,
            ));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::DivergingIterates {
            max_abs_x,
            threshold,
            context,
        }) => {
            assert_abs_diff_eq!(max_abs_x, 0.9, epsilon = 1e-12);
            assert_eq!(threshold, diverging_tol);
            context
                .final_state
                .expect("NLIP diverging failure should retain the current iterate")
        }
        other => panic!("native diverging-iterates status mismatch: {other:?}"),
    };
    assert_eq!(native_state.iteration, 0);
    assert_eq!(native_state.phase, InteriorPointIterationPhase::Initial);

    let ipopt_snapshots = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::DivergingIterates);
            assert_eq!(iterations, 0);
            assert!(
                partial_solution.is_some(),
                "IPOPT diverging-iterates failure should retain a partial solution"
            );
            snapshots
        }
        other => panic!("IPOPT diverging-iterates status mismatch: {other:?}"),
    };
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 0 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT diverging-iterates failure should retain an iteration-0 snapshot");
    assert_vec_close(
        "diverging_iterates x",
        &native_state.x,
        &ipopt_state.x,
        1e-12,
    );
    assert!(
        ipopt_state
            .x
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            > diverging_tol,
        "IPOPT iteration should exceed diverging_iterates_tol"
    );
}

#[test]
fn compare_native_and_ipopt_wall_time_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let max_wall_time = 1.0e-12;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.max_wall_time = max_wall_time;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("max_wall_time", max_wall_time));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::WallTimeExceeded {
            elapsed_seconds,
            limit_seconds,
            context,
        }) => {
            assert!(
                elapsed_seconds >= limit_seconds,
                "NLIP wall time should trip only after elapsed >= limit"
            );
            assert_eq!(limit_seconds, max_wall_time);
            context
                .final_state
                .expect("NLIP wall-time failure should retain the current iterate")
        }
        other => panic!("native wall-time status mismatch: {other:?}"),
    };
    assert_eq!(native_state.iteration, 0);
    assert_eq!(native_state.phase, InteriorPointIterationPhase::Initial);

    let ipopt_snapshots = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumWallTimeExceeded);
            assert_eq!(iterations, 0);
            assert!(
                partial_solution.is_some(),
                "IPOPT wall-time failure should retain a partial solution"
            );
            snapshots
        }
        other => panic!("IPOPT wall-time status mismatch: {other:?}"),
    };
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 0 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT wall-time failure should retain an iteration-0 snapshot");
    assert_vec_close("wall_time x", &native_state.x, &ipopt_state.x, 1e-12);
}

#[test]
fn compare_native_and_ipopt_cpu_time_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let max_cpu_time = 1.0e-12;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.max_cpu_time = max_cpu_time;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("max_cpu_time", max_cpu_time));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::CpuTimeExceeded {
            elapsed_seconds,
            limit_seconds,
            context,
        }) => {
            assert!(
                elapsed_seconds >= limit_seconds,
                "NLIP CPU time should trip only after elapsed >= limit"
            );
            assert_eq!(limit_seconds, max_cpu_time);
            context
                .final_state
                .expect("NLIP CPU-time failure should retain the current iterate")
        }
        other => panic!("native CPU-time status mismatch: {other:?}"),
    };
    assert_eq!(native_state.iteration, 0);
    assert_eq!(native_state.phase, InteriorPointIterationPhase::Initial);

    let ipopt_snapshots = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumCpuTimeExceeded);
            assert_eq!(iterations, 0);
            assert!(
                partial_solution.is_some(),
                "IPOPT CPU-time failure should retain a partial solution"
            );
            snapshots
        }
        other => panic!("IPOPT CPU-time status mismatch: {other:?}"),
    };
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 0 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT CPU-time failure should retain an iteration-0 snapshot");
    assert_vec_close("cpu_time x", &native_state.x, &ipopt_state.x, 1e-12);
}

#[test]
fn compare_native_and_ipopt_user_requested_stop_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_nlp_interior_point_with_control_callback(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options(),
        |snapshot| snapshot.phase != InteriorPointIterationPhase::AcceptedStep,
    );
    let ipopt = solve_nlp_ipopt_with_control_callback(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options(),
        |snapshot| snapshot.iteration < 1,
    );

    let native_state = match native {
        Err(InteriorPointSolveError::UserRequestedStop { context }) => {
            let final_state = context
                .final_state
                .expect("NLIP user stop should retain the current iterate");
            assert_eq!(final_state.phase, InteriorPointIterationPhase::AcceptedStep);
            assert!(
                context.last_accepted_state.is_some(),
                "NLIP user stop after an accepted step should retain last accepted state"
            );
            final_state
        }
        other => panic!("native user-stop status mismatch: {other:?}"),
    };

    let ipopt_snapshots = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::UserRequestedStop);
            assert_eq!(iterations, 1);
            assert!(
                partial_solution.is_some(),
                "IPOPT user-stop failure should retain a partial solution"
            );
            snapshots
        }
        other => panic!("IPOPT user-stop status mismatch: {other:?}"),
    };
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 1 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT user-stop failure should retain an iteration-1 snapshot");
    assert_vec_close("user_stop x", &native_state.x, &ipopt_state.x, 1e-10);
}

#[test]
fn compare_native_and_ipopt_restoration_user_requested_stop_status() {
    skip_without_native_spral!();
    let problem = ImpossibleSquareEqualityProblem;
    let mut native_restoration_callbacks = 0usize;
    let native = solve_nlp_interior_point_with_control_callback(
        &problem,
        &[1.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 40;
            options.second_order_correction = false;
        }),
        |snapshot| {
            if snapshot.phase == InteriorPointIterationPhase::Restoration {
                native_restoration_callbacks += 1;
                return false;
            }
            true
        },
    );
    let mut ipopt_restoration_callbacks = 0usize;
    let ipopt = solve_nlp_ipopt_with_control_callback(
        &problem,
        &[1.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 40;
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
        |snapshot| {
            if snapshot.phase == IpoptIterationPhase::Restoration {
                ipopt_restoration_callbacks += 1;
                return false;
            }
            true
        },
    );

    match native {
        Err(InteriorPointSolveError::UserRequestedStop { context }) => {
            assert!(
                context.final_state.is_some(),
                "NLIP restoration user stop should retain current failure state"
            );
        }
        other => panic!("native restoration user-stop status mismatch: {other:?}"),
    }
    assert_eq!(
        native_restoration_callbacks, 1,
        "NLIP should expose a restoration callback before honoring the stop"
    );

    match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::UserRequestedStop);
            assert!(
                partial_solution.is_some(),
                "IPOPT restoration user stop should retain a partial solution"
            );
            assert!(
                snapshots
                    .iter()
                    .any(|snapshot| snapshot.phase == IpoptIterationPhase::Restoration),
                "IPOPT should record the restoration-phase callback snapshot"
            );
        }
        other => panic!("IPOPT restoration user-stop status mismatch: {other:?}"),
    }
    assert_eq!(
        ipopt_restoration_callbacks, 1,
        "IPOPT should stop on the first restoration callback"
    );
}

#[test]
fn compare_native_and_ipopt_start_with_restoration_status() {
    skip_without_native_spral!();
    let problem = ImpossibleSquareEqualityProblem;
    let mut native_restoration_callbacks = 0usize;
    let native = solve_nlp_interior_point_with_control_callback(
        &problem,
        &[1.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 40;
            options.second_order_correction = false;
            options.start_with_restoration = true;
        }),
        |snapshot| {
            if snapshot.phase == InteriorPointIterationPhase::Restoration {
                native_restoration_callbacks += 1;
            }
            true
        },
    );
    let mut ipopt_restoration_callbacks = 0usize;
    let ipopt = solve_nlp_ipopt_with_control_callback(
        &problem,
        &[1.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 40;
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
            options
                .raw_options
                .push(IpoptRawOption::text("start_with_resto", "yes"));
        }),
        |snapshot| {
            if snapshot.phase == IpoptIterationPhase::Restoration {
                ipopt_restoration_callbacks += 1;
            }
            true
        },
    );

    match native {
        Err(InteriorPointSolveError::LocalInfeasibility { context }) => {
            assert!(
                context.failed_linear_solve.is_none(),
                "start_with_resto should enter restoration without emergency linear-solve context"
            );
            assert!(
                context.final_state.is_some(),
                "NLIP start-with-restoration failure should retain final state"
            );
        }
        other => panic!("native start-with-restoration status mismatch: {other:?}"),
    }
    assert!(
        native_restoration_callbacks > 0,
        "NLIP should expose restoration callbacks when start_with_restoration is set"
    );

    match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            snapshots,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::InfeasibleProblemDetected);
            assert!(
                partial_solution.is_some(),
                "IPOPT start-with-restoration failure should retain a partial solution"
            );
            assert!(
                snapshots
                    .iter()
                    .any(|snapshot| snapshot.phase == IpoptIterationPhase::Restoration),
                "IPOPT should record start-with-restoration callback snapshots"
            );
        }
        other => panic!("IPOPT start-with-restoration status mismatch: {other:?}"),
    }
    assert!(
        ipopt_restoration_callbacks > 0,
        "IPOPT should expose restoration callbacks when start_with_resto=yes"
    );
}

#[test]
fn compare_native_and_ipopt_expect_infeasible_multiplier_restoration() {
    skip_without_native_spral!();
    let problem = SquareEqualityQuadraticProblem;
    let mut native_restoration_callbacks = 0usize;
    let native = solve_nlp_interior_point_with_control_callback(
        &problem,
        &[8.0, -3.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 60;
            options.least_square_init_duals = true;
            options.second_order_correction = false;
            options.expect_infeasible_problem = true;
            options.expect_infeasible_problem_ytol = 1.0e-12;
        }),
        |snapshot| {
            if snapshot.phase == InteriorPointIterationPhase::Restoration {
                native_restoration_callbacks += 1;
            }
            true
        },
    )
    .expect("NLIP expect-infeasible square witness should solve");
    let mut ipopt_restoration_callbacks = 0usize;
    let ipopt = solve_nlp_ipopt_with_control_callback(
        &problem,
        &[8.0, -3.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 60;
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_duals", "yes"));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
            options
                .raw_options
                .push(IpoptRawOption::text("expect_infeasible_problem", "yes"));
            options.raw_options.push(IpoptRawOption::number(
                "expect_infeasible_problem_ytol",
                1.0e-12,
            ));
        }),
        |snapshot| {
            if snapshot.phase == IpoptIterationPhase::Restoration {
                ipopt_restoration_callbacks += 1;
            }
            true
        },
    )
    .expect("IPOPT expect-infeasible square witness should solve");

    assert!(
        native_restoration_callbacks > 0,
        "NLIP should enter restoration when expect_infeasible ytol is exceeded"
    );
    assert!(
        ipopt_restoration_callbacks > 0,
        "IPOPT should enter restoration when expect_infeasible_problem_ytol is exceeded"
    );
    assert_eq!(native.termination, InteriorPointTermination::Converged);
    assert_eq!(ipopt.status, IpoptRawStatus::SolveSucceeded);
    assert_native_matches_ipopt(
        "square_equality_expect_infeasible_restoration",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_adaptive_restoration_mu_update() {
    skip_without_native_spral!();
    let problem = SquareEqualityQuadraticProblem;
    let mut native_restoration_callbacks = 0usize;
    let native = solve_nlp_interior_point_with_control_callback(
        &problem,
        &[8.0, -3.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 80;
            options.least_square_init_duals = true;
            options.second_order_correction = false;
            options.expect_infeasible_problem = true;
            options.expect_infeasible_problem_ytol = 1.0e-12;
            options.restoration_mu_strategy = Some(InteriorPointMuStrategy::Adaptive);
            options.restoration_adaptive_mu_oracle = Some(InteriorPointAdaptiveMuOracle::Loqo);
            options.restoration_adaptive_mu_globalization =
                Some(InteriorPointAdaptiveMuGlobalization::NeverMonotoneMode);
        }),
        |snapshot| {
            if snapshot.phase == InteriorPointIterationPhase::Restoration {
                native_restoration_callbacks += 1;
            }
            true
        },
    )
    .expect("NLIP adaptive-restoration square witness should solve");
    let mut ipopt_restoration_callbacks = 0usize;
    let ipopt = solve_nlp_ipopt_with_control_callback(
        &problem,
        &[8.0, -3.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 80;
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_duals", "yes"));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
            options
                .raw_options
                .push(IpoptRawOption::text("expect_infeasible_problem", "yes"));
            options.raw_options.push(IpoptRawOption::number(
                "expect_infeasible_problem_ytol",
                1.0e-12,
            ));
            options
                .raw_options
                .push(IpoptRawOption::text("resto.mu_strategy", "adaptive"));
            options
                .raw_options
                .push(IpoptRawOption::text("resto.mu_oracle", "loqo"));
            options.raw_options.push(IpoptRawOption::text(
                "resto.adaptive_mu_globalization",
                "never-monotone-mode",
            ));
        }),
        |snapshot| {
            if snapshot.phase == IpoptIterationPhase::Restoration {
                ipopt_restoration_callbacks += 1;
            }
            true
        },
    )
    .expect("IPOPT adaptive-restoration square witness should solve");

    assert!(
        native_restoration_callbacks > 0,
        "NLIP should enter restoration with resto.mu_strategy=adaptive"
    );
    assert!(
        ipopt_restoration_callbacks > 0,
        "IPOPT should enter restoration with resto.mu_strategy=adaptive"
    );
    assert_eq!(native.termination, InteriorPointTermination::Converged);
    assert_eq!(ipopt.status, IpoptRawStatus::SolveSucceeded);
    assert_native_matches_ipopt(
        "square_equality_adaptive_restoration_mu_update",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_soft_restoration_profile() {
    skip_without_native_spral!();
    let backend = CallbackBackend::Aot;
    let problem = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let native = solve_native_with_options_ok(
        &problem,
        &[2.5, 3.0, 0.75],
        &[],
        native_options_with(|options| {
            options.max_iters = 120;
            options.alpha_min_frac = 0.05;
            options.second_order_correction = true;
            options.max_second_order_corrections = 1;
            options.watchdog_shortened_iter_trigger = 0;
            options.soft_restoration_pderror_reduction_factor = 1.0e6;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[2.5, 3.0, 0.75],
        &[],
        ipopt_options_with(|options| {
            options.max_iters = 120;
            enable_ipopt_trace_journal(options);
            options
                .raw_options
                .push(IpoptRawOption::number("alpha_min_frac", 0.05));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 1));
            options.raw_options.push(IpoptRawOption::integer(
                "watchdog_shortened_iter_trigger",
                0,
            ));
            options.raw_options.push(IpoptRawOption::number(
                "soft_resto_pderror_reduction_factor",
                1.0e6,
            ));
        }),
    );

    assert!(
        native
            .snapshots
            .iter()
            .any(|snapshot| snapshot.step_tag == Some('s')),
        "NLIP should accept a primal-dual-error soft restoration step"
    );
    assert!(
        parse_ipopt_step_tags(ipopt.journal_output.as_deref())
            .values()
            .any(|tag| tag == "s"),
        "IPOPT witness should exercise BacktrackingLineSearch::TrySoftRestoStep"
    );
    assert_native_matches_ipopt(
        "casadi_rosenbrock_soft_restoration",
        Some(backend),
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_restoration_successive_iteration_limit_status() {
    skip_without_native_spral!();
    let problem = ImpossibleSquareEqualityProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[1.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 40;
            options.max_restoration_iters = 0;
            options.second_order_correction = false;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[1.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 40;
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_resto_iter", 0));
        }),
    );

    match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations,
            context,
        }) => {
            assert_eq!(iterations, 3);
            assert!(
                context.final_state.is_some(),
                "NLIP restoration max_resto_iter failure should retain current failure state"
            );
        }
        other => panic!("native restoration max_resto_iter status mismatch: {other:?}"),
    }
    match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            assert_eq!(iterations, 3);
            assert!(
                partial_solution.is_some(),
                "IPOPT restoration max_resto_iter failure should retain a partial solution"
            );
        }
        other => panic!("IPOPT restoration max_resto_iter status mismatch: {other:?}"),
    }
}

#[test]
fn compare_native_and_ipopt_restoration_eval_error_max_iter_status_sweep() {
    skip_without_native_spral!();
    for nan_radius in [0.5, 0.75, 0.9, 0.99] {
        let problem = RestorationObjectiveEvalErrorProblem { nan_radius };
        let native = solve_nlp_interior_point(
            &problem,
            &[1.0],
            &[],
            &native_options_with(|options| {
                options.max_iters = 40;
                options.second_order_correction = false;
                options.soft_restoration_pderror_reduction_factor = 0.0;
            }),
        );
        let ipopt = solve_nlp_ipopt(
            &problem,
            &[1.0],
            &[],
            &ipopt_options_with(|options| {
                options.max_iters = 40;
                options
                    .raw_options
                    .push(IpoptRawOption::integer("max_soc", 0));
                options.raw_options.push(IpoptRawOption::number(
                    "soft_resto_pderror_reduction_factor",
                    0.0,
                ));
            }),
        );

        match native {
            Err(InteriorPointSolveError::MaxIterations { iterations, .. }) => {
                assert_eq!(iterations, 40, "nan_radius={nan_radius}");
            }
            other => panic!("native restoration eval-error status mismatch: {other:?}"),
        }
        match ipopt {
            Err(IpoptSolveError::Solve {
                status, iterations, ..
            }) => {
                assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
                assert_eq!(iterations, 40, "nan_radius={nan_radius}");
            }
            other => panic!("IPOPT restoration eval-error status mismatch: {other:?}"),
        }
    }
}

#[test]
#[ignore = "diagnostic sweep for restoration original-objective evaluation-error status parity"]
fn print_restoration_original_objective_eval_error_status_sweep() {
    skip_without_native_spral!();
    for nan_radius in [0.5, 0.75, 0.9, 0.99] {
        let problem = RestorationObjectiveEvalErrorProblem { nan_radius };
        let mut native_trace = Vec::new();
        let native = solve_nlp_interior_point_with_callback(
            &problem,
            &[1.0],
            &[],
            &native_options_with(|options| {
                options.max_iters = 40;
                options.second_order_correction = false;
                options.soft_restoration_pderror_reduction_factor = 0.0;
            }),
            |snapshot| native_trace.push(snapshot.clone()),
        );
        let mut ipopt_trace = Vec::new();
        let ipopt = solve_nlp_ipopt_with_control_callback(
            &problem,
            &[1.0],
            &[],
            &ipopt_options_with(|options| {
                options.max_iters = 40;
                enable_ipopt_trace_journal(options);
                options.journal_print_level = Some(12);
                options
                    .raw_options
                    .push(IpoptRawOption::integer("max_soc", 0));
                options.raw_options.push(IpoptRawOption::number(
                    "soft_resto_pderror_reduction_factor",
                    0.0,
                ));
            }),
            |snapshot| {
                ipopt_trace.push(snapshot.clone());
                true
            },
        );
        let native_label = match &native {
            Ok(summary) => format!("ok:{:?}", summary.termination),
            Err(error) => nlip_error_label(error),
        };
        let ipopt_label = match &ipopt {
            Ok(summary) => format!("ok:{:?}", summary.status),
            Err(IpoptSolveError::Solve { status, .. }) => format!("{status:?}"),
            Err(error) => format!("{error}"),
        };
        println!(
            "[resto-eval-sweep] nan_radius={nan_radius:.2} native={native_label} ipopt={ipopt_label}"
        );
        if let Err(InteriorPointSolveError::RestorationFailed { context, .. }) = &native {
            if let Some(final_state) = context.final_state.as_ref() {
                println!(
                    "[resto-eval-sweep] native_final iter={} phase={:?} tag={:?} prim={:.3e} dual={:.3e} comp={:.3e} obj={:.3e}",
                    final_state.iteration,
                    final_state.phase,
                    final_state.step_tag,
                    final_state.barrier_primal_inf.unwrap_or(0.0),
                    final_state.dual_inf,
                    final_state.comp_inf.unwrap_or(0.0),
                    final_state.objective,
                );
                if let Some(line_search) = final_state.line_search.as_ref() {
                    println!(
                        "[resto-eval-sweep] native_final_ls alpha0={:.3e} last={:.3e} alpha_min={:.3e} backs={} tiny={} tag={:?} rejected={}",
                        line_search.initial_alpha_pr,
                        line_search.last_tried_alpha,
                        line_search.alpha_min,
                        line_search.backtrack_count,
                        line_search.tiny_step,
                        line_search.step_tag,
                        line_search.rejected_trials.len(),
                    );
                }
            }
            if let Some(line_search) = context.failed_line_search.as_ref() {
                println!(
                    "[resto-eval-sweep] native_failed_ls alpha0={:.3e} last={:.3e} alpha_min={:.3e} backs={} tiny={} rejected={}",
                    line_search.initial_alpha_pr,
                    line_search.last_tried_alpha,
                    line_search.alpha_min,
                    line_search.backtrack_count,
                    line_search.tiny_step,
                    line_search.rejected_trials.len(),
                );
                for (tail_index, trial) in line_search
                    .rejected_trials
                    .iter()
                    .rev()
                    .take(5)
                    .collect::<Vec<_>>()
                    .iter()
                    .rev()
                    .enumerate()
                {
                    println!(
                        "[resto-eval-sweep] native_reject_tail[{tail_index}] alpha={:.3e} obj={:?} barr={:?} prim={:?} dual={:?} filter={:?} local={:?} suff_obj={:?} suff_viol={:?} switch={:?}",
                        trial.alpha,
                        trial.objective,
                        trial.barrier_objective,
                        trial.primal_inf,
                        trial.dual_inf,
                        trial.filter_acceptable,
                        trial.local_filter_acceptable,
                        trial.filter_sufficient_objective_reduction,
                        trial.filter_sufficient_violation_reduction,
                        trial.switching_condition_satisfied,
                    );
                }
            }
        }
        if let Some(journal) = ipopt_result_journal_output(&ipopt) {
            println!(
                "[resto-eval-sweep] ipopt_tags={}",
                ipopt_result_step_tag_summary(&ipopt)
            );
            if let Err(IpoptSolveError::Solve { snapshots, .. }) = &ipopt
                && let Some(last) = snapshots.last()
            {
                println!(
                    "[resto-eval-sweep] ipopt_last iter={} phase={:?} obj={:.3e} prim={:.3e} dual={:.3e} alpha={:.3e} ls={}",
                    last.iteration,
                    last.phase,
                    last.objective,
                    last.primal_inf,
                    last.dual_inf,
                    last.alpha_pr,
                    last.line_search_trials,
                );
            }
            for line in journal
                .lines()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    trimmed
                        .as_bytes()
                        .first()
                        .is_some_and(|byte| byte.is_ascii_digit())
                })
                .rev()
                .take(5)
                .collect::<Vec<_>>()
                .iter()
                .rev()
            {
                println!("[resto-eval-sweep][ipopt_iter] {line}");
            }
            if nan_radius == 0.5 {
                for line in journal
                    .lines()
                    .filter(|line| {
                        let trimmed = line.trim_start();
                        trimmed
                            .as_bytes()
                            .first()
                            .is_some_and(|byte| byte.is_ascii_digit())
                    })
                    .take(24)
                {
                    println!("[resto-eval-sweep][ipopt_iter_head] {line}");
                }
                for line in journal
                    .lines()
                    .filter(|line| {
                        let trimmed = line.trim_start();
                        trimmed
                            .as_bytes()
                            .first()
                            .is_some_and(|byte| byte.is_ascii_digit())
                            && trimmed.contains('r')
                    })
                    .take(24)
                {
                    println!("[resto-eval-sweep][ipopt_resto_head] {line}");
                }
            }
            for line in journal
                .lines()
                .filter(|line| line.contains("ALPHA_MIN"))
                .rev()
                .take(5)
                .collect::<Vec<_>>()
                .iter()
                .rev()
            {
                println!("[resto-eval-sweep][ipopt_alpha_min] {line}");
            }
            let tail = journal
                .lines()
                .filter(|line| {
                    line.contains("restoration")
                        || line.contains("alpha")
                        || line.contains("ALPHA")
                        || line.contains("Tiny")
                        || line.contains("Warning")
                })
                .rev()
                .take(8)
                .collect::<Vec<_>>();
            for line in tail.iter().rev() {
                println!("[resto-eval-sweep][ipopt] {line}");
            }
        }
        if nan_radius == 0.5 {
            for snapshot in native_trace
                .iter()
                .filter(|snapshot| snapshot.phase == InteriorPointIterationPhase::Restoration)
                .take(8)
            {
                println!(
                    "[resto-eval-sweep] native_trace iter={} tag={:?} x0={:.6e} obj={:.6e} prim={:?} dual={:.6e} mu={:?} alpha={:?} ls={}",
                    snapshot.iteration,
                    snapshot.step_tag,
                    snapshot.x.first().copied().unwrap_or(0.0),
                    snapshot.objective,
                    snapshot.barrier_primal_inf,
                    snapshot.dual_inf,
                    snapshot.barrier_parameter,
                    snapshot.alpha_pr,
                    snapshot.line_search_trials,
                );
            }
            for snapshot in native_trace
                .iter()
                .filter(|snapshot| snapshot.phase == InteriorPointIterationPhase::Restoration)
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .iter()
                .rev()
            {
                println!(
                    "[resto-eval-sweep] native_trace_tail iter={} tag={:?} x0={:.6e} obj={:.6e} prim={:?} dual={:.6e} mu={:?} alpha={:?} ls={}",
                    snapshot.iteration,
                    snapshot.step_tag,
                    snapshot.x.first().copied().unwrap_or(0.0),
                    snapshot.objective,
                    snapshot.barrier_primal_inf,
                    snapshot.dual_inf,
                    snapshot.barrier_parameter,
                    snapshot.alpha_pr,
                    snapshot.line_search_trials,
                );
            }
            for snapshot in ipopt_trace
                .iter()
                .filter(|snapshot| snapshot.phase == IpoptIterationPhase::Restoration)
                .take(8)
            {
                println!(
                    "[resto-eval-sweep] ipopt_trace iter={} x0={:.6e} obj={:.6e} prim={:.6e} dual={:.6e} mu={:.6e} alpha={:.6e} ls={}",
                    snapshot.iteration,
                    snapshot.x.first().copied().unwrap_or(0.0),
                    snapshot.objective,
                    snapshot.primal_inf,
                    snapshot.dual_inf,
                    snapshot.barrier_parameter,
                    snapshot.alpha_pr,
                    snapshot.line_search_trials,
                );
            }
        }
        if matches!(
            (&native, &ipopt),
            (
                Err(InteriorPointSolveError::RestorationFailed { .. }),
                Err(IpoptSolveError::Solve {
                    status: IpoptRawStatus::RestorationFailed,
                    ..
                })
            )
        ) {
            return;
        }
    }
}

#[test]
fn compare_native_and_ipopt_nonfinite_trial_backtracks_like_eval_error() {
    skip_without_native_spral!();
    let problem = NonFiniteFirstTrialQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 1;
            options.second_order_correction = false;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 1;
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations: 1,
            context,
        }) => context
            .final_state
            .expect("NLIP nonfinite-trial max-iter witness should retain accepted state"),
        other => panic!("native nonfinite-trial status mismatch: {other:?}"),
    };
    let (ipopt_snapshots, ipopt_iterations) = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            (snapshots, iterations)
        }
        other => panic!("IPOPT nonfinite-trial status mismatch: {other:?}"),
    };
    assert_eq!(ipopt_iterations, 1);
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 1 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT nonfinite-trial max-iter witness should retain accepted iteration");

    assert_abs_diff_eq!(native_state.x[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(ipopt_state.x[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(native_state.x[0], ipopt_state.x[0], epsilon = 1e-12);
    assert_abs_diff_eq!(
        native_state.alpha_pr.expect("NLIP accepted alpha_pr"),
        ipopt_state.alpha_pr,
        epsilon = 1e-12
    );
    assert_eq!(ipopt_state.line_search_trials, 2);
    let native_line_search = native_state
        .line_search
        .as_ref()
        .expect("NLIP accepted state should retain line-search diagnostics");
    assert_eq!(native_line_search.backtrack_count, 1);
    assert_eq!(native_line_search.rejected_trials.len(), 1);
    assert!(
        native_line_search.rejected_trials[0].objective.is_none(),
        "evaluation-error trials should not retain finite trial metrics"
    );
}

#[test]
fn compare_native_and_ipopt_tiny_step_eval_error_falls_back_to_backtracking() {
    skip_without_native_spral!();
    let problem = NonFiniteFirstTrialQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 1;
            options.second_order_correction = false;
            options.tiny_step_tol = 1e6;
            options.tiny_step_y_tol = 1e6;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 1;
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_tol", 1e6));
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_y_tol", 1e6));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations: 1,
            context,
        }) => context
            .final_state
            .expect("NLIP tiny-step eval-error witness should retain accepted state"),
        other => panic!("native tiny-step eval-error status mismatch: {other:?}"),
    };
    let ipopt_state = match ipopt {
        Err(IpoptSolveError::Solve {
            status, snapshots, ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            snapshots
                .into_iter()
                .find(|snapshot| {
                    snapshot.iteration == 1
                        && snapshot.phase == optimization::IpoptIterationPhase::Regular
                })
                .expect("IPOPT tiny-step eval-error witness should retain accepted iteration")
        }
        other => panic!("IPOPT tiny-step eval-error status mismatch: {other:?}"),
    };
    assert_abs_diff_eq!(native_state.x[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(ipopt_state.x[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(
        native_state.alpha_pr.expect("NLIP accepted alpha_pr"),
        ipopt_state.alpha_pr,
        epsilon = 1e-12
    );
    assert!(
        !native_state
            .events
            .contains(&InteriorPointIterationEvent::TinyStep),
        "evaluation failure should clear the unchecked tiny-step accept path before the regular backtracking accept"
    );
    let native_line_search = native_state
        .line_search
        .as_ref()
        .expect("NLIP accepted state should retain line-search diagnostics");
    assert_eq!(native_line_search.backtrack_count, 1);
    assert_eq!(ipopt_state.line_search_trials, 2);
}

#[test]
fn compare_native_and_ipopt_accept_every_trial_step() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.max_iters = 1;
            options.accept_every_trial_step = true;
            options.second_order_correction = false;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 1;
            enable_ipopt_trace_journal(options);
            options
                .raw_options
                .push(IpoptRawOption::text("accept_every_trial_step", "yes"));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations: 1,
            context,
        }) => context
            .final_state
            .expect("NLIP accept-every max-iter witness should retain accepted state"),
        other => panic!("native accept-every status mismatch: {other:?}"),
    };
    let (ipopt_snapshots, journal_output) = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations: 1,
            snapshots,
            journal_output,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            (snapshots, journal_output)
        }
        other => panic!("IPOPT accept-every status mismatch: {other:?}"),
    };
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 1 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT accept-every max-iter witness should retain accepted iteration");
    let native_line_search = native_state
        .line_search
        .as_ref()
        .expect("NLIP accept-every state should retain line-search diagnostics");
    assert_eq!(native_line_search.backtrack_count, 0);
    assert_eq!(ipopt_state.line_search_trials.saturating_sub(1), 0);
    assert_vec_close("accept_every x", &native_state.x, &ipopt_state.x, 1e-10);
    assert_abs_diff_eq!(
        native_state.alpha_pr.expect("NLIP accepted alpha_pr"),
        ipopt_state.alpha_pr,
        epsilon = 1e-12
    );
    assert!(
        parse_ipopt_info_strings(journal_output.as_deref())
            .values()
            .any(|info| info.contains("MaxS")),
        "IPOPT should report BacktrackingLineSearch::accept_every_trial_step via MaxS"
    );
}

#[test]
fn compare_native_and_ipopt_accept_after_max_steps_after_eval_error() {
    skip_without_native_spral!();
    let problem = NonFiniteFirstTrialQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 1;
            options.accept_after_max_steps = Some(1);
            options.second_order_correction = false;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 1;
            enable_ipopt_trace_journal(options);
            options
                .raw_options
                .push(IpoptRawOption::integer("accept_after_max_steps", 1));
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations: 1,
            context,
        }) => context
            .final_state
            .expect("NLIP accept-after max-iter witness should retain accepted state"),
        other => panic!("native accept-after status mismatch: {other:?}"),
    };
    let (ipopt_snapshots, journal_output) = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            iterations: 1,
            snapshots,
            journal_output,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            (snapshots, journal_output)
        }
        other => panic!("IPOPT accept-after status mismatch: {other:?}"),
    };
    let ipopt_state = ipopt_snapshots
        .iter()
        .find(|snapshot| {
            snapshot.iteration == 1 && snapshot.phase == optimization::IpoptIterationPhase::Regular
        })
        .expect("IPOPT accept-after max-iter witness should retain accepted iteration");

    assert_abs_diff_eq!(native_state.x[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(ipopt_state.x[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(
        native_state.alpha_pr.expect("NLIP accepted alpha_pr"),
        ipopt_state.alpha_pr,
        epsilon = 1e-12
    );
    assert_eq!(ipopt_state.line_search_trials, 2);
    let native_line_search = native_state
        .line_search
        .as_ref()
        .expect("NLIP accept-after state should retain line-search diagnostics");
    assert_eq!(native_line_search.backtrack_count, 1);
    assert!(
        parse_ipopt_info_strings(journal_output.as_deref())
            .values()
            .any(|info| info.contains("e") && info.contains("MaxS")),
        "IPOPT should report evaluation-error cutback followed by accept_after_max_steps MaxS"
    );
}

#[test]
fn compare_native_and_ipopt_impossible_square_equality_restoration_status() {
    skip_without_native_spral!();
    let problem = ImpossibleSquareEqualityProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[1.0],
        &[],
        &native_options_with(|options| {
            options.max_iters = 40;
            options.second_order_correction = false;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[1.0],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 40;
            options
                .raw_options
                .push(IpoptRawOption::integer("max_soc", 0));
        }),
    );

    match native {
        Err(InteriorPointSolveError::LocalInfeasibility { context }) => {
            assert!(
                context.final_state.is_some(),
                "NLIP local-infeasibility failure should retain final state"
            );
        }
        other => panic!("native impossible-square-equality status mismatch: {other:?}"),
    }
    match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            partial_solution,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::InfeasibleProblemDetected);
            assert!(
                partial_solution.is_some(),
                "IPOPT local-infeasibility failure should retain a partial solution"
            );
        }
        other => panic!("IPOPT impossible-square-equality status mismatch: {other:?}"),
    }
}

#[test]
fn compare_native_and_ipopt_acceptable_termination_status() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.overall_tol = 1e-14;
            options.dual_tol = 1e-14;
            options.constraint_tol = 1e-14;
            options.complementarity_tol = 1e-14;
            options.acceptable_tol = 1e-6;
            options.acceptable_dual_inf_tol = 1e-6;
            options.acceptable_constr_viol_tol = 1e-6;
            options.acceptable_compl_inf_tol = 1e-6;
            options.acceptable_obj_change_tol = 1e20;
            options.acceptable_iter = 1;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.tol = 1e-14;
            options.dual_tol = Some(1e-14);
            options.constraint_tol = Some(1e-14);
            options.complementarity_tol = Some(1e-14);
            options.acceptable_tol = Some(1e-6);
            options
                .raw_options
                .push(IpoptRawOption::integer("acceptable_iter", 1));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_dual_inf_tol", 1e-6));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_constr_viol_tol", 1e-6));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_compl_inf_tol", 1e-6));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_obj_change_tol", 1e20));
        }),
    );

    assert_eq!(native.termination, InteriorPointTermination::Acceptable);
    assert_eq!(native.status_kind, InteriorPointStatusKind::Warning);
    assert_eq!(ipopt.status, IpoptRawStatus::SolvedToAcceptableLevel);
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_acceptable_termination",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_acceptable_iter_backup_before_termination() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.overall_tol = 1e-14;
            options.dual_tol = 1e-14;
            options.constraint_tol = 1e-14;
            options.complementarity_tol = 1e-14;
            options.acceptable_tol = 1e-6;
            options.acceptable_dual_inf_tol = 1e-6;
            options.acceptable_constr_viol_tol = 1e-6;
            options.acceptable_compl_inf_tol = 1e-6;
            options.acceptable_obj_change_tol = 1e20;
            options.acceptable_iter = 2;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options.tol = 1e-14;
            options.dual_tol = Some(1e-14);
            options.constraint_tol = Some(1e-14);
            options.complementarity_tol = Some(1e-14);
            options.acceptable_tol = Some(1e-6);
            options
                .raw_options
                .push(IpoptRawOption::integer("acceptable_iter", 2));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_dual_inf_tol", 1e-6));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_constr_viol_tol", 1e-6));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_compl_inf_tol", 1e-6));
            options
                .raw_options
                .push(IpoptRawOption::number("acceptable_obj_change_tol", 1e20));
        }),
    );

    assert_eq!(native.termination, InteriorPointTermination::Acceptable);
    assert_eq!(native.status_kind, InteriorPointStatusKind::Warning);
    assert_eq!(ipopt.status, IpoptRawStatus::SolvedToAcceptableLevel);
    assert!(
        nlip_accepted_trace(&native).len() >= 2,
        "acceptable_iter=2 should pass through an acceptable backup point before termination"
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_acceptable_iter_backup",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_forced_tiny_step_acceptance() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.tiny_step_tol = 1e6;
            options.tiny_step_y_tol = 1e6;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_tol", 1e6));
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_y_tol", 1e6));
        }),
    );

    assert!(
        native.snapshots.iter().any(|snapshot| snapshot
            .events
            .contains(&InteriorPointIterationEvent::TinyStep)),
        "forced tiny-step profile should exercise the NLIP tiny-step branch"
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_forced_tiny_step",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_forced_tiny_step_without_dual_tiny_state() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;
    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.max_iters = 1;
            options.tiny_step_tol = 1e6;
            options.tiny_step_y_tol = 0.0;
        }),
    );
    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options.max_iters = 1;
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_tol", 1e6));
            options
                .raw_options
                .push(IpoptRawOption::number("tiny_step_y_tol", 0.0));
        }),
    );

    let native_state = match native {
        Err(InteriorPointSolveError::MaxIterations {
            iterations: 1,
            context,
        }) => context
            .final_state
            .expect("NLIP tiny-step dual threshold witness should retain accepted state"),
        other => panic!("native tiny-step dual threshold status mismatch: {other:?}"),
    };
    let (ipopt_state, ipopt_tags) = match ipopt {
        Err(IpoptSolveError::Solve {
            status,
            snapshots,
            journal_output,
            ..
        }) => {
            assert_eq!(status, IpoptRawStatus::MaximumIterationsExceeded);
            let state = snapshots
                .into_iter()
                .find(|snapshot| {
                    snapshot.iteration == 1
                        && snapshot.phase == optimization::IpoptIterationPhase::Regular
                })
                .expect("IPOPT tiny-step dual threshold witness should retain accepted iteration");
            (state, parse_ipopt_step_tags(journal_output.as_deref()))
        }
        other => panic!("IPOPT tiny-step dual threshold status mismatch: {other:?}"),
    };

    assert_abs_diff_eq!(native_state.x[0], ipopt_state.x[0], epsilon = 1e-10);
    assert_abs_diff_eq!(native_state.x[1], ipopt_state.x[1], epsilon = 1e-10);
    assert_abs_diff_eq!(
        native_state.alpha_pr.expect("NLIP accepted alpha_pr"),
        ipopt_state.alpha_pr,
        epsilon = 1e-12
    );
    assert!(
        native_state.step_tag == Some('t'),
        "forced tiny-step profile should accept an unchecked tiny step"
    );
    assert!(
        native_state.step_tag != Some('T'),
        "zero tiny_step_y_tol should keep tiny_step_last_iteration false"
    );
    assert!(
        !ipopt_tags.values().any(|tag| tag == "T"),
        "IPOPT should not emit the repeated-tiny-step marker with zero tiny_step_y_tol"
    );
}

#[test]
fn compare_native_and_ipopt_with_least_square_primal_initialization() {
    skip_without_native_spral!();

    let problem = LinearlyConstrainedQuadraticProblem;
    let native_initial = solve_native_initial_snapshot(
        &problem,
        &[0.1, 0.9],
        native_options_with(|options| {
            options.max_iters = 0;
            options.least_square_init_primal = true;
        }),
    );
    let ipopt_initial = solve_ipopt_initial_snapshot(
        &problem,
        &[0.1, 0.9],
        ipopt_options_with(|options| {
            options.max_iters = 0;
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_primal", "yes"));
        }),
    );
    assert_vec_close(
        "least_square_init_primal x",
        &native_initial.x,
        &ipopt_initial.x,
        1e-9,
    );
    assert_vec_close(
        "least_square_init_primal slack",
        native_initial
            .slack_primal
            .as_deref()
            .expect("NLIP initial slack"),
        &ipopt_initial.internal_slack,
        1e-9,
    );
    assert_vec_close(
        "least_square_init_primal y_c",
        native_initial
            .equality_multipliers
            .as_deref()
            .expect("NLIP initial equality multipliers"),
        &ipopt_initial.equality_multipliers,
        1e-9,
    );
    assert_vec_close(
        "least_square_init_primal y_d",
        native_initial
            .inequality_multipliers
            .as_deref()
            .expect("NLIP initial inequality multipliers"),
        &ipopt_initial.inequality_multipliers,
        1e-9,
    );

    let native = solve_native_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.least_square_init_primal = true;
        }),
    );
    let ipopt = solve_ipopt_with_options_ok(
        &problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_primal", "yes"));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_least_square_init_primal",
        None,
        &native,
        &ipopt,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_with_least_square_dual_initialization() {
    skip_without_native_spral!();

    let constrained_problem = LinearlyConstrainedQuadraticProblem;
    let native_initial = solve_native_initial_snapshot(
        &constrained_problem,
        &[0.1, 0.9],
        native_options_with(|options| {
            options.max_iters = 0;
            options.least_square_init_duals = true;
        }),
    );
    let ipopt_initial = solve_ipopt_initial_snapshot(
        &constrained_problem,
        &[0.1, 0.9],
        ipopt_options_with(|options| {
            options.max_iters = 0;
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_duals", "yes"));
        }),
    );
    assert_vec_close(
        "least_square_init_duals y_c",
        native_initial
            .equality_multipliers
            .as_deref()
            .expect("NLIP initial y_c"),
        &ipopt_initial.equality_multipliers,
        1e-9,
    );
    assert_vec_close(
        "least_square_init_duals y_d",
        native_initial
            .inequality_multipliers
            .as_deref()
            .expect("NLIP initial y_d"),
        &ipopt_initial.inequality_multipliers,
        1e-9,
    );
    assert_vec_close(
        "least_square_init_duals v_U",
        native_initial
            .slack_multipliers
            .as_deref()
            .expect("NLIP initial slack multipliers"),
        &ipopt_initial.slack_upper_bound_multipliers,
        1e-9,
    );

    let native_constrained = solve_native_with_options_ok(
        &constrained_problem,
        &[0.1, 0.9],
        &[],
        native_options_with(|options| {
            options.least_square_init_duals = true;
        }),
    );
    let ipopt_constrained = solve_ipopt_with_options_ok(
        &constrained_problem,
        &[0.1, 0.9],
        &[],
        ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_duals", "yes"));
        }),
    );
    assert_native_matches_ipopt(
        "linearly_constrained_quadratic_least_square_init_duals",
        None,
        &native_constrained,
        &ipopt_constrained,
        1e-6,
        1e-6,
    );

    let bound_problem = BoundConstrainedQuadraticProblem;
    let native_initial = solve_native_initial_snapshot(
        &bound_problem,
        &[-10.0, 10.0],
        native_options_with(|options| {
            options.max_iters = 0;
            // IPOPT's OrigIpoptNLP::relax_bounds caps bound relaxation by
            // constr_viol_tol. Keep this focused witness on the historical
            // strict cap so it
            // isolates DefaultIterateInitializer::CalculateLeastSquareDuals.
            options.constraint_tol = 1e-8;
            options.least_square_init_duals = true;
        }),
    );
    let ipopt_initial = solve_ipopt_initial_snapshot(
        &bound_problem,
        &[-10.0, 10.0],
        ipopt_options_with(|options| {
            options.max_iters = 0;
            options.constraint_tol = Some(1e-8);
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_duals", "yes"));
        }),
    );
    assert_vec_close(
        "least_square_init_duals z_L",
        native_initial
            .lower_bound_multipliers
            .as_deref()
            .expect("NLIP initial z_L"),
        &ipopt_initial.lower_bound_multipliers,
        1e-9,
    );
    assert_vec_close(
        "least_square_init_duals z_U",
        native_initial
            .upper_bound_multipliers
            .as_deref()
            .expect("NLIP initial z_U"),
        &ipopt_initial.upper_bound_multipliers,
        1e-9,
    );

    let native_bound = solve_native_with_options_ok(
        &bound_problem,
        &[-10.0, 10.0],
        &[],
        native_options_with(|options| {
            options.constraint_tol = 1e-8;
            options.least_square_init_duals = true;
        }),
    );
    let ipopt_bound = solve_ipopt_with_options_ok(
        &bound_problem,
        &[-10.0, 10.0],
        &[],
        ipopt_options_with(|options| {
            options.constraint_tol = Some(1e-8);
            options
                .raw_options
                .push(IpoptRawOption::text("least_square_init_duals", "yes"));
        }),
    );
    assert_native_matches_ipopt(
        "bound_constrained_quadratic_least_square_init_duals",
        None,
        &native_bound,
        &ipopt_bound,
        1e-6,
        1e-6,
    );
}

#[test]
fn compare_native_and_ipopt_reject_invalid_initializer_push_options() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;

    macro_rules! assert_rejected {
        ($field:ident, $value:expr, $raw_name:literal) => {{
            let native = solve_nlp_interior_point(
                &problem,
                &[0.1, 0.9],
                &[],
                &native_options_with(|options| {
                    options.$field = $value;
                }),
            );
            assert!(
                matches!(
                    native,
                    Err(InteriorPointSolveError::InvalidInput(ref message))
                        if message.contains(stringify!($field))
                ),
                "NLIP should reject {}={}: {:?}",
                stringify!($field),
                $value,
                native
            );

            let ipopt = solve_nlp_ipopt(
                &problem,
                &[0.1, 0.9],
                &[],
                &ipopt_options_with(|options| {
                    options
                        .raw_options
                        .push(IpoptRawOption::number($raw_name, $value));
                }),
            );
            assert!(
                matches!(
                    ipopt,
                    Err(IpoptSolveError::OptionRejected { ref name }) if name == $raw_name
                ),
                "IPOPT should reject {}={}: {:?}",
                $raw_name,
                $value,
                ipopt
            );
        }};
    }

    // IpDefaultIterateInitializer::RegisterOptions makes these public options
    // strictly positive, with the fractional variants capped at 0.5.
    assert_rejected!(bound_push, 0.0, "bound_push");
    assert_rejected!(bound_frac, 0.0, "bound_frac");
    assert_rejected!(bound_frac, 0.5000000001, "bound_frac");
    assert_rejected!(slack_bound_push, 0.0, "slack_bound_push");
    assert_rejected!(slack_bound_frac, 0.0, "slack_bound_frac");
    assert_rejected!(slack_bound_frac, 0.5000000001, "slack_bound_frac");
}

#[test]
fn compare_native_and_ipopt_reject_invalid_filter_theta_factors() {
    skip_without_native_spral!();
    let problem = LinearlyConstrainedQuadraticProblem;

    let native = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.theta_min_fact = 1.0;
            options.theta_max_fact = 1.0;
        }),
    );
    assert!(
        matches!(
            native,
            Err(InteriorPointSolveError::InvalidInput(ref message))
                if message.contains("theta_min_fact")
        ),
        "NLIP should reject theta_min_fact >= theta_max_fact: {:?}",
        native
    );

    let ipopt = solve_nlp_ipopt(
        &problem,
        &[0.1, 0.9],
        &[],
        &ipopt_options_with(|options| {
            options
                .raw_options
                .push(IpoptRawOption::number("theta_min_fact", 1.0));
            options
                .raw_options
                .push(IpoptRawOption::number("theta_max_fact", 1.0));
        }),
    );
    assert!(
        matches!(
            ipopt,
            Err(IpoptSolveError::Solve {
                status: IpoptRawStatus::InvalidOption,
                ..
            })
        ),
        "IPOPT should reject theta_min_fact >= theta_max_fact: {:?}",
        ipopt
    );

    macro_rules! assert_lower_bound_rejected {
        ($field:ident, $raw_name:literal) => {{
            let native = solve_nlp_interior_point(
                &problem,
                &[0.1, 0.9],
                &[],
                &native_options_with(|options| {
                    options.$field = 0.0;
                }),
            );
            assert!(
                matches!(
                    native,
                    Err(InteriorPointSolveError::InvalidInput(ref message))
                        if message.contains(stringify!($field))
                ),
                "NLIP should reject {}=0: {:?}",
                stringify!($field),
                native
            );

            let ipopt = solve_nlp_ipopt(
                &problem,
                &[0.1, 0.9],
                &[],
                &ipopt_options_with(|options| {
                    options
                        .raw_options
                        .push(IpoptRawOption::number($raw_name, 0.0));
                }),
            );
            assert!(
                matches!(
                    ipopt,
                    Err(IpoptSolveError::OptionRejected { ref name }) if name == $raw_name
                ),
                "IPOPT should reject {}=0: {:?}",
                $raw_name,
                ipopt
            );
        }};
    }

    // IpFilterLSAcceptor::RegisterOptions enforces strict positivity, and
    // InitializeImpl rejects theta_min_fact >= theta_max_fact.
    assert_lower_bound_rejected!(theta_min_fact, "theta_min_fact");
    assert_lower_bound_rejected!(theta_max_fact, "theta_max_fact");
}

#[test]
fn compare_invalid_shape_rejected_by_both_solvers() {
    let invalid = invalid_shape_problem();
    assert!(solve_nlp_interior_point(&invalid, &[0.0, 0.0], &[], &native_options()).is_err());
    assert!(solve_nlp_ipopt(&invalid, &[0.0, 0.0], &[], &ipopt_options()).is_err());
}
