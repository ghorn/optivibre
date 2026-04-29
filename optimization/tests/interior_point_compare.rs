#![cfg(feature = "ipopt")]

use approx::assert_abs_diff_eq;
use optimization::{
    CCS, CompiledNlpProblem, ConstraintBounds, FilterAcceptanceMode,
    InteriorPointAlphaForYStrategy, InteriorPointBoundMultiplierInitMethod,
    InteriorPointIterationEvent, InteriorPointIterationPhase, InteriorPointIterationSnapshot,
    InteriorPointOptions, InteriorPointSolveError, InteriorPointStatusKind, InteriorPointStepKind,
    InteriorPointTermination, IpoptIterationSnapshot, IpoptOptions, IpoptRawOption, IpoptRawStatus,
    IpoptSolveError, ParameterMatrix, apply_native_spral_parity_to_ipopt_options,
    apply_native_spral_parity_to_nlip_options, solve_nlp_interior_point,
    solve_nlp_interior_point_with_callback, solve_nlp_ipopt,
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
        let Some(alpha_pr_token) = tokens.iter().rev().nth(1).copied() else {
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

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0.0, |acc, (lhs_value, rhs_value)| {
            acc.max((lhs_value - rhs_value).abs())
        })
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
fn native_rejects_unported_least_square_dual_initialization() {
    let problem = LinearlyConstrainedQuadraticProblem;
    let result = solve_nlp_interior_point(
        &problem,
        &[0.1, 0.9],
        &[],
        &native_options_with(|options| {
            options.least_square_init_duals = true;
        }),
    );
    match result {
        Err(InteriorPointSolveError::InvalidInput(message)) => {
            assert!(
                message.contains("DefaultIterateInitializer::CalculateLeastSquareDuals"),
                "unexpected unsupported-branch error: {message}"
            );
        }
        other => panic!("least_square_init_duals should be guarded until ported: {other:?}"),
    }
}

#[test]
fn compare_invalid_shape_rejected_by_both_solvers() {
    let invalid = invalid_shape_problem();
    assert!(solve_nlp_interior_point(&invalid, &[0.0, 0.0], &[], &native_options()).is_err());
    assert!(solve_nlp_ipopt(&invalid, &[0.0, 0.0], &[], &ipopt_options()).is_err());
}
