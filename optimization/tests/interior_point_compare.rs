#![cfg(feature = "ipopt")]

use approx::assert_abs_diff_eq;
use optimization::{
    CCS, CompiledNlpProblem, ConstraintBounds, FilterAcceptanceMode, InteriorPointIterationPhase,
    InteriorPointOptions, InteriorPointStepKind, IpoptOptions, ParameterMatrix,
    apply_native_spral_parity_to_ipopt_options, apply_native_spral_parity_to_nlip_options,
    solve_nlp_interior_point, solve_nlp_ipopt,
};
use rstest::rstest;
use std::collections::BTreeMap;
use std::sync::OnceLock;

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

fn assert_local_spral_ipopt_provenance(summary: &optimization::IpoptSummary) {
    let provenance = summary
        .provenance
        .as_ref()
        .expect("expected IPOPT provenance for native-SPRAL parity runs");
    assert_local_spral_ipopt_environment(provenance);
}

const LOCAL_SPRAL_IPOPT_VERSION: &str = "3.14.20";
const LOCAL_SPRAL_IPOPT_PREFIX: &str = "/Users/greg/local/ipopt-spral";

fn local_spral_ipopt_environment_error(
    provenance: &optimization::IpoptProvenance,
) -> Option<String> {
    match provenance.pkg_config_version.as_deref() {
        Some(LOCAL_SPRAL_IPOPT_VERSION) => {}
        Some(version) => {
            return Some(format!(
                "pkg-config ipopt version {version} does not match expected {LOCAL_SPRAL_IPOPT_VERSION}"
            ));
        }
        None => return Some("pkg-config ipopt version is unavailable".to_string()),
    }

    let Some(flags) = provenance.pkg_config_cflags_libs.as_deref() else {
        return Some("pkg-config ipopt cflags/libs are unavailable".to_string());
    };
    if !flags.contains(LOCAL_SPRAL_IPOPT_PREFIX) {
        return Some(format!(
            "pkg-config ipopt flags do not contain expected prefix {LOCAL_SPRAL_IPOPT_PREFIX}: {flags}"
        ));
    }

    match provenance.linear_solver_default.as_deref() {
        Some("spral") => None,
        Some(default) => Some(format!(
            "ipopt linear_solver default {default} does not match expected spral"
        )),
        None => Some("ipopt linear_solver default is unavailable".to_string()),
    }
}

fn assert_local_spral_ipopt_environment(provenance: &optimization::IpoptProvenance) {
    if let Some(error) = local_spral_ipopt_environment_error(provenance) {
        panic!("unsupported local SPRAL IPOPT environment: {error}; provenance={provenance:?}");
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

fn native_spral_available() -> bool {
    static AVAILABLE: OnceLock<Result<(), String>> = OnceLock::new();
    match AVAILABLE.get_or_init(|| {
        spral_ssids::NativeSpral::load()
            .map(|_| ())
            .map_err(|error| error.to_string())
    }) {
        Ok(()) => true,
        Err(error) => {
            eprintln!("skipping native SPRAL/IPOPT comparison: {error}");
            false
        }
    }
}

fn local_spral_ipopt_environment_available() -> bool {
    static AVAILABLE: OnceLock<Result<(), String>> = OnceLock::new();
    match AVAILABLE.get_or_init(|| {
        let provenance = optimization::capture_ipopt_provenance();
        if let Some(error) = local_spral_ipopt_environment_error(&provenance) {
            Err(format!("{error}; provenance={provenance:?}"))
        } else {
            Ok(())
        }
    }) {
        Ok(()) => true,
        Err(error) => {
            eprintln!("skipping native SPRAL/IPOPT comparison: {error}");
            false
        }
    }
}

macro_rules! skip_without_native_spral {
    () => {
        if !native_spral_available() || !local_spral_ipopt_environment_available() {
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
    eprintln!(
        "[compare] {problem_name} native_linear_solver={:?} ipopt_provenance={:?}",
        native.linear_solver, ipopt.provenance,
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
    assert_eq!(
        native.linear_solver,
        optimization::InteriorPointLinearSolver::NativeSpralSsids
    );
    assert_local_spral_ipopt_provenance(ipopt);
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
    assert_local_spral_ipopt_environment(&optimization::capture_ipopt_provenance());
    let solve_result = solve_nlp_ipopt(problem, x0, parameters, &ipopt_options());
    assert!(solve_result.is_ok(), "Ipopt solve failed: {solve_result:?}");
    match solve_result {
        Ok(summary) => {
            assert_local_spral_ipopt_provenance(&summary);
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
fn compare_invalid_shape_rejected_by_both_solvers() {
    let invalid = invalid_shape_problem();
    assert!(solve_nlp_interior_point(&invalid, &[0.0, 0.0], &[], &native_options()).is_err());
    assert!(solve_nlp_ipopt(&invalid, &[0.0, 0.0], &[], &ipopt_options()).is_err());
}
