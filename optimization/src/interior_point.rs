use clarabel::algebra::CscMatrix;
use clarabel::qdldl::{QDLDLFactorisation, QDLDLSettings};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use spral_ssids::{
    AnalyseInfo as SpralAnalyseInfo, Inertia as SpralInertia, NativeSpral as NativeSpralLibrary,
    NativeSpralSession, NumericFactor as SpralNumericFactor,
    NumericFactorOptions as SpralNumericFactorOptions, OrderingStrategy as SpralOrderingStrategy,
    PivotMethod as SpralPivotMethod, SsidsOptions as SpralSsidsOptions,
    SymbolicFactor as SpralSymbolicFactor, SymmetricCscMatrix as SpralSymmetricCscMatrix,
    analyse as spral_analyse,
    current_factorization_progress as spral_current_factorization_progress,
    factorize as spral_factorize,
};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

use super::{
    BackendTimingMetadata, BoundConstraints, CCS, CompiledNlpProblem, DUAL_INF_LABEL, EQ_INF_LABEL,
    EvalTimingStat, FilterAcceptanceMode, FilterInfo, INEQ_INF_LABEL, Index, OVERALL_INF_LABEL,
    PRIMAL_INF_LABEL, ParameterMatrix, SolverAdapterTiming, SqpEventLegendState,
    augment_inequality_values, boxed_line, choose_summary_duration_unit, collect_bound_constraints,
    compact_duration_text, complementarity_inf_norm, declared_box_constraint_count,
    dense_fill_percent, fmt_duration_in_unit, fmt_optional_duration_in_unit, inf_norm,
    log_boxed_section, lower_tri_fill_percent, positive_part_inf_norm, scaled_overall_inf_norm,
    sci_text, split_augmented_inequality_multipliers, style_bold, style_cyan_bold,
    style_green_bold, style_iteration_label_cell, style_metric_against_tolerance, style_red_bold,
    style_yellow_bold, time_callback, validate_nlp_problem_shapes, validate_parameter_inputs,
};

const IP_COMP_INF_LABEL: &str = "‖s∘z‖∞";

const LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM: f64 = 1e12;
const LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL: f64 = 1e-7;
const LINEAR_DEBUG_DELTA_TOLERANCE: f64 = 1e-5;
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointLinearSolver {
    Auto,
    SpralSsids,
    NativeSpralSsids,
    SparseQdldl,
}

impl InteriorPointLinearSolver {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::SpralSsids => "spral_ssids",
            Self::NativeSpralSsids => "native_spral_ssids",
            Self::SparseQdldl => "sparse_qdldl",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointSpralPivotMethod {
    AggressiveAposteriori,
    BlockAposteriori,
    ThresholdPartial,
}

impl InteriorPointSpralPivotMethod {
    pub const fn label(self) -> &'static str {
        match self {
            Self::AggressiveAposteriori => "aggressive_app",
            Self::BlockAposteriori => "block_app",
            Self::ThresholdPartial => "threshold_partial",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointLinearDebugSchedule {
    FirstIteration,
    FailuresOnly,
    EveryIteration,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteriorPointLinearDebugOptions {
    pub compare_solvers: Vec<InteriorPointLinearSolver>,
    pub schedule: InteriorPointLinearDebugSchedule,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub dump_dir: Option<PathBuf>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteriorPointLinearInertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointLinearDebugVerdict {
    Consistent,
    LinearSolverMismatch,
    ComparisonIncomplete,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointLinearDebugBackendResult {
    pub solver: InteriorPointLinearSolver,
    pub success: bool,
    pub regularization: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub inertia: Option<InteriorPointLinearInertia>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub residual_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub solution_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_inf: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::option_duration_seconds_serde")
    )]
    pub factorization_time: Option<Duration>,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::option_duration_seconds_serde")
    )]
    pub solve_time: Option<Duration>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub reused_symbolic: Option<bool>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_delta_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub dx_delta_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub d_lambda_delta_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub ds_delta_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub dz_delta_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub detail: Option<String>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointLinearDebugReport {
    pub primary_solver: InteriorPointLinearSolver,
    pub schedule: InteriorPointLinearDebugSchedule,
    pub verdict: InteriorPointLinearDebugVerdict,
    pub results: Vec<InteriorPointLinearDebugBackendResult>,
    pub notes: Vec<String>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointAlphaForYStrategy {
    Primal,
    BoundMultiplier,
    Min,
    Max,
    Full,
    PrimalAndFull,
    DualAndFull,
}

impl InteriorPointAlphaForYStrategy {
    fn label(self) -> &'static str {
        match self {
            Self::Primal => "primal",
            Self::BoundMultiplier => "bound-mult",
            Self::Min => "min",
            Self::Max => "max",
            Self::Full => "full",
            Self::PrimalAndFull => "primal-and-full",
            Self::DualAndFull => "dual-and-full",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointBoundMultiplierInitMethod {
    Constant,
    MuBased,
}

impl InteriorPointBoundMultiplierInitMethod {
    fn label(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::MuBased => "mu-based",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InteriorPointOptions {
    pub max_iters: Index,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub complementarity_tol: f64,
    pub overall_tol: f64,
    pub overall_scale_max: f64,
    pub fraction_to_boundary: f64,
    pub alpha_for_y: InteriorPointAlphaForYStrategy,
    pub alpha_for_y_tol: f64,
    pub bound_mult_init_method: InteriorPointBoundMultiplierInitMethod,
    pub bound_mult_init_val: f64,
    pub least_square_init_duals: bool,
    pub constr_mult_init_max: f64,
    pub bound_push: f64,
    pub bound_frac: f64,
    pub slack_bound_push: f64,
    pub slack_bound_frac: f64,
    pub line_search_beta: f64,
    pub line_search_c1: f64,
    pub min_step: f64,
    pub filter_gamma_objective: f64,
    pub filter_gamma_violation: f64,
    pub max_filter_resets: Index,
    pub filter_reset_trigger: Index,
    pub theta_max_fact: f64,
    pub theta_min_fact: f64,
    pub eta_phi: f64,
    pub delta: f64,
    pub s_phi: f64,
    pub s_theta: f64,
    pub alpha_min_frac: f64,
    pub obj_max_inc: f64,
    pub acceptable_tol: f64,
    pub acceptable_iter: Index,
    pub acceptable_dual_inf_tol: f64,
    pub acceptable_constr_viol_tol: f64,
    pub acceptable_compl_inf_tol: f64,
    pub acceptable_obj_change_tol: f64,
    pub mu_init: f64,
    pub barrier_tol_factor: f64,
    pub mu_linear_decrease_factor: f64,
    pub mu_superlinear_decrease_power: f64,
    pub mu_allow_fast_monotone_decrease: bool,
    pub mu_target: f64,
    pub kappa_sigma: f64,
    pub regularization: f64,
    pub adaptive_regularization_retries: Index,
    pub regularization_growth_factor: f64,
    pub regularization_max: f64,
    pub second_order_correction: bool,
    pub tiny_step_tol: f64,
    pub watchdog_shortened_iter_trigger: Index,
    pub watchdog_trial_iter_max: Index,
    pub mu_min: f64,
    pub linear_solver: InteriorPointLinearSolver,
    pub spral_pivot_method: InteriorPointSpralPivotMethod,
    pub spral_action_on_zero_pivot: bool,
    pub spral_small_pivot_tolerance: f64,
    pub spral_threshold_pivot_u: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linear_debug: Option<InteriorPointLinearDebugOptions>,
    pub verbose: bool,
}

impl Default for InteriorPointOptions {
    fn default() -> Self {
        Self {
            max_iters: 80,
            dual_tol: 2e-6,
            constraint_tol: 1e-6,
            complementarity_tol: 1e-6,
            overall_tol: 1e-6,
            overall_scale_max: 100.0,
            fraction_to_boundary: 0.995,
            alpha_for_y: InteriorPointAlphaForYStrategy::Primal,
            alpha_for_y_tol: 10.0,
            bound_mult_init_method: InteriorPointBoundMultiplierInitMethod::Constant,
            bound_mult_init_val: 1.0,
            least_square_init_duals: false,
            constr_mult_init_max: 1e3,
            bound_push: 1e-2,
            bound_frac: 1e-2,
            slack_bound_push: 1e-2,
            slack_bound_frac: 1e-2,
            line_search_beta: 0.5,
            line_search_c1: 1e-4,
            min_step: 1e-8,
            filter_gamma_objective: 1e-8,
            filter_gamma_violation: 1e-5,
            max_filter_resets: 5,
            filter_reset_trigger: 5,
            theta_max_fact: 1e4,
            theta_min_fact: 1e-4,
            eta_phi: 1e-8,
            delta: 1.0,
            s_phi: 2.3,
            s_theta: 1.1,
            alpha_min_frac: 0.05,
            obj_max_inc: 5.0,
            acceptable_tol: 1e-6,
            acceptable_iter: 0,
            acceptable_dual_inf_tol: 2e-6,
            acceptable_constr_viol_tol: 1e-6,
            acceptable_compl_inf_tol: 1e-6,
            acceptable_obj_change_tol: 1e20,
            mu_init: 1e-1,
            barrier_tol_factor: 10.0,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            mu_allow_fast_monotone_decrease: true,
            mu_target: 0.0,
            kappa_sigma: 1e10,
            regularization: 1e-6,
            adaptive_regularization_retries: 3,
            regularization_growth_factor: 10.0,
            regularization_max: 1e2,
            second_order_correction: true,
            tiny_step_tol: 1e-6,
            watchdog_shortened_iter_trigger: 3,
            watchdog_trial_iter_max: 3,
            mu_min: 1e-12,
            linear_solver: InteriorPointLinearSolver::SpralSsids,
            spral_pivot_method: InteriorPointSpralPivotMethod::BlockAposteriori,
            spral_action_on_zero_pivot: true,
            spral_small_pivot_tolerance: 1e-20,
            spral_threshold_pivot_u: 1e-8,
            linear_debug: None,
            verbose: true,
        }
    }
}

pub fn format_nlip_settings_summary(options: &InteriorPointOptions) -> String {
    format!(
        "filter={}; linear_solver={}; linear_debug={}; spral=[pivot={}, action={}, small={}, u={}]; beta={}; c1={}; min_step={}; tau={}; alpha_y=[strategy={}, tol={}] ; init=[bound_push={}, bound_frac={}, slack_push={}, slack_frac={}] ; dual_init=[method={}, val={}, least_square={}, max={}] ; regularization={} (retries={}, growth={}, max={}); soc={}; watchdog=[trigger={}, max={}]; filter_reset=[max={}, trigger={}]; tiny_step={}; mu=[init={}, target={}, min={}, barrier_tol={}, linear={}, superlinear={}, fast={}]; theta=[{}, {}]; acceptable_iter={}",
        "on",
        options.linear_solver.label(),
        format_nlip_linear_debug_summary(options.linear_debug.as_ref()),
        options.spral_pivot_method.label(),
        if options.spral_action_on_zero_pivot {
            "continue"
        } else {
            "abort"
        },
        sci_text(options.spral_small_pivot_tolerance),
        sci_text(options.spral_threshold_pivot_u),
        sci_text(options.line_search_beta),
        sci_text(options.line_search_c1),
        sci_text(options.min_step),
        sci_text(options.fraction_to_boundary),
        options.alpha_for_y.label(),
        sci_text(options.alpha_for_y_tol),
        sci_text(options.bound_push),
        sci_text(options.bound_frac),
        sci_text(options.slack_bound_push),
        sci_text(options.slack_bound_frac),
        options.bound_mult_init_method.label(),
        sci_text(options.bound_mult_init_val),
        if options.least_square_init_duals {
            "yes"
        } else {
            "no"
        },
        sci_text(options.constr_mult_init_max),
        sci_text(options.regularization),
        options.adaptive_regularization_retries,
        sci_text(options.regularization_growth_factor),
        sci_text(options.regularization_max),
        if options.second_order_correction {
            "on"
        } else {
            "off"
        },
        options.watchdog_shortened_iter_trigger,
        options.watchdog_trial_iter_max,
        options.max_filter_resets,
        options.filter_reset_trigger,
        sci_text(options.tiny_step_tol),
        sci_text(options.mu_init),
        sci_text(options.mu_target),
        sci_text(options.mu_min),
        sci_text(options.barrier_tol_factor),
        sci_text(options.mu_linear_decrease_factor),
        sci_text(options.mu_superlinear_decrease_power),
        if options.mu_allow_fast_monotone_decrease {
            "on"
        } else {
            "off"
        },
        sci_text(options.theta_min_fact),
        sci_text(options.theta_max_fact),
        options.acceptable_iter,
    )
}

fn format_nlip_linear_debug_summary(options: Option<&InteriorPointLinearDebugOptions>) -> String {
    let Some(options) = options else {
        return "off".into();
    };
    let compare = options
        .compare_solvers
        .iter()
        .map(|solver| solver.label())
        .collect::<Vec<_>>()
        .join(",");
    let schedule = match options.schedule {
        InteriorPointLinearDebugSchedule::FirstIteration => "first_iteration",
        InteriorPointLinearDebugSchedule::FailuresOnly => "failures_only",
        InteriorPointLinearDebugSchedule::EveryIteration => "every_iteration",
    };
    let dump = options
        .dump_dir
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "-".into());
    format!("on[schedule={schedule}, compare={compare}, dump={dump}]")
}

fn spral_numeric_factor_options(options: &InteriorPointOptions) -> SpralNumericFactorOptions {
    SpralNumericFactorOptions {
        action_on_zero_pivot: options.spral_action_on_zero_pivot,
        pivot_method: match options.spral_pivot_method {
            InteriorPointSpralPivotMethod::AggressiveAposteriori => {
                SpralPivotMethod::AggressiveAposteriori
            }
            InteriorPointSpralPivotMethod::BlockAposteriori => SpralPivotMethod::BlockAposteriori,
            InteriorPointSpralPivotMethod::ThresholdPartial => SpralPivotMethod::ThresholdPartial,
        },
        small_pivot_tolerance: options.spral_small_pivot_tolerance,
        threshold_pivot_u: options.spral_threshold_pivot_u,
        ..SpralNumericFactorOptions::default()
    }
}

fn system_spral_numeric_factor_options(system: &ReducedKktSystem<'_>) -> SpralNumericFactorOptions {
    SpralNumericFactorOptions {
        action_on_zero_pivot: system.spral_action_on_zero_pivot,
        pivot_method: match system.spral_pivot_method {
            InteriorPointSpralPivotMethod::AggressiveAposteriori => {
                SpralPivotMethod::AggressiveAposteriori
            }
            InteriorPointSpralPivotMethod::BlockAposteriori => SpralPivotMethod::BlockAposteriori,
            InteriorPointSpralPivotMethod::ThresholdPartial => SpralPivotMethod::ThresholdPartial,
        },
        small_pivot_tolerance: system.spral_small_pivot_tolerance,
        threshold_pivot_u: system.spral_threshold_pivot_u,
        ..SpralNumericFactorOptions::default()
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteriorPointProfiling {
    pub objective_value: EvalTimingStat,
    pub objective_gradient: EvalTimingStat,
    pub equality_values: EvalTimingStat,
    pub inequality_values: EvalTimingStat,
    pub equality_jacobian_values: EvalTimingStat,
    pub inequality_jacobian_values: EvalTimingStat,
    pub lagrangian_hessian_values: EvalTimingStat,
    pub kkt_assemblies: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub kkt_assembly_time: Duration,
    pub linear_solves: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub linear_solve_time: Duration,
    pub sparse_symbolic_analyses: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub sparse_symbolic_analysis_time: Duration,
    pub sparse_numeric_factorizations: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub sparse_numeric_factorization_time: Duration,
    pub sparse_numeric_refactorizations: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub sparse_numeric_refactorization_time: Duration,
    pub adapter_timing: Option<SolverAdapterTiming>,
    pub preprocessing_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub preprocessing_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub total_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub unaccounted_time: Duration,
    pub backend_timing: BackendTimingMetadata,
}

impl InteriorPointProfiling {
    fn total_callback_time(&self) -> Duration {
        self.objective_value.total_time
            + self.objective_gradient.total_time
            + self.equality_values.total_time
            + self.inequality_values.total_time
            + self.equality_jacobian_values.total_time
            + self.inequality_jacobian_values.total_time
            + self.lagrangian_hessian_values.total_time
    }

    fn total_callback_calls(&self) -> Index {
        self.objective_value.calls
            + self.objective_gradient.calls
            + self.equality_values.calls
            + self.inequality_values.calls
            + self.equality_jacobian_values.calls
            + self.inequality_jacobian_values.calls
            + self.lagrangian_hessian_values.calls
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InteriorPointSummary {
    pub x: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub slack: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub equality_inf_norm: f64,
    pub inequality_inf_norm: f64,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: f64,
    pub overall_inf_norm: f64,
    pub barrier_parameter: f64,
    pub termination: InteriorPointTermination,
    pub status_kind: InteriorPointStatusKind,
    pub snapshots: Vec<InteriorPointIterationSnapshot>,
    pub final_state: InteriorPointIterationSnapshot,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub last_accepted_state: Option<InteriorPointIterationSnapshot>,
    pub profiling: InteriorPointProfiling,
    pub linear_solver: InteriorPointLinearSolver,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointTermination {
    Converged,
    Acceptable,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointStatusKind {
    Success,
    Warning,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointIterationPhase {
    Initial,
    AcceptedStep,
    Converged,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointStepKind {
    Objective,
    Feasibility,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointIterationEvent {
    LongLineSearch,
    FilterAccepted,
    SecondOrderCorrectionAttempted,
    SecondOrderCorrectionAccepted,
    WatchdogArmed,
    WatchdogActivated,
    FilterReset,
    BoundMultiplierSafeguardApplied,
    BarrierParameterUpdated,
    AdaptiveRegularizationUsed,
    TinyStep,
    MaxIterationsReached,
}

pub fn nlip_event_legend_entries_for_events(
    events: &[InteriorPointIterationEvent],
) -> Vec<(char, &'static str)> {
    let mut entries = Vec::new();
    for event in events {
        match event {
            InteriorPointIterationEvent::LongLineSearch => {
                entries.push(('L', "L=line search backtracked >=4 times"))
            }
            InteriorPointIterationEvent::FilterAccepted => entries.push((
                'F',
                "F=filter accepted a feasibility-improving step without objective Armijo",
            )),
            InteriorPointIterationEvent::SecondOrderCorrectionAttempted => {
                entries.push(('s', "s=second-order correction was attempted"))
            }
            InteriorPointIterationEvent::SecondOrderCorrectionAccepted => {
                entries.push(('S', "S=accepted step used second-order correction"))
            }
            InteriorPointIterationEvent::WatchdogArmed => entries.push((
                'A',
                "A=watchdog reference armed after repeated shortened/tiny steps",
            )),
            InteriorPointIterationEvent::WatchdogActivated => {
                entries.push(('W', "W=watchdog accepted a residual-improving step"))
            }
            InteriorPointIterationEvent::FilterReset => entries.push((
                'X',
                "X=filter frontier was reset after repeated filter rejections",
            )),
            InteriorPointIterationEvent::BoundMultiplierSafeguardApplied => entries.push((
                'B',
                "B=accepted step corrected bound multipliers via IPOPT kappa_sigma safeguard",
            )),
            InteriorPointIterationEvent::BarrierParameterUpdated => entries.push((
                'U',
                "U=barrier parameter updated and IPOPT line-search state was reset",
            )),
            InteriorPointIterationEvent::AdaptiveRegularizationUsed => entries.push((
                'V',
                "V=adaptive KKT regularization increased before acceptance",
            )),
            InteriorPointIterationEvent::TinyStep => {
                entries.push(('T', "T=accepted step was tiny"))
            }
            InteriorPointIterationEvent::MaxIterationsReached => {
                entries.push(('M', "M=maximum NLIP iterations reached"))
            }
        }
    }
    entries
}

pub fn nlip_event_legend_entries(
    snapshot: &InteriorPointIterationSnapshot,
) -> Vec<(char, &'static str)> {
    nlip_event_legend_entries_for_events(&snapshot.events)
}

pub fn nlip_event_codes_for_events(events: &[InteriorPointIterationEvent]) -> String {
    let present_codes = nlip_event_legend_entries_for_events(events)
        .into_iter()
        .map(|(code, _)| code)
        .collect::<Vec<_>>();
    NLIP_EVENT_SLOT_ORDER
        .iter()
        .filter(|code| present_codes.contains(code))
        .copied()
        .collect()
}

pub fn nlip_event_codes(snapshot: &InteriorPointIterationSnapshot) -> String {
    nlip_event_codes_for_events(&snapshot.events)
}

const NLIP_EVENT_SLOT_ORDER: [char; 12] =
    ['L', 'F', 's', 'S', 'A', 'W', 'X', 'B', 'U', 'V', 'T', 'M'];
const NLIP_EVENT_CELL_WIDTH: usize = NLIP_EVENT_SLOT_ORDER.len();

fn nlip_event_slot_codes(snapshot: &InteriorPointIterationSnapshot) -> String {
    let present_codes = nlip_event_legend_entries(snapshot)
        .into_iter()
        .map(|(code, _)| code)
        .collect::<Vec<_>>();
    NLIP_EVENT_SLOT_ORDER
        .iter()
        .map(|code| {
            if present_codes.contains(code) {
                *code
            } else {
                ' '
            }
        })
        .collect()
}

fn push_unique_nlip_event(
    events: &mut Vec<InteriorPointIterationEvent>,
    event: InteriorPointIterationEvent,
) {
    if !events.contains(&event) {
        events.push(event);
    }
}

fn snapshot_with_nlip_events(
    mut snapshot: InteriorPointIterationSnapshot,
    events: &[InteriorPointIterationEvent],
) -> InteriorPointIterationSnapshot {
    snapshot.events = events.to_vec();
    snapshot
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteriorPointIterationTiming {
    pub adapter_timing: Option<SolverAdapterTiming>,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub callback: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub kkt_assembly: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub linear_solve: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub preprocess: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub total: Duration,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointIterationSnapshot {
    pub iteration: Index,
    pub phase: InteriorPointIterationPhase,
    pub x: Vec<f64>,
    pub objective: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub barrier_objective: Option<f64>,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub dual_inf: f64,
    pub comp_inf: Option<f64>,
    pub overall_inf: f64,
    pub barrier_parameter: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub filter_theta: Option<f64>,
    pub step_inf: Option<f64>,
    pub alpha: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_pr: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_du: Option<f64>,
    pub line_search_iterations: Option<Index>,
    pub line_search_trials: Index,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub regularization_size: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_kind: Option<InteriorPointStepKind>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_tag: Option<char>,
    pub watchdog_active: bool,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub line_search: Option<InteriorPointLineSearchInfo>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub direction_diagnostics: Option<InteriorPointDirectionDiagnostics>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linear_debug: Option<InteriorPointLinearDebugReport>,
    pub linear_solver: InteriorPointLinearSolver,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::option_duration_seconds_serde")
    )]
    pub linear_solve_time: Option<Duration>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub filter: Option<FilterInfo>,
    pub timing: InteriorPointIterationTiming,
    pub events: Vec<InteriorPointIterationEvent>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointBoundaryLimiter {
    pub index: Index,
    pub value: f64,
    pub direction: f64,
    pub alpha: f64,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointDirectionDiagnostics {
    pub dx_inf: f64,
    pub d_lambda_inf: f64,
    pub ds_inf: f64,
    pub dz_inf: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_pr_limiter: Option<InteriorPointBoundaryLimiter>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_du_limiter: Option<InteriorPointBoundaryLimiter>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointLinearSolveFailureKind {
    FactorizationFailed,
    InertiaMismatch,
    NonFiniteSolution,
    SolutionNormTooLarge,
    ResidualTooLarge,
}

impl InteriorPointLinearSolveFailureKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::FactorizationFailed => "factorization_failed",
            Self::InertiaMismatch => "inertia_mismatch",
            Self::NonFiniteSolution => "non_finite_solution",
            Self::SolutionNormTooLarge => "solution_norm_too_large",
            Self::ResidualTooLarge => "residual_too_large",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointLinearSolveAttempt {
    pub solver: InteriorPointLinearSolver,
    pub regularization: f64,
    pub failure_kind: InteriorPointLinearSolveFailureKind,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub detail: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub solution_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub solution_inf_limit: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub residual_inf: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub residual_inf_limit: Option<f64>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointLinearSolveDiagnostics {
    pub preferred_solver: InteriorPointLinearSolver,
    pub matrix_dimension: Index,
    pub attempts: Vec<InteriorPointLinearSolveAttempt>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub debug_report: Option<InteriorPointLinearDebugReport>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointLineSearchTrial {
    pub alpha: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_du: Option<f64>,
    pub slack_positive: bool,
    pub multipliers_positive: bool,
    pub objective: Option<f64>,
    pub barrier_objective: Option<f64>,
    pub merit: Option<f64>,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub primal_inf: Option<f64>,
    pub dual_inf: Option<f64>,
    pub comp_inf: Option<f64>,
    pub mu: Option<f64>,
    pub local_filter_acceptable: Option<bool>,
    pub filter_acceptable: Option<bool>,
    pub filter_dominated: Option<bool>,
    pub filter_sufficient_objective_reduction: Option<bool>,
    pub filter_sufficient_violation_reduction: Option<bool>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub switching_condition_satisfied: Option<bool>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointLineSearchInfo {
    pub initial_alpha_pr: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub initial_alpha_du: Option<f64>,
    pub accepted_alpha: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub accepted_alpha_du: Option<f64>,
    pub last_tried_alpha: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub last_tried_alpha_du: Option<f64>,
    pub backtrack_count: Index,
    pub sigma: f64,
    pub current_merit: f64,
    pub current_barrier_objective: f64,
    pub current_primal_inf: f64,
    pub alpha_min: f64,
    pub second_order_correction_attempted: bool,
    pub second_order_correction_used: bool,
    pub watchdog_active: bool,
    pub watchdog_accepted: bool,
    pub tiny_step: bool,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub filter_acceptance_mode: Option<FilterAcceptanceMode>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_kind: Option<InteriorPointStepKind>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_tag: Option<char>,
    pub rejected_trials: Vec<InteriorPointLineSearchTrial>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InteriorPointFailureContext {
    pub final_state: Option<InteriorPointIterationSnapshot>,
    pub last_accepted_state: Option<InteriorPointIterationSnapshot>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub failed_linear_solve: Option<InteriorPointLinearSolveDiagnostics>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub failed_line_search: Option<InteriorPointLineSearchInfo>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub failed_direction_diagnostics: Option<InteriorPointDirectionDiagnostics>,
    pub profiling: InteriorPointProfiling,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Error)]
pub enum InteriorPointSolveError {
    #[error("invalid NLIP input: {0}")]
    InvalidInput(String),
    #[error("NLIP linear solve failed using {solver:?}")]
    LinearSolve {
        solver: InteriorPointLinearSolver,
        context: Box<InteriorPointFailureContext>,
    },
    #[error(
        "NLIP line search failed (residual merit {merit}, mu {mu}, step inf-norm {step_inf_norm})"
    )]
    LineSearchFailed {
        merit: f64,
        mu: f64,
        step_inf_norm: f64,
        context: Box<InteriorPointFailureContext>,
    },
    #[error("NLIP failed to converge in {iterations} iterations")]
    MaxIterations {
        iterations: Index,
        context: Box<InteriorPointFailureContext>,
    },
}

struct NewtonDirection {
    dx: Vec<f64>,
    d_lambda: Vec<f64>,
    ds: Vec<f64>,
    dz: Vec<f64>,
    solver_used: InteriorPointLinearSolver,
    regularization_used: f64,
    dual_regularization_used: f64,
    primal_diagonal_shift_used: f64,
    linear_solution: Vec<f64>,
    backend_stats: LinearBackendRunStats,
    linear_debug: Option<InteriorPointLinearDebugReport>,
}

struct AcceptedInteriorPointTrial {
    x: Vec<f64>,
    lambda: Vec<f64>,
    slack: Vec<f64>,
    z: Vec<f64>,
    objective: f64,
    barrier_objective: f64,
    equality_inf: f64,
    inequality_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    overall_inf: f64,
    mu: f64,
    filter_entry: super::FilterEntry,
    filter_augment_entry: Option<super::FilterEntry>,
    filter_theta: f64,
    filter_acceptance_mode: Option<FilterAcceptanceMode>,
    step_kind: InteriorPointStepKind,
    step_tag: char,
    phase: InteriorPointIterationPhase,
    accepted_alpha_pr: f64,
    accepted_alpha_du: Option<f64>,
    line_search_initial_alpha_pr: f64,
    line_search_initial_alpha_du: Option<f64>,
    line_search_last_alpha_pr: f64,
    line_search_last_alpha_du: Option<f64>,
    line_search_backtrack_count: Index,
    second_order_correction_used: bool,
    watchdog_accepted: bool,
    bound_multiplier_corrected: bool,
}

struct AcceptedTrialMultiplierState {
    z: Vec<f64>,
    dual_inf: f64,
    complementarity_inf: f64,
    overall_inf: f64,
}

#[derive(Clone, Copy, Debug)]
struct InteriorPointAcceptableState {
    overall_error: f64,
    dual_inf: f64,
    constr_viol: f64,
    compl_inf: f64,
    objective: f64,
}

#[derive(Clone, Debug)]
struct WatchdogReferencePoint {
    barrier_objective: f64,
    filter_theta: f64,
    barrier_directional_derivative: f64,
}

#[derive(Clone, Debug, Default)]
struct InteriorPointWatchdogState {
    reference: Option<WatchdogReferencePoint>,
    remaining_iters: Index,
    shortened_step_streak: Index,
    tiny_step_streak: Index,
}

#[derive(Clone, Debug)]
struct SparseMatrixStructure {
    ccs: CCS,
    row_entries: Vec<Vec<(usize, usize)>>,
}

#[derive(Clone, Debug)]
struct SparseMatrix {
    structure: Arc<SparseMatrixStructure>,
    values: Vec<f64>,
}

impl SparseMatrix {
    fn nrows(&self) -> usize {
        self.structure.ccs.nrow
    }

    fn ncols(&self) -> usize {
        self.structure.ccs.ncol
    }
}

#[derive(Clone, Debug)]
struct SparseSymmetricMatrix {
    lower_triangle: Arc<CCS>,
    values: Vec<f64>,
}

struct ReducedKktSystem<'a> {
    hessian: &'a SparseSymmetricMatrix,
    equality_jacobian: &'a SparseMatrix,
    inequality_jacobian: &'a SparseMatrix,
    slack: &'a [f64],
    multipliers: &'a [f64],
    r_dual: &'a [f64],
    r_eq: &'a [f64],
    r_ineq: &'a [f64],
    r_cent: &'a [f64],
    solver: InteriorPointLinearSolver,
    regularization: f64,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_max: f64,
    spral_pivot_method: InteriorPointSpralPivotMethod,
    spral_action_on_zero_pivot: bool,
    spral_small_pivot_tolerance: f64,
    spral_threshold_pivot_u: f64,
}

#[derive(Clone, Debug)]
struct SpralAugmentedKktPattern {
    ccs: Arc<CCS>,
    x_dimension: usize,
    inequality_dimension: usize,
    equality_dimension: usize,
    p_offset: usize,
    lambda_offset: usize,
    z_offset: usize,
    hessian_value_indices: Vec<usize>,
    equality_jacobian_value_indices: Vec<usize>,
    inequality_jacobian_value_indices: Vec<usize>,
    x_diagonal_indices: Vec<usize>,
    p_diagonal_indices: Vec<usize>,
    lambda_diagonal_indices: Vec<usize>,
    z_diagonal_indices: Vec<usize>,
    pz_indices: Vec<usize>,
}

impl SpralAugmentedKktPattern {
    fn dimension(&self) -> usize {
        self.ccs.nrow
    }
}

struct SpralAugmentedKktWorkspace {
    pattern: SpralAugmentedKktPattern,
    values: Vec<f64>,
    symbolic: SpralSymbolicFactor,
    factor: Option<SpralNumericFactor>,
    factor_regularization: Option<f64>,
    auto_fallback_disabled: bool,
}

struct NativeSpralAugmentedKktWorkspace {
    pattern: SpralAugmentedKktPattern,
    values: Vec<f64>,
    session: NativeSpralSession,
    factor_regularization: Option<f64>,
}

#[derive(Clone, Debug)]
struct InteriorPointKktSnapshot {
    iteration: Index,
    phase: InteriorPointIterationPhase,
    primary_solver: InteriorPointLinearSolver,
    matrix_dimension: usize,
    x_dimension: usize,
    equality_dimension: usize,
    inequality_dimension: usize,
    hessian: SparseSymmetricMatrix,
    equality_jacobian: SparseMatrix,
    inequality_jacobian: SparseMatrix,
    slack: Vec<f64>,
    multipliers: Vec<f64>,
    r_dual: Vec<f64>,
    r_eq: Vec<f64>,
    r_ineq: Vec<f64>,
    r_cent: Vec<f64>,
    regularization: f64,
    primal_diagonal_shift: f64,
    dual_regularization: f64,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_max: f64,
    spral_pivot_method: InteriorPointSpralPivotMethod,
    spral_action_on_zero_pivot: bool,
    spral_small_pivot_tolerance: f64,
    spral_threshold_pivot_u: f64,
    augmented_pattern: SpralAugmentedKktPattern,
    augmented_values: Vec<f64>,
    augmented_rhs: Vec<f64>,
    expected_augmented_inertia: SpralInertia,
    barrier_parameter: f64,
    primal_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    line_search_trials: Index,
}

#[derive(Clone, Debug)]
struct LinearBackendRunStats {
    solver: InteriorPointLinearSolver,
    factorization_time: Duration,
    solve_time: Duration,
    reused_symbolic: Option<bool>,
    inertia: Option<InteriorPointLinearInertia>,
    residual_inf: f64,
    solution_inf: f64,
    detail: Option<String>,
}

struct InteriorPointLinearDebugState {
    options: InteriorPointLinearDebugOptions,
    rust_workspace: Option<SpralAugmentedKktWorkspace>,
    rust_workspace_unavailable: bool,
    rust_workspace_error: Option<String>,
    native_workspace: Option<NativeSpralAugmentedKktWorkspace>,
    native_workspace_unavailable: bool,
    native_workspace_error: Option<String>,
}

impl SpralAugmentedKktWorkspace {
    fn analyse(
        hessian_structure: &CCS,
        equality_jacobian: &SparseMatrixStructure,
        inequality_jacobian: &SparseMatrixStructure,
    ) -> std::result::Result<(Self, SpralAnalyseInfo), InteriorPointSolveError> {
        let pattern = build_spral_augmented_kkt_pattern(
            hessian_structure,
            equality_jacobian,
            inequality_jacobian,
        )?;
        let matrix = SpralSymmetricCscMatrix::new(
            pattern.dimension(),
            &pattern.ccs.col_ptrs,
            &pattern.ccs.row_indices,
            None,
        )
        .map_err(|error| {
            InteriorPointSolveError::InvalidInput(format!(
                "invalid sparse KKT structure for spral_ssids: {error}"
            ))
        })?;
        let (symbolic, info) = spral_analyse(
            matrix,
            &SpralSsidsOptions {
                // Large NLP KKT systems can spend a long time estimating natural-order fill
                // inside `Auto`; AMD is a safer sparse default for this integration seam.
                ordering: SpralOrderingStrategy::ApproximateMinimumDegree,
            },
        )
        .map_err(|error| {
            InteriorPointSolveError::InvalidInput(format!(
                "failed to analyse sparse KKT structure for spral_ssids: {error}"
            ))
        })?;
        let values = vec![0.0; pattern.ccs.nnz()];
        Ok((
            Self {
                pattern,
                values,
                symbolic,
                factor: None,
                factor_regularization: None,
                auto_fallback_disabled: false,
            },
            info,
        ))
    }
}

impl NativeSpralAugmentedKktWorkspace {
    fn analyse(
        hessian_structure: &CCS,
        equality_jacobian: &SparseMatrixStructure,
        inequality_jacobian: &SparseMatrixStructure,
        numeric_options: &SpralNumericFactorOptions,
    ) -> std::result::Result<(Self, SpralAnalyseInfo), InteriorPointSolveError> {
        let pattern = build_spral_augmented_kkt_pattern(
            hessian_structure,
            equality_jacobian,
            inequality_jacobian,
        )?;
        let matrix = SpralSymmetricCscMatrix::new(
            pattern.dimension(),
            &pattern.ccs.col_ptrs,
            &pattern.ccs.row_indices,
            None,
        )
        .map_err(|error| {
            InteriorPointSolveError::InvalidInput(format!(
                "invalid sparse KKT structure for native spral_ssids: {error}"
            ))
        })?;
        let native = NativeSpralLibrary::load().map_err(|error| {
            InteriorPointSolveError::InvalidInput(format!(
                "failed to load native spral_ssids: {error}"
            ))
        })?;
        let session = native
            .analyse_with_options(matrix, numeric_options)
            .map_err(|error| {
                InteriorPointSolveError::InvalidInput(format!(
                    "failed to analyse sparse KKT structure for native spral_ssids: {error}"
                ))
            })?;
        let analyse_info = session.analyse_info();
        let values = vec![0.0; pattern.ccs.nnz()];
        Ok((
            Self {
                pattern,
                values,
                session,
                factor_regularization: None,
            },
            SpralAnalyseInfo {
                estimated_fill_nnz: 0,
                supernode_count: analyse_info.supernode_count,
                max_supernode_width: analyse_info.max_supernode_width,
                ordering_kind: "native_spral_default",
            },
        ))
    }
}

fn prepare_spral_workspace(
    hessian_structure: &CCS,
    equality_jacobian: &SparseMatrixStructure,
    inequality_jacobian: &SparseMatrixStructure,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<SpralAugmentedKktWorkspace, InteriorPointSolveError> {
    let equality_count = equality_jacobian.ccs.nrow;
    let inequality_count = inequality_jacobian.ccs.nrow;
    let matrix_dimension = hessian_structure.nrow + equality_count + 2 * inequality_count;
    if verbose {
        println!(
            "[NLIP][SPRAL] Analysing sparse KKT structure: dim={} x={} eq={} ineq={} hess_nnz={} eq_jac_nnz={} ineq_jac_nnz={}",
            matrix_dimension,
            hessian_structure.nrow,
            equality_count,
            inequality_count,
            hessian_structure.nnz(),
            equality_jacobian.ccs.nnz(),
            inequality_jacobian.ccs.nnz(),
        );
    }
    let analyse_started = Instant::now();
    profiling.sparse_symbolic_analyses += 1;
    let result = SpralAugmentedKktWorkspace::analyse(
        hessian_structure,
        equality_jacobian,
        inequality_jacobian,
    );
    profiling.sparse_symbolic_analysis_time += analyse_started.elapsed();
    match result {
        Ok((workspace, info)) => {
            if verbose {
                println!(
                    "[NLIP][SPRAL] Analysis completed in {}: matrix_nnz={} est_fill={} supernodes={} max_width={} ordering={}",
                    compact_duration_text(analyse_started.elapsed().as_secs_f64()),
                    workspace.pattern.ccs.nnz(),
                    info.estimated_fill_nnz,
                    info.supernode_count,
                    info.max_supernode_width,
                    info.ordering_kind,
                );
            }
            Ok(workspace)
        }
        Err(error) => Err(error),
    }
}

fn prepare_native_spral_workspace(
    hessian_structure: &CCS,
    equality_jacobian: &SparseMatrixStructure,
    inequality_jacobian: &SparseMatrixStructure,
    numeric_options: &SpralNumericFactorOptions,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<NativeSpralAugmentedKktWorkspace, InteriorPointSolveError> {
    let equality_count = equality_jacobian.ccs.nrow;
    let inequality_count = inequality_jacobian.ccs.nrow;
    let matrix_dimension = hessian_structure.nrow + equality_count + 2 * inequality_count;
    if verbose {
        println!(
            "[NLIP][Native-SPRAL] Analysing sparse KKT structure: dim={} x={} eq={} ineq={} hess_nnz={} eq_jac_nnz={} ineq_jac_nnz={}",
            matrix_dimension,
            hessian_structure.nrow,
            equality_count,
            inequality_count,
            hessian_structure.nnz(),
            equality_jacobian.ccs.nnz(),
            inequality_jacobian.ccs.nnz(),
        );
    }
    let analyse_started = Instant::now();
    profiling.sparse_symbolic_analyses += 1;
    let result = NativeSpralAugmentedKktWorkspace::analyse(
        hessian_structure,
        equality_jacobian,
        inequality_jacobian,
        numeric_options,
    );
    profiling.sparse_symbolic_analysis_time += analyse_started.elapsed();
    match result {
        Ok((workspace, info)) => {
            if verbose {
                println!(
                    "[NLIP][Native-SPRAL] Analysis completed in {}: matrix_nnz={} est_fill={} supernodes={} max_width={} ordering={}",
                    compact_duration_text(analyse_started.elapsed().as_secs_f64()),
                    workspace.pattern.ccs.nnz(),
                    info.estimated_fill_nnz,
                    info.supernode_count,
                    info.max_supernode_width,
                    info.ordering_kind,
                );
            }
            Ok(workspace)
        }
        Err(error) => Err(error),
    }
}

impl InteriorPointKktSnapshot {
    fn reduced_system(&self, solver: InteriorPointLinearSolver) -> ReducedKktSystem<'_> {
        ReducedKktSystem {
            hessian: &self.hessian,
            equality_jacobian: &self.equality_jacobian,
            inequality_jacobian: &self.inequality_jacobian,
            slack: &self.slack,
            multipliers: &self.multipliers,
            r_dual: &self.r_dual,
            r_eq: &self.r_eq,
            r_ineq: &self.r_ineq,
            r_cent: &self.r_cent,
            solver,
            regularization: self.regularization,
            adaptive_regularization_retries: self.adaptive_regularization_retries,
            regularization_growth_factor: self.regularization_growth_factor,
            regularization_max: self.regularization_max,
            spral_pivot_method: self.spral_pivot_method,
            spral_action_on_zero_pivot: self.spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: self.spral_small_pivot_tolerance,
            spral_threshold_pivot_u: self.spral_threshold_pivot_u,
        }
    }
}

fn linear_direction_inf_norm(direction: &NewtonDirection) -> f64 {
    direction
        .dx
        .iter()
        .chain(direction.d_lambda.iter())
        .chain(direction.ds.iter())
        .chain(direction.dz.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn should_run_linear_debug(
    schedule: InteriorPointLinearDebugSchedule,
    iteration: Index,
    primary_failed: bool,
) -> bool {
    match schedule {
        InteriorPointLinearDebugSchedule::FirstIteration => iteration == 0,
        InteriorPointLinearDebugSchedule::FailuresOnly => primary_failed,
        InteriorPointLinearDebugSchedule::EveryIteration => true,
    }
}

fn normalized_compare_solvers(
    options: &InteriorPointLinearDebugOptions,
    primary_solver: InteriorPointLinearSolver,
) -> Vec<InteriorPointLinearSolver> {
    let mut deduped = Vec::new();
    for solver in &options.compare_solvers {
        if *solver == InteriorPointLinearSolver::Auto
            || *solver == primary_solver
            || deduped.contains(solver)
        {
            continue;
        }
        deduped.push(*solver);
    }
    deduped
}

#[expect(
    clippy::too_many_arguments,
    reason = "KKT snapshot capture intentionally records both solver and nonlinear context."
)]
fn build_interior_point_kkt_snapshot(
    iteration: Index,
    phase: InteriorPointIterationPhase,
    primary_solver: InteriorPointLinearSolver,
    system: &ReducedKktSystem<'_>,
    dual_regularization_used: f64,
    primal_diagonal_shift_used: f64,
    barrier_parameter: f64,
    primal_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    line_search_trials: Index,
) -> std::result::Result<InteriorPointKktSnapshot, InteriorPointSolveError> {
    let inequality_scalings = system
        .slack
        .iter()
        .zip(system.multipliers.iter())
        .map(|(slack_i, multiplier_i)| (slack_i.max(1e-16) / multiplier_i.max(1e-16)).sqrt())
        .collect::<Vec<_>>();
    let rhs_p = system
        .r_cent
        .iter()
        .zip(system.slack.iter())
        .zip(system.multipliers.iter())
        .map(|((r_cent_i, slack_i), multiplier_i)| {
            let scaling = (slack_i.max(1e-16) * multiplier_i.max(1e-16)).sqrt();
            -r_cent_i / scaling
        })
        .collect::<Vec<_>>();
    let pattern = build_spral_augmented_kkt_pattern(
        system.hessian.lower_triangle.as_ref(),
        system.equality_jacobian.structure.as_ref(),
        system.inequality_jacobian.structure.as_ref(),
    )?;
    let primal_shift = primal_diagonal_shift_used;
    let dual_shift = dual_regularization_used.max(1e-8);
    let mut augmented_values = vec![0.0; pattern.ccs.nnz()];
    fill_spral_augmented_kkt_values(
        &pattern,
        &mut augmented_values,
        system,
        primal_shift,
        dual_shift,
        &inequality_scalings,
    );
    let mut augmented_rhs = vec![0.0; pattern.dimension()];
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    augmented_rhs[..n]
        .iter_mut()
        .zip(system.r_dual.iter())
        .for_each(|(rhs_i, r_i)| *rhs_i = -*r_i);
    augmented_rhs[pattern.p_offset..pattern.p_offset + mineq].copy_from_slice(&rhs_p);
    for row in 0..meq {
        augmented_rhs[pattern.lambda_offset + row] = -system.r_eq[row];
    }
    for row in 0..mineq {
        augmented_rhs[pattern.z_offset + row] = -system.r_ineq[row];
    }
    Ok(InteriorPointKktSnapshot {
        iteration,
        phase,
        primary_solver,
        matrix_dimension: pattern.dimension(),
        x_dimension: n,
        equality_dimension: meq,
        inequality_dimension: mineq,
        hessian: system.hessian.clone(),
        equality_jacobian: system.equality_jacobian.clone(),
        inequality_jacobian: system.inequality_jacobian.clone(),
        slack: system.slack.to_vec(),
        multipliers: system.multipliers.to_vec(),
        r_dual: system.r_dual.to_vec(),
        r_eq: system.r_eq.to_vec(),
        r_ineq: system.r_ineq.to_vec(),
        r_cent: system.r_cent.to_vec(),
        regularization: dual_regularization_used,
        primal_diagonal_shift: primal_diagonal_shift_used,
        dual_regularization: dual_regularization_used,
        adaptive_regularization_retries: system.adaptive_regularization_retries,
        regularization_growth_factor: system.regularization_growth_factor,
        regularization_max: system.regularization_max,
        spral_pivot_method: system.spral_pivot_method,
        spral_action_on_zero_pivot: system.spral_action_on_zero_pivot,
        spral_small_pivot_tolerance: system.spral_small_pivot_tolerance,
        spral_threshold_pivot_u: system.spral_threshold_pivot_u,
        augmented_pattern: pattern.clone(),
        augmented_values,
        augmented_rhs,
        expected_augmented_inertia: spral_expected_augmented_inertia(&pattern),
        barrier_parameter,
        primal_inf,
        dual_inf,
        complementarity_inf,
        line_search_trials,
    })
}

fn linear_debug_result_from_stats(
    direction: &NewtonDirection,
    stats: &LinearBackendRunStats,
    regularization: f64,
) -> InteriorPointLinearDebugBackendResult {
    InteriorPointLinearDebugBackendResult {
        solver: stats.solver,
        success: true,
        regularization,
        inertia: stats.inertia,
        residual_inf: Some(stats.residual_inf),
        solution_inf: Some(stats.solution_inf),
        step_inf: Some(linear_direction_inf_norm(direction)),
        factorization_time: Some(stats.factorization_time),
        solve_time: Some(stats.solve_time),
        reused_symbolic: stats.reused_symbolic,
        step_delta_inf: None,
        dx_delta_inf: None,
        d_lambda_delta_inf: None,
        ds_delta_inf: None,
        dz_delta_inf: None,
        detail: stats.detail.clone(),
    }
}

fn linear_debug_result_from_attempt(
    solver: InteriorPointLinearSolver,
    attempt: &InteriorPointLinearSolveAttempt,
) -> InteriorPointLinearDebugBackendResult {
    InteriorPointLinearDebugBackendResult {
        solver,
        success: false,
        regularization: attempt.regularization,
        inertia: None,
        residual_inf: attempt.residual_inf,
        solution_inf: attempt.solution_inf,
        step_inf: None,
        factorization_time: None,
        solve_time: None,
        reused_symbolic: None,
        step_delta_inf: None,
        dx_delta_inf: None,
        d_lambda_delta_inf: None,
        ds_delta_inf: None,
        dz_delta_inf: None,
        detail: attempt.detail.clone(),
    }
}

fn annotate_linear_debug_delta(
    primary: &NewtonDirection,
    comparison: &NewtonDirection,
    result: &mut InteriorPointLinearDebugBackendResult,
) {
    result.step_delta_inf = Some(delta_inf_norm(
        &primary.linear_solution,
        &comparison.linear_solution,
    ));
    result.dx_delta_inf = Some(delta_inf_norm(&primary.dx, &comparison.dx));
    result.d_lambda_delta_inf = Some(delta_inf_norm(&primary.d_lambda, &comparison.d_lambda));
    result.ds_delta_inf = Some(delta_inf_norm(&primary.ds, &comparison.ds));
    result.dz_delta_inf = Some(delta_inf_norm(&primary.dz, &comparison.dz));
}

fn linear_debug_matches_primary(result: &InteriorPointLinearDebugBackendResult) -> bool {
    result.success
        && result
            .step_delta_inf
            .is_some_and(|delta| delta <= LINEAR_DEBUG_DELTA_TOLERANCE)
        && result
            .dx_delta_inf
            .is_some_and(|delta| delta <= LINEAR_DEBUG_DELTA_TOLERANCE)
        && result
            .d_lambda_delta_inf
            .is_some_and(|delta| delta <= LINEAR_DEBUG_DELTA_TOLERANCE)
        && result
            .ds_delta_inf
            .is_some_and(|delta| delta <= LINEAR_DEBUG_DELTA_TOLERANCE)
        && result
            .dz_delta_inf
            .is_some_and(|delta| delta <= LINEAR_DEBUG_DELTA_TOLERANCE)
}

fn replay_snapshot_with_solver(
    snapshot: &InteriorPointKktSnapshot,
    solver: InteriorPointLinearSolver,
    debug_state: &mut InteriorPointLinearDebugState,
) -> std::result::Result<NewtonDirection, Vec<InteriorPointLinearSolveAttempt>> {
    let system = snapshot.reduced_system(solver);
    let mut scratch_profiling = InteriorPointProfiling::default();
    match solver {
        InteriorPointLinearSolver::SpralSsids => {
            if debug_state.rust_workspace.is_none() && !debug_state.rust_workspace_unavailable {
                match prepare_spral_workspace(
                    snapshot.hessian.lower_triangle.as_ref(),
                    snapshot.equality_jacobian.structure.as_ref(),
                    snapshot.inequality_jacobian.structure.as_ref(),
                    &mut scratch_profiling,
                    false,
                ) {
                    Ok(workspace) => {
                        debug_state.rust_workspace = Some(workspace);
                    }
                    Err(error) => {
                        debug_state.rust_workspace_unavailable = true;
                        debug_state.rust_workspace_error = Some(error.to_string());
                    }
                }
            }
            let Some(workspace) = debug_state.rust_workspace.as_mut() else {
                return Err(vec![spral_error_attempt(
                    snapshot.regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    debug_state
                        .rust_workspace_error
                        .as_deref()
                        .unwrap_or("rust spral_ssids workspace unavailable"),
                )]);
            };
            solve_reduced_kkt_with_spral_ssids(&system, workspace, &mut scratch_profiling, false)
        }
        InteriorPointLinearSolver::NativeSpralSsids => {
            if debug_state.native_workspace.is_none() && !debug_state.native_workspace_unavailable {
                match prepare_native_spral_workspace(
                    snapshot.hessian.lower_triangle.as_ref(),
                    snapshot.equality_jacobian.structure.as_ref(),
                    snapshot.inequality_jacobian.structure.as_ref(),
                    &system_spral_numeric_factor_options(&system),
                    &mut scratch_profiling,
                    false,
                ) {
                    Ok(workspace) => {
                        debug_state.native_workspace = Some(workspace);
                    }
                    Err(error) => {
                        debug_state.native_workspace_unavailable = true;
                        debug_state.native_workspace_error = Some(error.to_string());
                    }
                }
            }
            let Some(workspace) = debug_state.native_workspace.as_mut() else {
                return Err(vec![native_spral_error_attempt(
                    snapshot.regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    debug_state
                        .native_workspace_error
                        .as_deref()
                        .unwrap_or("native spral_ssids workspace unavailable"),
                )]);
            };
            solve_reduced_kkt_with_native_spral_ssids(
                &system,
                workspace,
                &mut scratch_profiling,
                false,
            )
        }
        InteriorPointLinearSolver::SparseQdldl => solve_reduced_kkt_with_sparse_qdldl(&system),
        InteriorPointLinearSolver::Auto => Err(vec![InteriorPointLinearSolveAttempt {
            solver,
            regularization: snapshot.regularization,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: Some("auto is not a concrete comparison backend".into()),
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        }]),
    }
}

fn run_linear_debug_report_on_success(
    snapshot: &InteriorPointKktSnapshot,
    primary_direction: &NewtonDirection,
    debug_state: &mut InteriorPointLinearDebugState,
) -> InteriorPointLinearDebugReport {
    let compare_solvers =
        normalized_compare_solvers(&debug_state.options, primary_direction.solver_used);
    let mut results = vec![linear_debug_result_from_stats(
        primary_direction,
        &primary_direction.backend_stats,
        primary_direction.regularization_used,
    )];
    let mut notes = Vec::new();
    let mut mismatch = false;
    let mut incomplete = false;
    let mut matched_solvers = Vec::new();
    for solver in compare_solvers {
        match replay_snapshot_with_solver(snapshot, solver, debug_state) {
            Ok(direction) => {
                let mut result = linear_debug_result_from_stats(
                    &direction,
                    &direction.backend_stats,
                    direction.regularization_used,
                );
                annotate_linear_debug_delta(primary_direction, &direction, &mut result);
                if linear_debug_matches_primary(&result) {
                    matched_solvers.push(solver.label());
                } else {
                    mismatch = true;
                }
                results.push(result);
            }
            Err(attempts) => {
                incomplete = true;
                let fallback_attempt = InteriorPointLinearSolveAttempt {
                    solver,
                    regularization: snapshot.regularization,
                    failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    detail: Some("comparison backend failed without diagnostics".into()),
                    solution_inf: None,
                    solution_inf_limit: None,
                    residual_inf: None,
                    residual_inf_limit: None,
                };
                let attempt = attempts.last().unwrap_or(&fallback_attempt);
                notes.push(format!(
                    "{} comparison failed: {}",
                    solver.label(),
                    attempt
                        .detail
                        .clone()
                        .unwrap_or_else(|| attempt.failure_kind.label().into())
                ));
                results.push(linear_debug_result_from_attempt(solver, attempt));
            }
        }
    }
    if mismatch && !matched_solvers.is_empty() {
        notes.push(format!(
            "matched primary within tolerance: {}",
            matched_solvers.join(", ")
        ));
    }
    let successful_compares = results.iter().skip(1).any(|result| result.success);
    let verdict = if mismatch {
        InteriorPointLinearDebugVerdict::LinearSolverMismatch
    } else if incomplete {
        InteriorPointLinearDebugVerdict::ComparisonIncomplete
    } else if successful_compares || results.len() == 1 {
        InteriorPointLinearDebugVerdict::Consistent
    } else {
        InteriorPointLinearDebugVerdict::ComparisonIncomplete
    };
    InteriorPointLinearDebugReport {
        primary_solver: primary_direction.solver_used,
        schedule: debug_state.options.schedule,
        verdict,
        results,
        notes,
    }
}

fn run_linear_debug_report_on_failure(
    snapshot: &InteriorPointKktSnapshot,
    preferred_solver: InteriorPointLinearSolver,
    attempts: &[InteriorPointLinearSolveAttempt],
    debug_state: &mut InteriorPointLinearDebugState,
) -> InteriorPointLinearDebugReport {
    let compare_solvers = normalized_compare_solvers(&debug_state.options, preferred_solver);
    let mut results = vec![linear_debug_result_from_attempt(
        preferred_solver,
        attempts.last().unwrap_or(&InteriorPointLinearSolveAttempt {
            solver: preferred_solver,
            regularization: snapshot.regularization,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: Some("primary backend failed without diagnostics".into()),
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        }),
    )];
    let mut notes = attempts
        .iter()
        .filter_map(|attempt| {
            attempt.detail.as_ref().map(|detail| {
                format!(
                    "primary attempt reg={:.3e}: {detail}",
                    attempt.regularization
                )
            })
        })
        .collect::<Vec<_>>();
    let mut compare_success = false;
    for solver in compare_solvers {
        match replay_snapshot_with_solver(snapshot, solver, debug_state) {
            Ok(direction) => {
                compare_success = true;
                results.push(linear_debug_result_from_stats(
                    &direction,
                    &direction.backend_stats,
                    direction.regularization_used,
                ));
            }
            Err(compare_attempts) => {
                let fallback_attempt = InteriorPointLinearSolveAttempt {
                    solver,
                    regularization: snapshot.regularization,
                    failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    detail: Some("comparison backend failed without diagnostics".into()),
                    solution_inf: None,
                    solution_inf_limit: None,
                    residual_inf: None,
                    residual_inf_limit: None,
                };
                let attempt = compare_attempts.last().unwrap_or(&fallback_attempt);
                notes.push(format!(
                    "{} comparison failed: {}",
                    solver.label(),
                    attempt
                        .detail
                        .clone()
                        .unwrap_or_else(|| attempt.failure_kind.label().into())
                ));
                results.push(linear_debug_result_from_attempt(solver, attempt));
            }
        }
    }
    InteriorPointLinearDebugReport {
        primary_solver: preferred_solver,
        schedule: debug_state.options.schedule,
        verdict: if compare_success {
            InteriorPointLinearDebugVerdict::LinearSolverMismatch
        } else {
            InteriorPointLinearDebugVerdict::ComparisonIncomplete
        },
        results,
        notes,
    }
}

fn dump_linear_debug_snapshot(
    options: &InteriorPointLinearDebugOptions,
    snapshot: &InteriorPointKktSnapshot,
    report: &InteriorPointLinearDebugReport,
) {
    let Some(dir) = options.dump_dir.as_ref() else {
        return;
    };
    let _ = fs::create_dir_all(dir);
    let path = dir.join(format!("nlip_kkt_iter_{:04}.txt", snapshot.iteration));
    let mut body = String::new();
    body.push_str(&format!(
        "iteration={}\nphase={:?}\nprimary_solver={}\nmatrix_dimension={}\nx_dimension={}\ninequality_dimension={}\nequality_dimension={}\np_offset={}\nlambda_offset={}\nz_offset={}\nexpected_inertia=+{}/-{}/0={}\nregularization={:.6e}\nprimal_diagonal_shift={:.6e}\ndual_regularization={:.6e}\nbarrier_parameter={:.6e}\nprimal_inf={:.6e}\ndual_inf={:.6e}\ncomplementarity_inf={:.6e}\nline_search_trials={}\nverdict={:?}\n",
        snapshot.iteration,
        snapshot.phase,
        snapshot.primary_solver.label(),
        snapshot.matrix_dimension,
        snapshot.x_dimension,
        snapshot.inequality_dimension,
        snapshot.equality_dimension,
        snapshot.augmented_pattern.p_offset,
        snapshot.augmented_pattern.lambda_offset,
        snapshot.augmented_pattern.z_offset,
        snapshot.expected_augmented_inertia.positive,
        snapshot.expected_augmented_inertia.negative,
        snapshot.expected_augmented_inertia.zero,
        snapshot.regularization,
        snapshot.primal_diagonal_shift,
        snapshot.dual_regularization,
        snapshot.barrier_parameter,
        snapshot.primal_inf,
        snapshot.dual_inf,
        snapshot.complementarity_inf,
        snapshot.line_search_trials,
        report.verdict,
    ));
    for result in &report.results {
        body.push_str(&format!(
            "result solver={} success={} reg={:.6e} inertia={:?} residual={:?} solution_inf={:?} step_inf={:?} step_delta={:?} dx_delta={:?} dlambda_delta={:?} ds_delta={:?} dz_delta={:?} detail={:?}\n",
            result.solver.label(),
            result.success,
            result.regularization,
            result.inertia,
            result.residual_inf,
            result.solution_inf,
            result.step_inf,
            result.step_delta_inf,
            result.dx_delta_inf,
            result.d_lambda_delta_inf,
            result.ds_delta_inf,
            result.dz_delta_inf,
            result.detail,
        ));
    }
    for note in &report.notes {
        body.push_str(&format!("note={note}\n"));
    }
    body.push_str(&format!(
        "col_ptrs={:?}\nrow_indices={:?}\nvalues={:?}\nrhs={:?}\nslack={:?}\nmultipliers={:?}\n",
        snapshot.augmented_pattern.ccs.col_ptrs,
        snapshot.augmented_pattern.ccs.row_indices,
        snapshot.augmented_values,
        snapshot.augmented_rhs,
        snapshot.slack,
        snapshot.multipliers,
    ));
    let _ = fs::write(path, body);
}

fn spawn_spral_stage_heartbeat(label: &'static str) -> (Arc<AtomicBool>, thread::JoinHandle<()>) {
    let finished = Arc::new(AtomicBool::new(false));
    let finished_for_thread = Arc::clone(&finished);
    let handle = thread::spawn(move || {
        let started = Instant::now();
        let mut next_notice_after = Duration::from_secs(5);
        loop {
            let elapsed = started.elapsed();
            if elapsed < next_notice_after {
                thread::park_timeout(next_notice_after - elapsed);
            }
            if finished_for_thread.load(Ordering::Relaxed) {
                break;
            }
            if let Some(progress) = spral_current_factorization_progress() {
                let root_tail_status = if progress.current_root_delayed_block > 0 {
                    format!(
                        " root_delayed={}/{} size={} stage={}",
                        progress.current_root_delayed_block,
                        progress.total_root_delayed_blocks,
                        progress.current_root_delayed_block_size,
                        progress.root_delayed_stage.label(),
                    )
                } else if progress.completed_root_delayed_blocks > 0
                    && progress.completed_pivots < progress.total_pivots
                {
                    format!(
                        " root_delayed={}/{} stage=between_blocks",
                        progress.completed_root_delayed_blocks, progress.total_root_delayed_blocks,
                    )
                } else {
                    String::new()
                };
                println!(
                    "{label} still running after {}: weighted={:.0}% pivots={}/{} ({:.0}%) fronts={}/{} roots={}/{}{}",
                    compact_duration_text(started.elapsed().as_secs_f64()),
                    progress.weighted_percent(),
                    progress.completed_pivots,
                    progress.total_pivots,
                    progress.pivot_percent(),
                    progress.completed_fronts,
                    progress.total_fronts,
                    progress.completed_roots,
                    progress.total_roots,
                    root_tail_status,
                );
            } else {
                println!(
                    "{label} still running after {}",
                    compact_duration_text(started.elapsed().as_secs_f64()),
                );
            }
            next_notice_after += Duration::from_secs(10);
        }
    });
    (finished, handle)
}

#[derive(Clone)]
struct EvalState {
    objective_value: f64,
    gradient: Vec<f64>,
    equality_values: Vec<f64>,
    augmented_inequality_values: Vec<f64>,
    equality_jacobian: SparseMatrix,
    inequality_jacobian: SparseMatrix,
}

fn merit_residual(primal_inf: f64, dual_inf: f64, complementarity_inf: f64, mu: f64) -> f64 {
    primal_inf.max(dual_inf).max(complementarity_inf).max(mu)
}

fn l1_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).sum()
}

fn slack_form_inequality_residuals(augmented_inequality_values: &[f64], slack: &[f64]) -> Vec<f64> {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    augmented_inequality_values
        .iter()
        .zip(slack.iter())
        .map(|(&g_i, &s_i)| g_i + s_i)
        .collect()
}

fn slack_form_inequality_l1_norm(augmented_inequality_values: &[f64], slack: &[f64]) -> f64 {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    augmented_inequality_values
        .iter()
        .zip(slack.iter())
        .map(|(&g_i, &s_i)| (g_i + s_i).abs())
        .sum()
}

fn slack_form_inequality_inf_norm(augmented_inequality_values: &[f64], slack: &[f64]) -> f64 {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    augmented_inequality_values
        .iter()
        .zip(slack.iter())
        .fold(0.0, |acc, (&g_i, &s_i)| acc.max((g_i + s_i).abs()))
}

fn filter_theta_l1_norm(
    equality_values: &[f64],
    augmented_inequality_values: &[f64],
    slack: &[f64],
) -> f64 {
    l1_norm(equality_values) + slack_form_inequality_l1_norm(augmented_inequality_values, slack)
}

fn barrier_objective_value(objective_value: f64, slack: &[f64], barrier_parameter: f64) -> f64 {
    if slack.is_empty() || barrier_parameter <= 0.0 {
        return objective_value;
    }
    objective_value - barrier_parameter * slack.iter().map(|value| value.ln()).sum::<f64>()
}

fn barrier_objective_directional_derivative(
    gradient: &[f64],
    slack: &[f64],
    dx: &[f64],
    ds: &[f64],
    barrier_parameter: f64,
) -> f64 {
    let objective_term = gradient
        .iter()
        .zip(dx.iter())
        .map(|(gradient_i, dx_i)| gradient_i * dx_i)
        .sum::<f64>();
    if slack.is_empty() || barrier_parameter <= 0.0 {
        return objective_term;
    }
    let barrier_term = slack
        .iter()
        .zip(ds.iter())
        .map(|(slack_i, ds_i)| ds_i / slack_i.max(1e-16))
        .sum::<f64>();
    objective_term - barrier_parameter * barrier_term
}

fn switching_condition_satisfied(
    theta: f64,
    barrier_directional_derivative: f64,
    alpha_primal: f64,
    options: &InteriorPointOptions,
) -> bool {
    barrier_directional_derivative < 0.0
        && alpha_primal * (-barrier_directional_derivative).powf(options.s_phi)
            > options.delta * theta.max(0.0).powf(options.s_theta)
}

fn calculate_filter_alpha_min(
    theta: f64,
    theta_min: f64,
    barrier_directional_derivative: f64,
    options: &InteriorPointOptions,
) -> f64 {
    let mut alpha_min = options.filter_gamma_violation;
    if barrier_directional_derivative < 0.0 {
        alpha_min = alpha_min.min(
            options.filter_gamma_objective * theta
                / (-barrier_directional_derivative).max(f64::MIN_POSITIVE),
        );
        if theta <= theta_min {
            alpha_min = alpha_min.min(
                options.delta * theta.max(0.0).powf(options.s_theta)
                    / (-barrier_directional_derivative).powf(options.s_phi),
            );
        }
    }
    (options.alpha_min_frac * alpha_min).max(options.min_step)
}

fn barrier_objective_increase_too_large(
    reference_barrier_objective: f64,
    trial_barrier_objective: f64,
    obj_max_inc: f64,
) -> bool {
    if trial_barrier_objective <= reference_barrier_objective {
        return false;
    }
    let increase = trial_barrier_objective - reference_barrier_objective;
    let baseline = if reference_barrier_objective == 0.0 {
        0.0
    } else {
        reference_barrier_objective.abs().log10()
    };
    increase > 0.0 && increase.log10() > obj_max_inc + baseline
}

fn should_reduce_barrier_parameter(
    barrier_subproblem_error: f64,
    barrier_parameter: f64,
    options: &InteriorPointOptions,
) -> bool {
    barrier_subproblem_error <= options.barrier_tol_factor * barrier_parameter
}

fn minimum_monotone_barrier_parameter(options: &InteriorPointOptions) -> f64 {
    let tol = options.overall_tol.min(options.complementarity_tol);
    options
        .mu_target
        .max(options.mu_min)
        .max(tol / (options.barrier_tol_factor + 1.0))
}

fn next_barrier_parameter_once(
    barrier_parameter: f64,
    barrier_subproblem_error: f64,
    options: &InteriorPointOptions,
) -> f64 {
    let minimum_barrier = minimum_monotone_barrier_parameter(options);
    if barrier_parameter <= minimum_barrier {
        return minimum_barrier;
    }
    if !should_reduce_barrier_parameter(barrier_subproblem_error, barrier_parameter, options) {
        return barrier_parameter;
    }
    minimum_barrier.max(
        (options.mu_linear_decrease_factor * barrier_parameter)
            .min(barrier_parameter.powf(options.mu_superlinear_decrease_power)),
    )
}

fn next_barrier_parameter(
    barrier_parameter: f64,
    barrier_subproblem_error: f64,
    tiny_step: bool,
    options: &InteriorPointOptions,
) -> f64 {
    let mut current_barrier = barrier_parameter;
    let minimum_barrier = minimum_monotone_barrier_parameter(options);
    let mut first_pass = true;
    while first_pass
        || (options.mu_allow_fast_monotone_decrease
            && should_reduce_barrier_parameter(barrier_subproblem_error, current_barrier, options))
    {
        first_pass = false;
        if !tiny_step
            && !should_reduce_barrier_parameter(barrier_subproblem_error, current_barrier, options)
        {
            break;
        }
        let next_barrier =
            next_barrier_parameter_once(current_barrier, barrier_subproblem_error, options);
        if next_barrier >= current_barrier - 1e-18 {
            break;
        }
        current_barrier = next_barrier.max(minimum_barrier);
        if !options.mu_allow_fast_monotone_decrease {
            break;
        }
    }
    current_barrier.max(minimum_barrier)
}

fn interior_point_complementarity_target_inf_norm(
    slack: &[f64],
    multipliers: &[f64],
    mu_target: f64,
) -> f64 {
    slack
        .iter()
        .zip(multipliers.iter())
        .fold(0.0, |acc, (s, z)| acc.max((s * z - mu_target).abs()))
}

fn interior_point_current_is_acceptable(
    state: InteriorPointAcceptableState,
    last_objective: f64,
    options: &InteriorPointOptions,
) -> bool {
    let relative_objective_change =
        (state.objective - last_objective).abs() / state.objective.abs().max(1.0);
    state.overall_error <= options.acceptable_tol
        && state.dual_inf <= options.acceptable_dual_inf_tol
        && state.constr_viol <= options.acceptable_constr_viol_tol
        && state.compl_inf <= options.acceptable_compl_inf_tol
        && relative_objective_change <= options.acceptable_obj_change_tol
}

fn step_inf_norm(step: &[f64]) -> f64 {
    step.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn accepted_ip_step_inf_norm(direction: &NewtonDirection, alpha_pr: f64) -> f64 {
    alpha_pr * step_inf_norm(&direction.dx)
}

fn alpha_for_y(
    alpha_primal: f64,
    alpha_dual: f64,
    direction: &NewtonDirection,
    options: &InteriorPointOptions,
) -> f64 {
    let primal_step_norm = step_inf_norm(&direction.dx).max(step_inf_norm(&direction.ds));
    let alpha = match options.alpha_for_y {
        InteriorPointAlphaForYStrategy::Primal => alpha_primal,
        InteriorPointAlphaForYStrategy::BoundMultiplier => alpha_dual,
        InteriorPointAlphaForYStrategy::Min => alpha_primal.min(alpha_dual),
        InteriorPointAlphaForYStrategy::Max => alpha_primal.max(alpha_dual),
        InteriorPointAlphaForYStrategy::Full => 1.0,
        InteriorPointAlphaForYStrategy::PrimalAndFull => {
            if primal_step_norm <= options.alpha_for_y_tol {
                1.0
            } else {
                alpha_primal
            }
        }
        InteriorPointAlphaForYStrategy::DualAndFull => {
            if primal_step_norm <= options.alpha_for_y_tol {
                1.0
            } else {
                alpha_dual
            }
        }
    };
    alpha.clamp(0.0, 1.0)
}

fn kkt_regularization(
    has_inequalities: bool,
    primal_inf: f64,
    complementarity_inf: f64,
    dual_inf: f64,
    options: &InteriorPointOptions,
) -> f64 {
    let _ = (has_inequalities, primal_inf, complementarity_inf, dual_inf);
    options.regularization
}

#[derive(Clone, Copy, Debug)]
struct InteriorPointDisplayTolerances {
    constraint: f64,
    dual: f64,
    complementarity: f64,
    overall: f64,
}

#[derive(Clone, Copy, Debug)]
enum InteriorPointResidualMetric {
    Constraint,
    Dual,
    Complementarity,
    Overall,
}

#[derive(Clone, Copy, Debug)]
enum InteriorPointDisplayMode {
    Strict(InteriorPointDisplayTolerances),
    Acceptable {
        strict: InteriorPointDisplayTolerances,
        acceptable: InteriorPointDisplayTolerances,
    },
}

impl InteriorPointDisplayMode {
    fn strict(options: &InteriorPointOptions) -> Self {
        Self::Strict(interior_point_display_tolerances(
            options,
            InteriorPointTermination::Converged,
        ))
    }

    fn for_termination(
        options: &InteriorPointOptions,
        termination: InteriorPointTermination,
    ) -> Self {
        let strict =
            interior_point_display_tolerances(options, InteriorPointTermination::Converged);
        match termination {
            InteriorPointTermination::Converged => Self::Strict(strict),
            InteriorPointTermination::Acceptable => Self::Acceptable {
                strict,
                acceptable: interior_point_display_tolerances(options, termination),
            },
        }
    }

    fn tolerances(self, metric: InteriorPointResidualMetric) -> (f64, Option<f64>) {
        let select = |tolerances: InteriorPointDisplayTolerances| match metric {
            InteriorPointResidualMetric::Constraint => tolerances.constraint,
            InteriorPointResidualMetric::Dual => tolerances.dual,
            InteriorPointResidualMetric::Complementarity => tolerances.complementarity,
            InteriorPointResidualMetric::Overall => tolerances.overall,
        };
        match self {
            Self::Strict(strict) => (select(strict), None),
            Self::Acceptable { strict, acceptable } => (select(strict), Some(select(acceptable))),
        }
    }
}

fn interior_point_display_tolerances(
    options: &InteriorPointOptions,
    termination: InteriorPointTermination,
) -> InteriorPointDisplayTolerances {
    match termination {
        InteriorPointTermination::Converged => InteriorPointDisplayTolerances {
            constraint: options.constraint_tol,
            dual: options.dual_tol,
            complementarity: options.complementarity_tol,
            overall: options.overall_tol,
        },
        InteriorPointTermination::Acceptable => InteriorPointDisplayTolerances {
            constraint: options.acceptable_constr_viol_tol,
            dual: options.acceptable_dual_inf_tol,
            complementarity: options.acceptable_compl_inf_tol,
            overall: options.acceptable_tol,
        },
    }
}

fn is_shortened_ip_step(
    _alpha_pr: f64,
    _alpha_du: Option<f64>,
    line_search_iterations: Index,
) -> bool {
    line_search_iterations > 0
}

fn is_tiny_ip_step(step_inf: f64, options: &InteriorPointOptions) -> bool {
    step_inf <= options.tiny_step_tol
}

fn fraction_to_boundary_with_limiter(
    current: &[f64],
    direction: &[f64],
    tau: f64,
) -> (f64, Option<InteriorPointBoundaryLimiter>) {
    let mut alpha = 1.0_f64;
    let mut limiter = None;
    for (idx, (&value, &delta)) in current.iter().zip(direction.iter()).enumerate() {
        if delta < 0.0 {
            let candidate = (-tau * value / delta).clamp(0.0, 1.0);
            if candidate < alpha {
                alpha = candidate;
                limiter = Some(InteriorPointBoundaryLimiter {
                    index: idx,
                    value,
                    direction: delta,
                    alpha: candidate,
                });
            }
        }
    }
    (alpha.clamp(0.0, 1.0), limiter)
}

fn interior_point_direction_diagnostics(
    direction: &NewtonDirection,
    alpha_pr_limiter: Option<InteriorPointBoundaryLimiter>,
    alpha_du_limiter: Option<InteriorPointBoundaryLimiter>,
) -> InteriorPointDirectionDiagnostics {
    InteriorPointDirectionDiagnostics {
        dx_inf: step_inf_norm(&direction.dx),
        d_lambda_inf: step_inf_norm(&direction.d_lambda),
        ds_inf: step_inf_norm(&direction.ds),
        dz_inf: step_inf_norm(&direction.dz),
        alpha_pr_limiter,
        alpha_du_limiter,
    }
}

fn barrier_parameter(slack: &[f64], multipliers: &[f64]) -> f64 {
    if slack.is_empty() {
        0.0
    } else {
        slack
            .iter()
            .zip(multipliers.iter())
            .map(|(s, z)| s * z)
            .sum::<f64>()
            / slack.len() as f64
    }
}

fn correct_bound_multiplier_estimate(
    trial_z: &[f64],
    trial_slack: &[f64],
    barrier_parameter: f64,
    kappa_sigma: f64,
) -> (Vec<f64>, f64) {
    if kappa_sigma < 1.0 || trial_z.is_empty() {
        return (trial_z.to_vec(), 0.0);
    }

    let upper_complementarity = kappa_sigma * barrier_parameter;
    let lower_complementarity = barrier_parameter / kappa_sigma;
    let needs_correction = trial_z
        .iter()
        .zip(trial_slack.iter())
        .any(|(z_i, slack_i)| {
            let compl = z_i * slack_i;
            compl > upper_complementarity || compl < lower_complementarity
        });
    if !needs_correction {
        return (trial_z.to_vec(), 0.0);
    }

    let mut corrected_z = trial_z.to_vec();
    let mut max_correction = 0.0_f64;
    for (z_i, slack_i) in corrected_z.iter_mut().zip(trial_slack.iter()) {
        let slack_floor = slack_i.max(1e-16);
        let upper_multiplier = upper_complementarity / slack_floor;
        let lower_multiplier = lower_complementarity / slack_floor;
        let corrected = z_i.min(upper_multiplier).max(lower_multiplier);
        max_correction = max_correction.max((corrected - *z_i).abs());
        *z_i = corrected;
    }

    (corrected_z, max_correction)
}

fn apply_bound_multiplier_safeguard(
    state: &EvalState,
    primal_inf: f64,
    lambda: &[f64],
    slack: &[f64],
    z: &[f64],
    barrier_parameter_value: f64,
    options: &InteriorPointOptions,
) -> Option<AcceptedTrialMultiplierState> {
    let (corrected_z, max_correction) =
        correct_bound_multiplier_estimate(z, slack, barrier_parameter_value, options.kappa_sigma);
    if max_correction <= 0.0 {
        return None;
    }

    let corrected_dual_residual = lagrangian_gradient_sparse(
        &state.gradient,
        &state.equality_jacobian,
        lambda,
        &state.inequality_jacobian,
        &corrected_z,
    );
    let corrected_dual_inf = inf_norm(&corrected_dual_residual);
    let corrected_comp_inf = if corrected_z.is_empty() {
        0.0
    } else {
        complementarity_inf_norm(slack, &corrected_z)
    };
    let corrected_all_dual_multipliers = [lambda, corrected_z.as_slice()].concat();
    let corrected_overall_inf = scaled_overall_inf_norm(
        primal_inf,
        corrected_dual_inf,
        corrected_comp_inf,
        &corrected_all_dual_multipliers,
        &corrected_z,
        options.overall_scale_max,
    );

    Some(AcceptedTrialMultiplierState {
        z: corrected_z,
        dual_inf: corrected_dual_inf,
        complementarity_inf: corrected_comp_inf,
        overall_inf: corrected_overall_inf,
    })
}

fn adapter_timing_delta<P>(
    problem: &P,
    previous: &mut Option<SolverAdapterTiming>,
) -> Option<SolverAdapterTiming>
where
    P: CompiledNlpProblem,
{
    let current = problem.adapter_timing_snapshot();
    let delta = match (current, *previous) {
        (Some(current), Some(previous)) => Some(current.saturating_sub(previous)),
        (Some(current), None) => Some(current),
        (None, _) => None,
    };
    *previous = current;
    delta
}

fn push_scalar_to_bounds_interior(
    value: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    bound_push: f64,
    bound_frac: f64,
) -> f64 {
    let mut pushed = match (lower, upper) {
        (Some(lower), Some(upper)) if lower <= upper => value.clamp(lower, upper),
        (Some(lower), _) => value.max(lower),
        (_, Some(upper)) => value.min(upper),
        (None, None) => value,
    };
    let lower_margin = lower.map(|lower| {
        let absolute_margin = bound_push * lower.abs().max(1.0);
        match upper {
            Some(upper) if upper > lower => absolute_margin.min(bound_frac * (upper - lower)),
            _ => absolute_margin,
        }
    });
    let upper_margin = upper.map(|upper| {
        let absolute_margin = bound_push * upper.abs().max(1.0);
        match lower {
            Some(lower) if upper > lower => absolute_margin.min(bound_frac * (upper - lower)),
            _ => absolute_margin,
        }
    });
    if let Some(lower) = lower
        && let Some(lower_margin) = lower_margin
    {
        pushed = pushed.max(lower + lower_margin);
    }
    if let Some(upper) = upper
        && let Some(upper_margin) = upper_margin
    {
        pushed = pushed.min(upper - upper_margin);
    }
    if let (Some(lower), Some(upper), Some(lower_margin), Some(upper_margin)) =
        (lower, upper, lower_margin, upper_margin)
    {
        let interior_lower = lower + lower_margin;
        let interior_upper = upper - upper_margin;
        if interior_lower > interior_upper {
            pushed = 0.5 * (lower + upper);
        }
    }
    pushed
}

fn project_initial_point_into_box_interior(
    x: &mut [f64],
    bounds: &BoundConstraints,
    options: &InteriorPointOptions,
) {
    for (idx, x_i) in x.iter_mut().enumerate() {
        let lower = bounds
            .lower_indices
            .iter()
            .position(|lower_idx| *lower_idx == idx)
            .map(|position| bounds.lower_values[position]);
        let upper = bounds
            .upper_indices
            .iter()
            .position(|upper_idx| *upper_idx == idx)
            .map(|position| bounds.upper_values[position]);
        *x_i = push_scalar_to_bounds_interior(
            *x_i,
            lower,
            upper,
            options.bound_push,
            options.bound_frac,
        );
    }
}

fn initialise_slacks(
    augmented_inequality_values: &[f64],
    slack: &mut [f64],
    options: &InteriorPointOptions,
) {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    for (&g_i, s_i) in augmented_inequality_values.iter().zip(slack.iter_mut()) {
        *s_i = push_scalar_to_bounds_interior(
            (-g_i).max(0.0),
            Some(0.0),
            None,
            options.slack_bound_push,
            options.slack_bound_frac,
        );
    }
}

fn sparse_row_entries(ccs: &CCS) -> Vec<Vec<(usize, usize)>> {
    let mut row_entries = vec![Vec::new(); ccs.nrow];
    for col in 0..ccs.ncol {
        for index in ccs.col_ptrs[col]..ccs.col_ptrs[col + 1] {
            row_entries[ccs.row_indices[index]].push((col, index));
        }
    }
    row_entries
}

fn sparse_structure_from_ccs(ccs: &CCS) -> SparseMatrixStructure {
    SparseMatrixStructure {
        ccs: ccs.clone(),
        row_entries: sparse_row_entries(ccs),
    }
}

fn build_bound_jacobian_sparse(bounds: &BoundConstraints, dimension: Index) -> SparseMatrix {
    let mut columns = vec![Vec::<(usize, f64)>::new(); dimension];
    for (row, &idx) in bounds.lower_indices.iter().enumerate() {
        columns[idx].push((row, -1.0));
    }
    let row_offset = bounds.lower_indices.len();
    for (row, &idx) in bounds.upper_indices.iter().enumerate() {
        columns[idx].push((row_offset + row, 1.0));
    }

    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for column in columns {
        for (row, value) in column {
            row_indices.push(row);
            values.push(value);
        }
        col_ptrs.push(row_indices.len());
    }

    let ccs = CCS::new(bounds.total_count(), dimension, col_ptrs, row_indices);
    SparseMatrix {
        structure: Arc::new(sparse_structure_from_ccs(&ccs)),
        values,
    }
}

fn vstack_sparse_structures(
    top: &SparseMatrixStructure,
    bottom: &SparseMatrixStructure,
) -> SparseMatrixStructure {
    debug_assert_eq!(top.ccs.ncol, bottom.ccs.ncol);
    let row_offset = top.ccs.nrow;
    let mut col_ptrs = Vec::with_capacity(top.ccs.ncol + 1);
    let mut row_indices = Vec::with_capacity(top.ccs.nnz() + bottom.ccs.nnz());
    col_ptrs.push(0);
    for col in 0..top.ccs.ncol {
        row_indices.extend_from_slice(
            &top.ccs.row_indices[top.ccs.col_ptrs[col]..top.ccs.col_ptrs[col + 1]],
        );
        row_indices.extend(
            bottom.ccs.row_indices[bottom.ccs.col_ptrs[col]..bottom.ccs.col_ptrs[col + 1]]
                .iter()
                .map(|row| row_offset + row),
        );
        col_ptrs.push(row_indices.len());
    }
    sparse_structure_from_ccs(&CCS::new(
        top.ccs.nrow + bottom.ccs.nrow,
        top.ccs.ncol,
        col_ptrs,
        row_indices,
    ))
}

fn stack_sparse_values(top_ccs: &CCS, top_values: &[f64], bottom: &SparseMatrix) -> Vec<f64> {
    debug_assert_eq!(top_values.len(), top_ccs.nnz());
    debug_assert_eq!(top_ccs.ncol, bottom.structure.ccs.ncol);
    let mut values = Vec::with_capacity(top_values.len() + bottom.values.len());
    for col in 0..top_ccs.ncol {
        values.extend_from_slice(&top_values[top_ccs.col_ptrs[col]..top_ccs.col_ptrs[col + 1]]);
        values.extend_from_slice(
            &bottom.values
                [bottom.structure.ccs.col_ptrs[col]..bottom.structure.ccs.col_ptrs[col + 1]],
        );
    }
    values
}

fn sparse_add_transpose_mat_vec(out: &mut [f64], matrix: &SparseMatrix, vector: &[f64]) {
    debug_assert_eq!(matrix.nrows(), vector.len());
    debug_assert_eq!(matrix.ncols(), out.len());
    for (col, out_col) in out.iter_mut().enumerate().take(matrix.ncols()) {
        let mut sum = 0.0;
        for index in matrix.structure.ccs.col_ptrs[col]..matrix.structure.ccs.col_ptrs[col + 1] {
            sum += matrix.values[index] * vector[matrix.structure.ccs.row_indices[index]];
        }
        *out_col += sum;
    }
}

fn sparse_mat_vec(matrix: &SparseMatrix, vector: &[f64]) -> Vec<f64> {
    debug_assert_eq!(matrix.ncols(), vector.len());
    let mut product = vec![0.0; matrix.nrows()];
    for (col, &x) in vector.iter().enumerate().take(matrix.ncols()) {
        if x == 0.0 {
            continue;
        }
        for index in matrix.structure.ccs.col_ptrs[col]..matrix.structure.ccs.col_ptrs[col + 1] {
            product[matrix.structure.ccs.row_indices[index]] += matrix.values[index] * x;
        }
    }
    product
}

fn symmetric_ccs_lower_mat_vec(ccs: &CCS, values: &[f64], vector: &[f64]) -> Vec<f64> {
    debug_assert_eq!(ccs.nrow, ccs.ncol);
    debug_assert_eq!(ccs.nrow, vector.len());
    debug_assert_eq!(ccs.nnz(), values.len());
    let mut product = vec![0.0; ccs.nrow];
    for col in 0..ccs.ncol {
        let x_col = vector[col];
        let start = ccs.col_ptrs[col];
        let end = ccs.col_ptrs[col + 1];
        for (&row, &value) in ccs.row_indices[start..end]
            .iter()
            .zip(values[start..end].iter())
        {
            product[row] += value * x_col;
            if row != col {
                product[col] += value * vector[row];
            }
        }
    }
    product
}

fn symmetric_ccs_lower_abs_mat_vec(ccs: &CCS, values: &[f64], vector_abs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(ccs.nrow, ccs.ncol);
    debug_assert_eq!(ccs.nrow, vector_abs.len());
    debug_assert_eq!(ccs.nnz(), values.len());
    let mut product = vec![0.0; ccs.nrow];
    for col in 0..ccs.ncol {
        let x_col_abs = vector_abs[col];
        let start = ccs.col_ptrs[col];
        let end = ccs.col_ptrs[col + 1];
        for (&row, &value) in ccs.row_indices[start..end]
            .iter()
            .zip(values[start..end].iter())
        {
            let value_abs = value.abs();
            product[row] += value_abs * x_col_abs;
            if row != col {
                product[col] += value_abs * vector_abs[row];
            }
        }
    }
    product
}

fn build_spral_augmented_kkt_pattern(
    hessian_structure: &CCS,
    equality_jacobian: &SparseMatrixStructure,
    inequality_jacobian: &SparseMatrixStructure,
) -> std::result::Result<SpralAugmentedKktPattern, InteriorPointSolveError> {
    let n = hessian_structure.nrow;
    if hessian_structure.ncol != n {
        return Err(InteriorPointSolveError::InvalidInput(
            "interior-point Hessian structure must be square".into(),
        ));
    }
    let meq = equality_jacobian.ccs.nrow;
    let mineq = inequality_jacobian.ccs.nrow;
    let p_offset = n;
    let lambda_offset = p_offset + mineq;
    let z_offset = lambda_offset + meq;
    let total_dimension = z_offset + mineq;

    let mut columns = vec![Vec::<usize>::new(); total_dimension];
    for (col, column) in columns.iter_mut().enumerate().take(n) {
        column.push(col);
        for index in hessian_structure.col_ptrs[col]..hessian_structure.col_ptrs[col + 1] {
            column.push(hessian_structure.row_indices[index]);
        }
    }
    for (col, column) in columns
        .iter_mut()
        .enumerate()
        .take(equality_jacobian.ccs.ncol)
    {
        for index in equality_jacobian.ccs.col_ptrs[col]..equality_jacobian.ccs.col_ptrs[col + 1] {
            column.push(lambda_offset + equality_jacobian.ccs.row_indices[index]);
        }
    }
    for (col, column) in columns
        .iter_mut()
        .enumerate()
        .take(inequality_jacobian.ccs.ncol)
    {
        for index in
            inequality_jacobian.ccs.col_ptrs[col]..inequality_jacobian.ccs.col_ptrs[col + 1]
        {
            column.push(z_offset + inequality_jacobian.ccs.row_indices[index]);
        }
    }
    for row in 0..mineq {
        let p_index = p_offset + row;
        columns[p_index].push(p_index);
        columns[p_index].push(z_offset + row);
    }
    for row in 0..meq {
        let lambda_index = lambda_offset + row;
        columns[lambda_index].push(lambda_index);
    }
    for row in 0..mineq {
        let z_index = z_offset + row;
        columns[z_index].push(z_index);
    }

    let mut col_ptrs = Vec::with_capacity(total_dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for (col, rows) in columns.iter_mut().enumerate() {
        rows.sort_unstable();
        rows.dedup();
        if rows.iter().any(|&row| row < col) {
            return Err(InteriorPointSolveError::InvalidInput(format!(
                "invalid lower-triangular sparse KKT pattern in column {col}"
            )));
        }
        row_indices.extend_from_slice(rows);
        col_ptrs.push(row_indices.len());
    }
    let ccs = Arc::new(CCS::new(
        total_dimension,
        total_dimension,
        col_ptrs,
        row_indices,
    ));
    let mut column_maps = vec![HashMap::<usize, usize>::new(); total_dimension];
    for (col, column_map) in column_maps.iter_mut().enumerate().take(total_dimension) {
        for index in ccs.col_ptrs[col]..ccs.col_ptrs[col + 1] {
            column_map.insert(ccs.row_indices[index], index);
        }
    }
    let lookup = |col: usize, row: usize| -> std::result::Result<usize, InteriorPointSolveError> {
        column_maps[col].get(&row).copied().ok_or_else(|| {
            InteriorPointSolveError::InvalidInput(format!(
                "missing sparse KKT slot for entry ({row}, {col})"
            ))
        })
    };
    let mut hessian_value_indices = Vec::with_capacity(hessian_structure.nnz());
    for col in 0..n {
        for index in hessian_structure.col_ptrs[col]..hessian_structure.col_ptrs[col + 1] {
            hessian_value_indices.push(lookup(col, hessian_structure.row_indices[index])?);
        }
    }
    let mut equality_jacobian_value_indices = Vec::with_capacity(equality_jacobian.ccs.nnz());
    for col in 0..equality_jacobian.ccs.ncol {
        for index in equality_jacobian.ccs.col_ptrs[col]..equality_jacobian.ccs.col_ptrs[col + 1] {
            equality_jacobian_value_indices.push(lookup(
                col,
                lambda_offset + equality_jacobian.ccs.row_indices[index],
            )?);
        }
    }
    let mut inequality_jacobian_value_indices = Vec::with_capacity(inequality_jacobian.ccs.nnz());
    for col in 0..inequality_jacobian.ccs.ncol {
        for index in
            inequality_jacobian.ccs.col_ptrs[col]..inequality_jacobian.ccs.col_ptrs[col + 1]
        {
            inequality_jacobian_value_indices.push(lookup(
                col,
                z_offset + inequality_jacobian.ccs.row_indices[index],
            )?);
        }
    }
    let x_diagonal_indices = (0..n)
        .map(|diag| lookup(diag, diag))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let p_diagonal_indices = (0..mineq)
        .map(|diag| {
            let index = p_offset + diag;
            lookup(index, index)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let lambda_diagonal_indices = (0..meq)
        .map(|diag| {
            let index = lambda_offset + diag;
            lookup(index, index)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let z_diagonal_indices = (0..mineq)
        .map(|diag| {
            let index = z_offset + diag;
            lookup(index, index)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let pz_indices = (0..mineq)
        .map(|row| lookup(p_offset + row, z_offset + row))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(SpralAugmentedKktPattern {
        ccs,
        x_dimension: n,
        inequality_dimension: mineq,
        equality_dimension: meq,
        p_offset,
        lambda_offset,
        z_offset,
        hessian_value_indices,
        equality_jacobian_value_indices,
        inequality_jacobian_value_indices,
        x_diagonal_indices,
        p_diagonal_indices,
        lambda_diagonal_indices,
        z_diagonal_indices,
        pz_indices,
    })
}

fn lagrangian_gradient_sparse(
    gradient: &[f64],
    equality_jacobian: &SparseMatrix,
    equality_multipliers: &[f64],
    inequality_jacobian: &SparseMatrix,
    inequality_multipliers: &[f64],
) -> Vec<f64> {
    let mut residual = gradient.to_vec();
    sparse_add_transpose_mat_vec(&mut residual, equality_jacobian, equality_multipliers);
    sparse_add_transpose_mat_vec(&mut residual, inequality_jacobian, inequality_multipliers);
    residual
}

#[expect(
    clippy::too_many_arguments,
    reason = "State evaluation threads through compiled NLP data and profiling sinks explicitly."
)]
fn trial_state<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    bounds: &BoundConstraints,
    bound_jacobian: &SparseMatrix,
    equality_jacobian_structure: &Arc<SparseMatrixStructure>,
    inequality_jacobian_structure: &Arc<SparseMatrixStructure>,
    profiling: &mut InteriorPointProfiling,
    callback_time: &mut Duration,
) -> EvalState
where
    P: CompiledNlpProblem,
{
    let mut gradient = vec![0.0; problem.dimension()];
    let mut equality_values = vec![0.0; problem.equality_count()];
    let mut nonlinear_inequality_values = vec![0.0; problem.inequality_count()];
    let mut augmented_inequality_values =
        vec![0.0; problem.inequality_count() + bounds.total_count()];
    let mut equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    let objective_value = time_callback(&mut profiling.objective_value, callback_time, || {
        problem.objective_value(x, parameters)
    });
    time_callback(&mut profiling.objective_gradient, callback_time, || {
        problem.objective_gradient(x, parameters, &mut gradient);
    });
    time_callback(&mut profiling.equality_values, callback_time, || {
        problem.equality_values(x, parameters, &mut equality_values);
    });
    time_callback(&mut profiling.inequality_values, callback_time, || {
        problem.inequality_values(x, parameters, &mut nonlinear_inequality_values);
    });
    augment_inequality_values(
        &nonlinear_inequality_values,
        x,
        bounds,
        &mut augmented_inequality_values,
    );
    time_callback(
        &mut profiling.equality_jacobian_values,
        callback_time,
        || problem.equality_jacobian_values(x, parameters, &mut equality_jacobian_values),
    );
    time_callback(
        &mut profiling.inequality_jacobian_values,
        callback_time,
        || problem.inequality_jacobian_values(x, parameters, &mut inequality_jacobian_values),
    );
    EvalState {
        objective_value,
        gradient,
        equality_values,
        augmented_inequality_values,
        equality_jacobian: SparseMatrix {
            structure: Arc::clone(equality_jacobian_structure),
            values: equality_jacobian_values,
        },
        inequality_jacobian: SparseMatrix {
            structure: Arc::clone(inequality_jacobian_structure),
            values: stack_sparse_values(
                problem.inequality_jacobian_ccs(),
                &inequality_jacobian_values,
                bound_jacobian,
            ),
        },
    }
}

fn least_squares_initial_dual_state(
    state: &EvalState,
    initial_z: &[f64],
    regularization: f64,
) -> (Vec<f64>, Vec<f64>) {
    let meq = state.equality_values.len();
    let mineq = state.augmented_inequality_values.len();
    if meq + mineq == 0 {
        return (Vec::new(), Vec::new());
    }

    debug_assert_eq!(initial_z.len(), mineq);
    let lambda_eq = least_squares_equality_multipliers(state, initial_z, regularization);
    (lambda_eq, initial_z.to_vec())
}

fn least_squares_equality_multipliers(
    state: &EvalState,
    z: &[f64],
    regularization: f64,
) -> Vec<f64> {
    let dual_regularization = regularization.max(1e-8);
    let meq = state.equality_values.len();
    let mineq = state.augmented_inequality_values.len();
    if meq == 0 {
        return Vec::new();
    }
    let mut adjusted_gradient = state.gradient.clone();
    if mineq > 0 {
        sparse_add_transpose_mat_vec(&mut adjusted_gradient, &state.inequality_jacobian, z);
    }

    let n = state.gradient.len();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    let mut rhs = vec![0.0; meq];
    for (col, &adjusted_grad_col) in adjusted_gradient.iter().enumerate().take(n) {
        let start = state.equality_jacobian.structure.ccs.col_ptrs[col];
        let end = state.equality_jacobian.structure.ccs.col_ptrs[col + 1];
        let mut hits = Vec::new();
        for index in start..end {
            hits.push((
                state.equality_jacobian.structure.ccs.row_indices[index],
                state.equality_jacobian.values[index],
            ));
        }
        for &(row, value) in &hits {
            rhs[row] -= value * adjusted_grad_col;
        }
        for (offset, &(row_i, value_i)) in hits.iter().enumerate() {
            for &(row_j, value_j) in &hits[offset..] {
                rows.push(row_i.min(row_j));
                cols.push(row_i.max(row_j));
                values.push(value_i * value_j);
            }
        }
    }
    for diag in 0..meq {
        rows.push(diag);
        cols.push(diag);
        values.push(dual_regularization);
    }
    let normal_matrix = CscMatrix::new_from_triplets(meq, meq, rows, cols, values);
    match solve_symmetric_system(
        InteriorPointLinearSolver::SparseQdldl,
        &normal_matrix,
        &rhs,
        dual_regularization,
        None,
        0,
        1.0,
        dual_regularization,
    ) {
        Ok((solution, _, _)) => solution,
        Err(_) => vec![0.0; meq],
    }
}

fn sparse_second_order_correction_step(
    equality_jacobian: &SparseMatrix,
    inequality_jacobian: &SparseMatrix,
    trial_equality_values: &[f64],
    trial_augmented_inequality_values: &[f64],
    candidate_augmented_inequality_multipliers: &[f64],
    constraint_tol: f64,
) -> Option<Vec<f64>> {
    debug_assert_eq!(equality_jacobian.nrows(), trial_equality_values.len());
    debug_assert_eq!(
        inequality_jacobian.nrows(),
        trial_augmented_inequality_values.len()
    );
    debug_assert_eq!(
        candidate_augmented_inequality_multipliers.len(),
        trial_augmented_inequality_values.len()
    );

    let active_inequality_rows = trial_augmented_inequality_values
        .iter()
        .zip(candidate_augmented_inequality_multipliers.iter())
        .enumerate()
        .filter_map(|(row, (&trial_value, &multiplier))| {
            super::should_include_soc_inequality_row(trial_value, multiplier, constraint_tol)
                .then_some(row)
        })
        .collect::<Vec<_>>();
    let active_row_count = trial_equality_values.len() + active_inequality_rows.len();
    if active_row_count == 0 {
        return None;
    }

    let residual_inf = trial_equality_values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        .max(active_inequality_rows.iter().fold(0.0_f64, |acc, &row| {
            acc.max(trial_augmented_inequality_values[row].abs())
        }));
    if residual_inf <= constraint_tol {
        return None;
    }

    let n = equality_jacobian.ncols().max(inequality_jacobian.ncols());
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    let mut rhs = vec![0.0; n];
    for (row, residual) in trial_equality_values.iter().enumerate() {
        let target = -*residual;
        let entries = &equality_jacobian.structure.row_entries[row];
        for (offset, &(col_i, index_i)) in entries.iter().enumerate() {
            let value_i = equality_jacobian.values[index_i];
            rhs[col_i] += target * value_i;
            for &(col_j, index_j) in &entries[offset..] {
                rows.push(col_i.min(col_j));
                cols.push(col_i.max(col_j));
                values.push(value_i * equality_jacobian.values[index_j]);
            }
        }
    }
    for &row in &active_inequality_rows {
        let target = -trial_augmented_inequality_values[row];
        let entries = &inequality_jacobian.structure.row_entries[row];
        for (offset, &(col_i, index_i)) in entries.iter().enumerate() {
            let value_i = inequality_jacobian.values[index_i];
            rhs[col_i] += target * value_i;
            for &(col_j, index_j) in &entries[offset..] {
                rows.push(col_i.min(col_j));
                cols.push(col_i.max(col_j));
                values.push(value_i * inequality_jacobian.values[index_j]);
            }
        }
    }
    let regularization = super::soc_multiplier_tolerance(constraint_tol);
    for diag in 0..n {
        rows.push(diag);
        cols.push(diag);
        values.push(regularization);
    }
    let normal_matrix = CscMatrix::new_from_triplets(n, n, rows, cols, values);
    let (correction, _, _) = solve_symmetric_system(
        InteriorPointLinearSolver::SparseQdldl,
        &normal_matrix,
        &rhs,
        regularization,
        None,
        0,
        1.0,
        regularization,
    )
    .ok()?;
    (inf_norm(&correction) > 0.0 && correction.iter().all(|value| value.is_finite()))
        .then_some(correction)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LinearSolutionAssessment {
    solution_inf: f64,
    solution_inf_limit: f64,
    residual_inf: f64,
    residual_inf_limit: f64,
}

fn symmetric_csc_upper_mat_vec(matrix: &CscMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    debug_assert_eq!(matrix.m, matrix.n);
    debug_assert_eq!(matrix.n, vector.len());
    let mut product = vec![0.0; matrix.n];
    for col in 0..matrix.n {
        let x_col = vector[col];
        for index in matrix.colptr[col]..matrix.colptr[col + 1] {
            let row = matrix.rowval[index];
            let value = matrix.nzval[index];
            product[row] += value * x_col;
            if row != col {
                product[col] += value * vector[row];
            }
        }
    }
    product
}

fn symmetric_csc_upper_abs_mat_vec(matrix: &CscMatrix<f64>, vector_abs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(matrix.m, matrix.n);
    debug_assert_eq!(matrix.n, vector_abs.len());
    let mut product = vec![0.0; matrix.n];
    for col in 0..matrix.n {
        let x_col_abs = vector_abs[col];
        for index in matrix.colptr[col]..matrix.colptr[col + 1] {
            let row = matrix.rowval[index];
            let value_abs = matrix.nzval[index].abs();
            product[row] += value_abs * x_col_abs;
            if row != col {
                product[col] += value_abs * vector_abs[row];
            }
        }
    }
    product
}

fn assess_linear_solution(
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    solution: &[f64],
) -> std::result::Result<LinearSolutionAssessment, InteriorPointLinearSolveAttempt> {
    let rhs_inf = rhs.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let solution_inf_limit = LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM * (1.0 + rhs_inf);
    let residual_inf_limit = LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL * (1.0 + rhs_inf);
    let solution_inf = solution
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if !solution.iter().all(|value| value.is_finite()) {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::Auto,
            regularization: 0.0,
            failure_kind: InteriorPointLinearSolveFailureKind::NonFiniteSolution,
            detail: None,
            solution_inf: Some(solution_inf),
            solution_inf_limit: Some(solution_inf_limit),
            residual_inf: None,
            residual_inf_limit: Some(residual_inf_limit),
        });
    }
    if solution_inf > LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM * (1.0 + rhs_inf) {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::Auto,
            regularization: 0.0,
            failure_kind: InteriorPointLinearSolveFailureKind::SolutionNormTooLarge,
            detail: None,
            solution_inf: Some(solution_inf),
            solution_inf_limit: Some(solution_inf_limit),
            residual_inf: None,
            residual_inf_limit: Some(residual_inf_limit),
        });
    }

    let residual = symmetric_csc_upper_mat_vec(matrix, solution)
        .into_iter()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs_i)| lhs - rhs_i)
        .collect::<Vec<_>>();
    let abs_solution = solution.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let lhs_scale = symmetric_csc_upper_abs_mat_vec(matrix, &abs_solution);
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let backward_error_residual_limit =
        lhs_scale
            .iter()
            .zip(rhs.iter())
            .fold(0.0_f64, |acc, (lhs_scale_i, rhs_i)| {
                acc.max(LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL * (1.0 + lhs_scale_i + rhs_i.abs()))
            });
    let residual_inf_limit = residual_inf_limit.max(backward_error_residual_limit);
    if residual_inf > residual_inf_limit {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::Auto,
            regularization: 0.0,
            failure_kind: InteriorPointLinearSolveFailureKind::ResidualTooLarge,
            detail: None,
            solution_inf: Some(solution_inf),
            solution_inf_limit: Some(solution_inf_limit),
            residual_inf: Some(residual_inf),
            residual_inf_limit: Some(residual_inf_limit),
        });
    }

    Ok(LinearSolutionAssessment {
        solution_inf,
        solution_inf_limit,
        residual_inf,
        residual_inf_limit,
    })
}

fn assess_linear_solution_ccs(
    ccs: &CCS,
    values: &[f64],
    rhs: &[f64],
    solution: &[f64],
) -> std::result::Result<LinearSolutionAssessment, InteriorPointLinearSolveAttempt> {
    let rhs_inf = rhs.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let solution_inf_limit = LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM * (1.0 + rhs_inf);
    let residual_inf_limit = LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL * (1.0 + rhs_inf);
    let solution_inf = solution
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if !solution.iter().all(|value| value.is_finite()) {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::Auto,
            regularization: 0.0,
            failure_kind: InteriorPointLinearSolveFailureKind::NonFiniteSolution,
            detail: None,
            solution_inf: Some(solution_inf),
            solution_inf_limit: Some(solution_inf_limit),
            residual_inf: None,
            residual_inf_limit: Some(residual_inf_limit),
        });
    }
    if solution_inf > solution_inf_limit {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::Auto,
            regularization: 0.0,
            failure_kind: InteriorPointLinearSolveFailureKind::SolutionNormTooLarge,
            detail: None,
            solution_inf: Some(solution_inf),
            solution_inf_limit: Some(solution_inf_limit),
            residual_inf: None,
            residual_inf_limit: Some(residual_inf_limit),
        });
    }

    let residual = symmetric_ccs_lower_mat_vec(ccs, values, solution)
        .into_iter()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs_i)| lhs - rhs_i)
        .collect::<Vec<_>>();
    let abs_solution = solution.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let lhs_scale = symmetric_ccs_lower_abs_mat_vec(ccs, values, &abs_solution);
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let backward_error_residual_limit =
        lhs_scale
            .iter()
            .zip(rhs.iter())
            .fold(0.0_f64, |acc, (lhs_scale_i, rhs_i)| {
                acc.max(LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL * (1.0 + lhs_scale_i + rhs_i.abs()))
            });
    let residual_inf_limit = residual_inf_limit.max(backward_error_residual_limit);
    if residual_inf > residual_inf_limit {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::Auto,
            regularization: 0.0,
            failure_kind: InteriorPointLinearSolveFailureKind::ResidualTooLarge,
            detail: None,
            solution_inf: Some(solution_inf),
            solution_inf_limit: Some(solution_inf_limit),
            residual_inf: Some(residual_inf),
            residual_inf_limit: Some(residual_inf_limit),
        });
    }

    Ok(LinearSolutionAssessment {
        solution_inf,
        solution_inf_limit,
        residual_inf,
        residual_inf_limit,
    })
}

const LINEAR_REFINEMENT_MAX_STEPS: usize = 10;
const LINEAR_REFINEMENT_RELATIVE_RESIDUAL: f64 = 128.0 * f64::EPSILON;

fn refinement_residual_target_ccs(
    ccs: &CCS,
    values: &[f64],
    rhs: &[f64],
    solution: &[f64],
) -> (Vec<f64>, f64, f64) {
    let residual = symmetric_ccs_lower_mat_vec(ccs, values, solution)
        .into_iter()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs_i)| rhs_i - lhs)
        .collect::<Vec<_>>();
    let abs_solution = solution.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let lhs_scale = symmetric_ccs_lower_abs_mat_vec(ccs, values, &abs_solution);
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let target = lhs_scale
        .iter()
        .zip(rhs.iter())
        .fold(0.0_f64, |acc, (lhs_scale_i, rhs_i)| {
            acc.max(LINEAR_REFINEMENT_RELATIVE_RESIDUAL * (1.0 + lhs_scale_i + rhs_i.abs()))
        });
    (residual, residual_inf, target)
}

fn refine_linear_solution_ccs<E>(
    ccs: &CCS,
    values: &[f64],
    rhs: &[f64],
    solution: &mut [f64],
    solve_time: &mut Duration,
    mut solve_correction: impl FnMut(&[f64]) -> Result<Vec<f64>, E>,
) -> Result<usize, E> {
    let mut steps = 0;
    let mut previous_residual_inf = f64::INFINITY;
    loop {
        let (residual, residual_inf, target) =
            refinement_residual_target_ccs(ccs, values, rhs, solution);
        if residual_inf <= target
            || steps >= LINEAR_REFINEMENT_MAX_STEPS
            || residual_inf >= previous_residual_inf * (1.0 - 1e-6)
        {
            break;
        }
        previous_residual_inf = residual_inf;
        let correction_started = Instant::now();
        let correction = solve_correction(&residual)?;
        *solve_time += correction_started.elapsed();
        if correction.iter().all(|value| value.abs() <= f64::EPSILON) {
            break;
        }
        for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
            *solution_i += correction_i;
        }
        steps += 1;
    }
    Ok(steps)
}

fn interior_point_linear_inertia(inertia: SpralInertia) -> InteriorPointLinearInertia {
    InteriorPointLinearInertia {
        positive: inertia.positive,
        negative: inertia.negative,
        zero: inertia.zero,
    }
}

fn delta_inf_norm(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()))
}

fn spral_error_attempt(
    regularization: f64,
    failure_kind: InteriorPointLinearSolveFailureKind,
    error: impl ToString,
) -> InteriorPointLinearSolveAttempt {
    InteriorPointLinearSolveAttempt {
        solver: InteriorPointLinearSolver::SpralSsids,
        regularization,
        failure_kind,
        detail: Some(error.to_string()),
        solution_inf: None,
        solution_inf_limit: None,
        residual_inf: None,
        residual_inf_limit: None,
    }
}

fn native_spral_error_attempt(
    regularization: f64,
    failure_kind: InteriorPointLinearSolveFailureKind,
    error: impl ToString,
) -> InteriorPointLinearSolveAttempt {
    InteriorPointLinearSolveAttempt {
        solver: InteriorPointLinearSolver::NativeSpralSsids,
        regularization,
        failure_kind,
        detail: Some(error.to_string()),
        solution_inf: None,
        solution_inf_limit: None,
        residual_inf: None,
        residual_inf_limit: None,
    }
}

fn default_dsigns(dimension: usize) -> Vec<i8> {
    vec![1_i8; dimension]
}

fn quasidefinite_dsigns(primal_dimension: usize, dual_dimension: usize) -> Vec<i8> {
    let mut dsigns = default_dsigns(primal_dimension + dual_dimension);
    dsigns[primal_dimension..].fill(-1);
    dsigns
}

fn factor_solve_sparse_qdldl_with_metrics(
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
    dsigns: Option<&[i8]>,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats), InteriorPointLinearSolveAttempt> {
    let n = matrix.n;
    if matrix.m != n || rhs.len() != n {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SparseQdldl,
            regularization,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: None,
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        });
    }
    let dsigns = dsigns.map_or_else(|| default_dsigns(n), ToOwned::to_owned);
    if dsigns.len() != n {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SparseQdldl,
            regularization,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: None,
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        });
    }
    let settings = QDLDLSettings {
        amd_dense_scale: 1.5,
        Dsigns: Some(dsigns),
        regularize_enable: true,
        regularize_eps: regularization.max(1e-12),
        regularize_delta: regularization.max(1e-9),
        ..Default::default()
    };
    let factor_started = Instant::now();
    let mut factor = QDLDLFactorisation::new(matrix, Some(settings)).map_err(|_| {
        InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SparseQdldl,
            regularization,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: None,
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        }
    })?;
    let factorization_time = factor_started.elapsed();
    let inertia = InteriorPointLinearInertia {
        positive: factor.positive_inertia(),
        negative: n.saturating_sub(factor.positive_inertia()),
        zero: 0,
    };
    let mut solution = rhs.to_vec();
    let solve_started = Instant::now();
    factor.solve(&mut solution);
    let solve_time = solve_started.elapsed();
    let mut assessment = assess_linear_solution(matrix, rhs, &solution);
    if assessment.is_err() {
        for _ in 0..10 {
            let residual = symmetric_csc_upper_mat_vec(matrix, &solution)
                .into_iter()
                .zip(rhs.iter().copied())
                .map(|(lhs, rhs_i)| rhs_i - lhs)
                .collect::<Vec<_>>();
            if residual.iter().all(|value| value.abs() <= f64::EPSILON) {
                break;
            }
            let mut correction = residual;
            factor.solve(&mut correction);
            for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
                *solution_i += correction_i;
            }
            assessment = assess_linear_solution(matrix, rhs, &solution);
            if assessment.is_ok() {
                break;
            }
        }
    }
    assessment
        .map(|assessment| {
            (
                solution,
                LinearBackendRunStats {
                    solver: InteriorPointLinearSolver::SparseQdldl,
                    factorization_time,
                    solve_time,
                    reused_symbolic: None,
                    inertia: Some(inertia),
                    residual_inf: assessment.residual_inf,
                    solution_inf: assessment.solution_inf,
                    detail: None,
                },
            )
        })
        .map_err(|mut attempt| {
            attempt.solver = InteriorPointLinearSolver::SparseQdldl;
            attempt.regularization = regularization;
            attempt
        })
}

fn preferred_linear_solver(
    solver: InteriorPointLinearSolver,
    equality_count: usize,
    inequality_count: usize,
) -> InteriorPointLinearSolver {
    match solver {
        InteriorPointLinearSolver::Auto if equality_count == 0 && inequality_count == 0 => {
            InteriorPointLinearSolver::SparseQdldl
        }
        InteriorPointLinearSolver::Auto | InteriorPointLinearSolver::SpralSsids => {
            InteriorPointLinearSolver::SpralSsids
        }
        InteriorPointLinearSolver::NativeSpralSsids => InteriorPointLinearSolver::NativeSpralSsids,
        InteriorPointLinearSolver::SparseQdldl => InteriorPointLinearSolver::SparseQdldl,
    }
}

fn secondary_linear_solver(solver: InteriorPointLinearSolver) -> Option<InteriorPointLinearSolver> {
    match solver {
        InteriorPointLinearSolver::Auto => Some(InteriorPointLinearSolver::SparseQdldl),
        InteriorPointLinearSolver::SpralSsids
        | InteriorPointLinearSolver::NativeSpralSsids
        | InteriorPointLinearSolver::SparseQdldl => None,
    }
}

fn linear_solve_error(
    preferred_solver: InteriorPointLinearSolver,
    matrix_dimension: usize,
    attempts: Vec<InteriorPointLinearSolveAttempt>,
) -> InteriorPointSolveError {
    InteriorPointSolveError::LinearSolve {
        solver: preferred_solver,
        context: Box::new(InteriorPointFailureContext {
            final_state: None,
            last_accepted_state: None,
            failed_linear_solve: Some(InteriorPointLinearSolveDiagnostics {
                preferred_solver,
                matrix_dimension,
                attempts,
                debug_report: None,
            }),
            failed_line_search: None,
            failed_direction_diagnostics: None,
            profiling: InteriorPointProfiling::default(),
        }),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Linear-solver retries/regularization are passed explicitly at the solver boundary."
)]
fn try_solve_symmetric_system(
    _solver: InteriorPointLinearSolver,
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
    dsigns: Option<&[i8]>,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_max: f64,
) -> std::result::Result<
    (Vec<f64>, InteriorPointLinearSolver, f64),
    Vec<InteriorPointLinearSolveAttempt>,
> {
    try_solve_symmetric_system_with_metrics(
        InteriorPointLinearSolver::SparseQdldl,
        matrix,
        rhs,
        regularization,
        dsigns,
        adaptive_regularization_retries,
        regularization_growth_factor,
        regularization_max,
    )
    .map(|(solution, _stats, regularization_used)| {
        (
            solution,
            InteriorPointLinearSolver::SparseQdldl,
            regularization_used,
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Linear-solver retries/regularization are passed explicitly at the solver boundary."
)]
fn try_solve_symmetric_system_with_metrics(
    _solver: InteriorPointLinearSolver,
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
    dsigns: Option<&[i8]>,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_max: f64,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats, f64), Vec<InteriorPointLinearSolveAttempt>>
{
    let try_sparse_qdldl = |matrix: &CscMatrix<f64>,
                            rhs: &[f64],
                            retries: Index,
                            growth_factor: f64,
                            regularization_max: f64| {
        let mut attempts = Vec::new();
        let mut current_regularization = regularization.max(1e-12);
        let max_regularization = regularization_max.max(current_regularization);
        for retry_index in 0..=retries {
            match factor_solve_sparse_qdldl_with_metrics(
                matrix,
                rhs,
                current_regularization,
                dsigns,
            ) {
                Ok((solution, stats)) => {
                    return Ok((solution, stats, current_regularization));
                }
                Err(attempt) => attempts.push(attempt),
            }
            if retry_index == retries || current_regularization >= max_regularization {
                break;
            }
            let next_regularization =
                (current_regularization * growth_factor.max(1.0)).min(max_regularization);
            if next_regularization <= current_regularization {
                break;
            }
            current_regularization = next_regularization;
        }
        Err(attempts)
    };
    let sparse_result = try_sparse_qdldl(
        matrix,
        rhs,
        adaptive_regularization_retries,
        regularization_growth_factor,
        regularization_max,
    );
    let attempts = match sparse_result {
        Ok(result) => return Ok(result),
        Err(attempts) => attempts,
    };
    Err(attempts)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Error-wrapping symmetric solve retains the same explicit boundary inputs."
)]
fn solve_symmetric_system(
    solver: InteriorPointLinearSolver,
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
    dsigns: Option<&[i8]>,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_max: f64,
) -> std::result::Result<(Vec<f64>, InteriorPointLinearSolver, f64), InteriorPointSolveError> {
    let preferred_solver = preferred_linear_solver(solver, 0, 0);
    try_solve_symmetric_system(
        solver,
        matrix,
        rhs,
        regularization,
        dsigns,
        adaptive_regularization_retries,
        regularization_growth_factor,
        regularization_max,
    )
    .map_err(|attempts| linear_solve_error(preferred_solver, matrix.n, attempts))
}

fn finalised_interior_point_failure_profiling(
    profiling: &InteriorPointProfiling,
    solve_started: Instant,
) -> InteriorPointProfiling {
    let mut profiling = profiling.clone();
    finalise_interior_point_profiling(&mut profiling, solve_started);
    profiling
}

fn interior_point_failure_context(
    final_state: Option<InteriorPointIterationSnapshot>,
    last_accepted_state: Option<InteriorPointIterationSnapshot>,
    failed_linear_solve: Option<InteriorPointLinearSolveDiagnostics>,
    failed_line_search: Option<InteriorPointLineSearchInfo>,
    failed_direction_diagnostics: Option<InteriorPointDirectionDiagnostics>,
    profiling: &InteriorPointProfiling,
    solve_started: Instant,
) -> Box<InteriorPointFailureContext> {
    Box::new(InteriorPointFailureContext {
        final_state,
        last_accepted_state,
        failed_linear_solve,
        failed_line_search,
        failed_direction_diagnostics,
        profiling: finalised_interior_point_failure_profiling(profiling, solve_started),
    })
}

fn with_interior_point_failure_profiling(
    error: InteriorPointSolveError,
    final_state: Option<InteriorPointIterationSnapshot>,
    last_accepted_state: Option<InteriorPointIterationSnapshot>,
    profiling: &InteriorPointProfiling,
    solve_started: Instant,
) -> InteriorPointSolveError {
    match error {
        InteriorPointSolveError::LinearSolve { solver, context } => {
            InteriorPointSolveError::LinearSolve {
                solver,
                context: interior_point_failure_context(
                    final_state.or(context.final_state),
                    last_accepted_state.or(context.last_accepted_state),
                    context.failed_linear_solve,
                    context.failed_line_search,
                    context.failed_direction_diagnostics,
                    profiling,
                    solve_started,
                ),
            }
        }
        InteriorPointSolveError::LineSearchFailed {
            merit,
            mu,
            step_inf_norm,
            context,
        } => InteriorPointSolveError::LineSearchFailed {
            merit,
            mu,
            step_inf_norm,
            context: interior_point_failure_context(
                final_state.or(context.final_state),
                last_accepted_state.or(context.last_accepted_state),
                context.failed_linear_solve,
                context.failed_line_search,
                context.failed_direction_diagnostics,
                profiling,
                solve_started,
            ),
        },
        InteriorPointSolveError::MaxIterations {
            iterations,
            context,
        } => InteriorPointSolveError::MaxIterations {
            iterations,
            context: interior_point_failure_context(
                final_state.or(context.final_state),
                last_accepted_state.or(context.last_accepted_state),
                context.failed_linear_solve,
                context.failed_line_search,
                context.failed_direction_diagnostics,
                profiling,
                solve_started,
            ),
        },
        InteriorPointSolveError::InvalidInput(message) => {
            InteriorPointSolveError::InvalidInput(message)
        }
    }
}

fn with_linear_debug_report(
    error: InteriorPointSolveError,
    report: InteriorPointLinearDebugReport,
) -> InteriorPointSolveError {
    match error {
        InteriorPointSolveError::LinearSolve {
            solver,
            mut context,
        } => {
            if let Some(diagnostics) = context.failed_linear_solve.as_mut() {
                diagnostics.debug_report = Some(report);
            }
            InteriorPointSolveError::LinearSolve { solver, context }
        }
        other => other,
    }
}

fn append_symmetric_hessian_triplets(
    hessian: &SparseSymmetricMatrix,
    diagonal_shift: f64,
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    for col in 0..hessian.lower_triangle.ncol {
        for index in hessian.lower_triangle.col_ptrs[col]..hessian.lower_triangle.col_ptrs[col + 1]
        {
            let row = hessian.lower_triangle.row_indices[index];
            rows.push(col.min(row));
            cols.push(col.max(row));
            values.push(hessian.values[index]);
        }
    }
    if diagonal_shift != 0.0 {
        for diag in 0..hessian.lower_triangle.ncol {
            rows.push(diag);
            cols.push(diag);
            values.push(diagonal_shift);
        }
    }
}

fn sparse_hessian_diagonal_shift(hessian: &SparseSymmetricMatrix, minimum_shift: f64) -> f64 {
    let n = hessian.lower_triangle.nrow;
    let mut diagonal = vec![0.0; n];
    let mut off_diagonal_abs_sum = vec![0.0; n];
    for col in 0..hessian.lower_triangle.ncol {
        for index in hessian.lower_triangle.col_ptrs[col]..hessian.lower_triangle.col_ptrs[col + 1]
        {
            let row = hessian.lower_triangle.row_indices[index];
            let value = hessian.values[index];
            if row == col {
                diagonal[row] += value;
            } else {
                let abs_value = value.abs();
                off_diagonal_abs_sum[row] += abs_value;
                off_diagonal_abs_sum[col] += abs_value;
            }
        }
    }
    let mut shift = minimum_shift.max(0.0);
    for idx in 0..n {
        shift = shift.max(off_diagonal_abs_sum[idx] - diagonal[idx] + minimum_shift);
    }
    shift
}

fn append_selected_constraint_block_triplets(
    matrix: &SparseMatrix,
    selected_rows: &[usize],
    block_column_offset: usize,
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    for (offset, &selected_row) in selected_rows.iter().enumerate() {
        let dual_col = block_column_offset + offset;
        for &(col, index) in &matrix.structure.row_entries[selected_row] {
            rows.push(col);
            cols.push(dual_col);
            values.push(matrix.values[index]);
        }
    }
}

fn append_quasidefinite_dual_diagonal(
    dual_offset: usize,
    dual_dimension: usize,
    shift: f64,
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    for dual_index in 0..dual_dimension {
        let index = dual_offset + dual_index;
        rows.push(index);
        cols.push(index);
        values.push(-shift);
    }
}

fn fill_spral_augmented_kkt_values(
    pattern: &SpralAugmentedKktPattern,
    values: &mut [f64],
    system: &ReducedKktSystem<'_>,
    primal_shift: f64,
    dual_shift: f64,
    inequality_scalings: &[f64],
) {
    values.fill(0.0);
    for (index, &slot) in pattern.hessian_value_indices.iter().enumerate() {
        values[slot] += system.hessian.values[index];
    }
    for &slot in &pattern.x_diagonal_indices {
        values[slot] += primal_shift;
    }
    for (index, &slot) in pattern.equality_jacobian_value_indices.iter().enumerate() {
        values[slot] += system.equality_jacobian.values[index];
    }
    for (index, &slot) in pattern.inequality_jacobian_value_indices.iter().enumerate() {
        values[slot] += system.inequality_jacobian.values[index];
    }
    for &slot in &pattern.p_diagonal_indices {
        values[slot] += 1.0;
    }
    for &slot in &pattern.lambda_diagonal_indices {
        values[slot] += -dual_shift;
    }
    for &slot in &pattern.z_diagonal_indices {
        values[slot] += -dual_shift;
    }
    for (index, &slot) in pattern.pz_indices.iter().enumerate() {
        values[slot] += inequality_scalings[index];
    }
}

fn assemble_spral_augmented_kkt_values(
    workspace: &mut SpralAugmentedKktWorkspace,
    system: &ReducedKktSystem<'_>,
    primal_shift: f64,
    dual_shift: f64,
    inequality_scalings: &[f64],
) {
    fill_spral_augmented_kkt_values(
        &workspace.pattern,
        &mut workspace.values,
        system,
        primal_shift,
        dual_shift,
        inequality_scalings,
    );
}

fn spral_expected_augmented_inertia(pattern: &SpralAugmentedKktPattern) -> SpralInertia {
    SpralInertia {
        positive: pattern.x_dimension + pattern.inequality_dimension,
        negative: pattern.equality_dimension + pattern.inequality_dimension,
        zero: 0,
    }
}

fn factor_solve_spral_ssids(
    system: &ReducedKktSystem<'_>,
    workspace: &mut SpralAugmentedKktWorkspace,
    rhs: &[f64],
    regularization: f64,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats), InteriorPointLinearSolveAttempt> {
    let matrix = SpralSymmetricCscMatrix::new(
        workspace.pattern.dimension(),
        &workspace.pattern.ccs.col_ptrs,
        &workspace.pattern.ccs.row_indices,
        Some(&workspace.values),
    )
    .map_err(|error| {
        spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    // The assembled augmented KKT already includes the NLP regularization shifts.
    // Reusing that value as the LDL pivot regularization materially changes the
    // factorization path versus native SPRAL; keep the solver's own small pivot
    // floor instead.
    let numeric_options = system_spral_numeric_factor_options(system);

    let needs_new_factor = workspace.factor.is_none()
        || workspace
            .factor_regularization
            .is_none_or(|existing| (existing - regularization).abs() > 1e-18);
    let mut factorization_time = Duration::ZERO;
    let mut solve_time = Duration::ZERO;
    if !needs_new_factor {
        let started = Instant::now();
        profiling.sparse_numeric_refactorizations += 1;
        let refactorization_index = profiling.sparse_numeric_refactorizations;
        if let Some(factor) = workspace.factor.as_mut() {
            let refactor_heartbeat =
                verbose.then(|| spawn_spral_stage_heartbeat("[NLIP][SPRAL] Refactorization"));
            match factor.refactorize(matrix) {
                Ok(_) => {
                    if let Some((finished, handle)) = refactor_heartbeat {
                        finished.store(true, Ordering::Relaxed);
                        handle.thread().unpark();
                        let _ = handle.join();
                    }
                    let elapsed = started.elapsed();
                    factorization_time += elapsed;
                    profiling.sparse_numeric_refactorization_time += elapsed;
                    if verbose
                        && (refactorization_index <= 3 || refactorization_index.is_multiple_of(25))
                    {
                        println!(
                            "[NLIP][SPRAL] Refactorization #{} completed in {} (reg={:.3e})",
                            refactorization_index,
                            compact_duration_text(elapsed.as_secs_f64()),
                            regularization,
                        );
                    }
                }
                Err(error) => {
                    if let Some((finished, handle)) = refactor_heartbeat {
                        finished.store(true, Ordering::Relaxed);
                        handle.thread().unpark();
                        let _ = handle.join();
                    }
                    profiling.sparse_numeric_refactorization_time += started.elapsed();
                    workspace.factor = None;
                    workspace.factor_regularization = None;
                    return Err(spral_error_attempt(
                        regularization,
                        InteriorPointLinearSolveFailureKind::FactorizationFailed,
                        error,
                    ));
                }
            }
        }
    }
    if workspace.factor.is_none() {
        let started = Instant::now();
        profiling.sparse_numeric_factorizations += 1;
        if verbose {
            println!(
                "[NLIP][SPRAL] Starting numeric factorization: dim={} nnz={} reg={:.3e}",
                workspace.pattern.dimension(),
                workspace.pattern.ccs.nnz(),
                regularization,
            );
        }
        let factorization_heartbeat =
            verbose.then(|| spawn_spral_stage_heartbeat("[NLIP][SPRAL] Numeric factorization"));
        match spral_factorize(matrix, &workspace.symbolic, &numeric_options) {
            Ok((factor, _)) => {
                if let Some((finished, handle)) = factorization_heartbeat {
                    finished.store(true, Ordering::Relaxed);
                    handle.thread().unpark();
                    let _ = handle.join();
                }
                let elapsed = started.elapsed();
                factorization_time += elapsed;
                profiling.sparse_numeric_factorization_time += elapsed;
                if verbose {
                    println!(
                        "[NLIP][SPRAL] Numeric factorization completed in {}: stored_nnz={} factor_bytes={}",
                        compact_duration_text(elapsed.as_secs_f64()),
                        factor.stored_nnz(),
                        factor.factor_bytes(),
                    );
                }
                workspace.factor = Some(factor);
                workspace.factor_regularization = Some(regularization);
            }
            Err(error) => {
                if let Some((finished, handle)) = factorization_heartbeat {
                    finished.store(true, Ordering::Relaxed);
                    handle.thread().unpark();
                    let _ = handle.join();
                }
                profiling.sparse_numeric_factorization_time += started.elapsed();
                return Err(spral_error_attempt(
                    regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    error,
                ));
            }
        }
    }

    let reused_symbolic = !needs_new_factor;
    let factor = workspace.factor.as_mut().expect("factorization must exist");
    let expected_inertia = spral_expected_augmented_inertia(&workspace.pattern);
    let actual_inertia = factor.inertia();
    if actual_inertia != expected_inertia {
        return Err(spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::InertiaMismatch,
            format!(
                "expected inertia (+{}, -{}, 0={}), got (+{}, -{}, 0={})",
                expected_inertia.positive,
                expected_inertia.negative,
                expected_inertia.zero,
                actual_inertia.positive,
                actual_inertia.negative,
                actual_inertia.zero
            ),
        ));
    }

    let solve_started = Instant::now();
    let mut solution = factor.solve(rhs).map_err(|error| {
        spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    solve_time += solve_started.elapsed();
    let refinement_steps = refine_linear_solution_ccs(
        &workspace.pattern.ccs,
        &workspace.values,
        rhs,
        &mut solution,
        &mut solve_time,
        |residual| {
            factor.solve(residual).map_err(|error| {
                spral_error_attempt(
                    regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    error,
                )
            })
        },
    )?;
    let assessment =
        assess_linear_solution_ccs(&workspace.pattern.ccs, &workspace.values, rhs, &solution);
    assessment
        .map(|assessment| {
            (
                solution,
                LinearBackendRunStats {
                    solver: InteriorPointLinearSolver::SpralSsids,
                    factorization_time,
                    solve_time,
                    reused_symbolic: Some(reused_symbolic),
                    inertia: Some(interior_point_linear_inertia(actual_inertia)),
                    residual_inf: assessment.residual_inf,
                    solution_inf: assessment.solution_inf,
                    detail: (refinement_steps > 0)
                        .then(|| format!("iterative_refinement_steps={refinement_steps}")),
                },
            )
        })
        .map_err(|mut attempt| {
            attempt.solver = InteriorPointLinearSolver::SpralSsids;
            attempt.regularization = regularization;
            attempt
        })
}

fn factor_solve_native_spral_ssids(
    workspace: &mut NativeSpralAugmentedKktWorkspace,
    rhs: &[f64],
    regularization: f64,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats), InteriorPointLinearSolveAttempt> {
    let matrix = SpralSymmetricCscMatrix::new(
        workspace.pattern.dimension(),
        &workspace.pattern.ccs.col_ptrs,
        &workspace.pattern.ccs.row_indices,
        Some(&workspace.values),
    )
    .map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;

    let mut factorization_time = Duration::ZERO;
    let reused_symbolic = workspace.factor_regularization.is_some();
    let factor_started = Instant::now();
    let factor_info = if reused_symbolic {
        profiling.sparse_numeric_refactorizations += 1;
        workspace.session.refactorize(matrix).map_err(|error| {
            native_spral_error_attempt(
                regularization,
                InteriorPointLinearSolveFailureKind::FactorizationFailed,
                error,
            )
        })?
    } else {
        profiling.sparse_numeric_factorizations += 1;
        if verbose {
            println!(
                "[NLIP][Native-SPRAL] Starting numeric factorization: dim={} nnz={} reg={:.3e}",
                workspace.pattern.dimension(),
                workspace.pattern.ccs.nnz(),
                regularization,
            );
        }
        workspace.session.factorize(matrix).map_err(|error| {
            native_spral_error_attempt(
                regularization,
                InteriorPointLinearSolveFailureKind::FactorizationFailed,
                error,
            )
        })?
    };
    factorization_time += factor_started.elapsed();
    if reused_symbolic {
        profiling.sparse_numeric_refactorization_time += factorization_time;
    } else {
        profiling.sparse_numeric_factorization_time += factorization_time;
    }
    workspace.factor_regularization = Some(regularization);

    let expected_inertia = spral_expected_augmented_inertia(&workspace.pattern);
    if factor_info.inertia != expected_inertia {
        return Err(native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::InertiaMismatch,
            format!(
                "expected inertia (+{}, -{}, 0={}), got (+{}, -{}, 0={})",
                expected_inertia.positive,
                expected_inertia.negative,
                expected_inertia.zero,
                factor_info.inertia.positive,
                factor_info.inertia.negative,
                factor_info.inertia.zero
            ),
        ));
    }

    let solve_started = Instant::now();
    let mut solution = workspace.session.solve(rhs).map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let mut solve_time = solve_started.elapsed();
    let refinement_steps = refine_linear_solution_ccs(
        &workspace.pattern.ccs,
        &workspace.values,
        rhs,
        &mut solution,
        &mut solve_time,
        |residual| {
            workspace.session.solve(residual).map_err(|error| {
                native_spral_error_attempt(
                    regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    error,
                )
            })
        },
    )?;
    let assessment =
        assess_linear_solution_ccs(&workspace.pattern.ccs, &workspace.values, rhs, &solution);

    assessment
        .map(|assessment| {
            (
                solution,
                LinearBackendRunStats {
                    solver: InteriorPointLinearSolver::NativeSpralSsids,
                    factorization_time,
                    solve_time,
                    reused_symbolic: Some(reused_symbolic),
                    inertia: Some(interior_point_linear_inertia(factor_info.inertia)),
                    residual_inf: assessment.residual_inf,
                    solution_inf: assessment.solution_inf,
                    detail: (refinement_steps > 0)
                        .then(|| format!("iterative_refinement_steps={refinement_steps}")),
                },
            )
        })
        .map_err(|mut attempt| {
            attempt.solver = InteriorPointLinearSolver::NativeSpralSsids;
            attempt.regularization = regularization;
            attempt
        })
}

fn solve_reduced_kkt_with_native_spral_ssids(
    system: &ReducedKktSystem<'_>,
    workspace: &mut NativeSpralAugmentedKktWorkspace,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<NewtonDirection, Vec<InteriorPointLinearSolveAttempt>> {
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let inequality_scalings = system
        .slack
        .iter()
        .zip(system.multipliers.iter())
        .map(|(slack_i, multiplier_i)| (slack_i.max(1e-16) / multiplier_i.max(1e-16)).sqrt())
        .collect::<Vec<_>>();
    let rhs_p = system
        .r_cent
        .iter()
        .zip(system.slack.iter())
        .zip(system.multipliers.iter())
        .map(|((r_cent_i, slack_i), multiplier_i)| {
            let scaling = (slack_i.max(1e-16) * multiplier_i.max(1e-16)).sqrt();
            -r_cent_i / scaling
        })
        .collect::<Vec<_>>();
    let total_dimension = workspace.pattern.dimension();
    let mut rhs = vec![0.0; total_dimension];
    rhs[..n]
        .iter_mut()
        .zip(system.r_dual.iter())
        .for_each(|(rhs_i, r_i)| *rhs_i = -*r_i);
    rhs[workspace.pattern.p_offset..workspace.pattern.p_offset + mineq].copy_from_slice(&rhs_p);
    for row in 0..meq {
        rhs[workspace.pattern.lambda_offset + row] = -system.r_eq[row];
    }
    for row in 0..mineq {
        rhs[workspace.pattern.z_offset + row] = -system.r_ineq[row];
    }

    let mut attempts = Vec::new();
    let mut current_regularization = system.regularization.max(1e-12);
    let max_regularization = system.regularization_max.max(current_regularization);
    for retry_index in 0..=system.adaptive_regularization_retries {
        let primal_shift = sparse_hessian_diagonal_shift(system.hessian, current_regularization);
        let dual_shift = current_regularization.max(1e-8);
        fill_spral_augmented_kkt_values(
            &workspace.pattern,
            &mut workspace.values,
            system,
            primal_shift,
            dual_shift,
            &inequality_scalings,
        );
        match factor_solve_native_spral_ssids(
            workspace,
            &rhs,
            current_regularization,
            profiling,
            verbose,
        ) {
            Ok((solution, backend_stats)) => {
                let dx = solution[..n].to_vec();
                let p = solution[workspace.pattern.p_offset..workspace.pattern.p_offset + mineq]
                    .to_vec();
                let d_lambda = solution
                    [workspace.pattern.lambda_offset..workspace.pattern.lambda_offset + meq]
                    .to_vec();
                let dz = solution[workspace.pattern.z_offset..workspace.pattern.z_offset + mineq]
                    .to_vec();
                let ds = p
                    .iter()
                    .zip(inequality_scalings.iter())
                    .map(|(p_i, scaling_i)| p_i * scaling_i)
                    .collect::<Vec<_>>();
                return Ok(NewtonDirection {
                    dx,
                    d_lambda,
                    ds,
                    dz,
                    solver_used: InteriorPointLinearSolver::NativeSpralSsids,
                    regularization_used: current_regularization.max(primal_shift),
                    dual_regularization_used: current_regularization,
                    primal_diagonal_shift_used: primal_shift,
                    linear_solution: solution,
                    backend_stats,
                    linear_debug: None,
                });
            }
            Err(attempt) => attempts.push(attempt),
        }
        if retry_index == system.adaptive_regularization_retries
            || current_regularization >= max_regularization
        {
            break;
        }
        let next_regularization = (current_regularization
            * system.regularization_growth_factor.max(1.0))
        .min(max_regularization);
        if next_regularization <= current_regularization {
            break;
        }
        current_regularization = next_regularization;
    }
    Err(attempts)
}

fn solve_reduced_kkt_with_spral_ssids(
    system: &ReducedKktSystem<'_>,
    workspace: &mut SpralAugmentedKktWorkspace,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<NewtonDirection, Vec<InteriorPointLinearSolveAttempt>> {
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let inequality_scalings = system
        .slack
        .iter()
        .zip(system.multipliers.iter())
        .map(|(slack_i, multiplier_i)| (slack_i.max(1e-16) / multiplier_i.max(1e-16)).sqrt())
        .collect::<Vec<_>>();
    let rhs_p = system
        .r_cent
        .iter()
        .zip(system.slack.iter())
        .zip(system.multipliers.iter())
        .map(|((r_cent_i, slack_i), multiplier_i)| {
            let scaling = (slack_i.max(1e-16) * multiplier_i.max(1e-16)).sqrt();
            -r_cent_i / scaling
        })
        .collect::<Vec<_>>();
    let total_dimension = workspace.pattern.dimension();
    let mut rhs = vec![0.0; total_dimension];
    rhs[..n]
        .iter_mut()
        .zip(system.r_dual.iter())
        .for_each(|(rhs_i, r_i)| *rhs_i = -*r_i);
    rhs[workspace.pattern.p_offset..workspace.pattern.p_offset + mineq].copy_from_slice(&rhs_p);
    for row in 0..meq {
        rhs[workspace.pattern.lambda_offset + row] = -system.r_eq[row];
    }
    for row in 0..mineq {
        rhs[workspace.pattern.z_offset + row] = -system.r_ineq[row];
    }

    let mut attempts = Vec::new();
    let mut current_regularization = system.regularization.max(1e-12);
    let max_regularization = system.regularization_max.max(current_regularization);
    for retry_index in 0..=system.adaptive_regularization_retries {
        let primal_shift = sparse_hessian_diagonal_shift(system.hessian, current_regularization);
        let dual_shift = current_regularization.max(1e-8);
        assemble_spral_augmented_kkt_values(
            workspace,
            system,
            primal_shift,
            dual_shift,
            &inequality_scalings,
        );
        match factor_solve_spral_ssids(
            system,
            workspace,
            &rhs,
            current_regularization,
            profiling,
            verbose,
        ) {
            Ok((solution, backend_stats)) => {
                let dx = solution[..n].to_vec();
                let p = solution[workspace.pattern.p_offset..workspace.pattern.p_offset + mineq]
                    .to_vec();
                let d_lambda = solution
                    [workspace.pattern.lambda_offset..workspace.pattern.lambda_offset + meq]
                    .to_vec();
                let dz = solution[workspace.pattern.z_offset..workspace.pattern.z_offset + mineq]
                    .to_vec();
                let ds = p
                    .iter()
                    .zip(inequality_scalings.iter())
                    .map(|(p_i, scaling_i)| p_i * scaling_i)
                    .collect::<Vec<_>>();
                return Ok(NewtonDirection {
                    dx,
                    d_lambda,
                    ds,
                    dz,
                    solver_used: InteriorPointLinearSolver::SpralSsids,
                    regularization_used: current_regularization.max(primal_shift),
                    dual_regularization_used: current_regularization,
                    primal_diagonal_shift_used: primal_shift,
                    linear_solution: solution,
                    backend_stats,
                    linear_debug: None,
                });
            }
            Err(attempt) => {
                attempts.push(attempt);
                workspace.factor = None;
                workspace.factor_regularization = None;
            }
        }
        if retry_index == system.adaptive_regularization_retries
            || current_regularization >= max_regularization
        {
            break;
        }
        let next_regularization = (current_regularization
            * system.regularization_growth_factor.max(1.0))
        .min(max_regularization);
        if next_regularization <= current_regularization {
            break;
        }
        current_regularization = next_regularization;
    }
    Err(attempts)
}

fn solve_reduced_kkt_with_sparse_qdldl(
    system: &ReducedKktSystem<'_>,
) -> std::result::Result<NewtonDirection, Vec<InteriorPointLinearSolveAttempt>> {
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    let hessian_shift = if meq == 0 && mineq == 0 {
        system.regularization
    } else {
        sparse_hessian_diagonal_shift(system.hessian, system.regularization)
    };
    let fallback_hessian_shift =
        sparse_hessian_diagonal_shift(system.hessian, system.regularization);
    append_symmetric_hessian_triplets(
        system.hessian,
        hessian_shift,
        &mut rows,
        &mut cols,
        &mut values,
    );
    let mut rhs_top = system.r_dual.iter().map(|value| -value).collect::<Vec<_>>();

    if mineq > 0 {
        for row in 0..mineq {
            let scale = system.multipliers[row] / system.slack[row];
            let row_entries = &system.inequality_jacobian.structure.row_entries[row];
            for (offset, &(col_i, index_i)) in row_entries.iter().enumerate() {
                let value_i = system.inequality_jacobian.values[index_i];
                for &(col_j, index_j) in &row_entries[offset..] {
                    rows.push(col_i.min(col_j));
                    cols.push(col_i.max(col_j));
                    values.push(scale * value_i * system.inequality_jacobian.values[index_j]);
                }
            }
        }
        let sz_term = system
            .r_cent
            .iter()
            .zip(system.multipliers.iter())
            .zip(system.r_ineq.iter())
            .zip(system.slack.iter())
            .map(|(((r_cent_i, z_i), r_ineq_i), s_i)| (r_cent_i - z_i * r_ineq_i) / s_i)
            .collect::<Vec<_>>();
        sparse_add_transpose_mat_vec(&mut rhs_top, system.inequality_jacobian, &sz_term);
    }

    if meq == 0 {
        let matrix = CscMatrix::new_from_triplets(n, n, rows, cols, values);
        let (solution, backend_stats, regularization_used) =
            match try_solve_symmetric_system_with_metrics(
                InteriorPointLinearSolver::SparseQdldl,
                &matrix,
                &rhs_top,
                system.regularization,
                None,
                system.adaptive_regularization_retries,
                system.regularization_growth_factor,
                system.regularization_max,
            ) {
                Ok(result) => result,
                Err(primary_attempts) if fallback_hessian_shift > hessian_shift * (1.0 + 1e-12) => {
                    let mut fallback_rows = Vec::new();
                    let mut fallback_cols = Vec::new();
                    let mut fallback_values = Vec::new();
                    append_symmetric_hessian_triplets(
                        system.hessian,
                        fallback_hessian_shift,
                        &mut fallback_rows,
                        &mut fallback_cols,
                        &mut fallback_values,
                    );
                    let fallback_matrix = CscMatrix::new_from_triplets(
                        n,
                        n,
                        fallback_rows,
                        fallback_cols,
                        fallback_values,
                    );
                    match try_solve_symmetric_system_with_metrics(
                        InteriorPointLinearSolver::SparseQdldl,
                        &fallback_matrix,
                        &rhs_top,
                        system.regularization,
                        None,
                        system.adaptive_regularization_retries,
                        system.regularization_growth_factor,
                        system.regularization_max,
                    ) {
                        Ok(result) => result,
                        Err(mut fallback_attempts) => {
                            let mut attempts = primary_attempts;
                            attempts.append(&mut fallback_attempts);
                            return Err(attempts);
                        }
                    }
                }
                Err(attempts) => return Err(attempts),
            };
        let dx = solution[..n].to_vec();
        let (ds, dz) = if mineq > 0 {
            let jacobian_dx = sparse_mat_vec(system.inequality_jacobian, &dx);
            let ds = jacobian_dx
                .iter()
                .zip(system.r_ineq.iter())
                .map(|(gdx_i, r_ineq_i)| -r_ineq_i - gdx_i)
                .collect::<Vec<_>>();
            let dz = ds
                .iter()
                .zip(system.r_cent.iter())
                .zip(system.multipliers.iter())
                .zip(system.slack.iter())
                .map(|(((ds_i, r_cent_i), z_i), s_i)| (-r_cent_i - z_i * ds_i) / s_i.max(1e-16))
                .collect::<Vec<_>>();
            (ds, dz)
        } else {
            (Vec::new(), Vec::new())
        };
        return Ok(NewtonDirection {
            dx,
            d_lambda: Vec::new(),
            ds,
            dz,
            solver_used: InteriorPointLinearSolver::SparseQdldl,
            regularization_used,
            dual_regularization_used: regularization_used,
            primal_diagonal_shift_used: regularization_used,
            linear_solution: solution,
            backend_stats,
            linear_debug: None,
        });
    }

    let mut rhs = vec![0.0; n + meq];
    rhs[..n].copy_from_slice(&rhs_top);
    for row in 0..meq {
        rhs[n + row] = -system.r_eq[row];
    }
    let dsigns = quasidefinite_dsigns(n, meq);
    let equality_rows = (0..meq).collect::<Vec<_>>();
    let base_rows = rows;
    let base_cols = cols;
    let base_values = values;
    let mut attempts = Vec::new();
    let mut current_regularization = system.regularization.max(1e-12);
    let max_regularization = system.regularization_max.max(current_regularization);
    for retry_index in 0..=system.adaptive_regularization_retries {
        let mut attempt_rows = base_rows.clone();
        let mut attempt_cols = base_cols.clone();
        let mut attempt_values = base_values.clone();
        let primal_shift = sparse_hessian_diagonal_shift(system.hessian, current_regularization);
        if primal_shift > hessian_shift * (1.0 + 1e-12) {
            let additional_shift = primal_shift - hessian_shift;
            for diag in 0..n {
                attempt_rows.push(diag);
                attempt_cols.push(diag);
                attempt_values.push(additional_shift);
            }
        }
        append_selected_constraint_block_triplets(
            system.equality_jacobian,
            &equality_rows,
            n,
            &mut attempt_rows,
            &mut attempt_cols,
            &mut attempt_values,
        );
        append_quasidefinite_dual_diagonal(
            n,
            meq,
            current_regularization.max(1e-8),
            &mut attempt_rows,
            &mut attempt_cols,
            &mut attempt_values,
        );
        let matrix = CscMatrix::new_from_triplets(
            n + meq,
            n + meq,
            attempt_rows,
            attempt_cols,
            attempt_values,
        );
        match factor_solve_sparse_qdldl_with_metrics(
            &matrix,
            &rhs,
            current_regularization,
            Some(&dsigns),
        ) {
            Ok((solution, backend_stats)) => {
                let dx = solution[..n].to_vec();
                let d_lambda = solution[n..n + meq].to_vec();
                let (ds, dz) = if mineq > 0 {
                    let jacobian_dx = sparse_mat_vec(system.inequality_jacobian, &dx);
                    let ds = jacobian_dx
                        .iter()
                        .zip(system.r_ineq.iter())
                        .map(|(gdx_i, r_ineq_i)| -r_ineq_i - gdx_i)
                        .collect::<Vec<_>>();
                    let dz = ds
                        .iter()
                        .zip(system.r_cent.iter())
                        .zip(system.multipliers.iter())
                        .zip(system.slack.iter())
                        .map(|(((ds_i, r_cent_i), z_i), s_i)| {
                            (-r_cent_i - z_i * ds_i) / s_i.max(1e-16)
                        })
                        .collect::<Vec<_>>();
                    (ds, dz)
                } else {
                    (Vec::new(), Vec::new())
                };
                return Ok(NewtonDirection {
                    dx,
                    d_lambda,
                    ds,
                    dz,
                    solver_used: InteriorPointLinearSolver::SparseQdldl,
                    regularization_used: current_regularization.max(primal_shift),
                    dual_regularization_used: current_regularization,
                    primal_diagonal_shift_used: primal_shift,
                    linear_solution: solution,
                    backend_stats,
                    linear_debug: None,
                });
            }
            Err(attempt) => attempts.push(attempt),
        }
        if retry_index == system.adaptive_regularization_retries
            || current_regularization >= max_regularization
        {
            break;
        }
        let next_regularization = (current_regularization
            * system.regularization_growth_factor.max(1.0))
        .min(max_regularization);
        if next_regularization <= current_regularization {
            break;
        }
        current_regularization = next_regularization;
    }
    Err(attempts)
}

fn solve_reduced_kkt(
    system: &ReducedKktSystem<'_>,
    spral_workspace: Option<&mut SpralAugmentedKktWorkspace>,
    native_spral_workspace: Option<&mut NativeSpralAugmentedKktWorkspace>,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<NewtonDirection, InteriorPointSolveError> {
    let preferred_solver = preferred_linear_solver(
        system.solver,
        system.equality_jacobian.nrows(),
        system.inequality_jacobian.nrows(),
    );
    let spral_matrix_dimension = spral_workspace
        .as_ref()
        .map(|workspace| workspace.pattern.dimension());
    let mut attempts = Vec::new();

    if preferred_solver == InteriorPointLinearSolver::NativeSpralSsids
        && let Some(workspace) = native_spral_workspace
    {
        return solve_reduced_kkt_with_native_spral_ssids(system, workspace, profiling, verbose)
            .map_err(|attempts| {
                linear_solve_error(preferred_solver, workspace.pattern.dimension(), attempts)
            });
    }

    if preferred_solver == InteriorPointLinearSolver::SpralSsids
        && let Some(workspace) = spral_workspace
        && !workspace.auto_fallback_disabled
    {
        match solve_reduced_kkt_with_spral_ssids(system, workspace, profiling, verbose) {
            Ok(direction) => return Ok(direction),
            Err(mut spral_attempts) => {
                attempts.append(&mut spral_attempts);
                if secondary_linear_solver(system.solver).is_some() {
                    workspace.auto_fallback_disabled = true;
                } else {
                    return Err(linear_solve_error(
                        preferred_solver,
                        workspace.pattern.dimension(),
                        attempts,
                    ));
                }
            }
        }
    }

    match solve_reduced_kkt_with_sparse_qdldl(system) {
        Ok(direction) => Ok(direction),
        Err(mut qdldl_attempts) => {
            attempts.append(&mut qdldl_attempts);
            let matrix_dimension = if let Some(matrix_dimension) = spral_matrix_dimension {
                matrix_dimension
            } else if system.equality_jacobian.nrows() > 0 {
                system.hessian.lower_triangle.nrow + system.equality_jacobian.nrows()
            } else {
                system.hessian.lower_triangle.nrow
            };
            Err(linear_solve_error(
                preferred_solver,
                matrix_dimension,
                attempts,
            ))
        }
    }
}

fn style_ip_metric(
    text: &str,
    value: f64,
    metric: InteriorPointResidualMetric,
    mode: InteriorPointDisplayMode,
) -> String {
    let (strict_tolerance, acceptable_tolerance) = mode.tolerances(metric);
    if value <= strict_tolerance {
        return style_green_bold(text);
    }
    if let Some(acceptable_tolerance) = acceptable_tolerance {
        if value <= acceptable_tolerance {
            return style_yellow_bold(text);
        }
        return style_red_bold(text);
    }
    style_metric_against_tolerance(text, value, strict_tolerance)
}

fn style_ip_residual_text(
    value: f64,
    metric: InteriorPointResidualMetric,
    mode: InteriorPointDisplayMode,
    applicable: bool,
) -> String {
    if !applicable {
        return "--".to_string();
    }
    let text = sci_text(value);
    style_ip_metric(&text, value, metric, mode)
}

fn style_ip_residual_cell(
    value: f64,
    metric: InteriorPointResidualMetric,
    mode: InteriorPointDisplayMode,
    applicable: bool,
) -> String {
    if !applicable {
        return format!("{:>9}", "--");
    }
    let cell = format!("{:>9}", sci_text(value));
    style_ip_metric(&cell, value, metric, mode)
}

fn fmt_optional_ip_sci(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{:>9}", sci_text(value)),
        None => format!("{:>9}", "--"),
    }
}

fn fmt_optional_index(value: Option<Index>) -> String {
    match value {
        Some(value) => format!("{value:>5}"),
        None => format!("{:>5}", "--"),
    }
}

fn style_ip_line_search_cell(iterations: Option<Index>) -> String {
    let cell = fmt_optional_index(iterations);
    match iterations {
        Some(iterations) if iterations >= 10 => style_red_bold(&cell),
        Some(iterations) if iterations >= 4 => style_yellow_bold(&cell),
        _ => cell,
    }
}

struct InteriorPointIterationLog {
    iteration: Index,
    phase: InteriorPointIterationPhase,
    flags: InteriorPointIterationLogFlags,
    extra_events: Vec<InteriorPointIterationEvent>,
    display_mode: InteriorPointDisplayMode,
    objective_value: f64,
    barrier_objective: f64,
    equality_inf: f64,
    inequality_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    overall_inf: f64,
    barrier_parameter: f64,
    alpha: Option<f64>,
    alpha_pr: Option<f64>,
    alpha_du: Option<f64>,
    line_search_iterations: Option<Index>,
    regularization_size: Option<f64>,
    step_kind: Option<InteriorPointStepKind>,
    step_tag: Option<char>,
    linear_time_secs: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct InteriorPointIterationLogFlags {
    has_equalities: bool,
    has_inequalities: bool,
    filter_accepted: bool,
    soc_attempted: bool,
    soc_used: bool,
    watchdog_active: bool,
    tiny_step: bool,
    iteration_limit_reached: bool,
}

fn nlip_log_snapshot(log: &InteriorPointIterationLog) -> InteriorPointIterationSnapshot {
    let mut snapshot = InteriorPointIterationSnapshot {
        iteration: log.iteration,
        phase: log.phase,
        x: Vec::new(),
        objective: log.objective_value,
        barrier_objective: Some(log.barrier_objective),
        eq_inf: Some(log.equality_inf),
        ineq_inf: Some(log.inequality_inf),
        dual_inf: log.dual_inf,
        comp_inf: Some(log.complementarity_inf),
        overall_inf: log.overall_inf,
        barrier_parameter: Some(log.barrier_parameter),
        filter_theta: Some(log.equality_inf.max(log.inequality_inf)),
        step_inf: None,
        alpha: log.alpha,
        alpha_pr: log.alpha_pr,
        alpha_du: log.alpha_du,
        line_search_iterations: log.line_search_iterations,
        line_search_trials: log.line_search_iterations.unwrap_or(0),
        regularization_size: log.regularization_size,
        step_kind: log.step_kind,
        step_tag: log.step_tag,
        watchdog_active: log.flags.watchdog_active,
        line_search: None,
        direction_diagnostics: None,
        linear_debug: None,
        linear_solver: InteriorPointLinearSolver::Auto,
        linear_solve_time: log.linear_time_secs.map(Duration::from_secs_f64),
        filter: None,
        events: log.extra_events.clone(),
        timing: InteriorPointIterationTiming::default(),
    };
    if matches!(log.line_search_iterations, Some(iterations) if iterations >= 4) {
        push_unique_nlip_event(
            &mut snapshot.events,
            InteriorPointIterationEvent::LongLineSearch,
        );
    }
    if log.flags.filter_accepted {
        push_unique_nlip_event(
            &mut snapshot.events,
            InteriorPointIterationEvent::FilterAccepted,
        );
    }
    if log.flags.soc_attempted {
        push_unique_nlip_event(
            &mut snapshot.events,
            InteriorPointIterationEvent::SecondOrderCorrectionAttempted,
        );
    }
    if log.flags.soc_used {
        push_unique_nlip_event(
            &mut snapshot.events,
            InteriorPointIterationEvent::SecondOrderCorrectionAccepted,
        );
    }
    if log.flags.watchdog_active {
        push_unique_nlip_event(
            &mut snapshot.events,
            InteriorPointIterationEvent::WatchdogActivated,
        );
    }
    if log.flags.tiny_step {
        push_unique_nlip_event(&mut snapshot.events, InteriorPointIterationEvent::TinyStep);
    }
    if log.flags.iteration_limit_reached {
        push_unique_nlip_event(
            &mut snapshot.events,
            InteriorPointIterationEvent::MaxIterationsReached,
        );
    }
    snapshot
}

fn fmt_ip_event_codes(log: &InteriorPointIterationLog) -> String {
    nlip_event_slot_codes(&nlip_log_snapshot(log))
}

fn style_ip_event_cell(log: &InteriorPointIterationLog) -> String {
    let codes = fmt_ip_event_codes(log);
    if !codes.chars().any(|code| code != ' ') {
        return codes;
    }
    if log.flags.iteration_limit_reached {
        style_red_bold(&codes)
    } else {
        style_yellow_bold(&codes)
    }
}

fn fmt_ip_event_header() -> String {
    format!("{:^width$}", "evt", width = NLIP_EVENT_CELL_WIDTH)
}

fn fmt_ip_event_prefix_cell() -> String {
    format!("{:width$}", "", width = NLIP_EVENT_CELL_WIDTH)
}

fn ip_event_legend_prefix() -> String {
    [format!("{:>4}", ""), fmt_ip_event_prefix_cell()].join("  ")
}

fn fmt_ip_event_header_row() -> String {
    [
        format!("{:>4}", "iter"),
        fmt_ip_event_header(),
        format!("{:>9}", "f"),
        format!("{:>9}", EQ_INF_LABEL),
        format!("{:>9}", INEQ_INF_LABEL),
        format!("{:>9}", DUAL_INF_LABEL),
        format!("{:>9}", IP_COMP_INF_LABEL),
        format!("{:>9}", OVERALL_INF_LABEL),
        format!("{:>9}", "mu"),
        format!("{:>9}", "α"),
        format!("{:>5}", "ls_it"),
        format!("{:>7}", "lin_t"),
    ]
    .join("  ")
}

fn log_interior_point_iteration(
    log: &InteriorPointIterationLog,
    event_state: &mut SqpEventLegendState,
) {
    if log.iteration.is_multiple_of(10) {
        eprintln!();
        eprintln!("{}", style_bold(&fmt_ip_event_header_row()));
    }
    for legend_line in ip_event_legend_lines(log, event_state) {
        eprintln!("{legend_line}");
    }
    let iteration_label = match log.phase {
        InteriorPointIterationPhase::Initial => "pre".to_string(),
        InteriorPointIterationPhase::AcceptedStep => log.iteration.to_string(),
        InteriorPointIterationPhase::Converged => "post".to_string(),
    };
    let row = [
        style_iteration_label_cell(&iteration_label, log.flags.iteration_limit_reached),
        style_ip_event_cell(log),
        format!("{:>9}", sci_text(log.objective_value)),
        style_ip_residual_cell(
            log.equality_inf,
            InteriorPointResidualMetric::Constraint,
            log.display_mode,
            log.flags.has_equalities,
        ),
        style_ip_residual_cell(
            log.inequality_inf,
            InteriorPointResidualMetric::Constraint,
            log.display_mode,
            log.flags.has_inequalities,
        ),
        style_ip_residual_cell(
            log.dual_inf,
            InteriorPointResidualMetric::Dual,
            log.display_mode,
            true,
        ),
        style_ip_residual_cell(
            log.complementarity_inf,
            InteriorPointResidualMetric::Complementarity,
            log.display_mode,
            log.flags.has_inequalities,
        ),
        style_ip_residual_cell(
            log.overall_inf,
            InteriorPointResidualMetric::Overall,
            log.display_mode,
            true,
        ),
        format!("{:>9}", sci_text(log.barrier_parameter)),
        fmt_optional_ip_sci(log.alpha),
        style_ip_line_search_cell(log.line_search_iterations),
        match log.linear_time_secs {
            Some(seconds) => format!("{:>7}", compact_duration_text(seconds)),
            None => format!("{:>7}", "--"),
        },
    ];
    eprintln!("{}", row.join("  "));
}

fn ip_event_legend_lines(
    log: &InteriorPointIterationLog,
    state: &mut SqpEventLegendState,
) -> Vec<String> {
    let mut parts = Vec::new();
    let snapshot = nlip_log_snapshot(log);
    for (code, description) in nlip_event_legend_entries(&snapshot) {
        let is_new = match code {
            'L' => state.mark_line_search_if_new(),
            'F' => state.mark_filter_if_new(),
            'X' => state.mark_filter_reset_if_new(),
            's' => state.mark_soc_attempted_if_new(),
            'S' => state.mark_soc_if_new(),
            'A' => state.mark_watchdog_armed_if_new(),
            'W' => state.mark_watchdog_if_new(),
            'B' => state.mark_bound_multiplier_safeguard_if_new(),
            'U' => state.mark_barrier_update_if_new(),
            'V' => state.mark_adaptive_regularization_if_new(),
            'T' => state.mark_tiny_step_if_new(),
            'M' => state.mark_max_iter_if_new(),
            _ => false,
        };
        if is_new {
            parts.push(description);
        }
    }

    let prefix = ip_event_legend_prefix();
    parts
        .into_iter()
        .map(|part| format!("{prefix}  {part}"))
        .collect()
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    fn snapshot_with_events(
        events: impl Into<Vec<InteriorPointIterationEvent>>,
    ) -> InteriorPointIterationSnapshot {
        InteriorPointIterationSnapshot {
            iteration: 0,
            phase: InteriorPointIterationPhase::AcceptedStep,
            x: Vec::new(),
            objective: 0.0,
            barrier_objective: None,
            eq_inf: None,
            ineq_inf: None,
            dual_inf: 0.0,
            comp_inf: None,
            overall_inf: 0.0,
            barrier_parameter: None,
            filter_theta: None,
            step_inf: None,
            alpha: None,
            alpha_pr: None,
            alpha_du: None,
            line_search_iterations: None,
            line_search_trials: 0,
            regularization_size: None,
            step_kind: None,
            step_tag: None,
            watchdog_active: false,
            line_search: None,
            direction_diagnostics: None,
            linear_debug: None,
            linear_solver: InteriorPointLinearSolver::Auto,
            linear_solve_time: None,
            filter: None,
            timing: InteriorPointIterationTiming::default(),
            events: events.into(),
        }
    }

    #[test]
    fn nlip_event_slots_keep_letter_columns_stable() {
        let filter_soc = snapshot_with_events(vec![
            InteriorPointIterationEvent::FilterAccepted,
            InteriorPointIterationEvent::SecondOrderCorrectionAccepted,
        ]);
        let filter_watchdog = snapshot_with_events(vec![
            InteriorPointIterationEvent::FilterAccepted,
            InteriorPointIterationEvent::WatchdogActivated,
        ]);
        let filter_reset = snapshot_with_events(vec![InteriorPointIterationEvent::FilterReset]);
        let watchdog_only =
            snapshot_with_events(vec![InteriorPointIterationEvent::WatchdogActivated]);
        let barrier_update =
            snapshot_with_events(vec![InteriorPointIterationEvent::BarrierParameterUpdated]);

        assert_eq!(nlip_event_slot_codes(&filter_soc), " F S        ");
        assert_eq!(nlip_event_slot_codes(&filter_watchdog), " F   W      ");
        assert_eq!(nlip_event_slot_codes(&filter_reset), "      X     ");
        assert_eq!(nlip_event_slot_codes(&watchdog_only), "     W      ");
        assert_eq!(nlip_event_slot_codes(&barrier_update), "        U   ");
    }

    #[test]
    fn nlip_event_header_matches_slot_width() {
        assert_eq!(fmt_ip_event_header().chars().count(), NLIP_EVENT_CELL_WIDTH);
        assert_eq!(
            nlip_event_slot_codes(&snapshot_with_events(Vec::new()))
                .chars()
                .count(),
            NLIP_EVENT_CELL_WIDTH
        );
    }

    #[test]
    fn nlip_event_codes_follow_slot_order() {
        let codes = nlip_event_codes_for_events(&[
            InteriorPointIterationEvent::WatchdogActivated,
            InteriorPointIterationEvent::FilterAccepted,
        ]);
        assert_eq!(codes, "FW");
    }

    #[test]
    fn bound_multiplier_correction_clamps_complementarity_band() {
        let slack = vec![2.0, 4.0];
        let z = vec![100.0, 1e-6];
        let mu = 1.0;
        let kappa_sigma = 10.0;

        let (corrected_z, max_correction) =
            correct_bound_multiplier_estimate(&z, &slack, mu, kappa_sigma);

        assert!(max_correction > 0.0);
        for (slack_i, z_i) in slack.iter().zip(corrected_z.iter()) {
            let compl = slack_i * z_i;
            assert!(compl <= kappa_sigma * mu + 1e-12);
            assert!(compl >= mu / kappa_sigma - 1e-12);
        }
    }

    #[test]
    fn acceptable_display_mode_uses_yellow_between_strict_and_acceptable() {
        let mode = InteriorPointDisplayMode::Acceptable {
            strict: InteriorPointDisplayTolerances {
                constraint: 1e-6,
                dual: 1e-6,
                complementarity: 1e-6,
                overall: 1e-6,
            },
            acceptable: InteriorPointDisplayTolerances {
                constraint: 1e-2,
                dual: 2e-2,
                complementarity: 1e-2,
                overall: 2e-2,
            },
        };

        assert_eq!(
            style_ip_metric("x", 1e-7, InteriorPointResidualMetric::Dual, mode),
            style_green_bold("x")
        );
        assert_eq!(
            style_ip_metric("x", 1e-2, InteriorPointResidualMetric::Dual, mode),
            style_yellow_bold("x")
        );
        assert_eq!(
            style_ip_metric("x", 3e-2, InteriorPointResidualMetric::Dual, mode),
            style_red_bold("x")
        );
    }

    fn test_direction(dx: &[f64], ds: &[f64]) -> NewtonDirection {
        NewtonDirection {
            dx: dx.to_vec(),
            d_lambda: Vec::new(),
            ds: ds.to_vec(),
            dz: Vec::new(),
            solver_used: InteriorPointLinearSolver::SparseQdldl,
            regularization_used: 0.0,
            dual_regularization_used: 0.0,
            primal_diagonal_shift_used: 0.0,
            linear_solution: dx.iter().chain(ds.iter()).copied().collect(),
            backend_stats: LinearBackendRunStats {
                solver: InteriorPointLinearSolver::SparseQdldl,
                factorization_time: Duration::ZERO,
                solve_time: Duration::ZERO,
                reused_symbolic: None,
                inertia: None,
                residual_inf: 0.0,
                solution_inf: 0.0,
                detail: None,
            },
            linear_debug: None,
        }
    }

    #[test]
    fn alpha_for_y_respects_primal_and_full_threshold() {
        let direction = test_direction(&[1e-3], &[2e-3]);
        let mut options = InteriorPointOptions {
            alpha_for_y: InteriorPointAlphaForYStrategy::PrimalAndFull,
            alpha_for_y_tol: 1e-2,
            ..InteriorPointOptions::default()
        };

        assert_eq!(alpha_for_y(0.2, 0.8, &direction, &options), 1.0);

        options.alpha_for_y_tol = 1e-4;
        assert_eq!(alpha_for_y(0.2, 0.8, &direction, &options), 0.2);
    }

    #[test]
    fn alpha_for_y_can_follow_bound_multiplier_step() {
        let direction = test_direction(&[1.0], &[1.0]);
        let options = InteriorPointOptions {
            alpha_for_y: InteriorPointAlphaForYStrategy::BoundMultiplier,
            ..InteriorPointOptions::default()
        };

        assert_eq!(alpha_for_y(0.3, 0.7, &direction, &options), 0.7);
    }

    #[test]
    fn filter_alpha_min_keeps_ipopt_formula_near_feasibility() {
        let options = InteriorPointOptions::default();
        let alpha_min = calculate_filter_alpha_min(1.0e-3, 1.0e-2, -1.0e-2, &options);

        assert_eq!(alpha_min, options.min_step);
    }
}

fn log_interior_point_problem_header<P>(
    problem: &P,
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
) where
    P: CompiledNlpProblem,
{
    let n = problem.dimension();
    let equality_count = problem.equality_count();
    let inequality_count = problem.inequality_count();
    let declared_box_constraints = declared_box_constraint_count(problem);
    let degrees_of_freedom = n as i128 - equality_count as i128;
    let total_jacobian_rows = equality_count + inequality_count;
    let total_jacobian_nnz =
        problem.equality_jacobian_ccs().nnz() + problem.inequality_jacobian_ccs().nnz();
    let total_jacobian_fill = dense_fill_percent(total_jacobian_nnz, total_jacobian_rows, n);
    let hessian_nnz = problem.lagrangian_hessian_ccs().nnz();
    let hessian_fill = lower_tri_fill_percent(hessian_nnz, n);
    let parameter_nnz = parameters
        .iter()
        .map(|parameter| parameter.values.len())
        .sum::<usize>();
    let lines = vec![
        boxed_line(
            "dimensions",
            format!(
                "vars={n}  eq={equality_count}  ineq={inequality_count}  box={declared_box_constraints}  dof={degrees_of_freedom}"
            ),
        ),
        boxed_line(
            "sparsity",
            format!(
                "jac={total_jacobian_nnz} nnz ({total_jacobian_fill:.2}%)  hess={hessian_nnz} nnz ({hessian_fill:.2}% lower)"
            ),
        ),
        boxed_line(
            "parameters",
            format!("matrices={}  nnz={parameter_nnz}", parameters.len()),
        ),
        String::new(),
        boxed_line(
            "tolerances",
            format!(
                "dual={}  constraint={}  complementarity={}  overall={}  s_max={}",
                sci_text(options.dual_tol),
                sci_text(options.constraint_tol),
                sci_text(options.complementarity_tol),
                sci_text(options.overall_tol),
                sci_text(options.overall_scale_max),
            ),
        ),
        boxed_line("summary", format_nlip_settings_summary(options)),
        boxed_line(
            "line search",
            format!(
                "beta={}  c1={}  min_step={}  fraction_to_boundary={}  filter={}",
                sci_text(options.line_search_beta),
                sci_text(options.line_search_c1),
                sci_text(options.min_step),
                sci_text(options.fraction_to_boundary),
                "on",
            ),
        ),
        boxed_line(
            "filter",
            format!(
                "gamma_obj={}  gamma_violation={}",
                sci_text(options.filter_gamma_objective),
                sci_text(options.filter_gamma_violation),
            ),
        ),
        boxed_line("barrier", format!("mu_min={}", sci_text(options.mu_min))),
        boxed_line(
            "linear solve",
            format!(
                "solver={}  regularization={}",
                options.linear_solver.label(),
                sci_text(options.regularization),
            ),
        ),
        boxed_line(
            "iteration",
            format!(
                "max_iters={}  verbose={}",
                options.max_iters, options.verbose
            ),
        ),
    ];
    log_boxed_section("NLIP problem / settings", &lines, style_cyan_bold);
}

fn log_interior_point_status_summary(
    summary: &InteriorPointSummary,
    options: &InteriorPointOptions,
) {
    let display_mode = InteriorPointDisplayMode::for_termination(options, summary.termination);
    let has_inequality_like_constraints = !summary.inequality_multipliers.is_empty()
        || !summary.lower_bound_multipliers.is_empty()
        || !summary.upper_bound_multipliers.is_empty();
    let eq_text = if summary.equality_multipliers.is_empty() {
        "--".to_string()
    } else {
        style_ip_residual_text(
            summary.equality_inf_norm,
            InteriorPointResidualMetric::Constraint,
            display_mode,
            true,
        )
    };
    let ineq_text = if has_inequality_like_constraints {
        style_ip_residual_text(
            summary.inequality_inf_norm,
            InteriorPointResidualMetric::Constraint,
            display_mode,
            true,
        )
    } else {
        "--".to_string()
    };
    let comp_text = if has_inequality_like_constraints {
        style_ip_residual_text(
            summary.complementarity_inf_norm,
            InteriorPointResidualMetric::Complementarity,
            display_mode,
            true,
        )
    } else {
        "--".to_string()
    };
    let callback_total_time = summary.profiling.total_callback_time();
    let callback_rows = [
        ("objective", &summary.profiling.objective_value),
        ("gradient", &summary.profiling.objective_gradient),
        ("eq values", &summary.profiling.equality_values),
        ("ineq values", &summary.profiling.inequality_values),
        ("eq jac", &summary.profiling.equality_jacobian_values),
        ("ineq jac", &summary.profiling.inequality_jacobian_values),
        ("hessian", &summary.profiling.lagrangian_hessian_values),
    ];
    let callback_unit = choose_summary_duration_unit(&[
        callback_total_time,
        summary.profiling.objective_value.total_time,
        summary.profiling.objective_gradient.total_time,
        summary.profiling.equality_values.total_time,
        summary.profiling.inequality_values.total_time,
        summary.profiling.equality_jacobian_values.total_time,
        summary.profiling.inequality_jacobian_values.total_time,
        summary.profiling.lagrangian_hessian_values.total_time,
    ]);
    let timing_unit = choose_summary_duration_unit(&[
        summary.profiling.kkt_assembly_time,
        summary.profiling.linear_solve_time,
        summary.profiling.preprocessing_time,
        summary.profiling.unaccounted_time,
        summary.profiling.total_time,
        summary
            .profiling
            .adapter_timing
            .map(|timing| timing.callback_evaluation)
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .adapter_timing
            .map(|timing| timing.output_marshalling)
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .adapter_timing
            .map(|timing| timing.layout_projection)
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .backend_timing
            .function_creation_time
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .backend_timing
            .jit_time
            .unwrap_or(Duration::ZERO),
    ]);
    let callback_row = |name: &str, calls: Index, duration: Duration| {
        format!(
            "{name:<12}  calls={calls:>4}  time={}",
            fmt_duration_in_unit(duration, callback_unit)
        )
    };
    let timing_row = |name: &str, count: Option<Index>, duration: Duration| {
        let count_cell = match count {
            Some(count) => format!("{count:>4}"),
            None => format!("{:>4}", "--"),
        };
        format!(
            "{name:<12}  count={count_cell}  time={}",
            fmt_duration_in_unit(duration, timing_unit)
        )
    };
    let optional_timing_row = |name: &str, count: Option<Index>, duration: Option<Duration>| {
        let count_cell = match count {
            Some(count) => format!("{count:>4}"),
            None => format!("{:>4}", "--"),
        };
        let time_cell = duration
            .map(|duration| fmt_duration_in_unit(duration, timing_unit))
            .unwrap_or_else(|| format!("{:>7}", "--"));
        format!("{name:<12}  count={count_cell}  time={time_cell}")
    };
    let lines = vec![
        boxed_line(
            "result",
            format!(
                "objective={}  {}={}  {}={}  {}={}  mu={}",
                sci_text(summary.objective),
                PRIMAL_INF_LABEL,
                style_ip_residual_text(
                    summary.primal_inf_norm,
                    InteriorPointResidualMetric::Constraint,
                    display_mode,
                    true,
                ),
                DUAL_INF_LABEL,
                style_ip_residual_text(
                    summary.dual_inf_norm,
                    InteriorPointResidualMetric::Dual,
                    display_mode,
                    true,
                ),
                OVERALL_INF_LABEL,
                style_ip_residual_text(
                    summary.overall_inf_norm,
                    InteriorPointResidualMetric::Overall,
                    display_mode,
                    true,
                ),
                sci_text(summary.barrier_parameter),
            ),
        ),
        boxed_line(
            "",
            format!(
                "{}={}  {}={}  {}={}  iterations={}",
                EQ_INF_LABEL,
                eq_text,
                INEQ_INF_LABEL,
                ineq_text,
                IP_COMP_INF_LABEL,
                comp_text,
                summary.iterations,
            ),
        ),
        String::new(),
        boxed_line(
            "callbacks",
            callback_row(
                "total",
                summary.profiling.total_callback_calls(),
                callback_total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[0].0,
                callback_rows[0].1.calls,
                callback_rows[0].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[1].0,
                callback_rows[1].1.calls,
                callback_rows[1].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[2].0,
                callback_rows[2].1.calls,
                callback_rows[2].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[3].0,
                callback_rows[3].1.calls,
                callback_rows[3].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[4].0,
                callback_rows[4].1.calls,
                callback_rows[4].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[5].0,
                callback_rows[5].1.calls,
                callback_rows[5].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[6].0,
                callback_rows[6].1.calls,
                callback_rows[6].1.total_time,
            ),
        ),
        String::new(),
        boxed_line(
            "timing",
            timing_row(
                "kkt assembly",
                Some(summary.profiling.kkt_assemblies),
                summary.profiling.kkt_assembly_time,
            ),
        ),
        boxed_line(
            "",
            timing_row(
                "linear solve",
                Some(summary.profiling.linear_solves),
                summary.profiling.linear_solve_time,
            ),
        ),
        boxed_line(
            "",
            optional_timing_row(
                "adapter cb",
                None,
                summary
                    .profiling
                    .adapter_timing
                    .map(|timing| timing.callback_evaluation),
            ),
        ),
        boxed_line(
            "",
            optional_timing_row(
                "adapter io",
                None,
                summary
                    .profiling
                    .adapter_timing
                    .map(|timing| timing.output_marshalling),
            ),
        ),
        boxed_line(
            "",
            optional_timing_row(
                "layout",
                None,
                summary
                    .profiling
                    .adapter_timing
                    .map(|timing| timing.layout_projection),
            ),
        ),
        boxed_line(
            "",
            timing_row("preprocess", None, summary.profiling.preprocessing_time),
        ),
        boxed_line(
            "",
            timing_row("unaccounted", None, summary.profiling.unaccounted_time),
        ),
        boxed_line("", timing_row("total", None, summary.profiling.total_time)),
        String::new(),
        boxed_line(
            "backend",
            format!(
                "create={}  jit={}  linear_solver={}",
                fmt_optional_duration_in_unit(
                    summary.profiling.backend_timing.function_creation_time,
                    timing_unit,
                ),
                fmt_optional_duration_in_unit(
                    summary.profiling.backend_timing.jit_time,
                    timing_unit
                ),
                summary.linear_solver.label(),
            ),
        ),
    ];
    match summary.termination {
        InteriorPointTermination::Converged => {
            log_boxed_section("Interior-point converged", &lines, style_green_bold)
        }
        InteriorPointTermination::Acceptable => log_boxed_section(
            "Interior-point reached acceptable level",
            &lines,
            style_yellow_bold,
        ),
    }
}

fn finalise_interior_point_profiling(
    profiling: &mut InteriorPointProfiling,
    solve_started: Instant,
) {
    profiling.total_time = solve_started.elapsed();
    profiling.unaccounted_time = profiling.total_time.saturating_sub(
        profiling.total_callback_time()
            + profiling.kkt_assembly_time
            + profiling.linear_solve_time
            + profiling.preprocessing_time,
    );
}

fn validate_interior_point_options(
    options: &InteriorPointOptions,
) -> std::result::Result<(), InteriorPointSolveError> {
    if !options.overall_tol.is_finite() || options.overall_tol < 0.0 {
        return Err(InteriorPointSolveError::InvalidInput(format!(
            "overall_tol must be finite and non-negative, got {}",
            options.overall_tol
        )));
    }
    if !options.overall_scale_max.is_finite() || options.overall_scale_max <= 0.0 {
        return Err(InteriorPointSolveError::InvalidInput(format!(
            "overall_scale_max must be finite and positive, got {}",
            options.overall_scale_max
        )));
    }
    if let Some(linear_debug) = options.linear_debug.as_ref()
        && linear_debug
            .compare_solvers
            .contains(&InteriorPointLinearSolver::Auto)
    {
        return Err(InteriorPointSolveError::InvalidInput(
            "linear_debug.compare_solvers must not contain auto".into(),
        ));
    }
    Ok(())
}

pub fn solve_nlp_interior_point<P>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
) -> std::result::Result<InteriorPointSummary, InteriorPointSolveError>
where
    P: CompiledNlpProblem,
{
    solve_nlp_interior_point_with_callback(problem, x0, parameters, options, |_| {})
}

pub fn solve_nlp_interior_point_with_callback<P, C>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
    mut callback: C,
) -> std::result::Result<InteriorPointSummary, InteriorPointSolveError>
where
    P: CompiledNlpProblem,
    C: FnMut(&InteriorPointIterationSnapshot),
{
    let solve_started = Instant::now();
    let mut profiling = InteriorPointProfiling {
        backend_timing: problem.backend_timing_metadata(),
        ..InteriorPointProfiling::default()
    };
    let validation_started = Instant::now();
    validate_nlp_problem_shapes(problem)
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
    validate_parameter_inputs(problem, parameters)
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
    validate_interior_point_options(options)?;
    profiling.preprocessing_steps += 1;
    profiling.preprocessing_time += validation_started.elapsed();

    let n = problem.dimension();
    if x0.len() != n {
        return Err(InteriorPointSolveError::InvalidInput(format!(
            "x0 has length {}, expected {n}",
            x0.len()
        )));
    }

    let equality_count = problem.equality_count();
    let inequality_count = problem.inequality_count();
    let bounds = collect_bound_constraints(problem)
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
    let augmented_inequality_count = inequality_count + bounds.total_count();
    let lower_bound_count = bounds.lower_indices.len();
    let bound_jacobian = build_bound_jacobian_sparse(&bounds, n);
    let equality_jacobian_structure =
        Arc::new(sparse_structure_from_ccs(problem.equality_jacobian_ccs()));
    let nonlinear_inequality_structure =
        sparse_structure_from_ccs(problem.inequality_jacobian_ccs());
    let inequality_jacobian_structure = Arc::new(vstack_sparse_structures(
        &nonlinear_inequality_structure,
        bound_jacobian.structure.as_ref(),
    ));
    let hessian_structure = Arc::new(problem.lagrangian_hessian_ccs().clone());
    let preferred_solver = preferred_linear_solver(
        options.linear_solver,
        equality_count,
        augmented_inequality_count,
    );
    let mut spral_workspace = None;
    let mut spral_workspace_unavailable = false;
    let mut native_spral_workspace = None;
    let mut native_spral_workspace_unavailable = false;
    let mut linear_debug_state =
        options
            .linear_debug
            .clone()
            .map(|debug_options| InteriorPointLinearDebugState {
                options: debug_options,
                rust_workspace: None,
                rust_workspace_unavailable: false,
                rust_workspace_error: None,
                native_workspace: None,
                native_workspace_unavailable: false,
                native_workspace_error: None,
            });
    let mut x = x0.to_vec();
    project_initial_point_into_box_interior(&mut x, &bounds, options);
    let mut lambda_eq = vec![0.0; equality_count];
    let mut z = vec![1.0; augmented_inequality_count];
    let mut slack = vec![1.0; augmented_inequality_count];
    let mut event_state = SqpEventLegendState::default();
    let mut last_adapter_timing = problem.adapter_timing_snapshot();
    profiling.adapter_timing = last_adapter_timing;

    let setup_started = Instant::now();
    let mut setup_callback_time = Duration::ZERO;
    let initial_state = trial_state(
        problem,
        &x,
        parameters,
        &bounds,
        &bound_jacobian,
        &equality_jacobian_structure,
        &inequality_jacobian_structure,
        &mut profiling,
        &mut setup_callback_time,
    );
    profiling.preprocessing_steps += 1;
    profiling.preprocessing_time += setup_started.elapsed().saturating_sub(setup_callback_time);
    initialise_slacks(
        &initial_state.augmented_inequality_values,
        &mut slack,
        options,
    );
    match options.bound_mult_init_method {
        InteriorPointBoundMultiplierInitMethod::Constant => {
            z.fill(options.bound_mult_init_val);
        }
        InteriorPointBoundMultiplierInitMethod::MuBased => {
            for (slack_i, z_i) in slack.iter().zip(z.iter_mut()) {
                *z_i = options.mu_init / slack_i.max(1e-8);
            }
        }
    }
    if options.least_square_init_duals {
        let (initial_lambda_eq, initial_z) =
            least_squares_initial_dual_state(&initial_state, &z, options.regularization);
        if initial_z.len() == z.len() {
            for (z_i, restored_z_i) in z.iter_mut().zip(initial_z.into_iter()) {
                *z_i = (*z_i).max(restored_z_i);
            }
        }
        lambda_eq = initial_lambda_eq;
    }

    if options.verbose {
        log_interior_point_problem_header(problem, parameters, options);
    }

    let mut nonlinear_inequality_multipliers = vec![0.0; inequality_count];
    let mut last_linear_solver = preferred_solver;
    let mut filter_entries = Vec::new();
    let mut filter_reset_count = 0;
    let mut successive_filter_rejections = 0;
    let mut snapshots = Vec::new();
    let mut last_accepted_state: Option<InteriorPointIterationSnapshot> = None;
    let mut acceptable_counter = 0;
    let mut last_objective_value = initial_state.objective_value;
    let initial_theta = filter_theta_l1_norm(
        &initial_state.equality_values,
        &initial_state.augmented_inequality_values,
        &slack,
    );
    let theta_scale = initial_theta.max(1.0);
    let theta_max = options.theta_max_fact * theta_scale;
    let theta_min = options.theta_min_fact * theta_scale;
    let mut barrier_parameter_value = if augmented_inequality_count > 0 {
        let initial_complementarity = barrier_parameter(&slack, &z);
        options
            .mu_target
            .max(options.mu_min)
            .max(initial_complementarity.min(options.mu_init))
    } else {
        0.0
    };
    let mut watchdog_state = InteriorPointWatchdogState::default();
    let mut pending_iteration_events = Vec::new();

    if options.max_iters == 0 {
        let equality_inf = inf_norm(&initial_state.equality_values);
        let inequality_inf = positive_part_inf_norm(&initial_state.augmented_inequality_values);
        let dual_inf = inf_norm(&lagrangian_gradient_sparse(
            &initial_state.gradient,
            &initial_state.equality_jacobian,
            &lambda_eq,
            &initial_state.inequality_jacobian,
            &z,
        ));
        let complementarity_inf = if augmented_inequality_count > 0 {
            complementarity_inf_norm(&slack, &z)
        } else {
            0.0
        };
        let current_theta = filter_theta_l1_norm(
            &initial_state.equality_values,
            &initial_state.augmented_inequality_values,
            &slack,
        );
        let current_barrier_objective = barrier_objective_value(
            initial_state.objective_value,
            &slack,
            barrier_parameter_value,
        );
        let current_filter_entry = super::filter::entry(current_barrier_objective, current_theta);
        let all_dual_multipliers = [lambda_eq.as_slice(), z.as_slice()].concat();
        let overall_inf = scaled_overall_inf_norm(
            current_theta,
            dual_inf,
            complementarity_inf,
            &all_dual_multipliers,
            &z,
            options.overall_scale_max,
        );
        let snapshot = InteriorPointIterationSnapshot {
            iteration: 0,
            phase: InteriorPointIterationPhase::Initial,
            x: x.clone(),
            objective: initial_state.objective_value,
            barrier_objective: Some(current_barrier_objective),
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
            overall_inf,
            barrier_parameter: (augmented_inequality_count > 0).then_some(barrier_parameter_value),
            filter_theta: Some(current_theta),
            step_inf: None,
            alpha: None,
            alpha_pr: None,
            alpha_du: None,
            line_search_iterations: None,
            line_search_trials: 0,
            regularization_size: Some(options.regularization),
            step_kind: None,
            step_tag: None,
            watchdog_active: watchdog_state.reference.is_some()
                && watchdog_state.remaining_iters > 0,
            line_search: None,
            direction_diagnostics: None,
            linear_debug: None,
            linear_solver: last_linear_solver,
            linear_solve_time: None,
            filter: Some(FilterInfo {
                current: current_filter_entry,
                entries: filter_entries.clone(),
                accepted_mode: None,
            }),
            timing: InteriorPointIterationTiming {
                adapter_timing: profiling.adapter_timing,
                callback: setup_callback_time,
                kkt_assembly: Duration::ZERO,
                linear_solve: Duration::ZERO,
                preprocess: setup_started.elapsed().saturating_sub(setup_callback_time),
                total: setup_started.elapsed(),
            },
            events: vec![InteriorPointIterationEvent::MaxIterationsReached],
        };
        snapshots.push(snapshot.clone());
        callback(&snapshot);
        if options.verbose {
            let flags = InteriorPointIterationLogFlags {
                has_equalities: equality_count > 0,
                has_inequalities: augmented_inequality_count > 0,
                iteration_limit_reached: true,
                ..InteriorPointIterationLogFlags::default()
            };
            log_interior_point_iteration(
                &InteriorPointIterationLog {
                    iteration: 0,
                    phase: InteriorPointIterationPhase::Initial,
                    flags,
                    extra_events: Vec::new(),
                    display_mode: InteriorPointDisplayMode::strict(options),
                    objective_value: initial_state.objective_value,
                    barrier_objective: current_barrier_objective,
                    equality_inf,
                    inequality_inf,
                    dual_inf,
                    complementarity_inf,
                    overall_inf,
                    barrier_parameter: if augmented_inequality_count > 0 {
                        barrier_parameter_value
                    } else {
                        0.0
                    },
                    alpha: None,
                    alpha_pr: None,
                    alpha_du: None,
                    line_search_iterations: None,
                    regularization_size: Some(options.regularization),
                    step_kind: None,
                    step_tag: None,
                    linear_time_secs: None,
                },
                &mut event_state,
            );
        }
        return Err(InteriorPointSolveError::MaxIterations {
            iterations: options.max_iters,
            context: interior_point_failure_context(
                Some(snapshot),
                last_accepted_state.clone(),
                None,
                None,
                None,
                &profiling,
                solve_started,
            ),
        });
    }

    for iteration in 0..options.max_iters {
        let mut iteration_events = std::mem::take(&mut pending_iteration_events);
        let iteration_started = Instant::now();
        let mut iteration_callback_time = Duration::ZERO;
        let mut iteration_kkt_assembly_time = Duration::ZERO;
        let mut iteration_linear_solve_time = Duration::ZERO;
        let state = trial_state(
            problem,
            &x,
            parameters,
            &bounds,
            &bound_jacobian,
            &equality_jacobian_structure,
            &inequality_jacobian_structure,
            &mut profiling,
            &mut iteration_callback_time,
        );
        let equality_inf = inf_norm(&state.equality_values);
        let inequality_inf = positive_part_inf_norm(&state.augmented_inequality_values);
        let inequality_residual =
            slack_form_inequality_residuals(&state.augmented_inequality_values, &slack);
        let internal_inequality_inf = inf_norm(&inequality_residual);
        let primal_inf = equality_inf.max(internal_inequality_inf);
        let dual_residual = lagrangian_gradient_sparse(
            &state.gradient,
            &state.equality_jacobian,
            &lambda_eq,
            &state.inequality_jacobian,
            &z,
        );
        let dual_inf = inf_norm(&dual_residual);
        let debug_any_dual = std::env::var_os("NLIP_DEBUG_MAX_DUAL_ANY").is_some();
        if (std::env::var_os("NLIP_DEBUG_MAX_DUAL").is_some()
            && primal_inf <= 1.0e-4
            && dual_inf <= 1.0e-1
            || debug_any_dual)
            && iteration % 10 == 0
            && let Some((max_index, max_value)) = dual_residual
                .iter()
                .enumerate()
                .max_by(|(_, lhs), (_, rhs)| lhs.abs().total_cmp(&rhs.abs()))
        {
            let equality_contrib = state.equality_jacobian.structure.ccs.col_ptrs[max_index]
                ..state.equality_jacobian.structure.ccs.col_ptrs[max_index + 1];
            let equality_contrib = equality_contrib
                .map(|entry| {
                    state.equality_jacobian.values[entry]
                        * lambda_eq[state.equality_jacobian.structure.ccs.row_indices[entry]]
                })
                .sum::<f64>();
            let top_eq_terms = state.equality_jacobian.structure.ccs.col_ptrs[max_index]
                ..state.equality_jacobian.structure.ccs.col_ptrs[max_index + 1];
            let mut top_eq_terms = top_eq_terms
                .map(|entry| {
                    let row = state.equality_jacobian.structure.ccs.row_indices[entry];
                    let contribution = state.equality_jacobian.values[entry] * lambda_eq[row];
                    (row, contribution)
                })
                .collect::<Vec<_>>();
            top_eq_terms.sort_by(|lhs, rhs| rhs.1.abs().total_cmp(&lhs.1.abs()));
            let top_eq_summary = top_eq_terms
                .iter()
                .take(5)
                .map(|(row, contribution)| format!("{row}:{contribution:.3e}"))
                .collect::<Vec<_>>()
                .join(",");
            let inequality_contrib = state.inequality_jacobian.structure.ccs.col_ptrs[max_index]
                ..state.inequality_jacobian.structure.ccs.col_ptrs[max_index + 1];
            let inequality_contrib = inequality_contrib
                .map(|entry| {
                    state.inequality_jacobian.values[entry]
                        * z[state.inequality_jacobian.structure.ccs.row_indices[entry]]
                })
                .sum::<f64>();
            let top_ineq_terms = state.inequality_jacobian.structure.ccs.col_ptrs[max_index]
                ..state.inequality_jacobian.structure.ccs.col_ptrs[max_index + 1];
            let mut top_ineq_terms = top_ineq_terms
                .map(|entry| {
                    let row = state.inequality_jacobian.structure.ccs.row_indices[entry];
                    let contribution = state.inequality_jacobian.values[entry] * z[row];
                    (row, contribution)
                })
                .collect::<Vec<_>>();
            top_ineq_terms.sort_by(|lhs, rhs| rhs.1.abs().total_cmp(&lhs.1.abs()));
            let top_ineq_summary = top_ineq_terms
                .iter()
                .take(3)
                .map(|(row, contribution)| {
                    let kind = if *row < inequality_count {
                        "ineq"
                    } else if *row < inequality_count + lower_bound_count {
                        "lb"
                    } else {
                        "ub"
                    };
                    format!("{kind}:{row}:{contribution:.3e}")
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "NLIP_DEBUG_MAX_DUAL iter={} idx={} residual={:.6e} x={:.6e} grad={:.6e} eq={:.6e} ineq={:.6e} primal={:.6e} top_eq=[{}] top_ineq=[{}]",
                iteration,
                max_index,
                max_value,
                x.get(max_index).copied().unwrap_or(0.0),
                state.gradient.get(max_index).copied().unwrap_or(0.0),
                equality_contrib,
                inequality_contrib,
                primal_inf,
                top_eq_summary,
                top_ineq_summary,
            );
        }
        let complementarity_inf = if augmented_inequality_count > 0 {
            complementarity_inf_norm(&slack, &z)
        } else {
            0.0
        };
        let all_dual_multipliers = [lambda_eq.as_slice(), z.as_slice()].concat();
        let overall_inf = scaled_overall_inf_norm(
            primal_inf,
            dual_inf,
            complementarity_inf,
            &all_dual_multipliers,
            &z,
            options.overall_scale_max,
        );
        let current_barrier_objective =
            barrier_objective_value(state.objective_value, &slack, barrier_parameter_value);
        let current_theta = filter_theta_l1_norm(
            &state.equality_values,
            &state.augmented_inequality_values,
            &slack,
        );
        let current_filter_entry = super::filter::entry(current_barrier_objective, current_theta);
        let mut current_snapshot = InteriorPointIterationSnapshot {
            iteration,
            phase: if iteration == 0 {
                InteriorPointIterationPhase::Initial
            } else {
                InteriorPointIterationPhase::AcceptedStep
            },
            x: x.clone(),
            objective: state.objective_value,
            barrier_objective: Some(current_barrier_objective),
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
            overall_inf,
            barrier_parameter: (augmented_inequality_count > 0).then_some(barrier_parameter_value),
            filter_theta: Some(current_theta),
            step_inf: None,
            alpha: None,
            alpha_pr: None,
            alpha_du: None,
            line_search_iterations: None,
            line_search_trials: 0,
            regularization_size: Some(options.regularization),
            step_kind: None,
            step_tag: None,
            watchdog_active: watchdog_state.reference.is_some()
                && watchdog_state.remaining_iters > 0,
            line_search: None,
            direction_diagnostics: None,
            linear_debug: None,
            linear_solver: last_linear_solver,
            linear_solve_time: None,
            filter: Some(FilterInfo {
                current: current_filter_entry.clone(),
                entries: filter_entries.clone(),
                accepted_mode: None,
            }),
            timing: InteriorPointIterationTiming::default(),
            events: iteration_events.clone(),
        };
        let acceptable_state = InteriorPointAcceptableState {
            overall_error: overall_inf,
            dual_inf,
            constr_viol: primal_inf,
            compl_inf: interior_point_complementarity_target_inf_norm(
                &slack,
                &z,
                options.mu_target,
            ),
            objective: state.objective_value,
        };
        let acceptable_iterate = options.acceptable_iter > 0
            && interior_point_current_is_acceptable(
                acceptable_state,
                last_objective_value,
                options,
            );
        if acceptable_iterate {
            acceptable_counter += 1;
        } else {
            acceptable_counter = 0;
        }
        let converged = overall_inf <= options.overall_tol
            && primal_inf <= options.constraint_tol
            && dual_inf <= options.dual_tol
            && complementarity_inf <= options.complementarity_tol;
        let acceptable_converged = !converged
            && options.acceptable_iter > 0
            && acceptable_counter >= options.acceptable_iter;
        if converged || acceptable_converged {
            let adapter_timing = adapter_timing_delta(problem, &mut last_adapter_timing);
            profiling.adapter_timing = last_adapter_timing;
            let iteration_total = iteration_started.elapsed();
            let iteration_preprocess = iteration_total.saturating_sub(
                iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
            );
            let termination = if converged {
                InteriorPointTermination::Converged
            } else {
                InteriorPointTermination::Acceptable
            };
            let snapshot = InteriorPointIterationSnapshot {
                iteration,
                phase: InteriorPointIterationPhase::Converged,
                x: x.clone(),
                objective: state.objective_value,
                barrier_objective: Some(current_barrier_objective),
                eq_inf: (equality_count > 0).then_some(equality_inf),
                ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
                dual_inf,
                comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
                overall_inf,
                barrier_parameter: (augmented_inequality_count > 0)
                    .then_some(barrier_parameter_value),
                filter_theta: Some(current_theta),
                step_inf: None,
                alpha: None,
                alpha_pr: None,
                alpha_du: None,
                line_search_iterations: None,
                line_search_trials: 0,
                regularization_size: Some(options.regularization),
                step_kind: None,
                step_tag: None,
                watchdog_active: false,
                line_search: None,
                direction_diagnostics: None,
                linear_debug: None,
                linear_solver: last_linear_solver,
                linear_solve_time: None,
                filter: Some(FilterInfo {
                    current: current_filter_entry,
                    entries: filter_entries.clone(),
                    accepted_mode: None,
                }),
                timing: InteriorPointIterationTiming {
                    adapter_timing,
                    callback: iteration_callback_time,
                    kkt_assembly: iteration_kkt_assembly_time,
                    linear_solve: iteration_linear_solve_time,
                    preprocess: iteration_preprocess,
                    total: iteration_total,
                },
                events: iteration_events.clone(),
            };
            snapshots.push(snapshot.clone());
            callback(&snapshot);
            let (nonlinear_ineq, lower_bounds, upper_bounds) =
                split_augmented_inequality_multipliers(&z, inequality_count, lower_bound_count);
            let summary = InteriorPointSummary {
                x,
                equality_multipliers: lambda_eq,
                inequality_multipliers: nonlinear_ineq,
                lower_bound_multipliers: lower_bounds,
                upper_bound_multipliers: upper_bounds,
                slack,
                objective: state.objective_value,
                iterations: iteration,
                equality_inf_norm: equality_inf,
                inequality_inf_norm: inequality_inf,
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: complementarity_inf,
                overall_inf_norm: overall_inf,
                barrier_parameter: barrier_parameter_value,
                termination,
                status_kind: match termination {
                    InteriorPointTermination::Converged => InteriorPointStatusKind::Success,
                    InteriorPointTermination::Acceptable => InteriorPointStatusKind::Warning,
                },
                snapshots,
                final_state: snapshot.clone(),
                last_accepted_state: last_accepted_state.clone(),
                profiling,
                linear_solver: last_linear_solver,
            };
            let mut summary = summary;
            finalise_interior_point_profiling(&mut summary.profiling, solve_started);
            if options.verbose {
                log_interior_point_iteration(
                    &InteriorPointIterationLog {
                        iteration,
                        phase: InteriorPointIterationPhase::Converged,
                        flags: InteriorPointIterationLogFlags {
                            has_equalities: equality_count > 0,
                            has_inequalities: augmented_inequality_count > 0,
                            ..InteriorPointIterationLogFlags::default()
                        },
                        extra_events: iteration_events.clone(),
                        display_mode: InteriorPointDisplayMode::for_termination(
                            options,
                            termination,
                        ),
                        objective_value: state.objective_value,
                        barrier_objective: current_barrier_objective,
                        equality_inf,
                        inequality_inf,
                        dual_inf,
                        complementarity_inf,
                        overall_inf,
                        barrier_parameter: barrier_parameter_value,
                        alpha: None,
                        alpha_pr: None,
                        alpha_du: None,
                        line_search_iterations: None,
                        regularization_size: Some(options.regularization),
                        step_kind: None,
                        step_tag: None,
                        linear_time_secs: None,
                    },
                    &mut event_state,
                );
                log_interior_point_status_summary(&summary, options);
            }
            return Ok(summary);
        }
        last_objective_value = state.objective_value;

        let hessian_started = Instant::now();
        let mut hessian_values = vec![0.0; problem.lagrangian_hessian_ccs().nnz()];
        time_callback(
            &mut profiling.lagrangian_hessian_values,
            &mut iteration_callback_time,
            || {
                problem.lagrangian_hessian_values(
                    &x,
                    parameters,
                    &lambda_eq,
                    &nonlinear_inequality_multipliers,
                    &mut hessian_values,
                );
            },
        );
        let hessian = SparseSymmetricMatrix {
            lower_triangle: Arc::clone(&hessian_structure),
            values: hessian_values,
        };
        let hessian_elapsed = hessian_started.elapsed();
        profiling.kkt_assemblies += 1;
        profiling.kkt_assembly_time += hessian_elapsed;
        iteration_kkt_assembly_time += hessian_elapsed;

        let kkt_regularization = kkt_regularization(
            augmented_inequality_count > 0,
            primal_inf,
            complementarity_inf,
            dual_inf,
            options,
        );
        let sigma = if augmented_inequality_count > 0 {
            1.0
        } else {
            0.0
        };
        let r_cent = if augmented_inequality_count > 0 {
            slack
                .iter()
                .zip(z.iter())
                .map(|(s, z_i)| s * z_i - sigma * barrier_parameter_value)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let linear_started = Instant::now();
        let reduced_kkt_system = ReducedKktSystem {
            hessian: &hessian,
            equality_jacobian: &state.equality_jacobian,
            inequality_jacobian: &state.inequality_jacobian,
            slack: &slack,
            multipliers: &z,
            r_dual: &dual_residual,
            r_eq: &state.equality_values,
            r_ineq: &inequality_residual,
            r_cent: &r_cent,
            solver: options.linear_solver,
            regularization: kkt_regularization,
            adaptive_regularization_retries: options.adaptive_regularization_retries,
            regularization_growth_factor: options.regularization_growth_factor,
            regularization_max: options.regularization_max,
            spral_pivot_method: options.spral_pivot_method,
            spral_action_on_zero_pivot: options.spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: options.spral_small_pivot_tolerance,
            spral_threshold_pivot_u: options.spral_threshold_pivot_u,
        };
        current_snapshot.linear_solver = preferred_solver;
        if !spral_workspace_unavailable
            && spral_workspace.is_none()
            && preferred_solver == InteriorPointLinearSolver::SpralSsids
        {
            match prepare_spral_workspace(
                hessian_structure.as_ref(),
                equality_jacobian_structure.as_ref(),
                inequality_jacobian_structure.as_ref(),
                &mut profiling,
                options.verbose,
            ) {
                Ok(workspace) => {
                    spral_workspace = Some(workspace);
                }
                Err(error) => {
                    if options.linear_solver == InteriorPointLinearSolver::SpralSsids {
                        return Err(with_interior_point_failure_profiling(
                            error,
                            Some(snapshot_with_nlip_events(
                                current_snapshot.clone(),
                                &iteration_events,
                            )),
                            last_accepted_state.clone(),
                            &profiling,
                            solve_started,
                        ));
                    }
                    spral_workspace_unavailable = true;
                }
            }
        }
        if !native_spral_workspace_unavailable
            && native_spral_workspace.is_none()
            && preferred_solver == InteriorPointLinearSolver::NativeSpralSsids
        {
            match prepare_native_spral_workspace(
                hessian_structure.as_ref(),
                equality_jacobian_structure.as_ref(),
                inequality_jacobian_structure.as_ref(),
                &spral_numeric_factor_options(options),
                &mut profiling,
                options.verbose,
            ) {
                Ok(workspace) => {
                    native_spral_workspace = Some(workspace);
                }
                Err(error) => {
                    if options.linear_solver == InteriorPointLinearSolver::NativeSpralSsids {
                        return Err(with_interior_point_failure_profiling(
                            error,
                            Some(snapshot_with_nlip_events(
                                current_snapshot.clone(),
                                &iteration_events,
                            )),
                            last_accepted_state.clone(),
                            &profiling,
                            solve_started,
                        ));
                    }
                    native_spral_workspace_unavailable = true;
                }
            }
        }
        let solve_result = solve_reduced_kkt(
            &reduced_kkt_system,
            spral_workspace.as_mut(),
            native_spral_workspace.as_mut(),
            &mut profiling,
            options.verbose,
        );
        let direction = match solve_result {
            Ok(mut direction) => {
                current_snapshot.linear_solver = direction.solver_used;
                current_snapshot.linear_solve_time = Some(
                    direction.backend_stats.factorization_time + direction.backend_stats.solve_time,
                );
                if let Some(debug_state) = linear_debug_state.as_mut()
                    && should_run_linear_debug(debug_state.options.schedule, iteration, false)
                    && let Ok(snapshot) = build_interior_point_kkt_snapshot(
                        iteration,
                        current_snapshot.phase,
                        direction.solver_used,
                        &reduced_kkt_system,
                        direction.dual_regularization_used,
                        direction.primal_diagonal_shift_used,
                        barrier_parameter_value,
                        primal_inf,
                        dual_inf,
                        complementarity_inf,
                        0,
                    )
                {
                    let report =
                        run_linear_debug_report_on_success(&snapshot, &direction, debug_state);
                    dump_linear_debug_snapshot(&debug_state.options, &snapshot, &report);
                    current_snapshot.linear_debug = Some(report.clone());
                    direction.linear_debug = Some(report);
                }
                direction
            }
            Err(error) => {
                let mut error = error;
                if let Some(debug_state) = linear_debug_state.as_mut()
                    && should_run_linear_debug(debug_state.options.schedule, iteration, true)
                    && let InteriorPointSolveError::LinearSolve { solver, context } = &error
                    && let Some(diagnostics) = context.failed_linear_solve.as_ref()
                {
                    current_snapshot.linear_solver = *solver;
                    let regularization = diagnostics
                        .attempts
                        .last()
                        .map_or(reduced_kkt_system.regularization, |attempt| {
                            attempt.regularization
                        });
                    if let Ok(snapshot) = build_interior_point_kkt_snapshot(
                        iteration,
                        current_snapshot.phase,
                        *solver,
                        &reduced_kkt_system,
                        regularization,
                        sparse_hessian_diagonal_shift(reduced_kkt_system.hessian, regularization),
                        barrier_parameter_value,
                        primal_inf,
                        dual_inf,
                        complementarity_inf,
                        0,
                    ) {
                        let report = run_linear_debug_report_on_failure(
                            &snapshot,
                            *solver,
                            &diagnostics.attempts,
                            debug_state,
                        );
                        dump_linear_debug_snapshot(&debug_state.options, &snapshot, &report);
                        current_snapshot.linear_debug = Some(report.clone());
                        error = with_linear_debug_report(error, report);
                    }
                }
                return Err(with_interior_point_failure_profiling(
                    error,
                    Some(snapshot_with_nlip_events(
                        current_snapshot.clone(),
                        &iteration_events,
                    )),
                    last_accepted_state.clone(),
                    &profiling,
                    solve_started,
                ));
            }
        };
        last_linear_solver = direction.solver_used;
        let linear_elapsed = linear_started.elapsed();
        profiling.linear_solves += 1;
        profiling.linear_solve_time += linear_elapsed;
        iteration_linear_solve_time += linear_elapsed;
        if direction.regularization_used > kkt_regularization * (1.0 + 1e-12) {
            push_unique_nlip_event(
                &mut iteration_events,
                InteriorPointIterationEvent::AdaptiveRegularizationUsed,
            );
        }

        let (alpha_pr, alpha_pr_limiter) = if augmented_inequality_count > 0 {
            fraction_to_boundary_with_limiter(&slack, &direction.ds, options.fraction_to_boundary)
        } else {
            (1.0, None)
        };
        let (alpha_du, alpha_du_limiter) = if augmented_inequality_count > 0 {
            fraction_to_boundary_with_limiter(&z, &direction.dz, options.fraction_to_boundary)
        } else {
            (1.0, None)
        };
        let current_direction_diagnostics = Some(interior_point_direction_diagnostics(
            &direction,
            alpha_pr_limiter.clone(),
            alpha_du_limiter.clone(),
        ));
        let watchdog_active =
            watchdog_state.reference.is_some() && watchdog_state.remaining_iters > 0;
        let watchdog_reference = watchdog_state.reference.clone();
        let dual_alpha_limit = alpha_du.clamp(0.0, 1.0);
        let mut alpha = alpha_pr.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return Err(InteriorPointSolveError::LineSearchFailed {
                merit: merit_residual(
                    primal_inf,
                    dual_inf,
                    complementarity_inf,
                    barrier_parameter_value,
                ),
                mu: barrier_parameter_value,
                step_inf_norm: step_inf_norm(&direction.dx),
                context: interior_point_failure_context(
                    Some(snapshot_with_nlip_events(
                        current_snapshot.clone(),
                        &iteration_events,
                    )),
                    last_accepted_state.clone(),
                    None,
                    Some(InteriorPointLineSearchInfo {
                        initial_alpha_pr: alpha_pr,
                        initial_alpha_du: Some(alpha_du),
                        accepted_alpha: None,
                        accepted_alpha_du: None,
                        last_tried_alpha: 0.0,
                        last_tried_alpha_du: Some(0.0),
                        backtrack_count: 0,
                        sigma,
                        current_merit: merit_residual(
                            primal_inf,
                            dual_inf,
                            complementarity_inf,
                            barrier_parameter_value,
                        ),
                        current_barrier_objective,
                        current_primal_inf: primal_inf,
                        alpha_min: options.min_step,
                        second_order_correction_attempted: false,
                        second_order_correction_used: false,
                        watchdog_active,
                        watchdog_accepted: false,
                        tiny_step: false,
                        filter_acceptance_mode: None,
                        step_kind: None,
                        step_tag: None,
                        rejected_trials: Vec::new(),
                    }),
                    current_direction_diagnostics.clone(),
                    &profiling,
                    solve_started,
                ),
            });
        }

        let current_merit = merit_residual(
            primal_inf,
            dual_inf,
            complementarity_inf,
            barrier_parameter_value,
        );
        let current_primal_inf = primal_inf;
        let barrier_directional_derivative = barrier_objective_directional_derivative(
            &state.gradient,
            &slack,
            &direction.dx,
            &direction.ds,
            barrier_parameter_value,
        );
        let filter_parameters = super::filter::FilterParameters {
            gamma_phi: options.filter_gamma_objective,
            gamma_theta: options.filter_gamma_violation,
            eta_phi: options.eta_phi,
            theta_max,
        };
        let alpha_min = calculate_filter_alpha_min(
            current_theta,
            theta_min,
            barrier_directional_derivative,
            options,
        );
        let mut line_search_iterations = 0;
        let mut accepted = None;
        let mut last_tried_alpha_pr = alpha;
        let mut last_tried_alpha_du = dual_alpha_limit.min(alpha);
        let mut rejected_trials = Vec::new();
        let mut second_order_correction_attempted = false;
        let mut second_order_correction_used = false;
        let mut watchdog_accepted = false;
        while alpha >= alpha_min {
            let trial_alpha_pr = alpha;
            let trial_alpha_du = dual_alpha_limit;
            let trial_alpha_y = alpha_for_y(trial_alpha_pr, trial_alpha_du, &direction, options);
            last_tried_alpha_pr = trial_alpha_pr;
            last_tried_alpha_du = trial_alpha_du;
            let trial_x = x
                .iter()
                .zip(direction.dx.iter())
                .map(|(value, delta)| value + trial_alpha_pr * delta)
                .collect::<Vec<_>>();
            let trial_lambda = lambda_eq
                .iter()
                .zip(direction.d_lambda.iter())
                .map(|(value, delta)| value + trial_alpha_y * delta)
                .collect::<Vec<_>>();
            let trial_slack = slack
                .iter()
                .zip(direction.ds.iter())
                .map(|(value, delta)| value + trial_alpha_pr * delta)
                .collect::<Vec<_>>();
            let trial_z = z
                .iter()
                .zip(direction.dz.iter())
                .map(|(value, delta)| value + trial_alpha_du * delta)
                .collect::<Vec<_>>();
            if trial_slack.iter().any(|value| *value <= 0.0)
                || trial_z.iter().any(|value| *value <= 0.0)
            {
                rejected_trials.push(InteriorPointLineSearchTrial {
                    alpha: trial_alpha_pr,
                    alpha_du: Some(trial_alpha_du),
                    slack_positive: trial_slack.iter().all(|value| *value > 0.0),
                    multipliers_positive: trial_z.iter().all(|value| *value > 0.0),
                    objective: None,
                    barrier_objective: None,
                    merit: None,
                    eq_inf: None,
                    ineq_inf: None,
                    primal_inf: None,
                    dual_inf: None,
                    comp_inf: None,
                    mu: None,
                    local_filter_acceptable: None,
                    filter_acceptable: None,
                    filter_dominated: None,
                    filter_sufficient_objective_reduction: None,
                    filter_sufficient_violation_reduction: None,
                    switching_condition_satisfied: None,
                });
                alpha *= options.line_search_beta;
                line_search_iterations += 1;
                continue;
            }
            let mut trial_callback_time = Duration::ZERO;
            let trial_state = trial_state(
                problem,
                &trial_x,
                parameters,
                &bounds,
                &bound_jacobian,
                &equality_jacobian_structure,
                &inequality_jacobian_structure,
                &mut profiling,
                &mut trial_callback_time,
            );
            let trial_eq_inf = inf_norm(&trial_state.equality_values);
            let trial_ineq_inf = positive_part_inf_norm(&trial_state.augmented_inequality_values);
            let trial_internal_ineq_inf = slack_form_inequality_inf_norm(
                &trial_state.augmented_inequality_values,
                &trial_slack,
            );
            let trial_dual_residual = lagrangian_gradient_sparse(
                &trial_state.gradient,
                &trial_state.equality_jacobian,
                &trial_lambda,
                &trial_state.inequality_jacobian,
                &trial_z,
            );
            let trial_dual_inf = inf_norm(&trial_dual_residual);
            let trial_comp_inf = if augmented_inequality_count > 0 {
                complementarity_inf_norm(&trial_slack, &trial_z)
            } else {
                0.0
            };
            let trial_primal_inf = trial_eq_inf.max(trial_internal_ineq_inf);
            let trial_filter_theta = filter_theta_l1_norm(
                &trial_state.equality_values,
                &trial_state.augmented_inequality_values,
                &trial_slack,
            );
            let trial_merit = merit_residual(
                trial_primal_inf,
                trial_dual_inf,
                trial_comp_inf,
                barrier_parameter_value,
            );
            let trial_all_dual_multipliers = [trial_lambda.as_slice(), trial_z.as_slice()].concat();
            let trial_overall_inf = scaled_overall_inf_norm(
                trial_primal_inf,
                trial_dual_inf,
                trial_comp_inf,
                &trial_all_dual_multipliers,
                &trial_z,
                options.overall_scale_max,
            );
            let corrected_bound_multipliers = apply_bound_multiplier_safeguard(
                &trial_state,
                trial_primal_inf,
                &trial_lambda,
                &trial_slack,
                &trial_z,
                barrier_parameter_value,
                options,
            );
            let trial_barrier_objective = barrier_objective_value(
                trial_state.objective_value,
                &trial_slack,
                barrier_parameter_value,
            );
            let trial_filter_entry =
                super::filter::entry(trial_barrier_objective, trial_filter_theta);
            let switching_condition = switching_condition_satisfied(
                current_theta,
                barrier_directional_derivative,
                trial_alpha_pr,
                options,
            );
            let armijo_required =
                trial_alpha_pr > 0.0 && switching_condition && current_theta <= theta_min;
            let filter_assessment = super::filter::assess_trial(
                &filter_entries,
                &current_filter_entry,
                &trial_filter_entry,
                alpha,
                barrier_directional_derivative,
                switching_condition,
                armijo_required,
                filter_parameters,
            );
            let barrier_objective_too_large = barrier_objective_increase_too_large(
                current_barrier_objective,
                trial_barrier_objective,
                options.obj_max_inc,
            );
            let filter_acceptance_mode = (!barrier_objective_too_large)
                .then_some(filter_assessment.acceptance_mode)
                .flatten();
            if filter_acceptance_mode.is_some() {
                let step_kind =
                    if filter_acceptance_mode == Some(FilterAcceptanceMode::ObjectiveArmijo) {
                        InteriorPointStepKind::Objective
                    } else {
                        InteriorPointStepKind::Feasibility
                    };
                let step_tag = if step_kind == InteriorPointStepKind::Feasibility {
                    'h'
                } else {
                    'f'
                };
                let (
                    accepted_z,
                    accepted_dual_inf,
                    accepted_comp_inf,
                    accepted_mu,
                    accepted_overall_inf,
                    bound_multiplier_corrected,
                ) = if let Some(corrected) = corrected_bound_multipliers {
                    (
                        corrected.z,
                        corrected.dual_inf,
                        corrected.complementarity_inf,
                        barrier_parameter_value,
                        corrected.overall_inf,
                        true,
                    )
                } else {
                    (
                        trial_z,
                        trial_dual_inf,
                        trial_comp_inf,
                        barrier_parameter_value,
                        trial_overall_inf,
                        false,
                    )
                };
                accepted = Some(AcceptedInteriorPointTrial {
                    x: trial_x,
                    lambda: trial_lambda,
                    slack: trial_slack,
                    z: accepted_z,
                    objective: trial_state.objective_value,
                    barrier_objective: trial_barrier_objective,
                    equality_inf: trial_eq_inf,
                    inequality_inf: trial_ineq_inf,
                    dual_inf: accepted_dual_inf,
                    complementarity_inf: accepted_comp_inf,
                    overall_inf: accepted_overall_inf,
                    mu: accepted_mu,
                    filter_theta: trial_filter_theta,
                    filter_entry: trial_filter_entry.clone(),
                    filter_augment_entry: (step_kind == InteriorPointStepKind::Feasibility).then(
                        || {
                            super::filter::augment_entry(
                                current_barrier_objective,
                                current_theta,
                                options.filter_gamma_objective,
                                options.filter_gamma_violation,
                            )
                        },
                    ),
                    filter_acceptance_mode,
                    step_kind,
                    step_tag,
                    phase: InteriorPointIterationPhase::AcceptedStep,
                    accepted_alpha_pr: trial_alpha_pr,
                    accepted_alpha_du: Some(trial_alpha_du),
                    line_search_initial_alpha_pr: alpha_pr,
                    line_search_initial_alpha_du: Some(alpha_du),
                    line_search_last_alpha_pr: trial_alpha_pr,
                    line_search_last_alpha_du: Some(trial_alpha_du),
                    line_search_backtrack_count: line_search_iterations,
                    second_order_correction_used: false,
                    watchdog_accepted: false,
                    bound_multiplier_corrected,
                });
                break;
            }
            if options.second_order_correction
                && !second_order_correction_attempted
                && trial_primal_inf > current_primal_inf.max(options.constraint_tol)
                && let Some(correction) = sparse_second_order_correction_step(
                    &trial_state.equality_jacobian,
                    &trial_state.inequality_jacobian,
                    &trial_state.equality_values,
                    &trial_state.augmented_inequality_values,
                    &trial_z,
                    options.constraint_tol,
                )
            {
                second_order_correction_attempted = true;
                let corrected_x = trial_x
                    .iter()
                    .zip(correction.iter())
                    .map(|(value, delta)| value + delta)
                    .collect::<Vec<_>>();
                let corrected_lambda = trial_lambda.clone();
                let mut corrected_callback_time = Duration::ZERO;
                let corrected_state = self::trial_state(
                    problem,
                    &corrected_x,
                    parameters,
                    &bounds,
                    &bound_jacobian,
                    &equality_jacobian_structure,
                    &inequality_jacobian_structure,
                    &mut profiling,
                    &mut corrected_callback_time,
                );
                let corrected_slack = if augmented_inequality_count > 0 {
                    corrected_state
                        .augmented_inequality_values
                        .iter()
                        .map(|value| (-value).max(1e-8))
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let corrected_z = trial_z.clone();
                let corrected_eq_inf = inf_norm(&corrected_state.equality_values);
                let corrected_ineq_inf =
                    positive_part_inf_norm(&corrected_state.augmented_inequality_values);
                let corrected_primal_inf = corrected_eq_inf.max(slack_form_inequality_inf_norm(
                    &corrected_state.augmented_inequality_values,
                    &corrected_slack,
                ));
                let corrected_dual_residual = lagrangian_gradient_sparse(
                    &corrected_state.gradient,
                    &corrected_state.equality_jacobian,
                    &corrected_lambda,
                    &corrected_state.inequality_jacobian,
                    &corrected_z,
                );
                let corrected_dual_inf = inf_norm(&corrected_dual_residual);
                let corrected_comp_inf = if augmented_inequality_count > 0 {
                    complementarity_inf_norm(&corrected_slack, &corrected_z)
                } else {
                    0.0
                };
                let corrected_filter_theta = filter_theta_l1_norm(
                    &corrected_state.equality_values,
                    &corrected_state.augmented_inequality_values,
                    &corrected_slack,
                );
                let corrected_barrier_objective = barrier_objective_value(
                    corrected_state.objective_value,
                    &corrected_slack,
                    barrier_parameter_value,
                );
                let corrected_filter_entry =
                    super::filter::entry(corrected_barrier_objective, corrected_filter_theta);
                let corrected_switching_condition = switching_condition_satisfied(
                    current_theta,
                    barrier_directional_derivative,
                    trial_alpha_pr,
                    options,
                );
                let corrected_armijo_required = trial_alpha_pr > 0.0
                    && corrected_switching_condition
                    && current_theta <= theta_min;
                let corrected_filter_assessment = super::filter::assess_trial(
                    &filter_entries,
                    &current_filter_entry,
                    &corrected_filter_entry,
                    trial_alpha_pr,
                    barrier_directional_derivative,
                    corrected_switching_condition,
                    corrected_armijo_required,
                    filter_parameters,
                );
                let corrected_barrier_objective_too_large = barrier_objective_increase_too_large(
                    current_barrier_objective,
                    corrected_barrier_objective,
                    options.obj_max_inc,
                );
                let corrected_filter_acceptance_mode = (!corrected_barrier_objective_too_large)
                    .then_some(corrected_filter_assessment.acceptance_mode)
                    .flatten();
                if corrected_filter_acceptance_mode.is_some() {
                    let step_kind = if corrected_filter_acceptance_mode
                        == Some(FilterAcceptanceMode::ObjectiveArmijo)
                    {
                        InteriorPointStepKind::Objective
                    } else {
                        InteriorPointStepKind::Feasibility
                    };
                    let step_tag = if step_kind == InteriorPointStepKind::Feasibility {
                        'h'
                    } else {
                        'f'
                    };
                    let corrected_all_dual_multipliers =
                        [corrected_lambda.as_slice(), corrected_z.as_slice()].concat();
                    let corrected_overall_inf = scaled_overall_inf_norm(
                        corrected_primal_inf,
                        corrected_dual_inf,
                        corrected_comp_inf,
                        &corrected_all_dual_multipliers,
                        &corrected_z,
                        options.overall_scale_max,
                    );
                    let corrected_bound_multipliers = apply_bound_multiplier_safeguard(
                        &corrected_state,
                        corrected_primal_inf,
                        &corrected_lambda,
                        &corrected_slack,
                        &corrected_z,
                        barrier_parameter_value,
                        options,
                    );
                    second_order_correction_used = true;
                    let (
                        accepted_z,
                        accepted_dual_inf,
                        accepted_comp_inf,
                        accepted_mu,
                        accepted_overall_inf,
                        bound_multiplier_corrected,
                    ) = if let Some(corrected) = corrected_bound_multipliers {
                        (
                            corrected.z,
                            corrected.dual_inf,
                            corrected.complementarity_inf,
                            barrier_parameter_value,
                            corrected.overall_inf,
                            true,
                        )
                    } else {
                        (
                            corrected_z,
                            corrected_dual_inf,
                            corrected_comp_inf,
                            barrier_parameter_value,
                            corrected_overall_inf,
                            false,
                        )
                    };
                    accepted = Some(AcceptedInteriorPointTrial {
                        x: corrected_x,
                        lambda: corrected_lambda,
                        slack: corrected_slack,
                        z: accepted_z,
                        objective: corrected_state.objective_value,
                        barrier_objective: corrected_barrier_objective,
                        equality_inf: corrected_eq_inf,
                        inequality_inf: corrected_ineq_inf,
                        dual_inf: accepted_dual_inf,
                        complementarity_inf: accepted_comp_inf,
                        overall_inf: accepted_overall_inf,
                        mu: accepted_mu,
                        filter_theta: corrected_filter_theta,
                        filter_entry: corrected_filter_entry.clone(),
                        filter_augment_entry: (step_kind == InteriorPointStepKind::Feasibility)
                            .then(|| {
                                super::filter::augment_entry(
                                    current_barrier_objective,
                                    current_theta,
                                    options.filter_gamma_objective,
                                    options.filter_gamma_violation,
                                )
                            }),
                        filter_acceptance_mode: corrected_filter_acceptance_mode,
                        step_kind,
                        step_tag,
                        phase: InteriorPointIterationPhase::AcceptedStep,
                        accepted_alpha_pr: trial_alpha_pr,
                        accepted_alpha_du: Some(trial_alpha_du),
                        line_search_initial_alpha_pr: alpha_pr,
                        line_search_initial_alpha_du: Some(alpha_du),
                        line_search_last_alpha_pr: trial_alpha_pr,
                        line_search_last_alpha_du: Some(trial_alpha_du),
                        line_search_backtrack_count: line_search_iterations,
                        second_order_correction_used: true,
                        watchdog_accepted: false,
                        bound_multiplier_corrected,
                    });
                    break;
                }
            }
            let watchdog_accept = watchdog_active
                && watchdog_reference.as_ref().is_some_and(|reference| {
                    let watchdog_filter_entry =
                        super::filter::entry(reference.barrier_objective, reference.filter_theta);
                    let watchdog_switching_condition = switching_condition_satisfied(
                        reference.filter_theta,
                        reference.barrier_directional_derivative,
                        trial_alpha_pr,
                        options,
                    );
                    let watchdog_armijo_required = trial_alpha_pr > 0.0
                        && watchdog_switching_condition
                        && reference.filter_theta <= theta_min;
                    let watchdog_assessment = super::filter::assess_trial(
                        &filter_entries,
                        &watchdog_filter_entry,
                        &trial_filter_entry,
                        trial_alpha_pr,
                        reference.barrier_directional_derivative,
                        watchdog_switching_condition,
                        watchdog_armijo_required,
                        filter_parameters,
                    );
                    !barrier_objective_increase_too_large(
                        reference.barrier_objective,
                        trial_barrier_objective,
                        options.obj_max_inc,
                    ) && watchdog_assessment.acceptance_mode.is_some()
                });
            if watchdog_accept {
                let step_kind = if let Some(reference) = &watchdog_reference {
                    let watchdog_filter_entry =
                        super::filter::entry(reference.barrier_objective, reference.filter_theta);
                    let watchdog_switching_condition = switching_condition_satisfied(
                        reference.filter_theta,
                        reference.barrier_directional_derivative,
                        trial_alpha_pr,
                        options,
                    );
                    let watchdog_armijo_required = trial_alpha_pr > 0.0
                        && watchdog_switching_condition
                        && reference.filter_theta <= theta_min;
                    let watchdog_assessment = super::filter::assess_trial(
                        &filter_entries,
                        &watchdog_filter_entry,
                        &trial_filter_entry,
                        trial_alpha_pr,
                        reference.barrier_directional_derivative,
                        watchdog_switching_condition,
                        watchdog_armijo_required,
                        filter_parameters,
                    );
                    if watchdog_assessment.acceptance_mode
                        == Some(FilterAcceptanceMode::ObjectiveArmijo)
                    {
                        InteriorPointStepKind::Objective
                    } else {
                        InteriorPointStepKind::Feasibility
                    }
                } else {
                    InteriorPointStepKind::Feasibility
                };
                watchdog_accepted = true;
                let step_tag = if step_kind == InteriorPointStepKind::Feasibility {
                    'h'
                } else {
                    'f'
                };
                let corrected_bound_multipliers = apply_bound_multiplier_safeguard(
                    &trial_state,
                    trial_primal_inf,
                    &trial_lambda,
                    &trial_slack,
                    &trial_z,
                    barrier_parameter_value,
                    options,
                );
                let (
                    accepted_z,
                    accepted_dual_inf,
                    accepted_comp_inf,
                    accepted_mu,
                    accepted_overall_inf,
                    bound_multiplier_corrected,
                ) = if let Some(corrected) = corrected_bound_multipliers {
                    (
                        corrected.z,
                        corrected.dual_inf,
                        corrected.complementarity_inf,
                        barrier_parameter_value,
                        corrected.overall_inf,
                        true,
                    )
                } else {
                    (
                        trial_z,
                        trial_dual_inf,
                        trial_comp_inf,
                        barrier_parameter_value,
                        trial_overall_inf,
                        false,
                    )
                };
                accepted = Some(AcceptedInteriorPointTrial {
                    x: trial_x,
                    lambda: trial_lambda,
                    slack: trial_slack,
                    z: accepted_z,
                    objective: trial_state.objective_value,
                    barrier_objective: trial_barrier_objective,
                    equality_inf: trial_eq_inf,
                    inequality_inf: trial_ineq_inf,
                    dual_inf: accepted_dual_inf,
                    complementarity_inf: accepted_comp_inf,
                    overall_inf: accepted_overall_inf,
                    mu: accepted_mu,
                    filter_theta: trial_filter_theta,
                    filter_entry: trial_filter_entry.clone(),
                    filter_augment_entry: (step_kind == InteriorPointStepKind::Feasibility).then(
                        || {
                            super::filter::augment_entry(
                                current_barrier_objective,
                                current_theta,
                                options.filter_gamma_objective,
                                options.filter_gamma_violation,
                            )
                        },
                    ),
                    filter_acceptance_mode: None,
                    step_kind,
                    step_tag,
                    phase: InteriorPointIterationPhase::AcceptedStep,
                    accepted_alpha_pr: trial_alpha_pr,
                    accepted_alpha_du: Some(trial_alpha_du),
                    line_search_initial_alpha_pr: alpha_pr,
                    line_search_initial_alpha_du: Some(alpha_du),
                    line_search_last_alpha_pr: trial_alpha_pr,
                    line_search_last_alpha_du: Some(trial_alpha_du),
                    line_search_backtrack_count: line_search_iterations,
                    second_order_correction_used: false,
                    watchdog_accepted: true,
                    bound_multiplier_corrected,
                });
                break;
            }
            rejected_trials.push(InteriorPointLineSearchTrial {
                alpha: trial_alpha_pr,
                alpha_du: Some(trial_alpha_du),
                slack_positive: true,
                multipliers_positive: true,
                objective: Some(trial_state.objective_value),
                barrier_objective: Some(trial_barrier_objective),
                merit: Some(trial_merit),
                eq_inf: Some(trial_eq_inf),
                ineq_inf: Some(trial_ineq_inf),
                primal_inf: Some(trial_primal_inf),
                dual_inf: Some(trial_dual_inf),
                comp_inf: Some(trial_comp_inf),
                mu: Some(barrier_parameter_value),
                local_filter_acceptable: Some(filter_assessment.current_iterate_acceptable),
                filter_acceptable: Some(
                    filter_assessment.filter_acceptable && !barrier_objective_too_large,
                ),
                filter_dominated: Some(filter_assessment.filter_dominated),
                filter_sufficient_objective_reduction: Some(
                    filter_assessment.filter_sufficient_objective_reduction,
                ),
                filter_sufficient_violation_reduction: Some(
                    filter_assessment.filter_sufficient_violation_reduction,
                ),
                switching_condition_satisfied: Some(switching_condition),
            });
            alpha *= options.line_search_beta;
            line_search_iterations += 1;
        }
        let accepted_direction_diagnostics = current_direction_diagnostics.clone();
        let accepted_regularization_size = Some(direction.regularization_used);
        let Some(accepted_trial) = accepted else {
            return Err(InteriorPointSolveError::LineSearchFailed {
                merit: current_merit,
                mu: barrier_parameter_value,
                step_inf_norm: step_inf_norm(&direction.dx),
                context: interior_point_failure_context(
                    Some(snapshot_with_nlip_events(
                        current_snapshot.clone(),
                        &iteration_events,
                    )),
                    last_accepted_state.clone(),
                    None,
                    Some(InteriorPointLineSearchInfo {
                        initial_alpha_pr: alpha_pr,
                        initial_alpha_du: Some(alpha_du),
                        accepted_alpha: None,
                        accepted_alpha_du: None,
                        last_tried_alpha: last_tried_alpha_pr,
                        last_tried_alpha_du: Some(last_tried_alpha_du),
                        backtrack_count: line_search_iterations,
                        sigma,
                        current_merit,
                        current_barrier_objective,
                        current_primal_inf,
                        alpha_min,
                        second_order_correction_attempted,
                        second_order_correction_used,
                        watchdog_active,
                        watchdog_accepted,
                        tiny_step: false,
                        filter_acceptance_mode: None,
                        step_kind: None,
                        step_tag: None,
                        rejected_trials,
                    }),
                    current_direction_diagnostics.clone(),
                    &profiling,
                    solve_started,
                ),
            });
        };
        let accepted_step_inf =
            accepted_ip_step_inf_norm(&direction, accepted_trial.accepted_alpha_pr);
        let tiny_step = is_tiny_ip_step(accepted_step_inf, options);
        let shortened_step = is_shortened_ip_step(
            accepted_trial.accepted_alpha_pr,
            accepted_trial.accepted_alpha_du,
            accepted_trial.line_search_backtrack_count,
        );
        let next_shortened_step_streak = if shortened_step {
            watchdog_state.shortened_step_streak + 1
        } else {
            0
        };
        let watchdog_will_arm = !accepted_trial.watchdog_accepted
            && options.watchdog_trial_iter_max > 0
            && options.watchdog_shortened_iter_trigger > 0
            && next_shortened_step_streak >= options.watchdog_shortened_iter_trigger;

        let accepted_rejected_trials = rejected_trials.clone();
        let last_rejection_due_to_filter = accepted_rejected_trials.last().is_some_and(|trial| {
            trial.local_filter_acceptable == Some(true) && trial.filter_acceptable == Some(false)
        });
        let line_search_info = InteriorPointLineSearchInfo {
            initial_alpha_pr: accepted_trial.line_search_initial_alpha_pr,
            initial_alpha_du: accepted_trial.line_search_initial_alpha_du,
            accepted_alpha: Some(accepted_trial.accepted_alpha_pr),
            accepted_alpha_du: accepted_trial.accepted_alpha_du,
            last_tried_alpha: accepted_trial.line_search_last_alpha_pr,
            last_tried_alpha_du: accepted_trial.line_search_last_alpha_du,
            backtrack_count: accepted_trial.line_search_backtrack_count,
            sigma,
            current_merit,
            current_barrier_objective,
            current_primal_inf,
            alpha_min,
            second_order_correction_attempted,
            second_order_correction_used: accepted_trial.second_order_correction_used,
            watchdog_active,
            watchdog_accepted: accepted_trial.watchdog_accepted,
            tiny_step,
            filter_acceptance_mode: accepted_trial.filter_acceptance_mode,
            step_kind: Some(accepted_trial.step_kind),
            step_tag: Some(accepted_trial.step_tag),
            rejected_trials: accepted_rejected_trials,
        };

        let mut events = iteration_events.clone();
        if line_search_iterations >= 4 {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::LongLineSearch);
        }
        if accepted_trial.filter_acceptance_mode == Some(FilterAcceptanceMode::ViolationReduction) {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::FilterAccepted);
        }
        if second_order_correction_attempted {
            push_unique_nlip_event(
                &mut events,
                InteriorPointIterationEvent::SecondOrderCorrectionAttempted,
            );
        }
        if accepted_trial.second_order_correction_used {
            push_unique_nlip_event(
                &mut events,
                InteriorPointIterationEvent::SecondOrderCorrectionAccepted,
            );
        }
        if accepted_trial.watchdog_accepted {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::WatchdogActivated);
        }
        if accepted_trial.bound_multiplier_corrected {
            push_unique_nlip_event(
                &mut events,
                InteriorPointIterationEvent::BoundMultiplierSafeguardApplied,
            );
        }
        if watchdog_will_arm {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::WatchdogArmed);
        }
        if tiny_step {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::TinyStep);
        }
        if iteration + 1 == options.max_iters {
            push_unique_nlip_event(
                &mut events,
                InteriorPointIterationEvent::MaxIterationsReached,
            );
        }
        let mut next_filter_entries = filter_entries.clone();
        if let Some(entry) = accepted_trial.filter_augment_entry.clone() {
            super::filter::update_frontier(&mut next_filter_entries, entry);
        }
        let filter_reset_applied =
            if options.max_filter_resets > 0 && filter_reset_count < options.max_filter_resets {
                if last_rejection_due_to_filter {
                    successive_filter_rejections += 1;
                    if successive_filter_rejections >= options.filter_reset_trigger {
                        next_filter_entries.clear();
                        filter_reset_count += 1;
                        successive_filter_rejections = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    successive_filter_rejections = 0;
                    false
                }
            } else {
                false
            };
        let previous_barrier_parameter = barrier_parameter_value;
        let mut next_barrier_parameter_value = barrier_parameter_value;
        if augmented_inequality_count > 0 {
            next_barrier_parameter_value = next_barrier_parameter(
                barrier_parameter_value,
                accepted_trial.overall_inf,
                tiny_step,
                options,
            );
        }
        let barrier_parameter_updated = next_barrier_parameter_value
            < previous_barrier_parameter - 1e-18 * previous_barrier_parameter.abs().max(1.0);
        if barrier_parameter_updated {
            next_filter_entries.clear();
        }

        if barrier_parameter_updated {
            push_unique_nlip_event(
                &mut events,
                InteriorPointIterationEvent::BarrierParameterUpdated,
            );
        }
        if filter_reset_applied {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::FilterReset);
        }
        let adapter_timing = adapter_timing_delta(problem, &mut last_adapter_timing);
        profiling.adapter_timing = last_adapter_timing;
        let iteration_total = iteration_started.elapsed();
        let iteration_preprocess = iteration_total.saturating_sub(
            iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
        );
        if options.verbose {
            let flags = InteriorPointIterationLogFlags {
                has_equalities: equality_count > 0,
                has_inequalities: augmented_inequality_count > 0,
                filter_accepted: accepted_trial.filter_acceptance_mode
                    == Some(FilterAcceptanceMode::ViolationReduction),
                soc_attempted: second_order_correction_attempted,
                soc_used: accepted_trial.second_order_correction_used,
                watchdog_active: accepted_trial.watchdog_accepted,
                tiny_step,
                iteration_limit_reached: iteration + 1 == options.max_iters,
            };
            log_interior_point_iteration(
                &InteriorPointIterationLog {
                    iteration,
                    phase: accepted_trial.phase,
                    flags,
                    extra_events: events.clone(),
                    display_mode: InteriorPointDisplayMode::strict(options),
                    objective_value: accepted_trial.objective,
                    barrier_objective: accepted_trial.barrier_objective,
                    equality_inf: accepted_trial.equality_inf,
                    inequality_inf: accepted_trial.inequality_inf,
                    dual_inf: accepted_trial.dual_inf,
                    complementarity_inf: accepted_trial.complementarity_inf,
                    overall_inf: accepted_trial.overall_inf,
                    barrier_parameter: if augmented_inequality_count > 0 {
                        accepted_trial.mu
                    } else {
                        0.0
                    },
                    alpha: Some(accepted_trial.accepted_alpha_pr),
                    alpha_pr: Some(accepted_trial.accepted_alpha_pr),
                    alpha_du: accepted_trial.accepted_alpha_du,
                    line_search_iterations: Some(accepted_trial.line_search_backtrack_count),
                    regularization_size: accepted_regularization_size,
                    step_kind: Some(accepted_trial.step_kind),
                    step_tag: Some(accepted_trial.step_tag),
                    linear_time_secs: Some(
                        profiling.linear_solve_time.as_secs_f64()
                            / profiling.linear_solves.max(1) as f64,
                    ),
                },
                &mut event_state,
            );
        }
        let accepted_snapshot = InteriorPointIterationSnapshot {
            iteration,
            phase: accepted_trial.phase,
            x: accepted_trial.x.clone(),
            objective: accepted_trial.objective,
            barrier_objective: Some(accepted_trial.barrier_objective),
            eq_inf: (equality_count > 0).then_some(accepted_trial.equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(accepted_trial.inequality_inf),
            dual_inf: accepted_trial.dual_inf,
            comp_inf: (augmented_inequality_count > 0)
                .then_some(accepted_trial.complementarity_inf),
            overall_inf: accepted_trial.overall_inf,
            barrier_parameter: (augmented_inequality_count > 0).then_some(accepted_trial.mu),
            filter_theta: Some(accepted_trial.filter_theta),
            step_inf: Some(accepted_step_inf),
            alpha: Some(accepted_trial.accepted_alpha_pr),
            alpha_pr: Some(accepted_trial.accepted_alpha_pr),
            alpha_du: accepted_trial.accepted_alpha_du,
            line_search_iterations: Some(accepted_trial.line_search_backtrack_count),
            line_search_trials: accepted_trial.line_search_backtrack_count,
            regularization_size: accepted_regularization_size,
            step_kind: Some(accepted_trial.step_kind),
            step_tag: Some(accepted_trial.step_tag),
            watchdog_active: accepted_trial.watchdog_accepted || watchdog_active,
            line_search: Some(line_search_info.clone()),
            direction_diagnostics: accepted_direction_diagnostics.clone(),
            linear_debug: direction.linear_debug.clone(),
            linear_solver: direction.solver_used,
            linear_solve_time: Some(iteration_linear_solve_time),
            filter: Some(FilterInfo {
                current: accepted_trial.filter_entry.clone(),
                entries: next_filter_entries.clone(),
                accepted_mode: accepted_trial.filter_acceptance_mode,
            }),
            timing: InteriorPointIterationTiming {
                adapter_timing,
                callback: iteration_callback_time,
                kkt_assembly: iteration_kkt_assembly_time,
                linear_solve: iteration_linear_solve_time,
                preprocess: iteration_preprocess,
                total: iteration_total,
            },
            events: events.clone(),
        };
        snapshots.push(accepted_snapshot.clone());
        callback(&accepted_snapshot);
        last_accepted_state = Some(accepted_snapshot);

        profiling.preprocessing_steps += 1;
        profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
            iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
        );
        x = accepted_trial.x;
        lambda_eq = accepted_trial.lambda;
        slack = accepted_trial.slack;
        z = accepted_trial.z;
        filter_entries = next_filter_entries;
        if augmented_inequality_count > 0 {
            barrier_parameter_value = next_barrier_parameter_value;
        }
        let (nonlinear, _, _) =
            split_augmented_inequality_multipliers(&z, inequality_count, lower_bound_count);
        nonlinear_inequality_multipliers = nonlinear;
        if barrier_parameter_updated {
            watchdog_state = InteriorPointWatchdogState::default();
            continue;
        }
        if shortened_step {
            watchdog_state.shortened_step_streak += 1;
        } else {
            watchdog_state.shortened_step_streak = 0;
        }
        if tiny_step {
            watchdog_state.tiny_step_streak += 1;
        } else {
            watchdog_state.tiny_step_streak = 0;
        }
        if accepted_trial.watchdog_accepted {
            if watchdog_state.remaining_iters > 0 {
                watchdog_state.remaining_iters -= 1;
            }
            let keep_watchdog = watchdog_state.reference.as_ref().is_some_and(|reference| {
                accepted_trial.filter_theta
                    >= reference.filter_theta
                        - 1e-12
                            * reference
                                .filter_theta
                                .abs()
                                .max(accepted_trial.filter_theta.abs())
                                .max(1.0)
                    && watchdog_state.remaining_iters > 0
            });
            if !keep_watchdog {
                watchdog_state.reference = None;
                watchdog_state.remaining_iters = 0;
            }
        } else if options.watchdog_trial_iter_max > 0
            && options.watchdog_shortened_iter_trigger > 0
            && watchdog_state.shortened_step_streak >= options.watchdog_shortened_iter_trigger
        {
            watchdog_state.reference = Some(WatchdogReferencePoint {
                barrier_objective: accepted_trial.barrier_objective,
                filter_theta: accepted_trial.filter_theta,
                barrier_directional_derivative,
            });
            watchdog_state.remaining_iters = options.watchdog_trial_iter_max;
        } else if !shortened_step && !tiny_step {
            watchdog_state.reference = None;
            watchdog_state.remaining_iters = 0;
        }
    }

    Err(InteriorPointSolveError::MaxIterations {
        iterations: options.max_iters,
        context: interior_point_failure_context(
            last_accepted_state.clone(),
            last_accepted_state.clone(),
            None,
            None,
            None,
            &profiling,
            solve_started,
        ),
    })
}
