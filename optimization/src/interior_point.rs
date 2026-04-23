use clarabel::algebra::CscMatrix;
use clarabel::qdldl::{QDLDLFactorisation, QDLDLSettings};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use ssids_rs::{
    AnalyseInfo as SpralAnalyseInfo, Inertia as SpralInertia,
    NativeOrdering as SpralNativeOrdering, NativeSpral as NativeSpralLibrary, NativeSpralSession,
    NumericFactor as SpralNumericFactor, NumericFactorOptions as SpralNumericFactorOptions,
    OrderingStrategy as SpralOrderingStrategy, PivotMethod as SpralPivotMethod,
    SsidsOptions as SpralSsidsOptions, SymbolicFactor as SpralSymbolicFactor,
    SymmetricCscMatrix as SpralSymmetricCscMatrix, analyse as spral_analyse,
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
    BackendTimingMetadata, BoundConstraints, CCS, CompiledNlpProblem, ConstraintBounds,
    DUAL_INF_LABEL, EQ_INF_LABEL, EvalTimingStat, FilterAcceptanceMode, FilterInfo, INEQ_INF_LABEL,
    Index, OVERALL_INF_LABEL, PRIMAL_INF_LABEL, ParameterMatrix, SolverAdapterTiming,
    SqpEventLegendState, boxed_line, choose_summary_duration_unit, compact_duration_text,
    complementarity_inf_norm, declared_box_constraint_count, dense_fill_percent,
    fmt_duration_in_unit, fmt_optional_duration_in_unit, inf_norm, log_boxed_section,
    lower_tri_fill_percent, positive_part_inf_norm, scaled_overall_inf_norm, sci_text, style_bold,
    style_cyan_bold, style_green_bold, style_iteration_label_cell, style_metric_against_tolerance,
    style_red_bold, style_yellow_bold, time_callback, validate_nlp_problem_shapes,
    validate_parameter_inputs,
};

const IP_COMP_INF_LABEL: &str = "‖s∘z‖∞";

const LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM: f64 = 1e12;
const LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL: f64 = 1e-7;
const LINEAR_DEBUG_RELATIVE_DELTA_TOLERANCE: f64 = 1e-8;
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointLinearSolver {
    Auto,
    SsidsRs,
    SpralSrc,
    SparseQdldl,
}

impl InteriorPointLinearSolver {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::SsidsRs => "ssids_rs",
            Self::SpralSrc => "spral_src",
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
    pub bound_relax_factor: f64,
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
    pub kappa_d: f64,
    pub kappa_sigma: f64,
    pub regularization: f64,
    pub first_hessian_perturbation: f64,
    pub regularization_first_growth_factor: f64,
    pub adaptive_regularization_retries: Index,
    pub regularization_growth_factor: f64,
    pub regularization_decay_factor: f64,
    pub regularization_max: f64,
    pub jacobian_regularization_value: f64,
    pub jacobian_regularization_exponent: f64,
    pub second_order_correction: bool,
    pub max_second_order_corrections: Index,
    pub second_order_correction_reduction_factor: f64,
    pub restoration_phase: bool,
    pub tiny_step_tol: f64,
    pub tiny_step_y_tol: f64,
    pub watchdog_shortened_iter_trigger: Index,
    pub watchdog_trial_iter_max: Index,
    pub mu_min: f64,
    pub linear_solver: InteriorPointLinearSolver,
    pub spral_pivot_method: InteriorPointSpralPivotMethod,
    pub spral_action_on_zero_pivot: bool,
    pub spral_small_pivot_tolerance: f64,
    pub spral_threshold_pivot_u: f64,
    pub spral_pivot_tolerance_max: f64,
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
            fraction_to_boundary: 0.99,
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
            bound_relax_factor: 1e-8,
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
            kappa_d: 1e-5,
            kappa_sigma: 1e10,
            regularization: 1e-6,
            first_hessian_perturbation: 1e-4,
            regularization_first_growth_factor: 100.0,
            adaptive_regularization_retries: 30,
            regularization_growth_factor: 8.0,
            regularization_decay_factor: 1.0 / 3.0,
            regularization_max: 1e20,
            jacobian_regularization_value: 1e-8,
            jacobian_regularization_exponent: 0.25,
            second_order_correction: true,
            max_second_order_corrections: 4,
            second_order_correction_reduction_factor: 0.99,
            restoration_phase: true,
            tiny_step_tol: 10.0 * f64::EPSILON,
            tiny_step_y_tol: 1e-2,
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 3,
            mu_min: 1e-12,
            linear_solver: InteriorPointLinearSolver::SsidsRs,
            spral_pivot_method: InteriorPointSpralPivotMethod::BlockAposteriori,
            spral_action_on_zero_pivot: true,
            spral_small_pivot_tolerance: 1e-20,
            spral_threshold_pivot_u: 1e-8,
            spral_pivot_tolerance_max: 1e-4,
            linear_debug: None,
            verbose: true,
        }
    }
}

pub fn format_nlip_settings_summary(options: &InteriorPointOptions) -> String {
    format!(
        "filter={}; linear_solver={}; linear_debug={}; spral=[pivot={}, action={}, small={}, u={}, umax={}]; beta={}; c1={}; min_step={}; tau={}; alpha_y=[strategy={}, tol={}] ; init=[bound_push={}, bound_frac={}, slack_push={}, slack_frac={}, bound_relax={}]; dual_init=[method={}, val={}, least_square={}, max={}] ; regularization={} (first={}, first_growth={}, retries={}, growth={}, decay={}, max={}, jacobian={}, jac_exp={}); soc={} (max={}, kappa={}); restoration={}; watchdog=[trigger={}, max={}]; filter_reset=[max={}, trigger={}]; tiny_step=[x={}, y={}]; mu=[init={}, target={}, min={}, barrier_tol={}, linear={}, superlinear={}, fast={}, kappa_d={}]; theta=[{}, {}]; acceptable_iter={}",
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
        sci_text(options.spral_pivot_tolerance_max),
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
        sci_text(options.bound_relax_factor),
        options.bound_mult_init_method.label(),
        sci_text(options.bound_mult_init_val),
        if options.least_square_init_duals {
            "yes"
        } else {
            "no"
        },
        sci_text(options.constr_mult_init_max),
        sci_text(options.regularization),
        sci_text(options.first_hessian_perturbation),
        sci_text(options.regularization_first_growth_factor),
        options.adaptive_regularization_retries,
        sci_text(options.regularization_growth_factor),
        sci_text(options.regularization_decay_factor),
        sci_text(options.regularization_max),
        sci_text(options.jacobian_regularization_value),
        sci_text(options.jacobian_regularization_exponent),
        if options.second_order_correction {
            "on"
        } else {
            "off"
        },
        options.max_second_order_corrections,
        sci_text(options.second_order_correction_reduction_factor),
        if options.restoration_phase {
            "on"
        } else {
            "off"
        },
        options.watchdog_shortened_iter_trigger,
        options.watchdog_trial_iter_max,
        options.max_filter_resets,
        options.filter_reset_trigger,
        sci_text(options.tiny_step_tol),
        sci_text(options.tiny_step_y_tol),
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
        sci_text(options.kappa_d),
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
    Tiny,
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
    LinearSolverQualityIncreased,
    BoundMultiplierSafeguardApplied,
    BarrierParameterUpdated,
    AdaptiveRegularizationUsed,
    RestorationPhaseAccepted,
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
                "X=IPOPT filter reset heuristic cleared the previous filter frontier",
            )),
            InteriorPointIterationEvent::LinearSolverQualityIncreased => entries.push((
                'q',
                "q=SPRAL pivot tolerance was increased by IPOPT quality retry",
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
            InteriorPointIterationEvent::RestorationPhaseAccepted => entries.push((
                'R',
                "R=restoration phase returned an acceptable original iterate",
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

const NLIP_EVENT_SLOT_ORDER: [char; 14] = [
    'L', 'F', 's', 'S', 'A', 'W', 'X', 'q', 'B', 'U', 'V', 'R', 'T', 'M',
];
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
pub struct InteriorPointStepDirectionSnapshot {
    pub x: Vec<f64>,
    pub slack: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub slack_multipliers: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointIterationSnapshot {
    pub iteration: Index,
    pub phase: InteriorPointIterationPhase,
    pub x: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub slack_primal: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub equality_multipliers: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub inequality_multipliers: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub slack_multipliers: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub lower_bound_multipliers: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub upper_bound_multipliers: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub kkt_inequality_residual: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub kkt_slack_stationarity: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub kkt_slack_complementarity: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub kkt_slack_sigma: Option<Vec<f64>>,
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
    pub step_direction: Option<InteriorPointStepDirectionSnapshot>,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InteriorPointBoundaryLimiterKind {
    #[default]
    Unknown,
    Slack,
    VariableLowerBound,
    VariableUpperBound,
    Multiplier,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointBoundaryLimiter {
    #[cfg_attr(feature = "serde", serde(default))]
    pub kind: InteriorPointBoundaryLimiterKind,
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
    pub regularization_size: f64,
    pub primal_diagonal_shift: f64,
    pub dual_regularization: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_pr_limiter: Option<InteriorPointBoundaryLimiter>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alpha_du_limiter: Option<InteriorPointBoundaryLimiter>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub alpha_pr_limiters: Vec<InteriorPointBoundaryLimiter>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub alpha_du_limiters: Vec<InteriorPointBoundaryLimiter>,
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
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub inertia: Option<Box<InteriorPointLinearInertia>>,
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
    d_ineq: Vec<f64>,
    ds: Vec<f64>,
    dz: Vec<f64>,
    dz_lower: Vec<f64>,
    dz_upper: Vec<f64>,
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
    inequality_multipliers: Vec<f64>,
    slack: Vec<f64>,
    z: Vec<f64>,
    z_lower: Vec<f64>,
    z_upper: Vec<f64>,
    kkt_inequality_residual: Vec<f64>,
    kkt_slack_stationarity: Vec<f64>,
    kkt_slack_complementarity: Vec<f64>,
    kkt_slack_sigma: Vec<f64>,
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
    step_direction: Option<InteriorPointStepDirectionSnapshot>,
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
    tiny_step: bool,
    bound_multiplier_corrected: bool,
}

struct AcceptedTrialMultiplierState {
    z: Vec<f64>,
    z_lower: Vec<f64>,
    z_upper: Vec<f64>,
    dual_inf: f64,
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
    tiny_step_last_iteration: bool,
}

#[derive(Clone, Copy, Debug)]
enum RestorationJacobianEntry {
    Original(Index),
    PositiveResidual,
    NegativeResidual,
}

#[derive(Clone, Copy, Debug)]
struct RestorationHessianEntry {
    original: Option<Index>,
    proximity_diagonal_index: Option<Index>,
}

struct EqualityRestorationNlp<'a> {
    original: &'a dyn CompiledNlpProblem,
    x_ref: Vec<f64>,
    dr2_x: Vec<f64>,
    rho: f64,
    eta: f64,
    equality_jacobian_ccs: CCS,
    equality_jacobian_entries: Vec<RestorationJacobianEntry>,
    inequality_jacobian_ccs: CCS,
    lagrangian_hessian_ccs: CCS,
    lagrangian_hessian_entries: Vec<RestorationHessianEntry>,
    bounds: ConstraintBounds,
}

impl<'a> EqualityRestorationNlp<'a> {
    fn new(
        original: &'a dyn CompiledNlpProblem,
        x_ref: &[f64],
        rho: f64,
        eta: f64,
    ) -> Option<Self> {
        if original.inequality_count() != 0 {
            return None;
        }
        let dimension = original.dimension();
        let equality_count = original.equality_count();
        if equality_count == 0 {
            return None;
        }
        let total_dimension = dimension + 2 * equality_count;
        let dr2_x = x_ref
            .iter()
            .map(|value| {
                let scale = value.abs().max(1.0);
                1.0 / (scale * scale)
            })
            .collect::<Vec<_>>();
        let (equality_jacobian_ccs, equality_jacobian_entries) =
            build_equality_restoration_jacobian_ccs(original.equality_jacobian_ccs());
        let (lagrangian_hessian_ccs, lagrangian_hessian_entries) =
            build_equality_restoration_hessian_ccs(
                original.lagrangian_hessian_ccs(),
                dimension,
                total_dimension,
            );
        let mut lower = vec![None; total_dimension];
        let mut upper = vec![None; total_dimension];
        if let Some(original_bounds) = original.variable_bounds() {
            if let Some(values) = original_bounds.lower {
                for (index, value) in values.into_iter().enumerate().take(dimension) {
                    lower[index] = value;
                }
            }
            if let Some(values) = original_bounds.upper {
                for (index, value) in values.into_iter().enumerate().take(dimension) {
                    upper[index] = value;
                }
            }
        }
        for value in lower.iter_mut().skip(dimension) {
            *value = Some(0.0);
        }
        Some(Self {
            original,
            x_ref: x_ref.to_vec(),
            dr2_x,
            rho,
            eta,
            equality_jacobian_ccs,
            equality_jacobian_entries,
            inequality_jacobian_ccs: CCS::empty(0, total_dimension),
            lagrangian_hessian_ccs,
            lagrangian_hessian_entries,
            bounds: ConstraintBounds {
                lower: Some(lower),
                upper: Some(upper),
            },
        })
    }

    fn original_dimension(&self) -> Index {
        self.x_ref.len()
    }

    fn equality_residual_count(&self) -> Index {
        self.original.equality_count()
    }

    fn split_x<'x>(&self, x: &'x [f64]) -> (&'x [f64], &'x [f64], &'x [f64]) {
        let n = self.original_dimension();
        let m = self.equality_residual_count();
        (&x[..n], &x[n..n + m], &x[n + m..n + 2 * m])
    }
}

impl CompiledNlpProblem for EqualityRestorationNlp<'_> {
    fn dimension(&self) -> Index {
        self.original_dimension() + 2 * self.equality_residual_count()
    }

    fn parameter_count(&self) -> Index {
        self.original.parameter_count()
    }

    fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.original.parameter_ccs(parameter_index)
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        Some(self.bounds.clone())
    }

    fn equality_count(&self) -> Index {
        self.equality_residual_count()
    }

    fn inequality_count(&self) -> Index {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        let (x_original, negative_residual, positive_residual) = self.split_x(x);
        let residual_penalty =
            negative_residual.iter().sum::<f64>() + positive_residual.iter().sum::<f64>();
        let proximity = x_original
            .iter()
            .zip(self.x_ref.iter())
            .zip(self.dr2_x.iter())
            .map(|((value, reference), dr2)| dr2 * (value - reference).powi(2))
            .sum::<f64>();
        self.rho * residual_penalty + 0.5 * self.eta * proximity
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let n = self.original_dimension();
        let m = self.equality_residual_count();
        for index in 0..n {
            out[index] = self.eta * self.dr2_x[index] * (x[index] - self.x_ref[index]);
        }
        for value in &mut out[n..n + 2 * m] {
            *value = self.rho;
        }
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        &self.equality_jacobian_ccs
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let (x_original, negative_residual, positive_residual) = self.split_x(x);
        self.original.equality_values(x_original, parameters, out);
        for ((value, negative), positive) in out
            .iter_mut()
            .zip(negative_residual.iter())
            .zip(positive_residual.iter())
        {
            *value += negative - positive;
        }
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let n = self.original_dimension();
        let original_nnz = self.original.equality_jacobian_ccs().nnz();
        let mut original_values = vec![0.0; original_nnz];
        self.original
            .equality_jacobian_values(&x[..n], parameters, &mut original_values);
        for (slot, entry) in out.iter_mut().zip(self.equality_jacobian_entries.iter()) {
            *slot = match entry {
                RestorationJacobianEntry::Original(index) => original_values[*index],
                RestorationJacobianEntry::PositiveResidual => 1.0,
                RestorationJacobianEntry::NegativeResidual => -1.0,
            };
        }
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        &self.inequality_jacobian_ccs
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
        &self.lagrangian_hessian_ccs
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let n = self.original_dimension();
        let original_nnz = self.original.lagrangian_hessian_ccs().nnz();
        let mut with_constraints = vec![0.0; original_nnz];
        let mut objective_only = vec![0.0; original_nnz];
        self.original.lagrangian_hessian_values(
            &x[..n],
            parameters,
            equality_multipliers,
            &[],
            &mut with_constraints,
        );
        self.original.lagrangian_hessian_values(
            &x[..n],
            parameters,
            &vec![0.0; equality_multipliers.len()],
            &[],
            &mut objective_only,
        );
        for (slot, entry) in out.iter_mut().zip(self.lagrangian_hessian_entries.iter()) {
            let mut value = entry
                .original
                .map(|index| with_constraints[index] - objective_only[index])
                .unwrap_or(0.0);
            if let Some(index) = entry.proximity_diagonal_index {
                value += self.eta * self.dr2_x[index];
            }
            *slot = value;
        }
    }
}

fn build_equality_restoration_jacobian_ccs(original: &CCS) -> (CCS, Vec<RestorationJacobianEntry>) {
    let n = original.ncol;
    let m = original.nrow;
    let total_dimension = n + 2 * m;
    let mut col_ptrs = Vec::with_capacity(total_dimension + 1);
    let mut row_indices = Vec::with_capacity(original.nnz() + 2 * m);
    let mut entries = Vec::with_capacity(original.nnz() + 2 * m);
    col_ptrs.push(0);
    for col in 0..n {
        for index in original.col_ptrs[col]..original.col_ptrs[col + 1] {
            row_indices.push(original.row_indices[index]);
            entries.push(RestorationJacobianEntry::Original(index));
        }
        col_ptrs.push(row_indices.len());
    }
    for row in 0..m {
        row_indices.push(row);
        entries.push(RestorationJacobianEntry::PositiveResidual);
        col_ptrs.push(row_indices.len());
    }
    for row in 0..m {
        row_indices.push(row);
        entries.push(RestorationJacobianEntry::NegativeResidual);
        col_ptrs.push(row_indices.len());
    }
    (CCS::new(m, total_dimension, col_ptrs, row_indices), entries)
}

fn build_equality_restoration_hessian_ccs(
    original: &CCS,
    original_dimension: Index,
    total_dimension: Index,
) -> (CCS, Vec<RestorationHessianEntry>) {
    let mut original_by_position = HashMap::with_capacity(original.nnz());
    for col in 0..original.ncol {
        for index in original.col_ptrs[col]..original.col_ptrs[col + 1] {
            let row = original.row_indices[index];
            let (row, col) = if row >= col { (row, col) } else { (col, row) };
            original_by_position.insert((row, col), index);
        }
    }
    let mut rows_by_col = vec![Vec::<Index>::new(); original_dimension];
    for &(row, col) in original_by_position.keys() {
        if col < original_dimension && row < original_dimension {
            rows_by_col[col].push(row);
        }
    }
    for (col, rows) in rows_by_col.iter_mut().enumerate() {
        rows.push(col);
        rows.sort_unstable();
        rows.dedup();
    }
    let mut col_ptrs = Vec::with_capacity(total_dimension + 1);
    let mut row_indices = Vec::new();
    let mut entries = Vec::new();
    col_ptrs.push(0);
    for (col, rows) in rows_by_col.iter().enumerate() {
        for &row in rows {
            row_indices.push(row);
            entries.push(RestorationHessianEntry {
                original: original_by_position.get(&(row, col)).copied(),
                proximity_diagonal_index: (row == col).then_some(col),
            });
        }
        col_ptrs.push(row_indices.len());
    }
    for _ in original_dimension..total_dimension {
        col_ptrs.push(row_indices.len());
    }
    (
        CCS::new(total_dimension, total_dimension, col_ptrs, row_indices),
        entries,
    )
}

fn equality_restoration_initial_guess(
    x: &[f64],
    equality_values: &[f64],
    bound_push: f64,
) -> Vec<f64> {
    let push = bound_push.max(1.0e-8);
    let mut restoration_x = Vec::with_capacity(x.len() + 2 * equality_values.len());
    restoration_x.extend_from_slice(x);
    for &value in equality_values {
        restoration_x.push(if value < 0.0 { -value + push } else { push });
    }
    for &value in equality_values {
        restoration_x.push(if value >= 0.0 { value + push } else { push });
    }
    restoration_x
}

fn solve_equality_restoration_problem<P>(
    problem: &P,
    x: &[f64],
    equality_values: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
    current_primal_inf: f64,
) -> Option<InteriorPointSummary>
where
    P: CompiledNlpProblem,
{
    let restoration_mu = current_primal_inf.max(options.mu_init).max(1.0e-8);
    let restoration_problem =
        EqualityRestorationNlp::new(problem, x, 1.0e3, restoration_mu.sqrt())?;
    let restoration_x0 = equality_restoration_initial_guess(x, equality_values, options.bound_push);
    let mut restoration_options = options.clone();
    restoration_options.restoration_phase = false;
    restoration_options.verbose = false;
    restoration_options.linear_debug = None;
    restoration_options.max_iters = options.max_iters.clamp(10, 30);
    restoration_options.acceptable_iter = 0;
    restoration_options.mu_init = restoration_mu;
    restoration_options.mu_target = 0.0;
    solve_nlp_interior_point(
        &restoration_problem,
        &restoration_x0,
        parameters,
        &restoration_options,
    )
    .ok()
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

#[derive(Clone, Debug)]
struct FixedVariableElimination {
    free_indices: Vec<usize>,
    fixed_indices: Vec<usize>,
    fixed_values: Vec<f64>,
    free_position: Vec<Option<usize>>,
}

impl FixedVariableElimination {
    fn new(dimension: usize, fixed_indices: Vec<usize>, fixed_values: Vec<f64>) -> Self {
        debug_assert_eq!(fixed_indices.len(), fixed_values.len());
        let mut is_fixed = vec![false; dimension];
        for &index in &fixed_indices {
            is_fixed[index] = true;
        }
        let free_indices = (0..dimension)
            .filter(|index| !is_fixed[*index])
            .collect::<Vec<_>>();
        let mut free_position = vec![None; dimension];
        for (position, &index) in free_indices.iter().enumerate() {
            free_position[index] = Some(position);
        }
        Self {
            free_indices,
            fixed_indices,
            fixed_values,
            free_position,
        }
    }

    fn none(dimension: usize) -> Self {
        Self::new(dimension, Vec::new(), Vec::new())
    }

    fn has_fixed(&self) -> bool {
        !self.fixed_indices.is_empty()
    }

    fn reduced_dimension(&self) -> usize {
        self.free_indices.len()
    }

    fn project_fixed_values(&self, x: &mut [f64]) {
        for (&index, &value) in self.fixed_indices.iter().zip(self.fixed_values.iter()) {
            x[index] = value;
        }
    }

    fn reduce_vector(&self, values: &[f64]) -> Vec<f64> {
        if !self.has_fixed() {
            return values.to_vec();
        }
        self.free_indices
            .iter()
            .map(|&index| values[index])
            .collect()
    }

    fn expand_direction(&self, reduced: &[f64]) -> Vec<f64> {
        if !self.has_fixed() {
            return reduced.to_vec();
        }
        debug_assert_eq!(reduced.len(), self.free_indices.len());
        let mut full = vec![0.0; self.free_position.len()];
        for (&index, &value) in self.free_indices.iter().zip(reduced.iter()) {
            full[index] = value;
        }
        full
    }

    fn free_inf_norm(&self, values: &[f64]) -> f64 {
        if !self.has_fixed() {
            return inf_norm(values);
        }
        self.free_indices
            .iter()
            .fold(0.0_f64, |acc, &index| acc.max(values[index].abs()))
    }
}

#[derive(Clone, Debug)]
struct SparseColumnReduction {
    structure: Arc<SparseMatrixStructure>,
    source_value_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct SymmetricSubmatrixReduction {
    lower_triangle: Arc<CCS>,
    source_value_indices: Vec<usize>,
}

#[derive(Clone, Copy)]
struct ReducedBoundKktData<'a> {
    x: &'a [f64],
    bounds: &'a BoundConstraints,
    fixed_variables: &'a FixedVariableElimination,
    z_lower: &'a [f64],
    z_upper: &'a [f64],
}

struct ReducedKktSystem<'a> {
    hessian: &'a SparseSymmetricMatrix,
    equality_jacobian: &'a SparseMatrix,
    inequality_jacobian: &'a SparseMatrix,
    bound_diagonal: &'a [f64],
    bound_rhs: &'a [f64],
    bound_data: Option<ReducedBoundKktData<'a>>,
    slack: &'a [f64],
    multipliers: &'a [f64],
    r_dual: &'a [f64],
    r_eq: &'a [f64],
    r_ineq: &'a [f64],
    r_slack_stationarity: &'a [f64],
    r_cent: &'a [f64],
    barrier_parameter: f64,
    kappa_d: f64,
    solver: InteriorPointLinearSolver,
    regularization: f64,
    first_hessian_perturbation: f64,
    previous_hessian_perturbation: Option<f64>,
    regularization_first_growth_factor: f64,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_decay_factor: f64,
    regularization_max: f64,
    jacobian_regularization_value: f64,
    jacobian_regularization_exponent: f64,
    forced_jacobian_regularization: Option<f64>,
    spral_pivot_method: InteriorPointSpralPivotMethod,
    spral_action_on_zero_pivot: bool,
    spral_small_pivot_tolerance: f64,
    spral_threshold_pivot_u: f64,
    spral_pivot_tolerance_max: f64,
}

impl<'a> ReducedKktSystem<'a> {
    fn with_constraint_residuals<'b>(
        &'b self,
        r_eq: &'b [f64],
        r_ineq: &'b [f64],
    ) -> ReducedKktSystem<'b>
    where
        'a: 'b,
    {
        ReducedKktSystem {
            hessian: self.hessian,
            equality_jacobian: self.equality_jacobian,
            inequality_jacobian: self.inequality_jacobian,
            bound_diagonal: self.bound_diagonal,
            bound_rhs: self.bound_rhs,
            bound_data: self.bound_data,
            slack: self.slack,
            multipliers: self.multipliers,
            r_dual: self.r_dual,
            r_eq,
            r_ineq,
            r_slack_stationarity: self.r_slack_stationarity,
            r_cent: self.r_cent,
            barrier_parameter: self.barrier_parameter,
            kappa_d: self.kappa_d,
            solver: self.solver,
            regularization: self.regularization,
            first_hessian_perturbation: self.first_hessian_perturbation,
            previous_hessian_perturbation: self.previous_hessian_perturbation,
            regularization_first_growth_factor: self.regularization_first_growth_factor,
            adaptive_regularization_retries: self.adaptive_regularization_retries,
            regularization_growth_factor: self.regularization_growth_factor,
            regularization_decay_factor: self.regularization_decay_factor,
            regularization_max: self.regularization_max,
            jacobian_regularization_value: self.jacobian_regularization_value,
            jacobian_regularization_exponent: self.jacobian_regularization_exponent,
            forced_jacobian_regularization: self.forced_jacobian_regularization,
            spral_pivot_method: self.spral_pivot_method,
            spral_action_on_zero_pivot: self.spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: self.spral_small_pivot_tolerance,
            spral_threshold_pivot_u: self.spral_threshold_pivot_u,
            spral_pivot_tolerance_max: self.spral_pivot_tolerance_max,
        }
    }

    fn with_forced_perturbations<'b>(
        &'b self,
        primal_regularization: f64,
        jacobian_regularization: f64,
    ) -> ReducedKktSystem<'b>
    where
        'a: 'b,
    {
        let primal_regularization = primal_regularization.max(0.0);
        ReducedKktSystem {
            hessian: self.hessian,
            equality_jacobian: self.equality_jacobian,
            inequality_jacobian: self.inequality_jacobian,
            bound_diagonal: self.bound_diagonal,
            bound_rhs: self.bound_rhs,
            bound_data: self.bound_data,
            slack: self.slack,
            multipliers: self.multipliers,
            r_dual: self.r_dual,
            r_eq: self.r_eq,
            r_ineq: self.r_ineq,
            r_slack_stationarity: self.r_slack_stationarity,
            r_cent: self.r_cent,
            barrier_parameter: self.barrier_parameter,
            kappa_d: self.kappa_d,
            solver: self.solver,
            regularization: primal_regularization,
            first_hessian_perturbation: primal_regularization,
            previous_hessian_perturbation: Some(primal_regularization),
            regularization_first_growth_factor: self.regularization_first_growth_factor,
            adaptive_regularization_retries: 0,
            regularization_growth_factor: self.regularization_growth_factor,
            regularization_decay_factor: self.regularization_decay_factor,
            regularization_max: primal_regularization,
            jacobian_regularization_value: self.jacobian_regularization_value,
            jacobian_regularization_exponent: self.jacobian_regularization_exponent,
            forced_jacobian_regularization: Some(jacobian_regularization.max(0.0)),
            spral_pivot_method: self.spral_pivot_method,
            spral_action_on_zero_pivot: self.spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: self.spral_small_pivot_tolerance,
            spral_threshold_pivot_u: self.spral_threshold_pivot_u,
            spral_pivot_tolerance_max: self.spral_pivot_tolerance_max,
        }
    }
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
    native: NativeSpralLibrary,
    session: Option<NativeSpralSession>,
    numeric_options: SpralNumericFactorOptions,
    ordering: SpralNativeOrdering,
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
    bound_diagonal: Vec<f64>,
    bound_rhs: Vec<f64>,
    slack: Vec<f64>,
    multipliers: Vec<f64>,
    r_dual: Vec<f64>,
    r_eq: Vec<f64>,
    r_ineq: Vec<f64>,
    r_slack_stationarity: Vec<f64>,
    r_cent: Vec<f64>,
    regularization: f64,
    primal_diagonal_shift: f64,
    dual_regularization: f64,
    first_hessian_perturbation: f64,
    previous_hessian_perturbation: Option<f64>,
    regularization_first_growth_factor: f64,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_decay_factor: f64,
    regularization_max: f64,
    jacobian_regularization_value: f64,
    jacobian_regularization_exponent: f64,
    spral_pivot_method: InteriorPointSpralPivotMethod,
    spral_action_on_zero_pivot: bool,
    spral_small_pivot_tolerance: f64,
    spral_threshold_pivot_u: f64,
    spral_pivot_tolerance_max: f64,
    augmented_pattern: SpralAugmentedKktPattern,
    augmented_values: Vec<f64>,
    augmented_rhs: Vec<f64>,
    expected_augmented_inertia: SpralInertia,
    barrier_parameter: f64,
    kappa_d: f64,
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
                "invalid sparse KKT structure for ssids_rs: {error}"
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
                "failed to analyse sparse KKT structure for ssids_rs: {error}"
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
        let native = NativeSpralLibrary::load().map_err(|error| {
            InteriorPointSolveError::InvalidInput(format!(
                "failed to load native ssids_rs: {error}"
            ))
        })?;
        let values = vec![0.0; pattern.ccs.nnz()];
        Ok((
            Self {
                pattern,
                values,
                native,
                session: None,
                numeric_options: *numeric_options,
                ordering: SpralNativeOrdering::Matching,
                factor_regularization: None,
            },
            SpralAnalyseInfo {
                estimated_fill_nnz: 0,
                supernode_count: 0,
                max_supernode_width: 0,
                ordering_kind: "native_spral_matching_value_dependent",
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
            bound_diagonal: &self.bound_diagonal,
            bound_rhs: &self.bound_rhs,
            // KKT snapshots do not yet carry the full variable-bound state
            // needed for IPOPT's z_L/z_U residual rows, so replay diagnostics
            // use the algebraically equivalent eliminated bound branch.
            bound_data: None,
            slack: &self.slack,
            multipliers: &self.multipliers,
            r_dual: &self.r_dual,
            r_eq: &self.r_eq,
            r_ineq: &self.r_ineq,
            r_slack_stationarity: &self.r_slack_stationarity,
            r_cent: &self.r_cent,
            barrier_parameter: self.barrier_parameter,
            kappa_d: self.kappa_d,
            solver,
            regularization: self.regularization,
            first_hessian_perturbation: self.first_hessian_perturbation,
            previous_hessian_perturbation: self.previous_hessian_perturbation,
            regularization_first_growth_factor: self.regularization_first_growth_factor,
            adaptive_regularization_retries: self.adaptive_regularization_retries,
            regularization_growth_factor: self.regularization_growth_factor,
            regularization_decay_factor: self.regularization_decay_factor,
            regularization_max: self.regularization_max,
            jacobian_regularization_value: self.jacobian_regularization_value,
            jacobian_regularization_exponent: self.jacobian_regularization_exponent,
            forced_jacobian_regularization: None,
            spral_pivot_method: self.spral_pivot_method,
            spral_action_on_zero_pivot: self.spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: self.spral_small_pivot_tolerance,
            spral_threshold_pivot_u: self.spral_threshold_pivot_u,
            spral_pivot_tolerance_max: self.spral_pivot_tolerance_max,
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

fn step_direction_snapshot(direction: &NewtonDirection) -> InteriorPointStepDirectionSnapshot {
    InteriorPointStepDirectionSnapshot {
        x: direction.dx.clone(),
        slack: direction.ds.clone(),
        equality_multipliers: direction.d_lambda.clone(),
        inequality_multipliers: direction.d_ineq.clone(),
        slack_multipliers: direction.dz.clone(),
        lower_bound_multipliers: direction.dz_lower.clone(),
        upper_bound_multipliers: direction.dz_upper.clone(),
    }
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
    primal_diagonal_shift_used: f64,
    dual_diagonal_shift_used: f64,
    barrier_parameter: f64,
    primal_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    line_search_trials: Index,
) -> std::result::Result<InteriorPointKktSnapshot, InteriorPointSolveError> {
    let pattern = build_spral_augmented_kkt_pattern(
        system.hessian.lower_triangle.as_ref(),
        system.equality_jacobian.structure.as_ref(),
        system.inequality_jacobian.structure.as_ref(),
    )?;
    let primal_shift = primal_diagonal_shift_used;
    let slack_shift = primal_diagonal_shift_used;
    let dual_shift = dual_diagonal_shift_used.max(0.0);
    let mut augmented_values = vec![0.0; pattern.ccs.nnz()];
    fill_spral_augmented_kkt_values(
        &pattern,
        &mut augmented_values,
        system,
        primal_shift,
        slack_shift,
        dual_shift,
    );
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let augmented_rhs =
        build_ipopt_augmented_kkt_rhs(system, &pattern, IpoptLinearRhsOrientation::FinalDirection);
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
        bound_diagonal: system.bound_diagonal.to_vec(),
        bound_rhs: system.bound_rhs.to_vec(),
        slack: system.slack.to_vec(),
        multipliers: system.multipliers.to_vec(),
        r_dual: system.r_dual.to_vec(),
        r_eq: system.r_eq.to_vec(),
        r_ineq: system.r_ineq.to_vec(),
        r_slack_stationarity: system.r_slack_stationarity.to_vec(),
        r_cent: system.r_cent.to_vec(),
        regularization: primal_diagonal_shift_used,
        primal_diagonal_shift: primal_diagonal_shift_used,
        dual_regularization: dual_diagonal_shift_used,
        first_hessian_perturbation: system.first_hessian_perturbation,
        previous_hessian_perturbation: system.previous_hessian_perturbation,
        regularization_first_growth_factor: system.regularization_first_growth_factor,
        adaptive_regularization_retries: system.adaptive_regularization_retries,
        regularization_growth_factor: system.regularization_growth_factor,
        regularization_decay_factor: system.regularization_decay_factor,
        regularization_max: system.regularization_max,
        jacobian_regularization_value: system.jacobian_regularization_value,
        jacobian_regularization_exponent: system.jacobian_regularization_exponent,
        spral_pivot_method: system.spral_pivot_method,
        spral_action_on_zero_pivot: system.spral_action_on_zero_pivot,
        spral_small_pivot_tolerance: system.spral_small_pivot_tolerance,
        spral_threshold_pivot_u: system.spral_threshold_pivot_u,
        spral_pivot_tolerance_max: system.spral_pivot_tolerance_max,
        augmented_pattern: pattern.clone(),
        augmented_values,
        augmented_rhs,
        expected_augmented_inertia: spral_expected_augmented_inertia(&pattern),
        barrier_parameter,
        kappa_d: system.kappa_d,
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
        inertia: attempt.inertia.as_deref().copied(),
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

fn scaled_linear_debug_delta_matches(delta: Option<f64>, primary_scale: f64) -> bool {
    delta
        .is_some_and(|delta| delta <= LINEAR_DEBUG_RELATIVE_DELTA_TOLERANCE * (1.0 + primary_scale))
}

fn slice_inf_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn linear_debug_matches_primary(
    primary: &NewtonDirection,
    result: &InteriorPointLinearDebugBackendResult,
) -> bool {
    result.success
        && scaled_linear_debug_delta_matches(
            result.step_delta_inf,
            linear_direction_inf_norm(primary),
        )
        && scaled_linear_debug_delta_matches(result.dx_delta_inf, slice_inf_norm(&primary.dx))
        && scaled_linear_debug_delta_matches(
            result.d_lambda_delta_inf,
            slice_inf_norm(&primary.d_lambda),
        )
        && scaled_linear_debug_delta_matches(result.ds_delta_inf, slice_inf_norm(&primary.ds))
        && scaled_linear_debug_delta_matches(result.dz_delta_inf, slice_inf_norm(&primary.dz))
}

fn newton_direction_from_augmented_solution(
    snapshot: &InteriorPointKktSnapshot,
    solver: InteriorPointLinearSolver,
    solution: Vec<f64>,
    backend_stats: LinearBackendRunStats,
) -> NewtonDirection {
    let n = snapshot.x_dimension;
    let meq = snapshot.equality_dimension;
    let mineq = snapshot.inequality_dimension;
    let dx = solution[..n].to_vec();
    let ipopt_ds = solution
        [snapshot.augmented_pattern.p_offset..snapshot.augmented_pattern.p_offset + mineq]
        .to_vec();
    let d_lambda = solution
        [snapshot.augmented_pattern.lambda_offset..snapshot.augmented_pattern.lambda_offset + meq]
        .to_vec();
    let d_ineq = solution
        [snapshot.augmented_pattern.z_offset..snapshot.augmented_pattern.z_offset + mineq]
        .to_vec();
    let ds = ipopt_ds;
    let dz = snapshot
        .r_cent
        .iter()
        .zip(snapshot.multipliers.iter())
        .zip(snapshot.slack.iter())
        .zip(ds.iter())
        .map(|(((r_cent_i, z_i), s_i), ds_i)| {
            ipopt_upper_slack_bound_multiplier_step(*s_i, *z_i, *r_cent_i, *ds_i)
        })
        .collect::<Vec<_>>();
    NewtonDirection {
        dx,
        d_lambda,
        d_ineq,
        ds,
        dz,
        dz_lower: Vec::new(),
        dz_upper: Vec::new(),
        solver_used: solver,
        regularization_used: snapshot.regularization.max(snapshot.primal_diagonal_shift),
        dual_regularization_used: snapshot.dual_regularization,
        primal_diagonal_shift_used: snapshot.primal_diagonal_shift,
        linear_solution: solution,
        backend_stats,
        linear_debug: None,
    }
}

fn replay_snapshot_with_solver(
    snapshot: &InteriorPointKktSnapshot,
    solver: InteriorPointLinearSolver,
    debug_state: &mut InteriorPointLinearDebugState,
) -> std::result::Result<NewtonDirection, Vec<InteriorPointLinearSolveAttempt>> {
    let system = snapshot.reduced_system(solver);
    let mut scratch_profiling = InteriorPointProfiling::default();
    match solver {
        InteriorPointLinearSolver::SsidsRs => {
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
                        .unwrap_or("rust ssids_rs workspace unavailable"),
                )]);
            };
            if workspace.values.len() != snapshot.augmented_values.len() {
                return Err(vec![spral_error_attempt(
                    snapshot.regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    "snapshot pattern does not match rust SPRAL workspace",
                )]);
            }
            workspace.values.copy_from_slice(&snapshot.augmented_values);
            factor_solve_spral_ssids(
                &system,
                workspace,
                &snapshot.augmented_rhs,
                snapshot.regularization,
                &mut scratch_profiling,
                false,
            )
            .map(|(solution, backend_stats)| {
                newton_direction_from_augmented_solution(snapshot, solver, solution, backend_stats)
            })
            .map_err(|attempt| vec![attempt])
        }
        InteriorPointLinearSolver::SpralSrc => {
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
                        .unwrap_or("native ssids_rs workspace unavailable"),
                )]);
            };
            if workspace.values.len() != snapshot.augmented_values.len() {
                return Err(vec![native_spral_error_attempt(
                    snapshot.regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    "snapshot pattern does not match native SPRAL workspace",
                )]);
            }
            workspace.values.copy_from_slice(&snapshot.augmented_values);
            let prefinal_rhs = snapshot
                .augmented_rhs
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>();
            factor_solve_spral_src(
                &system,
                workspace,
                &prefinal_rhs,
                snapshot.regularization,
                IpoptLinearSolveContext {
                    shifts: IpoptLinearRefinementShifts {
                        primal: snapshot.primal_diagonal_shift,
                        slack: snapshot.primal_diagonal_shift,
                        dual: snapshot.dual_regularization,
                    },
                    rhs_orientation: IpoptLinearRhsOrientation::PreFinal,
                    native_spral_quality_was_increased: false,
                },
                &mut scratch_profiling,
                false,
            )
            .map(|(solution, backend_stats)| {
                newton_direction_from_augmented_solution(snapshot, solver, solution, backend_stats)
            })
            .map_err(|attempt| vec![attempt])
        }
        InteriorPointLinearSolver::SparseQdldl => solve_reduced_kkt_with_sparse_qdldl(&system),
        InteriorPointLinearSolver::Auto => Err(vec![InteriorPointLinearSolveAttempt {
            solver,
            regularization: snapshot.regularization,
            inertia: None,
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
                if linear_debug_matches_primary(primary_direction, &result) {
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
                    inertia: None,
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
            inertia: None,
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
                    inertia: None,
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
        .map(|(&g_i, &s_i)| g_i - s_i)
        .collect()
}

fn slack_form_inequality_l1_norm(augmented_inequality_values: &[f64], slack: &[f64]) -> f64 {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    augmented_inequality_values
        .iter()
        .zip(slack.iter())
        .map(|(&g_i, &s_i)| (g_i - s_i).abs())
        .sum()
}

fn slack_form_inequality_inf_norm(augmented_inequality_values: &[f64], slack: &[f64]) -> f64 {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    augmented_inequality_values
        .iter()
        .zip(slack.iter())
        .fold(0.0, |acc, (&g_i, &s_i)| acc.max((g_i - s_i).abs()))
}

fn filter_theta_l1_norm(
    equality_values: &[f64],
    augmented_inequality_values: &[f64],
    slack: &[f64],
) -> f64 {
    l1_norm(equality_values) + slack_form_inequality_l1_norm(augmented_inequality_values, slack)
}

fn native_lower_bound_slack(x: &[f64], index: usize, lower: f64) -> f64 {
    x[index] - lower
}

fn native_upper_bound_slack(x: &[f64], index: usize, upper: f64) -> f64 {
    upper - x[index]
}

fn bound_relaxation_amount(bound: f64, options: &InteriorPointOptions) -> f64 {
    if options.bound_relax_factor <= 0.0 {
        return 0.0;
    }
    options
        .constraint_tol
        .min(options.bound_relax_factor * bound.abs().max(1.0))
}

fn slack_bound_relaxation(options: &InteriorPointOptions) -> f64 {
    bound_relaxation_amount(0.0, options)
}

fn slack_upper_bound_values(count: usize, options: &InteriorPointOptions) -> Vec<f64> {
    vec![slack_bound_relaxation(options); count]
}

fn inequality_upper_bound_inf_norm(values: &[f64], upper_bounds: &[f64]) -> f64 {
    debug_assert_eq!(values.len(), upper_bounds.len());
    values
        .iter()
        .zip(upper_bounds.iter())
        .fold(0.0_f64, |acc, (&value, &upper)| {
            acc.max((value - upper).max(0.0))
        })
}

fn slack_barrier_values(slack: &[f64], upper_bounds: &[f64]) -> Vec<f64> {
    debug_assert_eq!(slack.len(), upper_bounds.len());
    // NLIP stores inequality slacks in IPOPT's internal upper-bound
    // convention: d(x) - s = 0 and s <= d_U. IPOPT's barrier,
    // complementarity, and sigma routines operate on curr_slack_s_U = d_U - s.
    // See IpIpoptCalculatedQuantities.cpp::curr_slack_s_U/curr_sigma_s and
    // OrigIpoptNLP.cpp::relax_bounds for the relaxed d_U construction.
    upper_bounds
        .iter()
        .zip(slack.iter())
        .map(|(&upper, &value)| upper - value)
        .collect()
}

fn slack_barrier_direction_values(slack_direction: &[f64]) -> Vec<f64> {
    slack_direction.iter().map(|value| -*value).collect()
}

fn native_bound_log_sum(x: &[f64], bounds: &BoundConstraints) -> f64 {
    let lower_sum = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .map(|(&index, &lower)| native_lower_bound_slack(x, index, lower).ln())
        .sum::<f64>();
    let upper_sum = bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .map(|(&index, &upper)| native_upper_bound_slack(x, index, upper).ln())
        .sum::<f64>();
    lower_sum + upper_sum
}

fn native_bound_damping_sum(x: &[f64], bounds: &BoundConstraints) -> f64 {
    let lower_sum = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .filter(|(index, _)| !bounds.upper_indices.contains(index))
        .map(|(&index, &lower)| native_lower_bound_slack(x, index, lower))
        .sum::<f64>();
    let upper_sum = bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .filter(|(index, _)| !bounds.lower_indices.contains(index))
        .map(|(&index, &upper)| native_upper_bound_slack(x, index, upper))
        .sum::<f64>();
    lower_sum + upper_sum
}

fn native_bound_damping_directional_derivative(dx: &[f64], bounds: &BoundConstraints) -> f64 {
    let lower_sum = bounds
        .lower_indices
        .iter()
        .filter(|index| !bounds.upper_indices.contains(index))
        .map(|&index| dx[index])
        .sum::<f64>();
    let upper_sum = bounds
        .upper_indices
        .iter()
        .filter(|index| !bounds.lower_indices.contains(index))
        .map(|&index| -dx[index])
        .sum::<f64>();
    lower_sum + upper_sum
}

fn positive_slack_damping(barrier_parameter: f64, kappa_d: f64) -> f64 {
    if barrier_parameter > 0.0 && kappa_d > 0.0 {
        barrier_parameter * kappa_d
    } else {
        0.0
    }
}

fn system_positive_slack_damping(system: &ReducedKktSystem<'_>) -> f64 {
    positive_slack_damping(system.barrier_parameter, system.kappa_d)
}

fn damped_slack_stationarity_residual(system: &ReducedKktSystem<'_>, index: usize) -> f64 {
    system.r_slack_stationarity[index] - system_positive_slack_damping(system)
}

fn ipopt_internal_slack_rhs(system: &ReducedKktSystem<'_>, index: usize) -> f64 {
    system.r_cent[index] / system.slack[index] - damped_slack_stationarity_residual(system, index)
}

fn build_ipopt_augmented_kkt_rhs(
    system: &ReducedKktSystem<'_>,
    pattern: &SpralAugmentedKktPattern,
    orientation: IpoptLinearRhsOrientation,
) -> Vec<f64> {
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let mut rhs = vec![0.0; pattern.dimension()];
    let sign = match orientation {
        IpoptLinearRhsOrientation::PreFinal => 1.0,
        IpoptLinearRhsOrientation::FinalDirection => -1.0,
    };
    rhs[..n]
        .iter_mut()
        .zip(system.r_dual.iter())
        .zip(system.bound_rhs.iter())
        .for_each(|((rhs_i, r_i), bound_rhs_i)| *rhs_i = sign * (*r_i - *bound_rhs_i));
    for row in 0..mineq {
        rhs[pattern.p_offset + row] = -sign * ipopt_internal_slack_rhs(system, row);
    }
    for row in 0..meq {
        rhs[pattern.lambda_offset + row] = sign * system.r_eq[row];
    }
    for row in 0..mineq {
        rhs[pattern.z_offset + row] = sign * system.r_ineq[row];
    }
    rhs
}

fn ipopt_upper_slack_bound_multiplier_step(
    slack: f64,
    multiplier: f64,
    complementarity_residual: f64,
    ipopt_internal_slack_step: f64,
) -> f64 {
    // Mirrors IPOPT PDFullSpaceSolver::SolveOnce Pd_U.SinvBlrmZMTdBr(1., ...)
    // followed by PDSearchDirCalc's Solve(-1., 0., ...) final scaling:
    // delta_v_U = (-rhs.v_U + v_U * delta_s) / slack_s_U. NLIP now stores the
    // raw IPOPT-internal upper-bound slack component and converts separately
    // when a positive upper-slack distance is needed.
    (-complementarity_residual + multiplier * ipopt_internal_slack_step) / slack
}

fn ipopt_prefinal_upper_slack_bound_multiplier_step(
    slack: f64,
    multiplier: f64,
    complementarity_residual: f64,
    ipopt_internal_slack_step: f64,
) -> f64 {
    // Mirrors IPOPT PDFullSpaceSolver::SolveOnce Pd_U.SinvBlrmZMTdBr(1., ...)
    // before PDSearchDirCalc's final Solve(-1., 0., ...) scaling.
    (complementarity_residual + multiplier * ipopt_internal_slack_step) / slack
}

fn slack_stationarity_residuals(lambda_ineq: &[f64], z: &[f64]) -> Vec<f64> {
    lambda_ineq
        .iter()
        .zip(z.iter())
        .map(|(y_i, z_i)| z_i - y_i)
        .collect()
}

fn slack_stationarity_inf_norm(lambda_ineq: &[f64], z: &[f64]) -> f64 {
    lambda_ineq
        .iter()
        .zip(z.iter())
        .fold(0.0_f64, |acc, (y_i, z_i)| acc.max((z_i - y_i).abs()))
}

fn damped_slack_stationarity_residuals(
    lambda_ineq: &[f64],
    z: &[f64],
    barrier_parameter: f64,
    kappa_d: f64,
) -> Vec<f64> {
    let damping = positive_slack_damping(barrier_parameter, kappa_d);
    slack_stationarity_residuals(lambda_ineq, z)
        .into_iter()
        .map(|residual| residual - damping)
        .collect()
}

fn slack_complementarity_residuals(slack_barrier: &[f64], z: &[f64], mu: f64) -> Vec<f64> {
    slack_barrier
        .iter()
        .zip(z.iter())
        .map(|(slack_i, z_i)| slack_i * z_i - mu)
        .collect()
}

fn slack_sigma_values(slack_barrier: &[f64], z: &[f64]) -> Vec<f64> {
    slack_barrier
        .iter()
        .zip(z.iter())
        .map(|(slack_i, z_i)| z_i / slack_i)
        .collect()
}

fn barrier_objective_value(
    objective_value: f64,
    slack: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    barrier_parameter: f64,
    kappa_d: f64,
) -> f64 {
    if (slack.is_empty() && bounds.total_count() == 0) || barrier_parameter <= 0.0 {
        return objective_value;
    }
    let slack_log_sum = slack.iter().map(|value| value.ln()).sum::<f64>();
    let damping_weight = positive_slack_damping(barrier_parameter, kappa_d);
    let damping = if damping_weight > 0.0 {
        damping_weight * (slack.iter().sum::<f64>() + native_bound_damping_sum(x, bounds))
    } else {
        0.0
    };
    objective_value - barrier_parameter * (slack_log_sum + native_bound_log_sum(x, bounds))
        + damping
}

#[expect(
    clippy::too_many_arguments,
    reason = "IPOPT filter derivative uses separate primal, slack, bound, and damping inputs."
)]
fn barrier_objective_directional_derivative(
    gradient: &[f64],
    slack: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    dx: &[f64],
    ds: &[f64],
    barrier_parameter: f64,
    kappa_d: f64,
) -> f64 {
    let objective_term = gradient
        .iter()
        .zip(dx.iter())
        .map(|(gradient_i, dx_i)| gradient_i * dx_i)
        .sum::<f64>();
    if (slack.is_empty() && bounds.total_count() == 0) || barrier_parameter <= 0.0 {
        return objective_term;
    }
    let slack_barrier_term = slack
        .iter()
        .zip(ds.iter())
        .map(|(slack_i, ds_i)| ds_i / slack_i)
        .sum::<f64>();
    let lower_barrier_term = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .map(|(&index, &lower)| dx[index] / native_lower_bound_slack(x, index, lower))
        .sum::<f64>();
    let upper_barrier_term = bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .map(|(&index, &upper)| -dx[index] / native_upper_bound_slack(x, index, upper))
        .sum::<f64>();
    let damping_weight = positive_slack_damping(barrier_parameter, kappa_d);
    let damping_term = if damping_weight > 0.0 {
        damping_weight
            * (ds.iter().sum::<f64>() + native_bound_damping_directional_derivative(dx, bounds))
    } else {
        0.0
    };
    objective_term
        - barrier_parameter * (slack_barrier_term + lower_barrier_term + upper_barrier_term)
        + damping_term
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
    // IPOPT `FilterLSAcceptor::CalculateAlphaMin` returns only
    // `alpha_min_frac * alpha_min`; the backtracking loop itself is
    // responsible for always checking the first trial point.
    options.alpha_min_frac * alpha_min
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
    let mut baseline = 1.0;
    if reference_barrier_objective.abs() > 10.0 {
        baseline = reference_barrier_objective.abs().log10();
    }
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
    // IPOPT MonotoneMuUpdate::CalcNewMuAndTau uses mu_target and the
    // tol/compl_inf_tol floor here. The separate mu_min option belongs to the
    // adaptive/free-mu path, not to monotone mu updates.
    options
        .mu_target
        .max(tol / (options.barrier_tol_factor + 1.0))
}

fn next_barrier_parameter_once(barrier_parameter: f64, options: &InteriorPointOptions) -> f64 {
    let minimum_barrier = minimum_monotone_barrier_parameter(options);
    if barrier_parameter <= minimum_barrier {
        return minimum_barrier;
    }
    minimum_barrier.max(
        (options.mu_linear_decrease_factor * barrier_parameter)
            .min(barrier_parameter.powf(options.mu_superlinear_decrease_power)),
    )
}

fn next_barrier_parameter<F>(
    barrier_parameter: f64,
    tiny_step: bool,
    mu_update_initialized: bool,
    options: &InteriorPointOptions,
    mut barrier_error: F,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut current_barrier = barrier_parameter;
    let mut current_tiny_step = tiny_step;
    let minimum_barrier = minimum_monotone_barrier_parameter(options);
    loop {
        let current_barrier_error = barrier_error(current_barrier);
        if !current_tiny_step
            && !should_reduce_barrier_parameter(current_barrier_error, current_barrier, options)
        {
            break;
        }
        let next_barrier = next_barrier_parameter_once(current_barrier, options);
        if next_barrier >= current_barrier - 1e-18 {
            break;
        }
        current_barrier = next_barrier.max(minimum_barrier);
        if mu_update_initialized && !options.mu_allow_fast_monotone_decrease {
            break;
        }
        current_tiny_step = false;
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

fn ipopt_primal_step_inf_norm(direction: &NewtonDirection) -> f64 {
    step_inf_norm(&direction.dx).max(step_inf_norm(&direction.ds))
}

fn is_tiny_ip_step(
    x: &[f64],
    slack: &[f64],
    direction: &NewtonDirection,
    primal_inf: f64,
    options: &InteriorPointOptions,
) -> bool {
    if options.tiny_step_tol == 0.0 || primal_inf > 1.0e-4 {
        return false;
    }
    let max_x = x
        .iter()
        .zip(direction.dx.iter())
        .fold(0.0_f64, |acc, (&value, &delta)| {
            acc.max(delta.abs() / (1.0 + value.abs()))
        });
    if max_x > options.tiny_step_tol {
        return false;
    }
    let max_s = slack
        .iter()
        .zip(direction.ds.iter())
        .fold(0.0_f64, |acc, (&value, &delta)| {
            acc.max(delta.abs() / (1.0 + value.abs()))
        });
    max_s <= options.tiny_step_tol
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

fn current_fraction_to_boundary_tau(barrier_parameter: f64, options: &InteriorPointOptions) -> f64 {
    options
        .fraction_to_boundary
        .max(1.0 - barrier_parameter.max(0.0))
        .clamp(0.0, 1.0)
}

fn ipopt_dense_frac_to_bound_candidate(tau: f64, value: f64, delta: f64) -> f64 {
    // IPOPT `DenseVector::FracToBoundImpl` evaluates this as
    // `-tau / values_delta[i] * values_x[i]`; keep that operation order so
    // fraction-to-boundary alphas do not drift by a bit on tight limiters.
    -tau / delta * value
}

fn fraction_to_boundary_with_limiter(
    current: &[f64],
    direction: &[f64],
    tau: f64,
    kind: InteriorPointBoundaryLimiterKind,
) -> (f64, Option<InteriorPointBoundaryLimiter>) {
    let mut alpha = 1.0_f64;
    let mut limiter = None;
    for (idx, (&value, &delta)) in current.iter().zip(direction.iter()).enumerate() {
        if delta < 0.0 {
            let candidate = ipopt_dense_frac_to_bound_candidate(tau, value, delta);
            if candidate < alpha {
                alpha = candidate;
                limiter = Some(InteriorPointBoundaryLimiter {
                    kind,
                    index: idx,
                    value,
                    direction: delta,
                    alpha: candidate,
                });
            }
        }
    }
    (alpha, limiter)
}

fn fraction_to_boundary_limiters(
    current: &[f64],
    direction: &[f64],
    tau: f64,
    max_count: usize,
    kind: InteriorPointBoundaryLimiterKind,
) -> Vec<InteriorPointBoundaryLimiter> {
    if max_count == 0 {
        return Vec::new();
    }
    let mut limiters = current
        .iter()
        .zip(direction.iter())
        .enumerate()
        .filter_map(|(index, (&value, &delta))| {
            if delta < 0.0 {
                Some(InteriorPointBoundaryLimiter {
                    kind,
                    index,
                    value,
                    direction: delta,
                    alpha: ipopt_dense_frac_to_bound_candidate(tau, value, delta),
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    limiters.sort_by(|lhs, rhs| {
        lhs.alpha
            .total_cmp(&rhs.alpha)
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    limiters.truncate(max_count);
    limiters
}

fn fraction_to_variable_bounds_with_limiter(
    x: &[f64],
    dx: &[f64],
    bounds: &BoundConstraints,
    tau: f64,
) -> (f64, Option<InteriorPointBoundaryLimiter>) {
    let mut alpha = 1.0_f64;
    let mut limiter = None;
    for (&index, &lower) in bounds.lower_indices.iter().zip(bounds.lower_values.iter()) {
        let value = native_lower_bound_slack(x, index, lower);
        let direction = dx[index];
        if direction < 0.0 {
            let candidate = ipopt_dense_frac_to_bound_candidate(tau, value, direction);
            if candidate < alpha {
                alpha = candidate;
                limiter = Some(InteriorPointBoundaryLimiter {
                    kind: InteriorPointBoundaryLimiterKind::VariableLowerBound,
                    index,
                    value,
                    direction,
                    alpha: candidate,
                });
            }
        }
    }
    for (&index, &upper) in bounds.upper_indices.iter().zip(bounds.upper_values.iter()) {
        let value = native_upper_bound_slack(x, index, upper);
        let direction = -dx[index];
        if direction < 0.0 {
            let candidate = ipopt_dense_frac_to_bound_candidate(tau, value, direction);
            if candidate < alpha {
                alpha = candidate;
                limiter = Some(InteriorPointBoundaryLimiter {
                    kind: InteriorPointBoundaryLimiterKind::VariableUpperBound,
                    index,
                    value,
                    direction,
                    alpha: candidate,
                });
            }
        }
    }
    (alpha, limiter)
}

fn add_native_bound_multiplier_terms(
    residual: &mut [f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
) {
    for (&index, &z_i) in bounds.lower_indices.iter().zip(z_lower.iter()) {
        residual[index] -= z_i;
    }
    for (&index, &z_i) in bounds.upper_indices.iter().zip(z_upper.iter()) {
        residual[index] += z_i;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Bound KKT assembly keeps IPOPT damping and complementarity terms explicit."
)]
fn native_bound_kkt_terms(
    x: &[f64],
    bounds: &BoundConstraints,
    fixed_variables: &FixedVariableElimination,
    z_lower: &[f64],
    z_upper: &[f64],
    barrier_parameter: f64,
    sigma: f64,
    kappa_d: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut diagonal = vec![0.0; fixed_variables.reduced_dimension()];
    let mut rhs = vec![0.0; fixed_variables.reduced_dimension()];
    let damping_weight = positive_slack_damping(barrier_parameter, kappa_d);
    for ((&index, &lower), &z_i) in bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .zip(z_lower.iter())
    {
        let Some(reduced_index) = fixed_variables.free_position[index] else {
            continue;
        };
        let slack = native_lower_bound_slack(x, index, lower);
        let residual = slack * z_i - sigma * barrier_parameter;
        diagonal[reduced_index] += z_i / slack;
        rhs[reduced_index] += -residual / slack;
        if damping_weight > 0.0 && !bounds.upper_indices.contains(&index) {
            rhs[reduced_index] -= damping_weight;
        }
    }
    for ((&index, &upper), &z_i) in bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .zip(z_upper.iter())
    {
        let Some(reduced_index) = fixed_variables.free_position[index] else {
            continue;
        };
        let slack = native_upper_bound_slack(x, index, upper);
        let residual = slack * z_i - sigma * barrier_parameter;
        diagonal[reduced_index] += z_i / slack;
        rhs[reduced_index] += residual / slack;
        if damping_weight > 0.0 && !bounds.lower_indices.contains(&index) {
            rhs[reduced_index] += damping_weight;
        }
    }
    (diagonal, rhs)
}

fn native_bound_multiplier_steps(
    x: &[f64],
    dx: &[f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
    barrier_parameter: f64,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>) {
    let dz_lower = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .zip(z_lower.iter())
        .map(|((&index, &lower), &z_i)| {
            let slack = native_lower_bound_slack(x, index, lower);
            let residual = slack * z_i - sigma * barrier_parameter;
            (-residual - z_i * dx[index]) / slack
        })
        .collect::<Vec<_>>();
    let dz_upper = bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .zip(z_upper.iter())
        .map(|((&index, &upper), &z_i)| {
            let slack = native_upper_bound_slack(x, index, upper);
            let residual = slack * z_i - sigma * barrier_parameter;
            (-residual + z_i * dx[index]) / slack
        })
        .collect::<Vec<_>>();
    (dz_lower, dz_upper)
}

fn interior_point_direction_diagnostics(
    direction: &NewtonDirection,
    alpha_pr_limiter: Option<InteriorPointBoundaryLimiter>,
    alpha_du_limiter: Option<InteriorPointBoundaryLimiter>,
    alpha_pr_limiters: Vec<InteriorPointBoundaryLimiter>,
    alpha_du_limiters: Vec<InteriorPointBoundaryLimiter>,
) -> InteriorPointDirectionDiagnostics {
    InteriorPointDirectionDiagnostics {
        dx_inf: step_inf_norm(&direction.dx),
        d_lambda_inf: step_inf_norm(&direction.d_lambda).max(step_inf_norm(&direction.d_ineq)),
        ds_inf: step_inf_norm(&direction.ds),
        dz_inf: step_inf_norm(&direction.dz)
            .max(step_inf_norm(&direction.dz_lower))
            .max(step_inf_norm(&direction.dz_upper)),
        regularization_size: direction.regularization_used,
        primal_diagonal_shift: direction.primal_diagonal_shift_used,
        dual_regularization: direction.dual_regularization_used,
        alpha_pr_limiter,
        alpha_du_limiter,
        alpha_pr_limiters,
        alpha_du_limiters,
    }
}

fn native_bound_complementarity_sum(
    x: &[f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
) -> f64 {
    let lower_sum = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .zip(z_lower.iter())
        .map(|((&index, &lower), &z_i)| native_lower_bound_slack(x, index, lower) * z_i)
        .sum::<f64>();
    let upper_sum = bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .zip(z_upper.iter())
        .map(|((&index, &upper), &z_i)| native_upper_bound_slack(x, index, upper) * z_i)
        .sum::<f64>();
    lower_sum + upper_sum
}

fn native_bound_complementarity_inf_norm(
    x: &[f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
) -> f64 {
    let lower_inf = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .zip(z_lower.iter())
        .fold(0.0_f64, |acc, ((&index, &lower), &z_i)| {
            acc.max((native_lower_bound_slack(x, index, lower) * z_i).abs())
        });
    bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .zip(z_upper.iter())
        .fold(lower_inf, |acc, ((&index, &upper), &z_i)| {
            acc.max((native_upper_bound_slack(x, index, upper) * z_i).abs())
        })
}

fn combined_barrier_parameter(
    slack: &[f64],
    z: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
) -> f64 {
    let count = slack.len() + bounds.total_count();
    if count == 0 {
        return 0.0;
    }
    let slack_sum = slack
        .iter()
        .zip(z.iter())
        .map(|(s, z_i)| s * z_i)
        .sum::<f64>();
    (slack_sum + native_bound_complementarity_sum(x, bounds, z_lower, z_upper)) / count as f64
}

fn combined_complementarity_inf_norm(
    slack: &[f64],
    z: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
) -> f64 {
    let slack_inf = if slack.is_empty() {
        0.0
    } else {
        complementarity_inf_norm(slack, z)
    };
    slack_inf.max(native_bound_complementarity_inf_norm(
        x, bounds, z_lower, z_upper,
    ))
}

fn combined_complementarity_target_inf_norm(
    slack: &[f64],
    z: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    z_lower: &[f64],
    z_upper: &[f64],
    mu_target: f64,
) -> f64 {
    let slack_inf = interior_point_complementarity_target_inf_norm(slack, z, mu_target);
    let lower_inf = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .zip(z_lower.iter())
        .fold(0.0_f64, |acc, ((&index, &lower), &z_i)| {
            acc.max((native_lower_bound_slack(x, index, lower) * z_i - mu_target).abs())
        });
    bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .zip(z_upper.iter())
        .fold(slack_inf.max(lower_inf), |acc, ((&index, &upper), &z_i)| {
            acc.max((native_upper_bound_slack(x, index, upper) * z_i - mu_target).abs())
        })
}

fn combined_multiplier_vector<'a>(slices: impl IntoIterator<Item = &'a [f64]>) -> Vec<f64> {
    slices
        .into_iter()
        .flat_map(|slice| slice.iter().copied())
        .collect()
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
        // IPOPT `IpoptAlgorithm::correct_bound_multiplier` computes a
        // correction vector using one_over_s, then adds the negative/positive
        // correction back to z.  Preserve that order instead of assigning the
        // mathematically equivalent clamp endpoints directly.
        let one_over_s = 1.0 / *slack_i;
        let step_to_upper = upper_complementarity * one_over_s - *z_i;
        let max_correction_up = (-step_to_upper).max(0.0);
        if step_to_upper < 0.0 {
            *z_i += step_to_upper;
        }
        let step_to_lower = lower_complementarity * one_over_s - *z_i;
        let max_correction_low = step_to_lower.max(0.0);
        if step_to_lower > 0.0 {
            *z_i += step_to_lower;
        }
        max_correction = max_correction.max(max_correction_up.max(max_correction_low));
    }

    (corrected_z, max_correction)
}

#[expect(
    clippy::too_many_arguments,
    reason = "IPOPT kappa_sigma correction needs the full accepted trial multiplier state."
)]
fn apply_bound_multiplier_safeguard(
    inequality_multipliers: &[f64],
    raw_dual_residual: &[f64],
    fixed_variables: &FixedVariableElimination,
    x: &[f64],
    bounds: &BoundConstraints,
    slack: &[f64],
    z: &[f64],
    z_lower: &[f64],
    z_upper: &[f64],
    barrier_parameter_value: f64,
    options: &InteriorPointOptions,
) -> Option<AcceptedTrialMultiplierState> {
    let (corrected_z, max_correction) =
        correct_bound_multiplier_estimate(z, slack, barrier_parameter_value, options.kappa_sigma);

    let lower_slack = bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .map(|(&index, &lower)| native_lower_bound_slack(x, index, lower))
        .collect::<Vec<_>>();
    let (corrected_z_lower, lower_max_correction) = correct_bound_multiplier_estimate(
        z_lower,
        &lower_slack,
        barrier_parameter_value,
        options.kappa_sigma,
    );

    let upper_slack = bounds
        .upper_indices
        .iter()
        .zip(bounds.upper_values.iter())
        .map(|(&index, &upper)| native_upper_bound_slack(x, index, upper))
        .collect::<Vec<_>>();
    let (corrected_z_upper, upper_max_correction) = correct_bound_multiplier_estimate(
        z_upper,
        &upper_slack,
        barrier_parameter_value,
        options.kappa_sigma,
    );

    if max_correction
        .max(lower_max_correction)
        .max(upper_max_correction)
        <= 0.0
    {
        return None;
    }

    let mut corrected_dual_residual = raw_dual_residual.to_vec();
    add_native_bound_multiplier_terms(
        &mut corrected_dual_residual,
        bounds,
        &corrected_z_lower,
        &corrected_z_upper,
    );
    let corrected_dual_x_inf = fixed_variables.free_inf_norm(&corrected_dual_residual);
    let corrected_slack_stationarity_inf =
        slack_stationarity_inf_norm(inequality_multipliers, &corrected_z);
    let corrected_dual_inf = corrected_dual_x_inf.max(corrected_slack_stationarity_inf);

    Some(AcceptedTrialMultiplierState {
        z: corrected_z,
        z_lower: corrected_z_lower,
        z_upper: corrected_z_upper,
        dual_inf: corrected_dual_inf,
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
    let snapped = match (lower, upper) {
        (Some(lower), Some(upper)) if lower <= upper => value.clamp(lower, upper),
        (Some(lower), _) => value.max(lower),
        (_, Some(upper)) => value.min(upper),
        (None, None) => value,
    };
    if bound_frac <= 0.0 {
        return snapped;
    }

    // Mirrors Ipopt::DefaultIterateInitializer::push_variables: IPOPT first
    // snaps to the original bounds, then applies bound_push/bound_frac margins
    // with the same asymmetric tiny-double handling for lower and upper sides.
    let tiny_double = 100.0 * f64::MIN_POSITIVE;
    let mut lower_correction = 0.0_f64;
    if let Some(lower) = lower {
        let mut lower_margin = bound_push * lower.abs().max(1.0);
        if let Some(upper) = upper
            && upper > lower
        {
            let fractional_margin = bound_frac * (upper - lower) - tiny_double;
            if fractional_margin > 0.0 {
                lower_margin = lower_margin.min(fractional_margin);
            }
        }
        lower_correction = (lower + lower_margin - snapped).max(0.0);
    }
    let mut upper_correction = 0.0_f64;
    if let Some(upper) = upper {
        let mut upper_margin = bound_push * upper.abs().max(1.0);
        if let Some(lower) = lower
            && upper > lower
        {
            let fractional_margin = bound_frac * (upper - lower) - tiny_double;
            if fractional_margin > 0.0 {
                upper_margin = upper_margin.min(fractional_margin);
            }
        }
        upper_margin += tiny_double;
        upper_correction = (snapped - (upper - upper_margin)).max(0.0);
    }
    snapped + lower_correction - upper_correction
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

fn collect_interior_point_bounds_and_fixed<P>(
    problem: &P,
    options: &InteriorPointOptions,
) -> std::result::Result<(BoundConstraints, FixedVariableElimination), InteriorPointSolveError>
where
    P: CompiledNlpProblem,
{
    let dimension = problem.dimension();
    let Some(bounds_view) = problem.variable_bounds() else {
        return Ok((
            BoundConstraints::default(),
            FixedVariableElimination::none(dimension),
        ));
    };
    let lower = bounds_view.lower.unwrap_or_default();
    let upper = bounds_view.upper.unwrap_or_default();

    let mut bounds = BoundConstraints::default();
    let mut fixed_indices = Vec::new();
    let mut fixed_values = Vec::new();
    for idx in 0..dimension {
        let lower_bound = lower.get(idx).copied().flatten();
        let upper_bound = upper.get(idx).copied().flatten();
        if let (Some(lower_bound), Some(upper_bound)) = (lower_bound, upper_bound) {
            if lower_bound > upper_bound {
                return Err(InteriorPointSolveError::InvalidInput(format!(
                    "variable bound interval is empty at index {idx}: lower={lower_bound} > upper={upper_bound}"
                )));
            }
            if lower_bound == upper_bound {
                fixed_indices.push(idx);
                fixed_values.push(lower_bound);
                continue;
            }
        }
        if let Some(lower_bound) = lower_bound {
            bounds.lower_indices.push(idx);
            bounds
                .lower_values
                .push(lower_bound - bound_relaxation_amount(lower_bound, options));
        }
        if let Some(upper_bound) = upper_bound {
            bounds.upper_indices.push(idx);
            bounds
                .upper_values
                .push(upper_bound + bound_relaxation_amount(upper_bound, options));
        }
    }

    Ok((
        bounds,
        FixedVariableElimination::new(dimension, fixed_indices, fixed_values),
    ))
}

fn initialise_slacks(
    augmented_inequality_values: &[f64],
    upper_bounds: &[f64],
    slack: &mut [f64],
    options: &InteriorPointOptions,
) {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    debug_assert_eq!(augmented_inequality_values.len(), upper_bounds.len());
    for ((&g_i, &upper), s_i) in augmented_inequality_values
        .iter()
        .zip(upper_bounds.iter())
        .zip(slack.iter_mut())
    {
        // Mirrors Ipopt::DefaultIterateInitializer::push_variables for the
        // slack variable s with only a relaxed upper bound d_U: start from
        // trial_d() = d(x), snap to d_U if needed, then push strictly inside.
        *s_i = push_scalar_to_bounds_interior(
            g_i,
            None,
            Some(upper),
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

fn build_sparse_column_reduction(
    source: &SparseMatrixStructure,
    fixed_variables: &FixedVariableElimination,
) -> SparseColumnReduction {
    if !fixed_variables.has_fixed() {
        return SparseColumnReduction {
            structure: Arc::new(source.clone()),
            source_value_indices: (0..source.ccs.nnz()).collect(),
        };
    }

    let mut col_ptrs = Vec::with_capacity(fixed_variables.reduced_dimension() + 1);
    let mut row_indices = Vec::new();
    let mut source_value_indices = Vec::new();
    col_ptrs.push(0);
    for &source_col in &fixed_variables.free_indices {
        for source_index in source.ccs.col_ptrs[source_col]..source.ccs.col_ptrs[source_col + 1] {
            row_indices.push(source.ccs.row_indices[source_index]);
            source_value_indices.push(source_index);
        }
        col_ptrs.push(row_indices.len());
    }
    let ccs = CCS::new(
        source.ccs.nrow,
        fixed_variables.reduced_dimension(),
        col_ptrs,
        row_indices,
    );
    SparseColumnReduction {
        structure: Arc::new(sparse_structure_from_ccs(&ccs)),
        source_value_indices,
    }
}

fn reduce_sparse_matrix_columns(
    matrix: &SparseMatrix,
    reduction: &SparseColumnReduction,
) -> SparseMatrix {
    SparseMatrix {
        structure: Arc::clone(&reduction.structure),
        values: reduction
            .source_value_indices
            .iter()
            .map(|&source_index| matrix.values[source_index])
            .collect(),
    }
}

fn build_symmetric_submatrix_reduction(
    source: &CCS,
    fixed_variables: &FixedVariableElimination,
) -> SymmetricSubmatrixReduction {
    if !fixed_variables.has_fixed() {
        return SymmetricSubmatrixReduction {
            lower_triangle: Arc::new(source.clone()),
            source_value_indices: (0..source.nnz()).collect(),
        };
    }

    let mut col_ptrs = Vec::with_capacity(fixed_variables.reduced_dimension() + 1);
    let mut row_indices = Vec::new();
    let mut source_value_indices = Vec::new();
    col_ptrs.push(0);
    for &source_col in &fixed_variables.free_indices {
        for source_index in source.col_ptrs[source_col]..source.col_ptrs[source_col + 1] {
            let source_row = source.row_indices[source_index];
            if let Some(reduced_row) = fixed_variables.free_position[source_row] {
                row_indices.push(reduced_row);
                source_value_indices.push(source_index);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    SymmetricSubmatrixReduction {
        lower_triangle: Arc::new(CCS::new(
            fixed_variables.reduced_dimension(),
            fixed_variables.reduced_dimension(),
            col_ptrs,
            row_indices,
        )),
        source_value_indices,
    }
}

fn reduce_symmetric_matrix(
    matrix: &SparseSymmetricMatrix,
    reduction: &SymmetricSubmatrixReduction,
) -> SparseSymmetricMatrix {
    SparseSymmetricMatrix {
        lower_triangle: Arc::clone(&reduction.lower_triangle),
        values: reduction
            .source_value_indices
            .iter()
            .map(|&source_index| matrix.values[source_index])
            .collect(),
    }
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

struct TrialEvaluationContext<'a> {
    equality_jacobian_structure: &'a Arc<SparseMatrixStructure>,
    inequality_jacobian_structure: &'a Arc<SparseMatrixStructure>,
}

fn trial_state<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    context: &TrialEvaluationContext<'_>,
    profiling: &mut InteriorPointProfiling,
    callback_time: &mut Duration,
) -> EvalState
where
    P: CompiledNlpProblem,
{
    let mut gradient = vec![0.0; problem.dimension()];
    let mut equality_values = vec![0.0; problem.equality_count()];
    let mut augmented_inequality_values = vec![0.0; problem.inequality_count()];
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
        problem.inequality_values(x, parameters, &mut augmented_inequality_values);
    });
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
            structure: Arc::clone(context.equality_jacobian_structure),
            values: equality_jacobian_values,
        },
        inequality_jacobian: SparseMatrix {
            structure: Arc::clone(context.inequality_jacobian_structure),
            values: inequality_jacobian_values,
        },
    }
}

fn least_squares_constraint_multipliers(
    state: &EvalState,
    z: &[f64],
    hessian_structure: &CCS,
    regularization: f64,
    solver: InteriorPointLinearSolver,
) -> (Vec<f64>, Vec<f64>) {
    let debug_ls_multipliers = std::env::var_os("NLIP_DEBUG_LS_MULT").is_some();
    let meq = state.equality_values.len();
    let mineq = state.augmented_inequality_values.len();
    let total_constraints = meq + mineq;
    if total_constraints == 0 {
        return (Vec::new(), Vec::new());
    }

    let n = state.gradient.len();
    let slack_offset = n;
    let lambda_offset = slack_offset + mineq;
    let ineq_offset = lambda_offset + meq;
    let dimension = ineq_offset + mineq;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    let mut rhs = vec![0.0; dimension];
    rhs[..n].copy_from_slice(&state.gradient);
    for (row, &z_i) in z.iter().enumerate().take(mineq) {
        rhs[slack_offset + row] = z_i;
    }

    // IpLeastSquareMults.cpp passes IpNLP().uninitialized_h() with
    // W_factor=0 into StdAugSystemSolver. The numeric contribution is zero,
    // but TSymLinearSolver still includes the Hessian sparsity in the KKT
    // pattern before SPRAL's value-dependent matching analysis.
    for col in 0..hessian_structure.ncol {
        for index in hessian_structure.col_ptrs[col]..hessian_structure.col_ptrs[col + 1] {
            let row = hessian_structure.row_indices[index];
            rows.push(col.min(row));
            cols.push(col.max(row));
            values.push(0.0);
        }
    }
    for diag in 0..n {
        rows.push(diag);
        cols.push(diag);
        values.push(1.0);
    }
    for diag in 0..mineq {
        let index = slack_offset + diag;
        rows.push(index);
        cols.push(index);
        values.push(1.0);
    }
    for col in 0..n {
        for index in state.equality_jacobian.structure.ccs.col_ptrs[col]
            ..state.equality_jacobian.structure.ccs.col_ptrs[col + 1]
        {
            let dual_col = lambda_offset + state.equality_jacobian.structure.ccs.row_indices[index];
            rows.push(col.min(dual_col));
            cols.push(col.max(dual_col));
            values.push(state.equality_jacobian.values[index]);
        }
        for index in state.inequality_jacobian.structure.ccs.col_ptrs[col]
            ..state.inequality_jacobian.structure.ccs.col_ptrs[col + 1]
        {
            let dual_col = ineq_offset + state.inequality_jacobian.structure.ccs.row_indices[index];
            rows.push(col.min(dual_col));
            cols.push(col.max(dual_col));
            values.push(state.inequality_jacobian.values[index]);
        }
    }
    for row in 0..mineq {
        let slack_index = slack_offset + row;
        let dual_index = ineq_offset + row;
        rows.push(slack_index.min(dual_index));
        cols.push(slack_index.max(dual_index));
        values.push(-1.0);
    }

    let augmented_matrix = CscMatrix::new_from_triplets(dimension, dimension, rows, cols, values);
    let dsigns = quasidefinite_dsigns(n + mineq, total_constraints);
    match try_solve_symmetric_system_with_metrics(
        solver,
        &augmented_matrix,
        &rhs,
        regularization.max(1e-12),
        Some(&dsigns),
        0,
        1.0,
        regularization.max(1e-12),
    ) {
        Ok((solution, stats, _regularization_used)) => {
            let lambda_eq = solution[lambda_offset..lambda_offset + meq]
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>();
            let lambda_ineq = solution[ineq_offset..ineq_offset + mineq]
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>();
            if debug_ls_multipliers {
                let residual_current = lagrangian_gradient_sparse(
                    &state.gradient,
                    &state.equality_jacobian,
                    &lambda_eq,
                    &state.inequality_jacobian,
                    &lambda_ineq,
                );
                let neg_lambda_eq = lambda_eq.iter().map(|value| -*value).collect::<Vec<_>>();
                let neg_lambda_ineq = lambda_ineq.iter().map(|value| -*value).collect::<Vec<_>>();
                let residual_neg_eq = lagrangian_gradient_sparse(
                    &state.gradient,
                    &state.equality_jacobian,
                    &neg_lambda_eq,
                    &state.inequality_jacobian,
                    &lambda_ineq,
                );
                let residual_neg_ineq = lagrangian_gradient_sparse(
                    &state.gradient,
                    &state.equality_jacobian,
                    &lambda_eq,
                    &state.inequality_jacobian,
                    &neg_lambda_ineq,
                );
                let residual_neg_both = lagrangian_gradient_sparse(
                    &state.gradient,
                    &state.equality_jacobian,
                    &neg_lambda_eq,
                    &state.inequality_jacobian,
                    &neg_lambda_ineq,
                );
                let slack_stationarity_inf = lambda_ineq
                    .iter()
                    .zip(z.iter())
                    .fold(0.0_f64, |acc, (y_i, z_i)| acc.max((y_i - z_i).abs()));
                let rhs_plus_slack_report = {
                    let mut alt_rhs = rhs.clone();
                    for (row, &z_i) in z.iter().enumerate().take(mineq) {
                        alt_rhs[slack_offset + row] = z_i;
                    }
                    try_solve_symmetric_system_with_metrics(
                        solver,
                        &augmented_matrix,
                        &alt_rhs,
                        regularization.max(1e-12),
                        Some(&dsigns),
                        0,
                        1.0,
                        regularization.max(1e-12),
                    )
                    .ok()
                    .map(|(alt_solution, _stats, _regularization_used)| {
                        let alt_lambda_eq =
                            alt_solution[lambda_offset..lambda_offset + meq].to_vec();
                        let alt_lambda_ineq =
                            alt_solution[ineq_offset..ineq_offset + mineq].to_vec();
                        let alt_residual = lagrangian_gradient_sparse(
                            &state.gradient,
                            &state.equality_jacobian,
                            &alt_lambda_eq,
                            &state.inequality_jacobian,
                            &alt_lambda_ineq,
                        );
                        let neg_alt_lambda_eq =
                            alt_lambda_eq.iter().map(|value| -*value).collect::<Vec<_>>();
                        let neg_alt_lambda_ineq =
                            alt_lambda_ineq.iter().map(|value| -*value).collect::<Vec<_>>();
                        let alt_neg_both_residual = lagrangian_gradient_sparse(
                            &state.gradient,
                            &state.equality_jacobian,
                            &neg_alt_lambda_eq,
                            &state.inequality_jacobian,
                            &neg_alt_lambda_ineq,
                        );
                        let alt_slack_stationarity_inf = alt_lambda_ineq
                            .iter()
                            .zip(z.iter())
                            .fold(0.0_f64, |acc, (y_i, z_i)| acc.max((y_i - z_i).abs()));
                        let alt_neg_slack_stationarity_inf = neg_alt_lambda_ineq
                            .iter()
                            .zip(z.iter())
                            .fold(0.0_f64, |acc, (y_i, z_i)| acc.max((y_i - z_i).abs()));
                        format!(
                            "rhs_plus_lag={:.6e} rhs_plus_slack_stat={:.6e} rhs_plus_neg_both_lag={:.6e} rhs_plus_neg_slack_stat={:.6e} rhs_plus_lambda_eq_inf={:.6e} rhs_plus_lambda_ineq_inf={:.6e}",
                            inf_norm(&alt_residual),
                            alt_slack_stationarity_inf,
                            inf_norm(&alt_neg_both_residual),
                            alt_neg_slack_stationarity_inf,
                            inf_norm(&alt_lambda_eq),
                            inf_norm(&alt_lambda_ineq),
                        )
                    })
                    .unwrap_or_else(|| "rhs_plus_failed".to_string())
                };
                eprintln!(
                    "NLIP_DEBUG_LS_MULT success solver={} n={} meq={} mineq={} lambda_eq_inf={:.6e} lambda_ineq_inf={:.6e} residual_inf={:.6e} solution_inf={:.6e} inertia={} lag_current={:.6e} lag_neg_eq={:.6e} lag_neg_ineq={:.6e} lag_neg_both={:.6e} slack_stat={:.6e} {}",
                    solver.label(),
                    n,
                    meq,
                    mineq,
                    inf_norm(&lambda_eq),
                    inf_norm(&lambda_ineq),
                    stats.residual_inf,
                    stats.solution_inf,
                    stats.inertia.map_or_else(
                        || "none".to_owned(),
                        |inertia| format!(
                            "+{}/-{}/0{}",
                            inertia.positive, inertia.negative, inertia.zero
                        )
                    ),
                    inf_norm(&residual_current),
                    inf_norm(&residual_neg_eq),
                    inf_norm(&residual_neg_ineq),
                    inf_norm(&residual_neg_both),
                    slack_stationarity_inf,
                    rhs_plus_slack_report,
                );
            }
            (lambda_eq, lambda_ineq)
        }
        Err(attempts) => {
            if debug_ls_multipliers {
                let details = attempts
                    .iter()
                    .map(|attempt| {
                        format!(
                            "{}:{} reg={:.3e} detail={}",
                            attempt.solver.label(),
                            attempt.failure_kind.label(),
                            attempt.regularization,
                            attempt.detail.as_deref().unwrap_or("none")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!(
                    "NLIP_DEBUG_LS_MULT fallback_zero n={} meq={} mineq={} attempts=[{}]",
                    n, meq, mineq, details
                );
            }
            (vec![0.0; meq], vec![0.0; mineq])
        }
    }
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

fn symmetric_csc_lower_from_any_triangle(matrix: &CscMatrix<f64>) -> CscMatrix<f64> {
    debug_assert_eq!(matrix.m, matrix.n);
    let mut rows = Vec::with_capacity(matrix.nzval.len());
    let mut cols = Vec::with_capacity(matrix.nzval.len());
    let mut values = Vec::with_capacity(matrix.nzval.len());
    for col in 0..matrix.n {
        for index in matrix.colptr[col]..matrix.colptr[col + 1] {
            let row = matrix.rowval[index];
            rows.push(row.max(col));
            cols.push(row.min(col));
            values.push(matrix.nzval[index]);
        }
    }
    CscMatrix::new_from_triplets(matrix.m, matrix.n, rows, cols, values)
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
            inertia: None,
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
            inertia: None,
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
            inertia: None,
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
            inertia: None,
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
            inertia: None,
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
            inertia: None,
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

const IPOPT_LINEAR_MIN_REFINEMENT_STEPS: usize = 1;
const IPOPT_LINEAR_MAX_REFINEMENT_STEPS: usize = 10;
const IPOPT_LINEAR_RESIDUAL_RATIO_MAX: f64 = 1e-10;
const IPOPT_LINEAR_RESIDUAL_RATIO_SINGULAR: f64 = 1e-5;
const IPOPT_LINEAR_RESIDUAL_IMPROVEMENT_FACTOR: f64 = 0.999_999_999;
const IPOPT_LINEAR_RESIDUAL_MAX_COND: f64 = 1e6;

fn ipopt_refinement_residual_ratio_ccs(
    ccs: &CCS,
    values: &[f64],
    rhs: &[f64],
    solution: &[f64],
) -> (Vec<f64>, f64) {
    let residual = symmetric_ccs_lower_mat_vec(ccs, values, solution)
        .into_iter()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs_i)| lhs - rhs_i)
        .collect::<Vec<_>>();
    let rhs_inf = rhs.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let solution_inf = solution
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let denominator = solution_inf.min(IPOPT_LINEAR_RESIDUAL_MAX_COND * rhs_inf) + rhs_inf;
    let ratio = if denominator == 0.0 {
        residual_inf
    } else {
        residual_inf / denominator
    };
    (residual, ratio)
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
    let (mut residual, mut residual_ratio) =
        ipopt_refinement_residual_ratio_ccs(ccs, values, rhs, solution);
    let mut previous_residual_ratio = residual_ratio;
    while steps < IPOPT_LINEAR_MIN_REFINEMENT_STEPS
        || residual_ratio > IPOPT_LINEAR_RESIDUAL_RATIO_MAX
    {
        let correction_started = Instant::now();
        let correction = solve_correction(&residual)?;
        *solve_time += correction_started.elapsed();
        for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
            *solution_i -= correction_i;
        }
        steps += 1;
        let (new_residual, new_residual_ratio) =
            ipopt_refinement_residual_ratio_ccs(ccs, values, rhs, solution);
        if new_residual_ratio > IPOPT_LINEAR_RESIDUAL_RATIO_MAX
            && steps > IPOPT_LINEAR_MIN_REFINEMENT_STEPS
            && (steps > IPOPT_LINEAR_MAX_REFINEMENT_STEPS
                || new_residual_ratio
                    > IPOPT_LINEAR_RESIDUAL_IMPROVEMENT_FACTOR * previous_residual_ratio)
        {
            break;
        }
        residual = new_residual;
        previous_residual_ratio = new_residual_ratio;
        residual_ratio = new_residual_ratio;
    }
    Ok(steps)
}

struct IpoptFullSpaceResidual {
    augmented_correction_rhs: Vec<f64>,
    residual_ratio: f64,
    rhs_inf: f64,
    solution_inf: f64,
    residual_inf: f64,
    residual_x_inf: f64,
    residual_s_inf: f64,
    residual_c_inf: f64,
    residual_d_inf: f64,
    residual_bound_inf: f64,
    residual_vu_inf: f64,
}

#[derive(Clone, Copy, Debug)]
struct IpoptRefinementReport {
    steps: usize,
    initial_residual_ratio: f64,
    initial_residual: IpoptFullSpaceResidualMetrics,
    residual_ratio: f64,
    final_residual: IpoptFullSpaceResidualMetrics,
    failed: bool,
}

#[derive(Clone, Copy, Debug)]
struct IpoptFullSpaceResidualMetrics {
    rhs_inf: f64,
    solution_inf: f64,
    residual_inf: f64,
    residual_x_inf: f64,
    residual_s_inf: f64,
    residual_c_inf: f64,
    residual_d_inf: f64,
    residual_bound_inf: f64,
    residual_vu_inf: f64,
}

impl IpoptFullSpaceResidual {
    fn metrics(&self) -> IpoptFullSpaceResidualMetrics {
        IpoptFullSpaceResidualMetrics {
            rhs_inf: self.rhs_inf,
            solution_inf: self.solution_inf,
            residual_inf: self.residual_inf,
            residual_x_inf: self.residual_x_inf,
            residual_s_inf: self.residual_s_inf,
            residual_c_inf: self.residual_c_inf,
            residual_d_inf: self.residual_d_inf,
            residual_bound_inf: self.residual_bound_inf,
            residual_vu_inf: self.residual_vu_inf,
        }
    }
}

fn ipopt_full_space_residual_metrics_text(metrics: IpoptFullSpaceResidualMetrics) -> String {
    format!(
        "rhs={:.3e},sol={:.3e},res={:.3e},x={:.3e},s={:.3e},c={:.3e},d={:.3e},bound={:.3e},vU={:.3e}",
        metrics.rhs_inf,
        metrics.solution_inf,
        metrics.residual_inf,
        metrics.residual_x_inf,
        metrics.residual_s_inf,
        metrics.residual_c_inf,
        metrics.residual_d_inf,
        metrics.residual_bound_inf,
        metrics.residual_vu_inf,
    )
}

#[derive(Clone, Copy)]
struct IpoptLinearRefinementShifts {
    primal: f64,
    slack: f64,
    dual: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IpoptLinearRhsOrientation {
    PreFinal,
    FinalDirection,
}

#[derive(Clone, Copy)]
struct IpoptLinearSolveContext {
    shifts: IpoptLinearRefinementShifts,
    rhs_orientation: IpoptLinearRhsOrientation,
    native_spral_quality_was_increased: bool,
}

fn ipopt_full_space_residual_ratio(
    system: &ReducedKktSystem<'_>,
    pattern: &SpralAugmentedKktPattern,
    solution: &[f64],
    rhs_orientation: IpoptLinearRhsOrientation,
    shifts: IpoptLinearRefinementShifts,
) -> IpoptFullSpaceResidual {
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let dx = &solution[..n];
    let ipopt_ds = &solution[pattern.p_offset..pattern.p_offset + mineq];
    let d_lambda = &solution[pattern.lambda_offset..pattern.lambda_offset + meq];
    let d_ineq = &solution[pattern.z_offset..pattern.z_offset + mineq];
    let prefinal_orientation = rhs_orientation == IpoptLinearRhsOrientation::PreFinal;

    let mut residual_x = symmetric_ccs_lower_mat_vec(
        system.hessian.lower_triangle.as_ref(),
        &system.hessian.values,
        dx,
    );
    for (index, residual_i) in residual_x.iter_mut().enumerate() {
        *residual_i += shifts.primal * dx[index];
    }
    sparse_add_transpose_mat_vec(&mut residual_x, system.equality_jacobian, d_lambda);
    sparse_add_transpose_mat_vec(&mut residual_x, system.inequality_jacobian, d_ineq);

    let equality_dx = sparse_mat_vec(system.equality_jacobian, dx);
    let inequality_dx = sparse_mat_vec(system.inequality_jacobian, dx);
    let mut augmented_correction_rhs = vec![0.0; pattern.dimension()];

    let mut rhs_inf = 0.0_f64;
    let mut solution_inf = step_inf_norm(dx);
    let mut bound_residual_inf = 0.0_f64;

    if let Some(bound_data) = system.bound_data {
        let mut rhs_x = system.r_dual.to_vec();
        for (residual_i, r_dual_i) in residual_x.iter_mut().zip(system.r_dual.iter()) {
            if prefinal_orientation {
                *residual_i -= *r_dual_i;
            } else {
                *residual_i += *r_dual_i;
            }
        }
        let damping = system_positive_slack_damping(system);
        for ((&index, &lower), &z_i) in bound_data
            .bounds
            .lower_indices
            .iter()
            .zip(bound_data.bounds.lower_values.iter())
            .zip(bound_data.z_lower.iter())
        {
            let Some(reduced_index) = bound_data.fixed_variables.free_position[index] else {
                continue;
            };
            let slack = native_lower_bound_slack(bound_data.x, index, lower);
            let complementarity = slack * z_i - system.barrier_parameter;
            let dz_i = if prefinal_orientation {
                (complementarity - z_i * dx[reduced_index]) / slack
            } else {
                (-complementarity - z_i * dx[reduced_index]) / slack
            };
            let residual_z = if prefinal_orientation {
                slack * dz_i + z_i * dx[reduced_index] - complementarity
            } else {
                slack * dz_i + z_i * dx[reduced_index] + complementarity
            };
            residual_x[reduced_index] -= dz_i;
            if damping > 0.0 && !bound_data.bounds.upper_indices.contains(&index) {
                if prefinal_orientation {
                    residual_x[reduced_index] -= damping;
                } else {
                    residual_x[reduced_index] += damping;
                }
                rhs_x[reduced_index] += damping;
            }
            augmented_correction_rhs[reduced_index] += residual_z / slack;
            bound_residual_inf = bound_residual_inf.max(residual_z.abs());
            rhs_inf = rhs_inf.max(complementarity.abs());
            solution_inf = solution_inf.max(dz_i.abs());
        }
        for ((&index, &upper), &z_i) in bound_data
            .bounds
            .upper_indices
            .iter()
            .zip(bound_data.bounds.upper_values.iter())
            .zip(bound_data.z_upper.iter())
        {
            let Some(reduced_index) = bound_data.fixed_variables.free_position[index] else {
                continue;
            };
            let slack = native_upper_bound_slack(bound_data.x, index, upper);
            let complementarity = slack * z_i - system.barrier_parameter;
            let dz_i = if prefinal_orientation {
                (complementarity + z_i * dx[reduced_index]) / slack
            } else {
                (-complementarity + z_i * dx[reduced_index]) / slack
            };
            let residual_z = if prefinal_orientation {
                slack * dz_i - z_i * dx[reduced_index] - complementarity
            } else {
                slack * dz_i - z_i * dx[reduced_index] + complementarity
            };
            residual_x[reduced_index] += dz_i;
            if damping > 0.0 && !bound_data.bounds.lower_indices.contains(&index) {
                if prefinal_orientation {
                    residual_x[reduced_index] += damping;
                } else {
                    residual_x[reduced_index] -= damping;
                }
                rhs_x[reduced_index] -= damping;
            }
            augmented_correction_rhs[reduced_index] -= residual_z / slack;
            bound_residual_inf = bound_residual_inf.max(residual_z.abs());
            rhs_inf = rhs_inf.max(complementarity.abs());
            solution_inf = solution_inf.max(dz_i.abs());
        }
        for (rhs_i, residual_i) in augmented_correction_rhs[..n]
            .iter_mut()
            .zip(residual_x.iter())
        {
            *rhs_i += *residual_i;
        }
        rhs_inf = rhs_inf.max(step_inf_norm(&rhs_x));
    } else {
        for (residual_i, (r_dual_i, bound_rhs_i)) in residual_x
            .iter_mut()
            .zip(system.r_dual.iter().zip(system.bound_rhs.iter()))
        {
            let rhs_i = if prefinal_orientation {
                *r_dual_i - *bound_rhs_i
            } else {
                -*r_dual_i + *bound_rhs_i
            };
            *residual_i -= rhs_i;
        }
        for (index, residual_i) in residual_x.iter_mut().enumerate() {
            *residual_i += system.bound_diagonal.get(index).copied().unwrap_or(0.0) * dx[index];
        }
        augmented_correction_rhs[..n].copy_from_slice(&residual_x);
        rhs_inf = system
            .r_dual
            .iter()
            .zip(system.bound_rhs.iter())
            .fold(0.0_f64, |acc, (r_dual_i, bound_rhs_i)| {
                acc.max((*r_dual_i - *bound_rhs_i).abs())
            });
    }

    let mut residual_inf = step_inf_norm(&residual_x).max(bound_residual_inf);
    let mut residual_s_inf = 0.0_f64;
    let mut residual_d_inf = 0.0_f64;
    let mut residual_vu_inf = 0.0_f64;

    for row in 0..mineq {
        let slack = system.slack[row];
        let multiplier = system.multipliers[row];
        let ipopt_ds_i = ipopt_ds[row];
        let dz_i = if prefinal_orientation {
            ipopt_prefinal_upper_slack_bound_multiplier_step(
                slack,
                multiplier,
                system.r_cent[row],
                ipopt_ds_i,
            )
        } else {
            ipopt_upper_slack_bound_multiplier_step(
                slack,
                multiplier,
                system.r_cent[row],
                ipopt_ds_i,
            )
        };
        let damped_slack_stationarity = damped_slack_stationarity_residual(system, row);
        let (residual_s, residual_v, residual_d) = if prefinal_orientation {
            (
                dz_i - d_ineq[row] - damped_slack_stationarity + shifts.slack * ipopt_ds_i,
                slack * dz_i - multiplier * ipopt_ds_i - system.r_cent[row],
                inequality_dx[row] - ipopt_ds_i - shifts.dual * d_ineq[row] - system.r_ineq[row],
            )
        } else {
            (
                dz_i - d_ineq[row] + damped_slack_stationarity + shifts.slack * ipopt_ds_i,
                slack * dz_i - multiplier * ipopt_ds_i + system.r_cent[row],
                inequality_dx[row] - ipopt_ds_i - shifts.dual * d_ineq[row] + system.r_ineq[row],
            )
        };
        residual_s_inf = residual_s_inf.max(residual_s.abs());
        residual_d_inf = residual_d_inf.max(residual_d.abs());
        residual_vu_inf = residual_vu_inf.max(residual_v.abs());
        // Mirrors Ipopt::PDFullSpaceSolver::SolveOnce: full-space residuals
        // from ComputeResiduals are converted to the augmented-system RHS with
        // Pd_U.AddMSinvZ(-1.0, ...), i.e. rhs_s - rhs_v_U / slack_s_U.
        augmented_correction_rhs[pattern.p_offset + row] = residual_s - residual_v / slack;
        augmented_correction_rhs[pattern.z_offset + row] = residual_d;
        residual_inf = residual_inf
            .max(residual_s.abs())
            .max(residual_v.abs())
            .max(residual_d.abs());
        rhs_inf = rhs_inf
            .max(damped_slack_stationarity.abs())
            .max(system.r_cent[row].abs())
            .max(system.r_ineq[row].abs());
        solution_inf = solution_inf
            .max(ipopt_ds_i.abs())
            .max(d_ineq[row].abs())
            .max(dz_i.abs());
    }

    let mut residual_c_inf = 0.0_f64;
    for row in 0..meq {
        let residual_c = if prefinal_orientation {
            equality_dx[row] - shifts.dual * d_lambda[row] - system.r_eq[row]
        } else {
            equality_dx[row] - shifts.dual * d_lambda[row] + system.r_eq[row]
        };
        augmented_correction_rhs[pattern.lambda_offset + row] = residual_c;
        residual_c_inf = residual_c_inf.max(residual_c.abs());
        residual_inf = residual_inf.max(residual_c.abs());
        rhs_inf = rhs_inf.max(system.r_eq[row].abs());
        solution_inf = solution_inf.max(d_lambda[row].abs());
    }

    let residual_ratio = if rhs_inf + solution_inf == 0.0 {
        residual_inf
    } else {
        let denominator = solution_inf.min(IPOPT_LINEAR_RESIDUAL_MAX_COND * rhs_inf) + rhs_inf;
        residual_inf / denominator
    };
    IpoptFullSpaceResidual {
        augmented_correction_rhs,
        residual_ratio,
        rhs_inf,
        solution_inf,
        residual_inf,
        residual_x_inf: step_inf_norm(&residual_x),
        residual_s_inf,
        residual_c_inf,
        residual_d_inf,
        residual_bound_inf: bound_residual_inf,
        residual_vu_inf,
    }
}

fn refine_ipopt_full_space_solution<E>(
    system: &ReducedKktSystem<'_>,
    pattern: &SpralAugmentedKktPattern,
    solution: &mut [f64],
    rhs_orientation: IpoptLinearRhsOrientation,
    shifts: IpoptLinearRefinementShifts,
    solve_time: &mut Duration,
    mut solve_correction: impl FnMut(&[f64]) -> Result<Vec<f64>, E>,
) -> Result<IpoptRefinementReport, E> {
    // IPOPT refines PDFullSpaceSolver systems on the full unsymmetric
    // primal-dual residual, not on the already-eliminated symmetric augmented
    // matrix residual; see IpPDFullSpaceSolver.cpp::ComputeResiduals,
    // SolveOnce, and ComputeResidualRatio.
    let mut steps = 0;
    let mut residual =
        ipopt_full_space_residual_ratio(system, pattern, solution, rhs_orientation, shifts);
    let mut residual_ratio = residual.residual_ratio;
    let initial_residual_ratio = residual_ratio;
    let initial_residual = residual.metrics();
    let mut previous_residual_ratio = residual_ratio;
    let mut failed = false;
    while steps < IPOPT_LINEAR_MIN_REFINEMENT_STEPS
        || residual_ratio > IPOPT_LINEAR_RESIDUAL_RATIO_MAX
    {
        let correction_started = Instant::now();
        let correction = solve_correction(&residual.augmented_correction_rhs)?;
        *solve_time += correction_started.elapsed();
        for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
            *solution_i -= correction_i;
        }
        steps += 1;
        let new_residual =
            ipopt_full_space_residual_ratio(system, pattern, solution, rhs_orientation, shifts);
        let new_residual_ratio = new_residual.residual_ratio;
        residual_ratio = new_residual_ratio;
        if new_residual_ratio > IPOPT_LINEAR_RESIDUAL_RATIO_MAX
            && steps > IPOPT_LINEAR_MIN_REFINEMENT_STEPS
            && (steps > IPOPT_LINEAR_MAX_REFINEMENT_STEPS
                || new_residual_ratio
                    > IPOPT_LINEAR_RESIDUAL_IMPROVEMENT_FACTOR * previous_residual_ratio)
        {
            failed = true;
            residual = new_residual;
            break;
        }
        residual = new_residual;
        previous_residual_ratio = new_residual_ratio;
    }
    Ok(IpoptRefinementReport {
        steps,
        initial_residual_ratio,
        initial_residual,
        residual_ratio,
        final_residual: residual.metrics(),
        failed,
    })
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
        solver: InteriorPointLinearSolver::SsidsRs,
        regularization,
        inertia: None,
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
        solver: InteriorPointLinearSolver::SpralSrc,
        regularization,
        inertia: None,
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
            inertia: None,
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
            inertia: None,
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
            inertia: None,
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

fn factor_solve_spral_ssids_symmetric_with_metrics(
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats), InteriorPointLinearSolveAttempt> {
    let n = matrix.n;
    if matrix.m != n || rhs.len() != n {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SsidsRs,
            regularization,
            inertia: None,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: None,
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        });
    }
    let lower_matrix = symmetric_csc_lower_from_any_triangle(matrix);
    let spral_matrix = SpralSymmetricCscMatrix::new(
        n,
        &lower_matrix.colptr,
        &lower_matrix.rowval,
        Some(&lower_matrix.nzval),
    )
    .map_err(|error| {
        spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let ccs = CCS::new(
        n,
        n,
        lower_matrix.colptr.clone(),
        lower_matrix.rowval.clone(),
    );
    let factor_started = Instant::now();
    let (symbolic, _) = spral_analyse(
        spral_matrix,
        &SpralSsidsOptions {
            // IPOPT's SPRAL path does not have this crate's `Auto` heuristic.
            // Keep sparse auxiliary solves on explicit AMD, matching the Rust
            // augmented-KKT integration and avoiding Auto's natural-order fill
            // simulation on glider-sized least-squares multiplier systems.
            ordering: SpralOrderingStrategy::ApproximateMinimumDegree,
        },
    )
    .map_err(|error| {
        spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let mut factor = spral_factorize(
        spral_matrix,
        &symbolic,
        &SpralNumericFactorOptions::default(),
    )
    .map_err(|error| {
        spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?
    .0;
    let factorization_time = factor_started.elapsed();
    let inertia = interior_point_linear_inertia(factor.inertia());
    let solve_started = Instant::now();
    let mut solution = factor.solve(rhs).map_err(|error| {
        spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let mut solve_time = solve_started.elapsed();
    let refinement_steps = refine_linear_solution_ccs(
        &ccs,
        &lower_matrix.nzval,
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
    let assessment = assess_linear_solution_ccs(&ccs, &lower_matrix.nzval, rhs, &solution);
    assessment
        .map(|assessment| {
            (
                solution,
                LinearBackendRunStats {
                    solver: InteriorPointLinearSolver::SsidsRs,
                    factorization_time,
                    solve_time,
                    reused_symbolic: Some(false),
                    inertia: Some(inertia),
                    residual_inf: assessment.residual_inf,
                    solution_inf: assessment.solution_inf,
                    detail: (refinement_steps > 0)
                        .then(|| format!("iterative_refinement_steps={refinement_steps}")),
                },
            )
        })
        .map_err(|mut attempt| {
            attempt.solver = InteriorPointLinearSolver::SsidsRs;
            attempt.regularization = regularization;
            attempt
        })
}

fn factor_solve_spral_src_symmetric_with_metrics(
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats), InteriorPointLinearSolveAttempt> {
    let n = matrix.n;
    if matrix.m != n || rhs.len() != n {
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SpralSrc,
            regularization,
            inertia: None,
            failure_kind: InteriorPointLinearSolveFailureKind::FactorizationFailed,
            detail: None,
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        });
    }
    let lower_matrix = symmetric_csc_lower_from_any_triangle(matrix);
    let spral_matrix = SpralSymmetricCscMatrix::new(
        n,
        &lower_matrix.colptr,
        &lower_matrix.rowval,
        Some(&lower_matrix.nzval),
    )
    .map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let ccs = CCS::new(
        n,
        n,
        lower_matrix.colptr.clone(),
        lower_matrix.rowval.clone(),
    );
    let native = NativeSpralLibrary::load().map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let factor_started = Instant::now();
    let mut session = native
        // IpLeastSquareMults.cpp reaches SPRAL through StdAugSystemSolver and
        // IpSpralSolverInterface, not the generic native wrapper defaults.
        // Keep auxiliary multiplier solves on the same matching, one-based
        // ptr32 path used by IPOPT's SPRAL interface.
        .analyse_ipopt_compatible_with_options_and_ordering(
            spral_matrix,
            &SpralNumericFactorOptions::default(),
            SpralNativeOrdering::Matching,
        )
        .map_err(|error| {
            native_spral_error_attempt(
                regularization,
                InteriorPointLinearSolveFailureKind::FactorizationFailed,
                error,
            )
        })?;
    let factor_info = session.factorize(spral_matrix).map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let factorization_time = factor_started.elapsed();
    let inertia = interior_point_linear_inertia(factor_info.inertia);
    let solve_started = Instant::now();
    // IPOPT's SpralSolverInterface::MultiSolve calls spral_ssids_solve with
    // nrhs=1 even for a single right-hand side. Keep the NLIP source-built
    // SPRAL path on the same C entrypoint instead of spral_ssids_solve1.
    let solution = session.solve_ipopt_single_rhs(rhs).map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let solve_time = solve_started.elapsed();
    // IPOPT LeastSquareMultipliers calls StdAugSystemSolver::Solve directly;
    // the full-space PDFullSpaceSolver iterative refinement loop is not used
    // for this auxiliary multiplier estimate.
    let assessment = assess_linear_solution_ccs(&ccs, &lower_matrix.nzval, rhs, &solution);
    let (solution_inf, residual_inf) = match assessment {
        Ok(assessment) => (assessment.solution_inf, assessment.residual_inf),
        Err(attempt) => (
            attempt
                .solution_inf
                .unwrap_or_else(|| step_inf_norm(&solution)),
            attempt.residual_inf.unwrap_or(f64::NAN),
        ),
    };
    Ok((
        solution,
        LinearBackendRunStats {
            solver: InteriorPointLinearSolver::SpralSrc,
            factorization_time,
            solve_time,
            reused_symbolic: Some(false),
            inertia: Some(inertia),
            residual_inf,
            solution_inf,
            detail: Some("ipopt_least_square_multipliers_std_aug_solver_no_refinement".to_string()),
        },
    ))
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
        InteriorPointLinearSolver::Auto | InteriorPointLinearSolver::SsidsRs => {
            InteriorPointLinearSolver::SsidsRs
        }
        InteriorPointLinearSolver::SpralSrc => InteriorPointLinearSolver::SpralSrc,
        InteriorPointLinearSolver::SparseQdldl => InteriorPointLinearSolver::SparseQdldl,
    }
}

fn secondary_linear_solver(solver: InteriorPointLinearSolver) -> Option<InteriorPointLinearSolver> {
    match solver {
        InteriorPointLinearSolver::Auto => Some(InteriorPointLinearSolver::SparseQdldl),
        InteriorPointLinearSolver::SsidsRs
        | InteriorPointLinearSolver::SpralSrc
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
fn try_solve_symmetric_system_with_metrics(
    solver: InteriorPointLinearSolver,
    matrix: &CscMatrix<f64>,
    rhs: &[f64],
    regularization: f64,
    dsigns: Option<&[i8]>,
    adaptive_regularization_retries: Index,
    regularization_growth_factor: f64,
    regularization_max: f64,
) -> std::result::Result<(Vec<f64>, LinearBackendRunStats, f64), Vec<InteriorPointLinearSolveAttempt>>
{
    let preferred_solver = preferred_linear_solver(solver, 0, 0);
    let try_solver = |matrix: &CscMatrix<f64>,
                      rhs: &[f64],
                      retries: Index,
                      growth_factor: f64,
                      regularization_max: f64| {
        let mut attempts = Vec::new();
        let mut current_regularization = regularization.max(1e-12);
        let max_regularization = regularization_max.max(current_regularization);
        for retry_index in 0..=retries {
            let attempt_result = match preferred_solver {
                InteriorPointLinearSolver::SpralSrc => {
                    factor_solve_spral_src_symmetric_with_metrics(
                        matrix,
                        rhs,
                        current_regularization,
                    )
                }
                InteriorPointLinearSolver::SsidsRs => {
                    factor_solve_spral_ssids_symmetric_with_metrics(
                        matrix,
                        rhs,
                        current_regularization,
                    )
                }
                InteriorPointLinearSolver::SparseQdldl | InteriorPointLinearSolver::Auto => {
                    factor_solve_sparse_qdldl_with_metrics(
                        matrix,
                        rhs,
                        current_regularization,
                        dsigns,
                    )
                }
            };
            match attempt_result {
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
    let sparse_result = try_solver(
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

fn spral_augmented_kkt_regularization_shifts(
    system: &ReducedKktSystem<'_>,
    solver: InteriorPointLinearSolver,
    regularization: f64,
) -> (f64, f64) {
    match solver {
        InteriorPointLinearSolver::SpralSrc => (regularization.max(0.0), 0.0),
        InteriorPointLinearSolver::SsidsRs | InteriorPointLinearSolver::Auto => (
            sparse_hessian_diagonal_shift(system.hessian, regularization),
            regularization.max(1e-8),
        ),
        InteriorPointLinearSolver::SparseQdldl => (
            sparse_hessian_diagonal_shift(system.hessian, regularization),
            regularization.max(1e-8),
        ),
    }
}

fn next_ipopt_hessian_perturbation(
    current: f64,
    previous_successful: Option<f64>,
    first: f64,
    first_growth_factor: f64,
    growth_factor: f64,
    decay_factor: f64,
    max_regularization: f64,
) -> Option<f64> {
    let previous_successful = previous_successful.unwrap_or(0.0).max(0.0);
    let next = if current <= 0.0 {
        if previous_successful <= 0.0 {
            first.max(0.0)
        } else {
            (previous_successful * decay_factor.clamp(0.0, 1.0)).max(1e-20)
        }
    } else if previous_successful <= 0.0 || 1e5 * previous_successful < current {
        current * first_growth_factor.max(1.0)
    } else {
        current * growth_factor.max(1.0)
    };
    if next <= current || next > max_regularization {
        None
    } else {
        Some(next.min(max_regularization))
    }
}

fn ipopt_jacobian_perturbation(system: &ReducedKktSystem<'_>) -> f64 {
    system.jacobian_regularization_value
        * system
            .barrier_parameter
            .max(0.0)
            .powf(system.jacobian_regularization_exponent)
}

fn is_factorization_failure(attempt: &InteriorPointLinearSolveAttempt) -> bool {
    attempt.failure_kind == InteriorPointLinearSolveFailureKind::FactorizationFailed
}

fn is_singularity_like_linear_failure(attempt: &InteriorPointLinearSolveAttempt) -> bool {
    is_factorization_failure(attempt)
        || (attempt.failure_kind == InteriorPointLinearSolveFailureKind::InertiaMismatch
            && attempt
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("negative_eigenvalues_too_few")))
        || (attempt.failure_kind == InteriorPointLinearSolveFailureKind::ResidualTooLarge
            && attempt.detail.as_deref().is_some_and(|detail| {
                detail.contains("ipopt_pretend_singular_after_refinement_failure")
            }))
}

fn is_negative_eigenvalues_too_few(attempt: &InteriorPointLinearSolveAttempt) -> bool {
    attempt.failure_kind == InteriorPointLinearSolveFailureKind::InertiaMismatch
        && attempt
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("negative_eigenvalues_too_few"))
}

fn append_attempt_detail(attempt: &mut InteriorPointLinearSolveAttempt, detail: impl Into<String>) {
    let detail = detail.into();
    attempt.detail = Some(match attempt.detail.take() {
        Some(existing) => format!("{existing}; {detail}"),
        None => detail,
    });
}

fn native_spral_quality_can_increase(
    workspace: &NativeSpralAugmentedKktWorkspace,
    system: &ReducedKktSystem<'_>,
) -> bool {
    let old_u = workspace.numeric_options.threshold_pivot_u;
    let max_u = system.spral_pivot_tolerance_max.max(0.0);
    if !old_u.is_finite() || !max_u.is_finite() || old_u >= max_u {
        return false;
    }
    let next_u = old_u.powf(0.75).min(max_u);
    next_u.is_finite() && next_u > old_u
}

fn try_increase_native_spral_quality(
    workspace: &mut NativeSpralAugmentedKktWorkspace,
    system: &ReducedKktSystem<'_>,
) -> Option<(f64, f64)> {
    if !native_spral_quality_can_increase(workspace, system) {
        return None;
    }
    let old_u = workspace.numeric_options.threshold_pivot_u;
    let next_u = old_u
        .powf(0.75)
        .min(system.spral_pivot_tolerance_max.max(0.0));
    workspace.numeric_options.threshold_pivot_u = next_u;
    workspace.session = None;
    workspace.factor_regularization = None;
    Some((old_u, next_u))
}

fn linear_solver_quality_was_increased(stats: &LinearBackendRunStats) -> bool {
    stats
        .detail
        .as_deref()
        .is_some_and(|detail| detail.contains("spral_quality_retry="))
}

fn inertia_mismatch_detail(expected: SpralInertia, actual: SpralInertia) -> String {
    let prefix = if actual.negative < expected.negative {
        "negative_eigenvalues_too_few; "
    } else if actual.negative > expected.negative {
        "negative_eigenvalues_too_many; "
    } else {
        ""
    };
    format!(
        "{prefix}expected inertia (+{}, -{}, 0={}), got (+{}, -{}, 0={})",
        expected.positive,
        expected.negative,
        expected.zero,
        actual.positive,
        actual.negative,
        actual.zero
    )
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
    slack_shift: f64,
    dual_shift: f64,
) {
    values.fill(0.0);
    for (index, &slot) in pattern.hessian_value_indices.iter().enumerate() {
        values[slot] += system.hessian.values[index];
    }
    for (index, &slot) in pattern.x_diagonal_indices.iter().enumerate() {
        values[slot] += primal_shift;
        values[slot] += system.bound_diagonal.get(index).copied().unwrap_or(0.0);
    }
    for (index, &slot) in pattern.equality_jacobian_value_indices.iter().enumerate() {
        values[slot] += system.equality_jacobian.values[index];
    }
    for (index, &slot) in pattern.inequality_jacobian_value_indices.iter().enumerate() {
        values[slot] += system.inequality_jacobian.values[index];
    }
    for (index, &slot) in pattern.p_diagonal_indices.iter().enumerate() {
        values[slot] += system.multipliers[index] / system.slack[index] + slack_shift;
    }
    for &slot in &pattern.lambda_diagonal_indices {
        values[slot] += -dual_shift;
    }
    for &slot in &pattern.z_diagonal_indices {
        values[slot] += -dual_shift;
    }
    for &slot in &pattern.pz_indices {
        values[slot] += -1.0;
    }
}

fn assemble_spral_augmented_kkt_values(
    workspace: &mut SpralAugmentedKktWorkspace,
    system: &ReducedKktSystem<'_>,
    primal_shift: f64,
    slack_shift: f64,
    dual_shift: f64,
) {
    fill_spral_augmented_kkt_values(
        &workspace.pattern,
        &mut workspace.values,
        system,
        primal_shift,
        slack_shift,
        dual_shift,
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
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SsidsRs,
            regularization,
            inertia: Some(Box::new(interior_point_linear_inertia(actual_inertia))),
            failure_kind: InteriorPointLinearSolveFailureKind::InertiaMismatch,
            detail: Some(inertia_mismatch_detail(expected_inertia, actual_inertia)),
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        });
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
                    solver: InteriorPointLinearSolver::SsidsRs,
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
            attempt.solver = InteriorPointLinearSolver::SsidsRs;
            attempt.regularization = regularization;
            attempt
        })
}

fn factor_solve_spral_src(
    system: &ReducedKktSystem<'_>,
    workspace: &mut NativeSpralAugmentedKktWorkspace,
    rhs: &[f64],
    regularization: f64,
    context: IpoptLinearSolveContext,
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

    let value_dependent_analysis = matches!(workspace.ordering, SpralNativeOrdering::Matching);
    let mut reused_symbolic =
        workspace.factor_regularization.is_some() && !value_dependent_analysis;
    if value_dependent_analysis || workspace.session.is_none() {
        let analyse_started = Instant::now();
        profiling.sparse_symbolic_analyses += 1;
        workspace.session = Some(
            workspace
                .native
                // IpSpralSolverInterface.cpp exposes SPRAL through the ptr32
                // one-based compressed sparse path. Keep NLIP's source-built
                // SPRAL parity lane on the same analyse/factor entrypoints.
                .analyse_ipopt_compatible_with_options_and_ordering(
                    matrix,
                    &workspace.numeric_options,
                    workspace.ordering,
                )
                .map_err(|error| {
                    native_spral_error_attempt(
                        regularization,
                        InteriorPointLinearSolveFailureKind::FactorizationFailed,
                        format!("native SPRAL analyse failed: {error}"),
                    )
                })?,
        );
        profiling.sparse_symbolic_analysis_time += analyse_started.elapsed();
        workspace.factor_regularization = None;
        reused_symbolic = false;
    }

    let mut factorization_time = Duration::ZERO;
    let session = workspace
        .session
        .as_mut()
        .expect("native SPRAL session must exist after analysis");
    let factor_started = Instant::now();
    let factor_info = if reused_symbolic {
        profiling.sparse_numeric_refactorizations += 1;
        session.refactorize(matrix).map_err(|error| {
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
        session.factorize(matrix).map_err(|error| {
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
        return Err(InteriorPointLinearSolveAttempt {
            solver: InteriorPointLinearSolver::SpralSrc,
            regularization,
            inertia: Some(Box::new(interior_point_linear_inertia(factor_info.inertia))),
            failure_kind: InteriorPointLinearSolveFailureKind::InertiaMismatch,
            detail: Some(inertia_mismatch_detail(
                expected_inertia,
                factor_info.inertia,
            )),
            solution_inf: None,
            solution_inf_limit: None,
            residual_inf: None,
            residual_inf_limit: None,
        });
    }

    let solve_started = Instant::now();
    let session = workspace
        .session
        .as_ref()
        .expect("native SPRAL session must exist after factorization");
    // IPOPT's SpralSolverInterface::MultiSolve calls spral_ssids_solve with
    // nrhs=1 even for a single right-hand side. Keep the live SpralSrc KKT path
    // on the same C entrypoint instead of spral_ssids_solve1.
    let mut solution = session.solve_ipopt_single_rhs(rhs).map_err(|error| {
        native_spral_error_attempt(
            regularization,
            InteriorPointLinearSolveFailureKind::FactorizationFailed,
            error,
        )
    })?;
    let mut solve_time = solve_started.elapsed();
    let refinement = refine_ipopt_full_space_solution(
        system,
        &workspace.pattern,
        &mut solution,
        context.rhs_orientation,
        context.shifts,
        &mut solve_time,
        |residual| {
            session.solve_ipopt_single_rhs(residual).map_err(|error| {
                native_spral_error_attempt(
                    regularization,
                    InteriorPointLinearSolveFailureKind::FactorizationFailed,
                    error,
                )
            })
        },
    )?;
    let mut detail = (refinement.steps > 0).then(|| {
        format!(
            "full_space_iterative_refinement_steps={} residual_ratio={:.3e}->{:.3e} residuals=[{}]->[{}]",
            refinement.steps,
            refinement.initial_residual_ratio,
            refinement.residual_ratio,
            ipopt_full_space_residual_metrics_text(refinement.initial_residual),
            ipopt_full_space_residual_metrics_text(refinement.final_residual),
        )
    });
    let mut accepted_after_failed_refinement = false;
    if refinement.failed {
        // IpPDFullSpaceSolver.cpp retries failed full-space iterative
        // refinement by asking the augmented solver IncreaseQuality() once;
        // if that has already happened, residual_ratio_singular decides
        // whether to accept the current solution or pretend singular.
        let refinement_detail = format!(
            "ipopt_full_space_iterative_refinement_failed residual_ratio={:.3e}",
            refinement.residual_ratio
        );
        if !context.native_spral_quality_was_increased
            && native_spral_quality_can_increase(workspace, system)
        {
            return Err(InteriorPointLinearSolveAttempt {
                solver: InteriorPointLinearSolver::SpralSrc,
                regularization,
                inertia: Some(Box::new(interior_point_linear_inertia(factor_info.inertia))),
                failure_kind: InteriorPointLinearSolveFailureKind::ResidualTooLarge,
                detail: Some(format!(
                    "{refinement_detail}; spral_quality_retry_requested"
                )),
                solution_inf: Some(
                    solution
                        .iter()
                        .fold(0.0_f64, |acc, value| acc.max(value.abs())),
                ),
                solution_inf_limit: None,
                residual_inf: Some(refinement.residual_ratio),
                residual_inf_limit: Some(IPOPT_LINEAR_RESIDUAL_RATIO_MAX),
            });
        }
        if refinement.residual_ratio >= IPOPT_LINEAR_RESIDUAL_RATIO_SINGULAR {
            return Err(InteriorPointLinearSolveAttempt {
                solver: InteriorPointLinearSolver::SpralSrc,
                regularization,
                inertia: Some(Box::new(interior_point_linear_inertia(factor_info.inertia))),
                failure_kind: InteriorPointLinearSolveFailureKind::ResidualTooLarge,
                detail: Some(format!(
                    "{refinement_detail}; ipopt_pretend_singular_after_refinement_failure"
                )),
                solution_inf: Some(
                    solution
                        .iter()
                        .fold(0.0_f64, |acc, value| acc.max(value.abs())),
                ),
                solution_inf_limit: None,
                residual_inf: Some(refinement.residual_ratio),
                residual_inf_limit: Some(IPOPT_LINEAR_RESIDUAL_RATIO_SINGULAR),
            });
        }
        detail = Some(match detail.take() {
            Some(existing) => format!(
                "{existing}; {refinement_detail}; ipopt_accept_current_solution_after_refinement_failure"
            ),
            None => {
                format!(
                    "{refinement_detail}; ipopt_accept_current_solution_after_refinement_failure"
                )
            }
        });
        accepted_after_failed_refinement = true;
    }
    if context.rhs_orientation == IpoptLinearRhsOrientation::PreFinal {
        // IPOPT's PDSearchDirCalc calls PDFullSpaceSolver::Solve(-1., 0., ...)
        // and PDFullSpaceSolver::Solve applies the final scaling only after
        // SolveOnce and full-space iterative refinement have finished.
        for value in &mut solution {
            *value = -*value;
        }
    }
    let final_rhs_storage;
    let final_rhs = if context.rhs_orientation == IpoptLinearRhsOrientation::PreFinal {
        final_rhs_storage = rhs.iter().map(|value| -*value).collect::<Vec<_>>();
        final_rhs_storage.as_slice()
    } else {
        rhs
    };
    let assessment = assess_linear_solution_ccs(
        &workspace.pattern.ccs,
        &workspace.values,
        final_rhs,
        &solution,
    );
    if accepted_after_failed_refinement {
        // IpPDFullSpaceSolver.cpp accepts this branch directly after the
        // residual_ratio_singular check. Keep the generic augmented residual
        // assessment as reporting metadata only.
        let (solution_inf, residual_inf) = match assessment {
            Ok(assessment) => (assessment.solution_inf, assessment.residual_inf),
            Err(attempt) => (
                attempt
                    .solution_inf
                    .unwrap_or_else(|| step_inf_norm(&solution)),
                attempt.residual_inf.unwrap_or(f64::NAN),
            ),
        };
        return Ok((
            solution,
            LinearBackendRunStats {
                solver: InteriorPointLinearSolver::SpralSrc,
                factorization_time,
                solve_time,
                reused_symbolic: Some(reused_symbolic),
                inertia: Some(interior_point_linear_inertia(factor_info.inertia)),
                residual_inf,
                solution_inf,
                detail,
            },
        ));
    }

    assessment
        .map(|assessment| {
            (
                solution,
                LinearBackendRunStats {
                    solver: InteriorPointLinearSolver::SpralSrc,
                    factorization_time,
                    solve_time,
                    reused_symbolic: Some(reused_symbolic),
                    inertia: Some(interior_point_linear_inertia(factor_info.inertia)),
                    residual_inf: assessment.residual_inf,
                    solution_inf: assessment.solution_inf,
                    detail,
                },
            )
        })
        .map_err(|mut attempt| {
            attempt.solver = InteriorPointLinearSolver::SpralSrc;
            attempt.regularization = regularization;
            attempt
        })
}

fn solve_reduced_kkt_with_spral_src(
    system: &ReducedKktSystem<'_>,
    workspace: &mut NativeSpralAugmentedKktWorkspace,
    profiling: &mut InteriorPointProfiling,
    verbose: bool,
) -> std::result::Result<NewtonDirection, Vec<InteriorPointLinearSolveAttempt>> {
    let n = system.hessian.lower_triangle.nrow;
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();
    let rhs = build_ipopt_augmented_kkt_rhs(
        system,
        &workspace.pattern,
        IpoptLinearRhsOrientation::PreFinal,
    );

    let mut attempts: Vec<InteriorPointLinearSolveAttempt> = Vec::new();
    let mut current_regularization = system.regularization.max(0.0);
    let mut current_jacobian_regularization = system.forced_jacobian_regularization.unwrap_or(0.0);
    let mut tried_jacobian_regularization = system.forced_jacobian_regularization.is_some();
    let mut quality_retry_detail: Option<String> = None;
    // Mirrors IpPDFullSpaceSolver::augsys_improved_: scoped to this
    // primal-dual system solve, and not reset by perturbation retries because
    // IPOPT's cache dependencies exclude the perturbation shifts.
    let mut solver_quality_improved = false;
    let max_regularization = system
        .regularization_max
        .max(current_regularization)
        .max(system.first_hessian_perturbation);
    let max_retry_count = system.adaptive_regularization_retries + 2;
    for retry_index in 0..=max_retry_count {
        let primal_shift = current_regularization;
        let slack_shift = current_regularization;
        let dual_shift = current_jacobian_regularization;
        fill_spral_augmented_kkt_values(
            &workspace.pattern,
            &mut workspace.values,
            system,
            primal_shift,
            slack_shift,
            dual_shift,
        );
        match factor_solve_spral_src(
            system,
            workspace,
            &rhs,
            current_regularization,
            IpoptLinearSolveContext {
                shifts: IpoptLinearRefinementShifts {
                    primal: primal_shift,
                    slack: slack_shift,
                    dual: dual_shift,
                },
                rhs_orientation: IpoptLinearRhsOrientation::PreFinal,
                native_spral_quality_was_increased: solver_quality_improved,
            },
            profiling,
            verbose,
        ) {
            Ok((solution, mut backend_stats)) => {
                if !attempts.is_empty() {
                    let attempt_detail = attempts
                        .iter()
                        .map(|attempt| {
                            format!(
                                "reg={:.3e}:{}:{}",
                                attempt.regularization,
                                attempt.failure_kind.label(),
                                attempt.detail.as_deref().unwrap_or("--")
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    let detail = match backend_stats.detail.take() {
                        Some(detail) => format!("{detail}; prior_attempts=[{attempt_detail}]"),
                        None => format!("prior_attempts=[{attempt_detail}]"),
                    };
                    backend_stats.detail = Some(match quality_retry_detail.take() {
                        Some(quality_detail) => format!("{detail}; {quality_detail}"),
                        None => detail,
                    });
                } else if let Some(quality_detail) = quality_retry_detail.take() {
                    backend_stats.detail = Some(match backend_stats.detail.take() {
                        Some(detail) => format!("{detail}; {quality_detail}"),
                        None => quality_detail,
                    });
                }
                let dx = solution[..n].to_vec();
                let ipopt_ds = solution
                    [workspace.pattern.p_offset..workspace.pattern.p_offset + mineq]
                    .to_vec();
                let ds = ipopt_ds;
                let d_lambda = solution
                    [workspace.pattern.lambda_offset..workspace.pattern.lambda_offset + meq]
                    .to_vec();
                let d_ineq = solution
                    [workspace.pattern.z_offset..workspace.pattern.z_offset + mineq]
                    .to_vec();
                let dz = system
                    .r_cent
                    .iter()
                    .zip(system.multipliers.iter())
                    .zip(system.slack.iter())
                    .zip(ds.iter())
                    .map(|(((r_cent_i, z_i), s_i), ds_i)| {
                        ipopt_upper_slack_bound_multiplier_step(*s_i, *z_i, *r_cent_i, *ds_i)
                    })
                    .collect::<Vec<_>>();
                return Ok(NewtonDirection {
                    dx,
                    d_lambda,
                    d_ineq,
                    ds,
                    dz,
                    dz_lower: Vec::new(),
                    dz_upper: Vec::new(),
                    solver_used: InteriorPointLinearSolver::SpralSrc,
                    regularization_used: current_regularization,
                    dual_regularization_used: current_jacobian_regularization,
                    primal_diagonal_shift_used: primal_shift,
                    linear_solution: solution,
                    backend_stats,
                    linear_debug: None,
                });
            }
            Err(attempt) => {
                let singularity_like_failure = is_singularity_like_linear_failure(&attempt);
                let too_few_negative_eigenvalues = is_negative_eigenvalues_too_few(&attempt);
                let full_space_refinement_quality_retry = attempt.failure_kind
                    == InteriorPointLinearSolveFailureKind::ResidualTooLarge
                    && attempt
                        .detail
                        .as_deref()
                        .is_some_and(|detail| detail.contains("spral_quality_retry_requested"));
                attempts.push(attempt);
                if (too_few_negative_eigenvalues || full_space_refinement_quality_retry)
                    && !solver_quality_improved
                    && let Some((old_u, new_u)) =
                        try_increase_native_spral_quality(workspace, system)
                {
                    solver_quality_improved = true;
                    let quality_detail = format!("spral_quality_retry=u:{old_u:.3e}->{new_u:.3e}");
                    if let Some(last_attempt) = attempts.last_mut() {
                        append_attempt_detail(last_attempt, quality_detail.clone());
                    }
                    quality_retry_detail = Some(quality_detail);
                    continue;
                }
                if singularity_like_failure
                    && meq + mineq > 0
                    && !tried_jacobian_regularization
                    && current_jacobian_regularization <= 0.0
                {
                    current_jacobian_regularization = ipopt_jacobian_perturbation(system);
                    tried_jacobian_regularization = true;
                    if current_jacobian_regularization > 0.0 {
                        continue;
                    }
                }
                if singularity_like_failure
                    && current_jacobian_regularization > 0.0
                    && current_regularization <= 0.0
                {
                    current_jacobian_regularization = 0.0;
                }
            }
        }
        if retry_index == max_retry_count || current_regularization >= max_regularization {
            break;
        }
        let Some(next_regularization) = next_ipopt_hessian_perturbation(
            current_regularization,
            system.previous_hessian_perturbation,
            system.first_hessian_perturbation,
            system.regularization_first_growth_factor,
            system.regularization_growth_factor,
            system.regularization_decay_factor,
            max_regularization,
        ) else {
            break;
        };
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
    let rhs = build_ipopt_augmented_kkt_rhs(
        system,
        &workspace.pattern,
        IpoptLinearRhsOrientation::FinalDirection,
    );

    let mut attempts = Vec::new();
    let mut current_regularization = if system.forced_jacobian_regularization.is_some() {
        system.regularization.max(0.0)
    } else {
        system.regularization.max(1e-12)
    };
    let max_regularization = system.regularization_max.max(current_regularization);
    for retry_index in 0..=system.adaptive_regularization_retries {
        let primal_shift = sparse_hessian_diagonal_shift(system.hessian, current_regularization);
        let slack_shift = primal_shift;
        let dual_shift = system
            .forced_jacobian_regularization
            .unwrap_or_else(|| current_regularization.max(1e-8));
        assemble_spral_augmented_kkt_values(
            workspace,
            system,
            primal_shift,
            slack_shift,
            dual_shift,
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
                let ipopt_ds = solution
                    [workspace.pattern.p_offset..workspace.pattern.p_offset + mineq]
                    .to_vec();
                let ds = ipopt_ds;
                let d_lambda = solution
                    [workspace.pattern.lambda_offset..workspace.pattern.lambda_offset + meq]
                    .to_vec();
                let d_ineq = solution
                    [workspace.pattern.z_offset..workspace.pattern.z_offset + mineq]
                    .to_vec();
                let dz = system
                    .r_cent
                    .iter()
                    .zip(system.multipliers.iter())
                    .zip(system.slack.iter())
                    .zip(ds.iter())
                    .map(|(((r_cent_i, z_i), s_i), ds_i)| {
                        ipopt_upper_slack_bound_multiplier_step(*s_i, *z_i, *r_cent_i, *ds_i)
                    })
                    .collect::<Vec<_>>();
                return Ok(NewtonDirection {
                    dx,
                    d_lambda,
                    d_ineq,
                    ds,
                    dz,
                    dz_lower: Vec::new(),
                    dz_upper: Vec::new(),
                    solver_used: InteriorPointLinearSolver::SsidsRs,
                    regularization_used: current_regularization.max(primal_shift),
                    dual_regularization_used: dual_shift,
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
    for (diag, &value) in system.bound_diagonal.iter().enumerate() {
        if value != 0.0 {
            rows.push(diag);
            cols.push(diag);
            values.push(value);
        }
    }
    let mut rhs_top = system
        .r_dual
        .iter()
        .zip(system.bound_rhs.iter())
        .map(|(value, bound_rhs)| -value + bound_rhs)
        .collect::<Vec<_>>();

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
            .enumerate()
            .map(|(index, (((r_cent_i, z_i), r_ineq_i), s_i))| {
                (r_cent_i - z_i * r_ineq_i) / s_i
                    - damped_slack_stationarity_residual(system, index)
            })
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
        let (ds, d_ineq, dz) = if mineq > 0 {
            let jacobian_dx = sparse_mat_vec(system.inequality_jacobian, &dx);
            let ds = jacobian_dx
                .iter()
                .zip(system.r_ineq.iter())
                .map(|(gdx_i, r_ineq_i)| r_ineq_i + gdx_i)
                .collect::<Vec<_>>();
            let d_ineq = ds
                .iter()
                .zip(system.r_cent.iter())
                .zip(system.multipliers.iter())
                .zip(system.slack.iter())
                .enumerate()
                .map(|(index, (((ds_i, r_cent_i), z_i), s_i))| {
                    (-r_cent_i
                        + s_i * damped_slack_stationarity_residual(system, index)
                        + z_i * ds_i)
                        / s_i
                })
                .collect::<Vec<_>>();
            let dz = d_ineq
                .iter()
                .enumerate()
                .map(|(index, dy_i)| *dy_i - damped_slack_stationarity_residual(system, index))
                .collect::<Vec<_>>();
            (ds, d_ineq, dz)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        return Ok(NewtonDirection {
            dx,
            d_lambda: Vec::new(),
            d_ineq,
            ds,
            dz,
            dz_lower: Vec::new(),
            dz_upper: Vec::new(),
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
                let (ds, d_ineq, dz) = if mineq > 0 {
                    let jacobian_dx = sparse_mat_vec(system.inequality_jacobian, &dx);
                    let ds = jacobian_dx
                        .iter()
                        .zip(system.r_ineq.iter())
                        .map(|(gdx_i, r_ineq_i)| r_ineq_i + gdx_i)
                        .collect::<Vec<_>>();
                    let d_ineq = ds
                        .iter()
                        .zip(system.r_cent.iter())
                        .zip(system.multipliers.iter())
                        .zip(system.slack.iter())
                        .enumerate()
                        .map(|(index, (((ds_i, r_cent_i), z_i), s_i))| {
                            (-r_cent_i
                                + s_i * damped_slack_stationarity_residual(system, index)
                                + z_i * ds_i)
                                / s_i
                        })
                        .collect::<Vec<_>>();
                    let dz = d_ineq
                        .iter()
                        .enumerate()
                        .map(|(index, dy_i)| {
                            *dy_i - damped_slack_stationarity_residual(system, index)
                        })
                        .collect::<Vec<_>>();
                    (ds, d_ineq, dz)
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };
                return Ok(NewtonDirection {
                    dx,
                    d_lambda,
                    d_ineq,
                    ds,
                    dz,
                    dz_lower: Vec::new(),
                    dz_upper: Vec::new(),
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

    if preferred_solver == InteriorPointLinearSolver::SpralSrc
        && let Some(workspace) = native_spral_workspace
    {
        return solve_reduced_kkt_with_spral_src(system, workspace, profiling, verbose).map_err(
            |attempts| {
                linear_solve_error(preferred_solver, workspace.pattern.dimension(), attempts)
            },
        );
    }

    if preferred_solver == InteriorPointLinearSolver::SsidsRs
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
        slack_primal: None,
        equality_multipliers: None,
        inequality_multipliers: None,
        slack_multipliers: None,
        lower_bound_multipliers: None,
        upper_bound_multipliers: None,
        kkt_inequality_residual: None,
        kkt_slack_stationarity: None,
        kkt_slack_complementarity: None,
        kkt_slack_sigma: None,
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
        step_direction: None,
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
            'q' => state.mark_linear_solver_quality_if_new(),
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
            slack_primal: None,
            equality_multipliers: None,
            inequality_multipliers: None,
            slack_multipliers: None,
            lower_bound_multipliers: None,
            upper_bound_multipliers: None,
            kkt_inequality_residual: None,
            kkt_slack_stationarity: None,
            kkt_slack_complementarity: None,
            kkt_slack_sigma: None,
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
            step_direction: None,
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
        let linear_quality = snapshot_with_events(vec![
            InteriorPointIterationEvent::LinearSolverQualityIncreased,
        ]);
        let barrier_update =
            snapshot_with_events(vec![InteriorPointIterationEvent::BarrierParameterUpdated]);

        assert_eq!(nlip_event_slot_codes(&filter_soc), " F S          ");
        assert_eq!(nlip_event_slot_codes(&filter_watchdog), " F   W        ");
        assert_eq!(nlip_event_slot_codes(&filter_reset), "      X       ");
        assert_eq!(nlip_event_slot_codes(&watchdog_only), "     W        ");
        assert_eq!(nlip_event_slot_codes(&linear_quality), "       q      ");
        assert_eq!(nlip_event_slot_codes(&barrier_update), "         U    ");
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
    fn bound_multiplier_correction_keeps_ipopt_step_order() {
        let mu = 1.7646745086707415e-7;
        let kappa_sigma = 5.918795724267679;
        let slack = vec![0.2958312539029896];
        let z = vec![278.59313097706615];

        let (corrected_z, _) = correct_bound_multiplier_estimate(&z, &slack, mu, kappa_sigma);
        let direct_endpoint = (kappa_sigma * mu) / slack[0];

        assert_eq!(corrected_z[0].to_bits(), 0x3ecd_9dff_f800_0000);
        assert_eq!(direct_endpoint.to_bits(), 0x3ecd_9dff_fa07_9d96);
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
            d_ineq: Vec::new(),
            ds: ds.to_vec(),
            dz: Vec::new(),
            dz_lower: Vec::new(),
            dz_upper: Vec::new(),
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
    fn monotone_mu_floor_ignores_adaptive_mu_min() {
        let options = InteriorPointOptions {
            overall_tol: 1e-12,
            complementarity_tol: 1e-12,
            mu_min: 1e-2,
            ..InteriorPointOptions::default()
        };

        // IPOPT IpMonotoneMuUpdate::CalcNewMuAndTau does not include mu_min
        // in the monotone lower bound. With mu=1e-4, the next value is
        // min(0.2*mu, mu^1.5) = 1e-6, not the larger adaptive mu_min.
        let next = next_barrier_parameter_once(1e-4, &options);

        assert!((next - 1e-6).abs() <= 10.0 * f64::EPSILON);
    }

    #[test]
    fn monotone_mu_tiny_step_forces_only_one_drop() {
        let options = InteriorPointOptions::default();

        // IPOPT IpMonotoneMuUpdate::UpdateBarrierParameter consumes
        // tiny_step_flag after the first loop pass. A large barrier error
        // therefore stops after the forced one-step update.
        let next = next_barrier_parameter(1e-1, true, true, &options, |_| 1e6);

        assert!((next - 2e-2).abs() <= f64::EPSILON);
    }

    #[test]
    fn monotone_mu_first_call_can_fast_decrease_even_when_disabled() {
        let options = InteriorPointOptions {
            overall_tol: 1e-12,
            complementarity_tol: 1e-12,
            mu_allow_fast_monotone_decrease: false,
            ..InteriorPointOptions::default()
        };

        // IPOPT keeps initialized_=false until the end of the first
        // UpdateBarrierParameter call, so the first call can still take the
        // fast monotone loop even when later calls would stop after one drop.
        let first_call = next_barrier_parameter(1e-1, false, false, &options, |_| 0.0);
        let later_call = next_barrier_parameter(1e-1, false, true, &options, |_| 0.0);

        assert!((first_call - 1e-12 / 11.0).abs() <= f64::EPSILON);
        assert!((later_call - 2e-2).abs() <= f64::EPSILON);
    }

    #[test]
    fn push_scalar_to_bounds_interior_keeps_ipopt_tiny_margin_order() {
        let lower = 0.0;
        let upper = 1.0e-305;
        let bound_push = 1.0;
        let bound_frac = 0.5;
        let tiny_double = 100.0 * f64::MIN_POSITIVE;

        let pushed_from_lower =
            push_scalar_to_bounds_interior(lower, Some(lower), Some(upper), bound_push, bound_frac);
        let pushed_from_upper =
            push_scalar_to_bounds_interior(upper, Some(lower), Some(upper), bound_push, bound_frac);

        // Ipopt::DefaultIterateInitializer::push_variables subtracts
        // tiny_double from the lower-side fraction margin, then adds it back
        // only for the upper-side margin.
        assert_eq!(
            pushed_from_lower.to_bits(),
            (bound_frac * (upper - lower) - tiny_double).to_bits()
        );
        assert_eq!(
            pushed_from_upper.to_bits(),
            (upper - bound_frac * (upper - lower)).to_bits()
        );
    }

    #[test]
    fn push_scalar_to_bounds_interior_snaps_before_pushing() {
        let pushed = push_scalar_to_bounds_interior(-10.0, Some(2.0), Some(8.0), 1.0e-1, 1.0e-1);

        assert_eq!(pushed, 2.2);
    }

    #[test]
    fn push_scalar_to_bounds_interior_without_bound_frac_only_snaps() {
        let pushed = push_scalar_to_bounds_interior(-10.0, Some(2.0), Some(8.0), 1.0e-1, 0.0);

        assert_eq!(pushed, 2.0);
    }

    #[test]
    fn filter_alpha_min_keeps_ipopt_formula_near_feasibility() {
        let options = InteriorPointOptions::default();
        let alpha_min = calculate_filter_alpha_min(1.0e-3, 1.0e-2, -1.0e-2, &options);

        assert!((alpha_min - 5.0e-11).abs() <= 1.0e-25);
        assert!(alpha_min < options.min_step);
    }

    #[test]
    fn fraction_to_boundary_candidate_keeps_ipopt_operation_order() {
        let tau = 0.7782901536485674;
        let value = 2.59793708066258e-11;
        let delta = -2.2047607129366904e-6;

        let ipopt_order = ipopt_dense_frac_to_bound_candidate(tau, value, delta);
        let old_nlip_order = -tau * value / delta;

        assert_eq!(ipopt_order.to_bits(), 0x3ee3_3b8d_73e4_0424);
        assert_eq!(old_nlip_order.to_bits(), 0x3ee3_3b8d_73e4_0425);
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
    let (bounds, fixed_variables) = collect_interior_point_bounds_and_fixed(problem, options)?;
    let augmented_inequality_count = inequality_count;
    let barrier_bound_count = bounds.total_count();
    let barrier_pair_count = augmented_inequality_count + barrier_bound_count;
    let equality_jacobian_structure =
        Arc::new(sparse_structure_from_ccs(problem.equality_jacobian_ccs()));
    let inequality_jacobian_structure =
        Arc::new(sparse_structure_from_ccs(problem.inequality_jacobian_ccs()));
    let equality_column_reduction =
        build_sparse_column_reduction(equality_jacobian_structure.as_ref(), &fixed_variables);
    let inequality_column_reduction =
        build_sparse_column_reduction(inequality_jacobian_structure.as_ref(), &fixed_variables);
    let full_hessian_structure = Arc::new(problem.lagrangian_hessian_ccs().clone());
    let hessian_reduction =
        build_symmetric_submatrix_reduction(full_hessian_structure.as_ref(), &fixed_variables);
    let hessian_structure = Arc::clone(&hessian_reduction.lower_triangle);
    let preferred_solver = preferred_linear_solver(
        options.linear_solver,
        equality_count,
        augmented_inequality_count,
    );
    let mut spral_workspace = None;
    let mut spral_workspace_unavailable = false;
    let mut native_spral_workspace = None;
    let mut native_spral_workspace_unavailable = false;
    let mut previous_hessian_perturbation = None;
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
    fixed_variables.project_fixed_values(&mut x);
    let trial_evaluation_context = TrialEvaluationContext {
        equality_jacobian_structure: &equality_jacobian_structure,
        inequality_jacobian_structure: &inequality_jacobian_structure,
    };
    let mut lambda_eq = vec![0.0; equality_count];
    let mut z = vec![1.0; augmented_inequality_count];
    let mut z_lower = vec![1.0; bounds.lower_indices.len()];
    let mut z_upper = vec![1.0; bounds.upper_indices.len()];
    let mut slack = vec![1.0; augmented_inequality_count];
    let slack_upper_bounds = slack_upper_bound_values(augmented_inequality_count, options);
    let mut event_state = SqpEventLegendState::default();
    let mut last_adapter_timing = problem.adapter_timing_snapshot();
    profiling.adapter_timing = last_adapter_timing;

    let setup_started = Instant::now();
    let mut setup_callback_time = Duration::ZERO;
    let initial_state = trial_state(
        problem,
        &x,
        parameters,
        &trial_evaluation_context,
        &mut profiling,
        &mut setup_callback_time,
    );
    profiling.preprocessing_steps += 1;
    profiling.preprocessing_time += setup_started.elapsed().saturating_sub(setup_callback_time);
    initialise_slacks(
        &initial_state.augmented_inequality_values,
        &slack_upper_bounds,
        &mut slack,
        options,
    );
    let initial_slack_barrier = slack_barrier_values(&slack, &slack_upper_bounds);
    match options.bound_mult_init_method {
        InteriorPointBoundMultiplierInitMethod::Constant => {
            z.fill(options.bound_mult_init_val);
            z_lower.fill(options.bound_mult_init_val);
            z_upper.fill(options.bound_mult_init_val);
        }
        InteriorPointBoundMultiplierInitMethod::MuBased => {
            for (slack_i, z_i) in initial_slack_barrier.iter().zip(z.iter_mut()) {
                *z_i = options.mu_init / slack_i;
            }
            for ((&index, &lower), z_i) in bounds
                .lower_indices
                .iter()
                .zip(bounds.lower_values.iter())
                .zip(z_lower.iter_mut())
            {
                *z_i = options.mu_init / native_lower_bound_slack(&x, index, lower);
            }
            for ((&index, &upper), z_i) in bounds
                .upper_indices
                .iter()
                .zip(bounds.upper_values.iter())
                .zip(z_upper.iter_mut())
            {
                *z_i = options.mu_init / native_upper_bound_slack(&x, index, upper);
            }
        }
    }
    let mut lambda_ineq = z.clone();
    if (equality_count > 0 || augmented_inequality_count > 0) && options.constr_mult_init_max > 0.0
    {
        let mut initial_ls_gradient = initial_state.gradient.clone();
        add_native_bound_multiplier_terms(&mut initial_ls_gradient, &bounds, &z_lower, &z_upper);
        let initial_linear_state = EvalState {
            objective_value: initial_state.objective_value,
            gradient: fixed_variables.reduce_vector(&initial_ls_gradient),
            equality_values: initial_state.equality_values.clone(),
            augmented_inequality_values: initial_state.augmented_inequality_values.clone(),
            equality_jacobian: reduce_sparse_matrix_columns(
                &initial_state.equality_jacobian,
                &equality_column_reduction,
            ),
            inequality_jacobian: reduce_sparse_matrix_columns(
                &initial_state.inequality_jacobian,
                &inequality_column_reduction,
            ),
        };
        let (initial_lambda_eq, initial_lambda_ineq) = least_squares_constraint_multipliers(
            &initial_linear_state,
            &z,
            hessian_structure.as_ref(),
            options.regularization,
            options.linear_solver,
        );
        let initial_lambda_inf = inf_norm(&initial_lambda_eq).max(inf_norm(&initial_lambda_ineq));
        if initial_lambda_inf <= options.constr_mult_init_max {
            lambda_eq = initial_lambda_eq;
            lambda_ineq = initial_lambda_ineq;
        } else {
            lambda_eq.fill(0.0);
            lambda_ineq.fill(0.0);
        }
    }

    if options.verbose {
        log_interior_point_problem_header(problem, parameters, options);
    }

    let mut nonlinear_inequality_multipliers = lambda_ineq.clone();
    let mut last_linear_solver = preferred_solver;
    let mut filter_entries = Vec::new();
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
    let mut barrier_parameter_value = if barrier_pair_count > 0 {
        let initial_complementarity =
            combined_barrier_parameter(&initial_slack_barrier, &z, &x, &bounds, &z_lower, &z_upper);
        options
            .mu_target
            .max(options.mu_min)
            .max(initial_complementarity.min(options.mu_init))
    } else {
        0.0
    };
    let mut watchdog_state = InteriorPointWatchdogState::default();
    let mut monotone_mu_update_initialized = false;
    let mut pending_iteration_events = Vec::new();

    if options.max_iters == 0 {
        let equality_inf = inf_norm(&initial_state.equality_values);
        let inequality_inf = inequality_upper_bound_inf_norm(
            &initial_state.augmented_inequality_values,
            &slack_upper_bounds,
        );
        let initial_inequality_residual =
            slack_form_inequality_residuals(&initial_state.augmented_inequality_values, &slack);
        let initial_dual_residual = lagrangian_gradient_sparse(
            &initial_state.gradient,
            &initial_state.equality_jacobian,
            &lambda_eq,
            &initial_state.inequality_jacobian,
            &lambda_ineq,
        );
        let mut initial_dual_residual = initial_dual_residual;
        add_native_bound_multiplier_terms(&mut initial_dual_residual, &bounds, &z_lower, &z_upper);
        let initial_slack_stationarity_inf = slack_stationarity_inf_norm(&lambda_ineq, &z);
        let dual_inf = fixed_variables
            .free_inf_norm(&initial_dual_residual)
            .max(initial_slack_stationarity_inf);
        let complementarity_inf = if barrier_pair_count > 0 {
            combined_complementarity_inf_norm(
                &initial_slack_barrier,
                &z,
                &x,
                &bounds,
                &z_lower,
                &z_upper,
            )
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
            &initial_slack_barrier,
            &x,
            &bounds,
            barrier_parameter_value,
            options.kappa_d,
        );
        let current_filter_entry = super::filter::entry(current_barrier_objective, current_theta);
        let all_dual_multipliers = combined_multiplier_vector([
            lambda_eq.as_slice(),
            lambda_ineq.as_slice(),
            z.as_slice(),
            z_lower.as_slice(),
            z_upper.as_slice(),
        ]);
        let complementarity_multipliers =
            combined_multiplier_vector([z.as_slice(), z_lower.as_slice(), z_upper.as_slice()]);
        let overall_inf = scaled_overall_inf_norm(
            current_theta,
            dual_inf,
            complementarity_inf,
            &all_dual_multipliers,
            &complementarity_multipliers,
            options.overall_scale_max,
        );
        let snapshot = InteriorPointIterationSnapshot {
            iteration: 0,
            phase: InteriorPointIterationPhase::Initial,
            x: x.clone(),
            slack_primal: Some(slack.clone()),
            equality_multipliers: Some(lambda_eq.clone()),
            inequality_multipliers: Some(lambda_ineq.clone()),
            slack_multipliers: Some(z.clone()),
            lower_bound_multipliers: Some(z_lower.clone()),
            upper_bound_multipliers: Some(z_upper.clone()),
            kkt_inequality_residual: Some(initial_inequality_residual),
            kkt_slack_stationarity: Some(damped_slack_stationarity_residuals(
                &lambda_ineq,
                &z,
                barrier_parameter_value,
                options.kappa_d,
            )),
            kkt_slack_complementarity: Some(slack_complementarity_residuals(
                &initial_slack_barrier,
                &z,
                barrier_parameter_value,
            )),
            kkt_slack_sigma: Some(slack_sigma_values(&initial_slack_barrier, &z)),
            objective: initial_state.objective_value,
            barrier_objective: Some(current_barrier_objective),
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (barrier_pair_count > 0).then_some(complementarity_inf),
            overall_inf,
            barrier_parameter: (barrier_pair_count > 0).then_some(barrier_parameter_value),
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
            step_direction: None,
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
                has_inequalities: barrier_pair_count > 0,
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
                    barrier_parameter: if barrier_pair_count > 0 {
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
            &trial_evaluation_context,
            &mut profiling,
            &mut iteration_callback_time,
        );
        let equality_inf = inf_norm(&state.equality_values);
        let inequality_inf = inequality_upper_bound_inf_norm(
            &state.augmented_inequality_values,
            &slack_upper_bounds,
        );
        let inequality_residual =
            slack_form_inequality_residuals(&state.augmented_inequality_values, &slack);
        let slack_barrier = slack_barrier_values(&slack, &slack_upper_bounds);
        let internal_inequality_inf = inf_norm(&inequality_residual);
        let primal_inf = equality_inf.max(internal_inequality_inf);
        let full_dual_residual = lagrangian_gradient_sparse(
            &state.gradient,
            &state.equality_jacobian,
            &lambda_eq,
            &state.inequality_jacobian,
            &lambda_ineq,
        );
        let mut full_dual_residual = full_dual_residual;
        add_native_bound_multiplier_terms(&mut full_dual_residual, &bounds, &z_lower, &z_upper);
        let dual_residual = fixed_variables.reduce_vector(&full_dual_residual);
        let slack_stationarity_residual = slack_stationarity_residuals(&lambda_ineq, &z);
        // IPOPT reports and tests convergence with curr_dual_infeasibility(), which uses the
        // undamped Lagrangian gradients. The damped vectors are kept as separate KKT diagnostics.
        let dual_x_inf = fixed_variables.free_inf_norm(&full_dual_residual);
        let slack_stationarity_inf = slack_stationarity_inf_norm(&lambda_ineq, &z);
        let dual_inf = dual_x_inf.max(slack_stationarity_inf);
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
            let full_index = fixed_variables
                .free_indices
                .get(max_index)
                .copied()
                .unwrap_or(max_index);
            let equality_contrib = state.equality_jacobian.structure.ccs.col_ptrs[full_index]
                ..state.equality_jacobian.structure.ccs.col_ptrs[full_index + 1];
            let equality_contrib = equality_contrib
                .map(|entry| {
                    state.equality_jacobian.values[entry]
                        * lambda_eq[state.equality_jacobian.structure.ccs.row_indices[entry]]
                })
                .sum::<f64>();
            let top_eq_terms = state.equality_jacobian.structure.ccs.col_ptrs[full_index]
                ..state.equality_jacobian.structure.ccs.col_ptrs[full_index + 1];
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
            let inequality_contrib = state.inequality_jacobian.structure.ccs.col_ptrs[full_index]
                ..state.inequality_jacobian.structure.ccs.col_ptrs[full_index + 1];
            let inequality_contrib = inequality_contrib
                .map(|entry| {
                    state.inequality_jacobian.values[entry]
                        * lambda_ineq[state.inequality_jacobian.structure.ccs.row_indices[entry]]
                })
                .sum::<f64>();
            let top_ineq_terms = state.inequality_jacobian.structure.ccs.col_ptrs[full_index]
                ..state.inequality_jacobian.structure.ccs.col_ptrs[full_index + 1];
            let mut top_ineq_terms = top_ineq_terms
                .map(|entry| {
                    let row = state.inequality_jacobian.structure.ccs.row_indices[entry];
                    let contribution = state.inequality_jacobian.values[entry] * lambda_ineq[row];
                    (row, contribution)
                })
                .collect::<Vec<_>>();
            top_ineq_terms.sort_by(|lhs, rhs| rhs.1.abs().total_cmp(&lhs.1.abs()));
            let top_ineq_summary = top_ineq_terms
                .iter()
                .take(3)
                .map(|(row, contribution)| {
                    let kind = "ineq";
                    format!("{kind}:{row}:{contribution:.3e}")
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "NLIP_DEBUG_MAX_DUAL iter={} idx={} residual={:.6e} x={:.6e} grad={:.6e} eq={:.6e} ineq={:.6e} primal={:.6e} top_eq=[{}] top_ineq=[{}]",
                iteration,
                full_index,
                max_value,
                x.get(full_index).copied().unwrap_or(0.0),
                state.gradient.get(full_index).copied().unwrap_or(0.0),
                equality_contrib,
                inequality_contrib,
                primal_inf,
                top_eq_summary,
                top_ineq_summary,
            );
        }
        let complementarity_inf = if barrier_pair_count > 0 {
            combined_complementarity_inf_norm(&slack_barrier, &z, &x, &bounds, &z_lower, &z_upper)
        } else {
            0.0
        };
        let all_dual_multipliers = combined_multiplier_vector([
            lambda_eq.as_slice(),
            lambda_ineq.as_slice(),
            z.as_slice(),
            z_lower.as_slice(),
            z_upper.as_slice(),
        ]);
        let complementarity_multipliers =
            combined_multiplier_vector([z.as_slice(), z_lower.as_slice(), z_upper.as_slice()]);
        let overall_inf = scaled_overall_inf_norm(
            primal_inf,
            dual_inf,
            complementarity_inf,
            &all_dual_multipliers,
            &complementarity_multipliers,
            options.overall_scale_max,
        );
        let mut current_barrier_objective = barrier_objective_value(
            state.objective_value,
            &slack_barrier,
            &x,
            &bounds,
            barrier_parameter_value,
            options.kappa_d,
        );
        let current_theta = filter_theta_l1_norm(
            &state.equality_values,
            &state.augmented_inequality_values,
            &slack,
        );
        let mut current_filter_entry =
            super::filter::entry(current_barrier_objective, current_theta);
        let mut current_snapshot = InteriorPointIterationSnapshot {
            iteration,
            phase: if iteration == 0 {
                InteriorPointIterationPhase::Initial
            } else {
                InteriorPointIterationPhase::AcceptedStep
            },
            x: x.clone(),
            slack_primal: Some(slack.clone()),
            equality_multipliers: Some(lambda_eq.clone()),
            inequality_multipliers: Some(lambda_ineq.clone()),
            slack_multipliers: Some(z.clone()),
            lower_bound_multipliers: Some(z_lower.clone()),
            upper_bound_multipliers: Some(z_upper.clone()),
            kkt_inequality_residual: Some(inequality_residual.clone()),
            kkt_slack_stationarity: Some(damped_slack_stationarity_residuals(
                &lambda_ineq,
                &z,
                barrier_parameter_value,
                options.kappa_d,
            )),
            kkt_slack_complementarity: Some(slack_complementarity_residuals(
                &slack_barrier,
                &z,
                barrier_parameter_value,
            )),
            kkt_slack_sigma: Some(slack_sigma_values(&slack_barrier, &z)),
            objective: state.objective_value,
            barrier_objective: Some(current_barrier_objective),
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (barrier_pair_count > 0).then_some(complementarity_inf),
            overall_inf,
            barrier_parameter: (barrier_pair_count > 0).then_some(barrier_parameter_value),
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
            step_direction: None,
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
            compl_inf: combined_complementarity_target_inf_norm(
                &slack_barrier,
                &z,
                &x,
                &bounds,
                &z_lower,
                &z_upper,
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
                slack_primal: Some(slack.clone()),
                equality_multipliers: Some(lambda_eq.clone()),
                inequality_multipliers: Some(lambda_ineq.clone()),
                slack_multipliers: Some(z.clone()),
                lower_bound_multipliers: Some(z_lower.clone()),
                upper_bound_multipliers: Some(z_upper.clone()),
                kkt_inequality_residual: Some(inequality_residual.clone()),
                kkt_slack_stationarity: Some(damped_slack_stationarity_residuals(
                    &lambda_ineq,
                    &z,
                    barrier_parameter_value,
                    options.kappa_d,
                )),
                kkt_slack_complementarity: Some(slack_complementarity_residuals(
                    &slack_barrier,
                    &z,
                    barrier_parameter_value,
                )),
                kkt_slack_sigma: Some(slack_sigma_values(&slack_barrier, &z)),
                objective: state.objective_value,
                barrier_objective: Some(current_barrier_objective),
                eq_inf: (equality_count > 0).then_some(equality_inf),
                ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
                dual_inf,
                comp_inf: (barrier_pair_count > 0).then_some(complementarity_inf),
                overall_inf,
                barrier_parameter: (barrier_pair_count > 0).then_some(barrier_parameter_value),
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
                step_direction: None,
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
            let summary = InteriorPointSummary {
                x,
                equality_multipliers: lambda_eq,
                inequality_multipliers: lambda_ineq,
                lower_bound_multipliers: z_lower,
                upper_bound_multipliers: z_upper,
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
                            has_inequalities: barrier_pair_count > 0,
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

        if barrier_pair_count > 0 {
            let previous_barrier_parameter = barrier_parameter_value;
            let next_barrier_parameter_value = next_barrier_parameter(
                barrier_parameter_value,
                watchdog_state.tiny_step_last_iteration,
                monotone_mu_update_initialized,
                options,
                |candidate_barrier_parameter| {
                    let current_target_complementarity_inf =
                        combined_complementarity_target_inf_norm(
                            &slack_barrier,
                            &z,
                            &x,
                            &bounds,
                            &z_lower,
                            &z_upper,
                            candidate_barrier_parameter,
                        );
                    scaled_overall_inf_norm(
                        primal_inf,
                        dual_inf,
                        current_target_complementarity_inf,
                        &all_dual_multipliers,
                        &complementarity_multipliers,
                        options.overall_scale_max,
                    )
                },
            );
            monotone_mu_update_initialized = true;
            if next_barrier_parameter_value
                < previous_barrier_parameter - 1e-18 * previous_barrier_parameter.abs().max(1.0)
            {
                // IPOPT calls MonotoneMuUpdate::UpdateBarrierParameter before
                // ComputeSearchDirection.  When mu changes, it calls
                // BacktrackingLineSearch::Reset, clearing the filter acceptor;
                // the subsequent FindAcceptableTrialPoint mu check clears the
                // watchdog counters before trial search starts.
                barrier_parameter_value = next_barrier_parameter_value;
                filter_entries.clear();
                successive_filter_rejections = 0;
                watchdog_state = InteriorPointWatchdogState::default();
                push_unique_nlip_event(
                    &mut iteration_events,
                    InteriorPointIterationEvent::BarrierParameterUpdated,
                );
                current_barrier_objective = barrier_objective_value(
                    state.objective_value,
                    &slack_barrier,
                    &x,
                    &bounds,
                    barrier_parameter_value,
                    options.kappa_d,
                );
                current_filter_entry =
                    super::filter::entry(current_barrier_objective, current_theta);
                current_snapshot.barrier_objective = Some(current_barrier_objective);
                current_snapshot.barrier_parameter = Some(barrier_parameter_value);
                current_snapshot.kkt_slack_stationarity =
                    Some(damped_slack_stationarity_residuals(
                        &lambda_ineq,
                        &z,
                        barrier_parameter_value,
                        options.kappa_d,
                    ));
                current_snapshot.kkt_slack_complementarity = Some(slack_complementarity_residuals(
                    &slack_barrier,
                    &z,
                    barrier_parameter_value,
                ));
                current_snapshot.filter = Some(FilterInfo {
                    current: current_filter_entry.clone(),
                    entries: filter_entries.clone(),
                    accepted_mode: None,
                });
                current_snapshot.events = iteration_events.clone();
            }
        }

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
        let full_hessian = SparseSymmetricMatrix {
            lower_triangle: Arc::clone(&full_hessian_structure),
            values: hessian_values,
        };
        let hessian = reduce_symmetric_matrix(&full_hessian, &hessian_reduction);
        let linear_equality_jacobian =
            reduce_sparse_matrix_columns(&state.equality_jacobian, &equality_column_reduction);
        let linear_inequality_jacobian =
            reduce_sparse_matrix_columns(&state.inequality_jacobian, &inequality_column_reduction);
        let hessian_elapsed = hessian_started.elapsed();
        profiling.kkt_assemblies += 1;
        profiling.kkt_assembly_time += hessian_elapsed;
        iteration_kkt_assembly_time += hessian_elapsed;

        let kkt_regularization = kkt_regularization(
            barrier_pair_count > 0,
            primal_inf,
            complementarity_inf,
            dual_inf,
            options,
        );
        let sigma = if barrier_pair_count > 0 { 1.0 } else { 0.0 };
        let r_cent = if augmented_inequality_count > 0 {
            slack_barrier
                .iter()
                .zip(z.iter())
                .map(|(s, z_i)| s * z_i - sigma * barrier_parameter_value)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let (bound_diagonal, bound_rhs) = native_bound_kkt_terms(
            &x,
            &bounds,
            &fixed_variables,
            &z_lower,
            &z_upper,
            barrier_parameter_value,
            sigma,
            options.kappa_d,
        );
        let linear_started = Instant::now();
        let reduced_kkt_system = ReducedKktSystem {
            hessian: &hessian,
            equality_jacobian: &linear_equality_jacobian,
            inequality_jacobian: &linear_inequality_jacobian,
            bound_diagonal: &bound_diagonal,
            bound_rhs: &bound_rhs,
            bound_data: Some(ReducedBoundKktData {
                x: &x,
                bounds: &bounds,
                fixed_variables: &fixed_variables,
                z_lower: &z_lower,
                z_upper: &z_upper,
            }),
            slack: &slack_barrier,
            multipliers: &z,
            r_dual: &dual_residual,
            r_eq: &state.equality_values,
            r_ineq: &inequality_residual,
            r_slack_stationarity: &slack_stationarity_residual,
            r_cent: &r_cent,
            barrier_parameter: barrier_parameter_value,
            kappa_d: options.kappa_d,
            solver: options.linear_solver,
            regularization: kkt_regularization,
            first_hessian_perturbation: options.first_hessian_perturbation,
            previous_hessian_perturbation,
            regularization_first_growth_factor: options.regularization_first_growth_factor,
            adaptive_regularization_retries: options.adaptive_regularization_retries,
            regularization_growth_factor: options.regularization_growth_factor,
            regularization_decay_factor: options.regularization_decay_factor,
            regularization_max: options.regularization_max,
            jacobian_regularization_value: options.jacobian_regularization_value,
            jacobian_regularization_exponent: options.jacobian_regularization_exponent,
            forced_jacobian_regularization: None,
            spral_pivot_method: options.spral_pivot_method,
            spral_action_on_zero_pivot: options.spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: options.spral_small_pivot_tolerance,
            spral_threshold_pivot_u: options.spral_threshold_pivot_u,
            spral_pivot_tolerance_max: options.spral_pivot_tolerance_max,
        };
        current_snapshot.linear_solver = preferred_solver;
        if !spral_workspace_unavailable
            && spral_workspace.is_none()
            && preferred_solver == InteriorPointLinearSolver::SsidsRs
        {
            match prepare_spral_workspace(
                hessian_structure.as_ref(),
                equality_column_reduction.structure.as_ref(),
                inequality_column_reduction.structure.as_ref(),
                &mut profiling,
                options.verbose,
            ) {
                Ok(workspace) => {
                    spral_workspace = Some(workspace);
                }
                Err(error) => {
                    if options.linear_solver == InteriorPointLinearSolver::SsidsRs {
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
            && preferred_solver == InteriorPointLinearSolver::SpralSrc
        {
            match prepare_native_spral_workspace(
                hessian_structure.as_ref(),
                equality_column_reduction.structure.as_ref(),
                inequality_column_reduction.structure.as_ref(),
                &spral_numeric_factor_options(options),
                &mut profiling,
                options.verbose,
            ) {
                Ok(workspace) => {
                    native_spral_workspace = Some(workspace);
                }
                Err(error) => {
                    if options.linear_solver == InteriorPointLinearSolver::SpralSrc {
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
        let mut direction = match solve_result {
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
                        direction.primal_diagonal_shift_used,
                        direction.dual_regularization_used,
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
                    let (primal_shift, dual_shift) = spral_augmented_kkt_regularization_shifts(
                        &reduced_kkt_system,
                        *solver,
                        regularization,
                    );
                    if let Ok(snapshot) = build_interior_point_kkt_snapshot(
                        iteration,
                        current_snapshot.phase,
                        *solver,
                        &reduced_kkt_system,
                        primal_shift,
                        dual_shift,
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
        direction.dx = fixed_variables.expand_direction(&direction.dx);
        let (dz_lower_direction, dz_upper_direction) = native_bound_multiplier_steps(
            &x,
            &direction.dx,
            &bounds,
            &z_lower,
            &z_upper,
            barrier_parameter_value,
            sigma,
        );
        direction.dz_lower = dz_lower_direction;
        direction.dz_upper = dz_upper_direction;
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
        if linear_solver_quality_was_increased(&direction.backend_stats) {
            push_unique_nlip_event(
                &mut iteration_events,
                InteriorPointIterationEvent::LinearSolverQualityIncreased,
            );
        }

        let fraction_to_boundary_tau =
            current_fraction_to_boundary_tau(barrier_parameter_value, options);
        let slack_barrier_direction = slack_barrier_direction_values(&direction.ds);
        let (slack_alpha_pr, slack_alpha_pr_limiter) = if augmented_inequality_count > 0 {
            fraction_to_boundary_with_limiter(
                &slack_barrier,
                &slack_barrier_direction,
                fraction_to_boundary_tau,
                InteriorPointBoundaryLimiterKind::Slack,
            )
        } else {
            (1.0, None)
        };
        let (bound_alpha_pr, bound_alpha_pr_limiter) = fraction_to_variable_bounds_with_limiter(
            &x,
            &direction.dx,
            &bounds,
            fraction_to_boundary_tau,
        );
        let (alpha_pr, alpha_pr_limiter) = if bound_alpha_pr < slack_alpha_pr {
            (bound_alpha_pr, bound_alpha_pr_limiter)
        } else {
            (slack_alpha_pr, slack_alpha_pr_limiter)
        };
        let combined_z =
            combined_multiplier_vector([z.as_slice(), z_lower.as_slice(), z_upper.as_slice()]);
        let combined_dz = combined_multiplier_vector([
            direction.dz.as_slice(),
            direction.dz_lower.as_slice(),
            direction.dz_upper.as_slice(),
        ]);
        let (alpha_du, alpha_du_limiter) = if barrier_pair_count > 0 {
            fraction_to_boundary_with_limiter(
                &combined_z,
                &combined_dz,
                fraction_to_boundary_tau,
                InteriorPointBoundaryLimiterKind::Multiplier,
            )
        } else {
            (1.0, None)
        };
        let alpha_pr_limiters = alpha_pr_limiter.iter().cloned().collect::<Vec<_>>();
        let alpha_du_limiters = if barrier_pair_count > 0 {
            fraction_to_boundary_limiters(
                &combined_z,
                &combined_dz,
                fraction_to_boundary_tau,
                8,
                InteriorPointBoundaryLimiterKind::Multiplier,
            )
        } else {
            Vec::new()
        };
        let current_direction_diagnostics = Some(interior_point_direction_diagnostics(
            &direction,
            alpha_pr_limiter.clone(),
            alpha_du_limiter.clone(),
            alpha_pr_limiters,
            alpha_du_limiters,
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
            &slack_barrier,
            &x,
            &bounds,
            &direction.dx,
            &slack_barrier_direction,
            barrier_parameter_value,
            options.kappa_d,
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
        let tiny_step_unchecked_accept = !watchdog_active
            && is_tiny_ip_step(&x, &slack_barrier, &direction, primal_inf, options);
        let tiny_step_barrier_update =
            tiny_step_unchecked_accept && watchdog_state.tiny_step_last_iteration;
        // IPOPT `BacktrackingLineSearch::DoBacktrackingLineSearch` uses
        // `alpha_primal > alpha_min || n_steps == 0`, so the initial trial is
        // evaluated even when the maximum feasible step is already tiny.
        while alpha > alpha_min || line_search_iterations == 0 {
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
            let trial_lambda_ineq = lambda_ineq
                .iter()
                .zip(direction.d_ineq.iter())
                .map(|(value, delta)| value + trial_alpha_y * delta)
                .collect::<Vec<_>>();
            let trial_slack = slack
                .iter()
                .zip(direction.ds.iter())
                .map(|(value, delta)| value + trial_alpha_pr * delta)
                .collect::<Vec<_>>();
            let trial_slack_barrier = slack_barrier_values(&trial_slack, &slack_upper_bounds);
            let trial_z = z
                .iter()
                .zip(direction.dz.iter())
                .map(|(value, delta)| value + trial_alpha_du * delta)
                .collect::<Vec<_>>();
            let trial_z_lower = z_lower
                .iter()
                .zip(direction.dz_lower.iter())
                .map(|(value, delta)| value + trial_alpha_du * delta)
                .collect::<Vec<_>>();
            let trial_z_upper = z_upper
                .iter()
                .zip(direction.dz_upper.iter())
                .map(|(value, delta)| value + trial_alpha_du * delta)
                .collect::<Vec<_>>();
            let trial_bounds_positive = bounds
                .lower_indices
                .iter()
                .zip(bounds.lower_values.iter())
                .all(|(&index, &lower)| trial_x[index] > lower)
                && bounds
                    .upper_indices
                    .iter()
                    .zip(bounds.upper_values.iter())
                    .all(|(&index, &upper)| trial_x[index] < upper);
            let trial_bound_multipliers_positive = trial_z_lower.iter().all(|value| *value > 0.0)
                && trial_z_upper.iter().all(|value| *value > 0.0);
            if trial_slack_barrier.iter().any(|value| *value <= 0.0)
                || trial_z.iter().any(|value| *value <= 0.0)
                || !trial_bounds_positive
                || !trial_bound_multipliers_positive
            {
                rejected_trials.push(InteriorPointLineSearchTrial {
                    alpha: trial_alpha_pr,
                    alpha_du: Some(trial_alpha_du),
                    slack_positive: trial_slack_barrier.iter().all(|value| *value > 0.0)
                        && trial_bounds_positive,
                    multipliers_positive: trial_z.iter().all(|value| *value > 0.0)
                        && trial_bound_multipliers_positive,
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
                &trial_evaluation_context,
                &mut profiling,
                &mut trial_callback_time,
            );
            let trial_eq_inf = inf_norm(&trial_state.equality_values);
            let trial_ineq_inf = inequality_upper_bound_inf_norm(
                &trial_state.augmented_inequality_values,
                &slack_upper_bounds,
            );
            let trial_internal_ineq_inf = slack_form_inequality_inf_norm(
                &trial_state.augmented_inequality_values,
                &trial_slack,
            );
            let trial_raw_dual_residual = lagrangian_gradient_sparse(
                &trial_state.gradient,
                &trial_state.equality_jacobian,
                &trial_lambda,
                &trial_state.inequality_jacobian,
                &trial_lambda_ineq,
            );
            let mut trial_dual_residual = trial_raw_dual_residual.clone();
            add_native_bound_multiplier_terms(
                &mut trial_dual_residual,
                &bounds,
                &trial_z_lower,
                &trial_z_upper,
            );
            let trial_dual_x_inf = fixed_variables.free_inf_norm(&trial_dual_residual);
            let trial_slack_stationarity_inf =
                slack_stationarity_inf_norm(&trial_lambda_ineq, &trial_z);
            let trial_dual_inf = trial_dual_x_inf.max(trial_slack_stationarity_inf);
            let trial_comp_inf = if barrier_pair_count > 0 {
                combined_complementarity_inf_norm(
                    &trial_slack_barrier,
                    &trial_z,
                    &trial_x,
                    &bounds,
                    &trial_z_lower,
                    &trial_z_upper,
                )
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
            let trial_all_dual_multipliers = combined_multiplier_vector([
                trial_lambda.as_slice(),
                trial_lambda_ineq.as_slice(),
                trial_z.as_slice(),
                trial_z_lower.as_slice(),
                trial_z_upper.as_slice(),
            ]);
            let trial_complementarity_multipliers = combined_multiplier_vector([
                trial_z.as_slice(),
                trial_z_lower.as_slice(),
                trial_z_upper.as_slice(),
            ]);
            let trial_overall_inf = scaled_overall_inf_norm(
                trial_primal_inf,
                trial_dual_inf,
                trial_comp_inf,
                &trial_all_dual_multipliers,
                &trial_complementarity_multipliers,
                options.overall_scale_max,
            );
            let corrected_bound_multipliers = apply_bound_multiplier_safeguard(
                &trial_lambda_ineq,
                &trial_raw_dual_residual,
                &fixed_variables,
                &trial_x,
                &bounds,
                &trial_slack_barrier,
                &trial_z,
                &trial_z_lower,
                &trial_z_upper,
                barrier_parameter_value,
                options,
            );
            let trial_barrier_objective = barrier_objective_value(
                trial_state.objective_value,
                &trial_slack_barrier,
                &trial_x,
                &bounds,
                barrier_parameter_value,
                options.kappa_d,
            );
            let trial_filter_entry =
                super::filter::entry(trial_barrier_objective, trial_filter_theta);
            if tiny_step_unchecked_accept {
                let (
                    accepted_z,
                    accepted_z_lower,
                    accepted_z_upper,
                    accepted_dual_inf,
                    accepted_comp_inf,
                    accepted_mu,
                    accepted_overall_inf,
                    bound_multiplier_corrected,
                ) = if let Some(corrected) = corrected_bound_multipliers {
                    let corrected_comp_inf = combined_complementarity_inf_norm(
                        &trial_slack_barrier,
                        &corrected.z,
                        &trial_x,
                        &bounds,
                        &corrected.z_lower,
                        &corrected.z_upper,
                    );
                    let corrected_complementarity_multipliers = combined_multiplier_vector([
                        corrected.z.as_slice(),
                        corrected.z_lower.as_slice(),
                        corrected.z_upper.as_slice(),
                    ]);
                    let corrected_all_dual_multipliers = combined_multiplier_vector([
                        trial_lambda.as_slice(),
                        trial_lambda_ineq.as_slice(),
                        corrected.z.as_slice(),
                        corrected.z_lower.as_slice(),
                        corrected.z_upper.as_slice(),
                    ]);
                    let corrected_overall_inf = scaled_overall_inf_norm(
                        trial_primal_inf,
                        corrected.dual_inf,
                        corrected_comp_inf,
                        &corrected_all_dual_multipliers,
                        &corrected_complementarity_multipliers,
                        options.overall_scale_max,
                    );
                    (
                        corrected.z,
                        corrected.z_lower,
                        corrected.z_upper,
                        corrected.dual_inf,
                        corrected_comp_inf,
                        barrier_parameter_value,
                        corrected_overall_inf,
                        true,
                    )
                } else {
                    (
                        trial_z,
                        trial_z_lower,
                        trial_z_upper,
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
                    inequality_multipliers: trial_lambda_ineq.clone(),
                    kkt_inequality_residual: slack_form_inequality_residuals(
                        &trial_state.augmented_inequality_values,
                        &trial_slack,
                    ),
                    kkt_slack_stationarity: damped_slack_stationarity_residuals(
                        &trial_lambda_ineq,
                        &accepted_z,
                        barrier_parameter_value,
                        options.kappa_d,
                    ),
                    kkt_slack_complementarity: slack_complementarity_residuals(
                        &trial_slack_barrier,
                        &accepted_z,
                        barrier_parameter_value,
                    ),
                    kkt_slack_sigma: slack_sigma_values(&trial_slack_barrier, &accepted_z),
                    slack: trial_slack,
                    z: accepted_z,
                    z_lower: accepted_z_lower,
                    z_upper: accepted_z_upper,
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
                    filter_augment_entry: None,
                    filter_acceptance_mode: None,
                    step_kind: InteriorPointStepKind::Tiny,
                    step_tag: if tiny_step_barrier_update { 'T' } else { 't' },
                    step_direction: Some(step_direction_snapshot(&direction)),
                    phase: InteriorPointIterationPhase::AcceptedStep,
                    accepted_alpha_pr: trial_alpha_pr,
                    accepted_alpha_du: Some(trial_alpha_du),
                    line_search_initial_alpha_pr: alpha_pr,
                    line_search_initial_alpha_du: Some(alpha_du),
                    line_search_last_alpha_pr: trial_alpha_pr,
                    line_search_last_alpha_du: Some(trial_alpha_du),
                    line_search_backtrack_count: 0,
                    second_order_correction_used: false,
                    watchdog_accepted: false,
                    tiny_step: true,
                    bound_multiplier_corrected,
                });
                break;
            }
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
                    accepted_z_lower,
                    accepted_z_upper,
                    accepted_dual_inf,
                    accepted_comp_inf,
                    accepted_mu,
                    accepted_overall_inf,
                    bound_multiplier_corrected,
                ) = if let Some(corrected) = corrected_bound_multipliers {
                    let corrected_comp_inf = combined_complementarity_inf_norm(
                        &trial_slack_barrier,
                        &corrected.z,
                        &trial_x,
                        &bounds,
                        &corrected.z_lower,
                        &corrected.z_upper,
                    );
                    let corrected_complementarity_multipliers = combined_multiplier_vector([
                        corrected.z.as_slice(),
                        corrected.z_lower.as_slice(),
                        corrected.z_upper.as_slice(),
                    ]);
                    let corrected_all_dual_multipliers = combined_multiplier_vector([
                        trial_lambda.as_slice(),
                        trial_lambda_ineq.as_slice(),
                        corrected.z.as_slice(),
                        corrected.z_lower.as_slice(),
                        corrected.z_upper.as_slice(),
                    ]);
                    let corrected_overall_inf = scaled_overall_inf_norm(
                        trial_primal_inf,
                        corrected.dual_inf,
                        corrected_comp_inf,
                        &corrected_all_dual_multipliers,
                        &corrected_complementarity_multipliers,
                        options.overall_scale_max,
                    );
                    (
                        corrected.z,
                        corrected.z_lower,
                        corrected.z_upper,
                        corrected.dual_inf,
                        corrected_comp_inf,
                        barrier_parameter_value,
                        corrected_overall_inf,
                        true,
                    )
                } else {
                    (
                        trial_z,
                        trial_z_lower,
                        trial_z_upper,
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
                    inequality_multipliers: trial_lambda_ineq.clone(),
                    kkt_inequality_residual: slack_form_inequality_residuals(
                        &trial_state.augmented_inequality_values,
                        &trial_slack,
                    ),
                    kkt_slack_stationarity: damped_slack_stationarity_residuals(
                        &trial_lambda_ineq,
                        &accepted_z,
                        barrier_parameter_value,
                        options.kappa_d,
                    ),
                    kkt_slack_complementarity: slack_complementarity_residuals(
                        &trial_slack_barrier,
                        &accepted_z,
                        barrier_parameter_value,
                    ),
                    kkt_slack_sigma: slack_sigma_values(&trial_slack_barrier, &accepted_z),
                    slack: trial_slack,
                    z: accepted_z,
                    z_lower: accepted_z_lower,
                    z_upper: accepted_z_upper,
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
                    step_direction: Some(step_direction_snapshot(&direction)),
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
                    tiny_step: false,
                    bound_multiplier_corrected,
                });
                break;
            }
            if options.second_order_correction
                && options.max_second_order_corrections > 0
                && !second_order_correction_attempted
                && line_search_iterations == 0
                && current_theta <= trial_filter_theta
            {
                second_order_correction_attempted = true;
                let mut soc_count = 0;
                let mut soc_theta_old = 0.0;
                let mut soc_theta_trial = trial_filter_theta;
                let mut soc_alpha_pr = trial_alpha_pr;
                let mut soc_eq_accumulator = state.equality_values.clone();
                let mut soc_ineq_accumulator = inequality_residual.clone();
                let mut soc_trial_eq_values = trial_state.equality_values.clone();
                let mut soc_trial_ineq_residual = slack_form_inequality_residuals(
                    &trial_state.augmented_inequality_values,
                    &trial_slack,
                );

                while soc_count < options.max_second_order_corrections
                    && (soc_count == 0
                        || soc_theta_trial
                            <= options.second_order_correction_reduction_factor * soc_theta_old)
                {
                    soc_theta_old = soc_theta_trial;
                    let soc_eq_residual = soc_trial_eq_values
                        .iter()
                        .zip(soc_eq_accumulator.iter())
                        .map(|(trial_i, previous_i)| trial_i + soc_alpha_pr * previous_i)
                        .collect::<Vec<_>>();
                    let soc_ineq_residual = soc_trial_ineq_residual
                        .iter()
                        .zip(soc_ineq_accumulator.iter())
                        .map(|(trial_i, previous_i)| trial_i + soc_alpha_pr * previous_i)
                        .collect::<Vec<_>>();
                    soc_eq_accumulator = soc_eq_residual.clone();
                    soc_ineq_accumulator = soc_ineq_residual.clone();

                    let soc_constraint_kkt_system = reduced_kkt_system
                        .with_constraint_residuals(&soc_eq_residual, &soc_ineq_residual);
                    let soc_reduced_kkt_system = soc_constraint_kkt_system
                        .with_forced_perturbations(
                            direction.primal_diagonal_shift_used,
                            direction.dual_regularization_used,
                        );
                    let soc_linear_started = Instant::now();
                    let mut soc_direction = match solve_reduced_kkt(
                        &soc_reduced_kkt_system,
                        spral_workspace.as_mut(),
                        native_spral_workspace.as_mut(),
                        &mut profiling,
                        options.verbose,
                    ) {
                        Ok(direction) => direction,
                        Err(_) => break,
                    };
                    let soc_linear_elapsed = soc_linear_started.elapsed();
                    profiling.linear_solves += 1;
                    profiling.linear_solve_time += soc_linear_elapsed;
                    iteration_linear_solve_time += soc_linear_elapsed;
                    if soc_direction.regularization_used > kkt_regularization * (1.0 + 1e-12) {
                        push_unique_nlip_event(
                            &mut iteration_events,
                            InteriorPointIterationEvent::AdaptiveRegularizationUsed,
                        );
                    }
                    if linear_solver_quality_was_increased(&soc_direction.backend_stats) {
                        push_unique_nlip_event(
                            &mut iteration_events,
                            InteriorPointIterationEvent::LinearSolverQualityIncreased,
                        );
                    }
                    soc_direction.dx = fixed_variables.expand_direction(&soc_direction.dx);
                    let (soc_dz_lower, soc_dz_upper) = native_bound_multiplier_steps(
                        &x,
                        &soc_direction.dx,
                        &bounds,
                        &z_lower,
                        &z_upper,
                        barrier_parameter_value,
                        sigma,
                    );
                    soc_direction.dz_lower = soc_dz_lower;
                    soc_direction.dz_upper = soc_dz_upper;

                    let (soc_slack_alpha_pr, _) = if augmented_inequality_count > 0 {
                        let soc_slack_barrier_direction =
                            slack_barrier_direction_values(&soc_direction.ds);
                        fraction_to_boundary_with_limiter(
                            &slack_barrier,
                            &soc_slack_barrier_direction,
                            fraction_to_boundary_tau,
                            InteriorPointBoundaryLimiterKind::Slack,
                        )
                    } else {
                        (1.0, None)
                    };
                    let (soc_bound_alpha_pr, _) = fraction_to_variable_bounds_with_limiter(
                        &x,
                        &soc_direction.dx,
                        &bounds,
                        fraction_to_boundary_tau,
                    );
                    soc_alpha_pr = soc_slack_alpha_pr.min(soc_bound_alpha_pr).clamp(0.0, 1.0);
                    if soc_alpha_pr <= 0.0 {
                        break;
                    }

                    let corrected_x = x
                        .iter()
                        .zip(soc_direction.dx.iter())
                        .map(|(value, delta)| value + soc_alpha_pr * delta)
                        .collect::<Vec<_>>();
                    let corrected_slack = slack
                        .iter()
                        .zip(soc_direction.ds.iter())
                        .map(|(value, delta)| value + soc_alpha_pr * delta)
                        .collect::<Vec<_>>();
                    let corrected_slack_barrier =
                        slack_barrier_values(&corrected_slack, &slack_upper_bounds);
                    let corrected_bounds_positive = bounds
                        .lower_indices
                        .iter()
                        .zip(bounds.lower_values.iter())
                        .all(|(&index, &lower)| corrected_x[index] > lower)
                        && bounds
                            .upper_indices
                            .iter()
                            .zip(bounds.upper_values.iter())
                            .all(|(&index, &upper)| corrected_x[index] < upper);
                    if corrected_slack_barrier.iter().any(|value| *value <= 0.0)
                        || !corrected_bounds_positive
                    {
                        break;
                    }

                    let mut corrected_callback_time = Duration::ZERO;
                    let corrected_state = self::trial_state(
                        problem,
                        &corrected_x,
                        parameters,
                        &trial_evaluation_context,
                        &mut profiling,
                        &mut corrected_callback_time,
                    );
                    let corrected_eq_inf = inf_norm(&corrected_state.equality_values);
                    let corrected_ineq_inf = inequality_upper_bound_inf_norm(
                        &corrected_state.augmented_inequality_values,
                        &slack_upper_bounds,
                    );
                    let corrected_primal_inf =
                        corrected_eq_inf.max(slack_form_inequality_inf_norm(
                            &corrected_state.augmented_inequality_values,
                            &corrected_slack,
                        ));
                    let corrected_filter_theta = filter_theta_l1_norm(
                        &corrected_state.equality_values,
                        &corrected_state.augmented_inequality_values,
                        &corrected_slack,
                    );
                    let corrected_barrier_objective = barrier_objective_value(
                        corrected_state.objective_value,
                        &corrected_slack_barrier,
                        &corrected_x,
                        &bounds,
                        barrier_parameter_value,
                        options.kappa_d,
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
                    let corrected_barrier_objective_too_large =
                        barrier_objective_increase_too_large(
                            current_barrier_objective,
                            corrected_barrier_objective,
                            options.obj_max_inc,
                        );
                    let corrected_filter_acceptance_mode = (!corrected_barrier_objective_too_large)
                        .then_some(corrected_filter_assessment.acceptance_mode)
                        .flatten();

                    if let Some(corrected_filter_acceptance_mode) = corrected_filter_acceptance_mode
                    {
                        let soc_combined_z = combined_multiplier_vector([
                            z.as_slice(),
                            z_lower.as_slice(),
                            z_upper.as_slice(),
                        ]);
                        let soc_combined_dz = combined_multiplier_vector([
                            soc_direction.dz.as_slice(),
                            soc_direction.dz_lower.as_slice(),
                            soc_direction.dz_upper.as_slice(),
                        ]);
                        let soc_alpha_du = if barrier_pair_count > 0 {
                            fraction_to_boundary_with_limiter(
                                &soc_combined_z,
                                &soc_combined_dz,
                                fraction_to_boundary_tau,
                                InteriorPointBoundaryLimiterKind::Multiplier,
                            )
                            .0
                            .clamp(0.0, 1.0)
                        } else {
                            1.0
                        };
                        let soc_alpha_y =
                            alpha_for_y(soc_alpha_pr, soc_alpha_du, &soc_direction, options);
                        let corrected_lambda = lambda_eq
                            .iter()
                            .zip(soc_direction.d_lambda.iter())
                            .map(|(value, delta)| value + soc_alpha_y * delta)
                            .collect::<Vec<_>>();
                        let corrected_lambda_ineq = lambda_ineq
                            .iter()
                            .zip(soc_direction.d_ineq.iter())
                            .map(|(value, delta)| value + soc_alpha_y * delta)
                            .collect::<Vec<_>>();
                        let corrected_z = z
                            .iter()
                            .zip(soc_direction.dz.iter())
                            .map(|(value, delta)| value + soc_alpha_du * delta)
                            .collect::<Vec<_>>();
                        let corrected_z_lower = z_lower
                            .iter()
                            .zip(soc_direction.dz_lower.iter())
                            .map(|(value, delta)| value + soc_alpha_du * delta)
                            .collect::<Vec<_>>();
                        let corrected_z_upper = z_upper
                            .iter()
                            .zip(soc_direction.dz_upper.iter())
                            .map(|(value, delta)| value + soc_alpha_du * delta)
                            .collect::<Vec<_>>();
                        let corrected_raw_dual_residual = lagrangian_gradient_sparse(
                            &corrected_state.gradient,
                            &corrected_state.equality_jacobian,
                            &corrected_lambda,
                            &corrected_state.inequality_jacobian,
                            &corrected_lambda_ineq,
                        );
                        let mut corrected_dual_residual = corrected_raw_dual_residual.clone();
                        add_native_bound_multiplier_terms(
                            &mut corrected_dual_residual,
                            &bounds,
                            &corrected_z_lower,
                            &corrected_z_upper,
                        );
                        let corrected_dual_x_inf =
                            fixed_variables.free_inf_norm(&corrected_dual_residual);
                        let corrected_slack_stationarity_inf =
                            slack_stationarity_inf_norm(&corrected_lambda_ineq, &corrected_z);
                        let corrected_dual_inf =
                            corrected_dual_x_inf.max(corrected_slack_stationarity_inf);
                        let corrected_comp_inf = if barrier_pair_count > 0 {
                            combined_complementarity_inf_norm(
                                &corrected_slack_barrier,
                                &corrected_z,
                                &corrected_x,
                                &bounds,
                                &corrected_z_lower,
                                &corrected_z_upper,
                            )
                        } else {
                            0.0
                        };
                        let step_kind = if corrected_filter_acceptance_mode
                            == FilterAcceptanceMode::ObjectiveArmijo
                        {
                            InteriorPointStepKind::Objective
                        } else {
                            InteriorPointStepKind::Feasibility
                        };
                        let step_tag = if step_kind == InteriorPointStepKind::Feasibility {
                            'H'
                        } else {
                            'F'
                        };
                        let corrected_all_dual_multipliers = combined_multiplier_vector([
                            corrected_lambda.as_slice(),
                            corrected_lambda_ineq.as_slice(),
                            corrected_z.as_slice(),
                            corrected_z_lower.as_slice(),
                            corrected_z_upper.as_slice(),
                        ]);
                        let corrected_complementarity_multipliers = combined_multiplier_vector([
                            corrected_z.as_slice(),
                            corrected_z_lower.as_slice(),
                            corrected_z_upper.as_slice(),
                        ]);
                        let corrected_overall_inf = scaled_overall_inf_norm(
                            corrected_primal_inf,
                            corrected_dual_inf,
                            corrected_comp_inf,
                            &corrected_all_dual_multipliers,
                            &corrected_complementarity_multipliers,
                            options.overall_scale_max,
                        );
                        let corrected_bound_multipliers = apply_bound_multiplier_safeguard(
                            &corrected_lambda_ineq,
                            &corrected_raw_dual_residual,
                            &fixed_variables,
                            &corrected_x,
                            &bounds,
                            &corrected_slack_barrier,
                            &corrected_z,
                            &corrected_z_lower,
                            &corrected_z_upper,
                            barrier_parameter_value,
                            options,
                        );
                        second_order_correction_used = true;
                        let (
                            accepted_z,
                            accepted_z_lower,
                            accepted_z_upper,
                            accepted_dual_inf,
                            accepted_comp_inf,
                            accepted_mu,
                            accepted_overall_inf,
                            bound_multiplier_corrected,
                        ) = if let Some(corrected) = corrected_bound_multipliers {
                            let corrected_comp_inf = combined_complementarity_inf_norm(
                                &corrected_slack_barrier,
                                &corrected.z,
                                &corrected_x,
                                &bounds,
                                &corrected.z_lower,
                                &corrected.z_upper,
                            );
                            let corrected_complementarity_multipliers =
                                combined_multiplier_vector([
                                    corrected.z.as_slice(),
                                    corrected.z_lower.as_slice(),
                                    corrected.z_upper.as_slice(),
                                ]);
                            let corrected_all_dual_multipliers = combined_multiplier_vector([
                                corrected_lambda.as_slice(),
                                corrected_lambda_ineq.as_slice(),
                                corrected.z.as_slice(),
                                corrected.z_lower.as_slice(),
                                corrected.z_upper.as_slice(),
                            ]);
                            let corrected_overall_inf = scaled_overall_inf_norm(
                                corrected_primal_inf,
                                corrected.dual_inf,
                                corrected_comp_inf,
                                &corrected_all_dual_multipliers,
                                &corrected_complementarity_multipliers,
                                options.overall_scale_max,
                            );
                            (
                                corrected.z,
                                corrected.z_lower,
                                corrected.z_upper,
                                corrected.dual_inf,
                                corrected_comp_inf,
                                barrier_parameter_value,
                                corrected_overall_inf,
                                true,
                            )
                        } else {
                            (
                                corrected_z,
                                corrected_z_lower,
                                corrected_z_upper,
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
                            inequality_multipliers: corrected_lambda_ineq.clone(),
                            kkt_inequality_residual: slack_form_inequality_residuals(
                                &corrected_state.augmented_inequality_values,
                                &corrected_slack,
                            ),
                            kkt_slack_stationarity: damped_slack_stationarity_residuals(
                                &corrected_lambda_ineq,
                                &accepted_z,
                                barrier_parameter_value,
                                options.kappa_d,
                            ),
                            kkt_slack_complementarity: slack_complementarity_residuals(
                                &corrected_slack_barrier,
                                &accepted_z,
                                barrier_parameter_value,
                            ),
                            kkt_slack_sigma: slack_sigma_values(
                                &corrected_slack_barrier,
                                &accepted_z,
                            ),
                            slack: corrected_slack,
                            z: accepted_z,
                            z_lower: accepted_z_lower,
                            z_upper: accepted_z_upper,
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
                            filter_acceptance_mode: Some(corrected_filter_acceptance_mode),
                            step_kind,
                            step_tag,
                            step_direction: Some(step_direction_snapshot(&soc_direction)),
                            phase: InteriorPointIterationPhase::AcceptedStep,
                            accepted_alpha_pr: soc_alpha_pr,
                            accepted_alpha_du: Some(soc_alpha_du),
                            line_search_initial_alpha_pr: alpha_pr,
                            line_search_initial_alpha_du: Some(alpha_du),
                            line_search_last_alpha_pr: soc_alpha_pr,
                            line_search_last_alpha_du: Some(soc_alpha_du),
                            line_search_backtrack_count: line_search_iterations,
                            second_order_correction_used: true,
                            watchdog_accepted: false,
                            tiny_step: false,
                            bound_multiplier_corrected,
                        });
                        break;
                    }

                    soc_count += 1;
                    soc_theta_trial = corrected_filter_theta;
                    soc_trial_eq_values = corrected_state.equality_values.clone();
                    soc_trial_ineq_residual = slack_form_inequality_residuals(
                        &corrected_state.augmented_inequality_values,
                        &corrected_slack,
                    );
                }
                if accepted.is_some() {
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
                    &trial_lambda_ineq,
                    &trial_raw_dual_residual,
                    &fixed_variables,
                    &trial_x,
                    &bounds,
                    &trial_slack_barrier,
                    &trial_z,
                    &trial_z_lower,
                    &trial_z_upper,
                    barrier_parameter_value,
                    options,
                );
                let (
                    accepted_z,
                    accepted_z_lower,
                    accepted_z_upper,
                    accepted_dual_inf,
                    accepted_comp_inf,
                    accepted_mu,
                    accepted_overall_inf,
                    bound_multiplier_corrected,
                ) = if let Some(corrected) = corrected_bound_multipliers {
                    let corrected_comp_inf = combined_complementarity_inf_norm(
                        &trial_slack_barrier,
                        &corrected.z,
                        &trial_x,
                        &bounds,
                        &corrected.z_lower,
                        &corrected.z_upper,
                    );
                    let corrected_complementarity_multipliers = combined_multiplier_vector([
                        corrected.z.as_slice(),
                        corrected.z_lower.as_slice(),
                        corrected.z_upper.as_slice(),
                    ]);
                    let corrected_all_dual_multipliers = combined_multiplier_vector([
                        trial_lambda.as_slice(),
                        trial_lambda_ineq.as_slice(),
                        corrected.z.as_slice(),
                        corrected.z_lower.as_slice(),
                        corrected.z_upper.as_slice(),
                    ]);
                    let corrected_overall_inf = scaled_overall_inf_norm(
                        trial_primal_inf,
                        corrected.dual_inf,
                        corrected_comp_inf,
                        &corrected_all_dual_multipliers,
                        &corrected_complementarity_multipliers,
                        options.overall_scale_max,
                    );
                    (
                        corrected.z,
                        corrected.z_lower,
                        corrected.z_upper,
                        corrected.dual_inf,
                        corrected_comp_inf,
                        barrier_parameter_value,
                        corrected_overall_inf,
                        true,
                    )
                } else {
                    (
                        trial_z,
                        trial_z_lower,
                        trial_z_upper,
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
                    inequality_multipliers: trial_lambda_ineq.clone(),
                    kkt_inequality_residual: slack_form_inequality_residuals(
                        &trial_state.augmented_inequality_values,
                        &trial_slack,
                    ),
                    kkt_slack_stationarity: damped_slack_stationarity_residuals(
                        &trial_lambda_ineq,
                        &accepted_z,
                        barrier_parameter_value,
                        options.kappa_d,
                    ),
                    kkt_slack_complementarity: slack_complementarity_residuals(
                        &trial_slack_barrier,
                        &accepted_z,
                        barrier_parameter_value,
                    ),
                    kkt_slack_sigma: slack_sigma_values(&trial_slack_barrier, &accepted_z),
                    slack: trial_slack,
                    z: accepted_z,
                    z_lower: accepted_z_lower,
                    z_upper: accepted_z_upper,
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
                    step_direction: Some(step_direction_snapshot(&direction)),
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
                    tiny_step: false,
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
        if accepted.is_none()
            && options.restoration_phase
            && augmented_inequality_count == 0
            && let Some(restoration_summary) = solve_equality_restoration_problem(
                problem,
                &x,
                &state.equality_values,
                parameters,
                options,
                current_primal_inf,
            )
        {
            let restored_x = restoration_summary.x[..problem.dimension()].to_vec();
            let restored_state = trial_state(
                problem,
                &restored_x,
                parameters,
                &trial_evaluation_context,
                &mut profiling,
                &mut iteration_callback_time,
            );
            let restored_eq_inf = inf_norm(&restored_state.equality_values);
            let restored_ineq_inf =
                positive_part_inf_norm(&restored_state.augmented_inequality_values);
            let restored_primal_inf = restored_eq_inf.max(restored_ineq_inf);
            if restored_primal_inf
                <= 0.9 * current_primal_inf.max(options.constraint_tol)
                    + 100.0 * f64::EPSILON * current_primal_inf.abs().max(1.0)
            {
                let restored_linear_state = EvalState {
                    objective_value: restored_state.objective_value,
                    gradient: fixed_variables.reduce_vector(&restored_state.gradient),
                    equality_values: restored_state.equality_values.clone(),
                    augmented_inequality_values: restored_state.augmented_inequality_values.clone(),
                    equality_jacobian: reduce_sparse_matrix_columns(
                        &restored_state.equality_jacobian,
                        &equality_column_reduction,
                    ),
                    inequality_jacobian: reduce_sparse_matrix_columns(
                        &restored_state.inequality_jacobian,
                        &inequality_column_reduction,
                    ),
                };
                let (restored_lambda, restored_lambda_ineq) = least_squares_constraint_multipliers(
                    &restored_linear_state,
                    &[],
                    hessian_structure.as_ref(),
                    options.regularization,
                    options.linear_solver,
                );
                let restored_raw_dual_residual = lagrangian_gradient_sparse(
                    &restored_state.gradient,
                    &restored_state.equality_jacobian,
                    &restored_lambda,
                    &restored_state.inequality_jacobian,
                    &restored_lambda_ineq,
                );
                let mut restored_dual_residual = restored_raw_dual_residual;
                add_native_bound_multiplier_terms(
                    &mut restored_dual_residual,
                    &bounds,
                    &z_lower,
                    &z_upper,
                );
                let restored_dual_inf = fixed_variables.free_inf_norm(&restored_dual_residual);
                let restored_complementarity_inf = if barrier_pair_count > 0 {
                    combined_complementarity_inf_norm(
                        &[],
                        &[],
                        &restored_x,
                        &bounds,
                        &z_lower,
                        &z_upper,
                    )
                } else {
                    0.0
                };
                let restored_all_dual_multipliers = combined_multiplier_vector([
                    restored_lambda.as_slice(),
                    restored_lambda_ineq.as_slice(),
                    z.as_slice(),
                    z_lower.as_slice(),
                    z_upper.as_slice(),
                ]);
                let restored_complementarity_multipliers = combined_multiplier_vector([
                    z.as_slice(),
                    z_lower.as_slice(),
                    z_upper.as_slice(),
                ]);
                let restored_overall_inf = scaled_overall_inf_norm(
                    restored_primal_inf,
                    restored_dual_inf,
                    restored_complementarity_inf,
                    &restored_all_dual_multipliers,
                    &restored_complementarity_multipliers,
                    options.overall_scale_max,
                );
                let restored_filter_theta = filter_theta_l1_norm(
                    &restored_state.equality_values,
                    &restored_state.augmented_inequality_values,
                    &[],
                );
                let restored_barrier_objective = barrier_objective_value(
                    restored_state.objective_value,
                    &[],
                    &restored_x,
                    &bounds,
                    barrier_parameter_value,
                    options.kappa_d,
                );
                let restored_filter_entry =
                    super::filter::entry(restored_barrier_objective, restored_filter_theta);
                let restoration_filter_assessment = super::filter::assess_trial(
                    &filter_entries,
                    &current_filter_entry,
                    &restored_filter_entry,
                    0.0,
                    barrier_directional_derivative,
                    false,
                    false,
                    filter_parameters,
                );
                if restoration_filter_assessment.current_iterate_acceptable
                    && restoration_filter_assessment.filter_acceptable
                {
                    push_unique_nlip_event(
                        &mut iteration_events,
                        InteriorPointIterationEvent::RestorationPhaseAccepted,
                    );
                    accepted = Some(AcceptedInteriorPointTrial {
                        x: restored_x,
                        lambda: restored_lambda,
                        inequality_multipliers: restored_lambda_ineq.clone(),
                        slack: Vec::new(),
                        z: z.clone(),
                        z_lower: z_lower.clone(),
                        z_upper: z_upper.clone(),
                        kkt_inequality_residual: Vec::new(),
                        kkt_slack_stationarity: Vec::new(),
                        kkt_slack_complementarity: Vec::new(),
                        kkt_slack_sigma: Vec::new(),
                        objective: restored_state.objective_value,
                        barrier_objective: restored_barrier_objective,
                        equality_inf: restored_eq_inf,
                        inequality_inf: restored_ineq_inf,
                        dual_inf: restored_dual_inf,
                        complementarity_inf: restored_complementarity_inf,
                        overall_inf: restored_overall_inf,
                        mu: barrier_parameter_value,
                        filter_entry: restored_filter_entry,
                        filter_augment_entry: Some(super::filter::augment_entry(
                            current_barrier_objective,
                            current_theta,
                            options.filter_gamma_objective,
                            options.filter_gamma_violation,
                        )),
                        filter_theta: restored_filter_theta,
                        filter_acceptance_mode: None,
                        step_kind: InteriorPointStepKind::Feasibility,
                        step_tag: 'r',
                        step_direction: None,
                        phase: InteriorPointIterationPhase::AcceptedStep,
                        accepted_alpha_pr: last_tried_alpha_pr,
                        accepted_alpha_du: Some(0.0),
                        line_search_initial_alpha_pr: alpha_pr,
                        line_search_initial_alpha_du: Some(alpha_du),
                        line_search_last_alpha_pr: last_tried_alpha_pr,
                        line_search_last_alpha_du: Some(last_tried_alpha_du),
                        line_search_backtrack_count: line_search_iterations,
                        second_order_correction_used: false,
                        watchdog_accepted: false,
                        tiny_step: false,
                        bound_multiplier_corrected: false,
                    });
                }
            }
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
        let step_inf = ipopt_primal_step_inf_norm(&direction);
        let tiny_step = accepted_trial.tiny_step;
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
        let filter_reset_applied = if options.max_filter_resets > 0 {
            if last_rejection_due_to_filter {
                successive_filter_rejections += 1;
                if successive_filter_rejections >= options.filter_reset_trigger {
                    // IPOPT 3.14's n_filter_resets_ is checked but not incremented, so
                    // max_filter_resets behaves as an enable/disable flag in practice.
                    next_filter_entries.clear();
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
        if filter_reset_applied {
            push_unique_nlip_event(&mut events, InteriorPointIterationEvent::FilterReset);
        }
        if let Some(entry) = accepted_trial.filter_augment_entry.clone() {
            super::filter::update_frontier(&mut next_filter_entries, entry);
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
                has_inequalities: barrier_pair_count > 0,
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
                    barrier_parameter: if barrier_pair_count > 0 {
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
            slack_primal: Some(accepted_trial.slack.clone()),
            equality_multipliers: Some(accepted_trial.lambda.clone()),
            inequality_multipliers: Some(accepted_trial.inequality_multipliers.clone()),
            slack_multipliers: Some(accepted_trial.z.clone()),
            lower_bound_multipliers: Some(accepted_trial.z_lower.clone()),
            upper_bound_multipliers: Some(accepted_trial.z_upper.clone()),
            kkt_inequality_residual: Some(accepted_trial.kkt_inequality_residual.clone()),
            kkt_slack_stationarity: Some(accepted_trial.kkt_slack_stationarity.clone()),
            kkt_slack_complementarity: Some(accepted_trial.kkt_slack_complementarity.clone()),
            kkt_slack_sigma: Some(accepted_trial.kkt_slack_sigma.clone()),
            objective: accepted_trial.objective,
            barrier_objective: Some(accepted_trial.barrier_objective),
            eq_inf: (equality_count > 0).then_some(accepted_trial.equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(accepted_trial.inequality_inf),
            dual_inf: accepted_trial.dual_inf,
            comp_inf: (barrier_pair_count > 0).then_some(accepted_trial.complementarity_inf),
            overall_inf: accepted_trial.overall_inf,
            barrier_parameter: (barrier_pair_count > 0).then_some(accepted_trial.mu),
            filter_theta: Some(accepted_trial.filter_theta),
            step_inf: Some(step_inf),
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
            step_direction: accepted_trial.step_direction.clone(),
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
        if direction.primal_diagonal_shift_used > 0.0 {
            previous_hessian_perturbation = Some(direction.primal_diagonal_shift_used);
        }

        profiling.preprocessing_steps += 1;
        profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
            iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
        );
        x = accepted_trial.x;
        lambda_eq = accepted_trial.lambda;
        lambda_ineq = accepted_trial.inequality_multipliers;
        slack = accepted_trial.slack;
        z = accepted_trial.z;
        z_lower = accepted_trial.z_lower;
        z_upper = accepted_trial.z_upper;
        filter_entries = next_filter_entries;
        nonlinear_inequality_multipliers = lambda_ineq.clone();
        if shortened_step {
            watchdog_state.shortened_step_streak += 1;
        } else {
            watchdog_state.shortened_step_streak = 0;
        }
        let dual_step_norm =
            step_inf_norm(&direction.d_lambda).max(step_inf_norm(&direction.d_ineq));
        watchdog_state.tiny_step_last_iteration =
            tiny_step && dual_step_norm < options.tiny_step_y_tol;
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
