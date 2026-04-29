use crate::{
    BackendTimingMetadata, CCS, CompiledNlpProblem, EvalTimingStat, Index, ParameterMatrix,
    SolverAdapterTiming, complementarity_inf_norm, inf_norm, lagrangian_gradient,
    positive_part_inf_norm, validate_nlp_problem_shapes, validate_parameter_inputs,
};
use anyhow::{Result, bail};
use ipopt::{
    AlgorithmMode, BasicProblem, ConstrainedProblem, Index as IpoptIndex, IntermediateCallbackData,
    Ipopt, NewtonProblem, Number, SolveStatus,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

const IPOPT_INF: f64 = 1e20;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptMuStrategy {
    Adaptive,
    Monotone,
}

impl IpoptMuStrategy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Monotone => "monotone",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptNlpScalingMethod {
    None,
    GradientBased,
}

impl IpoptNlpScalingMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::GradientBased => "gradient-based",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptLinearSolver {
    Spral,
}

impl IpoptLinearSolver {
    fn as_str(self) -> &'static str {
        match self {
            Self::Spral => "spral",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptSpralPivotMethod {
    Block,
}

impl IpoptSpralPivotMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptSpralOrdering {
    Metis,
    Matching,
}

impl IpoptSpralOrdering {
    fn as_str(self) -> &'static str {
        match self {
            Self::Metis => "metis",
            Self::Matching => "matching",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptSpralScaling {
    None,
    Mc64,
    Auction,
    Matching,
    Ruiz,
    Dynamic,
}

impl IpoptSpralScaling {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Mc64 => "mc64",
            Self::Auction => "auction",
            Self::Matching => "matching",
            Self::Ruiz => "ruiz",
            Self::Dynamic => "dynamic",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum IpoptRawOptionValue {
    Number(f64),
    Integer(i32),
    Text(String),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IpoptRawOption {
    pub name: String,
    pub value: IpoptRawOptionValue,
}

impl IpoptRawOption {
    pub fn number(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value: IpoptRawOptionValue::Number(value),
        }
    }

    pub fn integer(name: impl Into<String>, value: i32) -> Self {
        Self {
            name: name.into(),
            value: IpoptRawOptionValue::Integer(value),
        }
    }

    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: IpoptRawOptionValue::Text(value.into()),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IpoptProvenance {
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_solver_stack: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_ipopt_version: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_ipopt_source_commit: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_spral_version: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_metis_version: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_openblas_version: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_openblas_threading: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_static_solver_math: Option<bool>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_ipopt_lib_dir: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_spral_lib_dir: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_metis_lib_dir: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_openblas_lib_dir: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linked_openmp_lib: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub pkg_config_version: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub pkg_config_cflags_libs: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub ipopt_binary: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linear_solver_default: Option<String>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct IpoptOptions {
    pub max_iters: Index,
    pub tol: f64,
    pub acceptable_tol: Option<f64>,
    pub constraint_tol: Option<f64>,
    pub complementarity_tol: Option<f64>,
    pub dual_tol: Option<f64>,
    pub print_level: i32,
    pub suppress_banner: bool,
    pub mu_strategy: IpoptMuStrategy,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub nlp_scaling_method: Option<IpoptNlpScalingMethod>,
    pub kappa_d: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub linear_solver: Option<IpoptLinearSolver>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_pivot_method: Option<IpoptSpralPivotMethod>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_ordering: Option<IpoptSpralOrdering>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_scaling: Option<IpoptSpralScaling>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_small_pivot_tolerance: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_threshold_pivot_u: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_pivot_tolerance_max: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub spral_use_gpu: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub raw_options: Vec<IpoptRawOption>,
    pub capture_provenance: bool,
}

impl Default for IpoptOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tol: 1e-8,
            acceptable_tol: Some(1e-6),
            constraint_tol: Some(1e-8),
            complementarity_tol: Some(1e-8),
            dual_tol: Some(1e-8),
            print_level: 0,
            suppress_banner: true,
            mu_strategy: IpoptMuStrategy::Adaptive,
            nlp_scaling_method: None,
            kappa_d: 1e-5,
            linear_solver: None,
            spral_pivot_method: None,
            spral_ordering: None,
            spral_scaling: None,
            spral_small_pivot_tolerance: None,
            spral_threshold_pivot_u: None,
            spral_pivot_tolerance_max: None,
            spral_use_gpu: None,
            raw_options: Vec::new(),
            capture_provenance: false,
        }
    }
}

pub fn format_ipopt_settings_summary(options: &IpoptOptions) -> String {
    format!(
        "mu_strategy={}; nlp_scaling={}; kappa_d={:.1e}; acceptable_tol={}; print_level={}; banner={}; linear_solver={}; spral_pivot={}; spral_order={}; spral_scaling={}; spral_small={}; spral_u={}; spral_umax={}; spral_gpu={}; raw_options={}; provenance={}",
        options.mu_strategy.as_str(),
        options
            .nlp_scaling_method
            .map_or("problem/default", IpoptNlpScalingMethod::as_str),
        options.kappa_d,
        options
            .acceptable_tol
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "off".to_string()),
        options.print_level,
        if options.suppress_banner { "off" } else { "on" },
        options
            .linear_solver
            .map(|value| value.as_str().to_string())
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_pivot_method
            .map(|value| value.as_str().to_string())
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_ordering
            .map(|value| value.as_str().to_string())
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_scaling
            .map(|value| value.as_str().to_string())
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_small_pivot_tolerance
            .map(|value| format!("{value:.1e}"))
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_threshold_pivot_u
            .map(|value| format!("{value:.1e}"))
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_pivot_tolerance_max
            .map(|value| format!("{value:.1e}"))
            .unwrap_or_else(|| "default".to_string()),
        options
            .spral_use_gpu
            .map(|value| if value { "yes" } else { "no" }.to_string())
            .unwrap_or_else(|| "default".to_string()),
        options.raw_options.len(),
        if options.capture_provenance {
            "on"
        } else {
            "off"
        },
    )
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptIterationPhase {
    Regular,
    Restoration,
}

impl From<AlgorithmMode> for IpoptIterationPhase {
    fn from(value: AlgorithmMode) -> Self {
        match value {
            AlgorithmMode::Regular => Self::Regular,
            AlgorithmMode::RestorationPhase => Self::Restoration,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptRawStatus {
    SolveSucceeded,
    SolvedToAcceptableLevel,
    FeasiblePointFound,
    InfeasibleProblemDetected,
    SearchDirectionBecomesTooSmall,
    DivergingIterates,
    UserRequestedStop,
    MaximumIterationsExceeded,
    MaximumCpuTimeExceeded,
    RestorationFailed,
    ErrorInStepComputation,
    InvalidOption,
    NotEnoughDegreesOfFreedom,
    InvalidProblemDefinition,
    InvalidNumberDetected,
    UnrecoverableException,
    NonIpoptExceptionThrown,
    InsufficientMemory,
    InternalError,
    UnknownError,
}

impl From<SolveStatus> for IpoptRawStatus {
    fn from(status: SolveStatus) -> Self {
        match status {
            SolveStatus::SolveSucceeded => Self::SolveSucceeded,
            SolveStatus::SolvedToAcceptableLevel => Self::SolvedToAcceptableLevel,
            SolveStatus::FeasiblePointFound => Self::FeasiblePointFound,
            SolveStatus::InfeasibleProblemDetected => Self::InfeasibleProblemDetected,
            SolveStatus::SearchDirectionBecomesTooSmall => Self::SearchDirectionBecomesTooSmall,
            SolveStatus::DivergingIterates => Self::DivergingIterates,
            SolveStatus::UserRequestedStop => Self::UserRequestedStop,
            SolveStatus::MaximumIterationsExceeded => Self::MaximumIterationsExceeded,
            SolveStatus::MaximumCpuTimeExceeded => Self::MaximumCpuTimeExceeded,
            SolveStatus::RestorationFailed => Self::RestorationFailed,
            SolveStatus::ErrorInStepComputation => Self::ErrorInStepComputation,
            SolveStatus::InvalidOption => Self::InvalidOption,
            SolveStatus::NotEnoughDegreesOfFreedom => Self::NotEnoughDegreesOfFreedom,
            SolveStatus::InvalidProblemDefinition => Self::InvalidProblemDefinition,
            SolveStatus::InvalidNumberDetected => Self::InvalidNumberDetected,
            SolveStatus::UnrecoverableException => Self::UnrecoverableException,
            SolveStatus::NonIpoptExceptionThrown => Self::NonIpoptExceptionThrown,
            SolveStatus::InsufficientMemory => Self::InsufficientMemory,
            SolveStatus::InternalError => Self::InternalError,
            SolveStatus::UnknownError => Self::UnknownError,
        }
    }
}

impl std::fmt::Display for IpoptRawStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IpoptIterationTiming {
    pub adapter_timing: Option<SolverAdapterTiming>,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub objective_value: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub objective_gradient: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub constraint_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub constraint_jacobian_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub hessian_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub total_callback: Duration,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IpoptIterationSnapshot {
    pub iteration: Index,
    pub phase: IpoptIterationPhase,
    pub x: Vec<f64>,
    pub internal_slack: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub slack_lower_bound_multipliers: Vec<f64>,
    pub slack_upper_bound_multipliers: Vec<f64>,
    pub direction_x: Vec<f64>,
    pub direction_slack: Vec<f64>,
    pub direction_equality_multipliers: Vec<f64>,
    pub direction_inequality_multipliers: Vec<f64>,
    pub direction_lower_bound_multipliers: Vec<f64>,
    pub direction_upper_bound_multipliers: Vec<f64>,
    pub direction_slack_lower_bound_multipliers: Vec<f64>,
    pub direction_slack_upper_bound_multipliers: Vec<f64>,
    pub kkt_x_stationarity: Vec<f64>,
    pub kkt_slack_stationarity: Vec<f64>,
    pub kkt_equality_residual: Vec<f64>,
    pub kkt_inequality_residual: Vec<f64>,
    pub kkt_slack_complementarity: Vec<f64>,
    pub kkt_slack_sigma: Vec<f64>,
    pub kkt_slack_distance: Vec<f64>,
    pub curr_grad_f: Vec<f64>,
    pub curr_jac_c_t_y_c: Vec<f64>,
    pub curr_jac_d_t_y_d: Vec<f64>,
    pub curr_grad_lag_x: Vec<f64>,
    pub curr_grad_lag_s: Vec<f64>,
    pub curr_barrier_error: f64,
    pub curr_primal_infeasibility: f64,
    pub curr_dual_infeasibility: f64,
    pub curr_complementarity: f64,
    pub curr_nlp_error: f64,
    pub objective: f64,
    pub primal_inf: f64,
    pub dual_inf: f64,
    pub barrier_parameter: f64,
    pub step_inf: f64,
    pub regularization_size: f64,
    pub alpha_pr: f64,
    pub alpha_du: f64,
    pub line_search_trials: Index,
    pub timing: IpoptIterationTiming,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IpoptProfiling {
    pub objective_value: EvalTimingStat,
    pub objective_gradient: EvalTimingStat,
    pub constraint_values: EvalTimingStat,
    pub constraint_jacobian_values: EvalTimingStat,
    pub hessian_values: EvalTimingStat,
    pub adapter_timing: Option<SolverAdapterTiming>,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub total_time: Duration,
    pub backend_timing: BackendTimingMetadata,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct IpoptPartialSolution {
    pub x: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub objective: f64,
    pub equality_inf_norm: f64,
    pub inequality_inf_norm: f64,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: f64,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct IpoptSummary {
    pub x: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub status: IpoptRawStatus,
    pub equality_inf_norm: f64,
    pub inequality_inf_norm: f64,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: f64,
    pub snapshots: Vec<IpoptIterationSnapshot>,
    pub journal_output: Option<String>,
    pub profiling: IpoptProfiling,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub provenance: Option<IpoptProvenance>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Error)]
pub enum IpoptSolveError {
    #[error("invalid IPOPT input: {0}")]
    InvalidInput(String),
    #[error("ipopt setup failed: {0}")]
    Setup(String),
    #[error("ipopt rejected option `{name}`")]
    OptionRejected { name: String },
    #[error("ipopt failed with status {status:?}")]
    Solve {
        status: IpoptRawStatus,
        iterations: Index,
        snapshots: Vec<IpoptIterationSnapshot>,
        partial_solution: Option<Box<IpoptPartialSolution>>,
        journal_output: Option<String>,
        profiling: Box<IpoptProfiling>,
    },
}

#[derive(Clone, Copy, Debug, Default)]
struct IpoptTimingCheckpoint {
    objective_value: Duration,
    objective_gradient: Duration,
    constraint_values: Duration,
    constraint_jacobian_values: Duration,
    hessian_values: Duration,
    adapter_timing: Option<SolverAdapterTiming>,
}

impl IpoptTimingCheckpoint {
    fn from_profiling(profiling: &IpoptProfiling) -> Self {
        Self {
            objective_value: profiling.objective_value.total_time,
            objective_gradient: profiling.objective_gradient.total_time,
            constraint_values: profiling.constraint_values.total_time,
            constraint_jacobian_values: profiling.constraint_jacobian_values.total_time,
            hessian_values: profiling.hessian_values.total_time,
            adapter_timing: profiling.adapter_timing,
        }
    }

    fn delta_since(self, baseline: Self) -> IpoptIterationTiming {
        let adapter_timing = match (self.adapter_timing, baseline.adapter_timing) {
            (Some(current), Some(previous)) => Some(current.saturating_sub(previous)),
            (Some(current), None) => Some(current),
            (None, _) => None,
        };
        let objective_value = self
            .objective_value
            .saturating_sub(baseline.objective_value);
        let objective_gradient = self
            .objective_gradient
            .saturating_sub(baseline.objective_gradient);
        let constraint_values = self
            .constraint_values
            .saturating_sub(baseline.constraint_values);
        let constraint_jacobian_values = self
            .constraint_jacobian_values
            .saturating_sub(baseline.constraint_jacobian_values);
        let hessian_values = self.hessian_values.saturating_sub(baseline.hessian_values);
        IpoptIterationTiming {
            adapter_timing,
            objective_value,
            objective_gradient,
            constraint_values,
            constraint_jacobian_values,
            hessian_values,
            total_callback: objective_value
                + objective_gradient
                + constraint_values
                + constraint_jacobian_values
                + hessian_values,
        }
    }
}

#[derive(Clone, Debug)]
struct IpoptRuntimeState {
    profiling: IpoptProfiling,
    last_checkpoint: IpoptTimingCheckpoint,
}

trait IpoptIterationConsumer {
    fn accept(&mut self, snapshot: &IpoptIterationSnapshot);
}

struct NoopIpoptIterationConsumer;

impl IpoptIterationConsumer for NoopIpoptIterationConsumer {
    fn accept(&mut self, _snapshot: &IpoptIterationSnapshot) {}
}

impl<F> IpoptIterationConsumer for F
where
    F: FnMut(&IpoptIterationSnapshot),
{
    fn accept(&mut self, snapshot: &IpoptIterationSnapshot) {
        self(snapshot);
    }
}

fn ccs_triplet_indices(
    ccs: &CCS,
    row_offset: Index,
) -> std::result::Result<(Vec<IpoptIndex>, Vec<IpoptIndex>), IpoptSolveError> {
    let mut rows = Vec::with_capacity(ccs.nnz());
    let mut cols = Vec::with_capacity(ccs.nnz());
    for col in 0..ccs.ncol {
        let start = ccs.col_ptrs[col];
        let end = ccs.col_ptrs[col + 1];
        let col_index = IpoptIndex::try_from(col).map_err(|_| {
            IpoptSolveError::InvalidInput(format!(
                "column index {col} does not fit into Ipopt index type"
            ))
        })?;
        for &row in &ccs.row_indices[start..end] {
            let row_index = row + row_offset;
            rows.push(IpoptIndex::try_from(row_index).map_err(|_| {
                IpoptSolveError::InvalidInput(format!(
                    "row index {row_index} does not fit into Ipopt index type"
                ))
            })?);
            cols.push(col_index);
        }
    }
    Ok((rows, cols))
}

fn validate_ipopt_compatibility<P>(problem: &P) -> Result<()>
where
    P: CompiledNlpProblem,
{
    validate_nlp_problem_shapes(problem)?;
    let hessian_ccs = problem.lagrangian_hessian_ccs();
    if hessian_ccs.nrow != hessian_ccs.ncol {
        bail!("Ipopt requires a square lower-triangular Hessian pattern");
    }
    for col in 0..hessian_ccs.ncol {
        let start = hessian_ccs.col_ptrs[col];
        let end = hessian_ccs.col_ptrs[col + 1];
        for &row in &hessian_ccs.row_indices[start..end] {
            if row < col {
                bail!("Ipopt Hessian CCS must contain only lower-triangular entries");
            }
        }
    }
    Ok(())
}

struct IpoptProblemAdapter<'a, P, C = NoopIpoptIterationConsumer> {
    problem: &'a P,
    x0: &'a [f64],
    parameters: &'a [ParameterMatrix<'a>],
    constraint_rows: Vec<IpoptIndex>,
    constraint_cols: Vec<IpoptIndex>,
    hessian_rows: Vec<IpoptIndex>,
    hessian_cols: Vec<IpoptIndex>,
    iterations: Index,
    snapshots: Vec<IpoptIterationSnapshot>,
    runtime: RefCell<IpoptRuntimeState>,
    callback: C,
}

impl<'a, P, C> IpoptProblemAdapter<'a, P, C>
where
    P: CompiledNlpProblem,
    C: IpoptIterationConsumer,
{
    fn new(
        problem: &'a P,
        x0: &'a [f64],
        parameters: &'a [ParameterMatrix<'a>],
        callback: C,
    ) -> std::result::Result<Self, IpoptSolveError> {
        let (mut constraint_rows, mut constraint_cols) =
            ccs_triplet_indices(problem.equality_jacobian_ccs(), 0)?;
        let (inequality_rows, inequality_cols) =
            ccs_triplet_indices(problem.inequality_jacobian_ccs(), problem.equality_count())?;
        constraint_rows.extend(inequality_rows);
        constraint_cols.extend(inequality_cols);
        let (hessian_rows, hessian_cols) =
            ccs_triplet_indices(problem.lagrangian_hessian_ccs(), 0)?;
        Ok(Self {
            problem,
            x0,
            parameters,
            constraint_rows,
            constraint_cols,
            hessian_rows,
            hessian_cols,
            iterations: 0,
            snapshots: Vec::new(),
            runtime: RefCell::new(IpoptRuntimeState {
                profiling: IpoptProfiling {
                    backend_timing: problem.backend_timing_metadata(),
                    adapter_timing: problem.adapter_timing_snapshot(),
                    ..IpoptProfiling::default()
                },
                last_checkpoint: IpoptTimingCheckpoint::default(),
            }),
            callback,
        })
    }

    fn record_iteration(&mut self, data: IntermediateCallbackData) -> bool {
        self.iterations = data.iter_count as usize;
        let timing = {
            let mut runtime = self.runtime.borrow_mut();
            runtime.profiling.adapter_timing = self.problem.adapter_timing_snapshot();
            let checkpoint = IpoptTimingCheckpoint::from_profiling(&runtime.profiling);
            let timing = checkpoint.delta_since(runtime.last_checkpoint);
            runtime.last_checkpoint = checkpoint;
            timing
        };
        let snapshot = IpoptIterationSnapshot {
            iteration: data.iter_count as usize,
            phase: data.alg_mod.into(),
            x: data.x.to_vec(),
            internal_slack: data.s.to_vec(),
            equality_multipliers: data.y_c.to_vec(),
            inequality_multipliers: data.y_d.to_vec(),
            lower_bound_multipliers: data.z_l.to_vec(),
            upper_bound_multipliers: data.z_u.to_vec(),
            slack_lower_bound_multipliers: data.v_l.to_vec(),
            slack_upper_bound_multipliers: data.v_u.to_vec(),
            direction_x: data.delta_x.to_vec(),
            direction_slack: data.delta_s.to_vec(),
            direction_equality_multipliers: data.delta_y_c.to_vec(),
            direction_inequality_multipliers: data.delta_y_d.to_vec(),
            direction_lower_bound_multipliers: data.delta_z_l.to_vec(),
            direction_upper_bound_multipliers: data.delta_z_u.to_vec(),
            direction_slack_lower_bound_multipliers: data.delta_v_l.to_vec(),
            direction_slack_upper_bound_multipliers: data.delta_v_u.to_vec(),
            kkt_x_stationarity: data.kkt_x_stationarity.to_vec(),
            kkt_slack_stationarity: data.kkt_slack_stationarity.to_vec(),
            kkt_equality_residual: data.kkt_equality_residual.to_vec(),
            kkt_inequality_residual: data.kkt_inequality_residual.to_vec(),
            kkt_slack_complementarity: data.kkt_slack_complementarity.to_vec(),
            kkt_slack_sigma: data.kkt_slack_sigma.to_vec(),
            kkt_slack_distance: data.kkt_slack_distance.to_vec(),
            curr_grad_f: data.curr_grad_f.to_vec(),
            curr_jac_c_t_y_c: data.curr_jac_c_t_y_c.to_vec(),
            curr_jac_d_t_y_d: data.curr_jac_d_t_y_d.to_vec(),
            curr_grad_lag_x: data.curr_grad_lag_x.to_vec(),
            curr_grad_lag_s: data.curr_grad_lag_s.to_vec(),
            curr_barrier_error: data.curr_barrier_error,
            curr_primal_infeasibility: data.curr_primal_infeasibility,
            curr_dual_infeasibility: data.curr_dual_infeasibility,
            curr_complementarity: data.curr_complementarity,
            curr_nlp_error: data.curr_nlp_error,
            objective: data.obj_value,
            primal_inf: data.inf_pr,
            dual_inf: data.inf_du,
            barrier_parameter: data.mu,
            step_inf: data.d_norm,
            regularization_size: data.regularization_size,
            alpha_pr: data.alpha_pr,
            alpha_du: data.alpha_du,
            line_search_trials: data.ls_trials as usize,
            timing,
        };
        self.callback.accept(&snapshot);
        self.snapshots.push(snapshot);
        true
    }

    fn objective_hessian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        let equality_multipliers = vec![0.0; self.problem.equality_count()];
        let inequality_multipliers = vec![0.0; self.problem.inequality_count()];
        let started = Instant::now();
        self.problem.lagrangian_hessian_values(
            x,
            self.parameters,
            &equality_multipliers,
            &inequality_multipliers,
            vals,
        );
        self.runtime
            .borrow_mut()
            .profiling
            .hessian_values
            .record(started.elapsed());
        true
    }

    fn profiling(&self) -> IpoptProfiling {
        let mut profiling = self.runtime.borrow().profiling.clone();
        profiling.adapter_timing = self.problem.adapter_timing_snapshot();
        profiling
    }
}

impl<P, C> BasicProblem for IpoptProblemAdapter<'_, P, C>
where
    P: CompiledNlpProblem,
    C: IpoptIterationConsumer,
{
    fn num_variables(&self) -> usize {
        self.problem.dimension()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.fill(-IPOPT_INF);
        x_u.fill(IPOPT_INF);
        if let Some(bounds) = self.problem.variable_bounds() {
            if let Some(lower) = bounds.lower {
                for (slot, bound) in x_l.iter_mut().zip(lower.into_iter()) {
                    if let Some(bound) = bound {
                        *slot = bound;
                    }
                }
            }
            if let Some(upper) = bounds.upper {
                for (slot, bound) in x_u.iter_mut().zip(upper.into_iter()) {
                    if let Some(bound) = bound {
                        *slot = bound;
                    }
                }
            }
        }
        true
    }

    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.copy_from_slice(self.x0);
        true
    }

    fn objective(&self, x: &[Number], _new_x: bool, obj: &mut Number) -> bool {
        let started = Instant::now();
        *obj = self.problem.objective_value(x, self.parameters);
        self.runtime
            .borrow_mut()
            .profiling
            .objective_value
            .record(started.elapsed());
        true
    }

    fn objective_grad(&self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        let started = Instant::now();
        self.problem.objective_gradient(x, self.parameters, grad_f);
        self.runtime
            .borrow_mut()
            .profiling
            .objective_gradient
            .record(started.elapsed());
        true
    }
}

impl<P, C> NewtonProblem for IpoptProblemAdapter<'_, P, C>
where
    P: CompiledNlpProblem,
    C: IpoptIterationConsumer,
{
    fn num_hessian_non_zeros(&self) -> usize {
        self.problem.lagrangian_hessian_ccs().nnz()
    }

    fn hessian_indices(&self, rows: &mut [IpoptIndex], cols: &mut [IpoptIndex]) -> bool {
        rows.copy_from_slice(&self.hessian_rows);
        cols.copy_from_slice(&self.hessian_cols);
        true
    }

    fn hessian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        self.objective_hessian_values(x, vals)
    }
}

impl<P, C> ConstrainedProblem for IpoptProblemAdapter<'_, P, C>
where
    P: CompiledNlpProblem,
    C: IpoptIterationConsumer,
{
    fn num_constraints(&self) -> usize {
        self.problem.equality_count() + self.problem.inequality_count()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        self.constraint_rows.len()
    }

    fn constraint(&self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let started = Instant::now();
        let equality_count = self.problem.equality_count();
        let (equality_out, inequality_out) = g.split_at_mut(equality_count);
        self.problem
            .equality_values(x, self.parameters, equality_out);
        self.problem
            .inequality_values(x, self.parameters, inequality_out);
        self.runtime
            .borrow_mut()
            .profiling
            .constraint_values
            .record(started.elapsed());
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let equality_count = self.problem.equality_count();
        let (equality_lower, inequality_lower) = g_l.split_at_mut(equality_count);
        let (equality_upper, inequality_upper) = g_u.split_at_mut(equality_count);
        equality_lower.fill(0.0);
        equality_upper.fill(0.0);
        inequality_lower.fill(-IPOPT_INF);
        inequality_upper.fill(0.0);
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [IpoptIndex],
        cols: &mut [IpoptIndex],
    ) -> bool {
        rows.copy_from_slice(&self.constraint_rows);
        cols.copy_from_slice(&self.constraint_cols);
        true
    }

    fn constraint_jacobian_values(&self, x: &[Number], _new_x: bool, vals: &mut [Number]) -> bool {
        let started = Instant::now();
        let equality_nnz = self.problem.equality_jacobian_ccs().nnz();
        let (equality_vals, inequality_vals) = vals.split_at_mut(equality_nnz);
        self.problem
            .equality_jacobian_values(x, self.parameters, equality_vals);
        self.problem
            .inequality_jacobian_values(x, self.parameters, inequality_vals);
        self.runtime
            .borrow_mut()
            .profiling
            .constraint_jacobian_values
            .record(started.elapsed());
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        self.problem.lagrangian_hessian_ccs().nnz()
    }

    fn hessian_indices(&self, rows: &mut [IpoptIndex], cols: &mut [IpoptIndex]) -> bool {
        rows.copy_from_slice(&self.hessian_rows);
        cols.copy_from_slice(&self.hessian_cols);
        true
    }

    fn hessian_values(
        &self,
        x: &[Number],
        _new_x: bool,
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let started = Instant::now();
        let equality_count = self.problem.equality_count();
        let (equality_multipliers, inequality_multipliers) = lambda.split_at(equality_count);
        let mut objective_hessian = vec![0.0; vals.len()];
        self.problem.lagrangian_hessian_values(
            x,
            self.parameters,
            &vec![0.0; equality_count],
            &vec![0.0; self.problem.inequality_count()],
            &mut objective_hessian,
        );
        self.problem.lagrangian_hessian_values(
            x,
            self.parameters,
            equality_multipliers,
            inequality_multipliers,
            vals,
        );
        for (value, objective_value) in vals.iter_mut().zip(objective_hessian.iter()) {
            *value += (obj_factor - 1.0) * objective_value;
        }
        self.runtime
            .borrow_mut()
            .profiling
            .hessian_values
            .record(started.elapsed());
        true
    }
}

fn set_ipopt_option<'a, P, O>(
    solver: &mut Ipopt<P>,
    name: &str,
    value: O,
) -> std::result::Result<(), IpoptSolveError>
where
    P: BasicProblem,
    O: Into<ipopt::IpoptOption<'a>>,
{
    if solver.set_option(name, value).is_none() {
        return Err(IpoptSolveError::OptionRejected {
            name: name.to_owned(),
        });
    }
    Ok(())
}

fn solve_status_is_success(status: SolveStatus) -> bool {
    matches!(
        status,
        SolveStatus::SolveSucceeded
            | SolveStatus::SolvedToAcceptableLevel
            | SolveStatus::FeasiblePointFound
    )
}

fn open_ipopt_journal<P>(solver: &mut Ipopt<P>, print_level: i32) -> Option<PathBuf>
where
    P: BasicProblem,
{
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ad_codegen_ipopt_{}_{}.log",
        std::process::id(),
        timestamp
    ));
    let path_text = path.to_string_lossy().into_owned();
    solver
        .open_output_file(&path_text, print_level)
        .map(|_| path)
}

fn read_ipopt_journal(path: Option<PathBuf>) -> Option<String> {
    let path = path?;
    let journal = fs::read_to_string(&path).ok();
    let _ = fs::remove_file(&path);
    journal.filter(|text| !text.trim().is_empty())
}

fn command_stdout(mut command: Command) -> Option<String> {
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn find_executable_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|entry| entry.join(name))
        .find(|candidate| candidate.is_file())
}

fn parse_linear_solver_default(options: &str) -> Option<String> {
    options.lines().find_map(|line| {
        let trimmed = line.trim();
        if !trimmed.starts_with("linear_solver") {
            return None;
        }
        let start = trimmed.find("(\"")? + 2;
        let end = trimmed[start..].find("\")")?;
        Some(trimmed[start..start + end].to_string())
    })
}

fn ipopt_linear_solver_default(ipopt_binary: &Path) -> Option<String> {
    command_stdout({
        let mut command = Command::new(ipopt_binary);
        command.arg("--print-options");
        command
    })
    .and_then(|options| parse_linear_solver_default(&options))
}

pub fn capture_ipopt_provenance() -> IpoptProvenance {
    let linked = ipopt::linked_ipopt_provenance();
    let pkg_config_version = command_stdout({
        let mut command = Command::new("pkg-config");
        command.arg("--modversion").arg("ipopt");
        command
    });
    let pkg_config_cflags_libs = command_stdout({
        let mut command = Command::new("pkg-config");
        command.arg("--cflags").arg("--libs").arg("ipopt");
        command
    });
    let ipopt_binary = find_executable_on_path("ipopt");
    let linear_solver_default = ipopt_binary
        .as_deref()
        .and_then(ipopt_linear_solver_default);
    IpoptProvenance {
        linked_solver_stack: linked.link_mode.map(str::to_string),
        linked_ipopt_version: linked.ipopt_version.map(str::to_string),
        linked_ipopt_source_commit: linked.ipopt_source_commit.map(str::to_string),
        linked_spral_version: linked.spral_version.map(str::to_string),
        linked_metis_version: linked.metis_version.map(str::to_string),
        linked_openblas_version: linked.openblas_version.map(str::to_string),
        linked_openblas_threading: linked.openblas_threading.map(str::to_string),
        linked_static_solver_math: linked.static_solver_math,
        linked_ipopt_lib_dir: linked.ipopt_lib_dir.map(str::to_string),
        linked_spral_lib_dir: linked.spral_lib_dir.map(str::to_string),
        linked_metis_lib_dir: linked.metis_lib_dir.map(str::to_string),
        linked_openblas_lib_dir: linked.openblas_lib_dir.map(str::to_string),
        linked_openmp_lib: linked.openmp_lib.map(str::to_string),
        pkg_config_version,
        pkg_config_cflags_libs,
        ipopt_binary: ipopt_binary.map(|path| path.display().to_string()),
        linear_solver_default,
    }
}

fn apply_ipopt_options<P>(
    solver: &mut Ipopt<P>,
    nlp_scaling_method: Option<&str>,
    options: &IpoptOptions,
) -> std::result::Result<(), IpoptSolveError>
where
    P: BasicProblem,
{
    set_ipopt_option(solver, "max_iter", options.max_iters as i32)?;
    set_ipopt_option(solver, "tol", options.tol)?;
    if let Some(value) = options.acceptable_tol {
        set_ipopt_option(solver, "acceptable_tol", value)?;
    }
    if let Some(value) = options.constraint_tol {
        set_ipopt_option(solver, "constr_viol_tol", value)?;
    }
    if let Some(value) = options.complementarity_tol {
        set_ipopt_option(solver, "compl_inf_tol", value)?;
    }
    if let Some(value) = options.dual_tol {
        set_ipopt_option(solver, "dual_inf_tol", value)?;
    }
    if let Some(method) = options
        .nlp_scaling_method
        .map(IpoptNlpScalingMethod::as_str)
        .or(nlp_scaling_method)
    {
        set_ipopt_option(solver, "nlp_scaling_method", method)?;
    }
    set_ipopt_option(solver, "mu_strategy", options.mu_strategy.as_str())?;
    set_ipopt_option(solver, "kappa_d", options.kappa_d)?;
    set_ipopt_option(solver, "print_level", options.print_level)?;
    if let Some(linear_solver) = options.linear_solver {
        set_ipopt_option(solver, "linear_solver", linear_solver.as_str())?;
    }
    if let Some(pivot_method) = options.spral_pivot_method {
        set_ipopt_option(solver, "spral_pivot_method", pivot_method.as_str())?;
    }
    if let Some(ordering) = options.spral_ordering {
        set_ipopt_option(solver, "spral_order", ordering.as_str())?;
    }
    if let Some(scaling) = options.spral_scaling {
        set_ipopt_option(solver, "spral_scaling", scaling.as_str())?;
    }
    if let Some(value) = options.spral_small_pivot_tolerance {
        set_ipopt_option(solver, "spral_small", value)?;
    }
    if let Some(value) = options.spral_threshold_pivot_u {
        set_ipopt_option(solver, "spral_u", value)?;
    }
    if let Some(value) = options.spral_pivot_tolerance_max {
        set_ipopt_option(solver, "spral_umax", value)?;
    }
    if let Some(value) = options.spral_use_gpu {
        set_ipopt_option(solver, "spral_use_gpu", if value { "yes" } else { "no" })?;
    }
    for raw_option in &options.raw_options {
        match &raw_option.value {
            IpoptRawOptionValue::Number(value) => {
                set_ipopt_option(solver, raw_option.name.as_str(), *value)?;
            }
            IpoptRawOptionValue::Integer(value) => {
                set_ipopt_option(solver, raw_option.name.as_str(), *value)?;
            }
            IpoptRawOptionValue::Text(value) => {
                set_ipopt_option(solver, raw_option.name.as_str(), value.as_str())?;
            }
        }
    }
    if options.suppress_banner {
        set_ipopt_option(solver, "sb", "yes")?;
    }
    Ok(())
}

pub fn solve_nlp_ipopt<'a, P>(
    problem: &'a P,
    x0: &'a [f64],
    parameters: &'a [ParameterMatrix<'a>],
    options: &IpoptOptions,
) -> std::result::Result<IpoptSummary, IpoptSolveError>
where
    P: CompiledNlpProblem,
{
    solve_nlp_ipopt_with_callback(problem, x0, parameters, options, |_| {})
}

pub fn solve_nlp_ipopt_with_callback<'a, P, C>(
    problem: &'a P,
    x0: &'a [f64],
    parameters: &'a [ParameterMatrix<'a>],
    options: &IpoptOptions,
    callback: C,
) -> std::result::Result<IpoptSummary, IpoptSolveError>
where
    P: CompiledNlpProblem,
    C: FnMut(&IpoptIterationSnapshot),
{
    validate_ipopt_compatibility(problem)
        .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
    validate_parameter_inputs(problem, parameters)
        .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
    if x0.len() != problem.dimension() {
        return Err(IpoptSolveError::InvalidInput(format!(
            "x0 has length {}, expected {}",
            x0.len(),
            problem.dimension()
        )));
    }
    let solve_started = Instant::now();
    let provenance = options.capture_provenance.then(capture_ipopt_provenance);

    let adapter = IpoptProblemAdapter::new(problem, x0, parameters, callback)?;
    let total_constraint_count = problem.equality_count() + problem.inequality_count();
    if total_constraint_count == 0 {
        let mut solver =
            Ipopt::new_newton(adapter).map_err(|err| IpoptSolveError::Setup(format!("{err:?}")))?;
        apply_ipopt_options(&mut solver, problem.ipopt_nlp_scaling_method(), options)?;
        let journal_path = open_ipopt_journal(&mut solver, options.print_level);
        solver.set_intermediate_callback(Some(IpoptProblemAdapter::<P, C>::record_iteration));
        let solve_result = solver.solve();
        let status = solve_result.status;
        let raw_status = IpoptRawStatus::from(status);
        let objective = solve_result.objective_value;
        let x = solve_result.solver_data.solution.primal_variables.to_vec();
        let lower_bound_multipliers = solve_result
            .solver_data
            .solution
            .lower_bound_multipliers
            .to_vec();
        let upper_bound_multipliers = solve_result
            .solver_data
            .solution
            .upper_bound_multipliers
            .to_vec();
        let iterations = solve_result.solver_data.problem.iterations;
        let snapshots = solve_result.solver_data.problem.snapshots.clone();
        let mut profiling = solve_result.solver_data.problem.profiling();
        profiling.total_time = solve_started.elapsed();
        let journal_output = read_ipopt_journal(journal_path);
        let partial_solution = IpoptPartialSolution {
            x: x.clone(),
            lower_bound_multipliers: lower_bound_multipliers.clone(),
            upper_bound_multipliers: upper_bound_multipliers.clone(),
            equality_multipliers: Vec::new(),
            inequality_multipliers: Vec::new(),
            objective,
            equality_inf_norm: 0.0,
            inequality_inf_norm: 0.0,
            primal_inf_norm: 0.0,
            dual_inf_norm: 0.0,
            complementarity_inf_norm: 0.0,
        };
        if !solve_status_is_success(status) {
            return Err(IpoptSolveError::Solve {
                status: raw_status,
                iterations,
                snapshots,
                partial_solution: Some(Box::new(partial_solution)),
                journal_output,
                profiling: Box::new(profiling),
            });
        }
        return Ok(IpoptSummary {
            x,
            lower_bound_multipliers,
            upper_bound_multipliers,
            equality_multipliers: Vec::new(),
            inequality_multipliers: Vec::new(),
            objective,
            iterations,
            status: raw_status,
            equality_inf_norm: 0.0,
            inequality_inf_norm: 0.0,
            primal_inf_norm: 0.0,
            dual_inf_norm: 0.0,
            complementarity_inf_norm: 0.0,
            snapshots,
            journal_output,
            profiling,
            provenance,
        });
    }

    let mut solver =
        Ipopt::new(adapter).map_err(|err| IpoptSolveError::Setup(format!("{err:?}")))?;
    apply_ipopt_options(&mut solver, problem.ipopt_nlp_scaling_method(), options)?;
    let journal_path = open_ipopt_journal(&mut solver, options.print_level);
    solver.set_intermediate_callback(Some(IpoptProblemAdapter::<P, C>::record_iteration));

    let solve_result = solver.solve();
    let status = solve_result.status;
    let raw_status = IpoptRawStatus::from(status);
    let objective = solve_result.objective_value;
    let x = solve_result.solver_data.solution.primal_variables.to_vec();
    let lower_bound_multipliers = solve_result
        .solver_data
        .solution
        .lower_bound_multipliers
        .to_vec();
    let upper_bound_multipliers = solve_result
        .solver_data
        .solution
        .upper_bound_multipliers
        .to_vec();
    let iterations = solve_result.solver_data.problem.iterations;
    let snapshots = solve_result.solver_data.problem.snapshots.clone();
    let mut profiling = solve_result.solver_data.problem.profiling();
    profiling.total_time = solve_started.elapsed();
    let journal_output = read_ipopt_journal(journal_path);
    let constraint_multipliers = solve_result
        .solver_data
        .solution
        .constraint_multipliers
        .to_vec();
    let equality_count = problem.equality_count();
    let equality_multipliers = constraint_multipliers[..equality_count].to_vec();
    let inequality_multipliers = constraint_multipliers[equality_count..].to_vec();
    let mut gradient = vec![0.0; problem.dimension()];
    problem.objective_gradient(&x, parameters, &mut gradient);
    let mut equality_values = vec![0.0; equality_count];
    let mut inequality_values = vec![0.0; problem.inequality_count()];
    problem.equality_values(&x, parameters, &mut equality_values);
    problem.inequality_values(&x, parameters, &mut inequality_values);
    let mut equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    problem.equality_jacobian_values(&x, parameters, &mut equality_jacobian_values);
    problem.inequality_jacobian_values(&x, parameters, &mut inequality_jacobian_values);
    let equality_jacobian =
        super::ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
    let inequality_jacobian = super::ccs_to_dense(
        problem.inequality_jacobian_ccs(),
        &inequality_jacobian_values,
    );
    let lagrangian_residual = lagrangian_gradient(
        &gradient,
        &equality_jacobian,
        &equality_multipliers,
        &inequality_jacobian,
        &inequality_multipliers,
    );
    let equality_inf_norm = inf_norm(&equality_values);
    let inequality_inf_norm = positive_part_inf_norm(&inequality_values);
    let primal_inf_norm = equality_inf_norm.max(inequality_inf_norm);
    let dual_inf_norm = inf_norm(&lagrangian_residual);
    let complementarity_inf_norm =
        complementarity_inf_norm(&inequality_values, &inequality_multipliers);
    let partial_solution = IpoptPartialSolution {
        x: x.clone(),
        lower_bound_multipliers: lower_bound_multipliers.clone(),
        upper_bound_multipliers: upper_bound_multipliers.clone(),
        equality_multipliers: equality_multipliers.clone(),
        inequality_multipliers: inequality_multipliers.clone(),
        objective,
        equality_inf_norm,
        inequality_inf_norm,
        primal_inf_norm,
        dual_inf_norm,
        complementarity_inf_norm,
    };

    if !solve_status_is_success(status) {
        return Err(IpoptSolveError::Solve {
            status: raw_status,
            iterations,
            snapshots,
            partial_solution: Some(Box::new(partial_solution)),
            journal_output,
            profiling: Box::new(profiling),
        });
    }

    Ok(IpoptSummary {
        x,
        lower_bound_multipliers,
        upper_bound_multipliers,
        equality_multipliers,
        inequality_multipliers,
        objective,
        iterations,
        status: raw_status,
        equality_inf_norm,
        inequality_inf_norm,
        primal_inf_norm,
        dual_inf_norm,
        complementarity_inf_norm,
        snapshots,
        journal_output,
        profiling,
        provenance,
    })
}
