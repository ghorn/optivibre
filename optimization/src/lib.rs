use anyhow::{Result, bail};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{NonnegativeConeT, SecondOrderConeT, ZeroConeT};
use clarabel::solver::implementations::default::DefaultSettingsBuilder;
use clarabel::solver::{DefaultSolver, IPSolver, SolverStatus};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::fs::File;
use std::io::{self, IsTerminal, Read, Write};
#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU8, Ordering},
};
use std::time::{Duration, Instant};
pub use sx_codegen_llvm::{FunctionCompileOptions, LlvmOptimizationLevel};
pub use sx_core::{CallPolicy, CallPolicyConfig, CompileStats, CompileWarning};
use thiserror::Error;

mod filter;
mod interior_point;
#[cfg(feature = "ipopt")]
mod ipopt_backend;
mod symbolic;
mod validation;
mod vectorize;

pub use filter::{FilterAcceptanceMode, FilterEntry, FilterInfo};
pub use filter::{
    FilterAcceptanceMode as SqpFilterAcceptanceMode, FilterEntry as SqpFilterEntry,
    FilterInfo as SqpFilterInfo,
};
pub use interior_point::{
    InteriorPointBoundaryLimiter, InteriorPointDirectionDiagnostics, InteriorPointFailureContext,
    InteriorPointIterationEvent, InteriorPointIterationPhase, InteriorPointIterationSnapshot,
    InteriorPointIterationTiming, InteriorPointLineSearchInfo, InteriorPointLineSearchTrial,
    InteriorPointLinearSolveAttempt, InteriorPointLinearSolveDiagnostics,
    InteriorPointLinearSolveFailureKind, InteriorPointLinearSolver, InteriorPointOptions,
    InteriorPointProfiling, InteriorPointSolveError, InteriorPointStatusKind,
    InteriorPointStepKind, InteriorPointSummary, InteriorPointTermination,
    format_nlip_settings_summary, nlip_event_codes, nlip_event_codes_for_events,
    nlip_event_legend_entries, nlip_event_legend_entries_for_events, solve_nlp_interior_point,
    solve_nlp_interior_point_with_callback,
};
#[cfg(feature = "ipopt")]
pub use ipopt::SolveStatus as IpoptSolveStatus;
#[cfg(feature = "ipopt")]
pub use ipopt_backend::{
    IpoptIterationPhase, IpoptIterationSnapshot, IpoptIterationTiming, IpoptMuStrategy,
    IpoptOptions, IpoptProfiling, IpoptRawStatus, IpoptSolveError, IpoptSummary,
    format_ipopt_settings_summary, solve_nlp_ipopt, solve_nlp_ipopt_with_callback,
};
pub use optimization_derive::Vectorize;
pub use symbolic::{
    ConstraintBounds, RuntimeBoundedJitNlp, RuntimeNlpBounds, SymbolicNlpBuildError,
    SymbolicNlpCompileError, SymbolicNlpCompileOptions, SymbolicNlpOutputs, TypedCompiledJitNlp,
    TypedNlpScaling, TypedRuntimeNlpBounds, TypedSymbolicNlp, symbolic_nlp,
};
pub use validation::{
    FiniteDifferenceValidationOptions, NlpDerivativeValidationReport, ValidationSparsitySummary,
    ValidationSummary, ValidationTolerances, ValidationWorstEntry,
    validate_compiled_nlp_problem_derivatives,
};
pub use vectorize::{
    ScalarLeaf, Vectorize, VectorizeLayoutError, extend_layout_name, flat_view,
    flatten_optional_value, flatten_value, rebind_from_flat, symbolic_column, symbolic_value,
    unflatten_value,
};

pub type Index = usize;
const BOX_LABEL_WIDTH: usize = 13;
pub(crate) const EQ_INF_LABEL: &str = "‖eq‖∞";
pub(crate) const INEQ_INF_LABEL: &str = "‖ineq₊‖∞";
pub(crate) const DUAL_INF_LABEL: &str = "‖∇L‖∞";
pub(crate) const SQP_COMP_INF_LABEL: &str = "‖g∘λ‖∞";
pub(crate) const OVERALL_INF_LABEL: &str = "‖overall‖∞";
pub(crate) const STEP_INF_LABEL: &str = "‖Δx‖∞";
pub(crate) const PRIMAL_INF_LABEL: &str = "max(‖eq‖∞, ‖ineq₊‖∞)";

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintSatisfaction {
    FullAccuracy,
    ReducedAccuracy,
    Violated,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintBoundSide {
    Equality,
    Lower,
    Upper,
    Both,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NlpInequalitySource {
    ConstraintRow { row: Index },
    VariableBound { index: Index },
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct NlpEqualityViolation {
    pub row: Index,
    pub value: f64,
    pub abs_violation: f64,
    pub satisfaction: ConstraintSatisfaction,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct NlpInequalityViolation {
    pub source: NlpInequalitySource,
    pub value: f64,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub lower_violation: f64,
    pub upper_violation: f64,
    pub worst_violation: f64,
    pub bound_side: ConstraintBoundSide,
    pub satisfaction: ConstraintSatisfaction,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NlpConstraintViolationReport {
    pub equalities: Vec<NlpEqualityViolation>,
    pub inequalities: Vec<NlpInequalityViolation>,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnsiColorMode {
    Auto = 0,
    Always = 1,
    Never = 2,
}

impl AnsiColorMode {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Always,
            2 => Self::Never,
            _ => Self::Auto,
        }
    }
}

static ANSI_COLOR_MODE: AtomicU8 = AtomicU8::new(AnsiColorMode::Auto as u8);

pub fn ansi_color_mode() -> AnsiColorMode {
    AnsiColorMode::from_u8(ANSI_COLOR_MODE.load(Ordering::Relaxed))
}

pub fn set_ansi_color_mode(mode: AnsiColorMode) -> AnsiColorMode {
    AnsiColorMode::from_u8(ANSI_COLOR_MODE.swap(mode as u8, Ordering::Relaxed))
}

#[cfg(feature = "serde")]
mod duration_seconds_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = f64::deserialize(deserializer)?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(serde::de::Error::custom(format!(
                "expected non-negative finite seconds, got {seconds}"
            )));
        }
        Ok(Duration::from_secs_f64(seconds))
    }
}

#[cfg(feature = "serde")]
mod option_duration_seconds_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration
            .map(|value: Duration| value.as_secs_f64())
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = Option::<f64>::deserialize(deserializer)?;
        seconds
            .map(|seconds| {
                if !seconds.is_finite() || seconds < 0.0 {
                    Err(serde::de::Error::custom(format!(
                        "expected non-negative finite seconds, got {seconds}"
                    )))
                } else {
                    Ok(Duration::from_secs_f64(seconds))
                }
            })
            .transpose()
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendTimingMetadata {
    #[cfg_attr(feature = "serde", serde(with = "option_duration_seconds_serde"))]
    pub function_creation_time: Option<Duration>,
    #[cfg_attr(feature = "serde", serde(with = "option_duration_seconds_serde"))]
    pub derivative_generation_time: Option<Duration>,
    #[cfg_attr(feature = "serde", serde(with = "option_duration_seconds_serde"))]
    pub jit_time: Option<Duration>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NlpCompileStats {
    pub variable_count: Index,
    pub parameter_scalar_count: Index,
    pub equality_count: Index,
    pub inequality_count: Index,
    pub objective_gradient_nnz: Index,
    pub equality_jacobian_nnz: Index,
    pub inequality_jacobian_nnz: Index,
    pub hessian_nnz: Index,
    pub jit_kernel_count: Index,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicCompileMetadata {
    pub timing: BackendTimingMetadata,
    pub setup_profile: SymbolicSetupProfile,
    pub stats: NlpCompileStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolicCompileStage {
    BuildProblem,
    ObjectiveGradient,
    EqualityJacobian,
    InequalityJacobian,
    LagrangianAssembly,
    HessianGeneration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicCompileStageProgress {
    pub stage: SymbolicCompileStage,
    pub metadata: SymbolicCompileMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicCompileProgress {
    Stage(SymbolicCompileStageProgress),
    Ready(SymbolicCompileMetadata),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicSetupProfile {
    pub symbolic_construction: Option<Duration>,
    pub objective_gradient: Option<Duration>,
    pub equality_jacobian: Option<Duration>,
    pub inequality_jacobian: Option<Duration>,
    pub lagrangian_assembly: Option<Duration>,
    pub hessian_generation: Option<Duration>,
    pub lowering: Option<Duration>,
    pub llvm_jit: Option<Duration>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackendCompileReport {
    pub timing: BackendTimingMetadata,
    pub setup_profile: SymbolicSetupProfile,
    pub stats: CompileStats,
    pub warnings: Vec<CompileWarning>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalTimingStat {
    pub calls: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub total_time: Duration,
}

impl EvalTimingStat {
    fn record(&mut self, elapsed: Duration) {
        self.calls += 1;
        self.total_time += elapsed;
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SqpAdapterTiming {
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub callback_evaluation: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub output_marshalling: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub layout_projection: Duration,
}

pub type SolverAdapterTiming = SqpAdapterTiming;

impl SqpAdapterTiming {
    pub(crate) fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            callback_evaluation: self
                .callback_evaluation
                .saturating_sub(baseline.callback_evaluation),
            output_marshalling: self
                .output_marshalling
                .saturating_sub(baseline.output_marshalling),
            layout_projection: self
                .layout_projection
                .saturating_sub(baseline.layout_projection),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClarabelSqpProfiling {
    pub objective_value: EvalTimingStat,
    pub objective_gradient: EvalTimingStat,
    pub equality_values: EvalTimingStat,
    pub inequality_values: EvalTimingStat,
    pub equality_jacobian_values: EvalTimingStat,
    pub inequality_jacobian_values: EvalTimingStat,
    pub lagrangian_hessian_values: EvalTimingStat,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub evaluation_time: Duration,
    pub jacobian_assembly_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub jacobian_assembly_time: Duration,
    pub hessian_assembly_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub hessian_assembly_time: Duration,
    pub regularization_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub regularization_time: Duration,
    pub subproblem_assembly_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub subproblem_assembly_time: Duration,
    pub preprocessing_other_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub preprocessing_other_time: Duration,
    pub qp_setups: Index,
    pub qp_setup_time: Duration,
    pub qp_solves: Index,
    pub qp_solve_time: Duration,
    pub multiplier_estimations: Index,
    pub multiplier_estimation_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub subproblem_solve_time: Duration,
    pub line_search_evaluations: Index,
    pub line_search_evaluation_time: Duration,
    pub line_search_condition_checks: Index,
    pub line_search_condition_check_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub line_search_time: Duration,
    pub convergence_checks: Index,
    pub convergence_check_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub convergence_time: Duration,
    pub elastic_recovery_activations: Index,
    pub elastic_recovery_qp_solves: Index,
    pub adapter_timing: Option<SqpAdapterTiming>,
    pub preprocessing_steps: Index,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub preprocessing_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub total_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub unaccounted_time: Duration,
    pub backend_timing: BackendTimingMetadata,
}

impl ClarabelSqpProfiling {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CCS {
    pub nrow: Index,
    pub ncol: Index,
    pub col_ptrs: Vec<Index>,
    pub row_indices: Vec<Index>,
}

impl CCS {
    pub fn new(nrow: Index, ncol: Index, col_ptrs: Vec<Index>, row_indices: Vec<Index>) -> Self {
        Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        }
    }

    pub fn empty(nrow: Index, ncol: Index) -> Self {
        Self {
            nrow,
            ncol,
            col_ptrs: vec![0; ncol + 1],
            row_indices: Vec::new(),
        }
    }

    pub fn dense(nrow: Index, ncol: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(ncol + 1);
        let mut row_indices = Vec::with_capacity(nrow * ncol);
        col_ptrs.push(0);
        for _ in 0..ncol {
            row_indices.extend(0..nrow);
            col_ptrs.push(row_indices.len());
        }
        Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        }
    }

    pub fn nnz(&self) -> Index {
        self.row_indices.len()
    }

    pub fn lower_triangular_dense(size: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(size + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col in 0..size {
            row_indices.extend(col..size);
            col_ptrs.push(row_indices.len());
        }
        Self {
            nrow: size,
            ncol: size,
            col_ptrs,
            row_indices,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParameterMatrix<'a> {
    pub ccs: &'a CCS,
    pub values: &'a [f64],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NlpEvaluationBenchmarkOptions {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
}

impl Default for NlpEvaluationBenchmarkOptions {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measured_iterations: 100,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NlpEvaluationKernelKind {
    ObjectiveValue,
    ObjectiveGradient,
    EqualityJacobianValues,
    InequalityJacobianValues,
    LagrangianHessianValues,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KernelOutputSummary {
    pub finite: bool,
    pub nonzero_count: usize,
    pub max_abs: f64,
}

impl KernelOutputSummary {
    pub fn is_all_zero(&self) -> bool {
        self.nonzero_count == 0
    }
}

fn summarize_output(values: &[f64]) -> KernelOutputSummary {
    let finite = values.iter().all(|value| value.is_finite());
    let nonzero_count = values.iter().filter(|&&value| value != 0.0).count();
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    KernelOutputSummary {
        finite,
        nonzero_count,
        max_abs,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KernelBenchmarkStats {
    pub output_len: usize,
    pub iterations: usize,
    pub total_time: Duration,
    pub min_time: Option<Duration>,
    pub max_time: Option<Duration>,
    pub stddev_seconds_accumulator: f64,
    pub preflight_output: KernelOutputSummary,
}

impl KernelBenchmarkStats {
    pub fn average_time(&self) -> Option<Duration> {
        (self.iterations > 0).then(|| self.total_time.div_f64(self.iterations as f64))
    }

    pub fn stddev_time(&self) -> Option<Duration> {
        if self.iterations == 0 {
            return None;
        }
        let mean_seconds = self.total_time.as_secs_f64() / self.iterations as f64;
        let variance_seconds = (self.stddev_seconds_accumulator / self.iterations as f64)
            - mean_seconds * mean_seconds;
        Some(Duration::from_secs_f64(variance_seconds.max(0.0).sqrt()))
    }

    fn record_sample(&mut self, elapsed: Duration) {
        self.iterations += 1;
        self.total_time += elapsed;
        let elapsed_seconds = elapsed.as_secs_f64();
        self.stddev_seconds_accumulator += elapsed_seconds * elapsed_seconds;
        self.min_time = Some(match self.min_time {
            Some(current) => current.min(elapsed),
            None => elapsed,
        });
        self.max_time = Some(match self.max_time {
            Some(current) => current.max(elapsed),
            None => elapsed,
        });
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NlpBenchmarkPointSummary {
    pub decision_inf_norm: f64,
    pub parameter_inf_norm: f64,
    pub objective_value: f64,
    pub objective_finite: bool,
    pub equality_inf_norm: Option<f64>,
    pub inequality_inf_norm: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NlpEvaluationBenchmark {
    pub benchmark_point: NlpBenchmarkPointSummary,
    pub objective_value: KernelBenchmarkStats,
    pub objective_gradient: KernelBenchmarkStats,
    pub equality_jacobian_values: Option<KernelBenchmarkStats>,
    pub inequality_jacobian_values: Option<KernelBenchmarkStats>,
    pub lagrangian_hessian_values: KernelBenchmarkStats,
}

pub trait CompiledNlpProblem {
    fn dimension(&self) -> Index;
    fn parameter_count(&self) -> Index;
    fn parameter_ccs(&self, parameter_index: Index) -> &CCS;
    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        None
    }
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        BackendTimingMetadata::default()
    }
    fn backend_compile_report(&self) -> Option<&BackendCompileReport> {
        None
    }
    fn adapter_timing_snapshot(&self) -> Option<SolverAdapterTiming> {
        self.sqp_adapter_timing_snapshot()
    }
    fn ipopt_nlp_scaling_method(&self) -> Option<&'static str> {
        None
    }
    fn sqp_adapter_timing_snapshot(&self) -> Option<SqpAdapterTiming> {
        None
    }
    fn equality_count(&self) -> Index;
    fn inequality_count(&self) -> Index;
    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64;
    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]);
    fn equality_jacobian_ccs(&self) -> &CCS;
    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]);
    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    );
    fn inequality_jacobian_ccs(&self) -> &CCS;
    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]);
    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    );
    fn lagrangian_hessian_ccs(&self) -> &CCS;
    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    );
}

fn benchmark_kernel(
    output_len: usize,
    preflight_output: KernelOutputSummary,
    options: NlpEvaluationBenchmarkOptions,
    mut eval: impl FnMut(),
) -> KernelBenchmarkStats {
    for _ in 0..options.warmup_iterations {
        eval();
    }
    let mut stats = KernelBenchmarkStats {
        output_len,
        preflight_output,
        ..KernelBenchmarkStats::default()
    };
    for _ in 0..options.measured_iterations {
        let started = Instant::now();
        eval();
        stats.record_sample(started.elapsed());
    }
    stats
}

pub fn benchmark_compiled_nlp_problem(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: NlpEvaluationBenchmarkOptions,
) -> NlpEvaluationBenchmark {
    benchmark_compiled_nlp_problem_with_progress(problem, x, parameters, options, |_| {})
}

pub fn benchmark_compiled_nlp_problem_with_progress(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: NlpEvaluationBenchmarkOptions,
    mut on_progress: impl FnMut(NlpEvaluationKernelKind),
) -> NlpEvaluationBenchmark {
    if options.warmup_iterations == 0 && options.measured_iterations == 0 {
        return NlpEvaluationBenchmark {
            benchmark_point: NlpBenchmarkPointSummary {
                decision_inf_norm: x.iter().map(|value| value.abs()).fold(0.0_f64, f64::max),
                parameter_inf_norm: parameters
                    .iter()
                    .flat_map(|parameter| parameter.values.iter())
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max),
                objective_value: f64::NAN,
                objective_finite: false,
                equality_inf_norm: None,
                inequality_inf_norm: None,
            },
            objective_value: KernelBenchmarkStats {
                output_len: 1,
                ..KernelBenchmarkStats::default()
            },
            objective_gradient: KernelBenchmarkStats {
                output_len: problem.dimension(),
                ..KernelBenchmarkStats::default()
            },
            equality_jacobian_values: (problem.equality_count() > 0).then(|| {
                KernelBenchmarkStats {
                    output_len: problem.equality_jacobian_ccs().nnz(),
                    ..KernelBenchmarkStats::default()
                }
            }),
            inequality_jacobian_values: (problem.inequality_count() > 0).then(|| {
                KernelBenchmarkStats {
                    output_len: problem.inequality_jacobian_ccs().nnz(),
                    ..KernelBenchmarkStats::default()
                }
            }),
            lagrangian_hessian_values: KernelBenchmarkStats {
                output_len: problem.lagrangian_hessian_ccs().nnz(),
                ..KernelBenchmarkStats::default()
            },
        };
    }

    let mut gradient = vec![0.0; problem.dimension()];
    let equality_value_len = problem.equality_count();
    let mut equality_values = vec![0.0; equality_value_len];
    let inequality_value_len = problem.inequality_count();
    let mut inequality_values = vec![0.0; inequality_value_len];
    let equality_jac_nnz = problem.equality_jacobian_ccs().nnz();
    let mut equality_jacobian = vec![0.0; equality_jac_nnz];
    let inequality_jac_nnz = problem.inequality_jacobian_ccs().nnz();
    let mut inequality_jacobian = vec![0.0; inequality_jac_nnz];
    let mut hessian = vec![0.0; problem.lagrangian_hessian_ccs().nnz()];
    let equality_multipliers = vec![0.0; problem.equality_count()];
    let inequality_multipliers = vec![0.0; problem.inequality_count()];

    let objective_value_preflight = problem.objective_value(x, parameters);
    problem.objective_gradient(x, parameters, &mut gradient);
    if equality_value_len > 0 {
        problem.equality_values(x, parameters, &mut equality_values);
    }
    if inequality_value_len > 0 {
        problem.inequality_values(x, parameters, &mut inequality_values);
    }
    if equality_jac_nnz > 0 {
        problem.equality_jacobian_values(x, parameters, &mut equality_jacobian);
    }
    if inequality_jac_nnz > 0 {
        problem.inequality_jacobian_values(x, parameters, &mut inequality_jacobian);
    }
    problem.lagrangian_hessian_values(
        x,
        parameters,
        &equality_multipliers,
        &inequality_multipliers,
        &mut hessian,
    );
    let benchmark_point = NlpBenchmarkPointSummary {
        decision_inf_norm: x.iter().map(|value| value.abs()).fold(0.0_f64, f64::max),
        parameter_inf_norm: parameters
            .iter()
            .flat_map(|parameter| parameter.values.iter())
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max),
        objective_value: objective_value_preflight,
        objective_finite: objective_value_preflight.is_finite(),
        equality_inf_norm: (!equality_values.is_empty()).then(|| {
            equality_values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
        }),
        inequality_inf_norm: (!inequality_values.is_empty()).then(|| {
            inequality_values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
        }),
    };
    let objective_preflight = summarize_output(&[objective_value_preflight]);
    let gradient_preflight = summarize_output(&gradient);
    let equality_jacobian_preflight = summarize_output(&equality_jacobian);
    let inequality_jacobian_preflight = summarize_output(&inequality_jacobian);
    let hessian_preflight = summarize_output(&hessian);

    on_progress(NlpEvaluationKernelKind::ObjectiveValue);
    let objective_value = benchmark_kernel(1, objective_preflight, options, || {
        std::hint::black_box(problem.objective_value(x, parameters));
    });
    on_progress(NlpEvaluationKernelKind::ObjectiveGradient);
    let objective_gradient = benchmark_kernel(gradient.len(), gradient_preflight, options, || {
        problem.objective_gradient(x, parameters, &mut gradient);
        std::hint::black_box(&gradient);
    });
    let equality_jacobian_values = (equality_jac_nnz > 0).then(|| {
        on_progress(NlpEvaluationKernelKind::EqualityJacobianValues);
        benchmark_kernel(
            equality_jacobian.len(),
            equality_jacobian_preflight,
            options,
            || {
                problem.equality_jacobian_values(x, parameters, &mut equality_jacobian);
                std::hint::black_box(&equality_jacobian);
            },
        )
    });
    let inequality_jacobian_values = (inequality_jac_nnz > 0).then(|| {
        on_progress(NlpEvaluationKernelKind::InequalityJacobianValues);
        benchmark_kernel(
            inequality_jacobian.len(),
            inequality_jacobian_preflight,
            options,
            || {
                problem.inequality_jacobian_values(x, parameters, &mut inequality_jacobian);
                std::hint::black_box(&inequality_jacobian);
            },
        )
    });
    on_progress(NlpEvaluationKernelKind::LagrangianHessianValues);
    let lagrangian_hessian_values =
        benchmark_kernel(hessian.len(), hessian_preflight, options, || {
            problem.lagrangian_hessian_values(
                x,
                parameters,
                &equality_multipliers,
                &inequality_multipliers,
                &mut hessian,
            );
            std::hint::black_box(&hessian);
        });

    NlpEvaluationBenchmark {
        benchmark_point,
        objective_value,
        objective_gradient,
        equality_jacobian_values,
        inequality_jacobian_values,
        lagrangian_hessian_values,
    }
}

#[derive(Clone, Debug)]
pub struct ClarabelSqpOptions {
    pub max_iters: Index,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub complementarity_tol: f64,
    pub overall_tol: f64,
    pub overall_scale_max: f64,
    pub merit_penalty: f64,
    pub hessian_regularization_enabled: bool,
    pub regularization: f64,
    pub globalization: SqpGlobalization,
    pub second_order_correction: bool,
    pub restoration_phase: bool,
    pub elastic_mode: bool,
    pub elastic_weight: f64,
    pub elastic_primal_regularization: f64,
    pub elastic_slack_regularization: f64,
    pub elastic_restore_reduction_factor: f64,
    pub elastic_restore_abs_tol: f64,
    pub elastic_restore_max_iters: Index,
    pub verbose: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpGlobalizationKind {
    LineSearchMerit,
    LineSearchFilter,
    TrustRegionMerit,
    TrustRegionFilter,
}

impl SqpGlobalizationKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::LineSearchMerit => "ls_merit",
            Self::LineSearchFilter => "ls_filter",
            Self::TrustRegionMerit => "tr_merit",
            Self::TrustRegionFilter => "tr_filter",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqpLineSearchOptions {
    pub armijo_c1: f64,
    pub wolfe_c2: Option<f64>,
    pub beta: f64,
    pub max_steps: Index,
    pub min_step: f64,
}

impl SqpGlobalization {
    pub const fn kind(&self) -> SqpGlobalizationKind {
        match self {
            Self::LineSearchMerit(_) => SqpGlobalizationKind::LineSearchMerit,
            Self::LineSearchFilter(_) => SqpGlobalizationKind::LineSearchFilter,
            Self::TrustRegionMerit(_) => SqpGlobalizationKind::TrustRegionMerit,
            Self::TrustRegionFilter(_) => SqpGlobalizationKind::TrustRegionFilter,
        }
    }
}

pub fn format_sqp_settings_summary(options: &ClarabelSqpOptions) -> String {
    let globalization = match &options.globalization {
        SqpGlobalization::LineSearchMerit(globalization) => format!(
            "globalization={} armijo={} wolfe={} beta={} max_ls={} min_step={} penalty={}x{}",
            options.globalization.kind().label(),
            sci_text(globalization.line_search.armijo_c1),
            globalization
                .line_search
                .wolfe_c2
                .map(sci_text)
                .unwrap_or_else(|| "off".to_string()),
            sci_text(globalization.line_search.beta),
            globalization.line_search.max_steps,
            sci_text(globalization.line_search.min_step),
            sci_text(globalization.exact_merit_penalty),
            globalization.penalty_increase_factor,
        ),
        SqpGlobalization::LineSearchFilter(globalization) => format!(
            "globalization={} armijo={} wolfe={} beta={} max_ls={} min_step={} filter_gamma=({}, {}) switching=({}, {}, {}) penalty={}",
            options.globalization.kind().label(),
            sci_text(globalization.line_search.armijo_c1),
            globalization
                .line_search
                .wolfe_c2
                .map(sci_text)
                .unwrap_or_else(|| "off".to_string()),
            sci_text(globalization.line_search.beta),
            globalization.line_search.max_steps,
            sci_text(globalization.line_search.min_step),
            sci_text(globalization.filter.gamma_objective),
            sci_text(globalization.filter.gamma_violation),
            sci_text(globalization.filter.switching_reference_min),
            sci_text(globalization.filter.switching_violation_factor),
            sci_text(globalization.filter.switching_linearized_reduction_factor),
            sci_text(globalization.exact_merit_penalty),
        ),
        SqpGlobalization::TrustRegionMerit(globalization) => format!(
            "globalization={} radius=({}->{}, min={}) shrink={} grow={} rho=({},{}) boundary={} max_contract={} penalty={} fixed={}",
            options.globalization.kind().label(),
            sci_text(globalization.trust_region.initial_radius),
            sci_text(globalization.trust_region.max_radius),
            sci_text(globalization.trust_region.min_radius),
            sci_text(globalization.trust_region.shrink_factor),
            sci_text(globalization.trust_region.grow_factor),
            sci_text(globalization.trust_region.accept_ratio),
            sci_text(globalization.trust_region.expand_ratio),
            sci_text(globalization.trust_region.boundary_fraction),
            globalization.trust_region.max_radius_contractions,
            sci_text(globalization.exact_merit_penalty),
            if globalization.fixed_penalty {
                "on"
            } else {
                "off"
            },
        ),
        SqpGlobalization::TrustRegionFilter(globalization) => format!(
            "globalization={} radius=({}->{}, min={}) shrink={} grow={} rho=({},{}) boundary={} max_contract={} filter_gamma=({}, {}) switching=({}, {}, {}) penalty={}",
            options.globalization.kind().label(),
            sci_text(globalization.trust_region.initial_radius),
            sci_text(globalization.trust_region.max_radius),
            sci_text(globalization.trust_region.min_radius),
            sci_text(globalization.trust_region.shrink_factor),
            sci_text(globalization.trust_region.grow_factor),
            sci_text(globalization.trust_region.accept_ratio),
            sci_text(globalization.trust_region.expand_ratio),
            sci_text(globalization.trust_region.boundary_fraction),
            globalization.trust_region.max_radius_contractions,
            sci_text(globalization.filter.gamma_objective),
            sci_text(globalization.filter.gamma_violation),
            sci_text(globalization.filter.switching_reference_min),
            sci_text(globalization.filter.switching_violation_factor),
            sci_text(globalization.filter.switching_linearized_reduction_factor),
            sci_text(globalization.exact_merit_penalty),
        ),
    };
    format!(
        "{globalization}; hessian_regularization={}; soc={}; restoration={}; elastic={}",
        if options.hessian_regularization_enabled {
            format!("on({})", sci_text(options.regularization))
        } else {
            "off".to_string()
        },
        if options.second_order_correction {
            "on"
        } else {
            "off"
        },
        if options.restoration_phase {
            "on"
        } else {
            "off"
        },
        if options.elastic_mode { "on" } else { "off" },
    )
}

impl Default for SqpLineSearchOptions {
    fn default() -> Self {
        Self {
            armijo_c1: 1e-4,
            wolfe_c2: None,
            beta: 0.5,
            max_steps: 32,
            min_step: 1e-8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqpFilterOptions {
    pub armijo_c1: f64,
    pub gamma_objective: f64,
    pub gamma_violation: f64,
    pub theta_max_factor: f64,
    pub switching_reference_min: f64,
    pub switching_violation_factor: f64,
    pub switching_linearized_reduction_factor: f64,
}

impl Default for SqpFilterOptions {
    fn default() -> Self {
        Self {
            armijo_c1: 1e-4,
            gamma_objective: 1e-4,
            gamma_violation: 1e-4,
            theta_max_factor: 1e3,
            switching_reference_min: 1e-3,
            switching_violation_factor: 10.0,
            switching_linearized_reduction_factor: 0.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqpTrustRegionOptions {
    pub initial_radius: f64,
    pub max_radius: f64,
    pub min_radius: f64,
    pub shrink_factor: f64,
    pub grow_factor: f64,
    pub accept_ratio: f64,
    pub expand_ratio: f64,
    pub boundary_fraction: f64,
    pub max_radius_contractions: Index,
}

impl Default for SqpTrustRegionOptions {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 100.0,
            min_radius: 1e-8,
            shrink_factor: 0.25,
            grow_factor: 2.0,
            accept_ratio: 0.1,
            expand_ratio: 0.75,
            boundary_fraction: 0.8,
            max_radius_contractions: 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineSearchMeritOptions {
    pub line_search: SqpLineSearchOptions,
    pub exact_merit_penalty: f64,
    pub penalty_increase_factor: f64,
    pub max_penalty_updates: Index,
}

impl Default for LineSearchMeritOptions {
    fn default() -> Self {
        Self {
            line_search: SqpLineSearchOptions::default(),
            exact_merit_penalty: 10.0,
            penalty_increase_factor: 10.0,
            max_penalty_updates: 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineSearchFilterOptions {
    pub line_search: SqpLineSearchOptions,
    pub filter: SqpFilterOptions,
    pub exact_merit_penalty: f64,
}

impl Default for LineSearchFilterOptions {
    fn default() -> Self {
        Self {
            line_search: SqpLineSearchOptions::default(),
            filter: SqpFilterOptions::default(),
            exact_merit_penalty: 10.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrustRegionMeritOptions {
    pub trust_region: SqpTrustRegionOptions,
    pub exact_merit_penalty: f64,
    pub fixed_penalty: bool,
}

impl Default for TrustRegionMeritOptions {
    fn default() -> Self {
        Self {
            trust_region: SqpTrustRegionOptions::default(),
            exact_merit_penalty: 10.0,
            fixed_penalty: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrustRegionFilterOptions {
    pub trust_region: SqpTrustRegionOptions,
    pub filter: SqpFilterOptions,
    pub exact_merit_penalty: f64,
}

impl Default for TrustRegionFilterOptions {
    fn default() -> Self {
        Self {
            trust_region: SqpTrustRegionOptions::default(),
            filter: SqpFilterOptions::default(),
            exact_merit_penalty: 10.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SqpGlobalization {
    LineSearchMerit(LineSearchMeritOptions),
    LineSearchFilter(LineSearchFilterOptions),
    TrustRegionMerit(TrustRegionMeritOptions),
    TrustRegionFilter(TrustRegionFilterOptions),
}

impl Default for SqpGlobalization {
    fn default() -> Self {
        Self::TrustRegionFilter(TrustRegionFilterOptions::default())
    }
}

impl Default for ClarabelSqpOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            dual_tol: 1e-6,
            constraint_tol: 1e-6,
            complementarity_tol: 1e-6,
            overall_tol: 1e-6,
            overall_scale_max: 100.0,
            merit_penalty: 10.0,
            hessian_regularization_enabled: false,
            regularization: 1e-6,
            globalization: SqpGlobalization::default(),
            second_order_correction: true,
            restoration_phase: true,
            elastic_mode: true,
            elastic_weight: 100.0,
            elastic_primal_regularization: 1.0,
            elastic_slack_regularization: 1e-8,
            elastic_restore_reduction_factor: 1e-2,
            elastic_restore_abs_tol: 1e-4,
            elastic_restore_max_iters: 5,
            verbose: true,
        }
    }
}

fn sqp_globalization_exact_merit_penalty(options: &ClarabelSqpOptions) -> f64 {
    match &options.globalization {
        SqpGlobalization::LineSearchMerit(globalization) => globalization.exact_merit_penalty,
        SqpGlobalization::LineSearchFilter(globalization) => globalization.exact_merit_penalty,
        SqpGlobalization::TrustRegionMerit(globalization) => globalization.exact_merit_penalty,
        SqpGlobalization::TrustRegionFilter(globalization) => globalization.exact_merit_penalty,
    }
}

fn sqp_effective_exact_merit_penalty(options: &ClarabelSqpOptions) -> f64 {
    let globalization_penalty = sqp_globalization_exact_merit_penalty(options);
    if (options.merit_penalty - ClarabelSqpOptions::default().merit_penalty).abs() > f64::EPSILON
        && (globalization_penalty - 10.0).abs() <= f64::EPSILON
    {
        options.merit_penalty
    } else {
        globalization_penalty
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpIterationPhase {
    Initial,
    AcceptedStep,
    PostConvergence,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpIterationEvent {
    PenaltyUpdated,
    HessianShifted,
    LongLineSearch,
    ArmijoToleranceAdjusted,
    SecondOrderCorrectionUsed,
    FilterAccepted,
    RestorationStepAccepted,
    QpReducedAccuracy,
    ElasticRecoveryUsed,
    WolfeRejectedTrial,
    MaxIterationsReached,
}

pub fn sqp_event_legend_entries(snapshot: &SqpIterationSnapshot) -> Vec<(char, &'static str)> {
    let mut entries = Vec::new();
    if snapshot.events.contains(&SqpIterationEvent::PenaltyUpdated) {
        entries.push(('P', "P=merit penalty increased"));
    }
    if snapshot.events.contains(&SqpIterationEvent::HessianShifted) {
        entries.push(('H', "H=Hessian shifted beyond baseline regularization"));
    }
    if snapshot.events.contains(&SqpIterationEvent::LongLineSearch) {
        entries.push(('L', "L=line search backtracked >=4 times"));
    }
    if snapshot
        .events
        .contains(&SqpIterationEvent::ArmijoToleranceAdjusted)
    {
        entries.push(('A', "A=Armijo accepted using numerical tolerance slack"));
    }
    if snapshot
        .events
        .contains(&SqpIterationEvent::SecondOrderCorrectionUsed)
    {
        entries.push(('S', "S=accepted full step after second-order correction"));
    } else if snapshot
        .line_search
        .as_ref()
        .is_some_and(|info| info.second_order_correction_attempted)
    {
        entries.push(('s', "s=second-order correction attempted but not accepted"));
    }
    if snapshot.events.contains(&SqpIterationEvent::FilterAccepted) {
        entries.push((
            'F',
            "F=filter accepted a feasibility-improving step without objective Armijo",
        ));
    }
    if snapshot
        .events
        .contains(&SqpIterationEvent::QpReducedAccuracy)
    {
        entries.push(('R', "R=QP solved to reduced accuracy"));
    }
    if snapshot
        .events
        .contains(&SqpIterationEvent::WolfeRejectedTrial)
    {
        entries.push(('W', "W=rejected trial failed Wolfe curvature condition"));
    }
    if snapshot
        .events
        .contains(&SqpIterationEvent::ElasticRecoveryUsed)
    {
        entries.push((
            'E',
            "E=elastic recovery QP used after primal-infeasible linearization",
        ));
    } else if snapshot
        .line_search
        .as_ref()
        .is_some_and(|info| info.elastic_recovery_attempted)
    {
        entries.push(('e', "e=elastic recovery QP attempted but not accepted"));
    }
    if snapshot
        .events
        .contains(&SqpIterationEvent::MaxIterationsReached)
    {
        entries.push(('M', "M=maximum SQP iterations reached"));
    }
    entries
}

pub fn sqp_iteration_label(snapshot: &SqpIterationSnapshot) -> String {
    match snapshot.phase {
        SqpIterationPhase::Initial => "pre".to_string(),
        SqpIterationPhase::AcceptedStep => {
            if accepted_step_kind(
                snapshot.line_search.as_ref(),
                snapshot.trust_region.as_ref(),
            ) == Some(SqpStepKind::Restoration)
            {
                format!("{}r", snapshot.iteration)
            } else {
                snapshot.iteration.to_string()
            }
        }
        SqpIterationPhase::PostConvergence => "post".to_string(),
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpQpStatus {
    Solved,
    ReducedAccuracy,
    Failed,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SqpQpRawStatus {
    Unsolved,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    AlmostSolved,
    AlmostPrimalInfeasible,
    AlmostDualInfeasible,
    MaxIterations,
    MaxTime,
    NumericalError,
    InsufficientProgress,
    CallbackTerminated,
    Other(String),
}

impl From<SolverStatus> for SqpQpRawStatus {
    fn from(status: SolverStatus) -> Self {
        match status {
            SolverStatus::Unsolved => Self::Unsolved,
            SolverStatus::Solved => Self::Solved,
            SolverStatus::PrimalInfeasible => Self::PrimalInfeasible,
            SolverStatus::DualInfeasible => Self::DualInfeasible,
            SolverStatus::AlmostSolved => Self::AlmostSolved,
            SolverStatus::AlmostPrimalInfeasible => Self::AlmostPrimalInfeasible,
            SolverStatus::AlmostDualInfeasible => Self::AlmostDualInfeasible,
            SolverStatus::MaxIterations => Self::MaxIterations,
            SolverStatus::MaxTime => Self::MaxTime,
            SolverStatus::NumericalError => Self::NumericalError,
            SolverStatus::InsufficientProgress => Self::InsufficientProgress,
            SolverStatus::CallbackTerminated => Self::CallbackTerminated,
        }
    }
}

impl std::fmt::Display for SqpQpRawStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other(label) => f.write_str(label),
            _ => write!(f, "{self:?}"),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpTermination {
    Converged,
    MaxIterations,
    QpSolve,
    LineSearchFailed,
    RestorationFailed,
    Stalled,
    NonFiniteInput,
    NonFiniteCallbackOutput,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpFinalStateKind {
    InitialPoint,
    AcceptedIterate,
    TrialPoint,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFiniteInputStage {
    InitialGuess,
    ParameterValues { parameter_index: Index },
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFiniteCallbackStage {
    ObjectiveValue,
    ObjectiveGradient,
    EqualityValues,
    InequalityValues,
    EqualityJacobianValues,
    InequalityJacobianValues,
    LagrangianHessianValues,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SqpIterationTiming {
    pub adapter_timing: Option<SqpAdapterTiming>,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub objective_value: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub objective_gradient: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub equality_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub inequality_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub equality_jacobian_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub inequality_jacobian_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub lagrangian_hessian_values: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub jacobian_assembly: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub hessian_assembly: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub regularization: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub subproblem_assembly: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub qp_setup: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub qp_solve: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub multiplier_estimation: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub line_search_evaluation: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub line_search_condition_checks: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub convergence_check: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub preprocess_other: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub preprocess: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub total: Duration,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpLineSearchTrial {
    pub alpha: f64,
    pub objective: f64,
    pub merit: f64,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub primal_inf: f64,
    pub armijo_satisfied: bool,
    pub armijo_tolerance_adjusted: bool,
    pub objective_armijo_satisfied: Option<bool>,
    pub objective_armijo_tolerance_adjusted: Option<bool>,
    pub wolfe_satisfied: Option<bool>,
    pub violation_satisfied: bool,
    pub filter_acceptable: Option<bool>,
    pub filter_dominated: Option<bool>,
    pub filter_theta_acceptable: Option<bool>,
    pub filter_sufficient_objective_reduction: Option<bool>,
    pub filter_sufficient_violation_reduction: Option<bool>,
    pub switching_condition_satisfied: Option<bool>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpStepKind {
    Objective,
    Feasibility,
    Restoration,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpTrustRegionTrial {
    pub radius: f64,
    pub step_norm: f64,
    pub boundary_active: bool,
    pub actual_reduction: f64,
    pub predicted_reduction: f64,
    pub ratio: Option<f64>,
    pub filter_acceptable: Option<bool>,
    pub filter_dominated: Option<bool>,
    pub filter_theta_acceptable: Option<bool>,
    pub filter_sufficient_objective_reduction: Option<bool>,
    pub filter_sufficient_violation_reduction: Option<bool>,
    pub switching_condition_satisfied: Option<bool>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub qp_status: Option<SqpQpStatus>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub qp_raw_status: Option<SqpQpRawStatus>,
    pub restoration_phase: bool,
    pub elastic_recovery_attempted: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpTrustRegionInfo {
    pub radius: f64,
    pub attempted_radius: f64,
    pub contraction_count: Index,
    pub qp_failure_retries: Index,
    pub step_norm: f64,
    pub boundary_active: bool,
    pub actual_reduction: f64,
    pub predicted_reduction: f64,
    pub ratio: Option<f64>,
    pub restoration_attempted: bool,
    pub elastic_recovery_attempted: bool,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_kind: Option<SqpStepKind>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub filter_acceptance_mode: Option<SqpFilterAcceptanceMode>,
    pub rejected_trials: Vec<SqpTrustRegionTrial>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpLineSearchInfo {
    pub accepted_alpha: f64,
    pub last_tried_alpha: f64,
    pub backtrack_count: Index,
    pub armijo_satisfied: bool,
    pub armijo_tolerance_adjusted: bool,
    pub objective_armijo_satisfied: Option<bool>,
    pub objective_armijo_tolerance_adjusted: Option<bool>,
    pub second_order_correction_attempted: bool,
    pub second_order_correction_used: bool,
    pub wolfe_satisfied: Option<bool>,
    pub violation_satisfied: bool,
    pub restoration_attempted: bool,
    pub elastic_recovery_attempted: bool,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_kind: Option<SqpStepKind>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub filter_acceptance_mode: Option<SqpFilterAcceptanceMode>,
    pub filter_acceptable: Option<bool>,
    pub filter_dominated: Option<bool>,
    pub filter_theta_acceptable: Option<bool>,
    pub filter_sufficient_objective_reduction: Option<bool>,
    pub filter_sufficient_violation_reduction: Option<bool>,
    pub switching_condition_satisfied: Option<bool>,
    pub rejected_trials: Vec<SqpLineSearchTrial>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpRegularizationInfo {
    pub enabled: bool,
    pub min_eigenvalue: f64,
    pub applied_shift: f64,
    pub shifted_by_analysis: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpStepDiagnostics {
    pub objective_directional_derivative: f64,
    pub exact_merit_directional_derivative: f64,
    pub qp_model_change: f64,
    pub current_violation: f64,
    pub theta_max: f64,
    pub linearized_eq_inf: Option<f64>,
    pub linearized_ineq_inf: Option<f64>,
    pub linearized_primal_inf: f64,
    pub switching_condition_satisfied: bool,
    pub restoration_phase: bool,
    pub elastic_recovery_used: bool,
    pub regularization: SqpRegularizationInfo,
}

#[expect(
    clippy::too_many_arguments,
    reason = "line-search failure reporting carries explicit diagnostic fields for snapshots"
)]
fn failed_sqp_line_search_info(
    last_tried_alpha: f64,
    backtrack_count: Index,
    rejected_trials: Vec<SqpLineSearchTrial>,
    wolfe_rejected: bool,
    step_kind: Option<SqpStepKind>,
    switching_condition_satisfied: Option<bool>,
    second_order_correction_attempted: bool,
    restoration_attempted: bool,
    elastic_recovery_attempted: bool,
) -> SqpLineSearchInfo {
    let last_trial = rejected_trials.last();
    SqpLineSearchInfo {
        accepted_alpha: 0.0,
        last_tried_alpha,
        backtrack_count,
        armijo_satisfied: last_trial.is_some_and(|trial| trial.armijo_satisfied),
        armijo_tolerance_adjusted: last_trial.is_some_and(|trial| trial.armijo_tolerance_adjusted),
        objective_armijo_satisfied: last_trial.and_then(|trial| trial.objective_armijo_satisfied),
        objective_armijo_tolerance_adjusted: last_trial
            .and_then(|trial| trial.objective_armijo_tolerance_adjusted),
        second_order_correction_attempted,
        second_order_correction_used: false,
        wolfe_satisfied: if wolfe_rejected { Some(false) } else { None },
        violation_satisfied: last_trial.is_some_and(|trial| trial.violation_satisfied),
        restoration_attempted,
        elastic_recovery_attempted,
        step_kind,
        filter_acceptance_mode: None,
        filter_acceptable: last_trial.and_then(|trial| trial.filter_acceptable),
        filter_dominated: last_trial.and_then(|trial| trial.filter_dominated),
        filter_theta_acceptable: last_trial.and_then(|trial| trial.filter_theta_acceptable),
        filter_sufficient_objective_reduction: last_trial
            .and_then(|trial| trial.filter_sufficient_objective_reduction),
        filter_sufficient_violation_reduction: last_trial
            .and_then(|trial| trial.filter_sufficient_violation_reduction),
        switching_condition_satisfied,
        rejected_trials,
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqpQpInfo {
    pub status: SqpQpStatus,
    pub raw_status: SqpQpRawStatus,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub setup_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "duration_seconds_serde"))]
    pub solve_time: Duration,
    pub iteration_count: Index,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SqpConeKind {
    Zero,
    Nonnegative,
    SecondOrder,
    Other(String),
}

impl std::fmt::Display for SqpConeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => f.write_str("zero"),
            Self::Nonnegative => f.write_str("nonnegative"),
            Self::SecondOrder => f.write_str("second_order"),
            Self::Other(label) => f.write_str(label),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqpConeSummary {
    pub kind: SqpConeKind,
    pub dim: Index,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpQpFailureDiagnostics {
    pub qp_info: SqpQpInfo,
    pub variable_count: Index,
    pub constraint_count: Index,
    pub linear_objective_inf_norm: f64,
    pub rhs_inf_norm: f64,
    pub hessian_diag_min: f64,
    pub hessian_diag_max: f64,
    pub elastic_recovery: bool,
    pub cones: Vec<SqpConeSummary>,
    pub transcript: Option<String>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SqpIterationSnapshot {
    pub iteration: Index,
    pub phase: SqpIterationPhase,
    pub globalization: SqpGlobalizationKind,
    pub x: Vec<f64>,
    pub objective: f64,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub dual_inf: f64,
    pub comp_inf: Option<f64>,
    pub overall_inf: f64,
    pub step_inf: Option<f64>,
    pub penalty: f64,
    pub line_search: Option<SqpLineSearchInfo>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub trust_region: Option<SqpTrustRegionInfo>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub step_diagnostics: Option<SqpStepDiagnostics>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub filter: Option<SqpFilterInfo>,
    pub qp: Option<SqpQpInfo>,
    pub timing: SqpIterationTiming,
    pub events: Vec<SqpIterationEvent>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct ClarabelSqpSummary {
    pub x: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub equality_inf_norm: Option<f64>,
    pub inequality_inf_norm: Option<f64>,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: Option<f64>,
    pub overall_inf_norm: f64,
    pub termination: SqpTermination,
    pub final_state: SqpIterationSnapshot,
    pub final_state_kind: SqpFinalStateKind,
    pub last_accepted_state: Option<SqpIterationSnapshot>,
    pub profiling: ClarabelSqpProfiling,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct SqpFailureContext {
    pub termination: SqpTermination,
    pub final_state: Option<SqpIterationSnapshot>,
    pub final_state_kind: Option<SqpFinalStateKind>,
    pub last_accepted_state: Option<SqpIterationSnapshot>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub failed_line_search: Option<SqpLineSearchInfo>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub failed_trust_region: Option<SqpTrustRegionInfo>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub failed_step_diagnostics: Option<SqpStepDiagnostics>,
    pub qp_failure: Option<SqpQpFailureDiagnostics>,
    pub profiling: ClarabelSqpProfiling,
}

#[derive(Debug, Error)]
pub enum ClarabelSqpError {
    #[error("invalid SQP input: {0}")]
    InvalidInput(String),
    #[error("non-finite SQP input at {stage:?}")]
    NonFiniteInput { stage: NonFiniteInputStage },
    #[error("clarabel SQP failed to converge in {iterations} iterations")]
    MaxIterations {
        iterations: Index,
        context: Box<SqpFailureContext>,
    },
    #[error("clarabel solver setup failed: {0}")]
    Setup(String),
    #[error("clarabel returned status {status:?}")]
    QpSolve {
        status: SqpQpRawStatus,
        context: Box<SqpFailureContext>,
    },
    #[error("unconstrained SQP subproblem solve failed")]
    UnconstrainedStepSolve { context: Box<SqpFailureContext> },
    #[error(
        "armijo line search failed to find sufficient decrease (directional derivative {directional_derivative}, step inf-norm {step_inf_norm}, penalty {penalty})"
    )]
    LineSearchFailed {
        directional_derivative: f64,
        step_inf_norm: f64,
        penalty: f64,
        context: Box<SqpFailureContext>,
    },
    #[error("sqp restoration failed with step inf-norm {step_inf_norm}")]
    RestorationFailed {
        step_inf_norm: f64,
        context: Box<SqpFailureContext>,
    },
    #[error(
        "sqp stalled with step inf-norm {step_inf_norm}, primal inf-norm {primal_inf_norm}, dual inf-norm {dual_inf_norm}, complementarity inf-norm {complementarity_inf_norm}"
    )]
    Stalled {
        step_inf_norm: f64,
        primal_inf_norm: f64,
        dual_inf_norm: f64,
        complementarity_inf_norm: f64,
        context: Box<SqpFailureContext>,
    },
    #[error("non-finite SQP callback output at {stage:?}")]
    NonFiniteCallbackOutput {
        stage: NonFiniteCallbackStage,
        context: Box<SqpFailureContext>,
    },
}

fn ccs_to_dense(sp: &CCS, values: &[f64]) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(sp.nrow, sp.ncol);
    for col in 0..sp.ncol {
        let start = sp.col_ptrs[col];
        let end = sp.col_ptrs[col + 1];
        for (offset, &row) in sp.row_indices[start..end].iter().enumerate() {
            dense[(row, col)] = values[start + offset];
        }
    }
    dense
}

fn lower_triangle_to_symmetric_dense(sp: &CCS, values: &[f64]) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(sp.nrow, sp.ncol);
    for col in 0..sp.ncol {
        let start = sp.col_ptrs[col];
        let end = sp.col_ptrs[col + 1];
        for (offset, &row) in sp.row_indices[start..end].iter().enumerate() {
            let value = values[start + offset];
            dense[(row, col)] = value;
            dense[(col, row)] = value;
        }
    }
    dense
}

fn dense_to_csc_upper(matrix: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = matrix.ncols();
    let mut col_ptrs = Vec::with_capacity(n + 1);
    let mut row_vals = Vec::new();
    let mut nz_vals = Vec::new();
    col_ptrs.push(0);
    for col in 0..n {
        for row in 0..=col {
            let value = matrix[(row, col)];
            if value.abs() > 1e-12 || row == col {
                row_vals.push(row);
                nz_vals.push(value);
            }
        }
        col_ptrs.push(row_vals.len());
    }
    CscMatrix::new(n, n, col_ptrs, row_vals, nz_vals)
}

fn dense_to_csc(matrix: &DMatrix<f64>) -> CscMatrix<f64> {
    let ncol = matrix.ncols();
    let nrow = matrix.nrows();
    let mut col_ptrs = Vec::with_capacity(ncol + 1);
    let mut row_vals = Vec::new();
    let mut nz_vals = Vec::new();
    col_ptrs.push(0);
    for col in 0..ncol {
        for row in 0..nrow {
            let value = matrix[(row, col)];
            if value.abs() > 1e-12 {
                row_vals.push(row);
                nz_vals.push(value);
            }
        }
        col_ptrs.push(row_vals.len());
    }
    CscMatrix::new(nrow, ncol, col_ptrs, row_vals, nz_vals)
}

fn stack_jacobians(
    equality_jacobian: &DMatrix<f64>,
    inequality_jacobian: &DMatrix<f64>,
) -> DMatrix<f64> {
    let ncol = equality_jacobian.ncols().max(inequality_jacobian.ncols());
    let total_rows = equality_jacobian.nrows() + inequality_jacobian.nrows();
    let mut stacked = DMatrix::<f64>::zeros(total_rows, ncol);
    for row in 0..equality_jacobian.nrows() {
        for col in 0..equality_jacobian.ncols() {
            stacked[(row, col)] = equality_jacobian[(row, col)];
        }
    }
    let row_offset = equality_jacobian.nrows();
    for row in 0..inequality_jacobian.nrows() {
        for col in 0..inequality_jacobian.ncols() {
            stacked[(row_offset + row, col)] = inequality_jacobian[(row, col)];
        }
    }
    stacked
}

#[derive(Clone, Debug, Default)]
struct BoundConstraints {
    lower_indices: Vec<Index>,
    lower_values: Vec<f64>,
    upper_indices: Vec<Index>,
    upper_values: Vec<f64>,
}

impl BoundConstraints {
    fn total_count(&self) -> Index {
        self.lower_indices.len() + self.upper_indices.len()
    }
}

fn collect_bound_constraints<P>(
    problem: &P,
) -> std::result::Result<BoundConstraints, ClarabelSqpError>
where
    P: CompiledNlpProblem,
{
    let Some(bounds_view) = problem.variable_bounds() else {
        return Ok(BoundConstraints::default());
    };
    let dimension = problem.dimension();
    let lower = bounds_view.lower.unwrap_or_default();
    let upper = bounds_view.upper.unwrap_or_default();

    let mut bounds = BoundConstraints::default();
    for idx in 0..dimension {
        let lower_bound = lower.get(idx).copied().flatten();
        let upper_bound = upper.get(idx).copied().flatten();
        if let (Some(lower_bound), Some(upper_bound)) = (lower_bound, upper_bound)
            && lower_bound > upper_bound
        {
            return Err(ClarabelSqpError::InvalidInput(format!(
                "variable bound interval is empty at index {idx}: lower={lower_bound} > upper={upper_bound}"
            )));
        }
        if let Some(lower_bound) = lower_bound {
            bounds.lower_indices.push(idx);
            bounds.lower_values.push(lower_bound);
        }
        if let Some(upper_bound) = upper_bound {
            bounds.upper_indices.push(idx);
            bounds.upper_values.push(upper_bound);
        }
    }
    Ok(bounds)
}

fn build_bound_jacobian(bounds: &BoundConstraints, dimension: Index) -> DMatrix<f64> {
    let mut jacobian = DMatrix::<f64>::zeros(bounds.total_count(), dimension);
    for (row, &idx) in bounds.lower_indices.iter().enumerate() {
        jacobian[(row, idx)] = -1.0;
    }
    let row_offset = bounds.lower_indices.len();
    for (row, &idx) in bounds.upper_indices.iter().enumerate() {
        jacobian[(row_offset + row, idx)] = 1.0;
    }
    jacobian
}

fn augment_inequality_values(
    nonlinear_values: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    out: &mut [f64],
) {
    debug_assert_eq!(
        out.len(),
        nonlinear_values.len() + bounds.lower_indices.len() + bounds.upper_indices.len()
    );
    out[..nonlinear_values.len()].copy_from_slice(nonlinear_values);
    let mut cursor = nonlinear_values.len();
    for (&idx, &bound) in bounds.lower_indices.iter().zip(bounds.lower_values.iter()) {
        out[cursor] = bound - x[idx];
        cursor += 1;
    }
    for (&idx, &bound) in bounds.upper_indices.iter().zip(bounds.upper_values.iter()) {
        out[cursor] = x[idx] - bound;
        cursor += 1;
    }
}

fn split_augmented_inequality_multipliers(
    multipliers: &[f64],
    nonlinear_count: Index,
    lower_bound_count: Index,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nonlinear = multipliers[..nonlinear_count].to_vec();
    let lower = multipliers[nonlinear_count..nonlinear_count + lower_bound_count].to_vec();
    let upper = multipliers[nonlinear_count + lower_bound_count..].to_vec();
    (nonlinear, lower, upper)
}

fn inf_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn average_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64
    }
}

fn positive_part(value: f64) -> f64 {
    value.max(0.0)
}

fn positive_part_inf_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0, |acc, value| acc.max(positive_part(*value)))
}

fn complementarity_inf_norm(inequality_values: &[f64], inequality_multipliers: &[f64]) -> f64 {
    inequality_values
        .iter()
        .zip(inequality_multipliers.iter())
        .fold(0.0, |acc, (value, multiplier)| {
            acc.max((value * multiplier).abs())
        })
}

fn soc_multiplier_tolerance(constraint_tol: f64) -> f64 {
    constraint_tol.max(1e-8)
}

fn soc_active_slack_tolerance(constraint_tol: f64) -> f64 {
    (10.0 * constraint_tol).max(1e-8)
}

fn should_include_soc_inequality_row(
    trial_value: f64,
    multiplier: f64,
    constraint_tol: f64,
) -> bool {
    trial_value > constraint_tol
        || (multiplier > soc_multiplier_tolerance(constraint_tol)
            && trial_value >= -soc_active_slack_tolerance(constraint_tol))
}

pub(crate) fn second_order_correction_step(
    equality_jacobian: &DMatrix<f64>,
    inequality_jacobian: &DMatrix<f64>,
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
            should_include_soc_inequality_row(trial_value, multiplier, constraint_tol)
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

    let mut active_jacobian = DMatrix::<f64>::zeros(active_row_count, equality_jacobian.ncols());
    let mut rhs = DVector::<f64>::zeros(active_row_count);
    let mut active_row = 0;
    for row in 0..trial_equality_values.len() {
        for col in 0..equality_jacobian.ncols() {
            active_jacobian[(active_row, col)] = equality_jacobian[(row, col)];
        }
        rhs[active_row] = -trial_equality_values[row];
        active_row += 1;
    }
    for &row in &active_inequality_rows {
        for col in 0..inequality_jacobian.ncols() {
            active_jacobian[(active_row, col)] = inequality_jacobian[(row, col)];
        }
        rhs[active_row] = -trial_augmented_inequality_values[row];
        active_row += 1;
    }

    let correction = active_jacobian
        .svd(true, true)
        .solve(&rhs, soc_multiplier_tolerance(constraint_tol))
        .ok()?;
    let correction = correction.column(0).iter().copied().collect::<Vec<_>>();
    (inf_norm(&correction) > 0.0 && correction.iter().all(|value| value.is_finite()))
        .then_some(correction)
}

pub(crate) fn scaled_overall_inf_norm(
    primal_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    all_dual_multipliers: &[f64],
    complementarity_multipliers: &[f64],
    scale_max: f64,
) -> f64 {
    let dual_scale = (average_abs(all_dual_multipliers) / scale_max).max(1.0);
    let complementarity_scale = (average_abs(complementarity_multipliers) / scale_max).max(1.0);
    primal_inf
        .max(dual_inf / dual_scale)
        .max(complementarity_inf / complementarity_scale)
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
        .sum()
}

fn mat_vec(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; matrix.nrows()];
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            product[row] += matrix[(row, col)] * vector[col];
        }
    }
    product
}

fn lagrangian_gradient(
    gradient: &[f64],
    equality_jacobian: &DMatrix<f64>,
    equality_multipliers: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    inequality_multipliers: &[f64],
) -> Vec<f64> {
    let mut residual = gradient.to_vec();
    for row in 0..equality_jacobian.nrows() {
        let lambda = equality_multipliers[row];
        for col in 0..equality_jacobian.ncols() {
            residual[col] += equality_jacobian[(row, col)] * lambda;
        }
    }
    for row in 0..inequality_jacobian.nrows() {
        let mu = inequality_multipliers[row];
        for col in 0..inequality_jacobian.ncols() {
            residual[col] += inequality_jacobian[(row, col)] * mu;
        }
    }
    residual
}

fn estimate_equality_multipliers(
    gradient: &[f64],
    equality_jacobian: &DMatrix<f64>,
) -> Option<Vec<f64>> {
    if equality_jacobian.nrows() == 0 {
        return Some(Vec::new());
    }
    let rhs = DVector::from_iterator(gradient.len(), gradient.iter().map(|value| -value));
    let estimate = equality_jacobian
        .transpose()
        .svd(true, true)
        .solve(&rhs, 1e-10)
        .ok()?;
    let estimate = estimate.column(0).iter().copied().collect::<Vec<_>>();
    estimate
        .iter()
        .all(|value| value.is_finite())
        .then_some(estimate)
}

fn exact_merit_value(
    objective_value: f64,
    equality_values: &[f64],
    inequality_values: &[f64],
    penalty: f64,
) -> f64 {
    objective_value
        + penalty
            * (equality_values.iter().map(|value| value.abs()).sum::<f64>()
                + inequality_values
                    .iter()
                    .map(|value| positive_part(*value))
                    .sum::<f64>())
}

fn exact_merit_directional_derivative(
    gradient: &[f64],
    equality_values: &[f64],
    equality_jacobian: &DMatrix<f64>,
    inequality_values: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    step: &[f64],
    penalty: f64,
) -> f64 {
    let equality_directional = mat_vec(equality_jacobian, step);
    let inequality_directional = mat_vec(inequality_jacobian, step);
    let equality_penalty_rate = equality_values
        .iter()
        .zip(equality_directional.iter())
        .map(|(value, directional)| {
            if *value > 0.0 {
                *directional
            } else if *value < 0.0 {
                -*directional
            } else {
                directional.abs()
            }
        })
        .sum::<f64>();
    let inequality_penalty_rate = inequality_values
        .iter()
        .zip(inequality_directional.iter())
        .map(|(value, directional)| {
            if *value > 0.0 {
                *directional
            } else if *value < 0.0 {
                0.0
            } else {
                positive_part(*directional)
            }
        })
        .sum::<f64>();
    dot(gradient, step) + penalty * (equality_penalty_rate + inequality_penalty_rate)
}

fn two_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[expect(
    clippy::too_many_arguments,
    reason = "SQP merit model helpers keep the inputs explicit to match the mathematical terms"
)]
fn exact_merit_model_value(
    objective_value: f64,
    gradient: &[f64],
    hessian: &DMatrix<f64>,
    equality_values: &[f64],
    equality_jacobian: &DMatrix<f64>,
    inequality_values: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    step: &[f64],
    penalty: f64,
) -> f64 {
    let linearized_eq = linearized_constraint_residual(equality_values, equality_jacobian, step);
    let linearized_ineq =
        linearized_constraint_residual(inequality_values, inequality_jacobian, step);
    objective_value
        + dot(gradient, step)
        + 0.5 * quadratic_form(hessian, step)
        + penalty
            * (linearized_eq.iter().map(|value| value.abs()).sum::<f64>()
                + linearized_ineq
                    .iter()
                    .map(|value| positive_part(*value))
                    .sum::<f64>())
}

#[expect(
    clippy::too_many_arguments,
    reason = "SQP merit model helpers keep the inputs explicit to match the mathematical terms"
)]
fn exact_merit_model_reduction(
    objective_value: f64,
    gradient: &[f64],
    hessian: &DMatrix<f64>,
    equality_values: &[f64],
    equality_jacobian: &DMatrix<f64>,
    inequality_values: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    step: &[f64],
    penalty: f64,
) -> f64 {
    exact_merit_value(objective_value, equality_values, inequality_values, penalty)
        - exact_merit_model_value(
            objective_value,
            gradient,
            hessian,
            equality_values,
            equality_jacobian,
            inequality_values,
            inequality_jacobian,
            step,
            penalty,
        )
}

fn sqp_filter_theta_reference(current_theta: f64, options: &ClarabelSqpOptions) -> f64 {
    match &options.globalization {
        SqpGlobalization::LineSearchFilter(globalization) => options
            .constraint_tol
            .max(globalization.filter.switching_reference_min),
        SqpGlobalization::TrustRegionFilter(globalization) => current_theta
            .max(options.constraint_tol)
            .max(globalization.filter.switching_reference_min),
        _ => panic!("filter settings required"),
    }
}

fn sqp_filter_parameters(options: &ClarabelSqpOptions, theta_max: f64) -> filter::FilterParameters {
    let filter = match &options.globalization {
        SqpGlobalization::LineSearchFilter(globalization) => &globalization.filter,
        SqpGlobalization::TrustRegionFilter(globalization) => &globalization.filter,
        _ => panic!("filter settings required"),
    };
    filter::FilterParameters {
        gamma_phi: filter.gamma_objective,
        gamma_theta: filter.gamma_violation,
        eta_phi: filter.armijo_c1,
        theta_max,
    }
}

fn sqp_switching_condition(
    current_theta: f64,
    theta_reference: f64,
    predicted_theta: f64,
    objective_directional_derivative: f64,
    options: &ClarabelSqpOptions,
) -> bool {
    let (violation_factor, linearized_reduction_factor) = match &options.globalization {
        SqpGlobalization::LineSearchFilter(globalization) => (
            globalization.filter.switching_violation_factor,
            globalization.filter.switching_linearized_reduction_factor,
        ),
        SqpGlobalization::TrustRegionFilter(globalization) => (
            globalization.filter.switching_violation_factor,
            globalization.filter.switching_linearized_reduction_factor,
        ),
        _ => return false,
    };
    objective_directional_derivative < 0.0
        && current_theta <= violation_factor * theta_reference + options.constraint_tol.max(1e-12)
        && predicted_theta
            <= linearized_reduction_factor * current_theta.max(options.constraint_tol)
}

fn update_merit_penalty(
    current_penalty: f64,
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
) -> f64 {
    let equality_multiplier_inf = inf_norm(equality_multipliers);
    let inequality_multiplier_inf = inequality_multipliers
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(positive_part(*value)));
    current_penalty.max(equality_multiplier_inf.max(inequality_multiplier_inf) + 1.0)
}

fn quadratic_form(matrix: &DMatrix<f64>, vector: &[f64]) -> f64 {
    let product = mat_vec(matrix, vector);
    dot(vector, &product)
}

fn regularize_hessian(hessian: &mut DMatrix<f64>, regularization: f64) -> SqpRegularizationInfo {
    let eigen = SymmetricEigen::new(hessian.clone());
    let min_eig = eigen
        .eigenvalues
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value));
    let shifted_by_analysis = min_eig < regularization;
    let shift = if min_eig < regularization {
        regularization - min_eig
    } else {
        regularization
    };
    for idx in 0..hessian.nrows() {
        hessian[(idx, idx)] += shift;
    }
    SqpRegularizationInfo {
        enabled: true,
        min_eigenvalue: min_eig,
        applied_shift: shift,
        shifted_by_analysis,
    }
}

fn disabled_hessian_regularization_info() -> SqpRegularizationInfo {
    SqpRegularizationInfo {
        enabled: false,
        min_eigenvalue: f64::NAN,
        applied_shift: 0.0,
        shifted_by_analysis: false,
    }
}

fn linearized_constraint_residual(
    values: &[f64],
    jacobian: &DMatrix<f64>,
    step: &[f64],
) -> Vec<f64> {
    let mut residual = values.to_vec();
    let directional = mat_vec(jacobian, step);
    for (value, delta) in residual.iter_mut().zip(directional.iter()) {
        *value += delta;
    }
    residual
}

#[expect(
    clippy::too_many_arguments,
    reason = "step diagnostics aggregate the full set of already-computed SQP quantities"
)]
fn sqp_step_diagnostics(
    gradient: &[f64],
    hessian: &DMatrix<f64>,
    current_violation: f64,
    theta_max: f64,
    equality_values: &[f64],
    equality_jacobian: &DMatrix<f64>,
    augmented_inequality_values: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    step: &[f64],
    exact_merit_directional_derivative: f64,
    switching_condition_satisfied: bool,
    restoration_phase: bool,
    elastic_recovery_used: bool,
    regularization: SqpRegularizationInfo,
) -> SqpStepDiagnostics {
    let linearized_eq = linearized_constraint_residual(equality_values, equality_jacobian, step);
    let linearized_ineq =
        linearized_constraint_residual(augmented_inequality_values, inequality_jacobian, step);
    let linearized_eq_inf = (!linearized_eq.is_empty()).then(|| inf_norm(&linearized_eq));
    let linearized_ineq_inf =
        (!linearized_ineq.is_empty()).then(|| positive_part_inf_norm(&linearized_ineq));
    let linearized_primal_inf = linearized_eq_inf
        .unwrap_or(0.0)
        .max(linearized_ineq_inf.unwrap_or(0.0));
    let objective_directional_derivative = dot(gradient, step);
    let qp_model_change = objective_directional_derivative + 0.5 * quadratic_form(hessian, step);
    SqpStepDiagnostics {
        objective_directional_derivative,
        exact_merit_directional_derivative,
        qp_model_change,
        current_violation,
        theta_max,
        linearized_eq_inf,
        linearized_ineq_inf,
        linearized_primal_inf,
        switching_condition_satisfied,
        restoration_phase,
        elastic_recovery_used,
        regularization,
    }
}

fn split_multipliers(multipliers: &[f64], equality_count: Index) -> (Vec<f64>, Vec<f64>) {
    let equality = multipliers[..equality_count].to_vec();
    let inequality = multipliers[equality_count..]
        .iter()
        .map(|value| positive_part(*value))
        .collect::<Vec<_>>();
    (equality, inequality)
}

fn solve_unconstrained_quadratic_step(
    hessian: &DMatrix<f64>,
    gradient: &[f64],
) -> std::result::Result<Vec<f64>, ()> {
    let rhs = DVector::<f64>::from_iterator(gradient.len(), gradient.iter().map(|value| -value));
    let lu = hessian.clone().lu();
    let Some(step) = lu.solve(&rhs) else {
        return Err(());
    };
    Ok(step.iter().copied().collect())
}

fn trial_merit<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    buffers: (&mut [f64], &mut [f64], &mut [f64]),
    bounds: &BoundConstraints,
    penalty: f64,
    timing: (&mut ClarabelSqpProfiling, &mut Duration),
) -> std::result::Result<TrialEvaluation, NonFiniteCallbackStage>
where
    P: CompiledNlpProblem,
{
    let (equality_values, inequality_values, augmented_inequality_values) = buffers;
    let (profiling, iteration_callback_time) = timing;
    time_callback(
        &mut profiling.equality_values,
        iteration_callback_time,
        || problem.equality_values(x, parameters, equality_values),
    );
    if equality_values.iter().any(|value| !value.is_finite()) {
        return Err(NonFiniteCallbackStage::EqualityValues);
    }
    time_callback(
        &mut profiling.inequality_values,
        iteration_callback_time,
        || problem.inequality_values(x, parameters, inequality_values),
    );
    if inequality_values.iter().any(|value| !value.is_finite()) {
        return Err(NonFiniteCallbackStage::InequalityValues);
    }
    augment_inequality_values(inequality_values, x, bounds, augmented_inequality_values);
    let objective_value = time_callback(
        &mut profiling.objective_value,
        iteration_callback_time,
        || problem.objective_value(x, parameters),
    );
    if !objective_value.is_finite() {
        return Err(NonFiniteCallbackStage::ObjectiveValue);
    }
    let eq_inf = (!equality_values.is_empty()).then_some(inf_norm(equality_values));
    let ineq_inf = (!augmented_inequality_values.is_empty())
        .then_some(positive_part_inf_norm(augmented_inequality_values));
    Ok(TrialEvaluation {
        objective: objective_value,
        merit: exact_merit_value(
            objective_value,
            equality_values,
            augmented_inequality_values,
            penalty,
        ),
        eq_inf,
        ineq_inf,
        primal_inf: eq_inf.into_iter().chain(ineq_inf).fold(0.0_f64, f64::max),
    })
}

fn trial_merit_directional_derivative<P>(
    problem: &P,
    request: TrialDerivativeRequest<'_>,
    profiling: &mut ClarabelSqpProfiling,
    iteration_callback_time: &mut Duration,
    workspace: TrialDerivativeWorkspace<'_>,
) -> std::result::Result<f64, NonFiniteCallbackStage>
where
    P: CompiledNlpProblem,
{
    time_callback(
        &mut profiling.objective_gradient,
        iteration_callback_time,
        || problem.objective_gradient(request.x, request.parameters, workspace.trial_gradient),
    );
    if workspace
        .trial_gradient
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(NonFiniteCallbackStage::ObjectiveGradient);
    }

    time_callback(
        &mut profiling.equality_jacobian_values,
        iteration_callback_time,
        || {
            problem.equality_jacobian_values(
                request.x,
                request.parameters,
                workspace.trial_equality_jacobian_values,
            )
        },
    );
    if workspace
        .trial_equality_jacobian_values
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(NonFiniteCallbackStage::EqualityJacobianValues);
    }

    time_callback(
        &mut profiling.inequality_jacobian_values,
        iteration_callback_time,
        || {
            problem.inequality_jacobian_values(
                request.x,
                request.parameters,
                workspace.trial_inequality_jacobian_values,
            )
        },
    );
    if workspace
        .trial_inequality_jacobian_values
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(NonFiniteCallbackStage::InequalityJacobianValues);
    }

    let equality_jacobian = ccs_to_dense(
        problem.equality_jacobian_ccs(),
        workspace.trial_equality_jacobian_values,
    );
    let nonlinear_inequality_jacobian = ccs_to_dense(
        problem.inequality_jacobian_ccs(),
        workspace.trial_inequality_jacobian_values,
    );
    let inequality_jacobian =
        stack_jacobians(&nonlinear_inequality_jacobian, workspace.bound_jacobian);
    Ok(exact_merit_directional_derivative(
        workspace.trial_gradient,
        request.equality_values,
        &equality_jacobian,
        request.augmented_inequality_values,
        &inequality_jacobian,
        request.step,
        request.penalty,
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "iteration event synthesis consumes independent state toggles from one snapshot boundary"
)]
fn snapshot_events(
    penalty_updated: bool,
    hessian_shifted: bool,
    line_search_backtracks: Option<Index>,
    armijo_tolerance_adjusted: bool,
    second_order_correction_used: bool,
    filter_accepted: bool,
    restoration_step_accepted: bool,
    wolfe_rejected: bool,
    qp_info: Option<&SqpQpInfo>,
    elastic_recovery_used: bool,
    max_iterations_reached: bool,
) -> Vec<SqpIterationEvent> {
    let mut events = Vec::new();
    if penalty_updated {
        events.push(SqpIterationEvent::PenaltyUpdated);
    }
    if hessian_shifted {
        events.push(SqpIterationEvent::HessianShifted);
    }
    if matches!(line_search_backtracks, Some(iterations) if iterations >= 4) {
        events.push(SqpIterationEvent::LongLineSearch);
    }
    if armijo_tolerance_adjusted {
        events.push(SqpIterationEvent::ArmijoToleranceAdjusted);
    }
    if second_order_correction_used {
        events.push(SqpIterationEvent::SecondOrderCorrectionUsed);
    }
    if filter_accepted {
        events.push(SqpIterationEvent::FilterAccepted);
    }
    if restoration_step_accepted {
        events.push(SqpIterationEvent::RestorationStepAccepted);
    }
    if wolfe_rejected {
        events.push(SqpIterationEvent::WolfeRejectedTrial);
    }
    if matches!(
        qp_info.map(|info| info.status),
        Some(SqpQpStatus::ReducedAccuracy)
    ) {
        events.push(SqpIterationEvent::QpReducedAccuracy);
    }
    if elastic_recovery_used {
        events.push(SqpIterationEvent::ElasticRecoveryUsed);
    }
    if max_iterations_reached {
        events.push(SqpIterationEvent::MaxIterationsReached);
    }
    events
}

fn accepted_filter_mode(
    line_search: Option<&SqpLineSearchInfo>,
    trust_region: Option<&SqpTrustRegionInfo>,
) -> Option<SqpFilterAcceptanceMode> {
    line_search
        .and_then(|info| info.filter_acceptance_mode)
        .or_else(|| trust_region.and_then(|info| info.filter_acceptance_mode))
}

fn accepted_step_kind(
    line_search: Option<&SqpLineSearchInfo>,
    trust_region: Option<&SqpTrustRegionInfo>,
) -> Option<SqpStepKind> {
    line_search
        .and_then(|info| info.step_kind)
        .or_else(|| trust_region.and_then(|info| info.step_kind))
}

fn final_state_kind(snapshot: &SqpIterationSnapshot) -> SqpFinalStateKind {
    match snapshot.phase {
        SqpIterationPhase::Initial => SqpFinalStateKind::InitialPoint,
        SqpIterationPhase::AcceptedStep | SqpIterationPhase::PostConvergence => {
            SqpFinalStateKind::AcceptedIterate
        }
    }
}

fn failure_context(
    termination: SqpTermination,
    final_state: Option<SqpIterationSnapshot>,
    last_accepted_state: Option<SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> Box<SqpFailureContext> {
    failure_context_with_qp_failure(
        termination,
        final_state,
        last_accepted_state,
        None,
        None,
        None,
        None,
        profiling,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "failure context preserves independent optional diagnostics for downstream reporting"
)]
fn failure_context_with_qp_failure(
    termination: SqpTermination,
    final_state: Option<SqpIterationSnapshot>,
    last_accepted_state: Option<SqpIterationSnapshot>,
    failed_line_search: Option<SqpLineSearchInfo>,
    failed_trust_region: Option<SqpTrustRegionInfo>,
    failed_step_diagnostics: Option<SqpStepDiagnostics>,
    qp_failure: Option<SqpQpFailureDiagnostics>,
    profiling: &ClarabelSqpProfiling,
) -> Box<SqpFailureContext> {
    let final_state_kind = final_state.as_ref().map(final_state_kind);
    Box::new(SqpFailureContext {
        termination,
        final_state,
        final_state_kind,
        last_accepted_state,
        failed_line_search,
        failed_trust_region,
        failed_step_diagnostics,
        qp_failure,
        profiling: profiling.clone(),
    })
}

#[derive(Clone, Copy, Debug)]
struct TrialEvaluation {
    objective: f64,
    merit: f64,
    eq_inf: Option<f64>,
    ineq_inf: Option<f64>,
    primal_inf: f64,
}

struct AcceptedLineSearchTrial {
    point: Vec<f64>,
    evaluation: TrialEvaluation,
    armijo_satisfied: bool,
    armijo_tolerance_adjusted: bool,
    objective_armijo_satisfied: Option<bool>,
    objective_armijo_tolerance_adjusted: Option<bool>,
    second_order_correction_used: bool,
    wolfe_satisfied: Option<bool>,
    violation_satisfied: bool,
    step_kind: Option<SqpStepKind>,
    filter_acceptance_mode: Option<SqpFilterAcceptanceMode>,
    filter_acceptable: Option<bool>,
    filter_dominated: Option<bool>,
    filter_theta_acceptable: Option<bool>,
    filter_sufficient_objective_reduction: Option<bool>,
    filter_sufficient_violation_reduction: Option<bool>,
    switching_condition_satisfied: Option<bool>,
}

struct SqpLineSearchAttempt {
    accepted_trial: Option<AcceptedLineSearchTrial>,
    last_tried_alpha: f64,
    backtrack_count: Index,
    rejected_trials: Vec<SqpLineSearchTrial>,
    wolfe_rejected: bool,
    second_order_correction_attempted: bool,
    restoration_attempted: bool,
    elastic_recovery_attempted: bool,
}

type LineSearchStageResult = (
    bool,
    SqpSubproblemSolution,
    AcceptedLineSearchTrial,
    SqpStepDiagnostics,
    f64,
    f64,
    f64,
    Index,
    Vec<SqpLineSearchTrial>,
    bool,
    bool,
    bool,
    bool,
);

struct SqpTrustRegionAttempt {
    penalty_updated: bool,
    solution: SqpSubproblemSolution,
    accepted_trial: AcceptedLineSearchTrial,
    step_diagnostics: SqpStepDiagnostics,
    step_inf_norm: f64,
    trust_region: SqpTrustRegionInfo,
    next_radius: f64,
}

fn trust_region_qp_failure_retries(rejected_trials: &[SqpTrustRegionTrial]) -> Index {
    rejected_trials
        .iter()
        .filter(|trial| trial.qp_raw_status.is_some())
        .count() as Index
}

struct TrialDerivativeWorkspace<'a> {
    trial_gradient: &'a mut [f64],
    trial_equality_jacobian_values: &'a mut [f64],
    trial_inequality_jacobian_values: &'a mut [f64],
    bound_jacobian: &'a DMatrix<f64>,
}

struct TrialDerivativeRequest<'a> {
    x: &'a [f64],
    parameters: &'a [ParameterMatrix<'a>],
    step: &'a [f64],
    equality_values: &'a [f64],
    augmented_inequality_values: &'a [f64],
    penalty: f64,
}

#[allow(clippy::too_many_arguments)]
fn attempt_sqp_line_search<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    solution: &SqpSubproblemSolution,
    candidate_all_inequality_multipliers: &[f64],
    objective_directional_derivative: f64,
    directional_derivative: f64,
    switching_condition_satisfied: bool,
    restoration_phase: bool,
    equality_jacobian: &DMatrix<f64>,
    inequality_jacobian: &DMatrix<f64>,
    primal_inf: f64,
    current_merit: f64,
    merit_penalty: f64,
    options: &ClarabelSqpOptions,
    filter_entries: &[filter::FilterEntry],
    current_filter_trial: &filter::FilterEntry,
    filter_theta_max: f64,
    current_snapshot: &SqpIterationSnapshot,
    last_accepted_state: &Option<SqpIterationSnapshot>,
    bounds: &BoundConstraints,
    bound_jacobian: &DMatrix<f64>,
    profiling: &mut ClarabelSqpProfiling,
    iteration_callback_time: &mut Duration,
    iteration_line_search_evaluation_time: &mut Duration,
    iteration_line_search_condition_check_time: &mut Duration,
    trial_equality_values: &mut [f64],
    trial_inequality_values: &mut [f64],
    trial_augmented_inequality_values: &mut [f64],
    trial_gradient: &mut [f64],
    trial_equality_jacobian_values: &mut [f64],
    trial_inequality_jacobian_values: &mut [f64],
) -> std::result::Result<SqpLineSearchAttempt, ClarabelSqpError>
where
    P: CompiledNlpProblem,
{
    let (line_search_options, use_filter) = match &options.globalization {
        SqpGlobalization::LineSearchMerit(globalization) => (&globalization.line_search, false),
        SqpGlobalization::LineSearchFilter(globalization) => (&globalization.line_search, true),
        _ => panic!("line search attempt requires line-search globalization options"),
    };
    let step = &solution.step;
    let step_inf_norm = inf_norm(step);
    let mut alpha = 1.0;
    let mut maybe_accepted_trial = None;
    let mut current_line_search_iterations = 0;
    let mut current_last_tried_alpha = alpha;
    let mut current_rejected_trials = Vec::new();
    let mut current_wolfe_rejected = false;
    let mut second_order_correction_attempted = false;
    let restoration_attempted = restoration_phase;
    let elastic_recovery_attempted = solution.elastic_recovery_used;
    while alpha * step_inf_norm >= line_search_options.min_step
        && current_line_search_iterations <= line_search_options.max_steps
    {
        let trial = x
            .iter()
            .zip(step.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect::<Vec<_>>();
        let line_search_eval_started = Instant::now();
        let trial_eval = trial_merit(
            problem,
            &trial,
            parameters,
            (
                trial_equality_values,
                trial_inequality_values,
                trial_augmented_inequality_values,
            ),
            bounds,
            merit_penalty,
            (profiling, iteration_callback_time),
        )
        .map_err(|stage| ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: failure_context(
                SqpTermination::NonFiniteCallbackOutput,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                profiling,
            ),
        })?;
        let trial_directional_derivative = line_search_options
            .wolfe_c2
            .map(|_| {
                trial_merit_directional_derivative(
                    problem,
                    TrialDerivativeRequest {
                        x: &trial,
                        parameters,
                        step,
                        equality_values: trial_equality_values,
                        augmented_inequality_values: trial_augmented_inequality_values,
                        penalty: merit_penalty,
                    },
                    profiling,
                    iteration_callback_time,
                    TrialDerivativeWorkspace {
                        trial_gradient,
                        trial_equality_jacobian_values,
                        trial_inequality_jacobian_values,
                        bound_jacobian,
                    },
                )
            })
            .transpose()
            .map_err(|stage| ClarabelSqpError::NonFiniteCallbackOutput {
                stage,
                context: failure_context(
                    SqpTermination::NonFiniteCallbackOutput,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    profiling,
                ),
            })?;
        let line_search_eval_elapsed = line_search_eval_started.elapsed();
        profiling.line_search_evaluations += 1;
        *iteration_line_search_evaluation_time += line_search_eval_elapsed;
        profiling.line_search_evaluation_time += line_search_eval_elapsed;

        let line_search_check_started = Instant::now();
        let armijo_rhs =
            current_merit + line_search_options.armijo_c1 * alpha * directional_derivative;
        let armijo_abs_tol = current_merit.abs().max(1.0) * 1e-12;
        let armijo_satisfied_strict = trial_eval.merit <= armijo_rhs;
        let armijo_tolerance_adjusted =
            !armijo_satisfied_strict && trial_eval.merit <= armijo_rhs + armijo_abs_tol;
        let armijo_satisfied = armijo_satisfied_strict || armijo_tolerance_adjusted;
        let wolfe_satisfied = line_search_options.wolfe_c2.map(|c2| {
            trial_directional_derivative.expect("wolfe_c2 implies a trial directional derivative")
                >= c2 * directional_derivative
        });
        let violation_satisfied = trial_eval.primal_inf <= primal_inf.max(options.constraint_tol);
        let filter_assessment = use_filter.then(|| {
            filter::assess_trial(
                filter_entries,
                current_filter_trial,
                &filter::entry(trial_eval.objective, trial_eval.primal_inf),
                alpha,
                objective_directional_derivative,
                switching_condition_satisfied,
                switching_condition_satisfied,
                sqp_filter_parameters(options, filter_theta_max),
            )
        });
        let line_search_check_elapsed = line_search_check_started.elapsed();
        profiling.line_search_condition_checks += 1;
        *iteration_line_search_condition_check_time += line_search_check_elapsed;
        profiling.line_search_condition_check_time += line_search_check_elapsed;

        let filter_acceptance_mode = filter_assessment.and_then(|assessment| {
            wolfe_satisfied
                .unwrap_or(true)
                .then_some(assessment.acceptance_mode)
                .flatten()
        });
        let merit_accepted = armijo_satisfied && wolfe_satisfied.unwrap_or(true);
        let accepted_step_kind = if restoration_phase {
            Some(SqpStepKind::Restoration)
        } else if !use_filter {
            Some(SqpStepKind::Objective)
        } else {
            match filter_acceptance_mode {
                Some(SqpFilterAcceptanceMode::ObjectiveArmijo) => Some(SqpStepKind::Objective),
                Some(SqpFilterAcceptanceMode::ViolationReduction) => Some(SqpStepKind::Feasibility),
                None => None,
            }
        };
        if (use_filter && filter_acceptance_mode.is_some()) || (!use_filter && merit_accepted) {
            maybe_accepted_trial = Some(AcceptedLineSearchTrial {
                point: trial,
                evaluation: trial_eval,
                armijo_satisfied,
                armijo_tolerance_adjusted,
                objective_armijo_satisfied: filter_assessment
                    .map(|assessment| assessment.objective_armijo_satisfied),
                objective_armijo_tolerance_adjusted: filter_assessment
                    .map(|assessment| assessment.objective_armijo_tolerance_adjusted),
                second_order_correction_used: false,
                wolfe_satisfied,
                violation_satisfied,
                step_kind: accepted_step_kind,
                filter_acceptance_mode,
                filter_acceptable: filter_assessment.map(|assessment| assessment.filter_acceptable),
                filter_dominated: filter_assessment.map(|assessment| assessment.filter_dominated),
                filter_theta_acceptable: filter_assessment
                    .map(|assessment| assessment.filter_theta_acceptable),
                filter_sufficient_objective_reduction: filter_assessment
                    .map(|assessment| assessment.filter_sufficient_objective_reduction),
                filter_sufficient_violation_reduction: filter_assessment
                    .map(|assessment| assessment.filter_sufficient_violation_reduction),
                switching_condition_satisfied: filter_assessment
                    .map(|assessment| assessment.switching_condition_satisfied),
            });
            current_last_tried_alpha = alpha;
            break;
        }
        if wolfe_satisfied == Some(false) {
            current_wolfe_rejected = true;
        }
        current_rejected_trials.push(SqpLineSearchTrial {
            alpha,
            objective: trial_eval.objective,
            merit: trial_eval.merit,
            eq_inf: trial_eval.eq_inf,
            ineq_inf: trial_eval.ineq_inf,
            primal_inf: trial_eval.primal_inf,
            armijo_satisfied,
            armijo_tolerance_adjusted,
            objective_armijo_satisfied: filter_assessment
                .map(|assessment| assessment.objective_armijo_satisfied),
            objective_armijo_tolerance_adjusted: filter_assessment
                .map(|assessment| assessment.objective_armijo_tolerance_adjusted),
            wolfe_satisfied,
            violation_satisfied,
            filter_acceptable: filter_assessment.map(|assessment| assessment.filter_acceptable),
            filter_dominated: filter_assessment.map(|assessment| assessment.filter_dominated),
            filter_theta_acceptable: filter_assessment
                .map(|assessment| assessment.filter_theta_acceptable),
            filter_sufficient_objective_reduction: filter_assessment
                .map(|assessment| assessment.filter_sufficient_objective_reduction),
            filter_sufficient_violation_reduction: filter_assessment
                .map(|assessment| assessment.filter_sufficient_violation_reduction),
            switching_condition_satisfied: filter_assessment
                .map(|assessment| assessment.switching_condition_satisfied),
        });
        if options.second_order_correction
            && alpha == 1.0
            && current_line_search_iterations == 0
            && line_search_options.wolfe_c2.is_none()
            && !restoration_phase
            && !solution.elastic_recovery_used
            && !violation_satisfied
            && let Some(correction) = second_order_correction_step(
                equality_jacobian,
                inequality_jacobian,
                trial_equality_values,
                trial_augmented_inequality_values,
                candidate_all_inequality_multipliers,
                options.constraint_tol,
            )
        {
            second_order_correction_attempted = true;
            let corrected_trial = x
                .iter()
                .zip(step.iter().zip(correction.iter()))
                .map(|(xi, (di, pi))| xi + di + pi)
                .collect::<Vec<_>>();
            let corrected_eval_started = Instant::now();
            let corrected_eval = trial_merit(
                problem,
                &corrected_trial,
                parameters,
                (
                    trial_equality_values,
                    trial_inequality_values,
                    trial_augmented_inequality_values,
                ),
                bounds,
                merit_penalty,
                (profiling, iteration_callback_time),
            )
            .map_err(|stage| ClarabelSqpError::NonFiniteCallbackOutput {
                stage,
                context: failure_context(
                    SqpTermination::NonFiniteCallbackOutput,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    profiling,
                ),
            })?;
            let corrected_eval_elapsed = corrected_eval_started.elapsed();
            profiling.line_search_evaluations += 1;
            *iteration_line_search_evaluation_time += corrected_eval_elapsed;
            profiling.line_search_evaluation_time += corrected_eval_elapsed;

            let corrected_check_started = Instant::now();
            let corrected_armijo_strict = corrected_eval.merit <= armijo_rhs;
            let corrected_armijo_tolerance_adjusted =
                !corrected_armijo_strict && corrected_eval.merit <= armijo_rhs + armijo_abs_tol;
            let corrected_armijo_satisfied =
                corrected_armijo_strict || corrected_armijo_tolerance_adjusted;
            let corrected_violation_satisfied =
                corrected_eval.primal_inf <= primal_inf.max(options.constraint_tol);
            let corrected_filter_assessment = use_filter.then(|| {
                filter::assess_trial(
                    filter_entries,
                    current_filter_trial,
                    &filter::entry(corrected_eval.objective, corrected_eval.primal_inf),
                    alpha,
                    objective_directional_derivative,
                    switching_condition_satisfied,
                    sqp_filter_parameters(options, filter_theta_max),
                )
            });
            let corrected_check_elapsed = corrected_check_started.elapsed();
            profiling.line_search_condition_checks += 1;
            *iteration_line_search_condition_check_time += corrected_check_elapsed;
            profiling.line_search_condition_check_time += corrected_check_elapsed;

            let corrected_filter_acceptance_mode =
                corrected_filter_assessment.and_then(|assessment| assessment.acceptance_mode);
            if (use_filter && corrected_filter_acceptance_mode.is_some())
                || (!use_filter && corrected_armijo_satisfied)
            {
                maybe_accepted_trial = Some(AcceptedLineSearchTrial {
                    point: corrected_trial,
                    evaluation: corrected_eval,
                    armijo_satisfied: corrected_armijo_satisfied,
                    armijo_tolerance_adjusted: corrected_armijo_tolerance_adjusted,
                    objective_armijo_satisfied: corrected_filter_assessment
                        .map(|assessment| assessment.objective_armijo_satisfied),
                    objective_armijo_tolerance_adjusted: corrected_filter_assessment
                        .map(|assessment| assessment.objective_armijo_tolerance_adjusted),
                    second_order_correction_used: true,
                    wolfe_satisfied: None,
                    violation_satisfied: corrected_violation_satisfied,
                    step_kind: match corrected_filter_acceptance_mode {
                        Some(SqpFilterAcceptanceMode::ObjectiveArmijo) => {
                            Some(SqpStepKind::Objective)
                        }
                        Some(SqpFilterAcceptanceMode::ViolationReduction) => {
                            Some(SqpStepKind::Feasibility)
                        }
                        None if !use_filter => Some(SqpStepKind::Objective),
                        None => None,
                    },
                    filter_acceptance_mode: corrected_filter_acceptance_mode,
                    filter_acceptable: corrected_filter_assessment
                        .map(|assessment| assessment.filter_acceptable),
                    filter_dominated: corrected_filter_assessment
                        .map(|assessment| assessment.filter_dominated),
                    filter_theta_acceptable: corrected_filter_assessment
                        .map(|assessment| assessment.filter_theta_acceptable),
                    filter_sufficient_objective_reduction: corrected_filter_assessment
                        .map(|assessment| assessment.filter_sufficient_objective_reduction),
                    filter_sufficient_violation_reduction: corrected_filter_assessment
                        .map(|assessment| assessment.filter_sufficient_violation_reduction),
                    switching_condition_satisfied: corrected_filter_assessment
                        .map(|assessment| assessment.switching_condition_satisfied),
                });
                current_last_tried_alpha = alpha;
                break;
            }
        }
        current_last_tried_alpha = alpha;
        alpha *= line_search_options.beta;
        current_line_search_iterations += 1;
    }
    Ok(SqpLineSearchAttempt {
        accepted_trial: maybe_accepted_trial,
        last_tried_alpha: current_last_tried_alpha,
        backtrack_count: current_line_search_iterations,
        rejected_trials: current_rejected_trials,
        wolfe_rejected: current_wolfe_rejected,
        second_order_correction_attempted,
        restoration_attempted,
        elastic_recovery_attempted,
    })
}

#[allow(clippy::too_many_arguments)]
fn attempt_sqp_trust_region<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    objective_value: f64,
    gradient: &[f64],
    hessian: &DMatrix<f64>,
    equality_values: &[f64],
    equality_jacobian: &DMatrix<f64>,
    _inequality_values: &[f64],
    _nonlinear_inequality_jacobian: &DMatrix<f64>,
    augmented_inequality_values: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    bounds: &BoundConstraints,
    _bound_jacobian: &DMatrix<f64>,
    elastic_model: &ElasticRecoveryModel<'_>,
    regularization_info: &SqpRegularizationInfo,
    equality_count: Index,
    inequality_count: Index,
    lower_bound_count: Index,
    primal_inf: f64,
    options: &ClarabelSqpOptions,
    merit_penalty: &mut f64,
    current_radius: f64,
    filter_entries: &[filter::FilterEntry],
    filter_theta_reference: f64,
    filter_theta_max: f64,
    current_snapshot: &SqpIterationSnapshot,
    last_accepted_state: &Option<SqpIterationSnapshot>,
    profiling: &mut ClarabelSqpProfiling,
    iteration_callback_time: &mut Duration,
    iteration_subproblem_assembly_time: &mut Duration,
    iteration_qp_setup_time: &mut Duration,
    iteration_qp_solve_time: &mut Duration,
    iteration_line_search_evaluation_time: &mut Duration,
    iteration_line_search_condition_check_time: &mut Duration,
    trial_equality_values: &mut [f64],
    trial_inequality_values: &mut [f64],
    trial_augmented_inequality_values: &mut [f64],
    _trial_gradient: &mut [f64],
    _trial_equality_jacobian_values: &mut [f64],
    _trial_inequality_jacobian_values: &mut [f64],
) -> std::result::Result<SqpTrustRegionAttempt, ClarabelSqpError>
where
    P: CompiledNlpProblem,
{
    let (trust_region, use_filter, fixed_penalty) = match &options.globalization {
        SqpGlobalization::TrustRegionMerit(globalization) => (
            &globalization.trust_region,
            false,
            globalization.fixed_penalty,
        ),
        SqpGlobalization::TrustRegionFilter(globalization) => {
            (&globalization.trust_region, true, true)
        }
        _ => panic!("trust-region attempt requires trust-region globalization options"),
    };
    let mut radius = current_radius.clamp(trust_region.min_radius, trust_region.max_radius);
    let restoration_available =
        has_restoration_constraints(equality_count, inequality_count, bounds.total_count());
    let mut rejected_trials = Vec::new();
    let mut elastic_recovery_attempted = false;
    let current_filter_trial = filter::entry(objective_value, primal_inf);
    let mut last_step_diagnostics = None;
    let mut last_step_inf_norm = 0.0;
    let mut last_attempted_radius = radius;

    for contraction in 0..=trust_region.max_radius_contractions {
        let subproblem_assembly_started = Instant::now();
        let assembled_subproblem = assemble_sqp_subproblem(
            equality_jacobian,
            inequality_jacobian,
            equality_values,
            augmented_inequality_values,
            Some(radius),
        );
        let subproblem_assembly_elapsed = subproblem_assembly_started.elapsed();
        *iteration_subproblem_assembly_time += subproblem_assembly_elapsed;
        record_iteration_duration(
            &mut profiling.subproblem_assembly_steps,
            &mut profiling.subproblem_assembly_time,
            subproblem_assembly_elapsed,
        );
        let profiling_snapshot = profiling.clone();
        let mut qp_ctx = QpSolveContext {
            profiling,
            iteration_qp_setup_time,
            iteration_qp_solve_time,
        };
        let solution = match solve_sqp_subproblem(
            hessian,
            gradient,
            &assembled_subproblem,
            elastic_model,
            options,
            &mut qp_ctx,
            x.len(),
            equality_count,
            inequality_count,
            lower_bound_count,
            current_snapshot,
            last_accepted_state,
            &profiling_snapshot,
        ) {
            Ok(solution) => solution,
            Err(ClarabelSqpError::QpSolve { status: _, context })
                if context.qp_failure.as_ref().is_some_and(|failure| {
                    failure.qp_info.raw_status == SqpQpRawStatus::NumericalError
                }) =>
            {
                let qp_raw_status = context
                    .qp_failure
                    .as_ref()
                    .map(|failure| failure.qp_info.raw_status.clone());
                let qp_status = context
                    .qp_failure
                    .as_ref()
                    .map(|failure| failure.qp_info.status);
                let elastic_attempted = context
                    .qp_failure
                    .as_ref()
                    .is_some_and(|failure| failure.elastic_recovery);
                elastic_recovery_attempted |= elastic_attempted;
                last_attempted_radius = radius;
                rejected_trials.push(SqpTrustRegionTrial {
                    radius,
                    step_norm: 0.0,
                    boundary_active: false,
                    actual_reduction: 0.0,
                    predicted_reduction: 0.0,
                    ratio: None,
                    filter_acceptable: None,
                    filter_dominated: None,
                    filter_theta_acceptable: None,
                    filter_sufficient_objective_reduction: None,
                    filter_sufficient_violation_reduction: None,
                    switching_condition_satisfied: None,
                    qp_status,
                    qp_raw_status,
                    restoration_phase: false,
                    elastic_recovery_attempted: elastic_attempted,
                });
                radius *= trust_region.shrink_factor;
                if radius < trust_region.min_radius {
                    break;
                }
                continue;
            }
            Err(error) => return Err(error),
        };
        elastic_recovery_attempted |= solution.elastic_recovery_used;

        let all_inequality_multipliers = [
            solution.inequality_multipliers.as_slice(),
            solution.lower_bound_multipliers.as_slice(),
            solution.upper_bound_multipliers.as_slice(),
        ]
        .concat();
        if !fixed_penalty && !solution.elastic_recovery_used {
            *merit_penalty = update_merit_penalty(
                *merit_penalty,
                &solution.equality_multipliers,
                &all_inequality_multipliers,
            );
        }

        let step_inf_norm = inf_norm(&solution.step);
        let step_norm = two_norm(&solution.step);
        let objective_directional_derivative = dot(gradient, &solution.step);
        let predicted_linearized_eq_inf = if equality_count > 0 {
            inf_norm(&linearized_constraint_residual(
                equality_values,
                equality_jacobian,
                &solution.step,
            ))
        } else {
            0.0
        };
        let predicted_linearized_ineq_inf = if !augmented_inequality_values.is_empty() {
            positive_part_inf_norm(&linearized_constraint_residual(
                augmented_inequality_values,
                inequality_jacobian,
                &solution.step,
            ))
        } else {
            0.0
        };
        let switching_condition_satisfied = if solution.elastic_recovery_used {
            false
        } else {
            sqp_switching_condition(
                primal_inf,
                filter_theta_reference,
                predicted_linearized_eq_inf.max(predicted_linearized_ineq_inf),
                objective_directional_derivative,
                options,
            )
        };
        let exact_directional_derivative = exact_merit_directional_derivative(
            gradient,
            equality_values,
            equality_jacobian,
            augmented_inequality_values,
            inequality_jacobian,
            &solution.step,
            *merit_penalty,
        );
        let step_diagnostics = sqp_step_diagnostics(
            gradient,
            hessian,
            primal_inf,
            filter_theta_max,
            equality_values,
            equality_jacobian,
            augmented_inequality_values,
            inequality_jacobian,
            &solution.step,
            exact_directional_derivative,
            switching_condition_satisfied,
            solution.elastic_recovery_used,
            solution.elastic_recovery_used,
            regularization_info.clone(),
        );
        last_step_diagnostics = Some(step_diagnostics.clone());
        last_step_inf_norm = step_inf_norm;
        last_attempted_radius = radius;

        let current_merit = exact_merit_value(
            objective_value,
            equality_values,
            augmented_inequality_values,
            *merit_penalty,
        );
        let predicted_reduction = exact_merit_model_reduction(
            objective_value,
            gradient,
            hessian,
            equality_values,
            equality_jacobian,
            augmented_inequality_values,
            inequality_jacobian,
            &solution.step,
            *merit_penalty,
        );

        let trial_point = x
            .iter()
            .zip(solution.step.iter())
            .map(|(xi, di)| xi + di)
            .collect::<Vec<_>>();
        let trial_eval_started = Instant::now();
        let trial_eval = trial_merit(
            problem,
            &trial_point,
            parameters,
            (
                trial_equality_values,
                trial_inequality_values,
                trial_augmented_inequality_values,
            ),
            bounds,
            *merit_penalty,
            (profiling, iteration_callback_time),
        )
        .map_err(|stage| ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: failure_context(
                SqpTermination::NonFiniteCallbackOutput,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                profiling,
            ),
        })?;
        let trial_eval_elapsed = trial_eval_started.elapsed();
        profiling.line_search_evaluations += 1;
        *iteration_line_search_evaluation_time += trial_eval_elapsed;
        profiling.line_search_evaluation_time += trial_eval_elapsed;

        let tr_check_started = Instant::now();
        let actual_reduction = current_merit - trial_eval.merit;
        let ratio = (predicted_reduction > 0.0).then_some(actual_reduction / predicted_reduction);
        let boundary_active = step_norm >= trust_region.boundary_fraction * radius;
        let filter_assessment = use_filter.then(|| {
            filter::assess_trial(
                filter_entries,
                &current_filter_trial,
                &filter::entry(trial_eval.objective, trial_eval.primal_inf),
                1.0,
                objective_directional_derivative,
                switching_condition_satisfied,
                switching_condition_satisfied,
                sqp_filter_parameters(options, filter_theta_max),
            )
        });
        let filter_acceptance_mode =
            filter_assessment.and_then(|assessment| assessment.acceptance_mode);
        let tr_check_elapsed = tr_check_started.elapsed();
        profiling.line_search_condition_checks += 1;
        *iteration_line_search_condition_check_time += tr_check_elapsed;
        profiling.line_search_condition_check_time += tr_check_elapsed;

        let acceptable_ratio = ratio.is_some_and(|rho| rho >= trust_region.accept_ratio);
        let accepted = if use_filter {
            filter_acceptance_mode.is_some() && predicted_reduction > 0.0 && acceptable_ratio
        } else {
            predicted_reduction > 0.0 && acceptable_ratio
        };
        if accepted {
            let next_radius =
                if ratio.is_some_and(|rho| rho >= trust_region.expand_ratio) && boundary_active {
                    (radius * trust_region.grow_factor).min(trust_region.max_radius)
                } else {
                    radius
                };
            let step_kind = if solution.elastic_recovery_used {
                Some(SqpStepKind::Restoration)
            } else if use_filter {
                match filter_acceptance_mode {
                    Some(SqpFilterAcceptanceMode::ObjectiveArmijo) => Some(SqpStepKind::Objective),
                    Some(SqpFilterAcceptanceMode::ViolationReduction) => {
                        Some(SqpStepKind::Feasibility)
                    }
                    None => None,
                }
            } else {
                Some(SqpStepKind::Objective)
            };
            return Ok(SqpTrustRegionAttempt {
                penalty_updated: false,
                solution,
                accepted_trial: AcceptedLineSearchTrial {
                    point: trial_point,
                    evaluation: trial_eval,
                    armijo_satisfied: false,
                    armijo_tolerance_adjusted: false,
                    objective_armijo_satisfied: filter_assessment
                        .map(|assessment| assessment.objective_armijo_satisfied),
                    objective_armijo_tolerance_adjusted: filter_assessment
                        .map(|assessment| assessment.objective_armijo_tolerance_adjusted),
                    second_order_correction_used: false,
                    wolfe_satisfied: None,
                    violation_satisfied: trial_eval.primal_inf
                        <= primal_inf.max(options.constraint_tol),
                    step_kind,
                    filter_acceptance_mode,
                    filter_acceptable: filter_assessment
                        .map(|assessment| assessment.filter_acceptable),
                    filter_dominated: filter_assessment
                        .map(|assessment| assessment.filter_dominated),
                    filter_theta_acceptable: filter_assessment
                        .map(|assessment| assessment.filter_theta_acceptable),
                    filter_sufficient_objective_reduction: filter_assessment
                        .map(|assessment| assessment.filter_sufficient_objective_reduction),
                    filter_sufficient_violation_reduction: filter_assessment
                        .map(|assessment| assessment.filter_sufficient_violation_reduction),
                    switching_condition_satisfied: filter_assessment
                        .map(|assessment| assessment.switching_condition_satisfied),
                },
                step_diagnostics,
                step_inf_norm,
                trust_region: SqpTrustRegionInfo {
                    radius,
                    attempted_radius: radius,
                    contraction_count: contraction,
                    qp_failure_retries: trust_region_qp_failure_retries(&rejected_trials),
                    step_norm,
                    boundary_active,
                    actual_reduction,
                    predicted_reduction,
                    ratio,
                    restoration_attempted: false,
                    elastic_recovery_attempted,
                    step_kind,
                    filter_acceptance_mode,
                    rejected_trials,
                },
                next_radius,
            });
        }

        rejected_trials.push(SqpTrustRegionTrial {
            radius,
            step_norm,
            boundary_active,
            actual_reduction,
            predicted_reduction,
            ratio,
            filter_acceptable: filter_assessment.map(|assessment| assessment.filter_acceptable),
            filter_dominated: filter_assessment.map(|assessment| assessment.filter_dominated),
            filter_theta_acceptable: filter_assessment
                .map(|assessment| assessment.filter_theta_acceptable),
            filter_sufficient_objective_reduction: filter_assessment
                .map(|assessment| assessment.filter_sufficient_objective_reduction),
            filter_sufficient_violation_reduction: filter_assessment
                .map(|assessment| assessment.filter_sufficient_violation_reduction),
            switching_condition_satisfied: filter_assessment
                .map(|assessment| assessment.switching_condition_satisfied),
            qp_status: None,
            qp_raw_status: None,
            restoration_phase: false,
            elastic_recovery_attempted: solution.elastic_recovery_used,
        });

        if solution.elastic_recovery_used {
            radius *= trust_region.shrink_factor;
            if radius < trust_region.min_radius {
                break;
            }
            continue;
        }

        radius *= trust_region.shrink_factor;
        if radius < trust_region.min_radius {
            break;
        }
    }

    if options.restoration_phase && restoration_available {
        let profiling_snapshot = profiling.clone();
        let mut qp_ctx = QpSolveContext {
            profiling,
            iteration_qp_setup_time,
            iteration_qp_solve_time,
        };
        let restoration_solution = match solve_restoration_subproblem(
            elastic_model,
            options,
            &mut qp_ctx,
            x.len(),
            equality_count,
            inequality_count,
            lower_bound_count,
            current_snapshot,
            last_accepted_state,
            &profiling_snapshot,
        ) {
            Ok(solution) => solution,
            Err(ClarabelSqpError::QpSolve { status: _, context }) => {
                let qp_raw_status = context
                    .qp_failure
                    .as_ref()
                    .map(|failure| failure.qp_info.raw_status.clone());
                let qp_status = context
                    .qp_failure
                    .as_ref()
                    .map(|failure| failure.qp_info.status);
                let elastic_attempted = context
                    .qp_failure
                    .as_ref()
                    .is_some_and(|failure| failure.elastic_recovery);
                elastic_recovery_attempted |= elastic_attempted;
                rejected_trials.push(SqpTrustRegionTrial {
                    radius,
                    step_norm: 0.0,
                    boundary_active: false,
                    actual_reduction: 0.0,
                    predicted_reduction: 0.0,
                    ratio: None,
                    filter_acceptable: None,
                    filter_dominated: None,
                    filter_theta_acceptable: None,
                    filter_sufficient_objective_reduction: None,
                    filter_sufficient_violation_reduction: None,
                    switching_condition_satisfied: None,
                    qp_status,
                    qp_raw_status,
                    restoration_phase: true,
                    elastic_recovery_attempted: elastic_attempted,
                });
                let failed_trust_region = SqpTrustRegionInfo {
                    radius,
                    attempted_radius: last_attempted_radius,
                    contraction_count: trust_region.max_radius_contractions,
                    qp_failure_retries: trust_region_qp_failure_retries(&rejected_trials),
                    step_norm: 0.0,
                    boundary_active: false,
                    actual_reduction: 0.0,
                    predicted_reduction: 0.0,
                    ratio: None,
                    restoration_attempted: true,
                    elastic_recovery_attempted,
                    step_kind: Some(SqpStepKind::Restoration),
                    filter_acceptance_mode: None,
                    rejected_trials,
                };
                return Err(ClarabelSqpError::RestorationFailed {
                    step_inf_norm: last_step_inf_norm,
                    context: failure_context_with_qp_failure(
                        SqpTermination::RestorationFailed,
                        Some(current_snapshot.clone()),
                        last_accepted_state.clone(),
                        None,
                        Some(failed_trust_region),
                        last_step_diagnostics.clone(),
                        context.qp_failure,
                        profiling,
                    ),
                });
            }
            Err(error) => return Err(error),
        };
        elastic_recovery_attempted = true;
        let restoration_step_inf_norm = inf_norm(&restoration_solution.step);
        let restoration_step_norm = two_norm(&restoration_solution.step);
        let restoration_objective_directional_derivative =
            dot(gradient, &restoration_solution.step);
        let restoration_switching_condition = false;
        let restoration_step_diagnostics = sqp_step_diagnostics(
            gradient,
            hessian,
            primal_inf,
            filter_theta_max,
            equality_values,
            equality_jacobian,
            augmented_inequality_values,
            inequality_jacobian,
            &restoration_solution.step,
            exact_merit_directional_derivative(
                gradient,
                equality_values,
                equality_jacobian,
                augmented_inequality_values,
                inequality_jacobian,
                &restoration_solution.step,
                *merit_penalty,
            ),
            restoration_switching_condition,
            true,
            true,
            regularization_info.clone(),
        );
        let current_merit = exact_merit_value(
            objective_value,
            equality_values,
            augmented_inequality_values,
            *merit_penalty,
        );
        let predicted_reduction = exact_merit_model_reduction(
            objective_value,
            gradient,
            hessian,
            equality_values,
            equality_jacobian,
            augmented_inequality_values,
            inequality_jacobian,
            &restoration_solution.step,
            *merit_penalty,
        );
        let restoration_point = x
            .iter()
            .zip(restoration_solution.step.iter())
            .map(|(xi, di)| xi + di)
            .collect::<Vec<_>>();
        let trial_eval = trial_merit(
            problem,
            &restoration_point,
            parameters,
            (
                trial_equality_values,
                trial_inequality_values,
                trial_augmented_inequality_values,
            ),
            bounds,
            *merit_penalty,
            (profiling, iteration_callback_time),
        )
        .map_err(|stage| ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: failure_context(
                SqpTermination::NonFiniteCallbackOutput,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                profiling,
            ),
        })?;
        let actual_reduction = current_merit - trial_eval.merit;
        let ratio = (predicted_reduction > 0.0).then_some(actual_reduction / predicted_reduction);
        let boundary_active = restoration_step_norm >= trust_region.boundary_fraction * radius;
        let filter_assessment = use_filter.then(|| {
            filter::assess_trial(
                filter_entries,
                &current_filter_trial,
                &filter::entry(trial_eval.objective, trial_eval.primal_inf),
                1.0,
                restoration_objective_directional_derivative,
                restoration_switching_condition,
                restoration_switching_condition,
                sqp_filter_parameters(options, filter_theta_max),
            )
        });
        let filter_acceptance_mode =
            filter_assessment.and_then(|assessment| assessment.acceptance_mode);
        let acceptable_ratio = ratio.is_some_and(|rho| rho >= trust_region.accept_ratio);
        let accepted = if use_filter {
            filter_acceptance_mode.is_some() && predicted_reduction > 0.0 && acceptable_ratio
        } else {
            predicted_reduction > 0.0 && acceptable_ratio
        };
        if accepted {
            return Ok(SqpTrustRegionAttempt {
                penalty_updated: false,
                solution: restoration_solution,
                accepted_trial: AcceptedLineSearchTrial {
                    point: restoration_point,
                    evaluation: trial_eval,
                    armijo_satisfied: false,
                    armijo_tolerance_adjusted: false,
                    objective_armijo_satisfied: filter_assessment
                        .map(|assessment| assessment.objective_armijo_satisfied),
                    objective_armijo_tolerance_adjusted: filter_assessment
                        .map(|assessment| assessment.objective_armijo_tolerance_adjusted),
                    second_order_correction_used: false,
                    wolfe_satisfied: None,
                    violation_satisfied: true,
                    step_kind: Some(SqpStepKind::Restoration),
                    filter_acceptance_mode,
                    filter_acceptable: filter_assessment
                        .map(|assessment| assessment.filter_acceptable),
                    filter_dominated: filter_assessment
                        .map(|assessment| assessment.filter_dominated),
                    filter_theta_acceptable: filter_assessment
                        .map(|assessment| assessment.filter_theta_acceptable),
                    filter_sufficient_objective_reduction: filter_assessment
                        .map(|assessment| assessment.filter_sufficient_objective_reduction),
                    filter_sufficient_violation_reduction: filter_assessment
                        .map(|assessment| assessment.filter_sufficient_violation_reduction),
                    switching_condition_satisfied: filter_assessment
                        .map(|assessment| assessment.switching_condition_satisfied),
                },
                step_diagnostics: restoration_step_diagnostics,
                step_inf_norm: restoration_step_inf_norm,
                trust_region: SqpTrustRegionInfo {
                    radius,
                    attempted_radius: last_attempted_radius,
                    contraction_count: trust_region.max_radius_contractions,
                    qp_failure_retries: trust_region_qp_failure_retries(&rejected_trials),
                    step_norm: restoration_step_norm,
                    boundary_active,
                    actual_reduction,
                    predicted_reduction,
                    ratio,
                    restoration_attempted: true,
                    elastic_recovery_attempted,
                    step_kind: Some(SqpStepKind::Restoration),
                    filter_acceptance_mode,
                    rejected_trials,
                },
                next_radius: radius,
            });
        }
    }

    let failed_trust_region = SqpTrustRegionInfo {
        radius,
        attempted_radius: last_attempted_radius,
        contraction_count: trust_region.max_radius_contractions,
        qp_failure_retries: trust_region_qp_failure_retries(&rejected_trials),
        step_norm: 0.0,
        boundary_active: false,
        actual_reduction: 0.0,
        predicted_reduction: 0.0,
        ratio: None,
        restoration_attempted: options.restoration_phase && restoration_available,
        elastic_recovery_attempted,
        step_kind: None,
        filter_acceptance_mode: None,
        rejected_trials,
    };
    let failed_step = last_step_diagnostics;
    Err(if options.restoration_phase && restoration_available {
        ClarabelSqpError::RestorationFailed {
            step_inf_norm: last_step_inf_norm,
            context: failure_context_with_qp_failure(
                SqpTermination::RestorationFailed,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                None,
                Some(failed_trust_region),
                failed_step,
                None,
                profiling,
            ),
        }
    } else {
        ClarabelSqpError::LineSearchFailed {
            directional_derivative: 0.0,
            step_inf_norm: last_step_inf_norm,
            penalty: *merit_penalty,
            context: failure_context_with_qp_failure(
                SqpTermination::LineSearchFailed,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                None,
                Some(failed_trust_region),
                failed_step,
                None,
                profiling,
            ),
        }
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct IterationTimingBaseline {
    adapter_timing: Option<SqpAdapterTiming>,
    objective_value: Duration,
    objective_gradient: Duration,
    equality_values: Duration,
    inequality_values: Duration,
    equality_jacobian_values: Duration,
    inequality_jacobian_values: Duration,
    lagrangian_hessian_values: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct IterationTimingBuckets {
    jacobian_assembly: Duration,
    hessian_assembly: Duration,
    regularization: Duration,
    subproblem_assembly: Duration,
    qp_setup: Duration,
    qp_solve: Duration,
    multiplier_estimation: Duration,
    line_search_evaluation: Duration,
    line_search_condition_checks: Duration,
    convergence_check: Duration,
    preprocess_other: Duration,
    total: Duration,
}

impl IterationTimingBaseline {
    fn capture(profiling: &ClarabelSqpProfiling) -> Self {
        Self {
            adapter_timing: profiling.adapter_timing,
            objective_value: profiling.objective_value.total_time,
            objective_gradient: profiling.objective_gradient.total_time,
            equality_values: profiling.equality_values.total_time,
            inequality_values: profiling.inequality_values.total_time,
            equality_jacobian_values: profiling.equality_jacobian_values.total_time,
            inequality_jacobian_values: profiling.inequality_jacobian_values.total_time,
            lagrangian_hessian_values: profiling.lagrangian_hessian_values.total_time,
        }
    }
}

fn build_iteration_timing(
    baseline: IterationTimingBaseline,
    profiling: &ClarabelSqpProfiling,
    buckets: IterationTimingBuckets,
) -> SqpIterationTiming {
    let preprocess = buckets.jacobian_assembly
        + buckets.hessian_assembly
        + buckets.regularization
        + buckets.subproblem_assembly
        + buckets.preprocess_other;
    SqpIterationTiming {
        adapter_timing: profiling
            .adapter_timing
            .map(|timing| timing.saturating_sub(baseline.adapter_timing.unwrap_or_default())),
        objective_value: profiling
            .objective_value
            .total_time
            .saturating_sub(baseline.objective_value),
        objective_gradient: profiling
            .objective_gradient
            .total_time
            .saturating_sub(baseline.objective_gradient),
        equality_values: profiling
            .equality_values
            .total_time
            .saturating_sub(baseline.equality_values),
        inequality_values: profiling
            .inequality_values
            .total_time
            .saturating_sub(baseline.inequality_values),
        equality_jacobian_values: profiling
            .equality_jacobian_values
            .total_time
            .saturating_sub(baseline.equality_jacobian_values),
        inequality_jacobian_values: profiling
            .inequality_jacobian_values
            .total_time
            .saturating_sub(baseline.inequality_jacobian_values),
        lagrangian_hessian_values: profiling
            .lagrangian_hessian_values
            .total_time
            .saturating_sub(baseline.lagrangian_hessian_values),
        jacobian_assembly: buckets.jacobian_assembly,
        hessian_assembly: buckets.hessian_assembly,
        regularization: buckets.regularization,
        subproblem_assembly: buckets.subproblem_assembly,
        qp_setup: buckets.qp_setup,
        qp_solve: buckets.qp_solve,
        multiplier_estimation: buckets.multiplier_estimation,
        line_search_evaluation: buckets.line_search_evaluation,
        line_search_condition_checks: buckets.line_search_condition_checks,
        convergence_check: buckets.convergence_check,
        preprocess_other: buckets.preprocess_other,
        preprocess,
        total: buckets.total,
    }
}

fn cone_summaries(cones: &[clarabel::solver::SupportedConeT<f64>]) -> Vec<SqpConeSummary> {
    cones
        .iter()
        .map(|cone| match cone {
            clarabel::solver::SupportedConeT::ZeroConeT(dim) => SqpConeSummary {
                kind: SqpConeKind::Zero,
                dim: *dim,
            },
            clarabel::solver::SupportedConeT::NonnegativeConeT(dim) => SqpConeSummary {
                kind: SqpConeKind::Nonnegative,
                dim: *dim,
            },
            clarabel::solver::SupportedConeT::SecondOrderConeT(dim) => SqpConeSummary {
                kind: SqpConeKind::SecondOrder,
                dim: *dim,
            },
            _ => SqpConeSummary {
                kind: SqpConeKind::Other(format!("{cone:?}")),
                dim: 0,
            },
        })
        .collect()
}

#[cfg(unix)]
fn capture_stdio_output<R>(f: impl FnOnce() -> R) -> io::Result<(String, R)> {
    static CAPTURE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = CAPTURE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    io::stdout().flush()?;
    io::stderr().flush()?;

    let mut pipe_fds = [0; 2];
    if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }

    let read_fd = pipe_fds[0];
    let write_fd = pipe_fds[1];
    let stdout_dup = unsafe { libc::dup(libc::STDOUT_FILENO) };
    let stderr_dup = unsafe { libc::dup(libc::STDERR_FILENO) };
    if stdout_dup < 0 || stderr_dup < 0 {
        unsafe {
            libc::close(read_fd);
            libc::close(write_fd);
            if stdout_dup >= 0 {
                libc::close(stdout_dup);
            }
            if stderr_dup >= 0 {
                libc::close(stderr_dup);
            }
        }
        return Err(io::Error::last_os_error());
    }

    if unsafe { libc::dup2(write_fd, libc::STDOUT_FILENO) } < 0
        || unsafe { libc::dup2(write_fd, libc::STDERR_FILENO) } < 0
    {
        unsafe {
            libc::close(read_fd);
            libc::close(write_fd);
            libc::close(stdout_dup);
            libc::close(stderr_dup);
        }
        return Err(io::Error::last_os_error());
    }
    unsafe {
        libc::close(write_fd);
    }

    let result = catch_unwind(AssertUnwindSafe(f));

    let _ = io::stdout().flush();
    let _ = io::stderr().flush();
    unsafe {
        libc::dup2(stdout_dup, libc::STDOUT_FILENO);
        libc::dup2(stderr_dup, libc::STDERR_FILENO);
        libc::close(stdout_dup);
        libc::close(stderr_dup);
    }

    let mut file = unsafe { File::from_raw_fd(read_fd) };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    drop(file);
    let output = String::from_utf8_lossy(&bytes).into_owned();

    match result {
        Ok(value) => Ok((output, value)),
        Err(payload) => resume_unwind(payload),
    }
}

#[cfg(not(unix))]
fn capture_stdio_output<R>(f: impl FnOnce() -> R) -> io::Result<(String, R)> {
    Ok((String::new(), f()))
}

fn maybe_capture_failed_qp_transcript(
    hessian: &DMatrix<f64>,
    linear_objective: &[f64],
    constraint_matrix: &DMatrix<f64>,
    rhs: &[f64],
    cones: &[clarabel::solver::SupportedConeT<f64>],
) -> Option<String> {
    let captured = capture_stdio_output(|| {
        let Ok(settings) = DefaultSettingsBuilder::default().verbose(true).build() else {
            return;
        };
        let p = dense_to_csc_upper(hessian);
        let a = dense_to_csc(constraint_matrix);
        let Ok(mut solver) = DefaultSolver::new(&p, linear_objective, &a, rhs, cones, settings)
        else {
            return;
        };
        solver.solve();
    })
    .ok()?;
    let transcript = captured.0.trim().to_string();
    (!transcript.is_empty()).then_some(transcript)
}

fn build_qp_failure_diagnostics(
    qp_info: SqpQpInfo,
    hessian: &DMatrix<f64>,
    linear_objective: &[f64],
    constraint_matrix: &DMatrix<f64>,
    rhs: &[f64],
    cones: &[clarabel::solver::SupportedConeT<f64>],
    elastic_recovery: bool,
) -> SqpQpFailureDiagnostics {
    let diag_values = (0..hessian.nrows())
        .map(|idx| hessian[(idx, idx)])
        .collect::<Vec<_>>();
    let hessian_diag_min = diag_values.iter().copied().fold(f64::INFINITY, f64::min);
    let hessian_diag_max = diag_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    SqpQpFailureDiagnostics {
        qp_info,
        variable_count: hessian.nrows(),
        constraint_count: constraint_matrix.nrows(),
        linear_objective_inf_norm: inf_norm(linear_objective),
        rhs_inf_norm: inf_norm(rhs),
        hessian_diag_min,
        hessian_diag_max,
        elastic_recovery,
        cones: cone_summaries(cones),
        transcript: maybe_capture_failed_qp_transcript(
            hessian,
            linear_objective,
            constraint_matrix,
            rhs,
            cones,
        ),
    }
}

#[derive(Default)]
struct SqpEventLegendState {
    penalty: bool,
    hessian_shift: bool,
    line_search: bool,
    filter: bool,
    filter_reset: bool,
    soc_attempted: bool,
    elastic_attempted: bool,
    qp: bool,
    max_iter: bool,
    elastic: bool,
    wolfe: bool,
    armijo_adjust: bool,
    soc: bool,
    watchdog_armed: bool,
    watchdog: bool,
    bound_multiplier_safeguard: bool,
    barrier_update: bool,
    adaptive_regularization: bool,
    tiny_step: bool,
}

impl SqpEventLegendState {
    fn mark_new(flag: &mut bool) -> bool {
        let is_new = !*flag;
        *flag = true;
        is_new
    }

    fn mark_penalty_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.penalty)
    }

    fn mark_hessian_shift_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.hessian_shift)
    }

    fn mark_line_search_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.line_search)
    }

    fn mark_filter_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.filter)
    }

    fn mark_filter_reset_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.filter_reset)
    }

    fn mark_soc_attempted_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.soc_attempted)
    }

    fn mark_elastic_attempted_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.elastic_attempted)
    }

    fn mark_qp_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.qp)
    }

    fn mark_max_iter_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.max_iter)
    }

    fn mark_elastic_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.elastic)
    }

    fn mark_wolfe_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.wolfe)
    }

    fn mark_armijo_adjust_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.armijo_adjust)
    }

    fn mark_soc_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.soc)
    }

    fn mark_watchdog_armed_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.watchdog_armed)
    }

    fn mark_watchdog_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.watchdog)
    }

    fn mark_bound_multiplier_safeguard_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.bound_multiplier_safeguard)
    }

    fn mark_barrier_update_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.barrier_update)
    }

    fn mark_adaptive_regularization_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.adaptive_regularization)
    }

    fn mark_tiny_step_if_new(&mut self) -> bool {
        Self::mark_new(&mut self.tiny_step)
    }
}

fn ansi_enabled() -> bool {
    match ansi_color_mode() {
        AnsiColorMode::Always => true,
        AnsiColorMode::Never => false,
        AnsiColorMode::Auto => {
            static ENABLED: OnceLock<bool> = OnceLock::new();
            *ENABLED.get_or_init(|| {
                io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
            })
        }
    }
}

fn style(text: &str, code: &str) -> String {
    if ansi_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn style_bold(text: &str) -> String {
    style(text, "1")
}

fn style_cyan_bold(text: &str) -> String {
    style(text, "1;36")
}

fn style_green_bold(text: &str) -> String {
    style(text, "1;32")
}

fn style_yellow_bold(text: &str) -> String {
    style(text, "1;33")
}

fn style_red_bold(text: &str) -> String {
    style(text, "1;31")
}

pub fn reduced_accuracy_tolerance(tolerance: f64) -> f64 {
    100.0 * tolerance
}

pub fn classify_constraint_satisfaction(value: f64, tolerance: f64) -> ConstraintSatisfaction {
    if value <= tolerance {
        ConstraintSatisfaction::FullAccuracy
    } else if value <= reduced_accuracy_tolerance(tolerance) {
        ConstraintSatisfaction::ReducedAccuracy
    } else {
        ConstraintSatisfaction::Violated
    }
}

pub(crate) fn style_metric_against_tolerance(text: &str, value: f64, tolerance: f64) -> String {
    match classify_constraint_satisfaction(value, tolerance) {
        ConstraintSatisfaction::FullAccuracy => style_green_bold(text),
        ConstraintSatisfaction::ReducedAccuracy => style_yellow_bold(text),
        ConstraintSatisfaction::Violated => style_red_bold(text),
    }
}

pub fn constraint_bound_side(lower_violation: f64, upper_violation: f64) -> ConstraintBoundSide {
    match (lower_violation > 0.0, upper_violation > 0.0) {
        (true, true) => ConstraintBoundSide::Both,
        (true, false) => ConstraintBoundSide::Lower,
        (false, true) => ConstraintBoundSide::Upper,
        (false, false) => ConstraintBoundSide::Equality,
    }
}

pub fn worst_bound_violation(
    value: f64,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) -> (f64, f64) {
    let lower_violation = lower_bound.map_or(0.0, |lower| (lower - value).max(0.0));
    let upper_violation = upper_bound.map_or(0.0, |upper| (value - upper).max(0.0));
    (lower_violation, upper_violation)
}

pub fn rank_nlp_constraint_violations(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    bounds: &RuntimeNlpBounds,
    tolerance: f64,
) -> Result<NlpConstraintViolationReport, symbolic::RuntimeNlpBoundsError> {
    symbolic::rank_nlp_constraint_violations(problem, x, parameters, bounds, tolerance)
}

fn sci_text(value: f64) -> String {
    let raw = format!("{value:.2e}");
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return raw;
    };
    let Ok(exponent_value) = exponent.parse::<i32>() else {
        return raw;
    };
    format!("{mantissa}e{exponent_value:+03}")
}

fn fmt_sci(value: f64) -> String {
    format!("{:>9}", sci_text(value))
}

fn fmt_optional_sci(value: Option<f64>) -> String {
    match value {
        Some(value) => fmt_sci(value),
        None => format!("{:>9}", "--"),
    }
}

fn fmt_qp_iterations(iterations: Option<u32>) -> String {
    match iterations {
        Some(iterations) => format!("{iterations:>5}"),
        None => format!("{:>5}", "--"),
    }
}

fn compact_duration_text(seconds: f64) -> String {
    let units = [
        (1e-9_f64, "ns"),
        (1e-6_f64, "us"),
        (1e-3_f64, "ms"),
        (1.0_f64, "s"),
    ];
    let mut value = seconds * 1e9;
    let mut unit = "ns";
    for &(scale, candidate_unit) in &units {
        let candidate_value = seconds / scale;
        if (0.1..100.0).contains(&candidate_value) {
            value = candidate_value;
            unit = candidate_unit;
            break;
        }
        if candidate_value >= 100.0 {
            value = candidate_value;
            unit = candidate_unit;
        }
    }
    if value < 9.95 {
        format!("{value:.1}{unit}")
    } else {
        format!("{value:.0}{unit}")
    }
}

#[derive(Clone, Copy)]
enum SummaryDurationUnit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
}

impl SummaryDurationUnit {
    fn suffix(self) -> &'static str {
        match self {
            Self::Nanoseconds => "ns",
            Self::Microseconds => "us",
            Self::Milliseconds => "ms",
            Self::Seconds => "s",
        }
    }

    fn scale_seconds(self) -> f64 {
        match self {
            Self::Nanoseconds => 1e-9,
            Self::Microseconds => 1e-6,
            Self::Milliseconds => 1e-3,
            Self::Seconds => 1.0,
        }
    }
}

fn choose_summary_duration_unit(durations: &[Duration]) -> SummaryDurationUnit {
    let max_seconds = durations
        .iter()
        .map(Duration::as_secs_f64)
        .fold(0.0_f64, f64::max);
    if max_seconds >= 0.1 {
        SummaryDurationUnit::Seconds
    } else if max_seconds >= 1e-4 {
        SummaryDurationUnit::Milliseconds
    } else if max_seconds >= 1e-7 {
        SummaryDurationUnit::Microseconds
    } else {
        SummaryDurationUnit::Nanoseconds
    }
}

fn fmt_duration_in_unit(duration: Duration, unit: SummaryDurationUnit) -> String {
    let value = duration.as_secs_f64() / unit.scale_seconds();
    format!("{value:>6.1}{}", unit.suffix())
}

fn fmt_optional_duration_in_unit(duration: Option<Duration>, unit: SummaryDurationUnit) -> String {
    match duration {
        Some(duration) => fmt_duration_in_unit(duration, unit),
        None => format!("{:>8}", "--"),
    }
}

fn fmt_qp_time(seconds: Option<f64>) -> String {
    match seconds {
        Some(seconds) => format!("{:>7}", compact_duration_text(seconds)),
        None => format!("{:>7}", "--"),
    }
}

fn fmt_alpha(alpha: Option<f64>) -> String {
    match alpha {
        Some(alpha) => fmt_sci(alpha),
        None => format!("{:>9}", "--"),
    }
}

fn fmt_line_search_iterations(iterations: Option<Index>) -> String {
    match iterations {
        Some(iterations) => format!("{iterations:>5}"),
        None => format!("{:>5}", "--"),
    }
}

pub(crate) fn fmt_iteration_label(label: &str) -> String {
    format!("{label:>5}")
}

pub(crate) fn style_iteration_label_cell(label: &str, iteration_limit_reached: bool) -> String {
    let cell = fmt_iteration_label(label);
    if iteration_limit_reached {
        style_red_bold(&cell)
    } else {
        cell
    }
}

fn style_iteration_cell(snapshot: &SqpIterationSnapshot) -> String {
    let label = sqp_iteration_label(snapshot);
    style_iteration_label_cell(
        &label,
        has_event(snapshot, SqpIterationEvent::MaxIterationsReached),
    )
}

fn time_callback<R>(
    stat: &mut EvalTimingStat,
    iteration_callback_time: &mut Duration,
    f: impl FnOnce() -> R,
) -> R {
    let started = Instant::now();
    let result = f();
    let elapsed = started.elapsed();
    stat.record(elapsed);
    *iteration_callback_time += elapsed;
    result
}

fn record_iteration_duration(counter: &mut Index, total: &mut Duration, elapsed: Duration) {
    *counter += 1;
    *total += elapsed;
}

fn finalize_profiling(profiling: &mut ClarabelSqpProfiling, solve_started: Instant) {
    profiling.total_time = solve_started.elapsed();
    profiling.unaccounted_time = profiling.total_time.saturating_sub(
        profiling.evaluation_time
            + profiling.preprocessing_time
            + profiling.subproblem_solve_time
            + profiling.line_search_time
            + profiling.convergence_time,
    );
}

fn dense_fill_percent(nnz: Index, nrow: Index, ncol: Index) -> f64 {
    let denominator = nrow.saturating_mul(ncol);
    if denominator == 0 {
        0.0
    } else {
        100.0 * nnz as f64 / denominator as f64
    }
}

fn lower_tri_fill_percent(nnz: Index, size: Index) -> f64 {
    let denominator = size.saturating_mul(size + 1) / 2;
    if denominator == 0 {
        0.0
    } else {
        100.0 * nnz as f64 / denominator as f64
    }
}

fn declared_box_constraint_count<P>(problem: &P) -> Index
where
    P: CompiledNlpProblem,
{
    let Some(bounds) = problem.variable_bounds() else {
        return 0;
    };
    bounds
        .lower
        .unwrap_or_default()
        .into_iter()
        .filter(|bound| bound.is_some())
        .count()
        + bounds
            .upper
            .unwrap_or_default()
            .into_iter()
            .filter(|bound| bound.is_some())
            .count()
}

fn visible_len(text: &str) -> usize {
    let mut chars = text.chars().peekable();
    let mut len = 0;
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if matches!(chars.peek(), Some('[')) {
                chars.next();
                while let Some(control) = chars.next() {
                    if ('@'..='~').contains(&control) {
                        break;
                    }
                }
            }
            continue;
        }
        len += 1;
    }
    len
}

fn boxed_line(label: &str, detail: impl Into<String>) -> String {
    format!("{label:<BOX_LABEL_WIDTH$}  {}", detail.into())
}

fn log_boxed_section(title: &str, lines: &[String], title_style: fn(&str) -> String) {
    let width = lines
        .iter()
        .map(|line| visible_len(line))
        .chain(std::iter::once(visible_len(title)))
        .max()
        .unwrap_or_else(|| visible_len(title));
    let border = format!("+{}+", "-".repeat(width + 2));
    eprintln!();
    eprintln!("{border}");
    let title_padding = " ".repeat(width.saturating_sub(title.len()));
    eprintln!("| {}{} |", title_style(title), title_padding);
    eprintln!("+{}+", "-".repeat(width + 2));
    for line in lines {
        let padding = " ".repeat(width.saturating_sub(visible_len(line)));
        eprintln!("| {line}{padding} |");
    }
    eprintln!("{border}");
}

#[cfg(test)]
mod tests {
    use super::visible_len;

    #[test]
    fn visible_len_ignores_ansi_and_counts_unicode_chars() {
        let styled = "\u{1b}[1m‖ineq₊‖∞=1.70e-08\u{1b}[0m";
        assert_eq!(visible_len(styled), "‖ineq₊‖∞=1.70e-08".chars().count());
    }
}

fn log_sqp_status_summary(summary: &ClarabelSqpSummary, options: &ClarabelSqpOptions) {
    let eq_text = summary.equality_inf_norm.map_or_else(
        || "--".to_string(),
        |value| style_residual_text(value, options.constraint_tol),
    );
    let ineq_text = summary.inequality_inf_norm.map_or_else(
        || "--".to_string(),
        |value| style_residual_text(value, options.constraint_tol),
    );
    let comp_text = summary.complementarity_inf_norm.map_or_else(
        || "--".to_string(),
        |value| style_residual_text(value, options.complementarity_tol),
    );
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
        summary.profiling.jacobian_assembly_time,
        summary.profiling.hessian_assembly_time,
        summary.profiling.regularization_time,
        summary.profiling.subproblem_assembly_time,
        summary.profiling.qp_setup_time,
        summary.profiling.qp_solve_time,
        summary.profiling.multiplier_estimation_time,
        summary.profiling.line_search_evaluation_time,
        summary.profiling.line_search_condition_check_time,
        summary.profiling.convergence_check_time,
        summary.profiling.preprocessing_other_time,
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
        summary.profiling.preprocessing_time,
        summary.profiling.unaccounted_time,
        summary.profiling.total_time,
        summary
            .profiling
            .backend_timing
            .function_creation_time
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .backend_timing
            .derivative_generation_time
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
    let mut lines = vec![
        boxed_line(
            "result",
            format!(
                "objective={}  {}={}  {}={}  {}={}",
                sci_text(summary.objective),
                PRIMAL_INF_LABEL,
                style_residual_text(summary.primal_inf_norm, options.constraint_tol),
                DUAL_INF_LABEL,
                style_residual_text(summary.dual_inf_norm, options.dual_tol),
                OVERALL_INF_LABEL,
                style_residual_text(summary.overall_inf_norm, options.overall_tol),
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
                SQP_COMP_INF_LABEL,
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
    ];
    for (name, stat) in callback_rows {
        lines.push(boxed_line(
            "",
            callback_row(name, stat.calls, stat.total_time),
        ));
    }
    lines.push(String::new());
    lines.push(boxed_line(
        "timing",
        timing_row(
            "qp setup",
            Some(summary.profiling.qp_setups),
            summary.profiling.qp_setup_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "qp solve",
            Some(summary.profiling.qp_solves),
            summary.profiling.qp_solve_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "mult est",
            None,
            summary.profiling.multiplier_estimation_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "ls eval",
            None,
            summary.profiling.line_search_evaluation_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "ls checks",
            None,
            summary.profiling.line_search_condition_check_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row("conv check", None, summary.profiling.convergence_check_time),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "jac build",
            Some(summary.profiling.jacobian_assembly_steps),
            summary.profiling.jacobian_assembly_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "hess build",
            Some(summary.profiling.hessian_assembly_steps),
            summary.profiling.hessian_assembly_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "regularize",
            Some(summary.profiling.regularization_steps),
            summary.profiling.regularization_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "subprob asm",
            Some(summary.profiling.subproblem_assembly_steps),
            summary.profiling.subproblem_assembly_time,
        ),
    ));
    lines.push(boxed_line(
        "elastic",
        format!(
            "activations={:>4}  recovery_qps={:>4}",
            summary.profiling.elastic_recovery_activations,
            summary.profiling.elastic_recovery_qp_solves,
        ),
    ));
    if let Some(adapter_timing) = summary.profiling.adapter_timing {
        lines.push(boxed_line(
            "adapter",
            timing_row("callback", None, adapter_timing.callback_evaluation),
        ));
        lines.push(boxed_line(
            "",
            timing_row("marshal", None, adapter_timing.output_marshalling),
        ));
        lines.push(boxed_line(
            "",
            timing_row("layout", None, adapter_timing.layout_projection),
        ));
    }
    lines.push(boxed_line(
        "",
        timing_row(
            "prep other",
            Some(summary.profiling.preprocessing_other_steps),
            summary.profiling.preprocessing_other_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "preprocess",
            Some(summary.profiling.preprocessing_steps),
            summary.profiling.preprocessing_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row("unaccounted", None, summary.profiling.unaccounted_time),
    ));
    lines.push(boxed_line(
        "",
        timing_row("total", None, summary.profiling.total_time),
    ));
    lines.push(String::new());
    lines.push(boxed_line(
        "backend",
        format!(
            "create={}  derive={}  jit={}",
            fmt_optional_duration_in_unit(
                summary.profiling.backend_timing.function_creation_time,
                timing_unit,
            ),
            fmt_optional_duration_in_unit(
                summary.profiling.backend_timing.derivative_generation_time,
                timing_unit,
            ),
            fmt_optional_duration_in_unit(summary.profiling.backend_timing.jit_time, timing_unit),
        ),
    ));
    log_boxed_section("SQP converged", &lines, style_green_bold);
}

fn log_sqp_problem_header<P>(
    problem: &P,
    parameters: &[ParameterMatrix<'_>],
    options: &ClarabelSqpOptions,
) where
    P: CompiledNlpProblem,
{
    let (
        globalization_kind,
        line_search,
        trust_region,
        filter_options,
        penalty_settings,
        uses_filter,
    ) = match &options.globalization {
        SqpGlobalization::LineSearchMerit(globalization) => (
            SqpGlobalizationKind::LineSearchMerit,
            Some(&globalization.line_search),
            None,
            None,
            Some((
                globalization.penalty_increase_factor,
                globalization.max_penalty_updates,
            )),
            false,
        ),
        SqpGlobalization::LineSearchFilter(globalization) => (
            SqpGlobalizationKind::LineSearchFilter,
            Some(&globalization.line_search),
            None,
            Some(&globalization.filter),
            None,
            true,
        ),
        SqpGlobalization::TrustRegionMerit(globalization) => (
            SqpGlobalizationKind::TrustRegionMerit,
            None,
            Some(&globalization.trust_region),
            None,
            None,
            false,
        ),
        SqpGlobalization::TrustRegionFilter(globalization) => (
            SqpGlobalizationKind::TrustRegionFilter,
            None,
            Some(&globalization.trust_region),
            Some(&globalization.filter),
            None,
            true,
        ),
    };
    let exact_merit_penalty = sqp_effective_exact_merit_penalty(options);
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
        boxed_line("summary", format_sqp_settings_summary(options)),
        boxed_line(
            "globalize",
            format!(
                "mode={}  filter={}  penalty0={}  regularization={}({})  soc={}  elastic={}",
                globalization_kind.label(),
                if uses_filter { "on" } else { "off" },
                sci_text(exact_merit_penalty),
                if options.hessian_regularization_enabled {
                    "on"
                } else {
                    "off"
                },
                sci_text(options.regularization),
                if options.second_order_correction {
                    "on"
                } else {
                    "off"
                },
                if options.elastic_mode { "on" } else { "off" },
            ),
        ),
        line_search
            .map(|cfg| {
                boxed_line(
                    "line search",
                    format!(
                        "armijo_c1={}  wolfe_c2={}  beta={}  min_step={}",
                        sci_text(cfg.armijo_c1),
                        cfg.wolfe_c2
                            .map(sci_text)
                            .unwrap_or_else(|| "--".to_string()),
                        sci_text(cfg.beta),
                        sci_text(cfg.min_step),
                    ),
                )
            })
            .unwrap_or_default(),
        line_search
            .map(|cfg| boxed_line("", format!("max_line_search_steps={}", cfg.max_steps)))
            .unwrap_or_default(),
        trust_region
            .map(|cfg| {
                boxed_line(
                    "trust region",
                    format!(
                        "radius0={}  min={}  max={}  shrink={}  grow={}",
                        sci_text(cfg.initial_radius),
                        sci_text(cfg.min_radius),
                        sci_text(cfg.max_radius),
                        sci_text(cfg.shrink_factor),
                        sci_text(cfg.grow_factor),
                    ),
                )
            })
            .unwrap_or_default(),
        trust_region
            .map(|cfg| {
                boxed_line(
                    "",
                    format!(
                        "accept_ratio={}  expand_ratio={}  boundary={}  max_contract={}",
                        sci_text(cfg.accept_ratio),
                        sci_text(cfg.expand_ratio),
                        sci_text(cfg.boundary_fraction),
                        cfg.max_radius_contractions,
                    ),
                )
            })
            .unwrap_or_default(),
        filter_options
            .map(|cfg| {
                boxed_line(
                    "",
                    format!(
                        "filter_gamma_obj={}  filter_gamma_violation={}",
                        sci_text(cfg.gamma_objective),
                        sci_text(cfg.gamma_violation),
                    ),
                )
            })
            .unwrap_or_default(),
        boxed_line(
            "",
            format!(
                "elastic_weight={}  elastic_primal_reg={}  elastic_slack_reg={}",
                sci_text(options.elastic_weight),
                sci_text(options.elastic_primal_regularization),
                sci_text(options.elastic_slack_regularization),
            ),
        ),
        boxed_line(
            "",
            format!(
                "elastic_restore_reduction={}  elastic_restore_abs_tol={}  elastic_restore_max_iters={}",
                sci_text(options.elastic_restore_reduction_factor),
                sci_text(options.elastic_restore_abs_tol),
                options.elastic_restore_max_iters,
            ),
        ),
        penalty_settings
            .map(|(factor, max_updates)| {
                boxed_line(
                    "",
                    format!(
                        "factor={}  max_penalty_updates={}",
                        sci_text(factor),
                        max_updates,
                    ),
                )
            })
            .unwrap_or_default(),
        boxed_line(
            "iteration",
            format!(
                "max_iters={}  verbose={}",
                options.max_iters, options.verbose
            ),
        ),
    ];
    log_boxed_section("SQP problem / settings", &lines, style_cyan_bold);
}

fn style_residual_text(value: f64, tolerance: f64) -> String {
    let text = sci_text(value);
    style_metric_against_tolerance(&text, value, tolerance)
}

fn style_residual_cell(value: f64, tolerance: f64, is_applicable: bool) -> String {
    if !is_applicable {
        return format!("{:>9}", "--");
    }
    let cell = fmt_sci(value);
    style_metric_against_tolerance(&cell, value, tolerance)
}

fn style_line_search_iterations_cell(iterations: Option<Index>) -> String {
    let cell = fmt_line_search_iterations(iterations);
    match iterations {
        Some(iterations) if iterations >= 10 => style_red_bold(&cell),
        Some(iterations) if iterations >= 4 => style_yellow_bold(&cell),
        _ => cell,
    }
}

fn has_event(snapshot: &SqpIterationSnapshot, event: SqpIterationEvent) -> bool {
    snapshot.events.contains(&event)
}

const SQP_EVENT_COLUMN_WIDTH: usize = 10;

pub fn sqp_event_codes(snapshot: &SqpIterationSnapshot) -> String {
    let line_search = snapshot.line_search.as_ref();
    [
        if has_event(snapshot, SqpIterationEvent::PenaltyUpdated) {
            'P'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::HessianShifted) {
            'H'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::LongLineSearch) {
            'L'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::ArmijoToleranceAdjusted) {
            'A'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::SecondOrderCorrectionUsed) {
            'S'
        } else if line_search.is_some_and(|info| info.second_order_correction_attempted) {
            's'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::FilterAccepted) {
            'F'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::QpReducedAccuracy) {
            'R'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::WolfeRejectedTrial) {
            'W'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::ElasticRecoveryUsed) {
            'E'
        } else if line_search.is_some_and(|info| info.elastic_recovery_attempted) {
            'e'
        } else {
            ' '
        },
        if has_event(snapshot, SqpIterationEvent::MaxIterationsReached) {
            'M'
        } else {
            ' '
        },
    ]
    .into_iter()
    .collect()
}

fn style_event_cell(snapshot: &SqpIterationSnapshot) -> String {
    sqp_event_codes(snapshot)
        .chars()
        .map(|code| match code {
            's' | 'e' | 'M' => style_red_bold(&code.to_string()),
            'P' | 'H' | 'L' | 'A' | 'S' | 'F' | 'R' | 'W' | 'E' => {
                style_yellow_bold(&code.to_string())
            }
            _ => " ".to_string(),
        })
        .collect::<Vec<_>>()
        .join("")
}

fn sqp_event_legend_prefix() -> String {
    [
        format!("{:>5}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>9}", ""),
        format!("{:>5}", ""),
        format!("{:>width$}", "", width = SQP_EVENT_COLUMN_WIDTH),
        format!("{:>5}", ""),
        format!("{:>7}", ""),
    ]
    .join("  ")
}

fn event_legend_lines(
    snapshot: &SqpIterationSnapshot,
    state: &mut SqpEventLegendState,
) -> Vec<String> {
    let mut parts = Vec::new();
    for (code, description) in sqp_event_legend_entries(snapshot) {
        let is_new = match code {
            'P' => state.mark_penalty_if_new(),
            'H' => state.mark_hessian_shift_if_new(),
            'L' => state.mark_line_search_if_new(),
            'A' => state.mark_armijo_adjust_if_new(),
            'S' => state.mark_soc_if_new(),
            's' => state.mark_soc_attempted_if_new(),
            'F' => state.mark_filter_if_new(),
            'R' => state.mark_qp_if_new(),
            'W' => state.mark_wolfe_if_new(),
            'E' => state.mark_elastic_if_new(),
            'e' => state.mark_elastic_attempted_if_new(),
            'M' => state.mark_max_iter_if_new(),
            _ => false,
        };
        if is_new {
            parts.push(description);
        }
    }

    let prefix = sqp_event_legend_prefix();
    parts
        .into_iter()
        .map(|part| format!("{prefix}  {part}"))
        .collect()
}

fn log_sqp_iteration(
    snapshot: &SqpIterationSnapshot,
    options: &ClarabelSqpOptions,
    event_state: &mut SqpEventLegendState,
) {
    if snapshot.iteration.is_multiple_of(10) {
        eprintln!();
        let header = [
            format!("{:>5}", "iter"),
            format!("{:>9}", "f"),
            format!("{:>9}", EQ_INF_LABEL),
            format!("{:>9}", INEQ_INF_LABEL),
            format!("{:>9}", DUAL_INF_LABEL),
            format!("{:>9}", SQP_COMP_INF_LABEL),
            format!("{:>9}", OVERALL_INF_LABEL),
            format!("{:>9}", STEP_INF_LABEL),
            format!("{:>9}", "penalty"),
            format!("{:>9}", "α"),
            format!("{:>5}", "ls_it"),
            format!("{:>width$}", "evt", width = SQP_EVENT_COLUMN_WIDTH),
            format!("{:>5}", "qp_it"),
            format!("{:>7}", "qp_time"),
        ];
        eprintln!("{}", style_bold(&header.join("  ")));
    }
    let line_search = snapshot.line_search.as_ref();
    let qp = snapshot.qp.as_ref();
    let row = [
        style_iteration_cell(snapshot),
        fmt_sci(snapshot.objective),
        style_residual_cell(
            snapshot.eq_inf.unwrap_or(0.0),
            options.constraint_tol,
            snapshot.eq_inf.is_some(),
        ),
        style_residual_cell(
            snapshot.ineq_inf.unwrap_or(0.0),
            options.constraint_tol,
            snapshot.ineq_inf.is_some(),
        ),
        style_residual_cell(snapshot.dual_inf, options.dual_tol, true),
        style_residual_cell(
            snapshot.comp_inf.unwrap_or(0.0),
            options.complementarity_tol,
            snapshot.comp_inf.is_some(),
        ),
        style_residual_cell(snapshot.overall_inf, options.overall_tol, true),
        fmt_optional_sci(snapshot.step_inf),
        fmt_sci(snapshot.penalty),
        fmt_alpha(line_search.map(|info| info.accepted_alpha)),
        style_line_search_iterations_cell(line_search.map(|info| info.backtrack_count)),
        style_event_cell(snapshot),
        fmt_qp_iterations(qp.and_then(|info| u32::try_from(info.iteration_count).ok())),
        fmt_qp_time(qp.map(|info| info.solve_time.as_secs_f64())),
    ];
    eprintln!("{}", row.join("  "));
    for legend_line in event_legend_lines(snapshot, event_state) {
        eprintln!("{legend_line}");
    }
}

fn validate_finite_inputs(
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
) -> std::result::Result<(), ClarabelSqpError> {
    if x0.iter().any(|value| !value.is_finite()) {
        return Err(ClarabelSqpError::NonFiniteInput {
            stage: NonFiniteInputStage::InitialGuess,
        });
    }
    for (parameter_index, parameter) in parameters.iter().enumerate() {
        if parameter.values.iter().any(|value| !value.is_finite()) {
            return Err(ClarabelSqpError::NonFiniteInput {
                stage: NonFiniteInputStage::ParameterValues { parameter_index },
            });
        }
    }
    Ok(())
}

fn validate_sqp_options(options: &ClarabelSqpOptions) -> std::result::Result<(), ClarabelSqpError> {
    if !options.overall_tol.is_finite() || options.overall_tol < 0.0 {
        return Err(ClarabelSqpError::InvalidInput(format!(
            "overall_tol must be finite and non-negative, got {}",
            options.overall_tol
        )));
    }
    if !options.overall_scale_max.is_finite() || options.overall_scale_max <= 0.0 {
        return Err(ClarabelSqpError::InvalidInput(format!(
            "overall_scale_max must be finite and positive, got {}",
            options.overall_scale_max
        )));
    }
    Ok(())
}

fn validate_finite_scalar_output(
    value: f64,
    stage: NonFiniteCallbackStage,
    current_state: Option<&SqpIterationSnapshot>,
    last_accepted_state: Option<&SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> std::result::Result<f64, ClarabelSqpError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: Box::new(SqpFailureContext {
                termination: SqpTermination::NonFiniteCallbackOutput,
                final_state: current_state.cloned(),
                final_state_kind: current_state.map(|_| SqpFinalStateKind::AcceptedIterate),
                last_accepted_state: last_accepted_state.cloned(),
                failed_line_search: None,
                failed_trust_region: None,
                failed_step_diagnostics: None,
                qp_failure: None,
                profiling: profiling.clone(),
            }),
        })
    }
}

fn validate_finite_slice_output(
    values: &[f64],
    stage: NonFiniteCallbackStage,
    current_state: Option<&SqpIterationSnapshot>,
    last_accepted_state: Option<&SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> std::result::Result<(), ClarabelSqpError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: Box::new(SqpFailureContext {
                termination: SqpTermination::NonFiniteCallbackOutput,
                final_state: current_state.cloned(),
                final_state_kind: current_state.map(|_| SqpFinalStateKind::AcceptedIterate),
                last_accepted_state: last_accepted_state.cloned(),
                failed_line_search: None,
                failed_trust_region: None,
                failed_step_diagnostics: None,
                qp_failure: None,
                profiling: profiling.clone(),
            }),
        })
    }
}

fn qp_status_info(
    raw_status: SolverStatus,
    setup_time: Duration,
    solve_time: Duration,
    iteration_count: u32,
) -> SqpQpInfo {
    let status = match raw_status {
        SolverStatus::Solved => SqpQpStatus::Solved,
        SolverStatus::AlmostSolved => SqpQpStatus::ReducedAccuracy,
        _ => SqpQpStatus::Failed,
    };
    SqpQpInfo {
        status,
        raw_status: raw_status.into(),
        setup_time,
        solve_time,
        iteration_count: iteration_count as Index,
    }
}

#[derive(Clone, Debug)]
struct RawQpSolve {
    primal: Vec<f64>,
    dual: Vec<f64>,
    solver_status: SolverStatus,
    qp_info: SqpQpInfo,
    qp_failure: Option<SqpQpFailureDiagnostics>,
}

#[derive(Clone, Debug)]
struct SqpSubproblemSolution {
    step: Vec<f64>,
    equality_multipliers: Vec<f64>,
    inequality_multipliers: Vec<f64>,
    lower_bound_multipliers: Vec<f64>,
    upper_bound_multipliers: Vec<f64>,
    qp_info: SqpQpInfo,
    elastic_recovery_used: bool,
}

#[derive(Clone, Debug)]
struct SqpAssembledSubproblem {
    constraint_matrix: DMatrix<f64>,
    rhs: Vec<f64>,
    cones: Vec<clarabel::solver::SupportedConeT<f64>>,
    trust_region_dim: Option<Index>,
}

struct QpSolveContext<'a> {
    profiling: &'a mut ClarabelSqpProfiling,
    iteration_qp_setup_time: &'a mut Duration,
    iteration_qp_solve_time: &'a mut Duration,
}

struct ElasticRecoveryModel<'a> {
    hessian: &'a DMatrix<f64>,
    gradient: &'a [f64],
    equality_values: &'a [f64],
    equality_jacobian: &'a DMatrix<f64>,
    nonlinear_inequality_values: &'a [f64],
    nonlinear_inequality_jacobian: &'a DMatrix<f64>,
    augmented_inequality_values: &'a [f64],
    bound_jacobian: &'a DMatrix<f64>,
}

fn assemble_sqp_subproblem(
    equality_jacobian: &DMatrix<f64>,
    inequality_jacobian: &DMatrix<f64>,
    equality_values: &[f64],
    augmented_inequality_values: &[f64],
    trust_region_radius: Option<f64>,
) -> SqpAssembledSubproblem {
    let n = equality_jacobian.ncols().max(inequality_jacobian.ncols());
    let equality_count = equality_values.len();
    let augmented_inequality_count = augmented_inequality_values.len();
    let trust_region_dim = trust_region_radius.map(|_| n + 1);
    let total_rows = equality_count + augmented_inequality_count + trust_region_dim.unwrap_or(0);
    let mut constraint_matrix = DMatrix::<f64>::zeros(total_rows, n);
    let mut rhs = Vec::with_capacity(total_rows);
    let mut cones = Vec::with_capacity(3);
    let mut row_cursor = 0;

    if equality_count > 0 {
        constraint_matrix
            .view_mut((row_cursor, 0), (equality_count, n))
            .copy_from(equality_jacobian);
        rhs.extend(equality_values.iter().map(|value| -value));
        cones.push(ZeroConeT(equality_count));
        row_cursor += equality_count;
    }
    if augmented_inequality_count > 0 {
        constraint_matrix
            .view_mut((row_cursor, 0), (augmented_inequality_count, n))
            .copy_from(inequality_jacobian);
        rhs.extend(augmented_inequality_values.iter().map(|value| -value));
        cones.push(NonnegativeConeT(augmented_inequality_count));
        row_cursor += augmented_inequality_count;
    }
    if let Some(radius) = trust_region_radius {
        rhs.push(radius);
        rhs.extend(std::iter::repeat_n(0.0, n));
        for col in 0..n {
            constraint_matrix[(row_cursor + 1 + col, col)] = 1.0;
        }
        cones.push(SecondOrderConeT(n + 1));
    }

    SqpAssembledSubproblem {
        constraint_matrix,
        rhs,
        cones,
        trust_region_dim,
    }
}

fn solve_clarabel_qp_from_dense(
    hessian: &DMatrix<f64>,
    linear_objective: &[f64],
    constraint_matrix: &DMatrix<f64>,
    rhs: &[f64],
    cones: &[clarabel::solver::SupportedConeT<f64>],
    elastic_recovery: bool,
    qp_ctx: &mut QpSolveContext<'_>,
) -> std::result::Result<RawQpSolve, ClarabelSqpError> {
    let qp_setup_started = Instant::now();
    let p = dense_to_csc_upper(hessian);
    let a = dense_to_csc(constraint_matrix);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
    let mut solver = DefaultSolver::new(&p, linear_objective, &a, rhs, cones, settings)
        .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
    let qp_setup_elapsed = qp_setup_started.elapsed();
    qp_ctx.profiling.qp_setups += 1;
    qp_ctx.profiling.qp_setup_time += qp_setup_elapsed;
    *qp_ctx.iteration_qp_setup_time += qp_setup_elapsed;

    let qp_solve_started = Instant::now();
    solver.solve();
    let qp_solve_elapsed = qp_solve_started.elapsed();
    qp_ctx.profiling.qp_solves += 1;
    qp_ctx.profiling.qp_solve_time += qp_solve_elapsed;
    *qp_ctx.iteration_qp_solve_time += qp_solve_elapsed;

    let qp_info = qp_status_info(
        solver.solution.status,
        qp_setup_elapsed,
        qp_solve_elapsed,
        solver.solution.iterations,
    );
    let qp_failure = (qp_info.status == SqpQpStatus::Failed).then(|| {
        build_qp_failure_diagnostics(
            qp_info.clone(),
            hessian,
            linear_objective,
            constraint_matrix,
            rhs,
            cones,
            elastic_recovery,
        )
    });

    Ok(RawQpSolve {
        primal: solver.solution.x.clone(),
        dual: solver.solution.z.clone(),
        solver_status: solver.solution.status,
        qp_info,
        qp_failure,
    })
}

fn should_try_elastic_recovery(
    status: SolverStatus,
    equality_count: Index,
    nonlinear_inequality_count: Index,
    bound_count: Index,
    options: &ClarabelSqpOptions,
) -> bool {
    options.restoration_phase
        && options.elastic_mode
        && has_restoration_constraints(equality_count, nonlinear_inequality_count, bound_count)
        && matches!(
            status,
            SolverStatus::PrimalInfeasible
                | SolverStatus::AlmostPrimalInfeasible
                | SolverStatus::NumericalError
                | SolverStatus::InsufficientProgress
        )
}

fn has_restoration_constraints(
    equality_count: Index,
    nonlinear_inequality_count: Index,
    bound_count: Index,
) -> bool {
    equality_count > 0 || nonlinear_inequality_count > 0 || bound_count > 0
}

#[expect(
    clippy::too_many_arguments,
    reason = "restoration solves need explicit dimensions and context for failure reporting"
)]
fn solve_restoration_subproblem(
    elastic_model: &ElasticRecoveryModel<'_>,
    options: &ClarabelSqpOptions,
    qp_ctx: &mut QpSolveContext<'_>,
    step_dimension: Index,
    equality_count: Index,
    inequality_count: Index,
    lower_bound_count: Index,
    current_snapshot: &SqpIterationSnapshot,
    last_accepted_state: &Option<SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> std::result::Result<SqpSubproblemSolution, ClarabelSqpError> {
    let elastic_qp = solve_elastic_recovery_qp(elastic_model, options, qp_ctx)?;
    match elastic_qp.solver_status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(decode_elastic_qp_solution(
            elastic_qp,
            step_dimension,
            equality_count,
            inequality_count,
            lower_bound_count,
        )),
        status => Err(ClarabelSqpError::QpSolve {
            status: status.into(),
            context: failure_context_with_qp_failure(
                SqpTermination::QpSolve,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                None,
                None,
                None,
                elastic_qp.qp_failure,
                profiling,
            ),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_sqp_subproblem(
    hessian: &DMatrix<f64>,
    gradient: &[f64],
    assembled_subproblem: &SqpAssembledSubproblem,
    elastic_model: &ElasticRecoveryModel<'_>,
    options: &ClarabelSqpOptions,
    qp_ctx: &mut QpSolveContext<'_>,
    step_dimension: Index,
    equality_count: Index,
    inequality_count: Index,
    lower_bound_count: Index,
    current_snapshot: &SqpIterationSnapshot,
    last_accepted_state: &Option<SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> std::result::Result<SqpSubproblemSolution, ClarabelSqpError> {
    let unconstrained = assembled_subproblem.constraint_matrix.nrows() == 0;
    if unconstrained {
        return Ok(SqpSubproblemSolution {
            step: solve_unconstrained_quadratic_step(hessian, gradient).map_err(|()| {
                ClarabelSqpError::UnconstrainedStepSolve {
                    context: failure_context(
                        SqpTermination::QpSolve,
                        Some(current_snapshot.clone()),
                        last_accepted_state.clone(),
                        profiling,
                    ),
                }
            })?,
            equality_multipliers: Vec::new(),
            inequality_multipliers: Vec::new(),
            lower_bound_multipliers: Vec::new(),
            upper_bound_multipliers: Vec::new(),
            qp_info: SqpQpInfo {
                status: SqpQpStatus::Solved,
                raw_status: SqpQpRawStatus::Solved,
                setup_time: Duration::ZERO,
                solve_time: Duration::ZERO,
                iteration_count: 0,
            },
            elastic_recovery_used: false,
        });
    }

    let normal_qp = solve_clarabel_qp_from_dense(
        hessian,
        gradient,
        &assembled_subproblem.constraint_matrix,
        &assembled_subproblem.rhs,
        &assembled_subproblem.cones,
        false,
        qp_ctx,
    )?;
    match normal_qp.solver_status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(decode_normal_qp_solution(
            normal_qp,
            equality_count,
            inequality_count,
            lower_bound_count,
            assembled_subproblem.trust_region_dim,
        )),
        status
            if (!matches!(status, SolverStatus::NumericalError)
                || assembled_subproblem.trust_region_dim.is_none())
                && should_try_elastic_recovery(
                    status,
                    equality_count,
                    inequality_count,
                    elastic_model.augmented_inequality_values.len() - inequality_count,
                    options,
                ) =>
        {
            qp_ctx.profiling.elastic_recovery_activations += 1;
            solve_restoration_subproblem(
                elastic_model,
                options,
                qp_ctx,
                step_dimension,
                equality_count,
                inequality_count,
                lower_bound_count,
                current_snapshot,
                last_accepted_state,
                profiling,
            )
        }
        status => Err(ClarabelSqpError::QpSolve {
            status: status.into(),
            context: failure_context_with_qp_failure(
                SqpTermination::QpSolve,
                Some(current_snapshot.clone()),
                last_accepted_state.clone(),
                None,
                None,
                None,
                normal_qp.qp_failure,
                profiling,
            ),
        }),
    }
}

fn decode_normal_qp_solution(
    raw: RawQpSolve,
    equality_count: Index,
    nonlinear_inequality_count: Index,
    lower_bound_count: Index,
    trust_region_dim: Option<Index>,
) -> SqpSubproblemSolution {
    let trust_region_rows = trust_region_dim.unwrap_or(0);
    let constraint_duals = &raw.dual[..raw.dual.len().saturating_sub(trust_region_rows)];
    let (equality_multipliers, candidate_augmented_inequality_multipliers) =
        split_multipliers(constraint_duals, equality_count);
    let (inequality_multipliers, lower_bound_multipliers, upper_bound_multipliers) =
        split_augmented_inequality_multipliers(
            &candidate_augmented_inequality_multipliers,
            nonlinear_inequality_count,
            lower_bound_count,
        );
    SqpSubproblemSolution {
        step: raw.primal,
        equality_multipliers,
        inequality_multipliers,
        lower_bound_multipliers,
        upper_bound_multipliers,
        qp_info: raw.qp_info,
        elastic_recovery_used: false,
    }
}

fn decode_elastic_qp_solution(
    raw: RawQpSolve,
    step_dimension: Index,
    equality_count: Index,
    nonlinear_inequality_count: Index,
    lower_bound_count: Index,
) -> SqpSubproblemSolution {
    let mut cursor = 0;
    let equality_upper = &raw.dual[cursor..cursor + equality_count];
    cursor += equality_count;
    let equality_lower = &raw.dual[cursor..cursor + equality_count];
    cursor += equality_count;
    let inequality_multipliers = raw.dual[cursor..cursor + nonlinear_inequality_count].to_vec();
    cursor += nonlinear_inequality_count;
    let augmented_bound_multipliers = &raw.dual[cursor..];
    let (lower_bound_multipliers, upper_bound_multipliers) =
        augmented_bound_multipliers.split_at(lower_bound_count);
    let equality_multipliers = equality_upper
        .iter()
        .zip(equality_lower.iter())
        .map(|(upper, lower)| upper - lower)
        .collect();
    SqpSubproblemSolution {
        step: raw.primal[..step_dimension].to_vec(),
        equality_multipliers,
        inequality_multipliers,
        lower_bound_multipliers: lower_bound_multipliers.to_vec(),
        upper_bound_multipliers: upper_bound_multipliers.to_vec(),
        qp_info: raw.qp_info,
        elastic_recovery_used: true,
    }
}

fn solve_elastic_recovery_qp(
    model: &ElasticRecoveryModel<'_>,
    options: &ClarabelSqpOptions,
    qp_ctx: &mut QpSolveContext<'_>,
) -> std::result::Result<RawQpSolve, ClarabelSqpError> {
    // Follow SNOPT-style elastic mode precedent: add nonnegative elastic variables to the
    // linearized nonlinear constraints, penalize their L1 norm in the QP objective, keep simple
    // bounds hard, and return to the normal SQP model on the next major iteration.
    let step_dimension = model.gradient.len();
    let equality_count = model.equality_values.len();
    let nonlinear_inequality_count = model.nonlinear_inequality_values.len();
    let bound_count = model.augmented_inequality_values.len() - nonlinear_inequality_count;
    let total_elastic = equality_count + nonlinear_inequality_count;
    let variable_count = step_dimension + total_elastic;
    let equality_elastic_offset = step_dimension;
    let inequality_elastic_offset = step_dimension + equality_count;

    let mut elastic_hessian = DMatrix::<f64>::zeros(variable_count, variable_count);
    for row in 0..step_dimension {
        for col in 0..step_dimension {
            elastic_hessian[(row, col)] = model.hessian[(row, col)];
        }
        elastic_hessian[(row, row)] += options.elastic_primal_regularization;
    }
    for idx in step_dimension..variable_count {
        elastic_hessian[(idx, idx)] = options.elastic_slack_regularization;
    }

    let mut linear_objective = model.gradient.to_vec();
    linear_objective.extend(std::iter::repeat_n(options.elastic_weight, total_elastic));

    let total_rows = 2 * equality_count
        + nonlinear_inequality_count
        + bound_count
        + equality_count
        + nonlinear_inequality_count;
    let mut constraint_matrix = DMatrix::<f64>::zeros(total_rows, variable_count);
    let mut rhs = vec![0.0; total_rows];
    let mut cones = Vec::new();
    let mut row = 0;

    if equality_count > 0 {
        for eq_row in 0..equality_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + eq_row, col)] = model.equality_jacobian[(eq_row, col)];
            }
            constraint_matrix[(row + eq_row, equality_elastic_offset + eq_row)] = -1.0;
            rhs[row + eq_row] = -model.equality_values[eq_row];
        }
        cones.push(NonnegativeConeT(equality_count));
        row += equality_count;

        for eq_row in 0..equality_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + eq_row, col)] = -model.equality_jacobian[(eq_row, col)];
            }
            constraint_matrix[(row + eq_row, equality_elastic_offset + eq_row)] = -1.0;
            rhs[row + eq_row] = model.equality_values[eq_row];
        }
        cones.push(NonnegativeConeT(equality_count));
        row += equality_count;
    }

    if nonlinear_inequality_count > 0 {
        for ineq_row in 0..nonlinear_inequality_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + ineq_row, col)] =
                    model.nonlinear_inequality_jacobian[(ineq_row, col)];
            }
            constraint_matrix[(row + ineq_row, inequality_elastic_offset + ineq_row)] = -1.0;
            rhs[row + ineq_row] = -model.nonlinear_inequality_values[ineq_row];
        }
        cones.push(NonnegativeConeT(nonlinear_inequality_count));
        row += nonlinear_inequality_count;
    }

    if bound_count > 0 {
        for bound_row in 0..bound_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + bound_row, col)] = model.bound_jacobian[(bound_row, col)];
            }
            rhs[row + bound_row] =
                -model.augmented_inequality_values[nonlinear_inequality_count + bound_row];
        }
        cones.push(NonnegativeConeT(bound_count));
        row += bound_count;
    }

    if equality_count > 0 {
        for eq_row in 0..equality_count {
            constraint_matrix[(row + eq_row, equality_elastic_offset + eq_row)] = -1.0;
        }
        cones.push(NonnegativeConeT(equality_count));
        row += equality_count;
    }

    if nonlinear_inequality_count > 0 {
        for ineq_row in 0..nonlinear_inequality_count {
            constraint_matrix[(row + ineq_row, inequality_elastic_offset + ineq_row)] = -1.0;
        }
        cones.push(NonnegativeConeT(nonlinear_inequality_count));
    }

    let solve = solve_clarabel_qp_from_dense(
        &elastic_hessian,
        &linear_objective,
        &constraint_matrix,
        &rhs,
        &cones,
        true,
        qp_ctx,
    )?;
    qp_ctx.profiling.elastic_recovery_qp_solves += 1;
    Ok(solve)
}

pub fn solve_nlp_sqp<P>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &ClarabelSqpOptions,
) -> std::result::Result<ClarabelSqpSummary, ClarabelSqpError>
where
    P: CompiledNlpProblem,
{
    solve_nlp_sqp_with_callback(problem, x0, parameters, options, |_| {})
}

pub fn solve_nlp_sqp_with_callback<P, C>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &ClarabelSqpOptions,
    mut callback: C,
) -> std::result::Result<ClarabelSqpSummary, ClarabelSqpError>
where
    P: CompiledNlpProblem,
    C: FnMut(&SqpIterationSnapshot),
{
    let solve_started = Instant::now();
    let mut profiling = ClarabelSqpProfiling {
        backend_timing: problem.backend_timing_metadata(),
        ..ClarabelSqpProfiling::default()
    };
    let validation_started = Instant::now();
    validate_nlp_problem_shapes(problem)
        .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
    validate_parameter_inputs(problem, parameters)
        .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
    validate_sqp_options(options)?;
    validate_finite_inputs(x0, parameters)?;
    let validation_elapsed = validation_started.elapsed();
    profiling.preprocessing_steps += 1;
    profiling.preprocessing_other_steps += 1;
    profiling.preprocessing_other_time += validation_elapsed;
    profiling.preprocessing_time += validation_elapsed;
    let n = problem.dimension();
    if x0.len() != n {
        return Err(ClarabelSqpError::InvalidInput(format!(
            "x0 has length {}, expected {n}",
            x0.len()
        )));
    }

    let equality_count = problem.equality_count();
    let inequality_count = problem.inequality_count();
    let bounds = collect_bound_constraints(problem)?;
    let lower_bound_count = bounds.lower_indices.len();
    let augmented_inequality_count = inequality_count + bounds.total_count();
    let total_constraint_count = equality_count + augmented_inequality_count;
    let bound_jacobian = build_bound_jacobian(&bounds, n);
    let mut x = x0.to_vec();
    let mut gradient = vec![0.0; n];
    let mut equality_values = vec![0.0; equality_count];
    let mut inequality_values = vec![0.0; inequality_count];
    let mut augmented_inequality_values = vec![0.0; augmented_inequality_count];
    let mut hessian_values = vec![0.0; problem.lagrangian_hessian_ccs().nnz()];
    let mut equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    let mut trial_equality_values = vec![0.0; equality_count];
    let mut trial_inequality_values = vec![0.0; inequality_count];
    let mut trial_augmented_inequality_values = vec![0.0; augmented_inequality_count];
    let mut trial_gradient = vec![0.0; n];
    let mut trial_equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut trial_inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    let mut equality_multipliers = vec![0.0; equality_count];
    let mut inequality_multipliers = vec![0.0; inequality_count];
    let mut lower_bound_multipliers = vec![0.0; lower_bound_count];
    let mut upper_bound_multipliers = vec![0.0; bounds.upper_indices.len()];
    let globalization_kind = match &options.globalization {
        SqpGlobalization::LineSearchMerit(_) => SqpGlobalizationKind::LineSearchMerit,
        SqpGlobalization::LineSearchFilter(_) => SqpGlobalizationKind::LineSearchFilter,
        SqpGlobalization::TrustRegionMerit(_) => SqpGlobalizationKind::TrustRegionMerit,
        SqpGlobalization::TrustRegionFilter(_) => SqpGlobalizationKind::TrustRegionFilter,
    };
    let line_search_options = match &options.globalization {
        SqpGlobalization::LineSearchMerit(globalization) => Some(&globalization.line_search),
        SqpGlobalization::LineSearchFilter(globalization) => Some(&globalization.line_search),
        _ => None,
    };
    let trust_region_options = match &options.globalization {
        SqpGlobalization::TrustRegionMerit(globalization) => Some(&globalization.trust_region),
        SqpGlobalization::TrustRegionFilter(globalization) => Some(&globalization.trust_region),
        _ => None,
    };
    let filter_settings = match &options.globalization {
        SqpGlobalization::LineSearchFilter(globalization) => Some(&globalization.filter),
        SqpGlobalization::TrustRegionFilter(globalization) => Some(&globalization.filter),
        _ => None,
    };
    let line_search_penalty_update_settings = match &options.globalization {
        SqpGlobalization::LineSearchMerit(globalization) => Some((
            globalization.penalty_increase_factor,
            globalization.max_penalty_updates,
        )),
        _ => None,
    };
    let exact_merit_penalty = sqp_effective_exact_merit_penalty(options);
    let uses_filter = filter_settings.is_some();
    let is_trust_region = trust_region_options.is_some();
    let min_line_search_step = line_search_options.map(|line_search| line_search.min_step);
    let filter_switching_reference_min =
        filter_settings.map(|filter| filter.switching_reference_min);
    let filter_theta_max_factor = filter_settings.map(|filter| filter.theta_max_factor);
    let mut merit_penalty = exact_merit_penalty;
    let mut filter_entries = Vec::new();
    let mut event_state = SqpEventLegendState::default();
    let mut previous_step_inf = None;
    let mut previous_line_search = None;
    let mut previous_trust_region = None;
    let mut previous_step_diagnostics = None;
    let mut previous_qp = None;
    let mut previous_elastic_recovery_used = false;
    let mut previous_events = Vec::new();
    let mut last_accepted_state = None;
    let mut filter_theta_reference = filter_switching_reference_min
        .unwrap_or(0.0)
        .max(options.constraint_tol);
    let mut filter_theta_max = filter_theta_max_factor.unwrap_or(0.0) * filter_theta_reference;
    let mut trust_region_radius =
        trust_region_options.map(|trust_region| trust_region.initial_radius);
    profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();

    if options.verbose {
        log_sqp_problem_header(problem, parameters, options);
    }

    for iteration in 0..options.max_iters.max(1) {
        let iteration_started = Instant::now();
        profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();
        let iteration_timing_baseline = IterationTimingBaseline::capture(&profiling);
        let mut iteration_callback_time = Duration::ZERO;
        let mut iteration_preprocess_time = Duration::ZERO;
        let mut iteration_jacobian_assembly_time = Duration::ZERO;
        let mut iteration_hessian_assembly_time = Duration::ZERO;
        let mut iteration_regularization_time = Duration::ZERO;
        let mut iteration_subproblem_assembly_time = Duration::ZERO;
        let mut iteration_qp_setup_time = Duration::ZERO;
        let mut iteration_qp_solve_time = Duration::ZERO;
        let mut iteration_multiplier_estimation_time = Duration::ZERO;
        let mut iteration_line_search_evaluation_time = Duration::ZERO;
        let mut iteration_line_search_condition_check_time = Duration::ZERO;
        let mut iteration_convergence_check_time = Duration::ZERO;
        let evaluation_started = Instant::now();
        let objective_value_result = (|| -> std::result::Result<f64, ClarabelSqpError> {
            let objective_value = validate_finite_scalar_output(
                time_callback(
                    &mut profiling.objective_value,
                    &mut iteration_callback_time,
                    || problem.objective_value(&x, parameters),
                ),
                NonFiniteCallbackStage::ObjectiveValue,
                None,
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            time_callback(
                &mut profiling.objective_gradient,
                &mut iteration_callback_time,
                || problem.objective_gradient(&x, parameters, &mut gradient),
            );
            validate_finite_slice_output(
                &gradient,
                NonFiniteCallbackStage::ObjectiveGradient,
                None,
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            time_callback(
                &mut profiling.equality_values,
                &mut iteration_callback_time,
                || problem.equality_values(&x, parameters, &mut equality_values),
            );
            validate_finite_slice_output(
                &equality_values,
                NonFiniteCallbackStage::EqualityValues,
                None,
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            time_callback(
                &mut profiling.inequality_values,
                &mut iteration_callback_time,
                || problem.inequality_values(&x, parameters, &mut inequality_values),
            );
            validate_finite_slice_output(
                &inequality_values,
                NonFiniteCallbackStage::InequalityValues,
                None,
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            augment_inequality_values(
                &inequality_values,
                &x,
                &bounds,
                &mut augmented_inequality_values,
            );
            time_callback(
                &mut profiling.equality_jacobian_values,
                &mut iteration_callback_time,
                || problem.equality_jacobian_values(&x, parameters, &mut equality_jacobian_values),
            );
            validate_finite_slice_output(
                &equality_jacobian_values,
                NonFiniteCallbackStage::EqualityJacobianValues,
                None,
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            time_callback(
                &mut profiling.inequality_jacobian_values,
                &mut iteration_callback_time,
                || {
                    problem.inequality_jacobian_values(
                        &x,
                        parameters,
                        &mut inequality_jacobian_values,
                    )
                },
            );
            validate_finite_slice_output(
                &inequality_jacobian_values,
                NonFiniteCallbackStage::InequalityJacobianValues,
                None,
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            Ok(objective_value)
        })();
        profiling.evaluation_time += evaluation_started.elapsed();
        let objective_value = objective_value_result?;
        let (equality_jacobian, nonlinear_inequality_jacobian, inequality_jacobian);
        let (
            equality_inf,
            inequality_inf,
            primal_inf,
            current_filter_entry,
            dual_inf,
            complementarity_inf,
            overall_inf,
        );
        {
            let preprocess_started = Instant::now();
            let jacobian_assembly_started = Instant::now();
            equality_jacobian =
                ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
            nonlinear_inequality_jacobian = ccs_to_dense(
                problem.inequality_jacobian_ccs(),
                &inequality_jacobian_values,
            );
            inequality_jacobian = stack_jacobians(&nonlinear_inequality_jacobian, &bound_jacobian);
            let jacobian_assembly_elapsed = jacobian_assembly_started.elapsed();
            iteration_jacobian_assembly_time += jacobian_assembly_elapsed;
            record_iteration_duration(
                &mut profiling.jacobian_assembly_steps,
                &mut profiling.jacobian_assembly_time,
                jacobian_assembly_elapsed,
            );
            equality_inf = inf_norm(&equality_values);
            inequality_inf = positive_part_inf_norm(&augmented_inequality_values);
            primal_inf = equality_inf.max(inequality_inf);
            current_filter_entry = filter::entry(objective_value, primal_inf);
            if uses_filter && filter_entries.is_empty() {
                filter_theta_reference = sqp_filter_theta_reference(primal_inf, options);
                filter_theta_max = filter_theta_max_factor.unwrap_or(0.0) * filter_theta_reference;
                filter::update_frontier(&mut filter_entries, current_filter_entry.clone());
            }
            if previous_elastic_recovery_used
                && augmented_inequality_count == 0
                && equality_count > 0
                && let Some(estimate) = {
                    let multiplier_estimation_started = Instant::now();
                    let estimate = estimate_equality_multipliers(&gradient, &equality_jacobian);
                    let multiplier_estimation_elapsed = multiplier_estimation_started.elapsed();
                    profiling.multiplier_estimations += 1;
                    iteration_multiplier_estimation_time += multiplier_estimation_elapsed;
                    profiling.multiplier_estimation_time += multiplier_estimation_elapsed;
                    estimate
                }
            {
                equality_multipliers = estimate;
            }
            let all_inequality_multipliers = [
                inequality_multipliers.as_slice(),
                lower_bound_multipliers.as_slice(),
                upper_bound_multipliers.as_slice(),
            ]
            .concat();
            let all_dual_multipliers = [
                equality_multipliers.as_slice(),
                all_inequality_multipliers.as_slice(),
            ]
            .concat();
            let dual_residual = lagrangian_gradient(
                &gradient,
                &equality_jacobian,
                &equality_multipliers,
                &inequality_jacobian,
                &all_inequality_multipliers,
            );
            dual_inf = inf_norm(&dual_residual);
            complementarity_inf =
                complementarity_inf_norm(&augmented_inequality_values, &all_inequality_multipliers);
            overall_inf = scaled_overall_inf_norm(
                primal_inf,
                dual_inf,
                complementarity_inf,
                &all_dual_multipliers,
                &all_inequality_multipliers,
                options.overall_scale_max,
            );
            let preprocess_elapsed = preprocess_started.elapsed();
            iteration_preprocess_time += preprocess_elapsed;
            profiling.preprocessing_time += preprocess_elapsed;
        }
        let converged;
        {
            let convergence_started = Instant::now();
            let convergence_check_started = Instant::now();
            converged = overall_inf <= options.overall_tol
                && primal_inf <= options.constraint_tol
                && dual_inf <= options.dual_tol
                && complementarity_inf <= options.complementarity_tol;
            let convergence_check_elapsed = convergence_check_started.elapsed();
            profiling.convergence_checks += 1;
            iteration_convergence_check_time += convergence_check_elapsed;
            profiling.convergence_check_time += convergence_check_elapsed;
            profiling.convergence_time += convergence_started.elapsed();
        }
        let current_iteration_elapsed = iteration_started.elapsed();
        let iteration_preprocess_other_time = iteration_preprocess_time.saturating_sub(
            iteration_jacobian_assembly_time
                + iteration_hessian_assembly_time
                + iteration_regularization_time
                + iteration_subproblem_assembly_time,
        );
        profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();
        let phase = if iteration == 0 {
            SqpIterationPhase::Initial
        } else if converged {
            SqpIterationPhase::PostConvergence
        } else {
            SqpIterationPhase::AcceptedStep
        };
        let current_snapshot = SqpIterationSnapshot {
            iteration,
            phase,
            globalization: globalization_kind,
            x: x.clone(),
            objective: objective_value,
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
            overall_inf,
            step_inf: previous_step_inf,
            penalty: merit_penalty,
            line_search: previous_line_search.clone(),
            trust_region: previous_trust_region.clone(),
            step_diagnostics: previous_step_diagnostics.clone(),
            filter: uses_filter.then(|| SqpFilterInfo {
                current: current_filter_entry.clone(),
                entries: filter_entries.clone(),
                accepted_mode: accepted_filter_mode(
                    previous_line_search.as_ref(),
                    previous_trust_region.as_ref(),
                ),
            }),
            qp: previous_qp.clone(),
            timing: build_iteration_timing(
                iteration_timing_baseline,
                &profiling,
                IterationTimingBuckets {
                    multiplier_estimation: iteration_multiplier_estimation_time,
                    line_search_evaluation: iteration_line_search_evaluation_time,
                    line_search_condition_checks: iteration_line_search_condition_check_time,
                    convergence_check: iteration_convergence_check_time,
                    jacobian_assembly: iteration_jacobian_assembly_time,
                    hessian_assembly: iteration_hessian_assembly_time,
                    regularization: iteration_regularization_time,
                    subproblem_assembly: iteration_subproblem_assembly_time,
                    preprocess_other: iteration_preprocess_other_time,
                    total: current_iteration_elapsed,
                    ..IterationTimingBuckets::default()
                },
            ),
            events: previous_events.clone(),
        };
        callback(&current_snapshot);
        if options.verbose {
            log_sqp_iteration(&current_snapshot, options, &mut event_state);
        }
        if iteration == options.max_iters {
            return Err(ClarabelSqpError::MaxIterations {
                iterations: options.max_iters,
                context: failure_context(
                    SqpTermination::MaxIterations,
                    Some(current_snapshot),
                    last_accepted_state,
                    &profiling,
                ),
            });
        }
        if converged {
            profiling.preprocessing_steps += 1;
            profiling.preprocessing_other_steps += 1;
            profiling.preprocessing_other_time += iteration_preprocess_other_time;
            profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();
            finalize_profiling(&mut profiling, solve_started);
            let summary = ClarabelSqpSummary {
                x: x.clone(),
                equality_multipliers,
                inequality_multipliers,
                lower_bound_multipliers,
                upper_bound_multipliers,
                objective: objective_value,
                iterations: iteration,
                equality_inf_norm: (equality_count > 0).then_some(equality_inf),
                inequality_inf_norm: (augmented_inequality_count > 0).then_some(inequality_inf),
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: (augmented_inequality_count > 0)
                    .then_some(complementarity_inf),
                overall_inf_norm: overall_inf,
                termination: SqpTermination::Converged,
                final_state: current_snapshot.clone(),
                final_state_kind: final_state_kind(&current_snapshot),
                last_accepted_state: (current_snapshot.phase != SqpIterationPhase::Initial)
                    .then_some(current_snapshot.clone()),
                profiling,
            };
            if options.verbose {
                log_sqp_status_summary(&summary, options);
            }
            return Ok(summary);
        }

        let hessian_evaluation_started = Instant::now();
        let hessian_evaluation_result = (|| -> std::result::Result<(), ClarabelSqpError> {
            time_callback(
                &mut profiling.lagrangian_hessian_values,
                &mut iteration_callback_time,
                || {
                    problem.lagrangian_hessian_values(
                        &x,
                        parameters,
                        &equality_multipliers,
                        &inequality_multipliers,
                        &mut hessian_values,
                    );
                },
            );
            validate_finite_slice_output(
                &hessian_values,
                NonFiniteCallbackStage::LagrangianHessianValues,
                Some(&current_snapshot),
                last_accepted_state.as_ref(),
                &profiling,
            )?;
            Ok(())
        })();
        profiling.evaluation_time += hessian_evaluation_started.elapsed();
        hessian_evaluation_result?;
        let mut hessian;
        let regularization_info;
        let base_subproblem;
        {
            let preprocess_started = Instant::now();
            let hessian_assembly_started = Instant::now();
            hessian = lower_triangle_to_symmetric_dense(
                problem.lagrangian_hessian_ccs(),
                &hessian_values,
            );
            let hessian_assembly_elapsed = hessian_assembly_started.elapsed();
            iteration_hessian_assembly_time += hessian_assembly_elapsed;
            record_iteration_duration(
                &mut profiling.hessian_assembly_steps,
                &mut profiling.hessian_assembly_time,
                hessian_assembly_elapsed,
            );
            if options.hessian_regularization_enabled {
                let regularization_started = Instant::now();
                regularization_info = regularize_hessian(&mut hessian, options.regularization);
                let regularization_elapsed = regularization_started.elapsed();
                iteration_regularization_time += regularization_elapsed;
                record_iteration_duration(
                    &mut profiling.regularization_steps,
                    &mut profiling.regularization_time,
                    regularization_elapsed,
                );
            } else {
                regularization_info = disabled_hessian_regularization_info();
            }

            base_subproblem = if is_trust_region {
                None
            } else {
                let subproblem_assembly_started = Instant::now();
                let assembled_subproblem = assemble_sqp_subproblem(
                    &equality_jacobian,
                    &inequality_jacobian,
                    &equality_values,
                    &augmented_inequality_values,
                    None,
                );
                let subproblem_assembly_elapsed = subproblem_assembly_started.elapsed();
                iteration_subproblem_assembly_time += subproblem_assembly_elapsed;
                record_iteration_duration(
                    &mut profiling.subproblem_assembly_steps,
                    &mut profiling.subproblem_assembly_time,
                    subproblem_assembly_elapsed,
                );
                Some(assembled_subproblem)
            };
            let preprocess_elapsed = preprocess_started.elapsed();
            iteration_preprocess_time += preprocess_elapsed;
            profiling.preprocessing_time += preprocess_elapsed;
        }
        if regularization_info.shifted_by_analysis {
            previous_events.push(SqpIterationEvent::HessianShifted);
        }

        if is_trust_region {
            let elastic_model = ElasticRecoveryModel {
                hessian: &hessian,
                gradient: &gradient,
                equality_values: &equality_values,
                equality_jacobian: &equality_jacobian,
                nonlinear_inequality_values: &inequality_values,
                nonlinear_inequality_jacobian: &nonlinear_inequality_jacobian,
                augmented_inequality_values: &augmented_inequality_values,
                bound_jacobian: &bound_jacobian,
            };
            let trust_region_attempt = attempt_sqp_trust_region(
                problem,
                &x,
                parameters,
                objective_value,
                &gradient,
                &hessian,
                &equality_values,
                &equality_jacobian,
                &inequality_values,
                &nonlinear_inequality_jacobian,
                &augmented_inequality_values,
                &inequality_jacobian,
                &bounds,
                &bound_jacobian,
                &elastic_model,
                &regularization_info,
                equality_count,
                inequality_count,
                lower_bound_count,
                primal_inf,
                options,
                &mut merit_penalty,
                trust_region_radius.expect("trust-region mode requires a radius"),
                &filter_entries,
                filter_theta_reference,
                filter_theta_max,
                &current_snapshot,
                &last_accepted_state,
                &mut profiling,
                &mut iteration_callback_time,
                &mut iteration_subproblem_assembly_time,
                &mut iteration_qp_setup_time,
                &mut iteration_qp_solve_time,
                &mut iteration_line_search_evaluation_time,
                &mut iteration_line_search_condition_check_time,
                &mut trial_equality_values,
                &mut trial_inequality_values,
                &mut trial_augmented_inequality_values,
                &mut trial_gradient,
                &mut trial_equality_jacobian_values,
                &mut trial_inequality_jacobian_values,
            )?;
            let iteration_preprocess_other_time = iteration_preprocess_time.saturating_sub(
                iteration_jacobian_assembly_time
                    + iteration_hessian_assembly_time
                    + iteration_regularization_time
                    + iteration_subproblem_assembly_time,
            );
            profiling.preprocessing_steps += 1;
            profiling.preprocessing_other_steps += 1;
            profiling.preprocessing_other_time += iteration_preprocess_other_time;
            if uses_filter {
                filter::update_frontier(
                    &mut filter_entries,
                    filter::entry(
                        trust_region_attempt.accepted_trial.evaluation.objective,
                        trust_region_attempt.accepted_trial.evaluation.primal_inf,
                    ),
                );
            }
            x = trust_region_attempt.accepted_trial.point.clone();
            equality_multipliers = trust_region_attempt.solution.equality_multipliers;
            inequality_multipliers = trust_region_attempt.solution.inequality_multipliers;
            lower_bound_multipliers = trust_region_attempt.solution.lower_bound_multipliers;
            upper_bound_multipliers = trust_region_attempt.solution.upper_bound_multipliers;
            previous_step_inf = Some(trust_region_attempt.step_inf_norm);
            previous_line_search = None;
            previous_trust_region = Some(trust_region_attempt.trust_region.clone());
            previous_qp = (total_constraint_count > 0)
                .then_some(trust_region_attempt.solution.qp_info.clone());
            previous_elastic_recovery_used = trust_region_attempt.solution.elastic_recovery_used;
            previous_events = snapshot_events(
                trust_region_attempt.penalty_updated,
                regularization_info.shifted_by_analysis,
                None,
                false,
                false,
                trust_region_attempt.accepted_trial.filter_acceptance_mode
                    == Some(SqpFilterAcceptanceMode::ViolationReduction),
                trust_region_attempt.accepted_trial.step_kind == Some(SqpStepKind::Restoration),
                false,
                previous_qp.as_ref(),
                previous_elastic_recovery_used,
                false,
            );
            previous_step_diagnostics = Some(trust_region_attempt.step_diagnostics);
            trust_region_radius = Some(trust_region_attempt.next_radius);
            last_accepted_state = Some(current_snapshot);
            continue;
        }

        let SqpSubproblemSolution {
            step,
            equality_multipliers: candidate_equality_multipliers,
            inequality_multipliers: candidate_inequality_multipliers,
            lower_bound_multipliers: candidate_lower_bound_multipliers,
            upper_bound_multipliers: candidate_upper_bound_multipliers,
            qp_info: current_qp_info,
            elastic_recovery_used,
        } = {
            let subproblem_started = Instant::now();
            let subproblem_result = {
                let elastic_model = ElasticRecoveryModel {
                    hessian: &hessian,
                    gradient: &gradient,
                    equality_values: &equality_values,
                    equality_jacobian: &equality_jacobian,
                    nonlinear_inequality_values: &inequality_values,
                    nonlinear_inequality_jacobian: &nonlinear_inequality_jacobian,
                    augmented_inequality_values: &augmented_inequality_values,
                    bound_jacobian: &bound_jacobian,
                };
                let profiling_snapshot = profiling.clone();
                let mut qp_ctx = QpSolveContext {
                    profiling: &mut profiling,
                    iteration_qp_setup_time: &mut iteration_qp_setup_time,
                    iteration_qp_solve_time: &mut iteration_qp_solve_time,
                };
                solve_sqp_subproblem(
                    &hessian,
                    &gradient,
                    base_subproblem
                        .as_ref()
                        .expect("line-search mode requires a base subproblem"),
                    &elastic_model,
                    options,
                    &mut qp_ctx,
                    n,
                    equality_count,
                    inequality_count,
                    lower_bound_count,
                    &current_snapshot,
                    &last_accepted_state,
                    &profiling_snapshot,
                )
            };
            profiling.subproblem_solve_time += subproblem_started.elapsed();
            subproblem_result?
        };
        let current_qp = (total_constraint_count > 0).then_some(current_qp_info.clone());

        let candidate_all_inequality_multipliers;
        let candidate_all_dual_multipliers;
        let candidate_dual_inf;
        let candidate_complementarity_inf;
        let candidate_overall_inf;
        {
            let multiplier_estimation_started_all = Instant::now();
            let multiplier_estimation_started = Instant::now();
            candidate_all_inequality_multipliers = [
                candidate_inequality_multipliers.as_slice(),
                candidate_lower_bound_multipliers.as_slice(),
                candidate_upper_bound_multipliers.as_slice(),
            ]
            .concat();
            candidate_all_dual_multipliers = [
                candidate_equality_multipliers.as_slice(),
                candidate_all_inequality_multipliers.as_slice(),
            ]
            .concat();
            candidate_dual_inf = inf_norm(&lagrangian_gradient(
                &gradient,
                &equality_jacobian,
                &candidate_equality_multipliers,
                &inequality_jacobian,
                &candidate_all_inequality_multipliers,
            ));
            candidate_complementarity_inf = complementarity_inf_norm(
                &augmented_inequality_values,
                &candidate_all_inequality_multipliers,
            );
            candidate_overall_inf = scaled_overall_inf_norm(
                primal_inf,
                candidate_dual_inf,
                candidate_complementarity_inf,
                &candidate_all_dual_multipliers,
                &candidate_all_inequality_multipliers,
                options.overall_scale_max,
            );
            let multiplier_estimation_elapsed = multiplier_estimation_started.elapsed();
            profiling.multiplier_estimations += 1;
            iteration_multiplier_estimation_time += multiplier_estimation_elapsed;
            profiling.multiplier_estimation_time += multiplier_estimation_elapsed;
            profiling.subproblem_solve_time += multiplier_estimation_started_all.elapsed();
        }

        let candidate_converged;
        {
            let convergence_started = Instant::now();
            let convergence_check_started = Instant::now();
            candidate_converged = candidate_overall_inf <= options.overall_tol
                && primal_inf <= options.constraint_tol
                && candidate_dual_inf <= options.dual_tol
                && candidate_complementarity_inf <= options.complementarity_tol;
            let convergence_check_elapsed = convergence_check_started.elapsed();
            profiling.convergence_checks += 1;
            iteration_convergence_check_time += convergence_check_elapsed;
            profiling.convergence_check_time += convergence_check_elapsed;
            profiling.convergence_time += convergence_started.elapsed();
        }

        if candidate_converged {
            let iteration_elapsed = iteration_started.elapsed();
            let iteration_preprocess_other_time = iteration_preprocess_time.saturating_sub(
                iteration_jacobian_assembly_time
                    + iteration_hessian_assembly_time
                    + iteration_regularization_time
                    + iteration_subproblem_assembly_time,
            );
            let post_convergence_state = SqpIterationSnapshot {
                iteration,
                phase: SqpIterationPhase::PostConvergence,
                globalization: globalization_kind,
                x: x.clone(),
                objective: objective_value,
                eq_inf: (equality_count > 0).then_some(equality_inf),
                ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
                dual_inf: candidate_dual_inf,
                comp_inf: (augmented_inequality_count > 0).then_some(candidate_complementarity_inf),
                overall_inf: candidate_overall_inf,
                step_inf: None,
                penalty: merit_penalty,
                line_search: None,
                trust_region: None,
                step_diagnostics: None,
                filter: uses_filter.then(|| SqpFilterInfo {
                    current: current_filter_entry,
                    entries: filter_entries.clone(),
                    accepted_mode: None,
                }),
                qp: current_qp.clone(),
                timing: build_iteration_timing(
                    iteration_timing_baseline,
                    &profiling,
                    IterationTimingBuckets {
                        jacobian_assembly: iteration_jacobian_assembly_time,
                        hessian_assembly: iteration_hessian_assembly_time,
                        regularization: iteration_regularization_time,
                        subproblem_assembly: iteration_subproblem_assembly_time,
                        qp_setup: iteration_qp_setup_time,
                        qp_solve: iteration_qp_solve_time,
                        multiplier_estimation: iteration_multiplier_estimation_time,
                        line_search_evaluation: iteration_line_search_evaluation_time,
                        line_search_condition_checks: iteration_line_search_condition_check_time,
                        convergence_check: iteration_convergence_check_time,
                        preprocess_other: iteration_preprocess_other_time,
                        total: iteration_elapsed,
                    },
                ),
                events: snapshot_events(
                    previous_events.contains(&SqpIterationEvent::PenaltyUpdated),
                    previous_events.contains(&SqpIterationEvent::HessianShifted),
                    previous_line_search
                        .as_ref()
                        .map(|info| info.backtrack_count),
                    previous_line_search
                        .as_ref()
                        .is_some_and(|info| info.armijo_tolerance_adjusted),
                    previous_line_search
                        .as_ref()
                        .is_some_and(|info| info.second_order_correction_used),
                    accepted_filter_mode(
                        previous_line_search.as_ref(),
                        previous_trust_region.as_ref(),
                    ) == Some(SqpFilterAcceptanceMode::ViolationReduction),
                    accepted_step_kind(
                        previous_line_search.as_ref(),
                        previous_trust_region.as_ref(),
                    ) == Some(SqpStepKind::Restoration),
                    previous_line_search.as_ref().is_some_and(|info| {
                        info.rejected_trials
                            .iter()
                            .any(|trial| trial.wolfe_satisfied == Some(false))
                    }),
                    current_qp.as_ref(),
                    previous_elastic_recovery_used,
                    false,
                ),
            };
            callback(&post_convergence_state);
            if options.verbose {
                log_sqp_iteration(&post_convergence_state, options, &mut event_state);
            }
            profiling.preprocessing_steps += 1;
            profiling.preprocessing_other_steps += 1;
            profiling.preprocessing_other_time += iteration_preprocess_other_time;
            profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();
            finalize_profiling(&mut profiling, solve_started);
            let summary = ClarabelSqpSummary {
                x: x.clone(),
                equality_multipliers: candidate_equality_multipliers,
                inequality_multipliers: candidate_inequality_multipliers,
                lower_bound_multipliers: candidate_lower_bound_multipliers,
                upper_bound_multipliers: candidate_upper_bound_multipliers,
                objective: objective_value,
                iterations: iteration,
                equality_inf_norm: (equality_count > 0).then_some(equality_inf),
                inequality_inf_norm: (augmented_inequality_count > 0).then_some(inequality_inf),
                primal_inf_norm: primal_inf,
                dual_inf_norm: candidate_dual_inf,
                complementarity_inf_norm: (augmented_inequality_count > 0)
                    .then_some(candidate_complementarity_inf),
                overall_inf_norm: candidate_overall_inf,
                termination: SqpTermination::Converged,
                final_state: post_convergence_state.clone(),
                final_state_kind: final_state_kind(&post_convergence_state),
                last_accepted_state: Some(current_snapshot.clone()),
                profiling,
            };
            if options.verbose {
                log_sqp_status_summary(&summary, options);
            }
            return Ok(summary);
        }

        let step_inf_norm = inf_norm(&step);
        if step_inf_norm <= min_line_search_step.unwrap_or(0.0)
            && !(uses_filter && options.restoration_phase)
        {
            return Err(ClarabelSqpError::Stalled {
                step_inf_norm,
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: complementarity_inf,
                context: failure_context(
                    SqpTermination::Stalled,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    &profiling,
                ),
            });
        }

        let mut current_step_diagnostics = None;
        let line_search_started = Instant::now();
        let line_search_result =
            (|| -> std::result::Result<LineSearchStageResult, ClarabelSqpError> {
                let penalty_before_updates = merit_penalty;
                if !uses_filter && !elastic_recovery_used {
                    merit_penalty = update_merit_penalty(
                        merit_penalty,
                        &candidate_equality_multipliers,
                        &[
                            candidate_inequality_multipliers.as_slice(),
                            candidate_lower_bound_multipliers.as_slice(),
                            candidate_upper_bound_multipliers.as_slice(),
                        ]
                        .concat(),
                    );
                }
                let current_merit = exact_merit_value(
                    objective_value,
                    &equality_values,
                    &augmented_inequality_values,
                    merit_penalty,
                );
                let objective_directional_derivative = dot(&gradient, &step);
                let mut directional_derivative = exact_merit_directional_derivative(
                    &gradient,
                    &equality_values,
                    &equality_jacobian,
                    &augmented_inequality_values,
                    &inequality_jacobian,
                    &step,
                    merit_penalty,
                );
                if !uses_filter {
                    let (penalty_increase_factor, max_penalty_updates) =
                        line_search_penalty_update_settings
                            .expect("merit line search requires penalty update settings");
                    for _ in 0..max_penalty_updates {
                        if directional_derivative < -1e-12 {
                            break;
                        }
                        merit_penalty *= penalty_increase_factor;
                        directional_derivative = exact_merit_directional_derivative(
                            &gradient,
                            &equality_values,
                            &equality_jacobian,
                            &augmented_inequality_values,
                            &inequality_jacobian,
                            &step,
                            merit_penalty,
                        );
                    }
                }
                let predicted_linearized_eq_inf = if equality_count > 0 {
                    inf_norm(&linearized_constraint_residual(
                        &equality_values,
                        &equality_jacobian,
                        &step,
                    ))
                } else {
                    0.0
                };
                let predicted_linearized_ineq_inf = if augmented_inequality_count > 0 {
                    positive_part_inf_norm(&linearized_constraint_residual(
                        &augmented_inequality_values,
                        &inequality_jacobian,
                        &step,
                    ))
                } else {
                    0.0
                };
                let normal_switching_condition = if elastic_recovery_used {
                    false
                } else {
                    sqp_switching_condition(
                        primal_inf,
                        filter_theta_reference,
                        predicted_linearized_eq_inf.max(predicted_linearized_ineq_inf),
                        objective_directional_derivative,
                        options,
                    )
                };
                let normal_step_diagnostics = sqp_step_diagnostics(
                    &gradient,
                    &hessian,
                    primal_inf,
                    filter_theta_max,
                    &equality_values,
                    &equality_jacobian,
                    &augmented_inequality_values,
                    &inequality_jacobian,
                    &step,
                    directional_derivative,
                    normal_switching_condition,
                    elastic_recovery_used,
                    elastic_recovery_used,
                    regularization_info.clone(),
                );
                current_step_diagnostics = Some(normal_step_diagnostics.clone());
                if !uses_filter && directional_derivative >= 0.0 {
                    return Err(ClarabelSqpError::LineSearchFailed {
                        directional_derivative,
                        step_inf_norm,
                        penalty: merit_penalty,
                        context: failure_context_with_qp_failure(
                            SqpTermination::LineSearchFailed,
                            Some(current_snapshot.clone()),
                            last_accepted_state.clone(),
                            None,
                            None,
                            Some(normal_step_diagnostics),
                            None,
                            &profiling,
                        ),
                    });
                }
                let penalty_updated = !uses_filter && merit_penalty != penalty_before_updates;
                let current_filter_trial = filter::entry(objective_value, primal_inf);
                let normal_solution = SqpSubproblemSolution {
                    step: step.clone(),
                    equality_multipliers: candidate_equality_multipliers.clone(),
                    inequality_multipliers: candidate_inequality_multipliers.clone(),
                    lower_bound_multipliers: candidate_lower_bound_multipliers.clone(),
                    upper_bound_multipliers: candidate_upper_bound_multipliers.clone(),
                    qp_info: current_qp_info.clone(),
                    elastic_recovery_used,
                };

                let normal_attempt = attempt_sqp_line_search(
                    problem,
                    &x,
                    parameters,
                    &normal_solution,
                    &candidate_all_inequality_multipliers,
                    objective_directional_derivative,
                    directional_derivative,
                    normal_switching_condition,
                    elastic_recovery_used,
                    &equality_jacobian,
                    &inequality_jacobian,
                    primal_inf,
                    current_merit,
                    merit_penalty,
                    options,
                    &filter_entries,
                    &current_filter_trial,
                    filter_theta_max,
                    &current_snapshot,
                    &last_accepted_state,
                    &bounds,
                    &bound_jacobian,
                    &mut profiling,
                    &mut iteration_callback_time,
                    &mut iteration_line_search_evaluation_time,
                    &mut iteration_line_search_condition_check_time,
                    &mut trial_equality_values,
                    &mut trial_inequality_values,
                    &mut trial_augmented_inequality_values,
                    &mut trial_gradient,
                    &mut trial_equality_jacobian_values,
                    &mut trial_inequality_jacobian_values,
                )?;
                let mut maybe_accepted_trial = normal_attempt.accepted_trial;
                let mut current_last_tried_alpha = normal_attempt.last_tried_alpha;
                let mut current_line_search_iterations = normal_attempt.backtrack_count;
                let mut current_rejected_trials = normal_attempt.rejected_trials;
                let mut current_wolfe_rejected = normal_attempt.wolfe_rejected;
                let mut second_order_correction_attempted =
                    normal_attempt.second_order_correction_attempted;
                let mut restoration_attempted = normal_attempt.restoration_attempted;
                let mut elastic_recovery_attempted = normal_attempt.elastic_recovery_attempted;

                let mut selected_solution = normal_solution;
                let mut selected_step_diagnostics = normal_step_diagnostics;
                let mut selected_step_inf_norm = step_inf_norm;

                if maybe_accepted_trial.is_none()
                    && uses_filter
                    && options.restoration_phase
                    && !selected_solution.elastic_recovery_used
                    && total_constraint_count > 0
                {
                    let elastic_model = ElasticRecoveryModel {
                        hessian: &hessian,
                        gradient: &gradient,
                        equality_values: &equality_values,
                        equality_jacobian: &equality_jacobian,
                        nonlinear_inequality_values: &inequality_values,
                        nonlinear_inequality_jacobian: &nonlinear_inequality_jacobian,
                        augmented_inequality_values: &augmented_inequality_values,
                        bound_jacobian: &bound_jacobian,
                    };
                    profiling.elastic_recovery_activations += 1;
                    let profiling_snapshot = profiling.clone();
                    let mut qp_ctx = QpSolveContext {
                        profiling: &mut profiling,
                        iteration_qp_setup_time: &mut iteration_qp_setup_time,
                        iteration_qp_solve_time: &mut iteration_qp_solve_time,
                    };
                    let restoration_solution = solve_restoration_subproblem(
                        &elastic_model,
                        options,
                        &mut qp_ctx,
                        n,
                        equality_count,
                        inequality_count,
                        lower_bound_count,
                        &current_snapshot,
                        &last_accepted_state,
                        &profiling_snapshot,
                    )?;
                    let restoration_step_inf_norm = inf_norm(&restoration_solution.step);
                    let restoration_objective_directional_derivative =
                        dot(&gradient, &restoration_solution.step);
                    let restoration_directional_derivative = exact_merit_directional_derivative(
                        &gradient,
                        &equality_values,
                        &equality_jacobian,
                        &augmented_inequality_values,
                        &inequality_jacobian,
                        &restoration_solution.step,
                        merit_penalty,
                    );
                    let restoration_step_diagnostics = sqp_step_diagnostics(
                        &gradient,
                        &hessian,
                        primal_inf,
                        filter_theta_max,
                        &equality_values,
                        &equality_jacobian,
                        &augmented_inequality_values,
                        &inequality_jacobian,
                        &restoration_solution.step,
                        restoration_directional_derivative,
                        false,
                        true,
                        true,
                        regularization_info.clone(),
                    );
                    let restoration_all_inequality_multipliers = [
                        restoration_solution.inequality_multipliers.as_slice(),
                        restoration_solution.lower_bound_multipliers.as_slice(),
                        restoration_solution.upper_bound_multipliers.as_slice(),
                    ]
                    .concat();
                    let restoration_attempt = attempt_sqp_line_search(
                        problem,
                        &x,
                        parameters,
                        &restoration_solution,
                        &restoration_all_inequality_multipliers,
                        restoration_objective_directional_derivative,
                        restoration_directional_derivative,
                        false,
                        true,
                        &equality_jacobian,
                        &inequality_jacobian,
                        primal_inf,
                        current_merit,
                        merit_penalty,
                        options,
                        &filter_entries,
                        &current_filter_trial,
                        filter_theta_max,
                        &current_snapshot,
                        &last_accepted_state,
                        &bounds,
                        &bound_jacobian,
                        &mut profiling,
                        &mut iteration_callback_time,
                        &mut iteration_line_search_evaluation_time,
                        &mut iteration_line_search_condition_check_time,
                        &mut trial_equality_values,
                        &mut trial_inequality_values,
                        &mut trial_augmented_inequality_values,
                        &mut trial_gradient,
                        &mut trial_equality_jacobian_values,
                        &mut trial_inequality_jacobian_values,
                    )?;
                    maybe_accepted_trial = restoration_attempt.accepted_trial;
                    current_last_tried_alpha = restoration_attempt.last_tried_alpha;
                    current_line_search_iterations = restoration_attempt.backtrack_count;
                    current_rejected_trials = restoration_attempt.rejected_trials;
                    current_wolfe_rejected = restoration_attempt.wolfe_rejected;
                    second_order_correction_attempted |=
                        restoration_attempt.second_order_correction_attempted;
                    restoration_attempted |= restoration_attempt.restoration_attempted;
                    elastic_recovery_attempted |= restoration_attempt.elastic_recovery_attempted;
                    selected_solution = restoration_solution;
                    selected_step_diagnostics = restoration_step_diagnostics;
                    selected_step_inf_norm = restoration_step_inf_norm;
                }

                let Some(line_search_accept) = maybe_accepted_trial else {
                    let restoration_failure = selected_solution.elastic_recovery_used;
                    let failed_line_search = failed_sqp_line_search_info(
                        current_last_tried_alpha,
                        current_line_search_iterations,
                        current_rejected_trials,
                        current_wolfe_rejected,
                        restoration_failure.then_some(SqpStepKind::Restoration),
                        Some(selected_step_diagnostics.switching_condition_satisfied),
                        second_order_correction_attempted,
                        restoration_attempted,
                        elastic_recovery_attempted,
                    );
                    let termination = if restoration_failure {
                        SqpTermination::RestorationFailed
                    } else {
                        SqpTermination::LineSearchFailed
                    };
                    let context = failure_context_with_qp_failure(
                        termination,
                        Some(current_snapshot.clone()),
                        last_accepted_state.clone(),
                        Some(failed_line_search),
                        None,
                        Some(selected_step_diagnostics.clone()),
                        None,
                        &profiling,
                    );
                    return Err(if restoration_failure {
                        ClarabelSqpError::RestorationFailed {
                            step_inf_norm: selected_step_inf_norm,
                            context,
                        }
                    } else {
                        ClarabelSqpError::LineSearchFailed {
                            directional_derivative,
                            step_inf_norm,
                            penalty: merit_penalty,
                            context,
                        }
                    });
                };
                Ok((
                    penalty_updated,
                    selected_solution,
                    line_search_accept,
                    selected_step_diagnostics,
                    selected_step_inf_norm,
                    current_last_tried_alpha,
                    current_last_tried_alpha,
                    current_line_search_iterations,
                    current_rejected_trials,
                    current_wolfe_rejected,
                    second_order_correction_attempted,
                    restoration_attempted,
                    elastic_recovery_attempted,
                ))
            })();
        profiling.line_search_time += line_search_started.elapsed();
        let (
            penalty_updated,
            selected_solution,
            accepted_line_search,
            selected_step_diagnostics,
            selected_step_inf_norm,
            accepted_alpha,
            last_tried_alpha,
            line_search_iterations,
            rejected_trials,
            wolfe_rejected,
            second_order_correction_attempted,
            restoration_attempted,
            elastic_recovery_attempted,
        ) = line_search_result?;
        current_step_diagnostics = Some(selected_step_diagnostics.clone());
        let iteration_preprocess_other_time = iteration_preprocess_time.saturating_sub(
            iteration_jacobian_assembly_time
                + iteration_hessian_assembly_time
                + iteration_regularization_time
                + iteration_subproblem_assembly_time,
        );

        profiling.preprocessing_steps += 1;
        profiling.preprocessing_other_steps += 1;
        profiling.preprocessing_other_time += iteration_preprocess_other_time;
        if uses_filter {
            filter::update_frontier(
                &mut filter_entries,
                filter::entry(
                    accepted_line_search.evaluation.objective,
                    accepted_line_search.evaluation.primal_inf,
                ),
            );
        }
        x = accepted_line_search.point.clone();
        equality_multipliers = selected_solution.equality_multipliers;
        inequality_multipliers = selected_solution.inequality_multipliers;
        lower_bound_multipliers = selected_solution.lower_bound_multipliers;
        upper_bound_multipliers = selected_solution.upper_bound_multipliers;
        previous_step_inf = Some(selected_step_inf_norm);
        previous_line_search = Some(SqpLineSearchInfo {
            accepted_alpha,
            last_tried_alpha,
            backtrack_count: line_search_iterations,
            armijo_satisfied: accepted_line_search.armijo_satisfied,
            armijo_tolerance_adjusted: accepted_line_search.armijo_tolerance_adjusted,
            objective_armijo_satisfied: accepted_line_search.objective_armijo_satisfied,
            objective_armijo_tolerance_adjusted: accepted_line_search
                .objective_armijo_tolerance_adjusted,
            second_order_correction_attempted,
            second_order_correction_used: accepted_line_search.second_order_correction_used,
            wolfe_satisfied: accepted_line_search.wolfe_satisfied,
            violation_satisfied: accepted_line_search.violation_satisfied,
            restoration_attempted,
            elastic_recovery_attempted,
            step_kind: accepted_line_search.step_kind,
            filter_acceptance_mode: accepted_line_search.filter_acceptance_mode,
            filter_acceptable: accepted_line_search.filter_acceptable,
            filter_dominated: accepted_line_search.filter_dominated,
            filter_theta_acceptable: accepted_line_search.filter_theta_acceptable,
            filter_sufficient_objective_reduction: accepted_line_search
                .filter_sufficient_objective_reduction,
            filter_sufficient_violation_reduction: accepted_line_search
                .filter_sufficient_violation_reduction,
            switching_condition_satisfied: accepted_line_search.switching_condition_satisfied,
            rejected_trials,
        });
        previous_trust_region = None;
        previous_qp = (total_constraint_count > 0).then_some(selected_solution.qp_info.clone());
        previous_elastic_recovery_used = selected_solution.elastic_recovery_used;
        previous_events = snapshot_events(
            penalty_updated,
            regularization_info.shifted_by_analysis,
            Some(line_search_iterations),
            accepted_line_search.armijo_tolerance_adjusted,
            accepted_line_search.second_order_correction_used,
            accepted_line_search.filter_acceptance_mode
                == Some(SqpFilterAcceptanceMode::ViolationReduction),
            accepted_line_search.step_kind == Some(SqpStepKind::Restoration),
            wolfe_rejected,
            previous_qp.as_ref(),
            previous_elastic_recovery_used,
            false,
        );
        previous_step_diagnostics = current_step_diagnostics;
        last_accepted_state = Some(current_snapshot);
    }

    let max_iteration = options.max_iters;
    let iteration_started = Instant::now();
    profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();
    let iteration_timing_baseline = IterationTimingBaseline::capture(&profiling);
    let mut iteration_callback_time = Duration::ZERO;
    let mut final_iteration_preprocess_time = Duration::ZERO;
    let mut iteration_jacobian_assembly_time = Duration::ZERO;
    let final_evaluation_started = Instant::now();
    let final_evaluation_result = (|| -> std::result::Result<(), ClarabelSqpError> {
        time_callback(
            &mut profiling.objective_gradient,
            &mut iteration_callback_time,
            || problem.objective_gradient(&x, parameters, &mut gradient),
        );
        validate_finite_slice_output(
            &gradient,
            NonFiniteCallbackStage::ObjectiveGradient,
            last_accepted_state.as_ref(),
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.equality_values,
            &mut iteration_callback_time,
            || problem.equality_values(&x, parameters, &mut equality_values),
        );
        validate_finite_slice_output(
            &equality_values,
            NonFiniteCallbackStage::EqualityValues,
            last_accepted_state.as_ref(),
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.inequality_values,
            &mut iteration_callback_time,
            || problem.inequality_values(&x, parameters, &mut inequality_values),
        );
        validate_finite_slice_output(
            &inequality_values,
            NonFiniteCallbackStage::InequalityValues,
            last_accepted_state.as_ref(),
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        augment_inequality_values(
            &inequality_values,
            &x,
            &bounds,
            &mut augmented_inequality_values,
        );
        time_callback(
            &mut profiling.equality_jacobian_values,
            &mut iteration_callback_time,
            || problem.equality_jacobian_values(&x, parameters, &mut equality_jacobian_values),
        );
        validate_finite_slice_output(
            &equality_jacobian_values,
            NonFiniteCallbackStage::EqualityJacobianValues,
            last_accepted_state.as_ref(),
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.inequality_jacobian_values,
            &mut iteration_callback_time,
            || problem.inequality_jacobian_values(&x, parameters, &mut inequality_jacobian_values),
        );
        validate_finite_slice_output(
            &inequality_jacobian_values,
            NonFiniteCallbackStage::InequalityJacobianValues,
            last_accepted_state.as_ref(),
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        Ok(())
    })();
    profiling.evaluation_time += final_evaluation_started.elapsed();
    final_evaluation_result?;
    let (equality_jacobian, nonlinear_inequality_jacobian, inequality_jacobian);
    let (
        objective_value,
        equality_inf,
        inequality_inf,
        dual_inf,
        complementarity_inf,
        final_filter_entry,
    );
    {
        let final_preprocess_started = Instant::now();
        let jacobian_assembly_started = Instant::now();
        equality_jacobian =
            ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
        nonlinear_inequality_jacobian = ccs_to_dense(
            problem.inequality_jacobian_ccs(),
            &inequality_jacobian_values,
        );
        inequality_jacobian = stack_jacobians(&nonlinear_inequality_jacobian, &bound_jacobian);
        let jacobian_assembly_elapsed = jacobian_assembly_started.elapsed();
        iteration_jacobian_assembly_time += jacobian_assembly_elapsed;
        record_iteration_duration(
            &mut profiling.jacobian_assembly_steps,
            &mut profiling.jacobian_assembly_time,
            jacobian_assembly_elapsed,
        );
        let final_objective_started = Instant::now();
        let final_objective_result = validate_finite_scalar_output(
            time_callback(
                &mut profiling.objective_value,
                &mut iteration_callback_time,
                || problem.objective_value(&x, parameters),
            ),
            NonFiniteCallbackStage::ObjectiveValue,
            last_accepted_state.as_ref(),
            last_accepted_state.as_ref(),
            &profiling,
        );
        profiling.evaluation_time += final_objective_started.elapsed();
        objective_value = final_objective_result?;
        equality_inf = inf_norm(&equality_values);
        inequality_inf = positive_part_inf_norm(&augmented_inequality_values);
        let all_inequality_multipliers = [
            inequality_multipliers.as_slice(),
            lower_bound_multipliers.as_slice(),
            upper_bound_multipliers.as_slice(),
        ]
        .concat();
        dual_inf = inf_norm(&lagrangian_gradient(
            &gradient,
            &equality_jacobian,
            &equality_multipliers,
            &inequality_jacobian,
            &all_inequality_multipliers,
        ));
        complementarity_inf =
            complementarity_inf_norm(&augmented_inequality_values, &all_inequality_multipliers);
        final_filter_entry = filter::entry(objective_value, equality_inf.max(inequality_inf));
        let final_preprocess_elapsed = final_preprocess_started.elapsed();
        final_iteration_preprocess_time += final_preprocess_elapsed;
        profiling.preprocessing_time += final_preprocess_elapsed;
    }
    let final_primal_inf = equality_inf.max(inequality_inf);
    let final_iteration_elapsed = iteration_started.elapsed();
    let final_iteration_subproblem_assembly_time = Duration::ZERO;
    let iteration_preprocess_other_time = final_iteration_preprocess_time.saturating_sub(
        iteration_jacobian_assembly_time + final_iteration_subproblem_assembly_time,
    );
    let all_inequality_multipliers = [
        inequality_multipliers.as_slice(),
        lower_bound_multipliers.as_slice(),
        upper_bound_multipliers.as_slice(),
    ]
    .concat();
    let all_dual_multipliers = [
        equality_multipliers.as_slice(),
        all_inequality_multipliers.as_slice(),
    ]
    .concat();
    let overall_inf = scaled_overall_inf_norm(
        final_primal_inf,
        dual_inf,
        complementarity_inf,
        &all_dual_multipliers,
        &all_inequality_multipliers,
        options.overall_scale_max,
    );
    let final_snapshot = SqpIterationSnapshot {
        iteration: max_iteration,
        phase: SqpIterationPhase::AcceptedStep,
        globalization: globalization_kind,
        x: x.clone(),
        objective: objective_value,
        eq_inf: (equality_count > 0).then_some(equality_inf),
        ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
        dual_inf,
        comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
        overall_inf,
        step_inf: previous_step_inf,
        penalty: merit_penalty,
        line_search: previous_line_search.clone(),
        trust_region: previous_trust_region.clone(),
        step_diagnostics: previous_step_diagnostics.clone(),
        filter: uses_filter.then(|| SqpFilterInfo {
            current: final_filter_entry,
            entries: filter_entries.clone(),
            accepted_mode: accepted_filter_mode(
                previous_line_search.as_ref(),
                previous_trust_region.as_ref(),
            ),
        }),
        qp: previous_qp.clone(),
        timing: build_iteration_timing(
            iteration_timing_baseline,
            &profiling,
            IterationTimingBuckets {
                jacobian_assembly: iteration_jacobian_assembly_time,
                subproblem_assembly: final_iteration_subproblem_assembly_time,
                preprocess_other: iteration_preprocess_other_time,
                total: final_iteration_elapsed,
                ..IterationTimingBuckets::default()
            },
        ),
        events: snapshot_events(
            false,
            previous_events.contains(&SqpIterationEvent::HessianShifted),
            previous_line_search
                .as_ref()
                .map(|info| info.backtrack_count),
            previous_line_search
                .as_ref()
                .is_some_and(|info| info.armijo_tolerance_adjusted),
            previous_line_search
                .as_ref()
                .is_some_and(|info| info.second_order_correction_used),
            accepted_filter_mode(
                previous_line_search.as_ref(),
                previous_trust_region.as_ref(),
            ) == Some(SqpFilterAcceptanceMode::ViolationReduction),
            accepted_step_kind(
                previous_line_search.as_ref(),
                previous_trust_region.as_ref(),
            ) == Some(SqpStepKind::Restoration),
            previous_line_search.as_ref().is_some_and(|info| {
                info.rejected_trials
                    .iter()
                    .any(|trial| trial.wolfe_satisfied == Some(false))
            }),
            previous_qp.as_ref(),
            previous_elastic_recovery_used,
            true,
        ),
    };
    callback(&final_snapshot);
    if options.verbose {
        log_sqp_iteration(&final_snapshot, options, &mut event_state);
    }
    profiling.preprocessing_steps += 1;
    profiling.preprocessing_other_steps += 1;
    profiling.preprocessing_other_time += iteration_preprocess_other_time;
    profiling.adapter_timing = problem.sqp_adapter_timing_snapshot();
    finalize_profiling(&mut profiling, solve_started);
    Err(ClarabelSqpError::MaxIterations {
        iterations: options.max_iters,
        context: failure_context(
            SqpTermination::MaxIterations,
            Some(final_snapshot),
            last_accepted_state,
            &profiling,
        ),
    })
}

pub fn validate_nlp_problem_shapes<P>(problem: &P) -> Result<()>
where
    P: CompiledNlpProblem,
{
    let dimension = problem.dimension();
    if problem.lagrangian_hessian_ccs().nrow != dimension
        || problem.lagrangian_hessian_ccs().ncol != dimension
    {
        bail!("Lagrangian Hessian CCS must be square with dimension {dimension}");
    }
    if problem.equality_jacobian_ccs().nrow != problem.equality_count()
        || problem.equality_jacobian_ccs().ncol != dimension
    {
        bail!("equality Jacobian CCS does not match declared dimensions");
    }
    if problem.inequality_jacobian_ccs().nrow != problem.inequality_count()
        || problem.inequality_jacobian_ccs().ncol != dimension
    {
        bail!("inequality Jacobian CCS does not match declared dimensions");
    }
    Ok(())
}

pub fn validate_parameter_inputs<P>(problem: &P, parameters: &[ParameterMatrix<'_>]) -> Result<()>
where
    P: CompiledNlpProblem,
{
    if parameters.len() != problem.parameter_count() {
        bail!(
            "parameter count mismatch: got {}, expected {}",
            parameters.len(),
            problem.parameter_count()
        );
    }
    for (index, parameter) in parameters.iter().enumerate() {
        let expected_ccs = problem.parameter_ccs(index);
        if parameter.ccs != expected_ccs {
            bail!("parameter {index} CCS does not match declared dimensions/pattern");
        }
        if parameter.values.len() != expected_ccs.nnz() {
            bail!(
                "parameter {index} value length mismatch: got {}, expected {}",
                parameter.values.len(),
                expected_ccs.nnz()
            );
        }
    }
    Ok(())
}
