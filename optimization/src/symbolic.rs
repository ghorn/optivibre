use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::Result as AnyResult;
use sx_codegen::LoweredFunction;
use sx_codegen_llvm::{
    CompiledJitFunction, FunctionCompileOptions, JitExecutionContext, LlvmOptimizationLevel,
};
use sx_core::{CCS as CoreCcs, HessianStrategy, NamedMatrix, SX, SXFunction, SXMatrix, SxError};
use thiserror::Error;

use crate::{
    BackendCompileReport, BackendTimingMetadata, CCS, ClarabelSqpError, ClarabelSqpOptions,
    ClarabelSqpSummary, CompiledNlpProblem, FiniteDifferenceValidationOptions, Index,
    InteriorPointIterationSnapshot, InteriorPointOptions, InteriorPointSolveError,
    InteriorPointSummary, NlpCompileStats, NlpConstraintViolationReport,
    NlpDerivativeValidationReport, NlpEqualityViolation, NlpEvaluationBenchmark,
    NlpEvaluationBenchmarkOptions, NlpEvaluationKernelKind, NlpInequalitySource,
    NlpInequalityViolation, ParameterMatrix, SqpAdapterTiming, SqpFailureContext,
    SqpIterationSnapshot, SymbolicCompileMetadata, SymbolicCompileProgress, SymbolicCompileStage,
    SymbolicCompileStageProgress, SymbolicSetupProfile, Vectorize,
    benchmark_compiled_nlp_problem_with_progress, classify_constraint_satisfaction,
    constraint_bound_side, flatten_optional_value, flatten_value, solve_nlp_interior_point,
    solve_nlp_interior_point_with_callback, solve_nlp_sqp, solve_nlp_sqp_with_callback,
    symbolic_column, symbolic_value, validate_compiled_nlp_problem_derivatives,
    worst_bound_violation,
};
#[cfg(feature = "ipopt")]
use crate::{
    IpoptIterationSnapshot, IpoptOptions, IpoptPartialSolution, IpoptSolveError, IpoptSummary,
    solve_nlp_ipopt, solve_nlp_ipopt_with_callback,
};

#[derive(Clone, Debug, PartialEq)]
struct SymbolicNlp {
    name: String,
    variables: SXMatrix,
    parameters: Option<NamedMatrix>,
    objective: SXMatrix,
    equalities: Option<SXMatrix>,
    inequalities: Option<SXMatrix>,
    construction_time: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicNlpOutputs<E = (), I = ()> {
    pub objective: SX,
    pub equalities: E,
    pub inequalities: I,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedSymbolicNlp<X, P, E, I> {
    symbolic: DynamicSymbolicNlp,
    _marker: TypedMarker<X, P, E, I>,
}

#[derive(Debug)]
struct CompiledJitNlp {
    dimension: Index,
    parameter_ccs: Vec<CCS>,
    equality_jacobian_ccs: CCS,
    inequality_base_jacobian_ccs: CCS,
    lagrangian_hessian_ccs: CCS,
    backend_timing: BackendTimingMetadata,
    backend_compile_report: BackendCompileReport,
    objective_value: JitKernel,
    objective_gradient: JitKernel,
    equality_values: Option<JitKernel>,
    equality_jacobian_values: Option<JitKernel>,
    inequality_values: Option<JitKernel>,
    inequality_jacobian_values: Option<JitKernel>,
    lagrangian_hessian_values: JitKernel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicSymbolicNlp {
    symbolic: SymbolicNlp,
}

#[derive(Debug)]
pub struct DynamicCompiledJitNlp {
    inner: CompiledJitNlp,
}

#[derive(Debug)]
pub struct TypedCompiledJitNlp<X, P, E, I> {
    inner: DynamicCompiledJitNlp,
    _marker: TypedMarker<X, P, E, I>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolicNlpCompileOptions {
    pub function_options: FunctionCompileOptions,
    pub hessian_strategy: HessianStrategy,
}

impl Default for SymbolicNlpCompileOptions {
    fn default() -> Self {
        Self::from(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }
}

impl From<FunctionCompileOptions> for SymbolicNlpCompileOptions {
    fn from(function_options: FunctionCompileOptions) -> Self {
        Self {
            function_options,
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
        }
    }
}

impl From<LlvmOptimizationLevel> for SymbolicNlpCompileOptions {
    fn from(opt_level: LlvmOptimizationLevel) -> Self {
        Self::from(FunctionCompileOptions::from(opt_level))
    }
}

/// User-facing NLP reference scales in the original problem units.
///
/// The solver internally normalizes with `q' = q / q_scale`.
pub struct TypedNlpScaling<X>
where
    X: Vectorize<SX>,
{
    pub variable: <X as Vectorize<SX>>::Rebind<f64>,
    pub constraints: Vec<f64>,
    pub objective: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeNlpScaling {
    pub variables: Vec<f64>,
    pub constraints: Vec<f64>,
    pub objective: f64,
}

impl<X, P, E, I> TypedCompiledJitNlp<X, P, E, I> {
    #[doc(hidden)]
    pub fn backend_compile_report_untyped(&self) -> &BackendCompileReport {
        self.inner.backend_compile_report()
    }
}

type TypedMarker<X, P, E, I> = PhantomData<fn() -> (X, P, E, I)>;

pub struct TypedRuntimeNlpBounds<X, I>
where
    X: Vectorize<SX>,
    I: Vectorize<SX>,
{
    pub variable_lower: Option<<X as Vectorize<SX>>::Rebind<Option<f64>>>,
    pub variable_upper: Option<<X as Vectorize<SX>>::Rebind<Option<f64>>>,
    pub inequality_lower: Option<<I as Vectorize<SX>>::Rebind<Option<f64>>>,
    pub inequality_upper: Option<<I as Vectorize<SX>>::Rebind<Option<f64>>>,
    pub scaling: Option<TypedNlpScaling<X>>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum RuntimeNlpParameterError {
    #[error("parameter value length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("parameter values were provided for a problem with no parameters")]
    UnexpectedValues,
    #[error("parameter values are required for this problem")]
    MissingValues,
}

impl<X, I> Default for TypedRuntimeNlpBounds<X, I>
where
    X: Vectorize<SX>,
    I: Vectorize<SX>,
{
    fn default() -> Self {
        Self {
            variable_lower: None,
            variable_upper: None,
            inequality_lower: None,
            inequality_upper: None,
            scaling: None,
        }
    }
}

#[derive(Clone, Debug)]
struct FlatNlpScaling {
    variable: Vec<f64>,
    equality: Vec<f64>,
    inequality: Vec<f64>,
    objective: f64,
}

#[derive(Clone, Debug)]
struct AppliedNlpScaling {
    variable: Vec<f64>,
    variable_inverse: Vec<f64>,
    equality: Vec<f64>,
    inequality: Vec<f64>,
    objective: f64,
    objective_inverse: f64,
    equality_jacobian_factors: Vec<f64>,
    inequality_jacobian_factors: Vec<f64>,
    hessian_factors: Vec<f64>,
}

struct ScaledNlpProblem<'a, P> {
    base: &'a P,
    scaling: AppliedNlpScaling,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintBounds {
    pub lower: Option<Vec<Option<f64>>>,
    pub upper: Option<Vec<Option<f64>>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeNlpBounds {
    pub variables: ConstraintBounds,
    pub inequalities: ConstraintBounds,
}

#[cfg(feature = "ipopt")]
fn fixed_bound_value(lower: Option<f64>, upper: Option<f64>) -> Option<f64> {
    let (Some(lower), Some(upper)) = (lower, upper) else {
        return None;
    };
    let tolerance = 1.0e-12 * lower.abs().max(upper.abs()).max(1.0);
    ((lower - upper).abs() <= tolerance).then_some(0.5 * (lower + upper))
}

#[cfg(feature = "ipopt")]
fn expand_ipopt_fixed_variable_snapshot(
    snapshot: &IpoptIterationSnapshot,
    full_dimension: usize,
    bounds: Option<&ConstraintBounds>,
) -> IpoptIterationSnapshot {
    if snapshot.x.len() == full_dimension {
        return snapshot.clone();
    }
    let Some(bounds) = bounds else {
        return snapshot.clone();
    };
    let reduced_dimension = snapshot.x.len();
    let mut expanded = Vec::with_capacity(full_dimension);
    let mut reduced_index = 0;
    let mut fixed_mask = Vec::with_capacity(full_dimension);
    for index in 0..full_dimension {
        let lower = bounds
            .lower
            .as_ref()
            .and_then(|values| values.get(index))
            .and_then(|value| *value);
        let upper = bounds
            .upper
            .as_ref()
            .and_then(|values| values.get(index))
            .and_then(|value| *value);
        if let Some(fixed_value) = fixed_bound_value(lower, upper) {
            expanded.push(fixed_value);
            fixed_mask.push(true);
        } else if let Some(&value) = snapshot.x.get(reduced_index) {
            expanded.push(value);
            fixed_mask.push(false);
            reduced_index += 1;
        } else {
            return snapshot.clone();
        }
    }
    if reduced_index != snapshot.x.len() {
        return snapshot.clone();
    }
    let mut snapshot = snapshot.clone();
    snapshot.x = expanded;
    if snapshot.lower_bound_multipliers.len() == reduced_dimension {
        snapshot.lower_bound_multipliers =
            expand_compact_ipopt_vector(&snapshot.lower_bound_multipliers, &fixed_mask);
    }
    if snapshot.upper_bound_multipliers.len() == reduced_dimension {
        snapshot.upper_bound_multipliers =
            expand_compact_ipopt_vector(&snapshot.upper_bound_multipliers, &fixed_mask);
    }
    if snapshot.direction_x.len() == reduced_dimension {
        snapshot.direction_x = expand_compact_ipopt_vector(&snapshot.direction_x, &fixed_mask);
    }
    if snapshot.direction_lower_bound_multipliers.len() == reduced_dimension {
        snapshot.direction_lower_bound_multipliers =
            expand_compact_ipopt_vector(&snapshot.direction_lower_bound_multipliers, &fixed_mask);
    }
    if snapshot.direction_upper_bound_multipliers.len() == reduced_dimension {
        snapshot.direction_upper_bound_multipliers =
            expand_compact_ipopt_vector(&snapshot.direction_upper_bound_multipliers, &fixed_mask);
    }
    if snapshot.kkt_x_stationarity.len() == reduced_dimension {
        snapshot.kkt_x_stationarity =
            expand_compact_ipopt_vector(&snapshot.kkt_x_stationarity, &fixed_mask);
    }
    if snapshot.curr_grad_f.len() == reduced_dimension {
        snapshot.curr_grad_f = expand_compact_ipopt_vector(&snapshot.curr_grad_f, &fixed_mask);
    }
    if snapshot.curr_jac_c_t_y_c.len() == reduced_dimension {
        snapshot.curr_jac_c_t_y_c =
            expand_compact_ipopt_vector(&snapshot.curr_jac_c_t_y_c, &fixed_mask);
    }
    if snapshot.curr_jac_d_t_y_d.len() == reduced_dimension {
        snapshot.curr_jac_d_t_y_d =
            expand_compact_ipopt_vector(&snapshot.curr_jac_d_t_y_d, &fixed_mask);
    }
    if snapshot.curr_grad_lag_x.len() == reduced_dimension {
        snapshot.curr_grad_lag_x =
            expand_compact_ipopt_vector(&snapshot.curr_grad_lag_x, &fixed_mask);
    }
    snapshot
}

#[cfg(feature = "ipopt")]
fn expand_compact_ipopt_vector(values: &[f64], fixed_mask: &[bool]) -> Vec<f64> {
    let mut expanded = Vec::with_capacity(fixed_mask.len());
    let mut reduced_index = 0;
    for &fixed in fixed_mask {
        if fixed {
            expanded.push(0.0);
        } else if let Some(&value) = values.get(reduced_index) {
            expanded.push(value);
            reduced_index += 1;
        } else {
            return values.to_vec();
        }
    }
    if reduced_index == values.len() {
        expanded
    } else {
        values.to_vec()
    }
}

#[derive(Debug)]
pub struct RuntimeBoundedJitNlp<'a> {
    base: &'a CompiledJitNlp,
    variable_bounds: ConstraintBounds,
    inequality_mapping: InequalityMapping,
    adapter_timing: Mutex<SqpAdapterTiming>,
}

#[derive(Debug)]
struct JitKernel {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
}

#[derive(Clone, Copy, Debug, Default)]
struct KernelEvalTiming {
    evaluation: Duration,
    output_marshalling: Duration,
}

#[derive(Clone, Debug)]
struct InequalityMapping {
    rows: Vec<ConstraintTransform>,
    inequality_jacobian_ccs: CCS,
    inequality_value_map: Vec<JacobianValueMap>,
}

#[derive(Clone, Copy, Debug)]
struct ConstraintTransform {
    source_index: Index,
    sign: f64,
    offset: f64,
}

#[derive(Clone, Copy, Debug)]
struct JacobianValueMap {
    source_value_index: Index,
    sign: f64,
}

#[derive(Debug, Error)]
pub enum SymbolicNlpBuildError {
    #[error("symbolic NLP name cannot be empty")]
    EmptyName,
    #[error(transparent)]
    Graph(#[from] SxError),
}

#[derive(Debug, Error)]
pub enum RuntimeNlpBoundsError {
    #[error(
        "variable bounds length mismatch: expected {expected}, got lower={lower_len}, upper={upper_len}"
    )]
    VariableBoundsLengthMismatch {
        expected: Index,
        lower_len: Index,
        upper_len: Index,
    },
    #[error(
        "constraint bounds length mismatch: expected {expected}, got lower={lower_len}, upper={upper_len}"
    )]
    ConstraintBoundsLengthMismatch {
        expected: Index,
        lower_len: Index,
        upper_len: Index,
    },
    #[error("invalid variable bounds at index {index}: lower={lower} > upper={upper}")]
    InvalidVariableBounds {
        index: Index,
        lower: f64,
        upper: f64,
    },
    #[error("invalid constraint bounds at index {index}: lower={lower} > upper={upper}")]
    InvalidConstraintBounds {
        index: Index,
        lower: f64,
        upper: f64,
    },
    #[error("variable scaling length mismatch: expected {expected}, got {actual}")]
    VariableScalingLengthMismatch { expected: Index, actual: Index },
    #[error("constraint scaling length mismatch: expected {expected}, got {actual}")]
    ConstraintScalingLengthMismatch { expected: Index, actual: Index },
    #[error(
        "invalid variable scaling at index {index}: expected positive finite value, got {value}"
    )]
    InvalidVariableScaling { index: Index, value: f64 },
    #[error(
        "invalid constraint scaling at index {index}: expected positive finite value, got {value}"
    )]
    InvalidConstraintScaling { index: Index, value: f64 },
    #[error("invalid objective scaling: expected positive finite value, got {value}")]
    InvalidObjectiveScaling { value: f64 },
}

#[derive(Debug, Error)]
pub enum SymbolicNlpCompileError {
    #[error(transparent)]
    Build(#[from] SymbolicNlpBuildError),
    #[error(transparent)]
    Graph(#[from] SxError),
    #[error("jit compilation failed: {0}")]
    Jit(#[from] anyhow::Error),
}

impl SymbolicNlp {
    pub fn new(
        name: impl Into<String>,
        variables: SXMatrix,
        parameters: Option<NamedMatrix>,
        objective: SXMatrix,
        equalities: Option<SXMatrix>,
        inequalities: Option<SXMatrix>,
    ) -> Result<Self, SymbolicNlpBuildError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(SymbolicNlpBuildError::EmptyName);
        }

        let objective_function = SXFunction::new(
            format!("{name}_objective_validation"),
            symbolic_inputs(&variables, &parameters)?,
            vec![NamedMatrix::new("objective", objective.clone())?],
        )?;
        debug_assert_eq!(
            objective_function.n_in(),
            usize::from(parameters.is_some()) + 1
        );
        let _ = objective.scalar_expr()?;

        let equalities = normalize_optional_matrix(equalities);
        if let Some(equalities) = &equalities {
            let validation = SXFunction::new(
                format!("{name}_equalities_validation"),
                symbolic_inputs(&variables, &parameters)?,
                vec![NamedMatrix::new("equalities", equalities.clone())?],
            )?;
            debug_assert_eq!(validation.n_in(), usize::from(parameters.is_some()) + 1);
        }
        let inequalities = normalize_optional_matrix(inequalities);
        if let Some(inequalities) = &inequalities {
            let validation = SXFunction::new(
                format!("{name}_inequalities_validation"),
                symbolic_inputs(&variables, &parameters)?,
                vec![NamedMatrix::new("inequalities", inequalities.clone())?],
            )?;
            debug_assert_eq!(validation.n_in(), usize::from(parameters.is_some()) + 1);
        }

        Ok(Self {
            name,
            variables,
            parameters,
            objective,
            equalities,
            inequalities,
            construction_time: None,
        })
    }
}

impl DynamicSymbolicNlp {
    pub fn construction_time(&self) -> Option<Duration> {
        self.symbolic.construction_time
    }

    pub fn compile_jit(&self) -> Result<DynamicCompiledJitNlp, SymbolicNlpCompileError> {
        self.compile_jit_with_options(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }

    pub fn compile_jit_with_options(
        &self,
        options: FunctionCompileOptions,
    ) -> Result<DynamicCompiledJitNlp, SymbolicNlpCompileError> {
        self.compile_jit_with_compile_options(SymbolicNlpCompileOptions::from(options))
    }

    pub fn compile_jit_with_compile_options(
        &self,
        options: SymbolicNlpCompileOptions,
    ) -> Result<DynamicCompiledJitNlp, SymbolicNlpCompileError> {
        self.compile_jit_with_compile_options_and_symbolic_progress_callback(options, |_| {})
    }

    pub fn compile_jit_with_compile_options_and_symbolic_progress_callback<CB>(
        &self,
        options: SymbolicNlpCompileOptions,
        on_symbolic_progress: CB,
    ) -> Result<DynamicCompiledJitNlp, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileProgress),
    {
        Ok(DynamicCompiledJitNlp {
            inner: compile_symbolic_nlp_with_symbolic_progress_callback(
                &self.symbolic,
                options,
                on_symbolic_progress,
            )?,
        })
    }
}

impl<X, P, E, I> TypedSymbolicNlp<X, P, E, I>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
{
    pub fn compile_jit(&self) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError> {
        self.compile_jit_with_options(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }

    pub fn compile_jit_with_symbolic_callback<CB>(
        &self,
        mut on_symbolic_ready: CB,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
    {
        self.compile_jit_with_options_and_symbolic_progress_callback(
            FunctionCompileOptions::from(LlvmOptimizationLevel::O3),
            |progress| {
                if let SymbolicCompileProgress::Ready(metadata) = progress {
                    on_symbolic_ready(metadata);
                }
            },
        )
    }

    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        BackendTimingMetadata {
            function_creation_time: self.symbolic.construction_time(),
            derivative_generation_time: None,
            jit_time: None,
        }
    }

    pub fn compile_jit_with_opt_level(
        &self,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError> {
        self.compile_jit_with_options(FunctionCompileOptions::from(opt_level))
    }

    pub fn compile_jit_with_options(
        &self,
        options: FunctionCompileOptions,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError> {
        self.compile_jit_with_compile_options(SymbolicNlpCompileOptions::from(options))
    }

    pub fn compile_jit_with_opt_level_and_symbolic_callback<CB>(
        &self,
        opt_level: LlvmOptimizationLevel,
        mut on_symbolic_ready: CB,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
    {
        self.compile_jit_with_options_and_symbolic_progress_callback(
            FunctionCompileOptions::from(opt_level),
            |progress| {
                if let SymbolicCompileProgress::Ready(metadata) = progress {
                    on_symbolic_ready(metadata);
                }
            },
        )
    }

    pub fn compile_jit_with_options_and_symbolic_callback<CB>(
        &self,
        options: FunctionCompileOptions,
        mut on_symbolic_ready: CB,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
    {
        self.compile_jit_with_compile_options_and_symbolic_progress_callback(
            SymbolicNlpCompileOptions::from(options),
            |progress| {
                if let SymbolicCompileProgress::Ready(metadata) = progress {
                    on_symbolic_ready(metadata);
                }
            },
        )
    }

    pub fn compile_jit_with_compile_options(
        &self,
        options: SymbolicNlpCompileOptions,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError> {
        self.compile_jit_with_compile_options_and_symbolic_progress_callback(options, |_| {})
    }

    pub fn compile_jit_with_compile_options_and_symbolic_callback<CB>(
        &self,
        options: SymbolicNlpCompileOptions,
        mut on_symbolic_ready: CB,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
    {
        self.compile_jit_with_compile_options_and_symbolic_progress_callback(options, |progress| {
            if let SymbolicCompileProgress::Ready(metadata) = progress {
                on_symbolic_ready(metadata);
            }
        })
    }

    pub fn compile_jit_with_options_and_symbolic_progress_callback<CB>(
        &self,
        options: FunctionCompileOptions,
        on_symbolic_progress: CB,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileProgress),
    {
        self.compile_jit_with_compile_options_and_symbolic_progress_callback(
            SymbolicNlpCompileOptions::from(options),
            on_symbolic_progress,
        )
    }

    pub fn compile_jit_with_compile_options_and_symbolic_progress_callback<CB>(
        &self,
        options: SymbolicNlpCompileOptions,
        on_symbolic_progress: CB,
    ) -> Result<TypedCompiledJitNlp<X, P, E, I>, SymbolicNlpCompileError>
    where
        CB: FnMut(SymbolicCompileProgress),
    {
        Ok(TypedCompiledJitNlp {
            inner: self
                .symbolic
                .compile_jit_with_compile_options_and_symbolic_progress_callback(
                    options,
                    on_symbolic_progress,
                )?,
            _marker: PhantomData,
        })
    }
}

pub fn symbolic_nlp<X, P, E, I, F>(
    name: impl Into<String>,
    model: F,
) -> Result<TypedSymbolicNlp<X, P, E, I>, SymbolicNlpBuildError>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    F: FnOnce(&X, &P) -> SymbolicNlpOutputs<E, I>,
{
    let started_at = Instant::now();
    let name = name.into();
    if name.trim().is_empty() {
        return Err(SymbolicNlpBuildError::EmptyName);
    }

    let variables = symbolic_value::<X>("x")?;
    let parameters = symbolic_value::<P>("p")?;
    let outputs = model(&variables, &parameters);

    let mut symbolic = symbolic_nlp_dynamic(
        name,
        symbolic_column(&variables)?,
        (P::LEN > 0)
            .then(|| symbolic_column(&parameters))
            .transpose()?,
        outputs.objective,
        (E::LEN > 0)
            .then(|| symbolic_column(&outputs.equalities))
            .transpose()?,
        (I::LEN > 0)
            .then(|| symbolic_column(&outputs.inequalities))
            .transpose()?,
    )?;
    symbolic.symbolic.construction_time = Some(started_at.elapsed());
    Ok(TypedSymbolicNlp {
        symbolic,
        _marker: PhantomData,
    })
}

pub fn symbolic_nlp_dynamic(
    name: impl Into<String>,
    variables: SXMatrix,
    parameters: Option<SXMatrix>,
    objective: SX,
    equalities: Option<SXMatrix>,
    inequalities: Option<SXMatrix>,
) -> Result<DynamicSymbolicNlp, SymbolicNlpBuildError> {
    let name = name.into();
    if name.trim().is_empty() {
        return Err(SymbolicNlpBuildError::EmptyName);
    }
    Ok(DynamicSymbolicNlp {
        symbolic: SymbolicNlp::new(
            name,
            variables,
            parameters
                .map(|matrix| NamedMatrix::new("p", matrix))
                .transpose()?,
            SXMatrix::scalar(objective),
            equalities,
            inequalities,
        )?,
    })
}

impl CompiledJitNlp {
    fn compile_stats(&self) -> NlpCompileStats {
        NlpCompileStats {
            variable_count: self.dimension(),
            parameter_scalar_count: self.parameter_ccs.iter().map(CCS::nnz).sum(),
            equality_count: self.equality_count(),
            inequality_count: self.inequality_base_count(),
            objective_gradient_nnz: self.dimension(),
            equality_jacobian_nnz: self.equality_jacobian_ccs().nnz(),
            inequality_jacobian_nnz: self.inequality_base_jacobian_ccs().nnz(),
            hessian_nnz: self.lagrangian_hessian_ccs().nnz(),
            jit_kernel_count: 3
                + 2 * usize::from(self.equality_values.is_some())
                + 2 * usize::from(self.inequality_values.is_some()),
        }
    }

    fn from_symbolic(
        symbolic: &SymbolicNlp,
        options: SymbolicNlpCompileOptions,
        mut on_symbolic_progress: impl FnMut(SymbolicCompileProgress),
    ) -> Result<Self, SymbolicNlpCompileError> {
        let derivative_started = Instant::now();
        let mut emit_symbolic_stage =
            |stage: SymbolicCompileStage,
             setup_profile: &SymbolicSetupProfile,
             stats: NlpCompileStats| {
                on_symbolic_progress(SymbolicCompileProgress::Stage(
                    SymbolicCompileStageProgress {
                        stage,
                        metadata: SymbolicCompileMetadata {
                            timing: BackendTimingMetadata {
                                function_creation_time: symbolic.construction_time,
                                derivative_generation_time: Some(derivative_started.elapsed()),
                                jit_time: None,
                            },
                            setup_profile: setup_profile.clone(),
                            stats,
                        },
                    },
                ));
            };
        emit_symbolic_stage(
            SymbolicCompileStage::BuildProblem,
            &SymbolicSetupProfile {
                symbolic_construction: symbolic.construction_time,
                ..SymbolicSetupProfile::default()
            },
            symbolic_compile_stats(symbolic, 0, 0, 0, 0),
        );
        let functions = derive_symbolic_functions(
            symbolic,
            options.hessian_strategy,
            &mut emit_symbolic_stage,
        )?;
        let derivative_generation_time = derivative_started.elapsed();
        let timing = BackendTimingMetadata {
            function_creation_time: symbolic.construction_time,
            derivative_generation_time: Some(derivative_generation_time),
            jit_time: None,
        };
        on_symbolic_progress(SymbolicCompileProgress::Ready(SymbolicCompileMetadata {
            timing,
            setup_profile: functions.setup_profile.clone(),
            stats: derived_symbolic_compile_stats(symbolic, &functions),
        }));

        let jit_started = Instant::now();
        let objective_value =
            JitKernel::compile_with_options(&functions.objective_value, options.function_options)?;
        let objective_gradient = JitKernel::compile_with_options(
            &functions.objective_gradient,
            options.function_options,
        )?;
        let equality_values = functions
            .equality_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options.function_options))
            .transpose()?;
        let equality_jacobian_values = functions
            .equality_jacobian_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options.function_options))
            .transpose()?;
        let inequality_values = functions
            .inequality_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options.function_options))
            .transpose()?;
        let inequality_jacobian_values = functions
            .inequality_jacobian_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options.function_options))
            .transpose()?;
        let lagrangian_hessian_values = JitKernel::compile_with_options(
            &functions.lagrangian_hessian_values,
            options.function_options,
        )?;
        let jit_time = jit_started.elapsed();
        let mut backend_compile_report = BackendCompileReport {
            timing: BackendTimingMetadata {
                jit_time: Some(jit_time),
                ..timing
            },
            setup_profile: functions.setup_profile.clone(),
            ..BackendCompileReport::default()
        };
        absorb_kernel_compile_report(&mut backend_compile_report, &objective_value);
        absorb_kernel_compile_report(&mut backend_compile_report, &objective_gradient);
        if let Some(kernel) = &equality_values {
            absorb_kernel_compile_report(&mut backend_compile_report, kernel);
        }
        if let Some(kernel) = &equality_jacobian_values {
            absorb_kernel_compile_report(&mut backend_compile_report, kernel);
        }
        if let Some(kernel) = &inequality_values {
            absorb_kernel_compile_report(&mut backend_compile_report, kernel);
        }
        if let Some(kernel) = &inequality_jacobian_values {
            absorb_kernel_compile_report(&mut backend_compile_report, kernel);
        }
        absorb_kernel_compile_report(&mut backend_compile_report, &lagrangian_hessian_values);

        Ok(Self {
            dimension: symbolic.variables.nnz(),
            parameter_ccs: symbolic
                .parameters
                .iter()
                .map(|parameter| ccs_from_core(parameter.matrix().ccs()))
                .collect(),
            equality_jacobian_ccs: functions.equality_jacobian_values.as_ref().map_or_else(
                || CCS::empty(0, symbolic.variables.nnz()),
                function_output_ccs,
            ),
            inequality_base_jacobian_ccs: functions
                .inequality_jacobian_values
                .as_ref()
                .map_or_else(
                    || CCS::empty(0, symbolic.variables.nnz()),
                    function_output_ccs,
                ),
            lagrangian_hessian_ccs: function_output_ccs(&functions.lagrangian_hessian_values),
            backend_timing: backend_compile_report.timing,
            backend_compile_report,
            objective_value,
            objective_gradient,
            equality_values,
            equality_jacobian_values,
            inequality_values,
            inequality_jacobian_values,
            lagrangian_hessian_values,
        })
    }

    pub fn dimension(&self) -> Index {
        self.dimension
    }

    pub fn parameter_count(&self) -> Index {
        self.parameter_ccs.len()
    }

    pub fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        &self.parameter_ccs[parameter_index]
    }

    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.backend_timing
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        &self.backend_compile_report
    }

    pub fn equality_count(&self) -> Index {
        self.equality_jacobian_ccs.nrow
    }

    pub fn inequality_base_count(&self) -> Index {
        self.inequality_base_jacobian_ccs.nrow
    }

    pub fn equality_jacobian_ccs(&self) -> &CCS {
        &self.equality_jacobian_ccs
    }

    pub fn inequality_base_jacobian_ccs(&self) -> &CCS {
        &self.inequality_base_jacobian_ccs
    }

    pub fn lagrangian_hessian_ccs(&self) -> &CCS {
        &self.lagrangian_hessian_ccs
    }

    fn objective_value_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
    ) -> (f64, SqpAdapterTiming) {
        let (value, timing) = self.objective_value.eval_scalar_timed(x, parameters);
        (
            value,
            SqpAdapterTiming {
                callback_evaluation: timing.evaluation,
                output_marshalling: timing.output_marshalling,
                layout_projection: Duration::ZERO,
            },
        )
    }

    fn objective_gradient_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) -> SqpAdapterTiming {
        let timing = self
            .objective_gradient
            .eval_vector_timed(x, parameters, out);
        SqpAdapterTiming {
            callback_evaluation: timing.evaluation,
            output_marshalling: timing.output_marshalling,
            layout_projection: Duration::ZERO,
        }
    }

    fn equality_values_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) -> SqpAdapterTiming {
        let timing = self
            .equality_values
            .as_ref()
            .map_or_else(KernelEvalTiming::default, |kernel| {
                kernel.eval_vector_timed(x, parameters, out)
            });
        SqpAdapterTiming {
            callback_evaluation: timing.evaluation,
            output_marshalling: timing.output_marshalling,
            layout_projection: Duration::ZERO,
        }
    }

    fn equality_jacobian_values_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) -> SqpAdapterTiming {
        let timing = self
            .equality_jacobian_values
            .as_ref()
            .map_or_else(KernelEvalTiming::default, |kernel| {
                kernel.eval_vector_timed(x, parameters, out)
            });
        SqpAdapterTiming {
            callback_evaluation: timing.evaluation,
            output_marshalling: timing.output_marshalling,
            layout_projection: Duration::ZERO,
        }
    }

    fn inequality_values_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) -> SqpAdapterTiming {
        let timing = self
            .inequality_values
            .as_ref()
            .map_or_else(KernelEvalTiming::default, |kernel| {
                kernel.eval_vector_timed(x, parameters, out)
            });
        SqpAdapterTiming {
            callback_evaluation: timing.evaluation,
            output_marshalling: timing.output_marshalling,
            layout_projection: Duration::ZERO,
        }
    }

    fn inequality_jacobian_values_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) -> SqpAdapterTiming {
        let timing = self
            .inequality_jacobian_values
            .as_ref()
            .map_or_else(KernelEvalTiming::default, |kernel| {
                kernel.eval_vector_timed(x, parameters, out)
            });
        SqpAdapterTiming {
            callback_evaluation: timing.evaluation,
            output_marshalling: timing.output_marshalling,
            layout_projection: Duration::ZERO,
        }
    }

    fn lagrangian_hessian_values_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) -> SqpAdapterTiming {
        let timing = self.lagrangian_hessian_values.eval_hessian_timed(
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
            out,
        );
        SqpAdapterTiming {
            callback_evaluation: timing.evaluation,
            output_marshalling: timing.output_marshalling,
            layout_projection: Duration::ZERO,
        }
    }

    pub fn bind_runtime_bounds(
        &self,
        bounds: RuntimeNlpBounds,
    ) -> Result<RuntimeBoundedJitNlp<'_>, RuntimeNlpBoundsError> {
        let projection_started = Instant::now();
        let variable_bounds = validate_bound_vectors(self.dimension, bounds.variables, true)?;
        let inequality_bounds =
            validate_bound_vectors(self.inequality_base_count(), bounds.inequalities, false)?;
        let mapping = InequalityMapping::from_runtime_bounds(
            self.inequality_base_jacobian_ccs(),
            &inequality_bounds,
        );
        Ok(RuntimeBoundedJitNlp {
            base: self,
            variable_bounds,
            inequality_mapping: mapping,
            adapter_timing: Mutex::new(SqpAdapterTiming {
                layout_projection: projection_started.elapsed(),
                ..SqpAdapterTiming::default()
            }),
        })
    }
}

impl DynamicCompiledJitNlp {
    #[doc(hidden)]
    pub fn debug_lagrangian_hessian_lowered(&self) -> &LoweredFunction {
        self.inner.lagrangian_hessian_values.function.lowered()
    }

    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.inner.backend_timing_metadata()
    }

    pub fn compile_stats(&self) -> NlpCompileStats {
        self.inner.compile_stats()
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        self.inner.backend_compile_report()
    }

    pub(crate) fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.inner.parameter_ccs(parameter_index)
    }

    pub(crate) fn equality_count(&self) -> Index {
        self.inner.equality_count()
    }

    pub(crate) fn inequality_base_count(&self) -> Index {
        self.inner.inequality_base_count()
    }

    pub fn parameter_vector_ccs(&self) -> Option<&CCS> {
        (self.inner.parameter_count() > 0).then(|| self.inner.parameter_ccs(0))
    }

    pub fn parameter_storage<'a>(
        &'a self,
        values: Option<&'a [f64]>,
    ) -> Result<Vec<ParameterMatrix<'a>>, RuntimeNlpParameterError> {
        match (self.parameter_vector_ccs(), values) {
            (None, None) => Ok(Vec::new()),
            (None, Some(_)) => Err(RuntimeNlpParameterError::UnexpectedValues),
            (Some(_ccs), None) => Err(RuntimeNlpParameterError::MissingValues),
            (Some(ccs), Some(values)) => {
                if values.len() != ccs.nnz() {
                    return Err(RuntimeNlpParameterError::LengthMismatch {
                        expected: ccs.nnz(),
                        actual: values.len(),
                    });
                }
                Ok(vec![ParameterMatrix { ccs, values }])
            }
        }
    }

    pub fn bind_runtime_bounds(
        &self,
        bounds: &RuntimeNlpBounds,
    ) -> Result<RuntimeBoundedJitNlp<'_>, RuntimeNlpBoundsError> {
        self.inner.bind_runtime_bounds(bounds.clone())
    }

    pub fn benchmark_bounded_evaluations_with_progress<CB>(
        &self,
        x: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> AnyResult<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let bound_problem = self.bind_runtime_bounds(bounds)?;
        let scaling = self.build_applied_scaling(scaling, &bound_problem)?;
        let mut x_values = x.to_vec();
        let parameter_storage = self.parameter_storage(parameters)?;
        if let Some(scaling) = scaling {
            x_values = scaling.scale_x(&x_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling,
            };
            Ok(benchmark_compiled_nlp_problem_with_progress(
                &scaled_problem,
                &x_values,
                &parameter_storage,
                options,
                on_progress,
            ))
        } else {
            Ok(benchmark_compiled_nlp_problem_with_progress(
                &bound_problem,
                &x_values,
                &parameter_storage,
                options,
                on_progress,
            ))
        }
    }

    pub fn benchmark_bounded_evaluations(
        &self,
        x: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: NlpEvaluationBenchmarkOptions,
    ) -> AnyResult<NlpEvaluationBenchmark> {
        self.benchmark_bounded_evaluations_with_progress(
            x,
            parameters,
            bounds,
            scaling,
            options,
            |_| {},
        )
    }

    pub fn validate_derivatives_flat_values(
        &self,
        x: &[f64],
        parameter_values: Option<&[f64]>,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let parameter_storage = self.parameter_storage(parameter_values)?;
        validate_compiled_nlp_problem_derivatives(
            &self.inner,
            x,
            &parameter_storage,
            equality_multipliers,
            inequality_multipliers,
            options,
        )
    }

    pub fn solve_sqp(
        &self,
        x0: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: &ClarabelSqpOptions,
    ) -> Result<ClarabelSqpSummary, ClarabelSqpError> {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(scaling, &bound_problem)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let mut x0_values = x0.to_vec();
        let parameter_storage = self
            .parameter_storage(parameters)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        bound_problem.record_layout_projection(projection_started.elapsed());
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            solve_nlp_sqp(&scaled_problem, &x0_values, &parameter_storage, options)
                .map(|summary| scaling.transform_sqp_summary(&summary))
                .map_err(|error| transform_sqp_error(&scaling, error))
        } else {
            solve_nlp_sqp(&bound_problem, &x0_values, &parameter_storage, options)
        }
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        x0: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<ClarabelSqpSummary, ClarabelSqpError>
    where
        CB: FnMut(&crate::SqpIterationSnapshot),
    {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(scaling, &bound_problem)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let mut x0_values = x0.to_vec();
        let parameter_storage = self
            .parameter_storage(parameters)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        bound_problem.record_layout_projection(projection_started.elapsed());
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            let mut callback = callback;
            solve_nlp_sqp_with_callback(
                &scaled_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| callback(&scaling.transform_sqp_snapshot(snapshot)),
            )
            .map(|summary| scaling.transform_sqp_summary(&summary))
            .map_err(|error| transform_sqp_error(&scaling, error))
        } else {
            solve_nlp_sqp_with_callback(
                &bound_problem,
                &x0_values,
                &parameter_storage,
                options,
                callback,
            )
        }
    }

    pub fn solve_interior_point(
        &self,
        x0: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: &InteriorPointOptions,
    ) -> Result<InteriorPointSummary, InteriorPointSolveError> {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(scaling, &bound_problem)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = x0.to_vec();
        let parameter_storage = self
            .parameter_storage(parameters)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        bound_problem.record_layout_projection(projection_started.elapsed());
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            solve_nlp_interior_point(&scaled_problem, &x0_values, &parameter_storage, options)
                .map(|summary| scaling.transform_interior_point_summary(&summary))
        } else {
            solve_nlp_interior_point(&bound_problem, &x0_values, &parameter_storage, options)
        }
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        x0: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<InteriorPointSummary, InteriorPointSolveError>
    where
        CB: FnMut(&InteriorPointIterationSnapshot),
    {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(scaling, &bound_problem)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = x0.to_vec();
        let parameter_storage = self
            .parameter_storage(parameters)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        bound_problem.record_layout_projection(projection_started.elapsed());
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            let mut callback = callback;
            solve_nlp_interior_point_with_callback(
                &scaled_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| callback(&scaling.transform_interior_point_snapshot(snapshot)),
            )
            .map(|summary| scaling.transform_interior_point_summary(&summary))
        } else {
            solve_nlp_interior_point_with_callback(
                &bound_problem,
                &x0_values,
                &parameter_storage,
                options,
                callback,
            )
        }
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        x0: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: &IpoptOptions,
    ) -> Result<IpoptSummary, IpoptSolveError> {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(scaling, &bound_problem)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = x0.to_vec();
        let parameter_storage = self
            .parameter_storage(parameters)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        bound_problem.record_layout_projection(projection_started.elapsed());
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            solve_nlp_ipopt(&scaled_problem, &x0_values, &parameter_storage, options)
                .map(|summary| scaling.transform_ipopt_summary(&summary))
                .map_err(|error| transform_ipopt_error(&scaling, error))
        } else {
            solve_nlp_ipopt(&bound_problem, &x0_values, &parameter_storage, options)
        }
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt_with_callback<CB>(
        &self,
        x0: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        scaling: Option<&RuntimeNlpScaling>,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<IpoptSummary, IpoptSolveError>
    where
        CB: FnMut(&IpoptIterationSnapshot),
    {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(scaling, &bound_problem)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = x0.to_vec();
        let parameter_storage = self
            .parameter_storage(parameters)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        bound_problem.record_layout_projection(projection_started.elapsed());
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            let scaled_variable_bounds = scaled_problem.variable_bounds();
            let full_dimension = x0_values.len();
            let mut callback = callback;
            solve_nlp_ipopt_with_callback(
                &scaled_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| {
                    let snapshot = expand_ipopt_fixed_variable_snapshot(
                        snapshot,
                        full_dimension,
                        scaled_variable_bounds.as_ref(),
                    );
                    callback(&scaling.transform_ipopt_snapshot(&snapshot));
                },
            )
            .map(|summary| scaling.transform_ipopt_summary(&summary))
            .map_err(|error| transform_ipopt_error(&scaling, error))
        } else {
            let variable_bounds = bound_problem.variable_bounds();
            let full_dimension = x0_values.len();
            let mut callback = callback;
            solve_nlp_ipopt_with_callback(
                &bound_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| {
                    let snapshot = expand_ipopt_fixed_variable_snapshot(
                        snapshot,
                        full_dimension,
                        variable_bounds.as_ref(),
                    );
                    callback(&snapshot);
                },
            )
        }
    }

    pub fn rank_constraint_violations(
        &self,
        x: &[f64],
        parameters: Option<&[f64]>,
        bounds: &RuntimeNlpBounds,
        tolerance: f64,
    ) -> AnyResult<NlpConstraintViolationReport> {
        let parameter_storage = self.parameter_storage(parameters)?;
        Ok(rank_nlp_constraint_violations(
            &self.inner,
            x,
            &parameter_storage,
            bounds,
            tolerance,
        )?)
    }

    pub fn evaluate_equalities_flat(
        &self,
        x: &[f64],
        parameters: Option<&[f64]>,
    ) -> Result<Vec<f64>, RuntimeNlpParameterError> {
        let parameter_storage = self.parameter_storage(parameters)?;
        let mut values = vec![0.0; self.inner.equality_count()];
        let _ = self
            .inner
            .equality_values_timed(x, &parameter_storage, &mut values);
        Ok(values)
    }

    pub fn evaluate_inequalities_flat(
        &self,
        x: &[f64],
        parameters: Option<&[f64]>,
    ) -> Result<Vec<f64>, RuntimeNlpParameterError> {
        let parameter_storage = self.parameter_storage(parameters)?;
        let mut values = vec![0.0; self.inner.inequality_base_count()];
        let _ = self
            .inner
            .inequality_values_timed(x, &parameter_storage, &mut values);
        Ok(values)
    }

    fn build_applied_scaling(
        &self,
        scaling: Option<&RuntimeNlpScaling>,
        problem: &RuntimeBoundedJitNlp<'_>,
    ) -> Result<Option<AppliedNlpScaling>, RuntimeNlpBoundsError> {
        let Some(scaling) = scaling else {
            return Ok(None);
        };
        let equality_count = problem.equality_count();
        if scaling.constraints.len() < equality_count {
            return Err(RuntimeNlpBoundsError::ConstraintScalingLengthMismatch {
                expected: equality_count,
                actual: scaling.constraints.len(),
            });
        }
        let variable = invert_scaling_vector(validate_scaling_vector(
            problem.dimension(),
            scaling.variables.clone(),
            true,
        )?);
        let constraint = invert_scaling_vector(validate_scaling_vector(
            problem.equality_count() + problem.base.inequality_base_count(),
            scaling.constraints.clone(),
            false,
        )?);
        if !scaling.objective.is_finite() || scaling.objective <= 0.0 {
            return Err(RuntimeNlpBoundsError::InvalidObjectiveScaling {
                value: scaling.objective,
            });
        }
        let flat = FlatNlpScaling {
            variable,
            equality: constraint[..equality_count].to_vec(),
            inequality: constraint[equality_count..].to_vec(),
            objective: 1.0 / scaling.objective,
        };
        AppliedNlpScaling::from_runtime_problem(problem, flat).map(Some)
    }
}

fn transform_sqp_error(scaling: &AppliedNlpScaling, error: ClarabelSqpError) -> ClarabelSqpError {
    match error {
        ClarabelSqpError::InvalidInput(message) => ClarabelSqpError::InvalidInput(message),
        ClarabelSqpError::NonFiniteInput { stage } => ClarabelSqpError::NonFiniteInput { stage },
        ClarabelSqpError::MaxIterations {
            iterations,
            context,
        } => ClarabelSqpError::MaxIterations {
            iterations,
            context: Box::new(scaling.transform_sqp_context(&context)),
        },
        ClarabelSqpError::Setup(message) => ClarabelSqpError::Setup(message),
        ClarabelSqpError::QpSolve { status, context } => ClarabelSqpError::QpSolve {
            status,
            context: Box::new(scaling.transform_sqp_context(&context)),
        },
        ClarabelSqpError::UnconstrainedStepSolve { context } => {
            ClarabelSqpError::UnconstrainedStepSolve {
                context: Box::new(scaling.transform_sqp_context(&context)),
            }
        }
        ClarabelSqpError::LineSearchFailed {
            directional_derivative,
            step_inf_norm,
            penalty,
            context,
        } => ClarabelSqpError::LineSearchFailed {
            directional_derivative,
            step_inf_norm,
            penalty,
            context: Box::new(scaling.transform_sqp_context(&context)),
        },
        ClarabelSqpError::Stalled {
            step_inf_norm,
            primal_inf_norm,
            dual_inf_norm,
            complementarity_inf_norm,
            context,
        } => ClarabelSqpError::Stalled {
            step_inf_norm,
            primal_inf_norm,
            dual_inf_norm,
            complementarity_inf_norm,
            context: Box::new(scaling.transform_sqp_context(&context)),
        },
        ClarabelSqpError::RestorationFailed {
            step_inf_norm,
            context,
        } => ClarabelSqpError::RestorationFailed {
            step_inf_norm,
            context: Box::new(scaling.transform_sqp_context(&context)),
        },
        ClarabelSqpError::NonFiniteCallbackOutput { stage, context } => {
            ClarabelSqpError::NonFiniteCallbackOutput {
                stage,
                context: Box::new(scaling.transform_sqp_context(&context)),
            }
        }
    }
}

#[cfg(feature = "ipopt")]
fn transform_ipopt_error(scaling: &AppliedNlpScaling, error: IpoptSolveError) -> IpoptSolveError {
    match error {
        IpoptSolveError::InvalidInput(message) => IpoptSolveError::InvalidInput(message),
        IpoptSolveError::Setup(message) => IpoptSolveError::Setup(message),
        IpoptSolveError::OptionRejected { name } => IpoptSolveError::OptionRejected { name },
        IpoptSolveError::Solve {
            status,
            iterations,
            snapshots,
            partial_solution,
            journal_output,
            profiling,
        } => IpoptSolveError::Solve {
            status,
            iterations,
            snapshots: snapshots
                .iter()
                .map(|snapshot| scaling.transform_ipopt_snapshot(snapshot))
                .collect(),
            partial_solution: partial_solution.map(|partial| {
                Box::new(scaling.transform_ipopt_partial_solution(partial.as_ref()))
            }),
            journal_output,
            profiling,
        },
    }
}

impl<X, P, E, I> TypedCompiledJitNlp<X, P, E, I>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <E as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
{
    #[doc(hidden)]
    pub fn debug_lagrangian_hessian_lowered(&self) -> &LoweredFunction {
        self.inner
            .inner
            .lagrangian_hessian_values
            .function
            .lowered()
    }

    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.inner.backend_timing_metadata()
    }

    pub fn compile_stats(&self) -> NlpCompileStats {
        self.inner.compile_stats()
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        self.inner.backend_compile_report()
    }

    pub fn evaluate_equalities_flat(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
    ) -> Vec<f64> {
        let x_values = flatten_value(x);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        let mut values = vec![0.0; self.inner.equality_count()];
        let _ = self
            .inner
            .inner
            .equality_values_timed(&x_values, &parameter_storage, &mut values);
        values
    }

    pub fn evaluate_inequalities_flat(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
    ) -> Vec<f64> {
        let x_values = flatten_value(x);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        let mut values = vec![0.0; self.inner.inequality_base_count()];
        let _ =
            self.inner
                .inner
                .inequality_values_timed(&x_values, &parameter_storage, &mut values);
        values
    }
}

impl<X, P, E, I> TypedCompiledJitNlp<X, P, E, I>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <X as Vectorize<SX>>::Rebind<Option<f64>>: Vectorize<Option<f64>>,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <E as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<Option<f64>>: Vectorize<Option<f64>>,
{
    pub fn bind_runtime_bounds(
        &self,
        bounds: &TypedRuntimeNlpBounds<X, I>,
    ) -> Result<RuntimeBoundedJitNlp<'_>, RuntimeNlpBoundsError> {
        self.inner
            .bind_runtime_bounds(&self.flatten_runtime_bounds(bounds))
    }

    fn flatten_runtime_bounds(&self, bounds: &TypedRuntimeNlpBounds<X, I>) -> RuntimeNlpBounds {
        RuntimeNlpBounds {
            variables: ConstraintBounds {
                lower: bounds.variable_lower.as_ref().map(flatten_optional_value),
                upper: bounds.variable_upper.as_ref().map(flatten_optional_value),
            },
            inequalities: ConstraintBounds {
                lower: bounds.inequality_lower.as_ref().map(flatten_optional_value),
                upper: bounds.inequality_upper.as_ref().map(flatten_optional_value),
            },
        }
    }

    fn parameter_storage<'a>(&'a self, parameter_values: &'a [f64]) -> Vec<ParameterMatrix<'a>> {
        if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: parameter_values,
            }]
        }
    }

    fn build_applied_scaling(
        &self,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        problem: &RuntimeBoundedJitNlp<'_>,
    ) -> Result<Option<AppliedNlpScaling>, RuntimeNlpBoundsError> {
        let Some(scaling) = bounds.scaling.as_ref() else {
            return Ok(None);
        };
        let variable = invert_scaling_vector(validate_scaling_vector(
            X::LEN,
            flatten_value(&scaling.variable),
            true,
        )?);
        let constraint = invert_scaling_vector(validate_scaling_vector(
            E::LEN + I::LEN,
            scaling.constraints.clone(),
            false,
        )?);
        if !scaling.objective.is_finite() || scaling.objective <= 0.0 {
            return Err(RuntimeNlpBoundsError::InvalidObjectiveScaling {
                value: scaling.objective,
            });
        }
        let flat = FlatNlpScaling {
            variable,
            equality: constraint[..E::LEN].to_vec(),
            inequality: constraint[E::LEN..].to_vec(),
            objective: 1.0 / scaling.objective,
        };
        AppliedNlpScaling::from_runtime_problem(problem, flat).map(Some)
    }

    fn transform_sqp_error(
        &self,
        scaling: &AppliedNlpScaling,
        error: ClarabelSqpError,
    ) -> ClarabelSqpError {
        match error {
            ClarabelSqpError::InvalidInput(message) => ClarabelSqpError::InvalidInput(message),
            ClarabelSqpError::NonFiniteInput { stage } => {
                ClarabelSqpError::NonFiniteInput { stage }
            }
            ClarabelSqpError::MaxIterations {
                iterations,
                context,
            } => ClarabelSqpError::MaxIterations {
                iterations,
                context: Box::new(scaling.transform_sqp_context(&context)),
            },
            ClarabelSqpError::Setup(message) => ClarabelSqpError::Setup(message),
            ClarabelSqpError::QpSolve { status, context } => ClarabelSqpError::QpSolve {
                status,
                context: Box::new(scaling.transform_sqp_context(&context)),
            },
            ClarabelSqpError::UnconstrainedStepSolve { context } => {
                ClarabelSqpError::UnconstrainedStepSolve {
                    context: Box::new(scaling.transform_sqp_context(&context)),
                }
            }
            ClarabelSqpError::LineSearchFailed {
                directional_derivative,
                step_inf_norm,
                penalty,
                context,
            } => ClarabelSqpError::LineSearchFailed {
                directional_derivative,
                step_inf_norm,
                penalty,
                context: Box::new(scaling.transform_sqp_context(&context)),
            },
            ClarabelSqpError::Stalled {
                step_inf_norm,
                primal_inf_norm,
                dual_inf_norm,
                complementarity_inf_norm,
                context,
            } => ClarabelSqpError::Stalled {
                step_inf_norm,
                primal_inf_norm,
                dual_inf_norm,
                complementarity_inf_norm,
                context: Box::new(scaling.transform_sqp_context(&context)),
            },
            ClarabelSqpError::RestorationFailed {
                step_inf_norm,
                context,
            } => ClarabelSqpError::RestorationFailed {
                step_inf_norm,
                context: Box::new(scaling.transform_sqp_context(&context)),
            },
            ClarabelSqpError::NonFiniteCallbackOutput { stage, context } => {
                ClarabelSqpError::NonFiniteCallbackOutput {
                    stage,
                    context: Box::new(scaling.transform_sqp_context(&context)),
                }
            }
        }
    }

    #[cfg(feature = "ipopt")]
    fn transform_ipopt_error(
        &self,
        scaling: &AppliedNlpScaling,
        error: IpoptSolveError,
    ) -> IpoptSolveError {
        match error {
            IpoptSolveError::InvalidInput(message) => IpoptSolveError::InvalidInput(message),
            IpoptSolveError::Setup(message) => IpoptSolveError::Setup(message),
            IpoptSolveError::OptionRejected { name } => IpoptSolveError::OptionRejected { name },
            IpoptSolveError::Solve {
                status,
                iterations,
                snapshots,
                partial_solution,
                journal_output,
                profiling,
            } => IpoptSolveError::Solve {
                status,
                iterations,
                snapshots: snapshots
                    .iter()
                    .map(|snapshot| scaling.transform_ipopt_snapshot(snapshot))
                    .collect(),
                partial_solution: partial_solution.map(|partial| {
                    Box::new(scaling.transform_ipopt_partial_solution(partial.as_ref()))
                }),
                journal_output,
                profiling,
            },
        }
    }

    pub fn benchmark_evaluations(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        options: NlpEvaluationBenchmarkOptions,
    ) -> NlpEvaluationBenchmark {
        self.benchmark_evaluations_with_progress(x, parameters, options, |_| {})
    }

    pub fn benchmark_evaluations_with_progress<CB>(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> NlpEvaluationBenchmark
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let x_values = flatten_value(x);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = self.parameter_storage(&parameter_values);
        benchmark_compiled_nlp_problem_with_progress(
            &self.inner,
            &x_values,
            &parameter_storage,
            options,
            on_progress,
        )
    }

    pub fn benchmark_bounded_evaluations(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: NlpEvaluationBenchmarkOptions,
    ) -> Result<NlpEvaluationBenchmark, RuntimeNlpBoundsError> {
        self.benchmark_bounded_evaluations_with_progress(x, parameters, bounds, options, |_| {})
    }

    pub fn benchmark_bounded_evaluations_with_progress<CB>(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark, RuntimeNlpBoundsError>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let bound_problem = self.bind_runtime_bounds(bounds)?;
        let scaling = self.build_applied_scaling(bounds, &bound_problem)?;
        let mut x_values = flatten_value(x);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x_values = scaling.scale_x(&x_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling,
            };
            Ok(benchmark_compiled_nlp_problem_with_progress(
                &scaled_problem,
                &x_values,
                &parameter_storage,
                options,
                on_progress,
            ))
        } else {
            Ok(benchmark_compiled_nlp_problem_with_progress(
                &bound_problem,
                &x_values,
                &parameter_storage,
                options,
                on_progress,
            ))
        }
    }

    pub fn validate_derivatives(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        equality_multipliers: &<E as Vectorize<SX>>::Rebind<f64>,
        inequality_multipliers: &<I as Vectorize<SX>>::Rebind<f64>,
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let x_values = flatten_value(x);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        let equality_multiplier_values = flatten_value(equality_multipliers);
        let inequality_multiplier_values = flatten_value(inequality_multipliers);
        validate_compiled_nlp_problem_derivatives(
            &self.inner,
            &x_values,
            &parameter_storage,
            &equality_multiplier_values,
            &inequality_multiplier_values,
            options,
        )
    }

    pub fn validate_derivatives_flat(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        validate_compiled_nlp_problem_derivatives(
            &self.inner,
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
            options,
        )
    }

    pub fn validate_derivatives_flat_values(
        &self,
        x: &[f64],
        parameter_values: &[f64],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: parameter_values,
            }]
        };
        validate_compiled_nlp_problem_derivatives(
            &self.inner,
            x,
            &parameter_storage,
            equality_multipliers,
            inequality_multipliers,
            options,
        )
    }

    pub fn solve_sqp(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &ClarabelSqpOptions,
    ) -> Result<ClarabelSqpSummary, ClarabelSqpError> {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(bounds, &bound_problem)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let mut x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            solve_nlp_sqp(&scaled_problem, &x0_values, &parameter_storage, options)
                .map(|summary| scaling.transform_sqp_summary(&summary))
                .map_err(|error| self.transform_sqp_error(&scaling, error))
        } else {
            solve_nlp_sqp(&bound_problem, &x0_values, &parameter_storage, options)
        }
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<ClarabelSqpSummary, ClarabelSqpError>
    where
        CB: FnMut(&crate::SqpIterationSnapshot),
    {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(bounds, &bound_problem)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let mut x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            let mut callback = callback;
            solve_nlp_sqp_with_callback(
                &scaled_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| callback(&scaling.transform_sqp_snapshot(snapshot)),
            )
            .map(|summary| scaling.transform_sqp_summary(&summary))
            .map_err(|error| self.transform_sqp_error(&scaling, error))
        } else {
            solve_nlp_sqp_with_callback(
                &bound_problem,
                &x0_values,
                &parameter_storage,
                options,
                callback,
            )
        }
    }

    pub fn solve_interior_point(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &InteriorPointOptions,
    ) -> Result<InteriorPointSummary, InteriorPointSolveError> {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(bounds, &bound_problem)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            solve_nlp_interior_point(&scaled_problem, &x0_values, &parameter_storage, options)
                .map(|summary| scaling.transform_interior_point_summary(&summary))
        } else {
            solve_nlp_interior_point(&bound_problem, &x0_values, &parameter_storage, options)
        }
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<InteriorPointSummary, InteriorPointSolveError>
    where
        CB: FnMut(&InteriorPointIterationSnapshot),
    {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(bounds, &bound_problem)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            let mut callback = callback;
            solve_nlp_interior_point_with_callback(
                &scaled_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| callback(&scaling.transform_interior_point_snapshot(snapshot)),
            )
            .map(|summary| scaling.transform_interior_point_summary(&summary))
        } else {
            solve_nlp_interior_point_with_callback(
                &bound_problem,
                &x0_values,
                &parameter_storage,
                options,
                callback,
            )
        }
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &IpoptOptions,
    ) -> Result<IpoptSummary, IpoptSolveError> {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(bounds, &bound_problem)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            solve_nlp_ipopt(&scaled_problem, &x0_values, &parameter_storage, options)
                .map(|summary| scaling.transform_ipopt_summary(&summary))
                .map_err(|error| self.transform_ipopt_error(&scaling, error))
        } else {
            solve_nlp_ipopt(&bound_problem, &x0_values, &parameter_storage, options)
        }
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt_with_callback<CB>(
        &self,
        x0: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<IpoptSummary, IpoptSolveError>
    where
        CB: FnMut(&IpoptIterationSnapshot),
    {
        let projection_started = Instant::now();
        let bound_problem = self
            .bind_runtime_bounds(bounds)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let scaling = self
            .build_applied_scaling(bounds, &bound_problem)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let mut x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = self.parameter_storage(&parameter_values);
        if let Some(scaling) = scaling {
            x0_values = scaling.scale_x(&x0_values);
            let scaled_problem = ScaledNlpProblem {
                base: &bound_problem,
                scaling: scaling.clone(),
            };
            let scaled_variable_bounds = scaled_problem.variable_bounds();
            let full_dimension = x0_values.len();
            let mut callback = callback;
            solve_nlp_ipopt_with_callback(
                &scaled_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| {
                    let snapshot = expand_ipopt_fixed_variable_snapshot(
                        snapshot,
                        full_dimension,
                        scaled_variable_bounds.as_ref(),
                    );
                    callback(&scaling.transform_ipopt_snapshot(&snapshot));
                },
            )
            .map(|summary| scaling.transform_ipopt_summary(&summary))
            .map_err(|error| self.transform_ipopt_error(&scaling, error))
        } else {
            let variable_bounds = bound_problem.variable_bounds();
            let full_dimension = x0_values.len();
            let mut callback = callback;
            solve_nlp_ipopt_with_callback(
                &bound_problem,
                &x0_values,
                &parameter_storage,
                options,
                |snapshot| {
                    let snapshot = expand_ipopt_fixed_variable_snapshot(
                        snapshot,
                        full_dimension,
                        variable_bounds.as_ref(),
                    );
                    callback(&snapshot);
                },
            )
        }
    }

    pub fn rank_constraint_violations(
        &self,
        x: &<X as Vectorize<SX>>::Rebind<f64>,
        parameters: &<P as Vectorize<SX>>::Rebind<f64>,
        bounds: &TypedRuntimeNlpBounds<X, I>,
        tolerance: f64,
    ) -> Result<NlpConstraintViolationReport, RuntimeNlpBoundsError> {
        let x_values = flatten_value(x);
        let parameter_values = flatten_value(parameters);
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        rank_nlp_constraint_violations(
            &self.inner,
            &x_values,
            &parameter_storage,
            &RuntimeNlpBounds {
                variables: ConstraintBounds {
                    lower: bounds.variable_lower.as_ref().map(flatten_optional_value),
                    upper: bounds.variable_upper.as_ref().map(flatten_optional_value),
                },
                inequalities: ConstraintBounds {
                    lower: bounds.inequality_lower.as_ref().map(flatten_optional_value),
                    upper: bounds.inequality_upper.as_ref().map(flatten_optional_value),
                },
            },
            tolerance,
        )
    }
}

pub fn rank_nlp_constraint_violations(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    bounds: &RuntimeNlpBounds,
    tolerance: f64,
) -> Result<NlpConstraintViolationReport, RuntimeNlpBoundsError> {
    let variable_bounds =
        validate_bound_vectors(problem.dimension(), bounds.variables.clone(), true)?;
    let inequality_bounds = validate_bound_vectors(
        problem.inequality_count(),
        bounds.inequalities.clone(),
        false,
    )?;

    let mut equalities = vec![0.0; problem.equality_count()];
    problem.equality_values(x, parameters, &mut equalities);

    let mut inequalities = vec![0.0; problem.inequality_count()];
    problem.inequality_values(x, parameters, &mut inequalities);

    let mut report = NlpConstraintViolationReport {
        equalities: equalities
            .into_iter()
            .enumerate()
            .map(|(row, value)| {
                let abs_violation = value.abs();
                NlpEqualityViolation {
                    row,
                    value,
                    abs_violation,
                    satisfaction: classify_constraint_satisfaction(abs_violation, tolerance),
                }
            })
            .collect(),
        ..NlpConstraintViolationReport::default()
    };
    report
        .equalities
        .sort_by(|lhs, rhs| rhs.abs_violation.total_cmp(&lhs.abs_violation));

    let inequality_lower = inequality_bounds.lower.unwrap_or_default();
    let inequality_upper = inequality_bounds.upper.unwrap_or_default();
    for (row, value) in inequalities.into_iter().enumerate() {
        let lower_bound = inequality_lower.get(row).copied().flatten();
        let upper_bound = inequality_upper.get(row).copied().flatten();
        if lower_bound.is_none() && upper_bound.is_none() {
            continue;
        }
        let (lower_violation, upper_violation) =
            worst_bound_violation(value, lower_bound, upper_bound);
        let worst_violation = lower_violation.max(upper_violation);
        report.inequalities.push(NlpInequalityViolation {
            source: NlpInequalitySource::ConstraintRow { row },
            value,
            lower_bound,
            upper_bound,
            lower_violation,
            upper_violation,
            worst_violation,
            bound_side: constraint_bound_side(lower_violation, upper_violation),
            satisfaction: classify_constraint_satisfaction(worst_violation, tolerance),
        });
    }

    let variable_lower = variable_bounds.lower.unwrap_or_default();
    let variable_upper = variable_bounds.upper.unwrap_or_default();
    for (index, &value) in x.iter().enumerate() {
        let lower_bound = variable_lower.get(index).copied().flatten();
        let upper_bound = variable_upper.get(index).copied().flatten();
        if lower_bound.is_none() && upper_bound.is_none() {
            continue;
        }
        let (lower_violation, upper_violation) =
            worst_bound_violation(value, lower_bound, upper_bound);
        let worst_violation = lower_violation.max(upper_violation);
        report.inequalities.push(NlpInequalityViolation {
            source: NlpInequalitySource::VariableBound { index },
            value,
            lower_bound,
            upper_bound,
            lower_violation,
            upper_violation,
            worst_violation,
            bound_side: constraint_bound_side(lower_violation, upper_violation),
            satisfaction: classify_constraint_satisfaction(worst_violation, tolerance),
        });
    }
    report
        .inequalities
        .sort_by(|lhs, rhs| rhs.worst_violation.total_cmp(&lhs.worst_violation));
    Ok(report)
}

fn compile_symbolic_nlp_with_symbolic_progress_callback(
    symbolic: &SymbolicNlp,
    options: SymbolicNlpCompileOptions,
    on_symbolic_progress: impl FnMut(SymbolicCompileProgress),
) -> Result<CompiledJitNlp, SymbolicNlpCompileError> {
    CompiledJitNlp::from_symbolic(symbolic, options, on_symbolic_progress)
}

fn add_duration(total: &mut Option<Duration>, elapsed: Duration) {
    *total = Some(total.unwrap_or_default() + elapsed);
}

fn absorb_kernel_compile_report(report: &mut BackendCompileReport, kernel: &JitKernel) {
    let compile_report = kernel.function.compile_report();
    add_duration(
        &mut report.setup_profile.lowering,
        compile_report.lowering_time,
    );
    add_duration(&mut report.setup_profile.llvm_jit, compile_report.llvm_time);
    if compile_report.cache.hit {
        report.llvm_jit_cache.hits += 1;
        report.llvm_jit_cache.load_time += compile_report.cache.load_time;
    } else {
        report.llvm_jit_cache.misses += 1;
    }
    report.stats.absorb(&compile_report.stats);
    report
        .warnings
        .extend(compile_report.warnings.iter().cloned());
}

impl RuntimeBoundedJitNlp<'_> {
    fn record_adapter_timing(&self, timing: SqpAdapterTiming) {
        let mut totals = lock_context(&self.adapter_timing);
        totals.callback_evaluation += timing.callback_evaluation;
        totals.output_marshalling += timing.output_marshalling;
        totals.layout_projection += timing.layout_projection;
    }

    fn record_layout_projection(&self, elapsed: Duration) {
        let mut totals = lock_context(&self.adapter_timing);
        totals.layout_projection += elapsed;
    }
}

fn validate_scaling_vector(
    expected: Index,
    values: Vec<f64>,
    is_variable: bool,
) -> Result<Vec<f64>, RuntimeNlpBoundsError> {
    if values.len() != expected {
        return Err(if is_variable {
            RuntimeNlpBoundsError::VariableScalingLengthMismatch {
                expected,
                actual: values.len(),
            }
        } else {
            RuntimeNlpBoundsError::ConstraintScalingLengthMismatch {
                expected,
                actual: values.len(),
            }
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(if is_variable {
                RuntimeNlpBoundsError::InvalidVariableScaling { index, value }
            } else {
                RuntimeNlpBoundsError::InvalidConstraintScaling { index, value }
            });
        }
    }
    Ok(values)
}

fn invert_scaling_vector(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(|value| 1.0 / value).collect()
}

fn scaled_jacobian_factors(ccs: &CCS, row_scale: &[f64], variable_inverse: &[f64]) -> Vec<f64> {
    let mut factors = Vec::with_capacity(ccs.nnz());
    for (col, &inverse) in variable_inverse.iter().enumerate().take(ccs.ncol) {
        for index in ccs.col_ptrs[col]..ccs.col_ptrs[col + 1] {
            let row = ccs.row_indices[index];
            factors.push(row_scale[row] * inverse);
        }
    }
    factors
}

fn scaled_hessian_factors(ccs: &CCS, objective: f64, variable_inverse: &[f64]) -> Vec<f64> {
    let mut factors = Vec::with_capacity(ccs.nnz());
    for col in 0..ccs.ncol {
        for index in ccs.col_ptrs[col]..ccs.col_ptrs[col + 1] {
            let row = ccs.row_indices[index];
            factors.push(objective * variable_inverse[row] * variable_inverse[col]);
        }
    }
    factors
}

impl AppliedNlpScaling {
    fn from_runtime_problem(
        problem: &RuntimeBoundedJitNlp<'_>,
        scaling: FlatNlpScaling,
    ) -> Result<Self, RuntimeNlpBoundsError> {
        if !scaling.objective.is_finite() || scaling.objective <= 0.0 {
            return Err(RuntimeNlpBoundsError::InvalidObjectiveScaling {
                value: scaling.objective,
            });
        }
        let variable = validate_scaling_vector(problem.dimension(), scaling.variable, true)?;
        let equality = validate_scaling_vector(problem.equality_count(), scaling.equality, false)?;
        let raw_inequality = validate_scaling_vector(
            problem.base.inequality_base_count(),
            scaling.inequality,
            false,
        )?;
        let variable_inverse = variable
            .iter()
            .map(|value| 1.0 / *value)
            .collect::<Vec<_>>();
        let inequality = problem
            .inequality_mapping
            .rows
            .iter()
            .map(|row| raw_inequality[row.source_index])
            .collect::<Vec<_>>();
        let equality_jacobian_factors = scaled_jacobian_factors(
            problem.equality_jacobian_ccs(),
            &equality,
            &variable_inverse,
        );
        let inequality_jacobian_factors = scaled_jacobian_factors(
            problem.inequality_jacobian_ccs(),
            &inequality,
            &variable_inverse,
        );
        let hessian_factors = scaled_hessian_factors(
            problem.lagrangian_hessian_ccs(),
            scaling.objective,
            &variable_inverse,
        );
        Ok(Self {
            variable,
            variable_inverse,
            equality,
            inequality,
            objective: scaling.objective,
            objective_inverse: 1.0 / scaling.objective,
            equality_jacobian_factors,
            inequality_jacobian_factors,
            hessian_factors,
        })
    }

    fn scale_x(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.variable.iter())
            .map(|(value, scale)| value * scale)
            .collect()
    }

    fn unscale_x(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.variable_inverse.iter())
            .map(|(value, scale)| value * scale)
            .collect()
    }

    fn unscale_objective(&self, objective: f64) -> f64 {
        objective * self.objective_inverse
    }

    fn unscale_equality_multipliers(&self, multipliers: &[f64]) -> Vec<f64> {
        multipliers
            .iter()
            .zip(self.equality.iter())
            .map(|(multiplier, scale)| multiplier * scale * self.objective_inverse)
            .collect()
    }

    fn unscale_inequality_multipliers(&self, multipliers: &[f64]) -> Vec<f64> {
        multipliers
            .iter()
            .zip(self.inequality.iter())
            .map(|(multiplier, scale)| multiplier * scale * self.objective_inverse)
            .collect()
    }

    fn unscale_inequality_values(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(self.inequality.iter())
            .map(|(value, scale)| value / scale)
            .collect()
    }

    fn unscale_bound_multipliers(&self, multipliers: &[f64]) -> Vec<f64> {
        multipliers
            .iter()
            .zip(self.variable.iter())
            .map(|(multiplier, scale)| multiplier * scale * self.objective_inverse)
            .collect()
    }

    fn scale_hessian_equality_multipliers(&self, multipliers: &[f64]) -> Vec<f64> {
        multipliers
            .iter()
            .zip(self.equality.iter())
            .map(|(multiplier, scale)| multiplier * scale * self.objective_inverse)
            .collect()
    }

    fn scale_hessian_inequality_multipliers(&self, multipliers: &[f64]) -> Vec<f64> {
        multipliers
            .iter()
            .zip(self.inequality.iter())
            .map(|(multiplier, scale)| multiplier * scale * self.objective_inverse)
            .collect()
    }

    fn scale_variable_bounds(&self, bounds: &mut ConstraintBounds) {
        if let Some(lower) = bounds.lower.as_mut() {
            for (bound, scale) in lower.iter_mut().zip(self.variable.iter().copied()) {
                if let Some(value) = bound.as_mut() {
                    *value *= scale;
                }
            }
        }
        if let Some(upper) = bounds.upper.as_mut() {
            for (bound, scale) in upper.iter_mut().zip(self.variable.iter().copied()) {
                if let Some(value) = bound.as_mut() {
                    *value *= scale;
                }
            }
        }
    }

    fn transform_sqp_snapshot(&self, snapshot: &SqpIterationSnapshot) -> SqpIterationSnapshot {
        let mut snapshot = snapshot.clone();
        snapshot.x = self.unscale_x(&snapshot.x);
        snapshot.objective = self.unscale_objective(snapshot.objective);
        snapshot
    }

    fn transform_sqp_context(&self, context: &SqpFailureContext) -> SqpFailureContext {
        SqpFailureContext {
            termination: context.termination,
            final_state: context
                .final_state
                .as_ref()
                .map(|snapshot| self.transform_sqp_snapshot(snapshot)),
            final_state_kind: context.final_state_kind,
            last_accepted_state: context
                .last_accepted_state
                .as_ref()
                .map(|snapshot| self.transform_sqp_snapshot(snapshot)),
            failed_line_search: context.failed_line_search.clone(),
            failed_trust_region: context.failed_trust_region.clone(),
            failed_step_diagnostics: context.failed_step_diagnostics.clone(),
            qp_failure: context.qp_failure.clone(),
            profiling: context.profiling.clone(),
        }
    }

    fn transform_sqp_summary(&self, summary: &ClarabelSqpSummary) -> ClarabelSqpSummary {
        ClarabelSqpSummary {
            x: self.unscale_x(&summary.x),
            equality_multipliers: self.unscale_equality_multipliers(&summary.equality_multipliers),
            inequality_multipliers: self
                .unscale_inequality_multipliers(&summary.inequality_multipliers),
            lower_bound_multipliers: self
                .unscale_bound_multipliers(&summary.lower_bound_multipliers),
            upper_bound_multipliers: self
                .unscale_bound_multipliers(&summary.upper_bound_multipliers),
            objective: self.unscale_objective(summary.objective),
            iterations: summary.iterations,
            equality_inf_norm: summary.equality_inf_norm,
            inequality_inf_norm: summary.inequality_inf_norm,
            primal_inf_norm: summary.primal_inf_norm,
            dual_inf_norm: summary.dual_inf_norm,
            complementarity_inf_norm: summary.complementarity_inf_norm,
            overall_inf_norm: summary.overall_inf_norm,
            termination: summary.termination,
            final_state: self.transform_sqp_snapshot(&summary.final_state),
            final_state_kind: summary.final_state_kind,
            last_accepted_state: summary
                .last_accepted_state
                .as_ref()
                .map(|snapshot| self.transform_sqp_snapshot(snapshot)),
            profiling: summary.profiling.clone(),
        }
    }

    fn transform_interior_point_snapshot(
        &self,
        snapshot: &InteriorPointIterationSnapshot,
    ) -> InteriorPointIterationSnapshot {
        let mut snapshot = snapshot.clone();
        snapshot.x = self.unscale_x(&snapshot.x);
        snapshot.slack_primal = snapshot
            .slack_primal
            .as_ref()
            .map(|values| self.unscale_inequality_values(values));
        snapshot.equality_multipliers = snapshot
            .equality_multipliers
            .as_ref()
            .map(|multipliers| self.unscale_equality_multipliers(multipliers));
        snapshot.inequality_multipliers = snapshot
            .inequality_multipliers
            .as_ref()
            .map(|multipliers| self.unscale_inequality_multipliers(multipliers));
        snapshot.slack_multipliers = snapshot
            .slack_multipliers
            .as_ref()
            .map(|multipliers| self.unscale_inequality_multipliers(multipliers));
        snapshot.lower_bound_multipliers = snapshot
            .lower_bound_multipliers
            .as_ref()
            .map(|multipliers| self.unscale_bound_multipliers(multipliers));
        snapshot.upper_bound_multipliers = snapshot
            .upper_bound_multipliers
            .as_ref()
            .map(|multipliers| self.unscale_bound_multipliers(multipliers));
        if let Some(direction) = snapshot.step_direction.as_mut() {
            direction.x = self.unscale_x(&direction.x);
            direction.slack = self.unscale_inequality_values(&direction.slack);
            direction.equality_multipliers =
                self.unscale_equality_multipliers(&direction.equality_multipliers);
            direction.inequality_multipliers =
                self.unscale_inequality_multipliers(&direction.inequality_multipliers);
            direction.slack_multipliers =
                self.unscale_inequality_multipliers(&direction.slack_multipliers);
            direction.lower_bound_multipliers =
                self.unscale_bound_multipliers(&direction.lower_bound_multipliers);
            direction.upper_bound_multipliers =
                self.unscale_bound_multipliers(&direction.upper_bound_multipliers);
        }
        // KKT diagnostic vectors are intentionally left in internal solver units so NLIP and
        // IPOPT can be compared before user-unit scaling is applied to display quantities.
        snapshot.objective = self.unscale_objective(snapshot.objective);
        snapshot
    }

    fn transform_interior_point_summary(
        &self,
        summary: &InteriorPointSummary,
    ) -> InteriorPointSummary {
        InteriorPointSummary {
            x: self.unscale_x(&summary.x),
            equality_multipliers: self.unscale_equality_multipliers(&summary.equality_multipliers),
            inequality_multipliers: self
                .unscale_inequality_multipliers(&summary.inequality_multipliers),
            lower_bound_multipliers: self
                .unscale_bound_multipliers(&summary.lower_bound_multipliers),
            upper_bound_multipliers: self
                .unscale_bound_multipliers(&summary.upper_bound_multipliers),
            slack: summary.slack.clone(),
            objective: self.unscale_objective(summary.objective),
            iterations: summary.iterations,
            equality_inf_norm: summary.equality_inf_norm,
            inequality_inf_norm: summary.inequality_inf_norm,
            primal_inf_norm: summary.primal_inf_norm,
            dual_inf_norm: summary.dual_inf_norm,
            complementarity_inf_norm: summary.complementarity_inf_norm,
            overall_inf_norm: summary.overall_inf_norm,
            barrier_parameter: summary.barrier_parameter,
            termination: summary.termination,
            status_kind: summary.status_kind,
            snapshots: summary
                .snapshots
                .iter()
                .map(|snapshot| self.transform_interior_point_snapshot(snapshot))
                .collect(),
            final_state: self.transform_interior_point_snapshot(&summary.final_state),
            last_accepted_state: summary
                .last_accepted_state
                .as_ref()
                .map(|snapshot| self.transform_interior_point_snapshot(snapshot)),
            profiling: summary.profiling.clone(),
            linear_solver: summary.linear_solver,
        }
    }

    #[cfg(feature = "ipopt")]
    fn transform_ipopt_snapshot(
        &self,
        snapshot: &IpoptIterationSnapshot,
    ) -> IpoptIterationSnapshot {
        let mut snapshot = snapshot.clone();
        snapshot.x = self.unscale_x(&snapshot.x);
        snapshot.internal_slack = self.unscale_inequality_values(&snapshot.internal_slack);
        snapshot.equality_multipliers =
            self.unscale_equality_multipliers(&snapshot.equality_multipliers);
        snapshot.inequality_multipliers =
            self.unscale_inequality_multipliers(&snapshot.inequality_multipliers);
        snapshot.lower_bound_multipliers =
            self.unscale_bound_multipliers(&snapshot.lower_bound_multipliers);
        snapshot.upper_bound_multipliers =
            self.unscale_bound_multipliers(&snapshot.upper_bound_multipliers);
        snapshot.slack_lower_bound_multipliers =
            self.unscale_inequality_multipliers(&snapshot.slack_lower_bound_multipliers);
        snapshot.slack_upper_bound_multipliers =
            self.unscale_inequality_multipliers(&snapshot.slack_upper_bound_multipliers);
        snapshot.direction_x = self.unscale_x(&snapshot.direction_x);
        snapshot.direction_slack = self.unscale_inequality_values(&snapshot.direction_slack);
        snapshot.direction_equality_multipliers =
            self.unscale_equality_multipliers(&snapshot.direction_equality_multipliers);
        snapshot.direction_inequality_multipliers =
            self.unscale_inequality_multipliers(&snapshot.direction_inequality_multipliers);
        snapshot.direction_lower_bound_multipliers =
            self.unscale_bound_multipliers(&snapshot.direction_lower_bound_multipliers);
        snapshot.direction_upper_bound_multipliers =
            self.unscale_bound_multipliers(&snapshot.direction_upper_bound_multipliers);
        snapshot.direction_slack_lower_bound_multipliers =
            self.unscale_inequality_multipliers(&snapshot.direction_slack_lower_bound_multipliers);
        snapshot.direction_slack_upper_bound_multipliers =
            self.unscale_inequality_multipliers(&snapshot.direction_slack_upper_bound_multipliers);
        // KKT diagnostic vectors remain in IPOPT's internal solver units. The slack distance is
        // state-like and is scaled back for direct comparison against NLIP's internal slack via
        // the upper-bound distance conversion.
        snapshot.kkt_slack_distance = self.unscale_inequality_values(&snapshot.kkt_slack_distance);
        snapshot.objective = self.unscale_objective(snapshot.objective);
        snapshot
    }

    #[cfg(feature = "ipopt")]
    fn transform_ipopt_partial_solution(
        &self,
        partial: &IpoptPartialSolution,
    ) -> IpoptPartialSolution {
        IpoptPartialSolution {
            x: self.unscale_x(&partial.x),
            lower_bound_multipliers: self
                .unscale_bound_multipliers(&partial.lower_bound_multipliers),
            upper_bound_multipliers: self
                .unscale_bound_multipliers(&partial.upper_bound_multipliers),
            equality_multipliers: self.unscale_equality_multipliers(&partial.equality_multipliers),
            inequality_multipliers: self
                .unscale_inequality_multipliers(&partial.inequality_multipliers),
            objective: self.unscale_objective(partial.objective),
            equality_inf_norm: partial.equality_inf_norm,
            inequality_inf_norm: partial.inequality_inf_norm,
            primal_inf_norm: partial.primal_inf_norm,
            dual_inf_norm: partial.dual_inf_norm,
            complementarity_inf_norm: partial.complementarity_inf_norm,
        }
    }

    #[cfg(feature = "ipopt")]
    fn transform_ipopt_summary(&self, summary: &IpoptSummary) -> IpoptSummary {
        IpoptSummary {
            x: self.unscale_x(&summary.x),
            lower_bound_multipliers: self
                .unscale_bound_multipliers(&summary.lower_bound_multipliers),
            upper_bound_multipliers: self
                .unscale_bound_multipliers(&summary.upper_bound_multipliers),
            equality_multipliers: self.unscale_equality_multipliers(&summary.equality_multipliers),
            inequality_multipliers: self
                .unscale_inequality_multipliers(&summary.inequality_multipliers),
            objective: self.unscale_objective(summary.objective),
            iterations: summary.iterations,
            status: summary.status,
            equality_inf_norm: summary.equality_inf_norm,
            inequality_inf_norm: summary.inequality_inf_norm,
            primal_inf_norm: summary.primal_inf_norm,
            dual_inf_norm: summary.dual_inf_norm,
            complementarity_inf_norm: summary.complementarity_inf_norm,
            snapshots: summary
                .snapshots
                .iter()
                .map(|snapshot| self.transform_ipopt_snapshot(snapshot))
                .collect(),
            journal_output: summary.journal_output.clone(),
            profiling: summary.profiling.clone(),
            provenance: summary.provenance.clone(),
        }
    }
}

impl<P> CompiledNlpProblem for ScaledNlpProblem<'_, P>
where
    P: CompiledNlpProblem,
{
    fn dimension(&self) -> Index {
        self.base.dimension()
    }

    fn parameter_count(&self) -> Index {
        self.base.parameter_count()
    }

    fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.base.parameter_ccs(parameter_index)
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        let mut bounds = self.base.variable_bounds()?;
        self.scaling.scale_variable_bounds(&mut bounds);
        Some(bounds)
    }

    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.base.backend_timing_metadata()
    }

    fn backend_compile_report(&self) -> Option<&BackendCompileReport> {
        self.base.backend_compile_report()
    }

    fn adapter_timing_snapshot(&self) -> Option<SqpAdapterTiming> {
        self.base.sqp_adapter_timing_snapshot()
    }

    fn sqp_adapter_timing_snapshot(&self) -> Option<SqpAdapterTiming> {
        self.base.sqp_adapter_timing_snapshot()
    }

    fn ipopt_nlp_scaling_method(&self) -> Option<&'static str> {
        Some("none")
    }

    fn equality_count(&self) -> Index {
        self.base.equality_count()
    }

    fn inequality_count(&self) -> Index {
        self.base.inequality_count()
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        let unscaled_x = self.scaling.unscale_x(x);
        self.scaling.objective * self.base.objective_value(&unscaled_x, parameters)
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let unscaled_x = self.scaling.unscale_x(x);
        self.base.objective_gradient(&unscaled_x, parameters, out);
        for (index, value) in out.iter_mut().enumerate() {
            *value *= self.scaling.objective * self.scaling.variable_inverse[index];
        }
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        self.base.equality_jacobian_ccs()
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let unscaled_x = self.scaling.unscale_x(x);
        self.base.equality_values(&unscaled_x, parameters, out);
        for (value, scale) in out.iter_mut().zip(self.scaling.equality.iter()) {
            *value *= scale;
        }
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let unscaled_x = self.scaling.unscale_x(x);
        self.base
            .equality_jacobian_values(&unscaled_x, parameters, out);
        for (value, factor) in out
            .iter_mut()
            .zip(self.scaling.equality_jacobian_factors.iter())
        {
            *value *= factor;
        }
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        self.base.inequality_jacobian_ccs()
    }

    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let unscaled_x = self.scaling.unscale_x(x);
        self.base.inequality_values(&unscaled_x, parameters, out);
        for (value, scale) in out.iter_mut().zip(self.scaling.inequality.iter()) {
            *value *= scale;
        }
    }

    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let unscaled_x = self.scaling.unscale_x(x);
        self.base
            .inequality_jacobian_values(&unscaled_x, parameters, out);
        for (value, factor) in out
            .iter_mut()
            .zip(self.scaling.inequality_jacobian_factors.iter())
        {
            *value *= factor;
        }
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        self.base.lagrangian_hessian_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let unscaled_x = self.scaling.unscale_x(x);
        let base_equality_multipliers = self
            .scaling
            .scale_hessian_equality_multipliers(equality_multipliers);
        let base_inequality_multipliers = self
            .scaling
            .scale_hessian_inequality_multipliers(inequality_multipliers);
        self.base.lagrangian_hessian_values(
            &unscaled_x,
            parameters,
            &base_equality_multipliers,
            &base_inequality_multipliers,
            out,
        );
        for (value, factor) in out.iter_mut().zip(self.scaling.hessian_factors.iter()) {
            *value *= factor;
        }
    }
}

impl CompiledNlpProblem for CompiledJitNlp {
    fn dimension(&self) -> Index {
        self.dimension()
    }

    fn parameter_count(&self) -> Index {
        self.parameter_count()
    }

    fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.parameter_ccs(parameter_index)
    }

    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.backend_timing_metadata()
    }

    fn backend_compile_report(&self) -> Option<&BackendCompileReport> {
        Some(self.backend_compile_report())
    }

    fn equality_count(&self) -> Index {
        self.equality_count()
    }

    fn inequality_count(&self) -> Index {
        self.inequality_base_count()
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        self.objective_value_timed(x, parameters).0
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let _ = self.objective_gradient_timed(x, parameters, out);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        self.equality_jacobian_ccs()
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let _ = self.equality_values_timed(x, parameters, out);
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let _ = self.equality_jacobian_values_timed(x, parameters, out);
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        self.inequality_base_jacobian_ccs()
    }

    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let _ = self.inequality_values_timed(x, parameters, out);
    }

    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let _ = self.inequality_jacobian_values_timed(x, parameters, out);
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        self.lagrangian_hessian_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let _ = self.lagrangian_hessian_values_timed(
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
            out,
        );
    }
}

impl CompiledNlpProblem for DynamicCompiledJitNlp {
    fn dimension(&self) -> Index {
        self.inner.dimension()
    }

    fn parameter_count(&self) -> Index {
        self.inner.parameter_count()
    }

    fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.inner.parameter_ccs(parameter_index)
    }

    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.inner.backend_timing_metadata()
    }

    fn backend_compile_report(&self) -> Option<&BackendCompileReport> {
        Some(self.inner.backend_compile_report())
    }

    fn equality_count(&self) -> Index {
        self.inner.equality_count()
    }

    fn inequality_count(&self) -> Index {
        self.inner.inequality_base_count()
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        self.inner.objective_value(x, parameters)
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        self.inner.objective_gradient(x, parameters, out);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        self.inner.equality_jacobian_ccs()
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        self.inner.equality_values(x, parameters, out);
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        self.inner.equality_jacobian_values(x, parameters, out);
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        self.inner.inequality_jacobian_ccs()
    }

    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        self.inner.inequality_values(x, parameters, out);
    }

    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        self.inner.inequality_jacobian_values(x, parameters, out);
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        self.inner.lagrangian_hessian_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        self.inner.lagrangian_hessian_values(
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
            out,
        );
    }
}

impl CompiledNlpProblem for RuntimeBoundedJitNlp<'_> {
    fn dimension(&self) -> Index {
        self.base.dimension()
    }

    fn parameter_count(&self) -> Index {
        self.base.parameter_count()
    }

    fn parameter_ccs(&self, parameter_index: Index) -> &CCS {
        self.base.parameter_ccs(parameter_index)
    }

    fn variable_bounds(&self) -> Option<ConstraintBounds> {
        let started = Instant::now();
        let bounds = self.variable_bounds.clone();
        self.record_layout_projection(started.elapsed());
        Some(bounds)
    }

    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.base.backend_timing_metadata()
    }

    fn backend_compile_report(&self) -> Option<&BackendCompileReport> {
        Some(self.base.backend_compile_report())
    }

    fn sqp_adapter_timing_snapshot(&self) -> Option<SqpAdapterTiming> {
        Some(*lock_context(&self.adapter_timing))
    }

    fn equality_count(&self) -> Index {
        self.base.equality_count()
    }

    fn inequality_count(&self) -> Index {
        self.inequality_mapping.rows.len()
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        let (value, timing) = self.base.objective_value_timed(x, parameters);
        self.record_adapter_timing(timing);
        value
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let timing = self.base.objective_gradient_timed(x, parameters, out);
        self.record_adapter_timing(timing);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        self.base.equality_jacobian_ccs()
    }

    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let timing = self.base.equality_values_timed(x, parameters, out);
        self.record_adapter_timing(timing);
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let timing = self.base.equality_jacobian_values_timed(x, parameters, out);
        self.record_adapter_timing(timing);
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        &self.inequality_mapping.inequality_jacobian_ccs
    }

    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let layout_started = Instant::now();
        let mut inequality_values = vec![0.0; self.base.inequality_base_count()];
        let base_timing = self
            .base
            .inequality_values_timed(x, parameters, &mut inequality_values);
        for (slot, transform) in out.iter_mut().zip(self.inequality_mapping.rows.iter()) {
            *slot = transform.sign * inequality_values[transform.source_index] + transform.offset;
        }
        self.record_adapter_timing(SqpAdapterTiming {
            callback_evaluation: base_timing.callback_evaluation,
            output_marshalling: base_timing.output_marshalling,
            layout_projection: layout_started.elapsed(),
        });
    }

    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        let layout_started = Instant::now();
        let mut source_values = vec![0.0; self.base.inequality_base_jacobian_ccs().nnz()];
        let base_timing =
            self.base
                .inequality_jacobian_values_timed(x, parameters, &mut source_values);
        for (slot, mapping) in out
            .iter_mut()
            .zip(self.inequality_mapping.inequality_value_map.iter())
        {
            *slot = mapping.sign * source_values[mapping.source_value_index];
        }
        self.record_adapter_timing(SqpAdapterTiming {
            callback_evaluation: base_timing.callback_evaluation,
            output_marshalling: base_timing.output_marshalling,
            layout_projection: layout_started.elapsed(),
        });
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        self.base.lagrangian_hessian_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let layout_started = Instant::now();
        let mut base_inequality_multipliers = vec![0.0; self.base.inequality_base_count()];
        for (multiplier, transform) in inequality_multipliers
            .iter()
            .zip(self.inequality_mapping.rows.iter())
        {
            base_inequality_multipliers[transform.source_index] += transform.sign * multiplier;
        }
        let base_timing = self.base.lagrangian_hessian_values_timed(
            x,
            parameters,
            equality_multipliers,
            &base_inequality_multipliers,
            out,
        );
        self.record_adapter_timing(SqpAdapterTiming {
            callback_evaluation: base_timing.callback_evaluation,
            output_marshalling: base_timing.output_marshalling,
            layout_projection: layout_started.elapsed(),
        });
    }
}

impl InequalityMapping {
    fn from_runtime_bounds(base_ccs: &CCS, bounds: &ConstraintBounds) -> Self {
        let count = base_ccs.nrow;
        let mut inequality_rows = Vec::new();
        let mut inequality_by_source = vec![Vec::<(Index, f64)>::new(); count];

        for (source_index, rows_for_source) in
            inequality_by_source.iter_mut().enumerate().take(count)
        {
            let lower_bound = bounds
                .lower
                .as_ref()
                .and_then(|values| values.get(source_index))
                .copied()
                .flatten();
            let upper_bound = bounds
                .upper
                .as_ref()
                .and_then(|values| values.get(source_index))
                .copied()
                .flatten();
            if lower_bound.is_none() && upper_bound.is_none() {
                continue;
            }
            if let Some(lower_bound) = lower_bound {
                let row = inequality_rows.len();
                inequality_rows.push(ConstraintTransform {
                    source_index,
                    sign: -1.0,
                    offset: lower_bound,
                });
                rows_for_source.push((row, -1.0));
            }
            if let Some(upper_bound) = upper_bound {
                let row = inequality_rows.len();
                inequality_rows.push(ConstraintTransform {
                    source_index,
                    sign: 1.0,
                    offset: -upper_bound,
                });
                rows_for_source.push((row, 1.0));
            }
        }

        let (inequality_jacobian_ccs, inequality_value_map) =
            remap_constraint_jacobian(base_ccs, &inequality_by_source);
        Self {
            rows: inequality_rows,
            inequality_jacobian_ccs,
            inequality_value_map,
        }
    }
}

impl JitKernel {
    fn compile_with_options(
        function: &SXFunction,
        options: FunctionCompileOptions,
    ) -> AnyResult<Self> {
        let compiled = CompiledJitFunction::compile_function_with_options(function, options)?;
        let context = Mutex::new(compiled.create_context());
        Ok(Self {
            function: compiled,
            context,
        })
    }

    fn eval_scalar_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
    ) -> (f64, KernelEvalTiming) {
        let mut context = lock_context(&self.context);
        load_jit_inputs(&self.function, &mut context, x, parameters, &[], &[]);
        let eval_started = Instant::now();
        self.function.eval(&mut context);
        let evaluation = eval_started.elapsed();
        let marshal_started = Instant::now();
        let value = context.output(0)[0];
        let output_marshalling = marshal_started.elapsed();
        (
            value,
            KernelEvalTiming {
                evaluation,
                output_marshalling,
            },
        )
    }

    fn eval_vector_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) -> KernelEvalTiming {
        let mut context = lock_context(&self.context);
        load_jit_inputs(&self.function, &mut context, x, parameters, &[], &[]);
        let eval_started = Instant::now();
        self.function.eval(&mut context);
        let evaluation = eval_started.elapsed();
        let marshal_started = Instant::now();
        out.copy_from_slice(context.output(0));
        let output_marshalling = marshal_started.elapsed();
        KernelEvalTiming {
            evaluation,
            output_marshalling,
        }
    }

    fn eval_hessian_timed(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    ) -> KernelEvalTiming {
        let mut context = lock_context(&self.context);
        load_jit_inputs(
            &self.function,
            &mut context,
            x,
            parameters,
            equality_multipliers,
            inequality_multipliers,
        );
        let eval_started = Instant::now();
        self.function.eval(&mut context);
        let evaluation = eval_started.elapsed();
        let marshal_started = Instant::now();
        out.copy_from_slice(context.output(0));
        let output_marshalling = marshal_started.elapsed();
        KernelEvalTiming {
            evaluation,
            output_marshalling,
        }
    }
}

fn symbolic_inputs(
    variables: &SXMatrix,
    parameters: &Option<NamedMatrix>,
) -> Result<Vec<NamedMatrix>, SymbolicNlpBuildError> {
    let mut inputs = Vec::with_capacity(usize::from(parameters.is_some()) + 1);
    inputs.push(NamedMatrix::new("x", variables.clone())?);
    inputs.extend(parameters.iter().cloned());
    Ok(inputs)
}

fn normalize_optional_matrix(matrix: Option<SXMatrix>) -> Option<SXMatrix> {
    match matrix {
        Some(matrix) if matrix.nnz() == 0 => None,
        other => other,
    }
}

fn function_output_ccs(function: &SXFunction) -> CCS {
    ccs_from_core(function.outputs()[0].matrix().ccs())
}

struct DerivedSymbolicFunctions {
    setup_profile: SymbolicSetupProfile,
    objective_value: SXFunction,
    objective_gradient: SXFunction,
    equality_values: Option<SXFunction>,
    equality_jacobian_values: Option<SXFunction>,
    inequality_values: Option<SXFunction>,
    inequality_jacobian_values: Option<SXFunction>,
    lagrangian_hessian_values: SXFunction,
}

fn symbolic_compile_stats(
    symbolic: &SymbolicNlp,
    objective_gradient_nnz: usize,
    equality_jacobian_nnz: usize,
    inequality_jacobian_nnz: usize,
    hessian_nnz: usize,
) -> NlpCompileStats {
    NlpCompileStats {
        variable_count: symbolic.variables.nnz(),
        parameter_scalar_count: symbolic
            .parameters
            .iter()
            .map(|matrix| matrix.matrix().nnz())
            .sum(),
        equality_count: symbolic.equalities.as_ref().map_or(0, SXMatrix::nnz),
        inequality_count: symbolic.inequalities.as_ref().map_or(0, SXMatrix::nnz),
        objective_gradient_nnz,
        equality_jacobian_nnz,
        inequality_jacobian_nnz,
        hessian_nnz,
        jit_kernel_count: 3
            + 2 * usize::from(symbolic.equalities.is_some())
            + 2 * usize::from(symbolic.inequalities.is_some()),
    }
}

fn derived_symbolic_compile_stats(
    symbolic: &SymbolicNlp,
    functions: &DerivedSymbolicFunctions,
) -> NlpCompileStats {
    symbolic_compile_stats(
        symbolic,
        function_output_ccs(&functions.objective_gradient).nnz(),
        functions
            .equality_jacobian_values
            .as_ref()
            .map_or(0, |function| function_output_ccs(function).nnz()),
        functions
            .inequality_jacobian_values
            .as_ref()
            .map_or(0, |function| function_output_ccs(function).nnz()),
        function_output_ccs(&functions.lagrangian_hessian_values).nnz(),
    )
}

fn derive_symbolic_functions(
    symbolic: &SymbolicNlp,
    hessian_strategy: HessianStrategy,
    on_symbolic_stage: &mut dyn FnMut(SymbolicCompileStage, &SymbolicSetupProfile, NlpCompileStats),
) -> Result<DerivedSymbolicFunctions, SymbolicNlpCompileError> {
    let base_inputs = symbolic_inputs(&symbolic.variables, &symbolic.parameters)?;
    let mut setup_profile = SymbolicSetupProfile {
        symbolic_construction: symbolic.construction_time,
        ..SymbolicSetupProfile::default()
    };
    let objective_value = SXFunction::new(
        format!("{}_objective", symbolic.name),
        base_inputs.clone(),
        vec![NamedMatrix::new("objective", symbolic.objective.clone())?],
    )?;
    let objective_gradient_started = Instant::now();
    let gradient = symbolic.objective.gradient(&symbolic.variables)?;
    setup_profile.objective_gradient = Some(objective_gradient_started.elapsed());
    let objective_gradient = SXFunction::new(
        format!("{}_gradient", symbolic.name),
        base_inputs.clone(),
        vec![NamedMatrix::new("gradient", gradient)?],
    )?;
    on_symbolic_stage(
        SymbolicCompileStage::ObjectiveGradient,
        &setup_profile,
        symbolic_compile_stats(
            symbolic,
            objective_gradient.outputs()[0].matrix().nnz(),
            0,
            0,
            0,
        ),
    );

    let equality_values = symbolic
        .equalities
        .as_ref()
        .map(|equalities| {
            SXFunction::new(
                format!("{}_equalities", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new("equalities", equalities.clone())?],
            )
        })
        .transpose()?;
    let equality_jacobian_values = symbolic
        .equalities
        .as_ref()
        .map(|equalities| {
            let started = Instant::now();
            let jacobian = equalities.jacobian(&symbolic.variables)?;
            setup_profile.equality_jacobian = Some(started.elapsed());
            SXFunction::new(
                format!("{}_equality_jacobian", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new("equality_jacobian", jacobian)?],
            )
        })
        .transpose()?;
    on_symbolic_stage(
        SymbolicCompileStage::EqualityJacobian,
        &setup_profile,
        symbolic_compile_stats(
            symbolic,
            objective_gradient.outputs()[0].matrix().nnz(),
            equality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            0,
            0,
        ),
    );
    let inequality_values = symbolic
        .inequalities
        .as_ref()
        .map(|inequalities| {
            SXFunction::new(
                format!("{}_inequalities", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new("inequalities", inequalities.clone())?],
            )
        })
        .transpose()?;
    let inequality_jacobian_values = symbolic
        .inequalities
        .as_ref()
        .map(|inequalities| {
            let started = Instant::now();
            let jacobian = inequalities.jacobian(&symbolic.variables)?;
            setup_profile.inequality_jacobian = Some(started.elapsed());
            SXFunction::new(
                format!("{}_inequality_jacobian", symbolic.name),
                base_inputs.clone(),
                vec![NamedMatrix::new("inequality_jacobian", jacobian)?],
            )
        })
        .transpose()?;
    on_symbolic_stage(
        SymbolicCompileStage::InequalityJacobian,
        &setup_profile,
        symbolic_compile_stats(
            symbolic,
            objective_gradient.outputs()[0].matrix().nnz(),
            equality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            inequality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            0,
        ),
    );

    let mut hessian_inputs = base_inputs.clone();
    let lagrangian_started = Instant::now();
    let mut lagrangian = symbolic.objective.scalar_expr()?;
    let equality_count = symbolic.equalities.as_ref().map_or(0, SXMatrix::nnz);
    if let Some(equalities) = &symbolic.equalities {
        let lambda = SXMatrix::sym("lambda_equalities", CoreCcs::column_vector(equality_count)?)?;
        for idx in 0..equalities.nnz() {
            lagrangian += lambda.nz(idx) * equalities.nz(idx);
        }
        hessian_inputs.push(NamedMatrix::new("lambda_equalities", lambda)?);
    }
    let inequality_count = symbolic.inequalities.as_ref().map_or(0, SXMatrix::nnz);
    if let Some(inequalities) = &symbolic.inequalities {
        let lambda = SXMatrix::sym(
            "lambda_inequalities",
            CoreCcs::column_vector(inequality_count)?,
        )?;
        for idx in 0..inequalities.nnz() {
            lagrangian += lambda.nz(idx) * inequalities.nz(idx);
        }
        hessian_inputs.push(NamedMatrix::new("lambda_inequalities", lambda)?);
    }
    setup_profile.lagrangian_assembly = Some(lagrangian_started.elapsed());
    on_symbolic_stage(
        SymbolicCompileStage::LagrangianAssembly,
        &setup_profile,
        symbolic_compile_stats(
            symbolic,
            objective_gradient.outputs()[0].matrix().nnz(),
            equality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            inequality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            0,
        ),
    );
    let hessian_started = Instant::now();
    let lagrangian_hessian = SXMatrix::scalar(lagrangian)
        .hessian_with_strategy(&symbolic.variables, hessian_strategy)?;
    setup_profile.hessian_generation = Some(hessian_started.elapsed());
    let lagrangian_hessian_values = SXFunction::new(
        format!("{}_lagrangian_hessian", symbolic.name),
        hessian_inputs,
        vec![NamedMatrix::new("lagrangian_hessian", lagrangian_hessian)?],
    )?;
    on_symbolic_stage(
        SymbolicCompileStage::HessianGeneration,
        &setup_profile,
        symbolic_compile_stats(
            symbolic,
            objective_gradient.outputs()[0].matrix().nnz(),
            equality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            inequality_jacobian_values
                .as_ref()
                .map_or(0, |function| function.outputs()[0].matrix().nnz()),
            lagrangian_hessian_values.outputs()[0].matrix().nnz(),
        ),
    );

    Ok(DerivedSymbolicFunctions {
        setup_profile,
        objective_value,
        objective_gradient,
        equality_values,
        equality_jacobian_values,
        inequality_values,
        inequality_jacobian_values,
        lagrangian_hessian_values,
    })
}

fn lock_context<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompiledNlpInputRole {
    DecisionVariables,
    EqualityMultipliers,
    InequalityMultipliers,
    Parameter,
}

fn compiled_nlp_input_role(slot_name: &str) -> CompiledNlpInputRole {
    match slot_name {
        "x" => CompiledNlpInputRole::DecisionVariables,
        "lambda_equalities" => CompiledNlpInputRole::EqualityMultipliers,
        "lambda_inequalities" => CompiledNlpInputRole::InequalityMultipliers,
        _ => CompiledNlpInputRole::Parameter,
    }
}

fn load_jit_inputs(
    function: &CompiledJitFunction,
    context: &mut JitExecutionContext,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
) {
    let mut parameter_index = 0;
    for (slot_index, slot) in function.lowered().inputs.iter().enumerate() {
        let input = context.input_mut(slot_index);
        match compiled_nlp_input_role(&slot.name) {
            CompiledNlpInputRole::DecisionVariables => input.copy_from_slice(x),
            CompiledNlpInputRole::EqualityMultipliers => {
                input.copy_from_slice(equality_multipliers)
            }
            CompiledNlpInputRole::InequalityMultipliers => {
                input.copy_from_slice(inequality_multipliers)
            }
            CompiledNlpInputRole::Parameter => {
                input.copy_from_slice(parameters[parameter_index].values);
                parameter_index += 1;
            }
        }
    }
    debug_assert_eq!(parameter_index, parameters.len());
}

fn ccs_from_core(ccs: &CoreCcs) -> CCS {
    CCS::new(
        ccs.nrow(),
        ccs.ncol(),
        ccs.col_ptrs().to_vec(),
        ccs.row_indices().to_vec(),
    )
}

fn validate_bound_vectors(
    expected: Index,
    bounds: ConstraintBounds,
    is_variable: bool,
) -> Result<ConstraintBounds, RuntimeNlpBoundsError> {
    let lower_len = bounds.lower.as_ref().map_or(expected, Vec::len);
    let upper_len = bounds.upper.as_ref().map_or(expected, Vec::len);
    if bounds
        .lower
        .as_ref()
        .is_some_and(|values| values.len() != expected)
        || bounds
            .upper
            .as_ref()
            .is_some_and(|values| values.len() != expected)
    {
        return Err(if is_variable {
            RuntimeNlpBoundsError::VariableBoundsLengthMismatch {
                expected,
                lower_len,
                upper_len,
            }
        } else {
            RuntimeNlpBoundsError::ConstraintBoundsLengthMismatch {
                expected,
                lower_len,
                upper_len,
            }
        });
    }

    if let (Some(lower), Some(upper)) = (&bounds.lower, &bounds.upper) {
        for (index, (&lower, &upper)) in lower.iter().zip(upper.iter()).enumerate() {
            if let (Some(lower), Some(upper)) = (lower, upper)
                && lower > upper
            {
                return Err(if is_variable {
                    RuntimeNlpBoundsError::InvalidVariableBounds {
                        index,
                        lower,
                        upper,
                    }
                } else {
                    RuntimeNlpBoundsError::InvalidConstraintBounds {
                        index,
                        lower,
                        upper,
                    }
                });
            }
        }
    }
    Ok(bounds)
}

fn remap_constraint_jacobian(
    base_ccs: &CCS,
    rows_by_source: &[Vec<(Index, f64)>],
) -> (CCS, Vec<JacobianValueMap>) {
    let mut col_ptrs = Vec::with_capacity(base_ccs.ncol + 1);
    let mut row_indices = Vec::new();
    let mut value_map = Vec::new();
    col_ptrs.push(0);
    for col in 0..base_ccs.ncol {
        for source_value_index in base_ccs.col_ptrs[col]..base_ccs.col_ptrs[col + 1] {
            let source_row = base_ccs.row_indices[source_value_index];
            for &(output_row, sign) in &rows_by_source[source_row] {
                row_indices.push(output_row);
                value_map.push(JacobianValueMap {
                    source_value_index,
                    sign,
                });
            }
        }
        col_ptrs.push(row_indices.len());
    }
    (
        CCS::new(
            rows_by_source.iter().flatten().count(),
            base_ccs.ncol,
            col_ptrs,
            row_indices,
        ),
        value_map,
    )
}
