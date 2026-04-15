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
    NlpInequalityViolation, ParameterMatrix, SqpAdapterTiming, SymbolicCompileMetadata,
    SymbolicCompileProgress, SymbolicCompileStage, SymbolicCompileStageProgress,
    SymbolicSetupProfile, Vectorize, benchmark_compiled_nlp_problem_with_progress,
    classify_constraint_satisfaction, constraint_bound_side, flatten_value,
    solve_nlp_interior_point, solve_nlp_interior_point_with_callback, solve_nlp_sqp,
    solve_nlp_sqp_with_callback, symbolic_column, symbolic_value,
    validate_compiled_nlp_problem_derivatives, worst_bound_violation,
};
#[cfg(feature = "ipopt")]
use crate::{
    IpoptIterationSnapshot, IpoptOptions, IpoptSolveError, IpoptSummary, solve_nlp_ipopt,
    solve_nlp_ipopt_with_callback,
};

#[derive(Clone, Debug, PartialEq)]
struct SymbolicNlp {
    name: String,
    variables: SXMatrix,
    parameters: Vec<NamedMatrix>,
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
    symbolic: SymbolicNlp,
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

#[derive(Debug)]
pub struct TypedCompiledJitNlp<X, P, E, I> {
    inner: CompiledJitNlp,
    _marker: TypedMarker<X, P, E, I>,
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
    pub variable_lower: Option<<X as Vectorize<SX>>::Rebind<f64>>,
    pub variable_upper: Option<<X as Vectorize<SX>>::Rebind<f64>>,
    pub inequality_lower: Option<<I as Vectorize<SX>>::Rebind<f64>>,
    pub inequality_upper: Option<<I as Vectorize<SX>>::Rebind<f64>>,
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
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintBounds {
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeNlpBounds {
    pub variables: ConstraintBounds,
    pub inequalities: ConstraintBounds,
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
        parameters: Vec<NamedMatrix>,
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
        debug_assert_eq!(objective_function.n_in(), parameters.len() + 1);
        let _ = objective.scalar_expr()?;

        let equalities = normalize_optional_matrix(equalities);
        if let Some(equalities) = &equalities {
            let validation = SXFunction::new(
                format!("{name}_equalities_validation"),
                symbolic_inputs(&variables, &parameters)?,
                vec![NamedMatrix::new("equalities", equalities.clone())?],
            )?;
            debug_assert_eq!(validation.n_in(), parameters.len() + 1);
        }
        let inequalities = normalize_optional_matrix(inequalities);
        if let Some(inequalities) = &inequalities {
            let validation = SXFunction::new(
                format!("{name}_inequalities_validation"),
                symbolic_inputs(&variables, &parameters)?,
                vec![NamedMatrix::new("inequalities", inequalities.clone())?],
            )?;
            debug_assert_eq!(validation.n_in(), parameters.len() + 1);
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
            function_creation_time: self.symbolic.construction_time,
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
        self.compile_jit_with_options_and_symbolic_progress_callback(options, |_| {})
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
        self.compile_jit_with_options_and_symbolic_progress_callback(options, |progress| {
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
        Ok(TypedCompiledJitNlp {
            inner: compile_symbolic_nlp_with_symbolic_progress_callback(
                &self.symbolic,
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

    let variable_matrix = symbolic_column(&variables)?;
    let parameter_matrices = if P::LEN == 0 {
        Vec::new()
    } else {
        vec![NamedMatrix::new("p", symbolic_column(&parameters)?)?]
    };
    let equalities = (E::LEN > 0)
        .then(|| symbolic_column(&outputs.equalities))
        .transpose()?;
    let inequalities = (I::LEN > 0)
        .then(|| symbolic_column(&outputs.inequalities))
        .transpose()?;
    let mut symbolic = SymbolicNlp::new(
        name,
        variable_matrix,
        parameter_matrices,
        SXMatrix::scalar(outputs.objective),
        equalities,
        inequalities,
    )?;
    symbolic.construction_time = Some(started_at.elapsed());
    Ok(TypedSymbolicNlp {
        symbolic,
        _marker: PhantomData,
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
        options: FunctionCompileOptions,
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
        let functions = derive_symbolic_functions(symbolic, &mut emit_symbolic_stage)?;
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
        let objective_value = JitKernel::compile_with_options(&functions.objective_value, options)?;
        let objective_gradient =
            JitKernel::compile_with_options(&functions.objective_gradient, options)?;
        let equality_values = functions
            .equality_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options))
            .transpose()?;
        let equality_jacobian_values = functions
            .equality_jacobian_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options))
            .transpose()?;
        let inequality_values = functions
            .inequality_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options))
            .transpose()?;
        let inequality_jacobian_values = functions
            .inequality_jacobian_values
            .as_ref()
            .map(|function| JitKernel::compile_with_options(function, options))
            .transpose()?;
        let lagrangian_hessian_values =
            JitKernel::compile_with_options(&functions.lagrangian_hessian_values, options)?;
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
        let _ = self
            .inner
            .inequality_values_timed(&x_values, &parameter_storage, &mut values);
        values
    }

    pub fn bind_runtime_bounds(
        &self,
        bounds: &TypedRuntimeNlpBounds<X, I>,
    ) -> Result<RuntimeBoundedJitNlp<'_>, RuntimeNlpBoundsError> {
        self.inner.bind_runtime_bounds(RuntimeNlpBounds {
            variables: ConstraintBounds {
                lower: bounds.variable_lower.as_ref().map(flatten_value),
                upper: bounds.variable_upper.as_ref().map(flatten_value),
            },
            inequalities: ConstraintBounds {
                lower: bounds.inequality_lower.as_ref().map(flatten_value),
                upper: bounds.inequality_upper.as_ref().map(flatten_value),
            },
        })
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
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
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
        Ok(benchmark_compiled_nlp_problem_with_progress(
            &bound_problem,
            &x_values,
            &parameter_storage,
            options,
            on_progress,
        ))
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
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_sqp(&bound_problem, &x0_values, &parameter_storage, options)
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
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_sqp_with_callback(
            &bound_problem,
            &x0_values,
            &parameter_storage,
            options,
            callback,
        )
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
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_interior_point(&bound_problem, &x0_values, &parameter_storage, options)
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
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_interior_point_with_callback(
            &bound_problem,
            &x0_values,
            &parameter_storage,
            options,
            callback,
        )
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
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_ipopt(&bound_problem, &x0_values, &parameter_storage, options)
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
        let x0_values = flatten_value(x0);
        let parameter_values = flatten_value(parameters);
        bound_problem.record_layout_projection(projection_started.elapsed());
        let parameter_storage = if P::LEN == 0 {
            Vec::new()
        } else {
            vec![ParameterMatrix {
                ccs: self.inner.parameter_ccs(0),
                values: &parameter_values,
            }]
        };
        solve_nlp_ipopt_with_callback(
            &bound_problem,
            &x0_values,
            &parameter_storage,
            options,
            callback,
        )
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
                    lower: bounds.variable_lower.as_ref().map(flatten_value),
                    upper: bounds.variable_upper.as_ref().map(flatten_value),
                },
                inequalities: ConstraintBounds {
                    lower: bounds.inequality_lower.as_ref().map(flatten_value),
                    upper: bounds.inequality_upper.as_ref().map(flatten_value),
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

    let mut report = NlpConstraintViolationReport::default();
    report.equalities = equalities
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
        .collect();
    report
        .equalities
        .sort_by(|lhs, rhs| rhs.abs_violation.total_cmp(&lhs.abs_violation));

    let inequality_lower = inequality_bounds.lower.unwrap_or_default();
    let inequality_upper = inequality_bounds.upper.unwrap_or_default();
    for (row, value) in inequalities.into_iter().enumerate() {
        let lower_bound = inequality_lower.get(row).copied();
        let upper_bound = inequality_upper.get(row).copied();
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
        let lower_bound = variable_lower.get(index).copied();
        let upper_bound = variable_upper.get(index).copied();
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
    options: FunctionCompileOptions,
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

    fn variable_bounds(&self, lower: &mut [f64], upper: &mut [f64]) -> bool {
        let started = Instant::now();
        lower.fill(-crate::NLP_INF);
        upper.fill(crate::NLP_INF);
        if let Some(bounds) = &self.variable_bounds.lower {
            lower.copy_from_slice(bounds);
        }
        if let Some(bounds) = &self.variable_bounds.upper {
            upper.copy_from_slice(bounds);
        }
        self.record_layout_projection(started.elapsed());
        true
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
        let lower = bounds.lower.as_deref().unwrap_or(&[]).to_vec();
        let upper = bounds.upper.as_deref().unwrap_or(&[]).to_vec();
        let count = base_ccs.nrow;
        let mut inequality_rows = Vec::new();
        let mut inequality_by_source = vec![Vec::<(Index, f64)>::new(); count];

        for (source_index, rows_for_source) in
            inequality_by_source.iter_mut().enumerate().take(count)
        {
            let lower_bound = lower.get(source_index).copied().unwrap_or(-crate::NLP_INF);
            let upper_bound = upper.get(source_index).copied().unwrap_or(crate::NLP_INF);
            if lower_bound == -crate::NLP_INF && upper_bound == crate::NLP_INF {
                continue;
            }
            if lower_bound > -crate::NLP_INF {
                let row = inequality_rows.len();
                inequality_rows.push(ConstraintTransform {
                    source_index,
                    sign: -1.0,
                    offset: lower_bound,
                });
                rows_for_source.push((row, -1.0));
            }
            if upper_bound < crate::NLP_INF {
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
    parameters: &[NamedMatrix],
) -> Result<Vec<NamedMatrix>, SymbolicNlpBuildError> {
    let mut inputs = Vec::with_capacity(parameters.len() + 1);
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
        .hessian_with_strategy(&symbolic.variables, HessianStrategy::LowerTriangleByColumn)?;
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
            if lower > upper {
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
