#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::extra_unused_type_parameters)]

pub mod runtime;

use anyhow::Result as AnyResult;
use optimization::{
    BackendCompileReport, BackendTimingMetadata, CallPolicy, ClarabelSqpError, ClarabelSqpOptions,
    ClarabelSqpSummary, ConstraintBoundSide, ConstraintSatisfaction,
    FiniteDifferenceValidationOptions, FunctionCompileOptions, InteriorPointIterationSnapshot,
    InteriorPointOptions, InteriorPointSolveError, InteriorPointSummary, LlvmOptimizationLevel,
    NlpCompileStats, NlpDerivativeValidationReport, NlpEvaluationBenchmark,
    NlpEvaluationBenchmarkOptions, NlpEvaluationKernelKind, ScalarLeaf, SqpIterationSnapshot,
    SymbolicCompileMetadata, SymbolicCompileProgress, SymbolicCompileStageProgress,
    SymbolicNlpBuildError, SymbolicNlpCompileError, SymbolicNlpCompileOptions, Vectorize,
    VectorizeLayoutError, classify_constraint_satisfaction, constraint_bound_side, flatten_value,
    symbolic_column, symbolic_value, unflatten_value, worst_bound_violation,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptIterationSnapshot, IpoptOptions, IpoptSolveError, IpoptSummary};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Mutex;
use std::time::Duration;
use sx_codegen_llvm::{CompiledJitFunction, JitExecutionContext};
use sx_core::{HessianStrategy, NamedMatrix, NodeView, SX, SXFunction, SXMatrix, SxError};
use thiserror::Error;

pub const MULTIPLE_SHOOTING_ARC_SAMPLES: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollocationFamily {
    GaussLegendre,
    RadauIIA,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bounds1D {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl ScalarLeaf for Bounds1D {}

#[derive(Clone, Debug, PartialEq)]
pub struct IntervalArc<T> {
    pub times: Vec<f64>,
    pub values: Vec<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterpolatedTrajectory<X, U, G = FinalTime<f64>> {
    pub sample_times: Vec<f64>,
    pub x_samples: Vec<X>,
    pub u_samples: Vec<U>,
    pub dudt_samples: Vec<U>,
    pub global: G,
    pub tf: f64,
}

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
pub struct FinalTime<T> {
    pub tf: T,
}

pub trait OcpGlobalDesign<T: optimization::ScalarLeaf>: Vectorize<T> + Clone {
    fn final_time(&self) -> T;
    fn from_final_time(tf: T) -> Self;
}

impl<T> OcpGlobalDesign<T> for FinalTime<T>
where
    T: Copy + optimization::ScalarLeaf,
{
    fn final_time(&self) -> T {
        self.tf
    }

    fn from_final_time(tf: T) -> Self {
        Self { tf }
    }
}

pub type ControllerFn<X, U, P> = dyn Fn(f64, &X, &U, &P) -> U + Send + Sync;

#[derive(Clone, Debug, PartialEq)]
pub struct OcpScaling<P, X, U, G = FinalTime<f64>> {
    /// User-facing OCP reference scales in the original problem units.
    ///
    /// The transcribed NLP is normalized internally with `q' = q / q_scale`.
    pub objective: f64,
    pub state: X,
    pub control: U,
    pub control_rate: U,
    pub global: G,
    pub parameters: P,
    pub path: Vec<f64>,
    pub boundary_equalities: Vec<f64>,
    pub boundary_inequalities: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcpSolveSetupTiming {
    pub initial_guess: Duration,
    pub runtime_bounds: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OcpConstraintCategory {
    BoundaryEquality,
    BoundaryInequality,
    Path,
    ContinuityState,
    ContinuityControl,
    CollocationState,
    CollocationControl,
    FinalTime,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcpEqualityViolationGroup {
    pub label: String,
    pub category: OcpConstraintCategory,
    pub worst_violation: f64,
    pub violating_instances: usize,
    pub total_instances: usize,
    pub satisfaction: ConstraintSatisfaction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcpInequalityViolationGroup {
    pub label: String,
    pub category: OcpConstraintCategory,
    pub worst_violation: f64,
    pub violating_instances: usize,
    pub total_instances: usize,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub lower_satisfaction: Option<ConstraintSatisfaction>,
    pub upper_satisfaction: Option<ConstraintSatisfaction>,
    pub bound_side: ConstraintBoundSide,
    pub satisfaction: ConstraintSatisfaction,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OcpConstraintViolationReport {
    pub equalities: Vec<OcpEqualityViolationGroup>,
    pub inequalities: Vec<OcpInequalityViolationGroup>,
}

type ObjectiveLagrangeFn<X, U, P, G> = dyn Fn(&X, &U, &U, &P, &G) -> SX + Send + Sync;
type ObjectiveMayerFn<X, U, P, G> = dyn Fn(&X, &U, &X, &U, &P, &G) -> SX + Send + Sync;
type OdeFn<X, U, P> = dyn Fn(&X, &U, &P) -> X + Send + Sync;
type PathConstraintsFn<X, U, P, C> = dyn Fn(&X, &U, &U, &P) -> C + Send + Sync;
type BoundaryFn<X, U, P, G, B> = dyn Fn(&X, &U, &X, &U, &P, &G) -> B + Send + Sync;

pub struct Ocp<X, U, P, C, Beq, Bineq, Scheme, G = FinalTime<SX>> {
    name: String,
    scheme: Scheme,
    objective_lagrange: Box<ObjectiveLagrangeFn<X, U, P, G>>,
    objective_mayer: Box<ObjectiveMayerFn<X, U, P, G>>,
    ode: Box<OdeFn<X, U, P>>,
    path_constraints: Box<PathConstraintsFn<X, U, P, C>>,
    boundary_equalities: Box<BoundaryFn<X, U, P, G, Beq>>,
    boundary_inequalities: Box<BoundaryFn<X, U, P, G, Bineq>>,
}

pub struct OcpBuilder<X, U, P, C, Beq, Bineq, Scheme, G = FinalTime<SX>> {
    name: String,
    scheme: Scheme,
    objective_lagrange: Option<Box<ObjectiveLagrangeFn<X, U, P, G>>>,
    objective_mayer: Option<Box<ObjectiveMayerFn<X, U, P, G>>>,
    ode: Option<Box<OdeFn<X, U, P>>>,
    path_constraints: Option<Box<PathConstraintsFn<X, U, P, C>>>,
    boundary_equalities: Option<Box<BoundaryFn<X, U, P, G, Beq>>>,
    boundary_inequalities: Option<Box<BoundaryFn<X, U, P, G, Bineq>>>,
}

#[derive(Debug, Error)]
pub enum OcpBuildError {
    #[error("OCP name cannot be empty")]
    EmptyName,
    #[error("missing OCP callback `{0}`")]
    MissingCallback(&'static str),
}

#[derive(Debug, Error)]
pub enum OcpCompileError {
    #[error(transparent)]
    Build(#[from] OcpBuildError),
    #[error(transparent)]
    SymbolicBuild(#[from] SymbolicNlpBuildError),
    #[error(transparent)]
    SymbolicCompile(#[from] SymbolicNlpCompileError),
    #[error(transparent)]
    Graph(#[from] SxError),
    #[error(transparent)]
    Jit(#[from] anyhow::Error),
    #[error("{0}")]
    InvalidConfiguration(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OcpKernelMode {
    Inline,
    #[default]
    Function,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcpKernelFunctionOptions {
    pub mode: OcpKernelMode,
    pub call_policy_override: Option<CallPolicy>,
}

impl OcpKernelFunctionOptions {
    pub const fn inline() -> Self {
        Self {
            mode: OcpKernelMode::Inline,
            call_policy_override: None,
        }
    }

    pub const fn function() -> Self {
        Self {
            mode: OcpKernelMode::Function,
            call_policy_override: None,
        }
    }

    pub const fn function_with_call_policy(policy: CallPolicy) -> Self {
        Self {
            mode: OcpKernelMode::Function,
            call_policy_override: Some(policy),
        }
    }

    pub const fn with_call_policy_override(self, policy: CallPolicy) -> Self {
        Self {
            call_policy_override: Some(policy),
            ..self
        }
    }
}

impl Default for OcpKernelFunctionOptions {
    fn default() -> Self {
        Self::function()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcpSymbolicFunctionOptions {
    pub ode: OcpKernelFunctionOptions,
    pub objective_lagrange: OcpKernelFunctionOptions,
    pub objective_mayer: OcpKernelFunctionOptions,
    pub path_constraints: OcpKernelFunctionOptions,
    pub boundary_equalities: OcpKernelFunctionOptions,
    pub boundary_inequalities: OcpKernelFunctionOptions,
    pub multiple_shooting_integrator: OcpKernelFunctionOptions,
}

impl OcpSymbolicFunctionOptions {
    pub const fn multiple_shooting_default() -> Self {
        Self {
            ode: OcpKernelFunctionOptions::function_with_call_policy(CallPolicy::InlineInLLVM),
            objective_lagrange: OcpKernelFunctionOptions::function_with_call_policy(
                CallPolicy::InlineInLLVM,
            ),
            objective_mayer: OcpKernelFunctionOptions::inline(),
            path_constraints: OcpKernelFunctionOptions::function_with_call_policy(
                CallPolicy::InlineInLLVM,
            ),
            boundary_equalities: OcpKernelFunctionOptions::inline(),
            boundary_inequalities: OcpKernelFunctionOptions::inline(),
            multiple_shooting_integrator: OcpKernelFunctionOptions::inline(),
        }
    }

    pub const fn direct_collocation_default() -> Self {
        Self::inline_all()
    }

    pub const fn function_all_with_call_policy(policy: CallPolicy) -> Self {
        let function = OcpKernelFunctionOptions::function_with_call_policy(policy);
        Self {
            ode: function,
            objective_lagrange: function,
            objective_mayer: function,
            path_constraints: function,
            boundary_equalities: function,
            boundary_inequalities: function,
            multiple_shooting_integrator: function,
        }
    }

    pub const fn inline_all() -> Self {
        let inline = OcpKernelFunctionOptions::inline();
        Self {
            ode: inline,
            objective_lagrange: inline,
            objective_mayer: inline,
            path_constraints: inline,
            boundary_equalities: inline,
            boundary_inequalities: inline,
            multiple_shooting_integrator: inline,
        }
    }
}

impl Default for OcpSymbolicFunctionOptions {
    fn default() -> Self {
        Self::multiple_shooting_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcpCompileOptions {
    pub function_options: FunctionCompileOptions,
    pub symbolic_functions: OcpSymbolicFunctionOptions,
    pub hessian_strategy: HessianStrategy,
}

impl Default for OcpCompileOptions {
    fn default() -> Self {
        Self::from(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }
}

impl OcpCompileOptions {
    pub fn for_multiple_shooting(function_options: FunctionCompileOptions) -> Self {
        Self {
            function_options,
            symbolic_functions: OcpSymbolicFunctionOptions::multiple_shooting_default(),
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
        }
    }

    pub fn for_direct_collocation(function_options: FunctionCompileOptions) -> Self {
        Self {
            function_options,
            symbolic_functions: OcpSymbolicFunctionOptions::direct_collocation_default(),
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
        }
    }
}

impl From<FunctionCompileOptions> for OcpCompileOptions {
    fn from(function_options: FunctionCompileOptions) -> Self {
        Self {
            function_options,
            symbolic_functions: OcpSymbolicFunctionOptions::default(),
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
        }
    }
}

impl From<LlvmOptimizationLevel> for OcpCompileOptions {
    fn from(opt_level: LlvmOptimizationLevel) -> Self {
        Self::from(FunctionCompileOptions::from(opt_level))
    }
}

#[derive(Debug, Error)]
enum GuessError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Graph(#[from] SxError),
    #[error(transparent)]
    Layout(#[from] optimization::VectorizeLayoutError),
    #[error(transparent)]
    Jit(#[from] anyhow::Error),
}

#[derive(Clone, Debug)]
struct EqualityGroupAccumulator {
    label: String,
    category: OcpConstraintCategory,
    worst_violation: f64,
    violating_instances: usize,
    total_instances: usize,
}

#[derive(Clone, Debug)]
struct InequalityGroupAccumulator {
    label: String,
    category: OcpConstraintCategory,
    worst_violation: f64,
    violating_instances: usize,
    total_instances: usize,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    lower_worst_violation: f64,
    upper_worst_violation: f64,
    lower_violated: bool,
    upper_violated: bool,
}

type OcpParameters<P, Beq> = (P, Beq);
type Numeric<S> = <S as Vectorize<SX>>::Rebind<f64>;
type BoundTemplate<S> = <S as Vectorize<SX>>::Rebind<Bounds1D>;
type OcpParametersNum<P, Beq> = (Numeric<P>, Numeric<Beq>);
struct PromotionPlan {
    rows: Vec<RawInequalityRow>,
}

#[derive(Clone, Debug)]
struct RawInequalityRow {
    kind: RawInequalityKind,
    promotion: Option<AffinePromotion>,
}

#[derive(Clone, Debug)]
enum RawInequalityKind {
    BoundaryEquality,
    BoundaryInequality,
    Path,
}

#[derive(Clone, Debug)]
struct AffinePromotion {
    variable_index: usize,
    scale: f64,
    offset_index: usize,
    offset_expr: SX,
}

#[derive(Debug)]
struct PromotionOffsets<P> {
    function: Option<CompiledScalarVector<P>>,
}

#[derive(Clone, Debug)]
struct CollocationCoefficients {
    nodes: Vec<f64>,
    c_matrix: Vec<Vec<f64>>,
    d_vector: Vec<f64>,
    b_vector: Vec<f64>,
}

#[derive(Debug)]
struct CompiledScalarVector<P> {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
    _marker: PhantomData<fn() -> P>,
}

#[derive(Debug)]
struct CompiledXdot<X, U, P> {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
    _marker: PhantomData<fn() -> (X, U, P)>,
}

#[derive(Debug)]
struct OcpSymbolicFunctionLibrary {
    ode: Option<SXFunction>,
    objective_lagrange: Option<SXFunction>,
    objective_mayer: Option<SXFunction>,
    path_constraints: Option<SXFunction>,
    boundary_equalities: Option<SXFunction>,
    boundary_inequalities: Option<SXFunction>,
    multiple_shooting_integrator: Option<SXFunction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcpCompileHelperKind {
    Xdot,
    MultipleShootingArc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OcpCompileProgress {
    SymbolicStage(SymbolicCompileStageProgress),
    SymbolicReady(SymbolicCompileMetadata),
    NlpKernelCompiled {
        elapsed: Duration,
        root_instructions: usize,
        total_instructions: usize,
    },
    HelperCompiled {
        helper: OcpCompileHelperKind,
        elapsed: Duration,
        root_instructions: usize,
        total_instructions: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcpHelperCompileStats {
    pub xdot_helper_time: Option<Duration>,
    pub multiple_shooting_arc_helper_time: Option<Duration>,
    pub xdot_helper_root_instructions: Option<usize>,
    pub xdot_helper_total_instructions: Option<usize>,
    pub multiple_shooting_arc_helper_root_instructions: Option<usize>,
    pub multiple_shooting_arc_helper_total_instructions: Option<usize>,
    pub llvm_cache_hits: usize,
    pub llvm_cache_misses: usize,
    pub llvm_cache_check_time: Duration,
    pub llvm_cache_read_time: Duration,
    pub llvm_cache_load_time: Duration,
    pub llvm_cache_materialize_time: Duration,
}

impl OcpHelperCompileStats {
    fn record_compile_report(&mut self, report: &sx_codegen_llvm::FunctionCompileReport) {
        self.llvm_cache_check_time += report.cache.check_time;
        self.llvm_cache_read_time += report.cache.read_time;
        self.llvm_cache_materialize_time += report.cache.materialize_time;
        if report.cache.hit {
            self.llvm_cache_hits += 1;
            self.llvm_cache_load_time += report.cache.load_time;
        } else {
            self.llvm_cache_misses += 1;
        }
    }
}

impl<X, U, P, C, Beq, Bineq, Scheme, G> Ocp<X, U, P, C, Beq, Bineq, Scheme, G> {
    pub fn new(
        name: impl Into<String>,
        scheme: Scheme,
    ) -> OcpBuilder<X, U, P, C, Beq, Bineq, Scheme, G> {
        OcpBuilder {
            name: name.into(),
            scheme,
            objective_lagrange: None,
            objective_mayer: None,
            ode: None,
            path_constraints: None,
            boundary_equalities: None,
            boundary_inequalities: None,
        }
    }
}

impl<X, U, P, C, Beq, Bineq, Scheme, G> OcpBuilder<X, U, P, C, Beq, Bineq, Scheme, G> {
    pub fn objective_lagrange<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &U, &P) -> SX + Send + Sync + 'static,
    {
        self.objective_lagrange = Some(Box::new(move |x, u, dudt, p, _global| f(x, u, dudt, p)));
        self
    }

    pub fn objective_lagrange_global<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &U, &P, &G) -> SX + Send + Sync + 'static,
    {
        self.objective_lagrange = Some(Box::new(f));
        self
    }

    pub fn objective_mayer<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &SX) -> SX + Send + Sync + 'static,
        G: OcpGlobalDesign<SX>,
    {
        self.objective_mayer = Some(Box::new(move |x0, u0, xf, uf, p, global| {
            let tf = global.final_time();
            f(x0, u0, xf, uf, p, &tf)
        }));
        self
    }

    pub fn objective_mayer_global<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &G) -> SX + Send + Sync + 'static,
    {
        self.objective_mayer = Some(Box::new(f));
        self
    }

    pub fn ode<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &P) -> X + Send + Sync + 'static,
    {
        self.ode = Some(Box::new(f));
        self
    }

    pub fn path_constraints<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &U, &P) -> C + Send + Sync + 'static,
    {
        self.path_constraints = Some(Box::new(f));
        self
    }

    pub fn boundary_equalities<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &SX) -> Beq + Send + Sync + 'static,
        G: OcpGlobalDesign<SX>,
    {
        self.boundary_equalities = Some(Box::new(move |x0, u0, xf, uf, p, global| {
            let tf = global.final_time();
            f(x0, u0, xf, uf, p, &tf)
        }));
        self
    }

    pub fn boundary_equalities_global<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &G) -> Beq + Send + Sync + 'static,
    {
        self.boundary_equalities = Some(Box::new(f));
        self
    }

    pub fn boundary_inequalities<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &SX) -> Bineq + Send + Sync + 'static,
        G: OcpGlobalDesign<SX>,
    {
        self.boundary_inequalities = Some(Box::new(move |x0, u0, xf, uf, p, global| {
            let tf = global.final_time();
            f(x0, u0, xf, uf, p, &tf)
        }));
        self
    }

    pub fn boundary_inequalities_global<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &G) -> Bineq + Send + Sync + 'static,
    {
        self.boundary_inequalities = Some(Box::new(f));
        self
    }

    pub fn build(self) -> Result<Ocp<X, U, P, C, Beq, Bineq, Scheme, G>, OcpBuildError> {
        if self.name.trim().is_empty() {
            return Err(OcpBuildError::EmptyName);
        }
        Ok(Ocp {
            name: self.name,
            scheme: self.scheme,
            objective_lagrange: self
                .objective_lagrange
                .ok_or(OcpBuildError::MissingCallback("objective_lagrange"))?,
            objective_mayer: self
                .objective_mayer
                .ok_or(OcpBuildError::MissingCallback("objective_mayer"))?,
            ode: self.ode.ok_or(OcpBuildError::MissingCallback("ode"))?,
            path_constraints: self
                .path_constraints
                .ok_or(OcpBuildError::MissingCallback("path_constraints"))?,
            boundary_equalities: self
                .boundary_equalities
                .ok_or(OcpBuildError::MissingCallback("boundary_equalities"))?,
            boundary_inequalities: self
                .boundary_inequalities
                .ok_or(OcpBuildError::MissingCallback("boundary_inequalities"))?,
        })
    }
}

impl<X, U, P, C, Beq, Bineq, Scheme, G> Ocp<X, U, P, C, Beq, Bineq, Scheme, G>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    G: OcpGlobalDesign<SX> + Vectorize<SX, Rebind<SX> = G>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
{
    fn build_symbolic_function_library(
        &self,
        options: OcpSymbolicFunctionOptions,
    ) -> Result<OcpSymbolicFunctionLibrary, SxError> {
        Ok(OcpSymbolicFunctionLibrary {
            ode: self.build_ode_symbolic_function(options.ode)?,
            objective_lagrange: self
                .build_objective_lagrange_symbolic_function(options.objective_lagrange)?,
            objective_mayer: self
                .build_objective_mayer_symbolic_function(options.objective_mayer)?,
            path_constraints: self
                .build_path_constraints_symbolic_function(options.path_constraints)?,
            boundary_equalities: self
                .build_boundary_equalities_symbolic_function(options.boundary_equalities)?,
            boundary_inequalities: self
                .build_boundary_inequalities_symbolic_function(options.boundary_inequalities)?,
            multiple_shooting_integrator: None,
        })
    }

    fn configured_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
        build: impl FnOnce() -> Result<SXFunction, SxError>,
    ) -> Result<Option<SXFunction>, SxError> {
        match options.mode {
            OcpKernelMode::Inline => Ok(None),
            OcpKernelMode::Function => {
                let function = build()?;
                Ok(Some(match options.call_policy_override {
                    Some(policy) => function.with_call_policy_override(policy),
                    None => function,
                }))
            }
        }
    }

    fn build_ode_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x = symbolic_value::<X>("x")?;
            let u = symbolic_value::<U>("u")?;
            let p = symbolic_value::<P>("p")?;
            let xdot = (self.ode)(&x, &u, &p);
            SXFunction::new(
                format!("{}_ode", self.name),
                vec![
                    NamedMatrix::new("x", symbolic_column(&x)?)?,
                    NamedMatrix::new("u", symbolic_column(&u)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                ],
                vec![NamedMatrix::new("xdot", symbolic_column(&xdot)?)?],
            )
        })
    }

    fn build_objective_lagrange_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x = symbolic_value::<X>("x")?;
            let u = symbolic_value::<U>("u")?;
            let dudt = symbolic_value::<U>("dudt")?;
            let p = symbolic_value::<P>("p")?;
            let g = symbolic_value::<G>("g")?;
            let objective = (self.objective_lagrange)(&x, &u, &dudt, &p, &g);
            SXFunction::new(
                format!("{}_objective_lagrange", self.name),
                vec![
                    NamedMatrix::new("x", symbolic_column(&x)?)?,
                    NamedMatrix::new("u", symbolic_column(&u)?)?,
                    NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("g", symbolic_column(&g)?)?,
                ],
                vec![NamedMatrix::new("objective", SXMatrix::scalar(objective))?],
            )
        })
    }

    fn build_objective_mayer_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x0 = symbolic_value::<X>("x0")?;
            let u0 = symbolic_value::<U>("u0")?;
            let xf = symbolic_value::<X>("xf")?;
            let uf = symbolic_value::<U>("uf")?;
            let p = symbolic_value::<P>("p")?;
            let g = symbolic_value::<G>("g")?;
            let objective = (self.objective_mayer)(&x0, &u0, &xf, &uf, &p, &g);
            SXFunction::new(
                format!("{}_objective_mayer", self.name),
                vec![
                    NamedMatrix::new("x0", symbolic_column(&x0)?)?,
                    NamedMatrix::new("u0", symbolic_column(&u0)?)?,
                    NamedMatrix::new("xf", symbolic_column(&xf)?)?,
                    NamedMatrix::new("uf", symbolic_column(&uf)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("g", symbolic_column(&g)?)?,
                ],
                vec![NamedMatrix::new("objective", SXMatrix::scalar(objective))?],
            )
        })
    }

    fn build_path_constraints_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x = symbolic_value::<X>("x")?;
            let u = symbolic_value::<U>("u")?;
            let dudt = symbolic_value::<U>("dudt")?;
            let p = symbolic_value::<P>("p")?;
            let path = (self.path_constraints)(&x, &u, &dudt, &p);
            SXFunction::new(
                format!("{}_path_constraints", self.name),
                vec![
                    NamedMatrix::new("x", symbolic_column(&x)?)?,
                    NamedMatrix::new("u", symbolic_column(&u)?)?,
                    NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                ],
                vec![NamedMatrix::new("path", symbolic_column(&path)?)?],
            )
        })
    }

    fn build_boundary_equalities_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x0 = symbolic_value::<X>("x0")?;
            let u0 = symbolic_value::<U>("u0")?;
            let xf = symbolic_value::<X>("xf")?;
            let uf = symbolic_value::<U>("uf")?;
            let p = symbolic_value::<P>("p")?;
            let g = symbolic_value::<G>("g")?;
            let values = (self.boundary_equalities)(&x0, &u0, &xf, &uf, &p, &g);
            SXFunction::new(
                format!("{}_boundary_equalities", self.name),
                vec![
                    NamedMatrix::new("x0", symbolic_column(&x0)?)?,
                    NamedMatrix::new("u0", symbolic_column(&u0)?)?,
                    NamedMatrix::new("xf", symbolic_column(&xf)?)?,
                    NamedMatrix::new("uf", symbolic_column(&uf)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("g", symbolic_column(&g)?)?,
                ],
                vec![NamedMatrix::new("boundary_eq", symbolic_column(&values)?)?],
            )
        })
    }

    fn build_boundary_inequalities_symbolic_function(
        &self,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x0 = symbolic_value::<X>("x0")?;
            let u0 = symbolic_value::<U>("u0")?;
            let xf = symbolic_value::<X>("xf")?;
            let uf = symbolic_value::<U>("uf")?;
            let p = symbolic_value::<P>("p")?;
            let g = symbolic_value::<G>("g")?;
            let values = (self.boundary_inequalities)(&x0, &u0, &xf, &uf, &p, &g);
            SXFunction::new(
                format!("{}_boundary_inequalities", self.name),
                vec![
                    NamedMatrix::new("x0", symbolic_column(&x0)?)?,
                    NamedMatrix::new("u0", symbolic_column(&u0)?)?,
                    NamedMatrix::new("xf", symbolic_column(&xf)?)?,
                    NamedMatrix::new("uf", symbolic_column(&uf)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("g", symbolic_column(&g)?)?,
                ],
                vec![NamedMatrix::new(
                    "boundary_ineq",
                    symbolic_column(&values)?,
                )?],
            )
        })
    }

    fn eval_ode_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x: &X,
        u: &U,
        parameters: &P,
    ) -> Result<X, SxError> {
        match &library.ode {
            Some(function) => call_typed_unary_output::<X>(
                function,
                vec![
                    symbolic_column(x)?,
                    symbolic_column(u)?,
                    symbolic_column(parameters)?,
                ],
            ),
            None => Ok((self.ode)(x, u, parameters)),
        }
    }

    fn eval_objective_lagrange_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x: &X,
        u: &U,
        dudt: &U,
        parameters: &P,
        global: &G,
    ) -> Result<SX, SxError> {
        match &library.objective_lagrange {
            Some(function) => function.call_scalar(&[
                symbolic_column(x)?,
                symbolic_column(u)?,
                symbolic_column(dudt)?,
                symbolic_column(parameters)?,
                symbolic_column(global)?,
            ]),
            None => Ok((self.objective_lagrange)(x, u, dudt, parameters, global)),
        }
    }

    fn eval_objective_mayer_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x0: &X,
        u0: &U,
        xf: &X,
        uf: &U,
        parameters: &P,
        global: &G,
    ) -> Result<SX, SxError> {
        match &library.objective_mayer {
            Some(function) => function.call_scalar(&[
                symbolic_column(x0)?,
                symbolic_column(u0)?,
                symbolic_column(xf)?,
                symbolic_column(uf)?,
                symbolic_column(parameters)?,
                symbolic_column(global)?,
            ]),
            None => Ok((self.objective_mayer)(x0, u0, xf, uf, parameters, global)),
        }
    }

    fn eval_path_constraints_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x: &X,
        u: &U,
        dudt: &U,
        parameters: &P,
    ) -> Result<C, SxError> {
        match &library.path_constraints {
            Some(function) => call_typed_unary_output::<C>(
                function,
                vec![
                    symbolic_column(x)?,
                    symbolic_column(u)?,
                    symbolic_column(dudt)?,
                    symbolic_column(parameters)?,
                ],
            ),
            None => Ok((self.path_constraints)(x, u, dudt, parameters)),
        }
    }

    fn eval_boundary_equalities_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x0: &X,
        u0: &U,
        xf: &X,
        uf: &U,
        parameters: &P,
        global: &G,
    ) -> Result<Beq, SxError> {
        match &library.boundary_equalities {
            Some(function) => call_typed_unary_output::<Beq>(
                function,
                vec![
                    symbolic_column(x0)?,
                    symbolic_column(u0)?,
                    symbolic_column(xf)?,
                    symbolic_column(uf)?,
                    symbolic_column(parameters)?,
                    symbolic_column(global)?,
                ],
            ),
            None => Ok((self.boundary_equalities)(
                x0, u0, xf, uf, parameters, global,
            )),
        }
    }

    fn eval_boundary_inequalities_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x0: &X,
        u0: &U,
        xf: &X,
        uf: &U,
        parameters: &P,
        global: &G,
    ) -> Result<Bineq, SxError> {
        match &library.boundary_inequalities {
            Some(function) => call_typed_unary_output::<Bineq>(
                function,
                vec![
                    symbolic_column(x0)?,
                    symbolic_column(u0)?,
                    symbolic_column(xf)?,
                    symbolic_column(uf)?,
                    symbolic_column(parameters)?,
                    symbolic_column(global)?,
                ],
            ),
            None => Ok((self.boundary_inequalities)(
                x0, u0, xf, uf, parameters, global,
            )),
        }
    }
}

impl<P> PromotionOffsets<P>
where
    P: Vectorize<SX>,
    <P as Vectorize<SX>>::Rebind<f64>:
        Vectorize<f64, Rebind<f64> = <P as Vectorize<SX>>::Rebind<f64>>,
{
    fn eval(&self, parameters: &<P as Vectorize<SX>>::Rebind<f64>) -> Result<Vec<f64>, GuessError> {
        match &self.function {
            Some(function) => function.eval(parameters).map_err(GuessError::Jit),
            None => Ok(Vec::new()),
        }
    }
}

impl<P> CompiledScalarVector<P>
where
    P: Vectorize<SX>,
    <P as Vectorize<SX>>::Rebind<f64>:
        Vectorize<f64, Rebind<f64> = <P as Vectorize<SX>>::Rebind<f64>>,
{
    fn eval(&self, parameters: &Numeric<P>) -> AnyResult<Vec<f64>> {
        let flat_parameters = flatten_value(parameters);
        let mut context = lock_mutex(&self.context);
        if P::LEN > 0 {
            context.input_mut(0).copy_from_slice(&flat_parameters);
        }
        self.function.eval(&mut context);
        Ok(context.output(0).to_vec())
    }
}

impl<X, U, P> CompiledXdot<X, U, P>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    fn eval(
        &self,
        x: &Numeric<X>,
        u: &Numeric<U>,
        parameters: &Numeric<P>,
    ) -> AnyResult<Numeric<X>> {
        let flat_x = flatten_value(x);
        let flat_u = flatten_value(u);
        let flat_parameters = flatten_value(parameters);
        let mut context = lock_mutex(&self.context);
        context.input_mut(0).copy_from_slice(&flat_x);
        context.input_mut(1).copy_from_slice(&flat_u);
        if P::LEN > 0 {
            context.input_mut(2).copy_from_slice(&flat_parameters);
        }
        self.function.eval(&mut context);
        unflatten_value::<Numeric<X>, f64>(context.output(0)).map_err(Into::into)
    }
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

fn compile_xdot_helper<X, U, P>(
    ode: &OdeFn<X, U, P>,
    _symbolic_ode: Option<&SXFunction>,
    options: FunctionCompileOptions,
) -> Result<CompiledXdot<X, U, P>, OcpCompileError>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    let x = symbolic_value::<X>("x")?;
    let u = symbolic_value::<U>("u")?;
    let p = symbolic_value::<P>("p")?;
    let xdot = ode(&x, &u, &p);
    let mut inputs = vec![
        NamedMatrix::new("x", symbolic_column(&x)?)?,
        NamedMatrix::new("u", symbolic_column(&u)?)?,
    ];
    if P::LEN > 0 {
        inputs.push(NamedMatrix::new("p", symbolic_column(&p)?)?);
    }
    let function = SXFunction::new(
        "ocp_xdot",
        inputs,
        vec![NamedMatrix::new("xdot", symbolic_column(&xdot)?)?],
    )?;
    let compiled = CompiledJitFunction::compile_function_with_options(&function, options)?;
    let context = Mutex::new(compiled.create_context());
    Ok(CompiledXdot {
        function: compiled,
        context,
        _marker: PhantomData,
    })
}

fn compile_promotion_offsets<P>(
    plan: &PromotionPlan,
    parameters: &P,
    options: FunctionCompileOptions,
) -> Result<PromotionOffsets<P>, OcpCompileError>
where
    P: Vectorize<SX, Rebind<SX> = P>,
    <P as Vectorize<SX>>::Rebind<f64>:
        Vectorize<f64, Rebind<f64> = <P as Vectorize<SX>>::Rebind<f64>>,
{
    if plan.rows.iter().all(|row| row.promotion.is_none()) {
        return Ok(PromotionOffsets { function: None });
    }

    let mut offset_values = vec![SX::zero(); plan.rows.len()];
    for row in &plan.rows {
        if let Some(promotion) = &row.promotion {
            offset_values[promotion.offset_index] = promotion.offset_expr;
        }
    }
    let inputs = if P::LEN == 0 {
        Vec::new()
    } else {
        vec![NamedMatrix::new("runtime", symbolic_column(parameters)?)?]
    };
    let function = SXFunction::new(
        "ocp_promoted_offsets",
        inputs,
        vec![NamedMatrix::new(
            "offsets",
            SXMatrix::dense_column(offset_values)?,
        )?],
    )?;
    let compiled = CompiledJitFunction::compile_function_with_options(&function, options)?;
    let context = Mutex::new(compiled.create_context());
    Ok(PromotionOffsets {
        function: Some(CompiledScalarVector {
            function: compiled,
            context,
            _marker: PhantomData,
        }),
    })
}

#[derive(Clone)]
struct AffineForm {
    target: Option<SX>,
    scale: f64,
    offset: SX,
}

fn classify_affine_row(
    expr: SX,
    decision_map: &HashMap<SX, usize>,
    decision_set: &HashSet<SX>,
    memo: &mut HashMap<SX, Option<AffineForm>>,
    offset_index: usize,
) -> Option<AffinePromotion> {
    let affine = affine_form(expr, decision_set, memo)?;
    let target = affine.target?;
    let scale = affine.scale;
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    Some(AffinePromotion {
        variable_index: *decision_map
            .get(&target)
            .expect("target symbol should exist in decision map"),
        scale,
        offset_index,
        offset_expr: affine.offset,
    })
}

fn affine_form(
    expr: SX,
    decision_set: &HashSet<SX>,
    memo: &mut HashMap<SX, Option<AffineForm>>,
) -> Option<AffineForm> {
    if let Some(existing) = memo.get(&expr) {
        return existing.clone();
    }

    let affine = match expr.inspect() {
        NodeView::Constant(value) => Some(AffineForm {
            target: None,
            scale: 0.0,
            offset: SX::from(value),
        }),
        NodeView::Symbol { .. } => Some(if decision_set.contains(&expr) {
            AffineForm {
                target: Some(expr),
                scale: 1.0,
                offset: SX::zero(),
            }
        } else {
            AffineForm {
                target: None,
                scale: 0.0,
                offset: expr,
            }
        }),
        NodeView::Unary { .. } => None,
        NodeView::Call { .. } => None,
        NodeView::Binary { op, lhs, rhs } => match op {
            sx_core::BinaryOp::Add => {
                let lhs = affine_form(lhs, decision_set, memo)?;
                let rhs = affine_form(rhs, decision_set, memo)?;
                let target = combine_affine_targets(lhs.target, rhs.target)?;
                Some(AffineForm {
                    target,
                    scale: lhs.scale + rhs.scale,
                    offset: lhs.offset + rhs.offset,
                })
            }
            sx_core::BinaryOp::Sub => {
                let lhs = affine_form(lhs, decision_set, memo)?;
                let rhs = affine_form(rhs, decision_set, memo)?;
                let target = combine_affine_targets(lhs.target, rhs.target)?;
                Some(AffineForm {
                    target,
                    scale: lhs.scale - rhs.scale,
                    offset: lhs.offset - rhs.offset,
                })
            }
            sx_core::BinaryOp::Mul => {
                if let NodeView::Constant(value) = lhs.inspect() {
                    let rhs = affine_form(rhs, decision_set, memo)?;
                    Some(AffineForm {
                        target: rhs.target,
                        scale: value * rhs.scale,
                        offset: SX::from(value) * rhs.offset,
                    })
                } else if let NodeView::Constant(value) = rhs.inspect() {
                    let lhs = affine_form(lhs, decision_set, memo)?;
                    Some(AffineForm {
                        target: lhs.target,
                        scale: value * lhs.scale,
                        offset: SX::from(value) * lhs.offset,
                    })
                } else {
                    None
                }
            }
            sx_core::BinaryOp::Div => {
                if let NodeView::Constant(value) = rhs.inspect() {
                    let lhs = affine_form(lhs, decision_set, memo)?;
                    Some(AffineForm {
                        target: lhs.target,
                        scale: lhs.scale / value,
                        offset: lhs.offset / value,
                    })
                } else {
                    None
                }
            }
            _ => None,
        },
    };

    memo.insert(expr, affine.clone());
    affine
}

fn combine_affine_targets(lhs: Option<SX>, rhs: Option<SX>) -> Option<Option<SX>> {
    match (lhs, rhs) {
        (None, None) => Some(None),
        (Some(target), None) | (None, Some(target)) => Some(Some(target)),
        (Some(lhs), Some(rhs)) if lhs == rhs => Some(Some(lhs)),
        (Some(_), Some(_)) => None,
    }
}

fn build_raw_bounds<C, Beq, Bineq, G>(
    plan: &PromotionPlan,
    offsets: &[f64],
    path_bounds: &BoundTemplate<C>,
    bineq_bounds: &BoundTemplate<Bineq>,
    global_bounds: &BoundTemplate<G>,
    variable_count: usize,
) -> Result<(Vec<Option<f64>>, Vec<Option<f64>>), GuessError>
where
    C: Vectorize<SX>,
    Bineq: Vectorize<SX>,
    Beq: Vectorize<SX>,
    G: Vectorize<SX>,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
    BoundTemplate<G>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<G>>,
{
    let mut variable_lower = vec![None; variable_count];
    let mut variable_upper = vec![None; variable_count];
    let global_start = variable_count - G::LEN;
    for (offset, bounds) in flatten_bounds(global_bounds).into_iter().enumerate() {
        apply_bounds_to_coordinate(
            &mut variable_lower,
            &mut variable_upper,
            global_start + offset,
            1.0,
            0.0,
            bounds.lower,
            bounds.upper,
        )?;
    }
    let boundary_ineq = flatten_bounds(bineq_bounds);
    let path = flatten_bounds(path_bounds);
    let mut boundary_ineq_index = 0usize;
    let mut path_index = 0usize;

    for row in &plan.rows {
        let bounds = match row.kind {
            RawInequalityKind::BoundaryEquality => Bounds1D {
                lower: Some(0.0),
                upper: Some(0.0),
            },
            RawInequalityKind::BoundaryInequality => {
                let bound = boundary_ineq
                    .get(boundary_ineq_index)
                    .cloned()
                    .unwrap_or_default();
                boundary_ineq_index += 1;
                bound
            }
            RawInequalityKind::Path => {
                let bound = if path.is_empty() {
                    Bounds1D::default()
                } else {
                    path[path_index % path.len()].clone()
                };
                path_index += 1;
                bound
            }
        };
        if let Some(promotion) = &row.promotion {
            let offset = offsets.get(promotion.offset_index).copied().unwrap_or(0.0);
            apply_bounds_to_coordinate(
                &mut variable_lower,
                &mut variable_upper,
                promotion.variable_index,
                promotion.scale,
                offset,
                bounds.lower,
                bounds.upper,
            )?;
        }
    }
    Ok((variable_lower, variable_upper))
}

fn build_inequality_lower<C, Beq, Bineq>(
    plan: &PromotionPlan,
    _offsets: &[f64],
    path_bounds: &BoundTemplate<C>,
    bineq_bounds: &BoundTemplate<Bineq>,
) -> Result<Vec<Option<f64>>, GuessError>
where
    C: Vectorize<SX>,
    Bineq: Vectorize<SX>,
    Beq: Vectorize<SX>,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
{
    let boundary_ineq = flatten_bounds(bineq_bounds);
    let path = flatten_bounds(path_bounds);
    let mut boundary_ineq_index = 0usize;
    let mut path_index = 0usize;
    let mut lower = Vec::with_capacity(plan.rows.len());
    for row in &plan.rows {
        if row.promotion.is_some() {
            lower.push(None);
            continue;
        }
        match row.kind {
            RawInequalityKind::BoundaryEquality => lower.push(Some(0.0)),
            RawInequalityKind::BoundaryInequality => {
                let bound = &boundary_ineq[boundary_ineq_index];
                lower.push(bound.lower);
                boundary_ineq_index += 1;
            }
            RawInequalityKind::Path => {
                let bound = &path[path_index % path.len()];
                lower.push(bound.lower);
                path_index += 1;
            }
        }
    }
    Ok(lower)
}

fn build_inequality_upper<C, Beq, Bineq>(
    plan: &PromotionPlan,
    _offsets: &[f64],
    path_bounds: &BoundTemplate<C>,
    bineq_bounds: &BoundTemplate<Bineq>,
) -> Result<Vec<Option<f64>>, GuessError>
where
    C: Vectorize<SX>,
    Bineq: Vectorize<SX>,
    Beq: Vectorize<SX>,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
{
    let boundary_ineq = flatten_bounds(bineq_bounds);
    let path = flatten_bounds(path_bounds);
    let mut boundary_ineq_index = 0usize;
    let mut path_index = 0usize;
    let mut upper = Vec::with_capacity(plan.rows.len());
    for row in &plan.rows {
        if row.promotion.is_some() {
            upper.push(None);
            continue;
        }
        match row.kind {
            RawInequalityKind::BoundaryEquality => upper.push(Some(0.0)),
            RawInequalityKind::BoundaryInequality => {
                let bound = &boundary_ineq[boundary_ineq_index];
                upper.push(bound.upper);
                boundary_ineq_index += 1;
            }
            RawInequalityKind::Path => {
                let bound = &path[path_index % path.len()];
                upper.push(bound.upper);
                path_index += 1;
            }
        }
    }
    Ok(upper)
}

fn flatten_bounds<T>(value: &T) -> Vec<Bounds1D>
where
    T: Vectorize<Bounds1D>,
{
    value.flatten_cloned()
}

fn prefixed_leaf_names<T>(prefix: &str) -> Vec<String>
where
    T: Vectorize<SX>,
{
    let mut out = Vec::with_capacity(T::LEN);
    T::flat_layout_names(prefix, &mut out);
    out
}

fn sort_ocp_constraint_report(report: &mut OcpConstraintViolationReport) {
    report.equalities.sort_by(|lhs, rhs| {
        rhs.worst_violation
            .total_cmp(&lhs.worst_violation)
            .then_with(|| lhs.label.cmp(&rhs.label))
    });
    report.inequalities.sort_by(|lhs, rhs| {
        rhs.worst_violation
            .total_cmp(&lhs.worst_violation)
            .then_with(|| lhs.label.cmp(&rhs.label))
    });
}

fn accumulate_equality_group(
    groups: &mut HashMap<(OcpConstraintCategory, String), EqualityGroupAccumulator>,
    label: &str,
    category: OcpConstraintCategory,
    value: f64,
    tolerance: f64,
) {
    let violation = value.abs();
    let entry = groups
        .entry((category, label.to_string()))
        .or_insert_with(|| EqualityGroupAccumulator {
            label: label.to_string(),
            category,
            worst_violation: 0.0,
            violating_instances: 0,
            total_instances: 0,
        });
    entry.worst_violation = entry.worst_violation.max(violation);
    entry.total_instances += 1;
    if violation > tolerance {
        entry.violating_instances += 1;
    }
}

fn accumulate_inequality_group(
    groups: &mut HashMap<(OcpConstraintCategory, String), InequalityGroupAccumulator>,
    label: &str,
    category: OcpConstraintCategory,
    value: f64,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    tolerance: f64,
) {
    if lower_bound.is_none() && upper_bound.is_none() {
        return;
    }
    let (lower_violation, upper_violation) = worst_bound_violation(value, lower_bound, upper_bound);
    let worst_violation = lower_violation.max(upper_violation);
    let entry = groups
        .entry((category, label.to_string()))
        .or_insert_with(|| InequalityGroupAccumulator {
            label: label.to_string(),
            category,
            worst_violation: 0.0,
            violating_instances: 0,
            total_instances: 0,
            lower_bound,
            upper_bound,
            lower_worst_violation: 0.0,
            upper_worst_violation: 0.0,
            lower_violated: false,
            upper_violated: false,
        });
    entry.lower_bound = entry.lower_bound.or(lower_bound);
    entry.upper_bound = entry.upper_bound.or(upper_bound);
    entry.worst_violation = entry.worst_violation.max(worst_violation);
    entry.lower_worst_violation = entry.lower_worst_violation.max(lower_violation);
    entry.upper_worst_violation = entry.upper_worst_violation.max(upper_violation);
    entry.total_instances += 1;
    entry.lower_violated |= lower_violation > 0.0;
    entry.upper_violated |= upper_violation > 0.0;
    if worst_violation > tolerance {
        entry.violating_instances += 1;
    }
}

fn equality_groups_from_map(
    groups: HashMap<(OcpConstraintCategory, String), EqualityGroupAccumulator>,
    tolerance: f64,
) -> Vec<OcpEqualityViolationGroup> {
    groups
        .into_values()
        .map(|group| OcpEqualityViolationGroup {
            label: group.label,
            category: group.category,
            worst_violation: group.worst_violation,
            violating_instances: group.violating_instances,
            total_instances: group.total_instances,
            satisfaction: classify_constraint_satisfaction(group.worst_violation, tolerance),
        })
        .collect()
}

fn inequality_groups_from_map(
    groups: HashMap<(OcpConstraintCategory, String), InequalityGroupAccumulator>,
    tolerance: f64,
) -> Vec<OcpInequalityViolationGroup> {
    groups
        .into_values()
        .map(|group| OcpInequalityViolationGroup {
            label: group.label,
            category: group.category,
            worst_violation: group.worst_violation,
            violating_instances: group.violating_instances,
            total_instances: group.total_instances,
            lower_bound: group.lower_bound,
            upper_bound: group.upper_bound,
            lower_satisfaction: group
                .lower_bound
                .map(|_| classify_constraint_satisfaction(group.lower_worst_violation, tolerance)),
            upper_satisfaction: group
                .upper_bound
                .map(|_| classify_constraint_satisfaction(group.upper_worst_violation, tolerance)),
            bound_side: constraint_bound_side(
                if group.lower_violated { 1.0 } else { 0.0 },
                if group.upper_violated { 1.0 } else { 0.0 },
            ),
            satisfaction: classify_constraint_satisfaction(group.worst_violation, tolerance),
        })
        .collect()
}

fn add_repeated_equalities(
    groups: &mut HashMap<(OcpConstraintCategory, String), EqualityGroupAccumulator>,
    values: &[f64],
    labels: &[String],
    category: OcpConstraintCategory,
    tolerance: f64,
) {
    if labels.is_empty() || values.is_empty() {
        return;
    }
    debug_assert_eq!(values.len() % labels.len(), 0);
    for (value, label) in values.iter().zip(labels.iter().cycle()) {
        accumulate_equality_group(groups, label, category, *value, tolerance);
    }
}

fn add_repeated_inequalities(
    groups: &mut HashMap<(OcpConstraintCategory, String), InequalityGroupAccumulator>,
    values: &[f64],
    labels: &[String],
    bounds: &[Bounds1D],
    category: OcpConstraintCategory,
    tolerance: f64,
) {
    if labels.is_empty() || bounds.is_empty() || values.is_empty() {
        return;
    }
    debug_assert_eq!(labels.len(), bounds.len());
    for (value, (label, bound)) in values
        .iter()
        .zip(labels.iter().cycle().zip(bounds.iter().cycle()))
    {
        accumulate_inequality_group(
            groups,
            label,
            category,
            *value,
            bound.lower,
            bound.upper,
            tolerance,
        );
    }
}

fn apply_bounds_to_coordinate(
    variable_lower: &mut [Option<f64>],
    variable_upper: &mut [Option<f64>],
    index: usize,
    scale: f64,
    offset: f64,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<(), GuessError> {
    let (candidate_lower, candidate_upper) = if scale > 0.0 {
        (
            lower.map(|value| (value - offset) / scale),
            upper.map(|value| (value - offset) / scale),
        )
    } else {
        (
            upper.map(|value| (value - offset) / scale),
            lower.map(|value| (value - offset) / scale),
        )
    };
    if let Some(value) = candidate_lower {
        variable_lower[index] = Some(match variable_lower[index] {
            Some(current) => current.max(value),
            None => value,
        });
    }
    if let Some(value) = candidate_upper {
        variable_upper[index] = Some(match variable_upper[index] {
            Some(current) => current.min(value),
            None => value,
        });
    }
    if let (Some(lower), Some(upper)) = (variable_lower[index], variable_upper[index])
        && lower > upper
    {
        return Err(GuessError::Invalid(format!(
            "promoted box bounds are inconsistent at variable index {index}"
        )));
    }
    Ok(())
}

fn subtract_vectorized<S, T>(lhs: &S, rhs: &S) -> Result<S::Rebind<T>, SxError>
where
    S: Vectorize<T>,
    T: ScalarLeaf + Clone + std::ops::Sub<Output = T>,
{
    let lhs_flat = lhs.flatten_cloned();
    let rhs_flat = rhs.flatten_cloned();
    unflatten_value::<S, T>(
        &lhs_flat
            .into_iter()
            .zip(rhs_flat)
            .map(|(lhs, rhs)| lhs - rhs)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| SxError::Graph(err.to_string()))
}

fn scale_vectorized<S>(value: &S, scalar: impl Into<SX>) -> Result<S, SxError>
where
    S: Vectorize<SX, Rebind<SX> = S>,
{
    let scalar = scalar.into();
    unflatten_value::<S, SX>(
        &value
            .flatten_cloned()
            .into_iter()
            .map(|entry| entry * scalar)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| SxError::Graph(err.to_string()))
}

fn weighted_sum_vectorized<S>(values: &[S], weights: &[f64]) -> Result<S, SxError>
where
    S: Vectorize<SX, Rebind<SX> = S>,
{
    let mut acc = vec![SX::zero(); S::LEN];
    for (value, weight) in values.iter().zip(weights.iter().copied()) {
        for (slot, entry) in acc.iter_mut().zip(value.flatten_cloned()) {
            *slot += weight * entry;
        }
    }
    unflatten_value::<S, SX>(&acc).map_err(|err| SxError::Graph(err.to_string()))
}

fn unflatten_typed_output<S>(output: &SXMatrix) -> Result<S, SxError>
where
    S: Vectorize<SX, Rebind<SX> = S>,
{
    unflatten_value::<S, SX>(output.nonzeros()).map_err(|err| SxError::Graph(err.to_string()))
}

fn call_typed_unary_output<S>(function: &SXFunction, inputs: Vec<SXMatrix>) -> Result<S, SxError>
where
    S: Vectorize<SX, Rebind<SX> = S>,
{
    let output = function.call_output(&inputs)?;
    unflatten_typed_output(&output)
}

fn rk4_integrate_symbolic<X, U>(
    x0: &X,
    u0: &U,
    dudt: &U,
    dt: SX,
    substeps: usize,
    mut ode: impl FnMut(&X, &U) -> Result<X, SxError>,
    mut lagrange: impl FnMut(&X, &U) -> Result<SX, SxError>,
) -> Result<(X, U, SX), SxError>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    U: Vectorize<SX, Rebind<SX> = U>,
{
    let h = dt / (substeps as f64);
    let dudt_flat = dudt.flatten_cloned();
    let mut x_flat = x0.flatten_cloned();
    let mut u_flat = u0.flatten_cloned();
    let mut objective = SX::zero();
    for _ in 0..substeps {
        let x = unflatten_value::<X, SX>(&x_flat).map_err(|err| SxError::Graph(err.to_string()))?;
        let u = unflatten_value::<U, SX>(&u_flat).map_err(|err| SxError::Graph(err.to_string()))?;
        let k1 = ode(&x, &u)?.flatten_cloned();
        let l1 = lagrange(&x, &u)?;

        let x_mid_1 = x_flat
            .iter()
            .zip(k1.iter())
            .map(|(state, deriv)| *state + 0.5 * h * *deriv)
            .collect::<Vec<_>>();
        let u_mid_1 = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + 0.5 * h * *rate)
            .collect::<Vec<_>>();
        let x2 =
            unflatten_value::<X, SX>(&x_mid_1).map_err(|err| SxError::Graph(err.to_string()))?;
        let u2 =
            unflatten_value::<U, SX>(&u_mid_1).map_err(|err| SxError::Graph(err.to_string()))?;
        let k2 = ode(&x2, &u2)?.flatten_cloned();
        let l2 = lagrange(&x2, &u2)?;

        let x_mid_2 = x_flat
            .iter()
            .zip(k2.iter())
            .map(|(state, deriv)| *state + 0.5 * h * *deriv)
            .collect::<Vec<_>>();
        let u_mid_2 = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + 0.5 * h * *rate)
            .collect::<Vec<_>>();
        let x3 =
            unflatten_value::<X, SX>(&x_mid_2).map_err(|err| SxError::Graph(err.to_string()))?;
        let u3 =
            unflatten_value::<U, SX>(&u_mid_2).map_err(|err| SxError::Graph(err.to_string()))?;
        let k3 = ode(&x3, &u3)?.flatten_cloned();
        let l3 = lagrange(&x3, &u3)?;

        let x_end = x_flat
            .iter()
            .zip(k3.iter())
            .map(|(state, deriv)| *state + h * *deriv)
            .collect::<Vec<_>>();
        let u_end = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + h * *rate)
            .collect::<Vec<_>>();
        let x4 = unflatten_value::<X, SX>(&x_end).map_err(|err| SxError::Graph(err.to_string()))?;
        let u4 = unflatten_value::<U, SX>(&u_end).map_err(|err| SxError::Graph(err.to_string()))?;
        let k4 = ode(&x4, &u4)?.flatten_cloned();
        let l4 = lagrange(&x4, &u4)?;

        for index in 0..X::LEN {
            x_flat[index] += h / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]);
        }
        for index in 0..U::LEN {
            u_flat[index] += h * dudt_flat[index];
        }
        objective += h / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4);
    }
    Ok((
        unflatten_value::<X, SX>(&x_flat).map_err(|err| SxError::Graph(err.to_string()))?,
        unflatten_value::<U, SX>(&u_flat).map_err(|err| SxError::Graph(err.to_string()))?,
        objective,
    ))
}

fn rk4_integrate_symbolic_state_only<X, U>(
    x0: &X,
    u0: &U,
    dudt: &U,
    dt: SX,
    substeps: usize,
    mut ode: impl FnMut(&X, &U) -> Result<X, SxError>,
) -> Result<(X, U), SxError>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    U: Vectorize<SX, Rebind<SX> = U>,
{
    let h = dt / (substeps as f64);
    let dudt_flat = dudt.flatten_cloned();
    let mut x_flat = x0.flatten_cloned();
    let mut u_flat = u0.flatten_cloned();
    for _ in 0..substeps {
        let x = unflatten_value::<X, SX>(&x_flat).map_err(|err| SxError::Graph(err.to_string()))?;
        let u = unflatten_value::<U, SX>(&u_flat).map_err(|err| SxError::Graph(err.to_string()))?;
        let k1 = ode(&x, &u)?.flatten_cloned();

        let x_mid_1 = x_flat
            .iter()
            .zip(k1.iter())
            .map(|(state, deriv)| *state + 0.5 * h * *deriv)
            .collect::<Vec<_>>();
        let u_mid_1 = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + 0.5 * h * *rate)
            .collect::<Vec<_>>();
        let x2 =
            unflatten_value::<X, SX>(&x_mid_1).map_err(|err| SxError::Graph(err.to_string()))?;
        let u2 =
            unflatten_value::<U, SX>(&u_mid_1).map_err(|err| SxError::Graph(err.to_string()))?;
        let k2 = ode(&x2, &u2)?.flatten_cloned();

        let x_mid_2 = x_flat
            .iter()
            .zip(k2.iter())
            .map(|(state, deriv)| *state + 0.5 * h * *deriv)
            .collect::<Vec<_>>();
        let u_mid_2 = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + 0.5 * h * *rate)
            .collect::<Vec<_>>();
        let x3 =
            unflatten_value::<X, SX>(&x_mid_2).map_err(|err| SxError::Graph(err.to_string()))?;
        let u3 =
            unflatten_value::<U, SX>(&u_mid_2).map_err(|err| SxError::Graph(err.to_string()))?;
        let k3 = ode(&x3, &u3)?.flatten_cloned();

        let x_end = x_flat
            .iter()
            .zip(k3.iter())
            .map(|(state, deriv)| *state + h * *deriv)
            .collect::<Vec<_>>();
        let u_end = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + h * *rate)
            .collect::<Vec<_>>();
        let x4 = unflatten_value::<X, SX>(&x_end).map_err(|err| SxError::Graph(err.to_string()))?;
        let u4 = unflatten_value::<U, SX>(&u_end).map_err(|err| SxError::Graph(err.to_string()))?;
        let k4 = ode(&x4, &u4)?.flatten_cloned();

        for index in 0..X::LEN {
            x_flat[index] += h / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]);
        }
        for index in 0..U::LEN {
            u_flat[index] += h * dudt_flat[index];
        }
    }
    Ok((
        unflatten_value::<X, SX>(&x_flat).map_err(|err| SxError::Graph(err.to_string()))?,
        unflatten_value::<U, SX>(&u_flat).map_err(|err| SxError::Graph(err.to_string()))?,
    ))
}

fn validate_interpolation_samples<X, U, G>(
    samples: &InterpolatedTrajectory<X, U, G>,
) -> Result<(), GuessError> {
    if samples.sample_times.len() < 2 {
        return Err(GuessError::Invalid(
            "interpolated guess requires at least two samples".to_string(),
        ));
    }
    if samples.sample_times.len() != samples.x_samples.len()
        || samples.sample_times.len() != samples.u_samples.len()
        || samples.sample_times.len() != samples.dudt_samples.len()
    {
        return Err(GuessError::Invalid(
            "interpolated guess sample arrays must have matching lengths".to_string(),
        ));
    }
    if !samples
        .sample_times
        .windows(2)
        .all(|window| window[0] < window[1])
    {
        return Err(GuessError::Invalid(
            "interpolated guess times must be strictly increasing".to_string(),
        ));
    }
    Ok(())
}

fn interpolate_at<T: Clone>(times: &[f64], values: &[T], target: f64) -> T {
    let index = times.partition_point(|time| *time < target);
    let clamped = index.min(values.len() - 1);
    values[clamped].clone()
}

fn rk4_rollout_numeric<X, U, P>(
    xdot: &CompiledXdot<X, U, P>,
    x0: &Numeric<X>,
    u0: &Numeric<U>,
    dudt: &Numeric<U>,
    parameters: &Numeric<P>,
    dt: f64,
    substeps: usize,
) -> Result<(Numeric<X>, Numeric<U>), GuessError>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    let h = dt / (substeps as f64);
    let dudt_flat = flatten_value(dudt);
    let mut x = x0.clone();
    let mut u = u0.clone();
    let mut x_flat = flatten_value(&x);
    let mut u_flat = flatten_value(&u);
    for _ in 0..substeps {
        let k1 = flatten_value(&xdot.eval(&x, &u, parameters)?);
        let x2_flat = x_flat
            .iter()
            .zip(k1.iter())
            .map(|(state, deriv)| *state + 0.5 * h * *deriv)
            .collect::<Vec<_>>();
        let u2_flat = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + 0.5 * h * *rate)
            .collect::<Vec<_>>();
        let x2 = unflatten_value::<Numeric<X>, f64>(&x2_flat)?;
        let u2 = unflatten_value::<Numeric<U>, f64>(&u2_flat)?;

        let k2 = flatten_value(&xdot.eval(&x2, &u2, parameters)?);
        let x3_flat = x_flat
            .iter()
            .zip(k2.iter())
            .map(|(state, deriv)| *state + 0.5 * h * *deriv)
            .collect::<Vec<_>>();
        let u3_flat = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + 0.5 * h * *rate)
            .collect::<Vec<_>>();
        let x3 = unflatten_value::<Numeric<X>, f64>(&x3_flat)?;
        let u3 = unflatten_value::<Numeric<U>, f64>(&u3_flat)?;

        let k3 = flatten_value(&xdot.eval(&x3, &u3, parameters)?);
        let x4_flat = x_flat
            .iter()
            .zip(k3.iter())
            .map(|(state, deriv)| *state + h * *deriv)
            .collect::<Vec<_>>();
        let u4_flat = u_flat
            .iter()
            .zip(dudt_flat.iter())
            .map(|(control, rate)| *control + h * *rate)
            .collect::<Vec<_>>();
        let x4 = unflatten_value::<Numeric<X>, f64>(&x4_flat)?;
        let u4 = unflatten_value::<Numeric<U>, f64>(&u4_flat)?;
        let k4 = flatten_value(&xdot.eval(&x4, &u4, parameters)?);

        for index in 0..X::LEN {
            x_flat[index] += h / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]);
        }
        for index in 0..U::LEN {
            u_flat[index] += h * dudt_flat[index];
        }
        x = unflatten_value::<Numeric<X>, f64>(&x_flat)?;
        u = unflatten_value::<Numeric<U>, f64>(&u_flat)?;
    }
    Ok((x, u))
}

fn collocation_coefficients(
    family: CollocationFamily,
    order: usize,
) -> Result<CollocationCoefficients, OcpCompileError> {
    let nodes = match family {
        CollocationFamily::GaussLegendre => shifted_gauss_legendre_roots(order),
        CollocationFamily::RadauIIA => shifted_radau_iia_roots(order),
    }?;
    let mut basis_nodes = Vec::with_capacity(order + 1);
    basis_nodes.push(0.0);
    basis_nodes.extend(nodes.iter().copied());
    let basis = basis_nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| lagrange_basis_polynomial(&basis_nodes, i, node))
        .collect::<Result<Vec<_>, _>>()
        .map_err(OcpCompileError::InvalidConfiguration)?;
    let c_matrix = nodes
        .iter()
        .map(|&node| {
            basis
                .iter()
                .map(|poly| evaluate_polynomial_derivative(poly, node))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let d_vector = basis
        .iter()
        .map(|poly| evaluate_polynomial(poly, 1.0))
        .collect::<Vec<_>>();
    let b_vector = basis
        .iter()
        .map(|poly| integrate_polynomial_unit_interval(poly))
        .collect::<Vec<_>>();
    Ok(CollocationCoefficients {
        nodes,
        c_matrix,
        d_vector,
        b_vector,
    })
}

fn shifted_gauss_legendre_roots(order: usize) -> Result<Vec<f64>, OcpCompileError> {
    bracket_roots(order, false, |x| legendre_value(order, x))
        .map(|roots| roots.into_iter().map(|root| 0.5 * (root + 1.0)).collect())
}

fn shifted_radau_iia_roots(order: usize) -> Result<Vec<f64>, OcpCompileError> {
    if order == 1 {
        return Ok(vec![1.0]);
    }
    let mut roots = bracket_roots(order - 1, false, |x| {
        legendre_value(order, x) - legendre_value(order - 1, x)
    })?;
    roots.push(1.0);
    roots.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("finite roots should compare"));
    Ok(roots.into_iter().map(|root| 0.5 * (root + 1.0)).collect())
}

fn bracket_roots(
    expected: usize,
    include_right_endpoint: bool,
    f: impl Fn(f64) -> f64,
) -> Result<Vec<f64>, OcpCompileError> {
    let samples = (expected.max(1) * 1024).max(4096);
    let mut roots = Vec::new();
    let mut left = -1.0;
    let mut f_left = f(left);
    for step in 1..=samples {
        let right = -1.0 + 2.0 * (step as f64) / (samples as f64);
        if !include_right_endpoint && step == samples {
            break;
        }
        let f_right = f(right);
        if f_left == 0.0 {
            roots.push(left);
        } else if f_left.signum() != f_right.signum() {
            roots.push(bisect_root(&f, left, right));
        }
        left = right;
        f_left = f_right;
    }
    dedup_roots(&mut roots);
    if roots.len() != expected {
        return Err(OcpCompileError::InvalidConfiguration(format!(
            "failed to bracket {expected} collocation roots, found {}",
            roots.len()
        )));
    }
    Ok(roots)
}

fn bisect_root(f: &impl Fn(f64) -> f64, mut left: f64, mut right: f64) -> f64 {
    for _ in 0..128 {
        let mid = 0.5 * (left + right);
        let f_mid = f(mid);
        if f_mid.abs() <= 1e-15 {
            return mid;
        }
        if f(left).signum() == f_mid.signum() {
            left = mid;
        } else {
            right = mid;
        }
    }
    0.5 * (left + right)
}

fn dedup_roots(roots: &mut Vec<f64>) {
    roots.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("finite roots should compare"));
    roots.dedup_by(|lhs, rhs| (*lhs - *rhs).abs() <= 1e-12);
}

fn legendre_value(order: usize, x: f64) -> f64 {
    match order {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p_nm2 = 1.0;
            let mut p_nm1 = x;
            for n in 2..=order {
                let n_f = n as f64;
                let p_n = ((2.0 * n_f - 1.0) * x * p_nm1 - (n_f - 1.0) * p_nm2) / n_f;
                p_nm2 = p_nm1;
                p_nm1 = p_n;
            }
            p_nm1
        }
    }
}

fn lagrange_basis_polynomial(
    nodes: &[f64],
    basis_index: usize,
    basis_node: f64,
) -> Result<Vec<f64>, String> {
    let mut coefficients = vec![1.0];
    let mut denominator = 1.0;
    for (index, &node) in nodes.iter().enumerate() {
        if index == basis_index {
            continue;
        }
        denominator *= basis_node - node;
        coefficients = multiply_polynomials(&coefficients, &[-node, 1.0]);
    }
    if denominator == 0.0 {
        return Err("duplicate collocation nodes".to_string());
    }
    for coefficient in &mut coefficients {
        *coefficient /= denominator;
    }
    Ok(coefficients)
}

fn multiply_polynomials(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; lhs.len() + rhs.len() - 1];
    for (lhs_index, lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, rhs_value) in rhs.iter().enumerate() {
            out[lhs_index + rhs_index] += lhs_value * rhs_value;
        }
    }
    out
}

fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .fold(0.0, |acc, (power, coefficient)| {
            acc + coefficient * x.powi(power as i32)
        })
}

fn evaluate_polynomial_derivative(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .fold(0.0, |acc, (power, coefficient)| {
            acc + (power as f64) * coefficient * x.powi(power as i32 - 1)
        })
}

fn integrate_polynomial_unit_interval(coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .fold(0.0, |acc, (power, coefficient)| {
            acc + coefficient / (power as f64 + 1.0)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::time::Instant;

    #[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
    struct State<T> {
        x: T,
        v: T,
    }

    #[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
    struct Control<T> {
        u: T,
    }

    #[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
    struct Params<T> {
        target: T,
    }

    #[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
    struct TestGlobals<T> {
        tf: T,
        terminal_target: T,
    }

    impl<T> OcpGlobalDesign<T> for TestGlobals<T>
    where
        T: Clone + ScalarLeaf,
    {
        fn final_time(&self) -> T {
            self.tf.clone()
        }

        fn from_final_time(tf: T) -> Self {
            Self {
                tf: tf.clone(),
                terminal_target: tf,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
    struct GlobalBoundary<T> {
        start: T,
        terminal_target: T,
    }

    fn lqr_ocp_ms<const N: usize, const RK4_SUBSTEPS: usize>()
    -> Ocp<State<SX>, Control<SX>, Params<SX>, (), State<SX>, (), runtime::MultipleShooting> {
        Ocp::new(
            "lqr_ms",
            runtime::MultipleShooting {
                intervals: N,
                rk4_substeps: RK4_SUBSTEPS,
            },
        )
        .objective_lagrange(
            |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &Params<SX>| {
                x.x.sqr() + x.v.sqr() + u.u.sqr() + 1e-3 * dudt.u.sqr()
            },
        )
        .objective_mayer(
            |_: &State<SX>,
             _: &Control<SX>,
             x_t: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             _: &SX| { 10.0 * x_t.x.sqr() + 10.0 * x_t.v.sqr() },
        )
        .ode(|x: &State<SX>, u: &Control<SX>, _: &Params<SX>| State { x: x.v, v: u.u })
        .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &Params<SX>| ())
        .boundary_equalities(
            |x0: &State<SX>,
             u0: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             p: &Params<SX>,
             tf: &SX| {
                let _ = (u0, p, tf);
                State { x: x0.x, v: x0.v }
            },
        )
        .boundary_inequalities(
            |_: &State<SX>,
             _: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             _: &SX| (),
        )
        .build()
        .expect("builder should succeed")
    }

    fn lqr_ocp_dc<const N: usize, const K: usize>()
    -> Ocp<State<SX>, Control<SX>, Params<SX>, (), State<SX>, (), runtime::DirectCollocation> {
        Ocp::new(
            "lqr_dc",
            runtime::DirectCollocation {
                intervals: N,
                order: K,
                family: CollocationFamily::RadauIIA,
                time_grid: Default::default(),
            },
        )
        .objective_lagrange(
            |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &Params<SX>| {
                x.x.sqr() + x.v.sqr() + u.u.sqr() + 1e-3 * dudt.u.sqr()
            },
        )
        .objective_mayer(
            |_: &State<SX>,
             _: &Control<SX>,
             x_t: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             _: &SX| { 10.0 * x_t.x.sqr() + 10.0 * x_t.v.sqr() },
        )
        .ode(|x: &State<SX>, u: &Control<SX>, _: &Params<SX>| State { x: x.v, v: u.u })
        .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &Params<SX>| ())
        .boundary_equalities(
            |x0: &State<SX>,
             u0: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             p: &Params<SX>,
             tf: &SX| {
                let _ = (u0, p, tf);
                State { x: x0.x, v: x0.v }
            },
        )
        .boundary_inequalities(
            |_: &State<SX>,
             _: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             _: &SX| (),
        )
        .build()
        .expect("builder should succeed")
    }

    fn global_design_ocp_ms<const N: usize, const RK4_SUBSTEPS: usize>() -> Ocp<
        State<SX>,
        Control<SX>,
        Params<SX>,
        (),
        GlobalBoundary<SX>,
        (),
        runtime::MultipleShooting,
        TestGlobals<SX>,
    > {
        Ocp::new(
            "global_design_ms",
            runtime::MultipleShooting {
                intervals: N,
                rk4_substeps: RK4_SUBSTEPS,
            },
        )
        .objective_lagrange(
            |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &Params<SX>| {
                x.x.sqr() + x.v.sqr() + u.u.sqr() + 1e-3 * dudt.u.sqr()
            },
        )
        .objective_mayer_global(
            |_: &State<SX>,
             _: &Control<SX>,
             x_t: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             g: &TestGlobals<SX>| { 10.0 * (x_t.x - g.terminal_target).sqr() },
        )
        .ode(|x: &State<SX>, u: &Control<SX>, _: &Params<SX>| State { x: x.v, v: u.u })
        .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &Params<SX>| ())
        .boundary_equalities_global(
            |x0: &State<SX>,
             _: &Control<SX>,
             x_t: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             g: &TestGlobals<SX>| GlobalBoundary {
                start: x0.x,
                terminal_target: x_t.x - g.terminal_target,
            },
        )
        .boundary_inequalities_global(
            |_: &State<SX>,
             _: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             _: &Params<SX>,
             _: &TestGlobals<SX>| (),
        )
        .build()
        .expect("builder should succeed")
    }

    #[test]
    fn multiple_shooting_scaling_maps_into_flat_nlp_scaling() {
        const N: usize = 2;
        const RK4_SUBSTEPS: usize = 2;

        let compiled = lqr_ocp_ms::<N, RK4_SUBSTEPS>()
            .compile_jit()
            .expect("multiple shooting OCP should compile");
        let runtime = runtime::MultipleShootingRuntimeValues {
            parameters: Params { target: 1.0 },
            beq: State { x: 0.0, v: 0.0 },
            bineq_bounds: (),
            path_bounds: (),
            global_bounds: FinalTime {
                tf: Bounds1D {
                    lower: Some(1.0),
                    upper: Some(1.0),
                },
            },
            initial_guess: runtime::MultipleShootingInitialGuess::Constant {
                x: State { x: 0.0, v: 0.0 },
                u: Control { u: 0.0 },
                dudt: Control { u: 0.0 },
                tf: 1.0,
            },
            scaling: Some(OcpScaling {
                objective: 7.0,
                state: State { x: 2.0, v: 3.0 },
                control: Control { u: 5.0 },
                control_rate: Control { u: 11.0 },
                global: FinalTime { tf: 13.0 },
                parameters: Params { target: 17.0 },
                path: Vec::new(),
                boundary_equalities: vec![19.0, 23.0],
                boundary_inequalities: Vec::new(),
            }),
        };

        let (_bounds, scaling) = compiled
            .test_runtime_bounds(&runtime)
            .expect("runtime bounds should build");
        let scaling = scaling.expect("runtime scaling should map into NLP scaling");

        assert_eq!(
            scaling.variables,
            vec![
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 5.0, 5.0, 11.0, 11.0, 13.0,
            ]
        );
        assert_eq!(
            scaling.constraints,
            vec![2.0, 3.0, 2.0, 3.0, 5.0, 5.0, 19.0, 23.0]
        );
        assert_abs_diff_eq!(scaling.objective, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn multiple_shooting_global_design_variables_are_vectorized() {
        const N: usize = 1;
        const RK4_SUBSTEPS: usize = 1;

        let compiled = global_design_ocp_ms::<N, RK4_SUBSTEPS>()
            .compile_jit()
            .expect("global-design multiple shooting OCP should compile");
        let runtime = runtime::MultipleShootingRuntimeValues {
            parameters: Params { target: 1.0 },
            beq: GlobalBoundary {
                start: 0.0,
                terminal_target: 0.0,
            },
            bineq_bounds: (),
            path_bounds: (),
            global_bounds: TestGlobals {
                tf: Bounds1D {
                    lower: Some(1.0),
                    upper: Some(1.0),
                },
                terminal_target: Bounds1D {
                    lower: Some(4.0),
                    upper: Some(5.0),
                },
            },
            initial_guess: runtime::MultipleShootingInitialGuess::ConstantGlobal {
                x: State { x: 0.0, v: 0.0 },
                u: Control { u: 0.0 },
                dudt: Control { u: 0.0 },
                global: TestGlobals {
                    tf: 1.0,
                    terminal_target: 4.5,
                },
            },
            scaling: None,
        };

        let initial_guess = compiled
            .test_initial_guess_flat(&runtime)
            .expect("initial guess should flatten");
        assert_eq!(&initial_guess[initial_guess.len() - 2..], &[1.0, 4.5]);

        let (bounds, _) = compiled
            .test_runtime_bounds(&runtime)
            .expect("runtime bounds should build");
        let lower = bounds
            .variables
            .lower
            .as_ref()
            .expect("variable lower bounds should exist");
        let upper = bounds
            .variables
            .upper
            .as_ref()
            .expect("variable upper bounds should exist");
        assert_eq!(&lower[lower.len() - 2..], &[Some(1.0), Some(4.0)]);
        assert_eq!(&upper[upper.len() - 2..], &[Some(1.0), Some(5.0)]);
    }

    #[test]
    fn direct_collocation_scaling_maps_into_flat_nlp_scaling() {
        const N: usize = 2;
        const K: usize = 2;

        let compiled = lqr_ocp_dc::<N, K>()
            .compile_jit()
            .expect("direct collocation OCP should compile");
        let runtime = runtime::DirectCollocationRuntimeValues {
            parameters: Params { target: 1.0 },
            beq: State { x: 0.0, v: 0.0 },
            bineq_bounds: (),
            path_bounds: (),
            global_bounds: FinalTime {
                tf: Bounds1D {
                    lower: Some(1.0),
                    upper: Some(1.0),
                },
            },
            initial_guess: runtime::DirectCollocationInitialGuess::Constant {
                x: State { x: 0.0, v: 0.0 },
                u: Control { u: 0.0 },
                dudt: Control { u: 0.0 },
                tf: 1.0,
            },
            scaling: Some(OcpScaling {
                objective: 7.0,
                state: State { x: 2.0, v: 3.0 },
                control: Control { u: 5.0 },
                control_rate: Control { u: 11.0 },
                global: FinalTime { tf: 13.0 },
                parameters: Params { target: 17.0 },
                path: Vec::new(),
                boundary_equalities: vec![19.0, 23.0],
                boundary_inequalities: Vec::new(),
            }),
        };

        let (_bounds, scaling) = compiled
            .test_runtime_bounds(&runtime)
            .expect("runtime bounds should build");
        let scaling = scaling.expect("runtime scaling should map into NLP scaling");

        assert_eq!(
            scaling.variables,
            vec![
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 5.0, 5.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0,
                3.0, 5.0, 5.0, 5.0, 5.0, 11.0, 11.0, 11.0, 11.0, 13.0,
            ]
        );
        assert_eq!(
            scaling.constraints,
            vec![
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 2.0, 3.0, 2.0, 3.0,
                5.0, 5.0, 19.0, 23.0,
            ]
        );
        assert_abs_diff_eq!(scaling.objective, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn runtime_mesh_helpers_track_nodes_and_terminal() {
        let mesh = runtime::Mesh {
            nodes: vec![State { x: 1.0, v: 2.0 }],
            terminal: State { x: 3.0, v: 4.0 },
        };
        assert_eq!(mesh.interval_count(), 1);
        assert_eq!(mesh.node_count(), 2);
        assert_eq!(
            mesh.states().cloned().collect::<Vec<_>>(),
            vec![State { x: 1.0, v: 2.0 }, State { x: 3.0, v: 4.0 }]
        );
    }

    fn time_grid_interval_lengths(mesh: &runtime::Mesh<f64>) -> Vec<f64> {
        mesh.nodes
            .iter()
            .copied()
            .chain(std::iter::once(mesh.terminal))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }

    #[test]
    fn cosine_time_grid_clusters_mesh_nodes_at_both_ends() {
        let uniform = runtime::time_grid_mesh(1.0, 4, runtime::TimeGrid::Uniform)
            .expect("uniform time grid should build");
        let cosine_zero =
            runtime::time_grid_mesh(1.0, 4, runtime::TimeGrid::Cosine { strength: 0.0 })
                .expect("zero-strength cosine time grid should build");
        assert_eq!(uniform, cosine_zero);

        let cosine = runtime::time_grid_mesh(1.0, 4, runtime::TimeGrid::Cosine { strength: 1.0 })
            .expect("cosine time grid should build");
        let mut times = cosine.nodes.clone();
        times.push(cosine.terminal);
        let intervals = times
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();

        assert_abs_diff_eq!(times[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(times[4], 1.0, epsilon = 1e-12);
        assert!(intervals[0] < intervals[1]);
        assert!(intervals[3] < intervals[2]);
    }

    #[test]
    fn tanh_time_grid_clusters_mesh_nodes_at_both_ends() {
        let tanh = runtime::time_grid_mesh(1.0, 8, runtime::TimeGrid::Tanh { strength: 1.0 })
            .expect("tanh time grid should build");
        let intervals = time_grid_interval_lengths(&tanh);

        assert!(intervals[0] < intervals[3]);
        assert!(intervals[7] < intervals[4]);
    }

    #[test]
    fn geometric_time_grid_clusters_toward_the_selected_end() {
        let start = runtime::time_grid_mesh(1.0, 5, runtime::TimeGrid::geometric_start(1.0))
            .expect("geometric start time grid should build");
        let end = runtime::time_grid_mesh(1.0, 5, runtime::TimeGrid::geometric_end(1.0))
            .expect("geometric end time grid should build");
        let start_intervals = time_grid_interval_lengths(&start);
        let end_intervals = time_grid_interval_lengths(&end);

        assert!(start_intervals[0] < start_intervals[4]);
        assert!(end_intervals[4] < end_intervals[0]);
    }

    #[test]
    fn focus_time_grid_clusters_near_requested_center() {
        let focus = runtime::time_grid_mesh(1.0, 9, runtime::TimeGrid::focus(0.5, 0.12, 1.0))
            .expect("focus time grid should build");
        let intervals = time_grid_interval_lengths(&focus);

        assert!(intervals[4] < intervals[0]);
        assert!(intervals[4] < intervals[8]);
    }

    #[test]
    fn piecewise_time_grid_places_breakpoint_at_interval_split() {
        let piecewise = runtime::time_grid_mesh(1.0, 4, runtime::TimeGrid::piecewise(0.25, 0.5))
            .expect("piecewise time grid should build");
        let intervals = time_grid_interval_lengths(&piecewise);

        assert_abs_diff_eq!(piecewise.nodes[2], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(intervals[0], intervals[1], epsilon = 1e-12);
        assert_abs_diff_eq!(intervals[2], intervals[3], epsilon = 1e-12);
        assert!(intervals[0] < intervals[2]);
    }

    #[test]
    fn adaptive_time_grid_mesh_clusters_at_larger_indicators() {
        let adaptive = runtime::adaptive_time_grid_mesh(1.0, &[0.0, 10.0, 0.0], 1.0)
            .expect("adaptive time grid should build");
        let intervals = time_grid_interval_lengths(&adaptive);

        assert!(intervals[1] < intervals[0]);
        assert!(intervals[1] < intervals[2]);
    }

    #[test]
    fn direct_collocation_time_grid_from_mesh_maps_roots_inside_existing_intervals() {
        let mesh = runtime::Mesh {
            nodes: vec![0.0, 0.1, 0.5],
            terminal: 1.0,
        };
        let family = CollocationFamily::RadauIIA;
        let order = 2;
        let coeffs =
            collocation_coefficients(family, order).expect("collocation coefficients should build");
        let grid = runtime::direct_collocation_time_grid_from_mesh(mesh.clone(), family, order)
            .expect("direct-collocation time grid should build from mesh");

        assert_eq!(grid.nodes, mesh);
        for interval in 0..grid.nodes.interval_count() {
            let start = grid.nodes.nodes[interval];
            let end = if interval + 1 < grid.nodes.interval_count() {
                grid.nodes.nodes[interval + 1]
            } else {
                grid.nodes.terminal
            };
            let step = end - start;
            for (root, &time) in grid.roots.intervals[interval].iter().enumerate() {
                assert_abs_diff_eq!((time - start) / step, coeffs.nodes[root], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn direct_collocation_time_grid_maps_roots_linearly_inside_warped_intervals() {
        let scheme = runtime::DirectCollocation {
            intervals: 4,
            order: 2,
            family: CollocationFamily::RadauIIA,
            time_grid: runtime::TimeGrid::Cosine { strength: 1.0 },
        };
        let coeffs = collocation_coefficients(scheme.family, scheme.order)
            .expect("collocation coefficients should build");
        let grid = runtime::direct_collocation_time_grid(2.0, scheme)
            .expect("direct-collocation time grid should build");

        let first_step = grid.nodes.nodes[1] - grid.nodes.nodes[0];
        let second_step = grid.nodes.nodes[2] - grid.nodes.nodes[1];
        assert!(first_step < second_step);
        for interval in 0..scheme.intervals {
            let start = grid.nodes.nodes[interval];
            let end = if interval + 1 < scheme.intervals {
                grid.nodes.nodes[interval + 1]
            } else {
                grid.nodes.terminal
            };
            let step = end - start;
            for (root, &time) in grid.roots.intervals[interval].iter().enumerate() {
                assert!(time >= start);
                assert!(time <= end);
                assert_abs_diff_eq!((time - start) / step, coeffs.nodes[root], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn multiple_shooting_builder_boxes_callbacks_under_the_hood() {
        let ocp = lqr_ocp_ms::<4, 2>();
        let _compiled = ocp.compile_jit().expect("compile should succeed");
    }

    #[test]
    fn multiple_shooting_defaults_to_reused_symbolic_functions() {
        let ocp = lqr_ocp_ms::<4, 2>();
        let compiled = ocp
            .compile_jit_with_ocp_options(OcpCompileOptions {
                function_options: FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
                symbolic_functions: OcpSymbolicFunctionOptions::default(),
                hessian_strategy: HessianStrategy::LowerTriangleByColumn,
            })
            .expect("compile should succeed");
        let stats = &compiled.backend_compile_report().stats;

        assert!(stats.symbolic_function_count > 0);
        assert!(stats.call_site_count > 0);
    }

    #[test]
    fn multiple_shooting_integrator_can_be_built_as_symbolic_function() {
        let ocp = lqr_ocp_ms::<4, 2>();
        let options = OcpSymbolicFunctionOptions {
            multiple_shooting_integrator: OcpKernelFunctionOptions::function_with_call_policy(
                CallPolicy::NoInlineLLVM,
            ),
            ..OcpSymbolicFunctionOptions::default()
        };
        let library = ocp
            .test_symbolic_function_library(options)
            .expect("symbolic function library should build");
        let integrator = library
            .multiple_shooting_integrator
            .expect("multiple shooting integrator function should be present");
        assert_eq!(
            integrator.call_policy_override(),
            Some(CallPolicy::NoInlineLLVM)
        );
    }

    #[test]
    fn multiple_shooting_inline_symbolic_function_option_removes_call_sites() {
        let ocp = lqr_ocp_ms::<4, 2>();
        let compiled = ocp
            .compile_jit_with_ocp_options(OcpCompileOptions {
                function_options: FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
                symbolic_functions: OcpSymbolicFunctionOptions::inline_all(),
                hessian_strategy: HessianStrategy::LowerTriangleByColumn,
            })
            .expect("compile should succeed");

        assert_eq!(compiled.backend_compile_report().stats.call_site_count, 0);
    }

    #[test]
    fn direct_collocation_defaults_to_reused_symbolic_functions() {
        let ocp = lqr_ocp_dc::<2, 1>();
        let compiled = ocp
            .compile_jit_with_ocp_options(OcpCompileOptions {
                function_options: FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
                symbolic_functions: OcpSymbolicFunctionOptions::default(),
                hessian_strategy: HessianStrategy::LowerTriangleByColumn,
            })
            .expect("compile should succeed");
        let stats = &compiled.backend_compile_report().stats;

        assert!(stats.symbolic_function_count > 0);
        assert!(stats.call_site_count > 0);
    }

    #[test]
    #[ignore = "manual profiling helper"]
    fn profile_small_direct_collocation_symbolic_setup() {
        let ocp = lqr_ocp_dc::<8, 2>();
        for (label, symbolic_functions) in [
            ("inline_all", OcpSymbolicFunctionOptions::inline_all()),
            ("default", OcpSymbolicFunctionOptions::default()),
            (
                "function_inline_at_call",
                OcpSymbolicFunctionOptions::function_all_with_call_policy(CallPolicy::InlineAtCall),
            ),
        ] {
            let started = Instant::now();
            let mut symbolic_metadata = None;
            let compiled = ocp
                .compile_jit_with_ocp_options_and_progress_callback(
                    OcpCompileOptions {
                        function_options: FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
                        symbolic_functions,
                        hessian_strategy: HessianStrategy::LowerTriangleByColumn,
                    },
                    |progress| {
                        if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                            symbolic_metadata = Some(metadata);
                        }
                    },
                )
                .expect("compile should succeed");
            println!(
                "{label}: total={:?} symbolic_ready={:?} setup_profile={:?} stats={:?}",
                started.elapsed(),
                symbolic_metadata,
                compiled.backend_compile_report().setup_profile,
                compiled.backend_compile_report().stats,
            );
        }
    }

    #[test]
    fn collocation_coefficients_match_known_small_radau_nodes() {
        let coeffs = collocation_coefficients(CollocationFamily::RadauIIA, 2)
            .expect("coefficients should build");
        assert_abs_diff_eq!(coeffs.nodes[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(coeffs.nodes[1], 1.0, epsilon = 1e-12);
    }
}

#[cfg(test)]
#[path = "../../test_support/jacobian_proptest/mod.rs"]
mod jacobian_proptest;

#[cfg(test)]
#[path = "ocp_jacobian_prop_tests.rs"]
mod ocp_jacobian_prop_tests;
