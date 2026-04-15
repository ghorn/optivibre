use anyhow::Result as AnyResult;
use optimization::{
    BackendCompileReport, BackendTimingMetadata, CallPolicy, ClarabelSqpError, ClarabelSqpOptions,
    ClarabelSqpSummary, ConstraintBoundSide, ConstraintSatisfaction, FunctionCompileOptions,
    FiniteDifferenceValidationOptions, InteriorPointIterationSnapshot, InteriorPointOptions,
    InteriorPointSolveError, InteriorPointSummary, LlvmOptimizationLevel,
    NlpCompileStats, NlpDerivativeValidationReport, NlpEvaluationBenchmark,
    NlpEvaluationBenchmarkOptions, NlpEvaluationKernelKind, ScalarLeaf,
    SqpIterationSnapshot, SymbolicCompileMetadata, SymbolicCompileProgress,
    SymbolicCompileStageProgress, SymbolicNlpBuildError, SymbolicNlpCompileError,
    SymbolicNlpOutputs, TypedCompiledJitNlp, TypedRuntimeNlpBounds, Vectorize,
    VectorizeLayoutError, classify_constraint_satisfaction, constraint_bound_side,
    flatten_value, symbolic_column, symbolic_nlp, symbolic_value, unflatten_value,
    worst_bound_violation,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptIterationSnapshot, IpoptOptions, IpoptSolveError, IpoptSummary};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use sx_codegen_llvm::{CompiledJitFunction, JitExecutionContext};
use sx_core::{NamedMatrix, NodeView, SX, SXFunction, SXMatrix, SxError};
use thiserror::Error;

const NLP_BOUND_INF: f64 = 1e20;
pub const MULTIPLE_SHOOTING_ARC_SAMPLES: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollocationFamily {
    GaussLegendre,
    RadauIIA,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MultipleShooting<const N: usize, const RK4_SUBSTEPS: usize>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCollocation<const N: usize, const K: usize> {
    pub family: CollocationFamily,
}

impl<const N: usize, const K: usize> Default for DirectCollocation<N, K> {
    fn default() -> Self {
        Self {
            family: CollocationFamily::RadauIIA,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bounds1D {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl Default for Bounds1D {
    fn default() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }
}

impl ScalarLeaf for Bounds1D {}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<T, const N: usize> {
    pub nodes: [T; N],
    pub terminal: T,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntervalGrid<T, const N: usize, const K: usize> {
    pub intervals: [[T; K]; N],
}

impl<S, V, const N: usize> Vectorize<S> for Mesh<V, N>
where
    S: ScalarLeaf,
    V: Vectorize<S>,
{
    type Rebind<U: ScalarLeaf> = Mesh<V::Rebind<U>, N>;
    type View<'a>
        = Mesh<V::View<'a>, N>
    where
        Self: 'a,
        S: 'a,
        V: 'a;

    const LEN: usize = N * V::LEN + V::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a S>) {
        for node in &self.nodes {
            node.flatten_refs(out);
        }
        self.terminal.flatten_refs(out);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        Mesh {
            nodes: std::array::from_fn(|_| V::from_flat_fn(f)),
            terminal: V::from_flat_fn(f),
        }
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        Self: 'a,
        S: 'a,
        V: 'a,
    {
        Mesh {
            nodes: std::array::from_fn(|index| self.nodes[index].view()),
            terminal: self.terminal.view(),
        }
    }

    fn view_from_flat_slice<'a>(slice: &'a [S], index: &mut usize) -> Self::View<'a>
    where
        S: 'a,
        V: 'a,
    {
        Mesh {
            nodes: std::array::from_fn(|_| V::view_from_flat_slice(slice, index)),
            terminal: V::view_from_flat_slice(slice, index),
        }
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        for index in 0..N {
            V::flat_layout_names(
                &optimization::extend_layout_name(prefix, &format!("nodes[{index}]")),
                out,
            );
        }
        V::flat_layout_names(&optimization::extend_layout_name(prefix, "terminal"), out);
    }
}

impl<S, V, const N: usize, const K: usize> Vectorize<S> for IntervalGrid<V, N, K>
where
    S: ScalarLeaf,
    V: Vectorize<S>,
{
    type Rebind<U: ScalarLeaf> = IntervalGrid<V::Rebind<U>, N, K>;
    type View<'a>
        = IntervalGrid<V::View<'a>, N, K>
    where
        Self: 'a,
        S: 'a,
        V: 'a;

    const LEN: usize = N * K * V::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a S>) {
        for interval in &self.intervals {
            for value in interval {
                value.flatten_refs(out);
            }
        }
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        IntervalGrid {
            intervals: std::array::from_fn(|_| std::array::from_fn(|_| V::from_flat_fn(f))),
        }
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        Self: 'a,
        S: 'a,
        V: 'a,
    {
        IntervalGrid {
            intervals: std::array::from_fn(|interval| {
                std::array::from_fn(|root| self.intervals[interval][root].view())
            }),
        }
    }

    fn view_from_flat_slice<'a>(slice: &'a [S], index: &mut usize) -> Self::View<'a>
    where
        S: 'a,
        V: 'a,
    {
        IntervalGrid {
            intervals: std::array::from_fn(|_| {
                std::array::from_fn(|_| V::view_from_flat_slice(slice, index))
            }),
        }
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        for interval in 0..N {
            for root in 0..K {
                V::flat_layout_names(
                    &optimization::extend_layout_name(
                        prefix,
                        &format!("intervals[{interval}][{root}]"),
                    ),
                    out,
                );
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultipleShootingTrajectories<X, U, const N: usize> {
    pub x: Mesh<X, N>,
    pub u: Mesh<U, N>,
    pub dudt: [U; N],
    pub tf: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCollocationTrajectories<X, U, const N: usize, const K: usize> {
    pub x: Mesh<X, N>,
    pub u: Mesh<U, N>,
    pub root_x: IntervalGrid<X, N, K>,
    pub root_u: IntervalGrid<U, N, K>,
    pub root_dudt: IntervalGrid<U, N, K>,
    pub tf: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultipleShootingTimeGrid<const N: usize> {
    pub nodes: Mesh<f64, N>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCollocationTimeGrid<const N: usize, const K: usize> {
    pub nodes: Mesh<f64, N>,
    pub roots: IntervalGrid<f64, N, K>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntervalArc<T> {
    pub times: Vec<f64>,
    pub values: Vec<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterpolatedTrajectory<X, U> {
    pub sample_times: Vec<f64>,
    pub x_samples: Vec<X>,
    pub u_samples: Vec<U>,
    pub dudt_samples: Vec<U>,
    pub tf: f64,
}

pub type ControllerFn<X, U, P> = dyn Fn(f64, &X, &U, &P) -> U + Send + Sync;

pub enum MultipleShootingInitialGuess<X, U, P, const N: usize> {
    Explicit(MultipleShootingTrajectories<X, U, N>),
    Constant {
        x: X,
        u: U,
        dudt: U,
        tf: f64,
    },
    Interpolated(InterpolatedTrajectory<X, U>),
    Rollout {
        x0: X,
        u0: U,
        tf: f64,
        controller: Box<ControllerFn<X, U, P>>,
    },
}

pub enum DirectCollocationInitialGuess<X, U, P, const N: usize, const K: usize> {
    Explicit(DirectCollocationTrajectories<X, U, N, K>),
    Constant {
        x: X,
        u: U,
        dudt: U,
        tf: f64,
    },
    Interpolated(InterpolatedTrajectory<X, U>),
    Rollout {
        x0: X,
        u0: U,
        tf: f64,
        controller: Box<ControllerFn<X, U, P>>,
    },
}

pub struct MultipleShootingRuntimeValues<P, C, Beq, Bineq, X, U, const N: usize> {
    pub parameters: P,
    pub beq: Beq,
    pub bineq_bounds: Bineq,
    pub path_bounds: C,
    pub tf_bounds: Bounds1D,
    pub initial_guess: MultipleShootingInitialGuess<X, U, P, N>,
}

pub struct DirectCollocationRuntimeValues<P, C, Beq, Bineq, X, U, const N: usize, const K: usize> {
    pub parameters: P,
    pub beq: Beq,
    pub bineq_bounds: Bineq,
    pub path_bounds: C,
    pub tf_bounds: Bounds1D,
    pub initial_guess: DirectCollocationInitialGuess<X, U, P, N, K>,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingSqpSnapshot<X, U, const N: usize> {
    pub trajectories: MultipleShootingTrajectories<X, U, N>,
    pub time_grid: MultipleShootingTimeGrid<N>,
    pub solver: SqpIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationSqpSnapshot<X, U, const N: usize, const K: usize> {
    pub trajectories: DirectCollocationTrajectories<X, U, N, K>,
    pub time_grid: DirectCollocationTimeGrid<N, K>,
    pub solver: SqpIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingInteriorPointSnapshot<X, U, const N: usize> {
    pub trajectories: MultipleShootingTrajectories<X, U, N>,
    pub time_grid: MultipleShootingTimeGrid<N>,
    pub solver: InteriorPointIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationInteriorPointSnapshot<X, U, const N: usize, const K: usize> {
    pub trajectories: DirectCollocationTrajectories<X, U, N, K>,
    pub time_grid: DirectCollocationTimeGrid<N, K>,
    pub solver: InteriorPointIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingSqpSolveResult<X, U, const N: usize> {
    pub trajectories: MultipleShootingTrajectories<X, U, N>,
    pub time_grid: MultipleShootingTimeGrid<N>,
    pub solver: ClarabelSqpSummary,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationSqpSolveResult<X, U, const N: usize, const K: usize> {
    pub trajectories: DirectCollocationTrajectories<X, U, N, K>,
    pub time_grid: DirectCollocationTimeGrid<N, K>,
    pub solver: ClarabelSqpSummary,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingInteriorPointSolveResult<X, U, const N: usize> {
    pub trajectories: MultipleShootingTrajectories<X, U, N>,
    pub time_grid: MultipleShootingTimeGrid<N>,
    pub solver: InteriorPointSummary,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationInteriorPointSolveResult<X, U, const N: usize, const K: usize> {
    pub trajectories: DirectCollocationTrajectories<X, U, N, K>,
    pub time_grid: DirectCollocationTimeGrid<N, K>,
    pub solver: InteriorPointSummary,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct MultipleShootingIpoptSolveResult<X, U, const N: usize> {
    pub trajectories: MultipleShootingTrajectories<X, U, N>,
    pub time_grid: MultipleShootingTimeGrid<N>,
    pub solver: IpoptSummary,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct MultipleShootingIpoptSnapshot<X, U, const N: usize> {
    pub trajectories: MultipleShootingTrajectories<X, U, N>,
    pub time_grid: MultipleShootingTimeGrid<N>,
    pub solver: IpoptIterationSnapshot,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct DirectCollocationIpoptSolveResult<X, U, const N: usize, const K: usize> {
    pub trajectories: DirectCollocationTrajectories<X, U, N, K>,
    pub time_grid: DirectCollocationTimeGrid<N, K>,
    pub solver: IpoptSummary,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct DirectCollocationIpoptSnapshot<X, U, const N: usize, const K: usize> {
    pub trajectories: DirectCollocationTrajectories<X, U, N, K>,
    pub time_grid: DirectCollocationTimeGrid<N, K>,
    pub solver: IpoptIterationSnapshot,
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

type ObjectiveLagrangeFn<X, U, P> = dyn Fn(&X, &U, &U, &P) -> SX + Send + Sync;
type ObjectiveMayerFn<X, U, P> = dyn Fn(&X, &U, &X, &U, &P, &SX) -> SX + Send + Sync;
type OdeFn<X, U, P> = dyn Fn(&X, &U, &P) -> X + Send + Sync;
type PathConstraintsFn<X, U, P, C> = dyn Fn(&X, &U, &U, &P) -> C + Send + Sync;
type BoundaryFn<X, U, P, B> = dyn Fn(&X, &U, &X, &U, &P, &SX) -> B + Send + Sync;

pub struct Ocp<X, U, P, C, Beq, Bineq, Scheme> {
    name: String,
    scheme: Scheme,
    objective_lagrange: Box<ObjectiveLagrangeFn<X, U, P>>,
    objective_mayer: Box<ObjectiveMayerFn<X, U, P>>,
    ode: Box<OdeFn<X, U, P>>,
    path_constraints: Box<PathConstraintsFn<X, U, P, C>>,
    boundary_equalities: Box<BoundaryFn<X, U, P, Beq>>,
    boundary_inequalities: Box<BoundaryFn<X, U, P, Bineq>>,
}

pub struct OcpBuilder<X, U, P, C, Beq, Bineq, Scheme> {
    name: String,
    scheme: Scheme,
    objective_lagrange: Option<Box<ObjectiveLagrangeFn<X, U, P>>>,
    objective_mayer: Option<Box<ObjectiveMayerFn<X, U, P>>>,
    ode: Option<Box<OdeFn<X, U, P>>>,
    path_constraints: Option<Box<PathConstraintsFn<X, U, P, C>>>,
    boundary_equalities: Option<Box<BoundaryFn<X, U, P, Beq>>>,
    boundary_inequalities: Option<Box<BoundaryFn<X, U, P, Bineq>>>,
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
        }
    }

    pub fn for_direct_collocation(function_options: FunctionCompileOptions) -> Self {
        Self {
            function_options,
            symbolic_functions: OcpSymbolicFunctionOptions::direct_collocation_default(),
        }
    }
}

impl From<FunctionCompileOptions> for OcpCompileOptions {
    fn from(function_options: FunctionCompileOptions) -> Self {
        Self {
            function_options,
            symbolic_functions: OcpSymbolicFunctionOptions::default(),
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
type MsVars<X, U, const N: usize> = (Mesh<X, N>, Mesh<U, N>, [U; N], SX);
type MsEqualities<X, U, const N: usize> = ([X; N], [U; N]);
type MsIneq<C, Beq, Bineq, const N: usize> = (Beq, Bineq, [C; N]);
type MsVarsNum<X, U, const N: usize> = (
    Mesh<Numeric<X>, N>,
    Mesh<Numeric<U>, N>,
    [Numeric<U>; N],
    f64,
);
type MsIneqNum<C, Beq, Bineq, const N: usize> = (Numeric<Beq>, Numeric<Bineq>, [Numeric<C>; N]);
type OcpParametersNum<P, Beq> = (Numeric<P>, Numeric<Beq>);
type DcVars<X, U, const N: usize, const K: usize> = (
    Mesh<X, N>,
    Mesh<U, N>,
    IntervalGrid<X, N, K>,
    IntervalGrid<U, N, K>,
    IntervalGrid<U, N, K>,
    SX,
);
type DcEqualities<X, U, const N: usize, const K: usize> =
    (IntervalGrid<X, N, K>, IntervalGrid<U, N, K>, [X; N], [U; N]);
type DcIneq<C, Beq, Bineq, const N: usize, const K: usize> = (Beq, Bineq, IntervalGrid<C, N, K>);
type DcVarsNum<X, U, const N: usize, const K: usize> = (
    Mesh<Numeric<X>, N>,
    Mesh<Numeric<U>, N>,
    IntervalGrid<Numeric<X>, N, K>,
    IntervalGrid<Numeric<U>, N, K>,
    IntervalGrid<Numeric<U>, N, K>,
    f64,
);
type DcIneqNum<C, Beq, Bineq, const N: usize, const K: usize> =
    (Numeric<Beq>, Numeric<Bineq>, IntervalGrid<Numeric<C>, N, K>);
type MsArcSampleOutputNum<X, U> = (Numeric<X>, Numeric<U>);
#[derive(Clone, Debug)]
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
struct CompiledMultipleShootingArc<X, U, P, const RK4_SUBSTEPS: usize> {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
    _marker: PhantomData<fn() -> (X, U, P)>,
}

#[derive(Clone, Debug, Default)]
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
}

pub struct CompiledMultipleShootingOcp<
    X,
    U,
    P,
    C,
    Beq,
    Bineq,
    const N: usize,
    const RK4_SUBSTEPS: usize,
> {
    compiled: TypedCompiledJitNlp<
        MsVars<X, U, N>,
        OcpParameters<P, Beq>,
        MsEqualities<X, U, N>,
        MsIneq<C, Beq, Bineq, N>,
    >,
    promotion_plan: PromotionPlan,
    promotion_offsets: PromotionOffsets<OcpParameters<P, Beq>>,
    xdot_helper: CompiledXdot<X, U, P>,
    rk4_arc_helper: CompiledMultipleShootingArc<X, U, P, RK4_SUBSTEPS>,
    helper_compile_stats: OcpHelperCompileStats,
    _marker: PhantomData<fn() -> (C, Bineq)>,
}

pub struct CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, const N: usize, const K: usize> {
    compiled: TypedCompiledJitNlp<
        DcVars<X, U, N, K>,
        OcpParameters<P, Beq>,
        DcEqualities<X, U, N, K>,
        DcIneq<C, Beq, Bineq, N, K>,
    >,
    promotion_plan: PromotionPlan,
    promotion_offsets: PromotionOffsets<OcpParameters<P, Beq>>,
    xdot_helper: CompiledXdot<X, U, P>,
    coefficients: CollocationCoefficients,
    helper_compile_stats: OcpHelperCompileStats,
    _marker: PhantomData<fn() -> (C, Bineq)>,
}

impl<X, U, P, C, Beq, Bineq, Scheme> Ocp<X, U, P, C, Beq, Bineq, Scheme> {
    pub fn new(
        name: impl Into<String>,
        scheme: Scheme,
    ) -> OcpBuilder<X, U, P, C, Beq, Bineq, Scheme> {
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

impl<X, U, P, C, Beq, Bineq, Scheme> OcpBuilder<X, U, P, C, Beq, Bineq, Scheme> {
    pub fn objective_lagrange<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &U, &P) -> SX + Send + Sync + 'static,
    {
        self.objective_lagrange = Some(Box::new(f));
        self
    }

    pub fn objective_mayer<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &SX) -> SX + Send + Sync + 'static,
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
    {
        self.boundary_equalities = Some(Box::new(f));
        self
    }

    pub fn boundary_inequalities<F>(mut self, f: F) -> Self
    where
        F: Fn(&X, &U, &X, &U, &P, &SX) -> Bineq + Send + Sync + 'static,
    {
        self.boundary_inequalities = Some(Box::new(f));
        self
    }

    pub fn build(self) -> Result<Ocp<X, U, P, C, Beq, Bineq, Scheme>, OcpBuildError> {
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

impl<X, U, P, C, Beq, Bineq, Scheme> Ocp<X, U, P, C, Beq, Bineq, Scheme>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
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
            let objective = (self.objective_lagrange)(&x, &u, &dudt, &p);
            SXFunction::new(
                format!("{}_objective_lagrange", self.name),
                vec![
                    NamedMatrix::new("x", symbolic_column(&x)?)?,
                    NamedMatrix::new("u", symbolic_column(&u)?)?,
                    NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
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
            let tf = SX::sym("tf");
            let objective = (self.objective_mayer)(&x0, &u0, &xf, &uf, &p, &tf);
            SXFunction::new(
                format!("{}_objective_mayer", self.name),
                vec![
                    NamedMatrix::new("x0", symbolic_column(&x0)?)?,
                    NamedMatrix::new("u0", symbolic_column(&u0)?)?,
                    NamedMatrix::new("xf", symbolic_column(&xf)?)?,
                    NamedMatrix::new("uf", symbolic_column(&uf)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("tf", SXMatrix::dense_column(vec![tf])?)?,
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
            let tf = SX::sym("tf");
            let values = (self.boundary_equalities)(&x0, &u0, &xf, &uf, &p, &tf);
            SXFunction::new(
                format!("{}_boundary_equalities", self.name),
                vec![
                    NamedMatrix::new("x0", symbolic_column(&x0)?)?,
                    NamedMatrix::new("u0", symbolic_column(&u0)?)?,
                    NamedMatrix::new("xf", symbolic_column(&xf)?)?,
                    NamedMatrix::new("uf", symbolic_column(&uf)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("tf", SXMatrix::dense_column(vec![tf])?)?,
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
            let tf = SX::sym("tf");
            let values = (self.boundary_inequalities)(&x0, &u0, &xf, &uf, &p, &tf);
            SXFunction::new(
                format!("{}_boundary_inequalities", self.name),
                vec![
                    NamedMatrix::new("x0", symbolic_column(&x0)?)?,
                    NamedMatrix::new("u0", symbolic_column(&u0)?)?,
                    NamedMatrix::new("xf", symbolic_column(&xf)?)?,
                    NamedMatrix::new("uf", symbolic_column(&uf)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("tf", SXMatrix::dense_column(vec![tf])?)?,
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
    ) -> Result<SX, SxError> {
        match &library.objective_lagrange {
            Some(function) => function.call_scalar(&[
                symbolic_column(x)?,
                symbolic_column(u)?,
                symbolic_column(dudt)?,
                symbolic_column(parameters)?,
            ]),
            None => Ok((self.objective_lagrange)(x, u, dudt, parameters)),
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
        tf: &SX,
    ) -> Result<SX, SxError> {
        match &library.objective_mayer {
            Some(function) => function.call_scalar(&[
                symbolic_column(x0)?,
                symbolic_column(u0)?,
                symbolic_column(xf)?,
                symbolic_column(uf)?,
                symbolic_column(parameters)?,
                SXMatrix::dense_column(vec![*tf])?,
            ]),
            None => Ok((self.objective_mayer)(x0, u0, xf, uf, parameters, tf)),
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
        tf: &SX,
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
                    SXMatrix::dense_column(vec![*tf])?,
                ],
            ),
            None => Ok((self.boundary_equalities)(x0, u0, xf, uf, parameters, tf)),
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
        tf: &SX,
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
                    SXMatrix::dense_column(vec![*tf])?,
                ],
            ),
            None => Ok((self.boundary_inequalities)(x0, u0, xf, uf, parameters, tf)),
        }
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const RK4_SUBSTEPS: usize>
    Ocp<X, U, P, C, Beq, Bineq, MultipleShooting<N, RK4_SUBSTEPS>>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
{
    fn build_multiple_shooting_symbolic_function_library(
        &self,
        options: OcpSymbolicFunctionOptions,
    ) -> Result<OcpSymbolicFunctionLibrary, SxError> {
        let mut library = self.build_symbolic_function_library(options)?;
        library.multiple_shooting_integrator = self
            .build_multiple_shooting_integrator_symbolic_function(
                &library,
                options.multiple_shooting_integrator,
            )?;
        Ok(library)
    }

    fn build_multiple_shooting_integrator_symbolic_function(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        options: OcpKernelFunctionOptions,
    ) -> Result<Option<SXFunction>, SxError> {
        self.configured_symbolic_function(options, || {
            let x = symbolic_value::<X>("x")?;
            let u = symbolic_value::<U>("u")?;
            let dudt = symbolic_value::<U>("dudt")?;
            let p = symbolic_value::<P>("p")?;
            let dt = SX::sym("dt");
            let (x_next, u_next, objective) = rk4_integrate_symbolic(
                &x,
                &u,
                &dudt,
                dt,
                RK4_SUBSTEPS,
                |x_eval, u_eval| self.eval_ode_symbolic(library, x_eval, u_eval, &p),
                |x_eval, u_eval| {
                    self.eval_objective_lagrange_symbolic(library, x_eval, u_eval, &dudt, &p)
                },
            )?;
            SXFunction::new(
                format!("{}_multiple_shooting_integrator", self.name),
                vec![
                    NamedMatrix::new("x", symbolic_column(&x)?)?,
                    NamedMatrix::new("u", symbolic_column(&u)?)?,
                    NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
                    NamedMatrix::new("p", symbolic_column(&p)?)?,
                    NamedMatrix::new("dt", SXMatrix::dense_column(vec![dt])?)?,
                ],
                vec![
                    NamedMatrix::new("x_next", symbolic_column(&x_next)?)?,
                    NamedMatrix::new("u_next", symbolic_column(&u_next)?)?,
                    NamedMatrix::new("objective", SXMatrix::scalar(objective))?,
                ],
            )
        })
    }

    fn eval_multiple_shooting_integrator_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x: &X,
        u: &U,
        dudt: &U,
        parameters: &P,
        dt: SX,
    ) -> Result<(X, U, SX), SxError> {
        match &library.multiple_shooting_integrator {
            Some(function) => {
                let outputs = function.call(&[
                    symbolic_column(x)?,
                    symbolic_column(u)?,
                    symbolic_column(dudt)?,
                    symbolic_column(parameters)?,
                    SXMatrix::dense_column(vec![dt])?,
                ])?;
                Ok((
                    unflatten_typed_output::<X>(&outputs[0])?,
                    unflatten_typed_output::<U>(&outputs[1])?,
                    outputs[2].scalar_expr()?,
                ))
            }
            None => rk4_integrate_symbolic(
                x,
                u,
                dudt,
                dt,
                RK4_SUBSTEPS,
                |x_eval, u_eval| self.eval_ode_symbolic(library, x_eval, u_eval, parameters),
                |x_eval, u_eval| {
                    self.eval_objective_lagrange_symbolic(library, x_eval, u_eval, dudt, parameters)
                },
            ),
        }
    }

    pub fn compile_jit(
        &self,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_options(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }

    pub fn compile_jit_with_symbolic_callback<CB>(
        &self,
        on_symbolic_ready: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_symbolic_callback(
            OcpCompileOptions::for_multiple_shooting(FunctionCompileOptions::from(
                LlvmOptimizationLevel::O3,
            )),
            on_symbolic_ready,
        )
    }

    pub fn compile_jit_with_progress_callback<CB>(
        &self,
        on_progress: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_opt_level_and_progress_callback(
            LlvmOptimizationLevel::O3,
            on_progress,
        )
    }

    pub fn compile_jit_with_opt_level(
        &self,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_options(FunctionCompileOptions::from(opt_level))
    }

    pub fn compile_jit_with_options(
        &self,
        options: FunctionCompileOptions,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options(OcpCompileOptions::for_multiple_shooting(options))
    }

    pub fn compile_jit_with_ocp_options(
        &self,
        options: OcpCompileOptions,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_symbolic_callback(options, |_| {})
    }

    pub fn compile_jit_with_opt_level_and_symbolic_callback<CB>(
        &self,
        opt_level: LlvmOptimizationLevel,
        mut on_symbolic_ready: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_opt_level_and_progress_callback(opt_level, |progress| {
            if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                on_symbolic_ready(metadata);
            }
        })
    }

    pub fn compile_jit_with_opt_level_and_progress_callback<CB>(
        &self,
        opt_level: LlvmOptimizationLevel,
        on_progress: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::from(opt_level),
            on_progress,
        )
    }

    pub fn compile_jit_with_options_and_progress_callback<CB>(
        &self,
        options: FunctionCompileOptions,
        on_progress: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::from(options),
            on_progress,
        )
    }

    pub fn compile_jit_with_ocp_options_and_progress_callback<CB>(
        &self,
        options: OcpCompileOptions,
        mut on_progress: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        validate_multiple_shooting::<N, RK4_SUBSTEPS>()?;
        let symbolic_library =
            self.build_multiple_shooting_symbolic_function_library(options.symbolic_functions)?;
        let symbolic_vars = symbolic_value::<MsVars<X, U, N>>("w")?;
        let symbolic_params = symbolic_value::<OcpParameters<P, Beq>>("runtime")?;
        let outputs =
            self.transcribe_multiple_shooting(&symbolic_vars, &symbolic_params, &symbolic_library)?;
        let promotion_plan = build_promotion_plan::<MsVars<X, U, N>, [C; N], Beq, Bineq>(
            &symbolic_vars,
            &outputs.inequalities,
        );
        let promotion_offsets =
            compile_promotion_offsets(&promotion_plan, &symbolic_params, options.function_options)?;
        let symbolic_library_for_nlp = symbolic_library.clone();
        let symbolic = symbolic_nlp::<
            MsVars<X, U, N>,
            OcpParameters<P, Beq>,
            MsEqualities<X, U, N>,
            MsIneq<C, Beq, Bineq, N>,
            _,
        >(self.name.clone(), |vars, params| {
            self.transcribe_multiple_shooting(vars, params, &symbolic_library_for_nlp)
                .expect("multiple shooting transcription should be infallible after validation")
        })?;
        let compiled = symbolic.compile_jit_with_options_and_symbolic_progress_callback(
            options.function_options,
            |progress| match progress {
                SymbolicCompileProgress::Stage(progress) => {
                    on_progress(OcpCompileProgress::SymbolicStage(progress));
                }
                SymbolicCompileProgress::Ready(metadata) => {
                    on_progress(OcpCompileProgress::SymbolicReady(metadata));
                }
            },
        )?;
        let nlp_compile_report = compiled.backend_compile_report_untyped();
        let nlp_jit_elapsed = nlp_compile_report
            .setup_profile
            .lowering
            .unwrap_or_default()
            + nlp_compile_report
                .setup_profile
                .llvm_jit
                .unwrap_or_default();
        on_progress(OcpCompileProgress::NlpKernelCompiled {
            elapsed: nlp_jit_elapsed,
            root_instructions: nlp_compile_report.stats.llvm_root_instructions_emitted,
            total_instructions: nlp_compile_report.stats.llvm_total_instructions_emitted,
        });
        let xdot_started = Instant::now();
        let xdot_helper = compile_xdot_helper::<X, U, P>(
            &*self.ode,
            symbolic_library.ode.as_ref(),
            options.function_options,
        )?;
        let xdot_helper_time = xdot_started.elapsed();
        let xdot_helper_root_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_root_instructions_emitted;
        let xdot_helper_total_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_total_instructions_emitted;
        on_progress(OcpCompileProgress::HelperCompiled {
            helper: OcpCompileHelperKind::Xdot,
            elapsed: xdot_helper_time,
            root_instructions: xdot_helper_root_instructions,
            total_instructions: xdot_helper_total_instructions,
        });
        let rk4_arc_started = Instant::now();
        let rk4_arc_helper = compile_multiple_shooting_arc_helper::<X, U, P, RK4_SUBSTEPS>(
            &*self.ode,
            &symbolic_library,
            options.function_options,
        )?;
        let multiple_shooting_arc_helper_time = rk4_arc_started.elapsed();
        let multiple_shooting_arc_helper_root_instructions = rk4_arc_helper
            .function
            .compile_report()
            .stats
            .llvm_root_instructions_emitted;
        let multiple_shooting_arc_helper_total_instructions = rk4_arc_helper
            .function
            .compile_report()
            .stats
            .llvm_total_instructions_emitted;
        on_progress(OcpCompileProgress::HelperCompiled {
            helper: OcpCompileHelperKind::MultipleShootingArc,
            elapsed: multiple_shooting_arc_helper_time,
            root_instructions: multiple_shooting_arc_helper_root_instructions,
            total_instructions: multiple_shooting_arc_helper_total_instructions,
        });
        Ok(CompiledMultipleShootingOcp {
            compiled,
            promotion_plan,
            promotion_offsets,
            xdot_helper,
            rk4_arc_helper,
            helper_compile_stats: OcpHelperCompileStats {
                xdot_helper_time: Some(xdot_helper_time),
                multiple_shooting_arc_helper_time: Some(multiple_shooting_arc_helper_time),
                xdot_helper_root_instructions: Some(xdot_helper_root_instructions),
                xdot_helper_total_instructions: Some(xdot_helper_total_instructions),
                multiple_shooting_arc_helper_root_instructions: Some(
                    multiple_shooting_arc_helper_root_instructions,
                ),
                multiple_shooting_arc_helper_total_instructions: Some(
                    multiple_shooting_arc_helper_total_instructions,
                ),
            },
            _marker: PhantomData,
        })
    }

    pub fn compile_jit_with_options_and_symbolic_callback<CB>(
        &self,
        options: FunctionCompileOptions,
        mut on_symbolic_ready: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::from(options),
            |progress| {
                if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                    on_symbolic_ready(metadata);
                }
            },
        )
    }

    pub fn compile_jit_with_ocp_options_and_symbolic_callback<CB>(
        &self,
        options: OcpCompileOptions,
        mut on_symbolic_ready: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        MsVars<X, U, N>:
            Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
        MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<
                SX,
                Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
                Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(options, |progress| {
            if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                on_symbolic_ready(metadata);
            }
        })
    }

    fn transcribe_multiple_shooting(
        &self,
        vars: &MsVars<X, U, N>,
        params: &OcpParameters<P, Beq>,
        symbolic_library: &OcpSymbolicFunctionLibrary,
    ) -> Result<SymbolicNlpOutputs<MsEqualities<X, U, N>, MsIneq<C, Beq, Bineq, N>>, SxError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    {
        let (x_mesh, u_mesh, dudt, tf) = vars;
        let (parameters, beq) = params;
        let step = *tf / (N as f64);
        let mut objective = SX::zero();
        let mut x_defects = std::array::from_fn(|_| {
            unflatten_value::<X, SX>(&vec![SX::zero(); X::LEN])
                .expect("zero state should unflatten")
        });
        let mut u_defects = std::array::from_fn(|_| {
            unflatten_value::<U, SX>(&vec![SX::zero(); U::LEN])
                .expect("zero control should unflatten")
        });
        let mut path = std::array::from_fn(|_| {
            self.eval_path_constraints_symbolic(
                symbolic_library,
                &x_mesh.nodes[0],
                &u_mesh.nodes[0],
                &dudt[0],
                parameters,
            )
            .expect("path constraint call should be infallible after validation")
        });

        for interval in 0..N {
            let x_start = &x_mesh.nodes[interval];
            let u_start = &u_mesh.nodes[interval];
            let dudt_interval = &dudt[interval];
            path[interval] = self.eval_path_constraints_symbolic(
                symbolic_library,
                x_start,
                u_start,
                dudt_interval,
                parameters,
            )?;
            let (x_end, u_end, q_end) = self.eval_multiple_shooting_integrator_symbolic(
                symbolic_library,
                x_start,
                u_start,
                dudt_interval,
                parameters,
                step,
            )?;
            objective += q_end;
            let x_next = if interval + 1 < N {
                x_mesh.nodes[interval + 1].clone()
            } else {
                x_mesh.terminal.clone()
            };
            let u_next = if interval + 1 < N {
                u_mesh.nodes[interval + 1].clone()
            } else {
                u_mesh.terminal.clone()
            };
            x_defects[interval] = subtract_vectorized(&x_next, &x_end)?;
            u_defects[interval] = subtract_vectorized(&u_next, &u_end)?;
        }

        let boundary_eq_values = self.eval_boundary_equalities_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            tf,
        )?;
        let boundary_eq_residual = subtract_vectorized(&boundary_eq_values, beq)?;
        let boundary_ineq = self.eval_boundary_inequalities_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            tf,
        )?;
        objective += self.eval_objective_mayer_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            tf,
        )?;
        Ok(SymbolicNlpOutputs {
            objective,
            equalities: (x_defects, u_defects),
            inequalities: (boundary_eq_residual, boundary_ineq, path),
        })
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const K: usize>
    Ocp<X, U, P, C, Beq, Bineq, DirectCollocation<N, K>>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
{
    pub fn compile_jit(
        &self,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_options(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }

    pub fn compile_jit_with_symbolic_callback<CB>(
        &self,
        on_symbolic_ready: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_symbolic_callback(
            OcpCompileOptions::for_direct_collocation(FunctionCompileOptions::from(
                LlvmOptimizationLevel::O3,
            )),
            on_symbolic_ready,
        )
    }

    pub fn compile_jit_with_progress_callback<CB>(
        &self,
        on_progress: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_opt_level_and_progress_callback(
            LlvmOptimizationLevel::O3,
            on_progress,
        )
    }

    pub fn compile_jit_with_opt_level(
        &self,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_options(FunctionCompileOptions::from(opt_level))
    }

    pub fn compile_jit_with_options(
        &self,
        options: FunctionCompileOptions,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options(OcpCompileOptions::for_direct_collocation(options))
    }

    pub fn compile_jit_with_ocp_options(
        &self,
        options: OcpCompileOptions,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_symbolic_callback(options, |_| {})
    }

    pub fn compile_jit_with_opt_level_and_symbolic_callback<CB>(
        &self,
        opt_level: LlvmOptimizationLevel,
        mut on_symbolic_ready: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_opt_level_and_progress_callback(opt_level, |progress| {
            if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                on_symbolic_ready(metadata);
            }
        })
    }

    pub fn compile_jit_with_opt_level_and_progress_callback<CB>(
        &self,
        opt_level: LlvmOptimizationLevel,
        on_progress: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::from(opt_level),
            on_progress,
        )
    }

    pub fn compile_jit_with_options_and_progress_callback<CB>(
        &self,
        options: FunctionCompileOptions,
        on_progress: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::from(options),
            on_progress,
        )
    }

    pub fn compile_jit_with_ocp_options_and_progress_callback<CB>(
        &self,
        options: OcpCompileOptions,
        mut on_progress: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        validate_direct_collocation::<N, K>()?;
        let symbolic_library = self.build_symbolic_function_library(options.symbolic_functions)?;
        let coefficients = collocation_coefficients(self.scheme.family, K)?;
        let symbolic_vars = symbolic_value::<DcVars<X, U, N, K>>("w")?;
        let symbolic_params = symbolic_value::<OcpParameters<P, Beq>>("runtime")?;
        let outputs = self.transcribe_direct_collocation(
            &symbolic_vars,
            &symbolic_params,
            &coefficients,
            &symbolic_library,
        )?;
        let promotion_plan =
            build_promotion_plan::<DcVars<X, U, N, K>, IntervalGrid<C, N, K>, Beq, Bineq>(
                &symbolic_vars,
                &outputs.inequalities,
            );
        let promotion_offsets =
            compile_promotion_offsets(&promotion_plan, &symbolic_params, options.function_options)?;
        let symbolic_library_for_nlp = symbolic_library.clone();
        let symbolic = symbolic_nlp::<
            DcVars<X, U, N, K>,
            OcpParameters<P, Beq>,
            DcEqualities<X, U, N, K>,
            DcIneq<C, Beq, Bineq, N, K>,
            _,
        >(self.name.clone(), |vars, params| {
            self.transcribe_direct_collocation(
                vars,
                params,
                &coefficients,
                &symbolic_library_for_nlp,
            )
            .expect("direct collocation transcription should be infallible after validation")
        })?;
        let compiled = symbolic.compile_jit_with_options_and_symbolic_progress_callback(
            options.function_options,
            |progress| match progress {
                SymbolicCompileProgress::Stage(progress) => {
                    on_progress(OcpCompileProgress::SymbolicStage(progress));
                }
                SymbolicCompileProgress::Ready(metadata) => {
                    on_progress(OcpCompileProgress::SymbolicReady(metadata));
                }
            },
        )?;
        let nlp_compile_report = compiled.backend_compile_report_untyped();
        let nlp_jit_elapsed = nlp_compile_report
            .setup_profile
            .lowering
            .unwrap_or_default()
            + nlp_compile_report
                .setup_profile
                .llvm_jit
                .unwrap_or_default();
        on_progress(OcpCompileProgress::NlpKernelCompiled {
            elapsed: nlp_jit_elapsed,
            root_instructions: nlp_compile_report.stats.llvm_root_instructions_emitted,
            total_instructions: nlp_compile_report.stats.llvm_total_instructions_emitted,
        });
        let xdot_started = Instant::now();
        let xdot_helper = compile_xdot_helper::<X, U, P>(
            &*self.ode,
            symbolic_library.ode.as_ref(),
            options.function_options,
        )?;
        let xdot_helper_time = xdot_started.elapsed();
        let xdot_helper_root_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_root_instructions_emitted;
        let xdot_helper_total_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_total_instructions_emitted;
        on_progress(OcpCompileProgress::HelperCompiled {
            helper: OcpCompileHelperKind::Xdot,
            elapsed: xdot_helper_time,
            root_instructions: xdot_helper_root_instructions,
            total_instructions: xdot_helper_total_instructions,
        });
        Ok(CompiledDirectCollocationOcp {
            compiled,
            promotion_plan,
            promotion_offsets,
            xdot_helper,
            coefficients,
            helper_compile_stats: OcpHelperCompileStats {
                xdot_helper_time: Some(xdot_helper_time),
                multiple_shooting_arc_helper_time: None,
                xdot_helper_root_instructions: Some(xdot_helper_root_instructions),
                xdot_helper_total_instructions: Some(xdot_helper_total_instructions),
                multiple_shooting_arc_helper_root_instructions: None,
                multiple_shooting_arc_helper_total_instructions: None,
            },
            _marker: PhantomData,
        })
    }

    pub fn compile_jit_with_options_and_symbolic_callback<CB>(
        &self,
        options: FunctionCompileOptions,
        mut on_symbolic_ready: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::from(options),
            |progress| {
                if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                    on_symbolic_ready(metadata);
                }
            },
        )
    }

    pub fn compile_jit_with_ocp_options_and_symbolic_callback<CB>(
        &self,
        options: OcpCompileOptions,
        mut on_symbolic_ready: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>, OcpCompileError>
    where
        CB: FnMut(SymbolicCompileMetadata),
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        DcVars<X, U, N, K>:
            Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
        DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<
                SX,
                Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
                Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
            >,
        OcpParameters<P, Beq>: Vectorize<
                SX,
                Rebind<SX> = OcpParameters<P, Beq>,
                Rebind<f64> = OcpParametersNum<P, Beq>,
            >,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
        OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
    {
        self.compile_jit_with_ocp_options_and_progress_callback(options, |progress| {
            if let OcpCompileProgress::SymbolicReady(metadata) = progress {
                on_symbolic_ready(metadata);
            }
        })
    }

    fn transcribe_direct_collocation(
        &self,
        vars: &DcVars<X, U, N, K>,
        params: &OcpParameters<P, Beq>,
        coeffs: &CollocationCoefficients,
        symbolic_library: &OcpSymbolicFunctionLibrary,
    ) -> Result<SymbolicNlpOutputs<DcEqualities<X, U, N, K>, DcIneq<C, Beq, Bineq, N, K>>, SxError>
    where
        Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
        Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
        IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
        IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
        IntervalGrid<C, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<C, N, K>>,
        [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
        [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    {
        let (x_mesh, u_mesh, root_x, root_u, root_dudt, tf) = vars;
        let (parameters, beq) = params;
        let step = *tf / (N as f64);

        let mut collocation_x = std::array::from_fn(|interval| {
            std::array::from_fn(|root| root_x.intervals[interval][root].clone())
        });
        let mut collocation_u = std::array::from_fn(|interval| {
            std::array::from_fn(|root| root_u.intervals[interval][root].clone())
        });
        let mut continuity_x = std::array::from_fn(|_| {
            unflatten_value::<X, SX>(&vec![SX::zero(); X::LEN])
                .expect("zero state should unflatten")
        });
        let mut continuity_u = std::array::from_fn(|_| {
            unflatten_value::<U, SX>(&vec![SX::zero(); U::LEN])
                .expect("zero control should unflatten")
        });
        let mut path = IntervalGrid {
            intervals: std::array::from_fn(|interval| {
                std::array::from_fn(|root| {
                    self.eval_path_constraints_symbolic(
                        symbolic_library,
                        &root_x.intervals[interval][root],
                        &root_u.intervals[interval][root],
                        &root_dudt.intervals[interval][root],
                        parameters,
                    )
                    .expect("path constraint call should be infallible after validation")
                })
            }),
        };
        let mut objective = SX::zero();

        for interval in 0..N {
            let x_start = if interval == 0 {
                x_mesh.nodes[0].clone()
            } else {
                x_mesh.nodes[interval].clone()
            };
            let u_start = if interval == 0 {
                u_mesh.nodes[0].clone()
            } else {
                u_mesh.nodes[interval].clone()
            };
            let mut basis_x = Vec::with_capacity(K + 1);
            basis_x.push(x_start.clone());
            basis_x.extend(root_x.intervals[interval].iter().cloned());
            let mut basis_u = Vec::with_capacity(K + 1);
            basis_u.push(u_start.clone());
            basis_u.extend(root_u.intervals[interval].iter().cloned());

            for root in 0..K {
                path.intervals[interval][root] = self.eval_path_constraints_symbolic(
                    symbolic_library,
                    &root_x.intervals[interval][root],
                    &root_u.intervals[interval][root],
                    &root_dudt.intervals[interval][root],
                    parameters,
                )?;
                let xdot = self.eval_ode_symbolic(
                    symbolic_library,
                    &root_x.intervals[interval][root],
                    &root_u.intervals[interval][root],
                    parameters,
                )?;
                let xpoly = weighted_sum_vectorized(&basis_x, &coeffs.c_matrix[root])?;
                let upoly = weighted_sum_vectorized(&basis_u, &coeffs.c_matrix[root])?;
                collocation_x[interval][root] =
                    subtract_vectorized(&scale_vectorized(&xdot, step)?, &xpoly)?;
                collocation_u[interval][root] = subtract_vectorized(
                    &scale_vectorized(&root_dudt.intervals[interval][root], step)?,
                    &upoly,
                )?;
                objective += step
                    * coeffs.b_vector[root + 1]
                    * self.eval_objective_lagrange_symbolic(
                        symbolic_library,
                        &root_x.intervals[interval][root],
                        &root_u.intervals[interval][root],
                        &root_dudt.intervals[interval][root],
                        parameters,
                    )?;
            }

            let x_end = weighted_sum_vectorized(&basis_x, &coeffs.d_vector)?;
            let u_end = weighted_sum_vectorized(&basis_u, &coeffs.d_vector)?;
            let x_next = if interval + 1 < N {
                x_mesh.nodes[interval + 1].clone()
            } else {
                x_mesh.terminal.clone()
            };
            let u_next = if interval + 1 < N {
                u_mesh.nodes[interval + 1].clone()
            } else {
                u_mesh.terminal.clone()
            };
            continuity_x[interval] = subtract_vectorized(&x_next, &x_end)?;
            continuity_u[interval] = subtract_vectorized(&u_next, &u_end)?;
        }

        let boundary_eq_values = self.eval_boundary_equalities_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            tf,
        )?;
        let boundary_eq_residual = subtract_vectorized(&boundary_eq_values, beq)?;
        let boundary_ineq = self.eval_boundary_inequalities_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            tf,
        )?;
        objective += self.eval_objective_mayer_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            tf,
        )?;

        Ok(SymbolicNlpOutputs {
            objective,
            equalities: (
                IntervalGrid {
                    intervals: collocation_x,
                },
                IntervalGrid {
                    intervals: collocation_u,
                },
                continuity_x,
                continuity_u,
            ),
            inequalities: (boundary_eq_residual, boundary_ineq, path),
        })
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const RK4_SUBSTEPS: usize>
    CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
    Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
    [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
    [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
    [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    MsVars<X, U, N>: Vectorize<SX, Rebind<SX> = MsVars<X, U, N>, Rebind<f64> = MsVarsNum<X, U, N>>,
    MsEqualities<X, U, N>: Vectorize<SX, Rebind<SX> = MsEqualities<X, U, N>>,
    <MsEqualities<X, U, N> as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    MsIneq<C, Beq, Bineq, N>: Vectorize<
            SX,
            Rebind<SX> = MsIneq<C, Beq, Bineq, N>,
            Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>,
        >,
    OcpParameters<P, Beq>:
        Vectorize<SX, Rebind<SX> = OcpParameters<P, Beq>, Rebind<f64> = OcpParametersNum<P, Beq>>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
    MsVarsNum<X, U, N>: Vectorize<f64, Rebind<f64> = MsVarsNum<X, U, N>>,
    MsIneqNum<C, Beq, Bineq, N>: Vectorize<f64, Rebind<f64> = MsIneqNum<C, Beq, Bineq, N>>,
    OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
{
    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.compiled.backend_timing_metadata()
    }

    pub fn nlp_compile_stats(&self) -> NlpCompileStats {
        self.compiled.compile_stats()
    }

    pub fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        self.helper_compile_stats
    }

    pub const fn helper_kernel_count(&self) -> usize {
        2
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        self.compiled.backend_compile_report()
    }

    #[doc(hidden)]
    pub fn debug_lagrangian_hessian_lowered(&self) -> &sx_codegen::LoweredFunction {
        self.compiled.debug_lagrangian_hessian_lowered()
    }

    pub fn benchmark_nlp_evaluations(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: NlpEvaluationBenchmarkOptions,
    ) -> AnyResult<NlpEvaluationBenchmark> {
        self.benchmark_nlp_evaluations_with_progress(values, options, |_| {})
    }

    pub fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> AnyResult<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let x0 = self.build_initial_guess(values)?;
        let bounds = self.build_runtime_bounds(values)?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        Ok(self.compiled.benchmark_bounded_evaluations_with_progress(
            &x0,
            &runtime_params,
            &bounds,
            options,
            on_progress,
        )?)
    }

    pub fn validate_nlp_derivatives(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let x0 = self.build_initial_guess(values)?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let x_values = flatten_value(&x0);
        let param_values = flatten_value(&runtime_params);
        Ok(self.compiled.validate_derivatives_flat_values(
            &x_values,
            &param_values,
            equality_multipliers,
            inequality_multipliers,
            options,
        )?)
    }

    pub fn solve_sqp(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<MultipleShootingSqpSolveResult<Numeric<X>, Numeric<U>, N>, ClarabelSqpError> {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self
            .compiled
            .solve_sqp(&x0, &runtime_params, &bounds, options)?;
        let trajectories = project_multiple_shooting::<X, U, N>(&summary.x)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        Ok(MultipleShootingSqpSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: &ClarabelSqpOptions,
        mut callback: CB,
    ) -> Result<MultipleShootingSqpSolveResult<Numeric<X>, Numeric<U>, N>, ClarabelSqpError>
    where
        CB: FnMut(&MultipleShootingSqpSnapshot<Numeric<X>, Numeric<U>, N>),
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self.compiled.solve_sqp_with_callback(
            &x0,
            &runtime_params,
            &bounds,
            options,
            |snapshot| {
                let trajectories = project_multiple_shooting::<X, U, N>(&snapshot.x)
                    .expect("solver iterate should match OCP variable layout");
                callback(&MultipleShootingSqpSnapshot {
                    time_grid: MultipleShootingTimeGrid {
                        nodes: mesh_times::<N>(trajectories.tf),
                    },
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_multiple_shooting::<X, U, N>(&summary.x)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        Ok(MultipleShootingSqpSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn solve_interior_point(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: &InteriorPointOptions,
    ) -> Result<
        MultipleShootingInteriorPointSolveResult<Numeric<X>, Numeric<U>, N>,
        InteriorPointSolveError,
    > {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self
            .compiled
            .solve_interior_point(&x0, &runtime_params, &bounds, options)?;
        let trajectories = project_multiple_shooting::<X, U, N>(&summary.x)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        Ok(MultipleShootingInteriorPointSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: &InteriorPointOptions,
        mut callback: CB,
    ) -> Result<
        MultipleShootingInteriorPointSolveResult<Numeric<X>, Numeric<U>, N>,
        InteriorPointSolveError,
    >
    where
        CB: FnMut(&MultipleShootingInteriorPointSnapshot<Numeric<X>, Numeric<U>, N>),
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self.compiled.solve_interior_point_with_callback(
            &x0,
            &runtime_params,
            &bounds,
            options,
            |snapshot| {
                let trajectories = project_multiple_shooting::<X, U, N>(&snapshot.x)
                    .expect("solver iterate should match OCP variable layout");
                callback(&MultipleShootingInteriorPointSnapshot {
                    time_grid: MultipleShootingTimeGrid {
                        nodes: mesh_times::<N>(trajectories.tf),
                    },
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_multiple_shooting::<X, U, N>(&summary.x)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        Ok(MultipleShootingInteriorPointSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: &IpoptOptions,
    ) -> Result<MultipleShootingIpoptSolveResult<Numeric<X>, Numeric<U>, N>, IpoptSolveError> {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self
            .compiled
            .solve_ipopt(&x0, &runtime_params, &bounds, options)?;
        let trajectories = project_multiple_shooting::<X, U, N>(&summary.x)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        Ok(MultipleShootingIpoptSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        options: &IpoptOptions,
        mut callback: CB,
    ) -> Result<MultipleShootingIpoptSolveResult<Numeric<X>, Numeric<U>, N>, IpoptSolveError>
    where
        CB: FnMut(&MultipleShootingIpoptSnapshot<Numeric<X>, Numeric<U>, N>),
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self.compiled.solve_ipopt_with_callback(
            &x0,
            &runtime_params,
            &bounds,
            options,
            |snapshot| {
                if let Ok(trajectories) = project_multiple_shooting::<X, U, N>(&snapshot.x) {
                    callback(&MultipleShootingIpoptSnapshot {
                        time_grid: MultipleShootingTimeGrid {
                            nodes: mesh_times::<N>(trajectories.tf),
                        },
                        trajectories,
                        solver: snapshot.clone(),
                    });
                }
            },
        )?;
        let trajectories = project_multiple_shooting::<X, U, N>(&summary.x)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        Ok(MultipleShootingIpoptSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn interval_arcs(
        &self,
        trajectories: &MultipleShootingTrajectories<Numeric<X>, Numeric<U>, N>,
        parameters: &Numeric<P>,
    ) -> AnyResult<(Vec<IntervalArc<Numeric<X>>>, Vec<IntervalArc<Numeric<U>>>)> {
        let step = trajectories.tf / (N as f64);
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times::<N>(trajectories.tf),
        };
        let mut x_arcs = Vec::with_capacity(N);
        let mut u_arcs = Vec::with_capacity(N);
        for interval in 0..N {
            let start_x = &trajectories.x.nodes[interval];
            let start_u = &trajectories.u.nodes[interval];
            let rate = &trajectories.dudt[interval];
            let start_time = time_grid.nodes.nodes[interval];
            let mut times = Vec::with_capacity(MULTIPLE_SHOOTING_ARC_SAMPLES + 1);
            times.push(start_time);
            times.extend((0..MULTIPLE_SHOOTING_ARC_SAMPLES).map(|sample| {
                start_time + ((sample + 1) as f64 / MULTIPLE_SHOOTING_ARC_SAMPLES as f64) * step
            }));

            let mut x_values = Vec::with_capacity(MULTIPLE_SHOOTING_ARC_SAMPLES + 1);
            x_values.push(start_x.clone());
            let mut u_values = Vec::with_capacity(MULTIPLE_SHOOTING_ARC_SAMPLES + 1);
            u_values.push(start_u.clone());
            for sample in 0..MULTIPLE_SHOOTING_ARC_SAMPLES {
                let fraction = (sample + 1) as f64 / (MULTIPLE_SHOOTING_ARC_SAMPLES as f64);
                let (x_sample, u_sample) = self.rk4_arc_helper.eval(
                    start_x,
                    start_u,
                    rate,
                    parameters,
                    fraction * step,
                )?;
                x_values.push(x_sample);
                u_values.push(u_sample);
            }

            x_arcs.push(IntervalArc {
                times: times.clone(),
                values: x_values,
            });
            u_arcs.push(IntervalArc {
                times,
                values: u_values,
            });
        }
        Ok((x_arcs, u_arcs))
    }

    pub fn rank_constraint_violations(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
        trajectories: &MultipleShootingTrajectories<Numeric<X>, Numeric<U>, N>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport, VectorizeLayoutError> {
        let decision: MsVarsNum<X, U, N> = (
            trajectories.x.clone(),
            trajectories.u.clone(),
            trajectories.dudt.clone(),
            trajectories.tf,
        );
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let equality_values = self
            .compiled
            .evaluate_equalities_flat(&decision, &runtime_params);
        let inequality_values = self
            .compiled
            .evaluate_inequalities_flat(&decision, &runtime_params);

        let mut equality_groups = HashMap::new();
        let mut inequality_groups = HashMap::new();

        let continuity_x_labels = prefixed_leaf_names::<X>("continuity.x");
        let continuity_u_labels = prefixed_leaf_names::<U>("continuity.u");
        let boundary_eq_labels = prefixed_leaf_names::<Beq>("boundary_eq");
        let boundary_ineq_labels = prefixed_leaf_names::<Bineq>("boundary_ineq");
        let path_labels = prefixed_leaf_names::<C>("path");

        let continuity_x_count = N * X::LEN;
        let continuity_u_count = N * U::LEN;
        let boundary_eq_count = Beq::LEN;
        let boundary_ineq_count = Bineq::LEN;
        let continuity_x_values = &equality_values[..continuity_x_count];
        let continuity_u_values =
            &equality_values[continuity_x_count..continuity_x_count + continuity_u_count];
        let boundary_eq_values = &inequality_values[..boundary_eq_count];
        let boundary_ineq_values =
            &inequality_values[boundary_eq_count..boundary_eq_count + boundary_ineq_count];
        let path_values = &inequality_values[boundary_eq_count + boundary_ineq_count..];

        add_repeated_equalities(
            &mut equality_groups,
            &continuity_x_values,
            &continuity_x_labels,
            OcpConstraintCategory::ContinuityState,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            &continuity_u_values,
            &continuity_u_labels,
            OcpConstraintCategory::ContinuityControl,
            tolerance,
        );
        for (value, label) in boundary_eq_values.iter().zip(boundary_eq_labels.iter()) {
            accumulate_equality_group(
                &mut equality_groups,
                label,
                OcpConstraintCategory::BoundaryEquality,
                *value,
                tolerance,
            );
        }

        let boundary_ineq_bounds = flatten_bounds(&values.bineq_bounds);
        let path_bounds = flatten_bounds(&values.path_bounds);
        add_repeated_inequalities(
            &mut inequality_groups,
            &boundary_ineq_values,
            &boundary_ineq_labels,
            &boundary_ineq_bounds,
            OcpConstraintCategory::BoundaryInequality,
            tolerance,
        );
        add_repeated_inequalities(
            &mut inequality_groups,
            &path_values,
            &path_labels,
            &path_bounds,
            OcpConstraintCategory::Path,
            tolerance,
        );
        accumulate_inequality_group(
            &mut inequality_groups,
            "T",
            OcpConstraintCategory::FinalTime,
            trajectories.tf,
            values.tf_bounds.lower,
            values.tf_bounds.upper,
            tolerance,
        );

        let mut report = OcpConstraintViolationReport {
            equalities: equality_groups_from_map(equality_groups, tolerance),
            inequalities: inequality_groups_from_map(inequality_groups, tolerance),
        };
        sort_ocp_constraint_report(&mut report);
        Ok(report)
    }

    fn build_initial_guess(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
    ) -> Result<MsVarsNum<X, U, N>, GuessError>
    where
        MsVars<X, U, N>: Vectorize<SX>,
    {
        match &values.initial_guess {
            MultipleShootingInitialGuess::Explicit(guess) => {
                let guess_value: MsVarsNum<X, U, N> = (
                    guess.x.clone(),
                    guess.u.clone(),
                    guess.dudt.clone(),
                    guess.tf,
                );
                Ok(guess_value)
            }
            MultipleShootingInitialGuess::Constant { x, u, dudt, tf } => {
                let guess_value: MsVarsNum<X, U, N> = (
                    Mesh {
                        nodes: std::array::from_fn(|_| x.clone()),
                        terminal: x.clone(),
                    },
                    Mesh {
                        nodes: std::array::from_fn(|_| u.clone()),
                        terminal: u.clone(),
                    },
                    std::array::from_fn(|_| dudt.clone()),
                    *tf,
                );
                Ok(guess_value)
            }
            MultipleShootingInitialGuess::Interpolated(samples) => {
                build_multiple_shooting_interpolated_guess::<Numeric<X>, Numeric<U>, N>(samples)
            }
            MultipleShootingInitialGuess::Rollout {
                x0,
                u0,
                tf,
                controller,
            } => build_multiple_shooting_rollout_guess::<X, U, P, N, RK4_SUBSTEPS>(
                &self.xdot_helper,
                x0,
                u0,
                *tf,
                &values.parameters,
                controller.as_ref(),
            ),
        }
    }

    fn build_runtime_bounds(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
        >,
    ) -> Result<TypedRuntimeNlpBounds<MsVars<X, U, N>, MsIneq<C, Beq, Bineq, N>>, GuessError>
    where
        MsVars<X, U, N>: Vectorize<SX>,
        MsIneq<C, Beq, Bineq, N>: Vectorize<SX>,
        OcpParameters<P, Beq>: Vectorize<SX>,
    {
        let param_values: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let offsets = self.promotion_offsets.eval(&param_values)?;
        let (lower, upper) = build_raw_bounds::<C, Beq, Bineq>(
            &self.promotion_plan,
            &offsets,
            &values.path_bounds,
            &values.bineq_bounds,
            values.tf_bounds.clone(),
            MsVars::<X, U, N>::LEN,
        )?;
        Ok(TypedRuntimeNlpBounds {
            variable_lower: Some(unflatten_value::<MsVarsNum<X, U, N>, f64>(&lower)?),
            variable_upper: Some(unflatten_value::<MsVarsNum<X, U, N>, f64>(&upper)?),
            inequality_lower: Some(unflatten_value::<MsIneqNum<C, Beq, Bineq, N>, f64>(
                &build_inequality_lower::<C, Beq, Bineq>(
                    &self.promotion_plan,
                    &offsets,
                    &values.path_bounds,
                    &values.bineq_bounds,
                )?,
            )?),
            inequality_upper: Some(unflatten_value::<MsIneqNum<C, Beq, Bineq, N>, f64>(
                &build_inequality_upper::<C, Beq, Bineq>(
                    &self.promotion_plan,
                    &offsets,
                    &values.path_bounds,
                    &values.bineq_bounds,
                )?,
            )?),
        })
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const K: usize>
    CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Mesh<X, N>: Vectorize<SX, Rebind<SX> = Mesh<X, N>>,
    Mesh<U, N>: Vectorize<SX, Rebind<SX> = Mesh<U, N>>,
    IntervalGrid<X, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<X, N, K>>,
    IntervalGrid<U, N, K>: Vectorize<SX, Rebind<SX> = IntervalGrid<U, N, K>>,
    DcVars<X, U, N, K>:
        Vectorize<SX, Rebind<SX> = DcVars<X, U, N, K>, Rebind<f64> = DcVarsNum<X, U, N, K>>,
    DcEqualities<X, U, N, K>: Vectorize<SX, Rebind<SX> = DcEqualities<X, U, N, K>>,
    <DcEqualities<X, U, N, K> as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    DcIneq<C, Beq, Bineq, N, K>: Vectorize<
            SX,
            Rebind<SX> = DcIneq<C, Beq, Bineq, N, K>,
            Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>,
        >,
    OcpParameters<P, Beq>:
        Vectorize<SX, Rebind<SX> = OcpParameters<P, Beq>, Rebind<f64> = OcpParametersNum<P, Beq>>,
    [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
    [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
    DcVarsNum<X, U, N, K>: Vectorize<f64, Rebind<f64> = DcVarsNum<X, U, N, K>>,
    DcIneqNum<C, Beq, Bineq, N, K>: Vectorize<f64, Rebind<f64> = DcIneqNum<C, Beq, Bineq, N, K>>,
    OcpParametersNum<P, Beq>: Vectorize<f64, Rebind<f64> = OcpParametersNum<P, Beq>>,
{
    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.compiled.backend_timing_metadata()
    }

    pub fn nlp_compile_stats(&self) -> NlpCompileStats {
        self.compiled.compile_stats()
    }

    pub fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        self.helper_compile_stats
    }

    pub const fn helper_kernel_count(&self) -> usize {
        1
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        self.compiled.backend_compile_report()
    }

    #[doc(hidden)]
    pub fn debug_lagrangian_hessian_lowered(&self) -> &sx_codegen::LoweredFunction {
        self.compiled.debug_lagrangian_hessian_lowered()
    }

    pub fn benchmark_nlp_evaluations(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: NlpEvaluationBenchmarkOptions,
    ) -> AnyResult<NlpEvaluationBenchmark> {
        self.benchmark_nlp_evaluations_with_progress(values, options, |_| {})
    }

    pub fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> AnyResult<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let x0 = self.build_initial_guess(values)?;
        let bounds = self.build_runtime_bounds(values)?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        Ok(self.compiled.benchmark_bounded_evaluations_with_progress(
            &x0,
            &runtime_params,
            &bounds,
            options,
            on_progress,
        )?)
    }

    pub fn validate_nlp_derivatives(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let x0 = self.build_initial_guess(values)?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let x_values = flatten_value(&x0);
        let param_values = flatten_value(&runtime_params);
        Ok(self.compiled.validate_derivatives_flat_values(
            &x_values,
            &param_values,
            equality_multipliers,
            inequality_multipliers,
            options,
        )?)
    }

    pub fn solve_sqp(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<DirectCollocationSqpSolveResult<Numeric<X>, Numeric<U>, N, K>, ClarabelSqpError>
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self
            .compiled
            .solve_sqp(&x0, &runtime_params, &bounds, options)?;
        let trajectories = project_direct_collocation::<X, U, N, K>(&summary.x)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let time_grid = direct_collocation_times::<N, K>(trajectories.tf, &self.coefficients);
        Ok(DirectCollocationSqpSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: &ClarabelSqpOptions,
        mut callback: CB,
    ) -> Result<DirectCollocationSqpSolveResult<Numeric<X>, Numeric<U>, N, K>, ClarabelSqpError>
    where
        CB: FnMut(&DirectCollocationSqpSnapshot<Numeric<X>, Numeric<U>, N, K>),
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self.compiled.solve_sqp_with_callback(
            &x0,
            &runtime_params,
            &bounds,
            options,
            |snapshot| {
                let trajectories = project_direct_collocation::<X, U, N, K>(&snapshot.x)
                    .expect("solver iterate should match OCP variable layout");
                callback(&DirectCollocationSqpSnapshot {
                    time_grid: direct_collocation_times::<N, K>(
                        trajectories.tf,
                        &self.coefficients,
                    ),
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_direct_collocation::<X, U, N, K>(&summary.x)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let time_grid = direct_collocation_times::<N, K>(trajectories.tf, &self.coefficients);
        Ok(DirectCollocationSqpSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn solve_interior_point(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: &InteriorPointOptions,
    ) -> Result<
        DirectCollocationInteriorPointSolveResult<Numeric<X>, Numeric<U>, N, K>,
        InteriorPointSolveError,
    > {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self
            .compiled
            .solve_interior_point(&x0, &runtime_params, &bounds, options)?;
        let trajectories = project_direct_collocation::<X, U, N, K>(&summary.x)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let time_grid = direct_collocation_times::<N, K>(trajectories.tf, &self.coefficients);
        Ok(DirectCollocationInteriorPointSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: &InteriorPointOptions,
        mut callback: CB,
    ) -> Result<
        DirectCollocationInteriorPointSolveResult<Numeric<X>, Numeric<U>, N, K>,
        InteriorPointSolveError,
    >
    where
        CB: FnMut(&DirectCollocationInteriorPointSnapshot<Numeric<X>, Numeric<U>, N, K>),
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self.compiled.solve_interior_point_with_callback(
            &x0,
            &runtime_params,
            &bounds,
            options,
            |snapshot| {
                let trajectories = project_direct_collocation::<X, U, N, K>(&snapshot.x)
                    .expect("solver iterate should match OCP variable layout");
                callback(&DirectCollocationInteriorPointSnapshot {
                    time_grid: direct_collocation_times::<N, K>(
                        trajectories.tf,
                        &self.coefficients,
                    ),
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_direct_collocation::<X, U, N, K>(&summary.x)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let time_grid = direct_collocation_times::<N, K>(trajectories.tf, &self.coefficients);
        Ok(DirectCollocationInteriorPointSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: &IpoptOptions,
    ) -> Result<DirectCollocationIpoptSolveResult<Numeric<X>, Numeric<U>, N, K>, IpoptSolveError>
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self
            .compiled
            .solve_ipopt(&x0, &runtime_params, &bounds, options)?;
        let trajectories = project_direct_collocation::<X, U, N, K>(&summary.x)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let time_grid = direct_collocation_times::<N, K>(trajectories.tf, &self.coefficients);
        Ok(DirectCollocationIpoptSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        options: &IpoptOptions,
        mut callback: CB,
    ) -> Result<DirectCollocationIpoptSolveResult<Numeric<X>, Numeric<U>, N, K>, IpoptSolveError>
    where
        CB: FnMut(&DirectCollocationIpoptSnapshot<Numeric<X>, Numeric<U>, N, K>),
    {
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let bounds = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let summary = self.compiled.solve_ipopt_with_callback(
            &x0,
            &runtime_params,
            &bounds,
            options,
            |snapshot| {
                if let Ok(trajectories) = project_direct_collocation::<X, U, N, K>(&snapshot.x) {
                    callback(&DirectCollocationIpoptSnapshot {
                        time_grid: direct_collocation_times::<N, K>(
                            trajectories.tf,
                            &self.coefficients,
                        ),
                        trajectories,
                        solver: snapshot.clone(),
                    });
                }
            },
        )?;
        let trajectories = project_direct_collocation::<X, U, N, K>(&summary.x)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let time_grid = direct_collocation_times::<N, K>(trajectories.tf, &self.coefficients);
        Ok(DirectCollocationIpoptSolveResult {
            trajectories,
            time_grid,
            solver: summary,
        })
    }

    pub fn rank_constraint_violations(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
        trajectories: &DirectCollocationTrajectories<Numeric<X>, Numeric<U>, N, K>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport, VectorizeLayoutError> {
        let decision: DcVarsNum<X, U, N, K> = (
            trajectories.x.clone(),
            trajectories.u.clone(),
            trajectories.root_x.clone(),
            trajectories.root_u.clone(),
            trajectories.root_dudt.clone(),
            trajectories.tf,
        );
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let equality_values = self
            .compiled
            .evaluate_equalities_flat(&decision, &runtime_params);
        let inequality_values = self
            .compiled
            .evaluate_inequalities_flat(&decision, &runtime_params);

        let mut equality_groups = HashMap::new();
        let mut inequality_groups = HashMap::new();

        let collocation_x_labels = prefixed_leaf_names::<X>("collocation.x");
        let collocation_u_labels = prefixed_leaf_names::<U>("collocation.u");
        let continuity_x_labels = prefixed_leaf_names::<X>("continuity.x");
        let continuity_u_labels = prefixed_leaf_names::<U>("continuity.u");
        let boundary_eq_labels = prefixed_leaf_names::<Beq>("boundary_eq");
        let boundary_ineq_labels = prefixed_leaf_names::<Bineq>("boundary_ineq");
        let path_labels = prefixed_leaf_names::<C>("path");

        let collocation_x_count = N * K * X::LEN;
        let collocation_u_count = N * K * U::LEN;
        let continuity_x_count = N * X::LEN;
        let continuity_u_count = N * U::LEN;
        let boundary_eq_count = Beq::LEN;
        let boundary_ineq_count = Bineq::LEN;
        let collocation_x_values = &equality_values[..collocation_x_count];
        let collocation_u_values =
            &equality_values[collocation_x_count..collocation_x_count + collocation_u_count];
        let continuity_x_values = &equality_values[collocation_x_count + collocation_u_count
            ..collocation_x_count + collocation_u_count + continuity_x_count];
        let continuity_u_values = &equality_values[collocation_x_count
            + collocation_u_count
            + continuity_x_count
            ..collocation_x_count + collocation_u_count + continuity_x_count + continuity_u_count];
        let boundary_eq_values = &inequality_values[..boundary_eq_count];
        let boundary_ineq_values =
            &inequality_values[boundary_eq_count..boundary_eq_count + boundary_ineq_count];
        let path_values = &inequality_values[boundary_eq_count + boundary_ineq_count..];

        add_repeated_equalities(
            &mut equality_groups,
            &collocation_x_values,
            &collocation_x_labels,
            OcpConstraintCategory::CollocationState,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            &collocation_u_values,
            &collocation_u_labels,
            OcpConstraintCategory::CollocationControl,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            &continuity_x_values,
            &continuity_x_labels,
            OcpConstraintCategory::ContinuityState,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            &continuity_u_values,
            &continuity_u_labels,
            OcpConstraintCategory::ContinuityControl,
            tolerance,
        );
        for (value, label) in boundary_eq_values.iter().zip(boundary_eq_labels.iter()) {
            accumulate_equality_group(
                &mut equality_groups,
                label,
                OcpConstraintCategory::BoundaryEquality,
                *value,
                tolerance,
            );
        }

        let boundary_ineq_bounds = flatten_bounds(&values.bineq_bounds);
        let path_bounds = flatten_bounds(&values.path_bounds);
        add_repeated_inequalities(
            &mut inequality_groups,
            &boundary_ineq_values,
            &boundary_ineq_labels,
            &boundary_ineq_bounds,
            OcpConstraintCategory::BoundaryInequality,
            tolerance,
        );
        add_repeated_inequalities(
            &mut inequality_groups,
            &path_values,
            &path_labels,
            &path_bounds,
            OcpConstraintCategory::Path,
            tolerance,
        );
        accumulate_inequality_group(
            &mut inequality_groups,
            "T",
            OcpConstraintCategory::FinalTime,
            trajectories.tf,
            values.tf_bounds.lower,
            values.tf_bounds.upper,
            tolerance,
        );

        let mut report = OcpConstraintViolationReport {
            equalities: equality_groups_from_map(equality_groups, tolerance),
            inequalities: inequality_groups_from_map(inequality_groups, tolerance),
        };
        sort_ocp_constraint_report(&mut report);
        Ok(report)
    }

    fn build_initial_guess(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
    ) -> Result<DcVarsNum<X, U, N, K>, GuessError>
    where
        DcVars<X, U, N, K>: Vectorize<SX>,
    {
        match &values.initial_guess {
            DirectCollocationInitialGuess::Explicit(guess) => {
                let guess_value: DcVarsNum<X, U, N, K> = (
                    guess.x.clone(),
                    guess.u.clone(),
                    guess.root_x.clone(),
                    guess.root_u.clone(),
                    guess.root_dudt.clone(),
                    guess.tf,
                );
                Ok(guess_value)
            }
            DirectCollocationInitialGuess::Constant { x, u, dudt, tf } => {
                let guess_value: DcVarsNum<X, U, N, K> = (
                    Mesh {
                        nodes: std::array::from_fn(|_| x.clone()),
                        terminal: x.clone(),
                    },
                    Mesh {
                        nodes: std::array::from_fn(|_| u.clone()),
                        terminal: u.clone(),
                    },
                    IntervalGrid {
                        intervals: std::array::from_fn(|_| std::array::from_fn(|_| x.clone())),
                    },
                    IntervalGrid {
                        intervals: std::array::from_fn(|_| std::array::from_fn(|_| u.clone())),
                    },
                    IntervalGrid {
                        intervals: std::array::from_fn(|_| std::array::from_fn(|_| dudt.clone())),
                    },
                    *tf,
                );
                Ok(guess_value)
            }
            DirectCollocationInitialGuess::Interpolated(samples) => {
                build_direct_collocation_interpolated_guess::<Numeric<X>, Numeric<U>, N, K>(
                    samples,
                    &self.coefficients,
                )
            }
            DirectCollocationInitialGuess::Rollout {
                x0,
                u0,
                tf,
                controller,
            } => build_direct_collocation_rollout_guess::<X, U, P, N, K>(
                &self.xdot_helper,
                x0,
                u0,
                *tf,
                &values.parameters,
                controller.as_ref(),
                &self.coefficients,
            ),
        }
    }

    fn build_runtime_bounds(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            N,
            K,
        >,
    ) -> Result<TypedRuntimeNlpBounds<DcVars<X, U, N, K>, DcIneq<C, Beq, Bineq, N, K>>, GuessError>
    where
        DcVars<X, U, N, K>: Vectorize<SX>,
        DcIneq<C, Beq, Bineq, N, K>: Vectorize<SX>,
        OcpParameters<P, Beq>: Vectorize<SX>,
    {
        let param_values: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let offsets = self.promotion_offsets.eval(&param_values)?;
        let (lower, upper) = build_raw_bounds::<C, Beq, Bineq>(
            &self.promotion_plan,
            &offsets,
            &values.path_bounds,
            &values.bineq_bounds,
            values.tf_bounds.clone(),
            DcVars::<X, U, N, K>::LEN,
        )?;
        Ok(TypedRuntimeNlpBounds {
            variable_lower: Some(unflatten_value::<DcVarsNum<X, U, N, K>, f64>(&lower)?),
            variable_upper: Some(unflatten_value::<DcVarsNum<X, U, N, K>, f64>(&upper)?),
            inequality_lower: Some(unflatten_value::<DcIneqNum<C, Beq, Bineq, N, K>, f64>(
                &build_inequality_lower::<C, Beq, Bineq>(
                    &self.promotion_plan,
                    &offsets,
                    &values.path_bounds,
                    &values.bineq_bounds,
                )?,
            )?),
            inequality_upper: Some(unflatten_value::<DcIneqNum<C, Beq, Bineq, N, K>, f64>(
                &build_inequality_upper::<C, Beq, Bineq>(
                    &self.promotion_plan,
                    &offsets,
                    &values.path_bounds,
                    &values.bineq_bounds,
                )?,
            )?),
        })
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

impl<X, U, P, const RK4_SUBSTEPS: usize> CompiledMultipleShootingArc<X, U, P, RK4_SUBSTEPS>
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
        dudt: &Numeric<U>,
        parameters: &Numeric<P>,
        dt: f64,
    ) -> AnyResult<MsArcSampleOutputNum<X, U>> {
        let flat_x = flatten_value(x);
        let flat_u = flatten_value(u);
        let flat_dudt = flatten_value(dudt);
        let flat_parameters = flatten_value(parameters);
        let mut context = lock_mutex(&self.context);
        context.input_mut(0).copy_from_slice(&flat_x);
        context.input_mut(1).copy_from_slice(&flat_u);
        context.input_mut(2).copy_from_slice(&flat_dudt);
        if P::LEN > 0 {
            context.input_mut(3).copy_from_slice(&flat_parameters);
        }
        context.input_mut(3 + usize::from(P::LEN > 0))[0] = dt;
        self.function.eval(&mut context);
        Ok((
            unflatten_value::<Numeric<X>, f64>(context.output(0))?,
            unflatten_value::<Numeric<U>, f64>(context.output(1))?,
        ))
    }
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

fn validate_multiple_shooting<const N: usize, const RK4_SUBSTEPS: usize>()
-> Result<(), OcpCompileError> {
    if N == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "multiple shooting requires at least one interval".to_string(),
        ));
    }
    if RK4_SUBSTEPS == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "multiple shooting requires at least one RK4 substep".to_string(),
        ));
    }
    Ok(())
}

fn validate_direct_collocation<const N: usize, const K: usize>() -> Result<(), OcpCompileError> {
    if N == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation requires at least one interval".to_string(),
        ));
    }
    if K == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation requires at least one collocation root".to_string(),
        ));
    }
    Ok(())
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

fn compile_multiple_shooting_arc_helper<X, U, P, const RK4_SUBSTEPS: usize>(
    ode: &OdeFn<X, U, P>,
    _symbolic_library: &OcpSymbolicFunctionLibrary,
    options: FunctionCompileOptions,
) -> Result<CompiledMultipleShootingArc<X, U, P, RK4_SUBSTEPS>, OcpCompileError>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    // This helper is only used for post-solve/callback arc reconstruction, not for the
    // compiled NLP itself, so prioritize JIT latency over per-call throughput.
    let helper_options = FunctionCompileOptions {
        opt_level: match options.opt_level {
            LlvmOptimizationLevel::O3 => LlvmOptimizationLevel::O0,
            other => other,
        },
        ..options
    };
    let x = symbolic_value::<X>("x")?;
    let u = symbolic_value::<U>("u")?;
    let dudt = symbolic_value::<U>("dudt")?;
    let p = symbolic_value::<P>("p")?;
    let dt = SX::sym("dt");
    let outputs =
        rk4_integrate_symbolic_state_only(&x, &u, &dudt, dt, RK4_SUBSTEPS, |x_eval, u_eval| {
            Ok(ode(x_eval, u_eval, &p))
        })
        .expect("symbolic RK4 arc generation should be infallible");
    let mut inputs = vec![
        NamedMatrix::new("x", symbolic_column(&x)?)?,
        NamedMatrix::new("u", symbolic_column(&u)?)?,
        NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
    ];
    if P::LEN > 0 {
        inputs.push(NamedMatrix::new("p", symbolic_column(&p)?)?);
    }
    inputs.push(NamedMatrix::new("dt", SXMatrix::dense_column(vec![dt])?)?);
    let function = SXFunction::new(
        "ocp_multiple_shooting_arc",
        inputs,
        vec![
            NamedMatrix::new("x_next", symbolic_column(&outputs.0)?)?,
            NamedMatrix::new("u_next", symbolic_column(&outputs.1)?)?,
        ],
    )?;
    let compiled = CompiledJitFunction::compile_function_with_options(&function, helper_options)?;
    let context = Mutex::new(compiled.create_context());
    Ok(CompiledMultipleShootingArc {
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

fn build_promotion_plan<Vars, Path, Beq, Bineq>(
    variables: &Vars,
    inequalities: &(Beq, Bineq, Path),
) -> PromotionPlan
where
    Vars: Vectorize<SX>,
    Beq: Vectorize<SX>,
    Bineq: Vectorize<SX>,
    Path: Vectorize<SX>,
{
    let decision_symbols = variables.flatten_cloned();
    let decision_map = decision_symbols
        .iter()
        .copied()
        .enumerate()
        .map(|(index, symbol)| (symbol, index))
        .collect::<HashMap<_, _>>();
    let decision_set = decision_symbols.iter().copied().collect::<HashSet<_>>();
    let mut affine_memo = HashMap::new();

    let mut rows = Vec::new();
    let boundary_eq = inequalities.0.flatten_cloned();
    let boundary_ineq = inequalities.1.flatten_cloned();
    let path = inequalities.2.flatten_cloned();

    for expr in boundary_eq {
        rows.push(RawInequalityRow {
            kind: RawInequalityKind::BoundaryEquality,
            promotion: classify_affine_row(
                expr,
                &decision_map,
                &decision_set,
                &mut affine_memo,
                rows.len(),
            ),
        });
    }
    for expr in boundary_ineq {
        rows.push(RawInequalityRow {
            kind: RawInequalityKind::BoundaryInequality,
            promotion: classify_affine_row(
                expr,
                &decision_map,
                &decision_set,
                &mut affine_memo,
                rows.len(),
            ),
        });
    }
    for expr in path {
        rows.push(RawInequalityRow {
            kind: RawInequalityKind::Path,
            promotion: classify_affine_row(
                expr,
                &decision_map,
                &decision_set,
                &mut affine_memo,
                rows.len(),
            ),
        });
    }

    PromotionPlan { rows }
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

fn build_raw_bounds<C, Beq, Bineq>(
    plan: &PromotionPlan,
    offsets: &[f64],
    path_bounds: &BoundTemplate<C>,
    bineq_bounds: &BoundTemplate<Bineq>,
    tf_bounds: Bounds1D,
    variable_count: usize,
) -> Result<(Vec<f64>, Vec<f64>), GuessError>
where
    C: Vectorize<SX>,
    Bineq: Vectorize<SX>,
    Beq: Vectorize<SX>,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
{
    let mut variable_lower = vec![-NLP_BOUND_INF; variable_count];
    let mut variable_upper = vec![NLP_BOUND_INF; variable_count];
    let tf_index = variable_count - 1;
    apply_bounds_to_coordinate(
        &mut variable_lower,
        &mut variable_upper,
        tf_index,
        1.0,
        0.0,
        tf_bounds.lower,
        tf_bounds.upper,
    )?;
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
) -> Result<Vec<f64>, GuessError>
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
            lower.push(-NLP_BOUND_INF);
            continue;
        }
        match row.kind {
            RawInequalityKind::BoundaryEquality => lower.push(0.0),
            RawInequalityKind::BoundaryInequality => {
                let bound = &boundary_ineq[boundary_ineq_index];
                lower.push(bound.lower.unwrap_or(-NLP_BOUND_INF));
                boundary_ineq_index += 1;
            }
            RawInequalityKind::Path => {
                let bound = &path[path_index % path.len()];
                lower.push(bound.lower.unwrap_or(-NLP_BOUND_INF));
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
) -> Result<Vec<f64>, GuessError>
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
            upper.push(NLP_BOUND_INF);
            continue;
        }
        match row.kind {
            RawInequalityKind::BoundaryEquality => upper.push(0.0),
            RawInequalityKind::BoundaryInequality => {
                let bound = &boundary_ineq[boundary_ineq_index];
                upper.push(bound.upper.unwrap_or(NLP_BOUND_INF));
                boundary_ineq_index += 1;
            }
            RawInequalityKind::Path => {
                let bound = &path[path_index % path.len()];
                upper.push(bound.upper.unwrap_or(NLP_BOUND_INF));
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
    variable_lower: &mut [f64],
    variable_upper: &mut [f64],
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
        variable_lower[index] = variable_lower[index].max(value);
    }
    if let Some(value) = candidate_upper {
        variable_upper[index] = variable_upper[index].min(value);
    }
    if variable_lower[index] > variable_upper[index] {
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

fn project_multiple_shooting<X, U, const N: usize>(
    values: &[f64],
) -> Result<
    MultipleShootingTrajectories<Numeric<X>, Numeric<U>, N>,
    optimization::VectorizeLayoutError,
>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    MsVarsNum<X, U, N>: Vectorize<f64, Rebind<f64> = MsVarsNum<X, U, N>>,
{
    let (x, u, dudt, tf) = unflatten_value::<MsVarsNum<X, U, N>, f64>(values)?;
    Ok(MultipleShootingTrajectories { x, u, dudt, tf })
}

fn project_direct_collocation<X, U, const N: usize, const K: usize>(
    values: &[f64],
) -> Result<
    DirectCollocationTrajectories<Numeric<X>, Numeric<U>, N, K>,
    optimization::VectorizeLayoutError,
>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    DcVarsNum<X, U, N, K>: Vectorize<f64, Rebind<f64> = DcVarsNum<X, U, N, K>>,
{
    let (x, u, root_x, root_u, root_dudt, tf) =
        unflatten_value::<DcVarsNum<X, U, N, K>, f64>(values)?;
    Ok(DirectCollocationTrajectories {
        x,
        u,
        root_x,
        root_u,
        root_dudt,
        tf,
    })
}

fn mesh_times<const N: usize>(tf: f64) -> Mesh<f64, N> {
    let step = tf / (N as f64);
    Mesh {
        nodes: std::array::from_fn(|index| index as f64 * step),
        terminal: tf,
    }
}

fn direct_collocation_times<const N: usize, const K: usize>(
    tf: f64,
    coeffs: &CollocationCoefficients,
) -> DirectCollocationTimeGrid<N, K> {
    let step = tf / (N as f64);
    DirectCollocationTimeGrid {
        nodes: mesh_times::<N>(tf),
        roots: IntervalGrid {
            intervals: std::array::from_fn(|interval| {
                std::array::from_fn(|root| (interval as f64 + coeffs.nodes[root]) * step)
            }),
        },
    }
}

pub fn direct_collocation_interval_times<const N: usize, const K: usize>(
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> Vec<Vec<f64>> {
    (0..N)
        .map(|interval| {
            let mut times = Vec::with_capacity(K + 2);
            times.push(time_grid.nodes.nodes[interval]);
            for root in 0..K {
                times.push(time_grid.roots.intervals[interval][root]);
            }
            let end_time = if interval + 1 < N {
                time_grid.nodes.nodes[interval + 1]
            } else {
                time_grid.nodes.terminal
            };
            times.push(end_time);
            times
        })
        .collect()
}

pub fn direct_collocation_root_times<const N: usize, const K: usize>(
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> Vec<Vec<f64>> {
    (0..N)
        .map(|interval| {
            (0..K)
                .map(|root| time_grid.roots.intervals[interval][root])
                .collect()
        })
        .collect()
}

pub fn direct_collocation_extrapolated_end<T, const N: usize, const K: usize>(
    start: &T,
    roots: &[T; K],
    time_grid: &DirectCollocationTimeGrid<N, K>,
    interval: usize,
) -> Result<T, optimization::VectorizeLayoutError>
where
    T: Vectorize<f64, Rebind<f64> = T> + Clone,
{
    let start_time = time_grid.nodes.nodes[interval];
    let end_time = if interval + 1 < N {
        time_grid.nodes.nodes[interval + 1]
    } else {
        time_grid.nodes.terminal
    };
    let step = end_time - start_time;
    let nodes = (0..K)
        .map(|root| {
            if step.abs() <= 1.0e-12 {
                0.0
            } else {
                (time_grid.roots.intervals[interval][root] - start_time) / step
            }
        })
        .collect::<Vec<_>>();
    let mut basis_nodes = Vec::with_capacity(K + 1);
    basis_nodes.push(0.0);
    basis_nodes.extend(nodes.iter().copied());
    let weights = basis_nodes
        .iter()
        .enumerate()
        .map(|(basis_index, &basis_node)| {
            lagrange_basis_polynomial(&basis_nodes, basis_index, basis_node)
                .map(|poly| evaluate_polynomial(&poly, 1.0))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("collocation nodes derived from the time grid should define a valid basis");

    let mut flat = vec![0.0; T::LEN];
    let start_flat = flatten_value(start);
    for (index, value) in start_flat.iter().enumerate() {
        flat[index] += weights[0] * value;
    }
    for root in 0..K {
        let root_flat = flatten_value(&roots[root]);
        for (index, value) in root_flat.iter().enumerate() {
            flat[index] += weights[root + 1] * value;
        }
    }
    unflatten_value::<T, f64>(&flat)
}

pub fn direct_collocation_state_like_arcs<T, const N: usize, const K: usize>(
    mesh: &Mesh<T, N>,
    roots: &IntervalGrid<T, N, K>,
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> Result<Vec<IntervalArc<T>>, optimization::VectorizeLayoutError>
where
    T: Vectorize<f64, Rebind<f64> = T> + Clone,
{
    let interval_times = direct_collocation_interval_times(time_grid);
    (0..N)
        .map(|interval| {
            let mut values = Vec::with_capacity(K + 2);
            values.push(mesh.nodes[interval].clone());
            for root in 0..K {
                values.push(roots.intervals[interval][root].clone());
            }
            values.push(direct_collocation_extrapolated_end(
                &mesh.nodes[interval],
                &roots.intervals[interval],
                time_grid,
                interval,
            )?);
            Ok(IntervalArc {
                times: interval_times[interval].clone(),
                values,
            })
        })
        .collect()
}

pub fn direct_collocation_root_arcs<T, const N: usize, const K: usize>(
    roots: &IntervalGrid<T, N, K>,
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> Vec<IntervalArc<T>>
where
    T: Clone,
{
    let root_times = direct_collocation_root_times(time_grid);
    (0..N)
        .map(|interval| IntervalArc {
            times: root_times[interval].clone(),
            values: (0..K)
                .map(|root| roots.intervals[interval][root].clone())
                .collect(),
        })
        .collect()
}

fn build_multiple_shooting_interpolated_guess<X, U, const N: usize>(
    samples: &InterpolatedTrajectory<X, U>,
) -> Result<(Mesh<X, N>, Mesh<U, N>, [U; N], f64), GuessError>
where
    X: Clone,
    U: Clone,
{
    validate_interpolation_samples(samples)?;
    let times = mesh_times::<N>(samples.tf);
    Ok((
        Mesh {
            nodes: std::array::from_fn(|index| {
                interpolate_at(
                    &samples.sample_times,
                    &samples.x_samples,
                    times.nodes[index],
                )
            }),
            terminal: interpolate_at(&samples.sample_times, &samples.x_samples, times.terminal),
        },
        Mesh {
            nodes: std::array::from_fn(|index| {
                interpolate_at(
                    &samples.sample_times,
                    &samples.u_samples,
                    times.nodes[index],
                )
            }),
            terminal: interpolate_at(&samples.sample_times, &samples.u_samples, times.terminal),
        },
        std::array::from_fn(|index| {
            interpolate_at(
                &samples.sample_times,
                &samples.dudt_samples,
                times.nodes[index],
            )
        }),
        samples.tf,
    ))
}

fn build_direct_collocation_interpolated_guess<X, U, const N: usize, const K: usize>(
    samples: &InterpolatedTrajectory<X, U>,
    coeffs: &CollocationCoefficients,
) -> Result<
    (
        Mesh<X, N>,
        Mesh<U, N>,
        IntervalGrid<X, N, K>,
        IntervalGrid<U, N, K>,
        IntervalGrid<U, N, K>,
        f64,
    ),
    GuessError,
>
where
    X: Clone,
    U: Clone,
{
    validate_interpolation_samples(samples)?;
    let times = direct_collocation_times::<N, K>(samples.tf, coeffs);
    Ok((
        Mesh {
            nodes: std::array::from_fn(|index| {
                interpolate_at(
                    &samples.sample_times,
                    &samples.x_samples,
                    times.nodes.nodes[index],
                )
            }),
            terminal: interpolate_at(
                &samples.sample_times,
                &samples.x_samples,
                times.nodes.terminal,
            ),
        },
        Mesh {
            nodes: std::array::from_fn(|index| {
                interpolate_at(
                    &samples.sample_times,
                    &samples.u_samples,
                    times.nodes.nodes[index],
                )
            }),
            terminal: interpolate_at(
                &samples.sample_times,
                &samples.u_samples,
                times.nodes.terminal,
            ),
        },
        IntervalGrid {
            intervals: std::array::from_fn(|interval| {
                std::array::from_fn(|root| {
                    interpolate_at(
                        &samples.sample_times,
                        &samples.x_samples,
                        times.roots.intervals[interval][root],
                    )
                })
            }),
        },
        IntervalGrid {
            intervals: std::array::from_fn(|interval| {
                std::array::from_fn(|root| {
                    interpolate_at(
                        &samples.sample_times,
                        &samples.u_samples,
                        times.roots.intervals[interval][root],
                    )
                })
            }),
        },
        IntervalGrid {
            intervals: std::array::from_fn(|interval| {
                std::array::from_fn(|root| {
                    interpolate_at(
                        &samples.sample_times,
                        &samples.dudt_samples,
                        times.roots.intervals[interval][root],
                    )
                })
            }),
        },
        samples.tf,
    ))
}

fn validate_interpolation_samples<X, U>(
    samples: &InterpolatedTrajectory<X, U>,
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

fn build_multiple_shooting_rollout_guess<X, U, P, const N: usize, const RK4_SUBSTEPS: usize>(
    xdot: &CompiledXdot<X, U, P>,
    x0: &Numeric<X>,
    u0: &Numeric<U>,
    tf: f64,
    parameters: &Numeric<P>,
    controller: &ControllerFn<Numeric<X>, Numeric<U>, Numeric<P>>,
) -> Result<
    (
        Mesh<Numeric<X>, N>,
        Mesh<Numeric<U>, N>,
        [Numeric<U>; N],
        f64,
    ),
    GuessError,
>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    let h = tf / (N as f64);
    let mut x_nodes = std::array::from_fn(|_| x0.clone());
    let mut u_nodes = std::array::from_fn(|_| u0.clone());
    let mut rates = std::array::from_fn(|_| u0.clone());
    let mut x = x0.clone();
    let mut u = u0.clone();
    for interval in 0..N {
        x_nodes[interval] = x.clone();
        u_nodes[interval] = u.clone();
        let t = interval as f64 * h;
        let dudt = controller(t, &x, &u, parameters);
        rates[interval] = dudt.clone();
        let (x_next, u_next) =
            rk4_rollout_numeric::<X, U, P>(xdot, &x, &u, &dudt, parameters, h, RK4_SUBSTEPS)?;
        x = x_next;
        u = u_next;
    }
    Ok((
        Mesh {
            nodes: x_nodes,
            terminal: x,
        },
        Mesh {
            nodes: u_nodes,
            terminal: u,
        },
        rates,
        tf,
    ))
}

fn build_direct_collocation_rollout_guess<X, U, P, const N: usize, const K: usize>(
    xdot: &CompiledXdot<X, U, P>,
    x0: &Numeric<X>,
    u0: &Numeric<U>,
    tf: f64,
    parameters: &Numeric<P>,
    controller: &ControllerFn<Numeric<X>, Numeric<U>, Numeric<P>>,
    coeffs: &CollocationCoefficients,
) -> Result<
    (
        Mesh<Numeric<X>, N>,
        Mesh<Numeric<U>, N>,
        IntervalGrid<Numeric<X>, N, K>,
        IntervalGrid<Numeric<U>, N, K>,
        IntervalGrid<Numeric<U>, N, K>,
        f64,
    ),
    GuessError,
>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    let h = tf / (N as f64);
    let mut x_nodes = std::array::from_fn(|_| x0.clone());
    let mut u_nodes = std::array::from_fn(|_| u0.clone());
    let mut root_x = std::array::from_fn(|_| std::array::from_fn(|_| x0.clone()));
    let mut root_u = std::array::from_fn(|_| std::array::from_fn(|_| u0.clone()));
    let mut root_dudt = std::array::from_fn(|_| std::array::from_fn(|_| u0.clone()));
    let mut x = x0.clone();
    let mut u = u0.clone();
    for interval in 0..N {
        x_nodes[interval] = x.clone();
        u_nodes[interval] = u.clone();
        let t = interval as f64 * h;
        let dudt = controller(t, &x, &u, parameters);
        for root in 0..K {
            let duration = coeffs.nodes[root] * h;
            let (x_root, u_root) =
                rk4_rollout_numeric::<X, U, P>(xdot, &x, &u, &dudt, parameters, duration, 8)?;
            root_x[interval][root] = x_root;
            root_u[interval][root] = u_root;
            root_dudt[interval][root] = dudt.clone();
        }
        let (x_next, u_next) =
            rk4_rollout_numeric::<X, U, P>(xdot, &x, &u, &dudt, parameters, h, 8)?;
        x = x_next;
        u = u_next;
    }
    Ok((
        Mesh {
            nodes: x_nodes,
            terminal: x,
        },
        Mesh {
            nodes: u_nodes,
            terminal: u,
        },
        IntervalGrid { intervals: root_x },
        IntervalGrid { intervals: root_u },
        IntervalGrid {
            intervals: root_dudt,
        },
        tf,
    ))
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

    fn lqr_ocp_ms<const N: usize, const RK4_SUBSTEPS: usize>()
    -> Ocp<State<SX>, Control<SX>, Params<SX>, (), State<SX>, (), MultipleShooting<N, RK4_SUBSTEPS>>
    {
        Ocp::new("lqr_ms", MultipleShooting::<N, RK4_SUBSTEPS>)
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
    -> Ocp<State<SX>, Control<SX>, Params<SX>, (), State<SX>, (), DirectCollocation<N, K>> {
        Ocp::new("lqr_dc", DirectCollocation::<N, K>::default())
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

    #[test]
    fn tuple_vectorize_roundtrip_supports_internal_ocp_layouts() {
        let value = (
            Mesh {
                nodes: [State { x: 1.0, v: 2.0 }],
                terminal: State { x: 3.0, v: 4.0 },
            },
            Mesh {
                nodes: [Control { u: 5.0 }],
                terminal: Control { u: 6.0 },
            },
            [Control { u: 7.0 }],
            8.0,
        );
        let flat = flatten_value(&value);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rebuilt = unflatten_value::<
            (
                Mesh<State<f64>, 1>,
                Mesh<Control<f64>, 1>,
                [Control<f64>; 1],
                f64,
            ),
            f64,
        >(&flat)
        .expect("tuple layout should unflatten");
        assert_eq!(rebuilt, value);
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
            .build_multiple_shooting_symbolic_function_library(options)
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
