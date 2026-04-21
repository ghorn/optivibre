use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use optimization::{
    BackendTimingMetadata, CallPolicy, CallPolicyConfig, ClarabelSqpError, FunctionCompileOptions,
    InteriorPointSolveError, LlvmOptimizationLevel, NlpCompileStats, SymbolicCompileMetadata,
    SymbolicCompileProgress, SymbolicCompileStage, SymbolicNlpCompileOptions, SymbolicNlpOutputs,
    TypedCompiledJitNlp, TypedRuntimeNlpBounds, Vectorize, flatten_value,
    format_nlip_settings_summary, format_sqp_settings_summary,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptSolveError, format_ipopt_settings_summary};
use sx_core::{HessianStrategy, SX};

use crate::benchmark_report::{BenchmarkCaseProgress, OcpBenchmarkPreset, OcpBenchmarkRecord};
use crate::common::{
    ArtifactVisualization, Chart, CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate,
    ConstraintPanelCategory, ConstraintPanelEntry, ConstraintPanelSeverity, ConstraintPanels,
    ControlSection, ControlSemantic, ControlSpec, ControlVisibility, FromMap, LatexSection,
    MetricKey, PlotMode, ProblemDerivativeCheck, ProblemId, ProblemSpec, Scene2D, SceneCircle,
    ScenePath, ScenePath3D, SharedCompileCache, SolveArtifact, SolveStage, SolveStatus,
    SolveStreamEvent, SolverConfig, SolverMethod, SolverPhaseDetail, SolverPhaseDetails,
    SolverReport, TimeSeries, TimeSeriesRole, append_nlip_termination_metric,
    append_termination_metric, cached_compile_with_progress, compile_cache_status,
    default_solver_config, metric_with_key, nlip_failure_solver_report, nlip_options,
    nlip_progress, numeric_metric_with_key, problem_slider_control, problem_spec, select_control,
    solver_config_from_map, solver_controls, solver_method_from_map, sqp_failure_solver_report,
    sqp_options, sqp_progress, summarize_backend_compile_report,
};
#[cfg(feature = "ipopt")]
use crate::common::{
    append_ipopt_termination_metric, ipopt_failure_solver_report, ipopt_options, ipopt_progress,
};

const CHAIN_LINKS: usize = 24;
const CHAIN_POINTS: usize = CHAIN_LINKS - 1;
const CHAIN_LINK_LENGTH: f64 = 1.0;
const DEFAULT_CHAIN_SPAN_RATIO: f64 = 0.75;
const MAX_3D_CHAIN_FRAMES: usize = 44;
const ROSENBROCK_CONTOUR_POINTS: usize = 68;

type HangingChainCompiled =
    TypedCompiledJitNlp<Chain<SX, CHAIN_POINTS>, ChainParams<SX>, VecN<SX, CHAIN_LINKS>, ()>;
type RosenbrockCompiled = TypedCompiledJitNlp<Pair<SX>, RosenbrockCoefficients<SX>, (), Disk<SX>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StaticCompileKey;

thread_local! {
    static HANGING_CHAIN_CACHE: RefCell<SharedCompileCache<StaticCompileKey, HangingChainCompiled>> =
        RefCell::new(SharedCompileCache::new());
    static ROSENBROCK_CACHE: RefCell<SharedCompileCache<StaticCompileKey, RosenbrockCompiled>> =
        RefCell::new(SharedCompileCache::new());
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Pair<T> {
    x: T,
    y: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Disk<T> {
    radius_sq: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct RosenbrockCoefficients<T> {
    a: T,
    b: T,
    tilt_x: T,
    tilt_y: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Chain<T, const N: usize> {
    points: [Point<T>; N],
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct VecN<T, const N: usize> {
    values: [T; N],
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct ChainParams<T> {
    span: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChainInitialCondition {
    LinearInfeasible,
    ZigZagFeasible,
    QuadraticFeasible,
    QuadraticUpsideDown,
}

impl ChainInitialCondition {
    fn from_value(value: f64) -> Result<Self> {
        match rounded_choice(value, "initial_condition")? {
            0 => Ok(Self::LinearInfeasible),
            1 => Ok(Self::ZigZagFeasible),
            2 => Ok(Self::QuadraticFeasible),
            3 => Ok(Self::QuadraticUpsideDown),
            other => Err(anyhow!("unsupported initial_condition {other}")),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::LinearInfeasible => "linear infeasible",
            Self::ZigZagFeasible => "zig-zag feasible",
            Self::QuadraticFeasible => "quadratic feasible",
            Self::QuadraticUpsideDown => "quadratic upside down",
        }
    }
}

#[derive(Clone, Debug)]
pub struct HangingChainParams {
    solver_method: SolverMethod,
    solver: SolverConfig,
    span_ratio: f64,
    initial_condition: ChainInitialCondition,
}

impl Default for HangingChainParams {
    fn default() -> Self {
        let mut solver = default_solver_config();
        solver.max_iters = 120;
        solver.constraint_tol = 1.0e-9;
        solver.dual_tol = 1.0e-7;
        solver.complementarity_tol = 1.0e-7;
        Self {
            solver_method: SolverMethod::Nlip,
            solver,
            span_ratio: DEFAULT_CHAIN_SPAN_RATIO,
            initial_condition: ChainInitialCondition::ZigZagFeasible,
        }
    }
}

impl FromMap for HangingChainParams {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            solver_method: solver_method_from_map(values, defaults.solver_method)?,
            solver: solver_config_from_map(values, defaults.solver)?,
            span_ratio: finite_in_range(
                sample_or_default(values, "chain_span_ratio", defaults.span_ratio),
                "chain_span_ratio",
                0.55,
                0.92,
            )?,
            initial_condition: ChainInitialCondition::from_value(sample_or_default(
                values,
                "chain_initial_condition",
                1.0,
            ))?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RosenbrockVariant {
    Classic,
    NarrowValley,
    Tilted,
    DiskConstrained,
}

impl RosenbrockVariant {
    fn from_value(value: f64) -> Result<Self> {
        match rounded_choice(value, "rosenbrock_variant")? {
            0 => Ok(Self::Classic),
            1 => Ok(Self::NarrowValley),
            2 => Ok(Self::Tilted),
            3 => Ok(Self::DiskConstrained),
            other => Err(anyhow!("unsupported rosenbrock_variant {other}")),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::NarrowValley => "narrow valley",
            Self::Tilted => "tilted valley",
            Self::DiskConstrained => "disk constrained",
        }
    }

    const fn coefficients(self) -> RosenbrockCoefficients<f64> {
        match self {
            Self::Classic => RosenbrockCoefficients {
                a: 1.0,
                b: 100.0,
                tilt_x: 0.0,
                tilt_y: 0.0,
            },
            Self::NarrowValley => RosenbrockCoefficients {
                a: 1.0,
                b: 500.0,
                tilt_x: 0.0,
                tilt_y: 0.0,
            },
            Self::Tilted => RosenbrockCoefficients {
                a: 1.0,
                b: 100.0,
                tilt_x: 0.12,
                tilt_y: -0.08,
            },
            Self::DiskConstrained => RosenbrockCoefficients {
                a: 1.0,
                b: 100.0,
                tilt_x: 0.0,
                tilt_y: 0.0,
            },
        }
    }

    const fn disk_radius_sq(self) -> Option<f64> {
        match self {
            Self::DiskConstrained => Some(1.5),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RosenbrockParams {
    solver_method: SolverMethod,
    solver: SolverConfig,
    variant: RosenbrockVariant,
    start_x: f64,
    start_y: f64,
}

impl Default for RosenbrockParams {
    fn default() -> Self {
        let mut solver = default_solver_config();
        solver.max_iters = 80;
        solver.dual_tol = 1.0e-8;
        solver.constraint_tol = 1.0e-9;
        solver.complementarity_tol = 1.0e-8;
        Self {
            solver_method: SolverMethod::Sqp,
            solver,
            variant: RosenbrockVariant::Classic,
            start_x: -1.2,
            start_y: 1.0,
        }
    }
}

impl FromMap for RosenbrockParams {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            solver_method: solver_method_from_map(values, defaults.solver_method)?,
            solver: solver_config_from_map(values, defaults.solver)?,
            variant: RosenbrockVariant::from_value(sample_or_default(
                values,
                "rosenbrock_variant",
                0.0,
            ))?,
            start_x: finite_in_range(
                sample_or_default(values, "rosenbrock_start_x", defaults.start_x),
                "rosenbrock_start_x",
                -2.5,
                2.5,
            )?,
            start_y: finite_in_range(
                sample_or_default(values, "rosenbrock_start_y", defaults.start_y),
                "rosenbrock_start_y",
                -1.5,
                3.5,
            )?,
        })
    }
}

pub(crate) fn hanging_chain_problem_entry() -> crate::ProblemEntry {
    crate::ProblemEntry {
        id: ProblemId::HangingChainStatic,
        spec: hanging_chain_spec,
        solve_from_map: hanging_chain_solve_from_map,
        prewarm_from_map: hanging_chain_prewarm_from_map,
        validate_derivatives_from_request: unsupported_derivatives,
        solve_with_progress_boxed: hanging_chain_solve_with_progress_boxed,
        prewarm_with_progress_boxed: hanging_chain_prewarm_with_progress_boxed,
        compile_cache_statuses: hanging_chain_compile_cache_statuses,
        benchmark_default_case_with_progress: unsupported_benchmark,
    }
}

pub(crate) fn rosenbrock_problem_entry() -> crate::ProblemEntry {
    crate::ProblemEntry {
        id: ProblemId::RosenbrockVariants,
        spec: rosenbrock_spec,
        solve_from_map: rosenbrock_solve_from_map,
        prewarm_from_map: rosenbrock_prewarm_from_map,
        validate_derivatives_from_request: unsupported_derivatives,
        solve_with_progress_boxed: rosenbrock_solve_with_progress_boxed,
        prewarm_with_progress_boxed: rosenbrock_prewarm_with_progress_boxed,
        compile_cache_statuses: rosenbrock_compile_cache_statuses,
        benchmark_default_case_with_progress: unsupported_benchmark,
    }
}

pub(crate) fn compile_variant_for_problem(problem: ProblemId) -> Option<(String, String)> {
    match problem {
        ProblemId::HangingChainStatic => Some(hanging_chain_variant()),
        ProblemId::RosenbrockVariants => Some(rosenbrock_variant()),
        _ => None,
    }
}

fn unsupported_derivatives(
    _: &crate::common::DerivativeCheckRequest,
) -> Result<ProblemDerivativeCheck> {
    Err(anyhow!(
        "derivative checks are not wired for static optimization webapp problems yet"
    ))
}

fn unsupported_benchmark(
    _: crate::common::TranscriptionMethod,
    _: OcpBenchmarkPreset,
    _: optimization::NlpEvaluationBenchmarkOptions,
    _: &mut dyn FnMut(BenchmarkCaseProgress),
) -> Result<OcpBenchmarkRecord> {
    Err(anyhow!(
        "OCP benchmark presets do not apply to static optimization problems"
    ))
}

fn sample_or_default(values: &BTreeMap<String, f64>, key: &str, default: f64) -> f64 {
    values.get(key).copied().unwrap_or(default)
}

fn rounded_choice(value: f64, label: &str) -> Result<i64> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    Ok(value.round() as i64)
}

fn finite_in_range(value: f64, label: &str, min: f64, max: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    if value < min || value > max {
        return Err(anyhow!("{label} must be in {min}..={max}"));
    }
    Ok(value)
}

fn static_select_control(
    id: &str,
    label: &str,
    default: f64,
    help: &str,
    choices: &[(f64, &str)],
) -> ControlSpec {
    select_control(
        id,
        label,
        default,
        "",
        help,
        choices,
        ControlSection::Problem,
        ControlVisibility::Always,
        ControlSemantic::ProblemParameter,
    )
}

fn static_controls(
    solver_method: SolverMethod,
    solver: SolverConfig,
    extra_controls: impl IntoIterator<Item = ControlSpec>,
) -> Vec<ControlSpec> {
    let mut controls = solver_controls(solver_method, solver);
    controls.extend(extra_controls);
    controls
}

pub fn hanging_chain_spec() -> ProblemSpec {
    let defaults = HangingChainParams::default();
    problem_spec(
        ProblemId::HangingChainStatic,
        "Static Hanging Chain",
        "A fixed-length chain between two anchors solved as a static constrained nonlinear program.",
        static_controls(
            defaults.solver_method,
            defaults.solver,
            vec![
                static_select_control(
                    "chain_initial_condition",
                    "Initial Shape",
                    1.0,
                    "Choose whether the first iterate starts infeasible, feasible and folded, feasible and sagging, or feasible and upside down.",
                    &[
                        (0.0, "Linear Infeasible"),
                        (1.0, "Zig-Zag Feasible"),
                        (2.0, "Quadratic Feasible"),
                        (3.0, "Quadratic Upside Down"),
                    ],
                ),
                problem_slider_control(
                    "chain_span_ratio",
                    "Anchor Span Ratio",
                    0.55,
                    0.92,
                    0.01,
                    defaults.span_ratio,
                    "",
                    "Horizontal anchor spacing as a fraction of total chain length.",
                ),
            ],
        ),
        vec![
            LatexSection {
                title: "Variables".to_string(),
                entries: vec![
                    r"q_i = \begin{bmatrix} x_i & y_i \end{bmatrix}^{\mathsf T}, \quad i=1,\dots,n-1"
                        .to_string(),
                ],
            },
            LatexSection {
                title: "Objective".to_string(),
                entries: vec![r"\min_q \sum_{i=1}^{n-1} y_i".to_string()],
            },
            LatexSection {
                title: "Link Equalities".to_string(),
                entries: vec![
                    r"\|q_1-a\|_2^2 = \ell^2,\quad \|q_i-q_{i-1}\|_2^2 = \ell^2,\quad \|b-q_{n-1}\|_2^2 = \ell^2"
                        .to_string(),
                ],
            },
        ],
        vec![
            "The 2D scene overlays the initial chain and the optimized equilibrium.".to_string(),
            "The 3D convergence plot uses iteration number as the third axis, so every accepted chain shape remains visible.".to_string(),
        ],
    )
}

pub fn rosenbrock_spec() -> ProblemSpec {
    let defaults = RosenbrockParams::default();
    problem_spec(
        ProblemId::RosenbrockVariants,
        "Rosenbrock Variants",
        "Two-variable Rosenbrock objectives with live iterate paths drawn directly on the objective contour map.",
        static_controls(
            defaults.solver_method,
            defaults.solver,
            vec![
                static_select_control(
                    "rosenbrock_variant",
                    "Variant",
                    0.0,
                    "Switch the objective curvature or add an active disk constraint.",
                    &[
                        (0.0, "Classic"),
                        (1.0, "Narrow Valley"),
                        (2.0, "Tilted Valley"),
                        (3.0, "Disk Constrained"),
                    ],
                ),
                problem_slider_control(
                    "rosenbrock_start_x",
                    "Start X",
                    -2.5,
                    2.5,
                    0.05,
                    defaults.start_x,
                    "",
                    "Initial x coordinate.",
                ),
                problem_slider_control(
                    "rosenbrock_start_y",
                    "Start Y",
                    -1.5,
                    3.5,
                    0.05,
                    defaults.start_y,
                    "",
                    "Initial y coordinate.",
                ),
            ],
        ),
        vec![
            LatexSection {
                title: "Classic Form".to_string(),
                entries: vec![r"f(x,y) = (a-x)^2 + b\,(y-x^2)^2 + c_x x + c_y y".to_string()],
            },
            LatexSection {
                title: "Disk Variant".to_string(),
                entries: vec![r"x^2 + y^2 \le 1.5".to_string()],
            },
        ],
        vec![
            "The contour view keeps the full accepted-iterate path visible while the solver runs.".to_string(),
            "The disk-constrained variant shows how the unconstrained minimizer is clipped by an active nonlinear inequality.".to_string(),
        ],
    )
}

fn static_compile_options() -> SymbolicNlpCompileOptions {
    SymbolicNlpCompileOptions {
        function_options: FunctionCompileOptions::new(
            LlvmOptimizationLevel::O3,
            CallPolicyConfig {
                default_policy: CallPolicy::InlineAtLowering,
                respect_function_overrides: true,
            },
        ),
        hessian_strategy: HessianStrategy::LowerTriangleByColumn,
    }
}

fn duration_label(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 10.0 {
        format!("{seconds:.1} s")
    } else if seconds >= 1.0 {
        format!("{seconds:.2} s")
    } else {
        format!("{:.1} ms", seconds * 1000.0)
    }
}

fn phase_detail(
    label: impl Into<String>,
    value: impl Into<String>,
    count: usize,
) -> SolverPhaseDetail {
    SolverPhaseDetail {
        label: label.into(),
        value: value.into(),
        count,
    }
}

fn push_duration_detail(
    details: &mut Vec<SolverPhaseDetail>,
    label: &str,
    duration: Option<Duration>,
) {
    if let Some(duration) = duration
        && duration > Duration::ZERO
    {
        details.push(phase_detail(label, duration_label(duration), 0));
    }
}

fn symbolic_stage_label(stage: SymbolicCompileStage) -> &'static str {
    match stage {
        SymbolicCompileStage::BuildProblem => "Build Problem",
        SymbolicCompileStage::ObjectiveGradient => "Objective Gradient",
        SymbolicCompileStage::EqualityJacobian => "Equality Jacobian",
        SymbolicCompileStage::InequalityJacobian => "Inequality Jacobian",
        SymbolicCompileStage::LagrangianAssembly => "Lagrangian Assembly",
        SymbolicCompileStage::HessianGeneration => "Hessian Generation",
    }
}

fn static_symbolic_details(
    stats: NlpCompileStats,
    metadata: Option<&SymbolicCompileMetadata>,
    active_stage: Option<SymbolicCompileStage>,
) -> Vec<SolverPhaseDetail> {
    let mut details = vec![
        phase_detail("Vars", stats.variable_count.to_string(), 0),
        phase_detail("Params", stats.parameter_scalar_count.to_string(), 0),
        phase_detail("Eq", stats.equality_count.to_string(), 0),
        phase_detail("Ineq", stats.inequality_count.to_string(), 0),
        phase_detail(
            "Jac NNZ",
            (stats.equality_jacobian_nnz + stats.inequality_jacobian_nnz).to_string(),
            0,
        ),
        phase_detail("Hess NNZ", stats.hessian_nnz.to_string(), 0),
    ];
    if let Some(stage) = active_stage {
        details.push(phase_detail("Stage", symbolic_stage_label(stage), 0));
    }
    if let Some(metadata) = metadata {
        push_duration_detail(
            &mut details,
            "Build Problem",
            metadata.setup_profile.symbolic_construction,
        );
        push_duration_detail(
            &mut details,
            "Objective Gradient",
            metadata.setup_profile.objective_gradient,
        );
        push_duration_detail(
            &mut details,
            "Equality Jacobian",
            metadata.setup_profile.equality_jacobian,
        );
        push_duration_detail(
            &mut details,
            "Inequality Jacobian",
            metadata.setup_profile.inequality_jacobian,
        );
        push_duration_detail(
            &mut details,
            "Lagrangian Assembly",
            metadata.setup_profile.lagrangian_assembly,
        );
        push_duration_detail(
            &mut details,
            "Hessian Generation",
            metadata.setup_profile.hessian_generation,
        );
    }
    details
}

fn static_jit_details(
    stats: NlpCompileStats,
    summary: Option<&crate::common::CompileReportSummary>,
) -> Vec<SolverPhaseDetail> {
    let mut details = vec![phase_detail(
        "NLP Kernels",
        stats.jit_kernel_count.to_string(),
        0,
    )];
    if let Some(summary) = summary {
        if summary.llvm_cache_hits > 0 || summary.llvm_cache_misses > 0 {
            details.push(phase_detail(
                "LLVM Cache Hits",
                summary.llvm_cache_hits.to_string(),
                0,
            ));
            details.push(phase_detail(
                "LLVM Cache Misses",
                summary.llvm_cache_misses.to_string(),
                0,
            ));
        }
    }
    details
}

fn compile_progress_update(progress: SymbolicCompileProgress) -> CompileProgressUpdate {
    match progress {
        SymbolicCompileProgress::Stage(stage) => CompileProgressUpdate {
            timing: stage.metadata.timing,
            phase_details: SolverPhaseDetails {
                symbolic_setup: static_symbolic_details(
                    stage.metadata.stats,
                    Some(&stage.metadata),
                    Some(stage.stage),
                ),
                jit: Vec::new(),
                solve: Vec::new(),
            },
            compile_cached: false,
        },
        SymbolicCompileProgress::Ready(metadata) => CompileProgressUpdate {
            timing: metadata.timing,
            phase_details: SolverPhaseDetails {
                symbolic_setup: static_symbolic_details(metadata.stats, Some(&metadata), None),
                jit: static_jit_details(metadata.stats, None),
                solve: Vec::new(),
            },
            compile_cached: false,
        },
    }
}

fn compile_report_is_fully_cached(summary: &crate::common::CompileReportSummary) -> bool {
    summary.llvm_cache_hits > 0 && summary.llvm_cache_misses == 0
}

fn compile_progress_info_from_static<Compiled>(compiled: &Compiled) -> CompileProgressInfo
where
    Compiled: StaticCompiledMetadata,
{
    let compile_report = summarize_backend_compile_report(compiled.backend_compile_report());
    CompileProgressInfo {
        timing: compiled.backend_timing_metadata(),
        compile_cached: compile_report_is_fully_cached(&compile_report),
        phase_details: SolverPhaseDetails {
            symbolic_setup: static_symbolic_details(compiled.compile_stats(), None, None),
            jit: static_jit_details(compiled.compile_stats(), Some(&compile_report)),
            solve: Vec::new(),
        },
        compile_report: Some(compile_report),
    }
}

trait StaticCompiledMetadata {
    fn backend_timing_metadata(&self) -> BackendTimingMetadata;
    fn compile_stats(&self) -> NlpCompileStats;
    fn backend_compile_report(&self) -> &optimization::BackendCompileReport;
}

impl<X, P, E, I> StaticCompiledMetadata for TypedCompiledJitNlp<X, P, E, I>
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
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        TypedCompiledJitNlp::backend_timing_metadata(self)
    }

    fn compile_stats(&self) -> NlpCompileStats {
        TypedCompiledJitNlp::compile_stats(self)
    }

    fn backend_compile_report(&self) -> &optimization::BackendCompileReport {
        TypedCompiledJitNlp::backend_compile_report(self)
    }
}

fn hanging_chain_variant() -> (String, String) {
    (
        format!("static_hanging_chain_links{CHAIN_LINKS}"),
        format!("Static NLP · {CHAIN_LINKS} links"),
    )
}

fn rosenbrock_variant() -> (String, String) {
    (
        "static_rosenbrock_2d".to_string(),
        "Static NLP · 2 variables".to_string(),
    )
}

fn compile_hanging_chain_with_progress(
    on_compile_progress: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(Rc<RefCell<HangingChainCompiled>>, CompileProgressInfo)> {
    HANGING_CHAIN_CACHE.with(|cache| {
        cached_compile_with_progress(
            &mut cache.borrow_mut(),
            StaticCompileKey,
            on_compile_progress,
            |callback| {
                let symbolic = optimization::symbolic_nlp::<
                    Chain<SX, CHAIN_POINTS>,
                    ChainParams<SX>,
                    VecN<SX, CHAIN_LINKS>,
                    (),
                    _,
                >("static_hanging_chain", |q, params| {
                    SymbolicNlpOutputs {
                        objective: q.points.iter().fold(SX::zero(), |acc, point| acc + point.y),
                        equalities: VecN {
                            values: std::array::from_fn(|idx| chain_link_residual(q, params, idx)),
                        },
                        inequalities: (),
                    }
                })?;
                symbolic
                    .compile_jit_with_compile_options_and_symbolic_progress_callback(
                        static_compile_options(),
                        |progress| callback(compile_progress_update(progress)),
                    )
                    .map_err(Into::into)
            },
            compile_progress_info_from_static,
        )
    })
}

fn compile_rosenbrock_with_progress(
    on_compile_progress: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(Rc<RefCell<RosenbrockCompiled>>, CompileProgressInfo)> {
    ROSENBROCK_CACHE.with(|cache| {
        cached_compile_with_progress(
            &mut cache.borrow_mut(),
            StaticCompileKey,
            on_compile_progress,
            |callback| {
                let symbolic = optimization::symbolic_nlp::<
                    Pair<SX>,
                    RosenbrockCoefficients<SX>,
                    (),
                    Disk<SX>,
                    _,
                >("static_rosenbrock", |x, params| {
                    let objective = (params.a - x.x).sqr()
                        + params.b * (x.y - x.x.sqr()).sqr()
                        + params.tilt_x * x.x
                        + params.tilt_y * x.y;
                    SymbolicNlpOutputs {
                        objective,
                        equalities: (),
                        inequalities: Disk {
                            radius_sq: x.x.sqr() + x.y.sqr(),
                        },
                    }
                })?;
                symbolic
                    .compile_jit_with_compile_options_and_symbolic_progress_callback(
                        static_compile_options(),
                        |progress| callback(compile_progress_update(progress)),
                    )
                    .map_err(Into::into)
            },
            compile_progress_info_from_static,
        )
    })
}

fn chain_link_residual(q: &Chain<SX, CHAIN_POINTS>, params: &ChainParams<SX>, idx: usize) -> SX {
    let link_sq = CHAIN_LINK_LENGTH * CHAIN_LINK_LENGTH;
    if idx == 0 {
        let point = &q.points[0];
        point.x.sqr() + point.y.sqr() - link_sq
    } else if idx == CHAIN_POINTS {
        let point = &q.points[CHAIN_POINTS - 1];
        (point.x - params.span).sqr() + point.y.sqr() - link_sq
    } else {
        let left = &q.points[idx - 1];
        let right = &q.points[idx];
        (right.x - left.x).sqr() + (right.y - left.y).sqr() - link_sq
    }
}

pub fn hanging_chain_prewarm(params: &HangingChainParams) -> Result<()> {
    let _ = params;
    compile_hanging_chain_with_progress(&mut |_| {}).map(|_| ())
}

pub fn hanging_chain_prewarm_with_progress<F>(params: &HangingChainParams, emit: F) -> Result<()>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    lifecycle.prewarm_with_progress(compile_hanging_chain_with_progress)
}

pub fn rosenbrock_prewarm(params: &RosenbrockParams) -> Result<()> {
    let _ = params;
    compile_rosenbrock_with_progress(&mut |_| {}).map(|_| ())
}

pub fn rosenbrock_prewarm_with_progress<F>(params: &RosenbrockParams, emit: F) -> Result<()>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    lifecycle.prewarm_with_progress(compile_rosenbrock_with_progress)
}

pub fn hanging_chain_solve(params: &HangingChainParams) -> Result<SolveArtifact> {
    let compiled = compile_hanging_chain_with_progress(&mut |_| {})?.0;
    let compiled = compiled.borrow();
    solve_hanging_chain_compiled(
        &compiled,
        params,
        |_| {},
        SolverReport::placeholder(),
        false,
    )
}

pub fn rosenbrock_solve(params: &RosenbrockParams) -> Result<SolveArtifact> {
    let compiled = compile_rosenbrock_with_progress(&mut |_| {})?.0;
    let compiled = compiled.borrow();
    solve_rosenbrock_compiled(
        &compiled,
        params,
        |_| {},
        SolverReport::placeholder(),
        false,
    )
}

pub fn hanging_chain_solve_with_progress<F>(
    params: &HangingChainParams,
    emit: F,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    let (compiled, running_solver, compile_report) =
        lifecycle.compile_with_progress(compile_hanging_chain_with_progress)?;
    let compiled = compiled.borrow();
    let mut artifact = solve_hanging_chain_compiled(
        &compiled,
        params,
        lifecycle.into_emit(),
        running_solver,
        true,
    )?;
    artifact.compile_report = compile_report;
    Ok(artifact)
}

pub fn rosenbrock_solve_with_progress<F>(
    params: &RosenbrockParams,
    emit: F,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    let (compiled, running_solver, compile_report) =
        lifecycle.compile_with_progress(compile_rosenbrock_with_progress)?;
    let compiled = compiled.borrow();
    let mut artifact = solve_rosenbrock_compiled(
        &compiled,
        params,
        lifecycle.into_emit(),
        running_solver,
        true,
    )?;
    artifact.compile_report = compile_report;
    Ok(artifact)
}

fn chain_span(params: &HangingChainParams) -> f64 {
    params.span_ratio * CHAIN_LINKS as f64 * CHAIN_LINK_LENGTH
}

fn chain_runtime(params: &HangingChainParams) -> ChainParams<f64> {
    ChainParams {
        span: chain_span(params),
    }
}

fn chain_x0(params: &HangingChainParams) -> Chain<f64, CHAIN_POINTS> {
    let span = chain_span(params);
    match params.initial_condition {
        ChainInitialCondition::LinearInfeasible => straight_chain(span),
        ChainInitialCondition::ZigZagFeasible => {
            shaped_feasible_chain(span, |idx| if idx % 2 == 0 { -1.0 } else { 1.0 })
        }
        ChainInitialCondition::QuadraticFeasible => {
            shaped_feasible_chain(span, |idx| -quadratic_shape(idx))
        }
        ChainInitialCondition::QuadraticUpsideDown => shaped_feasible_chain(span, quadratic_shape),
    }
}

fn straight_chain(span: f64) -> Chain<f64, CHAIN_POINTS> {
    Chain {
        points: std::array::from_fn(|idx| {
            let t = (idx + 1) as f64 / CHAIN_LINKS as f64;
            Point {
                x: span * t,
                y: 0.0,
            }
        }),
    }
}

fn quadratic_shape(idx: usize) -> f64 {
    let t = (idx + 1) as f64 / CHAIN_LINKS as f64;
    4.0 * t * (1.0 - t)
}

fn shaped_feasible_chain(span: f64, shape: impl Fn(usize) -> f64) -> Chain<f64, CHAIN_POINTS> {
    let coeffs: [f64; CHAIN_POINTS] = std::array::from_fn(shape);
    let mut dy_coeffs = [0.0; CHAIN_LINKS];
    for idx in 0..CHAIN_LINKS {
        let current = if idx < CHAIN_POINTS { coeffs[idx] } else { 0.0 };
        let previous = if idx == 0 { 0.0 } else { coeffs[idx - 1] };
        dy_coeffs[idx] = current - previous;
    }
    let max_coeff = dy_coeffs
        .iter()
        .fold(0.0_f64, |acc, coeff| acc.max(coeff.abs()));
    let mut low = 0.0;
    let mut high = if max_coeff > 0.0 {
        0.999 * CHAIN_LINK_LENGTH / max_coeff
    } else {
        0.0
    };
    for _ in 0..96 {
        let mid = 0.5 * (low + high);
        let horizontal = dy_coeffs.iter().fold(0.0, |acc, coeff| {
            let dy = mid * *coeff;
            acc + (CHAIN_LINK_LENGTH * CHAIN_LINK_LENGTH - dy * dy).sqrt()
        });
        if horizontal > span {
            low = mid;
        } else {
            high = mid;
        }
    }
    let amplitude = high;
    let y_values: [f64; CHAIN_POINTS] = std::array::from_fn(|idx| amplitude * coeffs[idx]);
    let mut x_cursor = 0.0;
    Chain {
        points: std::array::from_fn(|idx| {
            let previous_y = if idx == 0 { 0.0 } else { y_values[idx - 1] };
            let dy = y_values[idx] - previous_y;
            x_cursor += (CHAIN_LINK_LENGTH * CHAIN_LINK_LENGTH - dy * dy).sqrt();
            Point {
                x: x_cursor,
                y: y_values[idx],
            }
        }),
    }
}

fn rosenbrock_runtime(params: &RosenbrockParams) -> RosenbrockCoefficients<f64> {
    params.variant.coefficients()
}

fn rosenbrock_x0(params: &RosenbrockParams) -> Pair<f64> {
    Pair {
        x: params.start_x,
        y: params.start_y,
    }
}

fn rosenbrock_bounds(params: &RosenbrockParams) -> TypedRuntimeNlpBounds<Pair<SX>, Disk<SX>> {
    TypedRuntimeNlpBounds {
        variable_lower: None,
        variable_upper: None,
        inequality_lower: Some(Disk { radius_sq: None }),
        inequality_upper: Some(Disk {
            radius_sq: params.variant.disk_radius_sq(),
        }),
        scaling: None,
    }
}

fn hanging_chain_bounds() -> TypedRuntimeNlpBounds<Chain<SX, CHAIN_POINTS>, ()> {
    TypedRuntimeNlpBounds::default()
}

fn solve_hanging_chain_compiled<F>(
    compiled: &HangingChainCompiled,
    params: &HangingChainParams,
    emit: F,
    running_solver: SolverReport,
    stream_events: bool,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent),
{
    let x0 = chain_x0(params);
    let runtime = chain_runtime(params);
    let bounds = hanging_chain_bounds();
    solve_typed_static_with_progress(
        compiled,
        &x0,
        &runtime,
        &bounds,
        params.solver_method,
        &params.solver,
        emit,
        running_solver,
        stream_events,
        |x, history| hanging_chain_artifact(params, &flatten_value(&x0), x, history),
    )
}

fn solve_rosenbrock_compiled<F>(
    compiled: &RosenbrockCompiled,
    params: &RosenbrockParams,
    emit: F,
    running_solver: SolverReport,
    stream_events: bool,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent),
{
    let x0 = rosenbrock_x0(params);
    let runtime = rosenbrock_runtime(params);
    let bounds = rosenbrock_bounds(params);
    solve_typed_static_with_progress(
        compiled,
        &x0,
        &runtime,
        &bounds,
        params.solver_method,
        &params.solver,
        emit,
        running_solver,
        stream_events,
        |x, history| rosenbrock_artifact(params, x, history),
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_typed_static_with_progress<X, P, E, I, F, Build>(
    compiled: &TypedCompiledJitNlp<X, P, E, I>,
    x0: &<X as Vectorize<SX>>::Rebind<f64>,
    parameters: &<P as Vectorize<SX>>::Rebind<f64>,
    bounds: &TypedRuntimeNlpBounds<X, I>,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    mut emit: F,
    running_solver: SolverReport,
    stream_events: bool,
    mut build_artifact: Build,
) -> Result<SolveArtifact>
where
    X: Vectorize<SX, Rebind<SX> = X>,
    P: Vectorize<SX, Rebind<SX> = P>,
    E: Vectorize<SX, Rebind<SX> = E>,
    I: Vectorize<SX, Rebind<SX> = I>,
    <X as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone,
    <X as Vectorize<SX>>::Rebind<Option<f64>>: Vectorize<Option<f64>> + Clone,
    <P as Vectorize<SX>>::Rebind<f64>: Vectorize<f64> + Clone,
    <E as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <I as Vectorize<SX>>::Rebind<Option<f64>>: Vectorize<Option<f64>> + Clone,
    F: FnMut(SolveStreamEvent),
    Build: FnMut(&[f64], &[Vec<f64>]) -> SolveArtifact,
{
    let mut history = vec![flatten_value(x0)];
    if stream_events {
        emit(SolveStreamEvent::Status {
            status: SolveStatus {
                stage: SolveStage::Solving,
                solver_method: Some(solver_method),
                solver: running_solver.clone().with_solve_seconds(0.0),
            },
        });
    }

    match solver_method {
        SolverMethod::Sqp => {
            let options = sqp_options(solver_config);
            let solved = match compiled.solve_sqp_with_callback(
                x0,
                parameters,
                bounds,
                &options,
                |snapshot| {
                    push_history(&mut history, &snapshot.x);
                    let mut artifact = build_artifact(&snapshot.x, &history);
                    let progress = sqp_progress(snapshot);
                    artifact.solver = running_solver
                        .clone()
                        .with_iterations(progress.iteration)
                        .with_solve_seconds(0.0)
                        .with_phase_details(running_solver.phase_details.clone());
                    if stream_events {
                        emit(SolveStreamEvent::Iteration { progress, artifact });
                    }
                },
            ) {
                Ok(summary) => summary,
                Err(error) => {
                    if stream_events {
                        emit_solver_failure_status(
                            &mut emit,
                            solver_method,
                            sqp_failure_solver_report(&error, &running_solver, &options),
                        );
                    }
                    return Err(ClarabelSqpError::from(error).into());
                }
            };
            push_history(&mut history, &solved.x);
            let mut artifact = build_artifact(&solved.x, &history);
            append_termination_metric(&mut artifact, &solved);
            artifact.solver = merge_compile_solver_report(
                crate::common::sqp_solver_report(&solved, &options),
                &running_solver,
                format_sqp_settings_summary(&options),
            );
            if stream_events {
                emit(SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                });
            }
            Ok(artifact)
        }
        SolverMethod::Nlip => {
            let options = nlip_options(solver_config);
            let solved = match compiled.solve_interior_point_with_callback(
                x0,
                parameters,
                bounds,
                &options,
                |snapshot| {
                    push_history(&mut history, &snapshot.x);
                    let mut artifact = build_artifact(&snapshot.x, &history);
                    let progress = nlip_progress(snapshot);
                    artifact.solver = running_solver
                        .clone()
                        .with_iterations(progress.iteration)
                        .with_solve_seconds(0.0)
                        .with_phase_details(running_solver.phase_details.clone());
                    if stream_events {
                        emit(SolveStreamEvent::Iteration { progress, artifact });
                    }
                },
            ) {
                Ok(summary) => summary,
                Err(error) => {
                    if stream_events {
                        emit_solver_failure_status(
                            &mut emit,
                            solver_method,
                            nlip_failure_solver_report(&error, &running_solver, &options),
                        );
                    }
                    return Err(InteriorPointSolveError::from(error).into());
                }
            };
            push_history(&mut history, &solved.x);
            let mut artifact = build_artifact(&solved.x, &history);
            append_nlip_termination_metric(&mut artifact, &solved);
            artifact.solver = merge_compile_solver_report(
                crate::common::nlip_solver_report(&solved, &options),
                &running_solver,
                format_nlip_settings_summary(&options),
            );
            if stream_events {
                emit(SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                });
            }
            Ok(artifact)
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let options = ipopt_options(solver_config);
            let solved = match compiled.solve_ipopt_with_callback(
                x0,
                parameters,
                bounds,
                &options,
                |snapshot| {
                    push_history(&mut history, &snapshot.x);
                    let mut artifact = build_artifact(&snapshot.x, &history);
                    let progress = ipopt_progress(snapshot);
                    artifact.solver = running_solver
                        .clone()
                        .with_iterations(progress.iteration)
                        .with_solve_seconds(0.0)
                        .with_phase_details(running_solver.phase_details.clone());
                    if stream_events {
                        emit(SolveStreamEvent::Iteration { progress, artifact });
                    }
                },
            ) {
                Ok(summary) => summary,
                Err(error) => {
                    if stream_events {
                        emit_solver_failure_status(
                            &mut emit,
                            solver_method,
                            ipopt_failure_solver_report(&error, &running_solver, &options),
                        );
                    }
                    return Err(IpoptSolveError::from(error).into());
                }
            };
            push_history(&mut history, &solved.x);
            let mut artifact = build_artifact(&solved.x, &history);
            append_ipopt_termination_metric(&mut artifact, &solved);
            artifact.solver = merge_compile_solver_report(
                crate::common::ipopt_solver_report(&solved, &options),
                &running_solver,
                format_ipopt_settings_summary(&options),
            );
            if stream_events {
                emit(SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                });
            }
            Ok(artifact)
        }
    }
}

fn emit_solver_failure_status<F>(
    emit: &mut F,
    solver_method: SolverMethod,
    report: Option<SolverReport>,
) where
    F: FnMut(SolveStreamEvent),
{
    if let Some(report) = report {
        emit(SolveStreamEvent::Status {
            status: SolveStatus {
                stage: SolveStage::Solving,
                solver_method: Some(solver_method),
                solver: report,
            },
        });
    }
}

fn merge_compile_solver_report(
    mut report: SolverReport,
    running_solver: &SolverReport,
    settings: String,
) -> SolverReport {
    report.symbolic_setup_s = report.symbolic_setup_s.or(running_solver.symbolic_setup_s);
    report.jit_s = report.jit_s.or(running_solver.jit_s);
    report.compile_cached = report.compile_cached || running_solver.compile_cached;
    report.jit_disk_cache_hit = report.jit_disk_cache_hit || running_solver.jit_disk_cache_hit;
    let mut phase_details = running_solver.phase_details.clone();
    let mut solve_details = report.phase_details.solve;
    if !solve_details
        .iter()
        .any(|detail| detail.label == "Settings")
    {
        solve_details.insert(0, phase_detail("Settings", settings, 0));
    }
    phase_details.solve = solve_details;
    report.phase_details = phase_details;
    report
}

fn push_history(history: &mut Vec<Vec<f64>>, x: &[f64]) {
    if history
        .last()
        .is_some_and(|last| last.len() == x.len() && max_abs_delta(last, x) <= 1.0e-13)
    {
        return;
    }
    history.push(x.to_vec());
}

fn max_abs_delta(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()))
}

fn chain_points_from_flat(x: &[f64], span: f64) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(CHAIN_LINKS + 1);
    points.push((0.0, 0.0));
    points.extend(x.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])));
    points.push((span, 0.0));
    points
}

fn chain_link_residuals(points: &[(f64, f64)]) -> Vec<f64> {
    points
        .windows(2)
        .map(|window| {
            let dx = window[1].0 - window[0].0;
            let dy = window[1].1 - window[0].1;
            dx * dx + dy * dy - CHAIN_LINK_LENGTH * CHAIN_LINK_LENGTH
        })
        .collect()
}

fn chain_potential(points: &[(f64, f64)]) -> f64 {
    points
        .iter()
        .skip(1)
        .take(CHAIN_POINTS)
        .map(|(_, y)| *y)
        .sum()
}

fn chain_lowest_y(points: &[(f64, f64)]) -> f64 {
    points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min)
}

fn chain_max_link_error(points: &[(f64, f64)]) -> f64 {
    chain_link_residuals(points)
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max)
}

fn panel_severity(value: f64, tolerance: f64) -> ConstraintPanelSeverity {
    if value <= tolerance {
        ConstraintPanelSeverity::FullAccuracy
    } else if value <= 100.0 * tolerance {
        ConstraintPanelSeverity::ReducedAccuracy
    } else {
        ConstraintPanelSeverity::Violated
    }
}

fn equality_entry(
    label: impl Into<String>,
    violation: f64,
    tolerance: f64,
) -> ConstraintPanelEntry {
    ConstraintPanelEntry {
        label: label.into(),
        category: ConstraintPanelCategory::Path,
        worst_violation: violation.abs(),
        violating_instances: usize::from(violation.abs() > tolerance),
        total_instances: 1,
        severity: panel_severity(violation.abs(), tolerance),
        lower_bound: None,
        upper_bound: None,
        lower_severity: None,
        upper_severity: None,
    }
}

fn inequality_entry(
    label: impl Into<String>,
    value: f64,
    upper_bound: f64,
    tolerance: f64,
) -> ConstraintPanelEntry {
    let violation = (value - upper_bound).max(0.0);
    ConstraintPanelEntry {
        label: label.into(),
        category: ConstraintPanelCategory::Path,
        worst_violation: violation,
        violating_instances: usize::from(violation > tolerance),
        total_instances: 1,
        severity: panel_severity(violation, tolerance),
        lower_bound: None,
        upper_bound: Some(upper_bound),
        lower_severity: None,
        upper_severity: Some(panel_severity(violation, tolerance)),
    }
}

fn hanging_chain_constraint_panels(points: &[(f64, f64)], tolerance: f64) -> ConstraintPanels {
    let mut equalities = chain_link_residuals(points)
        .into_iter()
        .enumerate()
        .map(|(idx, residual)| equality_entry(format!("link {}", idx + 1), residual, tolerance))
        .collect::<Vec<_>>();
    equalities.sort_by(|left, right| right.worst_violation.total_cmp(&left.worst_violation));
    ConstraintPanels {
        equalities,
        inequalities: Vec::new(),
    }
}

fn rosenbrock_constraint_panels(
    params: &RosenbrockParams,
    x: &[f64],
    tolerance: f64,
) -> ConstraintPanels {
    let Some(radius_sq) = params.variant.disk_radius_sq() else {
        return ConstraintPanels::default();
    };
    let value = x[0] * x[0] + x[1] * x[1];
    ConstraintPanels {
        equalities: Vec::new(),
        inequalities: vec![inequality_entry("disk radius", value, radius_sq, tolerance)],
    }
}

fn line_series(name: impl Into<String>, x: Vec<f64>, y: Vec<f64>) -> TimeSeries {
    TimeSeries {
        name: name.into(),
        x,
        y,
        mode: Some(PlotMode::LinesMarkers),
        legend_group: None,
        show_legend: true,
        role: TimeSeriesRole::Data,
    }
}

fn hanging_chain_artifact(
    params: &HangingChainParams,
    initial_x: &[f64],
    x: &[f64],
    history: &[Vec<f64>],
) -> SolveArtifact {
    let span = chain_span(params);
    let initial_points = chain_points_from_flat(initial_x, span);
    let points = chain_points_from_flat(x, span);
    let initial_residual = chain_max_link_error(&initial_points);
    let final_residual = chain_max_link_error(&points);
    let lowest_y = chain_lowest_y(&points);
    let potential = chain_potential(&points);
    let link_index = (1..=CHAIN_LINKS).map(|idx| idx as f64).collect::<Vec<_>>();
    let residuals = chain_link_residuals(&points);
    let node_index = (0..=CHAIN_LINKS).map(|idx| idx as f64).collect::<Vec<_>>();
    let initial_y = initial_points.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    let final_y = points.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    let iteration = (0..history.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let lowest_history = history
        .iter()
        .map(|state| chain_lowest_y(&chain_points_from_flat(state, span)))
        .collect::<Vec<_>>();
    let residual_history = history
        .iter()
        .map(|state| chain_max_link_error(&chain_points_from_flat(state, span)))
        .collect::<Vec<_>>();

    let mut artifact = SolveArtifact::new(
        "Static Hanging Chain",
        vec![
            metric_with_key(
                MetricKey::Custom,
                "Initial Shape",
                params.initial_condition.label(),
            ),
            numeric_metric_with_key(
                MetricKey::Custom,
                "Lowest Y",
                lowest_y,
                format!("{lowest_y:.3}"),
            ),
            numeric_metric_with_key(
                MetricKey::Custom,
                "Potential",
                potential,
                format!("{potential:.3}"),
            ),
            numeric_metric_with_key(
                MetricKey::Custom,
                "Max Link Error",
                final_residual,
                format!("{final_residual:.2e}"),
            ),
        ],
        SolverReport::placeholder(),
        vec![
            Chart {
                title: "Chain Heights".to_string(),
                x_label: "Node".to_string(),
                y_label: "y".to_string(),
                series: vec![
                    line_series("Initial", node_index.clone(), initial_y),
                    line_series("Optimized", node_index, final_y),
                ],
            },
            Chart {
                title: "Link Length Residuals".to_string(),
                x_label: "Link".to_string(),
                y_label: "squared length error".to_string(),
                series: vec![line_series("Residual", link_index, residuals)],
            },
            Chart {
                title: "Static Convergence".to_string(),
                x_label: "Iteration".to_string(),
                y_label: "value".to_string(),
                series: vec![
                    line_series("Lowest Y", iteration.clone(), lowest_history),
                    line_series("Max Link Error", iteration, residual_history),
                ],
            },
        ],
        Scene2D {
            title: "Chain Geometry".to_string(),
            x_label: "x".to_string(),
            y_label: "y".to_string(),
            paths: vec![
                ScenePath {
                    name: "Initial".to_string(),
                    x: initial_points.iter().map(|(x, _)| *x).collect(),
                    y: initial_points.iter().map(|(_, y)| *y).collect(),
                },
                ScenePath {
                    name: "Optimized".to_string(),
                    x: points.iter().map(|(x, _)| *x).collect(),
                    y: points.iter().map(|(_, y)| *y).collect(),
                },
            ],
            circles: vec![
                SceneCircle {
                    cx: 0.0,
                    cy: 0.0,
                    radius: 0.08,
                    label: "left anchor".to_string(),
                },
                SceneCircle {
                    cx: span,
                    cy: 0.0,
                    radius: 0.08,
                    label: "right anchor".to_string(),
                },
            ],
            arrows: Vec::new(),
            animation: None,
        },
        vec![
            format!("Initial max squared-link residual: {initial_residual:.2e}."),
            "The optimization objective lowers the interior nodes while the equality constraints keep every segment at the same link length.".to_string(),
        ],
    );
    artifact.constraint_panels =
        hanging_chain_constraint_panels(&points, params.solver.constraint_tol);
    artifact
        .visualizations
        .push(ArtifactVisualization::Paths3D {
            title: "Chain Convergence".to_string(),
            x_label: "x".to_string(),
            y_label: "y".to_string(),
            z_label: "iteration".to_string(),
            paths: chain_history_paths_3d(history, span),
        });
    artifact
}

fn chain_history_paths_3d(history: &[Vec<f64>], span: f64) -> Vec<ScenePath3D> {
    let indices = sampled_history_indices(history.len(), MAX_3D_CHAIN_FRAMES);
    indices
        .into_iter()
        .map(|idx| {
            let points = chain_points_from_flat(&history[idx], span);
            ScenePath3D {
                name: format!("iter {idx}"),
                x: points.iter().map(|(x, _)| *x).collect(),
                y: points.iter().map(|(_, y)| *y).collect(),
                z: vec![idx as f64; points.len()],
            }
        })
        .collect()
}

fn sampled_history_indices(len: usize, max_count: usize) -> Vec<usize> {
    if len <= max_count {
        return (0..len).collect();
    }
    let step = (len - 1) as f64 / (max_count - 1) as f64;
    let mut indices = (0..max_count)
        .map(|idx| (idx as f64 * step).round() as usize)
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if indices.last().copied() != Some(len - 1) {
        indices.push(len - 1);
    }
    indices
}

fn rosenbrock_objective(coeffs: &RosenbrockCoefficients<f64>, x: f64, y: f64) -> f64 {
    (coeffs.a - x).powi(2) + coeffs.b * (y - x * x).powi(2) + coeffs.tilt_x * x + coeffs.tilt_y * y
}

fn rosenbrock_artifact(
    params: &RosenbrockParams,
    x: &[f64],
    history: &[Vec<f64>],
) -> SolveArtifact {
    let coeffs = params.variant.coefficients();
    let objective = rosenbrock_objective(&coeffs, x[0], x[1]);
    let distance_to_classic = ((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)).sqrt();
    let iteration = (0..history.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let x_history = history.iter().map(|state| state[0]).collect::<Vec<_>>();
    let y_history = history.iter().map(|state| state[1]).collect::<Vec<_>>();
    let f_history = history
        .iter()
        .map(|state| rosenbrock_objective(&coeffs, state[0], state[1]))
        .collect::<Vec<_>>();
    let disk_value = x[0] * x[0] + x[1] * x[1];
    let disk_metric = params.variant.disk_radius_sq().map(|radius_sq| {
        numeric_metric_with_key(
            MetricKey::Custom,
            "Disk x^2+y^2",
            disk_value,
            format!("{disk_value:.4} / {radius_sq:.1}"),
        )
    });
    let mut summary = vec![
        metric_with_key(MetricKey::Custom, "Variant", params.variant.label()),
        numeric_metric_with_key(
            MetricKey::Custom,
            "Objective",
            objective,
            format!("{objective:.4e}"),
        ),
        numeric_metric_with_key(MetricKey::FinalX, "x", x[0], format!("{:.5}", x[0])),
        numeric_metric_with_key(MetricKey::FinalY, "y", x[1], format!("{:.5}", x[1])),
        numeric_metric_with_key(
            MetricKey::Custom,
            "Distance to (1,1)",
            distance_to_classic,
            format!("{distance_to_classic:.3e}"),
        ),
    ];
    if let Some(metric) = disk_metric {
        summary.push(metric);
    }

    let mut artifact = SolveArtifact::new(
        "Rosenbrock Variants",
        summary,
        SolverReport::placeholder(),
        vec![
            Chart {
                title: "Decision Variables".to_string(),
                x_label: "Iteration".to_string(),
                y_label: "value".to_string(),
                series: vec![
                    line_series("x", iteration.clone(), x_history.clone()),
                    line_series("y", iteration.clone(), y_history.clone()),
                ],
            },
            Chart {
                title: "Objective History".to_string(),
                x_label: "Iteration".to_string(),
                y_label: "objective".to_string(),
                series: vec![line_series("f(x,y)", iteration, f_history)],
            },
        ],
        Scene2D {
            title: "Iterate Path".to_string(),
            x_label: "x".to_string(),
            y_label: "y".to_string(),
            paths: vec![ScenePath {
                name: "Iterates".to_string(),
                x: x_history.clone(),
                y: y_history.clone(),
            }],
            circles: params.variant.disk_radius_sq().map_or_else(Vec::new, |radius_sq| {
                vec![SceneCircle {
                    cx: 0.0,
                    cy: 0.0,
                    radius: radius_sq.sqrt(),
                    label: "disk constraint".to_string(),
                }]
            }),
            arrows: Vec::new(),
            animation: None,
        },
        vec![
            "The contour plot uses the same accepted-iterate history as the live convergence stream.".to_string(),
            "Narrow valley and disk-constrained variants stress different parts of the line-search and trust-region logic.".to_string(),
        ],
    );
    artifact.constraint_panels =
        rosenbrock_constraint_panels(params, x, params.solver.constraint_tol);
    artifact
        .visualizations
        .push(rosenbrock_contour_visualization(
            params, history, &x_history, &y_history,
        ));
    artifact
}

fn rosenbrock_contour_visualization(
    params: &RosenbrockParams,
    history: &[Vec<f64>],
    x_history: &[f64],
    y_history: &[f64],
) -> ArtifactVisualization {
    let coeffs = params.variant.coefficients();
    let min_x = history
        .iter()
        .map(|state| state[0])
        .fold(-2.0_f64, f64::min)
        .min(-1.6);
    let max_x = history
        .iter()
        .map(|state| state[0])
        .fold(2.0_f64, f64::max)
        .max(1.6);
    let min_y = history
        .iter()
        .map(|state| state[1])
        .fold(-0.8_f64, f64::min)
        .min(-0.6);
    let max_y = history
        .iter()
        .map(|state| state[1])
        .fold(2.2_f64, f64::max)
        .max(2.2);
    let pad_x = 0.1 * (max_x - min_x).max(1.0);
    let pad_y = 0.1 * (max_y - min_y).max(1.0);
    let x_axis = linspace(min_x - pad_x, max_x + pad_x, ROSENBROCK_CONTOUR_POINTS);
    let y_axis = linspace(min_y - pad_y, max_y + pad_y, ROSENBROCK_CONTOUR_POINTS);
    let z = y_axis
        .iter()
        .map(|&y| {
            x_axis
                .iter()
                .map(|&x| rosenbrock_objective(&coeffs, x, y).min(2.0e3))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    ArtifactVisualization::Contour2D {
        title: "Rosenbrock Contours".to_string(),
        x_label: "x".to_string(),
        y_label: "y".to_string(),
        x: x_axis,
        y: y_axis,
        z,
        paths: vec![ScenePath {
            name: "Iterates".to_string(),
            x: x_history.to_vec(),
            y: y_history.to_vec(),
        }],
        circles: params
            .variant
            .disk_radius_sq()
            .map_or_else(Vec::new, |radius_sq| {
                vec![SceneCircle {
                    cx: 0.0,
                    cy: 0.0,
                    radius: radius_sq.sqrt(),
                    label: "disk".to_string(),
                }]
            }),
    }
}

fn linspace(min: f64, max: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![min];
    }
    let step = (max - min) / (count - 1) as f64;
    (0..count).map(|idx| min + idx as f64 * step).collect()
}

fn hanging_chain_compile_cache_statuses() -> Vec<CompileCacheStatus> {
    HANGING_CHAIN_CACHE.with(|cache| {
        let (variant_id, variant_label) = hanging_chain_variant();
        cache
            .borrow()
            .cached_entries()
            .into_iter()
            .map(|(_, compiled)| {
                let compiled = compiled.borrow();
                let summary = summarize_backend_compile_report(compiled.backend_compile_report());
                compile_cache_status(
                    ProblemId::HangingChainStatic,
                    "Static Hanging Chain",
                    &variant_id,
                    &variant_label,
                    compiled.backend_timing_metadata(),
                    compile_report_is_fully_cached(&summary),
                )
            })
            .collect()
    })
}

fn rosenbrock_compile_cache_statuses() -> Vec<CompileCacheStatus> {
    ROSENBROCK_CACHE.with(|cache| {
        let (variant_id, variant_label) = rosenbrock_variant();
        cache
            .borrow()
            .cached_entries()
            .into_iter()
            .map(|(_, compiled)| {
                let compiled = compiled.borrow();
                let summary = summarize_backend_compile_report(compiled.backend_compile_report());
                compile_cache_status(
                    ProblemId::RosenbrockVariants,
                    "Rosenbrock Variants",
                    &variant_id,
                    &variant_label,
                    compiled.backend_timing_metadata(),
                    compile_report_is_fully_cached(&summary),
                )
            })
            .collect()
    })
}

fn hanging_chain_solve_from_map(values: &BTreeMap<String, f64>) -> Result<SolveArtifact> {
    hanging_chain_solve(&HangingChainParams::from_map(values)?)
}

fn hanging_chain_prewarm_from_map(values: &BTreeMap<String, f64>) -> Result<()> {
    hanging_chain_prewarm(&HangingChainParams::from_map(values)?)
}

fn hanging_chain_solve_with_progress_boxed(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<SolveArtifact> {
    hanging_chain_solve_with_progress(&HangingChainParams::from_map(values)?, emit)
}

fn hanging_chain_prewarm_with_progress_boxed(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<()> {
    hanging_chain_prewarm_with_progress(&HangingChainParams::from_map(values)?, emit)
}

fn rosenbrock_solve_from_map(values: &BTreeMap<String, f64>) -> Result<SolveArtifact> {
    rosenbrock_solve(&RosenbrockParams::from_map(values)?)
}

fn rosenbrock_prewarm_from_map(values: &BTreeMap<String, f64>) -> Result<()> {
    rosenbrock_prewarm(&RosenbrockParams::from_map(values)?)
}

fn rosenbrock_solve_with_progress_boxed(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<SolveArtifact> {
    rosenbrock_solve_with_progress(&RosenbrockParams::from_map(values)?, emit)
}

fn rosenbrock_prewarm_with_progress_boxed(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<()> {
    rosenbrock_prewarm_with_progress(&RosenbrockParams::from_map(values)?, emit)
}
