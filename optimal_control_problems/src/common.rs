use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::{Result, anyhow};
use optimal_control::{
    Bounds1D, CollocationFamily, ControllerFn, InterpolatedTrajectory, IntervalArc,
    OcpCompileHelperKind, OcpCompileOptions, OcpCompileProgress, OcpConstraintCategory,
    OcpConstraintViolationReport, OcpHelperCompileStats, OcpKernelFunctionOptions,
    OcpSymbolicFunctionOptions, runtime as ocp_runtime,
};
use optimization::{
    BackendCompileReport, BackendTimingMetadata, CallPolicy, CallPolicyConfig, ClarabelSqpError,
    ClarabelSqpOptions, ClarabelSqpProfiling, ClarabelSqpSummary, ConstraintSatisfaction,
    FilterAcceptanceMode, FiniteDifferenceValidationOptions, FunctionCompileOptions,
    InteriorPointIterationSnapshot, InteriorPointLinearSolver, InteriorPointOptions,
    InteriorPointProfiling, InteriorPointSolveError, InteriorPointSpralPivotMethod,
    InteriorPointSummary, LineSearchFilterOptions, LineSearchMeritOptions, LlvmOptimizationLevel,
    NlpCompileStats, NlpDerivativeValidationReport, NlpEvaluationBenchmark,
    NlpEvaluationBenchmarkOptions, NlpEvaluationKernelKind, SqpFilterOptions, SqpGlobalization,
    SqpIterationEvent, SqpIterationPhase, SqpIterationSnapshot, SqpLineSearchOptions,
    SqpTrustRegionOptions, SymbolicSetupProfile, TrustRegionFilterOptions, TrustRegionMeritOptions,
    ValidationTolerances, Vectorize, format_nlip_settings_summary, format_sqp_settings_summary,
};
#[cfg(feature = "ipopt")]
use optimization::{
    IpoptLinearSolver, IpoptOptions, IpoptRawOption, IpoptRawStatus, IpoptSummary,
    format_ipopt_settings_summary,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptProfiling, IpoptSolveError};
use serde::{Deserialize, Serialize};
use sx_core::{HessianStrategy, SX};

#[derive(Clone, Debug, Serialize)]
pub struct ControlChoice {
    pub value: f64,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlSection {
    Transcription,
    Solver,
    #[default]
    Problem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlPanel {
    SxFunctions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEditor {
    #[default]
    Slider,
    Select,
    Text,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlVisibility {
    #[default]
    Always,
    DirectCollocationOnly,
    MultipleShootingOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlSemantic {
    TranscriptionMethod,
    TranscriptionIntervals,
    CollocationFamily,
    CollocationDegree,
    TimeGrid,
    TimeGridStrength,
    TimeGridFocusCenter,
    TimeGridFocusWidth,
    TimeGridBreakpoint,
    TimeGridFirstIntervalFraction,
    SolverMethod,
    SolverProfile,
    SolverGlobalization,
    SolverMaxIterations,
    SolverOverallTolerance,
    SolverHessianRegularization,
    SolverNlipLinearSolver,
    SolverNlipSpralPivotMethod,
    SolverNlipSpralZeroPivotAction,
    SolverNlipSpralSmallPivot,
    SolverNlipSpralPivotU,
    SolverDualTolerance,
    SolverConstraintTolerance,
    SolverComplementarityTolerance,
    SolverExactMeritPenalty,
    SolverPenaltyIncreaseFactor,
    SolverMaxPenaltyUpdates,
    SolverArmijoC1,
    SolverWolfeC2,
    SolverLineSearchBeta,
    SolverLineSearchMaxSteps,
    SolverMinStep,
    SolverFilterGammaObjective,
    SolverFilterGammaViolation,
    SolverFilterThetaMaxFactor,
    SolverFilterSwitchingReferenceMin,
    SolverFilterSwitchingViolationFactor,
    SolverFilterSwitchingLinearizedReductionFactor,
    SolverTrustRegionInitialRadius,
    SolverTrustRegionMaxRadius,
    SolverTrustRegionMinRadius,
    SolverTrustRegionShrinkFactor,
    SolverTrustRegionGrowFactor,
    SolverTrustRegionAcceptRatio,
    SolverTrustRegionExpandRatio,
    SolverTrustRegionBoundaryFraction,
    SolverTrustRegionMaxContractions,
    SolverTrustRegionFixedPenalty,
    SxFunctionOption,
    #[default]
    ProblemParameter,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlValueDisplay {
    #[default]
    Scalar,
    Integer,
    Scientific,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ControlSpec {
    pub id: String,
    pub label: String,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default: f64,
    pub unit: String,
    pub help: String,
    #[serde(default)]
    pub section: ControlSection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel: Option<ControlPanel>,
    #[serde(default)]
    pub editor: ControlEditor,
    #[serde(default)]
    pub visibility: ControlVisibility,
    #[serde(default)]
    pub semantic: ControlSemantic,
    #[serde(default)]
    pub value_display: ControlValueDisplay,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<ControlChoice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub profile_defaults: Vec<ControlProfileDefault>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ControlProfileDefault {
    pub profile: f64,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LatexSection {
    pub title: String,
    pub entries: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemId {
    OptimalDistanceGlider,
    LinearSManeuver,
    SailboatUpwind,
    CraneTransfer,
    HangingChainStatic,
    RosenbrockVariants,
}

impl ProblemId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OptimalDistanceGlider => "optimal_distance_glider",
            Self::LinearSManeuver => "linear_s_maneuver",
            Self::SailboatUpwind => "sailboat_upwind",
            Self::CraneTransfer => "crane_transfer",
            Self::HangingChainStatic => "hanging_chain_static",
            Self::RosenbrockVariants => "rosenbrock_variants",
        }
    }
}

impl std::str::FromStr for ProblemId {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "optimal_distance_glider" => Ok(Self::OptimalDistanceGlider),
            "linear_s_maneuver" => Ok(Self::LinearSManeuver),
            "sailboat_upwind" => Ok(Self::SailboatUpwind),
            "crane_transfer" => Ok(Self::CraneTransfer),
            "hanging_chain_static" => Ok(Self::HangingChainStatic),
            "rosenbrock_variants" => Ok(Self::RosenbrockVariants),
            _ => Err(format!("unknown problem `{value}`")),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProblemSpec {
    pub id: ProblemId,
    pub name: String,
    pub description: String,
    pub controls: Vec<ControlSpec>,
    pub math_sections: Vec<LatexSection>,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompileCacheState {
    Warming,
    Ready,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompileCacheStatus {
    pub problem_id: ProblemId,
    pub problem_name: String,
    pub variant_id: String,
    pub variant_label: String,
    pub state: CompileCacheState,
    pub symbolic_setup_s: Option<f64>,
    pub jit_s: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub jit_disk_cache_hit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DirectCollocationCompileKey {
    Legendre,
    RadauIia,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeGridCompileKey {
    Uniform,
    Cosine {
        strength_bits: u64,
    },
    Tanh {
        strength_bits: u64,
    },
    Geometric {
        strength_bits: u64,
        bias: ocp_runtime::TimeGridBias,
    },
    Focus {
        center_bits: u64,
        width_bits: u64,
        strength_bits: u64,
    },
    Piecewise {
        breakpoint_bits: u64,
        first_interval_fraction_bits: u64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKey {
    #[default]
    Custom,
    TranscriptionMethod,
    IntervalCount,
    CollocationNodeCount,
    TimeGrid,
    Termination,
    Distance,
    FinalTime,
    BestGlideAlpha,
    TerminalLiftToDrag,
    PeakAltitude,
    TrimCost,
    FinalX,
    FinalY,
    MaxY,
    MinY,
    PeakJerk,
    TransferTime,
    TargetX,
    MaxSwing,
    MaxAccel,
    MaxJerk,
    Duration,
    UpwindTarget,
    UpwindDistance,
    MaxSpeed,
    TackCount,
    CenterlineError,
    MaxCrossTrack,
}

#[derive(Clone, Debug, Serialize)]
pub struct Metric {
    pub key: MetricKey,
    pub label: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric_value: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeSeriesRole {
    #[default]
    Data,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Debug, Serialize)]
pub struct TimeSeries {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<PlotMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legend_group: Option<String>,
    #[serde(skip_serializing_if = "is_true")]
    pub show_legend: bool,
    #[serde(default)]
    pub role: TimeSeriesRole,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlotMode {
    Lines,
    LinesMarkers,
    Markers,
}

#[derive(Clone, Debug, Serialize)]
pub struct Chart {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<TimeSeries>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScenePath {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SceneCircle {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    pub label: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SceneArrow {
    pub x: f64,
    pub y: f64,
    pub dx: f64,
    pub dy: f64,
    pub label: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SceneFrame {
    pub points: BTreeMap<String, [f64; 2]>,
    pub segments: Vec<([f64; 2], [f64; 2])>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SceneAnimation {
    pub times: Vec<f64>,
    pub frames: Vec<SceneFrame>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Scene2D {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub paths: Vec<ScenePath>,
    pub circles: Vec<SceneCircle>,
    pub arrows: Vec<SceneArrow>,
    pub animation: Option<SceneAnimation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScenePath3D {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArtifactVisualization {
    #[serde(rename = "contour_2d")]
    Contour2D {
        title: String,
        x_label: String,
        y_label: String,
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<Vec<f64>>,
        paths: Vec<ScenePath>,
        circles: Vec<SceneCircle>,
    },
    #[serde(rename = "paths_3d")]
    Paths3D {
        title: String,
        x_label: String,
        y_label: String,
        z_label: String,
        paths: Vec<ScenePath3D>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveArtifact {
    pub title: String,
    pub summary: Vec<Metric>,
    pub solver: SolverReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compile_report: Option<CompileReportSummary>,
    #[serde(default)]
    pub constraint_panels: ConstraintPanels,
    pub charts: Vec<Chart>,
    pub scene: Scene2D,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visualizations: Vec<ArtifactVisualization>,
    pub notes: Vec<String>,
}

impl SolveArtifact {
    pub fn new(
        title: impl Into<String>,
        summary: Vec<Metric>,
        solver: SolverReport,
        charts: Vec<Chart>,
        scene: Scene2D,
        notes: Vec<String>,
    ) -> Self {
        Self {
            title: title.into(),
            summary,
            solver,
            compile_report: None,
            constraint_panels: ConstraintPanels::default(),
            charts,
            scene,
            visualizations: Vec::new(),
            notes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintPanelSeverity {
    FullAccuracy,
    ReducedAccuracy,
    #[default]
    Violated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintPanelCategory {
    BoundaryEquality,
    BoundaryInequality,
    Path,
    ContinuityState,
    ContinuityControl,
    CollocationState,
    CollocationControl,
    FinalTime,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConstraintPanelEntry {
    pub label: String,
    pub category: ConstraintPanelCategory,
    pub worst_violation: f64,
    pub violating_instances: usize,
    pub total_instances: usize,
    pub severity: ConstraintPanelSeverity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_bound: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upper_bound: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_severity: Option<ConstraintPanelSeverity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upper_severity: Option<ConstraintPanelSeverity>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ConstraintPanels {
    #[serde(default)]
    pub equalities: Vec<ConstraintPanelEntry>,
    #[serde(default)]
    pub inequalities: Vec<ConstraintPanelEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolvePhase {
    Initial,
    AcceptedStep,
    PostConvergence,
    Converged,
    Regular,
    Restoration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveLogLevel {
    Console,
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveFilterPoint {
    pub objective: f64,
    pub violation: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveFilterAcceptanceMode {
    ObjectiveArmijo,
    ViolationReduction,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveFilterInfo {
    pub current: SolveFilterPoint,
    pub entries: Vec<SolveFilterPoint>,
    pub objective_label: &'static str,
    pub title: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_mode: Option<SolveFilterAcceptanceMode>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveTrustRegionInfo {
    pub radius: f64,
    pub attempted_radius: f64,
    pub step_norm: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub largest_attempted_step_norm: Option<f64>,
    pub contraction_count: usize,
    pub qp_failure_retries: usize,
    pub boundary_active: bool,
    pub restoration_attempted: bool,
    pub elastic_recovery_attempted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveProgress {
    pub iteration: usize,
    pub phase: SolvePhase,
    pub objective: f64,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub dual_inf: f64,
    pub step_inf: Option<f64>,
    pub penalty: f64,
    pub alpha: Option<f64>,
    pub line_search_iterations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<SolveFilterInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust_region: Option<SolveTrustRegionInfo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveStage {
    SymbolicSetup,
    JitCompilation,
    Solving,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveStatus {
    pub stage: SolveStage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_method: Option<SolverMethod>,
    pub solver: SolverReport,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolveStreamEvent {
    Status {
        status: SolveStatus,
    },
    Log {
        line: String,
        level: SolveLogLevel,
    },
    Iteration {
        progress: SolveProgress,
        artifact: SolveArtifact,
    },
    Final {
        artifact: SolveArtifact,
    },
    Error {
        message: String,
    },
}

pub fn emit_solve_status<F>(
    emit: &mut F,
    stage: SolveStage,
    solver_method: Option<SolverMethod>,
    solver: SolverReport,
) where
    F: FnMut(SolveStreamEvent),
{
    emit(SolveStreamEvent::Status {
        status: SolveStatus {
            stage,
            solver_method,
            solver,
        },
    });
}

pub fn emit_symbolic_setup_status<F>(emit: &mut F)
where
    F: FnMut(SolveStreamEvent),
{
    emit_solve_status(
        emit,
        SolveStage::SymbolicSetup,
        None,
        SolverReport::in_progress("Setting up symbolic model..."),
    );
}

#[derive(Debug)]
struct LatestOnlyState<T> {
    pending: Option<T>,
    closed: bool,
}

struct LatestOnlySender<T> {
    shared: Arc<(Mutex<LatestOnlyState<T>>, Condvar)>,
}

impl<T> Clone for LatestOnlySender<T> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl<T> LatestOnlySender<T> {
    fn submit(&self, value: T) {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().expect("latest-only state poisoned");
        state.pending = Some(value);
        wake.notify_one();
    }

    fn close(&self) {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().expect("latest-only state poisoned");
        state.closed = true;
        wake.notify_all();
    }
}

fn with_latest_only_worker<T, F, R>(mut process: F, run: impl FnOnce(LatestOnlySender<T>) -> R) -> R
where
    T: Send,
    F: FnMut(T) + Send,
{
    thread::scope(|scope| {
        let shared = Arc::new((
            Mutex::new(LatestOnlyState {
                pending: None,
                closed: false,
            }),
            Condvar::new(),
        ));
        let sender = LatestOnlySender { shared };
        let worker_sender = sender.clone();
        let worker = scope.spawn(move || {
            let (lock, wake) = &*worker_sender.shared;
            loop {
                let next = {
                    let mut state = lock.lock().expect("latest-only state poisoned");
                    while state.pending.is_none() && !state.closed {
                        state = wake.wait(state).expect("latest-only state poisoned");
                    }
                    if let Some(value) = state.pending.take() {
                        Some(value)
                    } else if state.closed {
                        None
                    } else {
                        unreachable!("latest-only worker woke without data or close")
                    }
                };
                match next {
                    Some(value) => process(value),
                    None => break,
                }
            }
        });
        let result = run(sender.clone());
        sender.close();
        worker.join().expect("latest-only worker panicked");
        result
    })
}

fn emit_event<F>(emit: &Arc<Mutex<F>>, event: SolveStreamEvent)
where
    F: FnMut(SolveStreamEvent),
{
    let mut callback = emit.lock().expect("stream emitter poisoned");
    (*callback)(event);
}

fn emit_failure_status<F>(emit: &Arc<Mutex<F>>, solver_method: SolverMethod, solver: SolverReport)
where
    F: FnMut(SolveStreamEvent),
{
    emit_event(
        emit,
        SolveStreamEvent::Status {
            status: SolveStatus {
                stage: SolveStage::Solving,
                solver_method: Some(solver_method),
                solver,
            },
        },
    );
}

fn emit_log_lines<F>(emit: &Arc<Mutex<F>>, lines: impl IntoIterator<Item = String>)
where
    F: FnMut(SolveStreamEvent),
{
    for line in lines {
        emit_event(
            emit,
            SolveStreamEvent::Log {
                line,
                level: SolveLogLevel::Console,
            },
        );
    }
}

fn fmt_diag_sci(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.3e}")
    } else if value.is_nan() {
        "NaN".to_string()
    } else if value.is_sign_positive() {
        "inf".to_string()
    } else {
        "-inf".to_string()
    }
}

fn fmt_diag_opt_sci(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), fmt_diag_sci)
}

fn fmt_diag_opt_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "--",
    }
}

fn fmt_diag_opt_qp_status(value: Option<optimization::SqpQpStatus>) -> &'static str {
    match value {
        Some(optimization::SqpQpStatus::Solved) => "Solved",
        Some(optimization::SqpQpStatus::ReducedAccuracy) => "ReducedAccuracy",
        Some(optimization::SqpQpStatus::Failed) => "Failed",
        None => "--",
    }
}

fn fmt_diag_opt_qp_raw_status(value: Option<&optimization::SqpQpRawStatus>) -> String {
    value.map_or_else(|| "--".to_string(), |status| status.to_string())
}

fn fmt_sqp_phase(phase: SqpIterationPhase) -> &'static str {
    match phase {
        SqpIterationPhase::Initial => "initial",
        SqpIterationPhase::AcceptedStep => "accepted_step",
        SqpIterationPhase::PostConvergence => "post_convergence",
    }
}

fn fmt_nlip_phase(phase: optimization::InteriorPointIterationPhase) -> &'static str {
    match phase {
        optimization::InteriorPointIterationPhase::Initial => "initial",
        optimization::InteriorPointIterationPhase::AcceptedStep => "accepted_step",
        optimization::InteriorPointIterationPhase::Restoration => "restoration",
        optimization::InteriorPointIterationPhase::Converged => "converged",
    }
}

fn fmt_filter_mode(mode: Option<FilterAcceptanceMode>) -> &'static str {
    match mode {
        Some(FilterAcceptanceMode::ObjectiveArmijo) => "objective_armijo",
        Some(FilterAcceptanceMode::ViolationReduction) => "violation_reduction",
        None => "--",
    }
}

fn fmt_sqp_step_kind(kind: Option<optimization::SqpStepKind>) -> &'static str {
    match kind {
        Some(optimization::SqpStepKind::Objective) => "f-step",
        Some(optimization::SqpStepKind::Feasibility) => "h-step",
        Some(optimization::SqpStepKind::Restoration) => "restoration",
        None => "--",
    }
}

fn fmt_nlip_step_kind(kind: Option<optimization::InteriorPointStepKind>) -> &'static str {
    match kind {
        Some(optimization::InteriorPointStepKind::Objective) => "f-step",
        Some(optimization::InteriorPointStepKind::Feasibility) => "h-step",
        Some(optimization::InteriorPointStepKind::Tiny) => "tiny-step",
        None => "--",
    }
}

fn fmt_sqp_globalization(kind: optimization::SqpGlobalizationKind) -> &'static str {
    kind.label()
}

fn append_filter_frontier_lines(lines: &mut Vec<String>, filter: &optimization::FilterInfo) {
    let entries = &filter.entries;
    if entries.is_empty() {
        lines.push("    frontier is empty".to_string());
        return;
    }

    let min_objective = entries
        .iter()
        .map(|entry| entry.objective)
        .fold(f64::INFINITY, f64::min);
    let max_objective = entries
        .iter()
        .map(|entry| entry.objective)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_violation = entries
        .iter()
        .map(|entry| entry.violation)
        .fold(f64::INFINITY, f64::min);
    let max_violation = entries
        .iter()
        .map(|entry| entry.violation)
        .fold(f64::NEG_INFINITY, f64::max);

    let nearest_idx = entries
        .iter()
        .enumerate()
        .min_by(|(_, lhs), (_, rhs)| {
            let lhs_delta = (lhs.violation - filter.current.violation).abs();
            let rhs_delta = (rhs.violation - filter.current.violation).abs();
            lhs_delta
                .partial_cmp(&rhs_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let nearest = &entries[nearest_idx];

    lines.push(format!(
        "    frontier span obj=[{}, {}] violation=[{}, {}]",
        fmt_diag_sci(min_objective),
        fmt_diag_sci(max_objective),
        fmt_diag_sci(min_violation),
        fmt_diag_sci(max_violation),
    ));
    lines.push(format!(
        "    frontier nearest idx={} obj={} violation={} delta_obj={} delta_violation={}",
        nearest_idx,
        fmt_diag_sci(nearest.objective),
        fmt_diag_sci(nearest.violation),
        fmt_diag_sci(filter.current.objective - nearest.objective),
        fmt_diag_sci(filter.current.violation - nearest.violation),
    ));

    if entries.len() <= 8 {
        for (idx, entry) in entries.iter().enumerate() {
            lines.push(format!(
                "    frontier[{idx}] obj={} violation={}",
                fmt_diag_sci(entry.objective),
                fmt_diag_sci(entry.violation),
            ));
        }
        return;
    }

    let mut sample_indices = vec![0, entries.len() / 2, entries.len() - 1];
    if nearest_idx > 0 {
        sample_indices.push(nearest_idx - 1);
    }
    sample_indices.push(nearest_idx);
    if nearest_idx + 1 < entries.len() {
        sample_indices.push(nearest_idx + 1);
    }
    sample_indices.sort_unstable();
    sample_indices.dedup();

    lines.push(format!(
        "    frontier samples ({} omitted):",
        entries.len().saturating_sub(sample_indices.len())
    ));
    for idx in sample_indices {
        let entry = &entries[idx];
        lines.push(format!(
            "      [{idx}] obj={} violation={}",
            fmt_diag_sci(entry.objective),
            fmt_diag_sci(entry.violation),
        ));
    }
}

fn append_sqp_step_diagnostics_lines(
    lines: &mut Vec<String>,
    diagnostics: &optimization::SqpStepDiagnostics,
) {
    lines.push(format!(
        "  step model obj_dir={} merit_dir={} qp_model={} linear_eq_inf={} linear_ineq_inf={} linear_primal={}",
        fmt_diag_sci(diagnostics.objective_directional_derivative),
        fmt_diag_sci(diagnostics.exact_merit_directional_derivative),
        fmt_diag_sci(diagnostics.qp_model_change),
        fmt_diag_opt_sci(diagnostics.linearized_eq_inf),
        fmt_diag_opt_sci(diagnostics.linearized_ineq_inf),
        fmt_diag_sci(diagnostics.linearized_primal_inf),
    ));
    lines.push(format!(
        "  filter current_theta={} theta_max={} switching={} restoration={} elastic_recovery={}",
        fmt_diag_sci(diagnostics.current_violation),
        fmt_diag_sci(diagnostics.theta_max),
        diagnostics.switching_condition_satisfied,
        diagnostics.restoration_phase,
        diagnostics.elastic_recovery_used,
    ));
    if diagnostics.regularization.enabled {
        lines.push(format!(
            "  regularization min_eig={} shift={} shifted_by_analysis={}",
            fmt_diag_sci(diagnostics.regularization.min_eigenvalue),
            fmt_diag_sci(diagnostics.regularization.applied_shift),
            diagnostics.regularization.shifted_by_analysis,
        ));
    } else {
        lines.push("  regularization disabled".to_string());
    }
}

fn append_nlip_direction_diagnostics_lines(
    lines: &mut Vec<String>,
    diagnostics: &optimization::InteriorPointDirectionDiagnostics,
) {
    lines.push(format!(
        "  direction dx_inf={} dlambda_inf={} ds_inf={} dz_inf={}",
        fmt_diag_sci(diagnostics.dx_inf),
        fmt_diag_sci(diagnostics.d_lambda_inf),
        fmt_diag_sci(diagnostics.ds_inf),
        fmt_diag_sci(diagnostics.dz_inf),
    ));
    lines.push(format!(
        "  direction regularization={} primal_shift={} dual_regularization={}",
        fmt_diag_sci(diagnostics.regularization_size),
        fmt_diag_sci(diagnostics.primal_diagonal_shift),
        fmt_diag_sci(diagnostics.dual_regularization),
    ));
    if let Some(limiter) = &diagnostics.alpha_pr_limiter {
        lines.push(format!(
            "  alpha_pr limiter idx={} value={} delta={} alpha={}",
            limiter.index,
            fmt_diag_sci(limiter.value),
            fmt_diag_sci(limiter.direction),
            fmt_diag_sci(limiter.alpha),
        ));
    }
    if let Some(limiter) = &diagnostics.alpha_du_limiter {
        lines.push(format!(
            "  alpha_du limiter idx={} value={} delta={} alpha={}",
            limiter.index,
            fmt_diag_sci(limiter.value),
            fmt_diag_sci(limiter.direction),
            fmt_diag_sci(limiter.alpha),
        ));
    }
}

fn append_sqp_snapshot_lines(
    lines: &mut Vec<String>,
    heading: &str,
    snapshot: &SqpIterationSnapshot,
) {
    lines.push(format!("{heading}:"));
    lines.push(format!(
        "  iter={} phase={} globalization={} obj={} eq_inf={} ineq_inf={} dual_inf={} comp_inf={} step_inf={} penalty={}",
        snapshot.iteration,
        fmt_sqp_phase(snapshot.phase),
        fmt_sqp_globalization(snapshot.globalization),
        fmt_diag_sci(snapshot.objective),
        fmt_diag_opt_sci(snapshot.eq_inf),
        fmt_diag_opt_sci(snapshot.ineq_inf),
        fmt_diag_sci(snapshot.dual_inf),
        fmt_diag_opt_sci(snapshot.comp_inf),
        fmt_diag_opt_sci(snapshot.step_inf),
        fmt_diag_sci(snapshot.penalty),
    ));
    if let Some(filter) = &snapshot.filter {
        lines.push(format!(
            "  filter current_obj={} current_violation={} frontier_entries={} accepted_mode={}",
            fmt_diag_sci(filter.current.objective),
            fmt_diag_sci(filter.current.violation),
            filter.entries.len(),
            fmt_filter_mode(filter.accepted_mode),
        ));
        append_filter_frontier_lines(lines, filter);
    }
    if let Some(line_search) = &snapshot.line_search {
        lines.push(format!(
            "  accepted line search alpha={} ls_it={} armijo={} obj_armijo={} wolfe={} violation={} filter_mode={} step_kind={} theta_ok={} switching={} soc={} soc_attempted={} restoration_attempted={} elastic_attempted={}",
            fmt_diag_sci(line_search.accepted_alpha),
            line_search.backtrack_count,
            line_search.armijo_satisfied,
            fmt_diag_opt_bool(line_search.objective_armijo_satisfied),
            fmt_diag_opt_bool(line_search.wolfe_satisfied),
            line_search.violation_satisfied,
            fmt_filter_mode(line_search.filter_acceptance_mode),
            fmt_sqp_step_kind(line_search.step_kind),
            fmt_diag_opt_bool(line_search.filter_theta_acceptable),
            fmt_diag_opt_bool(line_search.switching_condition_satisfied),
            line_search.second_order_correction_used,
            line_search.second_order_correction_attempted,
            line_search.restoration_attempted,
            line_search.elastic_recovery_attempted,
        ));
    }
    if let Some(trust_region) = &snapshot.trust_region {
        lines.push(format!(
            "  trust_region radius={} attempted_radius={} contractions={} qp_retries={} step_norm={} boundary={} ared={} pred={} rho={} step_kind={} filter_mode={} restoration_attempted={} elastic_attempted={}",
            fmt_diag_sci(trust_region.radius),
            fmt_diag_sci(trust_region.attempted_radius),
            trust_region.contraction_count,
            trust_region.qp_failure_retries,
            fmt_diag_sci(trust_region.step_norm),
            trust_region.boundary_active,
            fmt_diag_sci(trust_region.actual_reduction),
            fmt_diag_sci(trust_region.predicted_reduction),
            fmt_diag_opt_sci(trust_region.ratio),
            fmt_sqp_step_kind(trust_region.step_kind),
            fmt_filter_mode(trust_region.filter_acceptance_mode),
            trust_region.restoration_attempted,
            trust_region.elastic_recovery_attempted,
        ));
    }
    if let Some(diagnostics) = &snapshot.step_diagnostics {
        append_sqp_step_diagnostics_lines(lines, diagnostics);
    }
    if let Some(qp) = &snapshot.qp {
        lines.push(format!(
            "  qp status={:?} raw_status={} iters={} solve_time={} setup_time={}",
            qp.status,
            qp.raw_status,
            qp.iteration_count,
            fmt_diag_sci(qp.solve_time.as_secs_f64()),
            fmt_diag_sci(qp.setup_time.as_secs_f64()),
        ));
    }
}

fn append_sqp_trust_region_failure_lines(
    lines: &mut Vec<String>,
    info: &optimization::SqpTrustRegionInfo,
) {
    lines.push("trust region failure:".to_string());
    lines.push(format!(
        "  radius={} attempted_radius={} contractions={} qp_retries={} step_norm={} boundary={} ared={} pred={} rho={} step_kind={} filter_mode={} restoration_attempted={} elastic_attempted={}",
        fmt_diag_sci(info.radius),
        fmt_diag_sci(info.attempted_radius),
        info.contraction_count,
        info.qp_failure_retries,
        fmt_diag_sci(info.step_norm),
        info.boundary_active,
        fmt_diag_sci(info.actual_reduction),
        fmt_diag_sci(info.predicted_reduction),
        fmt_diag_opt_sci(info.ratio),
        fmt_sqp_step_kind(info.step_kind),
        fmt_filter_mode(info.filter_acceptance_mode),
        info.restoration_attempted,
        info.elastic_recovery_attempted,
    ));
    for trial in &info.rejected_trials {
        lines.push(format!(
            "  reject radius={} step_norm={} boundary={} ared={} pred={} rho={} filter_ok={} filter_dom={} theta_ok={} switching={} filter_obj={} filter_violation={} qp_status={} qp_raw={} restoration={} elastic_attempted={}",
            fmt_diag_sci(trial.radius),
            fmt_diag_sci(trial.step_norm),
            trial.boundary_active,
            fmt_diag_sci(trial.actual_reduction),
            fmt_diag_sci(trial.predicted_reduction),
            fmt_diag_opt_sci(trial.ratio),
            fmt_diag_opt_bool(trial.filter_acceptable),
            fmt_diag_opt_bool(trial.filter_dominated),
            fmt_diag_opt_bool(trial.filter_theta_acceptable),
            fmt_diag_opt_bool(trial.switching_condition_satisfied),
            fmt_diag_opt_bool(trial.filter_sufficient_objective_reduction),
            fmt_diag_opt_bool(trial.filter_sufficient_violation_reduction),
            fmt_diag_opt_qp_status(trial.qp_status),
            fmt_diag_opt_qp_raw_status(trial.qp_raw_status.as_ref()),
            trial.restoration_phase,
            trial.elastic_recovery_attempted,
        ));
    }
}

fn append_sqp_line_search_failure_lines(
    lines: &mut Vec<String>,
    error: &ClarabelSqpError,
    info: &optimization::SqpLineSearchInfo,
) {
    let (directional_derivative, step_inf_norm, penalty) = match error {
        ClarabelSqpError::LineSearchFailed {
            directional_derivative,
            step_inf_norm,
            penalty,
            ..
        } => (*directional_derivative, *step_inf_norm, *penalty),
        ClarabelSqpError::RestorationFailed { step_inf_norm, .. } => {
            (f64::NAN, *step_inf_norm, f64::NAN)
        }
        _ => return,
    };
    let armijo_rejects = info
        .rejected_trials
        .iter()
        .filter(|trial| !trial.armijo_satisfied)
        .count();
    let filter_rejects = info
        .rejected_trials
        .iter()
        .filter(|trial| trial.filter_acceptable == Some(false))
        .count();
    let wolfe_rejects = info
        .rejected_trials
        .iter()
        .filter(|trial| trial.wolfe_satisfied == Some(false))
        .count();
    let violation_rejects = info
        .rejected_trials
        .iter()
        .filter(|trial| !trial.violation_satisfied)
        .count();
    lines.push("line search failure:".to_string());
    lines.push(format!(
        "  directional_derivative={} step_inf={} penalty={} last_alpha={} rejected={} armijo_rejects={} filter_rejects={} wolfe_rejects={} violation_rejects={} step_kind={} switching={} soc_attempted={} restoration_attempted={} elastic_attempted={}",
        fmt_diag_sci(directional_derivative),
        fmt_diag_sci(step_inf_norm),
        fmt_diag_sci(penalty),
        fmt_diag_sci(info.last_tried_alpha),
        info.rejected_trials.len(),
        armijo_rejects,
        filter_rejects,
        wolfe_rejects,
        violation_rejects,
        fmt_sqp_step_kind(info.step_kind),
        fmt_diag_opt_bool(info.switching_condition_satisfied),
        info.second_order_correction_attempted,
        info.restoration_attempted,
        info.elastic_recovery_attempted,
    ));
    if let ClarabelSqpError::LineSearchFailed { context, .. }
    | ClarabelSqpError::RestorationFailed { context, .. } = error
    {
        if let Some(diagnostics) = &context.failed_step_diagnostics {
            append_sqp_step_diagnostics_lines(lines, diagnostics);
        }
    }
    for trial in &info.rejected_trials {
        lines.push(format!(
            "  reject alpha={} merit={} obj={} primal={} eq_inf={} ineq_inf={} armijo={} obj_armijo={} wolfe={} violation={} filter_ok={} filter_dom={} theta_ok={} switching={} filter_obj={} filter_violation={}",
            fmt_diag_sci(trial.alpha),
            fmt_diag_sci(trial.merit),
            fmt_diag_sci(trial.objective),
            fmt_diag_sci(trial.primal_inf),
            fmt_diag_opt_sci(trial.eq_inf),
            fmt_diag_opt_sci(trial.ineq_inf),
            trial.armijo_satisfied,
            fmt_diag_opt_bool(trial.objective_armijo_satisfied),
            fmt_diag_opt_bool(trial.wolfe_satisfied),
            trial.violation_satisfied,
            fmt_diag_opt_bool(trial.filter_acceptable),
            fmt_diag_opt_bool(trial.filter_dominated),
            fmt_diag_opt_bool(trial.filter_theta_acceptable),
            fmt_diag_opt_bool(trial.switching_condition_satisfied),
            fmt_diag_opt_bool(trial.filter_sufficient_objective_reduction),
            fmt_diag_opt_bool(trial.filter_sufficient_violation_reduction),
        ));
    }
}

fn sqp_failure_diagnostic_lines(error: &ClarabelSqpError) -> Vec<String> {
    let context = match error {
        ClarabelSqpError::MaxIterations { context, .. }
        | ClarabelSqpError::QpSolve { context, .. }
        | ClarabelSqpError::UnconstrainedStepSolve { context }
        | ClarabelSqpError::LineSearchFailed { context, .. }
        | ClarabelSqpError::RestorationFailed { context, .. }
        | ClarabelSqpError::Stalled { context, .. }
        | ClarabelSqpError::NonFiniteCallbackOutput { context, .. } => context.as_ref(),
        ClarabelSqpError::InvalidInput(_)
        | ClarabelSqpError::NonFiniteInput { .. }
        | ClarabelSqpError::Setup(_) => return Vec::new(),
    };
    let mut lines = vec![
        String::new(),
        "failure diagnostics (SQP):".to_string(),
        format!("  termination={:?}", context.termination),
    ];
    if let Some(snapshot) = &context.final_state {
        append_sqp_snapshot_lines(&mut lines, "current iterate", snapshot);
    }
    if let Some(snapshot) = &context.last_accepted_state {
        append_sqp_snapshot_lines(&mut lines, "last accepted iterate", snapshot);
    }
    if let Some(info) = &context.failed_line_search {
        append_sqp_line_search_failure_lines(&mut lines, error, info);
    }
    if let Some(info) = &context.failed_trust_region {
        append_sqp_trust_region_failure_lines(&mut lines, info);
    }
    if let Some(qp_failure) = &context.qp_failure {
        lines.push("qp failure:".to_string());
        lines.push(format!(
            "  failed while solving the next {}SQP subproblem",
            if qp_failure.elastic_recovery {
                "elastic-recovery "
            } else {
                ""
            },
        ));
        lines.push(format!(
            "  failed_qp status={:?} raw_status={} iters={} solve_time={} setup_time={}",
            qp_failure.qp_info.status,
            qp_failure.qp_info.raw_status,
            qp_failure.qp_info.iteration_count,
            fmt_diag_sci(qp_failure.qp_info.solve_time.as_secs_f64()),
            fmt_diag_sci(qp_failure.qp_info.setup_time.as_secs_f64()),
        ));
        lines.push(format!(
            "  vars={} constraints={} lin_obj_inf={} rhs_inf={} hdiag_min={} hdiag_max={} elastic_recovery={} cones={}",
            qp_failure.variable_count,
            qp_failure.constraint_count,
            fmt_diag_sci(qp_failure.linear_objective_inf_norm),
            fmt_diag_sci(qp_failure.rhs_inf_norm),
            fmt_diag_sci(qp_failure.hessian_diag_min),
            fmt_diag_sci(qp_failure.hessian_diag_max),
            qp_failure.elastic_recovery,
            qp_failure
                .cones
                .iter()
                .map(|cone| format!("{}:{}", cone.kind, cone.dim))
                .collect::<Vec<_>>()
                .join(","),
        ));
        if let Some(transcript) = &qp_failure.transcript {
            let transcript = transcript.trim_end();
            if !transcript.is_empty() {
                lines.push("clarabel qp transcript:".to_string());
                lines.extend(transcript.lines().map(str::to_owned));
            }
        }
    }
    lines
}

fn append_nlip_snapshot_lines(
    lines: &mut Vec<String>,
    heading: &str,
    snapshot: &InteriorPointIterationSnapshot,
) {
    lines.push(format!("{heading}:"));
    lines.push(format!(
        "  iter={} phase={} obj={} eq_inf={} ineq_inf={} dual_inf={} comp_inf={} mu={} step_inf={} alpha={} ls_it={} step_kind={} step_tag={} watchdog_active={} linear_solver={}",
        snapshot.iteration,
        fmt_nlip_phase(snapshot.phase),
        fmt_diag_sci(snapshot.objective),
        fmt_diag_opt_sci(snapshot.eq_inf),
        fmt_diag_opt_sci(snapshot.ineq_inf),
        fmt_diag_sci(snapshot.dual_inf),
        fmt_diag_opt_sci(snapshot.comp_inf),
        fmt_diag_opt_sci(snapshot.barrier_parameter),
        fmt_diag_opt_sci(snapshot.step_inf),
        fmt_diag_opt_sci(snapshot.alpha),
        snapshot
            .line_search_iterations
            .map_or_else(|| "--".to_string(), |value| value.to_string()),
        fmt_nlip_step_kind(snapshot.step_kind),
        snapshot
            .step_tag
            .map(|tag| tag.to_string())
            .unwrap_or_else(|| "--".to_string()),
        snapshot.watchdog_active,
        snapshot.linear_solver.label(),
    ));
    if !snapshot.events.is_empty() {
        lines.push(format!(
            "  events={}",
            optimization::nlip_event_codes_for_events(&snapshot.events),
        ));
    }
    if let Some(filter) = &snapshot.filter {
        lines.push(format!(
            "  filter current_obj={} current_violation={} frontier_entries={} accepted_mode={}",
            fmt_diag_sci(filter.current.objective),
            fmt_diag_sci(filter.current.violation),
            filter.entries.len(),
            fmt_filter_mode(filter.accepted_mode),
        ));
        append_filter_frontier_lines(lines, filter);
    }
    if let Some(line_search) = &snapshot.line_search {
        lines.push(format!(
            "  accepted line search alpha={} alpha_pr={} alpha_du={} sigma={} ls_it={} filter_mode={} step_kind={} soc={} soc_attempted={} watchdog_active={} watchdog_accepted={} tiny_step={} rejected={}",
            fmt_diag_opt_sci(line_search.accepted_alpha),
            fmt_diag_sci(line_search.initial_alpha_pr),
            fmt_diag_opt_sci(line_search.initial_alpha_du),
            fmt_diag_sci(line_search.sigma),
            line_search.backtrack_count,
            fmt_filter_mode(line_search.filter_acceptance_mode),
            fmt_nlip_step_kind(line_search.step_kind),
            line_search.second_order_correction_used,
            line_search.second_order_correction_attempted,
            line_search.watchdog_active,
            line_search.watchdog_accepted,
            line_search.tiny_step,
            line_search.rejected_trials.len(),
        ));
    }
    if let Some(diagnostics) = &snapshot.direction_diagnostics {
        append_nlip_direction_diagnostics_lines(lines, diagnostics);
    }
}

fn append_nlip_line_search_failure_lines(
    lines: &mut Vec<String>,
    error: &InteriorPointSolveError,
    info: &optimization::InteriorPointLineSearchInfo,
) {
    let InteriorPointSolveError::LineSearchFailed {
        merit,
        mu,
        step_inf_norm,
        ..
    } = error
    else {
        return;
    };
    let positivity_rejects = info
        .rejected_trials
        .iter()
        .filter(|trial| !trial.slack_positive || !trial.multipliers_positive)
        .count();
    let filter_rejects = info
        .rejected_trials
        .iter()
        .filter(|trial| trial.filter_acceptable == Some(false))
        .count();
    lines.push("line search failure:".to_string());
    lines.push(format!(
        "  merit={} mu={} step_inf={} alpha_pr={} alpha_du={} last_alpha={} sigma={} rejected={} positivity_rejects={} filter_rejects={} filter_mode={} soc={} soc_attempted={} watchdog_active={} watchdog_accepted={} tiny_step={}",
        fmt_diag_sci(*merit),
        fmt_diag_sci(*mu),
        fmt_diag_sci(*step_inf_norm),
        fmt_diag_sci(info.initial_alpha_pr),
        fmt_diag_opt_sci(info.initial_alpha_du),
        fmt_diag_sci(info.last_tried_alpha),
        fmt_diag_sci(info.sigma),
        info.rejected_trials.len(),
        positivity_rejects,
        filter_rejects,
        fmt_filter_mode(info.filter_acceptance_mode),
        info.second_order_correction_used,
        info.second_order_correction_attempted,
        info.watchdog_active,
        info.watchdog_accepted,
        info.tiny_step,
    ));
    let InteriorPointSolveError::LineSearchFailed { context, .. } = error else {
        return;
    };
    if let Some(snapshot) = &context.final_state
        && !snapshot.events.is_empty()
    {
        lines.push(format!(
            "  events={}",
            optimization::nlip_event_codes_for_events(&snapshot.events),
        ));
    }
    lines.push(format!(
        "  current_merit={} current_barrier_obj={} current_primal_inf={}",
        fmt_diag_sci(info.current_merit),
        fmt_diag_sci(info.current_barrier_objective),
        fmt_diag_sci(info.current_primal_inf),
    ));
    if let Some(diagnostics) = &context.failed_direction_diagnostics {
        append_nlip_direction_diagnostics_lines(lines, diagnostics);
    }
    for trial in &info.rejected_trials {
        lines.push(format!(
            "  reject alpha={} slack_positive={} multipliers_positive={} merit={} barrier_obj={} primal={} dual={} comp={} mu={} local_filter={} filter_ok={} filter_dom={} filter_obj={} filter_violation={}",
            fmt_diag_sci(trial.alpha),
            trial.slack_positive,
            trial.multipliers_positive,
            fmt_diag_opt_sci(trial.merit),
            fmt_diag_opt_sci(trial.barrier_objective),
            fmt_diag_opt_sci(trial.primal_inf),
            fmt_diag_opt_sci(trial.dual_inf),
            fmt_diag_opt_sci(trial.comp_inf),
            fmt_diag_opt_sci(trial.mu),
            fmt_diag_opt_bool(trial.local_filter_acceptable),
            fmt_diag_opt_bool(trial.filter_acceptable),
            fmt_diag_opt_bool(trial.filter_dominated),
            fmt_diag_opt_bool(trial.filter_sufficient_objective_reduction),
            fmt_diag_opt_bool(trial.filter_sufficient_violation_reduction),
        ));
    }
}

fn append_nlip_linear_solve_failure_lines(
    lines: &mut Vec<String>,
    diagnostics: &optimization::InteriorPointLinearSolveDiagnostics,
) {
    lines.push("linear solve failure:".to_string());
    lines.push(format!(
        "  preferred_solver={} matrix_dim={} attempts={}",
        diagnostics.preferred_solver.label(),
        diagnostics.matrix_dimension,
        diagnostics.attempts.len(),
    ));
    for attempt in &diagnostics.attempts {
        lines.push(format!(
            "  attempt solver={} regularization={} failure={} solution_inf={} solution_limit={} residual_inf={} residual_limit={} detail={}",
            attempt.solver.label(),
            fmt_diag_sci(attempt.regularization),
            attempt.failure_kind.label(),
            fmt_diag_opt_sci(attempt.solution_inf),
            fmt_diag_opt_sci(attempt.solution_inf_limit),
            fmt_diag_opt_sci(attempt.residual_inf),
            fmt_diag_opt_sci(attempt.residual_inf_limit),
            attempt.detail.as_deref().unwrap_or("-"),
        ));
    }
}

fn nlip_failure_diagnostic_lines(error: &InteriorPointSolveError) -> Vec<String> {
    let context = match error {
        InteriorPointSolveError::InvalidInput(_) => return Vec::new(),
        InteriorPointSolveError::LinearSolve { context, .. }
        | InteriorPointSolveError::LineSearchFailed { context, .. }
        | InteriorPointSolveError::RestorationFailed { context, .. }
        | InteriorPointSolveError::LocalInfeasibility { context }
        | InteriorPointSolveError::DivergingIterates { context, .. }
        | InteriorPointSolveError::CpuTimeExceeded { context, .. }
        | InteriorPointSolveError::WallTimeExceeded { context, .. }
        | InteriorPointSolveError::UserRequestedStop { context }
        | InteriorPointSolveError::SearchDirectionTooSmall { context }
        | InteriorPointSolveError::MaxIterations { context, .. } => context.as_ref(),
    };
    let mut lines = vec![String::new(), "failure diagnostics (NLIP):".to_string()];
    match error {
        InteriorPointSolveError::LinearSolve { solver, .. } => lines.push(format!(
            "  termination=linear_solve solver={}",
            solver.label()
        )),
        InteriorPointSolveError::LineSearchFailed { .. } => {
            lines.push("  termination=line_search".to_string())
        }
        InteriorPointSolveError::RestorationFailed { status, .. } => {
            lines.push(format!("  termination=restoration status={status:?}"))
        }
        InteriorPointSolveError::LocalInfeasibility { .. } => {
            lines.push("  termination=local_infeasibility".to_string())
        }
        InteriorPointSolveError::DivergingIterates {
            max_abs_x,
            threshold,
            ..
        } => lines.push(format!(
            "  termination=diverging_iterates max_abs_x={} threshold={}",
            fmt_diag_sci(*max_abs_x),
            fmt_diag_sci(*threshold)
        )),
        InteriorPointSolveError::CpuTimeExceeded {
            elapsed_seconds,
            limit_seconds,
            ..
        } => lines.push(format!(
            "  termination=cpu_time elapsed={} limit={}",
            fmt_diag_sci(*elapsed_seconds),
            fmt_diag_sci(*limit_seconds)
        )),
        InteriorPointSolveError::WallTimeExceeded {
            elapsed_seconds,
            limit_seconds,
            ..
        } => lines.push(format!(
            "  termination=wall_time elapsed={} limit={}",
            fmt_diag_sci(*elapsed_seconds),
            fmt_diag_sci(*limit_seconds)
        )),
        InteriorPointSolveError::UserRequestedStop { .. } => {
            lines.push("  termination=user_requested_stop".to_string())
        }
        InteriorPointSolveError::SearchDirectionTooSmall { .. } => {
            lines.push("  termination=search_direction_too_small".to_string())
        }
        InteriorPointSolveError::MaxIterations { iterations, .. } => {
            lines.push(format!("  termination=max_iterations limit={iterations}"))
        }
        InteriorPointSolveError::InvalidInput(_) => {}
    }
    if let Some(snapshot) = &context.final_state {
        append_nlip_snapshot_lines(&mut lines, "current iterate", snapshot);
    }
    if let Some(snapshot) = &context.last_accepted_state {
        append_nlip_snapshot_lines(&mut lines, "last accepted iterate", snapshot);
    }
    if let Some(diagnostics) = &context.failed_linear_solve {
        append_nlip_linear_solve_failure_lines(&mut lines, diagnostics);
    }
    if let Some(info) = &context.failed_line_search {
        append_nlip_line_search_failure_lines(&mut lines, error, info);
    }
    lines
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig {
    pub profile: SolverProfile,
    pub max_iters: usize,
    pub overall_tol: f64,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub complementarity_tol: f64,
    pub sqp: SqpMethodConfig,
    pub nlip: NlipConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverProfile {
    StrictOcp,
    IpoptDefault,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SolverProfileDefaults {
    max_iters: usize,
    overall_tol: f64,
    dual_tol: f64,
    constraint_tol: f64,
    complementarity_tol: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SqpMethodConfig {
    pub hessian_regularization_enabled: bool,
    pub globalization: SqpGlobalizationConfig,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NlipConfig {
    pub linear_solver: InteriorPointLinearSolver,
    pub spral_pivot_method: InteriorPointSpralPivotMethod,
    pub spral_action_on_zero_pivot: bool,
    pub spral_small_pivot_tolerance: f64,
    pub spral_threshold_pivot_u: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpGlobalizationMode {
    LineSearchFilter,
    LineSearchMerit,
    TrustRegionFilter,
    TrustRegionMerit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SqpLineSearchConfig {
    pub armijo_c1: f64,
    pub wolfe_c2: Option<f64>,
    pub beta: f64,
    pub max_steps: usize,
    pub min_step: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SqpFilterConfig {
    pub gamma_objective: f64,
    pub gamma_violation: f64,
    pub theta_max_factor: f64,
    pub switching_reference_min: f64,
    pub switching_violation_factor: f64,
    pub switching_linearized_reduction_factor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SqpTrustRegionConfig {
    pub initial_radius: f64,
    pub max_radius: f64,
    pub min_radius: f64,
    pub shrink_factor: f64,
    pub grow_factor: f64,
    pub accept_ratio: f64,
    pub expand_ratio: f64,
    pub boundary_fraction: f64,
    pub max_radius_contractions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SqpGlobalizationConfig {
    LineSearchFilter {
        line_search: SqpLineSearchConfig,
        filter: SqpFilterConfig,
        exact_merit_penalty: f64,
    },
    LineSearchMerit {
        line_search: SqpLineSearchConfig,
        exact_merit_penalty: f64,
        penalty_increase_factor: f64,
        max_penalty_updates: usize,
    },
    TrustRegionFilter {
        trust_region: SqpTrustRegionConfig,
        filter: SqpFilterConfig,
        exact_merit_penalty: f64,
    },
    TrustRegionMerit {
        trust_region: SqpTrustRegionConfig,
        exact_merit_penalty: f64,
        fixed_penalty: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverMethod {
    Sqp,
    Nlip,
    #[cfg(feature = "ipopt")]
    Ipopt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverStatusKind {
    Success,
    Warning,
    Error,
    Info,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SolverPhaseDetail {
    pub label: String,
    pub value: String,
    pub count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SolverPhaseDetails {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbolic_setup: Vec<SolverPhaseDetail>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub jit: Vec<SolverPhaseDetail>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub solve: Vec<SolverPhaseDetail>,
}

impl SolverPhaseDetails {
    pub fn is_empty(&self) -> bool {
        self.symbolic_setup.is_empty() && self.jit.is_empty() && self.solve.is_empty()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SolverReport {
    pub completed: bool,
    pub status_label: String,
    pub status_kind: SolverStatusKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbolic_setup_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jit_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solve_s: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub compile_cached: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub jit_disk_cache_hit: bool,
    #[serde(default, skip_serializing_if = "SolverPhaseDetails::is_empty")]
    pub phase_details: SolverPhaseDetails,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompileReportSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbolic_construction_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective_gradient_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equality_jacobian_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inequality_jacobian_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lagrangian_assembly_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_generation_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lowering_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llvm_jit_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llvm_cache_load_s: Option<f64>,
    pub llvm_cache_hits: usize,
    pub llvm_cache_misses: usize,
    pub symbolic_function_count: usize,
    pub call_site_count: usize,
    pub max_call_depth: usize,
    pub inlines_at_call: usize,
    pub inlines_at_lowering: usize,
    pub llvm_root_instructions_emitted: usize,
    pub llvm_total_instructions_emitted: usize,
    pub llvm_subfunctions_emitted: usize,
    pub llvm_call_instructions_emitted: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl SolverReport {
    pub fn placeholder() -> Self {
        Self {
            completed: false,
            status_label: "Not solved".to_string(),
            status_kind: SolverStatusKind::Info,
            iterations: None,
            symbolic_setup_s: None,
            jit_s: None,
            solve_s: None,
            compile_cached: false,
            jit_disk_cache_hit: false,
            phase_details: SolverPhaseDetails::default(),
        }
    }

    pub fn in_progress(status_label: impl Into<String>) -> Self {
        Self {
            completed: false,
            status_label: status_label.into(),
            status_kind: SolverStatusKind::Info,
            iterations: None,
            symbolic_setup_s: None,
            jit_s: None,
            solve_s: None,
            compile_cached: false,
            jit_disk_cache_hit: false,
            phase_details: SolverPhaseDetails::default(),
        }
    }

    pub fn with_backend_timing(mut self, timing: BackendTimingMetadata) -> Self {
        self.symbolic_setup_s = symbolic_setup_seconds(timing);
        self.jit_s = duration_seconds(timing.jit_time);
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = Some(iterations);
        self
    }

    pub fn with_solve_seconds(mut self, solve_s: f64) -> Self {
        if solve_s.is_finite() && solve_s >= 0.0 {
            self.solve_s = Some(solve_s);
        }
        self
    }

    pub fn with_phase_details(mut self, phase_details: SolverPhaseDetails) -> Self {
        self.phase_details = phase_details;
        self
    }

    pub fn with_compile_cached(mut self, compile_cached: bool) -> Self {
        self.compile_cached = compile_cached;
        self.jit_disk_cache_hit = compile_cached;
        self
    }

    pub fn with_symbolic_phase_details(mut self, details: Vec<SolverPhaseDetail>) -> Self {
        self.phase_details.symbolic_setup = details;
        self
    }
}

pub fn summarize_backend_compile_report(report: &BackendCompileReport) -> CompileReportSummary {
    CompileReportSummary {
        symbolic_construction_s: duration_seconds(report.setup_profile.symbolic_construction),
        objective_gradient_s: duration_seconds(report.setup_profile.objective_gradient),
        equality_jacobian_s: duration_seconds(report.setup_profile.equality_jacobian),
        inequality_jacobian_s: duration_seconds(report.setup_profile.inequality_jacobian),
        lagrangian_assembly_s: duration_seconds(report.setup_profile.lagrangian_assembly),
        hessian_generation_s: duration_seconds(report.setup_profile.hessian_generation),
        lowering_s: duration_seconds(report.setup_profile.lowering),
        llvm_jit_s: duration_seconds(report.setup_profile.llvm_jit),
        llvm_cache_load_s: duration_seconds(Some(report.llvm_jit_cache.load_time)),
        llvm_cache_hits: report.llvm_jit_cache.hits,
        llvm_cache_misses: report.llvm_jit_cache.misses,
        symbolic_function_count: report.stats.symbolic_function_count,
        call_site_count: report.stats.call_site_count,
        max_call_depth: report.stats.max_call_depth,
        inlines_at_call: report.stats.inlines_at_call,
        inlines_at_lowering: report.stats.inlines_at_lowering,
        llvm_root_instructions_emitted: report.stats.llvm_root_instructions_emitted,
        llvm_total_instructions_emitted: report.stats.llvm_total_instructions_emitted,
        llvm_subfunctions_emitted: report.stats.llvm_subfunctions_emitted,
        llvm_call_instructions_emitted: report.stats.llvm_call_instructions_emitted,
        warnings: report
            .warnings
            .iter()
            .map(|warning| warning.message.clone())
            .collect(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptionMethod {
    MultipleShooting,
    DirectCollocation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranscriptionConfig {
    pub method: TranscriptionMethod,
    pub intervals: usize,
    pub collocation_degree: usize,
    pub collocation_family: CollocationFamily,
    pub time_grid: ocp_runtime::TimeGrid,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OcpOverrideBehavior {
    #[default]
    RespectFunctionOverrides,
    StrictGlobalPolicy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OcpKernelStrategy {
    Inline,
    #[default]
    FunctionUseGlobalPolicy,
    FunctionInlineAtCall,
    FunctionInlineAtLowering,
    FunctionInlineInLlvm,
    FunctionNoInlineLlvm,
}

impl OcpKernelStrategy {
    pub const fn to_kernel_options(self) -> OcpKernelFunctionOptions {
        match self {
            Self::Inline => OcpKernelFunctionOptions::inline(),
            Self::FunctionUseGlobalPolicy => OcpKernelFunctionOptions::function(),
            Self::FunctionInlineAtCall => {
                OcpKernelFunctionOptions::function_with_call_policy(CallPolicy::InlineAtCall)
            }
            Self::FunctionInlineAtLowering => {
                OcpKernelFunctionOptions::function_with_call_policy(CallPolicy::InlineAtLowering)
            }
            Self::FunctionInlineInLlvm => {
                OcpKernelFunctionOptions::function_with_call_policy(CallPolicy::InlineInLLVM)
            }
            Self::FunctionNoInlineLlvm => {
                OcpKernelFunctionOptions::function_with_call_policy(CallPolicy::NoInlineLLVM)
            }
        }
    }

    const fn short_label(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::FunctionUseGlobalPolicy => "function/global",
            Self::FunctionInlineAtCall => "function/call",
            Self::FunctionInlineAtLowering => "function/lowering",
            Self::FunctionInlineInLlvm => "function/llvm",
            Self::FunctionNoInlineLlvm => "function/noinline",
        }
    }

    const fn variant_token(self) -> &'static str {
        match self {
            Self::Inline => "inl",
            Self::FunctionUseGlobalPolicy => "fng",
            Self::FunctionInlineAtCall => "fnc",
            Self::FunctionInlineAtLowering => "fnl",
            Self::FunctionInlineInLlvm => "fni",
            Self::FunctionNoInlineLlvm => "fnn",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OcpSxFunctionConfig {
    pub global_call_policy: CallPolicy,
    pub override_behavior: OcpOverrideBehavior,
    pub hessian_strategy: HessianStrategy,
    pub ode: OcpKernelStrategy,
    pub objective_lagrange: OcpKernelStrategy,
    pub objective_mayer: OcpKernelStrategy,
    pub path_constraints: OcpKernelStrategy,
    pub boundary_equalities: OcpKernelStrategy,
    pub boundary_inequalities: OcpKernelStrategy,
    pub multiple_shooting_integrator: OcpKernelStrategy,
}

impl OcpSxFunctionConfig {
    pub const fn inline_all() -> Self {
        Self {
            global_call_policy: CallPolicy::InlineAtLowering,
            override_behavior: OcpOverrideBehavior::RespectFunctionOverrides,
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
            ode: OcpKernelStrategy::Inline,
            objective_lagrange: OcpKernelStrategy::Inline,
            objective_mayer: OcpKernelStrategy::Inline,
            path_constraints: OcpKernelStrategy::Inline,
            boundary_equalities: OcpKernelStrategy::Inline,
            boundary_inequalities: OcpKernelStrategy::Inline,
            multiple_shooting_integrator: OcpKernelStrategy::Inline,
        }
    }

    pub const fn all_functions_with_global_policy(policy: CallPolicy) -> Self {
        Self {
            global_call_policy: policy,
            override_behavior: OcpOverrideBehavior::RespectFunctionOverrides,
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
            ode: OcpKernelStrategy::FunctionUseGlobalPolicy,
            objective_lagrange: OcpKernelStrategy::FunctionUseGlobalPolicy,
            objective_mayer: OcpKernelStrategy::FunctionUseGlobalPolicy,
            path_constraints: OcpKernelStrategy::FunctionUseGlobalPolicy,
            boundary_equalities: OcpKernelStrategy::FunctionUseGlobalPolicy,
            boundary_inequalities: OcpKernelStrategy::FunctionUseGlobalPolicy,
            multiple_shooting_integrator: OcpKernelStrategy::FunctionUseGlobalPolicy,
        }
    }

    pub const fn call_policy_config(self) -> CallPolicyConfig {
        CallPolicyConfig {
            default_policy: self.global_call_policy,
            respect_function_overrides: matches!(
                self.override_behavior,
                OcpOverrideBehavior::RespectFunctionOverrides
            ),
        }
    }

    pub const fn symbolic_functions(self) -> OcpSymbolicFunctionOptions {
        OcpSymbolicFunctionOptions {
            ode: self.ode.to_kernel_options(),
            objective_lagrange: self.objective_lagrange.to_kernel_options(),
            objective_mayer: self.objective_mayer.to_kernel_options(),
            path_constraints: self.path_constraints.to_kernel_options(),
            boundary_equalities: self.boundary_equalities.to_kernel_options(),
            boundary_inequalities: self.boundary_inequalities.to_kernel_options(),
            multiple_shooting_integrator: self.multiple_shooting_integrator.to_kernel_options(),
        }
    }

    pub const fn compile_options(self, opt_level: LlvmOptimizationLevel) -> OcpCompileOptions {
        OcpCompileOptions {
            function_options: FunctionCompileOptions::new(opt_level, self.call_policy_config()),
            symbolic_functions: self.symbolic_functions(),
            hessian_strategy: self.hessian_strategy,
        }
    }

    pub fn variant_id_suffix(self) -> String {
        format!(
            "g{}_ov{}_h{}_ode{}_lag{}_may{}_path{}_beq{}_biq{}_msi{}",
            call_policy_variant_token(self.global_call_policy),
            match self.override_behavior {
                OcpOverrideBehavior::RespectFunctionOverrides => "r",
                OcpOverrideBehavior::StrictGlobalPolicy => "s",
            },
            hessian_strategy_variant_token(self.hessian_strategy),
            self.ode.variant_token(),
            self.objective_lagrange.variant_token(),
            self.objective_mayer.variant_token(),
            self.path_constraints.variant_token(),
            self.boundary_equalities.variant_token(),
            self.boundary_inequalities.variant_token(),
            self.multiple_shooting_integrator.variant_token(),
        )
    }

    pub fn variant_label_suffix(self) -> Option<String> {
        if self == Self::default() {
            None
        } else if self == Self::inline_all() {
            Some("SXF Inline All".to_string())
        } else if self == Self::all_functions_with_global_policy(CallPolicy::InlineAtCall) {
            Some("SXF All Functions / Inline At Call".to_string())
        } else if self == Self::all_functions_with_global_policy(CallPolicy::InlineAtLowering) {
            Some("SXF All Functions / Inline At Lowering".to_string())
        } else if self == Self::all_functions_with_global_policy(CallPolicy::InlineInLLVM) {
            Some("SXF All Functions / Inline In LLVM".to_string())
        } else if self == Self::all_functions_with_global_policy(CallPolicy::NoInlineLLVM) {
            Some("SXF All Functions / NoInline LLVM".to_string())
        } else {
            Some("SXF Custom".to_string())
        }
    }

    pub fn delta_summary(self) -> Vec<String> {
        let default = Self::default();
        let mut deltas = Vec::new();
        if self.global_call_policy != default.global_call_policy {
            deltas.push(format!(
                "Global {}",
                call_policy_short_label(self.global_call_policy)
            ));
        }
        if self.override_behavior != default.override_behavior {
            deltas.push("Strict Global".to_string());
        }
        if self.hessian_strategy != default.hessian_strategy {
            deltas.push(format!(
                "Hessian {}",
                hessian_strategy_short_label(self.hessian_strategy)
            ));
        }
        append_kernel_delta(&mut deltas, "ODE", self.ode, default.ode);
        append_kernel_delta(
            &mut deltas,
            "Lagrange",
            self.objective_lagrange,
            default.objective_lagrange,
        );
        append_kernel_delta(
            &mut deltas,
            "Mayer",
            self.objective_mayer,
            default.objective_mayer,
        );
        append_kernel_delta(
            &mut deltas,
            "Path",
            self.path_constraints,
            default.path_constraints,
        );
        append_kernel_delta(
            &mut deltas,
            "BEq",
            self.boundary_equalities,
            default.boundary_equalities,
        );
        append_kernel_delta(
            &mut deltas,
            "BIneq",
            self.boundary_inequalities,
            default.boundary_inequalities,
        );
        append_kernel_delta(
            &mut deltas,
            "MS Integrator",
            self.multiple_shooting_integrator,
            default.multiple_shooting_integrator,
        );
        deltas
    }
}

impl Default for OcpSxFunctionConfig {
    fn default() -> Self {
        Self {
            global_call_policy: CallPolicy::InlineAtLowering,
            override_behavior: OcpOverrideBehavior::RespectFunctionOverrides,
            hessian_strategy: HessianStrategy::LowerTriangleByColumn,
            ode: OcpKernelStrategy::FunctionInlineInLlvm,
            objective_lagrange: OcpKernelStrategy::FunctionInlineInLlvm,
            objective_mayer: OcpKernelStrategy::Inline,
            path_constraints: OcpKernelStrategy::FunctionInlineInLlvm,
            boundary_equalities: OcpKernelStrategy::Inline,
            boundary_inequalities: OcpKernelStrategy::Inline,
            multiple_shooting_integrator: OcpKernelStrategy::Inline,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MultipleShootingCompileKey {
    pub intervals: usize,
    pub sx_functions: OcpSxFunctionConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DirectCollocationCompileVariantKey {
    pub intervals: usize,
    pub order: usize,
    pub family: DirectCollocationCompileKey,
    pub time_grid: TimeGridCompileKey,
    pub sx_functions: OcpSxFunctionConfig,
}

pub const fn multiple_shooting_compile_key(
    intervals: usize,
    sx_functions: OcpSxFunctionConfig,
) -> MultipleShootingCompileKey {
    MultipleShootingCompileKey {
        intervals,
        sx_functions,
    }
}

pub fn direct_collocation_compile_key_with_sx(
    intervals: usize,
    order: usize,
    family: CollocationFamily,
    time_grid: ocp_runtime::TimeGrid,
    sx_functions: OcpSxFunctionConfig,
) -> DirectCollocationCompileVariantKey {
    DirectCollocationCompileVariantKey {
        intervals,
        order,
        family: direct_collocation_compile_key(family),
        time_grid: time_grid_compile_key(time_grid),
        sx_functions,
    }
}

pub fn ocp_compile_options(
    opt_level: LlvmOptimizationLevel,
    sx_functions: OcpSxFunctionConfig,
) -> OcpCompileOptions {
    sx_functions.compile_options(opt_level)
}

fn append_kernel_delta(
    deltas: &mut Vec<String>,
    label: &str,
    value: OcpKernelStrategy,
    default: OcpKernelStrategy,
) {
    if value != default {
        deltas.push(format!("{label} {}", value.short_label()));
    }
}

fn call_policy_short_label(policy: CallPolicy) -> &'static str {
    match policy {
        CallPolicy::InlineAtCall => "Inline At Call",
        CallPolicy::InlineAtLowering => "Inline At Lowering",
        CallPolicy::InlineInLLVM => "Inline In LLVM",
        CallPolicy::NoInlineLLVM => "NoInline LLVM",
    }
}

fn hessian_strategy_short_label(strategy: HessianStrategy) -> &'static str {
    match strategy {
        HessianStrategy::LowerTriangleByColumn => "By Column",
        HessianStrategy::LowerTriangleSelectedOutputs => "Selected Outputs",
        HessianStrategy::LowerTriangleColored => "Colored",
    }
}

const fn hessian_strategy_variant_token(strategy: HessianStrategy) -> &'static str {
    match strategy {
        HessianStrategy::LowerTriangleByColumn => "c",
        HessianStrategy::LowerTriangleSelectedOutputs => "s",
        HessianStrategy::LowerTriangleColored => "k",
    }
}

const fn call_policy_variant_token(policy: CallPolicy) -> &'static str {
    match policy {
        CallPolicy::InlineAtCall => "c",
        CallPolicy::InlineAtLowering => "l",
        CallPolicy::InlineInLLVM => "i",
        CallPolicy::NoInlineLLVM => "n",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedControlId {
    TranscriptionMethod,
    TranscriptionIntervals,
    CollocationFamily,
    CollocationDegree,
    TimeGrid,
    TimeGridStrength,
    TimeGridFocusCenter,
    TimeGridFocusWidth,
    TimeGridBreakpoint,
    TimeGridFirstIntervalFraction,
    SxFunctionGlobalCallPolicy,
    SxFunctionOverrideBehavior,
    SxFunctionHessianStrategy,
    SxFunctionOde,
    SxFunctionObjectiveLagrange,
    SxFunctionObjectiveMayer,
    SxFunctionPathConstraints,
    SxFunctionBoundaryEqualities,
    SxFunctionBoundaryInequalities,
    SxFunctionMultipleShootingIntegrator,
    SolverMethod,
    SolverProfile,
    SolverGlobalization,
    SolverMaxIterations,
    SolverOverallTolerance,
    SolverHessianRegularization,
    SolverNlipLinearSolver,
    SolverNlipSpralPivotMethod,
    SolverNlipSpralZeroPivotAction,
    SolverNlipSpralSmallPivot,
    SolverNlipSpralPivotU,
    SolverDualTolerance,
    SolverConstraintTolerance,
    SolverComplementarityTolerance,
    SolverExactMeritPenalty,
    SolverPenaltyIncreaseFactor,
    SolverMaxPenaltyUpdates,
    SolverArmijoC1,
    SolverWolfeC2,
    SolverLineSearchBeta,
    SolverLineSearchMaxSteps,
    SolverMinStep,
    SolverFilterGammaObjective,
    SolverFilterGammaViolation,
    SolverFilterThetaMaxFactor,
    SolverFilterSwitchingReferenceMin,
    SolverFilterSwitchingViolationFactor,
    SolverFilterSwitchingLinearizedReductionFactor,
    SolverTrustRegionInitialRadius,
    SolverTrustRegionMaxRadius,
    SolverTrustRegionMinRadius,
    SolverTrustRegionShrinkFactor,
    SolverTrustRegionGrowFactor,
    SolverTrustRegionAcceptRatio,
    SolverTrustRegionExpandRatio,
    SolverTrustRegionBoundaryFraction,
    SolverTrustRegionMaxContractions,
    SolverTrustRegionFixedPenalty,
}

impl SharedControlId {
    const fn id(self) -> &'static str {
        match self {
            Self::TranscriptionMethod => "transcription_method",
            Self::TranscriptionIntervals => "transcription_intervals",
            Self::CollocationFamily => "collocation_family",
            Self::CollocationDegree => "collocation_degree",
            Self::TimeGrid => "time_grid",
            Self::TimeGridStrength => "time_grid_strength",
            Self::TimeGridFocusCenter => "time_grid_focus_center",
            Self::TimeGridFocusWidth => "time_grid_focus_width",
            Self::TimeGridBreakpoint => "time_grid_breakpoint",
            Self::TimeGridFirstIntervalFraction => "time_grid_first_interval_fraction",
            Self::SxFunctionGlobalCallPolicy => "sxf_global_call_policy",
            Self::SxFunctionOverrideBehavior => "sxf_override_behavior",
            Self::SxFunctionHessianStrategy => "sxf_hessian_strategy",
            Self::SxFunctionOde => "sxf_ode",
            Self::SxFunctionObjectiveLagrange => "sxf_objective_lagrange",
            Self::SxFunctionObjectiveMayer => "sxf_objective_mayer",
            Self::SxFunctionPathConstraints => "sxf_path_constraints",
            Self::SxFunctionBoundaryEqualities => "sxf_boundary_equalities",
            Self::SxFunctionBoundaryInequalities => "sxf_boundary_inequalities",
            Self::SxFunctionMultipleShootingIntegrator => "sxf_multiple_shooting_integrator",
            Self::SolverMethod => "solver_method",
            Self::SolverProfile => "solver_profile",
            Self::SolverGlobalization => "solver_globalization",
            Self::SolverMaxIterations => "solver_max_iters",
            Self::SolverOverallTolerance => "solver_overall_tol",
            Self::SolverHessianRegularization => "solver_hessian_regularization",
            Self::SolverNlipLinearSolver => "solver_nlip_linear_solver",
            Self::SolverNlipSpralPivotMethod => "solver_nlip_spral_pivot_method",
            Self::SolverNlipSpralZeroPivotAction => "solver_nlip_spral_zero_pivot_action",
            Self::SolverNlipSpralSmallPivot => "solver_nlip_spral_small_pivot",
            Self::SolverNlipSpralPivotU => "solver_nlip_spral_pivot_u",
            Self::SolverDualTolerance => "solver_dual_tol",
            Self::SolverConstraintTolerance => "solver_constraint_tol",
            Self::SolverComplementarityTolerance => "solver_complementarity_tol",
            Self::SolverExactMeritPenalty => "solver_exact_merit_penalty",
            Self::SolverPenaltyIncreaseFactor => "solver_penalty_increase_factor",
            Self::SolverMaxPenaltyUpdates => "solver_max_penalty_updates",
            Self::SolverArmijoC1 => "solver_armijo_c1",
            Self::SolverWolfeC2 => "solver_wolfe_c2",
            Self::SolverLineSearchBeta => "solver_line_search_beta",
            Self::SolverLineSearchMaxSteps => "solver_line_search_max_steps",
            Self::SolverMinStep => "solver_min_step",
            Self::SolverFilterGammaObjective => "solver_filter_gamma_objective",
            Self::SolverFilterGammaViolation => "solver_filter_gamma_violation",
            Self::SolverFilterThetaMaxFactor => "solver_filter_theta_max_factor",
            Self::SolverFilterSwitchingReferenceMin => "solver_filter_switching_reference_min",
            Self::SolverFilterSwitchingViolationFactor => {
                "solver_filter_switching_violation_factor"
            }
            Self::SolverFilterSwitchingLinearizedReductionFactor => {
                "solver_filter_switching_linearized_reduction_factor"
            }
            Self::SolverTrustRegionInitialRadius => "solver_trust_region_initial_radius",
            Self::SolverTrustRegionMaxRadius => "solver_trust_region_max_radius",
            Self::SolverTrustRegionMinRadius => "solver_trust_region_min_radius",
            Self::SolverTrustRegionShrinkFactor => "solver_trust_region_shrink_factor",
            Self::SolverTrustRegionGrowFactor => "solver_trust_region_grow_factor",
            Self::SolverTrustRegionAcceptRatio => "solver_trust_region_accept_ratio",
            Self::SolverTrustRegionExpandRatio => "solver_trust_region_expand_ratio",
            Self::SolverTrustRegionBoundaryFraction => "solver_trust_region_boundary_fraction",
            Self::SolverTrustRegionMaxContractions => "solver_trust_region_max_contractions",
            Self::SolverTrustRegionFixedPenalty => "solver_trust_region_fixed_penalty",
        }
    }
}

type Numeric<T> = <T as Vectorize<SX>>::Rebind<f64>;

pub type CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq> =
    ocp_runtime::CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq>;
pub type CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq> =
    ocp_runtime::CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq>;
pub type MultipleShootingInitialGuess<X, U, P> = ocp_runtime::MultipleShootingInitialGuess<X, U, P>;
pub type DirectCollocationInitialGuess<X, U, P> =
    ocp_runtime::DirectCollocationInitialGuess<X, U, P>;
pub type MultipleShootingRuntimeValues<P, C, Beq, Bineq, X, U> =
    ocp_runtime::MultipleShootingRuntimeValues<P, C, Beq, Bineq, X, U>;
pub type DirectCollocationRuntimeValues<P, C, Beq, Bineq, X, U> =
    ocp_runtime::DirectCollocationRuntimeValues<P, C, Beq, Bineq, X, U>;
pub type MultipleShootingTrajectories<X, U> = ocp_runtime::MultipleShootingTrajectories<X, U>;
pub type DirectCollocationTrajectories<X, U> = ocp_runtime::DirectCollocationTrajectories<X, U>;
pub type TimeGrid = ocp_runtime::TimeGrid;
pub type TimeGridBias = ocp_runtime::TimeGridBias;
pub type DirectCollocationTimeGrid = ocp_runtime::DirectCollocationTimeGrid;
pub type MultipleShootingSqpSnapshot<X, U> = ocp_runtime::MultipleShootingSqpSnapshot<X, U>;
pub type DirectCollocationSqpSnapshot<X, U> = ocp_runtime::DirectCollocationSqpSnapshot<X, U>;
pub type MultipleShootingInteriorPointSnapshot<X, U> =
    ocp_runtime::MultipleShootingInteriorPointSnapshot<X, U>;
pub type DirectCollocationInteriorPointSnapshot<X, U> =
    ocp_runtime::DirectCollocationInteriorPointSnapshot<X, U>;
pub type MultipleShootingSqpSolveResult<X, U> = ocp_runtime::MultipleShootingSqpSolveResult<X, U>;
pub type DirectCollocationSqpSolveResult<X, U> = ocp_runtime::DirectCollocationSqpSolveResult<X, U>;
pub type MultipleShootingInteriorPointSolveResult<X, U> =
    ocp_runtime::MultipleShootingInteriorPointSolveResult<X, U>;
pub type DirectCollocationInteriorPointSolveResult<X, U> =
    ocp_runtime::DirectCollocationInteriorPointSolveResult<X, U>;
#[cfg(feature = "ipopt")]
pub type MultipleShootingIpoptSolveResult<X, U> =
    ocp_runtime::MultipleShootingIpoptSolveResult<X, U>;
#[cfg(feature = "ipopt")]
pub type DirectCollocationIpoptSolveResult<X, U> =
    ocp_runtime::DirectCollocationIpoptSolveResult<X, U>;
#[cfg(feature = "ipopt")]
pub type MultipleShootingIpoptSnapshot<X, U> = ocp_runtime::MultipleShootingIpoptSnapshot<X, U>;
#[cfg(feature = "ipopt")]
pub type DirectCollocationIpoptSnapshot<X, U> = ocp_runtime::DirectCollocationIpoptSnapshot<X, U>;

pub enum ContinuousInitialGuess<X, U, P> {
    Interpolated(InterpolatedTrajectory<X, U>),
    #[allow(dead_code)]
    Rollout {
        x0: X,
        u0: U,
        tf: f64,
        controller: Box<ControllerFn<X, U, P>>,
    },
}

pub struct OcpRuntimeSpec<P, C, Beq, Bineq, X, U> {
    pub parameters: P,
    pub beq: Beq,
    pub bineq_bounds: Bineq,
    pub path_bounds: C,
    pub tf_bounds: Bounds1D,
    pub initial_guess: ContinuousInitialGuess<X, U, P>,
}

impl<X, U, P> ContinuousInitialGuess<X, U, P> {
    pub fn into_multiple_shooting(self) -> ocp_runtime::MultipleShootingInitialGuess<X, U, P> {
        match self {
            Self::Interpolated(trajectory) => {
                MultipleShootingInitialGuess::Interpolated(trajectory)
            }
            Self::Rollout {
                x0,
                u0,
                tf,
                controller,
            } => MultipleShootingInitialGuess::Rollout {
                x0,
                u0,
                tf,
                controller,
            },
        }
    }

    pub fn into_direct_collocation(self) -> ocp_runtime::DirectCollocationInitialGuess<X, U, P> {
        match self {
            Self::Interpolated(trajectory) => {
                DirectCollocationInitialGuess::Interpolated(trajectory)
            }
            Self::Rollout {
                x0,
                u0,
                tf,
                controller,
            } => DirectCollocationInitialGuess::Rollout {
                x0,
                u0,
                tf,
                controller,
            },
        }
    }
}

pub fn multiple_shooting_runtime_from_spec<P, C, Beq, Bineq, X, U>(
    spec: OcpRuntimeSpec<P, C, Beq, Bineq, X, U>,
) -> ocp_runtime::MultipleShootingRuntimeValues<P, C, Beq, Bineq, X, U> {
    ocp_runtime::MultipleShootingRuntimeValues {
        parameters: spec.parameters,
        beq: spec.beq,
        bineq_bounds: spec.bineq_bounds,
        path_bounds: spec.path_bounds,
        tf_bounds: spec.tf_bounds,
        initial_guess: spec.initial_guess.into_multiple_shooting(),
        scaling: None,
    }
}

pub fn direct_collocation_runtime_from_spec<P, C, Beq, Bineq, X, U>(
    spec: OcpRuntimeSpec<P, C, Beq, Bineq, X, U>,
) -> ocp_runtime::DirectCollocationRuntimeValues<P, C, Beq, Bineq, X, U> {
    ocp_runtime::DirectCollocationRuntimeValues {
        parameters: spec.parameters,
        beq: spec.beq,
        bineq_bounds: spec.bineq_bounds,
        path_bounds: spec.path_bounds,
        tf_bounds: spec.tf_bounds,
        initial_guess: spec.initial_guess.into_direct_collocation(),
        scaling: None,
    }
}

pub struct SharedCompileCache<K, V> {
    entries: HashMap<K, Rc<RefCell<V>>>,
}

pub struct CachedCompile<V> {
    pub compiled: Rc<RefCell<V>>,
    pub was_cached: bool,
}

#[derive(Clone, Debug)]
pub struct CompileProgressUpdate {
    pub timing: BackendTimingMetadata,
    pub phase_details: SolverPhaseDetails,
    pub compile_cached: bool,
}

#[derive(Clone, Debug)]
pub struct CompileProgressInfo {
    pub timing: BackendTimingMetadata,
    pub compile_cached: bool,
    pub phase_details: SolverPhaseDetails,
    pub compile_report: Option<CompileReportSummary>,
}

#[derive(Clone, Debug, Default)]
pub struct OcpCompileProgressState {
    timing: BackendTimingMetadata,
    phase_details: SolverPhaseDetails,
}

impl<K, V> SharedCompileCache<K, V>
where
    K: Eq + Hash + Copy,
{
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn get_or_try_init<F>(&mut self, key: K, build: F) -> Result<CachedCompile<V>>
    where
        F: FnOnce() -> Result<V>,
    {
        if let Some(compiled) = self.entries.get(&key) {
            return Ok(CachedCompile {
                compiled: compiled.clone(),
                was_cached: true,
            });
        }

        let compiled = Rc::new(RefCell::new(build()?));
        self.entries.insert(key, compiled.clone());
        Ok(CachedCompile {
            compiled,
            was_cached: false,
        })
    }

    pub fn cached_entries(&self) -> Vec<(K, Rc<RefCell<V>>)> {
        self.entries
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect()
    }
}

pub fn compile_progress_info_from_compiled<Compiled>(compiled: &Compiled) -> CompileProgressInfo
where
    Compiled: CompiledOcpMetadata,
{
    compile_progress_info(
        compiled.backend_timing_metadata(),
        compiled.nlp_compile_stats(),
        compiled.helper_kernel_count(),
        compiled.helper_compile_stats(),
        Some(summarize_backend_compile_report(
            compiled.backend_compile_report(),
        )),
    )
}

pub fn standard_ocp_compile_cache_statuses<Ms, Dc>(
    problem_id: ProblemId,
    problem_name: &str,
    multiple_shooting_cache: &SharedCompileCache<MultipleShootingCompileKey, Ms>,
    direct_collocation_cache: &SharedCompileCache<DirectCollocationCompileVariantKey, Dc>,
) -> Vec<CompileCacheStatus>
where
    Ms: CompiledOcpMetadata,
    Dc: CompiledOcpMetadata,
{
    let mut statuses = Vec::new();
    append_standard_compile_cache_statuses(
        &mut statuses,
        problem_id,
        problem_name,
        multiple_shooting_cache,
        direct_collocation_cache,
    );
    statuses
}

pub fn prewarm_standard_ocp<Params, MsCompiled, DcCompiled, CachedMs, CachedDc>(
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    cached_multiple_shooting: CachedMs,
    cached_direct_collocation: CachedDc,
) -> Result<()>
where
    CachedMs: Fn(&Params) -> Result<CachedCompile<MsCompiled>>,
    CachedDc: Fn(&Params, CollocationFamily) -> Result<CachedCompile<DcCompiled>>,
{
    match transcription {
        TranscriptionMethod::MultipleShooting => cached_multiple_shooting(params).map(|_| ()),
        TranscriptionMethod::DirectCollocation => {
            cached_direct_collocation(params, collocation_family).map(|_| ())
        }
    }
}

pub fn prewarm_standard_ocp_with_progress<
    Params,
    Emit,
    MsCompiled,
    DcCompiled,
    CompileMs,
    CompileDc,
>(
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    solver_method: SolverMethod,
    emit: Emit,
    compile_multiple_shooting: CompileMs,
    compile_direct_collocation: CompileDc,
) -> Result<()>
where
    Emit: FnMut(SolveStreamEvent) + Send,
    CompileMs: Fn(
        &Params,
        &mut dyn FnMut(CompileProgressUpdate),
    ) -> Result<(Rc<RefCell<MsCompiled>>, CompileProgressInfo)>,
    CompileDc: Fn(
        &Params,
        CollocationFamily,
        &mut dyn FnMut(CompileProgressUpdate),
    ) -> Result<(Rc<RefCell<DcCompiled>>, CompileProgressInfo)>,
{
    let mut lifecycle = SolveLifecycleReporter::new(emit, solver_method);
    match transcription {
        TranscriptionMethod::MultipleShooting => {
            lifecycle.prewarm_with_progress(|callback| compile_multiple_shooting(params, callback))
        }
        TranscriptionMethod::DirectCollocation => lifecycle.prewarm_with_progress(|callback| {
            compile_direct_collocation(params, collocation_family, callback)
        }),
    }
}

pub fn solve_standard_ocp<
    Params,
    MsCompiled,
    DcCompiled,
    CachedMs,
    CachedDc,
    MsRuntimeFn,
    DcRuntimeFn,
    MsArtifact,
    DcArtifact,
>(
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    cached_multiple_shooting: CachedMs,
    cached_direct_collocation: CachedDc,
    multiple_shooting_runtime: MsRuntimeFn,
    direct_collocation_runtime: DcRuntimeFn,
    multiple_shooting_artifact: MsArtifact,
    direct_collocation_artifact: DcArtifact,
) -> Result<SolveArtifact>
where
    MsCompiled: MultipleShootingCompiled,
    DcCompiled: DirectCollocationCompiled,
    CachedMs: Fn(&Params) -> Result<CachedCompile<MsCompiled>>,
    CachedDc: Fn(&Params, CollocationFamily) -> Result<CachedCompile<DcCompiled>>,
    MsRuntimeFn: Fn(
        &Params,
    ) -> MultipleShootingRuntimeValues<
        MsCompiled::PNum,
        MsCompiled::CBounds,
        MsCompiled::BeqNum,
        MsCompiled::BineqBounds,
        MsCompiled::XNum,
        MsCompiled::UNum,
    >,
    DcRuntimeFn: Fn(
        &Params,
    ) -> DirectCollocationRuntimeValues<
        DcCompiled::PNum,
        DcCompiled::CBounds,
        DcCompiled::BeqNum,
        DcCompiled::BineqBounds,
        DcCompiled::XNum,
        DcCompiled::UNum,
    >,
    MsArtifact: FnMut(
        &MultipleShootingTrajectories<MsCompiled::XNum, MsCompiled::UNum>,
        &[IntervalArc<MsCompiled::XNum>],
        &[IntervalArc<MsCompiled::UNum>],
    ) -> SolveArtifact,
    DcArtifact: FnMut(
        &DirectCollocationTrajectories<DcCompiled::XNum, DcCompiled::UNum>,
        &DirectCollocationTimeGrid,
    ) -> SolveArtifact,
{
    match transcription {
        TranscriptionMethod::MultipleShooting => {
            let compiled = cached_multiple_shooting(params)?;
            solve_cached_multiple_shooting_problem(
                &compiled.compiled,
                &multiple_shooting_runtime(params),
                solver_method,
                solver_config,
                multiple_shooting_artifact,
            )
        }
        TranscriptionMethod::DirectCollocation => {
            let compiled = cached_direct_collocation(params, collocation_family)?;
            solve_cached_direct_collocation_problem(
                &compiled.compiled,
                &direct_collocation_runtime(params),
                solver_method,
                solver_config,
                direct_collocation_artifact,
            )
        }
    }
}

pub fn solve_standard_ocp_with_progress<
    Params,
    Emit,
    MsCompiled,
    DcCompiled,
    CompileMs,
    CompileDc,
    MsRuntimeFn,
    DcRuntimeFn,
    MsArtifact,
    DcArtifact,
>(
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    emit: Emit,
    compile_multiple_shooting: CompileMs,
    compile_direct_collocation: CompileDc,
    multiple_shooting_runtime: MsRuntimeFn,
    direct_collocation_runtime: DcRuntimeFn,
    multiple_shooting_artifact: MsArtifact,
    direct_collocation_artifact: DcArtifact,
) -> Result<SolveArtifact>
where
    Emit: FnMut(SolveStreamEvent) + Send,
    MsCompiled: MultipleShootingCompiled,
    DcCompiled: DirectCollocationCompiled,
    CompileMs: Fn(
        &Params,
        &mut dyn FnMut(CompileProgressUpdate),
    ) -> Result<(Rc<RefCell<MsCompiled>>, CompileProgressInfo)>,
    CompileDc: Fn(
        &Params,
        CollocationFamily,
        &mut dyn FnMut(CompileProgressUpdate),
    ) -> Result<(Rc<RefCell<DcCompiled>>, CompileProgressInfo)>,
    MsRuntimeFn: Fn(
        &Params,
    ) -> MultipleShootingRuntimeValues<
        MsCompiled::PNum,
        MsCompiled::CBounds,
        MsCompiled::BeqNum,
        MsCompiled::BineqBounds,
        MsCompiled::XNum,
        MsCompiled::UNum,
    >,
    DcRuntimeFn: Fn(
        &Params,
    ) -> DirectCollocationRuntimeValues<
        DcCompiled::PNum,
        DcCompiled::CBounds,
        DcCompiled::BeqNum,
        DcCompiled::BineqBounds,
        DcCompiled::XNum,
        DcCompiled::UNum,
    >,
    MsArtifact: FnMut(
        &MultipleShootingTrajectories<MsCompiled::XNum, MsCompiled::UNum>,
        &[IntervalArc<MsCompiled::XNum>],
        &[IntervalArc<MsCompiled::UNum>],
    ) -> SolveArtifact,
    DcArtifact: FnMut(
        &DirectCollocationTrajectories<DcCompiled::XNum, DcCompiled::UNum>,
        &DirectCollocationTimeGrid,
    ) -> SolveArtifact,
{
    let mut lifecycle = SolveLifecycleReporter::new(emit, solver_method);
    match transcription {
        TranscriptionMethod::MultipleShooting => {
            let (compiled, running_solver, compile_report) = lifecycle
                .compile_with_progress(|callback| compile_multiple_shooting(params, callback))?;
            let mut artifact = solve_cached_multiple_shooting_problem_with_progress(
                &compiled,
                &multiple_shooting_runtime(params),
                solver_method,
                solver_config,
                lifecycle.into_emit(),
                running_solver,
                multiple_shooting_artifact,
            )?;
            artifact.compile_report = compile_report;
            Ok(artifact)
        }
        TranscriptionMethod::DirectCollocation => {
            let (compiled, running_solver, compile_report) =
                lifecycle.compile_with_progress(|callback| {
                    compile_direct_collocation(params, collocation_family, callback)
                })?;
            let mut artifact = solve_cached_direct_collocation_problem_with_progress(
                &compiled,
                &direct_collocation_runtime(params),
                solver_method,
                solver_config,
                lifecycle.into_emit(),
                running_solver,
                direct_collocation_artifact,
            )?;
            artifact.compile_report = compile_report;
            Ok(artifact)
        }
    }
}

pub fn benchmark_standard_ocp_case_with_progress<
    Params,
    MsCompiled,
    DcCompiled,
    CompileMs,
    CompileDc,
    MsRuntimeFn,
    DcRuntimeFn,
    E,
>(
    problem_id: ProblemId,
    problem_name: &str,
    transcription: TranscriptionMethod,
    preset: crate::benchmark_report::OcpBenchmarkPreset,
    eval_options: NlpEvaluationBenchmarkOptions,
    on_progress: &mut dyn FnMut(crate::benchmark_report::BenchmarkCaseProgress),
    compile_multiple_shooting: CompileMs,
    compile_direct_collocation: CompileDc,
    multiple_shooting_runtime: MsRuntimeFn,
    direct_collocation_runtime: DcRuntimeFn,
) -> Result<crate::benchmark_report::OcpBenchmarkRecord>
where
    Params: Default + StandardOcpParams,
    MsCompiled: MultipleShootingCompiled,
    DcCompiled: DirectCollocationCompiled,
    CompileMs: FnOnce(
        OcpCompileOptions,
        &mut dyn FnMut(OcpCompileProgress),
    ) -> std::result::Result<MsCompiled, E>,
    CompileDc: FnOnce(
        CollocationFamily,
        OcpCompileOptions,
        &mut dyn FnMut(OcpCompileProgress),
    ) -> std::result::Result<DcCompiled, E>,
    MsRuntimeFn: Fn(
        &Params,
    ) -> MultipleShootingRuntimeValues<
        MsCompiled::PNum,
        MsCompiled::CBounds,
        MsCompiled::BeqNum,
        MsCompiled::BineqBounds,
        MsCompiled::XNum,
        MsCompiled::UNum,
    >,
    DcRuntimeFn: Fn(
        &Params,
    ) -> DirectCollocationRuntimeValues<
        DcCompiled::PNum,
        DcCompiled::CBounds,
        DcCompiled::BeqNum,
        DcCompiled::BineqBounds,
        DcCompiled::XNum,
        DcCompiled::UNum,
    >,
    E: Into<anyhow::Error>,
{
    let mut params = Params::default();
    params.transcription_mut().method = transcription;
    let opt_level = crate::benchmark_report::opt_level_for_transcription(transcription);
    let compile_options = preset.compile_options(opt_level);
    match transcription {
        TranscriptionMethod::MultipleShooting => {
            let compiled = compile_multiple_shooting(compile_options, &mut |progress| {
                on_progress(crate::benchmark_report::BenchmarkCaseProgress::Compile(
                    progress,
                ));
            })
            .map_err(Into::into)?;
            let eval = compiled.benchmark_nlp_evaluations_with_progress(
                &multiple_shooting_runtime(&params),
                eval_options,
                |kernel| {
                    on_progress(
                        crate::benchmark_report::BenchmarkCaseProgress::EvalKernelStarted(kernel),
                    )
                },
            )?;
            Ok(crate::benchmark_report::build_record(
                problem_id,
                problem_name,
                transcription,
                None,
                preset,
                opt_level,
                summarize_backend_compile_report(compiled.backend_compile_report()),
                compiled.helper_compile_stats(),
                compiled.nlp_compile_stats(),
                compiled.helper_kernel_count(),
                eval,
            ))
        }
        TranscriptionMethod::DirectCollocation => {
            let family = params.transcription().collocation_family;
            let compiled = compile_direct_collocation(family, compile_options, &mut |progress| {
                on_progress(crate::benchmark_report::BenchmarkCaseProgress::Compile(
                    progress,
                ));
            })
            .map_err(Into::into)?;
            let eval = compiled.benchmark_nlp_evaluations_with_progress(
                &direct_collocation_runtime(&params),
                eval_options,
                |kernel| {
                    on_progress(
                        crate::benchmark_report::BenchmarkCaseProgress::EvalKernelStarted(kernel),
                    )
                },
            )?;
            Ok(crate::benchmark_report::build_record(
                problem_id,
                problem_name,
                transcription,
                Some(family),
                preset,
                opt_level,
                summarize_backend_compile_report(compiled.backend_compile_report()),
                compiled.helper_compile_stats(),
                compiled.nlp_compile_stats(),
                compiled.helper_kernel_count(),
                eval,
            ))
        }
    }
}

pub fn validate_standard_ocp_derivatives<
    Params,
    MsCompiled,
    DcCompiled,
    CachedMs,
    CachedDc,
    MsRuntimeFn,
    DcRuntimeFn,
>(
    problem_id: ProblemId,
    problem_name: &str,
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    sx_functions: OcpSxFunctionConfig,
    request: &DerivativeCheckRequest,
    cached_multiple_shooting: CachedMs,
    cached_direct_collocation: CachedDc,
    multiple_shooting_runtime: MsRuntimeFn,
    direct_collocation_runtime: DcRuntimeFn,
) -> Result<ProblemDerivativeCheck>
where
    MsCompiled: MultipleShootingCompiled,
    DcCompiled: DirectCollocationCompiled,
    CachedMs: Fn(&Params) -> Result<CachedCompile<MsCompiled>>,
    CachedDc: Fn(&Params, CollocationFamily) -> Result<CachedCompile<DcCompiled>>,
    MsRuntimeFn: Fn(
        &Params,
    ) -> MultipleShootingRuntimeValues<
        MsCompiled::PNum,
        MsCompiled::CBounds,
        MsCompiled::BeqNum,
        MsCompiled::BineqBounds,
        MsCompiled::XNum,
        MsCompiled::UNum,
    >,
    DcRuntimeFn: Fn(
        &Params,
    ) -> DirectCollocationRuntimeValues<
        DcCompiled::PNum,
        DcCompiled::CBounds,
        DcCompiled::BeqNum,
        DcCompiled::BineqBounds,
        DcCompiled::XNum,
        DcCompiled::UNum,
    >,
{
    match transcription {
        TranscriptionMethod::MultipleShooting => {
            let compiled = cached_multiple_shooting(params)?;
            let runtime = multiple_shooting_runtime(params);
            let compiled_ref = compiled.compiled.borrow();
            let stats = compiled_ref.nlp_compile_stats();
            let report = compiled_ref.validate_nlp_derivatives(
                &runtime,
                &vec![request.equality_multiplier_fill; stats.equality_count],
                &vec![request.inequality_multiplier_fill; stats.inequality_count],
                request.finite_difference,
            )?;
            Ok(ProblemDerivativeCheck {
                problem_id,
                problem_name: problem_name.to_string(),
                transcription,
                collocation_family: None,
                sx_functions,
                compile_cached: compile_is_fully_disk_cached(
                    Some(&summarize_backend_compile_report(
                        compiled_ref.backend_compile_report(),
                    )),
                    compiled_ref.helper_compile_stats(),
                ),
                compile_report: summarize_backend_compile_report(
                    compiled_ref.backend_compile_report(),
                ),
                report,
            })
        }
        TranscriptionMethod::DirectCollocation => {
            let compiled = cached_direct_collocation(params, collocation_family)?;
            let runtime = direct_collocation_runtime(params);
            let compiled_ref = compiled.compiled.borrow();
            let stats = compiled_ref.nlp_compile_stats();
            let report = compiled_ref.validate_nlp_derivatives(
                &runtime,
                &vec![request.equality_multiplier_fill; stats.equality_count],
                &vec![request.inequality_multiplier_fill; stats.inequality_count],
                request.finite_difference,
            )?;
            Ok(ProblemDerivativeCheck {
                problem_id,
                problem_name: problem_name.to_string(),
                transcription,
                collocation_family: Some(collocation_family),
                sx_functions,
                compile_cached: compile_is_fully_disk_cached(
                    Some(&summarize_backend_compile_report(
                        compiled_ref.backend_compile_report(),
                    )),
                    compiled_ref.helper_compile_stats(),
                ),
                compile_report: summarize_backend_compile_report(
                    compiled_ref.backend_compile_report(),
                ),
                report,
            })
        }
    }
}

pub fn solve_from_value_map<Params, Solve>(
    values: &BTreeMap<String, f64>,
    solve: Solve,
) -> Result<SolveArtifact>
where
    Params: FromMap,
    Solve: FnOnce(&Params) -> Result<SolveArtifact>,
{
    solve(&Params::from_map(values)?)
}

pub fn prewarm_from_value_map<Params, Prewarm>(
    values: &BTreeMap<String, f64>,
    prewarm: Prewarm,
) -> Result<()>
where
    Params: FromMap,
    Prewarm: FnOnce(&Params) -> Result<()>,
{
    prewarm(&Params::from_map(values)?)
}

pub fn solve_with_progress_from_value_map<Params, Solve>(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
    solve: Solve,
) -> Result<SolveArtifact>
where
    Params: FromMap,
    Solve: FnOnce(&Params, Box<dyn FnMut(SolveStreamEvent) + Send>) -> Result<SolveArtifact>,
{
    solve(&Params::from_map(values)?, emit)
}

pub fn prewarm_with_progress_from_value_map<Params, Prewarm>(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
    prewarm: Prewarm,
) -> Result<()>
where
    Params: FromMap,
    Prewarm: FnOnce(&Params, Box<dyn FnMut(SolveStreamEvent) + Send>) -> Result<()>,
{
    prewarm(&Params::from_map(values)?, emit)
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

fn prepend_solver_settings_detail(
    mut details: Vec<SolverPhaseDetail>,
    settings: String,
) -> Vec<SolverPhaseDetail> {
    details.insert(0, phase_detail("Settings", settings, 0));
    details
}

pub fn compile_cache_status(
    problem_id: ProblemId,
    problem_name: &str,
    variant_id: &str,
    variant_label: &str,
    timing: BackendTimingMetadata,
    jit_disk_cache_hit: bool,
) -> CompileCacheStatus {
    CompileCacheStatus {
        problem_id,
        problem_name: problem_name.to_string(),
        variant_id: variant_id.to_string(),
        variant_label: variant_label.to_string(),
        state: CompileCacheState::Ready,
        symbolic_setup_s: symbolic_setup_seconds(timing),
        jit_s: duration_seconds(timing.jit_time),
        jit_disk_cache_hit,
    }
}

pub fn multiple_shooting_variant() -> (&'static str, &'static str) {
    ("multiple_shooting", "Multiple Shooting")
}

pub fn multiple_shooting_variant_with_sx(key: MultipleShootingCompileKey) -> (String, String) {
    let (base_id, base_label) = multiple_shooting_variant();
    let (base_id, base_label) =
        with_transcription_size_suffix(base_id, base_label, key.intervals, None);
    with_sx_variant_suffix(&base_id, &base_label, key.sx_functions)
}

pub fn direct_collocation_compile_key(family: CollocationFamily) -> DirectCollocationCompileKey {
    match family {
        CollocationFamily::GaussLegendre => DirectCollocationCompileKey::Legendre,
        CollocationFamily::RadauIIA => DirectCollocationCompileKey::RadauIia,
    }
}

pub fn time_grid_compile_key(time_grid: TimeGrid) -> TimeGridCompileKey {
    match time_grid {
        TimeGrid::Uniform => TimeGridCompileKey::Uniform,
        TimeGrid::Cosine { strength } => TimeGridCompileKey::Cosine {
            strength_bits: strength.to_bits(),
        },
        TimeGrid::Tanh { strength } => TimeGridCompileKey::Tanh {
            strength_bits: strength.to_bits(),
        },
        TimeGrid::Geometric { strength, bias } => TimeGridCompileKey::Geometric {
            strength_bits: strength.to_bits(),
            bias,
        },
        TimeGrid::Focus {
            center,
            width,
            strength,
        } => TimeGridCompileKey::Focus {
            center_bits: center.to_bits(),
            width_bits: width.to_bits(),
            strength_bits: strength.to_bits(),
        },
        TimeGrid::Piecewise {
            breakpoint,
            first_interval_fraction,
        } => TimeGridCompileKey::Piecewise {
            breakpoint_bits: breakpoint.to_bits(),
            first_interval_fraction_bits: first_interval_fraction.to_bits(),
        },
    }
}

pub fn direct_collocation_variant(
    key: DirectCollocationCompileKey,
) -> (&'static str, &'static str) {
    match key {
        DirectCollocationCompileKey::Legendre => (
            "direct_collocation_legendre",
            "Direct Collocation · Legendre",
        ),
        DirectCollocationCompileKey::RadauIia => (
            "direct_collocation_radau_iia",
            "Direct Collocation · Radau IIA",
        ),
    }
}

pub fn direct_collocation_variant_with_sx(
    key: DirectCollocationCompileVariantKey,
) -> (String, String) {
    let (base_id, base_label) = direct_collocation_variant(key.family);
    let (base_id, base_label) =
        with_transcription_size_suffix(base_id, base_label, key.intervals, Some(key.order));
    let (grid_id, grid_label) = time_grid_variant_suffix(key.time_grid);
    let variant_id = if let Some(grid_id) = grid_id {
        format!("{base_id}__{grid_id}")
    } else {
        base_id
    };
    let variant_label = if let Some(grid_label) = grid_label {
        format!("{base_label} · {grid_label}")
    } else {
        base_label
    };
    with_sx_variant_suffix(&variant_id, &variant_label, key.sx_functions)
}

fn with_transcription_size_suffix(
    base_id: &str,
    base_label: &str,
    intervals: usize,
    collocation_order: Option<usize>,
) -> (String, String) {
    let mut variant_id = format!("{base_id}__n{intervals}");
    let mut variant_label = format!("{base_label} · {intervals} intervals");
    if let Some(order) = collocation_order {
        variant_id = format!("{variant_id}_k{order}");
        variant_label = format!("{variant_label} · {order} nodes");
    }
    (variant_id, variant_label)
}

fn time_grid_variant_suffix(key: TimeGridCompileKey) -> (Option<String>, Option<String>) {
    match key {
        TimeGridCompileKey::Uniform => (None, None),
        TimeGridCompileKey::Cosine { strength_bits } => {
            let strength = f64::from_bits(strength_bits);
            (
                Some(format!("cosine_{strength_bits:016x}")),
                Some(format!("Cosine Grid s={strength:.2}")),
            )
        }
        TimeGridCompileKey::Tanh { strength_bits } => {
            let strength = f64::from_bits(strength_bits);
            (
                Some(format!("tanh_{strength_bits:016x}")),
                Some(format!("Tanh Grid s={strength:.2}")),
            )
        }
        TimeGridCompileKey::Geometric {
            strength_bits,
            bias,
        } => {
            let strength = f64::from_bits(strength_bits);
            let (bias_id, bias_label) = match bias {
                TimeGridBias::Start => ("start", "Start"),
                TimeGridBias::End => ("end", "End"),
            };
            (
                Some(format!("geometric_{bias_id}_{strength_bits:016x}")),
                Some(format!("Geometric {bias_label} Grid s={strength:.2}")),
            )
        }
        TimeGridCompileKey::Focus {
            center_bits,
            width_bits,
            strength_bits,
        } => {
            let center = f64::from_bits(center_bits);
            let width = f64::from_bits(width_bits);
            let strength = f64::from_bits(strength_bits);
            (
                Some(format!(
                    "focus_{center_bits:016x}_{width_bits:016x}_{strength_bits:016x}"
                )),
                Some(format!(
                    "Focus Grid c={center:.2} w={width:.2} s={strength:.2}"
                )),
            )
        }
        TimeGridCompileKey::Piecewise {
            breakpoint_bits,
            first_interval_fraction_bits,
        } => {
            let breakpoint = f64::from_bits(breakpoint_bits);
            let first_interval_fraction = f64::from_bits(first_interval_fraction_bits);
            (
                Some(format!(
                    "piecewise_{breakpoint_bits:016x}_{first_interval_fraction_bits:016x}"
                )),
                Some(format!(
                    "Piecewise Grid b={breakpoint:.2} n1={first_interval_fraction:.2}"
                )),
            )
        }
    }
}

fn with_sx_variant_suffix(
    base_id: &str,
    base_label: &str,
    sx_functions: OcpSxFunctionConfig,
) -> (String, String) {
    let variant_id = format!("{base_id}__{}", sx_functions.variant_id_suffix());
    let variant_label = if let Some(suffix) = sx_functions.variant_label_suffix() {
        format!("{base_label} · {suffix}")
    } else {
        base_label.to_string()
    };
    (variant_id, variant_label)
}

pub fn append_standard_compile_cache_statuses<Ms, Dc>(
    statuses: &mut Vec<CompileCacheStatus>,
    problem_id: ProblemId,
    problem_name: &str,
    multiple_shooting_cache: &SharedCompileCache<MultipleShootingCompileKey, Ms>,
    direct_collocation_cache: &SharedCompileCache<DirectCollocationCompileVariantKey, Dc>,
) where
    Ms: CompiledOcpMetadata,
    Dc: CompiledOcpMetadata,
{
    statuses.extend(collect_compile_cache_statuses(
        problem_id,
        problem_name,
        multiple_shooting_cache,
        multiple_shooting_variant_with_sx,
    ));
    statuses.extend(collect_compile_cache_statuses(
        problem_id,
        problem_name,
        direct_collocation_cache,
        direct_collocation_variant_with_sx,
    ));
}

pub fn collect_compile_cache_statuses<K, V, G>(
    problem_id: ProblemId,
    problem_name: &str,
    cache: &SharedCompileCache<K, V>,
    describe_variant: G,
) -> Vec<CompileCacheStatus>
where
    K: Eq + Hash + Copy,
    V: CompiledOcpMetadata,
    G: Fn(K) -> (String, String),
{
    cache
        .cached_entries()
        .into_iter()
        .map(|(key, compiled)| {
            let (variant_id, variant_label) = describe_variant(key);
            let compiled = compiled.borrow();
            compile_cache_status(
                problem_id,
                problem_name,
                &variant_id,
                &variant_label,
                compiled.backend_timing_metadata(),
                compile_is_fully_disk_cached(
                    Some(&summarize_backend_compile_report(
                        compiled.backend_compile_report(),
                    )),
                    compiled.helper_compile_stats(),
                ),
            )
        })
        .collect()
}

fn upsert_phase_detail(
    details: &mut Vec<SolverPhaseDetail>,
    label: impl Into<String>,
    value: impl Into<String>,
) {
    let label = label.into();
    let value = value.into();
    if let Some(detail) = details.iter_mut().find(|detail| detail.label == label) {
        detail.value = value;
    } else {
        details.push(phase_detail(label, value, 0));
    }
}

fn upsert_timing_phase_detail(
    details: &mut Vec<SolverPhaseDetail>,
    label: &str,
    count: usize,
    duration: Duration,
) {
    if duration <= Duration::ZERO {
        return;
    }
    let value = format_phase_duration(duration);
    if let Some(detail) = details.iter_mut().find(|detail| detail.label == label) {
        detail.value = value;
        detail.count = count;
    } else {
        details.push(phase_detail(label, value, count));
    }
}

fn append_ocp_setup_timing_details(
    solver: &mut SolverReport,
    timing: optimal_control::OcpSolveSetupTiming,
) {
    upsert_timing_phase_detail(
        &mut solver.phase_details.solve,
        "Initial Guess Construction",
        usize::from(timing.initial_guess > Duration::ZERO),
        timing.initial_guess,
    );
    upsert_timing_phase_detail(
        &mut solver.phase_details.solve,
        "Runtime Bounds Construction",
        usize::from(timing.runtime_bounds > Duration::ZERO),
        timing.runtime_bounds,
    );
}

fn format_phase_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 10.0 {
        format!("{seconds:.1} s")
    } else if seconds >= 1.0 {
        format!("{seconds:.2} s")
    } else if seconds >= 0.1 {
        format!("{:.0} ms", seconds * 1000.0)
    } else {
        format!("{:.1} ms", seconds * 1000.0)
    }
}

fn symbolic_phase_details(
    stats: NlpCompileStats,
    setup_profile: Option<&SymbolicSetupProfile>,
) -> Vec<SolverPhaseDetail> {
    let total_jacobian_nnz = stats.equality_jacobian_nnz + stats.inequality_jacobian_nnz;
    let mut details = vec![
        phase_detail("Vars", stats.variable_count.to_string(), 0),
        phase_detail("Params", stats.parameter_scalar_count.to_string(), 0),
        phase_detail("Eq", stats.equality_count.to_string(), 0),
        phase_detail("Ineq", stats.inequality_count.to_string(), 0),
        phase_detail("Jac NNZ", total_jacobian_nnz.to_string(), 0),
        phase_detail("Hess NNZ", stats.hessian_nnz.to_string(), 0),
    ];
    if let Some(profile) = setup_profile {
        if let Some(duration) = profile.symbolic_construction {
            details.push(phase_detail(
                "Build Problem",
                format_phase_duration(duration),
                0,
            ));
        }
        if let Some(duration) = profile.objective_gradient {
            details.push(phase_detail(
                "Objective Gradient",
                format_phase_duration(duration),
                0,
            ));
        }
        if let Some(duration) = profile.equality_jacobian {
            details.push(phase_detail(
                "Equality Jacobian",
                format_phase_duration(duration),
                0,
            ));
        }
        if let Some(duration) = profile.inequality_jacobian {
            details.push(phase_detail(
                "Inequality Jacobian",
                format_phase_duration(duration),
                0,
            ));
        }
        if let Some(duration) = profile.lagrangian_assembly {
            details.push(phase_detail(
                "Lagrangian Assembly",
                format_phase_duration(duration),
                0,
            ));
        }
        if let Some(duration) = profile.hessian_generation {
            details.push(phase_detail(
                "Hessian Generation",
                format_phase_duration(duration),
                0,
            ));
        }
    }
    details
}

fn jit_phase_details(stats: NlpCompileStats, helper_kernel_count: usize) -> Vec<SolverPhaseDetail> {
    vec![
        phase_detail("NLP Kernels", stats.jit_kernel_count.to_string(), 0),
        phase_detail("Helper Kernels", helper_kernel_count.to_string(), 0),
    ]
}

fn helper_compile_detail_label(helper: OcpCompileHelperKind) -> &'static str {
    match helper {
        OcpCompileHelperKind::Xdot => "Xdot Helper",
        OcpCompileHelperKind::MultipleShootingArc => "RK4 Arc Helper",
    }
}

fn helper_compile_phase_details(helper_stats: OcpHelperCompileStats) -> Vec<SolverPhaseDetail> {
    let mut details = Vec::new();
    if let Some(duration) = helper_stats.xdot_helper_time {
        details.push(phase_detail(
            "Xdot Helper",
            format_phase_duration(duration),
            0,
        ));
    }
    if let Some(duration) = helper_stats.multiple_shooting_arc_helper_time {
        details.push(phase_detail(
            "RK4 Arc Helper",
            format_phase_duration(duration),
            0,
        ));
    }
    details
}

fn total_llvm_cache_hits(
    compile_report: Option<&CompileReportSummary>,
    helper_stats: OcpHelperCompileStats,
) -> usize {
    compile_report.map_or(0, |report| report.llvm_cache_hits) + helper_stats.llvm_cache_hits
}

fn total_llvm_cache_misses(
    compile_report: Option<&CompileReportSummary>,
    helper_stats: OcpHelperCompileStats,
) -> usize {
    compile_report.map_or(0, |report| report.llvm_cache_misses) + helper_stats.llvm_cache_misses
}

fn llvm_cache_load_seconds(
    compile_report: Option<&CompileReportSummary>,
    helper_stats: OcpHelperCompileStats,
) -> Option<f64> {
    let nlp = compile_report
        .and_then(|report| report.llvm_cache_load_s)
        .unwrap_or(0.0);
    let helper = duration_seconds(Some(helper_stats.llvm_cache_load_time)).unwrap_or(0.0);
    let total = nlp + helper;
    (total > 0.0).then_some(total)
}

fn compile_is_fully_disk_cached(
    compile_report: Option<&CompileReportSummary>,
    helper_stats: OcpHelperCompileStats,
) -> bool {
    let hits = total_llvm_cache_hits(compile_report, helper_stats);
    let misses = total_llvm_cache_misses(compile_report, helper_stats);
    hits > 0 && misses == 0
}

fn llvm_cache_phase_details(
    compile_report: Option<&CompileReportSummary>,
    helper_stats: OcpHelperCompileStats,
) -> Vec<SolverPhaseDetail> {
    let hits = total_llvm_cache_hits(compile_report, helper_stats);
    let misses = total_llvm_cache_misses(compile_report, helper_stats);
    if hits == 0 && misses == 0 {
        return Vec::new();
    }

    let mut details = vec![
        phase_detail("LLVM Cache Hits", hits.to_string(), 0),
        phase_detail("LLVM Cache Misses", misses.to_string(), 0),
    ];
    if let Some(load_s) = llvm_cache_load_seconds(compile_report, helper_stats) {
        if load_s > 0.0 {
            details.push(phase_detail(
                "LLVM Cache Load",
                format_phase_duration(Duration::from_secs_f64(load_s)),
                0,
            ));
        }
    }
    details
}

pub fn ocp_compile_progress_update(
    progress: OcpCompileProgress,
    state: &mut OcpCompileProgressState,
) -> CompileProgressUpdate {
    match progress {
        OcpCompileProgress::SymbolicStage(progress) => {
            state.timing = progress.metadata.timing;
            state.phase_details.symbolic_setup = symbolic_phase_details(
                progress.metadata.stats,
                Some(&progress.metadata.setup_profile),
            );
        }
        OcpCompileProgress::SymbolicReady(metadata) => {
            state.timing = metadata.timing;
            state.phase_details.symbolic_setup =
                symbolic_phase_details(metadata.stats, Some(&metadata.setup_profile));
            state.phase_details.jit = vec![phase_detail(
                "NLP Kernels",
                metadata.stats.jit_kernel_count.to_string(),
                0,
            )];
        }
        OcpCompileProgress::NlpKernelCompiled { elapsed, .. } => {
            upsert_phase_detail(
                &mut state.phase_details.jit,
                "NLP Kernel JIT",
                format_phase_duration(elapsed),
            );
        }
        OcpCompileProgress::HelperCompiled {
            helper, elapsed, ..
        } => {
            upsert_phase_detail(
                &mut state.phase_details.jit,
                helper_compile_detail_label(helper),
                format_phase_duration(elapsed),
            );
        }
    }
    CompileProgressUpdate {
        timing: state.timing,
        phase_details: state.phase_details.clone(),
        compile_cached: false,
    }
}

pub fn compile_progress_info(
    timing: BackendTimingMetadata,
    stats: NlpCompileStats,
    helper_kernel_count: usize,
    helper_stats: OcpHelperCompileStats,
    compile_report: Option<CompileReportSummary>,
) -> CompileProgressInfo {
    let mut jit = jit_phase_details(stats, helper_kernel_count);
    jit.extend(helper_compile_phase_details(helper_stats));
    jit.extend(llvm_cache_phase_details(
        compile_report.as_ref(),
        helper_stats,
    ));
    CompileProgressInfo {
        timing,
        compile_cached: compile_is_fully_disk_cached(compile_report.as_ref(), helper_stats),
        phase_details: SolverPhaseDetails {
            symbolic_setup: symbolic_phase_details(stats, None),
            jit,
            solve: Vec::new(),
        },
        compile_report,
    }
}

pub fn cached_multiple_shooting_ocp_compile<Compiled, Build, E>(
    cache: &mut SharedCompileCache<MultipleShootingCompileKey, Compiled>,
    intervals: usize,
    sx_functions: OcpSxFunctionConfig,
    build: Build,
) -> Result<CachedCompile<Compiled>>
where
    Build: FnOnce(OcpCompileOptions) -> std::result::Result<Compiled, E>,
    E: Into<anyhow::Error>,
{
    cache.get_or_try_init(
        multiple_shooting_compile_key(intervals, sx_functions),
        || {
            build(ocp_compile_options(
                interactive_multiple_shooting_opt_level(),
                sx_functions,
            ))
            .map_err(Into::into)
        },
    )
}

pub fn cached_direct_collocation_ocp_compile<Compiled, Build, E>(
    cache: &mut SharedCompileCache<DirectCollocationCompileVariantKey, Compiled>,
    intervals: usize,
    order: usize,
    family: CollocationFamily,
    time_grid: TimeGrid,
    sx_functions: OcpSxFunctionConfig,
    build: Build,
) -> Result<CachedCompile<Compiled>>
where
    Build: FnOnce(OcpCompileOptions) -> std::result::Result<Compiled, E>,
    E: Into<anyhow::Error>,
{
    cache.get_or_try_init(
        direct_collocation_compile_key_with_sx(intervals, order, family, time_grid, sx_functions),
        || {
            build(ocp_compile_options(
                interactive_direct_collocation_opt_level(),
                sx_functions,
            ))
            .map_err(Into::into)
        },
    )
}

pub fn cached_multiple_shooting_ocp_compile_with_progress<Compiled, Build, Summary, E>(
    cache: &mut SharedCompileCache<MultipleShootingCompileKey, Compiled>,
    intervals: usize,
    sx_functions: OcpSxFunctionConfig,
    on_symbolic_ready: &mut dyn FnMut(CompileProgressUpdate),
    build: Build,
    summary: Summary,
) -> Result<(Rc<RefCell<Compiled>>, CompileProgressInfo)>
where
    Build: FnOnce(
        OcpCompileOptions,
        &mut dyn FnMut(OcpCompileProgress),
    ) -> std::result::Result<Compiled, E>,
    Summary: Fn(&Compiled) -> CompileProgressInfo,
    E: Into<anyhow::Error>,
{
    cached_compile_with_progress(
        cache,
        multiple_shooting_compile_key(intervals, sx_functions),
        on_symbolic_ready,
        |on_compile_progress| {
            let mut progress_state = OcpCompileProgressState::default();
            build(
                ocp_compile_options(interactive_multiple_shooting_opt_level(), sx_functions),
                &mut |progress| {
                    on_compile_progress(ocp_compile_progress_update(progress, &mut progress_state));
                },
            )
            .map_err(Into::into)
        },
        summary,
    )
}

pub fn cached_direct_collocation_ocp_compile_with_progress<Compiled, Build, Summary, E>(
    cache: &mut SharedCompileCache<DirectCollocationCompileVariantKey, Compiled>,
    intervals: usize,
    order: usize,
    family: CollocationFamily,
    time_grid: TimeGrid,
    sx_functions: OcpSxFunctionConfig,
    on_symbolic_ready: &mut dyn FnMut(CompileProgressUpdate),
    build: Build,
    summary: Summary,
) -> Result<(Rc<RefCell<Compiled>>, CompileProgressInfo)>
where
    Build: FnOnce(
        OcpCompileOptions,
        &mut dyn FnMut(OcpCompileProgress),
    ) -> std::result::Result<Compiled, E>,
    Summary: Fn(&Compiled) -> CompileProgressInfo,
    E: Into<anyhow::Error>,
{
    cached_compile_with_progress(
        cache,
        direct_collocation_compile_key_with_sx(intervals, order, family, time_grid, sx_functions),
        on_symbolic_ready,
        |on_compile_progress| {
            let mut progress_state = OcpCompileProgressState::default();
            build(
                ocp_compile_options(interactive_direct_collocation_opt_level(), sx_functions),
                &mut |progress| {
                    on_compile_progress(ocp_compile_progress_update(progress, &mut progress_state));
                },
            )
            .map_err(Into::into)
        },
        summary,
    )
}

pub fn cached_compile_with_progress<K, V, Build, Timing>(
    cache: &mut SharedCompileCache<K, V>,
    key: K,
    on_symbolic_ready: &mut dyn FnMut(CompileProgressUpdate),
    build: Build,
    timing_of: Timing,
) -> Result<(Rc<RefCell<V>>, CompileProgressInfo)>
where
    K: Eq + Hash + Copy,
    Build: FnOnce(&mut dyn FnMut(CompileProgressUpdate)) -> Result<V>,
    Timing: Fn(&V) -> CompileProgressInfo,
{
    let cached = cache.get_or_try_init(key, || build(on_symbolic_ready))?;
    let progress = timing_of(&cached.compiled.borrow());
    if cached.was_cached {
        on_symbolic_ready(CompileProgressUpdate {
            timing: pre_jit_backend_timing(progress.timing),
            phase_details: progress.phase_details.clone(),
            compile_cached: progress.compile_cached,
        });
    }
    Ok((
        cached.compiled,
        CompileProgressInfo {
            compile_cached: progress.compile_cached,
            ..progress
        },
    ))
}

pub fn default_transcription(intervals: usize) -> TranscriptionConfig {
    TranscriptionConfig {
        method: TranscriptionMethod::DirectCollocation,
        intervals,
        collocation_degree: 3,
        collocation_family: CollocationFamily::GaussLegendre,
        time_grid: TimeGrid::Uniform,
    }
}

pub fn default_sqp_line_search_config() -> SqpLineSearchConfig {
    SqpLineSearchConfig {
        armijo_c1: 1.0e-4,
        wolfe_c2: None,
        beta: 5.0e-1,
        max_steps: 32,
        min_step: 1.0e-8,
    }
}

pub fn default_sqp_filter_config() -> SqpFilterConfig {
    SqpFilterConfig {
        gamma_objective: 1.0e-4,
        gamma_violation: 1.0e-4,
        theta_max_factor: 1.0e3,
        switching_reference_min: 1.0e-3,
        switching_violation_factor: 10.0,
        switching_linearized_reduction_factor: 5.0e-1,
    }
}

pub fn default_sqp_trust_region_config() -> SqpTrustRegionConfig {
    SqpTrustRegionConfig {
        initial_radius: 1.0,
        max_radius: 100.0,
        min_radius: 1.0e-8,
        shrink_factor: 2.5e-1,
        grow_factor: 2.0,
        accept_ratio: 1.0e-1,
        expand_ratio: 7.5e-1,
        boundary_fraction: 8.0e-1,
        max_radius_contractions: 8,
    }
}

pub fn default_sqp_method_config() -> SqpMethodConfig {
    SqpMethodConfig {
        hessian_regularization_enabled: false,
        globalization: SqpGlobalizationConfig::TrustRegionFilter {
            trust_region: default_sqp_trust_region_config(),
            filter: default_sqp_filter_config(),
            exact_merit_penalty: 10.0,
        },
    }
}

pub const fn default_nlip_config() -> NlipConfig {
    NlipConfig {
        // IPOPT 3.14.20 `IpSpralSolverInterface` defaults to native SPRAL with
        // matching ordering/scaling and `spral_pivot_method=block`. IPOPT maps
        // that option to raw SPRAL control value 1, which corresponds to
        // SPRAL's aggressive APP constant in the native C/Fortran API.
        linear_solver: InteriorPointLinearSolver::SpralSrc,
        spral_pivot_method: InteriorPointSpralPivotMethod::AggressiveAposteriori,
        spral_action_on_zero_pivot: true,
        spral_small_pivot_tolerance: 1.0e-20,
        spral_threshold_pivot_u: 1.0e-8,
    }
}

pub fn default_solver_profile() -> SolverProfile {
    SolverProfile::StrictOcp
}

fn strict_ocp_solver_profile_defaults() -> SolverProfileDefaults {
    SolverProfileDefaults {
        max_iters: 200,
        overall_tol: 1.0e-8,
        dual_tol: 1.0e-6,
        constraint_tol: 1.0e-8,
        complementarity_tol: 1.0e-6,
    }
}

fn ipopt_default_solver_profile_defaults() -> SolverProfileDefaults {
    SolverProfileDefaults {
        max_iters: 3000,
        overall_tol: 1.0e-8,
        dual_tol: 1.0,
        constraint_tol: 1.0e-4,
        complementarity_tol: 1.0e-4,
    }
}

fn solver_profile_defaults_for(
    strict_defaults: SolverConfig,
    profile: SolverProfile,
) -> SolverProfileDefaults {
    match profile {
        SolverProfile::StrictOcp => SolverProfileDefaults {
            max_iters: strict_defaults.max_iters,
            overall_tol: strict_defaults.overall_tol,
            dual_tol: strict_defaults.dual_tol,
            constraint_tol: strict_defaults.constraint_tol,
            complementarity_tol: strict_defaults.complementarity_tol,
        },
        SolverProfile::IpoptDefault => ipopt_default_solver_profile_defaults(),
    }
}

pub fn default_solver_config() -> SolverConfig {
    let defaults = strict_ocp_solver_profile_defaults();
    SolverConfig {
        profile: default_solver_profile(),
        max_iters: defaults.max_iters,
        overall_tol: defaults.overall_tol,
        dual_tol: defaults.dual_tol,
        constraint_tol: defaults.constraint_tol,
        complementarity_tol: defaults.complementarity_tol,
        sqp: default_sqp_method_config(),
        nlip: default_nlip_config(),
    }
}

pub fn default_solver_method() -> SolverMethod {
    SolverMethod::Nlip
}

fn sqp_globalization_mode(config: &SolverConfig) -> SqpGlobalizationMode {
    match config.sqp.globalization {
        SqpGlobalizationConfig::LineSearchFilter { .. } => SqpGlobalizationMode::LineSearchFilter,
        SqpGlobalizationConfig::LineSearchMerit { .. } => SqpGlobalizationMode::LineSearchMerit,
        SqpGlobalizationConfig::TrustRegionFilter { .. } => SqpGlobalizationMode::TrustRegionFilter,
        SqpGlobalizationConfig::TrustRegionMerit { .. } => SqpGlobalizationMode::TrustRegionMerit,
    }
}

fn sqp_globalization_mode_value(mode: SqpGlobalizationMode) -> f64 {
    match mode {
        SqpGlobalizationMode::LineSearchFilter => 0.0,
        SqpGlobalizationMode::LineSearchMerit => 1.0,
        SqpGlobalizationMode::TrustRegionFilter => 2.0,
        SqpGlobalizationMode::TrustRegionMerit => 3.0,
    }
}

fn config_line_search(config: &SolverConfig) -> SqpLineSearchConfig {
    match config.sqp.globalization {
        SqpGlobalizationConfig::LineSearchFilter { line_search, .. }
        | SqpGlobalizationConfig::LineSearchMerit { line_search, .. } => line_search,
        _ => default_sqp_line_search_config(),
    }
}

fn config_filter(config: &SolverConfig) -> SqpFilterConfig {
    match config.sqp.globalization {
        SqpGlobalizationConfig::LineSearchFilter { filter, .. }
        | SqpGlobalizationConfig::TrustRegionFilter { filter, .. } => filter,
        _ => default_sqp_filter_config(),
    }
}

fn config_trust_region(config: &SolverConfig) -> SqpTrustRegionConfig {
    match config.sqp.globalization {
        SqpGlobalizationConfig::TrustRegionFilter { trust_region, .. }
        | SqpGlobalizationConfig::TrustRegionMerit { trust_region, .. } => trust_region,
        _ => default_sqp_trust_region_config(),
    }
}

fn config_exact_merit_penalty(config: &SolverConfig) -> f64 {
    match config.sqp.globalization {
        SqpGlobalizationConfig::LineSearchFilter {
            exact_merit_penalty,
            ..
        }
        | SqpGlobalizationConfig::LineSearchMerit {
            exact_merit_penalty,
            ..
        }
        | SqpGlobalizationConfig::TrustRegionFilter {
            exact_merit_penalty,
            ..
        }
        | SqpGlobalizationConfig::TrustRegionMerit {
            exact_merit_penalty,
            ..
        } => exact_merit_penalty,
    }
}

fn config_penalty_increase_factor(config: &SolverConfig) -> f64 {
    match config.sqp.globalization {
        SqpGlobalizationConfig::LineSearchMerit {
            penalty_increase_factor,
            ..
        } => penalty_increase_factor,
        _ => LineSearchMeritOptions::default().penalty_increase_factor,
    }
}

fn config_max_penalty_updates(config: &SolverConfig) -> usize {
    match config.sqp.globalization {
        SqpGlobalizationConfig::LineSearchMerit {
            max_penalty_updates,
            ..
        } => max_penalty_updates,
        _ => LineSearchMeritOptions::default().max_penalty_updates,
    }
}

fn config_trust_region_fixed_penalty(config: &SolverConfig) -> bool {
    match config.sqp.globalization {
        SqpGlobalizationConfig::TrustRegionMerit { fixed_penalty, .. } => fixed_penalty,
        _ => TrustRegionMeritOptions::default().fixed_penalty,
    }
}

pub const fn interactive_multiple_shooting_opt_level() -> LlvmOptimizationLevel {
    // The interactive demos are dominated by cold-start latency here, and multiple-shooting
    // kernels expand the RK4 dynamics directly into the NLP. Favor faster JIT over peak kernel
    // throughput so the webapp actually becomes usable.
    LlvmOptimizationLevel::O0
}

pub const fn interactive_direct_collocation_opt_level() -> LlvmOptimizationLevel {
    LlvmOptimizationLevel::O3
}

pub fn solver_running_label(method: SolverMethod) -> &'static str {
    match method {
        SolverMethod::Sqp => "Running SQP...",
        SolverMethod::Nlip => "Running NLIP solver...",
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => "Running IPOPT...",
    }
}

pub(crate) struct SolveLifecycleReporter<Emit> {
    emit: Emit,
    solver_method: SolverMethod,
}

impl<Emit> SolveLifecycleReporter<Emit>
where
    Emit: FnMut(SolveStreamEvent),
{
    pub fn new(emit: Emit, solver_method: SolverMethod) -> Self {
        Self {
            emit,
            solver_method,
        }
    }

    fn compile_progress<Compiled, Compile>(
        &mut self,
        compile: Compile,
    ) -> Result<(Compiled, CompileProgressInfo)>
    where
        Compile: FnOnce(
            &mut dyn FnMut(CompileProgressUpdate),
        ) -> Result<(Compiled, CompileProgressInfo)>,
    {
        emit_symbolic_setup_status(&mut self.emit);
        let mut on_symbolic_ready = |update: CompileProgressUpdate| {
            emit_solve_status(
                &mut self.emit,
                SolveStage::JitCompilation,
                None,
                SolverReport::in_progress("Compiling JIT...")
                    .with_backend_timing(update.timing)
                    .with_compile_cached(update.compile_cached)
                    .with_phase_details(update.phase_details),
            );
        };
        compile(&mut on_symbolic_ready)
    }

    pub fn compile_with_progress<Compiled, Compile>(
        &mut self,
        compile: Compile,
    ) -> Result<(Compiled, SolverReport, Option<CompileReportSummary>)>
    where
        Compile: FnOnce(
            &mut dyn FnMut(CompileProgressUpdate),
        ) -> Result<(Compiled, CompileProgressInfo)>,
    {
        let (compiled, progress) = self.compile_progress(compile)?;
        let running_solver = SolverReport::in_progress(solver_running_label(self.solver_method))
            .with_backend_timing(progress.timing)
            .with_compile_cached(progress.compile_cached)
            .with_phase_details(progress.phase_details);
        Ok((compiled, running_solver, progress.compile_report))
    }

    pub fn prewarm_with_progress<Compiled, Compile>(&mut self, compile: Compile) -> Result<()>
    where
        Compile: FnOnce(
            &mut dyn FnMut(CompileProgressUpdate),
        ) -> Result<(Compiled, CompileProgressInfo)>,
    {
        let (_, progress) = self.compile_progress(compile)?;
        emit_solve_status(
            &mut self.emit,
            SolveStage::JitCompilation,
            None,
            SolverReport::in_progress("Compiling JIT...")
                .with_backend_timing(progress.timing)
                .with_compile_cached(progress.compile_cached)
                .with_phase_details(progress.phase_details),
        );
        Ok(())
    }

    pub fn into_emit(self) -> Emit {
        self.emit
    }
}

fn with_control_panel(mut control: ControlSpec, panel: ControlPanel) -> ControlSpec {
    control.panel = Some(panel);
    control
}

fn with_solver_profile_defaults(
    mut control: ControlSpec,
    strict_value: f64,
    ipopt_default_value: f64,
) -> ControlSpec {
    control.profile_defaults = vec![
        ControlProfileDefault {
            profile: solver_profile_value(SolverProfile::StrictOcp),
            value: strict_value,
        },
        ControlProfileDefault {
            profile: solver_profile_value(SolverProfile::IpoptDefault),
            value: ipopt_default_value,
        },
    ];
    control
}

fn call_policy_control_choices() -> [(f64, &'static str); 4] {
    [
        (0.0, "Inline At Call"),
        (1.0, "Inline At Lowering"),
        (2.0, "Inline In LLVM"),
        (3.0, "NoInline LLVM"),
    ]
}

fn override_behavior_control_choices() -> [(f64, &'static str); 2] {
    [(0.0, "Allow Overrides"), (1.0, "Strict Global")]
}

fn hessian_strategy_control_choices() -> [(f64, &'static str); 3] {
    [
        (0.0, "By Column"),
        (1.0, "Selected Outputs"),
        (2.0, "Colored"),
    ]
}

fn kernel_strategy_control_choices() -> [(f64, &'static str); 6] {
    [
        (0.0, "Inline"),
        (1.0, "Function / Global Policy"),
        (2.0, "Function / Inline At Call"),
        (3.0, "Function / Inline At Lowering"),
        (4.0, "Function / Inline In LLVM"),
        (5.0, "Function / NoInline LLVM"),
    ]
}

fn ocp_sx_function_controls(default: OcpSxFunctionConfig) -> Vec<ControlSpec> {
    let panel = ControlPanel::SxFunctions;
    vec![
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionGlobalCallPolicy.id(),
                "Global Call Policy",
                call_policy_choice_value(default.global_call_policy),
                "",
                "Default lowering/JIT policy for symbolic function calls when a kernel uses the global policy.",
                &call_policy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionOverrideBehavior.id(),
                "Override Behavior",
                override_behavior_choice_value(default.override_behavior),
                "",
                "Allow per-kernel call-policy overrides, or force every function to use the global policy.",
                &override_behavior_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionHessianStrategy.id(),
                "Hessian Strategy",
                hessian_strategy_choice_value(default.hessian_strategy),
                "",
                "Controls how the lower-triangle Lagrangian Hessian is generated before JIT compilation.",
                &hessian_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionOde.id(),
                "ODE Kernel",
                kernel_strategy_choice_value(default.ode),
                "",
                "Controls whether the dynamics RHS is emitted inline or through a reusable SXFunction.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionObjectiveLagrange.id(),
                "Lagrange Objective",
                kernel_strategy_choice_value(default.objective_lagrange),
                "",
                "Controls reuse/inlining for the repeated stage-cost kernel.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionObjectiveMayer.id(),
                "Mayer Objective",
                kernel_strategy_choice_value(default.objective_mayer),
                "",
                "Controls reuse/inlining for the terminal objective kernel.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionPathConstraints.id(),
                "Path Constraints",
                kernel_strategy_choice_value(default.path_constraints),
                "",
                "Controls reuse/inlining for repeated path-constraint evaluation.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionBoundaryEqualities.id(),
                "Boundary Equalities",
                kernel_strategy_choice_value(default.boundary_equalities),
                "",
                "Controls reuse/inlining for the boundary-equality kernel.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionBoundaryInequalities.id(),
                "Boundary Inequalities",
                kernel_strategy_choice_value(default.boundary_inequalities),
                "",
                "Controls reuse/inlining for the boundary-inequality kernel.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::Always,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
        with_control_panel(
            select_control(
                SharedControlId::SxFunctionMultipleShootingIntegrator.id(),
                "MS Integrator",
                kernel_strategy_choice_value(default.multiple_shooting_integrator),
                "",
                "Controls whether the RK4 multiple-shooting interval map is wrapped in a reusable SXFunction.",
                &kernel_strategy_control_choices(),
                ControlSection::Transcription,
                ControlVisibility::MultipleShootingOnly,
                ControlSemantic::SxFunctionOption,
            ),
            panel,
        ),
    ]
}

fn time_grid_choices() -> Vec<(f64, &'static str)> {
    vec![
        (0.0, "Uniform"),
        (1.0, "Cosine"),
        (2.0, "Tanh"),
        (3.0, "Geometric Start"),
        (4.0, "Geometric End"),
        (5.0, "Focus"),
        (6.0, "Piecewise"),
    ]
}

fn time_grid_choice_value(time_grid: TimeGrid) -> f64 {
    match time_grid {
        TimeGrid::Uniform => 0.0,
        TimeGrid::Cosine { .. } => 1.0,
        TimeGrid::Tanh { .. } => 2.0,
        TimeGrid::Geometric {
            bias: TimeGridBias::Start,
            ..
        } => 3.0,
        TimeGrid::Geometric {
            bias: TimeGridBias::End,
            ..
        } => 4.0,
        TimeGrid::Focus { .. } => 5.0,
        TimeGrid::Piecewise { .. } => 6.0,
    }
}

fn time_grid_strength_value(time_grid: TimeGrid) -> f64 {
    match time_grid {
        TimeGrid::Cosine { strength }
        | TimeGrid::Tanh { strength }
        | TimeGrid::Geometric { strength, .. }
        | TimeGrid::Focus { strength, .. } => strength,
        TimeGrid::Uniform | TimeGrid::Piecewise { .. } => 1.0,
    }
}

fn time_grid_focus_center_value(time_grid: TimeGrid) -> f64 {
    match time_grid {
        TimeGrid::Focus { center, .. } => center,
        _ => 0.5,
    }
}

fn time_grid_focus_width_value(time_grid: TimeGrid) -> f64 {
    match time_grid {
        TimeGrid::Focus { width, .. } => width,
        _ => 0.15,
    }
}

fn time_grid_breakpoint_value(time_grid: TimeGrid) -> f64 {
    match time_grid {
        TimeGrid::Piecewise { breakpoint, .. } => breakpoint,
        _ => 0.5,
    }
}

fn time_grid_first_interval_fraction_value(time_grid: TimeGrid) -> f64 {
    match time_grid {
        TimeGrid::Piecewise {
            first_interval_fraction,
            ..
        } => first_interval_fraction,
        _ => 0.5,
    }
}

fn time_grid_slider_control(
    id: SharedControlId,
    label: &str,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
    help: &str,
    semantic: ControlSemantic,
) -> ControlSpec {
    ControlSpec {
        id: id.id().to_string(),
        label: label.to_string(),
        min,
        max,
        step,
        default,
        unit: "".to_string(),
        help: help.to_string(),
        section: ControlSection::Transcription,
        panel: None,
        editor: ControlEditor::Slider,
        visibility: ControlVisibility::DirectCollocationOnly,
        semantic,
        value_display: ControlValueDisplay::Scalar,
        choices: Vec::new(),
        profile_defaults: Vec::new(),
    }
}

pub fn transcription_controls(
    default: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
) -> Vec<ControlSpec> {
    let mut controls = vec![
        transcription_interval_control(default.intervals, supported_intervals),
        select_control(
            SharedControlId::TranscriptionMethod.id(),
            "Transcription",
            match default.method {
                TranscriptionMethod::MultipleShooting => 0.0,
                TranscriptionMethod::DirectCollocation => 1.0,
            },
            "",
            "Switch between RK4 multiple shooting and direct collocation.",
            &[(0.0, "Multiple Shooting"), (1.0, "Direct Collocation")],
            ControlSection::Transcription,
            ControlVisibility::Always,
            ControlSemantic::TranscriptionMethod,
        ),
        select_control(
            SharedControlId::CollocationFamily.id(),
            "Collocation Family",
            match default.collocation_family {
                CollocationFamily::GaussLegendre => 0.0,
                CollocationFamily::RadauIIA => 1.0,
            },
            "",
            "Family used when direct collocation is selected.",
            &[(0.0, "Legendre"), (1.0, "Radau IIA")],
            ControlSection::Transcription,
            ControlVisibility::DirectCollocationOnly,
            ControlSemantic::CollocationFamily,
        ),
        select_control(
            SharedControlId::CollocationDegree.id(),
            "Collocation Nodes",
            default.collocation_degree as f64,
            "",
            "Number of collocation nodes per interval when direct collocation is selected.",
            &supported_degrees
                .iter()
                .map(|value| (*value as f64, value.to_string()))
                .collect::<Vec<_>>(),
            ControlSection::Transcription,
            ControlVisibility::DirectCollocationOnly,
            ControlSemantic::CollocationDegree,
        ),
        select_control(
            SharedControlId::TimeGrid.id(),
            "Time Grid",
            time_grid_choice_value(default.time_grid),
            "",
            "Distribution of mesh interval lengths for direct collocation.",
            &time_grid_choices(),
            ControlSection::Transcription,
            ControlVisibility::DirectCollocationOnly,
            ControlSemantic::TimeGrid,
        ),
        time_grid_slider_control(
            SharedControlId::TimeGridStrength,
            "Grid Strength",
            0.0,
            1.0,
            0.05,
            time_grid_strength_value(default.time_grid),
            "Blend from uniform spacing at 0 to the selected warped spacing at 1.",
            ControlSemantic::TimeGridStrength,
        ),
        time_grid_slider_control(
            SharedControlId::TimeGridFocusCenter,
            "Focus Center",
            0.0,
            1.0,
            0.05,
            time_grid_focus_center_value(default.time_grid),
            "Normalized location where focus gridding uses the shortest intervals.",
            ControlSemantic::TimeGridFocusCenter,
        ),
        time_grid_slider_control(
            SharedControlId::TimeGridFocusWidth,
            "Focus Width",
            0.01,
            1.0,
            0.01,
            time_grid_focus_width_value(default.time_grid),
            "Normalized spread of the focus gridding region.",
            ControlSemantic::TimeGridFocusWidth,
        ),
        time_grid_slider_control(
            SharedControlId::TimeGridBreakpoint,
            "Phase Break",
            0.05,
            0.95,
            0.05,
            time_grid_breakpoint_value(default.time_grid),
            "Normalized time reached at the piecewise grid transition.",
            ControlSemantic::TimeGridBreakpoint,
        ),
        time_grid_slider_control(
            SharedControlId::TimeGridFirstIntervalFraction,
            "Phase Interval Share",
            0.05,
            0.95,
            0.05,
            time_grid_first_interval_fraction_value(default.time_grid),
            "Fraction of mesh intervals assigned before the piecewise grid transition.",
            ControlSemantic::TimeGridFirstIntervalFraction,
        ),
    ];
    controls.extend(ocp_sx_function_controls(OcpSxFunctionConfig::default()));
    controls
}

fn transcription_interval_control(default: usize, supported_intervals: &[usize]) -> ControlSpec {
    let (min, max) = interval_bounds(supported_intervals, default);
    ControlSpec {
        id: SharedControlId::TranscriptionIntervals.id().to_string(),
        label: "Intervals".to_string(),
        min: min as f64,
        max: max as f64,
        step: 1.0,
        default: default as f64,
        unit: String::new(),
        help: format!(
            "Number of mesh intervals. Enter an integer from {min} to {max}; changing this compiles a matching transcription variant on demand."
        ),
        section: ControlSection::Transcription,
        panel: None,
        editor: ControlEditor::Text,
        visibility: ControlVisibility::Always,
        semantic: ControlSemantic::TranscriptionIntervals,
        value_display: ControlValueDisplay::Integer,
        choices: Vec::new(),
        profile_defaults: Vec::new(),
    }
}

fn interval_bounds(supported_intervals: &[usize], default: usize) -> (usize, usize) {
    let min = supported_intervals
        .iter()
        .copied()
        .min()
        .unwrap_or(default)
        .max(1);
    let max = supported_intervals
        .iter()
        .copied()
        .max()
        .unwrap_or(default)
        .max(min);
    (min, max)
}

pub fn solver_controls(default_method: SolverMethod, default: SolverConfig) -> Vec<ControlSpec> {
    let ipopt_profile_defaults = ipopt_default_solver_profile_defaults();
    let default_line_search = config_line_search(&default);
    let default_filter = config_filter(&default);
    let default_trust_region = config_trust_region(&default);
    let default_globalization = sqp_globalization_mode(&default);
    let default_exact_merit_penalty = config_exact_merit_penalty(&default);
    let default_penalty_increase_factor = config_penalty_increase_factor(&default);
    let default_max_penalty_updates = config_max_penalty_updates(&default);
    let default_fixed_penalty = config_trust_region_fixed_penalty(&default);
    vec![
        select_control(
            SharedControlId::SolverMethod.id(),
            "Solver",
            match default_method {
                SolverMethod::Sqp => 0.0,
                SolverMethod::Nlip => 1.0,
                #[cfg(feature = "ipopt")]
                SolverMethod::Ipopt => 2.0,
            },
            "",
            "Select the runtime NLP solver for the compiled OCP transcription.",
            &solver_method_choices(),
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverMethod,
        ),
        select_control(
            SharedControlId::SolverProfile.id(),
            "Profile",
            solver_profile_value(default.profile),
            "",
            "Choose the default solver option profile. Text fields left blank inherit the selected profile; entered values are explicit overrides.",
            &solver_profile_choices(),
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverProfile,
        ),
        select_control(
            SharedControlId::SolverGlobalization.id(),
            "SQP Globalization",
            sqp_globalization_mode_value(default_globalization),
            "",
            "Choose the SQP globalization strategy and acceptance model.",
            &solver_globalization_choices(),
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverGlobalization,
        ),
        with_solver_profile_defaults(
            text_control(
                SharedControlId::SolverMaxIterations.id(),
                "Max Iterations",
                default.max_iters as f64,
                "",
                "Maximum nonlinear iterations before the selected solver terminates.",
                ControlSemantic::SolverMaxIterations,
                ControlValueDisplay::Integer,
            ),
            default.max_iters as f64,
            ipopt_profile_defaults.max_iters as f64,
        ),
        with_solver_profile_defaults(
            text_control(
                SharedControlId::SolverOverallTolerance.id(),
                "Overall Tolerance",
                default.overall_tol,
                "",
                "Termination threshold on the scaled overall NLP error.",
                ControlSemantic::SolverOverallTolerance,
                ControlValueDisplay::Scientific,
            ),
            default.overall_tol,
            ipopt_profile_defaults.overall_tol,
        ),
        select_control(
            SharedControlId::SolverHessianRegularization.id(),
            "Hessian Regularization",
            if default.sqp.hessian_regularization_enabled {
                1.0
            } else {
                0.0
            },
            "",
            "Whether SQP should diagonally regularize the Hessian before solving the QP subproblem.",
            &[(0.0, "Off"), (1.0, "On")],
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverHessianRegularization,
        ),
        select_control(
            SharedControlId::SolverNlipLinearSolver.id(),
            "NLIP Linear Solver",
            nlip_linear_solver_value(default.nlip.linear_solver),
            "",
            "Choose the sparse linear solver backend for NLIP KKT systems.",
            &nlip_linear_solver_choices(),
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverNlipLinearSolver,
        ),
        select_control(
            SharedControlId::SolverNlipSpralPivotMethod.id(),
            "NLIP SPRAL Pivoting",
            nlip_spral_pivot_method_value(default.nlip.spral_pivot_method),
            "",
            "Pivoting strategy for SPRAL-based NLIP KKT solves. IPOPT's spral_pivot_method=block default maps to native SPRAL raw control value 1.",
            &nlip_spral_pivot_method_choices(),
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverNlipSpralPivotMethod,
        ),
        select_control(
            SharedControlId::SolverNlipSpralZeroPivotAction.id(),
            "NLIP SPRAL Zero Pivot",
            if default.nlip.spral_action_on_zero_pivot {
                1.0
            } else {
                0.0
            },
            "",
            "Whether SPRAL continues factorization after encountering a zero pivot.",
            &[(1.0, "Continue"), (0.0, "Abort")],
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverNlipSpralZeroPivotAction,
        ),
        text_control(
            SharedControlId::SolverNlipSpralSmallPivot.id(),
            "NLIP SPRAL Small Pivot",
            default.nlip.spral_small_pivot_tolerance,
            "",
            "Absolute minimum pivot size accepted by SPRAL.",
            ControlSemantic::SolverNlipSpralSmallPivot,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverNlipSpralPivotU.id(),
            "NLIP SPRAL Pivot u",
            default.nlip.spral_threshold_pivot_u,
            "",
            "SPRAL threshold-pivoting parameter u.",
            ControlSemantic::SolverNlipSpralPivotU,
            ControlValueDisplay::Scientific,
        ),
        with_solver_profile_defaults(
            text_control(
                SharedControlId::SolverDualTolerance.id(),
                "Dual Tolerance",
                default.dual_tol,
                "",
                "Termination threshold on the dual infeasibility norm.",
                ControlSemantic::SolverDualTolerance,
                ControlValueDisplay::Scientific,
            ),
            default.dual_tol,
            ipopt_profile_defaults.dual_tol,
        ),
        with_solver_profile_defaults(
            text_control(
                SharedControlId::SolverConstraintTolerance.id(),
                "Constraint Tolerance",
                default.constraint_tol,
                "",
                "Termination threshold on equality and inequality infeasibility.",
                ControlSemantic::SolverConstraintTolerance,
                ControlValueDisplay::Scientific,
            ),
            default.constraint_tol,
            ipopt_profile_defaults.constraint_tol,
        ),
        with_solver_profile_defaults(
            text_control(
                SharedControlId::SolverComplementarityTolerance.id(),
                "Complementarity Tolerance",
                default.complementarity_tol,
                "",
                "Termination threshold on complementarity residuals.",
                ControlSemantic::SolverComplementarityTolerance,
                ControlValueDisplay::Scientific,
            ),
            default.complementarity_tol,
            ipopt_profile_defaults.complementarity_tol,
        ),
        text_control(
            SharedControlId::SolverExactMeritPenalty.id(),
            "Exact Merit Penalty",
            default_exact_merit_penalty,
            "",
            "Exact-penalty weight used in merit globalization and trust-region model quality.",
            ControlSemantic::SolverExactMeritPenalty,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverArmijoC1.id(),
            "Armijo c1",
            default_line_search.armijo_c1,
            "",
            "Armijo sufficient-decrease coefficient for line-search globalization.",
            ControlSemantic::SolverArmijoC1,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverWolfeC2.id(),
            "Wolfe c2 (Adv)",
            default_line_search.wolfe_c2.unwrap_or(0.9),
            "",
            "Wolfe curvature coefficient for line-search globalization. Positive values enable Wolfe; non-positive disables it.",
            ControlSemantic::SolverWolfeC2,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverLineSearchBeta.id(),
            "LS Backtrack Beta",
            default_line_search.beta,
            "",
            "Backtracking contraction factor for line search.",
            ControlSemantic::SolverLineSearchBeta,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverLineSearchMaxSteps.id(),
            "LS Max Steps",
            default_line_search.max_steps as f64,
            "",
            "Maximum number of backtracking reductions per line-search attempt.",
            ControlSemantic::SolverLineSearchMaxSteps,
            ControlValueDisplay::Integer,
        ),
        text_control(
            SharedControlId::SolverMinStep.id(),
            "LS Min Step",
            default_line_search.min_step,
            "",
            "Minimum accepted step scale in line-search globalization.",
            ControlSemantic::SolverMinStep,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverPenaltyIncreaseFactor.id(),
            "Penalty Increase Factor (Adv)",
            default_penalty_increase_factor,
            "",
            "Multiplier applied when merit line search needs a larger exact-penalty weight.",
            ControlSemantic::SolverPenaltyIncreaseFactor,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverMaxPenaltyUpdates.id(),
            "Max Penalty Updates (Adv)",
            default_max_penalty_updates as f64,
            "",
            "Maximum penalty updates before a merit line-search step is rejected.",
            ControlSemantic::SolverMaxPenaltyUpdates,
            ControlValueDisplay::Integer,
        ),
        text_control(
            SharedControlId::SolverFilterGammaObjective.id(),
            "Filter Gamma Objective",
            default_filter.gamma_objective,
            "",
            "Sufficient objective reduction threshold for filter globalization.",
            ControlSemantic::SolverFilterGammaObjective,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverFilterGammaViolation.id(),
            "Filter Gamma Violation",
            default_filter.gamma_violation,
            "",
            "Sufficient feasibility reduction threshold for filter globalization.",
            ControlSemantic::SolverFilterGammaViolation,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverFilterThetaMaxFactor.id(),
            "Filter Theta Max Factor (Adv)",
            default_filter.theta_max_factor,
            "",
            "Maximum filter violation envelope as a multiple of the switching reference.",
            ControlSemantic::SolverFilterThetaMaxFactor,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverFilterSwitchingReferenceMin.id(),
            "Filter Ref Min (Adv)",
            default_filter.switching_reference_min,
            "",
            "Minimum switching-condition reference violation.",
            ControlSemantic::SolverFilterSwitchingReferenceMin,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverFilterSwitchingViolationFactor.id(),
            "Filter Switching Violation (Adv)",
            default_filter.switching_violation_factor,
            "",
            "Switching-condition multiplier on current violation.",
            ControlSemantic::SolverFilterSwitchingViolationFactor,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverFilterSwitchingLinearizedReductionFactor.id(),
            "Filter Switching Linearized (Adv)",
            default_filter.switching_linearized_reduction_factor,
            "",
            "Switching-condition factor on predicted linearized feasibility reduction.",
            ControlSemantic::SolverFilterSwitchingLinearizedReductionFactor,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionInitialRadius.id(),
            "TR Initial Radius",
            default_trust_region.initial_radius,
            "",
            "Initial trust-region radius.",
            ControlSemantic::SolverTrustRegionInitialRadius,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionMaxRadius.id(),
            "TR Max Radius",
            default_trust_region.max_radius,
            "",
            "Maximum trust-region radius.",
            ControlSemantic::SolverTrustRegionMaxRadius,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionMinRadius.id(),
            "TR Min Radius",
            default_trust_region.min_radius,
            "",
            "Minimum trust-region radius before restoration/failure.",
            ControlSemantic::SolverTrustRegionMinRadius,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionShrinkFactor.id(),
            "TR Shrink Factor",
            default_trust_region.shrink_factor,
            "",
            "Radius contraction factor after a rejected trust-region trial.",
            ControlSemantic::SolverTrustRegionShrinkFactor,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionGrowFactor.id(),
            "TR Grow Factor",
            default_trust_region.grow_factor,
            "",
            "Radius expansion factor after a high-quality boundary step.",
            ControlSemantic::SolverTrustRegionGrowFactor,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionAcceptRatio.id(),
            "TR Accept Ratio",
            default_trust_region.accept_ratio,
            "",
            "Minimum actual/predicted reduction ratio for a trust-region step to be accepted.",
            ControlSemantic::SolverTrustRegionAcceptRatio,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionExpandRatio.id(),
            "TR Expand Ratio",
            default_trust_region.expand_ratio,
            "",
            "Ratio threshold for expanding the trust-region radius.",
            ControlSemantic::SolverTrustRegionExpandRatio,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionBoundaryFraction.id(),
            "TR Boundary Fraction (Adv)",
            default_trust_region.boundary_fraction,
            "",
            "Fraction of the radius used to classify a step as boundary-active.",
            ControlSemantic::SolverTrustRegionBoundaryFraction,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverTrustRegionMaxContractions.id(),
            "TR Max Contractions",
            default_trust_region.max_radius_contractions as f64,
            "",
            "Maximum trust-region radius contractions within one SQP iteration.",
            ControlSemantic::SolverTrustRegionMaxContractions,
            ControlValueDisplay::Integer,
        ),
        select_control(
            SharedControlId::SolverTrustRegionFixedPenalty.id(),
            "TR Fixed Penalty",
            if default_fixed_penalty { 1.0 } else { 0.0 },
            "",
            "Whether trust-region merit mode keeps the exact-penalty weight fixed instead of updating it from multipliers.",
            &[(1.0, "On"), (0.0, "Off")],
            ControlSection::Solver,
            ControlVisibility::Always,
            ControlSemantic::SolverTrustRegionFixedPenalty,
        ),
    ]
}

fn solver_method_choices() -> Vec<(f64, &'static str)> {
    #[cfg(feature = "ipopt")]
    let mut out = vec![(0.0, "SQP"), (1.0, "NLIP")];
    #[cfg(not(feature = "ipopt"))]
    let out = vec![(0.0, "SQP"), (1.0, "NLIP")];
    #[cfg(feature = "ipopt")]
    out.push((2.0, "IPOPT"));
    out
}

fn solver_profile_value(profile: SolverProfile) -> f64 {
    match profile {
        SolverProfile::StrictOcp => 0.0,
        SolverProfile::IpoptDefault => 1.0,
    }
}

fn solver_profile_choices() -> Vec<(f64, &'static str)> {
    vec![(0.0, "Strict OCP"), (1.0, "IPOPT Default")]
}

fn solver_globalization_choices() -> Vec<(f64, &'static str)> {
    vec![
        (0.0, "LS Filter"),
        (1.0, "LS Merit"),
        (2.0, "TR Filter"),
        (3.0, "TR Merit"),
    ]
}

fn nlip_linear_solver_value(solver: InteriorPointLinearSolver) -> f64 {
    match solver {
        InteriorPointLinearSolver::SsidsRs => 0.0,
        InteriorPointLinearSolver::SpralSrc => 1.0,
        InteriorPointLinearSolver::SparseQdldl => 2.0,
        InteriorPointLinearSolver::Auto => 3.0,
    }
}

fn nlip_linear_solver_choices() -> Vec<(f64, &'static str)> {
    vec![
        (0.0, "SSIDS-RS"),
        (1.0, "SPRAL-SRC"),
        (2.0, "QDLDL"),
        (3.0, "Auto"),
    ]
}

fn nlip_spral_pivot_method_value(method: InteriorPointSpralPivotMethod) -> f64 {
    match method {
        InteriorPointSpralPivotMethod::AggressiveAposteriori => 0.0,
        InteriorPointSpralPivotMethod::BlockAposteriori => 1.0,
        InteriorPointSpralPivotMethod::ThresholdPartial => 2.0,
    }
}

fn nlip_spral_pivot_method_choices() -> Vec<(f64, &'static str)> {
    vec![
        (0.0, "Aggressive APP / IPOPT block"),
        (1.0, "Block APP / SPRAL raw 2"),
        (2.0, "Threshold Partial"),
    ]
}

fn sample_shared_or_default(
    values: &BTreeMap<String, f64>,
    key: SharedControlId,
    default: f64,
) -> f64 {
    sample_or_default(values, key.id(), default)
}

fn parse_enum_choice(value: f64, key: SharedControlId, variants: &[f64]) -> Result<usize> {
    if !value.is_finite() {
        return Err(anyhow!("{} must be finite", key.id()));
    }
    variants
        .iter()
        .position(|candidate| (value - *candidate).abs() <= 1.0e-9)
        .ok_or_else(|| anyhow!("invalid {} value {value}", key.id()))
}

pub fn solver_config_from_map(
    values: &BTreeMap<String, f64>,
    default: SolverConfig,
) -> Result<SolverConfig> {
    let default_line_search = config_line_search(&default);
    let default_filter = config_filter(&default);
    let default_trust_region = config_trust_region(&default);
    let profile = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SolverProfile,
            solver_profile_value(default.profile),
        ),
        SharedControlId::SolverProfile,
        &[0.0, 1.0],
    )? {
        0 => SolverProfile::StrictOcp,
        1 => SolverProfile::IpoptDefault,
        _ => unreachable!("validated solver profile choice"),
    };
    let profile_defaults = solver_profile_defaults_for(default, profile);
    let max_iters = expect_nonnegative_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverMaxIterations,
            profile_defaults.max_iters as f64,
        ),
        SharedControlId::SolverMaxIterations.id(),
    )?
    .round() as usize;
    let overall_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverOverallTolerance,
            profile_defaults.overall_tol,
        ),
        SharedControlId::SolverOverallTolerance.id(),
    )?;
    let hessian_regularization_enabled = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SolverHessianRegularization,
            if default.sqp.hessian_regularization_enabled {
                1.0
            } else {
                0.0
            },
        ),
        SharedControlId::SolverHessianRegularization,
        &[0.0, 1.0],
    )? {
        0 => false,
        1 => true,
        _ => unreachable!("validated Hessian regularization toggle"),
    };
    let dual_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverDualTolerance,
            profile_defaults.dual_tol,
        ),
        SharedControlId::SolverDualTolerance.id(),
    )?;
    let constraint_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverConstraintTolerance,
            profile_defaults.constraint_tol,
        ),
        SharedControlId::SolverConstraintTolerance.id(),
    )?;
    let complementarity_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverComplementarityTolerance,
            profile_defaults.complementarity_tol,
        ),
        SharedControlId::SolverComplementarityTolerance.id(),
    )?;
    let nlip_linear_solver = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SolverNlipLinearSolver,
            nlip_linear_solver_value(default.nlip.linear_solver),
        ),
        SharedControlId::SolverNlipLinearSolver,
        &[0.0, 1.0, 2.0, 3.0],
    )? {
        0 => InteriorPointLinearSolver::SsidsRs,
        1 => InteriorPointLinearSolver::SpralSrc,
        2 => InteriorPointLinearSolver::SparseQdldl,
        3 => InteriorPointLinearSolver::Auto,
        _ => unreachable!("validated NLIP linear solver choice"),
    };
    let nlip_spral_pivot_method = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SolverNlipSpralPivotMethod,
            nlip_spral_pivot_method_value(default.nlip.spral_pivot_method),
        ),
        SharedControlId::SolverNlipSpralPivotMethod,
        &[0.0, 1.0, 2.0],
    )? {
        0 => InteriorPointSpralPivotMethod::AggressiveAposteriori,
        1 => InteriorPointSpralPivotMethod::BlockAposteriori,
        2 => InteriorPointSpralPivotMethod::ThresholdPartial,
        _ => unreachable!("validated NLIP SPRAL pivot method"),
    };
    let nlip_spral_action_on_zero_pivot = parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SolverNlipSpralZeroPivotAction,
            if default.nlip.spral_action_on_zero_pivot {
                1.0
            } else {
                0.0
            },
        ),
        SharedControlId::SolverNlipSpralZeroPivotAction,
        &[0.0, 1.0],
    )? == 1;
    let nlip_spral_small_pivot_tolerance = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverNlipSpralSmallPivot,
            default.nlip.spral_small_pivot_tolerance,
        ),
        SharedControlId::SolverNlipSpralSmallPivot.id(),
    )?;
    let nlip_spral_threshold_pivot_u = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverNlipSpralPivotU,
            default.nlip.spral_threshold_pivot_u,
        ),
        SharedControlId::SolverNlipSpralPivotU.id(),
    )?;
    let globalization = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SolverGlobalization,
            sqp_globalization_mode_value(sqp_globalization_mode(&default)),
        ),
        SharedControlId::SolverGlobalization,
        &[0.0, 1.0, 2.0, 3.0],
    )? {
        0 => SqpGlobalizationConfig::LineSearchFilter {
            line_search: SqpLineSearchConfig {
                armijo_c1: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverArmijoC1,
                        default_line_search.armijo_c1,
                    ),
                    SharedControlId::SolverArmijoC1.id(),
                )?,
                wolfe_c2: {
                    let value = sample_shared_or_default(
                        values,
                        SharedControlId::SolverWolfeC2,
                        default_line_search.wolfe_c2.unwrap_or(0.0),
                    );
                    (value > 0.0).then_some(value)
                },
                beta: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverLineSearchBeta,
                        default_line_search.beta,
                    ),
                    SharedControlId::SolverLineSearchBeta.id(),
                )?,
                max_steps: expect_nonnegative_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverLineSearchMaxSteps,
                        default_line_search.max_steps as f64,
                    ),
                    SharedControlId::SolverLineSearchMaxSteps.id(),
                )?
                .round() as usize,
                min_step: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverMinStep,
                        default_line_search.min_step,
                    ),
                    SharedControlId::SolverMinStep.id(),
                )?,
            },
            filter: SqpFilterConfig {
                gamma_objective: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterGammaObjective,
                        default_filter.gamma_objective,
                    ),
                    SharedControlId::SolverFilterGammaObjective.id(),
                )?,
                gamma_violation: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterGammaViolation,
                        default_filter.gamma_violation,
                    ),
                    SharedControlId::SolverFilterGammaViolation.id(),
                )?,
                theta_max_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterThetaMaxFactor,
                        default_filter.theta_max_factor,
                    ),
                    SharedControlId::SolverFilterThetaMaxFactor.id(),
                )?,
                switching_reference_min: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterSwitchingReferenceMin,
                        default_filter.switching_reference_min,
                    ),
                    SharedControlId::SolverFilterSwitchingReferenceMin.id(),
                )?,
                switching_violation_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterSwitchingViolationFactor,
                        default_filter.switching_violation_factor,
                    ),
                    SharedControlId::SolverFilterSwitchingViolationFactor.id(),
                )?,
                switching_linearized_reduction_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterSwitchingLinearizedReductionFactor,
                        default_filter.switching_linearized_reduction_factor,
                    ),
                    SharedControlId::SolverFilterSwitchingLinearizedReductionFactor.id(),
                )?,
            },
            exact_merit_penalty: expect_positive_finite(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverExactMeritPenalty,
                    config_exact_merit_penalty(&default),
                ),
                SharedControlId::SolverExactMeritPenalty.id(),
            )?,
        },
        1 => SqpGlobalizationConfig::LineSearchMerit {
            line_search: SqpLineSearchConfig {
                armijo_c1: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverArmijoC1,
                        default_line_search.armijo_c1,
                    ),
                    SharedControlId::SolverArmijoC1.id(),
                )?,
                wolfe_c2: {
                    let value = sample_shared_or_default(
                        values,
                        SharedControlId::SolverWolfeC2,
                        default_line_search.wolfe_c2.unwrap_or(0.0),
                    );
                    (value > 0.0).then_some(value)
                },
                beta: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverLineSearchBeta,
                        default_line_search.beta,
                    ),
                    SharedControlId::SolverLineSearchBeta.id(),
                )?,
                max_steps: expect_nonnegative_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverLineSearchMaxSteps,
                        default_line_search.max_steps as f64,
                    ),
                    SharedControlId::SolverLineSearchMaxSteps.id(),
                )?
                .round() as usize,
                min_step: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverMinStep,
                        default_line_search.min_step,
                    ),
                    SharedControlId::SolverMinStep.id(),
                )?,
            },
            exact_merit_penalty: expect_positive_finite(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverExactMeritPenalty,
                    config_exact_merit_penalty(&default),
                ),
                SharedControlId::SolverExactMeritPenalty.id(),
            )?,
            penalty_increase_factor: expect_positive_finite(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverPenaltyIncreaseFactor,
                    config_penalty_increase_factor(&default),
                ),
                SharedControlId::SolverPenaltyIncreaseFactor.id(),
            )?,
            max_penalty_updates: expect_nonnegative_finite(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverMaxPenaltyUpdates,
                    config_max_penalty_updates(&default) as f64,
                ),
                SharedControlId::SolverMaxPenaltyUpdates.id(),
            )?
            .round() as usize,
        },
        2 => SqpGlobalizationConfig::TrustRegionFilter {
            trust_region: SqpTrustRegionConfig {
                initial_radius: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionInitialRadius,
                        default_trust_region.initial_radius,
                    ),
                    SharedControlId::SolverTrustRegionInitialRadius.id(),
                )?,
                max_radius: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionMaxRadius,
                        default_trust_region.max_radius,
                    ),
                    SharedControlId::SolverTrustRegionMaxRadius.id(),
                )?,
                min_radius: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionMinRadius,
                        default_trust_region.min_radius,
                    ),
                    SharedControlId::SolverTrustRegionMinRadius.id(),
                )?,
                shrink_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionShrinkFactor,
                        default_trust_region.shrink_factor,
                    ),
                    SharedControlId::SolverTrustRegionShrinkFactor.id(),
                )?,
                grow_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionGrowFactor,
                        default_trust_region.grow_factor,
                    ),
                    SharedControlId::SolverTrustRegionGrowFactor.id(),
                )?,
                accept_ratio: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionAcceptRatio,
                        default_trust_region.accept_ratio,
                    ),
                    SharedControlId::SolverTrustRegionAcceptRatio.id(),
                )?,
                expand_ratio: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionExpandRatio,
                        default_trust_region.expand_ratio,
                    ),
                    SharedControlId::SolverTrustRegionExpandRatio.id(),
                )?,
                boundary_fraction: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionBoundaryFraction,
                        default_trust_region.boundary_fraction,
                    ),
                    SharedControlId::SolverTrustRegionBoundaryFraction.id(),
                )?,
                max_radius_contractions: expect_nonnegative_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionMaxContractions,
                        default_trust_region.max_radius_contractions as f64,
                    ),
                    SharedControlId::SolverTrustRegionMaxContractions.id(),
                )?
                .round() as usize,
            },
            filter: SqpFilterConfig {
                gamma_objective: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterGammaObjective,
                        default_filter.gamma_objective,
                    ),
                    SharedControlId::SolverFilterGammaObjective.id(),
                )?,
                gamma_violation: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterGammaViolation,
                        default_filter.gamma_violation,
                    ),
                    SharedControlId::SolverFilterGammaViolation.id(),
                )?,
                theta_max_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterThetaMaxFactor,
                        default_filter.theta_max_factor,
                    ),
                    SharedControlId::SolverFilterThetaMaxFactor.id(),
                )?,
                switching_reference_min: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterSwitchingReferenceMin,
                        default_filter.switching_reference_min,
                    ),
                    SharedControlId::SolverFilterSwitchingReferenceMin.id(),
                )?,
                switching_violation_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterSwitchingViolationFactor,
                        default_filter.switching_violation_factor,
                    ),
                    SharedControlId::SolverFilterSwitchingViolationFactor.id(),
                )?,
                switching_linearized_reduction_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverFilterSwitchingLinearizedReductionFactor,
                        default_filter.switching_linearized_reduction_factor,
                    ),
                    SharedControlId::SolverFilterSwitchingLinearizedReductionFactor.id(),
                )?,
            },
            exact_merit_penalty: expect_positive_finite(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverExactMeritPenalty,
                    config_exact_merit_penalty(&default),
                ),
                SharedControlId::SolverExactMeritPenalty.id(),
            )?,
        },
        3 => SqpGlobalizationConfig::TrustRegionMerit {
            trust_region: SqpTrustRegionConfig {
                initial_radius: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionInitialRadius,
                        default_trust_region.initial_radius,
                    ),
                    SharedControlId::SolverTrustRegionInitialRadius.id(),
                )?,
                max_radius: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionMaxRadius,
                        default_trust_region.max_radius,
                    ),
                    SharedControlId::SolverTrustRegionMaxRadius.id(),
                )?,
                min_radius: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionMinRadius,
                        default_trust_region.min_radius,
                    ),
                    SharedControlId::SolverTrustRegionMinRadius.id(),
                )?,
                shrink_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionShrinkFactor,
                        default_trust_region.shrink_factor,
                    ),
                    SharedControlId::SolverTrustRegionShrinkFactor.id(),
                )?,
                grow_factor: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionGrowFactor,
                        default_trust_region.grow_factor,
                    ),
                    SharedControlId::SolverTrustRegionGrowFactor.id(),
                )?,
                accept_ratio: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionAcceptRatio,
                        default_trust_region.accept_ratio,
                    ),
                    SharedControlId::SolverTrustRegionAcceptRatio.id(),
                )?,
                expand_ratio: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionExpandRatio,
                        default_trust_region.expand_ratio,
                    ),
                    SharedControlId::SolverTrustRegionExpandRatio.id(),
                )?,
                boundary_fraction: expect_positive_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionBoundaryFraction,
                        default_trust_region.boundary_fraction,
                    ),
                    SharedControlId::SolverTrustRegionBoundaryFraction.id(),
                )?,
                max_radius_contractions: expect_nonnegative_finite(
                    sample_shared_or_default(
                        values,
                        SharedControlId::SolverTrustRegionMaxContractions,
                        default_trust_region.max_radius_contractions as f64,
                    ),
                    SharedControlId::SolverTrustRegionMaxContractions.id(),
                )?
                .round() as usize,
            },
            exact_merit_penalty: expect_positive_finite(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverExactMeritPenalty,
                    config_exact_merit_penalty(&default),
                ),
                SharedControlId::SolverExactMeritPenalty.id(),
            )?,
            fixed_penalty: parse_enum_choice(
                sample_shared_or_default(
                    values,
                    SharedControlId::SolverTrustRegionFixedPenalty,
                    if config_trust_region_fixed_penalty(&default) {
                        1.0
                    } else {
                        0.0
                    },
                ),
                SharedControlId::SolverTrustRegionFixedPenalty,
                &[0.0, 1.0],
            )? == 1,
        },
        _ => unreachable!("validated SQP globalization choice"),
    };
    Ok(SolverConfig {
        profile,
        max_iters,
        overall_tol,
        dual_tol,
        constraint_tol,
        complementarity_tol,
        sqp: SqpMethodConfig {
            hessian_regularization_enabled,
            globalization,
        },
        nlip: NlipConfig {
            linear_solver: nlip_linear_solver,
            spral_pivot_method: nlip_spral_pivot_method,
            spral_action_on_zero_pivot: nlip_spral_action_on_zero_pivot,
            spral_small_pivot_tolerance: nlip_spral_small_pivot_tolerance,
            spral_threshold_pivot_u: nlip_spral_threshold_pivot_u,
        },
    })
}

pub fn solver_method_from_map(
    values: &BTreeMap<String, f64>,
    default: SolverMethod,
) -> Result<SolverMethod> {
    let default_value = match default {
        SolverMethod::Sqp => 0.0,
        SolverMethod::Nlip => 1.0,
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => 2.0,
    };
    let choice = parse_enum_choice(
        sample_shared_or_default(values, SharedControlId::SolverMethod, default_value),
        SharedControlId::SolverMethod,
        &[
            0.0,
            1.0,
            #[cfg(feature = "ipopt")]
            2.0,
        ],
    )?;
    Ok(match choice {
        0 => SolverMethod::Sqp,
        1 => SolverMethod::Nlip,
        #[cfg(feature = "ipopt")]
        2 => SolverMethod::Ipopt,
        _ => unreachable!("validated solver choice index"),
    })
}

pub fn transcription_from_map(
    values: &BTreeMap<String, f64>,
    default: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
) -> Result<TranscriptionConfig> {
    let method = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::TranscriptionMethod,
            match default.method {
                TranscriptionMethod::MultipleShooting => 0.0,
                TranscriptionMethod::DirectCollocation => 1.0,
            },
        ),
        SharedControlId::TranscriptionMethod,
        &[0.0, 1.0],
    )? {
        0 => TranscriptionMethod::MultipleShooting,
        1 => TranscriptionMethod::DirectCollocation,
        _ => unreachable!("validated transcription choice index"),
    };

    let intervals = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::TranscriptionIntervals,
            default.intervals as f64,
        ),
        SharedControlId::TranscriptionIntervals.id(),
    )?
    .round() as usize;
    let (min_intervals, max_intervals) = interval_bounds(supported_intervals, default.intervals);
    if intervals < min_intervals || intervals > max_intervals {
        return Err(anyhow!(
            "unsupported {} {}; expected {min_intervals}..={max_intervals}",
            SharedControlId::TranscriptionIntervals.id(),
            intervals
        ));
    }

    let collocation_family = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::CollocationFamily,
            match default.collocation_family {
                CollocationFamily::GaussLegendre => 0.0,
                CollocationFamily::RadauIIA => 1.0,
            },
        ),
        SharedControlId::CollocationFamily,
        &[0.0, 1.0],
    )? {
        0 => CollocationFamily::GaussLegendre,
        1 => CollocationFamily::RadauIIA,
        _ => unreachable!("validated collocation family index"),
    };

    let collocation_degree = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::CollocationDegree,
            default.collocation_degree as f64,
        ),
        SharedControlId::CollocationDegree.id(),
    )?
    .round() as usize;
    if !supported_degrees.contains(&collocation_degree) {
        return Err(anyhow!(
            "unsupported {} {}",
            SharedControlId::CollocationDegree.id(),
            collocation_degree
        ));
    }

    let time_grid_choice = parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::TimeGrid,
            time_grid_choice_value(default.time_grid),
        ),
        SharedControlId::TimeGrid,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?;
    let time_grid_strength = expect_unit_interval_finite(
        sample_shared_or_default(
            values,
            SharedControlId::TimeGridStrength,
            time_grid_strength_value(default.time_grid),
        ),
        SharedControlId::TimeGridStrength.id(),
    )?;
    let time_grid_focus_center = expect_unit_interval_finite(
        sample_shared_or_default(
            values,
            SharedControlId::TimeGridFocusCenter,
            time_grid_focus_center_value(default.time_grid),
        ),
        SharedControlId::TimeGridFocusCenter.id(),
    )?;
    let time_grid_focus_width = expect_open_unit_interval_finite(
        sample_shared_or_default(
            values,
            SharedControlId::TimeGridFocusWidth,
            time_grid_focus_width_value(default.time_grid),
        ),
        SharedControlId::TimeGridFocusWidth.id(),
    )?;
    let time_grid_breakpoint = expect_open_unit_interval_finite(
        sample_shared_or_default(
            values,
            SharedControlId::TimeGridBreakpoint,
            time_grid_breakpoint_value(default.time_grid),
        ),
        SharedControlId::TimeGridBreakpoint.id(),
    )?;
    let time_grid_first_interval_fraction = expect_open_unit_interval_finite(
        sample_shared_or_default(
            values,
            SharedControlId::TimeGridFirstIntervalFraction,
            time_grid_first_interval_fraction_value(default.time_grid),
        ),
        SharedControlId::TimeGridFirstIntervalFraction.id(),
    )?;
    let time_grid = match time_grid_choice {
        0 => TimeGrid::Uniform,
        1 => TimeGrid::Cosine {
            strength: time_grid_strength,
        },
        2 => TimeGrid::Tanh {
            strength: time_grid_strength,
        },
        3 => TimeGrid::Geometric {
            strength: time_grid_strength,
            bias: TimeGridBias::Start,
        },
        4 => TimeGrid::Geometric {
            strength: time_grid_strength,
            bias: TimeGridBias::End,
        },
        5 => TimeGrid::Focus {
            center: time_grid_focus_center,
            width: time_grid_focus_width,
            strength: time_grid_strength,
        },
        6 => TimeGrid::Piecewise {
            breakpoint: time_grid_breakpoint,
            first_interval_fraction: time_grid_first_interval_fraction,
        },
        _ => unreachable!("validated time grid choice index"),
    };

    Ok(TranscriptionConfig {
        method,
        intervals,
        collocation_degree,
        collocation_family,
        time_grid,
    })
}

pub fn time_grid_from_map_lossy(values: &BTreeMap<String, f64>, default: TimeGrid) -> TimeGrid {
    let default_choice = time_grid_choice_value(default);
    let default_strength = time_grid_strength_value(default);
    let strength =
        sample_shared_or_default(values, SharedControlId::TimeGridStrength, default_strength);
    let focus_center = sample_shared_or_default(
        values,
        SharedControlId::TimeGridFocusCenter,
        time_grid_focus_center_value(default),
    );
    let focus_width = sample_shared_or_default(
        values,
        SharedControlId::TimeGridFocusWidth,
        time_grid_focus_width_value(default),
    );
    let breakpoint = sample_shared_or_default(
        values,
        SharedControlId::TimeGridBreakpoint,
        time_grid_breakpoint_value(default),
    );
    let first_interval_fraction = sample_shared_or_default(
        values,
        SharedControlId::TimeGridFirstIntervalFraction,
        time_grid_first_interval_fraction_value(default),
    );
    let valid_strength = strength.is_finite() && (0.0..=1.0).contains(&strength);
    let valid_focus_center = focus_center.is_finite() && (0.0..=1.0).contains(&focus_center);
    let valid_focus_width = focus_width.is_finite() && (0.0..1.0).contains(&focus_width);
    let valid_breakpoint = breakpoint.is_finite() && (0.0..1.0).contains(&breakpoint);
    let valid_first_interval_fraction =
        first_interval_fraction.is_finite() && (0.0..1.0).contains(&first_interval_fraction);
    match sample_shared_or_default(values, SharedControlId::TimeGrid, default_choice).round() as i32
    {
        1 if valid_strength => TimeGrid::Cosine { strength },
        2 if valid_strength => TimeGrid::Tanh { strength },
        3 if valid_strength => TimeGrid::Geometric {
            strength,
            bias: TimeGridBias::Start,
        },
        4 if valid_strength => TimeGrid::Geometric {
            strength,
            bias: TimeGridBias::End,
        },
        5 if valid_strength && valid_focus_center && valid_focus_width => TimeGrid::Focus {
            center: focus_center,
            width: focus_width,
            strength,
        },
        6 if valid_breakpoint && valid_first_interval_fraction => TimeGrid::Piecewise {
            breakpoint,
            first_interval_fraction,
        },
        _ => TimeGrid::Uniform,
    }
}

fn parse_call_policy_choice(value: f64, key: SharedControlId) -> Result<CallPolicy> {
    Ok(
        match parse_enum_choice(value, key, &[0.0, 1.0, 2.0, 3.0])? {
            0 => CallPolicy::InlineAtCall,
            1 => CallPolicy::InlineAtLowering,
            2 => CallPolicy::InlineInLLVM,
            3 => CallPolicy::NoInlineLLVM,
            _ => unreachable!("validated call policy choice index"),
        },
    )
}

fn parse_kernel_strategy_choice(value: f64, key: SharedControlId) -> Result<OcpKernelStrategy> {
    Ok(
        match parse_enum_choice(value, key, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])? {
            0 => OcpKernelStrategy::Inline,
            1 => OcpKernelStrategy::FunctionUseGlobalPolicy,
            2 => OcpKernelStrategy::FunctionInlineAtCall,
            3 => OcpKernelStrategy::FunctionInlineAtLowering,
            4 => OcpKernelStrategy::FunctionInlineInLlvm,
            5 => OcpKernelStrategy::FunctionNoInlineLlvm,
            _ => unreachable!("validated kernel strategy choice index"),
        },
    )
}

fn parse_hessian_strategy_choice(value: f64, key: SharedControlId) -> Result<HessianStrategy> {
    Ok(match parse_enum_choice(value, key, &[0.0, 1.0, 2.0])? {
        0 => HessianStrategy::LowerTriangleByColumn,
        1 => HessianStrategy::LowerTriangleSelectedOutputs,
        2 => HessianStrategy::LowerTriangleColored,
        _ => unreachable!("validated Hessian strategy choice index"),
    })
}

pub fn ocp_sx_function_config_from_map(
    values: &BTreeMap<String, f64>,
    default: OcpSxFunctionConfig,
) -> Result<OcpSxFunctionConfig> {
    let global_call_policy = parse_call_policy_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SxFunctionGlobalCallPolicy,
            call_policy_choice_value(default.global_call_policy),
        ),
        SharedControlId::SxFunctionGlobalCallPolicy,
    )?;
    let override_behavior = match parse_enum_choice(
        sample_shared_or_default(
            values,
            SharedControlId::SxFunctionOverrideBehavior,
            override_behavior_choice_value(default.override_behavior),
        ),
        SharedControlId::SxFunctionOverrideBehavior,
        &[0.0, 1.0],
    )? {
        0 => OcpOverrideBehavior::RespectFunctionOverrides,
        1 => OcpOverrideBehavior::StrictGlobalPolicy,
        _ => unreachable!("validated override behavior choice index"),
    };
    Ok(OcpSxFunctionConfig {
        global_call_policy,
        override_behavior,
        hessian_strategy: parse_hessian_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionHessianStrategy,
                hessian_strategy_choice_value(default.hessian_strategy),
            ),
            SharedControlId::SxFunctionHessianStrategy,
        )?,
        ode: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionOde,
                kernel_strategy_choice_value(default.ode),
            ),
            SharedControlId::SxFunctionOde,
        )?,
        objective_lagrange: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionObjectiveLagrange,
                kernel_strategy_choice_value(default.objective_lagrange),
            ),
            SharedControlId::SxFunctionObjectiveLagrange,
        )?,
        objective_mayer: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionObjectiveMayer,
                kernel_strategy_choice_value(default.objective_mayer),
            ),
            SharedControlId::SxFunctionObjectiveMayer,
        )?,
        path_constraints: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionPathConstraints,
                kernel_strategy_choice_value(default.path_constraints),
            ),
            SharedControlId::SxFunctionPathConstraints,
        )?,
        boundary_equalities: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionBoundaryEqualities,
                kernel_strategy_choice_value(default.boundary_equalities),
            ),
            SharedControlId::SxFunctionBoundaryEqualities,
        )?,
        boundary_inequalities: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionBoundaryInequalities,
                kernel_strategy_choice_value(default.boundary_inequalities),
            ),
            SharedControlId::SxFunctionBoundaryInequalities,
        )?,
        multiple_shooting_integrator: parse_kernel_strategy_choice(
            sample_shared_or_default(
                values,
                SharedControlId::SxFunctionMultipleShootingIntegrator,
                kernel_strategy_choice_value(default.multiple_shooting_integrator),
            ),
            SharedControlId::SxFunctionMultipleShootingIntegrator,
        )?,
    })
}

pub fn ocp_sx_function_config_from_map_lossy(
    values: &BTreeMap<String, f64>,
    default: OcpSxFunctionConfig,
) -> OcpSxFunctionConfig {
    ocp_sx_function_config_from_map(values, default).unwrap_or(default)
}

fn call_policy_choice_value(policy: CallPolicy) -> f64 {
    match policy {
        CallPolicy::InlineAtCall => 0.0,
        CallPolicy::InlineAtLowering => 1.0,
        CallPolicy::InlineInLLVM => 2.0,
        CallPolicy::NoInlineLLVM => 3.0,
    }
}

fn override_behavior_choice_value(behavior: OcpOverrideBehavior) -> f64 {
    match behavior {
        OcpOverrideBehavior::RespectFunctionOverrides => 0.0,
        OcpOverrideBehavior::StrictGlobalPolicy => 1.0,
    }
}

fn hessian_strategy_choice_value(strategy: HessianStrategy) -> f64 {
    match strategy {
        HessianStrategy::LowerTriangleByColumn => 0.0,
        HessianStrategy::LowerTriangleSelectedOutputs => 1.0,
        HessianStrategy::LowerTriangleColored => 2.0,
    }
}

fn kernel_strategy_choice_value(strategy: OcpKernelStrategy) -> f64 {
    match strategy {
        OcpKernelStrategy::Inline => 0.0,
        OcpKernelStrategy::FunctionUseGlobalPolicy => 1.0,
        OcpKernelStrategy::FunctionInlineAtCall => 2.0,
        OcpKernelStrategy::FunctionInlineAtLowering => 3.0,
        OcpKernelStrategy::FunctionInlineInLlvm => 4.0,
        OcpKernelStrategy::FunctionNoInlineLlvm => 5.0,
    }
}

pub fn sqp_options(config: &SolverConfig) -> ClarabelSqpOptions {
    ClarabelSqpOptions {
        max_iters: config.max_iters,
        hessian_regularization_enabled: config.sqp.hessian_regularization_enabled,
        dual_tol: config.dual_tol,
        constraint_tol: config.constraint_tol,
        complementarity_tol: config.complementarity_tol,
        globalization: match config.sqp.globalization {
            SqpGlobalizationConfig::LineSearchFilter {
                line_search,
                filter,
                exact_merit_penalty,
            } => SqpGlobalization::LineSearchFilter(LineSearchFilterOptions {
                line_search: SqpLineSearchOptions {
                    armijo_c1: line_search.armijo_c1,
                    wolfe_c2: line_search.wolfe_c2,
                    beta: line_search.beta,
                    max_steps: line_search.max_steps,
                    min_step: line_search.min_step,
                },
                filter: SqpFilterOptions {
                    armijo_c1: line_search.armijo_c1,
                    gamma_objective: filter.gamma_objective,
                    gamma_violation: filter.gamma_violation,
                    theta_max_factor: filter.theta_max_factor,
                    switching_reference_min: filter.switching_reference_min,
                    switching_violation_factor: filter.switching_violation_factor,
                    switching_linearized_reduction_factor: filter
                        .switching_linearized_reduction_factor,
                },
                exact_merit_penalty,
            }),
            SqpGlobalizationConfig::LineSearchMerit {
                line_search,
                exact_merit_penalty,
                penalty_increase_factor,
                max_penalty_updates,
            } => SqpGlobalization::LineSearchMerit(LineSearchMeritOptions {
                line_search: SqpLineSearchOptions {
                    armijo_c1: line_search.armijo_c1,
                    wolfe_c2: line_search.wolfe_c2,
                    beta: line_search.beta,
                    max_steps: line_search.max_steps,
                    min_step: line_search.min_step,
                },
                exact_merit_penalty,
                penalty_increase_factor,
                max_penalty_updates,
            }),
            SqpGlobalizationConfig::TrustRegionFilter {
                trust_region,
                filter,
                exact_merit_penalty,
            } => SqpGlobalization::TrustRegionFilter(TrustRegionFilterOptions {
                trust_region: SqpTrustRegionOptions {
                    initial_radius: trust_region.initial_radius,
                    max_radius: trust_region.max_radius,
                    min_radius: trust_region.min_radius,
                    shrink_factor: trust_region.shrink_factor,
                    grow_factor: trust_region.grow_factor,
                    accept_ratio: trust_region.accept_ratio,
                    expand_ratio: trust_region.expand_ratio,
                    boundary_fraction: trust_region.boundary_fraction,
                    max_radius_contractions: trust_region.max_radius_contractions,
                },
                filter: SqpFilterOptions {
                    armijo_c1: default_sqp_line_search_config().armijo_c1,
                    gamma_objective: filter.gamma_objective,
                    gamma_violation: filter.gamma_violation,
                    theta_max_factor: filter.theta_max_factor,
                    switching_reference_min: filter.switching_reference_min,
                    switching_violation_factor: filter.switching_violation_factor,
                    switching_linearized_reduction_factor: filter
                        .switching_linearized_reduction_factor,
                },
                exact_merit_penalty,
            }),
            SqpGlobalizationConfig::TrustRegionMerit {
                trust_region,
                exact_merit_penalty,
                fixed_penalty,
            } => SqpGlobalization::TrustRegionMerit(TrustRegionMeritOptions {
                trust_region: SqpTrustRegionOptions {
                    initial_radius: trust_region.initial_radius,
                    max_radius: trust_region.max_radius,
                    min_radius: trust_region.min_radius,
                    shrink_factor: trust_region.shrink_factor,
                    grow_factor: trust_region.grow_factor,
                    accept_ratio: trust_region.accept_ratio,
                    expand_ratio: trust_region.expand_ratio,
                    boundary_fraction: trust_region.boundary_fraction,
                    max_radius_contractions: trust_region.max_radius_contractions,
                },
                exact_merit_penalty,
                fixed_penalty,
            }),
        },
        ..ClarabelSqpOptions::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AcceptableTerminationOptions {
    tol: f64,
    iter: usize,
    dual_inf_tol: f64,
    constr_viol_tol: f64,
    compl_inf_tol: f64,
    obj_change_tol: f64,
}

fn acceptable_termination_options(config: &SolverConfig) -> AcceptableTerminationOptions {
    let tol = (100.0 * config.overall_tol).max(config.overall_tol);
    match config.profile {
        SolverProfile::StrictOcp => AcceptableTerminationOptions {
            tol,
            iter: 15,
            dual_inf_tol: config.dual_tol,
            constr_viol_tol: config.constraint_tol,
            compl_inf_tol: config.complementarity_tol,
            obj_change_tol: 1.0e20,
        },
        SolverProfile::IpoptDefault => AcceptableTerminationOptions {
            tol,
            iter: 15,
            dual_inf_tol: 1.0e10,
            constr_viol_tol: 1.0e-2,
            compl_inf_tol: 1.0e-2,
            obj_change_tol: 1.0e20,
        },
    }
}

pub fn nlip_options(config: &SolverConfig) -> InteriorPointOptions {
    let acceptable = acceptable_termination_options(config);
    InteriorPointOptions {
        max_iters: config.max_iters,
        dual_tol: config.dual_tol,
        constraint_tol: config.constraint_tol,
        complementarity_tol: config.complementarity_tol,
        overall_tol: config.overall_tol,
        acceptable_tol: acceptable.tol,
        acceptable_iter: acceptable.iter,
        acceptable_dual_inf_tol: acceptable.dual_inf_tol,
        acceptable_constr_viol_tol: acceptable.constr_viol_tol,
        acceptable_compl_inf_tol: acceptable.compl_inf_tol,
        acceptable_obj_change_tol: acceptable.obj_change_tol,
        linear_solver: config.nlip.linear_solver,
        spral_pivot_method: config.nlip.spral_pivot_method,
        spral_action_on_zero_pivot: config.nlip.spral_action_on_zero_pivot,
        spral_small_pivot_tolerance: config.nlip.spral_small_pivot_tolerance,
        spral_threshold_pivot_u: config.nlip.spral_threshold_pivot_u,
        ..InteriorPointOptions::default()
    }
}

#[cfg(feature = "ipopt")]
pub fn ipopt_options(config: &SolverConfig) -> IpoptOptions {
    let acceptable = acceptable_termination_options(config);
    IpoptOptions {
        max_iters: config.max_iters,
        tol: config.overall_tol,
        acceptable_tol: Some(acceptable.tol),
        acceptable_iter: Some(acceptable.iter),
        acceptable_dual_tol: Some(acceptable.dual_inf_tol),
        acceptable_constraint_tol: Some(acceptable.constr_viol_tol),
        acceptable_complementarity_tol: Some(acceptable.compl_inf_tol),
        acceptable_obj_change_tol: Some(acceptable.obj_change_tol),
        constraint_tol: Some(config.constraint_tol),
        complementarity_tol: Some(config.complementarity_tol),
        dual_tol: Some(config.dual_tol),
        print_level: 5,
        suppress_banner: false,
        linear_solver: Some(IpoptLinearSolver::Spral),
        raw_options: vec![IpoptRawOption::text("print_timing_statistics", "yes")],
        ..IpoptOptions::default()
    }
}

pub fn transcription_metrics(config: &TranscriptionConfig) -> [Metric; 4] {
    [
        metric_with_key(
            MetricKey::TranscriptionMethod,
            "Transcription",
            match config.method {
                TranscriptionMethod::MultipleShooting => "Multiple Shooting".to_string(),
                TranscriptionMethod::DirectCollocation => format!(
                    "{} Collocation",
                    match config.collocation_family {
                        CollocationFamily::GaussLegendre => "Legendre",
                        CollocationFamily::RadauIIA => "Radau IIA",
                    }
                ),
            },
        ),
        metric_with_key(
            MetricKey::IntervalCount,
            "Intervals",
            config.intervals.to_string(),
        ),
        metric_with_key(
            MetricKey::CollocationNodeCount,
            "Collocation Nodes",
            match config.method {
                TranscriptionMethod::MultipleShooting => "--".to_string(),
                TranscriptionMethod::DirectCollocation => config.collocation_degree.to_string(),
            },
        ),
        metric_with_key(
            MetricKey::TimeGrid,
            "Time Grid",
            time_grid_label(config.time_grid),
        ),
    ]
}

fn time_grid_label(time_grid: TimeGrid) -> String {
    match time_grid {
        TimeGrid::Uniform => "Uniform".to_string(),
        TimeGrid::Cosine { strength } => format!("Cosine ({strength:.2})"),
        TimeGrid::Tanh { strength } => format!("Tanh ({strength:.2})"),
        TimeGrid::Geometric {
            strength,
            bias: TimeGridBias::Start,
        } => format!("Geometric Start ({strength:.2})"),
        TimeGrid::Geometric {
            strength,
            bias: TimeGridBias::End,
        } => format!("Geometric End ({strength:.2})"),
        TimeGrid::Focus {
            center,
            width,
            strength,
        } => format!("Focus (c {center:.2}, w {width:.2}, s {strength:.2})"),
        TimeGrid::Piecewise {
            breakpoint,
            first_interval_fraction,
        } => format!("Piecewise (b {breakpoint:.2}, n1 {first_interval_fraction:.2})"),
    }
}

pub fn chart(
    title: impl Into<String>,
    y_label: impl Into<String>,
    series: Vec<TimeSeries>,
) -> Chart {
    Chart {
        title: title.into(),
        x_label: "Time (s)".to_string(),
        y_label: y_label.into(),
        series,
    }
}

pub fn metric(label: impl Into<String>, value: impl Into<String>) -> Metric {
    metric_with_key(MetricKey::Custom, label, value)
}

pub fn metric_with_key(
    key: MetricKey,
    label: impl Into<String>,
    value: impl Into<String>,
) -> Metric {
    Metric {
        key,
        label: label.into(),
        value: value.into(),
        numeric_value: None,
    }
}

pub fn numeric_metric_with_key(
    key: MetricKey,
    label: impl Into<String>,
    numeric_value: f64,
    value: impl Into<String>,
) -> Metric {
    Metric {
        key,
        label: label.into(),
        value: value.into(),
        numeric_value: Some(numeric_value),
    }
}

pub fn find_metric(summary: &[Metric], key: MetricKey) -> Option<&Metric> {
    summary.iter().find(|metric| metric.key == key)
}

pub fn select_control<S: AsRef<str>>(
    id: impl Into<String>,
    label: impl Into<String>,
    default: f64,
    unit: impl Into<String>,
    help: impl Into<String>,
    choices: &[(f64, S)],
    section: ControlSection,
    visibility: ControlVisibility,
    semantic: ControlSemantic,
) -> ControlSpec {
    let choice_list = choices
        .iter()
        .map(|(value, label)| ControlChoice {
            value: *value,
            label: label.as_ref().to_string(),
        })
        .collect::<Vec<_>>();
    ControlSpec {
        id: id.into(),
        label: label.into(),
        min: choice_list
            .first()
            .map(|choice| choice.value)
            .unwrap_or(default),
        max: choice_list
            .last()
            .map(|choice| choice.value)
            .unwrap_or(default),
        step: 1.0,
        default,
        unit: unit.into(),
        help: help.into(),
        section,
        panel: None,
        editor: ControlEditor::Select,
        visibility,
        semantic,
        value_display: ControlValueDisplay::Scalar,
        choices: choice_list,
        profile_defaults: Vec::new(),
    }
}

pub fn text_control(
    id: impl Into<String>,
    label: impl Into<String>,
    default: f64,
    unit: impl Into<String>,
    help: impl Into<String>,
    semantic: ControlSemantic,
    value_display: ControlValueDisplay,
) -> ControlSpec {
    ControlSpec {
        id: id.into(),
        label: label.into(),
        min: 0.0,
        max: f64::MAX,
        step: default,
        default,
        unit: unit.into(),
        help: help.into(),
        section: ControlSection::Solver,
        panel: None,
        editor: ControlEditor::Text,
        visibility: ControlVisibility::Always,
        semantic,
        value_display,
        choices: Vec::new(),
        profile_defaults: Vec::new(),
    }
}

pub fn problem_slider_control(
    id: impl Into<String>,
    label: impl Into<String>,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
    unit: impl Into<String>,
    help: impl Into<String>,
) -> ControlSpec {
    problem_slider_control_with_display(
        id,
        label,
        min,
        max,
        step,
        default,
        unit,
        help,
        ControlValueDisplay::Scalar,
    )
}

pub fn problem_scientific_slider_control(
    id: impl Into<String>,
    label: impl Into<String>,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
    unit: impl Into<String>,
    help: impl Into<String>,
) -> ControlSpec {
    problem_slider_control_with_display(
        id,
        label,
        min,
        max,
        step,
        default,
        unit,
        help,
        ControlValueDisplay::Scientific,
    )
}

fn problem_slider_control_with_display(
    id: impl Into<String>,
    label: impl Into<String>,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
    unit: impl Into<String>,
    help: impl Into<String>,
    value_display: ControlValueDisplay,
) -> ControlSpec {
    ControlSpec {
        id: id.into(),
        label: label.into(),
        min,
        max,
        step,
        default,
        unit: unit.into(),
        help: help.into(),
        section: ControlSection::Problem,
        panel: None,
        editor: ControlEditor::Slider,
        visibility: ControlVisibility::Always,
        semantic: ControlSemantic::ProblemParameter,
        value_display,
        choices: Vec::new(),
        profile_defaults: Vec::new(),
    }
}

pub fn problem_controls(
    transcription: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
    solver_method: SolverMethod,
    solver: SolverConfig,
    extra_controls: impl IntoIterator<Item = ControlSpec>,
) -> Vec<ControlSpec> {
    let mut controls =
        transcription_controls(transcription, supported_intervals, supported_degrees);
    controls.extend(solver_controls(solver_method, solver));
    controls.extend(extra_controls);
    controls
}

pub fn problem_spec(
    id: ProblemId,
    name: impl Into<String>,
    description: impl Into<String>,
    controls: Vec<ControlSpec>,
    math_sections: Vec<LatexSection>,
    notes: Vec<String>,
) -> ProblemSpec {
    ProblemSpec {
        id,
        name: name.into(),
        description: description.into(),
        controls,
        math_sections,
        notes,
    }
}

pub fn segmented_series(
    name: impl Into<String>,
    segments: impl IntoIterator<Item = (Vec<f64>, Vec<f64>)>,
    mode: PlotMode,
) -> Vec<TimeSeries> {
    let name = name.into();
    segments
        .into_iter()
        .enumerate()
        .map(|(index, (x, y))| TimeSeries {
            name: name.clone(),
            x,
            y,
            mode: Some(mode),
            legend_group: Some(name.clone()),
            show_legend: index == 0,
            role: TimeSeriesRole::Data,
        })
        .collect()
}

fn constant_segmented_series_with_role(
    name: impl Into<String>,
    segments: impl IntoIterator<Item = Vec<f64>>,
    value: f64,
    mode: PlotMode,
    role: TimeSeriesRole,
) -> Vec<TimeSeries> {
    let mut out = segmented_series(
        name,
        segments.into_iter().map(|times| {
            let y = vec![value; times.len()];
            (times, y)
        }),
        mode,
    );
    for series in &mut out {
        series.role = role;
    }
    out
}

pub fn interval_arc_series<T, F>(
    name: impl Into<String>,
    arcs: &[IntervalArc<T>],
    mode: PlotMode,
    extract: F,
) -> Vec<TimeSeries>
where
    F: Fn(&T) -> f64,
{
    let name = name.into();
    segmented_series(
        name,
        arcs.iter().map(|arc| {
            (
                arc.times.clone(),
                arc.values.iter().map(&extract).collect::<Vec<_>>(),
            )
        }),
        mode,
    )
}

pub fn segmented_bound_series(
    segments: impl IntoIterator<Item = Vec<f64>>,
    lower: Option<f64>,
    upper: Option<f64>,
    mode: PlotMode,
) -> Vec<TimeSeries> {
    let segment_list = segments.into_iter().collect::<Vec<_>>();
    let mut out = Vec::new();
    if let Some(value) = lower {
        out.extend(constant_segmented_series_with_role(
            "Lower Bound",
            segment_list.iter().cloned(),
            value,
            mode,
            TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(value) = upper {
        out.extend(constant_segmented_series_with_role(
            "Upper Bound",
            segment_list.iter().cloned(),
            value,
            mode,
            TimeSeriesRole::UpperBound,
        ));
    }
    out
}

pub fn interval_arc_bound_series<T>(
    arcs: &[IntervalArc<T>],
    lower: Option<f64>,
    upper: Option<f64>,
    mode: PlotMode,
) -> Vec<TimeSeries> {
    segmented_bound_series(arcs.iter().map(|arc| arc.times.clone()), lower, upper, mode)
}

fn is_true(value: &bool) -> bool {
    *value
}

pub fn sample_or_default(values: &BTreeMap<String, f64>, key: &str, default: f64) -> f64 {
    values.get(key).copied().unwrap_or(default)
}

pub fn expect_finite(value: f64, label: &str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(anyhow!("{label} must be finite"))
    }
}

pub fn expect_positive_finite(value: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    if value <= 0.0 {
        return Err(anyhow!("{label} must be positive"));
    }
    Ok(value)
}

pub fn expect_nonnegative_finite(value: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    if value < 0.0 {
        return Err(anyhow!("{label} must be nonnegative"));
    }
    Ok(value)
}

pub fn expect_unit_interval_finite(value: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(anyhow!("{label} must be in [0, 1]"));
    }
    Ok(value)
}

pub fn expect_open_unit_interval_finite(value: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(anyhow!("{label} must be finite"));
    }
    if !(0.0..1.0).contains(&value) {
        return Err(anyhow!("{label} must be in (0, 1)"));
    }
    Ok(value)
}

pub fn deg_to_rad(value: f64) -> f64 {
    value.to_radians()
}

pub fn rad_to_deg(value: f64) -> f64 {
    value.to_degrees()
}

pub fn node_times(tf: f64, intervals: usize) -> Vec<f64> {
    let step = tf / intervals as f64;
    (0..=intervals).map(|index| index as f64 * step).collect()
}

pub fn trapezoid_integral(times: &[f64], values: &[f64]) -> f64 {
    times
        .windows(2)
        .zip(values.windows(2))
        .map(|(time, value)| 0.5 * (time[1] - time[0]) * (value[0] + value[1]))
        .sum()
}

pub fn with_solve_time(mut artifact: SolveArtifact, started: Instant) -> SolveArtifact {
    if artifact.solver.solve_s.is_none() {
        artifact.solver.solve_s = Some(started.elapsed().as_secs_f64());
    }
    artifact
}

fn panel_severity_from_core(value: ConstraintSatisfaction) -> ConstraintPanelSeverity {
    match value {
        ConstraintSatisfaction::FullAccuracy => ConstraintPanelSeverity::FullAccuracy,
        ConstraintSatisfaction::ReducedAccuracy => ConstraintPanelSeverity::ReducedAccuracy,
        ConstraintSatisfaction::Violated => ConstraintPanelSeverity::Violated,
    }
}

fn panel_category_from_core(value: OcpConstraintCategory) -> ConstraintPanelCategory {
    match value {
        OcpConstraintCategory::BoundaryEquality => ConstraintPanelCategory::BoundaryEquality,
        OcpConstraintCategory::BoundaryInequality => ConstraintPanelCategory::BoundaryInequality,
        OcpConstraintCategory::Path => ConstraintPanelCategory::Path,
        OcpConstraintCategory::ContinuityState => ConstraintPanelCategory::ContinuityState,
        OcpConstraintCategory::ContinuityControl => ConstraintPanelCategory::ContinuityControl,
        OcpConstraintCategory::CollocationState => ConstraintPanelCategory::CollocationState,
        OcpConstraintCategory::CollocationControl => ConstraintPanelCategory::CollocationControl,
        OcpConstraintCategory::FinalTime => ConstraintPanelCategory::FinalTime,
    }
}

fn constraint_panels_from_report(report: OcpConstraintViolationReport) -> ConstraintPanels {
    ConstraintPanels {
        equalities: report
            .equalities
            .into_iter()
            .map(|group| ConstraintPanelEntry {
                label: group.label,
                category: panel_category_from_core(group.category),
                worst_violation: group.worst_violation,
                violating_instances: group.violating_instances,
                total_instances: group.total_instances,
                severity: panel_severity_from_core(group.satisfaction),
                lower_bound: None,
                upper_bound: None,
                lower_severity: None,
                upper_severity: None,
            })
            .collect(),
        inequalities: report
            .inequalities
            .into_iter()
            .map(|group| ConstraintPanelEntry {
                label: group.label,
                category: panel_category_from_core(group.category),
                worst_violation: group.worst_violation,
                violating_instances: group.violating_instances,
                total_instances: group.total_instances,
                severity: panel_severity_from_core(group.satisfaction),
                lower_bound: group.lower_bound,
                upper_bound: group.upper_bound,
                lower_severity: group.lower_satisfaction.map(panel_severity_from_core),
                upper_severity: group.upper_satisfaction.map(panel_severity_from_core),
            })
            .collect(),
    }
}

fn attach_constraint_panels(artifact: &mut SolveArtifact, report: OcpConstraintViolationReport) {
    artifact.constraint_panels = constraint_panels_from_report(report);
}

fn try_attach_multiple_shooting_constraint_panels<Compiled>(
    artifact: &mut SolveArtifact,
    compiled: &Compiled,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    trajectories: &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum>,
    tolerance: f64,
) -> Result<()>
where
    Compiled: MultipleShootingCompiled,
{
    let report = compiled.build_constraint_violation_report(runtime, trajectories, tolerance)?;
    attach_constraint_panels(artifact, report);
    Ok(())
}

fn try_attach_direct_collocation_constraint_panels<Compiled>(
    artifact: &mut SolveArtifact,
    compiled: &Compiled,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    trajectories: &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum>,
    tolerance: f64,
) -> Result<()>
where
    Compiled: DirectCollocationCompiled,
{
    let report = compiled.build_constraint_violation_report(runtime, trajectories, tolerance)?;
    attach_constraint_panels(artifact, report);
    Ok(())
}

pub trait CompiledOcpMetadata {
    fn backend_timing_metadata(&self) -> BackendTimingMetadata;
    fn nlp_compile_stats(&self) -> NlpCompileStats;
    fn helper_compile_stats(&self) -> OcpHelperCompileStats;
    fn helper_kernel_count(&self) -> usize;
    fn backend_compile_report(&self) -> &BackendCompileReport;
}

pub trait MultipleShootingCompiled: CompiledOcpMetadata {
    type PNum;
    type CBounds;
    type BeqNum;
    type BineqBounds;
    type XNum: Clone;
    type UNum: Clone;

    fn run_sqp(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<MultipleShootingSqpSolveResult<Self::XNum, Self::UNum>>;

    fn run_sqp_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<MultipleShootingSqpSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&MultipleShootingSqpSnapshot<Self::XNum, Self::UNum>);

    fn run_nlip(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
    ) -> Result<MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum>>;

    fn run_nlip_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&MultipleShootingInteriorPointSnapshot<Self::XNum, Self::UNum>);

    #[cfg(feature = "ipopt")]
    fn run_ipopt(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
    ) -> Result<MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum>>;

    #[cfg(feature = "ipopt")]
    fn run_ipopt_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&MultipleShootingIpoptSnapshot<Self::XNum, Self::UNum>);

    fn build_interval_arcs(
        &self,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum>,
        parameters: &Self::PNum,
    ) -> Result<(Vec<IntervalArc<Self::XNum>>, Vec<IntervalArc<Self::UNum>>)>;

    fn build_constraint_violation_report(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport>;

    fn validate_nlp_derivatives(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> Result<NlpDerivativeValidationReport>;

    fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind);
}

pub trait DirectCollocationCompiled: CompiledOcpMetadata {
    type PNum;
    type CBounds;
    type BeqNum;
    type BineqBounds;
    type XNum: Clone;
    type UNum: Clone;

    fn run_sqp(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<DirectCollocationSqpSolveResult<Self::XNum, Self::UNum>>;

    fn run_sqp_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<DirectCollocationSqpSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&DirectCollocationSqpSnapshot<Self::XNum, Self::UNum>);

    fn run_nlip(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
    ) -> Result<DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum>>;

    fn run_nlip_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&DirectCollocationInteriorPointSnapshot<Self::XNum, Self::UNum>);

    #[cfg(feature = "ipopt")]
    fn run_ipopt(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
    ) -> Result<DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum>>;

    #[cfg(feature = "ipopt")]
    fn run_ipopt_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&DirectCollocationIpoptSnapshot<Self::XNum, Self::UNum>);

    fn build_constraint_violation_report(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        trajectories: &DirectCollocationTrajectories<Self::XNum, Self::UNum>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport>;

    fn validate_nlp_derivatives(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> Result<NlpDerivativeValidationReport>;

    fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind);
}

impl<X, U, P, C, Beq, Bineq> CompiledOcpMetadata
    for ocp_runtime::CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq>
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
    <C as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <C as Vectorize<SX>>::Rebind<Bounds1D>>,
    <Bineq as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>>,
{
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::backend_timing_metadata(
            self,
        )
    }

    fn nlp_compile_stats(&self) -> NlpCompileStats {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::nlp_compile_stats(self)
    }

    fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::helper_compile_stats(
            self,
        )
    }

    fn helper_kernel_count(&self) -> usize {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::helper_kernel_count(
            self,
        )
    }

    fn backend_compile_report(&self) -> &BackendCompileReport {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::backend_compile_report(
            self,
        )
    }
}

impl<X, U, P, C, Beq, Bineq> MultipleShootingCompiled
    for ocp_runtime::CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq>
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
    <C as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <C as Vectorize<SX>>::Rebind<Bounds1D>>,
    <Bineq as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>>,
{
    type PNum = Numeric<P>;
    type CBounds = <C as Vectorize<SX>>::Rebind<Bounds1D>;
    type BeqNum = Numeric<Beq>;
    type BineqBounds = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>;
    type XNum = Numeric<X>;
    type UNum = Numeric<U>;

    fn run_sqp(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<MultipleShootingSqpSolveResult<Self::XNum, Self::UNum>> {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::solve_sqp(
            self, values, options,
        )
        .map_err(Into::into)
    }

    fn run_sqp_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<MultipleShootingSqpSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&MultipleShootingSqpSnapshot<Self::XNum, Self::UNum>),
    {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::solve_sqp_with_callback(
            self, values, options, callback,
        )
        .map_err(Into::into)
    }

    fn run_nlip(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
    ) -> Result<MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum>> {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::solve_interior_point(
            self, values, options,
        )
        .map_err(Into::into)
    }

    fn run_nlip_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&MultipleShootingInteriorPointSnapshot<Self::XNum, Self::UNum>),
    {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::solve_interior_point_with_callback(
            self,
            values,
            options,
            callback,
        )
        .map_err(Into::into)
    }

    #[cfg(feature = "ipopt")]
    fn run_ipopt(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
    ) -> Result<MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum>> {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::solve_ipopt(
            self, values, options,
        )
        .map_err(Into::into)
    }

    #[cfg(feature = "ipopt")]
    fn run_ipopt_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&MultipleShootingIpoptSnapshot<Self::XNum, Self::UNum>),
    {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::solve_ipopt_with_callback(
            self,
            values,
            options,
            callback,
        )
        .map_err(Into::into)
    }

    fn build_interval_arcs(
        &self,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum>,
        parameters: &Self::PNum,
    ) -> Result<(Vec<IntervalArc<Self::XNum>>, Vec<IntervalArc<Self::UNum>>)> {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::interval_arcs(
            self,
            trajectories,
            parameters,
        )
        .map_err(Into::into)
    }

    fn build_constraint_violation_report(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport> {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::rank_constraint_violations(
            self,
            values,
            trajectories,
            tolerance,
        )
        .map_err(Into::into)
    }

    fn validate_nlp_derivatives(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> Result<NlpDerivativeValidationReport> {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::validate_nlp_derivatives(
            self,
            values,
            equality_multipliers,
            inequality_multipliers,
            options,
        )
        .map_err(Into::into)
    }

    fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        ocp_runtime::CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq>::benchmark_nlp_evaluations_with_progress(
            self,
            values,
            options,
            on_progress,
        )
        .map_err(Into::into)
    }
}

impl<X, U, P, C, Beq, Bineq> CompiledOcpMetadata
    for ocp_runtime::CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq>
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
    <C as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <C as Vectorize<SX>>::Rebind<Bounds1D>>,
    <Bineq as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>>,
{
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::backend_timing_metadata(
            self,
        )
    }

    fn nlp_compile_stats(&self) -> NlpCompileStats {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::nlp_compile_stats(self)
    }

    fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::helper_compile_stats(
            self,
        )
    }

    fn helper_kernel_count(&self) -> usize {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::helper_kernel_count(
            self,
        )
    }

    fn backend_compile_report(&self) -> &BackendCompileReport {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::backend_compile_report(
            self,
        )
    }
}

impl<X, U, P, C, Beq, Bineq> DirectCollocationCompiled
    for ocp_runtime::CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq>
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
    <C as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <C as Vectorize<SX>>::Rebind<Bounds1D>>,
    <Bineq as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>>,
{
    type PNum = Numeric<P>;
    type CBounds = <C as Vectorize<SX>>::Rebind<Bounds1D>;
    type BeqNum = Numeric<Beq>;
    type BineqBounds = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>;
    type XNum = Numeric<X>;
    type UNum = Numeric<U>;

    fn run_sqp(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<DirectCollocationSqpSolveResult<Self::XNum, Self::UNum>> {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::solve_sqp(
            self, values, options,
        )
        .map_err(Into::into)
    }

    fn run_sqp_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<DirectCollocationSqpSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&DirectCollocationSqpSnapshot<Self::XNum, Self::UNum>),
    {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::solve_sqp_with_callback(
            self, values, options, callback,
        )
        .map_err(Into::into)
    }

    fn run_nlip(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
    ) -> Result<DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum>> {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::solve_interior_point(
            self, values, options,
        )
        .map_err(Into::into)
    }

    fn run_nlip_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&DirectCollocationInteriorPointSnapshot<Self::XNum, Self::UNum>),
    {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::solve_interior_point_with_callback(
            self,
            values,
            options,
            callback,
        )
        .map_err(Into::into)
    }

    #[cfg(feature = "ipopt")]
    fn run_ipopt(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
    ) -> Result<DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum>> {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::solve_ipopt(
            self, values, options,
        )
        .map_err(Into::into)
    }

    #[cfg(feature = "ipopt")]
    fn run_ipopt_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum>>
    where
        CB: FnMut(&DirectCollocationIpoptSnapshot<Self::XNum, Self::UNum>),
    {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::solve_ipopt_with_callback(
            self, values, options, callback,
        )
        .map_err(Into::into)
    }

    fn build_constraint_violation_report(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        trajectories: &DirectCollocationTrajectories<Self::XNum, Self::UNum>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport> {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::rank_constraint_violations(
            self,
            values,
            trajectories,
            tolerance,
        )
        .map_err(Into::into)
    }

    fn validate_nlp_derivatives(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> Result<NlpDerivativeValidationReport> {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::validate_nlp_derivatives(
            self,
            values,
            equality_multipliers,
            inequality_multipliers,
            options,
        )
        .map_err(Into::into)
    }

    fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        ocp_runtime::CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq>::benchmark_nlp_evaluations_with_progress(
            self,
            values,
            options,
            on_progress,
        )
        .map_err(Into::into)
    }
}

pub fn solve_multiple_shooting_problem<Compiled, Build>(
    compiled: &Compiled,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    mut build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum>,
        &[IntervalArc<Compiled::XNum>],
        &[IntervalArc<Compiled::UNum>],
    ) -> SolveArtifact,
{
    let started = Instant::now();
    match solver_method {
        SolverMethod::Sqp => {
            let options = sqp_options(solver_config);
            let solved = compiled.run_sqp(runtime, &options)?;
            let (x_arcs, u_arcs) =
                compiled.build_interval_arcs(&solved.trajectories, &runtime.parameters)?;
            let mut artifact = build_artifact(&solved.trajectories, &x_arcs, &u_arcs);
            let _ = try_attach_multiple_shooting_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            Ok(with_solve_time(artifact, started))
        }
        SolverMethod::Nlip => {
            let options = nlip_options(solver_config);
            let solved = compiled.run_nlip(runtime, &options)?;
            let (x_arcs, u_arcs) =
                compiled.build_interval_arcs(&solved.trajectories, &runtime.parameters)?;
            let mut artifact = build_artifact(&solved.trajectories, &x_arcs, &u_arcs);
            let _ = try_attach_multiple_shooting_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            Ok(with_solve_time(artifact, started))
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let options = ipopt_options(solver_config);
            let solved = compiled.run_ipopt(runtime, &options)?;
            let (x_arcs, u_arcs) =
                compiled.build_interval_arcs(&solved.trajectories, &runtime.parameters)?;
            let mut artifact = build_artifact(&solved.trajectories, &x_arcs, &u_arcs);
            let _ = try_attach_multiple_shooting_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_ipopt_termination_metric(&mut artifact, &solved.solver);
            attach_ipopt_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            Ok(with_solve_time(artifact, started))
        }
    }
}

pub fn solve_cached_multiple_shooting_problem<Compiled, Build>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum>,
        &[IntervalArc<Compiled::XNum>],
        &[IntervalArc<Compiled::UNum>],
    ) -> SolveArtifact,
{
    let compiled = compiled.borrow();
    solve_multiple_shooting_problem(
        &*compiled,
        runtime,
        solver_method,
        solver_config,
        build_artifact,
    )
}

pub fn solve_multiple_shooting_problem_with_progress<Compiled, Emit, Build>(
    compiled: &Compiled,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum>,
        &[IntervalArc<Compiled::XNum>],
        &[IntervalArc<Compiled::UNum>],
    ) -> SolveArtifact,
{
    let started = Instant::now();
    let emit = Arc::new(Mutex::new(emit));
    let build_artifact = Arc::new(Mutex::new(build_artifact));
    match solver_method {
        SolverMethod::Sqp => {
            let options = sqp_options(solver_config);
            emit_event(
                &emit,
                SolveStreamEvent::Status {
                    status: SolveStatus {
                        stage: SolveStage::Solving,
                        solver_method: Some(solver_method),
                        solver: running_solver.clone().with_solve_seconds(0.0),
                    },
                },
            );
            let emit_for_worker = emit.clone();
            let solve_started = started;
            let running_solver_for_callback = running_solver.clone();
            let solved = match with_latest_only_worker(
                move |event| {
                    emit_event(&emit_for_worker, event);
                },
                |submit| {
                    let build_for_callback = build_artifact.clone();
                    let options_for_callback = options.clone();
                    compiled.run_sqp_with_callback(
                        runtime,
                        &options_for_callback,
                        move |snapshot| match compiled
                            .build_interval_arcs(&snapshot.trajectories, &runtime.parameters)
                        {
                            Ok((x_arcs, u_arcs)) => {
                                let mut builder = build_for_callback
                                    .lock()
                                    .expect("artifact builder poisoned");
                                let mut artifact =
                                    (*builder)(&snapshot.trajectories, &x_arcs, &u_arcs);
                                drop(builder);
                                let progress = sqp_progress(&snapshot.solver);
                                artifact.solver = running_solver_for_callback
                                    .clone()
                                    .with_iterations(progress.iteration)
                                    .with_solve_seconds(solve_started.elapsed().as_secs_f64());
                                if let Err(error) = try_attach_multiple_shooting_constraint_panels(
                                    &mut artifact,
                                    compiled,
                                    runtime,
                                    &snapshot.trajectories,
                                    solver_config.constraint_tol,
                                ) {
                                    submit.submit(SolveStreamEvent::Log {
                                        line: format!(
                                            "[constraint violation report failed: {error}]"
                                        ),
                                        level: SolveLogLevel::Info,
                                    });
                                }
                                submit.submit(SolveStreamEvent::Iteration { progress, artifact });
                            }
                            Err(error) => submit.submit(SolveStreamEvent::Log {
                                line: format!("[iteration visualization failed: {error}]"),
                                level: SolveLogLevel::Info,
                            }),
                        },
                    )
                },
            ) {
                Ok(solved) => solved,
                Err(error) => {
                    if let Some(typed) = error.downcast_ref::<ClarabelSqpError>() {
                        emit_log_lines(&emit, sqp_failure_diagnostic_lines(typed));
                        if let Some(report) =
                            sqp_failure_solver_report(typed, &running_solver, &options)
                        {
                            emit_failure_status(&emit, solver_method, report);
                        }
                    }
                    return Err(error.into());
                }
            };
            let (x_arcs, u_arcs) =
                compiled.build_interval_arcs(&solved.trajectories, &runtime.parameters)?;
            let mut artifact = {
                let mut builder = build_artifact.lock().expect("artifact builder poisoned");
                (*builder)(&solved.trajectories, &x_arcs, &u_arcs)
            };
            let _ = try_attach_multiple_shooting_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            let artifact = with_solve_time(artifact, started);
            emit_event(
                &emit,
                SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                },
            );
            Ok(artifact)
        }
        SolverMethod::Nlip => {
            let options = nlip_options(solver_config);
            emit_event(
                &emit,
                SolveStreamEvent::Status {
                    status: SolveStatus {
                        stage: SolveStage::Solving,
                        solver_method: Some(solver_method),
                        solver: running_solver.clone().with_solve_seconds(0.0),
                    },
                },
            );
            let emit_for_worker = emit.clone();
            let solve_started = started;
            let running_solver_for_callback = running_solver.clone();
            let solved = match with_latest_only_worker(
                move |event| {
                    emit_event(&emit_for_worker, event);
                },
                |submit| {
                    let build_for_callback = build_artifact.clone();
                    let options_for_callback = options.clone();
                    compiled.run_nlip_with_callback(
                        runtime,
                        &options_for_callback,
                        move |snapshot| match compiled
                            .build_interval_arcs(&snapshot.trajectories, &runtime.parameters)
                        {
                            Ok((x_arcs, u_arcs)) => {
                                let mut builder = build_for_callback
                                    .lock()
                                    .expect("artifact builder poisoned");
                                let mut artifact =
                                    (*builder)(&snapshot.trajectories, &x_arcs, &u_arcs);
                                drop(builder);
                                let progress = nlip_progress(&snapshot.solver);
                                artifact.solver = running_solver_for_callback
                                    .clone()
                                    .with_iterations(progress.iteration)
                                    .with_solve_seconds(solve_started.elapsed().as_secs_f64());
                                if let Err(error) = try_attach_multiple_shooting_constraint_panels(
                                    &mut artifact,
                                    compiled,
                                    runtime,
                                    &snapshot.trajectories,
                                    solver_config.constraint_tol,
                                ) {
                                    submit.submit(SolveStreamEvent::Log {
                                        line: format!(
                                            "[constraint violation report failed: {error}]"
                                        ),
                                        level: SolveLogLevel::Info,
                                    });
                                }
                                submit.submit(SolveStreamEvent::Iteration { progress, artifact });
                            }
                            Err(error) => submit.submit(SolveStreamEvent::Log {
                                line: format!("[iteration visualization failed: {error}]"),
                                level: SolveLogLevel::Info,
                            }),
                        },
                    )
                },
            ) {
                Ok(solved) => solved,
                Err(error) => {
                    if let Some(typed) = error.downcast_ref::<InteriorPointSolveError>() {
                        emit_log_lines(&emit, nlip_failure_diagnostic_lines(typed));
                        if let Some(report) =
                            nlip_failure_solver_report(typed, &running_solver, &options)
                        {
                            emit_failure_status(&emit, solver_method, report);
                        }
                    }
                    return Err(error.into());
                }
            };
            let (x_arcs, u_arcs) =
                compiled.build_interval_arcs(&solved.trajectories, &runtime.parameters)?;
            let mut artifact = {
                let mut builder = build_artifact.lock().expect("artifact builder poisoned");
                (*builder)(&solved.trajectories, &x_arcs, &u_arcs)
            };
            let _ = try_attach_multiple_shooting_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            let artifact = with_solve_time(artifact, started);
            emit_event(
                &emit,
                SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                },
            );
            Ok(artifact)
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let options = ipopt_options(solver_config);
            emit_event(
                &emit,
                SolveStreamEvent::Status {
                    status: SolveStatus {
                        stage: SolveStage::Solving,
                        solver_method: Some(solver_method),
                        solver: running_solver.clone().with_solve_seconds(0.0),
                    },
                },
            );
            let emit_for_worker = emit.clone();
            let solve_started = started;
            let running_solver_for_callback = running_solver.clone();
            let solved = match with_latest_only_worker(
                move |event| {
                    emit_event(&emit_for_worker, event);
                },
                |submit| {
                    let build_for_callback = build_artifact.clone();
                    compiled.run_ipopt_with_callback(
                        runtime,
                        &ipopt_options(solver_config),
                        move |snapshot| match compiled
                            .build_interval_arcs(&snapshot.trajectories, &runtime.parameters)
                        {
                            Ok((x_arcs, u_arcs)) => {
                                let mut builder = build_for_callback
                                    .lock()
                                    .expect("artifact builder poisoned");
                                let mut artifact =
                                    (*builder)(&snapshot.trajectories, &x_arcs, &u_arcs);
                                drop(builder);
                                let progress = ipopt_progress(&snapshot.solver);
                                artifact.solver = running_solver_for_callback
                                    .clone()
                                    .with_iterations(progress.iteration)
                                    .with_solve_seconds(solve_started.elapsed().as_secs_f64());
                                if let Err(error) = try_attach_multiple_shooting_constraint_panels(
                                    &mut artifact,
                                    compiled,
                                    runtime,
                                    &snapshot.trajectories,
                                    solver_config.constraint_tol,
                                ) {
                                    submit.submit(SolveStreamEvent::Log {
                                        line: format!(
                                            "[constraint violation report failed: {error}]"
                                        ),
                                        level: SolveLogLevel::Info,
                                    });
                                }
                                submit.submit(SolveStreamEvent::Iteration { progress, artifact });
                            }
                            Err(error) => submit.submit(SolveStreamEvent::Log {
                                line: format!("[iteration visualization failed: {error}]"),
                                level: SolveLogLevel::Info,
                            }),
                        },
                    )
                },
            ) {
                Ok(solved) => solved,
                Err(error) => {
                    if let Some(report) =
                        error.downcast_ref::<IpoptSolveError>().and_then(|typed| {
                            ipopt_failure_solver_report(typed, &running_solver, &options)
                        })
                    {
                        emit_failure_status(&emit, solver_method, report);
                    }
                    return Err(error.into());
                }
            };
            let (x_arcs, u_arcs) =
                compiled.build_interval_arcs(&solved.trajectories, &runtime.parameters)?;
            let mut artifact = {
                let mut builder = build_artifact.lock().expect("artifact builder poisoned");
                (*builder)(&solved.trajectories, &x_arcs, &u_arcs)
            };
            let _ = try_attach_multiple_shooting_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_ipopt_termination_metric(&mut artifact, &solved.solver);
            attach_ipopt_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            let artifact = with_solve_time(artifact, started);
            emit_event(
                &emit,
                SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                },
            );
            Ok(artifact)
        }
    }
}

pub fn solve_cached_multiple_shooting_problem_with_progress<Compiled, Emit, Build>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum>,
        &[IntervalArc<Compiled::XNum>],
        &[IntervalArc<Compiled::UNum>],
    ) -> SolveArtifact,
{
    let compiled = compiled.borrow();
    solve_multiple_shooting_problem_with_progress(
        &*compiled,
        runtime,
        solver_method,
        solver_config,
        emit,
        running_solver,
        build_artifact,
    )
}

pub fn solve_direct_collocation_problem<Compiled, Build>(
    compiled: &Compiled,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    mut build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum>,
        &DirectCollocationTimeGrid,
    ) -> SolveArtifact,
{
    let started = Instant::now();
    match solver_method {
        SolverMethod::Sqp => {
            let options = sqp_options(solver_config);
            let solved = compiled.run_sqp(runtime, &options)?;
            let mut artifact = build_artifact(&solved.trajectories, &solved.time_grid);
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            Ok(with_solve_time(artifact, started))
        }
        SolverMethod::Nlip => {
            let options = nlip_options(solver_config);
            let solved = compiled.run_nlip(runtime, &options)?;
            let mut artifact = build_artifact(&solved.trajectories, &solved.time_grid);
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            Ok(with_solve_time(artifact, started))
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let options = ipopt_options(solver_config);
            let solved = compiled.run_ipopt(runtime, &options)?;
            let mut artifact = build_artifact(&solved.trajectories, &solved.time_grid);
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_ipopt_termination_metric(&mut artifact, &solved.solver);
            attach_ipopt_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            Ok(with_solve_time(artifact, started))
        }
    }
}

pub fn solve_cached_direct_collocation_problem<Compiled, Build>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum>,
        &DirectCollocationTimeGrid,
    ) -> SolveArtifact,
{
    let compiled = compiled.borrow();
    solve_direct_collocation_problem(
        &*compiled,
        runtime,
        solver_method,
        solver_config,
        build_artifact,
    )
}

pub fn solve_direct_collocation_problem_with_progress<Compiled, Emit, Build>(
    compiled: &Compiled,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum>,
        &DirectCollocationTimeGrid,
    ) -> SolveArtifact,
{
    let started = Instant::now();
    let emit = Arc::new(Mutex::new(emit));
    let build_artifact = Arc::new(Mutex::new(build_artifact));
    match solver_method {
        SolverMethod::Sqp => {
            let options = sqp_options(solver_config);
            emit_event(
                &emit,
                SolveStreamEvent::Status {
                    status: SolveStatus {
                        stage: SolveStage::Solving,
                        solver_method: Some(solver_method),
                        solver: running_solver.clone().with_solve_seconds(0.0),
                    },
                },
            );
            let emit_for_worker = emit.clone();
            let solve_started = started;
            let running_solver_for_callback = running_solver.clone();
            let solved = match with_latest_only_worker(
                move |event| {
                    emit_event(&emit_for_worker, event);
                },
                |submit| {
                    let build_for_callback = build_artifact.clone();
                    let options_for_callback = options.clone();
                    compiled.run_sqp_with_callback(
                        runtime,
                        &options_for_callback,
                        move |snapshot| {
                            let mut artifact = {
                                let mut builder = build_for_callback
                                    .lock()
                                    .expect("artifact builder poisoned");
                                (*builder)(&snapshot.trajectories, &snapshot.time_grid)
                            };
                            let progress = sqp_progress(&snapshot.solver);
                            artifact.solver = running_solver_for_callback
                                .clone()
                                .with_iterations(progress.iteration)
                                .with_solve_seconds(solve_started.elapsed().as_secs_f64());
                            if let Err(error) = try_attach_direct_collocation_constraint_panels(
                                &mut artifact,
                                compiled,
                                runtime,
                                &snapshot.trajectories,
                                solver_config.constraint_tol,
                            ) {
                                submit.submit(SolveStreamEvent::Log {
                                    line: format!("[constraint violation report failed: {error}]"),
                                    level: SolveLogLevel::Info,
                                });
                            }
                            submit.submit(SolveStreamEvent::Iteration { progress, artifact });
                        },
                    )
                },
            ) {
                Ok(solved) => solved,
                Err(error) => {
                    if let Some(typed) = error.downcast_ref::<ClarabelSqpError>() {
                        emit_log_lines(&emit, sqp_failure_diagnostic_lines(typed));
                        if let Some(report) =
                            sqp_failure_solver_report(typed, &running_solver, &options)
                        {
                            emit_failure_status(&emit, solver_method, report);
                        }
                    }
                    return Err(error.into());
                }
            };
            let mut artifact = {
                let mut builder = build_artifact.lock().expect("artifact builder poisoned");
                (*builder)(&solved.trajectories, &solved.time_grid)
            };
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            let artifact = with_solve_time(artifact, started);
            emit_event(
                &emit,
                SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                },
            );
            Ok(artifact)
        }
        SolverMethod::Nlip => {
            let options = nlip_options(solver_config);
            emit_event(
                &emit,
                SolveStreamEvent::Status {
                    status: SolveStatus {
                        stage: SolveStage::Solving,
                        solver_method: Some(solver_method),
                        solver: running_solver.clone().with_solve_seconds(0.0),
                    },
                },
            );
            let emit_for_worker = emit.clone();
            let solve_started = started;
            let running_solver_for_callback = running_solver.clone();
            let solved = match with_latest_only_worker(
                move |event| {
                    emit_event(&emit_for_worker, event);
                },
                |submit| {
                    let build_for_callback = build_artifact.clone();
                    let options_for_callback = options.clone();
                    compiled.run_nlip_with_callback(
                        runtime,
                        &options_for_callback,
                        move |snapshot| {
                            let mut artifact = {
                                let mut builder = build_for_callback
                                    .lock()
                                    .expect("artifact builder poisoned");
                                (*builder)(&snapshot.trajectories, &snapshot.time_grid)
                            };
                            let progress = nlip_progress(&snapshot.solver);
                            artifact.solver = running_solver_for_callback
                                .clone()
                                .with_iterations(progress.iteration)
                                .with_solve_seconds(solve_started.elapsed().as_secs_f64());
                            if let Err(error) = try_attach_direct_collocation_constraint_panels(
                                &mut artifact,
                                compiled,
                                runtime,
                                &snapshot.trajectories,
                                solver_config.constraint_tol,
                            ) {
                                submit.submit(SolveStreamEvent::Log {
                                    line: format!("[constraint violation report failed: {error}]"),
                                    level: SolveLogLevel::Info,
                                });
                            }
                            submit.submit(SolveStreamEvent::Iteration { progress, artifact });
                        },
                    )
                },
            ) {
                Ok(solved) => solved,
                Err(error) => {
                    if let Some(typed) = error.downcast_ref::<InteriorPointSolveError>() {
                        emit_log_lines(&emit, nlip_failure_diagnostic_lines(typed));
                        if let Some(report) =
                            nlip_failure_solver_report(typed, &running_solver, &options)
                        {
                            emit_failure_status(&emit, solver_method, report);
                        }
                    }
                    return Err(error.into());
                }
            };
            let mut artifact = {
                let mut builder = build_artifact.lock().expect("artifact builder poisoned");
                (*builder)(&solved.trajectories, &solved.time_grid)
            };
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            let artifact = with_solve_time(artifact, started);
            emit_event(
                &emit,
                SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                },
            );
            Ok(artifact)
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let options = ipopt_options(solver_config);
            emit_event(
                &emit,
                SolveStreamEvent::Status {
                    status: SolveStatus {
                        stage: SolveStage::Solving,
                        solver_method: Some(solver_method),
                        solver: running_solver.clone().with_solve_seconds(0.0),
                    },
                },
            );
            let emit_for_worker = emit.clone();
            let solve_started = started;
            let running_solver_for_callback = running_solver.clone();
            let solved = match with_latest_only_worker(
                move |event| {
                    emit_event(&emit_for_worker, event);
                },
                |submit| {
                    let build_for_callback = build_artifact.clone();
                    let options_for_callback = options.clone();
                    compiled.run_ipopt_with_callback(
                        runtime,
                        &options_for_callback,
                        move |snapshot| {
                            let mut artifact = {
                                let mut builder = build_for_callback
                                    .lock()
                                    .expect("artifact builder poisoned");
                                (*builder)(&snapshot.trajectories, &snapshot.time_grid)
                            };
                            let progress = ipopt_progress(&snapshot.solver);
                            artifact.solver = running_solver_for_callback
                                .clone()
                                .with_iterations(progress.iteration)
                                .with_solve_seconds(solve_started.elapsed().as_secs_f64());
                            if let Err(error) = try_attach_direct_collocation_constraint_panels(
                                &mut artifact,
                                compiled,
                                runtime,
                                &snapshot.trajectories,
                                solver_config.constraint_tol,
                            ) {
                                submit.submit(SolveStreamEvent::Log {
                                    line: format!("[constraint violation report failed: {error}]"),
                                    level: SolveLogLevel::Info,
                                });
                            }
                            submit.submit(SolveStreamEvent::Iteration { progress, artifact });
                        },
                    )
                },
            ) {
                Ok(solved) => solved,
                Err(error) => {
                    if let Some(report) =
                        error.downcast_ref::<IpoptSolveError>().and_then(|typed| {
                            ipopt_failure_solver_report(typed, &running_solver, &options)
                        })
                    {
                        emit_failure_status(&emit, solver_method, report);
                    }
                    return Err(error.into());
                }
            };
            let mut artifact = {
                let mut builder = build_artifact.lock().expect("artifact builder poisoned");
                (*builder)(&solved.trajectories, &solved.time_grid)
            };
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_ipopt_termination_metric(&mut artifact, &solved.solver);
            attach_ipopt_solver_report(&mut artifact, &solved.solver, &options);
            append_ocp_setup_timing_details(&mut artifact.solver, solved.setup_timing);
            let artifact = with_solve_time(artifact, started);
            emit_event(
                &emit,
                SolveStreamEvent::Final {
                    artifact: artifact.clone(),
                },
            );
            Ok(artifact)
        }
    }
}

pub fn solve_cached_direct_collocation_problem_with_progress<Compiled, Emit, Build>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
    >,
    solver_method: SolverMethod,
    solver_config: &SolverConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum>,
        &DirectCollocationTimeGrid,
    ) -> SolveArtifact,
{
    let compiled = compiled.borrow();
    solve_direct_collocation_problem_with_progress(
        &*compiled,
        runtime,
        solver_method,
        solver_config,
        emit,
        running_solver,
        build_artifact,
    )
}

pub fn sqp_progress(snapshot: &SqpIterationSnapshot) -> SolveProgress {
    SolveProgress {
        iteration: snapshot.iteration,
        phase: match snapshot.phase {
            SqpIterationPhase::Initial => SolvePhase::Initial,
            SqpIterationPhase::AcceptedStep => SolvePhase::AcceptedStep,
            SqpIterationPhase::PostConvergence => SolvePhase::PostConvergence,
        },
        objective: snapshot.objective,
        eq_inf: snapshot.eq_inf,
        ineq_inf: snapshot.ineq_inf,
        dual_inf: snapshot.dual_inf,
        step_inf: snapshot.step_inf,
        penalty: snapshot.penalty,
        alpha: snapshot
            .line_search
            .as_ref()
            .map(|info| info.accepted_alpha),
        line_search_iterations: snapshot
            .line_search
            .as_ref()
            .map(|info| info.backtrack_count),
        filter: snapshot.filter.as_ref().map(|filter| SolveFilterInfo {
            current: SolveFilterPoint {
                objective: filter.current.objective,
                violation: filter.current.violation,
            },
            entries: filter
                .entries
                .iter()
                .map(|entry| SolveFilterPoint {
                    objective: entry.objective,
                    violation: entry.violation,
                })
                .collect(),
            objective_label: "Objective",
            title: "SQP filter frontier",
            accepted_mode: filter.accepted_mode.map(|mode| match mode {
                FilterAcceptanceMode::ObjectiveArmijo => SolveFilterAcceptanceMode::ObjectiveArmijo,
                FilterAcceptanceMode::ViolationReduction => {
                    SolveFilterAcceptanceMode::ViolationReduction
                }
            }),
        }),
        trust_region: snapshot
            .trust_region
            .as_ref()
            .map(|trust_region| SolveTrustRegionInfo {
                radius: trust_region.radius,
                attempted_radius: trust_region.attempted_radius,
                step_norm: trust_region.step_norm,
                largest_attempted_step_norm: trust_region
                    .rejected_trials
                    .iter()
                    .map(|trial| trial.step_norm)
                    .chain(std::iter::once(trust_region.step_norm))
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .reduce(f64::max),
                contraction_count: trust_region.contraction_count,
                qp_failure_retries: trust_region.qp_failure_retries,
                boundary_active: trust_region.boundary_active,
                restoration_attempted: trust_region.restoration_attempted,
                elastic_recovery_attempted: trust_region.elastic_recovery_attempted,
            }),
    }
}

pub fn nlip_progress(snapshot: &InteriorPointIterationSnapshot) -> SolveProgress {
    SolveProgress {
        iteration: snapshot.iteration,
        phase: match snapshot.phase {
            optimization::InteriorPointIterationPhase::Initial => SolvePhase::Initial,
            optimization::InteriorPointIterationPhase::AcceptedStep => SolvePhase::AcceptedStep,
            optimization::InteriorPointIterationPhase::Restoration => SolvePhase::Restoration,
            optimization::InteriorPointIterationPhase::Converged => SolvePhase::Converged,
        },
        objective: snapshot.objective,
        eq_inf: snapshot.eq_inf,
        ineq_inf: snapshot.ineq_inf,
        dual_inf: snapshot.dual_inf,
        step_inf: snapshot.step_inf,
        penalty: snapshot.barrier_parameter.unwrap_or(0.0),
        alpha: snapshot.alpha,
        line_search_iterations: snapshot.line_search_iterations.map(|value| value as usize),
        filter: snapshot.filter.as_ref().map(|filter| SolveFilterInfo {
            current: SolveFilterPoint {
                objective: filter.current.objective,
                violation: filter.current.violation,
            },
            entries: filter
                .entries
                .iter()
                .map(|entry| SolveFilterPoint {
                    objective: entry.objective,
                    violation: entry.violation,
                })
                .collect(),
            objective_label: "Objective",
            title: "NLIP filter frontier",
            accepted_mode: filter.accepted_mode.map(|mode| match mode {
                FilterAcceptanceMode::ObjectiveArmijo => SolveFilterAcceptanceMode::ObjectiveArmijo,
                FilterAcceptanceMode::ViolationReduction => {
                    SolveFilterAcceptanceMode::ViolationReduction
                }
            }),
        }),
        trust_region: None,
    }
}

#[cfg(feature = "ipopt")]
#[allow(dead_code)]
pub fn ipopt_progress(snapshot: &optimization::IpoptIterationSnapshot) -> SolveProgress {
    SolveProgress {
        iteration: snapshot.iteration,
        phase: match snapshot.phase {
            optimization::IpoptIterationPhase::Regular => SolvePhase::Regular,
            optimization::IpoptIterationPhase::Restoration => SolvePhase::Restoration,
        },
        objective: snapshot.objective,
        eq_inf: Some(snapshot.primal_inf),
        ineq_inf: None,
        dual_inf: snapshot.dual_inf,
        step_inf: Some(snapshot.step_inf),
        penalty: snapshot.barrier_parameter,
        alpha: Some(snapshot.alpha_pr),
        line_search_iterations: Some(snapshot.line_search_trials as usize),
        filter: None,
        trust_region: None,
    }
}

pub fn sqp_termination_label(summary: &ClarabelSqpSummary) -> String {
    let reduced_accuracy = summary
        .final_state
        .events
        .contains(&SqpIterationEvent::QpReducedAccuracy)
        || summary
            .last_accepted_state
            .as_ref()
            .is_some_and(|state| state.events.contains(&SqpIterationEvent::QpReducedAccuracy));
    match (summary.termination, reduced_accuracy) {
        (optimization::SqpTermination::Converged, true) => {
            "Converged with reduced QP accuracy".to_string()
        }
        (optimization::SqpTermination::Converged, false) => "Converged".to_string(),
        (optimization::SqpTermination::MaxIterations, _) => {
            "Failed: maximum iterations reached".to_string()
        }
        (optimization::SqpTermination::QpSolve, _) => "Failed: QP solve".to_string(),
        (optimization::SqpTermination::LineSearchFailed, _) => "Failed: line search".to_string(),
        (optimization::SqpTermination::RestorationFailed, _) => "Failed: restoration".to_string(),
        (optimization::SqpTermination::Stalled, _) => "Failed: stalled".to_string(),
        (optimization::SqpTermination::NonFiniteInput, _) => "Failed: non-finite input".to_string(),
        (optimization::SqpTermination::NonFiniteCallbackOutput, _) => {
            "Failed: non-finite callback output".to_string()
        }
    }
}

pub fn sqp_status_kind(summary: &ClarabelSqpSummary) -> SolverStatusKind {
    let reduced_accuracy = summary
        .final_state
        .events
        .contains(&SqpIterationEvent::QpReducedAccuracy)
        || summary
            .last_accepted_state
            .as_ref()
            .is_some_and(|state| state.events.contains(&SqpIterationEvent::QpReducedAccuracy));
    match (summary.termination, reduced_accuracy) {
        (optimization::SqpTermination::Converged, false) => SolverStatusKind::Success,
        (optimization::SqpTermination::Converged, true) => SolverStatusKind::Warning,
        _ => SolverStatusKind::Error,
    }
}

pub fn nlip_termination_label(summary: &InteriorPointSummary) -> String {
    match summary.termination {
        optimization::InteriorPointTermination::Converged => "Converged".to_string(),
        optimization::InteriorPointTermination::Acceptable => {
            "Converged to acceptable level".to_string()
        }
        optimization::InteriorPointTermination::FeasiblePointFound => {
            "Feasible point found".to_string()
        }
    }
}

pub fn nlip_status_kind(summary: &InteriorPointSummary) -> SolverStatusKind {
    match summary.status_kind {
        optimization::InteriorPointStatusKind::Success => SolverStatusKind::Success,
        optimization::InteriorPointStatusKind::Warning => SolverStatusKind::Warning,
    }
}

#[cfg(feature = "ipopt")]
pub fn ipopt_status_label(status: IpoptRawStatus) -> String {
    match status {
        IpoptRawStatus::SolveSucceeded => "Converged".to_string(),
        IpoptRawStatus::SolvedToAcceptableLevel => "Converged to acceptable level".to_string(),
        IpoptRawStatus::FeasiblePointFound => "Feasible point found".to_string(),
        other => format!("Failed: {other:?}"),
    }
}

#[cfg(feature = "ipopt")]
pub fn ipopt_status_kind(status: IpoptRawStatus) -> SolverStatusKind {
    match status {
        IpoptRawStatus::SolveSucceeded => SolverStatusKind::Success,
        IpoptRawStatus::SolvedToAcceptableLevel | IpoptRawStatus::FeasiblePointFound => {
            SolverStatusKind::Warning
        }
        _ => SolverStatusKind::Error,
    }
}

pub fn sqp_solver_report(
    summary: &ClarabelSqpSummary,
    options: &ClarabelSqpOptions,
) -> SolverReport {
    SolverReport {
        completed: true,
        status_label: sqp_termination_label(summary),
        status_kind: sqp_status_kind(summary),
        iterations: Some(summary.iterations),
        symbolic_setup_s: symbolic_setup_seconds(summary.profiling.backend_timing),
        jit_s: duration_seconds(summary.profiling.backend_timing.jit_time),
        solve_s: Some(summary.profiling.total_time.as_secs_f64()),
        compile_cached: false,
        jit_disk_cache_hit: false,
        phase_details: SolverPhaseDetails {
            solve: prepend_solver_settings_detail(
                sqp_solve_phase_details(&summary.profiling),
                format_sqp_settings_summary(options),
            ),
            ..SolverPhaseDetails::default()
        },
    }
}

pub fn nlip_solver_report(
    summary: &InteriorPointSummary,
    options: &InteriorPointOptions,
) -> SolverReport {
    SolverReport {
        completed: true,
        status_label: nlip_termination_label(summary),
        status_kind: nlip_status_kind(summary),
        iterations: Some(summary.iterations),
        symbolic_setup_s: symbolic_setup_seconds(summary.profiling.backend_timing),
        jit_s: duration_seconds(summary.profiling.backend_timing.jit_time),
        solve_s: Some(summary.profiling.total_time.as_secs_f64()),
        compile_cached: false,
        jit_disk_cache_hit: false,
        phase_details: SolverPhaseDetails {
            solve: prepend_solver_settings_detail(
                nlip_solve_phase_details(&summary.profiling),
                format_nlip_settings_summary(options),
            ),
            ..SolverPhaseDetails::default()
        },
    }
}

#[cfg(feature = "ipopt")]
pub fn ipopt_solver_report(summary: &IpoptSummary, options: &IpoptOptions) -> SolverReport {
    SolverReport {
        completed: true,
        status_label: ipopt_status_label(summary.status),
        status_kind: ipopt_status_kind(summary.status),
        iterations: Some(summary.iterations),
        symbolic_setup_s: symbolic_setup_seconds(summary.profiling.backend_timing),
        jit_s: duration_seconds(summary.profiling.backend_timing.jit_time),
        solve_s: Some(summary.profiling.total_time.as_secs_f64()),
        compile_cached: false,
        jit_disk_cache_hit: false,
        phase_details: SolverPhaseDetails {
            solve: prepend_solver_settings_detail(
                ipopt_solve_phase_details(&summary.profiling),
                format_ipopt_settings_summary(options),
            ),
            ..SolverPhaseDetails::default()
        },
    }
}

pub fn attach_sqp_solver_report(
    artifact: &mut SolveArtifact,
    summary: &ClarabelSqpSummary,
    options: &ClarabelSqpOptions,
) {
    artifact.solver = sqp_solver_report(summary, options);
}

pub fn attach_nlip_solver_report(
    artifact: &mut SolveArtifact,
    summary: &InteriorPointSummary,
    options: &InteriorPointOptions,
) {
    artifact.solver = nlip_solver_report(summary, options);
}

#[cfg(feature = "ipopt")]
pub fn attach_ipopt_solver_report(
    artifact: &mut SolveArtifact,
    summary: &IpoptSummary,
    options: &IpoptOptions,
) {
    artifact.solver = ipopt_solver_report(summary, options);
}

fn push_eval_timing_detail(
    details: &mut Vec<SolverPhaseDetail>,
    label: &str,
    calls: usize,
    total_time: Duration,
) {
    if calls > 0 {
        details.push(SolverPhaseDetail {
            label: label.to_string(),
            value: format_phase_duration(total_time),
            count: calls,
        });
    }
}

fn push_optional_timing_detail(
    details: &mut Vec<SolverPhaseDetail>,
    label: &str,
    count: usize,
    duration: Duration,
) {
    if duration > Duration::ZERO {
        details.push(phase_detail(label, format_phase_duration(duration), count));
    }
}

fn push_adapter_timing_details(
    details: &mut Vec<SolverPhaseDetail>,
    timing: Option<optimization::SolverAdapterTiming>,
) {
    if let Some(timing) = timing {
        push_optional_timing_detail(details, "Adapter Callback", 0, timing.callback_evaluation);
        push_optional_timing_detail(details, "Adapter IO", 0, timing.output_marshalling);
        push_optional_timing_detail(details, "Layout Projection", 0, timing.layout_projection);
    }
}

fn adapter_total_time(timing: Option<optimization::SolverAdapterTiming>) -> Duration {
    timing.map_or(Duration::ZERO, |timing| {
        timing.callback_evaluation + timing.output_marshalling + timing.layout_projection
    })
}

fn nlip_linear_solve_other_time(profiling: &InteriorPointProfiling) -> Duration {
    profiling.linear_solve_time.saturating_sub(
        profiling.linear_rhs_assembly_time
            + profiling.linear_kkt_value_assembly_time
            + profiling.sparse_symbolic_analysis_time
            + profiling.sparse_numeric_factorization_time
            + profiling.sparse_numeric_refactorization_time
            + profiling.linear_backsolve_time,
    )
}

fn sqp_evaluation_other_time(profiling: &ClarabelSqpProfiling) -> Duration {
    let callback_total = profiling.objective_value.total_time
        + profiling.objective_gradient.total_time
        + profiling.equality_values.total_time
        + profiling.inequality_values.total_time
        + profiling.equality_jacobian_values.total_time
        + profiling.inequality_jacobian_values.total_time
        + profiling.lagrangian_hessian_values.total_time;
    profiling.evaluation_time.saturating_sub(callback_total)
}

fn sqp_preprocess_other_time(profiling: &ClarabelSqpProfiling) -> Duration {
    profiling.preprocessing_time.saturating_sub(
        profiling.jacobian_assembly_time
            + profiling.hessian_assembly_time
            + profiling.regularization_time
            + profiling.subproblem_assembly_time,
    )
}

fn sqp_subproblem_solve_other_time(profiling: &ClarabelSqpProfiling) -> Duration {
    profiling.subproblem_solve_time.saturating_sub(
        profiling.qp_setup_time + profiling.qp_solve_time + profiling.multiplier_estimation_time,
    )
}

fn sqp_line_search_other_time(profiling: &ClarabelSqpProfiling) -> Duration {
    profiling.line_search_time.saturating_sub(
        profiling.line_search_evaluation_time + profiling.line_search_condition_check_time,
    )
}

fn sqp_convergence_other_time(profiling: &ClarabelSqpProfiling) -> Duration {
    profiling
        .convergence_time
        .saturating_sub(profiling.convergence_check_time)
}

fn sqp_solve_phase_details(profiling: &ClarabelSqpProfiling) -> Vec<SolverPhaseDetail> {
    let mut details = Vec::new();
    push_optional_timing_detail(
        &mut details,
        "Evaluate Functions",
        0,
        profiling.evaluation_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Objective",
        profiling.objective_value.calls,
        profiling.objective_value.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Gradient",
        profiling.objective_gradient.calls,
        profiling.objective_gradient.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Equality Values",
        profiling.equality_values.calls,
        profiling.equality_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Inequality Values",
        profiling.inequality_values.calls,
        profiling.inequality_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Equality Jacobian",
        profiling.equality_jacobian_values.calls,
        profiling.equality_jacobian_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Inequality Jacobian",
        profiling.inequality_jacobian_values.calls,
        profiling.inequality_jacobian_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Hessian",
        profiling.lagrangian_hessian_values.calls,
        profiling.lagrangian_hessian_values.total_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Evaluate Other",
        0,
        sqp_evaluation_other_time(profiling),
    );
    push_optional_timing_detail(
        &mut details,
        "Adapter / Runtime Plumbing",
        0,
        adapter_total_time(profiling.adapter_timing),
    );
    push_adapter_timing_details(&mut details, profiling.adapter_timing);
    push_optional_timing_detail(&mut details, "Preprocess", 0, profiling.preprocessing_time);
    push_optional_timing_detail(
        &mut details,
        "Jacobian Build",
        profiling.jacobian_assembly_steps,
        profiling.jacobian_assembly_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Hessian Build",
        profiling.hessian_assembly_steps,
        profiling.hessian_assembly_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Regularization",
        profiling.regularization_steps,
        profiling.regularization_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Subproblem Assembly",
        profiling.subproblem_assembly_steps,
        profiling.subproblem_assembly_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Preprocess Other",
        0,
        sqp_preprocess_other_time(profiling),
    );
    push_optional_timing_detail(
        &mut details,
        "Subproblem Solve",
        0,
        profiling.subproblem_solve_time,
    );
    push_optional_timing_detail(
        &mut details,
        "QP Setup",
        profiling.qp_setups,
        profiling.qp_setup_time,
    );
    push_optional_timing_detail(
        &mut details,
        "QP Solve",
        profiling.qp_solves,
        profiling.qp_solve_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Multiplier Estimation",
        profiling.multiplier_estimations,
        profiling.multiplier_estimation_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Subproblem Solve Other",
        0,
        sqp_subproblem_solve_other_time(profiling),
    );
    push_optional_timing_detail(&mut details, "Line Search", 0, profiling.line_search_time);
    push_optional_timing_detail(
        &mut details,
        "Line Search Eval",
        profiling.line_search_evaluations,
        profiling.line_search_evaluation_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Line Search Check",
        profiling.line_search_condition_checks,
        profiling.line_search_condition_check_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Line Search Other",
        0,
        sqp_line_search_other_time(profiling),
    );
    push_optional_timing_detail(&mut details, "Convergence", 0, profiling.convergence_time);
    push_optional_timing_detail(
        &mut details,
        "Convergence Check",
        profiling.convergence_checks,
        profiling.convergence_check_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Convergence Other",
        0,
        sqp_convergence_other_time(profiling),
    );
    push_optional_timing_detail(&mut details, "Unaccounted", 0, profiling.unaccounted_time);
    push_optional_timing_detail(&mut details, "Total", 0, profiling.total_time);
    details
}

fn nlip_solve_phase_details(profiling: &InteriorPointProfiling) -> Vec<SolverPhaseDetail> {
    let mut details = Vec::new();
    push_eval_timing_detail(
        &mut details,
        "Objective",
        profiling.objective_value.calls,
        profiling.objective_value.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Gradient",
        profiling.objective_gradient.calls,
        profiling.objective_gradient.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Equality Values",
        profiling.equality_values.calls,
        profiling.equality_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Inequality Values",
        profiling.inequality_values.calls,
        profiling.inequality_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Equality Jacobian",
        profiling.equality_jacobian_values.calls,
        profiling.equality_jacobian_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Inequality Jacobian",
        profiling.inequality_jacobian_values.calls,
        profiling.inequality_jacobian_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Hessian",
        profiling.lagrangian_hessian_values.calls,
        profiling.lagrangian_hessian_values.total_time,
    );
    push_adapter_timing_details(&mut details, profiling.adapter_timing);
    push_optional_timing_detail(
        &mut details,
        "Preprocess",
        profiling.preprocessing_steps,
        profiling.preprocessing_time,
    );
    push_optional_timing_detail(
        &mut details,
        "KKT Assembly",
        profiling.kkt_assemblies,
        profiling.kkt_assembly_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear Solve",
        profiling.linear_solves,
        profiling.linear_solve_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear RHS Assembly",
        profiling.linear_rhs_assemblies,
        profiling.linear_rhs_assembly_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear KKT Values",
        profiling.linear_kkt_value_assemblies,
        profiling.linear_kkt_value_assembly_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear Symbolic Analysis",
        profiling.sparse_symbolic_analyses,
        profiling.sparse_symbolic_analysis_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear Numeric Factorization",
        profiling.sparse_numeric_factorizations,
        profiling.sparse_numeric_factorization_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear Numeric Refactorization",
        profiling.sparse_numeric_refactorizations,
        profiling.sparse_numeric_refactorization_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear Backsolve / Refinement",
        profiling.linear_backsolves,
        profiling.linear_backsolve_time,
    );
    push_optional_timing_detail(
        &mut details,
        "Linear Solve Other",
        0,
        nlip_linear_solve_other_time(profiling),
    );
    push_optional_timing_detail(&mut details, "Unaccounted", 0, profiling.unaccounted_time);
    push_optional_timing_detail(&mut details, "Total", 0, profiling.total_time);
    details
}

#[cfg(feature = "ipopt")]
fn ipopt_solve_phase_details(profiling: &IpoptProfiling) -> Vec<SolverPhaseDetail> {
    let mut details = Vec::new();
    push_eval_timing_detail(
        &mut details,
        "Objective",
        profiling.objective_value.calls,
        profiling.objective_value.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Gradient",
        profiling.objective_gradient.calls,
        profiling.objective_gradient.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Constraints",
        profiling.constraint_values.calls,
        profiling.constraint_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Jacobian",
        profiling.constraint_jacobian_values.calls,
        profiling.constraint_jacobian_values.total_time,
    );
    push_eval_timing_detail(
        &mut details,
        "Hessian",
        profiling.hessian_values.calls,
        profiling.hessian_values.total_time,
    );
    push_adapter_timing_details(&mut details, profiling.adapter_timing);
    push_optional_timing_detail(&mut details, "Total", 0, profiling.total_time);
    details
}

fn merge_failure_solver_report(
    mut report: SolverReport,
    fallback: &SolverReport,
    solve_details: Vec<SolverPhaseDetail>,
) -> SolverReport {
    let mut phase_details = fallback.phase_details.clone();
    phase_details.solve = solve_details;
    report.symbolic_setup_s = report.symbolic_setup_s.or(fallback.symbolic_setup_s);
    report.jit_s = report.jit_s.or(fallback.jit_s);
    report.solve_s = report.solve_s.or(fallback.solve_s);
    report.compile_cached = report.compile_cached || fallback.compile_cached;
    report.jit_disk_cache_hit = report.jit_disk_cache_hit || fallback.jit_disk_cache_hit;
    report.phase_details = phase_details;
    report
}

pub fn sqp_failure_solver_report(
    error: &ClarabelSqpError,
    fallback: &SolverReport,
    options: &ClarabelSqpOptions,
) -> Option<SolverReport> {
    let context = match error {
        ClarabelSqpError::MaxIterations { context, .. }
        | ClarabelSqpError::QpSolve { context, .. }
        | ClarabelSqpError::UnconstrainedStepSolve { context }
        | ClarabelSqpError::LineSearchFailed { context, .. }
        | ClarabelSqpError::RestorationFailed { context, .. }
        | ClarabelSqpError::Stalled { context, .. }
        | ClarabelSqpError::NonFiniteCallbackOutput { context, .. } => context.as_ref(),
        ClarabelSqpError::InvalidInput(_)
        | ClarabelSqpError::NonFiniteInput { .. }
        | ClarabelSqpError::Setup(_) => return None,
    };
    Some(merge_failure_solver_report(
        SolverReport {
            completed: true,
            status_label: match context.termination {
                optimization::SqpTermination::MaxIterations => {
                    "Failed: maximum iterations reached".to_string()
                }
                optimization::SqpTermination::QpSolve => "Failed: QP solve".to_string(),
                optimization::SqpTermination::LineSearchFailed => "Failed: line search".to_string(),
                optimization::SqpTermination::RestorationFailed => {
                    "Failed: restoration".to_string()
                }
                optimization::SqpTermination::Stalled => "Failed: stalled".to_string(),
                optimization::SqpTermination::NonFiniteInput => {
                    "Failed: non-finite input".to_string()
                }
                optimization::SqpTermination::NonFiniteCallbackOutput => {
                    "Failed: non-finite callback output".to_string()
                }
                optimization::SqpTermination::Converged => "Failed".to_string(),
            },
            status_kind: SolverStatusKind::Error,
            iterations: context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            symbolic_setup_s: symbolic_setup_seconds(context.profiling.backend_timing),
            jit_s: duration_seconds(context.profiling.backend_timing.jit_time),
            solve_s: Some(context.profiling.total_time.as_secs_f64()),
            compile_cached: false,
            jit_disk_cache_hit: false,
            phase_details: SolverPhaseDetails::default(),
        },
        fallback,
        prepend_solver_settings_detail(
            sqp_solve_phase_details(&context.profiling),
            format_sqp_settings_summary(options),
        ),
    ))
}

pub fn nlip_failure_solver_report(
    error: &InteriorPointSolveError,
    fallback: &SolverReport,
    options: &InteriorPointOptions,
) -> Option<SolverReport> {
    let (status_label, iterations, profiling) = match error {
        InteriorPointSolveError::InvalidInput(_) => return None,
        InteriorPointSolveError::LinearSolve { solver, context } => (
            format!("Failed: linear solve ({})", solver.label()),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::LineSearchFailed { context, .. } => (
            "Failed: line search".to_string(),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::RestorationFailed { status, context } => (
            format!("Failed: restoration ({status:?})"),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::LocalInfeasibility { context } => (
            "Failed: local infeasibility".to_string(),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::DivergingIterates {
            max_abs_x,
            threshold,
            context,
        } => (
            format!(
                "Failed: diverging iterates (max |x| {}, limit {})",
                fmt_diag_sci(*max_abs_x),
                fmt_diag_sci(*threshold)
            ),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::WallTimeExceeded {
            elapsed_seconds,
            limit_seconds,
            context,
        } => (
            format!(
                "Failed: wall time exceeded (elapsed {}, limit {})",
                fmt_diag_sci(*elapsed_seconds),
                fmt_diag_sci(*limit_seconds)
            ),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::CpuTimeExceeded {
            elapsed_seconds,
            limit_seconds,
            context,
        } => (
            format!(
                "Failed: CPU time exceeded (elapsed {}, limit {})",
                fmt_diag_sci(*elapsed_seconds),
                fmt_diag_sci(*limit_seconds)
            ),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::UserRequestedStop { context } => (
            "Stopped: user requested stop".to_string(),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::SearchDirectionTooSmall { context } => (
            "Failed: search direction too small".to_string(),
            context
                .final_state
                .as_ref()
                .map(|state| state.iteration as usize),
            &context.profiling,
        ),
        InteriorPointSolveError::MaxIterations {
            iterations,
            context,
        } => (
            "Failed: maximum iterations reached".to_string(),
            Some(
                context
                    .final_state
                    .as_ref()
                    .map(|state| state.iteration as usize)
                    .unwrap_or(*iterations as usize),
            ),
            &context.profiling,
        ),
    };
    Some(merge_failure_solver_report(
        SolverReport {
            completed: true,
            status_label,
            status_kind: SolverStatusKind::Error,
            iterations,
            symbolic_setup_s: symbolic_setup_seconds(profiling.backend_timing),
            jit_s: duration_seconds(profiling.backend_timing.jit_time),
            solve_s: Some(profiling.total_time.as_secs_f64()),
            compile_cached: false,
            jit_disk_cache_hit: false,
            phase_details: SolverPhaseDetails::default(),
        },
        fallback,
        prepend_solver_settings_detail(
            nlip_solve_phase_details(profiling),
            format_nlip_settings_summary(options),
        ),
    ))
}

#[cfg(feature = "ipopt")]
pub fn ipopt_failure_solver_report(
    error: &IpoptSolveError,
    fallback: &SolverReport,
    options: &IpoptOptions,
) -> Option<SolverReport> {
    let (status_label, iterations, profiling) = match error {
        IpoptSolveError::Solve {
            status,
            iterations,
            profiling,
            ..
        } => (
            ipopt_status_label(*status),
            Some(*iterations as usize),
            profiling.as_ref(),
        ),
        IpoptSolveError::InvalidInput(_)
        | IpoptSolveError::Setup(_)
        | IpoptSolveError::OptionRejected { .. } => return None,
    };
    Some(merge_failure_solver_report(
        SolverReport {
            completed: true,
            status_label,
            status_kind: SolverStatusKind::Error,
            iterations,
            symbolic_setup_s: symbolic_setup_seconds(profiling.backend_timing),
            jit_s: duration_seconds(profiling.backend_timing.jit_time),
            solve_s: Some(profiling.total_time.as_secs_f64()),
            compile_cached: false,
            jit_disk_cache_hit: false,
            phase_details: SolverPhaseDetails::default(),
        },
        fallback,
        prepend_solver_settings_detail(
            ipopt_solve_phase_details(profiling),
            format_ipopt_settings_summary(options),
        ),
    ))
}

pub fn append_termination_metric(artifact: &mut SolveArtifact, summary: &ClarabelSqpSummary) {
    artifact.summary.push(metric_with_key(
        MetricKey::Termination,
        "Termination",
        sqp_termination_label(summary),
    ));
}

pub fn append_nlip_termination_metric(
    artifact: &mut SolveArtifact,
    summary: &InteriorPointSummary,
) {
    artifact.summary.push(metric_with_key(
        MetricKey::Termination,
        "Termination",
        nlip_termination_label(summary),
    ));
}

#[cfg(feature = "ipopt")]
pub fn append_ipopt_termination_metric(artifact: &mut SolveArtifact, summary: &IpoptSummary) {
    artifact.summary.push(metric_with_key(
        MetricKey::Termination,
        "Termination",
        ipopt_status_label(summary.status),
    ));
}

fn duration_seconds(duration: Option<Duration>) -> Option<f64> {
    duration.map(|value| value.as_secs_f64())
}

pub fn pre_jit_backend_timing(timing: BackendTimingMetadata) -> BackendTimingMetadata {
    BackendTimingMetadata {
        function_creation_time: timing.function_creation_time,
        derivative_generation_time: timing.derivative_generation_time,
        jit_time: None,
    }
}

fn symbolic_setup_seconds(timing: BackendTimingMetadata) -> Option<f64> {
    let function_creation = duration_seconds(timing.function_creation_time).unwrap_or(0.0);
    let derivative_generation = duration_seconds(timing.derivative_generation_time).unwrap_or(0.0);
    let total = function_creation + derivative_generation;
    if total > 0.0 { Some(total) } else { None }
}

pub trait FromMap: Sized {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self>;
}

pub trait StandardOcpParams {
    fn transcription(&self) -> &TranscriptionConfig;
    fn transcription_mut(&mut self) -> &mut TranscriptionConfig;
}

pub fn apply_derivative_request_overrides<Params>(
    params: &mut Params,
    request: &DerivativeCheckRequest,
) where
    Params: StandardOcpParams + HasOcpSxFunctionConfig,
{
    if let Some(sx_functions) = request.sx_functions_override {
        *params.sx_functions_mut() = sx_functions;
    }
}

pub trait HasOcpSxFunctionConfig {
    fn sx_functions_mut(&mut self) -> &mut OcpSxFunctionConfig;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolveRequest {
    pub values: BTreeMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct DerivativeCheckRequest {
    pub values: BTreeMap<String, f64>,
    pub finite_difference: FiniteDifferenceValidationOptions,
    pub equality_multiplier_fill: f64,
    pub inequality_multiplier_fill: f64,
    pub sx_functions_override: Option<OcpSxFunctionConfig>,
}

impl Default for DerivativeCheckRequest {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
            finite_difference: FiniteDifferenceValidationOptions::default(),
            equality_multiplier_fill: 1.0,
            inequality_multiplier_fill: 1.0,
            sx_functions_override: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeCheckOrder {
    First,
    Second,
    All,
}

#[derive(Clone, Debug)]
pub struct ProblemDerivativeCheck {
    pub problem_id: ProblemId,
    pub problem_name: String,
    pub transcription: TranscriptionMethod,
    pub collocation_family: Option<CollocationFamily>,
    pub sx_functions: OcpSxFunctionConfig,
    pub compile_cached: bool,
    pub compile_report: CompileReportSummary,
    pub report: NlpDerivativeValidationReport,
}

impl ProblemDerivativeCheck {
    pub fn first_order_is_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        self.report.first_order_is_within_tolerances(tolerances)
    }

    pub fn second_order_is_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        self.report.second_order_is_within_tolerances(tolerances)
    }

    pub fn all_orders_are_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        self.report.all_orders_are_within_tolerances(tolerances)
    }

    pub fn order_is_within_tolerances(
        &self,
        order: DerivativeCheckOrder,
        first_order: ValidationTolerances,
        second_order: ValidationTolerances,
    ) -> bool {
        match order {
            DerivativeCheckOrder::First => self.first_order_is_within_tolerances(first_order),
            DerivativeCheckOrder::Second => self.second_order_is_within_tolerances(second_order),
            DerivativeCheckOrder::All => {
                self.first_order_is_within_tolerances(first_order)
                    && self.second_order_is_within_tolerances(second_order)
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn assert_shared_progress_lifecycle(events: &[SolveStreamEvent]) {
    let status_count = events
        .iter()
        .filter(|event| matches!(event, SolveStreamEvent::Status { .. }))
        .count();
    let saw_jit_status = events.iter().any(|event| {
        matches!(
            event,
            SolveStreamEvent::Status { status }
                if status.stage == SolveStage::JitCompilation
        )
    });
    let saw_solving_timing = events.iter().any(|event| {
        matches!(
            event,
            SolveStreamEvent::Status { status }
                if status.stage == SolveStage::Solving
                    && status.solver.symbolic_setup_s.is_some()
                    && status.solver.jit_s.is_some()
        )
    });
    assert!(
        status_count >= 3,
        "expected symbolic, jit, and solve status updates"
    );
    assert!(
        saw_jit_status,
        "expected an explicit jit-compilation status"
    );
    assert!(
        saw_solving_timing,
        "expected solving status to include setup and jit timings"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let value = actual.expect("expected timing value");
        assert!(
            (value - expected).abs() < 1.0e-12,
            "expected {expected}, got {value}"
        );
    }

    fn assert_nlip_ipopt_termination_options_match(config: SolverConfig) {
        let nlip = nlip_options(&config);
        assert_eq!(nlip.overall_tol, config.overall_tol);
        assert_eq!(nlip.acceptable_iter, 15);

        #[cfg(feature = "ipopt")]
        {
            let ipopt = ipopt_options(&config);
            assert_eq!(nlip.max_iters, ipopt.max_iters);
            assert_eq!(nlip.overall_tol, ipopt.tol);
            assert_eq!(nlip.dual_tol, ipopt.dual_tol.unwrap());
            assert_eq!(nlip.constraint_tol, ipopt.constraint_tol.unwrap());
            assert_eq!(nlip.complementarity_tol, ipopt.complementarity_tol.unwrap());
            assert_eq!(Some(nlip.acceptable_tol), ipopt.acceptable_tol);
            assert_eq!(Some(nlip.acceptable_iter), ipopt.acceptable_iter);
            assert_eq!(
                Some(nlip.acceptable_dual_inf_tol),
                ipopt.acceptable_dual_tol
            );
            assert_eq!(
                Some(nlip.acceptable_constr_viol_tol),
                ipopt.acceptable_constraint_tol
            );
            assert_eq!(
                Some(nlip.acceptable_compl_inf_tol),
                ipopt.acceptable_complementarity_tol
            );
            assert_eq!(
                Some(nlip.acceptable_obj_change_tol),
                ipopt.acceptable_obj_change_tol
            );
        }
    }

    #[test]
    fn nlip_and_ipopt_strict_ocp_profile_termination_options_match() {
        let config = SolverConfig {
            profile: SolverProfile::StrictOcp,
            overall_tol: 1.0e-8,
            dual_tol: 1.0e-5,
            constraint_tol: 1.0e-8,
            complementarity_tol: 1.0e-6,
            ..default_solver_config()
        };

        let nlip = nlip_options(&config);
        assert_eq!(nlip.overall_tol, 1.0e-8);
        assert_eq!(nlip.acceptable_tol, 1.0e-6);
        assert_eq!(nlip.acceptable_iter, 15);
        assert_eq!(nlip.acceptable_dual_inf_tol, config.dual_tol);
        assert_eq!(nlip.acceptable_constr_viol_tol, config.constraint_tol);
        assert_eq!(nlip.acceptable_compl_inf_tol, config.complementarity_tol);

        assert_nlip_ipopt_termination_options_match(config);
    }

    #[test]
    fn nlip_and_ipopt_source_default_profile_termination_options_match() {
        let parsed = solver_config_from_map(
            &BTreeMap::from([("solver_profile".to_string(), 1.0)]),
            default_solver_config(),
        )
        .expect("solver config should parse source-default profile");

        assert_eq!(parsed.profile, SolverProfile::IpoptDefault);
        assert_eq!(parsed.max_iters, 3000);
        assert_eq!(parsed.overall_tol, 1.0e-8);
        assert_eq!(parsed.dual_tol, 1.0);
        assert_eq!(parsed.constraint_tol, 1.0e-4);
        assert_eq!(parsed.complementarity_tol, 1.0e-4);

        let nlip = nlip_options(&parsed);
        assert_eq!(nlip.acceptable_tol, 1.0e-6);
        assert_eq!(nlip.acceptable_iter, 15);
        assert_eq!(nlip.acceptable_dual_inf_tol, 1.0e10);
        assert_eq!(nlip.acceptable_constr_viol_tol, 1.0e-2);
        assert_eq!(nlip.acceptable_compl_inf_tol, 1.0e-2);

        assert_nlip_ipopt_termination_options_match(parsed);
    }

    #[test]
    fn solve_lifecycle_reporter_emits_shared_compile_status_updates() {
        let symbolic_timing = BackendTimingMetadata {
            function_creation_time: Some(Duration::from_millis(250)),
            derivative_generation_time: Some(Duration::from_millis(500)),
            jit_time: None,
        };
        let compiled_timing = BackendTimingMetadata {
            jit_time: Some(Duration::from_secs(2)),
            ..symbolic_timing
        };
        let mut events = Vec::new();
        let (compiled, running_solver, compile_report) = {
            let mut reporter =
                SolveLifecycleReporter::new(|event| events.push(event), SolverMethod::Sqp);
            let result = reporter
                .compile_with_progress(|on_symbolic_ready| {
                    on_symbolic_ready(CompileProgressUpdate {
                        timing: symbolic_timing,
                        phase_details: SolverPhaseDetails {
                            symbolic_setup: vec![phase_detail("Vars", "2", 0)],
                            jit: vec![phase_detail("NLP Kernels", "3", 0)],
                            solve: Vec::new(),
                        },
                        compile_cached: false,
                    });
                    Ok::<_, anyhow::Error>((
                        "compiled",
                        CompileProgressInfo {
                            timing: compiled_timing,
                            compile_cached: false,
                            phase_details: SolverPhaseDetails {
                                symbolic_setup: vec![phase_detail("Vars", "2", 0)],
                                jit: vec![phase_detail("NLP Kernels", "3", 0)],
                                solve: Vec::new(),
                            },
                            compile_report: None,
                        },
                    ))
                })
                .expect("compile should succeed");
            let _ = reporter.into_emit();
            result
        };

        assert_eq!(compiled, "compiled");
        assert!(compile_report.is_none());
        assert_eq!(events.len(), 2, "expected symbolic and timed jit statuses");

        match &events[0] {
            SolveStreamEvent::Status { status } => {
                assert_eq!(status.stage, SolveStage::SymbolicSetup);
                assert_eq!(status.solver.status_label, "Setting up symbolic model...");
                assert!(status.solver.symbolic_setup_s.is_none());
                assert!(status.solver.jit_s.is_none());
            }
            event => panic!("expected initial symbolic status, got {event:?}"),
        }

        match &events[1] {
            SolveStreamEvent::Status { status } => {
                assert_eq!(status.stage, SolveStage::JitCompilation);
                assert_eq!(status.solver.status_label, "Compiling JIT...");
                assert_close(status.solver.symbolic_setup_s, 0.75);
                assert!(status.solver.jit_s.is_none());
                assert!(!status.solver.compile_cached);
                assert_eq!(status.solver.phase_details.symbolic_setup.len(), 1);
            }
            event => panic!("expected timed jit status, got {event:?}"),
        }

        assert_eq!(running_solver.status_label, "Running SQP...");
        assert_close(running_solver.symbolic_setup_s, 0.75);
        assert_close(running_solver.jit_s, 2.0);
        assert!(running_solver.solve_s.is_none());
        assert!(running_solver.iterations.is_none());
        assert!(!running_solver.compile_cached);
        assert_eq!(running_solver.phase_details.symbolic_setup.len(), 1);
        assert_eq!(running_solver.phase_details.jit.len(), 1);
    }

    #[test]
    fn sqp_failure_diagnostics_print_failing_qp_status_explicitly() {
        let error = ClarabelSqpError::QpSolve {
            status: optimization::SqpQpRawStatus::NumericalError,
            context: Box::new(optimization::SqpFailureContext {
                termination: optimization::SqpTermination::QpSolve,
                final_state: None,
                final_state_kind: None,
                last_accepted_state: None,
                failed_trust_region: None,
                failed_line_search: None,
                failed_step_diagnostics: None,
                qp_failure: Some(optimization::SqpQpFailureDiagnostics {
                    qp_info: optimization::SqpQpInfo {
                        status: optimization::SqpQpStatus::Failed,
                        raw_status: optimization::SqpQpRawStatus::NumericalError,
                        setup_time: Duration::from_micros(750),
                        solve_time: Duration::from_millis(12),
                        iteration_count: 17,
                    },
                    variable_count: 12,
                    constraint_count: 24,
                    linear_objective_inf_norm: 1.0,
                    rhs_inf_norm: 2.0,
                    hessian_diag_min: -3.0,
                    hessian_diag_max: 4.0,
                    elastic_recovery: false,
                    cones: vec![optimization::SqpConeSummary {
                        kind: optimization::SqpConeKind::Zero,
                        dim: 24,
                    }],
                    transcript: Some(
                        "iter    pcost        dcost\n0   1.0e0   2.0e0\nstatus: NumericalError"
                            .to_string(),
                    ),
                }),
                profiling: ClarabelSqpProfiling::default(),
            }),
        };

        let lines = sqp_failure_diagnostic_lines(&error);
        let rendered = lines.join("\n");
        assert!(rendered.contains("termination=QpSolve"));
        assert!(rendered.contains("failed while solving the next SQP subproblem"));
        assert!(rendered.contains("failed_qp status=Failed raw_status=NumericalError iters=17"));
        assert!(rendered.contains("clarabel qp transcript:"));
        assert!(rendered.contains("status: NumericalError"));
    }

    #[test]
    fn prewarm_with_progress_emits_final_jit_timing_status() {
        let symbolic_timing = BackendTimingMetadata {
            function_creation_time: Some(Duration::from_millis(250)),
            derivative_generation_time: Some(Duration::from_millis(500)),
            jit_time: None,
        };
        let compiled_timing = BackendTimingMetadata {
            function_creation_time: Some(Duration::from_millis(250)),
            derivative_generation_time: Some(Duration::from_millis(500)),
            jit_time: Some(Duration::from_secs(2)),
        };
        let mut events = Vec::new();
        {
            let mut reporter =
                SolveLifecycleReporter::new(|event| events.push(event), SolverMethod::Sqp);
            reporter
                .prewarm_with_progress(|on_symbolic_ready| {
                    on_symbolic_ready(CompileProgressUpdate {
                        timing: symbolic_timing,
                        phase_details: SolverPhaseDetails {
                            symbolic_setup: vec![phase_detail("Vars", "2", 0)],
                            jit: vec![phase_detail("NLP Kernels", "3", 0)],
                            solve: Vec::new(),
                        },
                        compile_cached: false,
                    });
                    Ok::<_, anyhow::Error>((
                        "compiled",
                        CompileProgressInfo {
                            timing: compiled_timing,
                            compile_cached: false,
                            phase_details: SolverPhaseDetails {
                                symbolic_setup: vec![phase_detail("Vars", "2", 0)],
                                jit: vec![phase_detail("NLP Kernels", "3", 0)],
                                solve: Vec::new(),
                            },
                            compile_report: None,
                        },
                    ))
                })
                .expect("prewarm should succeed");
        }

        assert_eq!(
            events.len(),
            3,
            "expected symbolic, in-progress jit, and final jit statuses"
        );

        match &events[2] {
            SolveStreamEvent::Status { status } => {
                assert_eq!(status.stage, SolveStage::JitCompilation);
                assert_eq!(status.solver.status_label, "Compiling JIT...");
                assert_close(status.solver.symbolic_setup_s, 0.75);
                assert_close(status.solver.jit_s, 2.0);
                assert!(!status.solver.compile_cached);
                assert_eq!(status.solver.phase_details.jit.len(), 1);
            }
            event => panic!("expected final jit status, got {event:?}"),
        }
    }

    #[test]
    fn ocp_compile_progress_updates_include_helper_timings() {
        let symbolic_timing = BackendTimingMetadata {
            function_creation_time: Some(Duration::from_millis(250)),
            derivative_generation_time: Some(Duration::from_millis(500)),
            jit_time: None,
        };
        let mut state = OcpCompileProgressState::default();
        let symbolic = ocp_compile_progress_update(
            OcpCompileProgress::SymbolicReady(optimization::SymbolicCompileMetadata {
                timing: symbolic_timing,
                setup_profile: SymbolicSetupProfile::default(),
                stats: NlpCompileStats {
                    jit_kernel_count: 3,
                    ..NlpCompileStats::default()
                },
            }),
            &mut state,
        );
        assert_eq!(symbolic.phase_details.jit.len(), 1);
        assert_eq!(symbolic.phase_details.jit[0].label, "NLP Kernels");

        let xdot = ocp_compile_progress_update(
            OcpCompileProgress::HelperCompiled {
                helper: OcpCompileHelperKind::Xdot,
                elapsed: Duration::from_millis(125),
                root_instructions: 42,
                total_instructions: 42,
            },
            &mut state,
        );
        assert_eq!(xdot.phase_details.jit.len(), 2);
        assert_eq!(xdot.phase_details.jit[1].label, "Xdot Helper");
        assert_eq!(xdot.phase_details.jit[1].value, "125 ms");

        let arc = ocp_compile_progress_update(
            OcpCompileProgress::HelperCompiled {
                helper: OcpCompileHelperKind::MultipleShootingArc,
                elapsed: Duration::from_secs_f64(4.5),
                root_instructions: 84,
                total_instructions: 84,
            },
            &mut state,
        );
        assert_eq!(arc.phase_details.jit.len(), 3);
        assert_eq!(arc.phase_details.jit[2].label, "RK4 Arc Helper");
        assert_eq!(arc.phase_details.jit[2].value, "4.50 s");
    }

    #[test]
    fn shared_progress_lifecycle_helper_accepts_timed_solving_status() {
        let symbolic_timing = BackendTimingMetadata {
            function_creation_time: Some(Duration::from_millis(250)),
            derivative_generation_time: Some(Duration::from_millis(500)),
            jit_time: Some(Duration::from_secs(2)),
        };
        let events = vec![
            SolveStreamEvent::Status {
                status: SolveStatus {
                    stage: SolveStage::SymbolicSetup,
                    solver_method: None,
                    solver: SolverReport::in_progress("Setting up symbolic model..."),
                },
            },
            SolveStreamEvent::Status {
                status: SolveStatus {
                    stage: SolveStage::JitCompilation,
                    solver_method: None,
                    solver: SolverReport::in_progress("Compiling JIT...")
                        .with_backend_timing(symbolic_timing),
                },
            },
            SolveStreamEvent::Status {
                status: SolveStatus {
                    stage: SolveStage::Solving,
                    solver_method: Some(SolverMethod::Sqp),
                    solver: SolverReport::in_progress("Running SQP...")
                        .with_backend_timing(symbolic_timing)
                        .with_solve_seconds(0.0),
                },
            },
        ];

        assert_shared_progress_lifecycle(&events);
    }

    #[test]
    fn transcription_controls_include_collapsible_sx_function_panel() {
        let controls = transcription_controls(default_transcription(30), &[30], &[3]);
        let ode = controls
            .iter()
            .find(|control| control.id == "sxf_ode")
            .expect("expected sx function ODE control");
        let hessian = controls
            .iter()
            .find(|control| control.id == "sxf_hessian_strategy")
            .expect("expected sx function Hessian strategy control");
        assert_eq!(ode.section, ControlSection::Transcription);
        assert_eq!(ode.panel, Some(ControlPanel::SxFunctions));
        assert_eq!(ode.semantic, ControlSemantic::SxFunctionOption);
        assert_eq!(hessian.section, ControlSection::Transcription);
        assert_eq!(hessian.panel, Some(ControlPanel::SxFunctions));
        assert_eq!(hessian.semantic, ControlSemantic::SxFunctionOption);
    }

    #[test]
    fn transcription_controls_use_text_entry_for_interval_count() {
        let controls =
            transcription_controls(default_transcription(30), &[10, 20, 30, 40], &[2, 3, 4]);
        let intervals = controls
            .iter()
            .find(|control| control.id == "transcription_intervals")
            .expect("expected interval control");
        let collocation_nodes = controls
            .iter()
            .find(|control| control.id == "collocation_degree")
            .expect("expected collocation node control");

        assert_eq!(intervals.section, ControlSection::Transcription);
        assert_eq!(intervals.editor, ControlEditor::Text);
        assert_eq!(intervals.value_display, ControlValueDisplay::Integer);
        assert!(intervals.choices.is_empty());
        assert_eq!(intervals.min, 10.0);
        assert_eq!(intervals.max, 40.0);
        assert_eq!(collocation_nodes.editor, ControlEditor::Select);
        assert_eq!(collocation_nodes.choices.len(), 3);
    }

    #[test]
    fn transcription_from_map_accepts_any_interval_inside_configured_range() {
        let default = default_transcription(30);
        let mut values = BTreeMap::new();
        values.insert("transcription_intervals".to_string(), 25.0);

        let parsed = transcription_from_map(&values, default, &[10, 20, 30, 40], &[3])
            .expect("interval inside range should parse");
        assert_eq!(parsed.intervals, 25);

        values.insert("transcription_intervals".to_string(), 45.0);
        let error = transcription_from_map(&values, default, &[10, 20, 30, 40], &[3])
            .expect_err("interval outside range should fail");
        assert!(error.to_string().contains("expected 10..=40"));
    }

    #[test]
    fn transcription_controls_include_time_grid_warping_parameters() {
        let controls = transcription_controls(default_transcription(30), &[30], &[3]);
        let time_grid = controls
            .iter()
            .find(|control| control.id == "time_grid")
            .expect("expected time grid control");
        let focus_center = controls
            .iter()
            .find(|control| control.id == "time_grid_focus_center")
            .expect("expected focus center control");
        let piecewise_share = controls
            .iter()
            .find(|control| control.id == "time_grid_first_interval_fraction")
            .expect("expected piecewise interval share control");

        assert_eq!(time_grid.semantic, ControlSemantic::TimeGrid);
        assert_eq!(time_grid.choices.len(), 7);
        assert_eq!(focus_center.semantic, ControlSemantic::TimeGridFocusCenter);
        assert_eq!(
            piecewise_share.semantic,
            ControlSemantic::TimeGridFirstIntervalFraction
        );
        assert_eq!(
            focus_center.visibility,
            ControlVisibility::DirectCollocationOnly
        );
    }

    #[test]
    fn transcription_from_map_parses_time_grid_warping_parameters() {
        let mut values = BTreeMap::new();
        values.insert("time_grid".to_string(), 5.0);
        values.insert("time_grid_strength".to_string(), 0.7);
        values.insert("time_grid_focus_center".to_string(), 0.25);
        values.insert("time_grid_focus_width".to_string(), 0.2);
        let focus = transcription_from_map(&values, default_transcription(30), &[30], &[3])
            .expect("focus time grid should parse");
        assert_eq!(
            focus.time_grid,
            TimeGrid::Focus {
                center: 0.25,
                width: 0.2,
                strength: 0.7,
            }
        );

        values.insert("time_grid".to_string(), 6.0);
        values.insert("time_grid_breakpoint".to_string(), 0.35);
        values.insert("time_grid_first_interval_fraction".to_string(), 0.6);
        let piecewise = transcription_from_map(&values, default_transcription(30), &[30], &[3])
            .expect("piecewise time grid should parse");
        assert_eq!(
            piecewise.time_grid,
            TimeGrid::Piecewise {
                breakpoint: 0.35,
                first_interval_fraction: 0.6,
            }
        );
    }

    #[test]
    fn solver_controls_include_sqp_hessian_regularization_toggle() {
        let controls = solver_controls(default_solver_method(), default_solver_config());
        let toggle = controls
            .iter()
            .find(|control| control.id == "solver_hessian_regularization")
            .expect("expected SQP Hessian regularization control");
        assert_eq!(toggle.section, ControlSection::Solver);
        assert_eq!(
            toggle.semantic,
            ControlSemantic::SolverHessianRegularization
        );
        assert_eq!(toggle.editor, ControlEditor::Select);
        assert_eq!(toggle.default, 0.0);
        assert_eq!(toggle.choices.len(), 2);
        assert_eq!(toggle.choices[0].label, "Off");
        assert_eq!(toggle.choices[1].label, "On");
    }

    #[test]
    fn solver_config_from_map_parses_hessian_regularization_toggle() {
        let mut values = BTreeMap::new();
        values.insert("solver_hessian_regularization".to_string(), 1.0);
        let parsed = solver_config_from_map(&values, default_solver_config())
            .expect("solver config should parse");
        assert!(parsed.sqp.hessian_regularization_enabled);
    }

    #[test]
    fn solver_controls_allow_max_iterations_above_default() {
        let controls = solver_controls(default_solver_method(), default_solver_config());
        let max_iters = controls
            .iter()
            .find(|control| control.id == "solver_max_iters")
            .expect("expected max-iterations control");
        assert_eq!(max_iters.editor, ControlEditor::Text);
        assert!(max_iters.max > max_iters.default);

        let mut values = BTreeMap::new();
        values.insert("solver_max_iters".to_string(), 1000.0);
        let parsed = solver_config_from_map(&values, default_solver_config())
            .expect("solver config should parse");
        assert_eq!(parsed.max_iters, 1000);
    }

    #[test]
    fn solver_controls_expose_profile_defaults_for_text_overrides() {
        let controls = solver_controls(default_solver_method(), default_solver_config());
        let profile = controls
            .iter()
            .find(|control| control.id == "solver_profile")
            .expect("expected solver profile control");
        assert_eq!(profile.editor, ControlEditor::Select);
        assert_eq!(
            profile.default,
            solver_profile_value(SolverProfile::StrictOcp)
        );
        assert_eq!(profile.choices[0].label, "Strict OCP");
        assert_eq!(profile.choices[1].label, "IPOPT Default");

        let max_iters = controls
            .iter()
            .find(|control| control.id == "solver_max_iters")
            .expect("expected max-iterations control");
        assert_eq!(
            max_iters.profile_defaults,
            vec![
                ControlProfileDefault {
                    profile: solver_profile_value(SolverProfile::StrictOcp),
                    value: 200.0,
                },
                ControlProfileDefault {
                    profile: solver_profile_value(SolverProfile::IpoptDefault),
                    value: 3000.0,
                },
            ]
        );

        let dual_tol = controls
            .iter()
            .find(|control| control.id == "solver_dual_tol")
            .expect("expected dual tolerance control");
        assert_eq!(dual_tol.profile_defaults[0].value, 1.0e-6);
        assert_eq!(dual_tol.profile_defaults[1].value, 1.0);
    }

    #[test]
    fn solver_config_from_map_uses_selected_profile_defaults_and_text_overrides() {
        let parsed = solver_config_from_map(
            &BTreeMap::from([("solver_profile".to_string(), 1.0)]),
            default_solver_config(),
        )
        .expect("solver config should parse IPOPT profile");
        assert_eq!(parsed.profile, SolverProfile::IpoptDefault);
        assert_eq!(parsed.max_iters, 3000);
        assert_eq!(parsed.overall_tol, 1.0e-8);
        assert_eq!(parsed.dual_tol, 1.0);
        assert_eq!(parsed.constraint_tol, 1.0e-4);
        assert_eq!(parsed.complementarity_tol, 1.0e-4);

        let parsed = solver_config_from_map(
            &BTreeMap::from([
                ("solver_profile".to_string(), 1.0),
                ("solver_max_iters".to_string(), 42.0),
                ("solver_overall_tol".to_string(), 1.0e-10),
                ("solver_dual_tol".to_string(), 1.0e-9),
            ]),
            default_solver_config(),
        )
        .expect("solver config should parse overridden IPOPT profile");
        assert_eq!(parsed.profile, SolverProfile::IpoptDefault);
        assert_eq!(parsed.max_iters, 42);
        assert_eq!(parsed.overall_tol, 1.0e-10);
        assert_eq!(parsed.dual_tol, 1.0e-9);
        assert_eq!(parsed.constraint_tol, 1.0e-4);
    }

    #[test]
    fn solver_config_from_map_parses_nlip_linear_solver_choice() {
        let mut values = BTreeMap::new();
        values.insert("solver_nlip_linear_solver".to_string(), 2.0);
        let parsed = solver_config_from_map(&values, default_solver_config())
            .expect("solver config should parse");
        assert_eq!(
            parsed.nlip.linear_solver,
            InteriorPointLinearSolver::SparseQdldl
        );
    }

    #[test]
    fn solver_config_from_map_parses_native_nlip_linear_solver_choice() {
        let mut values = BTreeMap::new();
        values.insert("solver_nlip_linear_solver".to_string(), 1.0);
        let parsed = solver_config_from_map(&values, default_solver_config())
            .expect("solver config should parse");
        assert_eq!(
            parsed.nlip.linear_solver,
            InteriorPointLinearSolver::SpralSrc
        );
    }

    #[cfg(feature = "ipopt")]
    #[test]
    fn solver_method_from_map_parses_ipopt_choice() {
        let mut values = BTreeMap::new();
        values.insert("solver_method".to_string(), 2.0);
        let parsed = solver_method_from_map(&values, default_solver_method())
            .expect("solver method should parse");
        assert_eq!(parsed, SolverMethod::Ipopt);
    }

    #[cfg(feature = "ipopt")]
    #[test]
    fn default_ipopt_options_select_spral_explicitly() {
        let options = ipopt_options(&default_solver_config());
        assert_eq!(options.max_iters, 200);
        assert_eq!(options.tol, 1.0e-8);
        assert_eq!(options.dual_tol, Some(1.0e-6));
        assert_eq!(options.constraint_tol, Some(1.0e-8));
        assert_eq!(options.complementarity_tol, Some(1.0e-6));
        assert_eq!(options.acceptable_tol, Some(1.0e-6));
        assert_eq!(options.acceptable_iter, Some(15));
        assert_eq!(options.acceptable_dual_tol, Some(1.0e-6));
        assert_eq!(options.acceptable_constraint_tol, Some(1.0e-8));
        assert_eq!(options.acceptable_complementarity_tol, Some(1.0e-6));
        assert_eq!(options.linear_solver, Some(IpoptLinearSolver::Spral));
        assert_eq!(options.mu_strategy, optimization::IpoptMuStrategy::Monotone);
        assert!(format_ipopt_settings_summary(&options).contains("linear_solver=spral"));
        assert!(format_ipopt_settings_summary(&options).contains("mu_strategy=monotone"));
        assert!(format_ipopt_settings_summary(&options).contains("acceptable_iter=15"));
    }

    #[cfg(feature = "ipopt")]
    #[test]
    fn default_ipopt_options_print_timing_statistics() {
        let options = ipopt_options(&default_solver_config());
        assert!(options.raw_options.iter().any(|option| {
            option.name == "print_timing_statistics"
                && matches!(
                    &option.value,
                    optimization::IpoptRawOptionValue::Text(value) if value == "yes"
                )
        }));
    }

    #[test]
    fn default_nlip_config_matches_ipopt_spral_defaults() {
        let default = default_nlip_config();
        assert_eq!(default.linear_solver, InteriorPointLinearSolver::SpralSrc);
        assert_eq!(
            default.spral_pivot_method,
            InteriorPointSpralPivotMethod::AggressiveAposteriori
        );
        assert!(default.spral_action_on_zero_pivot);
        assert_eq!(default.spral_small_pivot_tolerance, 1.0e-20);
        assert_eq!(default.spral_threshold_pivot_u, 1.0e-8);
    }

    #[test]
    fn solver_config_from_map_parses_nlip_spral_controls() {
        let mut values = BTreeMap::new();
        values.insert("solver_nlip_spral_pivot_method".to_string(), 2.0);
        values.insert("solver_nlip_spral_zero_pivot_action".to_string(), 0.0);
        values.insert("solver_nlip_spral_small_pivot".to_string(), 1.0e-18);
        values.insert("solver_nlip_spral_pivot_u".to_string(), 1.0e-6);
        let parsed = solver_config_from_map(&values, default_solver_config())
            .expect("solver config should parse");
        assert_eq!(
            parsed.nlip.spral_pivot_method,
            InteriorPointSpralPivotMethod::ThresholdPartial
        );
        assert!(!parsed.nlip.spral_action_on_zero_pivot);
        assert_eq!(parsed.nlip.spral_small_pivot_tolerance, 1.0e-18);
        assert_eq!(parsed.nlip.spral_threshold_pivot_u, 1.0e-6);
    }

    #[test]
    fn sx_function_config_variant_suffix_changes_for_custom_settings() {
        let default = OcpSxFunctionConfig::default();
        let custom = OcpSxFunctionConfig {
            global_call_policy: CallPolicy::NoInlineLLVM,
            hessian_strategy: HessianStrategy::LowerTriangleColored,
            multiple_shooting_integrator: OcpKernelStrategy::FunctionUseGlobalPolicy,
            ..default
        };
        assert_ne!(default.variant_id_suffix(), custom.variant_id_suffix());
        assert_eq!(custom.variant_label_suffix().as_deref(), Some("SXF Custom"));
    }

    #[test]
    fn sqp_solve_phase_details_break_out_regularization_and_preprocess() {
        let profiling = optimization::ClarabelSqpProfiling {
            objective_value: optimization::EvalTimingStat {
                calls: 5,
                total_time: Duration::from_millis(10),
            },
            objective_gradient: optimization::EvalTimingStat {
                calls: 5,
                total_time: Duration::from_millis(12),
            },
            lagrangian_hessian_values: optimization::EvalTimingStat {
                calls: 4,
                total_time: Duration::from_millis(18),
            },
            jacobian_assembly_steps: 5,
            jacobian_assembly_time: Duration::from_millis(20),
            hessian_assembly_steps: 4,
            hessian_assembly_time: Duration::from_millis(22),
            regularization_steps: 4,
            regularization_time: Duration::from_millis(44),
            preprocessing_other_steps: 5,
            preprocessing_other_time: Duration::from_millis(16),
            preprocessing_steps: 5,
            preprocessing_time: Duration::from_millis(102),
            evaluation_time: Duration::from_millis(45),
            ..optimization::ClarabelSqpProfiling::default()
        };

        let details = sqp_solve_phase_details(&profiling);

        assert!(
            details
                .iter()
                .any(|detail| detail.label == "Jacobian Build" && detail.count == 5)
        );
        assert!(
            details
                .iter()
                .any(|detail| detail.label == "Hessian Build" && detail.count == 4)
        );
        assert!(
            details
                .iter()
                .any(|detail| detail.label == "Regularization" && detail.count == 4)
        );
        assert!(
            details
                .iter()
                .any(|detail| detail.label == "Preprocess Other" && detail.count == 0)
        );
        assert!(
            details
                .iter()
                .any(|detail| detail.label == "Preprocess" && detail.count == 0)
        );
        assert!(
            details
                .iter()
                .any(|detail| detail.label == "Evaluate Functions" && detail.count == 0)
        );
    }

    #[test]
    fn nlip_failure_solver_report_preserves_timing_details() {
        let profiling = optimization::InteriorPointProfiling {
            objective_value: optimization::EvalTimingStat {
                calls: 4,
                total_time: Duration::from_millis(12),
            },
            objective_gradient: optimization::EvalTimingStat {
                calls: 4,
                total_time: Duration::from_millis(8),
            },
            equality_jacobian_values: optimization::EvalTimingStat {
                calls: 4,
                total_time: Duration::from_millis(21),
            },
            lagrangian_hessian_values: optimization::EvalTimingStat {
                calls: 3,
                total_time: Duration::from_millis(34),
            },
            kkt_assemblies: 3,
            kkt_assembly_time: Duration::from_millis(55),
            linear_solves: 6,
            linear_solve_time: Duration::from_millis(89),
            linear_rhs_assemblies: 6,
            linear_rhs_assembly_time: Duration::from_millis(3),
            linear_kkt_value_assemblies: 7,
            linear_kkt_value_assembly_time: Duration::from_millis(4),
            sparse_symbolic_analyses: 1,
            sparse_symbolic_analysis_time: Duration::from_millis(5),
            sparse_numeric_factorizations: 2,
            sparse_numeric_factorization_time: Duration::from_millis(11),
            sparse_numeric_refactorizations: 4,
            sparse_numeric_refactorization_time: Duration::from_millis(13),
            linear_backsolves: 6,
            linear_backsolve_time: Duration::from_millis(17),
            preprocessing_steps: 2,
            preprocessing_time: Duration::from_millis(13),
            total_time: Duration::from_secs_f64(0.321),
            backend_timing: BackendTimingMetadata {
                function_creation_time: Some(Duration::from_millis(150)),
                derivative_generation_time: Some(Duration::from_millis(250)),
                jit_time: Some(Duration::from_millis(500)),
            },
            ..optimization::InteriorPointProfiling::default()
        };
        let error = optimization::InteriorPointSolveError::LineSearchFailed {
            merit: 1.0,
            mu: 1.0e-3,
            step_inf_norm: 0.25,
            context: Box::new(optimization::InteriorPointFailureContext {
                final_state: None,
                last_accepted_state: None,
                failed_linear_solve: None,
                failed_line_search: None,
                failed_direction_diagnostics: None,
                profiling,
            }),
        };
        let fallback = SolverReport::in_progress("Running NLIP...")
            .with_backend_timing(BackendTimingMetadata {
                function_creation_time: Some(Duration::from_millis(150)),
                derivative_generation_time: Some(Duration::from_millis(250)),
                jit_time: Some(Duration::from_millis(500)),
            })
            .with_compile_cached(true)
            .with_phase_details(SolverPhaseDetails {
                symbolic_setup: vec![phase_detail("Vars", "10", 0)],
                jit: vec![phase_detail("NLP Kernels", "3", 0)],
                solve: Vec::new(),
            });
        let report =
            nlip_failure_solver_report(&error, &fallback, &InteriorPointOptions::default())
                .expect("expected failure report");

        assert!(report.completed);
        assert_eq!(report.status_kind, SolverStatusKind::Error);
        assert_eq!(report.status_label, "Failed: line search");
        assert_eq!(report.symbolic_setup_s, Some(0.4));
        assert_eq!(report.jit_s, Some(0.5));
        assert_eq!(report.solve_s, Some(0.321));
        assert!(report.compile_cached);
        assert_eq!(report.phase_details.symbolic_setup.len(), 1);
        assert_eq!(report.phase_details.jit.len(), 1);
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "KKT Assembly")
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Linear Solve")
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Hessian")
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "KKT Assembly" && detail.count == 3)
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Linear Solve" && detail.count == 6)
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Linear RHS Assembly" && detail.count == 6)
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Linear KKT Values" && detail.count == 7)
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Linear Symbolic Analysis" && detail.count == 1)
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Linear Numeric Factorization" && detail.count == 2)
        );
        assert!(report.phase_details.solve.iter().any(|detail| {
            detail.label == "Linear Numeric Refactorization" && detail.count == 4
        }));
        assert!(report.phase_details.solve.iter().any(|detail| {
            detail.label == "Linear Backsolve / Refinement" && detail.count == 6
        }));
        assert!(
            report.phase_details.solve.iter().any(|detail| {
                detail.label == "Linear Solve Other" && detail.value == "36.0 ms"
            })
        );
        assert!(
            report
                .phase_details
                .solve
                .iter()
                .any(|detail| detail.label == "Preprocess" && detail.count == 2)
        );
    }
}
