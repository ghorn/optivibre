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
    Bounds1D, CollocationFamily, CompiledDirectCollocationOcp, CompiledMultipleShootingOcp,
    ControllerFn, DirectCollocationInitialGuess, DirectCollocationInteriorPointSnapshot,
    DirectCollocationRuntimeValues, DirectCollocationSqpSnapshot, DirectCollocationTimeGrid,
    DirectCollocationTrajectories, InterpolatedTrajectory, IntervalArc,
    MultipleShootingInitialGuess, MultipleShootingInteriorPointSnapshot,
    MultipleShootingRuntimeValues, MultipleShootingSqpSnapshot, MultipleShootingTrajectories,
    OcpCompileHelperKind, OcpCompileOptions, OcpCompileProgress, OcpConstraintCategory,
    OcpConstraintViolationReport, OcpHelperCompileStats, OcpKernelFunctionOptions,
    OcpSymbolicFunctionOptions,
};
#[cfg(feature = "ipopt")]
use optimal_control::{DirectCollocationIpoptSnapshot, MultipleShootingIpoptSnapshot};
use optimization::{
    BackendCompileReport, BackendTimingMetadata, CallPolicy, CallPolicyConfig, ClarabelSqpError,
    ClarabelSqpOptions, ClarabelSqpProfiling, ClarabelSqpSummary, ConstraintSatisfaction,
    FunctionCompileOptions, InteriorPointIterationSnapshot, InteriorPointOptions,
    InteriorPointProfiling, InteriorPointSolveError, InteriorPointSummary, LlvmOptimizationLevel,
    NlpCompileStats, NlpEvaluationBenchmark, NlpEvaluationBenchmarkOptions,
    NlpEvaluationKernelKind, SqpIterationEvent, SqpIterationPhase, SqpIterationSnapshot,
    SymbolicSetupProfile, Vectorize,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptOptions, IpoptRawStatus, IpoptSummary};
#[cfg(feature = "ipopt")]
use optimization::{IpoptProfiling, IpoptSolveError};
use serde::{Deserialize, Serialize};
use sx_core::SX;

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
    SolverMethod,
    SolverMaxIterations,
    SolverDualTolerance,
    SolverConstraintTolerance,
    SolverComplementarityTolerance,
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
}

impl ProblemId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OptimalDistanceGlider => "optimal_distance_glider",
            Self::LinearSManeuver => "linear_s_maneuver",
            Self::SailboatUpwind => "sailboat_upwind",
            Self::CraneTransfer => "crane_transfer",
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DirectCollocationCompileKey {
    Legendre,
    RadauIia,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKey {
    #[default]
    Custom,
    TranscriptionMethod,
    IntervalCount,
    CollocationNodeCount,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SqpConfig {
    pub max_iters: usize,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub complementarity_tol: f64,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptionConfig {
    pub method: TranscriptionMethod,
    pub intervals: usize,
    pub collocation_degree: usize,
    pub collocation_family: CollocationFamily,
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
        }
    }

    pub fn variant_id_suffix(self) -> String {
        format!(
            "g{}_ov{}_ode{}_lag{}_may{}_path{}_beq{}_biq{}_msi{}",
            call_policy_variant_token(self.global_call_policy),
            match self.override_behavior {
                OcpOverrideBehavior::RespectFunctionOverrides => "r",
                OcpOverrideBehavior::StrictGlobalPolicy => "s",
            },
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
    pub family: DirectCollocationCompileKey,
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
    family: CollocationFamily,
    sx_functions: OcpSxFunctionConfig,
) -> DirectCollocationCompileVariantKey {
    DirectCollocationCompileVariantKey {
        family: direct_collocation_compile_key(family),
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
    SxFunctionGlobalCallPolicy,
    SxFunctionOverrideBehavior,
    SxFunctionOde,
    SxFunctionObjectiveLagrange,
    SxFunctionObjectiveMayer,
    SxFunctionPathConstraints,
    SxFunctionBoundaryEqualities,
    SxFunctionBoundaryInequalities,
    SxFunctionMultipleShootingIntegrator,
    SolverMethod,
    SolverMaxIterations,
    SolverDualTolerance,
    SolverConstraintTolerance,
    SolverComplementarityTolerance,
}

impl SharedControlId {
    const fn id(self) -> &'static str {
        match self {
            Self::TranscriptionMethod => "transcription_method",
            Self::TranscriptionIntervals => "transcription_intervals",
            Self::CollocationFamily => "collocation_family",
            Self::CollocationDegree => "collocation_degree",
            Self::SxFunctionGlobalCallPolicy => "sxf_global_call_policy",
            Self::SxFunctionOverrideBehavior => "sxf_override_behavior",
            Self::SxFunctionOde => "sxf_ode",
            Self::SxFunctionObjectiveLagrange => "sxf_objective_lagrange",
            Self::SxFunctionObjectiveMayer => "sxf_objective_mayer",
            Self::SxFunctionPathConstraints => "sxf_path_constraints",
            Self::SxFunctionBoundaryEqualities => "sxf_boundary_equalities",
            Self::SxFunctionBoundaryInequalities => "sxf_boundary_inequalities",
            Self::SxFunctionMultipleShootingIntegrator => "sxf_multiple_shooting_integrator",
            Self::SolverMethod => "solver_method",
            Self::SolverMaxIterations => "solver_max_iters",
            Self::SolverDualTolerance => "solver_dual_tol",
            Self::SolverConstraintTolerance => "solver_constraint_tol",
            Self::SolverComplementarityTolerance => "solver_complementarity_tol",
        }
    }
}

type Numeric<T> = <T as Vectorize<SX>>::Rebind<f64>;

pub enum ContinuousInitialGuess<X, U, P> {
    Interpolated(InterpolatedTrajectory<X, U>),
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
    pub fn into_multiple_shooting<const N: usize>(
        self,
    ) -> MultipleShootingInitialGuess<X, U, P, N> {
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

    pub fn into_direct_collocation<const N: usize, const K: usize>(
        self,
    ) -> DirectCollocationInitialGuess<X, U, P, N, K> {
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

pub fn multiple_shooting_runtime_from_spec<P, C, Beq, Bineq, X, U, const N: usize>(
    spec: OcpRuntimeSpec<P, C, Beq, Bineq, X, U>,
) -> MultipleShootingRuntimeValues<P, C, Beq, Bineq, X, U, N> {
    MultipleShootingRuntimeValues {
        parameters: spec.parameters,
        beq: spec.beq,
        bineq_bounds: spec.bineq_bounds,
        path_bounds: spec.path_bounds,
        tf_bounds: spec.tf_bounds,
        initial_guess: spec.initial_guess.into_multiple_shooting(),
    }
}

pub fn direct_collocation_runtime_from_spec<
    P,
    C,
    Beq,
    Bineq,
    X,
    U,
    const N: usize,
    const K: usize,
>(
    spec: OcpRuntimeSpec<P, C, Beq, Bineq, X, U>,
) -> DirectCollocationRuntimeValues<P, C, Beq, Bineq, X, U, N, K> {
    DirectCollocationRuntimeValues {
        parameters: spec.parameters,
        beq: spec.beq,
        bineq_bounds: spec.bineq_bounds,
        path_bounds: spec.path_bounds,
        tf_bounds: spec.tf_bounds,
        initial_guess: spec.initial_guess.into_direct_collocation(),
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
        |compiled| compiled.backend_timing_metadata(),
        |compiled| compiled.backend_timing_metadata(),
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
    const N: usize,
    const K: usize,
>(
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    cached_multiple_shooting: CachedMs,
    cached_direct_collocation: CachedDc,
    multiple_shooting_runtime: MsRuntimeFn,
    direct_collocation_runtime: DcRuntimeFn,
    multiple_shooting_artifact: MsArtifact,
    direct_collocation_artifact: DcArtifact,
) -> Result<SolveArtifact>
where
    MsCompiled: MultipleShootingCompiled<N>,
    DcCompiled: DirectCollocationCompiled<N, K>,
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
        N,
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
        N,
        K,
    >,
    MsArtifact: FnMut(
        &MultipleShootingTrajectories<MsCompiled::XNum, MsCompiled::UNum, N>,
        &[IntervalArc<MsCompiled::XNum>],
        &[IntervalArc<MsCompiled::UNum>],
    ) -> SolveArtifact,
    DcArtifact: FnMut(
        &DirectCollocationTrajectories<DcCompiled::XNum, DcCompiled::UNum, N, K>,
        &DirectCollocationTimeGrid<N, K>,
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
    const N: usize,
    const K: usize,
>(
    params: &Params,
    transcription: TranscriptionMethod,
    collocation_family: CollocationFamily,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
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
    MsCompiled: MultipleShootingCompiled<N>,
    DcCompiled: DirectCollocationCompiled<N, K>,
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
        N,
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
        N,
        K,
    >,
    MsArtifact: FnMut(
        &MultipleShootingTrajectories<MsCompiled::XNum, MsCompiled::UNum, N>,
        &[IntervalArc<MsCompiled::XNum>],
        &[IntervalArc<MsCompiled::UNum>],
    ) -> SolveArtifact,
    DcArtifact: FnMut(
        &DirectCollocationTrajectories<DcCompiled::XNum, DcCompiled::UNum, N, K>,
        &DirectCollocationTimeGrid<N, K>,
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
    const N: usize,
    const K: usize,
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
    MsCompiled: MultipleShootingCompiled<N>,
    DcCompiled: DirectCollocationCompiled<N, K>,
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
        N,
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
        N,
        K,
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

pub fn compile_cache_status(
    problem_id: ProblemId,
    problem_name: &str,
    variant_id: &str,
    variant_label: &str,
    timing: BackendTimingMetadata,
) -> CompileCacheStatus {
    CompileCacheStatus {
        problem_id,
        problem_name: problem_name.to_string(),
        variant_id: variant_id.to_string(),
        variant_label: variant_label.to_string(),
        state: CompileCacheState::Ready,
        symbolic_setup_s: symbolic_setup_seconds(timing),
        jit_s: duration_seconds(timing.jit_time),
    }
}

pub fn multiple_shooting_variant() -> (&'static str, &'static str) {
    ("multiple_shooting", "Multiple Shooting")
}

pub fn multiple_shooting_variant_with_sx(key: MultipleShootingCompileKey) -> (String, String) {
    let (base_id, base_label) = multiple_shooting_variant();
    with_sx_variant_suffix(base_id, base_label, key.sx_functions)
}

pub fn direct_collocation_compile_key(family: CollocationFamily) -> DirectCollocationCompileKey {
    match family {
        CollocationFamily::GaussLegendre => DirectCollocationCompileKey::Legendre,
        CollocationFamily::RadauIIA => DirectCollocationCompileKey::RadauIia,
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
    with_sx_variant_suffix(base_id, base_label, key.sx_functions)
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

pub fn append_standard_compile_cache_statuses<Ms, Dc, MsTiming, DcTiming>(
    statuses: &mut Vec<CompileCacheStatus>,
    problem_id: ProblemId,
    problem_name: &str,
    multiple_shooting_cache: &SharedCompileCache<MultipleShootingCompileKey, Ms>,
    direct_collocation_cache: &SharedCompileCache<DirectCollocationCompileVariantKey, Dc>,
    multiple_shooting_timing_of: MsTiming,
    direct_collocation_timing_of: DcTiming,
) where
    MsTiming: Fn(&Ms) -> BackendTimingMetadata,
    DcTiming: Fn(&Dc) -> BackendTimingMetadata,
{
    statuses.extend(collect_compile_cache_statuses(
        problem_id,
        problem_name,
        multiple_shooting_cache,
        multiple_shooting_variant_with_sx,
        multiple_shooting_timing_of,
    ));
    statuses.extend(collect_compile_cache_statuses(
        problem_id,
        problem_name,
        direct_collocation_cache,
        direct_collocation_variant_with_sx,
        direct_collocation_timing_of,
    ));
}

pub fn collect_compile_cache_statuses<K, V, F, G>(
    problem_id: ProblemId,
    problem_name: &str,
    cache: &SharedCompileCache<K, V>,
    describe_variant: G,
    timing_of: F,
) -> Vec<CompileCacheStatus>
where
    K: Eq + Hash + Copy,
    F: Fn(&V) -> BackendTimingMetadata,
    G: Fn(K) -> (String, String),
{
    cache
        .cached_entries()
        .into_iter()
        .map(|(key, compiled)| {
            let (variant_id, variant_label) = describe_variant(key);
            compile_cache_status(
                problem_id,
                problem_name,
                &variant_id,
                &variant_label,
                timing_of(&compiled.borrow()),
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
    CompileProgressInfo {
        timing,
        compile_cached: false,
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
    family: CollocationFamily,
    sx_functions: OcpSxFunctionConfig,
    build: Build,
) -> Result<CachedCompile<Compiled>>
where
    Build: FnOnce(OcpCompileOptions) -> std::result::Result<Compiled, E>,
    E: Into<anyhow::Error>,
{
    cache.get_or_try_init(
        direct_collocation_compile_key_with_sx(family, sx_functions),
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
    family: CollocationFamily,
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
        direct_collocation_compile_key_with_sx(family, sx_functions),
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
            compile_cached: true,
        });
    }
    Ok((
        cached.compiled,
        CompileProgressInfo {
            compile_cached: cached.was_cached,
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
    }
}

pub fn default_sqp_config() -> SqpConfig {
    SqpConfig {
        max_iters: 200,
        dual_tol: 5.0e-2,
        constraint_tol: 1.0e-8,
        complementarity_tol: 1.0e-6,
    }
}

pub fn default_solver_method() -> SolverMethod {
    SolverMethod::Sqp
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

pub fn transcription_controls(
    default: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
) -> Vec<ControlSpec> {
    let mut controls = vec![
        select_control(
            SharedControlId::TranscriptionIntervals.id(),
            "Intervals",
            default.intervals as f64,
            "",
            "Number of mesh intervals. The current build exposes the compiled interval count only.",
            &supported_intervals
                .iter()
                .map(|value| (*value as f64, value.to_string()))
                .collect::<Vec<_>>(),
            ControlSection::Transcription,
            ControlVisibility::Always,
            ControlSemantic::TranscriptionIntervals,
        ),
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
            "Number of collocation nodes per interval when direct collocation is selected. The current build exposes the compiled degree only.",
            &supported_degrees
                .iter()
                .map(|value| (*value as f64, value.to_string()))
                .collect::<Vec<_>>(),
            ControlSection::Transcription,
            ControlVisibility::DirectCollocationOnly,
            ControlSemantic::CollocationDegree,
        ),
    ];
    controls.extend(ocp_sx_function_controls(OcpSxFunctionConfig::default()));
    controls
}

pub fn solver_controls(default_method: SolverMethod, default: SqpConfig) -> Vec<ControlSpec> {
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
        text_control(
            SharedControlId::SolverMaxIterations.id(),
            "Max Iterations",
            default.max_iters as f64,
            "",
            "Maximum nonlinear iterations before the selected solver terminates.",
            ControlSemantic::SolverMaxIterations,
            ControlValueDisplay::Integer,
        ),
        text_control(
            SharedControlId::SolverDualTolerance.id(),
            "Dual Tolerance",
            default.dual_tol,
            "",
            "Termination threshold on the dual infeasibility norm.",
            ControlSemantic::SolverDualTolerance,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverConstraintTolerance.id(),
            "Constraint Tolerance",
            default.constraint_tol,
            "",
            "Termination threshold on equality and inequality infeasibility.",
            ControlSemantic::SolverConstraintTolerance,
            ControlValueDisplay::Scientific,
        ),
        text_control(
            SharedControlId::SolverComplementarityTolerance.id(),
            "Complementarity Tolerance",
            default.complementarity_tol,
            "",
            "Termination threshold on complementarity residuals.",
            ControlSemantic::SolverComplementarityTolerance,
            ControlValueDisplay::Scientific,
        ),
    ]
}

fn solver_method_choices() -> Vec<(f64, &'static str)> {
    let mut out = vec![(0.0, "SQP"), (1.0, "NLIP")];
    #[cfg(feature = "ipopt")]
    out.push((2.0, "IPOPT"));
    out
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
    default: SqpConfig,
) -> Result<SqpConfig> {
    let max_iters = expect_nonnegative_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverMaxIterations,
            default.max_iters as f64,
        ),
        SharedControlId::SolverMaxIterations.id(),
    )?
    .round() as usize;
    let dual_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverDualTolerance,
            default.dual_tol,
        ),
        SharedControlId::SolverDualTolerance.id(),
    )?;
    let constraint_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverConstraintTolerance,
            default.constraint_tol,
        ),
        SharedControlId::SolverConstraintTolerance.id(),
    )?;
    let complementarity_tol = expect_positive_finite(
        sample_shared_or_default(
            values,
            SharedControlId::SolverComplementarityTolerance,
            default.complementarity_tol,
        ),
        SharedControlId::SolverComplementarityTolerance.id(),
    )?;
    Ok(SqpConfig {
        max_iters,
        dual_tol,
        constraint_tol,
        complementarity_tol,
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
    if !supported_intervals.contains(&intervals) {
        return Err(anyhow!(
            "unsupported {} {}",
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

    Ok(TranscriptionConfig {
        method,
        intervals,
        collocation_degree,
        collocation_family,
    })
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

pub fn sqp_options(config: &SqpConfig) -> ClarabelSqpOptions {
    ClarabelSqpOptions {
        max_iters: config.max_iters,
        dual_tol: config.dual_tol,
        constraint_tol: config.constraint_tol,
        complementarity_tol: config.complementarity_tol,
        ..ClarabelSqpOptions::default()
    }
}

pub fn nlip_options(config: &SqpConfig) -> InteriorPointOptions {
    InteriorPointOptions {
        max_iters: config.max_iters,
        dual_tol: config.dual_tol,
        constraint_tol: config.constraint_tol,
        complementarity_tol: config.complementarity_tol,
        ..InteriorPointOptions::default()
    }
}

#[cfg(feature = "ipopt")]
pub fn ipopt_options(config: &SqpConfig) -> IpoptOptions {
    let tol = config
        .dual_tol
        .min(config.constraint_tol)
        .min(config.complementarity_tol);
    IpoptOptions {
        max_iters: config.max_iters,
        tol,
        acceptable_tol: Some((100.0 * tol).max(tol)),
        constraint_tol: Some(config.constraint_tol),
        complementarity_tol: Some(config.complementarity_tol),
        dual_tol: Some(config.dual_tol),
        print_level: 5,
        suppress_banner: false,
        ..IpoptOptions::default()
    }
}

pub fn transcription_metrics(config: &TranscriptionConfig) -> [Metric; 3] {
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
    ]
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
        max: default,
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
    }
}

pub fn problem_controls(
    transcription: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
    solver_method: SolverMethod,
    solver: SqpConfig,
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

pub fn deg_to_rad(value: f64) -> f64 {
    value.to_radians()
}

pub fn rad_to_deg(value: f64) -> f64 {
    value.to_degrees()
}

pub fn node_times<const N: usize>(tf: f64) -> Vec<f64> {
    let step = tf / N as f64;
    (0..=N).map(|index| index as f64 * step).collect()
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

fn try_attach_multiple_shooting_constraint_panels<Compiled, const N: usize>(
    artifact: &mut SolveArtifact,
    compiled: &Compiled,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
    >,
    trajectories: &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum, N>,
    tolerance: f64,
) -> Result<()>
where
    Compiled: MultipleShootingCompiled<N>,
{
    let report = compiled.build_constraint_violation_report(runtime, trajectories, tolerance)?;
    attach_constraint_panels(artifact, report);
    Ok(())
}

fn try_attach_direct_collocation_constraint_panels<Compiled, const N: usize, const K: usize>(
    artifact: &mut SolveArtifact,
    compiled: &Compiled,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
        K,
    >,
    trajectories: &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum, N, K>,
    tolerance: f64,
) -> Result<()>
where
    Compiled: DirectCollocationCompiled<N, K>,
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

pub trait MultipleShootingCompiled<const N: usize>: CompiledOcpMetadata {
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
            N,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<optimal_control::MultipleShootingSqpSolveResult<Self::XNum, Self::UNum, N>>;

    fn run_sqp_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<optimal_control::MultipleShootingSqpSolveResult<Self::XNum, Self::UNum, N>>
    where
        CB: FnMut(&MultipleShootingSqpSnapshot<Self::XNum, Self::UNum, N>);

    fn run_nlip(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
        >,
        options: &InteriorPointOptions,
    ) -> Result<optimal_control::MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum, N>>;

    fn run_nlip_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<optimal_control::MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum, N>>
    where
        CB: FnMut(&MultipleShootingInteriorPointSnapshot<Self::XNum, Self::UNum, N>);

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
            N,
        >,
        options: &IpoptOptions,
    ) -> Result<optimal_control::MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum, N>>;

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
            N,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<optimal_control::MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum, N>>
    where
        CB: FnMut(&MultipleShootingIpoptSnapshot<Self::XNum, Self::UNum, N>);

    fn build_interval_arcs(
        &self,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum, N>,
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
            N,
        >,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum, N>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport>;

    fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind);
}

pub trait DirectCollocationCompiled<const N: usize, const K: usize>: CompiledOcpMetadata {
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
            N,
            K,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<optimal_control::DirectCollocationSqpSolveResult<Self::XNum, Self::UNum, N, K>>;

    fn run_sqp_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
            K,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<optimal_control::DirectCollocationSqpSolveResult<Self::XNum, Self::UNum, N, K>>
    where
        CB: FnMut(&DirectCollocationSqpSnapshot<Self::XNum, Self::UNum, N, K>);

    fn run_nlip(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
            K,
        >,
        options: &InteriorPointOptions,
    ) -> Result<
        optimal_control::DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum, N, K>,
    >;

    fn run_nlip_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
            K,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<
        optimal_control::DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum, N, K>,
    >
    where
        CB: FnMut(&DirectCollocationInteriorPointSnapshot<Self::XNum, Self::UNum, N, K>);

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
            N,
            K,
        >,
        options: &IpoptOptions,
    ) -> Result<optimal_control::DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum, N, K>>;

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
            N,
            K,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<optimal_control::DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum, N, K>>
    where
        CB: FnMut(&DirectCollocationIpoptSnapshot<Self::XNum, Self::UNum, N, K>);

    fn build_constraint_violation_report(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
            K,
        >,
        trajectories: &DirectCollocationTrajectories<Self::XNum, Self::UNum, N, K>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport>;

    fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Self::PNum,
            Self::CBounds,
            Self::BeqNum,
            Self::BineqBounds,
            Self::XNum,
            Self::UNum,
            N,
            K,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind);
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const RK4_SUBSTEPS: usize> CompiledOcpMetadata
    for CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    optimal_control::Mesh<X, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<X, N>>,
    optimal_control::Mesh<U, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<U, N>>,
    [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
    [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
    [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    <([X; N], [U; N]) as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    (
        optimal_control::Mesh<X, N>,
        optimal_control::Mesh<U, N>,
        [U; N],
        SX,
    ): Vectorize<
            SX,
            Rebind<SX> = (
                optimal_control::Mesh<X, N>,
                optimal_control::Mesh<U, N>,
                [U; N],
                SX,
            ),
            Rebind<f64> = (
                optimal_control::Mesh<Numeric<X>, N>,
                optimal_control::Mesh<Numeric<U>, N>,
                [Numeric<U>; N],
                f64,
            ),
        >,
    ([X; N], [U; N]): Vectorize<SX, Rebind<SX> = ([X; N], [U; N])>,
    (Beq, Bineq, [C; N]): Vectorize<
            SX,
            Rebind<SX> = (Beq, Bineq, [C; N]),
            Rebind<f64> = (Numeric<Beq>, Numeric<Bineq>, [Numeric<C>; N]),
        >,
    (P, Beq): Vectorize<SX, Rebind<SX> = (P, Beq), Rebind<f64> = (Numeric<P>, Numeric<Beq>)>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    Numeric<Bineq>: Vectorize<f64, Rebind<f64> = Numeric<Bineq>>,
    Numeric<C>: Vectorize<f64, Rebind<f64> = Numeric<C>>,
    <C as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <C as Vectorize<SX>>::Rebind<Bounds1D>>,
    <Bineq as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>>,
{
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        Self::backend_timing_metadata(self)
    }

    fn nlp_compile_stats(&self) -> NlpCompileStats {
        Self::nlp_compile_stats(self)
    }

    fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        Self::helper_compile_stats(self)
    }

    fn helper_kernel_count(&self) -> usize {
        Self::helper_kernel_count(self)
    }

    fn backend_compile_report(&self) -> &BackendCompileReport {
        Self::backend_compile_report(self)
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const RK4_SUBSTEPS: usize> MultipleShootingCompiled<N>
    for CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    optimal_control::Mesh<X, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<X, N>>,
    optimal_control::Mesh<U, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<U, N>>,
    [C; N]: Vectorize<SX, Rebind<SX> = [C; N]>,
    [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
    [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    <([X; N], [U; N]) as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    (
        optimal_control::Mesh<X, N>,
        optimal_control::Mesh<U, N>,
        [U; N],
        SX,
    ): Vectorize<
            SX,
            Rebind<SX> = (
                optimal_control::Mesh<X, N>,
                optimal_control::Mesh<U, N>,
                [U; N],
                SX,
            ),
            Rebind<f64> = (
                optimal_control::Mesh<Numeric<X>, N>,
                optimal_control::Mesh<Numeric<U>, N>,
                [Numeric<U>; N],
                f64,
            ),
        >,
    ([X; N], [U; N]): Vectorize<SX, Rebind<SX> = ([X; N], [U; N])>,
    (Beq, Bineq, [C; N]): Vectorize<
            SX,
            Rebind<SX> = (Beq, Bineq, [C; N]),
            Rebind<f64> = (Numeric<Beq>, Numeric<Bineq>, [Numeric<C>; N]),
        >,
    (P, Beq): Vectorize<SX, Rebind<SX> = (P, Beq), Rebind<f64> = (Numeric<P>, Numeric<Beq>)>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    Numeric<Bineq>: Vectorize<f64, Rebind<f64> = Numeric<Bineq>>,
    Numeric<C>: Vectorize<f64, Rebind<f64> = Numeric<C>>,
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
            N,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<optimal_control::MultipleShootingSqpSolveResult<Self::XNum, Self::UNum, N>> {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::solve_sqp(
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
            N,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<optimal_control::MultipleShootingSqpSolveResult<Self::XNum, Self::UNum, N>>
    where
        CB: FnMut(&MultipleShootingSqpSnapshot<Self::XNum, Self::UNum, N>),
    {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::solve_sqp_with_callback(
            self,
            values,
            options,
            callback,
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
            N,
        >,
        options: &InteriorPointOptions,
    ) -> Result<optimal_control::MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum, N>>
    {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::solve_interior_point(
            self,
            values,
            options,
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
            N,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<optimal_control::MultipleShootingInteriorPointSolveResult<Self::XNum, Self::UNum, N>>
    where
        CB: FnMut(&MultipleShootingInteriorPointSnapshot<Self::XNum, Self::UNum, N>),
    {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::solve_interior_point_with_callback(
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
            N,
        >,
        options: &IpoptOptions,
    ) -> Result<optimal_control::MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum, N>> {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::solve_ipopt(
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
            N,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<optimal_control::MultipleShootingIpoptSolveResult<Self::XNum, Self::UNum, N>>
    where
        CB: FnMut(&MultipleShootingIpoptSnapshot<Self::XNum, Self::UNum, N>),
    {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::solve_ipopt_with_callback(
            self,
            values,
            options,
            callback,
        )
        .map_err(Into::into)
    }

    fn build_interval_arcs(
        &self,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum, N>,
        parameters: &Self::PNum,
    ) -> Result<(Vec<IntervalArc<Self::XNum>>, Vec<IntervalArc<Self::UNum>>)> {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::interval_arcs(
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
            N,
        >,
        trajectories: &MultipleShootingTrajectories<Self::XNum, Self::UNum, N>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport> {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::rank_constraint_violations(
            self,
            values,
            trajectories,
            tolerance,
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
            N,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        CompiledMultipleShootingOcp::<X, U, P, C, Beq, Bineq, N, RK4_SUBSTEPS>::benchmark_nlp_evaluations_with_progress(
            self,
            values,
            options,
            on_progress,
        )
        .map_err(Into::into)
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const K: usize> CompiledOcpMetadata
    for CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    optimal_control::Mesh<X, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<X, N>>,
    optimal_control::Mesh<U, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<U, N>>,
    optimal_control::IntervalGrid<X, N, K>:
        Vectorize<SX, Rebind<SX> = optimal_control::IntervalGrid<X, N, K>>,
    optimal_control::IntervalGrid<U, N, K>:
        Vectorize<SX, Rebind<SX> = optimal_control::IntervalGrid<U, N, K>>,
    optimal_control::IntervalGrid<C, N, K>:
        Vectorize<SX, Rebind<SX> = optimal_control::IntervalGrid<C, N, K>>,
    (
        optimal_control::Mesh<X, N>,
        optimal_control::Mesh<U, N>,
        optimal_control::IntervalGrid<X, N, K>,
        optimal_control::IntervalGrid<U, N, K>,
        optimal_control::IntervalGrid<U, N, K>,
        SX,
    ): Vectorize<
            SX,
            Rebind<SX> = (
                optimal_control::Mesh<X, N>,
                optimal_control::Mesh<U, N>,
                optimal_control::IntervalGrid<X, N, K>,
                optimal_control::IntervalGrid<U, N, K>,
                optimal_control::IntervalGrid<U, N, K>,
                SX,
            ),
            Rebind<f64> = (
                optimal_control::Mesh<Numeric<X>, N>,
                optimal_control::Mesh<Numeric<U>, N>,
                optimal_control::IntervalGrid<Numeric<X>, N, K>,
                optimal_control::IntervalGrid<Numeric<U>, N, K>,
                optimal_control::IntervalGrid<Numeric<U>, N, K>,
                f64,
            ),
        >,
    (
        [X; N],
        [U; N],
        optimal_control::IntervalGrid<X, N, K>,
        optimal_control::IntervalGrid<U, N, K>,
    ): Vectorize<
            SX,
            Rebind<SX> = (
                [X; N],
                [U; N],
                optimal_control::IntervalGrid<X, N, K>,
                optimal_control::IntervalGrid<U, N, K>,
            ),
        >,
    (Beq, Bineq, optimal_control::IntervalGrid<C, N, K>): Vectorize<
            SX,
            Rebind<SX> = (Beq, Bineq, optimal_control::IntervalGrid<C, N, K>),
            Rebind<f64> = (
                Numeric<Beq>,
                Numeric<Bineq>,
                optimal_control::IntervalGrid<Numeric<C>, N, K>,
            ),
        >,
    (P, Beq): Vectorize<SX, Rebind<SX> = (P, Beq), Rebind<f64> = (Numeric<P>, Numeric<Beq>)>,
    [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
    [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    <optimal_control::IntervalGrid<X, N, K> as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <optimal_control::IntervalGrid<U, N, K> as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <[X; N] as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <[U; N] as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    Numeric<Bineq>: Vectorize<f64, Rebind<f64> = Numeric<Bineq>>,
    Numeric<C>: Vectorize<f64, Rebind<f64> = Numeric<C>>,
    <C as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <C as Vectorize<SX>>::Rebind<Bounds1D>>,
    <Bineq as Vectorize<SX>>::Rebind<Bounds1D>:
        Vectorize<Bounds1D, Rebind<Bounds1D> = <Bineq as Vectorize<SX>>::Rebind<Bounds1D>>,
{
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        Self::backend_timing_metadata(self)
    }

    fn nlp_compile_stats(&self) -> NlpCompileStats {
        Self::nlp_compile_stats(self)
    }

    fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        Self::helper_compile_stats(self)
    }

    fn helper_kernel_count(&self) -> usize {
        Self::helper_kernel_count(self)
    }

    fn backend_compile_report(&self) -> &BackendCompileReport {
        Self::backend_compile_report(self)
    }
}

impl<X, U, P, C, Beq, Bineq, const N: usize, const K: usize> DirectCollocationCompiled<N, K>
    for CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, N, K>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    optimal_control::Mesh<X, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<X, N>>,
    optimal_control::Mesh<U, N>: Vectorize<SX, Rebind<SX> = optimal_control::Mesh<U, N>>,
    optimal_control::IntervalGrid<X, N, K>:
        Vectorize<SX, Rebind<SX> = optimal_control::IntervalGrid<X, N, K>>,
    optimal_control::IntervalGrid<U, N, K>:
        Vectorize<SX, Rebind<SX> = optimal_control::IntervalGrid<U, N, K>>,
    optimal_control::IntervalGrid<C, N, K>:
        Vectorize<SX, Rebind<SX> = optimal_control::IntervalGrid<C, N, K>>,
    (
        optimal_control::Mesh<X, N>,
        optimal_control::Mesh<U, N>,
        optimal_control::IntervalGrid<X, N, K>,
        optimal_control::IntervalGrid<U, N, K>,
        optimal_control::IntervalGrid<U, N, K>,
        SX,
    ): Vectorize<
            SX,
            Rebind<SX> = (
                optimal_control::Mesh<X, N>,
                optimal_control::Mesh<U, N>,
                optimal_control::IntervalGrid<X, N, K>,
                optimal_control::IntervalGrid<U, N, K>,
                optimal_control::IntervalGrid<U, N, K>,
                SX,
            ),
            Rebind<f64> = (
                optimal_control::Mesh<Numeric<X>, N>,
                optimal_control::Mesh<Numeric<U>, N>,
                optimal_control::IntervalGrid<Numeric<X>, N, K>,
                optimal_control::IntervalGrid<Numeric<U>, N, K>,
                optimal_control::IntervalGrid<Numeric<U>, N, K>,
                f64,
            ),
        >,
    (
        [X; N],
        [U; N],
        optimal_control::IntervalGrid<X, N, K>,
        optimal_control::IntervalGrid<U, N, K>,
    ): Vectorize<
            SX,
            Rebind<SX> = (
                [X; N],
                [U; N],
                optimal_control::IntervalGrid<X, N, K>,
                optimal_control::IntervalGrid<U, N, K>,
            ),
        >,
    (Beq, Bineq, optimal_control::IntervalGrid<C, N, K>): Vectorize<
            SX,
            Rebind<SX> = (Beq, Bineq, optimal_control::IntervalGrid<C, N, K>),
            Rebind<f64> = (
                Numeric<Beq>,
                Numeric<Bineq>,
                optimal_control::IntervalGrid<Numeric<C>, N, K>,
            ),
        >,
    (P, Beq): Vectorize<SX, Rebind<SX> = (P, Beq), Rebind<f64> = (Numeric<P>, Numeric<Beq>)>,
    [X; N]: Vectorize<SX, Rebind<SX> = [X; N]>,
    [U; N]: Vectorize<SX, Rebind<SX> = [U; N]>,
    <optimal_control::IntervalGrid<X, N, K> as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <optimal_control::IntervalGrid<U, N, K> as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <[X; N] as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    <[U; N] as Vectorize<SX>>::Rebind<f64>: Vectorize<f64>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    Numeric<Bineq>: Vectorize<f64, Rebind<f64> = Numeric<Bineq>>,
    Numeric<C>: Vectorize<f64, Rebind<f64> = Numeric<C>>,
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
            N,
            K,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<optimal_control::DirectCollocationSqpSolveResult<Self::XNum, Self::UNum, N, K>>
    {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::solve_sqp(
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
            N,
            K,
        >,
        options: &ClarabelSqpOptions,
        callback: CB,
    ) -> Result<optimal_control::DirectCollocationSqpSolveResult<Self::XNum, Self::UNum, N, K>>
    where
        CB: FnMut(&DirectCollocationSqpSnapshot<Self::XNum, Self::UNum, N, K>),
    {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::solve_sqp_with_callback(
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
            N,
            K,
        >,
        options: &InteriorPointOptions,
    ) -> Result<
        optimal_control::DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum, N, K>,
    > {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::solve_interior_point(
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
            N,
            K,
        >,
        options: &InteriorPointOptions,
        callback: CB,
    ) -> Result<
        optimal_control::DirectCollocationInteriorPointSolveResult<Self::XNum, Self::UNum, N, K>,
    >
    where
        CB: FnMut(&DirectCollocationInteriorPointSnapshot<Self::XNum, Self::UNum, N, K>),
    {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::solve_interior_point_with_callback(
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
            N,
            K,
        >,
        options: &IpoptOptions,
    ) -> Result<optimal_control::DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum, N, K>>
    {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::solve_ipopt(
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
            N,
            K,
        >,
        options: &IpoptOptions,
        callback: CB,
    ) -> Result<optimal_control::DirectCollocationIpoptSolveResult<Self::XNum, Self::UNum, N, K>>
    where
        CB: FnMut(&DirectCollocationIpoptSnapshot<Self::XNum, Self::UNum, N, K>),
    {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::solve_ipopt_with_callback(
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
            N,
            K,
        >,
        trajectories: &DirectCollocationTrajectories<Self::XNum, Self::UNum, N, K>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport> {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::rank_constraint_violations(
            self,
            values,
            trajectories,
            tolerance,
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
            N,
            K,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> Result<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        CompiledDirectCollocationOcp::<X, U, P, C, Beq, Bineq, N, K>::benchmark_nlp_evaluations_with_progress(
            self,
            values,
            options,
            on_progress,
        )
        .map_err(Into::into)
    }
}

pub fn solve_multiple_shooting_problem<Compiled, Build, const N: usize>(
    compiled: &Compiled,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    mut build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled<N>,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum, N>,
        &[IntervalArc<Compiled::XNum>],
        &[IntervalArc<Compiled::UNum>],
    ) -> SolveArtifact,
{
    let started = Instant::now();
    match solver_method {
        SolverMethod::Sqp => {
            let solved = compiled.run_sqp(runtime, &sqp_options(solver_config))?;
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
            attach_sqp_solver_report(&mut artifact, &solved.solver);
            Ok(with_solve_time(artifact, started))
        }
        SolverMethod::Nlip => {
            let solved = compiled.run_nlip(runtime, &nlip_options(solver_config))?;
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
            attach_nlip_solver_report(&mut artifact, &solved.solver);
            Ok(with_solve_time(artifact, started))
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let solved = compiled.run_ipopt(runtime, &ipopt_options(solver_config))?;
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
            attach_ipopt_solver_report(&mut artifact, &solved.solver);
            Ok(with_solve_time(artifact, started))
        }
    }
}

pub fn solve_cached_multiple_shooting_problem<Compiled, Build, const N: usize>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled<N>,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum, N>,
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

pub fn solve_multiple_shooting_problem_with_progress<Compiled, Emit, Build, const N: usize>(
    compiled: &Compiled,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled<N>,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum, N>,
        &[IntervalArc<Compiled::XNum>],
        &[IntervalArc<Compiled::UNum>],
    ) -> SolveArtifact,
{
    let started = Instant::now();
    let emit = Arc::new(Mutex::new(emit));
    let build_artifact = Arc::new(Mutex::new(build_artifact));
    match solver_method {
        SolverMethod::Sqp => {
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
                    compiled.run_sqp_with_callback(
                        runtime,
                        &sqp_options(solver_config),
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
                    if let Some(report) = error
                        .downcast_ref::<ClarabelSqpError>()
                        .and_then(|typed| sqp_failure_solver_report(typed, &running_solver))
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
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver);
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
                    compiled.run_nlip_with_callback(
                        runtime,
                        &nlip_options(solver_config),
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
                    if let Some(report) = error
                        .downcast_ref::<InteriorPointSolveError>()
                        .and_then(|typed| nlip_failure_solver_report(typed, &running_solver))
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
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver);
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
                    if let Some(report) = error
                        .downcast_ref::<IpoptSolveError>()
                        .and_then(|typed| ipopt_failure_solver_report(typed, &running_solver))
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
            attach_ipopt_solver_report(&mut artifact, &solved.solver);
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

pub fn solve_cached_multiple_shooting_problem_with_progress<Compiled, Emit, Build, const N: usize>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &MultipleShootingRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: MultipleShootingCompiled<N>,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &MultipleShootingTrajectories<Compiled::XNum, Compiled::UNum, N>,
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

pub fn solve_direct_collocation_problem<Compiled, Build, const N: usize, const K: usize>(
    compiled: &Compiled,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
        K,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    mut build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled<N, K>,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum, N, K>,
        &DirectCollocationTimeGrid<N, K>,
    ) -> SolveArtifact,
{
    let started = Instant::now();
    match solver_method {
        SolverMethod::Sqp => {
            let solved = compiled.run_sqp(runtime, &sqp_options(solver_config))?;
            let mut artifact = build_artifact(&solved.trajectories, &solved.time_grid);
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver);
            Ok(with_solve_time(artifact, started))
        }
        SolverMethod::Nlip => {
            let solved = compiled.run_nlip(runtime, &nlip_options(solver_config))?;
            let mut artifact = build_artifact(&solved.trajectories, &solved.time_grid);
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver);
            Ok(with_solve_time(artifact, started))
        }
        #[cfg(feature = "ipopt")]
        SolverMethod::Ipopt => {
            let solved = compiled.run_ipopt(runtime, &ipopt_options(solver_config))?;
            let mut artifact = build_artifact(&solved.trajectories, &solved.time_grid);
            let _ = try_attach_direct_collocation_constraint_panels(
                &mut artifact,
                compiled,
                runtime,
                &solved.trajectories,
                solver_config.constraint_tol,
            );
            append_ipopt_termination_metric(&mut artifact, &solved.solver);
            attach_ipopt_solver_report(&mut artifact, &solved.solver);
            Ok(with_solve_time(artifact, started))
        }
    }
}

pub fn solve_cached_direct_collocation_problem<Compiled, Build, const N: usize, const K: usize>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
        K,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled<N, K>,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum, N, K>,
        &DirectCollocationTimeGrid<N, K>,
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

pub fn solve_direct_collocation_problem_with_progress<
    Compiled,
    Emit,
    Build,
    const N: usize,
    const K: usize,
>(
    compiled: &Compiled,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
        K,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled<N, K>,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum, N, K>,
        &DirectCollocationTimeGrid<N, K>,
    ) -> SolveArtifact,
{
    let started = Instant::now();
    let emit = Arc::new(Mutex::new(emit));
    let build_artifact = Arc::new(Mutex::new(build_artifact));
    match solver_method {
        SolverMethod::Sqp => {
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
                    compiled.run_sqp_with_callback(
                        runtime,
                        &sqp_options(solver_config),
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
                    if let Some(report) = error
                        .downcast_ref::<ClarabelSqpError>()
                        .and_then(|typed| sqp_failure_solver_report(typed, &running_solver))
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
            append_termination_metric(&mut artifact, &solved.solver);
            attach_sqp_solver_report(&mut artifact, &solved.solver);
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
                    compiled.run_nlip_with_callback(
                        runtime,
                        &nlip_options(solver_config),
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
                    if let Some(report) = error
                        .downcast_ref::<InteriorPointSolveError>()
                        .and_then(|typed| nlip_failure_solver_report(typed, &running_solver))
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
            append_nlip_termination_metric(&mut artifact, &solved.solver);
            attach_nlip_solver_report(&mut artifact, &solved.solver);
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
                    if let Some(report) = error
                        .downcast_ref::<IpoptSolveError>()
                        .and_then(|typed| ipopt_failure_solver_report(typed, &running_solver))
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
            attach_ipopt_solver_report(&mut artifact, &solved.solver);
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

pub fn solve_cached_direct_collocation_problem_with_progress<
    Compiled,
    Emit,
    Build,
    const N: usize,
    const K: usize,
>(
    compiled: &Rc<RefCell<Compiled>>,
    runtime: &DirectCollocationRuntimeValues<
        Compiled::PNum,
        Compiled::CBounds,
        Compiled::BeqNum,
        Compiled::BineqBounds,
        Compiled::XNum,
        Compiled::UNum,
        N,
        K,
    >,
    solver_method: SolverMethod,
    solver_config: &SqpConfig,
    emit: Emit,
    running_solver: SolverReport,
    build_artifact: Build,
) -> Result<SolveArtifact>
where
    Compiled: DirectCollocationCompiled<N, K>,
    Emit: FnMut(SolveStreamEvent) + Send,
    Build: FnMut(
        &DirectCollocationTrajectories<Compiled::XNum, Compiled::UNum, N, K>,
        &DirectCollocationTimeGrid<N, K>,
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
    }
}

pub fn nlip_progress(snapshot: &InteriorPointIterationSnapshot) -> SolveProgress {
    SolveProgress {
        iteration: snapshot.iteration,
        phase: match snapshot.phase {
            optimization::InteriorPointIterationPhase::Initial => SolvePhase::Initial,
            optimization::InteriorPointIterationPhase::AcceptedStep => SolvePhase::AcceptedStep,
            optimization::InteriorPointIterationPhase::Converged => SolvePhase::Converged,
        },
        objective: snapshot.objective,
        eq_inf: snapshot.eq_inf,
        ineq_inf: snapshot.ineq_inf,
        dual_inf: snapshot.dual_inf,
        step_inf: None,
        penalty: snapshot.barrier_parameter.unwrap_or(0.0),
        alpha: snapshot.alpha,
        line_search_iterations: snapshot.line_search_iterations.map(|value| value as usize),
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

pub fn nlip_termination_label(_: &InteriorPointSummary) -> String {
    "Converged".to_string()
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

pub fn sqp_solver_report(summary: &ClarabelSqpSummary) -> SolverReport {
    SolverReport {
        completed: true,
        status_label: sqp_termination_label(summary),
        status_kind: sqp_status_kind(summary),
        iterations: Some(summary.iterations),
        symbolic_setup_s: symbolic_setup_seconds(summary.profiling.backend_timing),
        jit_s: duration_seconds(summary.profiling.backend_timing.jit_time),
        solve_s: Some(summary.profiling.total_time.as_secs_f64()),
        compile_cached: false,
        phase_details: SolverPhaseDetails::default(),
    }
}

pub fn nlip_solver_report(summary: &InteriorPointSummary) -> SolverReport {
    SolverReport {
        completed: true,
        status_label: "Converged".to_string(),
        status_kind: SolverStatusKind::Success,
        iterations: Some(summary.iterations),
        symbolic_setup_s: symbolic_setup_seconds(summary.profiling.backend_timing),
        jit_s: duration_seconds(summary.profiling.backend_timing.jit_time),
        solve_s: Some(summary.profiling.total_time.as_secs_f64()),
        compile_cached: false,
        phase_details: SolverPhaseDetails::default(),
    }
}

#[cfg(feature = "ipopt")]
pub fn ipopt_solver_report(summary: &IpoptSummary) -> SolverReport {
    SolverReport {
        completed: true,
        status_label: ipopt_status_label(summary.status),
        status_kind: ipopt_status_kind(summary.status),
        iterations: Some(summary.iterations),
        symbolic_setup_s: symbolic_setup_seconds(summary.profiling.backend_timing),
        jit_s: duration_seconds(summary.profiling.backend_timing.jit_time),
        solve_s: Some(summary.profiling.total_time.as_secs_f64()),
        compile_cached: false,
        phase_details: SolverPhaseDetails::default(),
    }
}

pub fn attach_sqp_solver_report(artifact: &mut SolveArtifact, summary: &ClarabelSqpSummary) {
    artifact.solver = sqp_solver_report(summary);
}

pub fn attach_nlip_solver_report(artifact: &mut SolveArtifact, summary: &InteriorPointSummary) {
    artifact.solver = nlip_solver_report(summary);
}

#[cfg(feature = "ipopt")]
pub fn attach_ipopt_solver_report(artifact: &mut SolveArtifact, summary: &IpoptSummary) {
    artifact.solver = ipopt_solver_report(summary);
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

fn sqp_solve_phase_details(profiling: &ClarabelSqpProfiling) -> Vec<SolverPhaseDetail> {
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
        "Convergence Check",
        profiling.convergence_checks,
        profiling.convergence_check_time,
    );
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
        "Preprocess",
        profiling.preprocessing_steps,
        profiling.preprocessing_time,
    );
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
    report.phase_details = phase_details;
    report
}

pub fn sqp_failure_solver_report(
    error: &ClarabelSqpError,
    fallback: &SolverReport,
) -> Option<SolverReport> {
    let context = match error {
        ClarabelSqpError::MaxIterations { context, .. }
        | ClarabelSqpError::QpSolve { context, .. }
        | ClarabelSqpError::UnconstrainedStepSolve { context }
        | ClarabelSqpError::LineSearchFailed { context, .. }
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
            phase_details: SolverPhaseDetails::default(),
        },
        fallback,
        sqp_solve_phase_details(&context.profiling),
    ))
}

pub fn nlip_failure_solver_report(
    error: &InteriorPointSolveError,
    fallback: &SolverReport,
) -> Option<SolverReport> {
    let (status_label, iterations, profiling) = match error {
        InteriorPointSolveError::InvalidInput(_) => return None,
        InteriorPointSolveError::LinearSolve { solver, profiling } => (
            format!("Failed: linear solve ({})", solver.label()),
            None,
            profiling.as_ref(),
        ),
        InteriorPointSolveError::LineSearchFailed { profiling, .. } => {
            ("Failed: line search".to_string(), None, profiling.as_ref())
        }
        InteriorPointSolveError::MaxIterations {
            iterations,
            profiling,
        } => (
            "Failed: maximum iterations reached".to_string(),
            Some(*iterations as usize),
            profiling.as_ref(),
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
            phase_details: SolverPhaseDetails::default(),
        },
        fallback,
        nlip_solve_phase_details(profiling),
    ))
}

#[cfg(feature = "ipopt")]
pub fn ipopt_failure_solver_report(
    error: &IpoptSolveError,
    fallback: &SolverReport,
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
            phase_details: SolverPhaseDetails::default(),
        },
        fallback,
        ipopt_solve_phase_details(profiling),
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolveRequest {
    pub values: BTreeMap<String, f64>,
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
        assert_eq!(ode.section, ControlSection::Transcription);
        assert_eq!(ode.panel, Some(ControlPanel::SxFunctions));
        assert_eq!(ode.semantic, ControlSemantic::SxFunctionOption);
    }

    #[test]
    fn sx_function_config_variant_suffix_changes_for_custom_settings() {
        let default = OcpSxFunctionConfig::default();
        let custom = OcpSxFunctionConfig {
            global_call_policy: CallPolicy::NoInlineLLVM,
            multiple_shooting_integrator: OcpKernelStrategy::FunctionUseGlobalPolicy,
            ..default
        };
        assert_ne!(default.variant_id_suffix(), custom.variant_id_suffix());
        assert_eq!(custom.variant_label_suffix().as_deref(), Some("SXF Custom"));
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
            profiling: Box::new(profiling),
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
            nlip_failure_solver_report(&error, &fallback).expect("expected failure report");

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
                .any(|detail| detail.label == "Preprocess" && detail.count == 2)
        );
    }
}
