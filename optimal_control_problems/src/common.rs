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
    DirectCollocationInteriorPointSnapshot, DirectCollocationRuntimeValues,
    DirectCollocationSqpSnapshot, DirectCollocationTimeGrid, DirectCollocationTrajectories,
    IntervalArc, MultipleShootingInteriorPointSnapshot, MultipleShootingRuntimeValues,
    MultipleShootingSqpSnapshot, MultipleShootingTrajectories, OcpCompileHelperKind,
    OcpCompileProgress, OcpConstraintCategory, OcpConstraintViolationReport, OcpHelperCompileStats,
};
#[cfg(feature = "ipopt")]
use optimal_control::{DirectCollocationIpoptSnapshot, MultipleShootingIpoptSnapshot};
use optimization::{
    BackendTimingMetadata, ClarabelSqpOptions, ClarabelSqpSummary, ConstraintSatisfaction,
    InteriorPointIterationSnapshot, InteriorPointOptions, InteriorPointSummary,
    LlvmOptimizationLevel, NlpCompileStats, SqpIterationEvent, SqpIterationPhase,
    SqpIterationSnapshot, Vectorize,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptOptions, IpoptRawStatus, IpoptSummary};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedControlId {
    TranscriptionMethod,
    TranscriptionIntervals,
    CollocationFamily,
    CollocationDegree,
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
            Self::SolverMethod => "solver_method",
            Self::SolverMaxIterations => "solver_max_iters",
            Self::SolverDualTolerance => "solver_dual_tol",
            Self::SolverConstraintTolerance => "solver_constraint_tol",
            Self::SolverComplementarityTolerance => "solver_complementarity_tol",
        }
    }
}

type Numeric<T> = <T as Vectorize<SX>>::Rebind<f64>;

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

#[macro_export]
macro_rules! standard_ocp_compile_caches {
    ($multiple_shooting_cache:ident : $multiple_shooting_ty:ty, $direct_collocation_cache:ident : $direct_collocation_ty:ty) => {
        thread_local! {
            static $multiple_shooting_cache: std::cell::RefCell<$crate::common::SharedCompileCache<usize, $multiple_shooting_ty>> =
                std::cell::RefCell::new($crate::common::SharedCompileCache::new());
            static $direct_collocation_cache: std::cell::RefCell<
                $crate::common::SharedCompileCache<$crate::common::DirectCollocationCompileKey, $direct_collocation_ty>
            > = std::cell::RefCell::new($crate::common::SharedCompileCache::new());
        }
    };
}

#[macro_export]
macro_rules! standard_ocp_compile_cache_statuses {
    ($problem_id:expr, $problem_name:expr, $multiple_shooting_cache:ident, $direct_collocation_cache:ident) => {{
        let mut statuses = Vec::new();
        $multiple_shooting_cache.with(|cache| {
            $direct_collocation_cache.with(|dc_cache| {
                $crate::common::append_standard_compile_cache_statuses(
                    &mut statuses,
                    $problem_id,
                    $problem_name,
                    &cache.borrow(),
                    &dc_cache.borrow(),
                    |compiled| compiled.backend_timing_metadata(),
                    |compiled| compiled.backend_timing_metadata(),
                );
            });
        });
        statuses
    }};
}

fn phase_detail(label: impl Into<String>, value: impl Into<String>) -> SolverPhaseDetail {
    SolverPhaseDetail {
        label: label.into(),
        value: value.into(),
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

pub fn append_standard_compile_cache_statuses<Ms, Dc, MsTiming, DcTiming>(
    statuses: &mut Vec<CompileCacheStatus>,
    problem_id: ProblemId,
    problem_name: &str,
    multiple_shooting_cache: &SharedCompileCache<usize, Ms>,
    direct_collocation_cache: &SharedCompileCache<DirectCollocationCompileKey, Dc>,
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
        |_| multiple_shooting_variant(),
        multiple_shooting_timing_of,
    ));
    statuses.extend(collect_compile_cache_statuses(
        problem_id,
        problem_name,
        direct_collocation_cache,
        direct_collocation_variant,
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
    G: Fn(K) -> (&'static str, &'static str),
{
    cache
        .cached_entries()
        .into_iter()
        .map(|(key, compiled)| {
            let (variant_id, variant_label) = describe_variant(key);
            compile_cache_status(
                problem_id,
                problem_name,
                variant_id,
                variant_label,
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
        details.push(phase_detail(label, value));
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

fn symbolic_phase_details(stats: NlpCompileStats) -> Vec<SolverPhaseDetail> {
    let total_jacobian_nnz = stats.equality_jacobian_nnz + stats.inequality_jacobian_nnz;
    vec![
        phase_detail("Vars", stats.variable_count.to_string()),
        phase_detail("Params", stats.parameter_scalar_count.to_string()),
        phase_detail("Eq", stats.equality_count.to_string()),
        phase_detail("Ineq", stats.inequality_count.to_string()),
        phase_detail("Jac NNZ", total_jacobian_nnz.to_string()),
        phase_detail("Hess NNZ", stats.hessian_nnz.to_string()),
    ]
}

fn jit_phase_details(stats: NlpCompileStats, helper_kernel_count: usize) -> Vec<SolverPhaseDetail> {
    vec![
        phase_detail("NLP Kernels", stats.jit_kernel_count.to_string()),
        phase_detail("Helper Kernels", helper_kernel_count.to_string()),
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
        details.push(phase_detail("Xdot Helper", format_phase_duration(duration)));
    }
    if let Some(duration) = helper_stats.multiple_shooting_arc_helper_time {
        details.push(phase_detail(
            "RK4 Arc Helper",
            format_phase_duration(duration),
        ));
    }
    details
}

pub fn ocp_compile_progress_update(
    progress: OcpCompileProgress,
    state: &mut OcpCompileProgressState,
) -> CompileProgressUpdate {
    match progress {
        OcpCompileProgress::SymbolicReady(metadata) => {
            state.timing = metadata.timing;
            state.phase_details.symbolic_setup = symbolic_phase_details(metadata.stats);
            state.phase_details.jit = vec![phase_detail(
                "NLP Kernels",
                metadata.stats.jit_kernel_count.to_string(),
            )];
        }
        OcpCompileProgress::HelperCompiled { helper, elapsed } => {
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
) -> CompileProgressInfo {
    let mut jit = jit_phase_details(stats, helper_kernel_count);
    jit.extend(helper_compile_phase_details(helper_stats));
    CompileProgressInfo {
        timing,
        compile_cached: false,
        phase_details: SolverPhaseDetails {
            symbolic_setup: symbolic_phase_details(stats),
            jit,
            solve: Vec::new(),
        },
    }
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
    ) -> Result<(Compiled, SolverReport)>
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
        Ok((compiled, running_solver))
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

pub fn transcription_controls(
    default: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
) -> Vec<ControlSpec> {
    vec![
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
    ]
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

pub trait MultipleShootingCompiled<const N: usize> {
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
}

pub trait DirectCollocationCompiled<const N: usize, const K: usize> {
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
            let running_solver = running_solver.clone();
            let solved = with_latest_only_worker(
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
                                artifact.solver = running_solver
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
            )?;
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
            let running_solver = running_solver.clone();
            let solved = with_latest_only_worker(
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
                                artifact.solver = running_solver
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
            )?;
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
            let running_solver = running_solver.clone();
            let solved = with_latest_only_worker(
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
                                artifact.solver = running_solver
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
            )?;
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
            let running_solver = running_solver.clone();
            let solved = with_latest_only_worker(
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
                            artifact.solver = running_solver
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
            )?;
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
            let running_solver = running_solver.clone();
            let solved = with_latest_only_worker(
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
                            artifact.solver = running_solver
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
            )?;
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
            let running_solver = running_solver.clone();
            let solved = with_latest_only_worker(
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
                            artifact.solver = running_solver
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
            )?;
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
        let (compiled, running_solver) = {
            let mut reporter =
                SolveLifecycleReporter::new(|event| events.push(event), SolverMethod::Sqp);
            let result = reporter
                .compile_with_progress(|on_symbolic_ready| {
                    on_symbolic_ready(CompileProgressUpdate {
                        timing: symbolic_timing,
                        phase_details: SolverPhaseDetails {
                            symbolic_setup: vec![phase_detail("Vars", "2")],
                            jit: vec![phase_detail("NLP Kernels", "3")],
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
                                symbolic_setup: vec![phase_detail("Vars", "2")],
                                jit: vec![phase_detail("NLP Kernels", "3")],
                                solve: Vec::new(),
                            },
                        },
                    ))
                })
                .expect("compile should succeed");
            let _ = reporter.into_emit();
            result
        };

        assert_eq!(compiled, "compiled");
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
                            symbolic_setup: vec![phase_detail("Vars", "2")],
                            jit: vec![phase_detail("NLP Kernels", "3")],
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
                                symbolic_setup: vec![phase_detail("Vars", "2")],
                                jit: vec![phase_detail("NLP Kernels", "3")],
                                solve: Vec::new(),
                            },
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
}
