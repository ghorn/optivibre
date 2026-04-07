use std::collections::BTreeMap;
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
    MultipleShootingSqpSnapshot, MultipleShootingTrajectories, OcpConstraintCategory,
    OcpConstraintViolationReport,
};
#[cfg(feature = "ipopt")]
use optimal_control::{DirectCollocationIpoptSnapshot, MultipleShootingIpoptSnapshot};
use optimization::{
    BackendTimingMetadata, ClarabelSqpOptions, ClarabelSqpSummary, ConstraintSatisfaction,
    InteriorPointIterationSnapshot, InteriorPointOptions, InteriorPointSummary, SqpIterationEvent,
    SqpIterationPhase, SqpIterationSnapshot, Vectorize,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolveStreamEvent {
    Status {
        message: String,
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

pub const SYMBOLIC_SETUP_STATUS: &str = "Setting up symbolic model...";

pub fn emit_symbolic_setup_status<F>(emit: &mut F)
where
    F: FnMut(SolveStreamEvent),
{
    emit(SolveStreamEvent::Status {
        message: SYMBOLIC_SETUP_STATUS.to_string(),
    });
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        }
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

pub fn transcription_controls(
    default: TranscriptionConfig,
    supported_intervals: &[usize],
    supported_degrees: &[usize],
) -> Vec<ControlSpec> {
    vec![
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
                    message: "Running SQP...".to_string(),
                },
            );
            let emit_for_worker = emit.clone();
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
                                        level: SolveLogLevel::Warning,
                                    });
                                }
                                submit.submit(SolveStreamEvent::Iteration {
                                    progress: sqp_progress(&snapshot.solver),
                                    artifact,
                                });
                            }
                            Err(error) => submit.submit(SolveStreamEvent::Log {
                                line: format!("[iteration visualization failed: {error}]"),
                                level: SolveLogLevel::Warning,
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
                    message: "Running NLIP solver...".to_string(),
                },
            );
            let emit_for_worker = emit.clone();
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
                                        level: SolveLogLevel::Warning,
                                    });
                                }
                                submit.submit(SolveStreamEvent::Iteration {
                                    progress: nlip_progress(&snapshot.solver),
                                    artifact,
                                });
                            }
                            Err(error) => submit.submit(SolveStreamEvent::Log {
                                line: format!("[iteration visualization failed: {error}]"),
                                level: SolveLogLevel::Warning,
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
                    message: "Running IPOPT...".to_string(),
                },
            );
            let emit_for_worker = emit.clone();
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
                                        level: SolveLogLevel::Warning,
                                    });
                                }
                                submit.submit(SolveStreamEvent::Iteration {
                                    progress: ipopt_progress(&snapshot.solver),
                                    artifact,
                                });
                            }
                            Err(error) => submit.submit(SolveStreamEvent::Log {
                                line: format!("[iteration visualization failed: {error}]"),
                                level: SolveLogLevel::Warning,
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
                    message: "Running SQP...".to_string(),
                },
            );
            let emit_for_worker = emit.clone();
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
                            if let Err(error) = try_attach_direct_collocation_constraint_panels(
                                &mut artifact,
                                compiled,
                                runtime,
                                &snapshot.trajectories,
                                solver_config.constraint_tol,
                            ) {
                                submit.submit(SolveStreamEvent::Log {
                                    line: format!("[constraint violation report failed: {error}]"),
                                    level: SolveLogLevel::Warning,
                                });
                            }
                            submit.submit(SolveStreamEvent::Iteration {
                                progress: sqp_progress(&snapshot.solver),
                                artifact,
                            });
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
                    message: "Running NLIP solver...".to_string(),
                },
            );
            let emit_for_worker = emit.clone();
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
                            if let Err(error) = try_attach_direct_collocation_constraint_panels(
                                &mut artifact,
                                compiled,
                                runtime,
                                &snapshot.trajectories,
                                solver_config.constraint_tol,
                            ) {
                                submit.submit(SolveStreamEvent::Log {
                                    line: format!("[constraint violation report failed: {error}]"),
                                    level: SolveLogLevel::Warning,
                                });
                            }
                            submit.submit(SolveStreamEvent::Iteration {
                                progress: nlip_progress(&snapshot.solver),
                                artifact,
                            });
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
                    message: "Running IPOPT...".to_string(),
                },
            );
            let emit_for_worker = emit.clone();
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
                            if let Err(error) = try_attach_direct_collocation_constraint_panels(
                                &mut artifact,
                                compiled,
                                runtime,
                                &snapshot.trajectories,
                                solver_config.constraint_tol,
                            ) {
                                submit.submit(SolveStreamEvent::Log {
                                    line: format!("[constraint violation report failed: {error}]"),
                                    level: SolveLogLevel::Warning,
                                });
                            }
                            submit.submit(SolveStreamEvent::Iteration {
                                progress: ipopt_progress(&snapshot.solver),
                                artifact,
                            });
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
