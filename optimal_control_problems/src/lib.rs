mod common;
pub mod crane;
pub mod glider;
pub mod linear_s;
pub mod sailboat;

use std::collections::BTreeMap;

use anyhow::Result;
use common::FromMap;
pub use common::{
    Chart, ControlChoice, ControlEditor, ControlSection, ControlSemantic, ControlSpec,
    ControlValueDisplay, ControlVisibility, LatexSection, Metric, MetricKey, PlotMode, ProblemId,
    ProblemSpec, Scene2D, SceneAnimation, SceneArrow, SceneCircle, SceneFrame, ScenePath,
    SolveArtifact, SolveLogLevel, SolvePhase, SolveProgress, SolveRequest, SolveStreamEvent,
    SolverMethod, SolverReport, SolverStatusKind, TimeSeries, TimeSeriesRole, TranscriptionConfig,
    TranscriptionMethod, find_metric, metric, metric_with_key, numeric_metric_with_key,
};

pub fn problem_specs() -> Vec<ProblemSpec> {
    vec![
        glider::spec(),
        linear_s::spec(),
        sailboat::spec(),
        crane::spec(),
    ]
}

pub fn solve_problem(id: ProblemId, values: &BTreeMap<String, f64>) -> Result<SolveArtifact> {
    match id {
        ProblemId::OptimalDistanceGlider => glider::solve(&glider::Params::from_map(values)?),
        ProblemId::LinearSManeuver => linear_s::solve(&linear_s::Params::from_map(values)?),
        ProblemId::SailboatUpwind => sailboat::solve(&sailboat::Params::from_map(values)?),
        ProblemId::CraneTransfer => crane::solve(&crane::Params::from_map(values)?),
    }
}

pub fn solve_problem_with_progress<F>(
    id: ProblemId,
    values: &BTreeMap<String, f64>,
    emit: F,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    match id {
        ProblemId::OptimalDistanceGlider => {
            glider::solve_with_progress(&glider::Params::from_map(values)?, emit)
        }
        ProblemId::LinearSManeuver => {
            linear_s::solve_with_progress(&linear_s::Params::from_map(values)?, emit)
        }
        ProblemId::SailboatUpwind => {
            sailboat::solve_with_progress(&sailboat::Params::from_map(values)?, emit)
        }
        ProblemId::CraneTransfer => {
            crane::solve_with_progress(&crane::Params::from_map(values)?, emit)
        }
    }
}
