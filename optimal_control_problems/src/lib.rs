mod common;
pub mod crane;
pub mod glider;
pub mod linear_s;
pub mod sailboat;

use std::collections::BTreeMap;

use anyhow::Result;
use common::FromMap;
pub use common::{
    Chart, CompileCacheState, CompileCacheStatus, ControlChoice, ControlEditor, ControlSection,
    ControlSemantic, ControlSpec, ControlValueDisplay, ControlVisibility,
    DirectCollocationCompileKey, LatexSection, Metric, MetricKey, PlotMode, ProblemId, ProblemSpec,
    Scene2D, SceneAnimation, SceneArrow, SceneCircle, SceneFrame, ScenePath, SolveArtifact,
    SolveLogLevel, SolvePhase, SolveProgress, SolveRequest, SolveStage, SolveStatus,
    SolveStreamEvent, SolverMethod, SolverReport, SolverStatusKind, TimeSeries, TimeSeriesRole,
    TranscriptionConfig, TranscriptionMethod, direct_collocation_variant, find_metric, metric,
    metric_with_key, multiple_shooting_variant, numeric_metric_with_key,
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

pub fn prewarm_problem(id: ProblemId, values: &BTreeMap<String, f64>) -> Result<()> {
    match id {
        ProblemId::OptimalDistanceGlider => glider::prewarm(&glider::Params::from_map(values)?),
        ProblemId::LinearSManeuver => linear_s::prewarm(&linear_s::Params::from_map(values)?),
        ProblemId::SailboatUpwind => sailboat::prewarm(&sailboat::Params::from_map(values)?),
        ProblemId::CraneTransfer => crane::prewarm(&crane::Params::from_map(values)?),
    }
}

pub fn prewarm_problem_with_progress<F>(
    id: ProblemId,
    values: &BTreeMap<String, f64>,
    emit: F,
) -> Result<()>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    match id {
        ProblemId::OptimalDistanceGlider => {
            glider::prewarm_with_progress(&glider::Params::from_map(values)?, emit)
        }
        ProblemId::LinearSManeuver => {
            linear_s::prewarm_with_progress(&linear_s::Params::from_map(values)?, emit)
        }
        ProblemId::SailboatUpwind => {
            sailboat::prewarm_with_progress(&sailboat::Params::from_map(values)?, emit)
        }
        ProblemId::CraneTransfer => {
            crane::prewarm_with_progress(&crane::Params::from_map(values)?, emit)
        }
    }
}

pub fn compile_cache_statuses() -> Vec<CompileCacheStatus> {
    let mut statuses = Vec::new();
    statuses.extend(glider::compile_cache_statuses());
    statuses.extend(linear_s::compile_cache_statuses());
    statuses.extend(sailboat::compile_cache_statuses());
    statuses.extend(crane::compile_cache_statuses());
    statuses
}

pub fn compile_variant_for_problem(
    id: ProblemId,
    values: &BTreeMap<String, f64>,
) -> Option<(String, String)> {
    let spec = problem_specs().into_iter().find(|spec| spec.id == id)?;
    let method = spec
        .controls
        .iter()
        .find(|control| control.semantic == ControlSemantic::TranscriptionMethod)
        .map(|control| values.get(&control.id).copied().unwrap_or(control.default))
        .unwrap_or(0.0);
    let (variant_id, variant_label) = if method.round() as i32 == 0 {
        multiple_shooting_variant()
    } else {
        let family = spec
            .controls
            .iter()
            .find(|control| control.semantic == ControlSemantic::CollocationFamily)
            .map(|control| values.get(&control.id).copied().unwrap_or(control.default))
            .unwrap_or(0.0);
        if family.round() as i32 == 1 {
            direct_collocation_variant(DirectCollocationCompileKey::RadauIia)
        } else {
            direct_collocation_variant(DirectCollocationCompileKey::Legendre)
        }
    };
    Some((variant_id.to_string(), variant_label.to_string()))
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
