#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::collapsible_if)]

mod benchmark_report;
mod common;
pub mod crane;
pub mod glider;
pub mod linear_s;
pub mod sailboat;

use std::collections::BTreeMap;
use std::sync::LazyLock;

use anyhow::Result;
pub use benchmark_report::{
    OcpBenchmarkCase, OcpBenchmarkPreset, OcpBenchmarkProgress, OcpBenchmarkRecord,
    OcpBenchmarkSuite, OcpBenchmarkSuiteConfig, default_benchmark_jobs,
    render_ocp_benchmark_report, run_ocp_benchmark_suite, run_ocp_benchmark_suite_with_progress,
    write_ocp_benchmark_report,
};
pub use common::{
    Chart, CompileCacheState, CompileCacheStatus, ControlChoice, ControlEditor, ControlPanel,
    ControlSection, ControlSemantic, ControlSpec, ControlValueDisplay, ControlVisibility,
    DerivativeCheckOrder, DerivativeCheckRequest, DirectCollocationCompileKey, LatexSection,
    Metric, MetricKey, OcpKernelStrategy, OcpOverrideBehavior, OcpSxFunctionConfig, PlotMode,
    ProblemDerivativeCheck, ProblemId, ProblemSpec, Scene2D, SceneAnimation, SceneArrow,
    SceneCircle, SceneFrame, ScenePath, SolveArtifact, SolveLogLevel, SolvePhase, SolveProgress,
    SolveRequest, SolveStage, SolveStatus, SolveStreamEvent, SolverMethod, SolverReport,
    SolverStatusKind, TimeSeries, TimeSeriesRole, TranscriptionConfig, TranscriptionMethod,
    direct_collocation_variant, direct_collocation_variant_with_sx, find_metric, metric,
    metric_with_key, multiple_shooting_variant, multiple_shooting_variant_with_sx,
    numeric_metric_with_key, ocp_sx_function_config_from_map_lossy,
};

pub(crate) struct ProblemEntry {
    pub(crate) id: ProblemId,
    pub(crate) spec: fn() -> ProblemSpec,
    pub(crate) solve_from_map: fn(&BTreeMap<String, f64>) -> Result<SolveArtifact>,
    pub(crate) prewarm_from_map: fn(&BTreeMap<String, f64>) -> Result<()>,
    pub(crate) validate_derivatives_from_request:
        fn(&common::DerivativeCheckRequest) -> Result<common::ProblemDerivativeCheck>,
    pub(crate) solve_with_progress_boxed: fn(
        &BTreeMap<String, f64>,
        Box<dyn FnMut(SolveStreamEvent) + Send>,
    ) -> Result<SolveArtifact>,
    pub(crate) prewarm_with_progress_boxed:
        fn(&BTreeMap<String, f64>, Box<dyn FnMut(SolveStreamEvent) + Send>) -> Result<()>,
    pub(crate) compile_cache_statuses: fn() -> Vec<CompileCacheStatus>,
    pub(crate) benchmark_default_case_with_progress: fn(
        TranscriptionMethod,
        OcpBenchmarkPreset,
        optimization::NlpEvaluationBenchmarkOptions,
        &mut dyn FnMut(benchmark_report::BenchmarkCaseProgress),
    ) -> Result<OcpBenchmarkRecord>,
}

fn problem_entries() -> &'static [ProblemEntry] {
    static ENTRIES: LazyLock<Vec<ProblemEntry>> = LazyLock::new(|| {
        vec![
            glider::problem_entry(),
            linear_s::problem_entry(),
            sailboat::problem_entry(),
            crane::problem_entry(),
        ]
    });
    &ENTRIES
}

fn problem_entry(id: ProblemId) -> &'static ProblemEntry {
    problem_entries()
        .iter()
        .find(|entry| entry.id == id)
        .expect("problem entry should exist")
}

pub fn problem_specs() -> Vec<ProblemSpec> {
    problem_entries()
        .iter()
        .map(|entry| (entry.spec)())
        .collect()
}

pub fn solve_problem(id: ProblemId, values: &BTreeMap<String, f64>) -> Result<SolveArtifact> {
    (problem_entry(id).solve_from_map)(values)
}

pub fn prewarm_problem(id: ProblemId, values: &BTreeMap<String, f64>) -> Result<()> {
    (problem_entry(id).prewarm_from_map)(values)
}

pub fn validate_problem_derivatives(
    id: ProblemId,
    request: &DerivativeCheckRequest,
) -> Result<ProblemDerivativeCheck> {
    (problem_entry(id).validate_derivatives_from_request)(request)
}

pub fn prewarm_problem_with_progress<F>(
    id: ProblemId,
    values: &BTreeMap<String, f64>,
    emit: F,
) -> Result<()>
where
    F: FnMut(SolveStreamEvent) + Send + 'static,
{
    (problem_entry(id).prewarm_with_progress_boxed)(values, Box::new(emit))
}

pub fn compile_cache_statuses() -> Vec<CompileCacheStatus> {
    problem_entries()
        .iter()
        .flat_map(|entry| (entry.compile_cache_statuses)())
        .collect()
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
    let sx_functions =
        ocp_sx_function_config_from_map_lossy(values, OcpSxFunctionConfig::default());
    let (variant_id, variant_label) = if method.round() as i32 == 0 {
        multiple_shooting_variant_with_sx(common::multiple_shooting_compile_key(0, sx_functions))
    } else {
        let family = spec
            .controls
            .iter()
            .find(|control| control.semantic == ControlSemantic::CollocationFamily)
            .map(|control| values.get(&control.id).copied().unwrap_or(control.default))
            .unwrap_or(0.0);
        if family.round() as i32 == 1 {
            direct_collocation_variant_with_sx(common::DirectCollocationCompileVariantKey {
                family: DirectCollocationCompileKey::RadauIia,
                sx_functions,
            })
        } else {
            direct_collocation_variant_with_sx(common::DirectCollocationCompileVariantKey {
                family: DirectCollocationCompileKey::Legendre,
                sx_functions,
            })
        }
    };
    Some((variant_id, variant_label))
}

pub fn solve_problem_with_progress<F>(
    id: ProblemId,
    values: &BTreeMap<String, f64>,
    emit: F,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send + 'static,
{
    (problem_entry(id).solve_with_progress_boxed)(values, Box::new(emit))
}

pub(crate) fn benchmark_problem_case_with_progress(
    id: ProblemId,
    transcription: TranscriptionMethod,
    preset: OcpBenchmarkPreset,
    eval_options: optimization::NlpEvaluationBenchmarkOptions,
    on_progress: &mut dyn FnMut(benchmark_report::BenchmarkCaseProgress),
) -> Result<OcpBenchmarkRecord> {
    (problem_entry(id).benchmark_default_case_with_progress)(
        transcription,
        preset,
        eval_options,
        on_progress,
    )
}
