#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::collapsible_if)]

pub mod albatross;
mod benchmark_report;
mod common;
pub mod crane;
pub mod glider;
pub mod linear_s;
pub mod sailboat;
mod static_optimization;

use std::cell::RefCell;
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
    ArtifactVisualization, Chart, CompileCacheState, CompileCacheStatus, ControlChoice,
    ControlEditor, ControlPanel, ControlSection, ControlSemantic, ControlSpec, ControlValueDisplay,
    ControlVisibility, DerivativeCheckOrder, DerivativeCheckRequest, DirectCollocationCompileKey,
    LatexSection, Metric, MetricKey, OcpKernelStrategy, OcpOverrideBehavior, OcpSxFunctionConfig,
    PlotMode, ProblemDerivativeCheck, ProblemId, ProblemSpec, Scene2D, SceneAnimation, SceneArrow,
    SceneCircle, SceneFrame, ScenePath, ScenePath3D, SolveArtifact, SolveLogLevel, SolvePhase,
    SolveProgress, SolveRequest, SolveStage, SolveStatus, SolveStreamEvent, SolverMethod,
    SolverPhaseDetail, SolverPhaseDetails, SolverReport, SolverStatusKind, TimeGrid, TimeSeries,
    TimeSeriesRole, TranscriptionConfig, TranscriptionMethod, direct_collocation_variant,
    direct_collocation_variant_with_sx, find_metric, metric, metric_with_key,
    multiple_shooting_variant, multiple_shooting_variant_with_sx, numeric_metric_with_key,
    ocp_sx_function_config_from_map_lossy, time_grid_from_map_lossy,
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

type SolveCancellationCheck = Box<dyn Fn() -> bool + Send>;

thread_local! {
    static SOLVE_CANCELLATION_CHECK: RefCell<Option<SolveCancellationCheck>> =
        RefCell::new(None);
}

struct SolveCancellationGuard {
    previous: Option<SolveCancellationCheck>,
}

impl Drop for SolveCancellationGuard {
    fn drop(&mut self) {
        SOLVE_CANCELLATION_CHECK.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

pub fn with_solve_cancellation_check<R, F, C>(should_continue: C, run: F) -> R
where
    F: FnOnce() -> R,
    C: Fn() -> bool + Send + 'static,
{
    let previous = SOLVE_CANCELLATION_CHECK.with(|slot| {
        slot.borrow_mut()
            .replace(Box::new(should_continue) as SolveCancellationCheck)
    });
    let _guard = SolveCancellationGuard { previous };
    run()
}

pub(crate) fn solve_should_continue() -> bool {
    SOLVE_CANCELLATION_CHECK
        .with(|slot| slot.borrow().as_ref().map(|check| check()).unwrap_or(true))
}

fn problem_entries() -> &'static [ProblemEntry] {
    static ENTRIES: LazyLock<Vec<ProblemEntry>> = LazyLock::new(|| {
        vec![
            albatross::problem_entry(),
            glider::problem_entry(),
            linear_s::problem_entry(),
            sailboat::problem_entry(),
            crane::problem_entry(),
            static_optimization::hanging_chain_problem_entry(),
            static_optimization::rosenbrock_problem_entry(),
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
    if id == ProblemId::AlbatrossDynamicSoaring {
        return albatross::compile_variant_for_values(values);
    }
    if let Some(variant) = static_optimization::compile_variant_for_problem(id) {
        return Some(variant);
    }
    let spec = problem_specs().into_iter().find(|spec| spec.id == id)?;
    let control_value = |semantic: ControlSemantic, fallback: f64| {
        spec.controls
            .iter()
            .find(|control| control.semantic == semantic)
            .map(|control| values.get(&control.id).copied().unwrap_or(control.default))
            .unwrap_or(fallback)
    };
    let method = control_value(ControlSemantic::TranscriptionMethod, 0.0);
    let intervals = control_value(ControlSemantic::TranscriptionIntervals, 0.0)
        .round()
        .max(0.0) as usize;
    let sx_functions =
        ocp_sx_function_config_from_map_lossy(values, OcpSxFunctionConfig::default());
    let (variant_id, variant_label) = if method.round() as i32 == 0 {
        multiple_shooting_variant_with_sx(common::multiple_shooting_compile_key(
            intervals,
            sx_functions,
        ))
    } else {
        let family = control_value(ControlSemantic::CollocationFamily, 0.0);
        let order = control_value(ControlSemantic::CollocationDegree, 0.0)
            .round()
            .max(0.0) as usize;
        let time_grid =
            common::time_grid_compile_key(time_grid_from_map_lossy(values, TimeGrid::default()));
        if family.round() as i32 == 1 {
            direct_collocation_variant_with_sx(common::DirectCollocationCompileVariantKey {
                intervals,
                order,
                family: DirectCollocationCompileKey::RadauIia,
                time_grid,
                sx_functions,
            })
        } else {
            direct_collocation_variant_with_sx(common::DirectCollocationCompileVariantKey {
                intervals,
                order,
                family: DirectCollocationCompileKey::Legendre,
                time_grid,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_variant_includes_multiple_shooting_interval_count() {
        let mut values = BTreeMap::new();
        values.insert("transcription_method".to_string(), 0.0);
        values.insert("transcription_intervals".to_string(), 20.0);
        let (variant_20, label_20) =
            compile_variant_for_problem(ProblemId::LinearSManeuver, &values)
                .expect("variant should exist");

        values.insert("transcription_intervals".to_string(), 40.0);
        let (variant_40, label_40) =
            compile_variant_for_problem(ProblemId::LinearSManeuver, &values)
                .expect("variant should exist");

        assert_ne!(variant_20, variant_40);
        assert!(variant_20.contains("__n20"));
        assert!(variant_40.contains("__n40"));
        assert!(label_20.contains("20 intervals"));
        assert!(label_40.contains("40 intervals"));
    }

    #[test]
    fn compile_variant_includes_direct_collocation_interval_and_order() {
        let mut values = BTreeMap::new();
        values.insert("transcription_method".to_string(), 1.0);
        values.insert("transcription_intervals".to_string(), 20.0);
        values.insert("collocation_degree".to_string(), 2.0);
        let (variant_k2, label_k2) =
            compile_variant_for_problem(ProblemId::LinearSManeuver, &values)
                .expect("variant should exist");

        values.insert("collocation_degree".to_string(), 4.0);
        let (variant_k4, label_k4) =
            compile_variant_for_problem(ProblemId::LinearSManeuver, &values)
                .expect("variant should exist");

        assert_ne!(variant_k2, variant_k4);
        assert!(variant_k2.contains("__n20_k2"));
        assert!(variant_k4.contains("__n20_k4"));
        assert!(label_k2.contains("20 intervals"));
        assert!(label_k2.contains("2 nodes"));
        assert!(label_k4.contains("4 nodes"));
    }
}
