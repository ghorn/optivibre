mod dashboard;
mod manifest;
mod model;
mod problems;
mod registry;
mod report;
mod runner;
mod transcript;

pub use dashboard::{render_dashboard_html, write_dashboard};
pub use manifest::{
    DEFAULT_MAX_ITERS, IterationLimits, KnownStatus, ProblemManifestEntry, ProblemSpeed,
    ProblemTestSet, burkardt_manifest_entry, manifest_entries, manifest_entry,
    manifest_entry_by_id, slow_burkardt_manifest_entry, slow_manifest_entry,
};
pub use model::{
    CallPolicyMode, JitOptLevel, NlipLinearSolverMode, ProblemCase, ProblemDescriptor,
    ProblemRunOptions, ProblemRunRecord, ResultCacheInfo, ResultCacheStatus, RunArtifacts,
    RunStatus, SolverKind, SolverMetrics, SolverTimingBreakdown, ValidationOutcome,
};
pub use registry::registry;
pub use report::{
    render_html_report, render_markdown_report, render_terminal_report, write_html_report,
    write_json_report,
};
pub use runner::{
    PlannedRunTask, RunCacheOptions, RunPreviewEntry, RunProgressEvent, RunRequest, RunResults,
    RunStage, TEST_PROBLEMS_BUILD_FINGERPRINT, cached_result_records, clear_result_cache,
    default_result_cache_dir, planned_run_tasks, preview_run, run_cases, run_cases_with_cache,
};
pub use transcript::write_transcript_artifacts;
