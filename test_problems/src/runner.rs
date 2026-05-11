use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::manifest::{
    KnownStatus, ProblemManifestEntry, ProblemSpeed, ProblemTestSet, manifest_entry_by_id,
};
use crate::model::{
    CallPolicyMode, JitOptLevel, ProblemCase, ProblemDescriptor, ProblemRunOptions,
    ProblemRunRecord, ResultCacheInfo, ResultCacheStatus, RunStatus, SolverKind, ValidationOutcome,
    ValidationTier,
};
use crate::registry::registry;

pub const TEST_PROBLEMS_CACHE_SCHEMA_VERSION: u32 = 1;
pub const TEST_PROBLEMS_BUILD_FINGERPRINT: &str = env!("TEST_PROBLEMS_BUILD_FINGERPRINT");
const TEST_PROBLEMS_CACHE_ENV: &str = "OPTIVIBRE_TEST_PROBLEMS_CACHE_DIR";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRequest {
    pub problem_ids: Option<Vec<String>>,
    pub solvers: Vec<SolverKind>,
    pub run_options: Vec<ProblemRunOptions>,
    pub jobs: Option<usize>,
    pub include_skipped: bool,
    pub problem_set: Option<ProblemSpeed>,
    pub test_set: Option<ProblemTestSet>,
    pub progress: bool,
}

impl Default for RunRequest {
    fn default() -> Self {
        Self {
            problem_ids: None,
            solvers: vec![SolverKind::Nlip],
            run_options: vec![ProblemRunOptions {
                jit_opt_level: JitOptLevel::O3,
                call_policy: CallPolicyMode::InlineAtLowering,
            }],
            jobs: None,
            include_skipped: false,
            problem_set: None,
            test_set: None,
            progress: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunResults {
    pub records: Vec<ProblemRunRecord>,
}

impl RunResults {
    pub fn total(&self) -> usize {
        self.records.len()
    }
}

#[derive(Clone, Debug)]
pub struct RunCacheOptions {
    pub enabled: bool,
    pub force: bool,
    pub cache_dir: PathBuf,
}

impl Default for RunCacheOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            force: false,
            cache_dir: default_result_cache_dir(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "event")]
pub enum RunProgressEvent {
    Queued {
        total: usize,
    },
    CacheHit {
        completed: usize,
        total: usize,
        problem: String,
        solver: SolverKind,
    },
    Running {
        completed: usize,
        total: usize,
        problem: String,
        solver: SolverKind,
    },
    Completed {
        completed: usize,
        total: usize,
        problem: String,
        solver: SolverKind,
        status: RunStatus,
        cache_status: ResultCacheStatus,
    },
    Failed {
        completed: usize,
        total: usize,
        problem: String,
        solver: SolverKind,
        reason: String,
    },
    Finished {
        total: usize,
        accepted: usize,
        failed: usize,
    },
    Stage {
        stage: RunStage,
        completed: usize,
        total: usize,
        active: usize,
        desired_jobs: usize,
        message: String,
    },
    JobsChanged {
        jobs: usize,
    },
    StopRequested {
        completed: usize,
        total: usize,
        active: usize,
    },
    Stopped {
        total: usize,
        completed: usize,
        accepted: usize,
        failed: usize,
    },
}

pub type RunProgressSink = dyn Fn(RunProgressEvent) + Send + Sync;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStage {
    Planning,
    Queued,
    Solving,
    Stopping,
    CollectingResults,
    RefreshingCache,
    Complete,
    Stopped,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunPreviewEntry {
    pub problem_id: String,
    pub test_set: ProblemTestSet,
    pub family: String,
    pub solver: SolverKind,
    pub problem_speed: ProblemSpeed,
    pub cache_status: ResultCacheStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlannedRunTask {
    pub problem_id: String,
    pub solver: SolverKind,
    pub options: ProblemRunOptions,
    pub expected: KnownStatus,
    pub max_iters_limit: usize,
    pub problem_speed: ProblemSpeed,
    pub test_set: ProblemTestSet,
}

pub fn run_cases(request: &RunRequest) -> Result<RunResults> {
    run_cases_uncached(request, None)
}

pub fn run_cases_with_cache(
    request: &RunRequest,
    cache_options: &RunCacheOptions,
    progress_sink: Option<&RunProgressSink>,
) -> Result<RunResults> {
    if request.solvers.is_empty() {
        bail!("at least one solver must be selected");
    }
    if request.run_options.is_empty() {
        bail!("at least one run option must be selected");
    }

    let cases = registry()?;
    let tasks = plan_tasks(&cases, request)?;
    run_planned_tasks(&tasks, request, Some(cache_options), progress_sink)
}

fn run_cases_uncached(
    request: &RunRequest,
    progress_sink: Option<&RunProgressSink>,
) -> Result<RunResults> {
    if request.solvers.is_empty() {
        bail!("at least one solver must be selected");
    }
    if request.run_options.is_empty() {
        bail!("at least one run option must be selected");
    }

    let cases = registry()?;
    let tasks = plan_tasks(&cases, request)?;
    run_planned_tasks(&tasks, request, None, progress_sink)
}

fn run_planned_tasks(
    tasks: &[RunTask<'_>],
    request: &RunRequest,
    cache_options: Option<&RunCacheOptions>,
    progress_sink: Option<&RunProgressSink>,
) -> Result<RunResults> {
    if let Some(sink) = progress_sink {
        sink(RunProgressEvent::Queued { total: tasks.len() });
    }

    let progress = request.progress.then(|| make_progress_bar(tasks.len()));
    let completed = std::sync::atomic::AtomicUsize::new(0);
    let execute = || {
        tasks
            .iter()
            .par_bridge()
            .filter_map(|task| {
                if matches!(task.planned.expected, KnownStatus::Skipped) && !request.include_skipped
                {
                    return None;
                }
                let record = run_task_with_optional_cache(task, cache_options, progress_sink)
                    .unwrap_or_else(|error| {
                        let mut record = panic_record(
                            task.case,
                            task.planned.solver,
                            task.planned.options,
                            task.planned.expected,
                            task.planned.max_iters_limit,
                            error.to_string(),
                        );
                        record.error = Some(error.to_string());
                        record
                    });
                let completed_now = completed.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                if let Some(sink) = progress_sink {
                    sink(RunProgressEvent::Completed {
                        completed: completed_now,
                        total: tasks.len(),
                        problem: record.id.clone(),
                        solver: record.solver,
                        status: record.status,
                        cache_status: record
                            .cache
                            .as_ref()
                            .map(|info| info.status)
                            .unwrap_or(ResultCacheStatus::Bypassed),
                    });
                }
                if let Some(progress) = &progress {
                    progress.inc(1);
                    if record.status.failed() {
                        progress.println(format!(
                            "fail  {:<28}  {:<3}  {:<2}  {}",
                            record.id,
                            task.planned.solver.label(),
                            task.planned.options.label(),
                            failure_brief(&record),
                        ));
                    }
                }
                Some(record)
            })
            .collect::<Vec<_>>()
    };

    let mut records = if let Some(jobs) = request.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()?
            .install(execute)
    } else {
        execute()
    };
    records.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.solver.label().cmp(right.solver.label()))
            .then_with(|| left.options.label().cmp(&right.options.label()))
    });

    if let Some(progress) = progress {
        progress.finish_with_message(format!(
            "{} runs complete ({} passed, {} not passed)",
            records.len(),
            records
                .iter()
                .filter(|record| record.status.accepted())
                .count(),
            records
                .iter()
                .filter(|record| !record.status.accepted())
                .count(),
        ));
    }
    if let Some(sink) = progress_sink {
        sink(RunProgressEvent::Finished {
            total: records.len(),
            accepted: records
                .iter()
                .filter(|record| record.status.accepted())
                .count(),
            failed: records
                .iter()
                .filter(|record| !record.status.accepted())
                .count(),
        });
    }

    Ok(RunResults { records })
}

#[derive(Clone)]
struct RunTask<'a> {
    case: &'a ProblemCase,
    planned: PlannedRunTask,
}

pub fn planned_run_tasks(request: &RunRequest) -> Result<Vec<PlannedRunTask>> {
    let cases = registry()?;
    Ok(plan_tasks(&cases, request)?
        .into_iter()
        .map(|task| task.planned)
        .collect())
}

pub fn preview_run(request: &RunRequest, cache_dir: &Path) -> Result<Vec<RunPreviewEntry>> {
    let cases = registry()?;
    Ok(plan_tasks(&cases, request)?
        .into_iter()
        .map(|task| {
            let key = cache_key(&task.planned);
            let cache_status = if try_load_cached_record(cache_dir, &key).is_some() {
                ResultCacheStatus::Hit
            } else {
                ResultCacheStatus::Miss
            };
            RunPreviewEntry {
                problem_id: task.planned.problem_id,
                test_set: task.planned.test_set,
                family: task.case.family.to_string(),
                solver: task.planned.solver,
                problem_speed: task.planned.problem_speed,
                cache_status,
            }
        })
        .collect())
}

fn plan_tasks<'a>(cases: &'a [ProblemCase], request: &RunRequest) -> Result<Vec<RunTask<'a>>> {
    let selected_cases = select_cases(cases, request.problem_ids.as_ref())?;
    let tasks = selected_cases
        .iter()
        .flat_map(|case| {
            request.solvers.iter().flat_map(move |solver| {
                request.run_options.iter().map(move |run_options| {
                    let manifest = manifest_entry_by_id(case.id)
                        .ok_or_else(|| anyhow!("missing manifest entry for {}", case.id))?;
                    if request.problem_set.is_some_and(|set| manifest.speed != set) {
                        return Ok(None);
                    }
                    if request.test_set.is_some_and(|set| manifest.test_set != set) {
                        return Ok(None);
                    }
                    Ok(Some(RunTask {
                        case,
                        planned: planned_task(case.id, *solver, *run_options, manifest),
                    }))
                })
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok(tasks)
}

fn planned_task(
    problem_id: &str,
    solver: SolverKind,
    options: ProblemRunOptions,
    manifest: &ProblemManifestEntry,
) -> PlannedRunTask {
    let expected = match solver {
        SolverKind::Sqp => manifest.sqp,
        SolverKind::Nlip => manifest.nlip,
        #[cfg(feature = "ipopt")]
        SolverKind::Ipopt => manifest.ipopt,
    };
    let max_iters_limit = match solver {
        SolverKind::Sqp => manifest.max_iters.sqp,
        SolverKind::Nlip => manifest.max_iters.nlip,
        #[cfg(feature = "ipopt")]
        SolverKind::Ipopt => manifest.max_iters.ipopt,
    };
    PlannedRunTask {
        problem_id: problem_id.to_string(),
        solver,
        options,
        expected,
        max_iters_limit,
        problem_speed: manifest.speed,
        test_set: manifest.test_set,
    }
}

fn run_task_with_optional_cache(
    task: &RunTask<'_>,
    cache_options: Option<&RunCacheOptions>,
    progress_sink: Option<&RunProgressSink>,
) -> Result<ProblemRunRecord> {
    let Some(cache_options) = cache_options else {
        return Ok(run_task(task));
    };
    let cache_key = cache_key(&task.planned);
    let lookup_started = Instant::now();
    if cache_options.enabled
        && !cache_options.force
        && let Some((mut record, written_at_unix_ms)) =
            try_load_cached_record(&cache_options.cache_dir, &cache_key)
    {
        let lookup_time_s = lookup_started.elapsed().as_secs_f64();
        record.cache = Some(ResultCacheInfo {
            status: ResultCacheStatus::Hit,
            key: cache_key.hash.clone(),
            written_at_unix_ms: Some(written_at_unix_ms),
            lookup_time_s,
        });
        if let Some(sink) = progress_sink {
            sink(RunProgressEvent::CacheHit {
                completed: 0,
                total: 0,
                problem: record.id.clone(),
                solver: record.solver,
            });
        }
        return Ok(record);
    }
    if let Some(sink) = progress_sink {
        sink(RunProgressEvent::Running {
            completed: 0,
            total: 0,
            problem: task.planned.problem_id.clone(),
            solver: task.planned.solver,
        });
    }
    let mut record = run_task(task);
    let status = if cache_options.force {
        ResultCacheStatus::Bypassed
    } else {
        ResultCacheStatus::Miss
    };
    let written_at_unix_ms = unix_time_ms();
    record.cache = Some(ResultCacheInfo {
        status,
        key: cache_key.hash.clone(),
        written_at_unix_ms: Some(written_at_unix_ms),
        lookup_time_s: lookup_started.elapsed().as_secs_f64(),
    });
    if cache_options.enabled {
        write_cached_record(
            &cache_options.cache_dir,
            &cache_key,
            &record,
            written_at_unix_ms,
        )?;
    }
    Ok(record)
}

fn run_task(task: &RunTask<'_>) -> ProblemRunRecord {
    match task.planned.expected {
        KnownStatus::Skipped => skipped_record(
            task.case,
            task.planned.solver,
            task.planned.options,
            task.planned.expected,
            task.planned.max_iters_limit,
        ),
        _ => match catch_unwind(AssertUnwindSafe(|| {
            task.case.run(
                task.planned.solver,
                task.planned.options,
                task.planned.max_iters_limit,
                task.planned.expected,
            )
        })) {
            Ok(record) => record,
            Err(payload) => panic_record(
                task.case,
                task.planned.solver,
                task.planned.options,
                task.planned.expected,
                task.planned.max_iters_limit,
                panic_payload_message(&payload),
            ),
        },
    }
}

fn select_cases<'a>(
    cases: &'a [ProblemCase],
    requested_ids: Option<&Vec<String>>,
) -> Result<Vec<&'a ProblemCase>> {
    let Some(requested_ids) = requested_ids else {
        return Ok(cases.iter().collect());
    };
    let mut selected = Vec::with_capacity(requested_ids.len());
    for requested_id in requested_ids {
        let case = cases
            .iter()
            .find(|case| case.id == requested_id)
            .ok_or_else(|| anyhow!("unknown problem id: {requested_id}"))?;
        selected.push(case);
    }
    Ok(selected)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheKey {
    schema_version: u32,
    build_fingerprint: String,
    ipopt_enabled: bool,
    task: PlannedRunTask,
    hash: String,
    #[serde(default)]
    written_at_unix_ms: Option<u128>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedRecordPayload {
    schema_version: u32,
    build_fingerprint: String,
    ipopt_enabled: bool,
    task: PlannedRunTask,
    written_at_unix_ms: u128,
    record: ProblemRunRecord,
    console_output: Option<String>,
}

fn cache_key(task: &PlannedRunTask) -> CacheKey {
    let mut hasher = DefaultHasher::new();
    TEST_PROBLEMS_CACHE_SCHEMA_VERSION.hash(&mut hasher);
    TEST_PROBLEMS_BUILD_FINGERPRINT.hash(&mut hasher);
    cfg!(feature = "ipopt").hash(&mut hasher);
    task.hash(&mut hasher);
    CacheKey {
        schema_version: TEST_PROBLEMS_CACHE_SCHEMA_VERSION,
        build_fingerprint: TEST_PROBLEMS_BUILD_FINGERPRINT.to_string(),
        ipopt_enabled: cfg!(feature = "ipopt"),
        task: task.clone(),
        hash: format!("{:016x}", hasher.finish()),
        written_at_unix_ms: None,
    }
}

fn cache_entry_path(cache_dir: &Path, key: &CacheKey) -> PathBuf {
    cache_dir
        .join(format!("v{}", TEST_PROBLEMS_CACHE_SCHEMA_VERSION))
        .join(&key.hash[..2])
        .join(format!("{}.json", key.hash))
}

fn try_load_cached_record(cache_dir: &Path, key: &CacheKey) -> Option<(ProblemRunRecord, u128)> {
    let path = cache_entry_path(cache_dir, key);
    let text = fs::read_to_string(path).ok()?;
    let payload = serde_json::from_str::<CachedRecordPayload>(&text).ok()?;
    if payload.schema_version != TEST_PROBLEMS_CACHE_SCHEMA_VERSION
        || payload.build_fingerprint != TEST_PROBLEMS_BUILD_FINGERPRINT
        || payload.ipopt_enabled != cfg!(feature = "ipopt")
        || payload.task != key.task
    {
        return None;
    }
    let mut record = payload.record;
    record.console_output = payload.console_output;
    record.console_output_path = None;
    Some((record, payload.written_at_unix_ms))
}

fn write_cached_record(
    cache_dir: &Path,
    key: &CacheKey,
    record: &ProblemRunRecord,
    written_at_unix_ms: u128,
) -> Result<()> {
    let path = cache_entry_path(cache_dir, key);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("cache entry path has no parent"))?;
    fs::create_dir_all(parent)?;
    let mut stored_record = record.clone();
    let console_output = stored_record.console_output.take();
    stored_record.console_output_path = None;
    let payload = CachedRecordPayload {
        schema_version: TEST_PROBLEMS_CACHE_SCHEMA_VERSION,
        build_fingerprint: TEST_PROBLEMS_BUILD_FINGERPRINT.to_string(),
        ipopt_enabled: cfg!(feature = "ipopt"),
        task: key.task.clone(),
        written_at_unix_ms,
        record: stored_record,
        console_output,
    };
    let temp_path = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temp_path, serde_json::to_vec_pretty(&payload)?)?;
    fs::rename(temp_path, path)?;
    Ok(())
}

pub fn default_result_cache_dir() -> PathBuf {
    std::env::var_os(TEST_PROBLEMS_CACHE_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/test-problems/cache"))
}

pub fn clear_result_cache(cache_dir: &Path) -> Result<()> {
    if cache_dir.exists() {
        fs::remove_dir_all(cache_dir)?;
    }
    Ok(())
}

pub fn cached_result_records(cache_dir: &Path) -> Result<Vec<ProblemRunRecord>> {
    let cache_root = cache_dir.join(format!("v{}", TEST_PROBLEMS_CACHE_SCHEMA_VERSION));
    let mut records = Vec::new();
    if !cache_root.exists() {
        return Ok(records);
    }
    collect_cached_result_records(&cache_root, &mut records)?;
    records.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.solver.label().cmp(right.solver.label()))
            .then_with(|| left.options.label().cmp(&right.options.label()))
    });
    Ok(records)
}

fn collect_cached_result_records(dir: &Path, records: &mut Vec<ProblemRunRecord>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_cached_result_records(&path, records)?;
            continue;
        }
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(payload) = serde_json::from_str::<CachedRecordPayload>(&text) else {
            continue;
        };
        if payload.schema_version != TEST_PROBLEMS_CACHE_SCHEMA_VERSION
            || payload.build_fingerprint != TEST_PROBLEMS_BUILD_FINGERPRINT
            || payload.ipopt_enabled != cfg!(feature = "ipopt")
        {
            continue;
        }
        let key = cache_key(&payload.task);
        let mut record = payload.record;
        record.console_output = payload.console_output;
        record.console_output_path = None;
        record.cache = Some(ResultCacheInfo {
            status: ResultCacheStatus::Hit,
            key: key.hash,
            written_at_unix_ms: Some(payload.written_at_unix_ms),
            lookup_time_s: 0.0,
        });
        records.push(record);
    }
    Ok(())
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn skipped_record(
    case: &ProblemCase,
    solver: SolverKind,
    options: ProblemRunOptions,
    expected: KnownStatus,
    max_iters_limit: usize,
) -> ProblemRunRecord {
    ProblemRunRecord {
        id: case.id.to_string(),
        solver,
        options,
        expected,
        max_iters_limit,
        status: RunStatus::Skipped,
        descriptor: ProblemDescriptor {
            id: case.id.to_string(),
            test_set: case.test_set,
            family: case.family.to_string(),
            variant: case.variant.to_string(),
            source: case.source.to_string(),
            description: case.description.to_string(),
            parameterized: case.parameterized,
            num_vars: 0,
            num_eq: 0,
            num_ineq: 0,
            num_box: 0,
            dof: 0,
            constrained: false,
        },
        solution: None,
        metrics: crate::model::SolverMetrics::default(),
        timing: crate::model::SolverTimingBreakdown::default(),
        validation: ValidationOutcome {
            tier: ValidationTier::Passed,
            tolerance: "skipped".to_string(),
            detail: "skipped by manifest".to_string(),
        },
        solver_thresholds: None,
        solver_settings: None,
        error: None,
        compile_report: None,
        console_output: None,
        console_output_path: None,
        filter_replay: None,
        cache: None,
    }
}

fn panic_record(
    case: &ProblemCase,
    solver: SolverKind,
    options: ProblemRunOptions,
    expected: KnownStatus,
    max_iters_limit: usize,
    panic_message: String,
) -> ProblemRunRecord {
    let detail = format!("solver panicked: {panic_message}");
    ProblemRunRecord {
        id: case.id.to_string(),
        solver,
        options,
        expected,
        max_iters_limit,
        status: RunStatus::SolveError,
        descriptor: ProblemDescriptor {
            id: case.id.to_string(),
            test_set: case.test_set,
            family: case.family.to_string(),
            variant: case.variant.to_string(),
            source: case.source.to_string(),
            description: case.description.to_string(),
            parameterized: case.parameterized,
            num_vars: 0,
            num_eq: 0,
            num_ineq: 0,
            num_box: 0,
            dof: 0,
            constrained: false,
        },
        solution: None,
        metrics: crate::model::SolverMetrics::default(),
        timing: crate::model::SolverTimingBreakdown::default(),
        validation: ValidationOutcome {
            tier: ValidationTier::Failed,
            tolerance: "solver must not panic".to_string(),
            detail: detail.clone(),
        },
        solver_thresholds: None,
        solver_settings: None,
        error: Some(detail),
        compile_report: None,
        console_output: None,
        console_output_path: None,
        filter_replay: None,
        cache: None,
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

fn make_progress_bar(total: usize) -> ProgressBar {
    let progress = ProgressBar::new(total as u64);
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>3}/{len:<3} {msg}",
    )
    .expect("valid progress template")
    .progress_chars("=> ");
    progress.set_style(style);
    progress.set_message("running");
    progress
}

fn failure_brief(record: &ProblemRunRecord) -> String {
    if let Some(error) = &record.error {
        if error.contains("failed to converge") {
            "max_iters".to_string()
        } else if error.contains("solver panicked") {
            "panic".to_string()
        } else if error.contains("line search failed") {
            "line_search".to_string()
        } else if error.contains("PrimalInfeasible") {
            "primal_infeasible".to_string()
        } else {
            error.clone()
        }
    } else {
        record.validation.detail.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock_sqp_request() -> RunRequest {
        RunRequest {
            problem_ids: Some(vec!["rosenbrock_2".to_string()]),
            solvers: vec![SolverKind::Sqp],
            jobs: Some(1),
            ..RunRequest::default()
        }
    }

    #[test]
    fn cache_key_changes_when_options_change() {
        let request = rosenbrock_sqp_request();
        let mut tasks = planned_run_tasks(&request).expect("planned tasks");
        let base = cache_key(&tasks.remove(0));

        let mut changed_request = request;
        changed_request.run_options = vec![ProblemRunOptions {
            jit_opt_level: JitOptLevel::O2,
            call_policy: CallPolicyMode::InlineAtLowering,
        }];
        let changed = cache_key(
            &planned_run_tasks(&changed_request)
                .expect("changed planned tasks")
                .remove(0),
        );
        assert_ne!(base.hash, changed.hash);
    }

    #[test]
    fn cached_record_preserves_solver_timing() {
        let cache_dir = PathBuf::from(format!(
            "target/test-problems/unit-cache-{}",
            std::process::id()
        ));
        let _ = clear_result_cache(&cache_dir);
        let request = rosenbrock_sqp_request();
        let cache_options = RunCacheOptions {
            enabled: true,
            force: false,
            cache_dir: cache_dir.clone(),
        };
        let fresh = run_cases_with_cache(&request, &cache_options, None).expect("fresh run");
        let cached = run_cases_with_cache(&request, &cache_options, None).expect("cached run");
        let fresh_record = &fresh.records[0];
        let cached_record = &cached.records[0];
        assert_eq!(fresh_record.status, cached_record.status);
        assert_eq!(
            fresh_record.timing.total_wall_time,
            cached_record.timing.total_wall_time
        );
        assert_eq!(
            fresh_record.cache.as_ref().map(|info| info.status),
            Some(ResultCacheStatus::Miss)
        );
        assert_eq!(
            cached_record.cache.as_ref().map(|info| info.status),
            Some(ResultCacheStatus::Hit)
        );
        let _ = clear_result_cache(&cache_dir);
    }

    #[test]
    fn force_bypasses_existing_cache_entry() {
        let cache_dir = PathBuf::from(format!(
            "target/test-problems/unit-cache-force-{}",
            std::process::id()
        ));
        let _ = clear_result_cache(&cache_dir);
        let request = rosenbrock_sqp_request();
        let cache_options = RunCacheOptions {
            enabled: true,
            force: false,
            cache_dir: cache_dir.clone(),
        };
        let _ = run_cases_with_cache(&request, &cache_options, None).expect("fresh run");
        let forced = run_cases_with_cache(
            &request,
            &RunCacheOptions {
                force: true,
                ..cache_options
            },
            None,
        )
        .expect("forced run");
        assert_eq!(
            forced.records[0].cache.as_ref().map(|info| info.status),
            Some(ResultCacheStatus::Bypassed)
        );
        let _ = clear_result_cache(&cache_dir);
    }
}
