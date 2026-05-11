use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc as std_mpsc,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow, bail};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use test_problems::{
    CallPolicyMode, JitOptLevel, KnownStatus, PlannedRunTask, ProblemRunOptions, ProblemSpeed,
    ProblemTestSet, RunCacheOptions, RunPreviewEntry, RunProgressEvent, RunRequest, RunResults,
    RunStage, SolverKind, cached_result_records, clear_result_cache, default_result_cache_dir,
    manifest_entry_by_id, planned_run_tasks, registry, render_dashboard_html,
    render_markdown_report, run_cases_with_cache, write_json_report, write_transcript_artifacts,
};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::UnboundedReceiverStream};

#[derive(Debug, Parser)]
#[command(
    name = "test_problem_studio",
    about = "Local web app and static exporter for cached test-problem runs."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long, env = "PORT", default_value_t = 3001)]
        port: u16,
        #[arg(long = "cache-dir")]
        cache_dir: Option<PathBuf>,
    },
    Export {
        #[arg(long = "output-dir", default_value = "target/test-problems/site")]
        output_dir: PathBuf,
        #[arg(long = "cache-dir")]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },
}

#[derive(Clone)]
struct AppState {
    cache_dir: PathBuf,
    runs: Arc<Mutex<HashMap<String, RunState>>>,
}

struct RunState {
    events: Option<mpsc::UnboundedReceiver<RunProgressEvent>>,
    results: Arc<Mutex<Option<RunResults>>>,
    error: Arc<Mutex<Option<String>>>,
    control: Arc<RunControl>,
}

#[derive(Debug)]
struct RunControl {
    desired_jobs: AtomicUsize,
    stop_requested: AtomicBool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RunCreateRequest {
    test_set: Option<String>,
    problem_set: Option<String>,
    problem_ids: Option<Vec<String>>,
    solver: Option<String>,
    jit_opt: Option<String>,
    call_policy: Option<String>,
    jobs: Option<usize>,
    include_skipped: Option<bool>,
    force: Option<bool>,
}

#[derive(Debug, Serialize)]
struct RunCreateResponse {
    run_id: String,
}

#[derive(Debug, Deserialize)]
struct RunJobsRequest {
    jobs: usize,
}

#[derive(Debug, Serialize)]
struct Catalog {
    suites: Vec<&'static str>,
    solvers: Vec<&'static str>,
    jit_opts: Vec<&'static str>,
    call_policies: Vec<&'static str>,
    problems: Vec<CatalogProblem>,
    cache_dir: String,
}

#[derive(Debug, Serialize)]
struct CatalogProblem {
    id: String,
    test_set: &'static str,
    family: String,
    variant: String,
    source: String,
    speed: ProblemSpeed,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command.unwrap_or(Command::Serve {
        port: 3001,
        cache_dir: None,
    }) {
        Command::Serve { port, cache_dir } => {
            serve(port, cache_dir.unwrap_or_else(default_result_cache_dir)).await
        }
        Command::Export {
            output_dir,
            cache_dir,
            force,
        } => export_static(
            output_dir,
            cache_dir.unwrap_or_else(default_result_cache_dir),
            force,
        ),
    }
}

async fn serve(port: u16, cache_dir: PathBuf) -> Result<()> {
    let state = AppState {
        cache_dir,
        runs: Arc::new(Mutex::new(HashMap::new())),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/api/catalog", get(catalog))
        .route("/api/preview", post(preview))
        .route("/api/runs", post(create_run))
        .route("/api/runs/{id}/events", get(run_events))
        .route("/api/runs/{id}/results", get(run_results))
        .route("/api/runs/{id}/dashboard", get(run_dashboard))
        .route("/api/runs/{id}/jobs", post(update_run_jobs))
        .route("/api/runs/{id}/stop", post(stop_run))
        .route("/api/results_cache/records", get(results_cache_records))
        .route("/api/results_cache/clear", post(clear_results_cache))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("test_problem_studio listening on http://127.0.0.1:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        )],
        INDEX_HTML,
    )
}

async fn catalog(State(state): State<AppState>) -> Result<Json<Catalog>, ApiError> {
    let cases = registry().map_err(internal_error)?;
    let mut problems = cases
        .into_iter()
        .map(|case| {
            let manifest = manifest_entry_by_id(case.id)
                .ok_or_else(|| anyhow!("missing manifest entry for {}", case.id))?;
            Ok(CatalogProblem {
                id: case.id.to_string(),
                test_set: case.test_set.label(),
                family: case.family.to_string(),
                variant: case.variant.to_string(),
                source: case.source.to_string(),
                speed: manifest.speed,
            })
        })
        .collect::<Result<Vec<_>>>()
        .map_err(internal_error)?;
    problems.sort_by(|left, right| left.id.cmp(&right.id));
    #[cfg(feature = "ipopt")]
    let solvers = vec!["sqp", "nlip", "ipopt"];
    #[cfg(not(feature = "ipopt"))]
    let solvers = vec!["sqp", "nlip"];
    Ok(Json(Catalog {
        suites: vec!["core", "burkardt_test_nonlin", "schittkowski_306"],
        solvers,
        jit_opts: vec!["o0", "o2", "o3", "os"],
        call_policies: vec![
            "inline_at_call",
            "inline_at_lowering",
            "inline_in_llvm",
            "no_inline_llvm",
        ],
        problems,
        cache_dir: state.cache_dir.display().to_string(),
    }))
}

async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<RunCreateRequest>,
) -> Result<Json<RunCreateResponse>, ApiError> {
    let run_id = format!("run-{}", unix_time_ms());
    let run_request = request.to_run_request().map_err(bad_request)?;
    let cache_options = RunCacheOptions {
        enabled: true,
        force: request.force.unwrap_or(false),
        cache_dir: state.cache_dir.clone(),
    };
    let (sender, receiver) = mpsc::unbounded_channel();
    let results = Arc::new(Mutex::new(None));
    let error = Arc::new(Mutex::new(None));
    let control = Arc::new(RunControl {
        desired_jobs: AtomicUsize::new(run_request.jobs.unwrap_or(4).max(1)),
        stop_requested: AtomicBool::new(false),
    });
    state.runs.lock().expect("runs lock").insert(
        run_id.clone(),
        RunState {
            events: Some(receiver),
            results: results.clone(),
            error: error.clone(),
            control: control.clone(),
        },
    );
    std::thread::spawn(move || {
        let sink = move |event| {
            let _ = sender.send(event);
        };
        match run_cases_with_dynamic_control(&run_request, &cache_options, &control, &sink) {
            Ok(run_results) => {
                *results.lock().expect("results lock") = Some(run_results);
            }
            Err(run_error) => {
                *error.lock().expect("error lock") = Some(run_error.to_string());
            }
        }
    });
    Ok(Json(RunCreateResponse { run_id }))
}

async fn preview(
    State(state): State<AppState>,
    Json(request): Json<RunCreateRequest>,
) -> Result<Json<Vec<RunPreviewEntry>>, ApiError> {
    let run_request = request.to_run_request().map_err(bad_request)?;
    test_problems::preview_run(&run_request, &state.cache_dir)
        .map(Json)
        .map_err(internal_error)
}

async fn run_events(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let receiver = {
        let mut runs = state.runs.lock().expect("runs lock");
        runs.get_mut(&run_id)
            .ok_or_else(|| not_found(format!("unknown run id {run_id}")))?
            .events
            .take()
            .ok_or_else(|| bad_request("events already consumed".to_string()))?
    };
    let stream = UnboundedReceiverStream::new(receiver).map(|event| {
        let mut line = serde_json::to_vec(&event).unwrap_or_else(|_| b"{}".to_vec());
        line.push(b'\n');
        Ok::<_, Infallible>(Bytes::from(line))
    });
    Ok((
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
        )],
        axum::body::Body::from_stream(stream),
    ))
}

async fn run_results(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<RunResults>, ApiError> {
    let runs = state.runs.lock().expect("runs lock");
    let run = runs
        .get(&run_id)
        .ok_or_else(|| not_found(format!("unknown run id {run_id}")))?;
    if let Some(error) = run.error.lock().expect("error lock").clone() {
        return Err(internal_error(anyhow!(error)));
    }
    let results = run
        .results
        .lock()
        .expect("results lock")
        .clone()
        .ok_or_else(|| bad_request("run is not finished yet".to_string()))?;
    Ok(Json(results))
}

async fn run_dashboard(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let runs = state.runs.lock().expect("runs lock");
    let run = runs
        .get(&run_id)
        .ok_or_else(|| not_found(format!("unknown run id {run_id}")))?;
    let mut results = run
        .results
        .lock()
        .expect("results lock")
        .clone()
        .ok_or_else(|| bad_request("run is not finished yet".to_string()))?;
    let output_dir = PathBuf::from("target/test-problems/studio").join(&run_id);
    write_transcript_artifacts(&mut results, &output_dir).map_err(internal_error)?;
    let html = render_dashboard_html(&results).map_err(internal_error)?;
    Ok((
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        )],
        html,
    ))
}

async fn update_run_jobs(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    Json(request): Json<RunJobsRequest>,
) -> Result<StatusCode, ApiError> {
    if request.jobs == 0 {
        return Err(bad_request("jobs must be at least 1".to_string()));
    }
    let runs = state.runs.lock().expect("runs lock");
    let run = runs
        .get(&run_id)
        .ok_or_else(|| not_found(format!("unknown run id {run_id}")))?;
    run.control
        .desired_jobs
        .store(request.jobs, Ordering::SeqCst);
    Ok(StatusCode::NO_CONTENT)
}

async fn stop_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let runs = state.runs.lock().expect("runs lock");
    let run = runs
        .get(&run_id)
        .ok_or_else(|| not_found(format!("unknown run id {run_id}")))?;
    run.control.stop_requested.store(true, Ordering::SeqCst);
    Ok(StatusCode::ACCEPTED)
}

async fn clear_results_cache(State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    clear_result_cache(&state.cache_dir).map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn results_cache_records(
    State(state): State<AppState>,
) -> Result<Json<RunResults>, ApiError> {
    cached_result_records(&state.cache_dir)
        .map(|records| Json(RunResults { records }))
        .map_err(internal_error)
}

fn run_cases_with_dynamic_control(
    request: &RunRequest,
    cache_options: &RunCacheOptions,
    control: &RunControl,
    sink: &(dyn Fn(RunProgressEvent) + Send + Sync),
) -> Result<RunResults> {
    emit_stage(
        sink,
        RunStage::Planning,
        0,
        0,
        0,
        control.desired_jobs.load(Ordering::SeqCst),
        "Planning selected solver-runs.",
    );
    let mut tasks = planned_run_tasks(request)?;
    if !request.include_skipped {
        tasks.retain(|task| task.expected != KnownStatus::Skipped);
    }
    let total = tasks.len();
    sink(RunProgressEvent::Queued { total });
    emit_stage(
        sink,
        RunStage::Queued,
        0,
        total,
        0,
        control.desired_jobs.load(Ordering::SeqCst),
        "Queued selected solver-runs.",
    );

    let mut queue = VecDeque::from(tasks);
    let (done_sender, done_receiver) = std_mpsc::channel();
    let mut records = Vec::new();
    let mut active = 0usize;
    let mut completed = 0usize;
    let mut last_jobs = control.desired_jobs.load(Ordering::SeqCst).max(1);
    let mut stop_announced = false;

    loop {
        let desired_jobs = control.desired_jobs.load(Ordering::SeqCst).max(1);
        if desired_jobs != last_jobs {
            last_jobs = desired_jobs;
            sink(RunProgressEvent::JobsChanged { jobs: desired_jobs });
            emit_stage(
                sink,
                current_solve_stage(control, stop_announced),
                completed,
                total,
                active,
                desired_jobs,
                "Updated dynamic job limit.",
            );
        }

        let stop_requested = control.stop_requested.load(Ordering::SeqCst);
        if stop_requested && !stop_announced {
            stop_announced = true;
            sink(RunProgressEvent::StopRequested {
                completed,
                total,
                active,
            });
            emit_stage(
                sink,
                RunStage::Stopping,
                completed,
                total,
                active,
                desired_jobs,
                "Stop requested. Waiting for active solver-runs to finish.",
            );
        }

        while !stop_requested && active < desired_jobs {
            let Some(task) = queue.pop_front() else {
                break;
            };
            active += 1;
            sink(RunProgressEvent::Running {
                completed,
                total,
                problem: task.problem_id.clone(),
                solver: task.solver,
            });
            let sender = done_sender.clone();
            let task_request = single_task_request(request, &task);
            let task_cache_options = cache_options.clone();
            std::thread::spawn(move || {
                let result = run_cases_with_cache(&task_request, &task_cache_options, None)
                    .map(|results| results.records)
                    .map_err(|error| error.to_string());
                let _ = sender.send((task, result));
            });
        }

        if active == 0 && (queue.is_empty() || stop_requested) {
            break;
        }

        match done_receiver.recv_timeout(Duration::from_millis(100)) {
            Ok((task, task_result)) => {
                active = active.saturating_sub(1);
                completed += 1;
                match task_result {
                    Ok(mut task_records) => {
                        if let Some(record) = task_records.pop() {
                            let cache_status = record
                                .cache
                                .as_ref()
                                .map(|info| info.status)
                                .unwrap_or(test_problems::ResultCacheStatus::Bypassed);
                            sink(RunProgressEvent::Completed {
                                completed,
                                total,
                                problem: record.id.clone(),
                                solver: record.solver,
                                status: record.status,
                                cache_status,
                            });
                            records.push(record);
                        } else {
                            sink(RunProgressEvent::Failed {
                                completed,
                                total,
                                problem: task.problem_id,
                                solver: task.solver,
                                reason: "task returned no record".to_string(),
                            });
                        }
                    }
                    Err(reason) => {
                        sink(RunProgressEvent::Failed {
                            completed,
                            total,
                            problem: task.problem_id,
                            solver: task.solver,
                            reason,
                        });
                    }
                }
                let stage = current_solve_stage(control, stop_announced);
                emit_stage(
                    sink,
                    stage,
                    completed,
                    total,
                    active,
                    control.desired_jobs.load(Ordering::SeqCst).max(1),
                    if stage == RunStage::Stopping {
                        "Stopping after active solver-runs finish."
                    } else {
                        "Solving selected problem matrix."
                    },
                );
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(done_sender);
    emit_stage(
        sink,
        RunStage::CollectingResults,
        completed,
        total,
        active,
        control.desired_jobs.load(Ordering::SeqCst).max(1),
        "Collecting and sorting completed records.",
    );
    records.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.solver.label().cmp(right.solver.label()))
            .then_with(|| left.options.label().cmp(&right.options.label()))
    });
    let accepted = records
        .iter()
        .filter(|record| record.status.accepted())
        .count();
    let failed = records
        .iter()
        .filter(|record| !record.status.accepted())
        .count();
    if stop_announced {
        sink(RunProgressEvent::Stopped {
            total,
            completed,
            accepted,
            failed,
        });
        emit_stage(
            sink,
            RunStage::Stopped,
            completed,
            total,
            0,
            control.desired_jobs.load(Ordering::SeqCst).max(1),
            "Run stopped after active solver-runs finished.",
        );
    } else {
        sink(RunProgressEvent::Finished {
            total,
            accepted,
            failed,
        });
        emit_stage(
            sink,
            RunStage::Complete,
            completed,
            total,
            0,
            control.desired_jobs.load(Ordering::SeqCst).max(1),
            "Solver-runs complete.",
        );
    }
    Ok(RunResults { records })
}

fn current_solve_stage(control: &RunControl, stop_announced: bool) -> RunStage {
    if stop_announced || control.stop_requested.load(Ordering::SeqCst) {
        RunStage::Stopping
    } else {
        RunStage::Solving
    }
}

fn emit_stage(
    sink: &(dyn Fn(RunProgressEvent) + Send + Sync),
    stage: RunStage,
    completed: usize,
    total: usize,
    active: usize,
    desired_jobs: usize,
    message: impl Into<String>,
) {
    sink(RunProgressEvent::Stage {
        stage,
        completed,
        total,
        active,
        desired_jobs,
        message: message.into(),
    });
}

fn single_task_request(request: &RunRequest, task: &PlannedRunTask) -> RunRequest {
    RunRequest {
        problem_ids: Some(vec![task.problem_id.clone()]),
        solvers: vec![task.solver],
        run_options: vec![task.options],
        jobs: Some(1),
        include_skipped: request.include_skipped,
        problem_set: None,
        test_set: None,
        progress: false,
    }
}

fn export_static(output_dir: PathBuf, cache_dir: PathBuf, force: bool) -> Result<()> {
    fs::create_dir_all(&output_dir)?;
    let request = RunCreateRequest {
        test_set: None,
        problem_set: Some("fast".to_string()),
        solver: Some("both".to_string()),
        jobs: Some(4),
        force: Some(force),
        ..RunCreateRequest::default()
    };
    let run_request = request.to_run_request()?;
    let cache_options = RunCacheOptions {
        enabled: true,
        force,
        cache_dir,
    };
    let mut results = run_cases_with_cache(&run_request, &cache_options, None)?;
    write_transcript_artifacts(&mut results, &output_dir)?;
    fs::write(
        output_dir.join("index.html"),
        render_dashboard_html(&results)?,
    )?;
    fs::write(
        output_dir.join("report.md"),
        render_markdown_report(&results),
    )?;
    write_json_report(&results, &output_dir.join("results.json"))?;
    println!("Wrote static test-problem site to {}", output_dir.display());
    Ok(())
}

impl RunCreateRequest {
    fn to_run_request(&self) -> Result<RunRequest> {
        Ok(RunRequest {
            problem_ids: self.problem_ids.clone(),
            solvers: parse_solver_selection(self.solver.as_deref().unwrap_or("both"))?,
            run_options: vec![ProblemRunOptions {
                jit_opt_level: parse_jit(self.jit_opt.as_deref().unwrap_or("o3"))?,
                call_policy: parse_call_policy(
                    self.call_policy.as_deref().unwrap_or("inline_at_lowering"),
                )?,
            }],
            jobs: self.jobs,
            include_skipped: self.include_skipped.unwrap_or(false),
            problem_set: self
                .problem_set
                .as_deref()
                .map(parse_problem_speed)
                .transpose()?,
            test_set: self.test_set.as_deref().map(parse_test_set).transpose()?,
            progress: false,
        })
    }
}

fn parse_solver_selection(value: &str) -> Result<Vec<SolverKind>> {
    match normalized(value).as_str() {
        "none" | "" => bail!("select at least one solver"),
        "sqp" => Ok(vec![SolverKind::Sqp]),
        "nlip" | "ip" => Ok(vec![SolverKind::Nlip]),
        "both" => Ok(vec![SolverKind::Sqp, SolverKind::Nlip]),
        "all" => {
            #[cfg(feature = "ipopt")]
            let solvers = vec![SolverKind::Sqp, SolverKind::Nlip, SolverKind::Ipopt];
            #[cfg(not(feature = "ipopt"))]
            let solvers = vec![SolverKind::Sqp, SolverKind::Nlip];
            Ok(solvers)
        }
        #[cfg(feature = "ipopt")]
        "ipopt" => Ok(vec![SolverKind::Ipopt]),
        other if other.contains(',') => {
            let mut solvers = Vec::new();
            for part in other.split(',') {
                solvers.extend(parse_solver_selection(part)?);
            }
            solvers.sort_by_key(|solver| solver.label());
            solvers.dedup();
            Ok(solvers)
        }
        other => bail!("unknown solver selection {other}"),
    }
}

fn parse_jit(value: &str) -> Result<JitOptLevel> {
    match normalized(value).as_str() {
        "o0" | "0" => Ok(JitOptLevel::O0),
        "o2" | "2" => Ok(JitOptLevel::O2),
        "o3" | "3" => Ok(JitOptLevel::O3),
        "os" | "s" => Ok(JitOptLevel::Os),
        other => bail!("unknown JIT opt level {other}"),
    }
}

fn parse_call_policy(value: &str) -> Result<CallPolicyMode> {
    match normalized(value).as_str() {
        "inline_at_call" => Ok(CallPolicyMode::InlineAtCall),
        "inline_at_lowering" => Ok(CallPolicyMode::InlineAtLowering),
        "inline_in_llvm" => Ok(CallPolicyMode::InlineInLlvm),
        "no_inline_llvm" => Ok(CallPolicyMode::NoInlineLlvm),
        other => bail!("unknown call policy {other}"),
    }
}

fn parse_problem_speed(value: &str) -> Result<ProblemSpeed> {
    match normalized(value).as_str() {
        "fast" => Ok(ProblemSpeed::Fast),
        "slow" => Ok(ProblemSpeed::Slow),
        "all" => bail!("use null problem_set for all"),
        other => bail!("unknown problem set {other}"),
    }
}

fn parse_test_set(value: &str) -> Result<ProblemTestSet> {
    match normalized(value).as_str() {
        "core" => Ok(ProblemTestSet::Core),
        "burkardt_test_nonlin" | "burkardt-test-nonlin" => Ok(ProblemTestSet::BurkardtTestNonlin),
        "schittkowski_306" | "schittkowski-306" => Ok(ProblemTestSet::Schittkowski306),
        "all" => bail!("use null test_set for all"),
        other => bail!("unknown test set {other}"),
    }
}

fn normalized(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

type ApiError = (StatusCode, Json<ErrorResponse>);

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

fn bad_request(error: impl ToString) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn not_found(error: impl ToString) -> ApiError {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn internal_error(error: impl ToString) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Test Problem Studio</title>
  <style>
    :root { color-scheme: dark; --bg:#0f172a; --panel:#111827; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#60a5fa; --pass:#34d399; --reduced:#fbbf24; --fail:#f87171; --hit:#38bdf8; --miss:#a78bfa; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif, system-ui, sans-serif; }
    main { max-width:1600px; margin:0 auto; padding:20px; display:grid; gap:14px; }
    h1 { margin:0; font-size:32px; }
    .sub { color:var(--muted); }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; }
    .config-grid { display:grid; grid-template-columns:minmax(260px, 1fr) minmax(520px, 2.9fr); gap:14px; align-items:stretch; }
    .config-main { display:grid; grid-template-columns:minmax(190px, .9fr) minmax(160px, .8fr) minmax(120px, .55fr) minmax(370px, 1.2fr); gap:12px; align-items:end; }
    .action-stack { display:grid; grid-template-columns:minmax(90px, .7fr) minmax(90px, .7fr) minmax(180px, 1.2fr); gap:10px; align-items:end; }
    label, .control-label { display:grid; gap:5px; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    select, input { width:100%; background:#0b1220; color:var(--text); border:1px solid var(--line); border-radius:9px; padding:8px 9px; }
    input[type="checkbox"], input[type="radio"] { width:auto; }
    .segmented, .checks { display:flex; flex-wrap:wrap; gap:6px; align-items:center; min-height:38px; }
    .segmented label, .checks label { display:flex; align-items:center; gap:6px; margin:0; padding:8px 10px; border:1px solid var(--line); border-radius:999px; background:#0b1220; color:var(--text); text-transform:none; letter-spacing:0; font-size:13px; cursor:pointer; }
    .checks label { border-radius:9px; }
    .standalone-check { display:flex; align-items:center; gap:8px; min-height:38px; padding:8px 10px; border:1px solid var(--line); border-radius:9px; background:#0b1220; color:var(--text); text-transform:none; letter-spacing:0; font-size:13px; cursor:pointer; }
    button { border:0; border-radius:9px; background:var(--accent); color:#06111f; font-weight:700; min-height:38px; padding:9px 12px; cursor:pointer; }
    button:disabled { opacity:.72; cursor:wait; }
    button.secondary { background:#1f2937; color:var(--text); border:1px solid var(--line); }
    button.danger { background:rgba(248,113,113,.16); color:#fecaca; border:1px solid rgba(248,113,113,.55); }
    .action-stack button { height:38px; white-space:nowrap; }
    .loading-line { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:13px; }
    .spinner { width:14px; height:14px; border-radius:999px; border:2px solid #334155; border-top-color:var(--accent); animation:spin .8s linear infinite; flex:0 0 auto; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .skeleton { height:12px; border-radius:999px; background:linear-gradient(90deg, #1e293b, #334155, #1e293b); background-size:200% 100%; animation:pulse 1.2s ease-in-out infinite; }
    .skeleton.short { width:42%; }
    .skeleton.medium { width:68%; }
    .skeleton.long { width:88%; }
    @keyframes pulse { from { background-position:200% 0; } to { background-position:-200% 0; } }
    .progress-shell {
      display:grid;
      grid-template-columns:minmax(520px, 1fr) minmax(420px, .82fr);
      gap:12px;
      align-items:start;
    }
    .progress-main, .progress-side { display:grid; gap:12px; align-content:start; }
    .progress-top { display:flex; justify-content:space-between; gap:16px; align-items:end; flex-wrap:wrap; }
    .progress-title { font-size:18px; font-weight:800; }
    .progress-sub { color:var(--muted); font-size:13px; }
    .progress-percent { font-size:34px; line-height:1; font-weight:900; letter-spacing:.01em; }
    .progress-track { position:relative; height:22px; overflow:hidden; border-radius:999px; background:#020617; border:1px solid var(--line); box-shadow:inset 0 1px 10px rgba(0,0,0,.35); }
    .progress-fill { height:100%; width:0%; border-radius:999px; background:linear-gradient(90deg, var(--accent), #5eead4); transition:width .35s ease; box-shadow:0 0 24px rgba(96,165,250,.35); }
    .progress-fill.running::after { content:""; display:block; height:100%; width:100%; background:linear-gradient(110deg, transparent 0%, rgba(255,255,255,.22) 45%, transparent 70%); animation:sheen 1.4s linear infinite; }
    @keyframes sheen { from { transform:translateX(-100%); } to { transform:translateX(100%); } }
    .segment-track { display:flex; height:10px; overflow:hidden; border-radius:999px; background:#020617; border:1px solid #1e293b; }
    .segment { width:0%; transition:width .3s ease; }
    .segment.pass { background:var(--pass); }
    .segment.reduced { background:var(--reduced); }
    .segment.fail { background:var(--fail); }
    .segment.hit { background:var(--hit); }
    .segment.miss { background:var(--miss); }
    .progress-track, .segment-track, .segment, .stat[title], .donut, .mini-row, .mini-track, .scope-row, .scope-track, .solver-cell { cursor:help; }
    .stat-grid { display:grid; grid-template-columns:repeat(3, minmax(110px, 1fr)); gap:10px; }
    .stat { padding:10px 12px; border:1px solid var(--line); border-radius:12px; background:#0b1220; }
    .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
    .stat .v { margin-top:4px; font-size:20px; font-weight:850; }
    .active-list { display:flex; gap:8px; flex-wrap:wrap; min-height:30px; }
    .task-chip { padding:6px 9px; border:1px solid var(--line); border-radius:999px; background:#0b1220; color:#cbd5e1; font-size:12px; }
    .stage-panel { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:8px; }
    .stage-step { border:1px solid #253348; border-radius:11px; background:#0b1220; padding:8px 9px; min-width:0; }
    .stage-step.done { border-color:rgba(52,211,153,.48); background:rgba(52,211,153,.08); }
    .stage-step.active { border-color:rgba(96,165,250,.75); background:rgba(96,165,250,.12); box-shadow:0 0 0 1px rgba(96,165,250,.12); }
    .stage-step.blocked { border-color:rgba(248,113,113,.55); background:rgba(248,113,113,.1); }
    .stage-name { color:#dbeafe; font-size:11px; font-weight:850; text-transform:uppercase; letter-spacing:.05em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .stage-detail { margin-top:4px; color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .preview-grid { display:grid; grid-template-columns: 180px 1fr; gap:12px; align-items:stretch; }
    .preview-grid .stat:last-child { grid-column:1 / -1; }
    .donut-card { display:grid; place-items:center; border:1px solid var(--line); border-radius:14px; background:#0b1220; padding:12px; }
    .donut { width:142px; height:142px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--hit) 0deg, var(--hit) 0deg, var(--miss) 0deg 360deg); box-shadow:0 0 40px rgba(56,189,248,.08); }
    .donut::after { content:""; position:absolute; width:92px; height:92px; border-radius:50%; background:#0b1220; border:1px solid rgba(148,163,184,.18); }
    .donut-inner { position:relative; z-index:1; text-align:center; }
    .donut-inner .big { font-size:26px; font-weight:900; }
    .mini-bars { display:grid; gap:9px; }
    .mini-row { display:grid; grid-template-columns: 140px 1fr 58px; gap:8px; align-items:center; }
    .mini-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#cbd5e1; font-size:12px; }
    .mini-track { height:10px; border-radius:999px; background:#020617; overflow:hidden; border:1px solid #1e293b; }
    .mini-fill { height:100%; border-radius:999px; background:linear-gradient(90deg, var(--accent), #5eead4); }
    .scope-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .scope-row { display:grid; grid-template-columns: 170px 1fr 86px; gap:8px; align-items:center; }
    .scope-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#cbd5e1; font-size:12px; }
    .scope-label.unselected { color:#64748b; }
    .scope-track { height:12px; border-radius:999px; background:#334155; overflow:hidden; box-shadow:inset 0 0 0 1px rgba(148,163,184,.18); }
    .scope-fill { height:100%; border-radius:999px; background:linear-gradient(90deg, var(--pass), var(--accent)); box-shadow:0 0 18px rgba(56,189,248,.18); }
    .scope-count { color:var(--muted); font-size:12px; text-align:right; }
    .preview-list { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:6px; }
    .preview-item { display:grid; grid-template-columns:minmax(0, 1fr) auto; align-items:center; gap:10px; border:1px solid #243244; border-radius:10px; background:#0b1220; padding:7px 9px; font-size:12px; min-width:0; }
    .preview-item span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .preview-item .muted { color:var(--muted); }
    .matrix-head { display:flex; justify-content:space-between; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
    .legend { display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:12px; }
    .legend-item { display:inline-flex; gap:6px; align-items:center; }
    .legend-dot { width:10px; height:10px; border-radius:999px; background:#475569; box-shadow:0 0 0 1px rgba(148,163,184,.2); }
    .legend-dot.passed { background:var(--pass); }
    .legend-dot.reduced { background:var(--reduced); }
    .legend-dot.failed { background:var(--fail); }
    .legend-dot.staged { background:var(--accent); }
    .legend-dot.running { background:#22d3ee; }
    .problem-matrix { display:grid; gap:4px; }
    .problem-row { display:grid; gap:6px; align-items:center; padding:5px 0; border-bottom:1px solid rgba(51,65,85,.55); }
    .problem-row.running-row { position:relative; border-radius:9px; background:linear-gradient(90deg, rgba(34,211,238,.12), rgba(34,211,238,.035) 40%, transparent 72%); box-shadow:inset 3px 0 0 rgba(34,211,238,.95), 0 0 18px rgba(34,211,238,.08); animation:runningRowGlow 1.8s ease-in-out infinite; }
    .problem-row.running-row .problem-name { color:#ecfeff; text-shadow:0 0 12px rgba(34,211,238,.28); }
    @keyframes runningRowGlow { 0%, 100% { background-position:0% 50%; box-shadow:inset 3px 0 0 rgba(34,211,238,.82), 0 0 12px rgba(34,211,238,.06); } 50% { background-position:100% 50%; box-shadow:inset 3px 0 0 rgba(103,232,249,1), 0 0 24px rgba(34,211,238,.16); } }
    .problem-row.matrix-header { position:sticky; top:0; z-index:1; background:#0b1220; border-bottom:1px solid var(--line); color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
    .problem-main { min-width:0; display:grid; gap:2px; }
    .problem-name { color:#dbeafe; font-size:12px; font-weight:750; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .problem-meta { color:var(--muted); font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .problem-context { min-width:0; display:grid; gap:4px; align-content:center; color:var(--muted); font-size:11px; }
    .context-line { display:flex; gap:5px; align-items:center; flex-wrap:wrap; min-width:0; }
    .context-chip { border:1px solid #263548; border-radius:999px; padding:2px 7px; background:#07101e; color:#cbd5e1; white-space:nowrap; }
    .context-chip.pass { border-color:rgba(52,211,153,.45); color:var(--pass); background:rgba(52,211,153,.08); }
    .context-chip.reduced { border-color:rgba(251,191,36,.45); color:var(--reduced); background:rgba(251,191,36,.08); }
    .context-chip.fail { border-color:rgba(248,113,113,.45); color:var(--fail); background:rgba(248,113,113,.08); }
    .context-chip.running { border-color:rgba(34,211,238,.65); color:#a5f3fc; background:rgba(34,211,238,.1); }
    .context-detail { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#94a3b8; }
    .context-meter { display:flex; height:6px; overflow:hidden; border-radius:999px; background:#020617; border:1px solid #1e293b; }
    .context-meter span { min-width:0; }
    .context-meter .pass { background:var(--pass); }
    .context-meter .reduced { background:var(--reduced); }
    .context-meter .fail { background:var(--fail); }
    .context-meter .running { background:#22d3ee; }
    .context-meter .staged { background:var(--accent); }
    .context-meter .unsolved { background:#475569; }
    .solver-cell { position:relative; overflow:hidden; isolation:isolate; min-height:28px; border:1px solid #263548; border-radius:8px; padding:4px 7px; display:flex; justify-content:space-between; align-items:center; gap:6px; font-size:11px; color:var(--muted); background:#07101e; }
    .solver-cell .solver-label, .solver-cell .cell-note { position:relative; z-index:1; }
    .solver-cell .solver-label { font-weight:800; }
    .solver-cell .cell-note { opacity:.78; font-size:10px; }
    .solver-cell.unsolved { color:#64748b; background:#07101e; }
    .solver-cell.staged { color:#bfdbfe; border-color:rgba(96,165,250,.72); background:rgba(96,165,250,.12); }
    .solver-cell.running { color:#a5f3fc; border-color:rgba(103,232,249,.95); background:linear-gradient(90deg, rgba(8,47,73,.82), rgba(6,78,59,.42)); box-shadow:0 0 0 1px rgba(34,211,238,.35), 0 0 18px rgba(34,211,238,.22); animation:runningCellPulse 1.25s ease-in-out infinite; }
    .solver-cell.running .solver-label { display:inline-flex; align-items:center; gap:6px; color:#ecfeff; }
    .solver-cell.running .solver-label::before { content:""; width:10px; height:10px; flex:0 0 auto; border-radius:999px; border:2px solid rgba(165,243,252,.32); border-top-color:#67e8f9; box-shadow:0 0 8px rgba(34,211,238,.55); animation:spin .7s linear infinite; }
    .solver-cell.running::after { content:""; position:absolute; inset:-1px; z-index:0; transform:translateX(-125%); background:linear-gradient(105deg, transparent 0%, rgba(255,255,255,.04) 32%, rgba(125,211,252,.34) 48%, rgba(255,255,255,.05) 62%, transparent 100%); animation:runningCellSweep 1.45s ease-in-out infinite; }
    @keyframes runningCellPulse { 0%, 100% { transform:translateY(0); } 50% { transform:translateY(-1px); } }
    @keyframes runningCellSweep { from { transform:translateX(-125%); } to { transform:translateX(125%); } }
    .solver-cell.cache_hit { color:#7dd3fc; border-color:rgba(56,189,248,.55); background:rgba(56,189,248,.1); }
    .solver-cell.passed { color:var(--pass); border-color:rgba(52,211,153,.55); background:rgba(52,211,153,.11); }
    .solver-cell.reduced_accuracy { color:var(--reduced); border-color:rgba(251,191,36,.55); background:rgba(251,191,36,.12); }
    .solver-cell.solve_error, .solver-cell.failed_validation { color:var(--fail); border-color:rgba(248,113,113,.58); background:rgba(248,113,113,.12); }
    .solver-cell.staged.passed, .solver-cell.staged.reduced_accuracy, .solver-cell.staged.solve_error, .solver-cell.staged.failed_validation { box-shadow:inset 0 0 0 1px rgba(96,165,250,.55); }
    .problem-row.all-unsolved:not(.staged-row) { opacity:.62; }
    @media (prefers-reduced-motion: reduce) {
      .problem-row.running-row, .solver-cell.running, .solver-cell.running .solver-label::before, .solver-cell.running::after { animation:none; }
      .solver-cell.running::after { display:none; }
    }
    .cache-hit { color:var(--hit); font-weight:800; }
    .cache-miss { color:var(--miss); font-weight:800; }
    .dashboard-shell { display:grid; gap:14px; }
    .dashboard-card h2 { margin:0; font-size:20px; }
    .dashboard-summary { display:grid; grid-template-columns:minmax(330px, .72fr) minmax(520px, 1.28fr); gap:12px; align-items:start; }
    .dashboard-breakdown-grid { display:grid; grid-template-columns:minmax(150px, .6fr) minmax(300px, 1.4fr) minmax(190px, .8fr); gap:10px; }
    .dashboard-summary .dashboard-grid { grid-template-columns:repeat(2, minmax(130px, 1fr)); }
    .dashboard-table-head { display:flex; justify-content:space-between; gap:10px; align-items:baseline; margin:14px 0 6px; }
    .dashboard-table-head h3 { margin:0; font-size:15px; }
    .dashboard-grid { display:grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap:10px; }
    .failure-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin-top:8px; }
    .failure-card { display:grid; grid-template-columns:82px 1fr; gap:10px; align-items:center; border:1px solid var(--line); border-radius:12px; background:#0b1220; padding:10px; min-width:0; }
    .failure-card.clean { border-color:rgba(52,211,153,.38); background:rgba(52,211,153,.07); }
    .failure-donut { position:relative; width:72px; height:72px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(#475569 0deg 360deg); box-shadow:0 0 24px rgba(248,113,113,.08); }
    .failure-donut::after { content:""; position:absolute; width:44px; height:44px; border-radius:50%; background:#0b1220; border:1px solid rgba(148,163,184,.18); }
    .failure-donut.clean::after { background:#0d1f22; }
    .failure-donut-inner { position:relative; z-index:1; text-align:center; font-size:11px; color:#e2e8f0; font-weight:850; }
    .failure-copy { min-width:0; display:grid; gap:5px; }
    .failure-title { display:flex; justify-content:space-between; gap:8px; align-items:baseline; font-size:12px; font-weight:850; color:#dbeafe; }
    .failure-title span:last-child { color:var(--muted); font-weight:700; }
    .failure-list { display:grid; gap:4px; }
    .failure-row { display:grid; grid-template-columns:10px minmax(0, 1fr) auto; gap:6px; align-items:center; color:#cbd5e1; font-size:11px; }
    .failure-swatch { width:9px; height:9px; border-radius:999px; box-shadow:0 0 0 1px rgba(255,255,255,.12); }
    .failure-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .dash-table { width:100%; border-collapse:collapse; font-size:13px; }
    .dash-table th, .dash-table td { border-bottom:1px solid #243244; padding:8px; text-align:left; vertical-align:top; }
    .dash-table th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
    .status-pill { border-radius:999px; padding:2px 8px; font-weight:800; font-size:11px; }
    .status-pill.passed { background:rgba(52,211,153,.18); color:var(--pass); }
    .status-pill.reduced_accuracy { background:rgba(251,191,36,.18); color:var(--reduced); }
    .status-pill.solve_error, .status-pill.failed_validation { background:rgba(248,113,113,.18); color:var(--fail); }
    details { border-top:1px solid #243244; padding-top:10px; }
    summary { cursor:pointer; color:var(--muted); font-weight:700; }
    #events { margin-top:10px; min-height:72px; max-height:180px; overflow:auto; white-space:pre-wrap; color:#cbd5e1; background:#020617; border-radius:10px; padding:10px; }
    @media (max-width: 1100px) {
      .progress-shell {
        grid-template-columns:1fr;
      }
      .config-grid, .config-main, .preview-grid, .dashboard-grid, .dashboard-summary, .dashboard-breakdown-grid { grid-template-columns: 1fr 1fr; }
      .preview-list { grid-template-columns:1fr; }
    }
    @media (max-width: 720px) { .progress-shell, .config-grid, .config-main, .action-stack, .preview-grid, .dashboard-grid, .dashboard-summary, .dashboard-breakdown-grid, .stat-grid, .stage-panel { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Test Problem Studio</h1>
    <div class="sub">Parallel solve runner with persistent result cache. Cached rows preserve original solver timings.</div>
  </header>
  <section class="card">
    <div class="config-grid">
      <div class="control-label">Test set <div id="test-set" class="segmented"></div></div>
      <div class="config-main">
        <div class="control-label">Problem set <div id="problem-set" class="segmented">
          <label><input type="radio" name="problem-set" value="fast" checked>Fast</label>
          <label><input type="radio" name="problem-set" value="slow">Slow</label>
          <label><input type="radio" name="problem-set" value="">All</label>
        </div></div>
        <div class="control-label">Solvers <div id="solver-checks" class="checks">
          <label><input type="checkbox" value="sqp" checked>SQP</label>
          <label><input type="checkbox" value="nlip" checked>NLIP</label>
        </div></div>
        <label>Jobs <input id="jobs" type="number" min="1" value="4"></label>
        <div class="action-stack">
          <button id="run">Run</button>
          <button id="stop-run" class="danger" type="button" disabled>Stop</button>
          <button id="clear-results-cache" class="secondary" type="button">Clear Results Cache</button>
        </div>
      </div>
    </div>
    <p class="sub" id="catalog"><span class="loading-line"><span class="spinner"></span>Loading catalog...</span></p>
  </section>
  <section class="card progress-shell">
    <div class="progress-main">
      <div class="progress-top">
        <div>
          <div class="progress-title" id="progress-title">Ready</div>
          <div class="progress-sub" id="progress-sub">Choose a suite and run the matrix.</div>
        </div>
        <div class="progress-percent" id="progress-percent">0%</div>
      </div>
      <div id="progress-track" class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
      <div id="result-segments" class="segment-track" aria-label="result mix">
        <div id="seg-pass" class="segment pass" title="passed"></div>
        <div id="seg-reduced" class="segment reduced" title="reduced accuracy"></div>
        <div id="seg-fail" class="segment fail" title="failed"></div>
      </div>
      <div id="cache-segments" class="segment-track" aria-label="cache mix">
        <div id="seg-hit" class="segment hit" title="cache hits"></div>
        <div id="seg-miss" class="segment miss" title="fresh solves"></div>
      </div>
      <div class="stat-grid">
        <div class="stat"><div class="k">Completed</div><div class="v" id="stat-completed">0/0</div></div>
        <div class="stat"><div class="k">Passed</div><div class="v" id="stat-pass">0</div></div>
        <div class="stat"><div class="k">Reduced</div><div class="v" id="stat-reduced">0</div></div>
        <div class="stat"><div class="k">Failed</div><div class="v" id="stat-fail">0</div></div>
        <div class="stat"><div class="k">Cache hits</div><div class="v" id="stat-hit">0</div></div>
        <div class="stat"><div class="k">Fresh</div><div class="v" id="stat-miss">0</div></div>
      </div>
      <div class="active-block">
        <div class="progress-sub" id="active-label" style="margin-bottom:6px">Running now</div>
        <div id="active-tasks" class="active-list"><span class="task-chip">idle</span></div>
      </div>
      <div id="stage-panel" class="stage-panel"></div>
    </div>
    <div class="progress-side">
      <div class="preview-grid">
        <div class="donut-card">
          <div id="cache-donut" class="donut"><div class="donut-inner"><div class="big" id="preview-total">0</div><div class="progress-sub">planned runs</div></div></div>
        </div>
        <div class="stat">
          <div class="k">Preview by solver</div>
          <div id="preview-solvers" class="mini-bars"></div>
        </div>
        <div class="stat">
          <div class="k">Preview by test set</div>
          <div id="preview-suites" class="mini-bars"></div>
        </div>
      </div>
      <div class="stat subset-card">
        <div class="k">Selected subset</div>
        <div id="preview-scope" class="scope-grid"></div>
      </div>
      <div class="stat families-card">
        <div class="k">Largest planned families</div>
        <div id="preview-families" class="preview-list"></div>
      </div>
    </div>
  </section>
  <section id="live-dashboard" class="dashboard-shell"></section>
  <section class="card matrix-card">
      <div class="matrix-head">
        <div>
          <div class="k">Problem status matrix</div>
          <div class="progress-sub" id="problem-matrix-summary">Loading problem statuses...</div>
        </div>
        <div class="legend">
          <span class="legend-item"><span class="legend-dot passed"></span>passed</span>
          <span class="legend-item"><span class="legend-dot reduced"></span>reduced</span>
          <span class="legend-item"><span class="legend-dot failed"></span>failed</span>
          <span class="legend-item"><span class="legend-dot staged"></span>staged</span>
          <span class="legend-item"><span class="legend-dot"></span>unsolved</span>
        </div>
      </div>
      <div id="problem-matrix" class="problem-matrix"></div>
  </section>
  <section class="card">
    <details>
      <summary>Detailed event log</summary>
      <div id="events">Ready.</div>
    </details>
  </section>
</main>
<script>
window.addEventListener('error', (event) => {
  const message = event && event.message ? event.message : 'unknown script error';
  document.documentElement.setAttribute('data-script-error', message);
  const box = document.getElementById('events');
  if (box) box.textContent = `UI script error: ${message}`;
});
window.addEventListener('unhandledrejection', (event) => {
  const message = event && event.reason ? String(event.reason.message || event.reason) : 'unknown rejected promise';
  document.documentElement.setAttribute('data-script-error', message);
  const box = document.getElementById('events');
  if (box) box.textContent = `UI async error: ${message}`;
});
</script>
<script>
const byId = (id) => document.getElementById(id);
let progress = newProgressState();
let previewEntries = [];
let resultRecords = [];
let cachedRecords = [];
let transientCellStates = new Map();
let previewTimer = null;
let jobsUpdateTimer = null;
let catalogData = null;
let currentRunId = null;
let runActive = false;
async function loadCatalog() {
  byId('catalog').innerHTML = loadingLine('Loading catalog...');
  const catalog = await fetch('/api/catalog').then((r) => r.json());
  catalogData = catalog;
  byId('test-set').innerHTML = '<label><input type="radio" name="test-set" value="" checked>All</label>'
    + catalog.suites.map((s) => `<label><input type="radio" name="test-set" value="${s}">${s}</label>`).join('');
  if (catalog.solvers.includes('ipopt') && !byId('solver-checks').querySelector('input[value="ipopt"]')) {
    byId('solver-checks').insertAdjacentHTML('beforeend', '<label><input type="checkbox" value="ipopt">IPOPT</label>');
  }
  byId('catalog').textContent = `${catalog.problems.length} problems · cache ${catalog.cache_dir}`;
  renderProblemMatrixLoading('Loading cached result statuses...');
  await loadCachedRecords();
  schedulePreview();
}
function log(line) {
  const box = byId('events');
  box.textContent += `\n${line}`;
  box.scrollTop = box.scrollHeight;
}
function newProgressState() {
  const jobsInput = byId('jobs');
  return {
    total:0,
    completed:0,
    pass:0,
    reduced:0,
    fail:0,
    hit:0,
    miss:0,
    active:new Map(),
    startedAt:0,
    jobs:Number(jobsInput ? jobsInput.value : 4),
    stage:'preview',
    stageMessage:'Choose a suite and run the matrix.',
    stopped:false,
  };
}
function resetProgress() {
  progress = newProgressState();
  progress.jobs = Number(byId('jobs').value || 4);
  resultRecords = [];
  transientCellStates = new Map();
  progress.startedAt = Date.now();
  progress.stage = 'planning';
  progress.stageMessage = 'Starting run and planning selected solver-runs.';
  progress.stopped = false;
  byId('events').textContent = 'Starting run...';
  renderProgress();
  renderProblemMatrix();
  renderLiveDashboard();
}
function renderProgress() {
  const total = Math.max(progress.total, 0);
  const completed = progress.completed;
  const pct = total ? Math.round(completed / total * 100) : 0;
  const hasStarted = progress.startedAt > 0;
  const stageTitle = stageTitleText(progress.stage);
  byId('progress-title').textContent = hasStarted ? stageTitle : 'Run preview';
  byId('progress-sub').textContent = total
    ? (hasStarted ? `${progress.stageMessage || stageTitle} · ${completed} of ${total} done · ${elapsedText()}` : `${total} planned runs for the current filters`)
    : (progress.stageMessage || 'Choose a suite and run the matrix.');
  const progressTitle = total ? `Completed ${completed} of ${total} solver-runs (${percentText(completed, total)}).` : 'No solver-runs planned yet.';
  byId('progress-track').title = progressTitle;
  byId('progress-fill').title = progressTitle;
  byId('progress-percent').title = progressTitle;
  byId('progress-percent').textContent = `${pct}%`;
  byId('progress-fill').style.width = `${pct}%`;
  byId('progress-fill').classList.toggle('running', total > 0 && completed < total);
  setStat('stat-completed', `${completed}/${total}`, `Completed solver-runs: ${completed}/${total || 0}.`);
  setStat('stat-pass', progress.pass, `Passed solver-runs: ${progress.pass}/${total || 0}.`);
  setStat('stat-reduced', progress.reduced, `Reduced-accuracy solver-runs: ${progress.reduced}/${total || 0}. These are accepted, but with relaxed validation.`);
  setStat('stat-fail', progress.fail, `Failed solver-runs: ${progress.fail}/${total || 0}. See the problem matrix and dashboard rows for failure types.`);
  setStat('stat-hit', progress.hit, `Result-cache hits: ${progress.hit}/${total || 0}. Cached rows preserve original solver timing.`);
  setStat('stat-miss', progress.miss, `Fresh solves: ${progress.miss}/${total || 0}. These were or will be computed instead of loaded from the result cache.`);
  setSegment('seg-pass', progress.pass, total, 'Passed');
  setSegment('seg-reduced', progress.reduced, total, 'Reduced accuracy');
  setSegment('seg-fail', progress.fail, total, 'Failed');
  setSegment('seg-hit', progress.hit, total, 'Result-cache hits');
  setSegment('seg-miss', progress.miss, total, 'Fresh solves');
  byId('result-segments').title = `Result mix: ${progress.pass} passed, ${progress.reduced} reduced accuracy, ${progress.fail} failed out of ${total || 0}.`;
  byId('cache-segments').title = `Cache mix: ${progress.hit} result-cache hits, ${progress.miss} fresh solves out of ${total || 0}.`;
  const active = Array.from(progress.active.values());
  const remaining = Math.max(0, total - completed);
  const slots = Math.max(progress.jobs || 0, active.length);
  byId('active-label').textContent = total ? `Running now ${active.length}/${slots} slots · ${remaining} remaining` : 'Running now';
  const visibleActive = active.slice(0, 12);
  const overflow = active.length - visibleActive.length;
  byId('active-tasks').innerHTML = visibleActive.length
    ? visibleActive.map((text) => `<span class="task-chip">${escapeHtml(text)}</span>`).join('') + (overflow > 0 ? `<span class="task-chip">+${overflow} more</span>` : '')
    : '<span class="task-chip">idle</span>';
  renderStagePanel();
}
function renderStagePanel() {
  const total = Math.max(progress.total, 0);
  const completed = Math.max(progress.completed, 0);
  const remaining = Math.max(0, total - completed);
  const active = progress.active.size;
  const stages = [
    { key:'planning', label:'Plan', detail: total ? `${total} queued` : 'select filters' },
    { key:'solving', label: progress.stage === 'stopping' ? 'Stopping' : 'Solve', detail: `${completed}/${total || 0} done · ${active}/${progress.jobs} active · ${remaining} left` },
    { key:'collecting_results', label:'Collect', detail: resultRecords.length ? `${resultRecords.length} records` : 'waiting' },
    { key:'refreshing_cache', label:'Refresh', detail:'cache + preview' },
    { key:'complete', label: progress.stopped ? 'Stopped' : 'Done', detail: progress.stopped ? `${completed}/${total || 0} completed` : `${progress.pass + progress.reduced}/${total || 0} accepted` },
  ];
  const current = stageGroup(progress.stage);
  byId('stage-panel').innerHTML = stages.map((stage) => {
    let className = 'stage-step';
    if (stage.key === current) className += progress.stage === 'stopping' || progress.stage === 'stopped' ? ' blocked' : ' active';
    else if (stageRank(stage.key) < stageRank(current)) className += ' done';
    const title = `${stage.label}: ${stage.detail}. Current stage: ${stageTitleText(progress.stage)}.`;
    return `<div class="${className}" title="${escapeHtml(title)}"><div class="stage-name">${escapeHtml(stage.label)}</div><div class="stage-detail">${escapeHtml(stage.detail)}</div></div>`;
  }).join('');
}
function stageGroup(stage) {
  if (stage === 'queued') return 'planning';
  if (stage === 'stopping') return 'solving';
  if (stage === 'stopped') return 'complete';
  if (stage === 'preview') return 'planning';
  return stage || 'planning';
}
function stageRank(stage) {
  const order = { planning:0, solving:1, collecting_results:2, refreshing_cache:3, complete:4 };
  return order[stageGroup(stage)] == null ? 0 : order[stageGroup(stage)];
}
function stageTitleText(stage) {
  if (stage === 'planning') return 'Planning run';
  if (stage === 'queued') return 'Run queued';
  if (stage === 'solving') return 'Solving problems';
  if (stage === 'stopping') return 'Stopping run';
  if (stage === 'collecting_results') return 'Collecting results';
  if (stage === 'refreshing_cache') return 'Refreshing dashboard';
  if (stage === 'complete') return 'Run complete';
  if (stage === 'stopped') return 'Run stopped';
  if (stage === 'error') return 'Run error';
  return 'Run preview';
}
function renderPreview(entries) {
  previewEntries = entries || [];
  const total = previewEntries.length;
  const hits = previewEntries.filter((entry) => entry.cache_status === 'hit').length;
  const misses = total - hits;
  if (!progress.startedAt || progress.completed === 0) {
    progress.total = total;
    renderProgress();
  }
  byId('preview-total').textContent = String(total);
  const hitDeg = total ? hits / total * 360 : 0;
  byId('cache-donut').title = `Planned solver-runs: ${total}. Cached results: ${hits}. Fresh solves: ${misses}.`;
  byId('cache-donut').style.background = `conic-gradient(var(--hit) 0deg ${hitDeg}deg, var(--miss) ${hitDeg}deg 360deg)`;
  const solverCounts = groupCountsWithCache(previewEntries, (entry) => entry.solver);
  const suiteCounts = groupCountsWithCache(previewEntries, (entry) => entry.test_set);
  byId('preview-solvers').innerHTML = miniBars(solverCounts.rows, total, solverCounts.details);
  byId('preview-suites').innerHTML = miniBars(suiteCounts.rows, total, suiteCounts.details);
  byId('preview-scope').innerHTML = renderSubsetScope();
  byId('preview-families').innerHTML = previewList(groupCounts(previewEntries, (entry) => `${entry.test_set} / ${entry.family}`), total, hits, misses);
  renderProblemMatrix();
}
function renderPreviewLoading(message) {
  if (!progress.startedAt) {
    byId('progress-title').textContent = 'Loading preview';
    byId('progress-sub').innerHTML = loadingLine(message);
  }
  byId('preview-total').textContent = '...';
  byId('cache-donut').style.background = 'conic-gradient(#334155 0deg, #334155 360deg)';
  byId('preview-solvers').innerHTML = skeletonBars(2);
  byId('preview-suites').innerHTML = skeletonBars(3);
  byId('preview-scope').innerHTML = `<div>${skeletonBars(4)}</div><div>${skeletonBars(3)}</div>`;
  byId('preview-families').innerHTML = skeletonItems(8);
}
async function loadCachedRecords() {
  const response = await fetch('/api/results_cache/records');
  if (!response.ok) throw new Error(await response.text());
  const results = await response.json();
  cachedRecords = results.records || [];
  renderProblemMatrix();
}
function renderProblemMatrixLoading(message) {
  byId('problem-matrix-summary').innerHTML = loadingLine(message);
  byId('problem-matrix').innerHTML = skeletonItems(10);
}
function renderProblemMatrix() {
  if (!catalogData) {
    renderProblemMatrixLoading('Loading problem catalog...');
    return;
  }
  const solvers = catalogData.solvers || checkedSolverValues();
  const staged = stagedCellMap();
  const records = solvedRecordMap();
  const counts = { passed:0, reduced:0, failed:0, staged:0, unsolved:0, running:0 };
  const columns = `minmax(270px, 340px) minmax(360px, 1fr) repeat(${solvers.length}, minmax(112px, 130px))`;
  const header = `<div class="problem-row matrix-header" style="grid-template-columns:${columns}"><div>Problem</div><div>Summary</div>${solvers.map((solver) => `<div>${escapeHtml(solver.toUpperCase())}</div>`).join('')}</div>`;
  const rowData = catalogData.problems.map((problem) => {
    let rowSolved = false;
    let rowStaged = false;
    let rowRunning = false;
    const rowStates = [];
    const cells = solvers.map((solver) => {
      const key = cellKey(problem.id, solver);
      const stagedEntry = staged.get(key);
      const live = transientCellStates.get(key);
      const record = records.get(key);
      const state = cellState(record, stagedEntry, live);
      rowStates.push({ solver, state, record, staged: Boolean(stagedEntry) });
      if (state.kind === 'solved') rowSolved = true;
      if (stagedEntry || state.kind === 'running') rowStaged = true;
      if (state.kind === 'running') rowRunning = true;
      if (state.kind === 'running') counts.running += 1;
      else if (state.kind === 'solved' && state.status === 'passed') counts.passed += 1;
      else if (state.kind === 'solved' && state.status === 'reduced_accuracy') counts.reduced += 1;
      else if (state.kind === 'solved') counts.failed += 1;
      else if (state.kind === 'staged') counts.staged += 1;
      else counts.unsolved += 1;
      return solverCellHtml(solver, state, Boolean(stagedEntry));
    }).join('');
    const rowClass = `${rowSolved ? '' : 'all-unsolved'} ${rowStaged ? 'staged-row' : ''} ${rowRunning ? 'running-row' : ''}`;
    const priority = rowRunning ? 0 : (rowStaged ? 1 : (rowSolved ? 2 : 3));
    return {
      problem,
      priority,
      html: `<div class="problem-row ${rowClass}" style="grid-template-columns:${columns}">
      <div class="problem-main">
        <div class="problem-name" title="${escapeHtml(problem.id)}">${escapeHtml(problem.id)}</div>
        <div class="problem-meta">${escapeHtml(problem.test_set)} · ${escapeHtml(problem.family)} · ${escapeHtml(problemSpeed(problem))}</div>
      </div>
      ${problemContextHtml(problem, rowStates)}
      ${cells}
    </div>`
    };
  }).sort((left, right) =>
    left.priority - right.priority
      || String(left.problem.test_set).localeCompare(String(right.problem.test_set))
      || String(left.problem.family).localeCompare(String(right.problem.family))
      || String(left.problem.id).localeCompare(String(right.problem.id))
  );
  const rows = rowData.map((row) => row.html).join('');
  const stagedRuns = previewEntries.length;
  const solvedRuns = counts.passed + counts.reduced + counts.failed;
  byId('problem-matrix-summary').textContent = `${catalogData.problems.length} problems · ${counts.running} running · ${solvedRuns} solved cells · ${stagedRuns} staged solver-runs · ${counts.unsolved} unsolved cells`;
  byId('problem-matrix').innerHTML = header + rows;
}
function stagedCellMap() {
  const map = new Map();
  for (const entry of previewEntries) map.set(cellKey(entry.problem_id, entry.solver), entry);
  return map;
}
function solvedRecordMap() {
  const map = new Map();
  for (const record of cachedRecords) map.set(cellKey(record.id, record.solver), record);
  for (const record of resultRecords) map.set(cellKey(record.id, record.solver), record);
  return map;
}
function cellKey(problemId, solver) {
  return `${problemId}::${solver}`;
}
function cellState(record, stagedEntry, live) {
  if (live && live.kind === 'running') return live;
  if (live && live.kind === 'solved') return live;
  if (record) return { kind:'solved', status:record.status, cacheStatus:record.cache ? record.cache.status : 'none', record };
  if (stagedEntry) return { kind:'staged', cacheStatus:stagedEntry.cache_status };
  return { kind:'unsolved' };
}
function problemContextHtml(problem, rowStates) {
  const total = Math.max(1, rowStates.length);
  const solved = rowStates.filter((entry) => entry.state.kind === 'solved');
  const running = rowStates.filter((entry) => entry.state.kind === 'running');
  const staged = rowStates.filter((entry) => entry.state.kind === 'staged' || entry.staged);
  const pass = solved.filter((entry) => entry.state.status === 'passed').length;
  const reduced = solved.filter((entry) => entry.state.status === 'reduced_accuracy').length;
  const fail = solved.length - pass - reduced;
  const unsolved = rowStates.filter((entry) => entry.state.kind === 'unsolved').length;
  const stagedOnly = Math.max(0, staged.length - running.length - solved.filter((entry) => entry.staged).length);
  const record = solved.map((entry) => entry.record || entry.state.record).find(Boolean);
  const descriptor = record ? record.descriptor : null;
  const sizeText = descriptor
    ? `n=${descriptor.num_vars} · dof=${descriptor.dof} · eq=${descriptor.num_eq} · ineq=${descriptor.num_ineq} · box=${descriptor.num_box}`
    : `${problem.variant || problem.family} · ${problem.source}`;
  const accepted = pass + reduced;
  const outcomeText = solved.length
    ? `${accepted}/${solved.length} accepted`
    : (staged.length ? `${staged.length}/${rowStates.length} staged` : 'not staged');
  const fastestRecord = solved.map((entry) => entry.record || entry.state.record).filter(Boolean).sort((a, b) => a.timing.total_wall_time - b.timing.total_wall_time)[0];
  const failureRows = groupCounts(solved.filter((entry) => !isAcceptedStatus(entry.state.status)).map((entry) => entry.record || entry.state.record || entry.state), failureCategory);
  const detailParts = [];
  if (running.length) detailParts.push(`${running.length} running`);
  if (failureRows.length) detailParts.push(`fail: ${failureRows.slice(0, 2).map(([label, count]) => `${label} ${count}`).join(', ')}`);
  if (fastestRecord) detailParts.push(`fastest ${fastestRecord.solver} ${formatDuration(fastestRecord.timing.total_wall_time)}`);
  if (!detailParts.length) detailParts.push(staged.length ? `${staged.length} selected solver-runs` : 'no selected or cached solver result');
  const title = `${problem.id}: ${sizeText}. ${outcomeText}. ${detailParts.join('. ')}.`;
  return `<div class="problem-context" title="${escapeHtml(title)}">
    <div class="context-line">
      <span class="context-chip">${escapeHtml(outcomeText)}</span>
      ${running.length ? `<span class="context-chip running">${running.length} running</span>` : ''}
      ${pass ? `<span class="context-chip pass">${pass} pass</span>` : ''}
      ${reduced ? `<span class="context-chip reduced">${reduced} reduced</span>` : ''}
      ${fail ? `<span class="context-chip fail">${fail} fail</span>` : ''}
    </div>
    ${contextMeter({ pass, reduced, fail, running:running.length, staged:stagedOnly, unsolved }, total)}
    <div class="context-detail">${escapeHtml(sizeText)} · ${escapeHtml(detailParts.join(' · '))}</div>
  </div>`;
}
function contextMeter(counts, total) {
  const segments = [
    ['pass', counts.pass],
    ['reduced', counts.reduced],
    ['fail', counts.fail],
    ['running', counts.running],
    ['staged', counts.staged],
    ['unsolved', counts.unsolved],
  ].filter(([, value]) => value > 0);
  return `<div class="context-meter" title="${escapeHtml(segments.map(([label, value]) => `${label}: ${value}`).join('; '))}">${segments.map(([label, value]) => `<span class="${label}" style="width:${value / total * 100}%"></span>`).join('')}</div>`;
}
function solverCellHtml(solver, state, staged) {
  const stagedClass = staged ? ' staged' : '';
  if (state.kind === 'running') {
    const title = `${solver}: currently running${state.cacheStatus === 'hit' ? ' from result cache' : ''}.`;
    return `<div class="solver-cell running${stagedClass}" title="${escapeHtml(title)}"><span class="solver-label">running</span><span class="cell-note">${escapeHtml(solver)}</span></div>`;
  }
  if (state.kind === 'solved') {
    const title = `${solver}: ${statusShortLabel(state.status)}${state.cacheStatus === 'hit' ? ' from the results cache' : ' from the current run'}${staged ? '; also staged by current filters' : ''}.`;
    return `<div class="solver-cell ${escapeHtml(state.status)}${stagedClass}" title="${escapeHtml(title)}"><span class="solver-label">${statusShortLabel(state.status)}</span><span class="cell-note">${state.cacheStatus === 'hit' ? 'cached' : 'done'}</span></div>`;
  }
  if (state.kind === 'staged') {
    const note = state.cacheStatus === 'hit' ? 'cached' : 'fresh';
    const title = `${solver}: staged to solve with the current filters; ${state.cacheStatus === 'hit' ? 'a cached result is available' : 'will require a fresh solve'}.`;
    return `<div class="solver-cell staged" title="${escapeHtml(title)}"><span class="solver-label">staged</span><span class="cell-note">${note}</span></div>`;
  }
  const title = `${solver}: no cached or current result; not staged by the current filters.`;
  return `<div class="solver-cell unsolved" title="${escapeHtml(title)}"><span class="solver-label">unsolved</span><span class="cell-note">--</span></div>`;
}
function statusShortLabel(status) {
  if (status === 'passed') return 'passed';
  if (status === 'reduced_accuracy') return 'reduced';
  return 'failed';
}
function groupCounts(items, keyFn) {
  const counts = new Map();
  for (const item of items) counts.set(keyFn(item), (counts.get(keyFn(item)) || 0) + 1);
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
}
function groupCountsWithCache(items, keyFn) {
  const details = new Map();
  for (const item of items) {
    const label = keyFn(item);
    const current = details.get(label) || { total: 0, hits: 0, misses: 0 };
    current.total += 1;
    if (item.cache_status === 'hit') current.hits += 1;
    else current.misses += 1;
    details.set(label, current);
  }
  return {
    rows: Array.from(details.entries()).map(([label, detail]) => [label, detail.total]).sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]))),
    details,
  };
}
function miniBars(rows, total, details) {
  if (!rows.length) return '<div class="progress-sub">No runs selected.</div>';
  return rows.map(([label, count]) => {
    const detail = details ? details.get(label) : null;
    const cacheText = detail ? ` Cached: ${detail.hits}; fresh: ${detail.misses}.` : '';
    const title = `${label}: ${count}/${total || 0} solver-runs (${percentText(count, total)}).${cacheText}`;
    return `<div class="mini-row" title="${escapeHtml(title)}"><div class="mini-label" title="${escapeHtml(title)}">${escapeHtml(label)}</div><div class="mini-track" title="${escapeHtml(title)}"><div class="mini-fill" style="width:${total ? count / total * 100 : 0}%"></div></div><div class="progress-sub">${count}</div></div>`;
  }).join('');
}
function skeletonBars(count) {
  return Array.from({ length: count }, (_, index) => `<div class="mini-row"><div class="skeleton ${index % 2 ? 'short' : 'medium'}"></div><div class="mini-track"><div class="skeleton long"></div></div><div class="skeleton short"></div></div>`).join('');
}
function skeletonItems(count) {
  return Array.from({ length: count }, (_, index) => `<div class="preview-item"><span class="skeleton ${index % 2 ? 'medium' : 'long'}"></span><span class="skeleton short"></span></div>`).join('');
}
function loadingLine(message) {
  return `<span class="loading-line"><span class="spinner"></span>${escapeHtml(message)}</span>`;
}
function renderSubsetScope() {
  if (!catalogData) return '<div class="progress-sub">Loading catalog...</div>';
  const suiteRows = catalogData.suites.map((suite) => subsetRow(
    suite,
    (problem) => problem.test_set === suite && matchesTestSet(problem) && matchesProblemSet(problem),
    (problem) => problem.test_set === suite
  ));
  const speedRows = ['fast', 'slow'].map((speed) => subsetRow(
    speed,
    (problem) => problemSpeed(problem) === speed && matchesTestSet(problem) && matchesProblemSet(problem),
    (problem) => problemSpeed(problem) === speed
  ));
  return `<div>${scopeBars('Test set', suiteRows)}</div><div>${scopeBars('Problem set', speedRows)}</div>`;
}
function subsetRow(label, selectedPredicate, basePredicate) {
  const multiplier = Math.max(1, checkedSolverValues().length);
  const baseProblems = catalogData.problems.filter(basePredicate);
  const selectedProblems = baseProblems.filter(selectedPredicate);
  return { label, selected: selectedProblems.length * multiplier, total: baseProblems.length * multiplier };
}
function scopeBars(title, rows) {
  const maxTotal = Math.max(1, ...rows.map((row) => row.total));
  return `<div class="mini-bars"><div class="progress-sub">${escapeHtml(title)}</div>${rows.map((row) => {
    const selectedPct = row.total ? row.selected / row.total * 100 : 0;
    const totalPct = row.total / maxTotal * 100;
    const unselected = row.selected === 0 ? ' unselected' : '';
    const titleText = `${row.label}: ${row.selected}/${row.total} solver-runs staged by the current filters (${percentText(row.selected, row.total)}). Grey width shows this bucket's size relative to the largest bucket.`;
    return `<div class="scope-row" title="${escapeHtml(titleText)}"><div class="scope-label${unselected}" title="${escapeHtml(titleText)}">${escapeHtml(row.label)}</div><div class="scope-track" title="${escapeHtml(titleText)}" style="width:${totalPct}%"><div class="scope-fill" style="width:${selectedPct}%"></div></div><div class="scope-count">${row.selected}/${row.total}</div></div>`;
  }).join('')}</div>`;
}
function matchesTestSet(problem) {
  const testSet = checkedRadio('test-set');
  return !testSet || problem.test_set === testSet;
}
function matchesProblemSet(problem) {
  const problemSet = checkedRadio('problem-set');
  return !problemSet || problemSpeed(problem) === problemSet;
}
function problemSpeed(problem) {
  return String(problem.speed || '').toLowerCase();
}
function previewList(rows, total) {
  if (!rows.length) return '<div class="progress-sub">No planned families.</div>';
  return rows.slice(0, 18).map(([label, count]) => {
    const familyEntries = previewEntries.filter((entry) => `${entry.test_set} / ${entry.family}` === label);
    const hits = familyEntries.filter((entry) => entry.cache_status === 'hit').length;
    return `<div class="preview-item"><span>${escapeHtml(label)}</span><span><span class="cache-hit">${hits} cached</span> · <span class="cache-miss">${count - hits} fresh</span> · ${count}</span></div>`;
  }).join('');
}
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(loadPreview, 120);
}
async function loadPreview() {
  try {
    renderPreviewLoading('Refreshing planned run preview...');
    const response = await fetch('/api/preview', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(requestBody()) });
    if (!response.ok) throw new Error(await response.text());
    renderPreview(await response.json());
  } catch (error) {
    byId('preview-solvers').innerHTML = `<div class="progress-sub">Preview unavailable: ${escapeHtml(error.message)}</div>`;
  }
}
function setWidth(id, value, total) {
  byId(id).style.width = `${total ? value / total * 100 : 0}%`;
}
function setSegment(id, value, total, label) {
  setWidth(id, value, total);
  byId(id).title = `${label}: ${value}/${total || 0} (${percentText(value, total)})`;
}
function setStat(id, value, title) {
  const element = byId(id);
  element.textContent = String(value);
  element.title = title;
  element.closest('.stat').title = title;
}
function percentText(value, total) {
  return total ? `${Math.round(value / total * 100)}%` : '0%';
}
function elapsedText() {
  if (!progress.startedAt) return '0s';
  const seconds = Math.max(0, (Date.now() - progress.startedAt) / 1000);
  return seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}
function escapeHtml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}
function handleProgressEvent(event) {
  if (event.event === 'queued') {
    progress.total = event.total || 0;
    progress.stage = 'queued';
    progress.stageMessage = 'Queued selected solver-runs.';
    log(`queued ${progress.total} runs`);
  } else if (event.event === 'stage') {
    progress.stage = event.stage || progress.stage;
    progress.stageMessage = event.message || progress.stageMessage;
    progress.jobs = Number(event.desired_jobs || progress.jobs || 1);
    if (event.total != null) progress.total = event.total;
    if (event.completed != null) progress.completed = event.completed;
    log(`stage ${progress.stage}: ${progress.stageMessage}`);
  } else if (event.event === 'jobs_changed') {
    progress.jobs = Number(event.jobs || progress.jobs || 1);
    progress.stageMessage = `Dynamic job limit set to ${progress.jobs}.`;
    log(`jobs changed: ${progress.jobs}`);
  } else if (event.event === 'stop_requested') {
    progress.stage = 'stopping';
    progress.stageMessage = 'Stop requested. Waiting for active solver-runs to finish.';
    progress.stopped = true;
    log('stop requested');
  } else if (event.event === 'running') {
    progress.stage = 'solving';
    progress.stageMessage = 'Solving selected problem matrix.';
    progress.active.set(`${event.problem}:${event.solver}`, `${event.problem} · ${event.solver}`);
    transientCellStates.set(cellKey(event.problem, event.solver), { kind:'running' });
    log(`running ${event.problem} ${event.solver}`);
  } else if (event.event === 'cache_hit') {
    progress.active.set(`${event.problem}:${event.solver}`, `${event.problem} · ${event.solver} · cache`);
    transientCellStates.set(cellKey(event.problem, event.solver), { kind:'running', cacheStatus:'hit' });
    log(`cache hit ${event.problem} ${event.solver}`);
  } else if (event.event === 'completed') {
    progress.completed = event.completed || progress.completed + 1;
    progress.active.delete(`${event.problem}:${event.solver}`);
    transientCellStates.set(cellKey(event.problem, event.solver), { kind:'solved', status:event.status, cacheStatus:event.cache_status });
    if (event.status === 'passed') progress.pass += 1;
    else if (event.status === 'reduced_accuracy') progress.reduced += 1;
    else progress.fail += 1;
    if (event.cache_status === 'hit') progress.hit += 1;
    else progress.miss += 1;
    log(`${event.completed}/${event.total} ${event.problem} ${event.solver} ${event.status} ${event.cache_status}`);
  } else if (event.event === 'failed') {
    progress.completed = event.completed || progress.completed + 1;
    progress.active.delete(`${event.problem}:${event.solver}`);
    transientCellStates.set(cellKey(event.problem, event.solver), { kind:'solved', status:'solve_error', cacheStatus:'none' });
    progress.fail += 1;
    progress.miss += 1;
    log(`${event.completed}/${event.total} ${event.problem} ${event.solver} failed: ${event.reason}`);
  } else if (event.event === 'finished') {
    progress.completed = event.total;
    progress.active.clear();
    progress.stage = 'collecting_results';
    progress.stageMessage = 'Solver-runs complete; collecting records.';
    log(`finished: ${event.accepted}/${event.total} accepted`);
  } else if (event.event === 'stopped') {
    progress.completed = event.completed;
    progress.active.clear();
    progress.stage = 'stopped';
    progress.stageMessage = 'Run stopped after active solver-runs finished.';
    progress.stopped = true;
    log(`stopped: ${event.completed}/${event.total} completed`);
  }
  renderProgress();
  renderProblemMatrix();
}
async function run() {
  resetProgress();
  const runButton = byId('run');
  const stopButton = byId('stop-run');
  const runLabel = runButton.textContent;
  runButton.disabled = true;
  runButton.textContent = 'Starting...';
  stopButton.disabled = true;
  const body = requestBody();
  try {
    const created = await fetch('/api/runs', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) }).then((r) => r.json());
    currentRunId = created.run_id;
    runActive = true;
    runButton.textContent = 'Running';
    stopButton.disabled = false;
    stopButton.textContent = 'Stop';
    const response = await fetch(`/api/runs/${created.run_id}/events`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream:true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        handleProgressEvent(JSON.parse(line));
      }
    }
    setClientStage('collecting_results', 'Loading run records from the server.');
    renderLiveDashboardLoading('Building dashboard from run results...');
    const results = await fetch(`/api/runs/${created.run_id}/results`).then((r) => r.json());
    resultRecords = results.records || [];
    setClientStage('refreshing_cache', 'Rendering solver dashboard.');
    renderLiveDashboard();
    setClientStage('refreshing_cache', 'Refreshing cached result index.');
    await loadCachedRecords();
    setClientStage('refreshing_cache', 'Refreshing run preview.');
    await loadPreview();
    setClientStage(progress.stopped ? 'stopped' : 'complete', progress.stopped ? 'Stopped run results loaded.' : 'Run complete.');
  } finally {
    runActive = false;
    currentRunId = null;
    runButton.disabled = false;
    runButton.textContent = runLabel;
    stopButton.disabled = true;
    stopButton.textContent = 'Stop';
  }
}
function setClientStage(stage, message) {
  progress.stage = stage;
  progress.stageMessage = message;
  renderProgress();
}
async function stopRun() {
  if (!currentRunId || !runActive) return;
  const button = byId('stop-run');
  button.disabled = true;
  button.textContent = 'Stopping...';
  setClientStage('stopping', 'Stop requested. Waiting for active solver-runs to finish.');
  try {
    const response = await fetch(`/api/runs/${currentRunId}/stop`, { method:'POST' });
    if (!response.ok) throw new Error(await response.text());
  } catch (error) {
    log(`stop error: ${error.message}`);
    button.disabled = false;
    button.textContent = 'Stop';
  }
}
function scheduleDynamicJobUpdate() {
  progress.jobs = Math.max(1, Number(byId('jobs').value || 1));
  renderProgress();
  clearTimeout(jobsUpdateTimer);
  jobsUpdateTimer = setTimeout(updateRunJobs, 120);
}
async function updateRunJobs() {
  if (!currentRunId || !runActive) return;
  const jobs = Math.max(1, Number(byId('jobs').value || 1));
  try {
    const response = await fetch(`/api/runs/${currentRunId}/jobs`, {
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({ jobs }),
    });
    if (!response.ok) throw new Error(await response.text());
    progress.jobs = jobs;
    progress.stageMessage = `Dynamic job limit set to ${jobs}.`;
    renderProgress();
  } catch (error) {
    log(`jobs update error: ${error.message}`);
  }
}
async function clearCache() {
  const button = byId('clear-results-cache');
  const original = button.textContent;
  button.disabled = true;
  button.textContent = 'Clearing...';
  let cleared = false;
  try {
    const response = await fetch('/api/results_cache/clear', { method:'POST' });
    if (!response.ok) throw new Error(await response.text());
    cleared = true;
    resetToUnsolvedState('Results cache cleared.');
  } catch (error) {
    log(`clear cache error: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
  if (cleared) loadPreview().catch((error) => log(`preview refresh error: ${error.message}`));
}
function resetToUnsolvedState(message) {
  const plannedTotal = previewEntries.length || progress.total;
  progress = newProgressState();
  progress.total = plannedTotal;
  progress.miss = plannedTotal;
  progress.stage = 'preview';
  progress.stageMessage = 'Results cache cleared.';
  resultRecords = [];
  cachedRecords = [];
  transientCellStates = new Map();
  previewEntries = previewEntries.map((entry) => ({ ...entry, cache_status: 'miss' }));
  renderProgress();
  if (previewEntries.length) renderPreview(previewEntries);
  else renderProblemMatrix();
  renderLiveDashboard();
  byId('events').textContent = message;
  log('preview reflects unsolved fresh-result state');
}
function requestBody() {
  return {
    test_set: checkedRadio('test-set') || null,
    problem_set: checkedRadio('problem-set') || null,
    solver: checkedSolvers(),
    jobs: Number(byId('jobs').value || 4),
    force: false,
  };
}
function checkedRadio(name) {
  const input = document.querySelector(`input[name="${name}"]:checked`);
  return input ? input.value : '';
}
function checkedSolvers() {
  const solvers = checkedSolverValues();
  if (solvers.length === 0) return 'none';
  if (solvers.includes('sqp') && solvers.includes('nlip') && solvers.includes('ipopt')) return 'all';
  if (solvers.length === 2 && solvers.includes('sqp') && solvers.includes('nlip')) return 'both';
  return solvers.join(',');
}
function checkedSolverValues() {
  return Array.from(document.querySelectorAll('#solver-checks input:checked')).map((input) => input.value);
}
function renderLiveDashboard() {
  const records = resultRecords;
  if (!records.length) {
    byId('live-dashboard').innerHTML = `
      <section class="card dashboard-card">
        <div class="matrix-head">
          <div>
            <h2>Solver Dashboard</h2>
            <div class="sub">Run the selected matrix to populate live solver results. The problem matrix below stays as the detailed drill-down.</div>
          </div>
        </div>
      </section>`;
    return;
  }
  const accepted = records.filter((record) => record.status === 'passed' || record.status === 'reduced_accuracy').length;
  const reduced = records.filter((record) => record.status === 'reduced_accuracy').length;
  const failed = records.length - accepted;
  const hit = records.filter((record) => record.cache && record.cache.status === 'hit').length;
  const totalTime = records.reduce((sum, record) => sum + record.timing.total_wall_time, 0);
  const slowest = [...records].sort((a, b) => b.timing.total_wall_time - a.timing.total_wall_time).slice(0, 8);
  byId('live-dashboard').innerHTML = `
    <section class="card dashboard-card">
      <div class="matrix-head">
        <div>
          <h2>Solver Dashboard</h2>
          <div class="sub">${accepted}/${records.length} accepted · ${failed} failed · ${formatDuration(totalTime)} solver time</div>
        </div>
        <div class="progress-sub">${hit} cache hits · ${records.length - hit} fresh results</div>
      </div>
      <div class="dashboard-summary">
        <div class="dashboard-grid">
          ${dashMetric('Runs', records.length)}
          ${dashMetric('Accepted', `${accepted}/${records.length}`)}
          ${dashMetric('Reduced', reduced)}
          ${dashMetric('Failed', failed)}
          ${dashMetric('Cache hits', hit)}
          ${dashMetric('Solver time', formatDuration(totalTime))}
        </div>
        <div class="dashboard-breakdown-grid">
          ${dashboardBreakdown('By solver', groupCounts(records, (r) => r.solver), records.length)}
          ${dashboardBreakdown('By test set', groupCounts(records, (r) => r.descriptor.test_set), records.length)}
          ${dashboardBreakdown('By status', groupCounts(records, (r) => r.status), records.length)}
        </div>
      </div>
      <div class="dashboard-table-head">
        <h3>Failures by Solver</h3>
        <div class="progress-sub">Donuts split failed solver-runs by failure type</div>
      </div>
      ${failureDonutGrid(records)}
      <div class="dashboard-table-head">
        <h3>Slowest Runs</h3>
        <div class="progress-sub">Top ${slowest.length} by preserved solver time</div>
      </div>
      <table class="dash-table"><thead><tr><th>Problem</th><th>Set / Family</th><th>Solver</th><th>Status</th><th>Cache</th><th>Iters</th><th>Total</th><th>Reason</th></tr></thead><tbody>
        ${slowest.map((record) => `<tr><td>${escapeHtml(record.id)}</td><td>${escapeHtml(record.descriptor.test_set)} / ${escapeHtml(record.descriptor.family)}</td><td>${escapeHtml(record.solver)}</td><td><span class="status-pill ${record.status}">${escapeHtml(record.status)}</span></td><td>${escapeHtml(record.cache ? record.cache.status : 'none')}</td><td>${record.metrics.iterations == null ? '--' : record.metrics.iterations}</td><td>${formatDuration(record.timing.total_wall_time)}</td><td>${escapeHtml(record.error || record.validation.detail || '--')}</td></tr>`).join('')}
      </tbody></table>
    </section>`;
}
function renderLiveDashboardLoading(message) {
  byId('live-dashboard').innerHTML = `
    <section class="card dashboard-card">
      <h2>Solver Dashboard</h2>
      ${loadingLine(message)}
      <div class="dashboard-grid" style="margin-top:12px">
        ${dashSkeletonMetric()}
        ${dashSkeletonMetric()}
        ${dashSkeletonMetric()}
        ${dashSkeletonMetric()}
      </div>
    </section>`;
}
function dashMetric(label, value) {
  return `<div class="stat"><div class="k">${escapeHtml(label)}</div><div class="v">${escapeHtml(value)}</div></div>`;
}
function dashSkeletonMetric() {
  return `<div class="stat"><div class="skeleton medium"></div><div class="v"><span class="skeleton short" style="display:block;height:20px;margin-top:8px"></span></div></div>`;
}
function dashboardBreakdown(title, rows, total) {
  return `<div class="stat"><div class="k">${escapeHtml(title)}</div><div class="mini-bars">${miniBars(rows, total)}</div></div>`;
}
const FAILURE_COLORS = ['#f87171', '#fb7185', '#f97316', '#fbbf24', '#c084fc', '#818cf8', '#94a3b8'];
function failureDonutGrid(records) {
  const solvers = groupCounts(records, (record) => record.solver).map(([solver]) => solver);
  return `<div class="failure-grid">${solvers.map((solver) => failureDonutCard(solver, records.filter((record) => record.solver === solver))).join('')}</div>`;
}
function failureDonutCard(solver, records) {
  const failures = records.filter((record) => !isAcceptedStatus(record.status) && record.status !== 'skipped');
  const rows = groupCounts(failures, failureCategory);
  const totalFailures = failures.length;
  const title = totalFailures
    ? `${solver}: ${totalFailures} failed of ${records.length} solver-runs. ${rows.map(([label, count]) => `${label}: ${count}`).join('; ')}.`
    : `${solver}: no failed solver-runs out of ${records.length}.`;
  const style = totalFailures ? failureDonutStyle(rows, totalFailures) : 'background:conic-gradient(var(--pass) 0deg 360deg)';
  return `<div class="failure-card${totalFailures ? '' : ' clean'}" title="${escapeHtml(title)}">
    <div class="failure-donut${totalFailures ? '' : ' clean'}" style="${style}"><div class="failure-donut-inner">${totalFailures}/${records.length}</div></div>
    <div class="failure-copy">
      <div class="failure-title"><span>${escapeHtml(solver.toUpperCase())}</span><span>${totalFailures ? `${totalFailures} failed` : 'clean'}</span></div>
      <div class="failure-list">${totalFailures ? rows.map(([label, count], index) => failureLegendRow(label, count, index)).join('') : '<div class="progress-sub">No failed solver-runs.</div>'}</div>
    </div>
  </div>`;
}
function failureDonutStyle(rows, total) {
  let start = 0;
  const segments = rows.map(([label, count], index) => {
    const end = start + (total ? count / total * 360 : 0);
    const segment = `${failureColor(index)} ${start.toFixed(2)}deg ${end.toFixed(2)}deg`;
    start = end;
    return segment;
  });
  return `background:conic-gradient(${segments.join(',')})`;
}
function failureLegendRow(label, count, index) {
  return `<div class="failure-row" title="${escapeHtml(label)}: ${count} failed solver-runs"><span class="failure-swatch" style="background:${failureColor(index)}"></span><span class="failure-label">${escapeHtml(label)}</span><span>${count}</span></div>`;
}
function failureColor(index) {
  return FAILURE_COLORS[index % FAILURE_COLORS.length];
}
function failureCategory(record) {
  if (record.status === 'failed_validation') return 'failed validation';
  const text = String(record.error || (record.validation && record.validation.detail) || record.status || '').toLowerCase();
  if (text.includes('failed to converge') || text.includes('max_iters') || text.includes('max iterations')) return 'max iterations';
  if (text.includes('restoration')) return 'restoration';
  if (text.includes('line search')) return 'line search';
  if (text.includes('infeasible')) return 'infeasible';
  if (text.includes('panic')) return 'panic';
  if (text.includes('step inf-norm')) return 'step failure';
  return record.status === 'solve_error' ? 'solve error' : String(record.status || 'other').replaceAll('_', ' ');
}
function isAcceptedStatus(status) {
  return status === 'passed' || status === 'reduced_accuracy';
}
function formatDuration(seconds) {
  if (seconds >= 1) return `${seconds.toFixed(2)}s`;
  if (seconds >= 1e-3) return `${(seconds * 1e3).toFixed(2)}ms`;
  return `${(seconds * 1e6).toFixed(2)}us`;
}
byId('run').addEventListener('click', () => run().catch((error) => log(`error: ${error.message}`)));
byId('stop-run').addEventListener('click', () => stopRun());
byId('clear-results-cache').addEventListener('click', () => clearCache());
document.addEventListener('change', (event) => {
  if (event.target.closest('.segmented') || event.target.closest('.checks')) schedulePreview();
});
byId('jobs').addEventListener('input', () => {
  if (runActive) scheduleDynamicJobUpdate();
  else schedulePreview();
});
renderPreview([]);
renderLiveDashboard();
loadCatalog().catch((error) => log(`catalog error: ${error.message}`));
</script>
</body>
</html>
"#;
