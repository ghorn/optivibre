use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::Infallible;
use std::io;
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;

use anyhow::Result;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::Path,
    http::{HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use optimal_control_problems::{
    CompileCacheState, CompileCacheStatus, ControlSemantic, ProblemId, SolveArtifact,
    SolveLogLevel, SolveRequest, SolveStage, SolveStatus, SolveStreamEvent, SolverMethod,
    SolverReport, compile_variant_for_problem, prewarm_problem_with_progress, problem_specs,
    solve_problem, solve_problem_with_progress,
};
use optimization::{AnsiColorMode, InteriorPointLinearSolver, set_ansi_color_mode};
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

static SOLVE_STDIO_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static PROBLEM_BACKEND: OnceLock<ProblemBackend> = OnceLock::new();
const TEXT_HTML_UTF8: &str = "text/html; charset=utf-8";
const TEXT_JAVASCRIPT_UTF8: &str = "text/javascript; charset=utf-8";
const TEXT_CSS_UTF8: &str = "text/css; charset=utf-8";
const APPLICATION_NDJSON_UTF8: &str = "application/x-ndjson; charset=utf-8";
const GENERATED_APP_JS: &str = include_str!(concat!(env!("OUT_DIR"), "/app.js"));

type ApiError = (StatusCode, Json<ErrorResponse>);
type ApiResult<T> = Result<T, ApiError>;
type StreamSender = mpsc::Sender<Result<Bytes, Infallible>>;

struct ProblemBackend {
    actors: Mutex<HashMap<CompileSignature, Arc<ProblemActor>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CompileSignature {
    problem_id: ProblemId,
    variant_id: String,
}

#[derive(Clone, Debug)]
struct CompileDescriptor {
    signature: CompileSignature,
    problem_name: String,
    variant_label: String,
    compile_values: BTreeMap<String, f64>,
}

type ActorShared = (Mutex<ProblemActorState>, Condvar);

struct ProblemActor {
    descriptor: CompileDescriptor,
    shared: Arc<ActorShared>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerPhase {
    Idle,
    Warming,
    Solving,
}

struct ProblemActorState {
    phase: WorkerPhase,
    ready: bool,
    latest_compile_status: Option<SolveStatus>,
    pending_prewarm_replies: Vec<oneshot::Sender<Result<(), String>>>,
    pending_solves: VecDeque<PendingSolveJob>,
}

enum PendingSolveJob {
    Sync {
        values: BTreeMap<String, f64>,
        reply: oneshot::Sender<Result<SolveArtifact, String>>,
    },
    Stream {
        values: BTreeMap<String, f64>,
        sender: StreamSender,
    },
}

#[derive(Clone, Copy, Debug, Default)]
struct SolveRequestContext {
    solver_method: Option<SolverMethod>,
    nlip_linear_solver: Option<InteriorPointLinearSolver>,
}

impl SolveRequestContext {
    fn from_values(problem: ProblemId, values: &BTreeMap<String, f64>) -> Self {
        Self {
            solver_method: solver_method_for_values(problem, values),
            nlip_linear_solver: nlip_linear_solver_for_values(problem, values),
        }
    }

    fn initial_notice(self) -> Option<(&'static str, SolveLogLevel)> {
        if matches!(self.solver_method, Some(SolverMethod::Nlip))
            && matches!(
                self.nlip_linear_solver,
                Some(InteriorPointLinearSolver::SsidsRs)
            )
        {
            Some((
                "[NLIP] Preparing SSIDS-RS sparse KKT analysis; the first warmed solve may pause before iteration 0.",
                SolveLogLevel::Info,
            ))
        } else {
            None
        }
    }
}

struct AnsiColorModeGuard(AnsiColorMode);

impl Drop for AnsiColorModeGuard {
    fn drop(&mut self) {
        set_ansi_color_mode(self.0);
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Clone, Debug, Default, Serialize)]
struct CompileCacheSnapshot {
    entries: Vec<CompileCacheStatus>,
}

#[derive(Debug, Parser)]
#[command(
    name = "optimization_studio",
    about = "Local interactive web app for optimization and optimal-control demos."
)]
struct WebappCli {
    #[arg(long, env = "PORT", default_value_t = 3000)]
    port: u16,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = WebappCli::parse();
    let port = cli.port;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("optimization_studio listening on http://127.0.0.1:{port}");

    let app = Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles_css))
        .route("/api/problems", get(problems))
        .route("/api/prewarm_status", get(prewarm_status))
        .route("/api/prewarm/{id}", post(prewarm))
        .route("/api/clear_jit_cache", post(clear_jit_cache))
        .route("/api/solve/{id}", post(solve))
        .route("/api/solve_stream/{id}", post(solve_stream));

    axum::serve(listener, app).await?;
    Ok(())
}

fn static_text_response(content_type: &'static str, body: &'static str) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, HeaderValue::from_static(content_type))],
        body,
    )
}

async fn index() -> impl IntoResponse {
    static_text_response(TEXT_HTML_UTF8, include_str!("../static/index.html"))
}

async fn app_js() -> impl IntoResponse {
    static_text_response(TEXT_JAVASCRIPT_UTF8, GENERATED_APP_JS)
}

async fn styles_css() -> impl IntoResponse {
    static_text_response(TEXT_CSS_UTF8, include_str!("../static/styles.css"))
}

async fn problems() -> Json<Vec<optimal_control_problems::ProblemSpec>> {
    Json(problem_specs())
}

async fn prewarm_status() -> ApiResult<Json<CompileCacheSnapshot>> {
    Ok(Json(problem_backend().compile_snapshot()))
}

async fn clear_jit_cache() -> ApiResult<StatusCode> {
    optimization::clear_optivibre_jit_cache()
        .map_err(|error| internal_error(format!("failed to clear LLVM JIT cache: {error}")))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn prewarm(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<StatusCode> {
    let reply_rx = problem_backend().prewarm(problem, request);
    match reply_rx.await {
        Ok(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Ok(Err(error)) => Err(internal_error(error)),
        Err(error) => Err(internal_error(format!(
            "prewarm task dropped response: {error}"
        ))),
    }
}

async fn solve(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<Json<optimal_control_problems::SolveArtifact>> {
    let reply_rx = problem_backend().solve(problem, request);
    match reply_rx.await {
        Ok(Ok(artifact)) => Ok(Json(artifact)),
        Ok(Err(error)) => Err(internal_error(error)),
        Err(error) => Err(internal_error(format!(
            "solve task dropped response: {error}"
        ))),
    }
}

async fn solve_stream(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<impl IntoResponse> {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(64);
    let context = SolveRequestContext::from_values(problem, &request.values);
    let initial_status = problem_backend().solve_stream(problem, request, sender.clone(), context);
    if let Some(status) = initial_status {
        send_stream_event_async(&sender, SolveStreamEvent::Status { status }).await;
    }
    if let Some((line, level)) = context.initial_notice() {
        send_stream_event_async(
            &sender,
            SolveStreamEvent::Log {
                line: line.to_string(),
                level,
            },
        )
        .await;
    }
    Ok(ndjson_stream_response(receiver))
}

fn problem_backend() -> &'static ProblemBackend {
    PROBLEM_BACKEND.get_or_init(ProblemBackend::start)
}

impl ProblemBackend {
    fn start() -> Self {
        Self {
            actors: Mutex::new(HashMap::new()),
        }
    }

    fn compile_snapshot(&self) -> CompileCacheSnapshot {
        let actors = self.actors.lock().expect("problem actor registry poisoned");
        let mut entries = actors
            .values()
            .filter_map(|actor| actor.compile_status())
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            left.problem_name
                .cmp(&right.problem_name)
                .then(left.variant_label.cmp(&right.variant_label))
        });
        CompileCacheSnapshot { entries }
    }

    fn prewarm(
        &self,
        problem: ProblemId,
        request: SolveRequest,
    ) -> oneshot::Receiver<Result<(), String>> {
        self.actor_for(problem, &request.values).enqueue_prewarm()
    }

    fn solve(
        &self,
        problem: ProblemId,
        request: SolveRequest,
    ) -> oneshot::Receiver<Result<SolveArtifact, String>> {
        self.actor_for(problem, &request.values)
            .enqueue_sync_solve(request.values)
    }

    fn solve_stream(
        &self,
        problem: ProblemId,
        request: SolveRequest,
        sender: StreamSender,
        context: SolveRequestContext,
    ) -> Option<SolveStatus> {
        self.actor_for(problem, &request.values)
            .enqueue_stream_solve(request.values, sender, context)
    }

    fn actor_for(&self, problem: ProblemId, values: &BTreeMap<String, f64>) -> Arc<ProblemActor> {
        let descriptor = compile_descriptor(problem, values);
        let mut actors = self.actors.lock().expect("problem actor registry poisoned");
        if let Some(actor) = actors.get(&descriptor.signature) {
            return actor.clone();
        }
        let actor = ProblemActor::spawn(problem, descriptor.clone());
        actors.insert(descriptor.signature, actor.clone());
        actor
    }
}

impl ProblemActor {
    fn spawn(problem: ProblemId, descriptor: CompileDescriptor) -> Arc<Self> {
        let shared = Arc::new((
            Mutex::new(ProblemActorState {
                phase: WorkerPhase::Idle,
                ready: false,
                latest_compile_status: None,
                pending_prewarm_replies: Vec::new(),
                pending_solves: VecDeque::new(),
            }),
            Condvar::new(),
        ));
        let worker_shared = shared.clone();
        let worker_descriptor = descriptor.clone();
        thread::spawn(move || actor_worker_loop(problem, worker_descriptor, worker_shared));
        Arc::new(Self { descriptor, shared })
    }

    fn compile_status(&self) -> Option<CompileCacheStatus> {
        let (lock, _) = &*self.shared;
        let state = lock.lock().expect("problem actor state poisoned");
        compile_status_from_state(&self.descriptor, &state)
    }

    fn enqueue_prewarm(&self) -> oneshot::Receiver<Result<(), String>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().expect("problem actor state poisoned");
        if state.ready {
            let _ = reply_tx.send(Ok(()));
            return reply_rx;
        }
        state.pending_prewarm_replies.push(reply_tx);
        cvar.notify_one();
        reply_rx
    }

    fn enqueue_sync_solve(
        &self,
        values: BTreeMap<String, f64>,
    ) -> oneshot::Receiver<Result<SolveArtifact, String>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().expect("problem actor state poisoned");
        state.pending_solves.push_back(PendingSolveJob::Sync {
            values,
            reply: reply_tx,
        });
        cvar.notify_one();
        reply_rx
    }

    fn enqueue_stream_solve(
        &self,
        values: BTreeMap<String, f64>,
        sender: StreamSender,
        context: SolveRequestContext,
    ) -> Option<SolveStatus> {
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().expect("problem actor state poisoned");
            let initial_status = if state.phase == WorkerPhase::Warming {
                state.latest_compile_status.clone()
            } else if state.ready {
                Some(ready_solve_status(
                    state.latest_compile_status.clone(),
                    context.solver_method,
                ))
            } else {
                None
            };
            state.pending_solves.push_back(PendingSolveJob::Stream {
                values,
                sender: sender.clone(),
            });
            cvar.notify_one();
            initial_status
        }
    }
}

fn actor_worker_loop(problem: ProblemId, descriptor: CompileDescriptor, shared: Arc<ActorShared>) {
    loop {
        let job = {
            let (lock, cvar) = &*shared;
            let mut state = lock.lock().expect("problem actor state poisoned");
            loop {
                if state.phase == WorkerPhase::Idle {
                    if !state.ready {
                        if !state.pending_prewarm_replies.is_empty()
                            || !state.pending_solves.is_empty()
                        {
                            state.phase = WorkerPhase::Warming;
                            state.latest_compile_status = None;
                            break ActorJob::Compile;
                        }
                    } else if let Some(job) = state.pending_solves.pop_front() {
                        state.phase = WorkerPhase::Solving;
                        break ActorJob::Solve(job);
                    }
                }
                state = cvar.wait(state).expect("problem actor state poisoned");
            }
        };

        match job {
            ActorJob::Compile => run_actor_compile(problem, &descriptor, &shared),
            ActorJob::Solve(job) => run_actor_solve(problem, &shared, job),
        }
    }
}

enum ActorJob {
    Compile,
    Solve(PendingSolveJob),
}

fn run_actor_compile(
    problem: ProblemId,
    descriptor: &CompileDescriptor,
    shared: &Arc<ActorShared>,
) {
    let shared_for_progress = Arc::clone(shared);
    let result = prewarm_problem_with_progress(problem, &descriptor.compile_values, move |event| {
        forward_compile_event(&shared_for_progress, &event);
    });

    let (prewarm_replies, pending_solves) = {
        let (lock, cvar) = &**shared;
        let mut state = lock.lock().expect("problem actor state poisoned");
        state.phase = WorkerPhase::Idle;
        let prewarm_replies = std::mem::take(&mut state.pending_prewarm_replies);
        let pending_solves = if result.is_err() {
            state.ready = false;
            state.latest_compile_status = None;
            state.pending_solves.drain(..).collect::<Vec<_>>()
        } else {
            state.ready = true;
            Vec::new()
        };
        cvar.notify_all();
        (prewarm_replies, pending_solves)
    };

    match result {
        Ok(()) => {
            for reply in prewarm_replies {
                let _ = reply.send(Ok(()));
            }
        }
        Err(error) => {
            let message = error.to_string();
            for reply in prewarm_replies {
                let _ = reply.send(Err(message.clone()));
            }
            for job in pending_solves {
                match job {
                    PendingSolveJob::Sync { reply, .. } => {
                        let _ = reply.send(Err(message.clone()));
                    }
                    PendingSolveJob::Stream { sender, .. } => {
                        send_stream_error(&sender, &message);
                    }
                }
            }
        }
    }
}

fn run_actor_solve(problem: ProblemId, shared: &Arc<ActorShared>, job: PendingSolveJob) {
    match job {
        PendingSolveJob::Sync { values, reply } => {
            let result = solve_problem(problem, &values).map_err(|error| error.to_string());
            let _ = reply.send(result);
        }
        PendingSolveJob::Stream { values, sender } => {
            run_solve_stream(problem, values, sender);
        }
    }

    let (lock, cvar) = &**shared;
    let mut state = lock.lock().expect("problem actor state poisoned");
    state.phase = WorkerPhase::Idle;
    cvar.notify_all();
}

fn forward_compile_event(shared: &Arc<ActorShared>, event: &SolveStreamEvent) {
    let status = match event {
        SolveStreamEvent::Status { status } if is_compile_stage(status.stage) => status.clone(),
        SolveStreamEvent::Log { .. }
        | SolveStreamEvent::Iteration { .. }
        | SolveStreamEvent::Final { .. }
        | SolveStreamEvent::Error { .. }
        | SolveStreamEvent::Status { .. } => return,
    };

    let senders = {
        let (lock, _) = &**shared;
        let mut state = lock.lock().expect("problem actor state poisoned");
        state.latest_compile_status = Some(status.clone());
        state
            .pending_solves
            .iter()
            .filter_map(|job| match job {
                PendingSolveJob::Stream { sender, .. } => Some(sender.clone()),
                PendingSolveJob::Sync { .. } => None,
            })
            .collect::<Vec<_>>()
    };

    for sender in senders {
        send_stream_event(
            &sender,
            SolveStreamEvent::Status {
                status: status.clone(),
            },
        );
    }
}

fn run_solve_stream(problem: ProblemId, values: BTreeMap<String, f64>, sender: StreamSender) {
    let _capture_lock = solve_stdio_lock()
        .lock()
        .expect("stdio capture lock poisoned");
    let _ansi_color_guard = AnsiColorModeGuard(set_ansi_color_mode(AnsiColorMode::Always));

    #[cfg(unix)]
    let capture_state = StreamStdIoCapture::start(&sender);

    let sender_for_progress = sender.clone();
    let result = solve_problem_with_progress(problem, &values, move |event| {
        if should_forward_solve_event(&event) {
            send_stream_event(&sender_for_progress, event);
        }
    });

    #[cfg(unix)]
    if let Some(capture_state) = capture_state {
        capture_state.finish();
    }

    if let Err(error) = result {
        send_stream_error(&sender, error);
    }
}

fn should_forward_solve_event(event: &SolveStreamEvent) -> bool {
    !matches!(
        event,
        SolveStreamEvent::Status { status }
            if matches!(status.stage, SolveStage::SymbolicSetup | SolveStage::JitCompilation)
    )
}

fn compile_status_from_state(
    descriptor: &CompileDescriptor,
    state: &ProblemActorState,
) -> Option<CompileCacheStatus> {
    if state.phase == WorkerPhase::Warming {
        Some(compile_cache_status_from_parts(
            descriptor,
            CompileCacheState::Warming,
            state.latest_compile_status.as_ref(),
        ))
    } else if state.ready {
        Some(compile_cache_status_from_parts(
            descriptor,
            CompileCacheState::Ready,
            state.latest_compile_status.as_ref(),
        ))
    } else {
        None
    }
}

fn compile_cache_status_from_parts(
    descriptor: &CompileDescriptor,
    state: CompileCacheState,
    status: Option<&SolveStatus>,
) -> CompileCacheStatus {
    CompileCacheStatus {
        problem_id: descriptor.signature.problem_id,
        problem_name: descriptor.problem_name.clone(),
        variant_id: descriptor.signature.variant_id.clone(),
        variant_label: descriptor.variant_label.clone(),
        state,
        symbolic_setup_s: status.and_then(|status| status.solver.symbolic_setup_s),
        jit_s: status.and_then(|status| status.solver.jit_s),
        jit_disk_cache_hit: status
            .is_some_and(|status| status.solver.jit_disk_cache_hit || status.solver.compile_cached),
    }
}

fn ready_solve_status(
    latest_compile_status: Option<SolveStatus>,
    solver_method: Option<SolverMethod>,
) -> SolveStatus {
    let mut status = latest_compile_status.unwrap_or_else(|| SolveStatus {
        stage: SolveStage::Solving,
        solver_method,
        solver: SolverReport::in_progress(solver_running_label(solver_method)),
    });
    status.stage = SolveStage::Solving;
    status.solver_method = solver_method;
    status.solver.status_label = solver_running_label(solver_method).to_string();
    status.solver.solve_s = Some(0.0);
    status
}

fn solver_running_label(solver_method: Option<SolverMethod>) -> &'static str {
    match solver_method {
        Some(SolverMethod::Nlip) => "Running NLIP solver...",
        Some(SolverMethod::Sqp) => "Running SQP...",
        Some(_) => "Running solver...",
        None => "Starting solve...",
    }
}

fn control_value_for_semantic(
    problem: ProblemId,
    values: &BTreeMap<String, f64>,
    semantic: ControlSemantic,
) -> Option<f64> {
    let spec = problem_specs()
        .into_iter()
        .find(|spec| spec.id == problem)?;
    let control = spec
        .controls
        .iter()
        .find(|control| control.semantic == semantic)?;
    Some(values.get(&control.id).copied().unwrap_or(control.default))
}

fn solver_method_for_values(
    problem: ProblemId,
    values: &BTreeMap<String, f64>,
) -> Option<SolverMethod> {
    let value = control_value_for_semantic(problem, values, ControlSemantic::SolverMethod)?;
    match value.round() as i64 {
        0 => Some(SolverMethod::Sqp),
        1 => Some(SolverMethod::Nlip),
        2 => Some(SolverMethod::Ipopt),
        _ => None,
    }
}

fn nlip_linear_solver_for_values(
    problem: ProblemId,
    values: &BTreeMap<String, f64>,
) -> Option<InteriorPointLinearSolver> {
    let value =
        control_value_for_semantic(problem, values, ControlSemantic::SolverNlipLinearSolver)?;
    match value.round() as i64 {
        0 => Some(InteriorPointLinearSolver::SsidsRs),
        1 => Some(InteriorPointLinearSolver::SpralSrc),
        2 => Some(InteriorPointLinearSolver::SparseQdldl),
        3 => Some(InteriorPointLinearSolver::Auto),
        _ => None,
    }
}

fn is_compile_stage(stage: SolveStage) -> bool {
    matches!(
        stage,
        SolveStage::SymbolicSetup | SolveStage::JitCompilation
    )
}

fn compile_descriptor(problem: ProblemId, values: &BTreeMap<String, f64>) -> CompileDescriptor {
    let spec = problem_specs()
        .into_iter()
        .find(|spec| spec.id == problem)
        .expect("problem spec missing");
    let (variant_id, variant_label) =
        compile_variant_for_problem(problem, values).expect("compile variant missing");
    CompileDescriptor {
        signature: CompileSignature {
            problem_id: problem,
            variant_id,
        },
        problem_name: spec.name,
        variant_label,
        compile_values: values.clone(),
    }
}

fn ndjson_stream_response(
    receiver: mpsc::Receiver<Result<Bytes, Infallible>>,
) -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static(APPLICATION_NDJSON_UTF8),
        )],
        Body::from_stream(ReceiverStream::new(receiver)),
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

fn send_stream_log(sender: &StreamSender, level: SolveLogLevel, line: impl Into<String>) {
    send_stream_event(
        sender,
        SolveStreamEvent::Log {
            line: line.into(),
            level,
        },
    );
}

fn send_stream_error(sender: &StreamSender, error: impl ToString) {
    send_stream_event(
        sender,
        SolveStreamEvent::Error {
            message: error.to_string(),
        },
    );
}

fn send_stream_event(sender: &StreamSender, event: SolveStreamEvent) {
    if let Ok(mut payload) = serde_json::to_vec(&event) {
        payload.push(b'\n');
        let _ = sender.blocking_send(Ok(Bytes::from(payload)));
    }
}

async fn send_stream_event_async(sender: &StreamSender, event: SolveStreamEvent) {
    if let Ok(mut payload) = serde_json::to_vec(&event) {
        payload.push(b'\n');
        let _ = sender.send(Ok(Bytes::from(payload))).await;
    }
}

fn solve_stdio_lock() -> &'static Mutex<()> {
    SOLVE_STDIO_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(unix)]
struct StreamStdIoCapture {
    capture_guard: SolveStdIoCapture,
    stdout_handle: thread::JoinHandle<()>,
    stderr_handle: thread::JoinHandle<()>,
}

#[cfg(unix)]
impl StreamStdIoCapture {
    fn start(sender: &StreamSender) -> Option<Self> {
        match SolveStdIoCapture::start() {
            Ok((capture_guard, stdout_reader, stderr_reader)) => Some(Self {
                capture_guard,
                stdout_handle: spawn_stdio_reader(stdout_reader, sender.clone()),
                stderr_handle: spawn_stdio_reader(stderr_reader, sender.clone()),
            }),
            Err(error) => {
                send_stream_log(
                    sender,
                    SolveLogLevel::Info,
                    format!("[stdout/stderr capture unavailable: {error}]"),
                );
                None
            }
        }
    }

    fn finish(self) {
        drop(self.capture_guard);
        let _ = self.stdout_handle.join();
        let _ = self.stderr_handle.join();
    }
}

#[cfg(unix)]
struct SolveStdIoCapture {
    saved_stdout: RawFd,
    saved_stderr: RawFd,
}

#[cfg(unix)]
impl SolveStdIoCapture {
    fn start() -> io::Result<(Self, std::fs::File, std::fs::File)> {
        std::io::stdout().flush()?;
        std::io::stderr().flush()?;
        unsafe {
            libc::fflush(std::ptr::null_mut());
        }

        let mut stdout_pipe = [0; 2];
        let mut stderr_pipe = [0; 2];
        unsafe {
            if libc::pipe(stdout_pipe.as_mut_ptr()) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::pipe(stderr_pipe.as_mut_ptr()) != 0 {
                let _ = libc::close(stdout_pipe[0]);
                let _ = libc::close(stdout_pipe[1]);
                return Err(io::Error::last_os_error());
            }

            let saved_stdout = libc::dup(libc::STDOUT_FILENO);
            let saved_stderr = libc::dup(libc::STDERR_FILENO);
            if saved_stdout < 0 || saved_stderr < 0 {
                let err = io::Error::last_os_error();
                close_fds([
                    stdout_pipe[0],
                    stdout_pipe[1],
                    stderr_pipe[0],
                    stderr_pipe[1],
                    saved_stdout,
                    saved_stderr,
                ]);
                return Err(err);
            }

            if libc::dup2(stdout_pipe[1], libc::STDOUT_FILENO) < 0
                || libc::dup2(stderr_pipe[1], libc::STDERR_FILENO) < 0
            {
                let err = io::Error::last_os_error();
                let _ = libc::dup2(saved_stdout, libc::STDOUT_FILENO);
                let _ = libc::dup2(saved_stderr, libc::STDERR_FILENO);
                close_fds([
                    saved_stdout,
                    saved_stderr,
                    stdout_pipe[0],
                    stdout_pipe[1],
                    stderr_pipe[0],
                    stderr_pipe[1],
                ]);
                return Err(err);
            }

            close_fds([stdout_pipe[1], stderr_pipe[1]]);

            Ok((
                Self {
                    saved_stdout,
                    saved_stderr,
                },
                std::fs::File::from_raw_fd(stdout_pipe[0]),
                std::fs::File::from_raw_fd(stderr_pipe[0]),
            ))
        }
    }
}

#[cfg(unix)]
fn close_fd(fd: RawFd) {
    if fd >= 0 {
        unsafe {
            let _ = libc::close(fd);
        }
    }
}

#[cfg(unix)]
fn close_fds<const N: usize>(fds: [RawFd; N]) {
    for fd in fds {
        close_fd(fd);
    }
}

#[cfg(unix)]
impl Drop for SolveStdIoCapture {
    fn drop(&mut self) {
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe {
            libc::fflush(std::ptr::null_mut());
            let _ = libc::dup2(self.saved_stdout, libc::STDOUT_FILENO);
            let _ = libc::dup2(self.saved_stderr, libc::STDERR_FILENO);
            let _ = libc::close(self.saved_stdout);
            let _ = libc::close(self.saved_stderr);
        }
    }
}

#[cfg(unix)]
fn spawn_stdio_reader(
    reader: std::fs::File,
    sender: mpsc::Sender<Result<Bytes, Infallible>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim_end_matches(['\n', '\r']);
                    if should_forward_console_line(trimmed) {
                        send_stream_log(&sender, SolveLogLevel::Console, trimmed);
                    }
                }
                Err(_) => break,
            }
        }
    })
}

fn should_forward_console_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    !trimmed.starts_with("[NLIP][SPRAL]") && !trimmed.starts_with("[NLIP][Native-SPRAL]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    use tempfile::TempDir;

    fn cache_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("cache env lock")
    }

    #[test]
    fn solver_method_for_values_parses_ipopt_choice() {
        let mut values = BTreeMap::new();
        values.insert("solver_method".to_string(), 2.0);
        assert_eq!(
            solver_method_for_values(ProblemId::OptimalDistanceGlider, &values),
            Some(SolverMethod::Ipopt)
        );
    }

    #[test]
    fn console_stream_filters_internal_nlip_spral_debug_lines() {
        assert!(!should_forward_console_line(""));
        assert!(!should_forward_console_line(
            "[NLIP][Native-SPRAL] Starting numeric factorization: dim=3958 nnz=12856 reg=1.000e-6",
        ));
        assert!(!should_forward_console_line(
            "[NLIP][SPRAL] Numeric factorization completed in 10ms",
        ));
        assert!(should_forward_console_line(
            "  70   FsS        -3.28e+01    7.29e-02"
        ));
        assert!(should_forward_console_line("EXIT: Optimal Solution Found."));
    }

    #[test]
    fn clear_jit_cache_endpoint_removes_disk_cache_contents() {
        let _guard = cache_env_lock();
        let cache_root = TempDir::new().expect("temp cache root");
        unsafe { std::env::set_var("OPTIVIBRE_JIT_CACHE_DIR", cache_root.path()) };

        let nested = cache_root.path().join("v1/test/object.o");
        std::fs::create_dir_all(nested.parent().expect("nested parent"))
            .expect("create nested cache dir");
        std::fs::write(&nested, b"cached-object").expect("write cached object");
        assert!(nested.exists());

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        let status = runtime
            .block_on(clear_jit_cache())
            .expect("clear cache handler should succeed");
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(!cache_root.path().exists());

        unsafe { std::env::remove_var("OPTIVIBRE_JIT_CACHE_DIR") };
    }
}
