use std::convert::Infallible;
use std::env;
use std::io;
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::sync::mpsc as std_mpsc;
use std::sync::{Mutex, OnceLock};
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
use optimal_control_problems::{
    CompileCacheStatus, ProblemId, SolveArtifact, SolveLogLevel, SolveRequest, SolveStreamEvent,
    compile_cache_statuses, prewarm_problem, problem_specs, solve_problem,
    solve_problem_with_progress,
};
use optimization::{AnsiColorMode, set_ansi_color_mode};
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
    requests: std_mpsc::Sender<BackendRequest>,
}

enum BackendRequest {
    CompileCacheStatus {
        reply: oneshot::Sender<Result<Vec<CompileCacheStatus>, String>>,
    },
    Prewarm {
        problem: ProblemId,
        request: SolveRequest,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Solve {
        problem: ProblemId,
        request: SolveRequest,
        reply: oneshot::Sender<Result<SolveArtifact, String>>,
    },
    SolveStream {
        problem: ProblemId,
        request: SolveRequest,
        sender: StreamSender,
    },
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

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(3000);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("optimal_control_webapp listening on http://127.0.0.1:{port}");

    let app = Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles_css))
        .route("/api/problems", get(problems))
        .route("/api/prewarm_status", get(prewarm_status))
        .route("/api/prewarm/{id}", post(prewarm))
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

async fn prewarm_status() -> ApiResult<Json<Vec<CompileCacheStatus>>> {
    let (reply_tx, reply_rx) = oneshot::channel();
    problem_backend()
        .requests
        .send(BackendRequest::CompileCacheStatus { reply: reply_tx })
        .map_err(|error| internal_error(format!("backend worker unavailable: {error}")))?;
    match reply_rx.await {
        Ok(Ok(statuses)) => Ok(Json(statuses)),
        Ok(Err(error)) => Err(internal_error(error)),
        Err(error) => Err(internal_error(format!(
            "backend worker dropped compile cache status response: {error}"
        ))),
    }
}

async fn prewarm(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<StatusCode> {
    let (reply_tx, reply_rx) = oneshot::channel();
    problem_backend()
        .requests
        .send(BackendRequest::Prewarm {
            problem,
            request,
            reply: reply_tx,
        })
        .map_err(|error| internal_error(format!("backend worker unavailable: {error}")))?;
    match reply_rx.await {
        Ok(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Ok(Err(error)) => Err(internal_error(error)),
        Err(error) => Err(internal_error(format!(
            "backend worker dropped prewarm response: {error}"
        ))),
    }
}

async fn solve(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<Json<optimal_control_problems::SolveArtifact>> {
    let (reply_tx, reply_rx) = oneshot::channel();
    problem_backend()
        .requests
        .send(BackendRequest::Solve {
            problem,
            request,
            reply: reply_tx,
        })
        .map_err(|error| internal_error(format!("backend worker unavailable: {error}")))?;
    match reply_rx.await {
        Ok(Ok(artifact)) => Ok(Json(artifact)),
        Ok(Err(error)) => Err(internal_error(error)),
        Err(error) => Err(internal_error(format!(
            "backend worker dropped solve response: {error}"
        ))),
    }
}

async fn solve_stream(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<impl IntoResponse> {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(64);

    problem_backend()
        .requests
        .send(BackendRequest::SolveStream {
            problem,
            request,
            sender,
        })
        .map_err(|error| internal_error(format!("backend worker unavailable: {error}")))?;

    Ok(ndjson_stream_response(receiver))
}

fn problem_backend() -> &'static ProblemBackend {
    PROBLEM_BACKEND.get_or_init(ProblemBackend::start)
}

impl ProblemBackend {
    fn start() -> Self {
        let (requests, receiver) = std_mpsc::channel();
        thread::spawn(move || backend_worker_loop(receiver));
        Self { requests }
    }
}

fn backend_worker_loop(receiver: std_mpsc::Receiver<BackendRequest>) {
    while let Ok(request) = receiver.recv() {
        match request {
            BackendRequest::CompileCacheStatus { reply } => {
                let _ = reply.send(Ok(compile_cache_statuses()));
            }
            BackendRequest::Prewarm {
                problem,
                request,
                reply,
            } => {
                let result =
                    prewarm_problem(problem, &request.values).map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
            BackendRequest::Solve {
                problem,
                request,
                reply,
            } => {
                let result =
                    solve_problem(problem, &request.values).map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
            BackendRequest::SolveStream {
                problem,
                request,
                sender,
            } => run_solve_stream(problem, request, sender),
        }
    }
}

fn run_solve_stream(problem: ProblemId, request: SolveRequest, sender: StreamSender) {
    let _capture_lock = solve_stdio_lock()
        .lock()
        .expect("stdio capture lock poisoned");
    let _ansi_color_guard = AnsiColorModeGuard(set_ansi_color_mode(AnsiColorMode::Always));

    #[cfg(unix)]
    let capture_state = StreamStdIoCapture::start(&sender);

    let sender_for_progress = sender.clone();
    let result = solve_problem_with_progress(problem, &request.values, |event| {
        send_stream_event(&sender_for_progress, event);
    });

    #[cfg(unix)]
    if let Some(capture_state) = capture_state {
        capture_state.finish();
    }

    if let Err(error) = result {
        send_stream_error(&sender, error);
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
                    if !trimmed.is_empty() {
                        send_stream_log(&sender, SolveLogLevel::Console, trimmed);
                    }
                }
                Err(_) => break,
            }
        }
    })
}
