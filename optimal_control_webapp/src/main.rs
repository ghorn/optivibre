use std::convert::Infallible;
use std::env;
use std::io;
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::sync::{Mutex, OnceLock};
#[cfg(unix)]
use std::thread;

use anyhow::Result;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::Path,
    http::{HeaderValue, StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use optimal_control_problems::{
    ProblemId, SolveLogLevel, SolveRequest, SolveStreamEvent, problem_specs, solve_problem,
    solve_problem_with_progress,
};
use optimization::{AnsiColorMode, set_ansi_color_mode};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

static SOLVE_STDIO_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

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
        .route("/api/solve/{id}", post(solve))
        .route("/api/solve_stream/{id}", post(solve_stream));

    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn app_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/javascript; charset=utf-8"),
        )],
        include_str!("../static/app.js"),
    )
}

async fn styles_css() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/css; charset=utf-8"),
        )],
        include_str!("../static/styles.css"),
    )
}

async fn problems() -> Json<Vec<optimal_control_problems::ProblemSpec>> {
    Json(problem_specs())
}

async fn solve(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> Result<Json<optimal_control_problems::SolveArtifact>, (StatusCode, Json<ErrorResponse>)> {
    solve_problem(problem, &request.values)
        .map(Json)
        .map_err(internal_error)
}

async fn solve_stream(
    Path(problem): Path<ProblemId>,
    Json(request): Json<SolveRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(64);

    tokio::task::spawn_blocking(move || {
        let _capture_lock = solve_stdio_lock()
            .lock()
            .expect("stdio capture lock poisoned");
        let _ansi_color_guard = AnsiColorModeGuard(set_ansi_color_mode(AnsiColorMode::Always));

        let sender_for_progress = sender.clone();
        #[cfg(unix)]
        let mut capture_state = match SolveStdIoCapture::start() {
            Ok((capture_guard, stdout_reader, stderr_reader)) => Some((
                capture_guard,
                spawn_stdio_reader(stdout_reader, sender.clone()),
                spawn_stdio_reader(stderr_reader, sender.clone()),
            )),
            Err(error) => {
                send_stream_event(
                    &sender,
                    SolveStreamEvent::Log {
                        line: format!("[stdout/stderr capture unavailable: {error}]"),
                        level: SolveLogLevel::Warning,
                    },
                );
                None
            }
        };

        let result = solve_problem_with_progress(problem, &request.values, |event| {
            send_stream_event(&sender_for_progress, event);
        });

        #[cfg(unix)]
        if let Some((capture_guard, stdout_handle, stderr_handle)) = capture_state.take() {
            drop(capture_guard);
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
        }

        if let Err(error) = result {
            send_stream_event(
                &sender,
                SolveStreamEvent::Error {
                    message: error.to_string(),
                },
            );
        }
    });

    Ok((
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
        )],
        Body::from_stream(ReceiverStream::new(receiver)),
    ))
}

fn internal_error(error: impl ToString) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn send_stream_event(sender: &mpsc::Sender<Result<Bytes, Infallible>>, event: SolveStreamEvent) {
    if let Ok(mut payload) = serde_json::to_vec(&event) {
        payload.push(b'\n');
        let _ = sender.blocking_send(Ok(Bytes::from(payload)));
    }
}

fn solve_stdio_lock() -> &'static Mutex<()> {
    SOLVE_STDIO_LOCK.get_or_init(|| Mutex::new(()))
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
                let _ = libc::close(stdout_pipe[0]);
                let _ = libc::close(stdout_pipe[1]);
                let _ = libc::close(stderr_pipe[0]);
                let _ = libc::close(stderr_pipe[1]);
                if saved_stdout >= 0 {
                    let _ = libc::close(saved_stdout);
                }
                if saved_stderr >= 0 {
                    let _ = libc::close(saved_stderr);
                }
                return Err(err);
            }

            if libc::dup2(stdout_pipe[1], libc::STDOUT_FILENO) < 0
                || libc::dup2(stderr_pipe[1], libc::STDERR_FILENO) < 0
            {
                let err = io::Error::last_os_error();
                let _ = libc::dup2(saved_stdout, libc::STDOUT_FILENO);
                let _ = libc::dup2(saved_stderr, libc::STDERR_FILENO);
                let _ = libc::close(saved_stdout);
                let _ = libc::close(saved_stderr);
                let _ = libc::close(stdout_pipe[0]);
                let _ = libc::close(stdout_pipe[1]);
                let _ = libc::close(stderr_pipe[0]);
                let _ = libc::close(stderr_pipe[1]);
                return Err(err);
            }

            let _ = libc::close(stdout_pipe[1]);
            let _ = libc::close(stderr_pipe[1]);

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
                        send_stream_event(
                            &sender,
                            SolveStreamEvent::Log {
                                line: trimmed.to_string(),
                                level: SolveLogLevel::Console,
                            },
                        );
                    }
                }
                Err(_) => break,
            }
        }
    })
}
