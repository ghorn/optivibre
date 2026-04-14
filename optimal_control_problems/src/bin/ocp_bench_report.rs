use std::collections::BTreeMap;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::panic;
use std::path::PathBuf;
use std::process;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use crossterm::{
    cursor::{Hide, Show},
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use libc::{SIGABRT, SIGBUS, SIGILL, SIGSEGV, SIGTRAP};
use optimal_control::{OcpCompileHelperKind, OcpCompileProgress};
use optimal_control_problems::{
    OcpBenchmarkCase, OcpBenchmarkPreset, OcpBenchmarkProgress, OcpBenchmarkRecord,
    OcpBenchmarkSuiteConfig, ProblemId, TranscriptionMethod, run_ocp_benchmark_suite_with_progress,
    write_ocp_benchmark_report,
};
use optimization::{NlpEvaluationKernelKind, SymbolicCompileStage, SymbolicCompileStageProgress};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, Paragraph, Row, Table, Wrap},
};
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};
use signal_hook::iterator::Signals;

fn main() -> Result<()> {
    let cli = OcpBenchCli::parse();
    let mut config = OcpBenchmarkSuiteConfig::default();
    let output = cli.output;

    ensure_release_mode()?;

    config.eval_options.measured_iterations = cli.eval_iterations;
    config.eval_options.warmup_iterations = cli.warmup_iterations;
    config.jobs = cli.jobs;
    if let Some(problems) = cli.problems {
        config.problems = expand_problem_selections(&problems)?;
    }
    if let Some(presets) = cli.presets {
        config.presets = expand_preset_selections(&presets)?;
    }
    if let Some(transcriptions) = cli.transcriptions {
        config.transcriptions = expand_transcription_selections(&transcriptions)?;
    }

    let mut progress = TerminalProgress::start(&config, &output);
    let suite = match run_ocp_benchmark_suite_with_progress(&config, |event| {
        progress.update(event);
    }) {
        Ok(suite) => suite,
        Err(error) => {
            progress.fail(error.to_string());
            return Err(error);
        }
    };
    progress.set_stage(
        "Rendering HTML report",
        "Assembling final dashboard".to_string(),
    );
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    progress.set_stage("Writing output", output.display().to_string());
    write_ocp_benchmark_report(&output, &suite)?;
    progress.finish(format!("wrote {}", output.display()));
    println!("wrote {}", output.display());
    Ok(())
}

#[derive(Debug, Parser)]
#[command(
    name = "ocp_bench_report",
    about = "Benchmark OCP setup/JIT/evaluation paths and render an HTML report."
)]
struct OcpBenchCli {
    #[arg(long, default_value = "target/ocp_bench_report.html")]
    output: PathBuf,
    #[arg(long = "eval-iterations", default_value_t = default_eval_iterations(), value_parser = parse_positive_usize)]
    eval_iterations: usize,
    #[arg(long = "warmup-iterations", default_value_t = default_warmup_iterations())]
    warmup_iterations: usize,
    #[arg(long, default_value_t = optimal_control_problems::default_benchmark_jobs(), value_parser = parse_positive_usize)]
    jobs: usize,
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    problems: Option<Vec<CliProblemSelection>>,
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    presets: Option<Vec<CliPresetSelection>>,
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    transcriptions: Option<Vec<CliTranscriptionSelection>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliProblemSelection {
    All,
    #[value(name = "optimal_distance_glider")]
    OptimalDistanceGlider,
    #[value(name = "linear_s_maneuver")]
    LinearSManeuver,
    #[value(name = "sailboat_upwind")]
    SailboatUpwind,
    #[value(name = "crane_transfer")]
    CraneTransfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliPresetSelection {
    All,
    #[value(name = "baseline")]
    Baseline,
    #[value(name = "inline_all")]
    InlineAll,
    #[value(name = "function_inline_at_call")]
    FunctionInlineAtCall,
    #[value(name = "function_inline_at_lowering")]
    FunctionInlineAtLowering,
    #[value(name = "function_inline_in_llvm")]
    FunctionInlineInLlvm,
    #[value(name = "function_noinline_llvm")]
    FunctionNoInlineLlvm,
    #[value(name = "baseline_with_ms_integrator")]
    BaselineWithMsIntegrator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliTranscriptionSelection {
    All,
    #[value(name = "ms", alias = "multiple_shooting")]
    Ms,
    #[value(name = "dc", alias = "direct_collocation")]
    Dc,
}

fn default_eval_iterations() -> usize {
    OcpBenchmarkSuiteConfig::default()
        .eval_options
        .measured_iterations
}

fn default_warmup_iterations() -> usize {
    OcpBenchmarkSuiteConfig::default()
        .eval_options
        .warmup_iterations
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid positive integer `{value}`"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn expand_problem_selections(selections: &[CliProblemSelection]) -> Result<Vec<ProblemId>> {
    expand_selection_list(
        selections,
        CliProblemSelection::All,
        OcpBenchmarkSuiteConfig::default().problems,
        |selection| match selection {
            CliProblemSelection::OptimalDistanceGlider => Some(ProblemId::OptimalDistanceGlider),
            CliProblemSelection::LinearSManeuver => Some(ProblemId::LinearSManeuver),
            CliProblemSelection::SailboatUpwind => Some(ProblemId::SailboatUpwind),
            CliProblemSelection::CraneTransfer => Some(ProblemId::CraneTransfer),
            CliProblemSelection::All => None,
        },
        "--problems",
    )
}

fn expand_preset_selections(selections: &[CliPresetSelection]) -> Result<Vec<OcpBenchmarkPreset>> {
    expand_selection_list(
        selections,
        CliPresetSelection::All,
        OcpBenchmarkPreset::all().to_vec(),
        |selection| match selection {
            CliPresetSelection::Baseline => Some(OcpBenchmarkPreset::Baseline),
            CliPresetSelection::InlineAll => Some(OcpBenchmarkPreset::InlineAll),
            CliPresetSelection::FunctionInlineAtCall => {
                Some(OcpBenchmarkPreset::FunctionInlineAtCall)
            }
            CliPresetSelection::FunctionInlineAtLowering => {
                Some(OcpBenchmarkPreset::FunctionInlineAtLowering)
            }
            CliPresetSelection::FunctionInlineInLlvm => {
                Some(OcpBenchmarkPreset::FunctionInlineInLlvm)
            }
            CliPresetSelection::FunctionNoInlineLlvm => {
                Some(OcpBenchmarkPreset::FunctionNoInlineLlvm)
            }
            CliPresetSelection::BaselineWithMsIntegrator => {
                Some(OcpBenchmarkPreset::BaselineWithMsIntegrator)
            }
            CliPresetSelection::All => None,
        },
        "--presets",
    )
}

fn expand_transcription_selections(
    selections: &[CliTranscriptionSelection],
) -> Result<Vec<TranscriptionMethod>> {
    expand_selection_list(
        selections,
        CliTranscriptionSelection::All,
        OcpBenchmarkSuiteConfig::default().transcriptions,
        |selection| match selection {
            CliTranscriptionSelection::Ms => Some(TranscriptionMethod::MultipleShooting),
            CliTranscriptionSelection::Dc => Some(TranscriptionMethod::DirectCollocation),
            CliTranscriptionSelection::All => None,
        },
        "--transcriptions",
    )
}

fn expand_selection_list<T, U>(
    selections: &[T],
    all_marker: T,
    all_values: Vec<U>,
    map: impl Fn(T) -> Option<U>,
    flag: &str,
) -> Result<Vec<U>>
where
    T: Copy + PartialEq,
    U: PartialEq,
{
    if selections.is_empty() {
        bail!("{flag} must not be empty");
    }
    if selections.contains(&all_marker) {
        if selections.len() > 1 {
            bail!("{flag} cannot combine `all` with specific selections");
        }
        return Ok(all_values);
    }

    let mut values = Vec::new();
    for selection in selections.iter().copied() {
        if let Some(value) = map(selection) {
            if !values.contains(&value) {
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        bail!("{flag} must not be empty");
    }
    Ok(values)
}

fn ensure_release_mode() -> Result<()> {
    if cfg!(debug_assertions) {
        bail!(
            "ocp_bench_report must be run in release mode\n\ntry:\n  cargo run -p optimal_control_problems --release --bin ocp_bench_report -- --output target/ocp_bench_report.html --presets all"
        );
    }
    Ok(())
}

struct TerminalProgress {
    interactive: bool,
    done: Arc<AtomicBool>,
    terminal: Option<Arc<TerminalCleanup>>,
    state: Arc<Mutex<ProgressState>>,
    worker: Option<thread::JoinHandle<()>>,
}

struct TerminalCleanup {
    active: AtomicBool,
    raw_mode: AtomicBool,
    done: Arc<AtomicBool>,
    last_snapshot: Mutex<Option<String>>,
}

const CRASH_TERMINAL_RECOVERY_BYTES: &[u8] =
    b"\r\n\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1004l\x1b[?1006l\x1b[?1015l\x1b[?2004l\x1b[?25h";
const CRASH_STATUS_PREFIX_BYTES: &[u8] = b"latest status: ";
const MAX_CRASH_STATUS_BYTES: usize = 240;

static LAST_CRASH_STATUS_PTR: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
static LAST_CRASH_STATUS_LEN: AtomicUsize = AtomicUsize::new(0);

fn crash_signal_message(signal: libc::c_int) -> &'static [u8] {
    match signal {
        SIGSEGV => b"\r\nocp_bench_report crashed with SIGSEGV; terminal reset\r\n",
        SIGTRAP => b"\r\nocp_bench_report crashed with SIGTRAP; terminal reset\r\n",
        SIGBUS => b"\r\nocp_bench_report crashed with SIGBUS; terminal reset\r\n",
        SIGILL => b"\r\nocp_bench_report crashed with SIGILL; terminal reset\r\n",
        SIGABRT => b"\r\nocp_bench_report crashed with SIGABRT; terminal reset\r\n",
        _ => b"\r\nocp_bench_report crashed; terminal reset\r\n",
    }
}

unsafe extern "C" fn crash_terminal_signal_handler(signal: libc::c_int) {
    // SAFETY: Async-signal-safe raw write of static bytes to stderr.
    let _ = unsafe {
        libc::write(
            libc::STDERR_FILENO,
            CRASH_TERMINAL_RECOVERY_BYTES.as_ptr().cast(),
            CRASH_TERMINAL_RECOVERY_BYTES.len(),
        )
    };
    let message = crash_signal_message(signal);
    // SAFETY: Async-signal-safe raw write of static bytes to stderr.
    let _ = unsafe { libc::write(libc::STDERR_FILENO, message.as_ptr().cast(), message.len()) };
    let status_len = LAST_CRASH_STATUS_LEN.load(Ordering::Acquire);
    let status_ptr = LAST_CRASH_STATUS_PTR.load(Ordering::Acquire);
    if !status_ptr.is_null() && status_len > 0 {
        // SAFETY: Async-signal-safe raw write of static bytes to stderr.
        let _ = unsafe {
            libc::write(
                libc::STDERR_FILENO,
                CRASH_STATUS_PREFIX_BYTES.as_ptr().cast(),
                CRASH_STATUS_PREFIX_BYTES.len(),
            )
        };
        // SAFETY: Pointer/length were published from leaked process-lifetime storage.
        let _ = unsafe { libc::write(libc::STDERR_FILENO, status_ptr.cast(), status_len) };
        // SAFETY: Async-signal-safe raw write of static bytes to stderr.
        let _ = unsafe { libc::write(libc::STDERR_FILENO, b"\r\n".as_ptr().cast(), 2) };
    }
    // SAFETY: Async-signal-safe immediate process exit from a fatal signal handler.
    unsafe { libc::_exit(128 + signal) };
}

fn truncate_to_boundary(message: &str, max_bytes: usize) -> &str {
    if message.len() <= max_bytes {
        return message;
    }
    let mut end = max_bytes;
    while end > 0 && !message.is_char_boundary(end) {
        end -= 1;
    }
    &message[..end]
}

fn publish_crash_status(status: &str) {
    let status = truncate_to_boundary(status, MAX_CRASH_STATUS_BYTES);
    let mut bytes = status.as_bytes().to_vec();
    if bytes.is_empty() {
        LAST_CRASH_STATUS_LEN.store(0, Ordering::Release);
        LAST_CRASH_STATUS_PTR.store(std::ptr::null_mut(), Ordering::Release);
        return;
    }
    let ptr = bytes.as_mut_ptr();
    let len = bytes.len();
    std::mem::forget(bytes);
    LAST_CRASH_STATUS_PTR.store(ptr, Ordering::Release);
    LAST_CRASH_STATUS_LEN.store(len, Ordering::Release);
}

fn install_terminal_crash_signal_handlers() {
    for signal in [SIGABRT, SIGBUS, SIGILL, SIGSEGV, SIGTRAP] {
        // SAFETY: The handler only writes static bytes to stderr and exits via `_exit`,
        // which are async-signal-safe operations. The sigaction struct is fully initialized.
        unsafe {
            let mut action = std::mem::zeroed::<libc::sigaction>();
            action.sa_sigaction = crash_terminal_signal_handler as *const () as libc::sighandler_t;
            action.sa_flags = 0;
            libc::sigemptyset(&mut action.sa_mask);
            let _ = libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
}

impl TerminalCleanup {
    fn new(done: Arc<AtomicBool>) -> Self {
        Self {
            active: AtomicBool::new(false),
            raw_mode: AtomicBool::new(false),
            done,
            last_snapshot: Mutex::new(None),
        }
    }

    fn request_shutdown(&self) {
        self.done.store(true, Ordering::SeqCst);
    }

    fn enter_terminal(&self) -> Result<Terminal<CrosstermBackend<io::Stderr>>> {
        enable_raw_mode()?;
        self.raw_mode.store(true, Ordering::SeqCst);
        let mut stderr = io::stderr();
        if let Err(error) = execute!(stderr, EnterAlternateScreen, EnableMouseCapture, Hide) {
            let _ = disable_raw_mode();
            self.raw_mode.store(false, Ordering::SeqCst);
            return Err(error.into());
        }
        self.active.store(true, Ordering::SeqCst);
        match Terminal::new(CrosstermBackend::new(stderr)) {
            Ok(terminal) => Ok(terminal),
            Err(error) => {
                self.restore_terminal();
                Err(error.into())
            }
        }
    }

    fn update_snapshot(&self, snapshot: String) {
        let mut guard = match self.last_snapshot.lock() {
            Ok(guard) => guard,
            Err(poison) => poison.into_inner(),
        };
        *guard = Some(snapshot);
    }

    fn restore_terminal(&self) {
        self.restore_terminal_inner(None);
    }

    fn restore_terminal_with_message(&self, message: &str) {
        self.restore_terminal_inner(Some(message));
    }

    fn restore_terminal_inner(&self, message: Option<&str>) {
        let was_active = self.active.swap(false, Ordering::SeqCst);
        if !was_active && message.is_none() {
            return;
        }

        let mut stderr = io::stderr();
        if was_active {
            let _ = execute!(stderr, Show, DisableMouseCapture, LeaveAlternateScreen);
        }
        if self.raw_mode.swap(false, Ordering::SeqCst) {
            let _ = disable_raw_mode();
        }

        let snapshot = match self.last_snapshot.lock() {
            Ok(guard) => guard.clone(),
            Err(poison) => poison.into_inner().clone(),
        };
        if let Some(snapshot) = snapshot {
            let _ = writeln!(stderr, "{snapshot}");
        }
        if let Some(message) = message {
            let _ = writeln!(stderr, "{message}");
        }
        let _ = stderr.flush();
    }
}

#[derive(Clone)]
struct ProgressState {
    problems: Vec<ProblemId>,
    transcriptions: Vec<TranscriptionMethod>,
    presets: Vec<OcpBenchmarkPreset>,
    row_keys: Vec<(ProblemId, TranscriptionMethod)>,
    case_cells: Vec<CaseCell>,
    total_cases: usize,
    started_cases: usize,
    latest_case: Option<OcpBenchmarkCase>,
    stage: String,
    detail: String,
    completed_cases: usize,
    started_at: Instant,
    latest_event_started_at: Instant,
    output_path: PathBuf,
    eval_iterations: usize,
    warmup_iterations: usize,
    jobs: usize,
    final_message: Option<String>,
    warning_count: usize,
}

#[derive(Clone)]
struct CaseCell {
    case: OcpBenchmarkCase,
    symbolic: StageCell,
    active_symbolic_stage: Option<MatrixStage>,
    jit: StageCell,
    nlp_jit: StageCell,
    xdot_helper_jit: StageCell,
    multiple_shooting_arc_helper_jit: StageCell,
    objective: StageCell,
    gradient: StageCell,
    jacobian: StageCell,
    hessian: StageCell,
    equality_count: usize,
    inequality_count: usize,
    compile_total_s: Option<f64>,
    symbolic_construction_s: Option<f64>,
    compile_objective_gradient_s: Option<f64>,
    compile_equality_jacobian_s: Option<f64>,
    compile_inequality_jacobian_s: Option<f64>,
    lagrangian_assembly_s: Option<f64>,
    compile_hessian_generation_s: Option<f64>,
    compile_nlp_jit_s: Option<f64>,
    xdot_helper_compile_s: Option<f64>,
    multiple_shooting_arc_helper_compile_s: Option<f64>,
    objective_avg_s: Option<f64>,
    objective_stddev_s: Option<f64>,
    gradient_avg_s: Option<f64>,
    gradient_stddev_s: Option<f64>,
    equality_jacobian_avg_s: Option<f64>,
    equality_jacobian_stddev_s: Option<f64>,
    inequality_jacobian_avg_s: Option<f64>,
    inequality_jacobian_stddev_s: Option<f64>,
    jacobian_avg_s: Option<f64>,
    jacobian_stddev_s: Option<f64>,
    hessian_avg_s: Option<f64>,
    hessian_stddev_s: Option<f64>,
    gradient_nnz: Option<usize>,
    jacobian_nnz: Option<usize>,
    hessian_nnz: Option<usize>,
    llvm_root_instruction_count: Option<usize>,
    llvm_total_instruction_count: Option<usize>,
    eval_samples: usize,
    sanity: SanityStatus,
    warnings: usize,
}

#[derive(Clone, Copy)]
struct ActiveJob {
    case: OcpBenchmarkCase,
    stage: MatrixStage,
    started_at: Instant,
}

const ACTIVE_JOB_TIME_COL_WIDTH: u16 = 8;

#[derive(Clone)]
enum StageCell {
    Pending,
    Running(Instant),
    Done(Option<f64>),
    Skipped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WinnerMetric {
    Symbolic,
    Jit,
    Eval100,
    Overall100,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SanityStatus {
    Ok,
    AllZero,
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixStage {
    Build,
    SymbolicGradient,
    EqualityJacobianBuild,
    InequalityJacobianBuild,
    LagrangianAssembly,
    HessianGeneration,
    NlpJit,
    XdotHelperJit,
    MultipleShootingArcHelperJit,
    Objective,
    Gradient,
    Jacobian,
    Hessian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NnzMetric {
    Gradient,
    Jacobian,
    Hessian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SizeMetric {
    RootInstructions,
    TotalInstructions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixRow {
    Time(MatrixStage),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixSection {
    Symbolic,
    Jit,
    Runtime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LabelDetail {
    Short,
    Medium,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotCellStyle {
    fg: Color,
    bg: Color,
    modifier: Modifier,
}

impl TerminalProgress {
    fn start(
        config: &optimal_control_problems::OcpBenchmarkSuiteConfig,
        output_path: &PathBuf,
    ) -> Self {
        let interactive = io::stderr().is_terminal();
        let row_keys = config
            .problems
            .iter()
            .flat_map(|&problem_id| {
                config
                    .transcriptions
                    .iter()
                    .map(move |&transcription| (problem_id, transcription))
            })
            .collect::<Vec<_>>();
        let case_cells = row_keys
            .iter()
            .flat_map(|&(problem_id, transcription)| {
                config.presets.iter().map(move |&preset| CaseCell {
                    case: OcpBenchmarkCase {
                        problem_id,
                        transcription,
                        preset,
                    },
                    symbolic: StageCell::Pending,
                    active_symbolic_stage: None,
                    jit: StageCell::Pending,
                    nlp_jit: StageCell::Pending,
                    xdot_helper_jit: StageCell::Pending,
                    multiple_shooting_arc_helper_jit: StageCell::Pending,
                    objective: StageCell::Pending,
                    gradient: StageCell::Pending,
                    jacobian: StageCell::Pending,
                    hessian: StageCell::Pending,
                    equality_count: 0,
                    inequality_count: 0,
                    compile_total_s: None,
                    symbolic_construction_s: None,
                    compile_objective_gradient_s: None,
                    compile_equality_jacobian_s: None,
                    compile_inequality_jacobian_s: None,
                    lagrangian_assembly_s: None,
                    compile_hessian_generation_s: None,
                    compile_nlp_jit_s: None,
                    xdot_helper_compile_s: None,
                    multiple_shooting_arc_helper_compile_s: None,
                    objective_avg_s: None,
                    objective_stddev_s: None,
                    gradient_avg_s: None,
                    gradient_stddev_s: None,
                    equality_jacobian_avg_s: None,
                    equality_jacobian_stddev_s: None,
                    inequality_jacobian_avg_s: None,
                    inequality_jacobian_stddev_s: None,
                    jacobian_avg_s: None,
                    jacobian_stddev_s: None,
                    hessian_avg_s: None,
                    hessian_stddev_s: None,
                    gradient_nnz: None,
                    jacobian_nnz: None,
                    hessian_nnz: None,
                    llvm_root_instruction_count: None,
                    llvm_total_instruction_count: None,
                    eval_samples: 0,
                    sanity: SanityStatus::Ok,
                    warnings: 0,
                })
            })
            .collect::<Vec<_>>();
        let state = Arc::new(Mutex::new(ProgressState {
            problems: config.problems.clone(),
            transcriptions: config.transcriptions.clone(),
            presets: config.presets.clone(),
            row_keys,
            case_cells,
            total_cases: config.problems.len() * config.transcriptions.len() * config.presets.len(),
            started_cases: 0,
            latest_case: None,
            stage: "Queued".to_string(),
            detail: "Preparing benchmark matrix".to_string(),
            completed_cases: 0,
            started_at: Instant::now(),
            latest_event_started_at: Instant::now(),
            output_path: output_path.clone(),
            eval_iterations: config.eval_options.measured_iterations,
            warmup_iterations: config.eval_options.warmup_iterations,
            jobs: config.jobs,
            final_message: None,
            warning_count: 0,
        }));
        {
            let state = match state.lock() {
                Ok(guard) => guard,
                Err(poison) => poison.into_inner(),
            };
            publish_crash_status(&format_crash_status(&state));
        }
        let done = Arc::new(AtomicBool::new(false));
        let terminal = interactive.then(|| Arc::new(TerminalCleanup::new(Arc::clone(&done))));
        if let Some(terminal) = &terminal {
            install_terminal_signal_handlers(Arc::clone(terminal));
            install_terminal_panic_hook(Arc::clone(terminal));
        }
        install_terminal_crash_signal_handlers();
        let worker = interactive.then(|| {
            let state = Arc::clone(&state);
            let terminal = Arc::clone(
                terminal
                    .as_ref()
                    .expect("interactive terminal cleanup should exist"),
            );
            thread::spawn(move || render_progress_loop(state, terminal))
        });
        Self {
            interactive,
            done,
            terminal,
            state,
            worker,
        }
    }

    fn update(&mut self, event: OcpBenchmarkProgress) {
        if self.interactive {
            let mut state = self.lock_state();
            apply_progress_event(&mut state, event);
            publish_crash_status(&format_crash_status(&state));
        } else {
            print_event_line(event);
        }
    }

    fn set_stage(&mut self, stage: &str, detail: String) {
        if self.interactive {
            let mut state = self.lock_state();
            state.stage = stage.to_string();
            state.detail = detail;
            state.latest_event_started_at = Instant::now();
            publish_crash_status(&format_crash_status(&state));
        } else {
            eprintln!("{stage}: {detail}");
        }
    }

    fn finish(&mut self, final_message: String) {
        if self.interactive {
            {
                let mut state = self.lock_state();
                state.final_message = Some(final_message);
                publish_crash_status(&format_crash_status(&state));
            }
            self.done.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn fail(&mut self, error_message: String) {
        if self.interactive {
            {
                let mut state = self.lock_state();
                state.stage = "Failed".to_string();
                state.detail = error_message;
                state.final_message = Some(String::new());
                state.latest_event_started_at = Instant::now();
                publish_crash_status(&format_crash_status(&state));
            }
            self.done.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        } else {
            eprintln!("benchmark failed");
        }
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, ProgressState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poison) => poison.into_inner(),
        }
    }
}

impl Drop for TerminalProgress {
    fn drop(&mut self) {
        if self.interactive {
            self.done.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
            if let Some(terminal) = &self.terminal {
                terminal.restore_terminal();
            }
        }
    }
}

fn apply_progress_event(state: &mut ProgressState, event: OcpBenchmarkProgress) {
    match event {
        OcpBenchmarkProgress::CaseStarted {
            current,
            total: _,
            case,
        } => {
            state.started_cases = state.started_cases.max(current);
            state.latest_case = Some(case);
            state.stage = "Compiling symbolic model".to_string();
            state.detail = format!(
                "{} | {} | {}",
                case.problem_label(),
                case.transcription_label(),
                case.preset_label(),
            );
            state.latest_event_started_at = Instant::now();
            update_case_started(state, case);
        }
        OcpBenchmarkProgress::CompileProgress {
            current,
            total: _,
            case,
            progress,
        } => {
            state.started_cases = state.started_cases.max(current);
            state.latest_case = Some(case);
            match &progress {
                OcpCompileProgress::SymbolicStage(progress) => {
                    state.stage = "Compiling symbolic model".to_string();
                    state.detail = format!(
                        "{} | vars={} eq={} ineq={} jac_nnz={} hess_nnz={}",
                        symbolic_compile_stage_label(progress.stage),
                        progress.metadata.stats.variable_count,
                        progress.metadata.stats.equality_count,
                        progress.metadata.stats.inequality_count,
                        progress.metadata.stats.equality_jacobian_nnz
                            + progress.metadata.stats.inequality_jacobian_nnz,
                        progress.metadata.stats.hessian_nnz,
                    );
                }
                OcpCompileProgress::SymbolicReady(metadata) => {
                    state.stage = "LLVM JIT compiling NLP kernels".to_string();
                    state.detail = format!(
                        "vars={} eq={} ineq={} jac_nnz={} hess_nnz={}",
                        metadata.stats.variable_count,
                        metadata.stats.equality_count,
                        metadata.stats.inequality_count,
                        metadata.stats.equality_jacobian_nnz
                            + metadata.stats.inequality_jacobian_nnz,
                        metadata.stats.hessian_nnz,
                    );
                }
                OcpCompileProgress::HelperCompiled { helper, elapsed } => {
                    state.stage = "Compiling helper kernels".to_string();
                    state.detail = format!(
                        "{} ready in {}",
                        match helper {
                            optimal_control::OcpCompileHelperKind::Xdot => "xdot helper",
                            optimal_control::OcpCompileHelperKind::MultipleShootingArc => {
                                "rk4 arc helper"
                            }
                        },
                        format_duration(*elapsed),
                    );
                }
            }
            state.latest_event_started_at = Instant::now();
            update_case_compile_progress(state, case, progress);
        }
        OcpBenchmarkProgress::EvalKernelStarted {
            current,
            total: _,
            case,
            kernel,
        } => {
            state.started_cases = state.started_cases.max(current);
            state.latest_case = Some(case);
            state.stage = format!("Benchmarking {}", kernel_label(kernel));
            state.detail = format!(
                "{} measured / {} warmup iterations",
                state.eval_iterations, state.warmup_iterations,
            );
            state.latest_event_started_at = Instant::now();
            update_case_eval_stage(state, case, kernel);
        }
        OcpBenchmarkProgress::CaseFinished {
            current,
            total: _,
            case,
            record,
        } => {
            state.completed_cases = current;
            state.latest_case = Some(case);
            state.stage = "Case complete".to_string();
            state.detail = format!(
                "{} | {} | {}",
                case.problem_label(),
                case.transcription_label(),
                case.preset_label(),
            );
            state.latest_event_started_at = Instant::now();
            update_case_finished(state, &record);
        }
    }
}

fn render_progress_loop(state: Arc<Mutex<ProgressState>>, terminal: Arc<TerminalCleanup>) {
    let mut terminal_ui = match terminal.enter_terminal() {
        Ok(terminal_ui) => terminal_ui,
        Err(error) => {
            eprintln!("failed to initialize terminal UI: {error}");
            return;
        }
    };
    let frames = ["◜", "◠", "◝", "◞", "◡", "◟"];
    let mut frame = 0usize;
    let _ = terminal_ui.clear();

    loop {
        let snapshot = match state.lock() {
            Ok(guard) => guard.clone(),
            Err(poison) => poison.into_inner().clone(),
        };
        let completed_frame = match terminal_ui.draw(|frame_buf| {
            render_dashboard(frame_buf, &snapshot, frames[frame % frames.len()]);
        }) {
            Ok(completed_frame) => completed_frame,
            Err(error) => {
                drop(terminal_ui);
                terminal.restore_terminal_with_message(&format!("terminal UI error: {error}"));
                break;
            }
        };
        terminal.update_snapshot(buffer_to_ansi_text(completed_frame.buffer));
        frame += 1;

        if let Err(error) = drain_terminal_events(&terminal) {
            drop(terminal_ui);
            terminal.restore_terminal_with_message(&format!("terminal input error: {error}"));
            break;
        }

        if terminal.done.load(Ordering::SeqCst) {
            drop(terminal_ui);
            terminal.restore_terminal();
            break;
        }

        thread::sleep(Duration::from_millis(120));
    }
}

fn format_crash_status(state: &ProgressState) -> String {
    match state.latest_case {
        Some(case) => format!(
            "{} / {} / {} | {} | {}",
            case.problem_label(),
            case.transcription_label(),
            case.preset_label(),
            state.stage,
            state.detail
        ),
        None => format!("{} | {}", state.stage, state.detail),
    }
}

fn drain_terminal_events(terminal: &TerminalCleanup) -> Result<()> {
    while event::poll(Duration::from_millis(0))? {
        match event::read()? {
            Event::Key(key)
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C')) =>
            {
                terminal.request_shutdown();
                terminal.restore_terminal();
                process::exit(130);
            }
            Event::Resize(_, _) => {}
            Event::Mouse(_) => {}
            Event::FocusGained | Event::FocusLost => {}
            Event::Paste(_) => {}
            Event::Key(_) => {}
        }
    }
    Ok(())
}

fn install_terminal_signal_handlers(terminal: Arc<TerminalCleanup>) {
    let mut signals = match Signals::new([SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGABRT]) {
        Ok(signals) => signals,
        Err(_) => return,
    };
    thread::spawn(move || {
        if let Some(signal) = signals.forever().next() {
            terminal.request_shutdown();
            terminal.restore_terminal();
            process::exit(128 + signal);
        }
    });
}

fn install_terminal_panic_hook(terminal: Arc<TerminalCleanup>) {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        terminal.request_shutdown();
        terminal.restore_terminal();
        previous(panic_info);
    }));
}

fn render_dashboard(frame: &mut Frame, state: &ProgressState, spinner: &str) {
    let area = frame.area();
    if area.height < 14 || area.width < 40 {
        frame.render_widget(Paragraph::new("terminal too small for dashboard"), area);
        return;
    }

    let sections = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            Constraint::Length(14),
            Constraint::Min(4),
            Constraint::Length(11),
        ])
        .split(area);

    render_overview(frame, sections[0], state, spinner);
    render_matrix_bands(frame, sections[1], state);
    render_bottom_summary_widgets(frame, sections[2], state);
}

fn render_overview(frame: &mut Frame, area: Rect, state: &ProgressState, spinner: &str) {
    let block = panel_block("Overview", Color::Cyan);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 8 {
        return;
    }
    let rows = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    let elapsed = state.started_at.elapsed();
    let remaining_cases = state.total_cases.saturating_sub(state.completed_cases);
    let eta = if state.completed_cases > 0 {
        Some(elapsed.div_f64(state.completed_cases as f64) * remaining_cases as u32)
    } else {
        None
    };
    let header = Line::from(vec![
        Span::styled(
            format!("{spinner} OCP Benchmark Dashboard"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(format_duration(elapsed), Style::default().fg(Color::White)),
        Span::raw("  eta "),
        Span::styled(
            eta.map(format_duration).unwrap_or_else(|| "--".to_string()),
            Style::default().fg(Color::White),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{}/{}", state.completed_cases, state.total_cases),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]);
    frame.render_widget(Paragraph::new(header), rows[0]);

    let ratio = if state.total_cases == 0 {
        0.0
    } else {
        state.completed_cases as f64 / state.total_cases as f64
    };
    let running_cases = state
        .case_cells
        .iter()
        .filter(|cell| case_is_running(cell))
        .count();
    let queued_cases = state.total_cases.saturating_sub(state.started_cases);
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(ratio)
        .label(format!(
            "running {}   jobs {}   queued {}   warnings {}",
            running_cases, state.jobs, queued_cases, state.warning_count
        ));
    frame.render_widget(gauge, rows[1]);

    let avg_compile = average_metric_value(state, |cell| cell.compile_total_s);
    let avg_objective = average_metric_value(state, |cell| cell.objective_avg_s);
    let avg_gradient = average_metric_value(state, |cell| cell.gradient_avg_s);
    let avg_jacobian = average_metric_value(state, |cell| cell.jacobian_avg_s);
    let avg_hessian = average_metric_value(state, |cell| cell.hessian_avg_s);
    let avg_objective_stddev = average_metric_value(state, |cell| cell.objective_stddev_s);
    let avg_gradient_stddev = average_metric_value(state, |cell| cell.gradient_stddev_s);
    let avg_jacobian_stddev = average_metric_value(state, |cell| cell.jacobian_stddev_s);
    let avg_hessian_stddev = average_metric_value(state, |cell| cell.hessian_stddev_s);
    let summary_detail = overview_summary_detail(inner.width);
    let summary = Line::from(vec![
        muted_label("Avg"),
        Span::raw(" "),
        Span::styled(
            format!("(n={})", state.eval_iterations),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw(" "),
        stat_span(
            summary_metric_label("compile", summary_detail),
            format_optional_seconds(avg_compile).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("objective", summary_detail),
            format_optional_seconds(avg_objective).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("gradient", summary_detail),
            format_optional_seconds(avg_gradient).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("jacobian", summary_detail),
            format_optional_seconds(avg_jacobian).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("hessian", summary_detail),
            format_optional_seconds(avg_hessian).as_str(),
        ),
    ]);
    frame.render_widget(Paragraph::new(summary), rows[2]);

    let variance = Line::from(vec![
        muted_label("StdDev"),
        Span::raw(" "),
        stat_span(summary_metric_label("compile", summary_detail), "--"),
        Span::raw("   "),
        stat_span(
            summary_metric_label("objective", summary_detail),
            format_optional_seconds(avg_objective_stddev).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("gradient", summary_detail),
            format_optional_seconds(avg_gradient_stddev).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("jacobian", summary_detail),
            format_optional_seconds(avg_jacobian_stddev).as_str(),
        ),
        Span::raw("   "),
        stat_span(
            summary_metric_label("hessian", summary_detail),
            format_optional_seconds(avg_hessian_stddev).as_str(),
        ),
    ]);
    frame.render_widget(Paragraph::new(variance), rows[3]);

    render_overview_status(frame, rows[4], state);
    render_overview_details(frame, rows[5], state);
}

fn muted_label(text: &'static str) -> Span<'static> {
    Span::styled(
        text,
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    )
}

fn stat_span(label: &'static str, value: &str) -> Span<'static> {
    Span::styled(
        format!("{label} {value}"),
        Style::default().fg(Color::White),
    )
}

fn overview_summary_detail(width: u16) -> LabelDetail {
    if width >= 120 {
        LabelDetail::Full
    } else if width >= 92 {
        LabelDetail::Medium
    } else {
        LabelDetail::Short
    }
}

fn summary_metric_label(metric: &'static str, detail: LabelDetail) -> &'static str {
    match (metric, detail) {
        ("compile", LabelDetail::Short) => "C",
        ("objective", LabelDetail::Short) => "O",
        ("gradient", LabelDetail::Short) => "G",
        ("jacobian", LabelDetail::Short) => "J",
        ("hessian", LabelDetail::Short) => "H",
        ("compile", _) => "compile",
        ("objective", _) => "objective",
        ("gradient", _) => "gradient",
        ("jacobian", _) => "jacobian",
        ("hessian", _) => "hessian",
        _ => metric,
    }
}

fn render_overview_details(frame: &mut Frame, area: Rect, state: &ProgressState) {
    if area.height < 2 || area.width < 40 {
        return;
    }

    if area.width >= 150 {
        let columns = Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([
                Constraint::Percentage(36),
                Constraint::Percentage(28),
                Constraint::Percentage(36),
            ])
            .split(area);
        render_active_jobs_panel(frame, columns[0], state);
        render_run_settings_panel(frame, columns[1], state);
        render_preset_legend_panel(frame, columns[2]);
        return;
    }

    let active_height = area.height.saturating_sub(4).clamp(4, 8);
    let sections = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Length(active_height), Constraint::Min(2)])
        .split(area);
    render_active_jobs_panel(frame, sections[0], state);

    let columns = Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints(if sections[1].width >= 110 {
            vec![Constraint::Percentage(44), Constraint::Percentage(56)]
        } else {
            vec![Constraint::Percentage(48), Constraint::Percentage(52)]
        })
        .split(sections[1]);
    render_run_settings_panel(frame, columns[0], state);
    render_preset_legend_panel(frame, columns[1]);
}

fn render_overview_status(frame: &mut Frame, area: Rect, state: &ProgressState) {
    if area.width < 20 {
        return;
    }
    let status_style = if state.stage == "Failed" {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else if state
        .final_message
        .as_deref()
        .is_some_and(|message| !message.is_empty())
    {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    };
    let line = Line::from(vec![
        muted_label("Status"),
        Span::raw(" "),
        Span::styled(state.stage.clone(), status_style),
        Span::raw("  "),
        Span::styled(state.detail.clone(), Style::default().fg(Color::White)),
    ]);
    frame.render_widget(Paragraph::new(line).wrap(Wrap { trim: true }), area);
}

fn render_active_jobs_panel(frame: &mut Frame, area: Rect, state: &ProgressState) {
    let block = panel_block("Active Jobs", Color::Cyan);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 1 || inner.width < ACTIVE_JOB_TIME_COL_WIDTH + 4 {
        return;
    }

    let jobs = active_jobs(state);
    if jobs.is_empty() {
        frame.render_widget(
            Paragraph::new("No active jobs").style(
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            ),
            inner,
        );
        return;
    }

    let text_width = inner
        .width
        .saturating_sub(ACTIVE_JOB_TIME_COL_WIDTH)
        .saturating_sub(1);
    let detail = if text_width >= 56 {
        LabelDetail::Full
    } else if text_width >= 38 {
        LabelDetail::Medium
    } else {
        LabelDetail::Short
    };

    let visible_limit = inner.height as usize;
    let hidden_count = jobs.len().saturating_sub(visible_limit);
    let visible_jobs = if hidden_count > 0 {
        visible_limit.saturating_sub(1)
    } else {
        visible_limit
    };

    let mut rows = jobs
        .into_iter()
        .take(visible_jobs)
        .map(|job| active_job_row(job, detail))
        .collect::<Vec<_>>();
    if hidden_count > 0 {
        rows.push(Row::new(vec![
            ratatui::widgets::Cell::from(""),
            ratatui::widgets::Cell::from(Span::styled(
                format!("+{hidden_count} more active jobs"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )),
        ]));
    }

    let table = Table::new(
        rows,
        [
            Constraint::Length(ACTIVE_JOB_TIME_COL_WIDTH),
            Constraint::Min(1),
        ],
    )
    .column_spacing(1);
    frame.render_widget(table, inner);
}

fn render_run_settings_panel(frame: &mut Frame, area: Rect, state: &ProgressState) {
    let run_block = panel_block("Run Settings", Color::Cyan);
    let run_inner = run_block.inner(area);
    frame.render_widget(run_block, area);
    frame.render_widget(
        Paragraph::new(overview_run_lines(state, run_inner.width)).wrap(Wrap { trim: true }),
        run_inner,
    );
}

fn render_preset_legend_panel(frame: &mut Frame, area: Rect) {
    let legend_block = panel_block("Preset Legend", Color::Cyan);
    let legend_inner = legend_block.inner(area);
    frame.render_widget(legend_block, area);
    frame.render_widget(
        Paragraph::new(overview_legend_lines(legend_inner.width)).wrap(Wrap { trim: true }),
        legend_inner,
    );
}

fn active_jobs(state: &ProgressState) -> Vec<ActiveJob> {
    let mut jobs = state
        .case_cells
        .iter()
        .filter_map(active_job_for_cell)
        .collect::<Vec<_>>();
    jobs.sort_by(|lhs, rhs| lhs.started_at.cmp(&rhs.started_at));
    jobs
}

fn active_job_for_cell(cell: &CaseCell) -> Option<ActiveJob> {
    let stage = if matches!(cell.hessian, StageCell::Running(_)) {
        Some(MatrixStage::Hessian)
    } else if matches!(cell.jacobian, StageCell::Running(_)) {
        Some(MatrixStage::Jacobian)
    } else if matches!(cell.gradient, StageCell::Running(_)) {
        Some(MatrixStage::Gradient)
    } else if matches!(cell.objective, StageCell::Running(_)) {
        Some(MatrixStage::Objective)
    } else if matches!(cell.multiple_shooting_arc_helper_jit, StageCell::Running(_)) {
        Some(MatrixStage::MultipleShootingArcHelperJit)
    } else if matches!(cell.xdot_helper_jit, StageCell::Running(_)) {
        Some(MatrixStage::XdotHelperJit)
    } else if matches!(cell.nlp_jit, StageCell::Running(_)) {
        Some(MatrixStage::NlpJit)
    } else if matches!(cell.symbolic, StageCell::Running(_)) {
        cell.active_symbolic_stage
    } else {
        None
    }?;

    let started_at = match stage {
        MatrixStage::Build
        | MatrixStage::SymbolicGradient
        | MatrixStage::EqualityJacobianBuild
        | MatrixStage::InequalityJacobianBuild
        | MatrixStage::LagrangianAssembly
        | MatrixStage::HessianGeneration => match &cell.symbolic {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::NlpJit => match &cell.nlp_jit {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::XdotHelperJit => match &cell.xdot_helper_jit {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::MultipleShootingArcHelperJit => match &cell.multiple_shooting_arc_helper_jit {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::Objective => match &cell.objective {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::Gradient => match &cell.gradient {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::Jacobian => match &cell.jacobian {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
        MatrixStage::Hessian => match &cell.hessian {
            StageCell::Running(started_at) => *started_at,
            _ => return None,
        },
    };

    Some(ActiveJob {
        case: cell.case,
        stage,
        started_at,
    })
}

fn active_job_row(job: ActiveJob, detail: LabelDetail) -> Row<'static> {
    let case = job.case;
    let stage = matrix_row_display_name_with_detail(MatrixRow::Time(job.stage), detail);
    let problem = if detail == LabelDetail::Full {
        case.problem_label()
    } else {
        short_problem_label(case.problem_id)
    };
    Row::new(vec![
        ratatui::widgets::Cell::from(Span::styled(
            format!(
                "{:>width$}",
                format_active_job_duration(job.started_at.elapsed()),
                width = ACTIVE_JOB_TIME_COL_WIDTH as usize
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        ratatui::widgets::Cell::from(Line::from(vec![
            Span::styled(problem, Style::default().fg(problem_category_color())),
            Span::raw(" / "),
            Span::styled(
                overview_transcription_label(
                    case.transcription,
                    panel_width_for_label_detail(detail),
                ),
                Style::default().fg(transcription_category_color()),
            ),
            Span::raw(" / "),
            Span::styled(stage, Style::default().fg(Color::Gray)),
            Span::raw("  "),
            Span::styled(
                overview_preset_label(case.preset, panel_width_for_label_detail(detail)),
                Style::default().fg(strategy_category_color()),
            ),
        ])),
    ])
}

fn format_active_job_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 60.0 {
        let whole_seconds = duration.as_secs();
        format!("{:02}:{:02}", whole_seconds / 60, whole_seconds % 60)
    } else if seconds >= 1.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{}ms", duration.as_millis())
    }
}

fn panel_width_for_label_detail(detail: LabelDetail) -> u16 {
    match detail {
        LabelDetail::Short => 80,
        LabelDetail::Medium => 100,
        LabelDetail::Full => 140,
    }
}

fn overview_run_lines(state: &ProgressState, width: u16) -> Vec<Line<'static>> {
    let problem_names = state
        .problems
        .iter()
        .map(|&problem| {
            if width >= 56 {
                matrix_problem_title(problem)
            } else {
                short_problem_label(problem)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let transcription_names = state
        .transcriptions
        .iter()
        .map(|&method| {
            if width >= 60 {
                transcription_display_name(method)
            } else {
                short_transcription_label(method)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let preset_detail = if width >= 72 {
        LabelDetail::Medium
    } else {
        LabelDetail::Short
    };
    let preset_names = state
        .presets
        .iter()
        .map(|&preset| preset_display_name_with_detail(preset, preset_detail))
        .collect::<Vec<_>>()
        .join(", ");
    vec![
        overview_field_line(
            "jobs",
            format!(
                "{}   eval {}   warmup {}",
                state.jobs, state.eval_iterations, state.warmup_iterations
            ),
            Color::White,
        ),
        overview_field_line(
            "transcriptions",
            transcription_names,
            transcription_category_color(),
        ),
        overview_field_line("problems", problem_names, problem_category_color()),
        overview_field_line("presets", preset_names, strategy_category_color()),
        overview_field_line(
            "output",
            state.output_path.display().to_string(),
            Color::Gray,
        ),
    ]
}

fn overview_legend_lines(width: u16) -> Vec<Line<'static>> {
    let entries = preset_legend_entries();
    let mut lines = Vec::new();
    if width >= 88 {
        let left_width = entries
            .chunks(2)
            .filter_map(|pair| pair.first().copied())
            .map(legend_entry_width)
            .max()
            .unwrap_or(0)
            + 4;
        for pair in entries.chunks(2) {
            let left = pair[0];
            if let Some(right) = pair.get(1).copied() {
                lines.push(legend_two_column_line(left, right, left_width));
            } else {
                lines.push(legend_single_line(left));
            }
        }
    } else {
        lines.extend(entries.into_iter().map(legend_single_line));
    }
    lines
}

fn overview_field_line(label: &'static str, value: String, value_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label}: "),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(value, Style::default().fg(value_color)),
    ])
}

fn preset_legend_entries() -> Vec<(OcpBenchmarkPreset, &'static str, &'static str)> {
    vec![
        (OcpBenchmarkPreset::Baseline, "baseline", "OCP defaults"),
        (
            OcpBenchmarkPreset::InlineAll,
            "inline",
            "inline repeated kernels immediately",
        ),
        (
            OcpBenchmarkPreset::FunctionInlineAtCall,
            "call",
            "functions inline at call construction",
        ),
        (
            OcpBenchmarkPreset::FunctionInlineAtLowering,
            "lower",
            "functions inline during lowering",
        ),
        (
            OcpBenchmarkPreset::FunctionInlineInLlvm,
            "llvm",
            "functions survive to LLVM; LLVM may inline",
        ),
        (
            OcpBenchmarkPreset::FunctionNoInlineLlvm,
            "noinline",
            "preserve internal LLVM function calls",
        ),
        (
            OcpBenchmarkPreset::BaselineWithMsIntegrator,
            "ms int",
            "baseline plus reusable MS integrator",
        ),
    ]
}

fn legend_entry_width(entry: (OcpBenchmarkPreset, &'static str, &'static str)) -> usize {
    entry.1.chars().count() + 2 + entry.2.chars().count()
}

fn legend_single_line(entry: (OcpBenchmarkPreset, &'static str, &'static str)) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{:>8}", entry.1),
            Style::default()
                .fg(strategy_category_color())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": ", Style::default().fg(Color::DarkGray)),
        Span::styled(entry.2, Style::default().fg(Color::Gray)),
    ])
}

fn legend_two_column_line(
    left: (OcpBenchmarkPreset, &'static str, &'static str),
    right: (OcpBenchmarkPreset, &'static str, &'static str),
    left_width: usize,
) -> Line<'static> {
    let left_len = legend_entry_width(left);
    let spacer = " ".repeat(left_width.saturating_sub(left_len));
    Line::from(vec![
        Span::styled(
            format!("{:>8}", left.1),
            Style::default()
                .fg(strategy_category_color())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": ", Style::default().fg(Color::DarkGray)),
        Span::styled(left.2, Style::default().fg(Color::Gray)),
        Span::raw(spacer),
        Span::styled(
            format!("{:>8}", right.1),
            Style::default()
                .fg(strategy_category_color())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": ", Style::default().fg(Color::DarkGray)),
        Span::styled(right.2, Style::default().fg(Color::Gray)),
    ])
}

fn overview_transcription_label(method: TranscriptionMethod, width: u16) -> &'static str {
    if width >= 96 {
        transcription_display_name(method)
    } else {
        short_transcription_label(method)
    }
}

fn overview_preset_label(preset: OcpBenchmarkPreset, width: u16) -> &'static str {
    if width >= 120 {
        preset_display_name_with_detail(preset, LabelDetail::Full)
    } else if width >= 96 {
        preset_display_name_with_detail(preset, LabelDetail::Medium)
    } else {
        preset_display_name_with_detail(preset, LabelDetail::Short)
    }
}

fn panel_block<'a>(title: &'a str, color: Color) -> Block<'a> {
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray))
}

fn render_matrix_bands(frame: &mut Frame, area: Rect, state: &ProgressState) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    if inner.height < 4 || inner.width < 60 {
        return;
    }

    let panel_areas = Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(inner);
    let section_specs = [
        (MatrixSection::Symbolic, "Symbolic", Color::LightGreen),
        (MatrixSection::Jit, "JIT", Color::LightBlue),
        (MatrixSection::Runtime, "Eval", Color::LightMagenta),
    ];
    for ((section, title, color), panel_area) in
        section_specs.into_iter().zip(panel_areas.iter().copied())
    {
        render_timing_section_panel(frame, panel_area, state, section, title, color);
    }
}

fn render_timing_section_panel(
    frame: &mut Frame,
    area: Rect,
    state: &ProgressState,
    section: MatrixSection,
    title: &'static str,
    title_color: Color,
) {
    let block = panel_block(title, title_color);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 3 {
        return;
    }

    let case_count = state.row_keys.len();
    let cell_width = transposed_matrix_cell_width(frame.area().width as usize, 3) as u16;
    let column_spacing = 1usize;
    let row_label_detail = matrix_label_detail(inner.width as usize, case_count);
    let row_label_width = matrix_row_label_width(
        inner.width as usize,
        case_count,
        cell_width as usize,
        column_spacing,
        &state.presets,
        row_label_detail,
        section,
    ) as u16;

    let sections = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(inner);

    let mut widths = vec![Constraint::Length(row_label_width)];
    widths.extend((0..case_count).map(|_| Constraint::Length(cell_width)));

    frame.render_widget(
        Paragraph::new(matrix_problem_group_header(
            &state.row_keys,
            row_label_width as usize,
            cell_width as usize,
            column_spacing,
        ))
        .style(
            Style::default()
                .fg(problem_category_color())
                .add_modifier(Modifier::BOLD),
        ),
        sections[0],
    );

    let mut header = vec![ratatui::widgets::Cell::from("")];
    for &(_, transcription) in &state.row_keys {
        header.push(ratatui::widgets::Cell::from(Span::styled(
            short_transcription_label(transcription),
            Style::default()
                .fg(transcription_category_color())
                .add_modifier(Modifier::BOLD),
        )));
    }

    let table_rows = section_matrix_rows(section)
        .iter()
        .copied()
        .flat_map(|stage| {
            let stage_header = std::iter::once(
                Row::new(
                    std::iter::once(ratatui::widgets::Cell::from(
                        matrix_row_display_name_with_detail(
                            MatrixRow::Time(stage),
                            row_label_detail,
                        ),
                    ))
                    .chain((0..case_count).map(|_| ratatui::widgets::Cell::from("")))
                    .collect::<Vec<_>>(),
                )
                .style(
                    Style::default()
                        .fg(Color::Gray)
                        .add_modifier(Modifier::BOLD),
                ),
            );
            let preset_rows = state.presets.iter().copied().map(move |preset| {
                let mut cells = vec![ratatui::widgets::Cell::from(Span::styled(
                    format!(
                        "  {}",
                        preset_display_name_with_detail(preset, row_label_detail)
                    ),
                    Style::default().fg(strategy_category_color()),
                ))];
                for &(problem_id, transcription) in &state.row_keys {
                    let case = OcpBenchmarkCase {
                        problem_id,
                        transcription,
                        preset,
                    };
                    let cell = state
                        .case_cells
                        .iter()
                        .find(|entry| entry.case == case)
                        .expect("matrix case should exist");
                    let (text, style) = timing_cell_widget(stage, cell, state);
                    cells.push(ratatui::widgets::Cell::from(text).style(style));
                }
                Row::new(cells)
            });
            stage_header.chain(preset_rows)
        });

    let table = Table::new(table_rows, widths)
        .column_spacing(column_spacing as u16)
        .header(
            Row::new(header).style(
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::BOLD),
            ),
        );
    frame.render_widget(table, sections[1]);
}

fn render_bottom_summary_widgets(frame: &mut Frame, area: Rect, state: &ProgressState) {
    if area.width >= 230 {
        let sections = Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([
                Constraint::Percentage(44),
                Constraint::Percentage(18),
                Constraint::Percentage(38),
            ])
            .split(area);
        render_size_summary_widget(frame, sections[0], state);
        render_nnz_summary_widget(frame, sections[1], state);
        render_best_summary_widget(frame, sections[2], state);
    } else if area.width >= 180 {
        let sections = Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Percentage(20),
                Constraint::Percentage(40),
            ])
            .split(area);
        render_size_summary_widget(frame, sections[0], state);
        render_nnz_summary_widget(frame, sections[1], state);
        render_best_summary_widget(frame, sections[2], state);
    } else if area.width >= 120 {
        let sections = Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
            .split(area);
        let left_sections = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(sections[0]);
        render_size_summary_widget(frame, left_sections[0], state);
        render_nnz_summary_widget(frame, left_sections[1], state);
        render_best_summary_widget(frame, sections[1], state);
    } else {
        let size_height = area.height.saturating_sub(12).max(5);
        let nnz_height = area.height.saturating_sub(size_height + 6).max(5);
        let sections = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([
                Constraint::Length(size_height),
                Constraint::Length(nnz_height),
                Constraint::Min(4),
            ])
            .split(area);
        render_size_summary_widget(frame, sections[0], state);
        render_nnz_summary_widget(frame, sections[1], state);
        render_best_summary_widget(frame, sections[2], state);
    }
}

fn strategy_category_color() -> Color {
    Color::LightBlue
}

fn problem_category_color() -> Color {
    Color::LightCyan
}

fn transcription_category_color() -> Color {
    Color::LightMagenta
}

fn preset_display_name_with_detail(
    preset: OcpBenchmarkPreset,
    detail: LabelDetail,
) -> &'static str {
    match (preset, detail) {
        (OcpBenchmarkPreset::Baseline, _) => "Baseline",
        (OcpBenchmarkPreset::InlineAll, LabelDetail::Short) => "Inline",
        (OcpBenchmarkPreset::InlineAll, _) => "Inline All",
        (OcpBenchmarkPreset::FunctionInlineAtCall, LabelDetail::Short) => "Call",
        (OcpBenchmarkPreset::FunctionInlineAtCall, LabelDetail::Medium) => "At Call",
        (OcpBenchmarkPreset::FunctionInlineAtCall, LabelDetail::Full) => "Inline At Call",
        (OcpBenchmarkPreset::FunctionInlineAtLowering, LabelDetail::Short) => "Lower",
        (OcpBenchmarkPreset::FunctionInlineAtLowering, LabelDetail::Medium) => "At Lowering",
        (OcpBenchmarkPreset::FunctionInlineAtLowering, LabelDetail::Full) => "Inline At Lowering",
        (OcpBenchmarkPreset::FunctionInlineInLlvm, LabelDetail::Short) => "LLVM",
        (OcpBenchmarkPreset::FunctionInlineInLlvm, LabelDetail::Medium) => "In LLVM",
        (OcpBenchmarkPreset::FunctionInlineInLlvm, LabelDetail::Full) => "Inline In LLVM",
        (OcpBenchmarkPreset::FunctionNoInlineLlvm, LabelDetail::Short) => "NoInline",
        (OcpBenchmarkPreset::FunctionNoInlineLlvm, LabelDetail::Medium) => "No Inline",
        (OcpBenchmarkPreset::FunctionNoInlineLlvm, LabelDetail::Full) => "No Inline LLVM",
        (OcpBenchmarkPreset::BaselineWithMsIntegrator, LabelDetail::Short) => "MS Int",
        (OcpBenchmarkPreset::BaselineWithMsIntegrator, _) => "MS Integrator",
    }
}

fn preset_display_name(preset: OcpBenchmarkPreset) -> &'static str {
    preset_display_name_with_detail(preset, LabelDetail::Short)
}

fn matrix_row_display_name_with_detail(row: MatrixRow, detail: LabelDetail) -> &'static str {
    match (row, detail) {
        (MatrixRow::Time(MatrixStage::Build), LabelDetail::Short) => "Bld",
        (MatrixRow::Time(MatrixStage::SymbolicGradient), LabelDetail::Short) => "SGr",
        (MatrixRow::Time(MatrixStage::EqualityJacobianBuild), LabelDetail::Short) => "EqJ",
        (MatrixRow::Time(MatrixStage::InequalityJacobianBuild), LabelDetail::Short) => "InJ",
        (MatrixRow::Time(MatrixStage::LagrangianAssembly), LabelDetail::Short) => "Lag",
        (MatrixRow::Time(MatrixStage::HessianGeneration), LabelDetail::Short) => "SHe",
        (MatrixRow::Time(MatrixStage::NlpJit), LabelDetail::Short) => "NJT",
        (MatrixRow::Time(MatrixStage::XdotHelperJit), LabelDetail::Short) => "Xdt",
        (MatrixRow::Time(MatrixStage::MultipleShootingArcHelperJit), LabelDetail::Short) => "Arc",
        (MatrixRow::Time(MatrixStage::Objective), LabelDetail::Short) => "EOb",
        (MatrixRow::Time(MatrixStage::Gradient), LabelDetail::Short) => "EGr",
        (MatrixRow::Time(MatrixStage::Jacobian), LabelDetail::Short) => "EJa",
        (MatrixRow::Time(MatrixStage::Hessian), LabelDetail::Short) => "EHe",
        (MatrixRow::Time(MatrixStage::Build), LabelDetail::Medium) => "Build",
        (MatrixRow::Time(MatrixStage::SymbolicGradient), LabelDetail::Medium) => "Sym Grad",
        (MatrixRow::Time(MatrixStage::EqualityJacobianBuild), LabelDetail::Medium) => "Eq Jac",
        (MatrixRow::Time(MatrixStage::InequalityJacobianBuild), LabelDetail::Medium) => "In Jac",
        (MatrixRow::Time(MatrixStage::LagrangianAssembly), LabelDetail::Medium) => "Lag Asm",
        (MatrixRow::Time(MatrixStage::HessianGeneration), LabelDetail::Medium) => "Sym Hess",
        (MatrixRow::Time(MatrixStage::NlpJit), LabelDetail::Medium) => "NLP JIT",
        (MatrixRow::Time(MatrixStage::XdotHelperJit), LabelDetail::Medium) => "Xdot JIT",
        (MatrixRow::Time(MatrixStage::MultipleShootingArcHelperJit), LabelDetail::Medium) => {
            "Arc JIT"
        }
        (MatrixRow::Time(MatrixStage::Objective), LabelDetail::Medium) => "Eval Obj",
        (MatrixRow::Time(MatrixStage::Gradient), LabelDetail::Medium) => "Eval Grad",
        (MatrixRow::Time(MatrixStage::Jacobian), LabelDetail::Medium) => "Eval Jac",
        (MatrixRow::Time(MatrixStage::Hessian), LabelDetail::Medium) => "Eval Hess",
        (MatrixRow::Time(MatrixStage::Build), LabelDetail::Full) => "Build Problem",
        (MatrixRow::Time(MatrixStage::SymbolicGradient), LabelDetail::Full) => "Objective Gradient",
        (MatrixRow::Time(MatrixStage::EqualityJacobianBuild), LabelDetail::Full) => {
            "Equality Jacobian"
        }
        (MatrixRow::Time(MatrixStage::InequalityJacobianBuild), LabelDetail::Full) => {
            "Inequality Jacobian"
        }
        (MatrixRow::Time(MatrixStage::LagrangianAssembly), LabelDetail::Full) => {
            "Lagrangian Assembly"
        }
        (MatrixRow::Time(MatrixStage::HessianGeneration), LabelDetail::Full) => {
            "Hessian Generation"
        }
        (MatrixRow::Time(MatrixStage::NlpJit), LabelDetail::Full) => "NLP Kernel JIT",
        (MatrixRow::Time(MatrixStage::XdotHelperJit), LabelDetail::Full) => "Xdot Helper JIT",
        (MatrixRow::Time(MatrixStage::MultipleShootingArcHelperJit), LabelDetail::Full) => {
            "RK4 Arc Helper JIT"
        }
        (MatrixRow::Time(MatrixStage::Objective), LabelDetail::Full) => "Objective Eval",
        (MatrixRow::Time(MatrixStage::Gradient), LabelDetail::Full) => "Gradient Eval",
        (MatrixRow::Time(MatrixStage::Jacobian), LabelDetail::Full) => "Jacobian Eval",
        (MatrixRow::Time(MatrixStage::Hessian), LabelDetail::Full) => "Hessian Eval",
    }
}

fn render_best_summary_widget(frame: &mut Frame, area: Rect, state: &ProgressState) {
    let block = panel_block("Best By Goal", Color::Green);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 3 || inner.width < 20 {
        return;
    }

    let panels = if inner.width >= 110 {
        Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(inner)
    } else {
        Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(inner)
    };
    render_best_summary_transcription_panel(
        frame,
        panels[0],
        state,
        TranscriptionMethod::MultipleShooting,
        "MS",
    );
    render_best_summary_transcription_panel(
        frame,
        panels[1],
        state,
        TranscriptionMethod::DirectCollocation,
        "DC",
    );
}

fn render_best_summary_transcription_panel(
    frame: &mut Frame,
    area: Rect,
    state: &ProgressState,
    transcription: TranscriptionMethod,
    title: &'static str,
) {
    let block = panel_block(title, transcription_category_color());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 3 {
        return;
    }

    let table = Table::new(
        best_summary_rows(state, transcription),
        [
            Constraint::Length(12),
            Constraint::Length(20),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(9),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new(vec![
            "Goal",
            "Case",
            "Symbolic",
            "JIT",
            "Eval100",
            "Overall100",
        ])
        .style(
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::BOLD),
        ),
    );
    frame.render_widget(table, inner);
}

fn best_summary_rows(
    state: &ProgressState,
    transcription: TranscriptionMethod,
) -> Vec<Row<'static>> {
    let metrics = [
        WinnerMetric::Symbolic,
        WinnerMetric::Jit,
        WinnerMetric::Eval100,
        WinnerMetric::Overall100,
    ];
    metrics
        .into_iter()
        .map(|metric| {
            if let Some(cell) = best_case_for_metric(state, transcription, metric) {
                let symbolic = symbolic_total_for_cell(cell);
                let jit = jit_total_for_cell(cell);
                let eval = eval100_total_for_cell(cell);
                let overall = overall100_total_for_cell(cell);
                Row::new(vec![
                    ratatui::widgets::Cell::from(winner_metric_label(metric)),
                    ratatui::widgets::Cell::from(winner_case_label(cell)),
                    ratatui::widgets::Cell::from(compact_optional_cell_seconds(symbolic)).style(
                        winner_metric_style(state, transcription, WinnerMetric::Symbolic, symbolic),
                    ),
                    ratatui::widgets::Cell::from(compact_optional_cell_seconds(jit)).style(
                        winner_metric_style(state, transcription, WinnerMetric::Jit, jit),
                    ),
                    ratatui::widgets::Cell::from(compact_optional_cell_seconds(eval)).style(
                        winner_metric_style(state, transcription, WinnerMetric::Eval100, eval),
                    ),
                    ratatui::widgets::Cell::from(compact_optional_cell_seconds(overall)).style(
                        winner_metric_style(
                            state,
                            transcription,
                            WinnerMetric::Overall100,
                            overall,
                        ),
                    ),
                ])
            } else {
                Row::new(vec![
                    ratatui::widgets::Cell::from(winner_metric_label(metric)),
                    ratatui::widgets::Cell::from("pending"),
                    ratatui::widgets::Cell::from("--"),
                    ratatui::widgets::Cell::from("--"),
                    ratatui::widgets::Cell::from("--"),
                    ratatui::widgets::Cell::from("--"),
                ])
            }
        })
        .collect()
}

fn render_nnz_summary_widget(frame: &mut Frame, area: Rect, state: &ProgressState) {
    let has_mismatch = state.row_keys.iter().any(|&(problem_id, transcription)| {
        nnz_case_has_mismatch(state, problem_id, transcription)
    });
    let block = panel_block(
        if has_mismatch {
            "NNZ Consistency Errors"
        } else {
            "NNZ Consistency"
        },
        if has_mismatch {
            Color::Red
        } else {
            Color::Yellow
        },
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 3 {
        return;
    }

    let table = Table::new(
        nnz_summary_rows(state),
        [
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Min(28),
        ],
    )
    .header(
        Row::new(vec!["Case", "Grad", "Jac", "Hess", "Details"]).style(
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::BOLD),
        ),
    );
    frame.render_widget(table, inner);
}

fn render_size_summary_widget(frame: &mut Frame, area: Rect, state: &ProgressState) {
    let has_large_spread = state.row_keys.iter().any(|&(problem_id, transcription)| {
        size_case_has_large_spread(state, problem_id, transcription)
    });
    let block = panel_block(
        "Pre-JIT Size",
        if has_large_spread {
            Color::Yellow
        } else {
            Color::LightBlue
        },
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 3 {
        return;
    }

    let table = Table::new(
        size_summary_rows(state),
        [
            Constraint::Length(10),
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Min(36),
        ],
    )
    .header(
        Row::new(vec!["Case", "Root Inst", "Total Inst", "Details"]).style(
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::BOLD),
        ),
    );
    frame.render_widget(table, inner);
}

fn size_summary_rows(state: &ProgressState) -> Vec<Row<'static>> {
    state
        .row_keys
        .iter()
        .map(|&(problem_id, transcription)| {
            let root = size_summary_metric_cell(
                state,
                problem_id,
                transcription,
                SizeMetric::RootInstructions,
            );
            let total = size_summary_metric_cell(
                state,
                problem_id,
                transcription,
                SizeMetric::TotalInstructions,
            );
            let details = size_summary_detail_cell(state, problem_id, transcription);
            Row::new(vec![
                ratatui::widgets::Cell::from(nnz_case_label(problem_id, transcription)),
                ratatui::widgets::Cell::from(root.0).style(root.1),
                ratatui::widgets::Cell::from(total.0).style(total.1),
                ratatui::widgets::Cell::from(details.0).style(details.1),
            ])
        })
        .collect()
}

fn size_summary_metric_cell(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: SizeMetric,
) -> (String, Style) {
    let values = size_metric_distinct_values(state, problem_id, transcription, metric);
    let complete = size_metric_complete(state, problem_id, transcription, metric);
    let style = size_metric_style(state, problem_id, transcription, metric);
    match values.as_slice() {
        [] => ("--".to_string(), Style::default().fg(Color::DarkGray)),
        [value] => (
            compact_count(*value),
            if complete {
                style.add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            },
        ),
        values => (summarize_size_spread(values), style),
    }
}

fn size_summary_detail_cell(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
) -> (String, Style) {
    let mut details = Vec::new();
    for (metric, label) in [
        (SizeMetric::RootInstructions, "root"),
        (SizeMetric::TotalInstructions, "total"),
    ] {
        let Some((min_value, min_presets, max_value, max_presets)) =
            size_metric_min_max_offenders(state, problem_id, transcription, metric)
        else {
            continue;
        };
        let values = size_metric_distinct_values(state, problem_id, transcription, metric);
        if values.len() > 1 {
            details.push(format!(
                "{} {}:{} -> {}:{}",
                label,
                compact_count(min_value),
                summarize_offending_presets(state, &min_presets),
                compact_count(max_value),
                summarize_offending_presets(state, &max_presets)
            ));
        }
    }

    if !details.is_empty() {
        let has_large_spread = size_case_has_large_spread(state, problem_id, transcription);
        return (
            details.join(" | "),
            if has_large_spread {
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            },
        );
    }

    let all_complete = [SizeMetric::RootInstructions, SizeMetric::TotalInstructions]
        .into_iter()
        .all(|metric| size_metric_complete(state, problem_id, transcription, metric));
    if all_complete {
        (
            "ok".to_string(),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        ("pending".to_string(), Style::default().fg(Color::DarkGray))
    }
}

fn nnz_summary_rows(state: &ProgressState) -> Vec<Row<'static>> {
    state
        .row_keys
        .iter()
        .map(|&(problem_id, transcription)| {
            let grad =
                nnz_summary_metric_cell(state, problem_id, transcription, NnzMetric::Gradient);
            let jac =
                nnz_summary_metric_cell(state, problem_id, transcription, NnzMetric::Jacobian);
            let hess =
                nnz_summary_metric_cell(state, problem_id, transcription, NnzMetric::Hessian);
            let status = nnz_summary_detail_cell(state, problem_id, transcription);
            Row::new(vec![
                ratatui::widgets::Cell::from(nnz_case_label(problem_id, transcription)),
                ratatui::widgets::Cell::from(grad.0).style(grad.1),
                ratatui::widgets::Cell::from(jac.0).style(jac.1),
                ratatui::widgets::Cell::from(hess.0).style(hess.1),
                ratatui::widgets::Cell::from(status.0).style(status.1),
            ])
        })
        .collect()
}

fn nnz_case_label(problem_id: ProblemId, transcription: TranscriptionMethod) -> String {
    format!(
        "{}/{}",
        short_problem_label(problem_id),
        short_transcription_label(transcription)
    )
}

fn nnz_summary_metric_cell(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: NnzMetric,
) -> (String, Style) {
    let values = nnz_metric_distinct_values(state, problem_id, transcription, metric);
    let complete = nnz_metric_complete(state, problem_id, transcription, metric);
    match values.as_slice() {
        [] => ("--".to_string(), Style::default().fg(Color::DarkGray)),
        [value] => (
            compact_nnz(*value),
            if complete {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            },
        ),
        values => (
            summarize_distinct_nnz(values),
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
    }
}

fn nnz_summary_detail_cell(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
) -> (String, Style) {
    let mismatch_details = [
        (NnzMetric::Gradient, "grad"),
        (NnzMetric::Jacobian, "jac"),
        (NnzMetric::Hessian, "hess"),
    ]
    .into_iter()
    .filter_map(|(metric, label)| {
        let offenders = nnz_metric_offender_groups(state, problem_id, transcription, metric);
        (!offenders.is_empty()).then(|| {
            let groups = offenders
                .into_iter()
                .map(|(value, presets)| {
                    format!(
                        "{} {}:{}",
                        label,
                        compact_nnz(value),
                        summarize_offending_presets(state, &presets)
                    )
                })
                .collect::<Vec<_>>();
            groups.join("; ")
        })
    })
    .collect::<Vec<_>>();
    if !mismatch_details.is_empty() {
        return (
            mismatch_details.join(" | "),
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        );
    }

    let all_complete = [NnzMetric::Gradient, NnzMetric::Jacobian, NnzMetric::Hessian]
        .into_iter()
        .all(|metric| nnz_metric_complete(state, problem_id, transcription, metric));
    if all_complete {
        (
            "ok".to_string(),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        ("pending".to_string(), Style::default().fg(Color::DarkGray))
    }
}

fn nnz_metric_offender_groups(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: NnzMetric,
) -> Vec<(usize, Vec<OcpBenchmarkPreset>)> {
    let values = state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id && cell.case.transcription == transcription
        })
        .filter_map(|cell| nnz_metric_value(cell, metric).map(|value| (cell.case.preset, value)))
        .collect::<Vec<_>>();
    let Some(min_value) = values.iter().map(|(_, value)| *value).min() else {
        return Vec::new();
    };

    let mut offenders = BTreeMap::<usize, Vec<OcpBenchmarkPreset>>::new();
    for (preset, value) in values {
        if value != min_value {
            offenders.entry(value).or_default().push(preset);
        }
    }
    offenders.into_iter().collect()
}

fn size_metric_distinct_values(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: SizeMetric,
) -> Vec<usize> {
    let mut values = state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id
                && cell.case.transcription == transcription
                && preset_applies_to_transcription(cell.case.preset, transcription)
        })
        .filter_map(|cell| size_metric_value(cell, metric))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn size_metric_complete(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: SizeMetric,
) -> bool {
    let applicable = state
        .presets
        .iter()
        .copied()
        .filter(|preset| preset_applies_to_transcription(*preset, transcription))
        .count();
    state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id
                && cell.case.transcription == transcription
                && preset_applies_to_transcription(cell.case.preset, transcription)
        })
        .filter(|cell| size_metric_value(cell, metric).is_some())
        .count()
        == applicable
}

fn size_metric_min_max_offenders(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: SizeMetric,
) -> Option<(usize, Vec<OcpBenchmarkPreset>, usize, Vec<OcpBenchmarkPreset>)> {
    let values = state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id
                && cell.case.transcription == transcription
                && preset_applies_to_transcription(cell.case.preset, transcription)
        })
        .filter_map(|cell| size_metric_value(cell, metric).map(|value| (cell.case.preset, value)))
        .collect::<Vec<_>>();
    let min_value = values.iter().map(|(_, value)| *value).min()?;
    let max_value = values.iter().map(|(_, value)| *value).max()?;
    let min_offenders = values
        .iter()
        .filter_map(|(preset, value)| (*value == min_value).then_some(*preset))
        .collect::<Vec<_>>();
    let max_offenders = values
        .into_iter()
        .filter_map(|(preset, value)| (value == max_value).then_some(preset))
        .collect::<Vec<_>>();
    Some((min_value, min_offenders, max_value, max_offenders))
}

fn size_metric_style(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: SizeMetric,
) -> Style {
    let values = size_metric_distinct_values(state, problem_id, transcription, metric);
    let Some(min_value) = values.first().copied() else {
        return Style::default().fg(Color::DarkGray);
    };
    let max_value = *values.last().expect("non-empty values has last");
    if min_value == max_value {
        return Style::default().fg(Color::White);
    }
    if max_value >= min_value.saturating_mul(10) {
        Style::default().fg(Color::White).bg(Color::Red)
    } else {
        Style::default().fg(Color::Yellow)
    }
}

fn size_case_has_large_spread(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
) -> bool {
    [SizeMetric::RootInstructions, SizeMetric::TotalInstructions]
        .into_iter()
        .any(|metric| {
            let values = size_metric_distinct_values(state, problem_id, transcription, metric);
            match (values.first(), values.last()) {
                (Some(min), Some(max)) => *max >= min.saturating_mul(10) && max != min,
                _ => false,
            }
        })
}

fn summarize_offending_presets(state: &ProgressState, presets: &[OcpBenchmarkPreset]) -> String {
    let mut labels = state
        .presets
        .iter()
        .copied()
        .filter(|preset| presets.contains(preset))
        .map(preset_display_name)
        .collect::<Vec<_>>();
    if labels.is_empty() {
        labels = presets.iter().copied().map(preset_display_name).collect();
    }
    labels.join(", ")
}

fn nnz_metric_distinct_values(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: NnzMetric,
) -> Vec<usize> {
    let mut values = state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id && cell.case.transcription == transcription
        })
        .filter_map(|cell| nnz_metric_value(cell, metric))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn nnz_metric_complete(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    metric: NnzMetric,
) -> bool {
    state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id && cell.case.transcription == transcription
        })
        .filter(|cell| nnz_metric_value(cell, metric).is_some())
        .count()
        == state.presets.len()
}

fn nnz_case_has_mismatch(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
) -> bool {
    [NnzMetric::Gradient, NnzMetric::Jacobian, NnzMetric::Hessian]
        .into_iter()
        .any(|metric| {
            nnz_metric_distinct_values(state, problem_id, transcription, metric).len() > 1
        })
}

fn summarize_distinct_nnz(values: &[usize]) -> String {
    match values {
        [] => "--".to_string(),
        [value] => compact_nnz(*value),
        [lhs, rhs] => format!("{}, {}", compact_nnz(*lhs), compact_nnz(*rhs)),
        values => format!(
            "{}..{} ({})",
            compact_nnz(*values.first().unwrap()),
            compact_nnz(*values.last().unwrap()),
            values.len()
        ),
    }
}

fn summarize_size_spread(values: &[usize]) -> String {
    match values {
        [] => "--".to_string(),
        [value] => compact_count(*value),
        values => {
            let min = *values.first().unwrap();
            let max = *values.last().unwrap();
            format!(
                "{}->{} ({})",
                compact_count(min),
                compact_count(max),
                format_ratio(max, min)
            )
        }
    }
}

fn winner_metric_label(metric: WinnerMetric) -> &'static str {
    match metric {
        WinnerMetric::Symbolic => "Best Symbolic",
        WinnerMetric::Jit => "Best JIT",
        WinnerMetric::Eval100 => "Best Eval",
        WinnerMetric::Overall100 => "Best Overall",
    }
}

fn winner_case_label(cell: &CaseCell) -> String {
    format!(
        "{}/{}",
        short_problem_label(cell.case.problem_id),
        preset_display_name(cell.case.preset),
    )
}

fn preset_applies_to_transcription(
    preset: OcpBenchmarkPreset,
    transcription: TranscriptionMethod,
) -> bool {
    match preset {
        OcpBenchmarkPreset::BaselineWithMsIntegrator => {
            matches!(transcription, TranscriptionMethod::MultipleShooting)
        }
        _ => true,
    }
}

fn best_case_for_metric(
    state: &ProgressState,
    transcription: TranscriptionMethod,
    metric: WinnerMetric,
) -> Option<&CaseCell> {
    state
        .case_cells
        .iter()
        .filter(|cell| cell.case.transcription == transcription)
        .filter(|cell| preset_applies_to_transcription(cell.case.preset, transcription))
        .filter_map(|cell| winner_metric_value(cell, metric).map(|value| (cell, value)))
        .min_by(|lhs, rhs| {
            lhs.1
                .partial_cmp(&rhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(cell, _)| cell)
}

fn winner_metric_style(
    state: &ProgressState,
    transcription: TranscriptionMethod,
    metric: WinnerMetric,
    value: Option<f64>,
) -> Style {
    let Some(value) = value else {
        return Style::default().fg(Color::DarkGray);
    };
    let values = state
        .case_cells
        .iter()
        .filter(|cell| cell.case.transcription == transcription)
        .filter(|cell| preset_applies_to_transcription(cell.case.preset, transcription))
        .filter_map(|cell| winner_metric_value(cell, metric))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Style::default().fg(Color::DarkGray);
    }
    let Some(best) = values
        .iter()
        .copied()
        .min_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    };
    let Some(worst) = values
        .iter()
        .copied()
        .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    };
    let spread = worst - best;
    if approximately_equal(best, worst) || spread <= f64::EPSILON {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    }
    let normalized = ((value - best) / spread).clamp(0.0, 1.0);
    Style::default()
        .fg(heatmap_color(normalized))
        .add_modifier(Modifier::BOLD)
}

fn winner_metric_value(cell: &CaseCell, metric: WinnerMetric) -> Option<f64> {
    match metric {
        WinnerMetric::Symbolic => symbolic_total_for_cell(cell),
        WinnerMetric::Jit => jit_total_for_cell(cell),
        WinnerMetric::Eval100 => eval100_total_for_cell(cell),
        WinnerMetric::Overall100 => overall100_total_for_cell(cell),
    }
}

fn symbolic_total_for_cell(cell: &CaseCell) -> Option<f64> {
    sum_f64_options([
        cell.symbolic_construction_s,
        cell.compile_objective_gradient_s,
        cell.compile_equality_jacobian_s,
        cell.compile_inequality_jacobian_s,
        cell.lagrangian_assembly_s,
        cell.compile_hessian_generation_s,
    ])
}

fn jit_total_for_cell(cell: &CaseCell) -> Option<f64> {
    sum_f64_options([
        cell.compile_nlp_jit_s,
        cell.xdot_helper_compile_s,
        cell.multiple_shooting_arc_helper_compile_s,
    ])
}

fn eval100_total_for_cell(cell: &CaseCell) -> Option<f64> {
    let objective = cell.objective_avg_s?;
    let gradient = cell.gradient_avg_s?;
    let jacobian = cell.jacobian_avg_s?;
    let hessian = cell.hessian_avg_s?;
    Some(objective + gradient + 100.0 * jacobian + 100.0 * hessian)
}

fn overall100_total_for_cell(cell: &CaseCell) -> Option<f64> {
    Some(symbolic_total_for_cell(cell)? + jit_total_for_cell(cell)? + eval100_total_for_cell(cell)?)
}

fn transposed_matrix_cell_width(terminal_width: usize, panel_count: usize) -> usize {
    let width_per_panel = terminal_width / panel_count.max(1);
    if width_per_panel < 56 {
        4
    } else if width_per_panel < 72 {
        5
    } else {
        6
    }
}

fn matrix_label_detail(panel_width: usize, case_count: usize) -> LabelDetail {
    let cell_width = transposed_matrix_cell_width(panel_width, 1);
    let base_available = panel_width.saturating_sub(case_count * (cell_width + 1));
    if base_available >= 13 {
        LabelDetail::Full
    } else if base_available >= 10 {
        LabelDetail::Medium
    } else {
        LabelDetail::Short
    }
}

fn matrix_row_label_width(
    panel_width: usize,
    case_count: usize,
    cell_width: usize,
    column_spacing: usize,
    presets: &[OcpBenchmarkPreset],
    detail: LabelDetail,
    section: MatrixSection,
) -> usize {
    let available = panel_width
        .saturating_sub(case_count * cell_width)
        .saturating_sub(case_count * column_spacing);
    let longest_preset = presets
        .iter()
        .map(|&preset| {
            preset_display_name_with_detail(preset, detail)
                .chars()
                .count()
        })
        .max()
        .unwrap_or(7);
    let longest_stage = section_matrix_rows(section)
        .iter()
        .map(|&stage| {
            matrix_row_display_name_with_detail(MatrixRow::Time(stage), detail)
                .chars()
                .count()
        })
        .max()
        .unwrap_or(5);
    available.clamp(7, longest_preset.max(longest_stage))
}

fn section_matrix_rows(section: MatrixSection) -> &'static [MatrixStage] {
    match section {
        MatrixSection::Symbolic => &[
            MatrixStage::Build,
            MatrixStage::SymbolicGradient,
            MatrixStage::EqualityJacobianBuild,
            MatrixStage::InequalityJacobianBuild,
            MatrixStage::LagrangianAssembly,
            MatrixStage::HessianGeneration,
        ],
        MatrixSection::Jit => &[
            MatrixStage::NlpJit,
            MatrixStage::XdotHelperJit,
            MatrixStage::MultipleShootingArcHelperJit,
        ],
        MatrixSection::Runtime => &[
            MatrixStage::Objective,
            MatrixStage::Gradient,
            MatrixStage::Jacobian,
            MatrixStage::Hessian,
        ],
    }
}

fn matrix_problem_group_header(
    row_keys: &[(ProblemId, TranscriptionMethod)],
    row_label_width: usize,
    cell_width: usize,
    column_spacing: usize,
) -> String {
    let groups = matrix_problem_groups(row_keys);
    let mut line = " ".repeat(row_label_width + column_spacing);
    for (group_index, (problem_id, span)) in groups.iter().enumerate() {
        let width = span * cell_width + span.saturating_sub(1) * column_spacing;
        line.push_str(&matrix_problem_group_label(
            matrix_problem_title(*problem_id),
            width,
        ));
        if group_index + 1 != groups.len() {
            line.push_str(&" ".repeat(column_spacing));
        }
    }
    line
}

fn matrix_problem_group_label(title: &str, width: usize) -> String {
    let title_len = title.chars().count();
    if width <= title_len + 2 {
        return center_or_truncate(title, width);
    }

    let left_fill = (width.saturating_sub(title_len + 2)) / 2;
    let right_fill = width.saturating_sub(title_len + 2 + left_fill);
    format!(
        "{} {} {}",
        "─".repeat(left_fill),
        title,
        "─".repeat(right_fill),
    )
}

fn matrix_problem_groups(row_keys: &[(ProblemId, TranscriptionMethod)]) -> Vec<(ProblemId, usize)> {
    let mut groups = Vec::new();
    for &(problem_id, _) in row_keys {
        match groups.last_mut() {
            Some((last_problem_id, count)) if *last_problem_id == problem_id => {
                *count += 1;
            }
            _ => groups.push((problem_id, 1)),
        }
    }
    groups
}

fn matrix_problem_title(problem_id: ProblemId) -> &'static str {
    match problem_id {
        ProblemId::OptimalDistanceGlider => "Glider",
        ProblemId::LinearSManeuver => "Linear-S",
        ProblemId::SailboatUpwind => "Sailboat",
        ProblemId::CraneTransfer => "Crane",
    }
}

fn center_or_truncate(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let text = if text.chars().count() > width {
        text.chars().take(width).collect::<String>()
    } else {
        text.to_string()
    };
    let text_width = text.chars().count();
    let left = (width.saturating_sub(text_width)) / 2;
    let right = width.saturating_sub(text_width + left);
    format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
}

fn stage_cell_widget(stage: &StageCell, is_best: bool) -> (String, Style) {
    match stage {
        StageCell::Pending => ("..".to_string(), Style::default().fg(Color::DarkGray)),
        StageCell::Running(started_at) => (
            compact_cell_seconds(started_at.elapsed().as_secs_f64()),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        StageCell::Done(Some(value)) => (
            compact_cell_seconds(*value),
            Style::default()
                .fg(if is_best { Color::Green } else { Color::White })
                .add_modifier(Modifier::BOLD),
        ),
        StageCell::Done(None) => (
            "ok".to_string(),
            Style::default()
                .fg(if is_best { Color::Green } else { Color::White })
                .add_modifier(Modifier::BOLD),
        ),
        StageCell::Skipped => ("--".to_string(), Style::default().fg(Color::DarkGray)),
    }
}

fn timing_cell_widget(
    stage: MatrixStage,
    cell: &CaseCell,
    state: &ProgressState,
) -> (String, Style) {
    match timing_metric_value(cell, stage) {
        Some(value) => (
            compact_cell_seconds(value),
            timing_metric_style(state, cell, stage, value),
        ),
        None => match stage_cell(cell, stage) {
            StageCell::Skipped => ("--".to_string(), Style::default().fg(Color::DarkGray)),
            stage_cell => stage_cell_widget(&stage_cell, false),
        },
    }
}

fn timing_metric_style(
    state: &ProgressState,
    cell: &CaseCell,
    stage: MatrixStage,
    value: f64,
) -> Style {
    let group = matrix_stage_heatmap_group(stage);
    let values = timing_heatmap_values(state, cell.case.problem_id, cell.case.transcription, group);
    if values.len() < 2 {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    }
    let Some(best) = values
        .iter()
        .copied()
        .min_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    };
    let Some(worst) = values
        .iter()
        .copied()
        .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    };
    let spread = worst - best;
    if approximately_equal(best, worst) || spread <= f64::EPSILON {
        return Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    }
    let normalized = ((value - best) / spread).clamp(0.0, 1.0);
    Style::default()
        .fg(heatmap_color(normalized))
        .add_modifier(Modifier::BOLD)
}

fn timing_metric_value(cell: &CaseCell, stage: MatrixStage) -> Option<f64> {
    match stage {
        MatrixStage::Build => cell.symbolic_construction_s,
        MatrixStage::SymbolicGradient => cell.compile_objective_gradient_s,
        MatrixStage::EqualityJacobianBuild => cell.compile_equality_jacobian_s,
        MatrixStage::InequalityJacobianBuild => cell.compile_inequality_jacobian_s,
        MatrixStage::LagrangianAssembly => cell.lagrangian_assembly_s,
        MatrixStage::HessianGeneration => cell.compile_hessian_generation_s,
        MatrixStage::NlpJit => cell.compile_nlp_jit_s,
        MatrixStage::XdotHelperJit => cell.xdot_helper_compile_s,
        MatrixStage::MultipleShootingArcHelperJit => cell.multiple_shooting_arc_helper_compile_s,
        MatrixStage::Objective => cell.objective_avg_s,
        MatrixStage::Gradient => cell.gradient_avg_s,
        MatrixStage::Jacobian => cell.jacobian_avg_s,
        MatrixStage::Hessian => cell.hessian_avg_s,
    }
}

fn nnz_metric_value(cell: &CaseCell, metric: NnzMetric) -> Option<usize> {
    match metric {
        NnzMetric::Gradient => cell.gradient_nnz,
        NnzMetric::Jacobian => cell.jacobian_nnz,
        NnzMetric::Hessian => cell.hessian_nnz,
    }
}

fn size_metric_value(cell: &CaseCell, metric: SizeMetric) -> Option<usize> {
    match metric {
        SizeMetric::RootInstructions => cell.llvm_root_instruction_count,
        SizeMetric::TotalInstructions => cell.llvm_total_instruction_count,
    }
}

fn compact_count(value: usize) -> String {
    compact_nnz(value)
}

fn format_ratio(max: usize, min: usize) -> String {
    if min == 0 {
        return "infx".to_string();
    }
    let ratio = max as f64 / min as f64;
    if ratio >= 100.0 {
        format!("{ratio:.0}x")
    } else if ratio >= 10.0 {
        format!("{ratio:.1}x")
    } else {
        format!("{ratio:.2}x")
    }
}

fn compact_nnz(value: usize) -> String {
    if value >= 10_000 {
        format!("{}k", value / 1_000)
    } else {
        value.to_string()
    }
}

fn sum_f64_options<const N: usize>(values: [Option<f64>; N]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.into_iter().flatten() {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum)
}

fn approximately_equal(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 1.0e-12_f64.max(lhs.abs().max(rhs.abs()) * 1.0e-9)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixHeatmapGroup {
    PreEval,
    Eval,
}

fn matrix_stage_heatmap_group(stage: MatrixStage) -> MatrixHeatmapGroup {
    match stage {
        MatrixStage::Objective
        | MatrixStage::Gradient
        | MatrixStage::Jacobian
        | MatrixStage::Hessian => MatrixHeatmapGroup::Eval,
        MatrixStage::Build
        | MatrixStage::SymbolicGradient
        | MatrixStage::EqualityJacobianBuild
        | MatrixStage::InequalityJacobianBuild
        | MatrixStage::LagrangianAssembly
        | MatrixStage::HessianGeneration
        | MatrixStage::NlpJit
        | MatrixStage::XdotHelperJit
        | MatrixStage::MultipleShootingArcHelperJit => MatrixHeatmapGroup::PreEval,
    }
}

fn timing_heatmap_values(
    state: &ProgressState,
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    group: MatrixHeatmapGroup,
) -> Vec<f64> {
    state
        .case_cells
        .iter()
        .filter(|cell| {
            cell.case.problem_id == problem_id && cell.case.transcription == transcription
        })
        .flat_map(|cell| {
            matrix_rows().iter().filter_map(move |row| match row {
                MatrixRow::Time(stage) if matrix_stage_heatmap_group(*stage) == group => {
                    timing_metric_value(cell, *stage)
                }
                _ => None,
            })
        })
        .collect()
}

fn heatmap_color(normalized: f64) -> Color {
    let t = normalized.clamp(0.0, 1.0);
    if t <= 0.5 {
        interpolate_heatmap_rgb(t * 2.0, (0, 255, 0), (255, 255, 0))
    } else {
        interpolate_heatmap_rgb((t - 0.5) * 2.0, (255, 255, 0), (255, 0, 0))
    }
}

fn interpolate_heatmap_rgb(t: f64, start: (u8, u8, u8), end: (u8, u8, u8)) -> Color {
    let blend = |lhs: u8, rhs: u8| -> u8 {
        let lhs = lhs as f64;
        let rhs = rhs as f64;
        (lhs + (rhs - lhs) * t).round().clamp(0.0, 255.0) as u8
    };
    Color::Rgb(
        blend(start.0, end.0),
        blend(start.1, end.1),
        blend(start.2, end.2),
    )
}

fn buffer_to_ansi_text(buffer: &Buffer) -> String {
    let width = buffer.area.width as usize;
    let height = buffer.area.height as usize;
    let mut lines = Vec::with_capacity(height);
    for row in 0..height {
        let start = row * width;
        let end = start + width;
        let cells = &buffer.content[start..end];
        let last_visible = cells
            .iter()
            .rposition(|cell| cell.symbol() != " ")
            .map(|index| index + 1)
            .unwrap_or(0);
        let mut line = String::new();
        let mut current_style = SnapshotCellStyle::default();
        for cell in &cells[..last_visible] {
            let next_style = SnapshotCellStyle::from_cell(cell);
            if next_style != current_style {
                line.push_str(&ansi_style_sequence(next_style));
                current_style = next_style;
            }
            line.push_str(cell.symbol());
        }
        if current_style != SnapshotCellStyle::default() {
            line.push_str("\x1b[0m");
        }
        lines.push(line);
    }
    while matches!(lines.last(), Some(line) if line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

impl SnapshotCellStyle {
    fn from_cell(cell: &ratatui::buffer::Cell) -> Self {
        Self {
            fg: cell.fg,
            bg: cell.bg,
            modifier: cell.modifier,
        }
    }
}

impl Default for SnapshotCellStyle {
    fn default() -> Self {
        Self {
            fg: Color::Reset,
            bg: Color::Reset,
            modifier: Modifier::empty(),
        }
    }
}

fn ansi_style_sequence(style: SnapshotCellStyle) -> String {
    let mut codes: Vec<String> = vec!["0".to_string()];
    push_modifier_codes(&mut codes, style.modifier);
    push_color_code(&mut codes, style.fg, true);
    push_color_code(&mut codes, style.bg, false);
    format!("\x1b[{}m", codes.join(";"))
}

fn push_modifier_codes(codes: &mut Vec<String>, modifier: Modifier) {
    if modifier.contains(Modifier::BOLD) {
        codes.push("1".to_string());
    }
    if modifier.contains(Modifier::DIM) {
        codes.push("2".to_string());
    }
    if modifier.contains(Modifier::ITALIC) {
        codes.push("3".to_string());
    }
    if modifier.contains(Modifier::UNDERLINED) {
        codes.push("4".to_string());
    }
    if modifier.contains(Modifier::SLOW_BLINK) {
        codes.push("5".to_string());
    }
    if modifier.contains(Modifier::RAPID_BLINK) {
        codes.push("6".to_string());
    }
    if modifier.contains(Modifier::REVERSED) {
        codes.push("7".to_string());
    }
    if modifier.contains(Modifier::HIDDEN) {
        codes.push("8".to_string());
    }
    if modifier.contains(Modifier::CROSSED_OUT) {
        codes.push("9".to_string());
    }
}

fn push_color_code(codes: &mut Vec<String>, color: Color, foreground: bool) {
    match color {
        Color::Reset => {}
        Color::Black => codes.push(if foreground { "30" } else { "40" }.to_string()),
        Color::Red => codes.push(if foreground { "31" } else { "41" }.to_string()),
        Color::Green => codes.push(if foreground { "32" } else { "42" }.to_string()),
        Color::Yellow => codes.push(if foreground { "33" } else { "43" }.to_string()),
        Color::Blue => codes.push(if foreground { "34" } else { "44" }.to_string()),
        Color::Magenta => codes.push(if foreground { "35" } else { "45" }.to_string()),
        Color::Cyan => codes.push(if foreground { "36" } else { "46" }.to_string()),
        Color::Gray => codes.push(if foreground { "37" } else { "47" }.to_string()),
        Color::DarkGray => codes.push(if foreground { "90" } else { "100" }.to_string()),
        Color::LightRed => codes.push(if foreground { "91" } else { "101" }.to_string()),
        Color::LightGreen => codes.push(if foreground { "92" } else { "102" }.to_string()),
        Color::LightYellow => codes.push(if foreground { "93" } else { "103" }.to_string()),
        Color::LightBlue => codes.push(if foreground { "94" } else { "104" }.to_string()),
        Color::LightMagenta => codes.push(if foreground { "95" } else { "105" }.to_string()),
        Color::LightCyan => codes.push(if foreground { "96" } else { "106" }.to_string()),
        Color::White => codes.push(if foreground { "97" } else { "107" }.to_string()),
        Color::Rgb(r, g, b) => {
            codes.push(if foreground { "38" } else { "48" }.to_string());
            codes.push("2".to_string());
            codes.push(r.to_string());
            codes.push(g.to_string());
            codes.push(b.to_string());
        }
        Color::Indexed(index) => {
            codes.push(if foreground { "38" } else { "48" }.to_string());
            codes.push("5".to_string());
            codes.push(index.to_string());
        }
    }
}

fn print_event_line(event: OcpBenchmarkProgress) {
    match event {
        OcpBenchmarkProgress::CaseStarted {
            current,
            total,
            case,
        } => eprintln!(
            "[{current}/{total}] {} | {} | {}",
            case.problem_label(),
            case.transcription_label(),
            case.preset_label(),
        ),
        OcpBenchmarkProgress::CompileProgress { progress, .. } => match progress {
            OcpCompileProgress::SymbolicStage(progress) => eprintln!(
                "  symbolic stage: {} vars={} eq={} ineq={}",
                symbolic_compile_stage_label(progress.stage),
                progress.metadata.stats.variable_count,
                progress.metadata.stats.equality_count,
                progress.metadata.stats.inequality_count,
            ),
            OcpCompileProgress::SymbolicReady(metadata) => eprintln!(
                "  symbolic ready: vars={} eq={} ineq={}",
                metadata.stats.variable_count,
                metadata.stats.equality_count,
                metadata.stats.inequality_count,
            ),
            OcpCompileProgress::HelperCompiled { helper, elapsed } => eprintln!(
                "  helper ready: {} in {}",
                match helper {
                    optimal_control::OcpCompileHelperKind::Xdot => "xdot",
                    optimal_control::OcpCompileHelperKind::MultipleShootingArc => "rk4 arc",
                },
                format_duration(elapsed),
            ),
        },
        OcpBenchmarkProgress::EvalKernelStarted { kernel, .. } => {
            eprintln!("  benchmarking {}", kernel_label(kernel));
        }
        OcpBenchmarkProgress::CaseFinished {
            current,
            total,
            record,
            ..
        } => {
            eprintln!(
                "  finished case [{current}/{total}] compile={} grad={} hess={}",
                format_optional_seconds(record.compile_total_s),
                format_optional_seconds(record.eval.objective_gradient.average_s),
                format_optional_seconds(record.eval.lagrangian_hessian_values.average_s),
            );
        }
    }
}

fn update_case_started(state: &mut ProgressState, case: OcpBenchmarkCase) {
    let now = Instant::now();
    if let Some(cell) = find_case_cell_mut(state, case) {
        cell.symbolic = StageCell::Running(now);
        cell.active_symbolic_stage = Some(MatrixStage::Build);
        cell.jit = StageCell::Pending;
        cell.nlp_jit = StageCell::Pending;
        cell.xdot_helper_jit = StageCell::Pending;
        cell.multiple_shooting_arc_helper_jit = StageCell::Pending;
        cell.objective = StageCell::Pending;
        cell.gradient = StageCell::Pending;
        cell.jacobian = StageCell::Pending;
        cell.hessian = StageCell::Pending;
        cell.equality_count = 0;
        cell.inequality_count = 0;
        cell.compile_total_s = None;
        cell.symbolic_construction_s = None;
        cell.compile_objective_gradient_s = None;
        cell.compile_equality_jacobian_s = None;
        cell.compile_inequality_jacobian_s = None;
        cell.lagrangian_assembly_s = None;
        cell.compile_hessian_generation_s = None;
        cell.compile_nlp_jit_s = None;
        cell.xdot_helper_compile_s = None;
        cell.multiple_shooting_arc_helper_compile_s = None;
        cell.objective_avg_s = None;
        cell.objective_stddev_s = None;
        cell.gradient_avg_s = None;
        cell.gradient_stddev_s = None;
        cell.equality_jacobian_avg_s = None;
        cell.equality_jacobian_stddev_s = None;
        cell.inequality_jacobian_avg_s = None;
        cell.inequality_jacobian_stddev_s = None;
        cell.jacobian_avg_s = None;
        cell.jacobian_stddev_s = None;
        cell.hessian_avg_s = None;
        cell.hessian_stddev_s = None;
        cell.gradient_nnz = None;
        cell.jacobian_nnz = None;
        cell.hessian_nnz = None;
        cell.llvm_root_instruction_count = None;
        cell.llvm_total_instruction_count = None;
        cell.eval_samples = 0;
        cell.sanity = SanityStatus::Ok;
        cell.warnings = 0;
    }
}

fn apply_symbolic_metadata(cell: &mut CaseCell, metadata: &optimization::SymbolicCompileMetadata) {
    cell.equality_count = metadata.stats.equality_count;
    cell.inequality_count = metadata.stats.inequality_count;
    cell.symbolic_construction_s = metadata
        .setup_profile
        .symbolic_construction
        .map(|duration| duration.as_secs_f64());
    cell.compile_objective_gradient_s = metadata
        .setup_profile
        .objective_gradient
        .map(|duration| duration.as_secs_f64());
    cell.compile_equality_jacobian_s = metadata
        .setup_profile
        .equality_jacobian
        .map(|duration| duration.as_secs_f64());
    cell.compile_inequality_jacobian_s = metadata
        .setup_profile
        .inequality_jacobian
        .map(|duration| duration.as_secs_f64());
    cell.lagrangian_assembly_s = metadata
        .setup_profile
        .lagrangian_assembly
        .map(|duration| duration.as_secs_f64());
    cell.compile_hessian_generation_s = metadata
        .setup_profile
        .hessian_generation
        .map(|duration| duration.as_secs_f64());
    cell.gradient_nnz = (metadata.stats.objective_gradient_nnz > 0)
        .then_some(metadata.stats.objective_gradient_nnz);
    let jacobian_nnz =
        metadata.stats.equality_jacobian_nnz + metadata.stats.inequality_jacobian_nnz;
    cell.jacobian_nnz = (jacobian_nnz > 0).then_some(jacobian_nnz);
    cell.hessian_nnz = (metadata.stats.hessian_nnz > 0).then_some(metadata.stats.hessian_nnz);
}

fn update_case_symbolic_stage_progress(
    cell: &mut CaseCell,
    progress: SymbolicCompileStageProgress,
    now: Instant,
) {
    apply_symbolic_metadata(cell, &progress.metadata);
    cell.active_symbolic_stage = next_symbolic_matrix_stage(&progress);
    cell.symbolic = StageCell::Running(now);
}

fn update_case_compile_progress(
    state: &mut ProgressState,
    case: OcpBenchmarkCase,
    progress: OcpCompileProgress,
) {
    let now = Instant::now();
    if let Some(cell) = find_case_cell_mut(state, case) {
        match progress {
            OcpCompileProgress::SymbolicStage(progress) => {
                update_case_symbolic_stage_progress(cell, progress, now);
            }
            OcpCompileProgress::SymbolicReady(metadata) => {
                let creation_s = metadata
                    .timing
                    .function_creation_time
                    .map(|duration| duration.as_secs_f64());
                let derivatives_s = metadata
                    .timing
                    .derivative_generation_time
                    .map(|duration| duration.as_secs_f64());
                let symbolic_total = match (creation_s, derivatives_s) {
                    (Some(lhs), Some(rhs)) => Some(lhs + rhs),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                };
                apply_symbolic_metadata(cell, &metadata);
                cell.active_symbolic_stage = None;
                cell.symbolic = done_cell(symbolic_total);
                cell.jit = StageCell::Running(now);
                cell.nlp_jit = StageCell::Running(now);
            }
            OcpCompileProgress::HelperCompiled { helper, elapsed } => {
                finalize_running_stage(&mut cell.symbolic);
                cell.active_symbolic_stage = None;
                if !matches!(cell.jit, StageCell::Done(_)) {
                    cell.jit = StageCell::Running(now);
                }
                match helper {
                    OcpCompileHelperKind::Xdot => {
                        finalize_running_stage(&mut cell.nlp_jit);
                        cell.xdot_helper_compile_s = Some(elapsed.as_secs_f64());
                        cell.xdot_helper_jit = done_cell(cell.xdot_helper_compile_s);
                        if matches!(case.transcription, TranscriptionMethod::MultipleShooting) {
                            cell.multiple_shooting_arc_helper_jit = StageCell::Running(now);
                        }
                    }
                    OcpCompileHelperKind::MultipleShootingArc => {
                        cell.multiple_shooting_arc_helper_compile_s = Some(elapsed.as_secs_f64());
                        cell.multiple_shooting_arc_helper_jit =
                            done_cell(cell.multiple_shooting_arc_helper_compile_s);
                    }
                }
            }
        }
    }
}

fn update_case_eval_stage(
    state: &mut ProgressState,
    case: OcpBenchmarkCase,
    kernel: NlpEvaluationKernelKind,
) {
    let now = Instant::now();
    if let Some(cell) = find_case_cell_mut(state, case) {
        finalize_running_stage(&mut cell.symbolic);
        cell.active_symbolic_stage = None;
        finalize_running_stage(&mut cell.jit);
        finalize_running_stage(&mut cell.nlp_jit);
        finalize_running_stage(&mut cell.xdot_helper_jit);
        finalize_running_stage(&mut cell.multiple_shooting_arc_helper_jit);
        match kernel {
            NlpEvaluationKernelKind::ObjectiveValue => {
                finalize_running_stage(&mut cell.objective);
                cell.objective = StageCell::Running(now);
            }
            NlpEvaluationKernelKind::ObjectiveGradient => {
                finalize_running_stage(&mut cell.objective);
                finalize_running_stage(&mut cell.gradient);
                cell.gradient = StageCell::Running(now);
            }
            NlpEvaluationKernelKind::EqualityJacobianValues
            | NlpEvaluationKernelKind::InequalityJacobianValues => {
                finalize_running_stage(&mut cell.objective);
                finalize_running_stage(&mut cell.gradient);
                if !matches!(cell.jacobian, StageCell::Running(_)) {
                    cell.jacobian = StageCell::Running(now);
                }
            }
            NlpEvaluationKernelKind::LagrangianHessianValues => {
                finalize_running_stage(&mut cell.objective);
                finalize_running_stage(&mut cell.gradient);
                finalize_running_stage(&mut cell.jacobian);
                finalize_running_stage(&mut cell.hessian);
                cell.hessian = StageCell::Running(now);
            }
        }
    }
}

fn update_case_finished(state: &mut ProgressState, record: &OcpBenchmarkRecord) {
    let finished_case = OcpBenchmarkCase {
        problem_id: record.problem_id,
        transcription: match record.transcription_id.as_str() {
            "multiple_shooting" => TranscriptionMethod::MultipleShooting,
            "direct_collocation" => TranscriptionMethod::DirectCollocation,
            other => panic!("unexpected transcription id {other}"),
        },
        preset: OcpBenchmarkPreset::parse(&record.preset_id)
            .expect("finished preset id should parse"),
    };
    if let Some(cell) = find_case_cell_mut(state, finished_case) {
        cell.symbolic = done_cell(record.symbolic_total_s);
        cell.active_symbolic_stage = None;
        cell.jit = done_cell(record.jit_total_s);
        cell.compile_nlp_jit_s =
            sum_f64_options([record.compile.lowering_s, record.compile.llvm_jit_s]);
        cell.xdot_helper_compile_s = record.helper_compile.xdot_helper_s;
        cell.multiple_shooting_arc_helper_compile_s =
            record.helper_compile.multiple_shooting_arc_helper_s;
        cell.nlp_jit = done_cell(cell.compile_nlp_jit_s);
        cell.xdot_helper_jit = if jit_stage_applicable(cell, MatrixStage::XdotHelperJit) {
            done_cell(cell.xdot_helper_compile_s)
        } else {
            StageCell::Skipped
        };
        cell.multiple_shooting_arc_helper_jit =
            if jit_stage_applicable(cell, MatrixStage::MultipleShootingArcHelperJit) {
                done_cell(cell.multiple_shooting_arc_helper_compile_s)
            } else {
                StageCell::Skipped
            };
        cell.objective = done_cell(record.eval.objective_value.average_s);
        cell.gradient = done_cell(record.eval.objective_gradient.average_s);
        cell.jacobian = match combined_jacobian_average(record) {
            Some(value) => StageCell::Done(Some(value)),
            None => StageCell::Skipped,
        };
        cell.hessian = done_cell(record.eval.lagrangian_hessian_values.average_s);
        cell.equality_count = record.nlp.equality_count;
        cell.inequality_count = record.nlp.inequality_count;
        cell.compile_total_s = record.compile_total_s;
        cell.symbolic_construction_s = record.compile.symbolic_construction_s;
        cell.compile_objective_gradient_s = record.compile.objective_gradient_s;
        cell.compile_equality_jacobian_s = record.compile.equality_jacobian_s;
        cell.compile_inequality_jacobian_s = record.compile.inequality_jacobian_s;
        cell.lagrangian_assembly_s = record.compile.lagrangian_assembly_s;
        cell.compile_hessian_generation_s = record.compile.hessian_generation_s;
        cell.objective_avg_s = record.eval.objective_value.average_s;
        cell.objective_stddev_s = record.eval.objective_value.stddev_s;
        cell.gradient_avg_s = record.eval.objective_gradient.average_s;
        cell.gradient_stddev_s = record.eval.objective_gradient.stddev_s;
        cell.equality_jacobian_avg_s = record
            .eval
            .equality_jacobian_values
            .as_ref()
            .and_then(|summary| summary.average_s);
        cell.equality_jacobian_stddev_s = record
            .eval
            .equality_jacobian_values
            .as_ref()
            .and_then(|summary| summary.stddev_s);
        cell.inequality_jacobian_avg_s = record
            .eval
            .inequality_jacobian_values
            .as_ref()
            .and_then(|summary| summary.average_s);
        cell.inequality_jacobian_stddev_s = record
            .eval
            .inequality_jacobian_values
            .as_ref()
            .and_then(|summary| summary.stddev_s);
        cell.jacobian_avg_s = combined_jacobian_average(record);
        cell.jacobian_stddev_s = combined_jacobian_stddev(record);
        cell.hessian_avg_s = record.eval.lagrangian_hessian_values.average_s;
        cell.hessian_stddev_s = record.eval.lagrangian_hessian_values.stddev_s;
        cell.gradient_nnz = Some(record.nlp.objective_gradient_nnz);
        cell.jacobian_nnz =
            Some(record.nlp.equality_jacobian_nnz + record.nlp.inequality_jacobian_nnz);
        cell.hessian_nnz = Some(record.nlp.hessian_nnz);
        cell.llvm_root_instruction_count = Some(record.compile.llvm_root_instructions_emitted);
        cell.llvm_total_instruction_count = Some(record.compile.llvm_total_instructions_emitted);
        cell.eval_samples = record.eval.objective_value.iterations;
        cell.sanity = benchmark_sanity(record);
        cell.warnings = record.compile.warnings.len();
    }
    state.warning_count = state
        .case_cells
        .iter()
        .map(|cell| cell.warnings)
        .sum::<usize>();
}

fn find_case_cell_mut(state: &mut ProgressState, case: OcpBenchmarkCase) -> Option<&mut CaseCell> {
    state.case_cells.iter_mut().find(|cell| cell.case == case)
}

fn average_metric_value(
    state: &ProgressState,
    project: impl Fn(&CaseCell) -> Option<f64>,
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in state.case_cells.iter().filter_map(project) {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn matrix_rows() -> &'static [MatrixRow] {
    &[
        MatrixRow::Time(MatrixStage::Build),
        MatrixRow::Time(MatrixStage::SymbolicGradient),
        MatrixRow::Time(MatrixStage::EqualityJacobianBuild),
        MatrixRow::Time(MatrixStage::InequalityJacobianBuild),
        MatrixRow::Time(MatrixStage::LagrangianAssembly),
        MatrixRow::Time(MatrixStage::HessianGeneration),
        MatrixRow::Time(MatrixStage::NlpJit),
        MatrixRow::Time(MatrixStage::XdotHelperJit),
        MatrixRow::Time(MatrixStage::MultipleShootingArcHelperJit),
        MatrixRow::Time(MatrixStage::Objective),
        MatrixRow::Time(MatrixStage::Gradient),
        MatrixRow::Time(MatrixStage::Jacobian),
        MatrixRow::Time(MatrixStage::Hessian),
    ]
}

fn done_cell(value: Option<f64>) -> StageCell {
    StageCell::Done(value)
}

fn combined_jacobian_average(record: &OcpBenchmarkRecord) -> Option<f64> {
    let equality = record
        .eval
        .equality_jacobian_values
        .as_ref()
        .and_then(|summary| summary.average_s);
    let inequality = record
        .eval
        .inequality_jacobian_values
        .as_ref()
        .and_then(|summary| summary.average_s);
    match (equality, inequality) {
        (Some(lhs), Some(rhs)) => Some(lhs + rhs),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn combined_jacobian_stddev(record: &OcpBenchmarkRecord) -> Option<f64> {
    let equality = record
        .eval
        .equality_jacobian_values
        .as_ref()
        .and_then(|summary| summary.stddev_s);
    let inequality = record
        .eval
        .inequality_jacobian_values
        .as_ref()
        .and_then(|summary| summary.stddev_s);
    match (equality, inequality) {
        (Some(lhs), Some(rhs)) => Some(lhs + rhs),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn benchmark_sanity(record: &OcpBenchmarkRecord) -> SanityStatus {
    let summaries = [
        Some(&record.eval.objective_gradient),
        record.eval.equality_jacobian_values.as_ref(),
        record.eval.inequality_jacobian_values.as_ref(),
        Some(&record.eval.lagrangian_hessian_values),
    ];
    if !record.eval.benchmark_point.objective_finite
        || summaries
            .iter()
            .flatten()
            .any(|summary| !summary.preflight_finite)
    {
        SanityStatus::NonFinite
    } else if summaries
        .iter()
        .flatten()
        .all(|summary| summary.preflight_nonzero_count == 0)
    {
        SanityStatus::AllZero
    } else {
        SanityStatus::Ok
    }
}

fn case_is_running(cell: &CaseCell) -> bool {
    matches!(cell.symbolic, StageCell::Running(_))
        || matches!(cell.jit, StageCell::Running(_))
        || matches!(cell.objective, StageCell::Running(_))
        || matches!(cell.gradient, StageCell::Running(_))
        || matches!(cell.jacobian, StageCell::Running(_))
        || matches!(cell.hessian, StageCell::Running(_))
}

fn finalize_running_stage(stage: &mut StageCell) {
    if matches!(stage, StageCell::Running(_)) {
        *stage = StageCell::Done(None);
    }
}

fn symbolic_compile_stage_label(stage: SymbolicCompileStage) -> &'static str {
    match stage {
        SymbolicCompileStage::BuildProblem => "build problem",
        SymbolicCompileStage::ObjectiveGradient => "objective gradient",
        SymbolicCompileStage::EqualityJacobian => "equality jacobian",
        SymbolicCompileStage::InequalityJacobian => "inequality jacobian",
        SymbolicCompileStage::LagrangianAssembly => "lagrangian assembly",
        SymbolicCompileStage::HessianGeneration => "hessian generation",
    }
}

fn matrix_stage_for_symbolic_compile_stage(stage: SymbolicCompileStage) -> MatrixStage {
    match stage {
        SymbolicCompileStage::BuildProblem => MatrixStage::Build,
        SymbolicCompileStage::ObjectiveGradient => MatrixStage::SymbolicGradient,
        SymbolicCompileStage::EqualityJacobian => MatrixStage::EqualityJacobianBuild,
        SymbolicCompileStage::InequalityJacobian => MatrixStage::InequalityJacobianBuild,
        SymbolicCompileStage::LagrangianAssembly => MatrixStage::LagrangianAssembly,
        SymbolicCompileStage::HessianGeneration => MatrixStage::HessianGeneration,
    }
}

fn next_symbolic_matrix_stage(progress: &SymbolicCompileStageProgress) -> Option<MatrixStage> {
    let ordered = [
        MatrixStage::Build,
        MatrixStage::SymbolicGradient,
        MatrixStage::EqualityJacobianBuild,
        MatrixStage::InequalityJacobianBuild,
        MatrixStage::LagrangianAssembly,
        MatrixStage::HessianGeneration,
    ];
    let current = matrix_stage_for_symbolic_compile_stage(progress.stage);
    let mut seen_current = false;
    for stage in ordered {
        if !seen_current {
            seen_current = stage == current;
            continue;
        }
        if symbolic_stage_applicable_counts(
            progress.metadata.stats.equality_count,
            progress.metadata.stats.inequality_count,
            stage,
        ) {
            return Some(stage);
        }
    }
    None
}

fn symbolic_stage_applicable_counts(
    equality_count: usize,
    inequality_count: usize,
    stage: MatrixStage,
) -> bool {
    match stage {
        MatrixStage::EqualityJacobianBuild => equality_count > 0,
        MatrixStage::InequalityJacobianBuild => inequality_count > 0,
        MatrixStage::Build
        | MatrixStage::SymbolicGradient
        | MatrixStage::LagrangianAssembly
        | MatrixStage::HessianGeneration => true,
        MatrixStage::NlpJit
        | MatrixStage::XdotHelperJit
        | MatrixStage::MultipleShootingArcHelperJit
        | MatrixStage::Objective
        | MatrixStage::Gradient
        | MatrixStage::Jacobian
        | MatrixStage::Hessian => true,
    }
}

fn symbolic_stage_applicable(cell: &CaseCell, stage: MatrixStage) -> bool {
    symbolic_stage_applicable_counts(cell.equality_count, cell.inequality_count, stage)
}

fn jit_stage_applicable(cell: &CaseCell, stage: MatrixStage) -> bool {
    match stage {
        MatrixStage::NlpJit | MatrixStage::XdotHelperJit => true,
        MatrixStage::MultipleShootingArcHelperJit => matches!(
            cell.case.transcription,
            TranscriptionMethod::MultipleShooting
        ),
        _ => true,
    }
}

fn stage_cell(cell: &CaseCell, stage: MatrixStage) -> StageCell {
    match stage {
        MatrixStage::Build
        | MatrixStage::SymbolicGradient
        | MatrixStage::EqualityJacobianBuild
        | MatrixStage::InequalityJacobianBuild
        | MatrixStage::LagrangianAssembly
        | MatrixStage::HessianGeneration => {
            if !symbolic_stage_applicable(cell, stage) {
                StageCell::Skipped
            } else if cell.active_symbolic_stage == Some(stage) {
                cell.symbolic.clone()
            } else {
                StageCell::Pending
            }
        }
        MatrixStage::NlpJit => {
            if !jit_stage_applicable(cell, stage) {
                StageCell::Skipped
            } else {
                cell.nlp_jit.clone()
            }
        }
        MatrixStage::XdotHelperJit => {
            if !jit_stage_applicable(cell, stage) {
                StageCell::Skipped
            } else {
                cell.xdot_helper_jit.clone()
            }
        }
        MatrixStage::MultipleShootingArcHelperJit => {
            if !jit_stage_applicable(cell, stage) {
                StageCell::Skipped
            } else {
                cell.multiple_shooting_arc_helper_jit.clone()
            }
        }
        MatrixStage::Objective => cell.objective.clone(),
        MatrixStage::Gradient => cell.gradient.clone(),
        MatrixStage::Jacobian => cell.jacobian.clone(),
        MatrixStage::Hessian => cell.hessian.clone(),
    }
}

fn short_problem_label(problem_id: ProblemId) -> &'static str {
    match problem_id {
        ProblemId::OptimalDistanceGlider => "Gld",
        ProblemId::LinearSManeuver => "Lin",
        ProblemId::SailboatUpwind => "Sai",
        ProblemId::CraneTransfer => "Crn",
    }
}

fn short_transcription_label(method: TranscriptionMethod) -> &'static str {
    match method {
        TranscriptionMethod::MultipleShooting => "MS",
        TranscriptionMethod::DirectCollocation => "DC",
    }
}

fn transcription_display_name(method: TranscriptionMethod) -> &'static str {
    match method {
        TranscriptionMethod::MultipleShooting => "Multiple Shooting",
        TranscriptionMethod::DirectCollocation => "Direct Collocation",
    }
}

fn kernel_label(kernel: NlpEvaluationKernelKind) -> &'static str {
    match kernel {
        NlpEvaluationKernelKind::ObjectiveValue => "objective",
        NlpEvaluationKernelKind::ObjectiveGradient => "gradient",
        NlpEvaluationKernelKind::EqualityJacobianValues => "equality jacobian",
        NlpEvaluationKernelKind::InequalityJacobianValues => "inequality jacobian",
        NlpEvaluationKernelKind::LagrangianHessianValues => "hessian",
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 60.0 {
        let whole_seconds = duration.as_secs();
        format!("{:02}:{:02}", whole_seconds / 60, whole_seconds % 60)
    } else if seconds >= 1.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{}ms", duration.as_millis())
    }
}

fn compact_seconds(seconds: f64) -> String {
    if seconds >= 10.0 {
        format!("{seconds:.0}s")
    } else if seconds >= 1.0 {
        format!("{seconds:.1}s")
    } else if seconds >= 1.0e-3 {
        format!("{:.0}ms", seconds * 1.0e3)
    } else {
        format!("{:.0}us", seconds * 1.0e6)
    }
}

fn compact_cell_seconds(seconds: f64) -> String {
    if seconds >= 10.0 {
        format!("{seconds:.0}s")
    } else if seconds >= 1.0 {
        format!("{seconds:.1}s")
    } else if seconds >= 1.0e-3 {
        format!("{:.0}ms", seconds * 1.0e3)
    } else {
        format!("{:.0}us", seconds * 1.0e6)
    }
}

fn compact_optional_cell_seconds(value: Option<f64>) -> String {
    value
        .map(compact_cell_seconds)
        .unwrap_or_else(|| "--".to_string())
}

fn format_optional_seconds(value: Option<f64>) -> String {
    value
        .map(compact_seconds)
        .unwrap_or_else(|| "--".to_string())
}
