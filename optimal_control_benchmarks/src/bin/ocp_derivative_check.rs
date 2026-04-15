use std::collections::{BTreeMap, HashMap};
use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::{ArgAction, Parser, ValueEnum};
use optimal_control_problems::{
    DerivativeCheckOrder, DerivativeCheckRequest, OcpBenchmarkPreset, ProblemDerivativeCheck,
    ProblemId, TranscriptionMethod, problem_specs, validate_problem_derivatives,
};
use optimization::{FiniteDifferenceValidationOptions, ValidationSummary, ValidationTolerances};

fn main() -> Result<()> {
    let cli = OcpDerivativeCli::parse();
    ensure_release_mode()?;
    let problems = expand_problem_selections(&cli.problems)?;
    let transcriptions = expand_transcription_selections(&cli.transcriptions)?;
    let presets = expand_preset_selections(&cli.presets)?;
    let overrides = parse_assignments(&cli.set)?;
    let problem_names = problem_specs()
        .into_iter()
        .map(|spec| (spec.id, spec.name))
        .collect::<HashMap<_, _>>();
    let cases = planned_cases(&problems, &transcriptions, &presets, &problem_names);

    let first_tolerances =
        ValidationTolerances::new(cli.first_max_abs_error, cli.first_max_rel_error);
    let second_tolerances =
        ValidationTolerances::new(cli.second_max_abs_error, cli.second_max_rel_error);

    let mut results = Vec::new();
    let mut progress =
        DerivativeProgress::start(&cases, cli.order, first_tolerances, second_tolerances);
    for case in &cases {
        progress.mark_running(case);
        let request = request_for_case(
            case.transcription,
            case.preset,
            cli.collocation_family,
            &overrides,
            FiniteDifferenceValidationOptions {
                first_order_step: cli.first_order_step,
                second_order_step: cli.second_order_step,
                zero_tolerance: cli.zero_tolerance,
            },
            cli.equality_multiplier_fill,
            cli.inequality_multiplier_fill,
        );
        let outcome = match validate_problem_derivatives(case.problem_id, &request) {
            Ok(check) => CaseOutcome::Check(check),
            Err(error) => CaseOutcome::Error(format!("{error:#}")),
        };
        let result = CaseResult {
            problem_id: case.problem_id,
            problem_name: case.problem_name.clone(),
            transcription: case.transcription,
            preset: case.preset,
            outcome,
        };
        progress.mark_finished(&result);
        results.push(result);
    }
    progress.finish();

    print_matrix(
        &results,
        &problems,
        &transcriptions,
        &presets,
        cli.order,
        first_tolerances,
        second_tolerances,
    );
    print_details(
        &results,
        cli.details,
        cli.order,
        first_tolerances,
        second_tolerances,
    );

    let failed = results
        .iter()
        .filter(|result| {
            matches!(result.outcome, CaseOutcome::Check(_))
                && !result.passed(cli.order, first_tolerances, second_tolerances)
        })
        .count();
    let errors = results
        .iter()
        .filter(|result| matches!(result.outcome, CaseOutcome::Error(_)))
        .count();
    let passed = results.len() - failed - errors;

    println!(
        "\nSummary: {} passed, {} failed, {} execution errors, order={:?}, first_tol=({:.1e}, {:.1e}), second_tol=({:.1e}, {:.1e})",
        passed,
        failed,
        errors,
        cli.order,
        first_tolerances.max_abs_error,
        first_tolerances.max_rel_error,
        second_tolerances.max_abs_error,
        second_tolerances.max_rel_error
    );

    if failed > 0 {
        bail!("{failed} derivative check cases failed");
    }
    Ok(())
}

fn ensure_release_mode() -> Result<()> {
    if cfg!(debug_assertions) {
        bail!(
            "ocp_derivative_check must be run in release mode\n\ntry:\n  cargo run -p optimal_control_benchmarks --release --bin ocp_derivative_check -- --problems all --transcriptions all --presets all --order first"
        );
    }
    Ok(())
}

#[derive(Debug, Parser)]
#[command(
    name = "ocp_derivative_check",
    about = "Run numerical derivative checks across OCP problems, transcriptions, and symbolic-function policy presets."
)]
struct OcpDerivativeCli {
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1.., default_values_t = [CliProblemSelection::All])]
    problems: Vec<CliProblemSelection>,
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1.., default_values_t = [CliPresetSelection::Baseline])]
    presets: Vec<CliPresetSelection>,
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1.., default_values_t = [CliTranscriptionSelection::All])]
    transcriptions: Vec<CliTranscriptionSelection>,
    #[arg(long, value_enum)]
    collocation_family: Option<CliCollocationFamily>,
    #[arg(long, value_enum, default_value_t = CliDerivativeOrder::First)]
    order: CliDerivativeOrder,
    #[arg(long, default_value_t = 5.0e-5)]
    first_max_abs_error: f64,
    #[arg(long, default_value_t = 5.0e-4)]
    first_max_rel_error: f64,
    #[arg(long, default_value_t = 1.0e-4)]
    second_max_abs_error: f64,
    #[arg(long, default_value_t = 1.0e-3)]
    second_max_rel_error: f64,
    #[arg(long, default_value_t = 1.0e-6)]
    first_order_step: f64,
    #[arg(long, default_value_t = 1.0e-4)]
    second_order_step: f64,
    #[arg(long, default_value_t = 1.0e-7)]
    zero_tolerance: f64,
    #[arg(long, default_value_t = 1.0)]
    equality_multiplier_fill: f64,
    #[arg(long, default_value_t = 1.0)]
    inequality_multiplier_fill: f64,
    #[arg(long = "set", action = ArgAction::Append, value_name = "KEY=VALUE")]
    set: Vec<String>,
    #[arg(long, value_enum, default_value_t = CliDetailMode::Failures)]
    details: CliDetailMode,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliTranscriptionSelection {
    All,
    #[value(name = "ms", alias = "multiple_shooting")]
    Ms,
    #[value(name = "dc", alias = "direct_collocation")]
    Dc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliCollocationFamily {
    #[value(name = "legendre")]
    Legendre,
    #[value(name = "radau_iia")]
    RadauIia,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliDerivativeOrder {
    First,
    Second,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliDetailMode {
    None,
    Failures,
    All,
}

#[derive(Clone, Debug)]
struct CaseResult {
    problem_id: ProblemId,
    problem_name: String,
    transcription: TranscriptionMethod,
    preset: OcpBenchmarkPreset,
    outcome: CaseOutcome,
}

#[derive(Clone, Debug)]
struct PlannedCase {
    problem_id: ProblemId,
    problem_name: String,
    transcription: TranscriptionMethod,
    preset: OcpBenchmarkPreset,
}

#[derive(Clone, Debug)]
enum CaseOutcome {
    Check(ProblemDerivativeCheck),
    Error(String),
}

impl CaseResult {
    fn passed(
        &self,
        order: CliDerivativeOrder,
        first: ValidationTolerances,
        second: ValidationTolerances,
    ) -> bool {
        match &self.outcome {
            CaseOutcome::Check(check) => {
                check.order_is_within_tolerances(order.into(), first, second)
            }
            CaseOutcome::Error(_) => false,
        }
    }

    fn status_code(
        &self,
        order: CliDerivativeOrder,
        first: ValidationTolerances,
        second: ValidationTolerances,
    ) -> &'static str {
        match &self.outcome {
            CaseOutcome::Error(_) => "ERR",
            CaseOutcome::Check(check) => {
                let first_ok = check.first_order_is_within_tolerances(first);
                let second_ok = check.second_order_is_within_tolerances(second);
                match order {
                    CliDerivativeOrder::First => {
                        if first_ok {
                            "PASS"
                        } else {
                            "F1"
                        }
                    }
                    CliDerivativeOrder::Second => {
                        if second_ok {
                            "PASS"
                        } else {
                            "F2"
                        }
                    }
                    CliDerivativeOrder::All => match (first_ok, second_ok) {
                        (true, true) => "PASS",
                        (false, true) => "F1",
                        (true, false) => "F2",
                        (false, false) => "F12",
                    },
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProgressCell {
    Pending,
    Running,
    Passed,
    Failed,
    Error,
}

struct DerivativeProgress {
    interactive: bool,
    cases: Vec<PlannedCase>,
    cells: HashMap<(ProblemId, &'static str, &'static str), ProgressCell>,
    order: CliDerivativeOrder,
    first: ValidationTolerances,
    second: ValidationTolerances,
    started_at: Instant,
    finished: usize,
    failed: usize,
    errors: usize,
    current_label: Option<String>,
}

impl DerivativeProgress {
    fn start(
        cases: &[PlannedCase],
        order: CliDerivativeOrder,
        first: ValidationTolerances,
        second: ValidationTolerances,
    ) -> Self {
        let interactive = io::stderr().is_terminal();
        let progress = Self {
            interactive,
            cases: cases.to_vec(),
            cells: cases
                .iter()
                .map(|case| {
                    (
                        progress_key(case.problem_id, case.transcription, case.preset),
                        ProgressCell::Pending,
                    )
                })
                .collect(),
            order,
            first,
            second,
            started_at: Instant::now(),
            finished: 0,
            failed: 0,
            errors: 0,
            current_label: None,
        };
        progress.render();
        progress
    }

    fn mark_running(&mut self, case: &PlannedCase) {
        self.current_label = Some(format!(
            "{} / {} / {}",
            case.problem_name,
            transcription_label(case.transcription),
            case.preset.id()
        ));
        self.cells.insert(
            progress_key(case.problem_id, case.transcription, case.preset),
            ProgressCell::Running,
        );
        self.render();
    }

    fn mark_finished(&mut self, result: &CaseResult) {
        let passed = result.passed(self.order, self.first, self.second);
        let cell = match &result.outcome {
            CaseOutcome::Error(_) => {
                self.errors += 1;
                ProgressCell::Error
            }
            CaseOutcome::Check(_) if passed => ProgressCell::Passed,
            CaseOutcome::Check(_) => {
                self.failed += 1;
                ProgressCell::Failed
            }
        };
        self.finished += 1;
        self.cells.insert(
            progress_key(result.problem_id, result.transcription, result.preset),
            cell,
        );
        self.current_label = None;
        self.render();
    }

    fn finish(&mut self) {
        self.current_label = None;
        self.render();
        if self.interactive {
            let _ = writeln!(io::stderr());
        }
    }

    fn render(&self) {
        if self.interactive {
            let mut stderr = io::stderr().lock();
            let _ = write!(stderr, "\x1b[2J\x1b[H{}", self.snapshot());
            let _ = stderr.flush();
        } else if let Some(label) = self.current_label.as_ref() {
            eprintln!("[{}/{}] {}", self.finished + 1, self.cases.len(), label);
        } else {
            eprintln!(
                "[done {}/{}] failed={} errors={}",
                self.finished,
                self.cases.len(),
                self.failed,
                self.errors
            );
        }
    }

    fn snapshot(&self) -> String {
        let total = self.cases.len().max(1);
        let filled = (self.finished * 30) / total;
        let bar = format!(
            "[{}{}]",
            "#".repeat(filled),
            ".".repeat(30usize.saturating_sub(filled))
        );
        let elapsed = self.started_at.elapsed();
        let eta = if self.finished > 0 && self.finished < self.cases.len() {
            let avg = elapsed.as_secs_f64() / self.finished as f64;
            Some(Duration::from_secs_f64(
                avg * (self.cases.len() - self.finished) as f64,
            ))
        } else {
            None
        };

        let mut out = String::new();
        out.push_str("OCP Derivative Sweep\n");
        out.push_str(&format!(
            "{} {}/{}  elapsed={}  eta={}  failed={}  errors={}\n",
            bar,
            self.finished,
            self.cases.len(),
            format_duration(elapsed),
            eta.map(format_duration)
                .unwrap_or_else(|| "--:--".to_string()),
            self.failed,
            self.errors
        ));
        out.push_str(&format!(
            "order={:?} first_tol=({:.1e}, {:.1e}) second_tol=({:.1e}, {:.1e})\n",
            self.order,
            self.first.max_abs_error,
            self.first.max_rel_error,
            self.second.max_abs_error,
            self.second.max_rel_error
        ));
        out.push_str(&format!(
            "running={}\n\n",
            self.current_label.as_deref().unwrap_or("<idle>")
        ));
        out.push_str(&self.matrix_snapshot());
        out.push_str("\nLegend: .. pending, >> running, OK pass, F1/F2/F12 derivative tolerance miss, ERR execution error\n");
        out
    }

    fn matrix_snapshot(&self) -> String {
        let mut problems = Vec::<ProblemId>::new();
        let mut transcriptions = Vec::<TranscriptionMethod>::new();
        let mut presets = Vec::<OcpBenchmarkPreset>::new();
        for case in &self.cases {
            if !problems.contains(&case.problem_id) {
                problems.push(case.problem_id);
            }
            if !transcriptions.contains(&case.transcription) {
                transcriptions.push(case.transcription);
            }
            if !presets.contains(&case.preset) {
                presets.push(case.preset);
            }
        }

        let row_labels = problems
            .iter()
            .flat_map(|problem| {
                transcriptions.iter().map(move |transcription| {
                    format!(
                        "{}/{}",
                        problem.as_str(),
                        transcription_label(*transcription)
                    )
                })
            })
            .collect::<Vec<_>>();
        let row_width = row_labels
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(4)
            .max(18);
        let col_widths = presets
            .iter()
            .map(|preset| preset.id().len().max(4))
            .collect::<Vec<_>>();

        let mut out = String::new();
        out.push_str(&format!("{:row_width$}", "case", row_width = row_width));
        for (preset, width) in presets.iter().zip(col_widths.iter()) {
            out.push_str(&format!(" | {:width$}", preset.id(), width = *width));
        }
        out.push('\n');
        out.push_str(&"-".repeat(row_width + col_widths.iter().sum::<usize>() + 3 * presets.len()));
        out.push('\n');

        for problem in problems {
            for transcription in &transcriptions {
                let row_label = format!(
                    "{}/{}",
                    problem.as_str(),
                    transcription_label(*transcription)
                );
                out.push_str(&format!("{:row_width$}", row_label, row_width = row_width));
                for (preset, width) in presets.iter().zip(col_widths.iter()) {
                    let cell = self
                        .cells
                        .get(&progress_key(problem, *transcription, *preset))
                        .copied()
                        .unwrap_or(ProgressCell::Pending);
                    out.push_str(&format!(
                        " | {:width$}",
                        progress_code(cell, self.order),
                        width = *width
                    ));
                }
                out.push('\n');
            }
        }
        out
    }
}

impl From<CliDerivativeOrder> for DerivativeCheckOrder {
    fn from(value: CliDerivativeOrder) -> Self {
        match value {
            CliDerivativeOrder::First => Self::First,
            CliDerivativeOrder::Second => Self::Second,
            CliDerivativeOrder::All => Self::All,
        }
    }
}

fn request_for_case(
    transcription: TranscriptionMethod,
    preset: OcpBenchmarkPreset,
    collocation_family: Option<CliCollocationFamily>,
    overrides: &BTreeMap<String, f64>,
    finite_difference: FiniteDifferenceValidationOptions,
    equality_multiplier_fill: f64,
    inequality_multiplier_fill: f64,
) -> DerivativeCheckRequest {
    let mut values = overrides.clone();
    values.insert(
        "transcription_method".to_string(),
        match transcription {
            TranscriptionMethod::MultipleShooting => 0.0,
            TranscriptionMethod::DirectCollocation => 1.0,
        },
    );
    if let Some(family) = collocation_family {
        values.insert(
            "collocation_family".to_string(),
            match family {
                CliCollocationFamily::Legendre => 0.0,
                CliCollocationFamily::RadauIia => 1.0,
            },
        );
    }
    DerivativeCheckRequest {
        values,
        finite_difference,
        equality_multiplier_fill,
        inequality_multiplier_fill,
        sx_functions_override: Some(preset.sx_function_config()),
    }
}

fn parse_assignments(assignments: &[String]) -> Result<BTreeMap<String, f64>> {
    let mut values = BTreeMap::new();
    for assignment in assignments {
        let (key, value) = assignment
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("expected KEY=VALUE, got `{assignment}`"))?;
        let parsed = value.parse::<f64>().map_err(|_| {
            anyhow::anyhow!("expected numeric VALUE in KEY=VALUE override, got `{assignment}`")
        })?;
        values.insert(key.to_string(), parsed);
    }
    Ok(values)
}

fn expand_problem_selections(selections: &[CliProblemSelection]) -> Result<Vec<ProblemId>> {
    expand_selection_list(
        selections,
        CliProblemSelection::All,
        vec![
            ProblemId::OptimalDistanceGlider,
            ProblemId::LinearSManeuver,
            ProblemId::SailboatUpwind,
            ProblemId::CraneTransfer,
        ],
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
        vec![
            TranscriptionMethod::MultipleShooting,
            TranscriptionMethod::DirectCollocation,
        ],
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

fn planned_cases(
    problems: &[ProblemId],
    transcriptions: &[TranscriptionMethod],
    presets: &[OcpBenchmarkPreset],
    problem_names: &HashMap<ProblemId, String>,
) -> Vec<PlannedCase> {
    let mut cases = Vec::new();
    for problem in problems {
        for transcription in transcriptions {
            for preset in presets {
                cases.push(PlannedCase {
                    problem_id: *problem,
                    problem_name: problem_names
                        .get(problem)
                        .cloned()
                        .unwrap_or_else(|| problem.as_str().to_string()),
                    transcription: *transcription,
                    preset: *preset,
                });
            }
        }
    }
    cases
}

fn print_matrix(
    results: &[CaseResult],
    problems: &[ProblemId],
    transcriptions: &[TranscriptionMethod],
    presets: &[OcpBenchmarkPreset],
    order: CliDerivativeOrder,
    first: ValidationTolerances,
    second: ValidationTolerances,
) {
    let row_width = problems
        .iter()
        .flat_map(|problem| {
            transcriptions.iter().map(move |transcription| {
                format!(
                    "{}/{}",
                    problem.as_str(),
                    transcription_label(*transcription)
                )
                .len()
            })
        })
        .max()
        .unwrap_or(4)
        .max(12);
    let col_widths = presets
        .iter()
        .map(|preset| preset.id().len().max(4))
        .collect::<Vec<_>>();

    print!("{:row_width$}", "case", row_width = row_width);
    for (preset, width) in presets.iter().zip(col_widths.iter()) {
        print!(" | {:width$}", preset.id(), width = *width);
    }
    println!();
    println!(
        "{}",
        "-".repeat(row_width + col_widths.iter().sum::<usize>() + 3 * presets.len())
    );

    for problem in problems {
        for transcription in transcriptions {
            let row_label = format!(
                "{}/{}",
                problem.as_str(),
                transcription_label(*transcription)
            );
            print!("{:row_width$}", row_label, row_width = row_width);
            for (preset, width) in presets.iter().zip(col_widths.iter()) {
                let result = results
                    .iter()
                    .find(|result| {
                        result.problem_id == *problem
                            && result.transcription == *transcription
                            && result.preset == *preset
                    })
                    .expect("matrix result should exist");
                print!(
                    " | {:width$}",
                    result.status_code(order, first, second),
                    width = *width
                );
            }
            println!();
        }
    }
}

fn print_details(
    results: &[CaseResult],
    details: CliDetailMode,
    order: CliDerivativeOrder,
    first: ValidationTolerances,
    second: ValidationTolerances,
) {
    for result in results {
        let passed = result.passed(order, first, second);
        let should_print = match details {
            CliDetailMode::None => false,
            CliDetailMode::Failures => !passed,
            CliDetailMode::All => true,
        };
        if !should_print {
            continue;
        }
        println!(
            "\n=== {} / {} / {} ===",
            result.problem_name,
            transcription_label(result.transcription),
            result.preset.id()
        );
        println!("status={}", result.status_code(order, first, second));
        match &result.outcome {
            CaseOutcome::Check(check) => println!("{}", format_check(check)),
            CaseOutcome::Error(error) => println!("error: {error}"),
        }
    }
}

fn summary_line(label: &str, summary: &ValidationSummary) -> String {
    let worst = summary.worst_entry.as_ref().map_or_else(
        || "worst=none".to_string(),
        |entry| {
            format!(
                "worst=({}, {}) analytic={:.3e} fd={:.3e} abs={:.3e} rel={:.3e}",
                entry.row,
                entry.col,
                entry.analytic,
                entry.finite_difference,
                entry.abs_error,
                entry.rel_error
            )
        },
    );
    let worst_missing = summary
        .sparsity
        .worst_missing_from_analytic
        .as_ref()
        .map_or_else(
            || "worst_missing=none".to_string(),
            |entry| {
                format!(
                    "worst_missing=({}, {}) analytic={:.3e} fd={:.3e} abs={:.3e} rel={:.3e}",
                    entry.row,
                    entry.col,
                    entry.analytic,
                    entry.finite_difference,
                    entry.abs_error,
                    entry.rel_error
                )
            },
        );
    let worst_extra = summary
        .sparsity
        .worst_extra_in_analytic
        .as_ref()
        .map_or_else(
            || "worst_extra=none".to_string(),
            |entry| {
                format!(
                    "worst_extra=({}, {}) analytic={:.3e} fd={:.3e} abs={:.3e} rel={:.3e}",
                    entry.row,
                    entry.col,
                    entry.analytic,
                    entry.finite_difference,
                    entry.abs_error,
                    entry.rel_error
                )
            },
        );
    format!(
        "{label}: max_abs={:.3e} max_rel={:.3e} rms_abs={:.3e} missing={} extra={} {worst} {worst_missing} {worst_extra}",
        summary.max_abs_error,
        summary.max_rel_error,
        summary.rms_abs_error,
        summary.sparsity.missing_from_analytic,
        summary.sparsity.extra_in_analytic,
    )
}

fn format_check(check: &ProblemDerivativeCheck) -> String {
    let mut lines = vec![
        format!(
            "{} {:?} family={:?} cached={} sx={:?}",
            check.problem_name,
            check.transcription,
            check.collocation_family,
            check.compile_cached,
            check.sx_functions,
        ),
        summary_line("objective_gradient", &check.report.objective_gradient),
    ];
    if let Some(summary) = check.report.equality_jacobian.as_ref() {
        lines.push(summary_line("equality_jacobian", summary));
    }
    if let Some(summary) = check.report.inequality_jacobian.as_ref() {
        lines.push(summary_line("inequality_jacobian", summary));
    }
    lines.push(summary_line(
        "lagrangian_hessian",
        &check.report.lagrangian_hessian,
    ));
    lines.push(format!(
        "compile_stats: functions={} calls={} depth={} llvm_calls={}",
        check.compile_report.symbolic_function_count,
        check.compile_report.call_site_count,
        check.compile_report.max_call_depth,
        check.compile_report.llvm_call_instructions_emitted,
    ));
    lines.join("\n")
}

fn transcription_label(transcription: TranscriptionMethod) -> &'static str {
    match transcription {
        TranscriptionMethod::MultipleShooting => "ms",
        TranscriptionMethod::DirectCollocation => "dc",
    }
}

fn progress_key(
    problem_id: ProblemId,
    transcription: TranscriptionMethod,
    preset: OcpBenchmarkPreset,
) -> (ProblemId, &'static str, &'static str) {
    (problem_id, transcription_label(transcription), preset.id())
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let minutes = seconds / 60;
    let rem = seconds % 60;
    format!("{minutes:02}:{rem:02}")
}

fn progress_code(cell: ProgressCell, order: CliDerivativeOrder) -> &'static str {
    match cell {
        ProgressCell::Pending => "..",
        ProgressCell::Running => ">>",
        ProgressCell::Passed => "OK",
        ProgressCell::Failed => match order {
            CliDerivativeOrder::First => "F1",
            CliDerivativeOrder::Second => "F2",
            CliDerivativeOrder::All => "F12",
        },
        ProgressCell::Error => "ERR",
    }
}
