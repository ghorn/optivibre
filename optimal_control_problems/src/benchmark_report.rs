use std::collections::{BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use optimal_control::{
    CollocationFamily, OcpCompileOptions, OcpCompileProgress, OcpHelperCompileStats,
};
use optimization::{
    CallPolicy, KernelBenchmarkStats, LlvmOptimizationLevel, NlpBenchmarkPointSummary,
    NlpCompileStats, NlpEvaluationBenchmark, NlpEvaluationBenchmarkOptions,
    NlpEvaluationKernelKind,
};
use serde::Serialize;
use sx_core::HessianStrategy;

use crate::OcpSxFunctionConfig;
use crate::common::{CompileReportSummary, ProblemId, TranscriptionMethod};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OcpBenchmarkPreset {
    Baseline,
    BaselineHessianSelectedOutputs,
    BaselineHessianColored,
    InlineAll,
    InlineAllHessianSelectedOutputs,
    InlineAllHessianColored,
    FunctionInlineAtCall,
    FunctionInlineAtLowering,
    FunctionInlineInLlvm,
    FunctionNoInlineLlvm,
}

impl OcpBenchmarkPreset {
    pub const fn default_matrix() -> &'static [Self] {
        &[
            Self::Baseline,
            Self::InlineAll,
            Self::FunctionInlineAtLowering,
            Self::FunctionInlineInLlvm,
            Self::FunctionNoInlineLlvm,
        ]
    }

    pub const fn all() -> &'static [Self] {
        &[
            Self::Baseline,
            Self::BaselineHessianSelectedOutputs,
            Self::BaselineHessianColored,
            Self::InlineAll,
            Self::InlineAllHessianSelectedOutputs,
            Self::InlineAllHessianColored,
            Self::FunctionInlineAtCall,
            Self::FunctionInlineAtLowering,
            Self::FunctionInlineInLlvm,
            Self::FunctionNoInlineLlvm,
        ]
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::BaselineHessianSelectedOutputs => "baseline_hessian_selected_outputs",
            Self::BaselineHessianColored => "baseline_hessian_colored",
            Self::InlineAll => "inline_all",
            Self::InlineAllHessianSelectedOutputs => "inline_all_hessian_selected_outputs",
            Self::InlineAllHessianColored => "inline_all_hessian_colored",
            Self::FunctionInlineAtCall => "function_inline_at_call",
            Self::FunctionInlineAtLowering => "function_inline_at_lowering",
            Self::FunctionInlineInLlvm => "function_inline_in_llvm",
            Self::FunctionNoInlineLlvm => "function_noinline_llvm",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "Baseline",
            Self::BaselineHessianSelectedOutputs => "Baseline / Hessian Selected Outputs",
            Self::BaselineHessianColored => "Baseline / Hessian Colored",
            Self::InlineAll => "Inline All",
            Self::InlineAllHessianSelectedOutputs => "Inline All / Hessian Selected Outputs",
            Self::InlineAllHessianColored => "Inline All / Hessian Colored",
            Self::FunctionInlineAtCall => "Functions / Inline At Call",
            Self::FunctionInlineAtLowering => "Functions / Inline At Lowering",
            Self::FunctionInlineInLlvm => "Functions / Inline In LLVM",
            Self::FunctionNoInlineLlvm => "Functions / NoInline LLVM",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::Baseline => {
                "Current OCP defaults: reusable leaf kernels where configured, with LLVM-level inlining overrides on the default reusable kernels."
            }
            Self::BaselineHessianSelectedOutputs => {
                "Current OCP defaults, but generate the Lagrangian Hessian with selected-output lower-triangle sweeps."
            }
            Self::BaselineHessianColored => {
                "Current OCP defaults, but generate the Lagrangian Hessian with colored lower-triangle sweeps."
            }
            Self::InlineAll => {
                "Disable reusable OCP symbolic functions and inline repeated kernels immediately."
            }
            Self::InlineAllHessianSelectedOutputs => {
                "Inline repeated OCP kernels immediately and generate the Lagrangian Hessian with selected-output lower-triangle sweeps."
            }
            Self::InlineAllHessianColored => {
                "Inline repeated OCP kernels immediately and generate the Lagrangian Hessian with colored lower-triangle sweeps."
            }
            Self::FunctionInlineAtCall => {
                "Build all OCP kernels as symbolic functions but expand them at call construction time."
            }
            Self::FunctionInlineAtLowering => {
                "Keep reusable symbolic call boundaries through setup/AD and inline only during lowering."
            }
            Self::FunctionInlineInLlvm => {
                "Keep reusable call boundaries through lowering and allow LLVM to inline internal subfunctions."
            }
            Self::FunctionNoInlineLlvm => {
                "Keep reusable call boundaries through lowering and preserve internal LLVM calls."
            }
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "baseline" => Some(Self::Baseline),
            "baseline_hessian_selected_outputs" => Some(Self::BaselineHessianSelectedOutputs),
            "baseline_hessian_colored" => Some(Self::BaselineHessianColored),
            "inline_all" => Some(Self::InlineAll),
            "inline_all_hessian_selected_outputs" => Some(Self::InlineAllHessianSelectedOutputs),
            "inline_all_hessian_colored" => Some(Self::InlineAllHessianColored),
            "function_inline_at_call" => Some(Self::FunctionInlineAtCall),
            "function_inline_at_lowering" => Some(Self::FunctionInlineAtLowering),
            "function_inline_in_llvm" => Some(Self::FunctionInlineInLlvm),
            "function_noinline_llvm" => Some(Self::FunctionNoInlineLlvm),
            _ => None,
        }
    }

    pub fn compile_options(self, opt_level: LlvmOptimizationLevel) -> OcpCompileOptions {
        self.sx_function_config()
            .with_opt_level(opt_level)
            .compile_options(opt_level)
    }

    pub fn sx_function_config(self) -> OcpSxFunctionConfig {
        match self {
            Self::Baseline => OcpSxFunctionConfig::default(),
            Self::BaselineHessianSelectedOutputs => OcpSxFunctionConfig::default()
                .with_hessian_strategy(HessianStrategy::LowerTriangleSelectedOutputs),
            Self::BaselineHessianColored => OcpSxFunctionConfig::default()
                .with_hessian_strategy(HessianStrategy::LowerTriangleColored),
            Self::InlineAll => OcpSxFunctionConfig::inline_all(),
            Self::InlineAllHessianSelectedOutputs => OcpSxFunctionConfig::inline_all()
                .with_hessian_strategy(HessianStrategy::LowerTriangleSelectedOutputs),
            Self::InlineAllHessianColored => OcpSxFunctionConfig::inline_all()
                .with_hessian_strategy(HessianStrategy::LowerTriangleColored),
            Self::FunctionInlineAtCall => {
                OcpSxFunctionConfig::all_functions_with_global_policy(CallPolicy::InlineAtCall)
            }
            Self::FunctionInlineAtLowering => {
                OcpSxFunctionConfig::all_functions_with_global_policy(CallPolicy::InlineAtLowering)
            }
            Self::FunctionInlineInLlvm => {
                OcpSxFunctionConfig::all_functions_with_global_policy(CallPolicy::InlineInLLVM)
            }
            Self::FunctionNoInlineLlvm => {
                OcpSxFunctionConfig::all_functions_with_global_policy(CallPolicy::NoInlineLLVM)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OcpBenchmarkSuiteConfig {
    pub problems: Vec<ProblemId>,
    pub transcriptions: Vec<TranscriptionMethod>,
    pub presets: Vec<OcpBenchmarkPreset>,
    pub eval_options: NlpEvaluationBenchmarkOptions,
    pub jobs: usize,
}

impl Default for OcpBenchmarkSuiteConfig {
    fn default() -> Self {
        Self {
            problems: vec![
                ProblemId::OptimalDistanceGlider,
                ProblemId::AlbatrossDynamicSoaring,
                ProblemId::LinearSManeuver,
                ProblemId::SailboatUpwind,
                ProblemId::CraneTransfer,
            ],
            transcriptions: vec![
                TranscriptionMethod::MultipleShooting,
                TranscriptionMethod::DirectCollocation,
            ],
            presets: OcpBenchmarkPreset::default_matrix().to_vec(),
            eval_options: NlpEvaluationBenchmarkOptions::default(),
            jobs: default_benchmark_jobs(),
        }
    }
}

pub fn default_benchmark_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| (parallelism.get() / 2).max(1))
        .unwrap_or(1)
}

#[derive(Clone, Debug, Serialize)]
pub struct KernelBenchmarkSummary {
    pub output_len: usize,
    pub iterations: usize,
    pub total_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stddev_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_s: Option<f64>,
    pub preflight_finite: bool,
    pub preflight_nonzero_count: usize,
    pub preflight_max_abs: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OcpBenchmarkPointSummary {
    pub decision_inf_norm: f64,
    pub parameter_inf_norm: f64,
    pub objective_value: f64,
    pub objective_finite: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equality_inf_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inequality_inf_norm: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OcpEvalBenchmarkSummary {
    pub benchmark_point: OcpBenchmarkPointSummary,
    pub objective_value: KernelBenchmarkSummary,
    pub objective_gradient: KernelBenchmarkSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equality_jacobian_values: Option<KernelBenchmarkSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inequality_jacobian_values: Option<KernelBenchmarkSummary>,
    pub lagrangian_hessian_values: KernelBenchmarkSummary,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OcpHelperCompileSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xdot_helper_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiple_shooting_arc_helper_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xdot_helper_root_instructions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xdot_helper_total_instructions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiple_shooting_arc_helper_root_instructions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiple_shooting_arc_helper_total_instructions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_s: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OcpNlpShapeSummary {
    pub variable_count: usize,
    pub parameter_scalar_count: usize,
    pub equality_count: usize,
    pub inequality_count: usize,
    pub objective_gradient_nnz: usize,
    pub equality_jacobian_nnz: usize,
    pub inequality_jacobian_nnz: usize,
    pub hessian_nnz: usize,
    pub nlp_kernel_count: usize,
    pub helper_kernel_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct OcpBenchmarkRecord {
    pub problem_id: ProblemId,
    pub problem_name: String,
    pub transcription_id: String,
    pub transcription_label: String,
    pub preset_id: String,
    pub preset_label: String,
    pub preset_description: String,
    pub opt_level: String,
    pub compile: CompileReportSummary,
    pub helper_compile: OcpHelperCompileSummary,
    pub nlp: OcpNlpShapeSummary,
    pub eval: OcpEvalBenchmarkSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbolic_total_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jit_total_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compile_total_s: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct OcpBenchmarkSuite {
    pub eval_options: NlpEvaluationBenchmarkOptions,
    pub records: Vec<OcpBenchmarkRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcpBenchmarkCase {
    pub problem_id: ProblemId,
    pub transcription: TranscriptionMethod,
    pub preset: OcpBenchmarkPreset,
}

impl OcpBenchmarkCase {
    pub const fn problem_label(self) -> &'static str {
        problem_label(self.problem_id)
    }

    pub const fn transcription_label(self) -> &'static str {
        match self.transcription {
            TranscriptionMethod::MultipleShooting => "Multiple Shooting",
            TranscriptionMethod::DirectCollocation => "Direct Collocation",
        }
    }

    pub const fn preset_label(self) -> &'static str {
        self.preset.label()
    }
}

#[derive(Clone, Debug)]
pub enum OcpBenchmarkProgress {
    CaseStarted {
        current: usize,
        total: usize,
        case: OcpBenchmarkCase,
    },
    CompileProgress {
        current: usize,
        total: usize,
        case: OcpBenchmarkCase,
        progress: OcpCompileProgress,
    },
    EvalKernelStarted {
        current: usize,
        total: usize,
        case: OcpBenchmarkCase,
        kernel: NlpEvaluationKernelKind,
    },
    CaseFinished {
        current: usize,
        total: usize,
        case: OcpBenchmarkCase,
        record: OcpBenchmarkRecord,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BenchmarkCaseProgress {
    Compile(OcpCompileProgress),
    EvalKernelStarted(NlpEvaluationKernelKind),
}

impl OcpBenchmarkRecord {
    fn sort_key(&self) -> (&str, &str, &str) {
        (
            self.problem_id.as_str(),
            self.transcription_id.as_str(),
            self.preset_id.as_str(),
        )
    }
}

pub(crate) fn opt_level_for_transcription(method: TranscriptionMethod) -> LlvmOptimizationLevel {
    match method {
        TranscriptionMethod::MultipleShooting => {
            crate::common::interactive_multiple_shooting_opt_level()
        }
        TranscriptionMethod::DirectCollocation => {
            crate::common::interactive_direct_collocation_opt_level()
        }
    }
}

pub(crate) fn build_record(
    problem_id: ProblemId,
    problem_name: &str,
    transcription: TranscriptionMethod,
    collocation_family: Option<CollocationFamily>,
    preset: OcpBenchmarkPreset,
    opt_level: LlvmOptimizationLevel,
    compile: CompileReportSummary,
    helper_stats: OcpHelperCompileStats,
    nlp_stats: NlpCompileStats,
    helper_kernel_count: usize,
    eval: NlpEvaluationBenchmark,
) -> OcpBenchmarkRecord {
    let helper_compile = summarize_helper_compile(helper_stats);
    let symbolic_total_s = sum_options([
        compile.symbolic_construction_s,
        compile.objective_gradient_s,
        compile.equality_jacobian_s,
        compile.inequality_jacobian_s,
        compile.lagrangian_assembly_s,
        compile.hessian_generation_s,
    ]);
    let jit_total_s = sum_options([
        compile.lowering_s,
        compile.llvm_jit_s,
        helper_compile.total_s,
    ]);
    let compile_total_s = sum_options([symbolic_total_s, jit_total_s]);
    OcpBenchmarkRecord {
        problem_id,
        problem_name: problem_name.to_string(),
        transcription_id: transcription_id(transcription).to_string(),
        transcription_label: transcription_label(transcription, collocation_family).to_string(),
        preset_id: preset.id().to_string(),
        preset_label: preset.label().to_string(),
        preset_description: preset.description().to_string(),
        opt_level: opt_level_label(opt_level).to_string(),
        compile,
        helper_compile,
        nlp: OcpNlpShapeSummary {
            variable_count: nlp_stats.variable_count,
            parameter_scalar_count: nlp_stats.parameter_scalar_count,
            equality_count: nlp_stats.equality_count,
            inequality_count: nlp_stats.inequality_count,
            objective_gradient_nnz: nlp_stats.objective_gradient_nnz,
            equality_jacobian_nnz: nlp_stats.equality_jacobian_nnz,
            inequality_jacobian_nnz: nlp_stats.inequality_jacobian_nnz,
            hessian_nnz: nlp_stats.hessian_nnz,
            nlp_kernel_count: nlp_stats.jit_kernel_count,
            helper_kernel_count,
        },
        eval: summarize_eval_benchmark(eval),
        symbolic_total_s,
        jit_total_s,
        compile_total_s,
    }
}

pub fn run_ocp_benchmark_suite(config: &OcpBenchmarkSuiteConfig) -> Result<OcpBenchmarkSuite> {
    run_ocp_benchmark_suite_with_progress(config, |_| {})
}

pub fn run_ocp_benchmark_suite_with_progress<CB>(
    config: &OcpBenchmarkSuiteConfig,
    mut on_progress: CB,
) -> Result<OcpBenchmarkSuite>
where
    CB: FnMut(OcpBenchmarkProgress),
{
    let cases = build_benchmark_cases(config);
    let total = cases.len();
    if total == 0 {
        return Ok(OcpBenchmarkSuite {
            eval_options: config.eval_options,
            records: Vec::new(),
        });
    }

    let worker_count = config.jobs.max(1).min(total);
    let queue = Arc::new(Mutex::new(VecDeque::from(cases)));
    let cancel = Arc::new(AtomicBool::new(false));
    let started_count = Arc::new(AtomicUsize::new(0));
    let completed_count = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<WorkerMessage>();
    let mut workers = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let queue = Arc::clone(&queue);
        let cancel = Arc::clone(&cancel);
        let started_count = Arc::clone(&started_count);
        let completed_count = Arc::clone(&completed_count);
        let tx = tx.clone();
        let eval_options = config.eval_options;
        workers.push(thread::spawn(move || {
            loop {
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                let case = {
                    let mut queue = match queue.lock() {
                        Ok(guard) => guard,
                        Err(poison) => poison.into_inner(),
                    };
                    queue.pop_front()
                };
                let Some(case) = case else {
                    break;
                };
                let started = started_count.fetch_add(1, Ordering::SeqCst) + 1;
                if tx
                    .send(WorkerMessage::Progress(OcpBenchmarkProgress::CaseStarted {
                        current: started,
                        total,
                        case,
                    }))
                    .is_err()
                {
                    break;
                }
                let mut forward_progress = |progress| {
                    let result = match progress {
                        BenchmarkCaseProgress::Compile(progress) => tx.send(
                            WorkerMessage::Progress(OcpBenchmarkProgress::CompileProgress {
                                current: started,
                                total,
                                case,
                                progress,
                            }),
                        ),
                        BenchmarkCaseProgress::EvalKernelStarted(kernel) => tx.send(
                            WorkerMessage::Progress(OcpBenchmarkProgress::EvalKernelStarted {
                                current: started,
                                total,
                                case,
                                kernel,
                            }),
                        ),
                    };
                    if result.is_err() {
                        cancel.store(true, Ordering::SeqCst);
                    }
                };
                match run_benchmark_case(case, eval_options, &mut forward_progress) {
                    Ok(record) => {
                        let completed = completed_count.fetch_add(1, Ordering::SeqCst) + 1;
                        if tx
                            .send(WorkerMessage::Progress(
                                OcpBenchmarkProgress::CaseFinished {
                                    current: completed,
                                    total,
                                    case,
                                    record,
                                },
                            ))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        cancel.store(true, Ordering::SeqCst);
                        let _ = tx.send(WorkerMessage::Error(error));
                        break;
                    }
                }
            }
        }));
    }
    drop(tx);

    let mut records = Vec::new();
    let mut first_error = None;
    for message in rx {
        match message {
            WorkerMessage::Progress(progress) => {
                if let OcpBenchmarkProgress::CaseFinished { record, .. } = &progress {
                    records.push(record.clone());
                }
                on_progress(progress);
            }
            WorkerMessage::Error(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    for worker in workers {
        let _ = worker.join();
    }
    if let Some(error) = first_error {
        return Err(error);
    }

    records.sort_by(|lhs, rhs| lhs.sort_key().cmp(&rhs.sort_key()));
    Ok(OcpBenchmarkSuite {
        eval_options: config.eval_options,
        records,
    })
}

fn build_benchmark_cases(config: &OcpBenchmarkSuiteConfig) -> Vec<OcpBenchmarkCase> {
    let mut cases = Vec::new();
    for &problem_id in &config.problems {
        for &transcription in &config.transcriptions {
            for &preset in &config.presets {
                cases.push(OcpBenchmarkCase {
                    problem_id,
                    transcription,
                    preset,
                });
            }
        }
    }
    cases
}

enum WorkerMessage {
    Progress(OcpBenchmarkProgress),
    Error(anyhow::Error),
}

fn run_benchmark_case(
    case: OcpBenchmarkCase,
    eval_options: NlpEvaluationBenchmarkOptions,
    on_progress: &mut dyn FnMut(BenchmarkCaseProgress),
) -> Result<OcpBenchmarkRecord> {
    crate::benchmark_problem_case_with_progress(
        case.problem_id,
        case.transcription,
        case.preset,
        eval_options,
        on_progress,
    )
}

pub fn write_ocp_benchmark_report(path: &Path, suite: &OcpBenchmarkSuite) -> Result<()> {
    fs::write(path, render_ocp_benchmark_report(suite))?;
    Ok(())
}

pub fn render_ocp_benchmark_report(suite: &OcpBenchmarkSuite) -> String {
    let problem_values = collect_unique_strings(
        suite
            .records
            .iter()
            .map(|record| (record.problem_id.as_str(), record.problem_name.as_str())),
    );
    let transcription_values = collect_unique_strings(suite.records.iter().map(|record| {
        (
            record.transcription_id.as_str(),
            record.transcription_label.as_str(),
        )
    }));
    let preset_values = collect_unique_strings(
        suite
            .records
            .iter()
            .map(|record| (record.preset_id.as_str(), record.preset_label.as_str())),
    );
    let preset_summary_rows = preset_summary_rows(suite);
    let best_strategy_rows = best_strategy_by_case_rows(suite);
    let preset_win_rows = preset_win_rows(suite);
    let scales = ReportScales::from_records(&suite.records);
    let total_compile_sum = suite
        .records
        .iter()
        .filter_map(|record| record.compile_total_s)
        .sum::<f64>();
    let slowest_compile = suite
        .records
        .iter()
        .filter_map(|record| record.compile_total_s.map(|value| (record, value)))
        .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
    let slowest_hessian = suite
        .records
        .iter()
        .filter_map(|record| {
            record
                .eval
                .lagrangian_hessian_values
                .average_s
                .map(|value| (record, value))
        })
        .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
    let slowest_gradient = suite
        .records
        .iter()
        .filter_map(|record| {
            record
                .eval
                .objective_gradient
                .average_s
                .map(|value| (record, value))
        })
        .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));

    let mut rows_html = String::new();
    for record in &suite.records {
        let mut details = String::new();
        write!(
            details,
            "<div class=\"details-grid\">\
               <div><span class=\"detail-label\">Compile stages</span>{}</div>\
               <div><span class=\"detail-label\">Helpers</span>{}</div>\
               <div><span class=\"detail-label\">NLP shape</span>{}</div>\
               <div><span class=\"detail-label\">Eval stats</span>{}</div>\
               <div><span class=\"detail-label\">Benchmark point</span>{}</div>\
               <div><span class=\"detail-label\">Warnings</span>{}</div>\
             </div>",
            render_compile_breakdown(record),
            render_helper_breakdown(record),
            render_shape_breakdown(record),
            render_eval_breakdown(record),
            render_benchmark_point_breakdown(record),
            render_warning_breakdown(record),
        )
        .expect("writing benchmark details should not fail");
        write!(
            rows_html,
            "<tr data-problem=\"{}\" data-transcription=\"{}\" data-preset=\"{}\">\
               <td><div class=\"label-stack\"><strong>{}</strong><span>{}</span></div></td>\
               <td>{}</td>\
               <td><div class=\"label-stack\"><strong>{}</strong><span>{}</span></div></td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
             </tr>",
            escape_attr(record.problem_id.as_str()),
            escape_attr(&record.transcription_id),
            escape_attr(&record.preset_id),
            escape_html(&record.problem_name),
            escape_html(record.problem_id.as_str()),
            escape_html(&record.transcription_label),
            escape_html(&record.preset_label),
            escape_html(&record.preset_id),
            escape_html(&record.opt_level),
            metric_bar_cell(record.symbolic_total_s, scales.symbolic_total_max, false),
            metric_bar_cell(record.jit_total_s, scales.jit_total_max, false),
            metric_bar_cell(record.compile_total_s, scales.compile_total_max, false),
            metric_bar_cell(
                record.eval.objective_value.average_s,
                scales.objective_value_max,
                true,
            ),
            metric_bar_cell(
                record.eval.objective_gradient.average_s,
                scales.objective_gradient_max,
                true,
            ),
            metric_bar_cell(
                record
                    .eval
                    .equality_jacobian_values
                    .as_ref()
                    .and_then(|summary| summary.average_s),
                scales.equality_jacobian_max,
                true,
            ),
            metric_bar_cell(
                record
                    .eval
                    .inequality_jacobian_values
                    .as_ref()
                    .and_then(|summary| summary.average_s),
                scales.inequality_jacobian_max,
                true,
            ),
            metric_bar_cell(
                record.eval.lagrangian_hessian_values.average_s,
                scales.hessian_max,
                true,
            ),
            details,
        )
        .expect("writing benchmark rows should not fail");
    }

    let mut preset_summary_html = String::new();
    for row in preset_summary_rows {
        write!(
            preset_summary_html,
            "<tr>\
               <td><div class=\"label-stack\"><strong>{}</strong><span>{}</span></div></td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
             </tr>",
            escape_html(row.preset_label),
            escape_html(row.preset_description),
            row.case_count,
            metric_bar_cell(row.avg_compile_total_s, scales.compile_total_max, false),
            metric_bar_cell(
                row.avg_objective_gradient_s,
                scales.objective_gradient_max,
                true
            ),
            metric_bar_cell(
                row.avg_equality_jacobian_s,
                scales.equality_jacobian_max,
                true
            ),
            metric_bar_cell(row.avg_hessian_s, scales.hessian_max, true),
        )
        .expect("writing preset summary rows should not fail");
    }

    let mut best_strategy_html = String::new();
    for row in best_strategy_rows {
        write!(
            best_strategy_html,
            "<tr>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
             </tr>",
            row.problem_html,
            row.transcription_html,
            row.symbolic_html,
            row.jit_html,
            row.compile_html,
            row.objective_html,
            row.gradient_html,
            row.jacobian_html,
            row.hessian_html,
        )
        .expect("writing best strategy rows should not fail");
    }

    let mut preset_win_html = String::new();
    for row in preset_win_rows {
        write!(
            preset_win_html,
            "<tr>\
               <td><div class=\"label-stack\"><strong>{}</strong><span>{}</span></div></td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
               <td>{}</td>\
             </tr>",
            escape_html(row.preset_label),
            escape_html(row.preset_description),
            row.symbolic_wins,
            row.jit_wins,
            row.compile_wins,
            row.objective_wins,
            row.gradient_wins,
            row.jacobian_wins,
            row.hessian_wins,
            row.total_wins,
        )
        .expect("writing preset win rows should not fail");
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OCP Compile and Eval Benchmark Report</title>
  <style>
    :root {{
      --bg0: #f3ede2;
      --bg1: #d8e6ef;
      --panel: rgba(255, 252, 247, 0.84);
      --panel-strong: rgba(255, 252, 247, 0.96);
      --ink: #13263a;
      --muted: #5c6b78;
      --line: rgba(19, 38, 58, 0.14);
      --accent: #0f766e;
      --accent-strong: #0b4f6c;
      --warm: #c26d20;
      --bar: rgba(15, 118, 110, 0.18);
      --bar-warm: rgba(194, 109, 32, 0.18);
      --chip-bg: rgba(19, 38, 58, 0.06);
      --chip-active: #13263a;
      --chip-active-ink: #f8fbff;
      --shadow: 0 16px 40px rgba(19, 38, 58, 0.12);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194, 109, 32, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.18), transparent 30%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      min-height: 100vh;
    }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 32px 28px 40px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1.2fr 1fr;
      align-items: stretch;
      margin-bottom: 24px;
    }}
    .hero-card, .panel {{
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero-card {{
      padding: 26px 28px;
    }}
    .hero-card h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}
    .hero-card p {{
      margin: 0;
      color: var(--muted);
      max-width: 62ch;
      line-height: 1.5;
    }}
    .summary-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .summary-card {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 20px;
    }}
    .summary-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      margin-bottom: 6px;
    }}
    .summary-value {{
      display: block;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .summary-subvalue {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .panel {{
      padding: 18px 20px 20px;
      margin-bottom: 18px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}
    .filters {{
      display: grid;
      gap: 14px;
    }}
    .filter-row {{
      display: grid;
      gap: 12px;
    }}
    .filter-row label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .filter-search {{
      width: 100%;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 12px 15px;
      font: inherit;
      background: rgba(255, 255, 255, 0.76);
      color: var(--ink);
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .chip {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--chip-bg);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, color 120ms ease;
    }}
    .chip:hover {{
      transform: translateY(-1px);
    }}
    .chip.active {{
      background: var(--chip-active);
      color: var(--chip-active-ink);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1180px;
    }}
    th, td {{
      padding: 12px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: rgba(255, 252, 247, 0.95);
      backdrop-filter: blur(10px);
      z-index: 1;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
    }}
    tbody tr:hover {{
      background: rgba(255, 255, 255, 0.38);
    }}
    .label-stack {{
      display: grid;
      gap: 3px;
    }}
    .label-stack span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .metric-bar {{
      position: relative;
      display: inline-block;
      min-width: 92px;
      padding: 5px 8px;
      border-radius: 999px;
      background:
        linear-gradient(90deg, var(--bar) 0, var(--bar) var(--pct, 0%), rgba(255,255,255,0) var(--pct, 0%), rgba(255,255,255,0) 100%),
        rgba(19, 38, 58, 0.04);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .metric-bar.eval {{
      background:
        linear-gradient(90deg, var(--bar-warm) 0, var(--bar-warm) var(--pct, 0%), rgba(255,255,255,0) var(--pct, 0%), rgba(255,255,255,0) 100%),
        rgba(19, 38, 58, 0.04);
    }}
    .metric-empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .rank-stack {{
      display: grid;
      gap: 3px;
    }}
    .rank-stack strong {{
      font-size: 13px;
    }}
    .rank-stack span {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .rank-tie {{
      color: var(--warm);
      font-weight: 600;
    }}
    details {{
      max-width: 360px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent-strong);
      font-weight: 700;
      list-style: none;
    }}
    summary::-webkit-details-marker {{
      display: none;
    }}
    .details-grid {{
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }}
    .detail-label {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .detail-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 4px;
      color: var(--ink);
    }}
    .detail-list li {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
    }}
    .empty-state {{
      padding: 24px 8px 8px;
      color: var(--muted);
      text-align: center;
    }}
    .footer-note {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    @media (max-width: 1100px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .summary-grid {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      main {{
        padding: 20px 14px 28px;
      }}
      .summary-grid {{
        grid-template-columns: 1fr;
      }}
      .panel, .hero-card {{
        padding: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-card">
        <h1>OCP Compile and Eval Benchmark Report</h1>
        <p>
          This report compiles each optimal-control example exactly once for every selected
          transcription and symbolic-function preset, then benchmarks cold-free objective,
          gradient, Jacobian, and Hessian evaluation at the example's initial guess without
          running a solve.
        </p>
      </div>
      <div class="summary-grid">
        <div class="summary-card">
          <span class="summary-label">Cases</span>
          <span class="summary-value">{case_count}</span>
          <span class="summary-subvalue">
            {problem_count} problems • {transcription_count} transcriptions • {preset_count} presets
          </span>
        </div>
        <div class="summary-card">
          <span class="summary-label">Eval Loop</span>
          <span class="summary-value">{eval_loop_label}</span>
          <span class="summary-subvalue">
            {eval_loop_detail}
          </span>
        </div>
        <div class="summary-card">
          <span class="summary-label">Compile Sum</span>
          <span class="summary-value">{total_compile_sum}</span>
          <span class="summary-subvalue">{slowest_compile}</span>
        </div>
        <div class="summary-card">
          <span class="summary-label">Slowest Evals</span>
          <span class="summary-value">{slowest_hessian}</span>
          <span class="summary-subvalue">{slowest_gradient}</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Filters</h2>
      <div class="filters">
        <div class="filter-row">
          <label for="search-input">Search</label>
          <input id="search-input" class="filter-search" type="search" placeholder="Filter problem, preset, or transcription..." />
        </div>
        <div class="filter-row">
          <label>Problem</label>
          <div class="chips" id="problem-filters">
            {problem_filters}
          </div>
        </div>
        <div class="filter-row">
          <label>Transcription</label>
          <div class="chips" id="transcription-filters">
            {transcription_filters}
          </div>
        </div>
        <div class="filter-row">
          <label>Preset</label>
          <div class="chips" id="preset-filters">
            {preset_filters}
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Preset Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Preset</th>
              <th>Cases</th>
              <th>Avg Compile</th>
              <th>Avg Grad</th>
              <th>Avg Eq Jac</th>
              <th>Avg Hess</th>
            </tr>
          </thead>
          <tbody>
            {preset_summary_html}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Best Strategy By Case</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Problem</th>
              <th>Transcription</th>
              <th>Best Symbolic</th>
              <th>Best JIT</th>
              <th>Best Compile</th>
              <th>Best Obj</th>
              <th>Best Grad</th>
              <th>Best Jac</th>
              <th>Best Hess</th>
            </tr>
          </thead>
          <tbody>
            {best_strategy_html}
          </tbody>
        </table>
      </div>
      <p class="footer-note">
        These winners are computed within each problem/transcription slice. Ties are shown as shared winners when timings are effectively equal.
      </p>
    </section>

    <section class="panel">
      <h2>Preset Win Counts</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Preset</th>
              <th>Symbolic</th>
              <th>JIT</th>
              <th>Compile</th>
              <th>Obj</th>
              <th>Grad</th>
              <th>Jac</th>
              <th>Hess</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {preset_win_html}
          </tbody>
        </table>
      </div>
      <p class="footer-note">
        Win counts are shared across ties. A strategy can receive credit from multiple metrics for the same case.
      </p>
    </section>

    <section class="panel">
      <h2>Per-Case Results</h2>
      <div class="table-wrap">
        <table id="results-table">
          <thead>
            <tr>
              <th>Problem</th>
              <th>Transcription</th>
              <th>Preset</th>
              <th>Opt</th>
              <th>Symbolic</th>
              <th>JIT</th>
              <th>Compile</th>
              <th>Obj</th>
              <th>Grad</th>
              <th>Eq Jac</th>
              <th>Ineq Jac</th>
              <th>Hess</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        <div id="empty-state" class="empty-state" hidden>No rows match the active filters.</div>
      </div>
      <p class="footer-note">
        Compile totals sum the reported symbolic stages, lowering, LLVM JIT, and any tracked OCP helper compilation. Evaluation timings are per-kernel wall-clock averages over the measured loop; the details panel also shows per-kernel standard deviation and a one-shot benchmark-point sanity snapshot. Benchmarks use the runtime-bounded NLP path with zero multipliers for the Hessian.
      </p>
    </section>
  </main>
  <script>
    const searchInput = document.getElementById('search-input');
    const rows = Array.from(document.querySelectorAll('#results-table tbody tr'));
    const emptyState = document.getElementById('empty-state');

    function installChips(containerId) {{
      return Array.from(document.querySelectorAll(`#${{containerId}} .chip`));
    }}

    const chipGroups = {{
      problem: installChips('problem-filters'),
      transcription: installChips('transcription-filters'),
      preset: installChips('preset-filters'),
    }};

    function activeValues(chips) {{
      return new Set(chips.filter((chip) => chip.classList.contains('active')).map((chip) => chip.dataset.value));
    }}

    function updateRows() {{
      const search = searchInput.value.trim().toLowerCase();
      const activeProblems = activeValues(chipGroups.problem);
      const activeTranscriptions = activeValues(chipGroups.transcription);
      const activePresets = activeValues(chipGroups.preset);
      let visible = 0;

      for (const row of rows) {{
        const haystack = row.textContent.toLowerCase();
        const matchesSearch = !search || haystack.includes(search);
        const matchesProblem = activeProblems.size === 0 || activeProblems.has(row.dataset.problem);
        const matchesTranscription =
          activeTranscriptions.size === 0 || activeTranscriptions.has(row.dataset.transcription);
        const matchesPreset = activePresets.size === 0 || activePresets.has(row.dataset.preset);
        const show = matchesSearch && matchesProblem && matchesTranscription && matchesPreset;
        row.hidden = !show;
        if (show) {{
          visible += 1;
        }}
      }}

      emptyState.hidden = visible !== 0;
    }}

    for (const chips of Object.values(chipGroups)) {{
      for (const chip of chips) {{
        chip.addEventListener('click', () => {{
          chip.classList.toggle('active');
          updateRows();
        }});
      }}
    }}

    searchInput.addEventListener('input', updateRows);
    updateRows();
  </script>
</body>
</html>"#,
        case_count = suite.records.len(),
        problem_count = problem_values.len(),
        transcription_count = transcription_values.len(),
        preset_count = preset_values.len(),
        eval_loop_label = if suite.eval_options.measured_iterations == 0
            && suite.eval_options.warmup_iterations == 0
        {
            "compile only".to_string()
        } else {
            format!("{}x", suite.eval_options.measured_iterations)
        },
        eval_loop_detail = if suite.eval_options.measured_iterations == 0
            && suite.eval_options.warmup_iterations == 0
        {
            "no eval warmup or timed iterations".to_string()
        } else {
            format!(
                "{} warmup iterations per kernel",
                suite.eval_options.warmup_iterations
            )
        },
        total_compile_sum = format_duration_seconds(Some(total_compile_sum), false),
        slowest_compile = format_summary_case("Slowest compile", slowest_compile, false),
        slowest_hessian = format_summary_case("Slowest hessian", slowest_hessian, true),
        slowest_gradient = format_summary_case("Slowest gradient", slowest_gradient, true),
        problem_filters = render_filter_chips(problem_values),
        transcription_filters = render_filter_chips(transcription_values),
        preset_filters = render_filter_chips(preset_values),
        preset_summary_html = preset_summary_html,
        best_strategy_html = best_strategy_html,
        preset_win_html = preset_win_html,
        rows_html = rows_html,
    )
}

#[derive(Clone, Copy)]
struct ReportScales {
    symbolic_total_max: Option<f64>,
    jit_total_max: Option<f64>,
    compile_total_max: Option<f64>,
    objective_value_max: Option<f64>,
    objective_gradient_max: Option<f64>,
    equality_jacobian_max: Option<f64>,
    inequality_jacobian_max: Option<f64>,
    hessian_max: Option<f64>,
}

impl ReportScales {
    fn from_records(records: &[OcpBenchmarkRecord]) -> Self {
        Self {
            symbolic_total_max: max_optional(
                records.iter().filter_map(|record| record.symbolic_total_s),
            ),
            jit_total_max: max_optional(records.iter().filter_map(|record| record.jit_total_s)),
            compile_total_max: max_optional(
                records.iter().filter_map(|record| record.compile_total_s),
            ),
            objective_value_max: max_optional(
                records
                    .iter()
                    .filter_map(|record| record.eval.objective_value.average_s),
            ),
            objective_gradient_max: max_optional(
                records
                    .iter()
                    .filter_map(|record| record.eval.objective_gradient.average_s),
            ),
            equality_jacobian_max: max_optional(records.iter().filter_map(|record| {
                record
                    .eval
                    .equality_jacobian_values
                    .as_ref()
                    .and_then(|summary| summary.average_s)
            })),
            inequality_jacobian_max: max_optional(records.iter().filter_map(|record| {
                record
                    .eval
                    .inequality_jacobian_values
                    .as_ref()
                    .and_then(|summary| summary.average_s)
            })),
            hessian_max: max_optional(
                records
                    .iter()
                    .filter_map(|record| record.eval.lagrangian_hessian_values.average_s),
            ),
        }
    }
}

struct PresetSummaryRow<'a> {
    preset_label: &'a str,
    preset_description: &'a str,
    case_count: usize,
    avg_compile_total_s: Option<f64>,
    avg_objective_gradient_s: Option<f64>,
    avg_equality_jacobian_s: Option<f64>,
    avg_hessian_s: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RankingMetric {
    Symbolic,
    Jit,
    Compile,
    Objective,
    Gradient,
    Jacobian,
    Hessian,
}

impl RankingMetric {
    const ALL: [Self; 7] = [
        Self::Symbolic,
        Self::Jit,
        Self::Compile,
        Self::Objective,
        Self::Gradient,
        Self::Jacobian,
        Self::Hessian,
    ];

    fn value(self, record: &OcpBenchmarkRecord) -> Option<f64> {
        match self {
            Self::Symbolic => record.symbolic_total_s,
            Self::Jit => record.jit_total_s,
            Self::Compile => record.compile_total_s,
            Self::Objective => record.eval.objective_value.average_s,
            Self::Gradient => record.eval.objective_gradient.average_s,
            Self::Jacobian => combined_jacobian_average(record),
            Self::Hessian => record.eval.lagrangian_hessian_values.average_s,
        }
    }

    fn eval_metric(self) -> bool {
        matches!(
            self,
            Self::Objective | Self::Gradient | Self::Jacobian | Self::Hessian
        )
    }
}

struct CaseGroup<'a> {
    problem_name: &'a str,
    transcription_label: &'a str,
    records: Vec<&'a OcpBenchmarkRecord>,
}

struct BestStrategyRow {
    problem_html: String,
    transcription_html: String,
    symbolic_html: String,
    jit_html: String,
    compile_html: String,
    objective_html: String,
    gradient_html: String,
    jacobian_html: String,
    hessian_html: String,
}

struct PresetWinRow<'a> {
    preset_label: &'a str,
    preset_description: &'a str,
    symbolic_wins: usize,
    jit_wins: usize,
    compile_wins: usize,
    objective_wins: usize,
    gradient_wins: usize,
    jacobian_wins: usize,
    hessian_wins: usize,
    total_wins: usize,
}

fn preset_summary_rows(suite: &OcpBenchmarkSuite) -> Vec<PresetSummaryRow<'static>> {
    let mut rows = Vec::new();
    for &preset in OcpBenchmarkPreset::all() {
        let records = suite
            .records
            .iter()
            .filter(|record| record.preset_id == preset.id())
            .collect::<Vec<_>>();
        if records.is_empty() {
            continue;
        }
        rows.push(PresetSummaryRow {
            preset_label: preset.label(),
            preset_description: preset.description(),
            case_count: records.len(),
            avg_compile_total_s: average_optional(
                records.iter().filter_map(|record| record.compile_total_s),
            ),
            avg_objective_gradient_s: average_optional(
                records
                    .iter()
                    .filter_map(|record| record.eval.objective_gradient.average_s),
            ),
            avg_equality_jacobian_s: average_optional(records.iter().filter_map(|record| {
                record
                    .eval
                    .equality_jacobian_values
                    .as_ref()
                    .and_then(|summary| summary.average_s)
            })),
            avg_hessian_s: average_optional(
                records
                    .iter()
                    .filter_map(|record| record.eval.lagrangian_hessian_values.average_s),
            ),
        });
    }
    rows
}

fn case_groups(records: &[OcpBenchmarkRecord]) -> Vec<CaseGroup<'_>> {
    let mut groups: Vec<CaseGroup<'_>> = Vec::new();
    for record in records {
        match groups.last_mut() {
            Some(group)
                if group.problem_name == record.problem_name
                    && group.transcription_label == record.transcription_label =>
            {
                group.records.push(record);
            }
            _ => groups.push(CaseGroup {
                problem_name: &record.problem_name,
                transcription_label: &record.transcription_label,
                records: vec![record],
            }),
        }
    }
    groups
}

fn best_records_for_metric<'a>(
    records: &[&'a OcpBenchmarkRecord],
    metric: RankingMetric,
) -> Option<(f64, Vec<&'a OcpBenchmarkRecord>)> {
    let ranked = records
        .iter()
        .filter_map(|record| metric.value(record).map(|value| (*record, value)))
        .collect::<Vec<_>>();
    let best = ranked
        .iter()
        .map(|(_, value)| *value)
        .min_by(|lhs, rhs| lhs.total_cmp(rhs))?;
    let winners = ranked
        .into_iter()
        .filter_map(|(record, value)| approximately_equal(value, best).then_some(record))
        .collect::<Vec<_>>();
    Some((best, winners))
}

fn render_winner_cell(records: &[&OcpBenchmarkRecord], metric: RankingMetric) -> String {
    let Some((value, winners)) = best_records_for_metric(records, metric) else {
        return "<span class=\"metric-empty\">n/a</span>".to_string();
    };
    let lead = winners[0];
    let tie_note = if winners.len() > 1 {
        let extra = winners
            .iter()
            .skip(1)
            .map(|record| escape_html(&record.preset_label))
            .collect::<Vec<_>>()
            .join(", ");
        format!("<span class=\"rank-tie\">tie with {extra}</span>")
    } else {
        String::new()
    };
    format!(
        "<div class=\"rank-stack\"><strong>{}</strong><span>{}</span>{}</div>",
        escape_html(&lead.preset_label),
        format_duration_seconds(Some(value), metric.eval_metric()),
        tie_note,
    )
}

fn best_strategy_by_case_rows(suite: &OcpBenchmarkSuite) -> Vec<BestStrategyRow> {
    case_groups(&suite.records)
        .into_iter()
        .map(|group| BestStrategyRow {
            problem_html: escape_html(group.problem_name),
            transcription_html: escape_html(group.transcription_label),
            symbolic_html: render_winner_cell(&group.records, RankingMetric::Symbolic),
            jit_html: render_winner_cell(&group.records, RankingMetric::Jit),
            compile_html: render_winner_cell(&group.records, RankingMetric::Compile),
            objective_html: render_winner_cell(&group.records, RankingMetric::Objective),
            gradient_html: render_winner_cell(&group.records, RankingMetric::Gradient),
            jacobian_html: render_winner_cell(&group.records, RankingMetric::Jacobian),
            hessian_html: render_winner_cell(&group.records, RankingMetric::Hessian),
        })
        .collect()
}

fn preset_win_rows(suite: &OcpBenchmarkSuite) -> Vec<PresetWinRow<'static>> {
    let groups = case_groups(&suite.records);
    let mut rows = OcpBenchmarkPreset::all()
        .iter()
        .copied()
        .filter(|preset| {
            suite
                .records
                .iter()
                .any(|record| record.preset_id == preset.id())
        })
        .map(|preset| PresetWinRow {
            preset_label: preset.label(),
            preset_description: preset.description(),
            symbolic_wins: 0,
            jit_wins: 0,
            compile_wins: 0,
            objective_wins: 0,
            gradient_wins: 0,
            jacobian_wins: 0,
            hessian_wins: 0,
            total_wins: 0,
        })
        .collect::<Vec<_>>();

    for group in &groups {
        for metric in RankingMetric::ALL {
            let Some((_, winners)) = best_records_for_metric(&group.records, metric) else {
                continue;
            };
            for winner in winners {
                if let Some(row) = rows
                    .iter_mut()
                    .find(|row| row.preset_label == winner.preset_label)
                {
                    match metric {
                        RankingMetric::Symbolic => row.symbolic_wins += 1,
                        RankingMetric::Jit => row.jit_wins += 1,
                        RankingMetric::Compile => row.compile_wins += 1,
                        RankingMetric::Objective => row.objective_wins += 1,
                        RankingMetric::Gradient => row.gradient_wins += 1,
                        RankingMetric::Jacobian => row.jacobian_wins += 1,
                        RankingMetric::Hessian => row.hessian_wins += 1,
                    }
                    row.total_wins += 1;
                }
            }
        }
    }

    rows.sort_by(|lhs, rhs| {
        rhs.total_wins
            .cmp(&lhs.total_wins)
            .then_with(|| rhs.compile_wins.cmp(&lhs.compile_wins))
            .then_with(|| rhs.gradient_wins.cmp(&lhs.gradient_wins))
            .then_with(|| lhs.preset_label.cmp(rhs.preset_label))
    });
    rows
}

fn summarize_helper_compile(stats: OcpHelperCompileStats) -> OcpHelperCompileSummary {
    let xdot_helper_s = duration_seconds(stats.xdot_helper_time);
    let multiple_shooting_arc_helper_s = duration_seconds(stats.multiple_shooting_arc_helper_time);
    OcpHelperCompileSummary {
        xdot_helper_s,
        multiple_shooting_arc_helper_s,
        xdot_helper_root_instructions: stats.xdot_helper_root_instructions,
        xdot_helper_total_instructions: stats.xdot_helper_total_instructions,
        multiple_shooting_arc_helper_root_instructions: stats
            .multiple_shooting_arc_helper_root_instructions,
        multiple_shooting_arc_helper_total_instructions: stats
            .multiple_shooting_arc_helper_total_instructions,
        total_s: sum_options([xdot_helper_s, multiple_shooting_arc_helper_s]),
    }
}

fn summarize_eval_benchmark(eval: NlpEvaluationBenchmark) -> OcpEvalBenchmarkSummary {
    OcpEvalBenchmarkSummary {
        benchmark_point: summarize_benchmark_point(eval.benchmark_point),
        objective_value: summarize_kernel(eval.objective_value),
        objective_gradient: summarize_kernel(eval.objective_gradient),
        equality_jacobian_values: eval.equality_jacobian_values.map(summarize_kernel),
        inequality_jacobian_values: eval.inequality_jacobian_values.map(summarize_kernel),
        lagrangian_hessian_values: summarize_kernel(eval.lagrangian_hessian_values),
    }
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

fn summarize_benchmark_point(point: NlpBenchmarkPointSummary) -> OcpBenchmarkPointSummary {
    OcpBenchmarkPointSummary {
        decision_inf_norm: point.decision_inf_norm,
        parameter_inf_norm: point.parameter_inf_norm,
        objective_value: point.objective_value,
        objective_finite: point.objective_finite,
        equality_inf_norm: point.equality_inf_norm,
        inequality_inf_norm: point.inequality_inf_norm,
    }
}

fn summarize_kernel(stats: KernelBenchmarkStats) -> KernelBenchmarkSummary {
    KernelBenchmarkSummary {
        output_len: stats.output_len,
        iterations: stats.iterations,
        total_s: stats.total_time.as_secs_f64(),
        average_s: stats.average_time().map(|duration| duration.as_secs_f64()),
        stddev_s: stats.stddev_time().map(|duration| duration.as_secs_f64()),
        min_s: stats.min_time.map(|duration| duration.as_secs_f64()),
        max_s: stats.max_time.map(|duration| duration.as_secs_f64()),
        preflight_finite: stats.preflight_output.finite,
        preflight_nonzero_count: stats.preflight_output.nonzero_count,
        preflight_max_abs: stats.preflight_output.max_abs,
    }
}

fn transcription_id(method: TranscriptionMethod) -> &'static str {
    match method {
        TranscriptionMethod::MultipleShooting => "multiple_shooting",
        TranscriptionMethod::DirectCollocation => "direct_collocation",
    }
}

const fn problem_label(problem_id: ProblemId) -> &'static str {
    match problem_id {
        ProblemId::OptimalDistanceGlider => "Optimal Distance Glider",
        ProblemId::AlbatrossDynamicSoaring => "Albatross Dynamic Soaring",
        ProblemId::LinearSManeuver => "Linear Point-to-Point S Maneuver",
        ProblemId::SailboatUpwind => "Sailboat Upwind",
        ProblemId::CraneTransfer => "Crane Transfer",
        ProblemId::HangingChainStatic => "Static Hanging Chain",
        ProblemId::RosenbrockVariants => "Rosenbrock Variants",
    }
}

fn transcription_label(
    method: TranscriptionMethod,
    family: Option<CollocationFamily>,
) -> &'static str {
    match (method, family) {
        (TranscriptionMethod::MultipleShooting, _) => "Multiple Shooting",
        (TranscriptionMethod::DirectCollocation, Some(CollocationFamily::GaussLegendre)) => {
            "Direct Collocation (Legendre)"
        }
        (TranscriptionMethod::DirectCollocation, Some(CollocationFamily::RadauIIA)) => {
            "Direct Collocation (Radau IIA)"
        }
        (TranscriptionMethod::DirectCollocation, None) => "Direct Collocation",
    }
}

fn opt_level_label(opt_level: LlvmOptimizationLevel) -> &'static str {
    opt_level.label()
}

fn render_compile_breakdown(record: &OcpBenchmarkRecord) -> String {
    let rows = [
        (
            "symbolic construction",
            record.compile.symbolic_construction_s,
        ),
        ("objective gradient", record.compile.objective_gradient_s),
        ("equality jacobian", record.compile.equality_jacobian_s),
        ("inequality jacobian", record.compile.inequality_jacobian_s),
        ("lagrangian assembly", record.compile.lagrangian_assembly_s),
        ("hessian generation", record.compile.hessian_generation_s),
        ("lowering", record.compile.lowering_s),
        ("llvm jit", record.compile.llvm_jit_s),
        ("symbolic total", record.symbolic_total_s),
        ("jit total", record.jit_total_s),
        ("compile total", record.compile_total_s),
    ];
    render_detail_list(&rows)
}

fn render_helper_breakdown(record: &OcpBenchmarkRecord) -> String {
    let rows = [
        ("xdot helper", record.helper_compile.xdot_helper_s),
        (
            "rk4 arc helper",
            record.helper_compile.multiple_shooting_arc_helper_s,
        ),
        ("helper total", record.helper_compile.total_s),
    ];
    render_detail_list(&rows)
}

fn render_shape_breakdown(record: &OcpBenchmarkRecord) -> String {
    format!(
        "<ul class=\"detail-list\">\
           <li><span>vars</span><span>{}</span></li>\
           <li><span>params</span><span>{}</span></li>\
           <li><span>equalities</span><span>{}</span></li>\
           <li><span>inequalities</span><span>{}</span></li>\
           <li><span>grad nnz</span><span>{}</span></li>\
           <li><span>eq jac nnz</span><span>{}</span></li>\
           <li><span>ineq jac nnz</span><span>{}</span></li>\
           <li><span>hessian nnz</span><span>{}</span></li>\
           <li><span>nlp kernels</span><span>{}</span></li>\
           <li><span>helper kernels</span><span>{}</span></li>\
           <li><span>call sites</span><span>{}</span></li>\
           <li><span>root inst.</span><span>{}</span></li>\
           <li><span>total inst.</span><span>{}</span></li>\
           <li><span>llvm subfunctions</span><span>{}</span></li>\
           <li><span>llvm call inst.</span><span>{}</span></li>\
         </ul>",
        record.nlp.variable_count,
        record.nlp.parameter_scalar_count,
        record.nlp.equality_count,
        record.nlp.inequality_count,
        record.nlp.objective_gradient_nnz,
        record.nlp.equality_jacobian_nnz,
        record.nlp.inequality_jacobian_nnz,
        record.nlp.hessian_nnz,
        record.nlp.nlp_kernel_count,
        record.nlp.helper_kernel_count,
        record.compile.call_site_count,
        record.compile.llvm_root_instructions_emitted,
        record.compile.llvm_total_instructions_emitted,
        record.compile.llvm_subfunctions_emitted,
        record.compile.llvm_call_instructions_emitted,
    )
}

fn render_eval_breakdown(record: &OcpBenchmarkRecord) -> String {
    let mut out = String::from("<ul class=\"detail-list\">");
    render_kernel_eval_detail(&mut out, "objective", &record.eval.objective_value);
    render_kernel_eval_detail(&mut out, "gradient", &record.eval.objective_gradient);
    if let Some(summary) = &record.eval.equality_jacobian_values {
        render_kernel_eval_detail(&mut out, "eq jacobian", summary);
    }
    if let Some(summary) = &record.eval.inequality_jacobian_values {
        render_kernel_eval_detail(&mut out, "ineq jacobian", summary);
    }
    render_kernel_eval_detail(&mut out, "hessian", &record.eval.lagrangian_hessian_values);
    out.push_str("</ul>");
    out
}

fn render_kernel_eval_detail(out: &mut String, label: &str, summary: &KernelBenchmarkSummary) {
    write!(
        out,
        "<li><span>{}</span><span>n={} avg {} σ {} min {} max {} | nz {} | max|y| {}</span></li>",
        escape_html(label),
        summary.iterations,
        format_duration_seconds(summary.average_s, true),
        format_duration_seconds(summary.stddev_s, true),
        format_duration_seconds(summary.min_s, true),
        format_duration_seconds(summary.max_s, true),
        summary.preflight_nonzero_count,
        format_scalar(summary.preflight_max_abs),
    )
    .expect("writing eval breakdown should not fail");
}

fn render_benchmark_point_breakdown(record: &OcpBenchmarkRecord) -> String {
    let point = &record.eval.benchmark_point;
    format!(
        "<ul class=\"detail-list\">\
           <li><span>decision ‖x‖∞</span><span>{}</span></li>\
           <li><span>parameter ‖p‖∞</span><span>{}</span></li>\
           <li><span>objective value</span><span>{}</span></li>\
           <li><span>objective finite</span><span>{}</span></li>\
           <li><span>eq residual ‖·‖∞</span><span>{}</span></li>\
           <li><span>ineq value ‖·‖∞</span><span>{}</span></li>\
         </ul>",
        format_scalar(point.decision_inf_norm),
        format_scalar(point.parameter_inf_norm),
        format_scalar(point.objective_value),
        if point.objective_finite { "yes" } else { "no" },
        format_optional_scalar(point.equality_inf_norm),
        format_optional_scalar(point.inequality_inf_norm),
    )
}

fn render_warning_breakdown(record: &OcpBenchmarkRecord) -> String {
    if record.compile.warnings.is_empty() {
        "<span class=\"metric-empty\">none</span>".to_string()
    } else {
        let mut out = String::from("<ul class=\"detail-list\">");
        for warning in &record.compile.warnings {
            write!(out, "<li><span>{}</span></li>", escape_html(warning))
                .expect("writing warnings should not fail");
        }
        out.push_str("</ul>");
        out
    }
}

fn render_detail_list(rows: &[(&str, Option<f64>)]) -> String {
    let mut out = String::from("<ul class=\"detail-list\">");
    for (label, value) in rows {
        write!(
            out,
            "<li><span>{}</span><span>{}</span></li>",
            escape_html(label),
            format_duration_seconds(*value, false),
        )
        .expect("writing detail list should not fail");
    }
    out.push_str("</ul>");
    out
}

fn format_scalar(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    if !(1.0e-3..1.0e4).contains(&abs) {
        format!("{value:.2e}")
    } else if abs >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    }
}

fn format_optional_scalar(value: Option<f64>) -> String {
    value.map(format_scalar).unwrap_or_else(|| "—".to_string())
}

fn metric_bar_cell(value: Option<f64>, max: Option<f64>, eval_metric: bool) -> String {
    match value {
        Some(value) => {
            let pct = max
                .filter(|max| *max > 0.0)
                .map(|max| (100.0 * value / max).clamp(0.0, 100.0))
                .unwrap_or(0.0);
            format!(
                "<span class=\"metric-bar{}\" style=\"--pct:{pct:.3}%\">{}</span>",
                if eval_metric { " eval" } else { "" },
                format_duration_seconds(Some(value), eval_metric),
            )
        }
        None => "<span class=\"metric-empty\">n/a</span>".to_string(),
    }
}

fn render_filter_chips(values: Vec<(&str, &str)>) -> String {
    let mut out = String::new();
    for (id, label) in values {
        write!(
            out,
            "<button class=\"chip\" type=\"button\" data-value=\"{}\">{}</button>",
            escape_attr(id),
            escape_html(label),
        )
        .expect("writing filter chips should not fail");
    }
    out
}

fn collect_unique_strings<'a>(
    iter: impl Iterator<Item = (&'a str, &'a str)>,
) -> Vec<(&'a str, &'a str)> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (id, label) in iter {
        if seen.insert(id) {
            out.push((id, label));
        }
    }
    out
}

fn format_summary_case(
    prefix: &str,
    case: Option<(&OcpBenchmarkRecord, f64)>,
    eval_metric: bool,
) -> String {
    case.map_or_else(
        || format!("{prefix}: n/a"),
        |(record, value)| {
            format!(
                "{prefix}: {} ({}, {}, {})",
                format_duration_seconds(Some(value), eval_metric),
                record.problem_name,
                record.transcription_label,
                record.preset_label,
            )
        },
    )
}

fn format_duration_seconds(value: Option<f64>, eval_metric: bool) -> String {
    match value {
        Some(value) => {
            let duration = Duration::from_secs_f64(value.max(0.0));
            format_duration(duration, eval_metric)
        }
        None => "n/a".to_string(),
    }
}

fn format_duration(duration: Duration, prefer_small_units: bool) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.2} s")
    } else if seconds >= 1.0e-3 {
        format!("{:.2} ms", seconds * 1.0e3)
    } else if seconds >= 1.0e-6 || prefer_small_units {
        format!("{:.2} us", seconds * 1.0e6)
    } else {
        format!("{:.2} ns", seconds * 1.0e9)
    }
}

fn duration_seconds(duration: Option<Duration>) -> Option<f64> {
    duration.map(|duration| duration.as_secs_f64())
}

fn sum_options<const N: usize>(values: [Option<f64>; N]) -> Option<f64> {
    let mut total = 0.0;
    let mut saw_value = false;
    for value in values.into_iter().flatten() {
        total += value;
        saw_value = true;
    }
    saw_value.then_some(total)
}

fn average_optional(iter: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in iter {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn max_optional(iter: impl Iterator<Item = f64>) -> Option<f64> {
    iter.max_by(|lhs, rhs| lhs.total_cmp(rhs))
}

fn approximately_equal(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 1.0e-12_f64.max(lhs.abs().max(rhs.abs()) * 1.0e-9)
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn escape_attr(input: &str) -> String {
    escape_html(input).replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_case_order_stays_within_problem_transcription_lane_before_next_lane() {
        let config = OcpBenchmarkSuiteConfig {
            problems: vec![ProblemId::OptimalDistanceGlider, ProblemId::LinearSManeuver],
            transcriptions: vec![
                TranscriptionMethod::MultipleShooting,
                TranscriptionMethod::DirectCollocation,
            ],
            presets: vec![OcpBenchmarkPreset::Baseline, OcpBenchmarkPreset::InlineAll],
            eval_options: NlpEvaluationBenchmarkOptions {
                warmup_iterations: 0,
                measured_iterations: 1,
            },
            jobs: 4,
        };

        let cases = build_benchmark_cases(&config);
        let actual = cases
            .iter()
            .map(|case| (case.problem_id, case.transcription, case.preset))
            .collect::<Vec<_>>();
        let expected = vec![
            (
                ProblemId::OptimalDistanceGlider,
                TranscriptionMethod::MultipleShooting,
                OcpBenchmarkPreset::Baseline,
            ),
            (
                ProblemId::OptimalDistanceGlider,
                TranscriptionMethod::MultipleShooting,
                OcpBenchmarkPreset::InlineAll,
            ),
            (
                ProblemId::OptimalDistanceGlider,
                TranscriptionMethod::DirectCollocation,
                OcpBenchmarkPreset::Baseline,
            ),
            (
                ProblemId::OptimalDistanceGlider,
                TranscriptionMethod::DirectCollocation,
                OcpBenchmarkPreset::InlineAll,
            ),
            (
                ProblemId::LinearSManeuver,
                TranscriptionMethod::MultipleShooting,
                OcpBenchmarkPreset::Baseline,
            ),
            (
                ProblemId::LinearSManeuver,
                TranscriptionMethod::MultipleShooting,
                OcpBenchmarkPreset::InlineAll,
            ),
            (
                ProblemId::LinearSManeuver,
                TranscriptionMethod::DirectCollocation,
                OcpBenchmarkPreset::Baseline,
            ),
            (
                ProblemId::LinearSManeuver,
                TranscriptionMethod::DirectCollocation,
                OcpBenchmarkPreset::InlineAll,
            ),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn render_report_contains_cases() {
        let suite = OcpBenchmarkSuite {
            eval_options: NlpEvaluationBenchmarkOptions {
                warmup_iterations: 1,
                measured_iterations: 2,
            },
            records: vec![OcpBenchmarkRecord {
                problem_id: ProblemId::LinearSManeuver,
                problem_name: "Linear Point-to-Point S Maneuver".to_string(),
                transcription_id: "multiple_shooting".to_string(),
                transcription_label: "Multiple Shooting".to_string(),
                preset_id: "baseline".to_string(),
                preset_label: "Baseline".to_string(),
                preset_description: "Current defaults".to_string(),
                opt_level: "O0".to_string(),
                compile: CompileReportSummary::default(),
                helper_compile: OcpHelperCompileSummary::default(),
                nlp: OcpNlpShapeSummary {
                    variable_count: 10,
                    parameter_scalar_count: 2,
                    equality_count: 3,
                    inequality_count: 4,
                    objective_gradient_nnz: 10,
                    equality_jacobian_nnz: 5,
                    inequality_jacobian_nnz: 6,
                    hessian_nnz: 7,
                    nlp_kernel_count: 8,
                    helper_kernel_count: 2,
                },
                eval: OcpEvalBenchmarkSummary {
                    benchmark_point: OcpBenchmarkPointSummary {
                        decision_inf_norm: 12.0,
                        parameter_inf_norm: 3.0,
                        objective_value: 1.25,
                        objective_finite: true,
                        equality_inf_norm: Some(0.5),
                        inequality_inf_norm: Some(0.25),
                    },
                    objective_value: KernelBenchmarkSummary {
                        output_len: 1,
                        iterations: 2,
                        total_s: 0.001,
                        average_s: Some(0.0005),
                        stddev_s: Some(0.00005),
                        min_s: Some(0.0004),
                        max_s: Some(0.0006),
                        preflight_finite: true,
                        preflight_nonzero_count: 1,
                        preflight_max_abs: 1.25,
                    },
                    objective_gradient: KernelBenchmarkSummary {
                        output_len: 10,
                        iterations: 2,
                        total_s: 0.004,
                        average_s: Some(0.002),
                        stddev_s: Some(0.0001),
                        min_s: Some(0.0019),
                        max_s: Some(0.0021),
                        preflight_finite: true,
                        preflight_nonzero_count: 7,
                        preflight_max_abs: 2.0,
                    },
                    equality_jacobian_values: None,
                    inequality_jacobian_values: None,
                    lagrangian_hessian_values: KernelBenchmarkSummary {
                        output_len: 7,
                        iterations: 2,
                        total_s: 0.006,
                        average_s: Some(0.003),
                        stddev_s: Some(0.0002),
                        min_s: Some(0.0028),
                        max_s: Some(0.0032),
                        preflight_finite: true,
                        preflight_nonzero_count: 5,
                        preflight_max_abs: 3.0,
                    },
                },
                symbolic_total_s: Some(0.1),
                jit_total_s: Some(0.2),
                compile_total_s: Some(0.3),
            }],
        };
        let html = render_ocp_benchmark_report(&suite);
        assert!(html.contains("Linear Point-to-Point S Maneuver"));
        assert!(html.contains("Baseline"));
        assert!(html.contains("Best Strategy By Case"));
        assert!(html.contains("Preset Win Counts"));
        assert!(html.contains("Per-Case Results"));
    }
}
