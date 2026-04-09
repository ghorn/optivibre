use std::env;
use std::fs;
use std::hint::black_box;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use anyhow::Result;
use bench_report::{
    CaseTimingStats, FunctionPairReportInput, HessianStrategyReport, PropertyStatus,
    PropertyVerdict, SuiteReport, TimingStats, suite_report_from_function_pairs,
};
use clap::Parser;
use examples_source::{
    AD_HESSIAN_SIZE, AdCostCase, ad_cost_cases, build_forward_sweep_augmented_function,
    build_forward_sweep_original_function, build_hessian_augmented_function,
    build_hessian_augmented_function_with_strategy, build_hessian_original_function,
    build_jacobian_augmented_function, build_jacobian_original_function,
    build_reverse_gradient_augmented_function, build_reverse_gradient_original_function,
    hessian_strategy_cases, hessian_strategy_expectation,
};
use sx_codegen::{LoweredFunction, lower_function};
use sx_codegen_llvm::{
    CompiledJitFunction, LlvmOptimizationLevel, LlvmTarget, emit_object_file_lowered,
};
use sx_core::HessianStrategy;
use tempfile::tempdir;

include!(concat!(env!("OUT_DIR"), "/generated_ad_cost_llvm_aot.rs"));

#[derive(Clone, Debug)]
struct SuiteOptions {
    samples: usize,
    target_ms: u64,
    json: bool,
}

#[derive(Debug, Parser)]
#[command(
    name = "ad_cost_suite",
    about = "Benchmark AD code generation and LLVM setup costs."
)]
struct AdCostSuiteCli {
    #[arg(long, default_value_t = 12, value_parser = parse_positive_usize)]
    samples: usize,
    #[arg(long = "target-ms", default_value_t = 30, value_parser = parse_positive_u64)]
    target_ms: u64,
    #[arg(long)]
    json: bool,
}

fn parse_args() -> SuiteOptions {
    let cli = AdCostSuiteCli::parse();
    SuiteOptions {
        samples: cli.samples,
        target_ms: cli.target_ms,
        json: cli.json,
    }
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

fn parse_positive_u64(value: &str) -> std::result::Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("invalid positive integer `{value}`"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn measure(samples: usize, target: Duration, mut op: impl FnMut()) -> TimingStats {
    op();

    let mut iterations = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            op();
        }
        if start.elapsed() >= target / 4 || iterations >= (1u64 << 20) {
            break;
        }
        iterations *= 2;
    }

    let mut per_iter_ns = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            op();
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
        per_iter_ns.push(elapsed_ns / iterations as f64);
    }
    per_iter_ns.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("timings must be finite"));
    let median_ns = per_iter_ns[per_iter_ns.len() / 2];

    TimingStats {
        samples,
        iterations_per_sample: iterations,
        min_ns: per_iter_ns[0],
        median_ns,
        max_ns: *per_iter_ns
            .last()
            .expect("timing sample set cannot be empty"),
    }
}

fn measure_once(mut op: impl FnMut()) -> TimingStats {
    let start = Instant::now();
    op();
    let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
    TimingStats {
        samples: 1,
        iterations_per_sample: 1,
        min_ns: elapsed_ns,
        median_ns: elapsed_ns,
        max_ns: elapsed_ns,
    }
}

fn lowered_op_count(function: &sx_core::SXFunction) -> usize {
    lower_function(function)
        .expect("lowering must succeed")
        .instructions
        .len()
}

fn fill_sequence(values: &mut [f64]) {
    for (idx, value) in values.iter_mut().enumerate() {
        *value = 0.05 + idx as f64 * 0.0025;
    }
}

fn llvm_opt_level() -> LlvmOptimizationLevel {
    if let Ok(value) = env::var("AD_COST_LLVM_OPT_LEVEL") {
        match value.to_ascii_lowercase().as_str() {
            "0" | "o0" => return LlvmOptimizationLevel::O0,
            "2" | "o2" => return LlvmOptimizationLevel::O2,
            "3" | "o3" => return LlvmOptimizationLevel::O3,
            "s" | "os" => return LlvmOptimizationLevel::Os,
            _ => {}
        }
    }
    if cfg!(debug_assertions) {
        LlvmOptimizationLevel::O0
    } else {
        LlvmOptimizationLevel::O3
    }
}

fn measure_llvm_setup(lowered: &LoweredFunction) -> TimingStats {
    measure_once(|| {
        let compiled = CompiledJitFunction::compile_lowered(lowered, llvm_opt_level()).unwrap();
        black_box(compiled.lowered().instructions.len());
    })
}

fn measure_llvm_aot_compile(lowered: &LoweredFunction) -> TimingStats {
    measure_once(|| {
        let tempdir = tempdir().unwrap();
        let object_path = tempdir.path().join(format!("{}.o", lowered.name));
        emit_object_file_lowered(&object_path, lowered, llvm_opt_level(), &LlvmTarget::Native)
            .unwrap();
        black_box(fs::metadata(&object_path).unwrap().len());
    })
}

fn fill_jit_inputs(lowered: &LoweredFunction, context: &mut sx_codegen_llvm::JitExecutionContext) {
    for slot_idx in 0..lowered.inputs.len() {
        let values = context.input_mut(slot_idx);
        if values.len() == 1 {
            values[0] = 0.75 + slot_idx as f64 * 0.1;
        } else {
            fill_sequence(values);
        }
    }
}

fn measure_llvm_eval(
    lowered: &LoweredFunction,
    derivative_output_slot: usize,
    samples: usize,
    target: Duration,
) -> TimingStats {
    let compiled = CompiledJitFunction::compile_lowered(lowered, llvm_opt_level()).unwrap();
    let mut context = compiled.create_context();
    fill_jit_inputs(lowered, &mut context);
    measure(samples, target, || {
        compiled.eval(&mut context);
        black_box(context.output(derivative_output_slot)[0]);
    })
}

fn measure_build_original(case: &AdCostCase) -> TimingStats {
    measure_once(|| match case.key {
        "reverse_gradient" => {
            let _ = black_box(build_reverse_gradient_original_function(case.size).unwrap());
        }
        "forward_sweep" => {
            let _ = black_box(build_forward_sweep_original_function(case.size).unwrap());
        }
        "jacobian" => {
            let _ = black_box(build_jacobian_original_function(case.size).unwrap());
        }
        "hessian" => {
            let _ = black_box(build_hessian_original_function(case.size).unwrap());
        }
        other => panic!("unknown case key: {other}"),
    })
}

fn measure_build_augmented(case: &AdCostCase) -> TimingStats {
    measure_once(|| match case.key {
        "reverse_gradient" => {
            let _ = black_box(build_reverse_gradient_augmented_function(case.size).unwrap());
        }
        "forward_sweep" => {
            let _ = black_box(build_forward_sweep_augmented_function(case.size).unwrap());
        }
        "jacobian" => {
            let _ = black_box(build_jacobian_augmented_function(case.size).unwrap());
        }
        "hessian" => {
            let _ = black_box(build_hessian_augmented_function(case.size).unwrap());
        }
        other => panic!("unknown case key: {other}"),
    })
}

fn measure_hessian_strategy_build(strategy: HessianStrategy) -> TimingStats {
    measure_once(|| {
        let _ = black_box(
            build_hessian_augmented_function_with_strategy(AD_HESSIAN_SIZE, strategy).unwrap(),
        );
    })
}

fn measure_eval_original(case: &AdCostCase, samples: usize, target: Duration) -> TimingStats {
    match case.key {
        "reverse_gradient" => {
            let mut ctx = ad_reverse_gradient_original_llvm_aot::AdReverseGradientOriginalLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.value);
            })
        }
        "forward_sweep" => {
            let mut ctx =
                ad_forward_sweep_original_llvm_aot::AdForwardSweepOriginalLlvmAotContext {
                    t: 0.75,
                    ..Default::default()
                };
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.outputs[0]);
            })
        }
        "jacobian" => {
            let mut ctx =
                ad_jacobian_original_llvm_aot::AdJacobianOriginalLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.outputs[0]);
            })
        }
        "hessian" => {
            let mut ctx = ad_hessian_original_llvm_aot::AdHessianOriginalLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.value);
            })
        }
        other => panic!("unknown case key: {other}"),
    }
}

fn measure_eval_augmented(case: &AdCostCase, samples: usize, target: Duration) -> TimingStats {
    match case.key {
        "reverse_gradient" => {
            let mut ctx = ad_reverse_gradient_with_gradient_llvm_aot::AdReverseGradientWithGradientLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.gradient[0]);
            })
        }
        "forward_sweep" => {
            let mut ctx = ad_forward_sweep_with_directional_llvm_aot::AdForwardSweepWithDirectionalLlvmAotContext {
                t: 0.75,
                ..Default::default()
            };
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.directional[0]);
            })
        }
        "jacobian" => {
            let mut ctx =
                ad_jacobian_with_jacobian_llvm_aot::AdJacobianWithJacobianLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.jacobian[0]);
            })
        }
        "hessian" => {
            let mut ctx =
                ad_hessian_with_hessian_llvm_aot::AdHessianWithHessianLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.hessian[0]);
            })
        }
        other => panic!("unknown case key: {other}"),
    }
}

fn measure_hessian_strategy_eval(
    strategy: HessianStrategy,
    samples: usize,
    target: Duration,
) -> TimingStats {
    match strategy {
        HessianStrategy::LowerTriangleByColumn => {
            let mut ctx = ad_hessian_with_hessian_by_column_llvm_aot::AdHessianWithHessianByColumnLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.hessian[0]);
            })
        }
        HessianStrategy::LowerTriangleSelectedOutputs => {
            let mut ctx = ad_hessian_with_hessian_selected_outputs_llvm_aot::AdHessianWithHessianSelectedOutputsLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.hessian[0]);
            })
        }
        HessianStrategy::LowerTriangleColored => {
            let mut ctx = ad_hessian_with_hessian_colored_llvm_aot::AdHessianWithHessianColoredLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            measure(samples, target, || {
                ctx.eval();
                black_box(ctx.hessian[0]);
            })
        }
    }
}

fn evaluate_hessian_strategy(strategy: HessianStrategy) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    match strategy {
        HessianStrategy::LowerTriangleByColumn => {
            let mut ctx = ad_hessian_with_hessian_by_column_llvm_aot::AdHessianWithHessianByColumnLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            ctx.eval();
            (
                ad_hessian_with_hessian_by_column_llvm_aot::HESSIAN_COL_PTRS.to_vec(),
                ad_hessian_with_hessian_by_column_llvm_aot::HESSIAN_ROW_INDICES.to_vec(),
                ctx.hessian,
            )
        }
        HessianStrategy::LowerTriangleSelectedOutputs => {
            let mut ctx = ad_hessian_with_hessian_selected_outputs_llvm_aot::AdHessianWithHessianSelectedOutputsLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            ctx.eval();
            (
                ad_hessian_with_hessian_selected_outputs_llvm_aot::HESSIAN_COL_PTRS.to_vec(),
                ad_hessian_with_hessian_selected_outputs_llvm_aot::HESSIAN_ROW_INDICES.to_vec(),
                ctx.hessian,
            )
        }
        HessianStrategy::LowerTriangleColored => {
            let mut ctx = ad_hessian_with_hessian_colored_llvm_aot::AdHessianWithHessianColoredLlvmAotContext::default();
            fill_sequence(&mut ctx.x);
            ctx.eval();
            (
                ad_hessian_with_hessian_colored_llvm_aot::HESSIAN_COL_PTRS.to_vec(),
                ad_hessian_with_hessian_colored_llvm_aot::HESSIAN_ROW_INDICES.to_vec(),
                ctx.hessian,
            )
        }
    }
}

fn property_rows(
    case: &AdCostCase,
    original_ops: usize,
    augmented_ops: usize,
) -> Vec<PropertyStatus> {
    let expectations = case.expectations();
    let ratio = augmented_ops as f64 / original_ops as f64;
    let normalized_ratio = ratio / case.sweep_count as f64;
    let mut rows = vec![PropertyStatus {
        key: format!("{}.baseline", case.key),
        description: "Exact lowered op counts stay pinned.".into(),
        verdict: if original_ops == expectations.exact_original_ops
            && augmented_ops == expectations.exact_augmented_ops
        {
            PropertyVerdict::Pass
        } else {
            PropertyVerdict::Fail
        },
        result: format!("({}, {})", original_ops, augmented_ops),
        expectation: format!(
            "== ({}, {})",
            expectations.exact_original_ops, expectations.exact_augmented_ops
        ),
    }];

    if let Some(limit) = expectations.directional_ratio_limit {
        rows.push(PropertyStatus {
            key: format!("{}.constant_factor", case.key),
            description: "Directional AD stays within a constant-factor op budget.".into(),
            verdict: verdict_for_limit(ratio, limit),
            result: format!("{ratio:.3}x"),
            expectation: format!("<= {limit:.3}x"),
        });
    }
    if let Some(limit) = expectations.normalized_ratio_limit {
        rows.push(PropertyStatus {
            key: format!("{}.per_sweep", case.key),
            description: "Higher-order AD cost stays bounded per sweep.".into(),
            verdict: verdict_for_limit(normalized_ratio, limit),
            result: format!("{normalized_ratio:.3}x"),
            expectation: format!("<= {limit:.3}x"),
        });
    }

    rows
}

fn hessian_strategy_property_rows() -> Vec<PropertyStatus> {
    let mut rows = vec![PropertyStatus {
        key: "hessian_strategy.default".into(),
        description: "The default Hessian strategy stays pinned.".into(),
        verdict: if HessianStrategy::default() == HessianStrategy::LowerTriangleSelectedOutputs {
            PropertyVerdict::Pass
        } else {
            PropertyVerdict::Fail
        },
        result: HessianStrategy::default().label().into(),
        expectation: HessianStrategy::LowerTriangleSelectedOutputs.label().into(),
    }];

    for strategy_case in hessian_strategy_cases(AD_HESSIAN_SIZE)
        .expect("Hessian strategy cases must build for baseline checks")
    {
        let exact_ops = lowered_op_count(&strategy_case.function);
        let expected_ops = hessian_strategy_expectation(strategy_case.strategy).exact_ops;
        rows.push(PropertyStatus {
            key: format!("hessian_strategy.{}.baseline", strategy_case.strategy.key()),
            description: "Exact lowered op count stays pinned for this Hessian strategy.".into(),
            verdict: if exact_ops == expected_ops {
                PropertyVerdict::Pass
            } else {
                PropertyVerdict::Fail
            },
            result: exact_ops.to_string(),
            expectation: format!("== {expected_ops}"),
        });
    }

    let (reference_col_ptrs, reference_rows, reference_values) =
        evaluate_hessian_strategy(HessianStrategy::LowerTriangleByColumn);
    for strategy in HessianStrategy::ALL {
        if strategy == HessianStrategy::LowerTriangleByColumn {
            continue;
        }
        let (candidate_col_ptrs, candidate_rows, candidate_values) =
            evaluate_hessian_strategy(strategy);
        let same_pattern =
            reference_col_ptrs == candidate_col_ptrs && reference_rows == candidate_rows;
        let same_values = reference_values.len() == candidate_values.len()
            && reference_values
                .iter()
                .zip(candidate_values.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1e-12);
        rows.push(PropertyStatus {
            key: format!("hessian_strategy.{}.equivalence", strategy.key()),
            description:
                "Generated Hessian values and lower-triangle CCS pattern match strategy 1.".into(),
            verdict: if same_pattern && same_values {
                PropertyVerdict::Pass
            } else {
                PropertyVerdict::Fail
            },
            result: if same_pattern && same_values {
                "matched".into()
            } else {
                "mismatch".into()
            },
            expectation: "== strategy 1".into(),
        });
    }

    rows
}

fn verdict_for_limit(result: f64, limit: f64) -> PropertyVerdict {
    if result > limit {
        PropertyVerdict::Fail
    } else if result > 0.8 * limit {
        PropertyVerdict::Warn
    } else {
        PropertyVerdict::Pass
    }
}

fn collect_report(options: &SuiteOptions) -> SuiteReport {
    let target = Duration::from_millis(options.target_ms);
    let mut function_pairs = Vec::new();
    let mut properties = Vec::new();

    for case in ad_cost_cases().expect("AD cost cases must build") {
        let expectations = case.expectations();
        let original_lowered =
            lower_function(&case.original).expect("lowering primal case must succeed");
        let augmented_lowered =
            lower_function(&case.augmented).expect("lowering augmented case must succeed");
        let original_ops = original_lowered.instructions.len();
        let augmented_ops = augmented_lowered.instructions.len();
        properties.extend(property_rows(&case, original_ops, augmented_ops));
        let derivative_output_slot = case
            .augmented
            .outputs()
            .len()
            .checked_sub(1)
            .expect("augmented case must have at least one output");
        let timings = CaseTimingStats {
            build_original: measure_build_original(&case),
            build_augmented: measure_build_augmented(&case),
            eval_original: measure_eval_original(&case, options.samples, target),
            eval_augmented: measure_eval_augmented(&case, options.samples, target),
            llvm_aot_compile_original: measure_llvm_aot_compile(&original_lowered),
            llvm_aot_compile_augmented: measure_llvm_aot_compile(&augmented_lowered),
            llvm_setup_original: measure_llvm_setup(&original_lowered),
            llvm_setup_augmented: measure_llvm_setup(&augmented_lowered),
            llvm_eval_original: measure_llvm_eval(&original_lowered, 0, options.samples, target),
            llvm_eval_augmented: measure_llvm_eval(
                &augmented_lowered,
                derivative_output_slot,
                options.samples,
                target,
            ),
        };
        function_pairs.push(FunctionPairReportInput {
            key: case.key.into(),
            description: expectations.description.into(),
            size: case.size,
            sweep_count: case.sweep_count,
            original: case.original,
            augmented: case.augmented,
            timings,
        });
    }
    let strategy_cases = hessian_strategy_cases(AD_HESSIAN_SIZE)
        .expect("Hessian strategy comparison cases must build");
    let by_column_ops = strategy_cases
        .iter()
        .find(|case| case.strategy == HessianStrategy::LowerTriangleByColumn)
        .map(|case| lowered_op_count(&case.function))
        .expect("strategy 1 must exist");
    let hessian_strategies = strategy_cases
        .into_iter()
        .map(|case| {
            let op_count = lowered_op_count(&case.function);
            HessianStrategyReport {
                key: case.strategy.key().into(),
                label: case.strategy.label().into(),
                description: case.strategy.description().into(),
                is_default: case.strategy == HessianStrategy::default(),
                op_count,
                ratio_to_by_column: op_count as f64 / by_column_ops as f64,
                build: measure_hessian_strategy_build(case.strategy),
                eval: measure_hessian_strategy_eval(case.strategy, options.samples, target),
            }
        })
        .collect();
    properties.extend(hessian_strategy_property_rows());

    suite_report_from_function_pairs(
        profile_name(),
        options.samples,
        options.target_ms,
        function_pairs,
        hessian_strategies,
        properties,
    )
    .expect("suite report must build")
}

fn main() -> Result<()> {
    let options = parse_args();
    let report = collect_report(&options);
    let _ = options.json;
    serde_json::to_writer_pretty(io::stdout(), &report)?;
    io::stdout().write_all(b"\n")?;
    Ok(())
}
