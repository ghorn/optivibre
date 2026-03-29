use anyhow::Result;
use bench_report::{
    CaseTimingStats, FunctionPairReportInput, MarkdownReportOptions, PropertyStatus,
    PropertyVerdict, TimingStats, render_markdown_report_with_options,
    suite_report_from_function_pairs,
};
use sx_core::{NamedMatrix, SXFunction, SXMatrix};

fn fixed_timing(median_ns: f64) -> TimingStats {
    TimingStats {
        samples: 8,
        iterations_per_sample: 1_024,
        min_ns: median_ns * 0.95,
        median_ns,
        max_ns: median_ns * 1.05,
    }
}

fn build_original_function() -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let value = SXMatrix::scalar(x.nz(0) * x.nz(1) + x.nz(0) / 2.0);
    SXFunction::new(
        "production_value",
        vec![NamedMatrix::new("x", x)?],
        vec![NamedMatrix::new("value", value)?],
    )
    .map_err(Into::into)
}

fn build_augmented_function() -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let value = SXMatrix::scalar(x.nz(0) * x.nz(1) + x.nz(0) / 2.0);
    let gradient = value.gradient(&x)?;
    SXFunction::new(
        "production_value_with_gradient",
        vec![NamedMatrix::new("x", x)?],
        vec![
            NamedMatrix::new("value", value)?,
            NamedMatrix::new("gradient", gradient)?,
        ],
    )
    .map_err(Into::into)
}

fn main() -> Result<()> {
    let debug = suite_report_from_function_pairs(
        "debug",
        8,
        25,
        vec![FunctionPairReportInput {
            key: "production_value".into(),
            description: "Scalar value plus gradient for a custom production function.".into(),
            size: 2,
            sweep_count: 1,
            original: build_original_function()?,
            augmented: build_augmented_function()?,
            timings: CaseTimingStats {
                build_original: fixed_timing(40_000.0),
                build_augmented: fixed_timing(75_000.0),
                eval_original: fixed_timing(140.0),
                eval_augmented: fixed_timing(320.0),
                llvm_aot_compile_original: fixed_timing(180_000.0),
                llvm_aot_compile_augmented: fixed_timing(300_000.0),
                llvm_setup_original: fixed_timing(240_000.0),
                llvm_setup_augmented: fixed_timing(410_000.0),
                llvm_eval_original: fixed_timing(180.0),
                llvm_eval_augmented: fixed_timing(430.0),
            },
        }],
        Vec::new(),
        vec![PropertyStatus {
            key: "production_value.smoke".into(),
            description: "Custom report input assembled successfully.".into(),
            verdict: PropertyVerdict::Pass,
            result: "ready".into(),
            expectation: "== ready".into(),
        }],
    )?;

    let release = suite_report_from_function_pairs(
        "release",
        8,
        25,
        vec![FunctionPairReportInput {
            key: "production_value".into(),
            description: "Scalar value plus gradient for a custom production function.".into(),
            size: 2,
            sweep_count: 1,
            original: build_original_function()?,
            augmented: build_augmented_function()?,
            timings: CaseTimingStats {
                build_original: fixed_timing(8_000.0),
                build_augmented: fixed_timing(14_000.0),
                eval_original: fixed_timing(48.0),
                eval_augmented: fixed_timing(110.0),
                llvm_aot_compile_original: fixed_timing(80_000.0),
                llvm_aot_compile_augmented: fixed_timing(140_000.0),
                llvm_setup_original: fixed_timing(95_000.0),
                llvm_setup_augmented: fixed_timing(170_000.0),
                llvm_eval_original: fixed_timing(65.0),
                llvm_eval_augmented: fixed_timing(150.0),
            },
        }],
        Vec::new(),
        vec![PropertyStatus {
            key: "production_value.smoke".into(),
            description: "Custom report input assembled successfully.".into(),
            verdict: PropertyVerdict::Pass,
            result: "ready".into(),
            expectation: "== ready".into(),
        }],
    )?;

    let markdown = render_markdown_report_with_options(
        &MarkdownReportOptions {
            title: "Minimal Bench Report".into(),
            command: Some("cargo run -p bench_report --example minimal".into()),
            include_lowered_op_explanation: false,
        },
        &debug,
        &release,
    );

    println!("{markdown}");
    Ok(())
}
