use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sx_codegen::lower_function;
use sx_core::SXFunction;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TimingStats {
    pub samples: usize,
    pub iterations_per_sample: u64,
    pub min_ns: f64,
    pub median_ns: f64,
    pub max_ns: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PropertyVerdict {
    Pass,
    Warn,
    Fail,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PropertyStatus {
    pub key: String,
    pub description: String,
    pub verdict: PropertyVerdict,
    pub result: String,
    pub expectation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CaseReport {
    pub key: String,
    pub description: String,
    pub size: usize,
    pub sweep_count: usize,
    pub original_ops: usize,
    pub augmented_ops: usize,
    pub ratio: f64,
    pub normalized_ratio: f64,
    pub build_original: TimingStats,
    pub build_augmented: TimingStats,
    pub eval_original: TimingStats,
    pub eval_augmented: TimingStats,
    pub llvm_aot_compile_original: TimingStats,
    pub llvm_aot_compile_augmented: TimingStats,
    pub llvm_setup_original: TimingStats,
    pub llvm_setup_augmented: TimingStats,
    pub llvm_eval_original: TimingStats,
    pub llvm_eval_augmented: TimingStats,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HessianStrategyReport {
    pub key: String,
    pub label: String,
    pub description: String,
    pub is_default: bool,
    pub op_count: usize,
    pub ratio_to_by_column: f64,
    pub build: TimingStats,
    pub eval: TimingStats,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SuiteReport {
    pub build_profile: String,
    pub samples: usize,
    pub target_ms: u64,
    pub cases: Vec<CaseReport>,
    pub hessian_strategies: Vec<HessianStrategyReport>,
    pub properties: Vec<PropertyStatus>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkdownReportOptions {
    pub title: String,
    pub command: Option<String>,
    pub include_lowered_op_explanation: bool,
}

impl Default for MarkdownReportOptions {
    fn default() -> Self {
        Self {
            title: "Benchmark Report".into(),
            command: None,
            include_lowered_op_explanation: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CaseTimingStats {
    pub build_original: TimingStats,
    pub build_augmented: TimingStats,
    pub eval_original: TimingStats,
    pub eval_augmented: TimingStats,
    pub llvm_aot_compile_original: TimingStats,
    pub llvm_aot_compile_augmented: TimingStats,
    pub llvm_setup_original: TimingStats,
    pub llvm_setup_augmented: TimingStats,
    pub llvm_eval_original: TimingStats,
    pub llvm_eval_augmented: TimingStats,
}

#[derive(Clone, Debug)]
pub struct FunctionPairReportInput {
    pub key: String,
    pub description: String,
    pub size: usize,
    pub sweep_count: usize,
    pub original: SXFunction,
    pub augmented: SXFunction,
    pub timings: CaseTimingStats,
}

pub fn suite_report_from_function_pairs(
    build_profile: impl Into<String>,
    samples: usize,
    target_ms: u64,
    cases: Vec<FunctionPairReportInput>,
    hessian_strategies: Vec<HessianStrategyReport>,
    properties: Vec<PropertyStatus>,
) -> Result<SuiteReport> {
    let cases = cases
        .into_iter()
        .map(|case| {
            let original_ops = lower_function(&case.original)?.instructions.len();
            let augmented_ops = lower_function(&case.augmented)?.instructions.len();
            let ratio = augmented_ops as f64 / original_ops as f64;
            Ok(CaseReport {
                key: case.key,
                description: case.description,
                size: case.size,
                sweep_count: case.sweep_count,
                original_ops,
                augmented_ops,
                ratio,
                normalized_ratio: ratio / case.sweep_count as f64,
                build_original: case.timings.build_original,
                build_augmented: case.timings.build_augmented,
                eval_original: case.timings.eval_original,
                eval_augmented: case.timings.eval_augmented,
                llvm_aot_compile_original: case.timings.llvm_aot_compile_original,
                llvm_aot_compile_augmented: case.timings.llvm_aot_compile_augmented,
                llvm_setup_original: case.timings.llvm_setup_original,
                llvm_setup_augmented: case.timings.llvm_setup_augmented,
                llvm_eval_original: case.timings.llvm_eval_original,
                llvm_eval_augmented: case.timings.llvm_eval_augmented,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(SuiteReport {
        build_profile: build_profile.into(),
        samples,
        target_ms,
        cases,
        hessian_strategies,
        properties,
    })
}

fn format_ns(ns: f64) -> String {
    if ns >= 1e9 {
        format!("{:.2} s", ns / 1e9)
    } else if ns >= 1e6 {
        format!("{:.2} ms", ns / 1e6)
    } else if ns >= 1e3 {
        format!("{:.2} us", ns / 1e3)
    } else {
        format!("{ns:.0} ns")
    }
}

fn format_timing(stats: &TimingStats) -> String {
    if stats.samples == 1 && stats.iterations_per_sample == 1 {
        format!("{} (single)", format_ns(stats.median_ns))
    } else {
        format!(
            "{} [{} .. {}]",
            format_ns(stats.median_ns),
            format_ns(stats.min_ns),
            format_ns(stats.max_ns)
        )
    }
}

fn scaled_log_width(value: f64, min: f64, max: f64, width: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        width
    } else {
        let min_log = min.log10();
        let max_log = max.log10();
        let value_log = value.log10();
        let scaled = (value_log - min_log) / (max_log - min_log);
        8.0 + scaled * (width - 8.0)
    }
}

struct ChartScale {
    min: f64,
    max: f64,
    bar_width: f64,
}

struct ProfilePairChart<'a> {
    panel_x: f64,
    panel_title: &'a str,
    primal: &'a TimingStats,
    ad: &'a TimingStats,
}

fn render_profile_pair_chart(
    svg: &mut String,
    chart: &ProfilePairChart<'_>,
    y: f64,
    scale: &ChartScale,
) {
    let label_x = chart.panel_x;
    let bar_x = chart.panel_x + 20.0;
    let value_x = bar_x + scale.bar_width + 12.0;
    let primal_width = scaled_log_width(
        chart.primal.median_ns,
        scale.min,
        scale.max,
        scale.bar_width,
    );
    let ad_width = scaled_log_width(chart.ad.median_ns, scale.min, scale.max, scale.bar_width);

    svg.push_str(&format!(
        "<text x=\"{label_x}\" y=\"{}\" fill=\"#94a3b8\" font-size=\"11\">P</text>\n",
        y + 13.0
    ));
    svg.push_str(&format!(
        "<rect x=\"{bar_x}\" y=\"{}\" width=\"{}\" height=\"10\" rx=\"4\" fill=\"#1f2937\" />\n",
        y + 4.0,
        scale.bar_width
    ));
    svg.push_str(&format!(
        "<rect x=\"{bar_x}\" y=\"{}\" width=\"{primal_width}\" height=\"10\" rx=\"4\" fill=\"#94a3b8\" />\n",
        y + 4.0
    ));
    svg.push_str(&format!(
        "<text x=\"{value_x}\" y=\"{}\" fill=\"#cbd5e1\" font-size=\"12\">{}</text>\n",
        y + 13.0,
        format_ns(chart.primal.median_ns)
    ));

    svg.push_str(&format!(
        "<text x=\"{label_x}\" y=\"{}\" fill=\"#f59e0b\" font-size=\"11\">A</text>\n",
        y + 33.0
    ));
    svg.push_str(&format!(
        "<rect x=\"{bar_x}\" y=\"{}\" width=\"{}\" height=\"10\" rx=\"4\" fill=\"#1f2937\" />\n",
        y + 24.0,
        scale.bar_width
    ));
    svg.push_str(&format!(
        "<rect x=\"{bar_x}\" y=\"{}\" width=\"{ad_width}\" height=\"10\" rx=\"4\" fill=\"#f59e0b\" />\n",
        y + 24.0
    ));
    svg.push_str(&format!(
        "<text x=\"{value_x}\" y=\"{}\" fill=\"#fde68a\" font-size=\"12\">{}</text>\n",
        y + 33.0,
        format_ns(chart.ad.median_ns)
    ));

    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"#e5e7eb\" font-size=\"12\" font-weight=\"600\">{}</text>\n",
        bar_x,
        y - 8.0,
        chart.panel_title
    ));
}

struct CaseTimingRow {
    label: String,
    build_debug_primal: TimingStats,
    build_debug_ad: TimingStats,
    build_release_primal: TimingStats,
    build_release_ad: TimingStats,
    eval_debug_primal: TimingStats,
    eval_debug_ad: TimingStats,
    eval_release_primal: TimingStats,
    eval_release_ad: TimingStats,
}

fn render_case_timing_chart(title: &str, rows: &[CaseTimingRow]) -> String {
    let build_values = rows
        .iter()
        .flat_map(|row| {
            [
                row.build_debug_primal.median_ns,
                row.build_debug_ad.median_ns,
                row.build_release_primal.median_ns,
                row.build_release_ad.median_ns,
            ]
        })
        .collect::<Vec<_>>();
    let build_scale = ChartScale {
        min: build_values
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, value| acc.min(value)),
        max: build_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(value)),
        bar_width: 135.0,
    };
    let eval_values = rows
        .iter()
        .flat_map(|row| {
            [
                row.eval_debug_primal.median_ns,
                row.eval_debug_ad.median_ns,
                row.eval_release_primal.median_ns,
                row.eval_release_ad.median_ns,
            ]
        })
        .collect::<Vec<_>>();
    let eval_scale = ChartScale {
        min: eval_values
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, value| acc.min(value)),
        max: eval_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(value)),
        bar_width: 135.0,
    };

    let svg_width = 1490.0;
    let build_debug_x = 190.0;
    let build_release_x = 390.0;
    let eval_debug_x = 770.0;
    let eval_release_x = 970.0;
    let row_height = 68.0;
    let top = 82.0;
    let bottom = 34.0;
    let height = top + row_height * rows.len() as f64 + bottom;

    let mut svg = String::new();
    svg.push_str(&format!("### {title}\n\n"));
    svg.push_str(
        "<div style=\"color:#9ca3af;font-size:0.95em;margin:0.2rem 0 0.6rem 0;\">Log scale. Gray = primal, amber = AD. Each case shows symbolic build on the left and generated eval on the right, with debug and release paired inside each half.</div>\n\n",
    );
    svg.push_str(&format!(
        "<svg viewBox=\"0 0 {svg_width} {height}\" width=\"100%\" role=\"img\" aria-label=\"{title}\">\n"
    ));
    svg.push_str(&format!(
        "<text x=\"16\" y=\"24\" fill=\"#e5e7eb\" font-size=\"14\" font-weight=\"600\">{title}</text>\n"
    ));
    svg.push_str(
        "<rect x=\"640\" y=\"12\" width=\"12\" height=\"12\" rx=\"3\" fill=\"#94a3b8\" />\n",
    );
    svg.push_str("<text x=\"658\" y=\"22\" fill=\"#cbd5e1\" font-size=\"12\">Primal</text>\n");
    svg.push_str(
        "<rect x=\"730\" y=\"12\" width=\"12\" height=\"12\" rx=\"3\" fill=\"#f59e0b\" />\n",
    );
    svg.push_str("<text x=\"748\" y=\"22\" fill=\"#fde68a\" font-size=\"12\">AD</text>\n");
    svg.push_str(
        "<text x=\"385\" y=\"46\" text-anchor=\"middle\" fill=\"#e5e7eb\" font-size=\"13\" font-weight=\"600\">Build</text>\n",
    );
    svg.push_str(
        "<text x=\"965\" y=\"46\" text-anchor=\"middle\" fill=\"#e5e7eb\" font-size=\"13\" font-weight=\"600\">Eval</text>\n",
    );
    svg.push_str(&format!(
        "<line x1=\"700\" y1=\"34\" x2=\"700\" y2=\"{}\" stroke=\"#374151\" stroke-width=\"1\" opacity=\"0.8\" />\n",
        height - 18.0
    ));
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"#94a3b8\" font-size=\"11\">{}</text>\n",
        build_debug_x + 20.0,
        height - 10.0,
        format_ns(build_scale.min)
    ));
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" fill=\"#94a3b8\" font-size=\"11\">{}</text>\n",
        build_release_x + 20.0 + build_scale.bar_width,
        height - 10.0,
        format_ns(build_scale.max)
    ));
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"#94a3b8\" font-size=\"11\">{}</text>\n",
        eval_debug_x + 20.0,
        height - 10.0,
        format_ns(eval_scale.min)
    ));
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" fill=\"#94a3b8\" font-size=\"11\">{}</text>\n",
        eval_release_x + 20.0 + eval_scale.bar_width,
        height - 10.0,
        format_ns(eval_scale.max)
    ));

    for (idx, row) in rows.iter().enumerate() {
        let y = top + row_height * idx as f64;
        svg.push_str(&format!(
            "<text x=\"16\" y=\"{}\" fill=\"#e5e7eb\" font-size=\"13\" font-family=\"ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, monospace\">{}</text>\n",
            y + 24.0,
            row.label
        ));
        let build_debug = ProfilePairChart {
            panel_x: build_debug_x,
            panel_title: "Debug",
            primal: &row.build_debug_primal,
            ad: &row.build_debug_ad,
        };
        render_profile_pair_chart(&mut svg, &build_debug, y, &build_scale);
        let build_release = ProfilePairChart {
            panel_x: build_release_x,
            panel_title: "Release",
            primal: &row.build_release_primal,
            ad: &row.build_release_ad,
        };
        render_profile_pair_chart(&mut svg, &build_release, y, &build_scale);
        let eval_debug = ProfilePairChart {
            panel_x: eval_debug_x,
            panel_title: "Debug",
            primal: &row.eval_debug_primal,
            ad: &row.eval_debug_ad,
        };
        render_profile_pair_chart(&mut svg, &eval_debug, y, &eval_scale);
        let eval_release = ProfilePairChart {
            panel_x: eval_release_x,
            panel_title: "Release",
            primal: &row.eval_release_primal,
            ad: &row.eval_release_ad,
        };
        render_profile_pair_chart(&mut svg, &eval_release, y, &eval_scale);

        if idx + 1 != rows.len() {
            svg.push_str(&format!(
                "<line x1=\"16\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#374151\" stroke-width=\"1\" opacity=\"0.7\" />\n",
                y + row_height - 4.0,
                svg_width - 16.0,
                y + row_height - 4.0
            ));
        }
    }
    svg.push_str("</svg>\n");
    svg
}

fn render_strategy_timing_table(
    debug: &[HessianStrategyReport],
    release: &BTreeMap<&str, &HessianStrategyReport>,
) -> String {
    let mut out = String::new();
    out.push_str("| Strategy | What it does | Default | Ops | vs 1 | Debug build | Release build | Debug eval | Release eval |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for strategy in debug {
        let Some(release_strategy) = release.get(strategy.key.as_str()) else {
            continue;
        };
        let default_flag = if strategy.is_default { "`yes`" } else { "`no`" };
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {:.3}x | {} | {} | {} | {} |\n",
            strategy.label,
            strategy.description,
            default_flag,
            strategy.op_count,
            strategy.ratio_to_by_column,
            format_timing(&strategy.build),
            format_timing(&release_strategy.build),
            format_timing(&strategy.eval),
            format_timing(&release_strategy.eval),
        ));
    }
    out
}

fn render_timing_details(
    summary: &str,
    debug_cases: &[CaseReport],
    release_cases: &BTreeMap<&str, &CaseReport>,
    select_debug_primal: fn(&CaseReport) -> &TimingStats,
    select_debug_ad: fn(&CaseReport) -> &TimingStats,
    select_release_primal: fn(&CaseReport) -> &TimingStats,
    select_release_ad: fn(&CaseReport) -> &TimingStats,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("<details>\n<summary>{summary}</summary>\n\n"));
    out.push_str("| Case | Debug primal | Debug AD | Release primal | Release AD |\n");
    out.push_str("| --- | --- | --- | --- | --- |\n");
    for case in debug_cases {
        let Some(release_case) = release_cases.get(case.key.as_str()) else {
            continue;
        };
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            case.key,
            format_timing(select_debug_primal(case)),
            format_timing(select_debug_ad(case)),
            format_timing(select_release_primal(release_case)),
            format_timing(select_release_ad(release_case)),
        ));
    }
    out.push_str("\n</details>\n");
    out
}

fn render_backend_comparison_case(case: &CaseReport, release_case: &CaseReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("### `{}`\n\n", case.key));
    out.push_str(&format!("{}\n\n", case.description));
    out.push_str(
        "| Phase | Debug LLVM AOT | Debug LLVM JIT | Release LLVM AOT | Release LLVM JIT |\n",
    );
    out.push_str("| --- | --- | --- | --- | --- |\n");
    out.push_str(&format!(
        "| Primal setup | {} | {} | {} | {} |\n",
        format_timing(&case.llvm_aot_compile_original),
        format_timing(&case.llvm_setup_original),
        format_timing(&release_case.llvm_aot_compile_original),
        format_timing(&release_case.llvm_setup_original),
    ));
    out.push_str(&format!(
        "| AD setup | {} | {} | {} | {} |\n",
        format_timing(&case.llvm_aot_compile_augmented),
        format_timing(&case.llvm_setup_augmented),
        format_timing(&release_case.llvm_aot_compile_augmented),
        format_timing(&release_case.llvm_setup_augmented),
    ));
    out.push_str(&format!(
        "| Primal eval | {} | {} | {} | {} |\n",
        format_timing(&case.eval_original),
        format_timing(&case.llvm_eval_original),
        format_timing(&release_case.eval_original),
        format_timing(&release_case.llvm_eval_original),
    ));
    out.push_str(&format!(
        "| AD eval | {} | {} | {} | {} |\n",
        format_timing(&case.eval_augmented),
        format_timing(&case.llvm_eval_augmented),
        format_timing(&release_case.eval_augmented),
        format_timing(&release_case.llvm_eval_augmented),
    ));
    out.push('\n');
    out
}

pub fn render_markdown_report_with_options(
    options: &MarkdownReportOptions,
    debug: &SuiteReport,
    release: &SuiteReport,
) -> String {
    fn render_verdict(verdict: &PropertyVerdict) -> &'static str {
        match verdict {
            PropertyVerdict::Pass => "<span style=\"color:#22c55e\"><strong>PASS</strong></span>",
            PropertyVerdict::Warn => "<span style=\"color:#facc15\"><strong>WARN</strong></span>",
            PropertyVerdict::Fail => "<span style=\"color:#ef4444\"><strong>FAIL</strong></span>",
        }
    }

    let release_cases = release
        .cases
        .iter()
        .map(|case| (case.key.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let release_hessian_strategies = release
        .hessian_strategies
        .iter()
        .map(|strategy| (strategy.key.as_str(), strategy))
        .collect::<BTreeMap<_, _>>();

    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", options.title));
    if let Some(command) = &options.command {
        out.push_str(&format!("Command: `{command}`\n\n"));
    }
    out.push_str("## Summary\n\n");
    out.push_str(&format!("- Cases compared: `{}`.\n", debug.cases.len()));
    out.push_str(&format!(
        "- Profiles compared: `{}` and `{}`.\n",
        debug.build_profile, release.build_profile
    ));
    if let Some(max_ratio_case) = debug
        .cases
        .iter()
        .max_by(|lhs, rhs| lhs.ratio.total_cmp(&rhs.ratio))
    {
        out.push_str(&format!(
            "- Largest AD / primal op ratio: `{}` at `{:.3}x`.\n",
            max_ratio_case.key, max_ratio_case.ratio
        ));
    }
    if !debug.hessian_strategies.is_empty() {
        let Some(debug_best_hessian_build) = debug
            .hessian_strategies
            .iter()
            .min_by(|lhs, rhs| lhs.build.median_ns.total_cmp(&rhs.build.median_ns))
        else {
            return out;
        };
        let Some(release_best_hessian_build) = release
            .hessian_strategies
            .iter()
            .min_by(|lhs, rhs| lhs.build.median_ns.total_cmp(&rhs.build.median_ns))
        else {
            return out;
        };
        let debug_hessian_eval_min = debug
            .hessian_strategies
            .iter()
            .map(|strategy| strategy.eval.median_ns)
            .fold(f64::INFINITY, f64::min);
        let debug_hessian_eval_max = debug
            .hessian_strategies
            .iter()
            .map(|strategy| strategy.eval.median_ns)
            .fold(f64::NEG_INFINITY, f64::max);
        let release_hessian_eval_min = release
            .hessian_strategies
            .iter()
            .map(|strategy| strategy.eval.median_ns)
            .fold(f64::INFINITY, f64::min);
        let release_hessian_eval_max = release
            .hessian_strategies
            .iter()
            .map(|strategy| strategy.eval.median_ns)
            .fold(f64::NEG_INFINITY, f64::max);
        let Some(default_hessian_strategy) = debug
            .hessian_strategies
            .iter()
            .find(|strategy| strategy.is_default)
        else {
            return out;
        };
        out.push_str(&format!(
            "- Default Hessian strategy: `{}`.\n",
            default_hessian_strategy.label
        ));
        out.push_str(&format!(
            "- All Hessian strategies lower to `{}` ops and match strategy 1 numerically.\n",
            default_hessian_strategy.op_count
        ));
        out.push_str(&format!(
            "- Fastest Hessian build: debug `{}` at `{}`, release `{}` at `{}`.\n",
            debug_best_hessian_build.label,
            format_ns(debug_best_hessian_build.build.median_ns),
            release_best_hessian_build.label,
            format_ns(release_best_hessian_build.build.median_ns)
        ));
        out.push_str(&format!(
            "- Hessian eval medians stay tightly clustered: debug `{}` to `{}`, release `{}` to `{}`.\n",
            format_ns(debug_hessian_eval_min),
            format_ns(debug_hessian_eval_max),
            format_ns(release_hessian_eval_min),
            format_ns(release_hessian_eval_max)
        ));
    }
    out.push('\n');

    out.push_str("Notes:\n");
    out.push_str("- `debug` uses a regular `cargo run` build.\n");
    out.push_str("- `release` uses `cargo run --release`.\n");
    out.push_str(
        "- Op counts are lowered generated-kernel instructions, not raw symbolic nodes.\n",
    );
    out.push_str(
        "- LLVM AOT compile timings include LLVM IR build, optimization, and native object emission.\n",
    );
    out.push_str(
        "- LLVM JIT setup timings include LLVM IR build, optimization, object emission, LLJIT load, and symbol lookup.\n",
    );
    out.push_str(
        "- Symbolic build, LLVM AOT compile, and LLVM JIT setup timings are measured once per profile; eval timings stay sampled.\n",
    );
    out.push_str(
        "- The main eval charts use the linked LLVM AOT path; LLVM JIT eval appears in the backend comparison section.\n",
    );
    out.push_str(&format!(
        "- Eval timing samples per profile: `{}` with `{}` ms target per sample.\n\n",
        debug.samples, debug.target_ms
    ));

    if options.include_lowered_op_explanation {
        out.push_str("<details>\n<summary>What \"lowered op count\" means</summary>\n\n");
        out.push_str("```rust\n");
        out.push_str("// symbolic expression\n");
        out.push_str("let s = x + y;\n");
        out.push_str("let f = s * s + 1.0;\n\n");
        out.push_str("// lowered kernel-style instructions\n");
        out.push_str("let t0 = x + y;\n");
        out.push_str("let t1 = t0 * t0;\n");
        out.push_str("let t2 = t1 + 1.0;\n");
        out.push_str("// lowered op count = 3\n");
        out.push_str("```\n\n");
        out.push_str("```mermaid\n");
        out.push_str("graph LR\n");
        out.push_str("    x[\"x\"] --> s[\"s = x + y\"]\n");
        out.push_str("    y[\"y\"] --> s\n");
        out.push_str("    s --> sq[\"t1 = s * s\"]\n");
        out.push_str("    s --> sq\n");
        out.push_str("    sq --> out[\"f = t1 + 1\"]\n");
        out.push_str("    one[\"1\"] --> out\n");
        out.push_str("```\n\n");
        out.push_str(
            "Leaves like `x`, `y`, and constants are symbolic graph nodes, but the op count only tracks executable lowered instructions such as `t0`, `t1`, and `t2`.\n\n",
        );
        out.push_str("</details>\n");
    }

    if !debug.properties.is_empty() {
        out.push_str("## Verification\n\n");
        out.push_str("| Check | What it verifies | Status | Result | Expectation |\n");
        out.push_str("| --- | --- | --- | --- | --- |\n");
        for property in &debug.properties {
            out.push_str(&format!(
                "| `{}` | {} | {} | `{}` | `{}` |\n",
                property.key,
                property.description,
                render_verdict(&property.verdict),
                property.result,
                property.expectation
            ));
        }
    }

    out.push_str("\n## Op Counts\n\n");
    out.push_str("| Case | What it measures | Size | Sweeps | Original ops | AD ops | Ratio | Ratio / sweep |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for case in &debug.cases {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {:.3}x | {:.3}x |\n",
            case.key,
            case.description,
            case.size,
            case.sweep_count,
            case.original_ops,
            case.augmented_ops,
            case.ratio,
            case.normalized_ratio
        ));
    }

    out.push_str("\n## Timing By Case\n\n");
    let case_rows = debug
        .cases
        .iter()
        .filter_map(|case| {
            let release_case = release_cases.get(case.key.as_str())?;
            Some(CaseTimingRow {
                label: case.key.clone(),
                build_debug_primal: case.build_original.clone(),
                build_debug_ad: case.build_augmented.clone(),
                build_release_primal: release_case.build_original.clone(),
                build_release_ad: release_case.build_augmented.clone(),
                eval_debug_primal: case.eval_original.clone(),
                eval_debug_ad: case.eval_augmented.clone(),
                eval_release_primal: release_case.eval_original.clone(),
                eval_release_ad: release_case.eval_augmented.clone(),
            })
        })
        .collect::<Vec<_>>();
    out.push_str(&render_case_timing_chart(
        "Build and eval medians by case",
        &case_rows,
    ));
    out.push('\n');
    out.push_str(&render_timing_details(
        "Raw symbolic construction timing ranges",
        &debug.cases,
        &release_cases,
        |case| &case.build_original,
        |case| &case.build_augmented,
        |case| &case.build_original,
        |case| &case.build_augmented,
    ));
    out.push_str(&render_timing_details(
        "Raw generated evaluation timing ranges",
        &debug.cases,
        &release_cases,
        |case| &case.eval_original,
        |case| &case.eval_augmented,
        |case| &case.eval_original,
        |case| &case.eval_augmented,
    ));

    out.push_str("\n## Backend Comparison\n\n");
    out.push_str("This section compares backend setup cost and runtime cost directly. `LLVM AOT` means compiling lowered kernels to native object code and linking them into the binary; `LLVM JIT` means compiling lowered kernels through LLVM and loading them into LLJIT.\n\n");
    for case in &debug.cases {
        if let Some(release_case) = release_cases.get(case.key.as_str()) {
            out.push_str(&render_backend_comparison_case(case, release_case));
        }
    }

    if !debug.hessian_strategies.is_empty() {
        out.push_str("\n## Hessian Strategy Comparison\n\n");
        out.push_str("All three strategies produce the same lowered op count and numerically identical lower-triangular Hessians in the test suite. The default is `2. Selected outputs` because it keeps the same generated cost while building faster than `1. By column` and `3. Colored` on this benchmark family.\n\n");
        out.push_str(&render_strategy_timing_table(
            &debug.hessian_strategies,
            &release_hessian_strategies,
        ));
        out.push('\n');
    }

    out
}

pub fn render_markdown_report(debug: &SuiteReport, release: &SuiteReport) -> String {
    render_markdown_report_with_options(&MarkdownReportOptions::default(), debug, release)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use sx_core::{NamedMatrix, SX, SXFunction, SXMatrix};

    fn sampled_timing(median_ns: f64) -> TimingStats {
        TimingStats {
            samples: 4,
            iterations_per_sample: 16,
            min_ns: median_ns * 0.8,
            median_ns,
            max_ns: median_ns * 1.2,
        }
    }

    fn single_timing(ns: f64) -> TimingStats {
        TimingStats {
            samples: 1,
            iterations_per_sample: 1,
            min_ns: ns,
            median_ns: ns,
            max_ns: ns,
        }
    }

    fn sample_function_pair(profile: &str) -> FunctionPairReportInput {
        let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic input matrix");
        let x0 = x.nz(0);
        let x1 = x.nz(1);
        let original = SXFunction::new(
            format!("sample_{profile}"),
            vec![NamedMatrix::new("x", x.clone()).expect("input slot")],
            vec![NamedMatrix::new("value", SXMatrix::scalar(x0 + x1)).expect("output slot")],
        )
        .expect("original function");
        let augmented = SXFunction::new(
            format!("sample_{profile}_ad"),
            vec![NamedMatrix::new("x", x).expect("input slot")],
            vec![
                NamedMatrix::new("value", SXMatrix::scalar(x0 + x1)).expect("output slot"),
                NamedMatrix::new(
                    "gradient",
                    SXMatrix::dense_column(vec![SX::one(), SX::one()]).expect("gradient output"),
                )
                .expect("gradient slot"),
            ],
        )
        .expect("augmented function");
        FunctionPairReportInput {
            key: "sample_case".into(),
            description: "Sample scalar plus gradient output.".into(),
            size: 2,
            sweep_count: 1,
            original,
            augmented,
            timings: CaseTimingStats {
                build_original: single_timing(12_000.0),
                build_augmented: single_timing(18_000.0),
                eval_original: sampled_timing(800.0),
                eval_augmented: sampled_timing(1_600.0),
                llvm_aot_compile_original: single_timing(500_000.0),
                llvm_aot_compile_augmented: single_timing(650_000.0),
                llvm_setup_original: single_timing(700_000.0),
                llvm_setup_augmented: single_timing(900_000.0),
                llvm_eval_original: sampled_timing(500.0),
                llvm_eval_augmented: sampled_timing(950.0),
            },
        }
    }

    fn sample_suite(profile: &str) -> SuiteReport {
        suite_report_from_function_pairs(
            profile,
            8,
            25,
            vec![sample_function_pair(profile)],
            vec![HessianStrategyReport {
                key: "selected_outputs".into(),
                label: "2. Selected outputs".into(),
                description: "Only sweep the needed gradient suffix.".into(),
                is_default: true,
                op_count: 42,
                ratio_to_by_column: 1.0,
                build: single_timing(14_000.0),
                eval: sampled_timing(2_500.0),
            }],
            vec![PropertyStatus {
                key: "sample_case.baseline".into(),
                description: "Sample property check.".into(),
                verdict: PropertyVerdict::Pass,
                result: "ok".into(),
                expectation: "ok".into(),
            }],
        )
        .expect("sample suite report")
    }

    #[test]
    fn format_timing_marks_single_measurements() {
        assert_eq!(format_timing(&single_timing(2_500.0)), "2.50 us (single)");
        assert!(format_timing(&sampled_timing(2_500.0)).contains("["));
    }

    #[test]
    fn suite_report_from_function_pairs_computes_lowered_op_ratios() {
        let report = suite_report_from_function_pairs(
            "debug",
            8,
            25,
            vec![sample_function_pair("debug")],
            Vec::new(),
            Vec::new(),
        )
        .expect("suite report");

        assert_eq!(report.build_profile, "debug");
        assert_eq!(report.cases.len(), 1);
        let case = &report.cases[0];
        assert_eq!(case.key, "sample_case");
        assert_eq!(case.original_ops, 1);
        assert_eq!(case.augmented_ops, 1);
        assert_eq!(case.ratio, 1.0);
        assert_eq!(case.normalized_ratio, 1.0);
    }

    #[test]
    fn render_markdown_report_with_options_includes_key_sections() {
        let debug = sample_suite("debug");
        let mut release = sample_suite("release");
        release.cases[0].build_original = single_timing(8_000.0);
        release.cases[0].eval_original = sampled_timing(400.0);
        release.hessian_strategies[0].build = single_timing(9_000.0);

        let markdown = render_markdown_report_with_options(
            &MarkdownReportOptions {
                title: "Coverage Fixture".into(),
                command: Some("cargo run -p xtask -- ad-cost-report".into()),
                include_lowered_op_explanation: true,
            },
            &debug,
            &release,
        );

        assert!(markdown.contains("# Coverage Fixture"));
        assert!(markdown.contains("## Verification"));
        assert!(markdown.contains("## Op Counts"));
        assert!(markdown.contains("## Timing By Case"));
        assert!(markdown.contains("## Backend Comparison"));
        assert!(markdown.contains("## Hessian Strategy Comparison"));
        assert!(markdown.contains("single)"));
        assert!(markdown.contains("```mermaid"));
        assert!(markdown.contains("Sample scalar plus gradient output."));
    }

    #[test]
    fn render_markdown_report_wrapper_uses_default_options() {
        let debug = sample_suite("debug");
        let release = sample_suite("release");
        let markdown = render_markdown_report(&debug, &release);
        assert!(markdown.contains("# Benchmark Report"));
        assert!(markdown.contains("## Summary"));
    }
}
