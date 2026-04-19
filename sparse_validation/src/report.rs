use crate::model::{
    BaselineSummary, OrderingValidationResult, SymbolicValidationResult, ValidationOutcome,
    ValidationSuiteReport, ValidationSummary,
};

pub fn summarize_report(report: &ValidationSuiteReport) -> ValidationSummary {
    ValidationSummary {
        total_cases: report.cases.len() + report.skipped_cases.len(),
        executed_cases: report.cases.len(),
        skipped_cases: report.skipped_cases.len(),
        failed_ordering_results: report
            .cases
            .iter()
            .flat_map(|case| case.ordering.iter())
            .filter(|result| result.outcome == ValidationOutcome::Failed)
            .count(),
        failed_symbolic_results: report
            .cases
            .iter()
            .flat_map(|case| case.symbolic.iter())
            .filter(|result| result.outcome == ValidationOutcome::Failed)
            .count(),
        failed_numeric_results: report
            .cases
            .iter()
            .flat_map(|case| case.numeric.iter())
            .filter(|result| result.outcome == ValidationOutcome::Failed)
            .count(),
        failed_robustness_results: report
            .robustness
            .iter()
            .filter(|result| result.outcome == ValidationOutcome::Failed)
            .count(),
    }
}

pub fn apply_baseline_summary(
    report: &mut ValidationSuiteReport,
    previous: &ValidationSuiteReport,
) {
    let mut ordering_ratios = Vec::new();
    let mut symbolic_ratios = Vec::new();
    let mut numeric_ratios = Vec::new();
    let mut slower_ordering_entries = Vec::new();
    let mut slower_symbolic_entries = Vec::new();
    let mut slower_numeric_entries = Vec::new();

    for case in &report.cases {
        if let Some(previous_case) = previous
            .cases
            .iter()
            .find(|candidate| candidate.case.case_id == case.case.case_id)
        {
            accumulate_ratios(
                &case.case.case_id,
                &case.ordering,
                &previous_case.ordering,
                &mut ordering_ratios,
                &mut slower_ordering_entries,
            );
            accumulate_symbolic_ratios(
                &case.case.case_id,
                &case.symbolic,
                &previous_case.symbolic,
                &mut symbolic_ratios,
                &mut slower_symbolic_entries,
            );
            accumulate_numeric_ratios(
                &case.case.case_id,
                &case.numeric,
                &previous_case.numeric,
                &mut numeric_ratios,
                &mut slower_numeric_entries,
            );
        }
    }

    report.baseline = Some(BaselineSummary {
        previous_generated_at_utc: previous.generated_at_utc.clone(),
        ordering_median_ratio: median(ordering_ratios),
        symbolic_median_ratio: median(symbolic_ratios),
        numeric_median_ratio: median(numeric_ratios),
        slower_ordering_entries,
        slower_symbolic_entries,
        slower_numeric_entries,
    });
}

fn accumulate_ratios(
    case_id: &str,
    current: &[OrderingValidationResult],
    previous: &[OrderingValidationResult],
    ratios: &mut Vec<f64>,
    slower_entries: &mut Vec<String>,
) {
    for result in current {
        if let Some(previous_result) = previous
            .iter()
            .find(|candidate| candidate.method == result.method)
            && previous_result.elapsed_ms > 0.0
            && result.elapsed_ms > 0.0
        {
            let ratio = result.elapsed_ms / previous_result.elapsed_ms;
            ratios.push(ratio);
            if ratio > 1.10 {
                slower_entries.push(format!("{case_id}: {:?} x{ratio:.3}", result.method));
            }
        }
    }
}

fn accumulate_symbolic_ratios(
    case_id: &str,
    current: &[SymbolicValidationResult],
    previous: &[SymbolicValidationResult],
    ratios: &mut Vec<f64>,
    slower_entries: &mut Vec<String>,
) {
    for result in current {
        if let Some(previous_result) = previous
            .iter()
            .find(|candidate| candidate.strategy == result.strategy)
            && previous_result.elapsed_ms > 0.0
            && result.elapsed_ms > 0.0
        {
            let ratio = result.elapsed_ms / previous_result.elapsed_ms;
            ratios.push(ratio);
            if ratio > 1.10 {
                slower_entries.push(format!("{case_id}: {:?} x{ratio:.3}", result.strategy));
            }
        }
    }
}

fn accumulate_numeric_ratios(
    case_id: &str,
    current: &[crate::model::NumericValidationResult],
    previous: &[crate::model::NumericValidationResult],
    ratios: &mut Vec<f64>,
    slower_entries: &mut Vec<String>,
) {
    for result in current {
        if let Some(previous_result) = previous
            .iter()
            .find(|candidate| candidate.strategy == result.strategy)
            && previous_result.factor_elapsed_ms > 0.0
            && result.factor_elapsed_ms > 0.0
        {
            let ratio = result.factor_elapsed_ms / previous_result.factor_elapsed_ms;
            ratios.push(ratio);
            if ratio > 1.10 {
                slower_entries.push(format!("{case_id}: {:?} x{ratio:.3}", result.strategy));
            }
        }
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    Some(values[values.len() / 2])
}

pub fn render_markdown_report(report: &ValidationSuiteReport) -> String {
    let mut out = String::new();
    out.push_str("# Sparse Validation Report\n\n");
    out.push_str(&format!(
        "- tier: `{}`\n- generated_at_utc: `{}`\n- rustc: `{}`\n- os/arch: `{}` / `{}`\n",
        report.tier.label(),
        report.generated_at_utc,
        report.environment.rustc_version,
        report.environment.operating_system,
        report.environment.architecture,
    ));
    if let Some(git_sha) = &report.environment.git_sha {
        out.push_str(&format!("- git_sha: `{git_sha}`\n"));
    }
    out.push_str(&format!(
        "- requested_native_metis: `{}`\n- requested_native_spral: `{}`\n\n",
        report.requested_native_metis, report.requested_native_spral
    ));

    out.push_str("## Summary\n\n");
    out.push_str(&format!(
        "- total_cases: {}\n- executed_cases: {}\n- skipped_cases: {}\n- failed_ordering_results: {}\n- failed_symbolic_results: {}\n- failed_numeric_results: {}\n- failed_robustness_results: {}\n\n",
        report.summary.total_cases,
        report.summary.executed_cases,
        report.summary.skipped_cases,
        report.summary.failed_ordering_results,
        report.summary.failed_symbolic_results,
        report.summary.failed_numeric_results,
        report.summary.failed_robustness_results,
    ));
    if let Some(baseline) = &report.baseline {
        out.push_str("## Baseline\n\n");
        out.push_str(&format!(
            "- previous_generated_at_utc: `{}`\n- ordering_median_ratio: `{}`\n- symbolic_median_ratio: `{}`\n- numeric_median_ratio: `{}`\n\n",
            baseline.previous_generated_at_utc,
            baseline
                .ordering_median_ratio
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".into()),
            baseline
                .symbolic_median_ratio
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".into()),
            baseline
                .numeric_median_ratio
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".into()),
        ));
    }
    if !report.skipped_cases.is_empty() {
        out.push_str("## Skipped Cases\n\n");
        for skipped in &report.skipped_cases {
            out.push_str(&format!("- `{}`: {}\n", skipped.case_id, skipped.reason));
        }
        out.push('\n');
    }

    for case in &report.cases {
        out.push_str(&format!("## Case `{}`\n\n", case.case.case_id));
        out.push_str(&format!(
            "- description: {}\n- source: `{:?}`\n- dimension: `{}`\n- nnz: `{}`\n- exact_oracle: `{}`\n\n",
            case.case.description,
            case.case.source,
            case.case.dimension,
            case.case.nnz,
            case.case.exact_oracle,
        ));
        out.push_str("### Ordering\n\n");
        out.push_str("| method | outcome | ms | fill_nnz | ratio_vs_natural | etree_height | max_separator_fraction |\n");
        out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
        for result in &case.ordering {
            let metrics = result.metrics.as_ref();
            out.push_str(&format!(
                "| `{:?}` | `{:?}` | {:.3} | {} | {} | {} | {} |\n",
                result.method,
                result.outcome,
                result.elapsed_ms,
                metrics
                    .map(|value| value.fill_nnz.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .and_then(|value| value.fill_ratio_vs_natural)
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.etree_height.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .and_then(|value| value.max_separator_fraction)
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
            ));
        }
        out.push('\n');
        out.push_str("### Symbolic Analysis\n\n");
        out.push_str("| strategy | outcome | ms | fill_nnz | fill_pattern_match | column_counts_match | etree_height | supernodes |\n");
        out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for result in &case.symbolic {
            let metrics = result.metrics.as_ref();
            out.push_str(&format!(
                "| `{:?}` | `{:?}` | {:.3} | {} | {} | {} | {} | {} |\n",
                result.strategy,
                result.outcome,
                result.elapsed_ms,
                metrics
                    .map(|value| value.fill_nnz.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .and_then(|value| value.exact_fill_pattern_match)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .and_then(|value| value.exact_column_counts_match)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.etree_height.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.supernode_count.to_string())
                    .unwrap_or_else(|| "n/a".into()),
            ));
        }
        out.push('\n');
        out.push_str("### Numeric Factorization\n\n");
        out.push_str("| strategy | outcome | factor_ms | solve_ms | refactor_ms | refactor_speedup | residual | solution_error | stored_nnz | supernodes | max_width | inertia_match | regularized_pivots | two_by_two_pivots | delayed_pivots |\n");
        out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for result in &case.numeric {
            let metrics = result.metrics.as_ref();
            out.push_str(&format!(
                "| `{:?}` | `{:?}` | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                result.strategy,
                result.outcome,
                result.factor_elapsed_ms,
                result.solve_elapsed_ms,
                result
                    .refactor_elapsed_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .and_then(|value| value.refactor_speedup_vs_factor)
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| format!("{:.3e}", value.solve_residual_inf_norm))
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| format!("{:.3e}", value.solution_inf_error))
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.stored_nnz.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.supernode_count.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.max_supernode_width.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .and_then(|value| value.inertia_match)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.regularized_pivots.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.two_by_two_pivots.to_string())
                    .unwrap_or_else(|| "n/a".into()),
                metrics
                    .map(|value| value.delayed_pivots.to_string())
                    .unwrap_or_else(|| "n/a".into()),
            ));
        }
        out.push('\n');
        if !case.failures.is_empty() {
            out.push_str("### Failures\n\n");
            for failure in &case.failures {
                out.push_str(&format!("- {failure}\n"));
            }
            out.push('\n');
        }
    }

    out.push_str("## Robustness\n\n");
    out.push_str("| scenario | target | outcome | ms | error_kind |\n");
    out.push_str("| --- | --- | --- | ---: | --- |\n");
    for result in &report.robustness {
        out.push_str(&format!(
            "| `{}` | `{}` | `{:?}` | {:.3} | {} |\n",
            result.scenario,
            result.target,
            result.outcome,
            result.duration_ms,
            result.error_kind.as_deref().unwrap_or("n/a"),
        ));
    }
    out
}

pub fn render_html_report(report: &ValidationSuiteReport) -> String {
    let markdown = render_markdown_report(report);
    format!(
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Sparse Validation Report</title><style>body{{font-family:ui-monospace, SFMono-Regular, Menlo, monospace; background:#f5f2ea; color:#1f2430; padding:24px;}} pre{{white-space:pre-wrap; word-break:break-word; background:white; padding:16px; border:1px solid #d8d2c4; border-radius:8px;}}</style></head><body><pre>{}</pre></body></html>",
        html_escape(&markdown)
    )
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
