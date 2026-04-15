use std::collections::BTreeMap;

use optimal_control_problems::{
    DerivativeCheckRequest, ProblemDerivativeCheck, ProblemId, TranscriptionMethod, problem_specs,
    validate_problem_derivatives,
};
use optimization::{FiniteDifferenceValidationOptions, ValidationSummary, ValidationTolerances};

const FIRST_ORDER_TOLERANCES: ValidationTolerances = ValidationTolerances::new(5.0e-5, 5.0e-4);

fn request_for(transcription: TranscriptionMethod) -> DerivativeCheckRequest {
    let mut values = BTreeMap::new();
    values.insert(
        "transcription_method".to_string(),
        match transcription {
            TranscriptionMethod::MultipleShooting => 0.0,
            TranscriptionMethod::DirectCollocation => 1.0,
        },
    );
    DerivativeCheckRequest {
        values,
        finite_difference: FiniteDifferenceValidationOptions {
            first_order_step: 1.0e-6,
            second_order_step: 1.0e-4,
            zero_tolerance: 1.0e-7,
        },
        ..DerivativeCheckRequest::default()
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
    format!(
        "{label}: max_abs={:.3e} max_rel={:.3e} rms_abs={:.3e} missing={} extra={} {worst}",
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

#[test]
fn glider_derivative_check_api_smoke() {
    let check = validate_problem_derivatives(
        ProblemId::OptimalDistanceGlider,
        &request_for(TranscriptionMethod::MultipleShooting),
    )
    .expect("glider derivative check should compile and validate");
    assert!(
        check.first_order_is_within_tolerances(FIRST_ORDER_TOLERANCES),
        "glider first-order derivative check failed\n{}",
        format_check(&check),
    );
}

#[test]
#[ignore = "manual full derivative sweep over all OCP problems and both transcriptions"]
fn all_ocp_problems_first_order_derivatives_stay_clean() {
    let mut failures = Vec::new();
    for spec in problem_specs() {
        for transcription in [
            TranscriptionMethod::MultipleShooting,
            TranscriptionMethod::DirectCollocation,
        ] {
            let check = match validate_problem_derivatives(spec.id, &request_for(transcription)) {
                Ok(check) => check,
                Err(err) => {
                    failures.push(format!(
                        "{} {:?}: derivative check failed to run: {err:#}",
                        spec.name, transcription
                    ));
                    continue;
                }
            };
            println!("{}", format_check(&check));
            if !check.first_order_is_within_tolerances(FIRST_ORDER_TOLERANCES) {
                failures.push(format_check(&check));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "first-order derivative failures:\n\n{}",
        failures.join("\n\n"),
    );
}
