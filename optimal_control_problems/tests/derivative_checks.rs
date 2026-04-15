use std::collections::BTreeMap;
use std::thread;

use optimal_control_problems::{
    DerivativeCheckOrder, DerivativeCheckRequest, OcpBenchmarkPreset, ProblemDerivativeCheck,
    ProblemId, TranscriptionMethod, problem_specs, validate_problem_derivatives,
};
use optimization::{FiniteDifferenceValidationOptions, ValidationSummary, ValidationTolerances};

const FIRST_ORDER_TOLERANCES: ValidationTolerances = ValidationTolerances::new(5.0e-5, 5.0e-4);
const SECOND_ORDER_TOLERANCES: ValidationTolerances = ValidationTolerances::new(1.0e-4, 1.0e-3);

fn require_release_mode_for_manual_derivative_sweeps() {
    assert!(
        !cfg!(debug_assertions),
        "manual derivative sweeps must be run in release mode\n\ntry:\n  cargo test -p optimal_control_problems --release --test derivative_checks all_ocp_problems_policy_matrix_first_order_derivatives_stay_clean -- --ignored --nocapture"
    );
}

fn run_manual_sweep_with_large_stack(task: impl FnOnce() + Send + 'static) {
    let handle = thread::Builder::new()
        .name("manual-derivative-sweep".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(task)
        .expect("manual derivative sweep thread should spawn");
    if let Err(payload) = handle.join() {
        std::panic::resume_unwind(payload);
    }
}

fn request_for(
    transcription: TranscriptionMethod,
    preset: Option<OcpBenchmarkPreset>,
) -> DerivativeCheckRequest {
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
        sx_functions_override: preset.map(OcpBenchmarkPreset::sx_function_config),
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

#[test]
fn glider_derivative_check_api_smoke() {
    let check = validate_problem_derivatives(
        ProblemId::OptimalDistanceGlider,
        &request_for(TranscriptionMethod::MultipleShooting, None),
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
    require_release_mode_for_manual_derivative_sweeps();
    run_manual_sweep_with_large_stack(|| {
        let mut failures = Vec::new();
        for spec in problem_specs() {
            for transcription in [
                TranscriptionMethod::MultipleShooting,
                TranscriptionMethod::DirectCollocation,
            ] {
                let check = match validate_problem_derivatives(
                    spec.id,
                    &request_for(transcription, None),
                ) {
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
    });
}

#[test]
#[ignore = "manual policy-matrix derivative sweep over all OCP problems"]
fn all_ocp_problems_policy_matrix_first_order_derivatives_stay_clean() {
    require_release_mode_for_manual_derivative_sweeps();
    run_manual_sweep_with_large_stack(|| {
        let mut failures = Vec::new();
        for spec in problem_specs() {
            for transcription in [
                TranscriptionMethod::MultipleShooting,
                TranscriptionMethod::DirectCollocation,
            ] {
                for preset in OcpBenchmarkPreset::all() {
                    let request = request_for(transcription, Some(*preset));
                    let check = match validate_problem_derivatives(spec.id, &request) {
                        Ok(check) => check,
                        Err(err) => {
                            failures.push(format!(
                                "{} {:?} {}: derivative check failed to run: {err:#}",
                                spec.name,
                                transcription,
                                preset.id()
                            ));
                            continue;
                        }
                    };
                    println!("preset={}\n{}", preset.id(), format_check(&check));
                    if !check.order_is_within_tolerances(
                        DerivativeCheckOrder::First,
                        FIRST_ORDER_TOLERANCES,
                        SECOND_ORDER_TOLERANCES,
                    ) {
                        failures.push(format!("preset={}\n{}", preset.id(), format_check(&check)));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "policy-matrix first-order derivative failures:\n\n{}",
            failures.join("\n\n"),
        );
    });
}

#[test]
#[ignore = "manual Hessian policy-matrix sweep over all OCP problems"]
fn all_ocp_problems_policy_matrix_second_order_derivatives_stay_clean() {
    require_release_mode_for_manual_derivative_sweeps();
    run_manual_sweep_with_large_stack(|| {
        let mut execution_failures = Vec::new();
        let mut tolerance_misses = Vec::new();
        for spec in problem_specs() {
            for transcription in [
                TranscriptionMethod::MultipleShooting,
                TranscriptionMethod::DirectCollocation,
            ] {
                for preset in OcpBenchmarkPreset::all() {
                    let request = request_for(transcription, Some(*preset));
                    let check = match validate_problem_derivatives(spec.id, &request) {
                        Ok(check) => check,
                        Err(err) => {
                            execution_failures.push(format!(
                                "{} {:?} {}: derivative check failed to run: {err:#}",
                                spec.name,
                                transcription,
                                preset.id()
                            ));
                            continue;
                        }
                    };
                    println!("preset={}\n{}", preset.id(), format_check(&check));
                    if !check.order_is_within_tolerances(
                        DerivativeCheckOrder::Second,
                        FIRST_ORDER_TOLERANCES,
                        SECOND_ORDER_TOLERANCES,
                    ) {
                        tolerance_misses.push(format!(
                            "preset={}\n{}",
                            preset.id(),
                            format_check(&check)
                        ));
                    }
                }
            }
        }
        assert!(
            execution_failures.is_empty(),
            "second-order survey had execution failures:\n\n{}",
            execution_failures.join("\n\n"),
        );
        assert!(
            tolerance_misses.is_empty(),
            "second-order derivative failures:\n\n{}",
            tolerance_misses.join("\n\n"),
        );
    });
}
