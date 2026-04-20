use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use metis_ordering::{
    NestedDissectionOptions, OrderingSummary, Permutation, nested_dissection_order,
};
use nalgebra::{DMatrix, SymmetricEigen};
use spral_ssids::{
    Inertia, NumericFactorOptions, OrderingStrategy, SsidsOptions, SymmetricCscMatrix, analyse,
    factorize,
};

use crate::corpus::{LoadedCorpusCase, load_cases_for_tier};
use crate::metrics::{
    connected_component_count, exact_symbolic_metrics, permutation_is_valid, tree_parent_validity,
};
use crate::model::{
    CaseValidationReport, EnvironmentStamp, NumericFactorizationMetrics, NumericStrategy,
    NumericValidationResult, OrderingMethod, OrderingQualityMetrics, OrderingValidationResult,
    RobustnessCaseResult, SkippedCase, SymbolicAnalysisMetrics, SymbolicStrategy,
    SymbolicValidationResult, ValidationOutcome, ValidationSuiteReport, ValidationTier,
};
use crate::references::{NativeMetisReference, NativeSpralReference, amd_permutation};
use crate::report::summarize_report;

const NUMERIC_VALIDATION_MAX_DIM: usize = 1024;
const NUMERIC_EXACT_INERTIA_MAX_DIM: usize = 64;
const NUMERIC_VALIDATION_TOL: f64 = 1e-8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationRunConfig {
    pub tier: ValidationTier,
    pub corpus_root: PathBuf,
    pub with_native_metis: bool,
    pub with_native_spral: bool,
}

impl Default for ValidationRunConfig {
    fn default() -> Self {
        Self {
            tier: ValidationTier::Pr,
            corpus_root: PathBuf::from("target/validation_corpus"),
            with_native_metis: false,
            with_native_spral: false,
        }
    }
}

pub fn run_validation_suite(config: &ValidationRunConfig) -> Result<ValidationSuiteReport> {
    let environment = collect_environment();
    let generated_at_utc = iso8601_utc();
    let (cases, skipped_cases) = load_cases_for_tier(config.tier, &config.corpus_root)?;
    let native_metis = if config.with_native_metis {
        NativeMetisReference::load().ok()
    } else {
        None
    };
    let native_spral = if config.with_native_spral {
        NativeSpralReference::load().ok()
    } else {
        None
    };

    let mut reports = Vec::new();
    for case in cases {
        reports.push(validate_case(
            &case,
            config,
            native_metis.as_ref(),
            native_spral.as_ref(),
        )?);
    }
    let robustness = run_robustness_suite();
    let skipped = skipped_cases
        .into_iter()
        .map(|(case_id, reason)| SkippedCase { case_id, reason })
        .collect::<Vec<_>>();

    let mut report = ValidationSuiteReport {
        tier: config.tier,
        generated_at_utc,
        environment,
        requested_native_metis: config.with_native_metis,
        requested_native_spral: config.with_native_spral,
        skipped_cases: skipped,
        cases: reports,
        robustness,
        summary: crate::model::ValidationSummary {
            total_cases: 0,
            executed_cases: 0,
            skipped_cases: 0,
            failed_ordering_results: 0,
            failed_symbolic_results: 0,
            failed_numeric_results: 0,
            failed_robustness_results: 0,
        },
        baseline: None,
    };
    report.summary = summarize_report(&report);
    Ok(report)
}

fn validate_case(
    case: &LoadedCorpusCase,
    config: &ValidationRunConfig,
    native_metis: Option<&NativeMetisReference>,
    native_spral: Option<&NativeSpralReference>,
) -> Result<CaseValidationReport> {
    let graph = case.matrix.to_graph()?;
    let component_count = connected_component_count(&graph);

    let natural_permutation = Permutation::identity(case.matrix.dimension());
    let natural_metrics = exact_symbolic_metrics(&case.matrix, &natural_permutation)?;
    let natural_fill_ratio = if natural_metrics.fill_nnz == 0 {
        None
    } else {
        Some(1.0)
    };
    let mut ordering = vec![OrderingValidationResult {
        method: OrderingMethod::Natural,
        outcome: ValidationOutcome::Passed,
        elapsed_ms: 0.0,
        metrics: Some(OrderingQualityMetrics {
            permutation_valid: true,
            fill_nnz: natural_metrics.fill_nnz,
            fill_ratio_vs_natural: natural_fill_ratio,
            etree_height: natural_metrics.etree_height,
            memory_bytes: natural_metrics.memory_bytes,
            max_separator_fraction: None,
            component_count,
        }),
        notes: Vec::new(),
    }];

    ordering.push(validate_external_permutation(
        case,
        component_count,
        OrderingMethod::Amd,
        timed(|| amd_permutation(&case.matrix)),
        natural_metrics.fill_nnz,
        None,
    )?);

    let rust_order = timed(|| {
        Ok(nested_dissection_order(
            &graph,
            &NestedDissectionOptions::default(),
        )?)
    });
    ordering.push(match rust_order {
        Ok((summary, elapsed_ms)) => validate_rust_nested_dissection(
            case,
            component_count,
            summary,
            elapsed_ms,
            natural_metrics.fill_nnz,
        )?,
        Err(error) => OrderingValidationResult {
            method: OrderingMethod::RustNestedDissection,
            outcome: ValidationOutcome::Failed,
            elapsed_ms: 0.0,
            metrics: None,
            notes: vec![error.to_string()],
        },
    });

    if config.with_native_metis {
        ordering.push(validate_external_permutation(
            case,
            component_count,
            OrderingMethod::NativeMetis,
            native_metis
                .ok_or_else(|| anyhow::anyhow!("native METIS reference unavailable"))
                .and_then(|reference| timed(|| reference.order(&graph))),
            natural_metrics.fill_nnz,
            None,
        )?);
    } else {
        ordering.push(OrderingValidationResult {
            method: OrderingMethod::NativeMetis,
            outcome: ValidationOutcome::SkippedNotRequested,
            elapsed_ms: 0.0,
            metrics: None,
            notes: Vec::new(),
        });
    }

    let mut failures = Vec::new();
    if case.metadata.protected_fill_regression
        && let Some(rust_metrics) = ordering
            .iter()
            .find(|result| result.method == OrderingMethod::RustNestedDissection)
            .and_then(|result| result.metrics.as_ref())
        && rust_metrics.fill_nnz > natural_metrics.fill_nnz
    {
        failures.push(format!(
            "rust nested dissection fill {} exceeded natural fill {}",
            rust_metrics.fill_nnz, natural_metrics.fill_nnz
        ));
    }

    let mut symbolic = vec![
        validate_symbolic_strategy(case, SymbolicStrategy::Natural, OrderingStrategy::Natural)?,
        validate_symbolic_strategy(
            case,
            SymbolicStrategy::RustNestedDissection,
            OrderingStrategy::NestedDissection(NestedDissectionOptions::default()),
        )?,
    ];
    if config.with_native_spral {
        symbolic.push(SymbolicValidationResult {
            strategy: SymbolicStrategy::NativeSpral,
            outcome: ValidationOutcome::SkippedNotImplemented,
            elapsed_ms: 0.0,
            metrics: None,
            notes: vec![
                "native SPRAL scaffold is deferred until numeric factorization exists".into(),
            ],
        });
    } else {
        symbolic.push(SymbolicValidationResult {
            strategy: SymbolicStrategy::NativeSpral,
            outcome: ValidationOutcome::SkippedNotRequested,
            elapsed_ms: 0.0,
            metrics: None,
            notes: Vec::new(),
        });
    }

    let mut numeric = vec![
        validate_numeric_strategy(case, NumericStrategy::Natural, OrderingStrategy::Natural)?,
        validate_numeric_strategy(
            case,
            NumericStrategy::ApproximateMinimumDegree,
            OrderingStrategy::ApproximateMinimumDegree,
        )?,
        validate_numeric_strategy(
            case,
            NumericStrategy::Auto,
            OrderingStrategy::Auto(NestedDissectionOptions::default()),
        )?,
        validate_numeric_strategy(
            case,
            NumericStrategy::RustNestedDissection,
            OrderingStrategy::NestedDissection(NestedDissectionOptions::default()),
        )?,
    ];
    if config.with_native_spral {
        numeric.push(validate_native_spral_strategy(case, native_spral)?);
    } else {
        numeric.push(NumericValidationResult {
            strategy: NumericStrategy::NativeSpral,
            outcome: ValidationOutcome::SkippedNotRequested,
            factor_elapsed_ms: 0.0,
            solve_elapsed_ms: 0.0,
            refactor_elapsed_ms: None,
            metrics: None,
            notes: Vec::new(),
        });
    }

    for result in &ordering {
        if result.outcome == ValidationOutcome::Failed {
            failures.push(format!("ordering failure in {:?}", result.method));
        }
    }
    for result in &symbolic {
        if result.outcome == ValidationOutcome::Failed {
            failures.push(format!(
                "symbolic analysis failure in {:?}",
                result.strategy
            ));
        }
    }
    for result in &numeric {
        if result.outcome == ValidationOutcome::Failed {
            failures.push(format!(
                "numeric factorization failure in {:?}",
                result.strategy
            ));
        }
    }

    Ok(CaseValidationReport {
        case: case.metadata.clone(),
        ordering,
        symbolic,
        numeric,
        failures,
    })
}

fn validate_rust_nested_dissection(
    case: &LoadedCorpusCase,
    component_count: usize,
    summary: OrderingSummary,
    elapsed_ms: f64,
    natural_fill_nnz: usize,
) -> Result<OrderingValidationResult> {
    let exact = exact_symbolic_metrics(&case.matrix, &summary.permutation)?;
    let separator_fraction = if case.matrix.dimension() == 0 {
        0.0
    } else {
        summary.stats.max_separator_size as f64 / case.matrix.dimension() as f64
    };
    Ok(OrderingValidationResult {
        method: OrderingMethod::RustNestedDissection,
        outcome: ValidationOutcome::Passed,
        elapsed_ms,
        metrics: Some(OrderingQualityMetrics {
            permutation_valid: permutation_is_valid(&summary.permutation, case.matrix.dimension()),
            fill_nnz: exact.fill_nnz,
            fill_ratio_vs_natural: ratio(exact.fill_nnz, natural_fill_nnz),
            etree_height: exact.etree_height,
            memory_bytes: exact.memory_bytes,
            max_separator_fraction: Some(separator_fraction),
            component_count,
        }),
        notes: vec![format!(
            "separator_vertices={}, separator_calls={}",
            summary.stats.separator_vertices, summary.stats.separator_calls
        )],
    })
}

fn validate_external_permutation(
    case: &LoadedCorpusCase,
    component_count: usize,
    method: OrderingMethod,
    permutation_result: Result<(Permutation, f64)>,
    natural_fill_nnz: usize,
    separator_fraction: Option<f64>,
) -> Result<OrderingValidationResult> {
    Ok(match permutation_result {
        Ok((permutation, elapsed_ms)) => {
            let exact = exact_symbolic_metrics(&case.matrix, &permutation)?;
            OrderingValidationResult {
                method,
                outcome: ValidationOutcome::Passed,
                elapsed_ms,
                metrics: Some(OrderingQualityMetrics {
                    permutation_valid: permutation_is_valid(&permutation, case.matrix.dimension()),
                    fill_nnz: exact.fill_nnz,
                    fill_ratio_vs_natural: ratio(exact.fill_nnz, natural_fill_nnz),
                    etree_height: exact.etree_height,
                    memory_bytes: exact.memory_bytes,
                    max_separator_fraction: separator_fraction,
                    component_count,
                }),
                notes: Vec::new(),
            }
        }
        Err(error) => OrderingValidationResult {
            method,
            outcome: if method == OrderingMethod::NativeMetis {
                ValidationOutcome::SkippedReferenceUnavailable
            } else {
                ValidationOutcome::Failed
            },
            elapsed_ms: 0.0,
            metrics: None,
            notes: vec![error.to_string()],
        },
    })
}

fn validate_symbolic_strategy(
    case: &LoadedCorpusCase,
    strategy: SymbolicStrategy,
    ordering: OrderingStrategy,
) -> Result<SymbolicValidationResult> {
    let started = Instant::now();
    let matrix = SymmetricCscMatrix::new(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        None,
    )?;
    let result = analyse(matrix, &SsidsOptions { ordering });
    Ok(match result {
        Ok((symbolic, info)) => {
            let exact = if case.metadata.exact_oracle {
                Some(exact_symbolic_metrics(&case.matrix, &symbolic.permutation)?)
            } else {
                None
            };
            let exact_fill_pattern_match = exact
                .as_ref()
                .map(|oracle| oracle.column_pattern == symbolic.column_pattern);
            let exact_column_counts_match = exact
                .as_ref()
                .map(|oracle| oracle.column_counts == symbolic.column_counts);
            let fill_nnz = symbolic.column_counts.iter().sum();
            let memory_bytes =
                crate::metrics::symbolic_memory_bytes(case.matrix.dimension(), fill_nnz);
            let outcome = if exact_fill_pattern_match.is_some_and(|value| !value)
                || exact_column_counts_match.is_some_and(|value| !value)
                || !tree_parent_validity(&symbolic.elimination_tree)
            {
                ValidationOutcome::Failed
            } else {
                ValidationOutcome::Passed
            };
            SymbolicValidationResult {
                strategy,
                outcome,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                metrics: Some(SymbolicAnalysisMetrics {
                    fill_nnz,
                    exact_fill_pattern_match,
                    exact_column_counts_match,
                    tree_parent_valid: tree_parent_validity(&symbolic.elimination_tree),
                    etree_height: crate::metrics::elimination_tree_height(
                        &symbolic.elimination_tree,
                    ),
                    supernode_count: info.supernode_count,
                    memory_bytes,
                }),
                notes: vec![format!("ordering_kind={}", info.ordering_kind)],
            }
        }
        Err(error) => SymbolicValidationResult {
            strategy,
            outcome: ValidationOutcome::Failed,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            metrics: None,
            notes: vec![error.to_string()],
        },
    })
}

fn validate_numeric_strategy(
    case: &LoadedCorpusCase,
    strategy: NumericStrategy,
    ordering: OrderingStrategy,
) -> Result<NumericValidationResult> {
    if case.matrix.dimension() > NUMERIC_VALIDATION_MAX_DIM {
        return Ok(NumericValidationResult {
            strategy,
            outcome: ValidationOutcome::SkippedNotImplemented,
            factor_elapsed_ms: 0.0,
            solve_elapsed_ms: 0.0,
            refactor_elapsed_ms: None,
            metrics: None,
            notes: vec![format!(
                "v1 sparse numeric validation is limited to dimension <= {NUMERIC_VALIDATION_MAX_DIM}"
            )],
        });
    }

    let values = synthetic_numeric_values(case);
    let matrix = SymmetricCscMatrix::new(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        Some(&values),
    )?;
    let numeric_options = numeric_factor_options(case);
    let (symbolic, info) = analyse(matrix, &SsidsOptions { ordering })?;
    let factor_started = Instant::now();
    let (mut factor, factor_info) = factorize(matrix, &symbolic, &numeric_options)?;
    let factor_elapsed_ms = factor_started.elapsed().as_secs_f64() * 1000.0;

    let expected_solution = synthetic_expected_solution(case.matrix.dimension());
    let rhs = lower_csc_matvec(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &values,
        &expected_solution,
    );
    let solve_started = Instant::now();
    let solution = factor.solve(&rhs)?;
    let solve_elapsed_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
    let solve_residual_inf_norm = lower_csc_residual_inf_norm(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &values,
        &solution,
        &rhs,
    );
    let solution_inf_error = inf_norm_diff(&solution, &expected_solution);

    let inertia_match = if case.matrix.dimension() <= NUMERIC_EXACT_INERTIA_MAX_DIM {
        let dense = dense_from_lower_csc(
            case.matrix.dimension(),
            case.matrix.col_ptrs(),
            case.matrix.row_indices(),
            &values,
        );
        let exact_inertia = exact_dense_inertia(&dense, numeric_options.inertia_zero_tol);
        Some(exact_inertia == factor.inertia())
    } else {
        None
    };

    let updated_values = synthetic_refactor_values(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &values,
    );
    let updated_matrix = SymmetricCscMatrix::new(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        Some(&updated_values),
    )?;
    let refactor_started = Instant::now();
    let refactor_info = factor.refactorize(updated_matrix)?;
    let refactor_elapsed_ms = Some(refactor_started.elapsed().as_secs_f64() * 1000.0);
    let refactor_speedup_vs_factor = refactor_elapsed_ms
        .filter(|value| *value > 0.0)
        .map(|value| factor_elapsed_ms / value);
    let refactorization_residual_max_abs = Some(refactor_info.factorization_residual_max_abs);
    let updated_rhs = lower_csc_matvec(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &updated_values,
        &expected_solution,
    );
    let updated_solution = factor.solve(&updated_rhs)?;
    let refactorization_solve_residual_inf_norm = Some(lower_csc_residual_inf_norm(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &updated_values,
        &updated_solution,
        &updated_rhs,
    ));
    let refactorization_solution_inf_error =
        Some(inf_norm_diff(&updated_solution, &expected_solution));

    let outcome = if factor_info.factorization_residual_max_abs <= NUMERIC_VALIDATION_TOL
        && solve_residual_inf_norm <= NUMERIC_VALIDATION_TOL
        && solution_inf_error <= NUMERIC_VALIDATION_TOL
        && inertia_match != Some(false)
        && refactorization_residual_max_abs.is_none_or(|value| value <= NUMERIC_VALIDATION_TOL)
        && refactorization_solve_residual_inf_norm
            .is_none_or(|value| value <= NUMERIC_VALIDATION_TOL)
        && refactorization_solution_inf_error.is_none_or(|value| value <= NUMERIC_VALIDATION_TOL)
    {
        ValidationOutcome::Passed
    } else {
        ValidationOutcome::Failed
    };

    let mut notes = vec![format!("ordering_kind={}", info.ordering_kind)];
    if case.matrix.dimension() > NUMERIC_EXACT_INERTIA_MAX_DIM {
        notes.push(format!(
            "exact dense inertia oracle skipped above dimension {NUMERIC_EXACT_INERTIA_MAX_DIM}"
        ));
    }

    Ok(NumericValidationResult {
        strategy,
        outcome,
        factor_elapsed_ms,
        solve_elapsed_ms,
        refactor_elapsed_ms,
        metrics: Some(NumericFactorizationMetrics {
            factorization_residual_max_abs: factor_info.factorization_residual_max_abs,
            solve_residual_inf_norm,
            solution_inf_error,
            refactorization_residual_max_abs,
            refactorization_solve_residual_inf_norm,
            refactorization_solution_inf_error,
            inertia_match,
            positive_eigenvalues: factor.inertia().positive,
            negative_eigenvalues: factor.inertia().negative,
            zero_eigenvalues: factor.inertia().zero,
            stored_nnz: factor.stored_nnz(),
            factor_storage_bytes: factor.factor_bytes(),
            supernode_count: factor.supernode_count(),
            max_supernode_width: factor.max_supernode_width(),
            two_by_two_pivots: factor.pivot_stats().two_by_two_pivots,
            delayed_pivots: factor.pivot_stats().delayed_pivots,
            refactor_speedup_vs_factor,
        }),
        notes,
    })
}

fn validate_native_spral_strategy(
    case: &LoadedCorpusCase,
    native_spral: Option<&NativeSpralReference>,
) -> Result<NumericValidationResult> {
    let Some(native_spral) = native_spral else {
        return Ok(NumericValidationResult {
            strategy: NumericStrategy::NativeSpral,
            outcome: ValidationOutcome::SkippedReferenceUnavailable,
            factor_elapsed_ms: 0.0,
            solve_elapsed_ms: 0.0,
            refactor_elapsed_ms: None,
            metrics: None,
            notes: vec!["native SPRAL library could not be loaded locally".into()],
        });
    };
    if case.matrix.dimension() > NUMERIC_VALIDATION_MAX_DIM {
        return Ok(NumericValidationResult {
            strategy: NumericStrategy::NativeSpral,
            outcome: ValidationOutcome::SkippedNotImplemented,
            factor_elapsed_ms: 0.0,
            solve_elapsed_ms: 0.0,
            refactor_elapsed_ms: None,
            metrics: None,
            notes: vec![format!(
                "native SPRAL validation is limited to dimension <= {NUMERIC_VALIDATION_MAX_DIM}"
            )],
        });
    }

    let values = synthetic_numeric_values(case);
    let matrix = SymmetricCscMatrix::new(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        Some(&values),
    )?;
    let expected_solution = synthetic_expected_solution(case.matrix.dimension());
    let rhs = lower_csc_matvec(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &values,
        &expected_solution,
    );
    let updated_values = synthetic_refactor_values(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &values,
    );
    let updated_rhs = lower_csc_matvec(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &updated_values,
        &expected_solution,
    );
    let result =
        native_spral.factor_solve(matrix, &rhs, Some(&updated_values), Some(&updated_rhs))?;
    let solve_residual_inf_norm = lower_csc_residual_inf_norm(
        case.matrix.dimension(),
        case.matrix.col_ptrs(),
        case.matrix.row_indices(),
        &values,
        &result.solution,
        &rhs,
    );
    let solution_inf_error = inf_norm_diff(&result.solution, &expected_solution);
    let refactorization_solve_residual_inf_norm =
        result.refactor_solution.as_ref().map(|solution| {
            lower_csc_residual_inf_norm(
                case.matrix.dimension(),
                case.matrix.col_ptrs(),
                case.matrix.row_indices(),
                &updated_values,
                solution,
                &updated_rhs,
            )
        });
    let refactorization_solution_inf_error = result
        .refactor_solution
        .as_ref()
        .map(|solution| inf_norm_diff(solution, &expected_solution));

    let outcome = if solve_residual_inf_norm <= NUMERIC_VALIDATION_TOL
        && solution_inf_error <= NUMERIC_VALIDATION_TOL
        && refactorization_solve_residual_inf_norm
            .is_none_or(|value| value <= NUMERIC_VALIDATION_TOL)
        && refactorization_solution_inf_error.is_none_or(|value| value <= NUMERIC_VALIDATION_TOL)
    {
        ValidationOutcome::Passed
    } else {
        ValidationOutcome::Failed
    };

    Ok(NumericValidationResult {
        strategy: NumericStrategy::NativeSpral,
        outcome,
        factor_elapsed_ms: result.factor_elapsed_ms,
        solve_elapsed_ms: result.solve_elapsed_ms,
        refactor_elapsed_ms: result.refactor_elapsed_ms,
        metrics: None,
        notes: vec![
            format!("num_neg={}", result.num_neg),
            format!("num_two={}", result.num_two),
            format!("num_delay={}", result.num_delay),
            format!("num_sup={}", result.num_sup),
            format!("max_supernode={}", result.max_supernode),
            format!("solve_residual={solve_residual_inf_norm:.3e}"),
            format!("solution_error={solution_inf_error:.3e}"),
        ],
    })
}

fn numeric_factor_options(_case: &LoadedCorpusCase) -> NumericFactorOptions {
    NumericFactorOptions::default()
}

fn run_robustness_suite() -> Vec<RobustnessCaseResult> {
    vec![
        robustness_expect_error("csr_empty_offsets", "metis_ordering::CsrGraph::new", || {
            metis_ordering::CsrGraph::new(Vec::new(), Vec::new())
                .map(|_| ())
                .map_err(|error| error.to_string())
        }),
        robustness_expect_error(
            "csr_unsorted_neighbors",
            "metis_ordering::CsrGraph::new",
            || {
                metis_ordering::CsrGraph::new(vec![0, 2], vec![1, 0])
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_error("csr_out_of_bounds", "metis_ordering::CsrGraph::new", || {
            metis_ordering::CsrGraph::new(vec![0, 1, 1], vec![4])
                .map(|_| ())
                .map_err(|error| error.to_string())
        }),
        robustness_expect_error(
            "csc_bad_value_length",
            "spral_ssids::SymmetricCscMatrix::new",
            || {
                SymmetricCscMatrix::new(2, &[0, 1, 2], &[0, 1], Some(&[1.0]))
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_error(
            "csc_nonmonotone_ptrs",
            "spral_ssids::SymmetricCscMatrix::new",
            || {
                SymmetricCscMatrix::new(2, &[0, 2, 1], &[0, 1], None)
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_error(
            "numeric_factorize_missing_values",
            "spral_ssids::factorize",
            || {
                let matrix = SymmetricCscMatrix::new(2, &[0, 2, 3], &[0, 1, 1], None)
                    .map_err(|error| error.to_string())?;
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions::default()).map_err(|error| error.to_string())?;
                factorize(matrix, &symbolic, &NumericFactorOptions::default())
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_error(
            "numeric_factorize_nonfinite_value",
            "spral_ssids::factorize",
            || {
                let matrix =
                    SymmetricCscMatrix::new(2, &[0, 2, 3], &[0, 1, 1], Some(&[2.0, f64::NAN, 1.0]))
                        .map_err(|error| error.to_string())?;
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions::default()).map_err(|error| error.to_string())?;
                factorize(matrix, &symbolic, &NumericFactorOptions::default())
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_success("analyse_adversarial_ladder", "spral_ssids::analyse", || {
            let matrix = crate::corpus::SymmetricPatternMatrix::from_undirected_edges(
                14,
                &[(0, 7), (0, 1), (7, 8), (1, 8), (1, 2), (8, 9)],
            )
            .map_err(|error| error.to_string())?;
            let matrix = SymmetricCscMatrix::new(
                matrix.dimension(),
                matrix.col_ptrs(),
                matrix.row_indices(),
                None,
            )
            .map_err(|error| error.to_string())?;
            analyse(matrix, &SsidsOptions::default())
                .map(|_| ())
                .map_err(|error| error.to_string())
        }),
        robustness_expect_success(
            "nested_dissection_high_degree_hub",
            "metis_ordering::nested_dissection_order",
            || {
                let edges = (1..32).map(|leaf| (0, leaf)).collect::<Vec<_>>();
                let graph = metis_ordering::CsrGraph::from_edges(32, &edges)
                    .map_err(|error| error.to_string())?;
                nested_dissection_order(&graph, &NestedDissectionOptions::default())
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_success(
            "numeric_factorize_small_indefinite",
            "spral_ssids::factorize",
            || {
                let col_ptrs = vec![0, 2, 4, 5];
                let row_indices = vec![0, 1, 1, 2, 2];
                let values = vec![3.0, 0.1, -2.0, 0.2, 1.5];
                let matrix = SymmetricCscMatrix::new(3, &col_ptrs, &row_indices, Some(&values))
                    .map_err(|error| error.to_string())?;
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions::default()).map_err(|error| error.to_string())?;
                let (mut factor, _) =
                    factorize(matrix, &symbolic, &NumericFactorOptions::default())
                        .map_err(|error| error.to_string())?;
                let mut rhs = vec![1.0, -2.0, 0.5];
                factor
                    .solve_in_place(&mut rhs)
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_error(
            "numeric_solve_dimension_mismatch",
            "spral_ssids::NumericFactor::solve_in_place",
            || {
                let col_ptrs = vec![0, 2, 4, 5];
                let row_indices = vec![0, 1, 1, 2, 2];
                let values = vec![3.0, 0.1, -2.0, 0.2, 1.5];
                let matrix = SymmetricCscMatrix::new(3, &col_ptrs, &row_indices, Some(&values))
                    .map_err(|error| error.to_string())?;
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions::default()).map_err(|error| error.to_string())?;
                let (mut factor, _) =
                    factorize(matrix, &symbolic, &NumericFactorOptions::default())
                        .map_err(|error| error.to_string())?;
                let mut rhs = vec![1.0, -2.0];
                factor
                    .solve_in_place(&mut rhs)
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_error(
            "numeric_refactorize_pattern_mismatch",
            "spral_ssids::NumericFactor::refactorize",
            || {
                let original_col_ptrs = vec![0, 2, 4, 6, 7];
                let original_row_indices = vec![0, 1, 1, 2, 2, 3, 3];
                let original_values = vec![4.0, -1.0, 4.0, -1.0, 4.0, -1.0, 3.0];
                let matrix = SymmetricCscMatrix::new(
                    4,
                    &original_col_ptrs,
                    &original_row_indices,
                    Some(&original_values),
                )
                .map_err(|error| error.to_string())?;
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions::default()).map_err(|error| error.to_string())?;
                let (mut factor, _) =
                    factorize(matrix, &symbolic, &NumericFactorOptions::default())
                        .map_err(|error| error.to_string())?;

                let changed_col_ptrs = vec![0, 2, 5, 7, 8];
                let changed_row_indices = vec![0, 1, 1, 2, 3, 2, 3, 3];
                let changed_values = vec![4.0, -1.0, 4.0, -1.0, 0.25, 4.0, -1.0, 3.0];
                let changed_matrix = SymmetricCscMatrix::new(
                    4,
                    &changed_col_ptrs,
                    &changed_row_indices,
                    Some(&changed_values),
                )
                .map_err(|error| error.to_string())?;
                factor
                    .refactorize(changed_matrix)
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
        ),
        robustness_expect_success(
            "numeric_refactorize_repeated_delayed_chain",
            "spral_ssids::NumericFactor::refactorize",
            || {
                let col_ptrs = vec![0, 2, 4, 6, 8, 10, 11];
                let row_indices = vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
                let base_values = delayed_chain_values(0.0);
                let matrix =
                    SymmetricCscMatrix::new(6, &col_ptrs, &row_indices, Some(&base_values))
                        .map_err(|error| error.to_string())?;
                let (symbolic, _) = analyse(
                    matrix,
                    &SsidsOptions {
                        ordering: OrderingStrategy::Natural,
                    },
                )
                .map_err(|error| error.to_string())?;
                let (mut factor, _) =
                    factorize(matrix, &symbolic, &NumericFactorOptions::default())
                        .map_err(|error| error.to_string())?;
                for shift in [0.08, -0.05, 0.14, -0.09, 0.2] {
                    let updated_values = delayed_chain_values(shift);
                    let updated_matrix =
                        SymmetricCscMatrix::new(6, &col_ptrs, &row_indices, Some(&updated_values))
                            .map_err(|error| error.to_string())?;
                    factor
                        .refactorize(updated_matrix)
                        .map_err(|error| error.to_string())?;
                    let expected = (0..6)
                        .map(|index| 0.75 + index as f64 * 0.15)
                        .collect::<Vec<_>>();
                    let rhs =
                        lower_csc_matvec(6, &col_ptrs, &row_indices, &updated_values, &expected);
                    let solution = factor.solve(&rhs).map_err(|error| error.to_string())?;
                    let residual = lower_csc_residual_inf_norm(
                        6,
                        &col_ptrs,
                        &row_indices,
                        &updated_values,
                        &solution,
                        &rhs,
                    );
                    if residual > 1e-7 {
                        return Err(format!(
                            "refactorized solve residual {residual:e} exceeded tolerance"
                        ));
                    }
                }
                Ok(())
            },
        ),
        robustness_expect_success(
            "numeric_factorize_deterministic_saddle_kkt",
            "spral_ssids::factorize",
            || {
                let col_ptrs = vec![0, 3, 7, 10, 13, 15, 16, 17, 18];
                let row_indices = vec![0, 1, 5, 1, 2, 5, 7, 2, 3, 6, 3, 4, 6, 4, 7, 5, 6, 7];
                let values = saddle_kkt_values(0.0);
                let matrix = SymmetricCscMatrix::new(8, &col_ptrs, &row_indices, Some(&values))
                    .map_err(|error| error.to_string())?;
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions::default()).map_err(|error| error.to_string())?;
                let rhs_expected = synthetic_expected_solution(8);
                let rhs = lower_csc_matvec(8, &col_ptrs, &row_indices, &values, &rhs_expected);

                let mut baseline = None;
                for _ in 0..3 {
                    let (mut factor, info) =
                        factorize(matrix, &symbolic, &NumericFactorOptions::default())
                            .map_err(|error| error.to_string())?;
                    let solution = factor.solve(&rhs).map_err(|error| error.to_string())?;
                    let residual = lower_csc_residual_inf_norm(
                        8,
                        &col_ptrs,
                        &row_indices,
                        &values,
                        &solution,
                        &rhs,
                    );
                    if residual > 1e-8 || info.factorization_residual_max_abs > 1e-8 {
                        return Err(format!(
                            "deterministic KKT solve residuals exceeded tolerance: factor={:.3e} solve={residual:.3e}",
                            info.factorization_residual_max_abs
                        ));
                    }
                    let snapshot = (
                        factor.inertia(),
                        factor.pivot_stats(),
                        factor.stored_nnz(),
                        factor.factor_bytes(),
                        solution,
                    );
                    if let Some(reference) = &baseline {
                        if &snapshot != reference {
                            return Err(
                                "repeated KKT factorization changed observable numeric state"
                                    .into(),
                            );
                        }
                    } else {
                        baseline = Some(snapshot);
                    }
                }
                Ok(())
            },
        ),
    ]
}

fn delayed_chain_values(shift: f64) -> Vec<f64> {
    vec![
        1e-8 * (1.0 + 0.25 * shift),
        1.0 - 0.05 * shift,
        2.0 + 0.15 * shift,
        0.25 + 0.04 * shift,
        3.0 - 0.1 * shift,
        -0.5 + 0.03 * shift,
        2.5 + 0.12 * shift,
        0.2 - 0.02 * shift,
        1.75 + 0.08 * shift,
        -0.4 + 0.01 * shift,
        1.5 - 0.06 * shift,
    ]
}

fn saddle_kkt_values(shift: f64) -> Vec<f64> {
    vec![
        4.0 + 0.1 * shift,
        -1.0 + 0.02 * shift,
        1.0,
        4.0 - 0.05 * shift,
        -1.0 + 0.03 * shift,
        -0.25 + 0.01 * shift,
        0.5 - 0.02 * shift,
        3.5 + 0.04 * shift,
        -0.5 + 0.02 * shift,
        0.75 - 0.03 * shift,
        3.25 - 0.06 * shift,
        -0.75 + 0.01 * shift,
        -1.0 + 0.04 * shift,
        2.75 + 0.05 * shift,
        0.8 - 0.03 * shift,
        -0.1 - 0.01 * shift,
        -0.15 - 0.01 * shift,
        -0.2 - 0.02 * shift,
    ]
}

fn synthetic_kkt_values(case: &LoadedCorpusCase, primal_dimension: usize, shift: f64) -> Vec<f64> {
    let dimension = case.matrix.dimension();
    let col_ptrs = case.matrix.col_ptrs();
    let row_indices = case.matrix.row_indices();
    let mut row_abs_sums = vec![0.0; dimension];
    let mut values = vec![0.0; row_indices.len()];
    for col in 0..dimension {
        let start = col_ptrs[col];
        let end = col_ptrs[col + 1];
        for (&row, value_slot) in row_indices[start..end]
            .iter()
            .zip(values[start..end].iter_mut())
        {
            if row == col {
                continue;
            }
            let primal_primal = row < primal_dimension && col < primal_dimension;
            let value = if primal_primal {
                -(0.2 + 0.025 * ((row + col) % 4) as f64) * (1.0 + 0.1 * shift)
            } else {
                let coupling_sign = if ((row + col) & 1) == 0 { 1.0 } else { -1.0 };
                coupling_sign * (0.55 + 0.1 * ((row + 2 * col) % 3) as f64) * (1.0 + 0.05 * shift)
            };
            *value_slot = value;
            row_abs_sums[row] += value.abs();
            row_abs_sums[col] += value.abs();
        }
    }
    for col in 0..dimension {
        let start = col_ptrs[col];
        let end = col_ptrs[col + 1];
        for (&row, value_slot) in row_indices[start..end]
            .iter()
            .zip(values[start..end].iter_mut())
        {
            if row == col {
                *value_slot = if col < primal_dimension {
                    row_abs_sums[col] + 1.25 + col as f64 * 0.05 + 0.1 * shift
                } else {
                    -(0.08 + 0.03 * (col - primal_dimension) as f64) * (1.0 + 0.1 * shift)
                };
                break;
            }
        }
    }
    values
}

fn robustness_expect_error(
    scenario: &str,
    target: &str,
    operation: impl FnMut() -> Result<(), String>,
) -> RobustnessCaseResult {
    robustness_result(scenario, target, operation, true)
}

fn robustness_expect_success(
    scenario: &str,
    target: &str,
    operation: impl FnMut() -> Result<(), String>,
) -> RobustnessCaseResult {
    robustness_result(scenario, target, operation, false)
}

fn robustness_result(
    scenario: &str,
    target: &str,
    mut operation: impl FnMut() -> Result<(), String>,
    expect_error: bool,
) -> RobustnessCaseResult {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(&mut operation));
    match outcome {
        Ok(Ok(())) => {
            let (outcome, notes) = if expect_error {
                (
                    ValidationOutcome::Failed,
                    vec!["invalid input was accepted unexpectedly".into()],
                )
            } else {
                (ValidationOutcome::Passed, Vec::new())
            };
            RobustnessCaseResult {
                scenario: scenario.to_string(),
                target: target.to_string(),
                outcome,
                duration_ms: started.elapsed().as_secs_f64() * 1000.0,
                error_kind: None,
                notes,
            }
        }
        Ok(Err(error)) => RobustnessCaseResult {
            scenario: scenario.to_string(),
            target: target.to_string(),
            outcome: if expect_error {
                ValidationOutcome::Passed
            } else {
                ValidationOutcome::Failed
            },
            duration_ms: started.elapsed().as_secs_f64() * 1000.0,
            error_kind: Some(error),
            notes: if expect_error {
                vec!["typed error path exercised successfully".into()]
            } else {
                vec!["valid adversarial case returned an error".into()]
            },
        },
        Err(_) => RobustnessCaseResult {
            scenario: scenario.to_string(),
            target: target.to_string(),
            outcome: ValidationOutcome::Failed,
            duration_ms: started.elapsed().as_secs_f64() * 1000.0,
            error_kind: Some("panic".into()),
            notes: vec!["unexpected panic during robustness scenario".into()],
        },
    }
}

fn timed<T>(operation: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let started = Instant::now();
    operation().map(|value| (value, started.elapsed().as_secs_f64() * 1000.0))
}

fn ratio(value: usize, baseline: usize) -> Option<f64> {
    (baseline != 0).then_some(value as f64 / baseline as f64)
}

fn synthetic_numeric_values(case: &LoadedCorpusCase) -> Vec<f64> {
    if case
        .metadata
        .tags
        .iter()
        .any(|tag| matches!(tag, crate::model::CorpusTag::TwoByTwoPivot))
    {
        return vec![0.0, 1.0, 0.25, 0.0, 0.5, 2.0];
    }
    if case
        .metadata
        .tags
        .iter()
        .any(|tag| matches!(tag, crate::model::CorpusTag::DelayedPivot))
    {
        if case.matrix.dimension() == 6 {
            return vec![1e-8, 1.0, 2.0, 0.25, 3.0, -0.5, 2.5, 0.2, 1.75, -0.4, 1.5];
        }
        return vec![1e-4, 1e-4, 1.0, 1e-4, 0.5, 0.0];
    }
    match case.metadata.case_id.as_str() {
        "saddle_kkt_8" => return saddle_kkt_values(0.0),
        "banded_kkt_12" => return synthetic_kkt_values(case, 8, 0.0),
        _ => {}
    }
    let dimension = case.matrix.dimension();
    let col_ptrs = case.matrix.col_ptrs();
    let row_indices = case.matrix.row_indices();
    let indefinite = case.metadata.tags.iter().any(|tag| {
        matches!(
            tag,
            crate::model::CorpusTag::ArrowKkt
                | crate::model::CorpusTag::Adversarial
                | crate::model::CorpusTag::Workspace
        )
    });
    let mut row_abs_sums = vec![0.0; dimension];
    let mut values = vec![0.0; row_indices.len()];
    for col in 0..dimension {
        let start = col_ptrs[col];
        let end = col_ptrs[col + 1];
        for (&row, value_slot) in row_indices[start..end]
            .iter()
            .zip(values[start..end].iter_mut())
        {
            if row == col {
                continue;
            }
            let magnitude = 0.05 * (1.0 + ((row + 3 * col) % 5) as f64);
            let sign = if indefinite && (row + col).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let value = sign * magnitude;
            *value_slot = value;
            row_abs_sums[row] += value.abs();
            row_abs_sums[col] += value.abs();
        }
    }
    for (col, row_abs_sum) in row_abs_sums.iter().copied().enumerate().take(dimension) {
        let start = col_ptrs[col];
        let end = col_ptrs[col + 1];
        for (&row, value_slot) in row_indices[start..end]
            .iter()
            .zip(values[start..end].iter_mut())
        {
            if row == col {
                let diagonal_sign = if indefinite && col % 4 == 1 {
                    -1.0
                } else {
                    1.0
                };
                *value_slot = diagonal_sign * (row_abs_sum + 1.0 + col as f64 * 0.1);
                break;
            }
        }
    }
    values
}

fn synthetic_refactor_values(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
) -> Vec<f64> {
    let mut updated = values.to_vec();
    for col in 0..dimension {
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[index];
            updated[index] = if row == col {
                values[index] * (1.05 + 0.01 * (col % 3) as f64)
            } else {
                values[index] * 0.9
            };
        }
    }
    updated
}

fn synthetic_expected_solution(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|index| 1.0 + index as f64 * 0.25)
        .collect()
}

fn lower_csc_matvec(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let mut result = vec![0.0; dimension];
    for col in 0..dimension {
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[index];
            let value = values[index];
            result[row] += value * x[col];
            if row != col {
                result[col] += value * x[row];
            }
        }
    }
    result
}

fn lower_csc_residual_inf_norm(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    x: &[f64],
    rhs: &[f64],
) -> f64 {
    lower_csc_matvec(dimension, col_ptrs, row_indices, values, x)
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn inf_norm_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn dense_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(dimension, dimension);
    for col in 0..dimension {
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[index];
            let value = values[index];
            dense[(row, col)] += value;
            if row != col {
                dense[(col, row)] += value;
            }
        }
    }
    dense
}

fn exact_dense_inertia(matrix: &DMatrix<f64>, zero_tol: f64) -> Inertia {
    if matrix.nrows() == 0 {
        return Inertia {
            positive: 0,
            negative: 0,
            zero: 0,
        };
    }
    let eigen = SymmetricEigen::new(matrix.clone());
    let mut inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    for value in eigen.eigenvalues.iter().copied() {
        if value > zero_tol {
            inertia.positive += 1;
        } else if value < -zero_tol {
            inertia.negative += 1;
        } else {
            inertia.zero += 1;
        }
    }
    inertia
}

fn collect_environment() -> EnvironmentStamp {
    EnvironmentStamp {
        operating_system: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        rustc_version: command_stdout("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
        git_sha: command_stdout("git", &["rev-parse", "HEAD"]),
        hostname: command_stdout("hostname", &[]),
    }
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    Some(stdout.trim().to_string())
}

fn iso8601_utc() -> String {
    command_stdout("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".into())
}
