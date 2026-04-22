use approx::assert_abs_diff_eq;
use ssids_rs::{
    Inertia, NumericFactorOptions, OrderingStrategy, SsidsError, SsidsOptions, SymmetricCscMatrix,
    analyse, factorize,
};

fn dense_mul(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(value, x)| value * x)
                .sum::<f64>()
        })
        .collect()
}

fn residual_inf_norm(matrix: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
    dense_mul(matrix, x)
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn dense_to_lower_csc(matrix: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let dimension = matrix.len();
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..dimension {
        for (row, dense_row) in matrix.iter().enumerate().skip(col) {
            let value = dense_row[col];
            if value.abs() > 1e-15 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices, values)
}

fn delayed_chain_dense(shift: f64) -> Vec<Vec<f64>> {
    vec![
        vec![
            1e-8 * (1.0 + 0.25 * shift),
            1.0 - 0.05 * shift,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        vec![
            1.0 - 0.05 * shift,
            2.0 + 0.15 * shift,
            0.25 + 0.04 * shift,
            0.0,
            0.0,
            0.0,
        ],
        vec![
            0.0,
            0.25 + 0.04 * shift,
            3.0 - 0.1 * shift,
            -0.5 + 0.03 * shift,
            0.0,
            0.0,
        ],
        vec![
            0.0,
            0.0,
            -0.5 + 0.03 * shift,
            2.5 + 0.12 * shift,
            0.2 - 0.02 * shift,
            0.0,
        ],
        vec![
            0.0,
            0.0,
            0.0,
            0.2 - 0.02 * shift,
            1.75 + 0.08 * shift,
            -0.4 + 0.01 * shift,
        ],
        vec![0.0, 0.0, 0.0, 0.0, -0.4 + 0.01 * shift, 1.5 - 0.06 * shift],
    ]
}

fn saddle_kkt_dense(shift: f64) -> Vec<Vec<f64>> {
    vec![
        vec![
            4.0 + 0.1 * shift,
            -1.0 + 0.02 * shift,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        vec![
            -1.0 + 0.02 * shift,
            4.0 - 0.05 * shift,
            -1.0 + 0.03 * shift,
            0.0,
            0.0,
            -0.25 + 0.01 * shift,
            0.0,
            0.5 - 0.02 * shift,
        ],
        vec![
            0.0,
            -1.0 + 0.03 * shift,
            3.5 + 0.04 * shift,
            -0.5 + 0.02 * shift,
            0.0,
            0.0,
            0.75 - 0.03 * shift,
            0.0,
        ],
        vec![
            0.0,
            0.0,
            -0.5 + 0.02 * shift,
            3.25 - 0.06 * shift,
            -0.75 + 0.01 * shift,
            0.0,
            -1.0 + 0.04 * shift,
            0.0,
        ],
        vec![
            0.0,
            0.0,
            0.0,
            -0.75 + 0.01 * shift,
            2.75 + 0.05 * shift,
            0.0,
            0.0,
            0.8 - 0.03 * shift,
        ],
        vec![
            1.0,
            -0.25 + 0.01 * shift,
            0.0,
            0.0,
            0.0,
            -0.1 - 0.01 * shift,
            0.0,
            0.0,
        ],
        vec![
            0.0,
            0.0,
            0.75 - 0.03 * shift,
            -1.0 + 0.04 * shift,
            0.0,
            0.0,
            -0.15 - 0.01 * shift,
            0.0,
        ],
        vec![
            0.0,
            0.5 - 0.02 * shift,
            0.0,
            0.0,
            0.8 - 0.03 * shift,
            0.0,
            0.0,
            -0.2 - 0.02 * shift,
        ],
    ]
}

#[test]
fn numeric_factorization_solves_spd_system() {
    let dense = vec![
        vec![4.0, -1.0, 0.0, 0.0],
        vec![-1.0, 4.0, -1.0, 0.0],
        vec![0.0, -1.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 3.0],
    ];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");
    assert!(info.factorization_residual_max_abs <= 1e-12);
    assert!(factor.supernode_count() > 0);
    assert!(factor.max_supernode_width() > 0);
    assert!(factor.factor_bytes() >= factor.stored_nnz() * std::mem::size_of::<f64>());
    assert_eq!(
        factor.inertia(),
        Inertia {
            positive: 4,
            negative: 0,
            zero: 0,
        }
    );

    let expected = vec![1.0, -2.0, 0.5, 3.0];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-10);
}

#[test]
fn numeric_factorization_uses_native_relaxed_supernode_ranges() {
    let dimension = 64;
    let mut dense = vec![vec![0.0; dimension]; dimension];
    for idx in 0..dimension {
        dense[idx][idx] = 4.0;
        if idx + 1 < dimension {
            dense[idx][idx + 1] = -1.0;
            dense[idx + 1][idx] = -1.0;
        }
    }
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix =
        SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");
    let (factor, _) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert_eq!(symbolic.supernodes.len(), 2);
    assert_eq!(factor.supernode_count(), symbolic.supernodes.len());
    assert_eq!(factor.max_supernode_width(), 32);
}

#[test]
fn numeric_factorization_reports_indefinite_inertia() {
    let dense = vec![
        vec![3.0, 0.1, 0.0, 0.2],
        vec![0.1, -2.5, 0.15, 0.0],
        vec![0.0, 0.15, 2.0, -0.1],
        vec![0.2, 0.0, -0.1, -1.75],
    ];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-12);
    assert_eq!(
        factor.inertia(),
        Inertia {
            positive: 2,
            negative: 2,
            zero: 0,
        }
    );

    let expected = vec![0.5, -1.0, 2.0, -0.25];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-10);
}

#[test]
fn numeric_factorization_zero_pivot_action_matches_option() {
    let dense = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");

    let (factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");
    assert!(info.factorization_residual_max_abs <= 1e-12);
    assert_eq!(
        factor.inertia(),
        Inertia {
            positive: 0,
            negative: 0,
            zero: 2,
        }
    );

    let error = factorize(
        matrix,
        &symbolic,
        &NumericFactorOptions {
            action_on_zero_pivot: false,
            ..NumericFactorOptions::default()
        },
    )
    .expect_err("zero pivot should fail when action is disabled");
    assert!(matches!(error, SsidsError::NumericalBreakdown { .. }));
}

#[test]
fn numeric_factorization_uses_two_by_two_pivot_for_coupled_indefinite_panel() {
    let dense = vec![
        vec![0.0, 1.0, 0.25],
        vec![1.0, 0.0, 0.5],
        vec![0.25, 0.5, 2.0],
    ];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(3, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-10);
    assert!(factor.pivot_stats().two_by_two_pivots >= 1);

    let expected = vec![1.25, -0.75, 0.5];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-10);
}

#[test]
fn numeric_factorization_uses_two_by_two_pivot_for_width_two_supernode() {
    let dense = vec![vec![0.0, 1.0], vec![1.0, 0.25]];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-10);
    assert_eq!(factor.pivot_stats().two_by_two_pivots, 1);

    let expected = vec![0.75, -1.25];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-10);
}

#[test]
fn numeric_factorization_solves_saddle_point_kkt_system() {
    let dense = saddle_kkt_dense(0.0);
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(8, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-10);
    assert!(factor.inertia().positive > 0);
    assert!(factor.inertia().negative > 0);
    assert_eq!(factor.inertia().zero, 0);

    let expected = vec![1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.75, 1.5];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-9);
}

#[test]
fn numeric_refactorization_updates_factor_values() {
    let original = vec![
        vec![4.0, -1.0, 0.0, 0.0],
        vec![-1.0, 4.0, -1.0, 0.0],
        vec![0.0, -1.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 3.0],
    ];
    let updated = vec![
        vec![5.0, -0.5, 0.0, 0.0],
        vec![-0.5, 4.5, -1.25, 0.0],
        vec![0.0, -1.25, 4.25, -0.75],
        vec![0.0, 0.0, -0.75, 2.5],
    ];

    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&original);
    let matrix = SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let (mut factor, _) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    let (updated_col_ptrs, updated_row_indices, updated_values) = dense_to_lower_csc(&updated);
    let updated_matrix = SymmetricCscMatrix::new(
        4,
        &updated_col_ptrs,
        &updated_row_indices,
        Some(&updated_values),
    )
    .expect("updated csc");
    let info = factor.refactorize(updated_matrix).expect("refactorize");
    assert!(info.factorization_residual_max_abs <= 1e-12);

    let expected = vec![1.5, -0.25, 0.75, 2.0];
    let rhs = dense_mul(&updated, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&updated, &solution, &rhs) <= 1e-10);
}

#[test]
fn multifrontal_factorization_handles_delayed_chain_with_relaxed_amalgamation() {
    let dense = delayed_chain_dense(0.0);
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(6, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-6);
    let expected = vec![1.0, -0.5, 0.75, -1.25, 0.25, 1.5];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-8);
}

#[test]
fn multifrontal_refactorization_reuses_symbolic_front_tree() {
    let original = vec![
        vec![6.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0, 5.0, -1.0, 0.0, 0.0, 0.0],
        vec![0.0, -1.0, 4.5, -0.5, 0.0, 0.0],
        vec![0.0, 0.0, -0.5, 4.0, -0.25, 0.0],
        vec![0.0, 0.0, 0.0, -0.25, 3.5, -0.2],
        vec![0.0, 0.0, 0.0, 0.0, -0.2, 3.0],
    ];
    let updated = vec![
        vec![5.75, -0.75, 0.0, 0.0, 0.0, 0.0],
        vec![-0.75, 5.25, -1.1, 0.0, 0.0, 0.0],
        vec![0.0, -1.1, 4.8, -0.45, 0.0, 0.0],
        vec![0.0, 0.0, -0.45, 3.9, -0.3, 0.0],
        vec![0.0, 0.0, 0.0, -0.3, 3.7, -0.25],
        vec![0.0, 0.0, 0.0, 0.0, -0.25, 2.8],
    ];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&original);
    let matrix = SymmetricCscMatrix::new(6, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let (mut factor, _) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    let (updated_col_ptrs, updated_row_indices, updated_values) = dense_to_lower_csc(&updated);
    let updated_matrix = SymmetricCscMatrix::new(
        6,
        &updated_col_ptrs,
        &updated_row_indices,
        Some(&updated_values),
    )
    .expect("updated csc");
    let info = factor.refactorize(updated_matrix).expect("refactorize");
    assert!(info.factorization_residual_max_abs <= 1e-10);

    let expected = vec![0.5, -1.25, 0.75, 1.0, -0.5, 1.5];
    let rhs = dense_mul(&updated, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
    }
    assert!(residual_inf_norm(&updated, &solution, &rhs) <= 1e-9);
}

#[test]
fn multifrontal_repeated_refactorization_stays_stable_on_relaxed_delayed_chain() {
    let dense = delayed_chain_dense(0.0);
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(6, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("symbolic");
    let (mut factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");
    assert!(info.factorization_residual_max_abs <= 1e-6);

    for (iteration, shift) in [0.08, -0.05, 0.14, -0.09, 0.2].into_iter().enumerate() {
        let updated = delayed_chain_dense(shift);
        let (updated_col_ptrs, updated_row_indices, updated_values) = dense_to_lower_csc(&updated);
        let updated_matrix = SymmetricCscMatrix::new(
            6,
            &updated_col_ptrs,
            &updated_row_indices,
            Some(&updated_values),
        )
        .expect("updated csc");
        let info = factor.refactorize(updated_matrix).expect("refactorize");
        assert!(info.factorization_residual_max_abs <= 1e-6);
        let expected = (0..6)
            .map(|index| 0.5 + iteration as f64 * 0.1 + index as f64 * 0.2)
            .collect::<Vec<_>>();
        let rhs = dense_mul(&updated, &expected);
        let solution = factor.solve(&rhs).expect("solve");
        for (actual, expected) in solution.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-7);
        }
        assert!(residual_inf_norm(&updated, &solution, &rhs) <= 1e-7);
    }
}

#[test]
fn multifrontal_factorization_is_deterministic_on_saddle_kkt() {
    let dense = saddle_kkt_dense(0.0);
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(8, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let expected = vec![0.75, -0.25, 1.0, -1.5, 0.5, 0.25, -0.5, 1.25];
    let rhs = dense_mul(&dense, &expected);

    let mut baseline_solution: Option<Vec<f64>> = None;
    let mut baseline_stats = None;
    for _ in 0..3 {
        let (mut factor, info) =
            factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");
        assert!(info.factorization_residual_max_abs <= 1e-10);
        let solution = factor.solve(&rhs).expect("solve");
        assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-9);

        let stats = (
            factor.inertia(),
            factor.pivot_stats(),
            factor.stored_nnz(),
            factor.factor_bytes(),
        );
        if let Some(reference) = &baseline_solution {
            assert_eq!(baseline_stats, Some(stats));
            for (actual, expected) in solution.iter().zip(reference.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
        } else {
            baseline_stats = Some(stats);
            baseline_solution = Some(solution);
        }
    }
}

#[test]
fn numeric_refactorization_rejects_pattern_change() {
    let original = vec![
        vec![4.0, -1.0, 0.0, 0.0],
        vec![-1.0, 4.0, -1.0, 0.0],
        vec![0.0, -1.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 3.0],
    ];
    let changed_pattern = vec![
        vec![4.0, -1.0, 0.25, 0.0],
        vec![-1.0, 4.0, -1.0, 0.0],
        vec![0.25, -1.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 3.0],
    ];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&original);
    let matrix = SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let (mut factor, _) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    let (updated_col_ptrs, updated_row_indices, updated_values) =
        dense_to_lower_csc(&changed_pattern);
    let updated_matrix = SymmetricCscMatrix::new(
        4,
        &updated_col_ptrs,
        &updated_row_indices,
        Some(&updated_values),
    )
    .expect("updated csc");
    let error = factor
        .refactorize(updated_matrix)
        .expect_err("refactorization should reject a changed sparsity pattern");
    assert!(matches!(error, SsidsError::PatternMismatch(_)));
}

#[test]
fn numeric_factorization_requires_values() {
    let col_ptrs = vec![0, 2, 3];
    let row_indices = vec![0, 1, 1];
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, None).expect("pattern");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let error = factorize(matrix, &symbolic, &NumericFactorOptions::default())
        .expect_err("factorization should reject missing values");
    assert!(matches!(error, ssids_rs::SsidsError::MissingValues));
}

#[test]
fn numeric_factorization_rejects_nonfinite_values() {
    let col_ptrs = vec![0, 2, 3];
    let row_indices = vec![0, 1, 1];
    let values = vec![2.0, f64::NAN, 3.0];
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let error = factorize(matrix, &symbolic, &NumericFactorOptions::default())
        .expect_err("factorization should reject non-finite values");
    assert!(matches!(error, SsidsError::InvalidMatrix(message) if message.contains("not finite")));
}
