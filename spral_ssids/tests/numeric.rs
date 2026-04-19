use approx::assert_abs_diff_eq;
use spral_ssids::{
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
    let (factor, info) =
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
    let (factor, info) =
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
    let (factor, info) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-10);
    assert!(factor.pivot_stats().two_by_two_pivots >= 1);
    assert_eq!(factor.pivot_stats().regularized_pivots, 0);

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
    let (factor, info) =
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
fn numeric_factorization_defers_unstable_panel_to_dense_block() {
    let dense = vec![
        vec![1e-4, 1e-4, 1.0],
        vec![1e-4, 1e-4, 0.5],
        vec![1.0, 0.5, 0.0],
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
    let options = NumericFactorOptions {
        pivot_regularization: 1e-4,
        ..NumericFactorOptions::default()
    };
    let (factor, info) = factorize(matrix, &symbolic, &options).expect("factor");

    assert!(info.factorization_residual_max_abs <= 1e-4);
    assert!(factor.pivot_stats().delayed_pivots >= 3);

    let expected = vec![0.25, -0.5, 1.0];
    let rhs = dense_mul(&dense, &expected);
    let solution = factor.solve(&rhs).expect("solve");
    for (actual, expected) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-8);
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
fn numeric_factorization_requires_values() {
    let col_ptrs = vec![0, 2, 3];
    let row_indices = vec![0, 1, 1];
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, None).expect("pattern");
    let (symbolic, _) = analyse(matrix, &SsidsOptions::default()).expect("symbolic");
    let error = factorize(matrix, &symbolic, &NumericFactorOptions::default())
        .expect_err("factorization should reject missing values");
    assert!(matches!(error, spral_ssids::SsidsError::MissingValues));
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
