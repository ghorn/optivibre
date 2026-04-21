use approx::assert_abs_diff_eq;
use spral_ssids::{Inertia, NativeOrdering, NativeSpral, NumericFactorOptions, SymmetricCscMatrix};

fn dense_mul(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(value, x_i)| value * x_i)
                .sum::<f64>()
        })
        .collect()
}

fn residual_inf_norm(matrix: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
    dense_mul(matrix, x)
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs_i)| (lhs - rhs_i).abs())
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

fn load_native_or_skip() -> Option<NativeSpral> {
    match NativeSpral::load() {
        Ok(native) => Some(native),
        Err(error) => {
            eprintln!("skipping native SPRAL test: {error}");
            None
        }
    }
}

#[test]
fn native_spral_session_solves_spd_system() {
    let Some(native) = load_native_or_skip() else {
        return;
    };
    let dense = vec![
        vec![4.0, -1.0, 0.0, 0.0],
        vec![-1.0, 4.0, -1.0, 0.0],
        vec![0.0, -1.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 3.0],
    ];
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
    let matrix = SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let mut session = native.analyse(matrix).expect("analyse");
    let analyse_info = session.analyse_info();
    assert!(analyse_info.supernode_count > 0);
    assert!(analyse_info.max_front_size > 0);

    let factor_info = session.factorize(matrix).expect("factorize");
    assert_eq!(
        factor_info.inertia,
        Inertia {
            positive: 4,
            negative: 0,
            zero: 0,
        }
    );
    assert!(factor_info.supernode_count > 0);

    let expected = vec![1.0, -2.0, 0.5, 3.0];
    let rhs = dense_mul(&dense, &expected);
    let solution = session.solve(&rhs).expect("solve");
    for (actual, expected_i) in solution.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected_i, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-10);
}

#[test]
fn native_spral_session_refactorizes_same_pattern() {
    let Some(native) = load_native_or_skip() else {
        return;
    };
    let dense0 = vec![
        vec![6.0, -1.0, 0.0, 0.0],
        vec![-1.0, 5.0, -1.5, 0.0],
        vec![0.0, -1.5, 4.5, -1.0],
        vec![0.0, 0.0, -1.0, 3.5],
    ];
    let dense1 = vec![
        vec![6.5, -1.0, 0.0, 0.0],
        vec![-1.0, 5.25, -1.5, 0.0],
        vec![0.0, -1.5, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 3.75],
    ];
    let (col_ptrs, row_indices, values0) = dense_to_lower_csc(&dense0);
    let (_, _, values1) = dense_to_lower_csc(&dense1);
    let matrix0 =
        SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values0)).expect("csc0");
    let matrix1 =
        SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values1)).expect("csc1");
    let mut session = native.analyse(matrix0).expect("analyse");

    session.factorize(matrix0).expect("factorize");
    let factor_info = session.refactorize(matrix1).expect("refactorize");
    assert_eq!(
        factor_info.inertia,
        Inertia {
            positive: 4,
            negative: 0,
            zero: 0,
        }
    );
    assert_eq!(session.factor_info(), Some(factor_info));

    let expected = vec![0.25, -1.0, 0.5, 2.0];
    let rhs = dense_mul(&dense1, &expected);
    let mut in_place = rhs.clone();
    session
        .solve_in_place(&mut in_place)
        .expect("solve in place");
    for (actual, expected_i) in in_place.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected_i, epsilon = 1e-10);
    }
    assert!(residual_inf_norm(&dense1, &in_place, &rhs) <= 1e-10);
}

#[test]
fn native_spral_enquires_indefinite_pivots() {
    let Some(native) = load_native_or_skip() else {
        return;
    };
    let col_ptrs = [0, 2, 3];
    let row_indices = [0, 1, 1];
    let values = [0.0, 1.0, 0.0];
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, Some(&values)).expect("csc");
    let options = NumericFactorOptions::default();
    let mut session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
        .expect("analyse");

    let factor_info = session.factorize(matrix).expect("factorize");
    assert_eq!(
        factor_info.inertia,
        Inertia {
            positive: 1,
            negative: 1,
            zero: 0,
        }
    );
    assert_eq!(factor_info.two_by_two_pivots, 1);

    let enquiry = session.enquire_indef().expect("enquire indefinite factor");
    assert_eq!(enquiry.pivot_order, vec![0, 1]);
    assert_eq!(
        enquiry.inverse_diagonal_entries,
        vec![[0.0, 1.0], [0.0, 0.0]]
    );
}
