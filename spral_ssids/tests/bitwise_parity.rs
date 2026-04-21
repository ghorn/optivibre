use spral_ssids::{
    NativeOrdering, NativeSpral, NumericFactorOptions, OrderingStrategy, SsidsOptions,
    SymmetricCscMatrix, analyse, factorize,
};

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

fn dense_to_lower_csc(matrix: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let dimension = matrix.len();
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..dimension {
        for (row, dense_row) in matrix.iter().enumerate().skip(col) {
            let value = dense_row[col];
            if row == col || value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices, values)
}

fn deterministic_complete_dyadic_matrix(dimension: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    let mut row = 0;
    while row < dimension {
        let mut col = 0;
        while col <= row {
            let value = if row == col {
                f64::from((row % 7) as i16 - 3) / 64.0
            } else {
                let lower_triangle_index = row * (row + 1) / 2 + col;
                let magnitude = f64::from(lower_triangle_index as u16 + 1) / 512.0;
                let sign = if (row * 13 + col * 19 + 5) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                sign * magnitude
            };
            matrix[row][col] = value;
            matrix[col][row] = value;
            col += 1;
        }
        row += 1;
    }
    matrix
}

fn load_native_or_skip() -> Option<NativeSpral> {
    match NativeSpral::load() {
        Ok(native) => Some(native),
        Err(error) => {
            eprintln!("skipping native SPRAL bitwise test: {error}");
            None
        }
    }
}

#[test]
fn rust_and_native_spral_match_app_block_boundary_33x33_pivot_stats() {
    let Some(native) = load_native_or_skip() else {
        return;
    };

    let matrix_dense = deterministic_complete_dyadic_matrix(33);
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&matrix_dense);
    let matrix = SymmetricCscMatrix::new(33, &col_ptrs, &row_indices, Some(&values))
        .expect("valid boundary CSC");
    let options = NumericFactorOptions::default();

    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("rust analyse");
    let (rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");

    let mut native_session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
        .expect("native analyse");
    let native_info = native_session.factorize(matrix).expect("native factorize");

    assert_eq!(rust_factor.inertia(), native_info.inertia);
    assert_eq!(
        rust_factor.pivot_stats().two_by_two_pivots,
        native_info.two_by_two_pivots
    );
    assert_eq!(
        rust_factor.pivot_stats().delayed_pivots,
        native_info.delayed_pivots
    );
    assert_eq!(rust_factor.pivot_stats().two_by_two_pivots, 2);
}

#[test]
fn rust_and_native_spral_match_app_block_boundary_33x33_solution_bits() {
    let matrix = deterministic_complete_dyadic_matrix(33);
    let expected_solution = (0..33)
        .map(|index| f64::from((index % 11) as i16 - 5) / 8.0)
        .collect::<Vec<_>>();
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
fn rust_and_native_spral_match_app_block_boundary_34x34_solution_bits() {
    let matrix = deterministic_complete_dyadic_matrix(34);
    let expected_solution = (0..34)
        .map(|index| f64::from((index % 11) as i16 - 5) / 8.0)
        .collect::<Vec<_>>();
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
fn rust_and_native_spral_match_app_dense_zero_reduction_64x64_solution_bits() {
    let matrix = deterministic_complete_dyadic_matrix(64);
    let expected_solution = (0..64)
        .map(|index| f64::from((index % 11) as i16 - 5) / 8.0)
        .collect::<Vec<_>>();
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

fn bit_patterns(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn format_scalar_bits(value: f64) -> String {
    format!("{value:?} (0x{:016x})", value.to_bits())
}

fn format_vector_bits(values: &[f64]) -> Vec<String> {
    values
        .iter()
        .map(|&value| format_scalar_bits(value))
        .collect()
}

fn format_matrix_bits(matrix: &[Vec<f64>]) -> Vec<Vec<String>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&value| format_scalar_bits(value)).collect())
        .collect()
}

fn assert_exact_bitwise_parity_witness(matrix_dense: &[Vec<f64>], expected_solution: &[f64]) {
    let Some(native) = load_native_or_skip() else {
        return;
    };

    let (col_ptrs, row_indices, values) = dense_to_lower_csc(matrix_dense);
    let matrix =
        SymmetricCscMatrix::new(matrix_dense.len(), &col_ptrs, &row_indices, Some(&values))
            .expect("valid witness CSC");
    let options = NumericFactorOptions::default();

    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("rust analyse");

    let rust_factorization = factorize(matrix, &symbolic, &options);
    let mut native_session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
        .expect("native analyse");
    let native_factorization = native_session.factorize(matrix);

    let rhs = dense_mul(matrix_dense, expected_solution);

    let mismatch_context =
        |rust_factorization_error: Option<String>,
         native_factorization_error: Option<String>,
         rust_solution: Option<Result<Vec<f64>, String>>,
         native_solution: Option<Result<Vec<f64>, String>>| {
            format!(
                concat!(
                    "dense_matrix={:?}\n",
                    "col_ptrs={:?}\n",
                    "row_indices={:?}\n",
                    "values={:?}\n",
                    "expected_solution={:?}\n",
                    "expected_solution_bits={:?}\n",
                    "rhs={:?}\n",
                    "rhs_bits={:?}\n",
                    "rust_factorization_error={:?}\n",
                    "native_factorization_error={:?}\n",
                    "rust_solution={:?}\n",
                    "native_solution={:?}\n",
                ),
                format_matrix_bits(matrix_dense),
                col_ptrs,
                row_indices,
                format_vector_bits(&values),
                expected_solution,
                bit_patterns(expected_solution),
                rhs,
                bit_patterns(&rhs),
                rust_factorization_error,
                native_factorization_error,
                rust_solution.as_ref().map(|result| {
                    result
                        .as_ref()
                        .map(|values| (values.clone(), bit_patterns(values)))
                }),
                native_solution.as_ref().map(|result| {
                    result
                        .as_ref()
                        .map(|values| (values.clone(), bit_patterns(values)))
                }),
            )
        };

    match (&rust_factorization, &native_factorization) {
        (Ok(_), Err(error)) => panic!(
            "rust factorized but native failed\n{}",
            mismatch_context(None, Some(error.to_string()), None, None)
        ),
        (Err(error), Ok(_)) => panic!(
            "native factorized but rust failed\n{}",
            mismatch_context(Some(error.to_string()), None, None, None)
        ),
        (Err(_), Err(_)) => return,
        (Ok((rust_factor, _)), Ok(native_info)) => {
            assert_eq!(
                rust_factor.inertia(),
                native_info.inertia,
                "inertia mismatch\n{}",
                mismatch_context(None, None, None, None)
            );
            assert_eq!(
                rust_factor.pivot_stats().two_by_two_pivots,
                native_info.two_by_two_pivots,
                "two_by_two_pivots mismatch\n{}",
                mismatch_context(None, None, None, None)
            );
            assert_eq!(
                rust_factor.pivot_stats().delayed_pivots,
                native_info.delayed_pivots,
                "delayed_pivots mismatch\n{}",
                mismatch_context(None, None, None, None)
            );
        }
    }

    let (mut rust_factor, _) = rust_factorization.expect("checked above");
    let rust_solution = rust_factor.solve(&rhs).map_err(|error| error.to_string());
    let native_solution = native_session
        .solve(&rhs)
        .map_err(|error| error.to_string());

    match (&rust_solution, &native_solution) {
        (Ok(rust_solution), Ok(native_solution)) => assert_eq!(
            bit_patterns(rust_solution),
            bit_patterns(native_solution),
            "solve bitwise mismatch\n{}",
            mismatch_context(
                None,
                None,
                Some(Ok(rust_solution.clone())),
                Some(Ok(native_solution.clone()))
            )
        ),
        (Ok(_), Err(_)) | (Err(_), Ok(_)) => panic!(
            "solve success mismatch\n{}",
            mismatch_context(None, None, Some(rust_solution), Some(native_solution))
        ),
        (Err(_), Err(_)) => {}
    }
}

fn assert_tiny_bitwise_solution_parity(
    matrix_dense: &[Vec<f64>],
    expected_solution: &[f64],
    expected_two_by_two_pivots: usize,
) {
    let Some(native) = load_native_or_skip() else {
        return;
    };

    let (col_ptrs, row_indices, values) = dense_to_lower_csc(matrix_dense);
    let matrix =
        SymmetricCscMatrix::new(matrix_dense.len(), &col_ptrs, &row_indices, Some(&values))
            .expect("valid tiny CSC");
    let options = NumericFactorOptions::default();

    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("rust analyse");
    let (mut rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");

    let mut native_session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
        .expect("native analyse");
    let native_info = native_session.factorize(matrix).expect("native factorize");

    assert_eq!(rust_factor.inertia(), native_info.inertia);
    assert_eq!(
        rust_factor.pivot_stats().two_by_two_pivots,
        native_info.two_by_two_pivots
    );
    assert_eq!(
        rust_factor.pivot_stats().delayed_pivots,
        native_info.delayed_pivots
    );
    assert_eq!(
        rust_factor.pivot_stats().two_by_two_pivots,
        expected_two_by_two_pivots
    );

    let rhs = dense_mul(matrix_dense, expected_solution);
    let rust_solution = rust_factor.solve(&rhs).expect("rust solve");
    let native_solution = native_session.solve(&rhs).expect("native solve");

    assert_eq!(
        bit_patterns(&rust_solution),
        bit_patterns(&native_solution),
        "expected bitwise parity on tiny system, rust={:?}, native={:?}",
        rust_solution,
        native_solution
    );
}

#[test]
fn rust_and_native_spral_match_bitwise_on_tiny_spd_system() {
    let matrix = vec![vec![2.0, 0.0], vec![0.0, 1.0]];
    let expected_solution = vec![0.5, -0.25];
    assert_tiny_bitwise_solution_parity(&matrix, &expected_solution, 0);
}

#[test]
fn rust_and_native_spral_match_bitwise_on_tiny_two_by_two_pivot_system() {
    let matrix = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let expected_solution = vec![0.5, -0.75];
    assert_tiny_bitwise_solution_parity(&matrix, &expected_solution, 1);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_singular_two_by_two_plus_zero_3x3() {
    let matrix = vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![-1.0, -1.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_singular_path_signed_zero_source_3x3() {
    let matrix = vec![
        vec![0.0, -1.0, 0.0],
        vec![-1.0, 0.0, -1.0],
        vec![0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![-1.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_singular_two_by_two_signed_zero_source_3x3() {
    let matrix = vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, -1.0],
        vec![0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![-3.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_3x3() {
    let matrix = vec![
        vec![0.0, -1.0, 1.5],
        vec![-1.0, 0.0, -1.0],
        vec![1.5, -1.0, 2.0],
    ];
    let expected_solution = vec![0.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_arrow_4x4() {
    let matrix = vec![
        vec![1.0, -3.0, 1.0, 1.0],
        vec![-3.0, -1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_arrow_two_pivot_4x4() {
    let matrix = vec![
        vec![0.0, -3.0, 0.25, 1.0],
        vec![-3.0, 0.0, 0.0, 0.0],
        vec![0.25, 0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ];
    let expected_solution = vec![0.0, -1.0, 1.0, -2.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_singular_arrow_4x4() {
    let matrix = vec![
        vec![1.0, -1.0, 0.25, 2.0],
        vec![-1.0, -2.0, 0.0, 0.0],
        vec![0.25, 0.0, 0.0, 0.0],
        vec![2.0, 0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![-0.75, -1.0, 1.0, -2.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_singular_arrow_solve_order_4x4() {
    let matrix = vec![
        vec![1.0, -1.0, 0.25, 1.0],
        vec![-1.0, -2.0, 0.0, 0.0],
        vec![0.25, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![0.0, -0.25, 2.0, -2.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_signed_zero_solve_4x4() {
    let matrix = vec![
        vec![2.0, -1.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, -1.0],
        vec![0.0, 0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![-1.0, -1.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_4x4() {
    let matrix = vec![
        vec![-4.0, 4.0, 1.0, 0.0],
        vec![4.0, -1.5, -6.0, 0.0],
        vec![1.0, -6.0, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_dense_4x4() {
    let matrix = vec![
        vec![-0.5, 0.125, 0.0, -0.1875],
        vec![0.125, -0.75, 3.0, 0.0],
        vec![0.0, 3.0, 0.75, -5.0],
        vec![-0.1875, 0.0, -5.0, 0.5],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_scaled_4x4() {
    let matrix = vec![
        vec![-0.5, 0.125, 0.0, -0.1875],
        vec![0.125, -0.75, 2.0, 0.0],
        vec![0.0, 2.0, 0.75, -3.0],
        vec![-0.1875, 0.0, -3.0, 0.5],
    ];
    let expected_solution = vec![0.0, 0.0, -6.0, -0.75];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_update_4x4() {
    let matrix = vec![
        vec![0.25, -0.375, -3.0, 0.0],
        vec![-0.375, -1.0, -1.0, 0.0],
        vec![-3.0, -1.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0, -5.0],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_cancellation_4x4() {
    let matrix = vec![
        vec![0.25, -0.375, -3.0, 0.0],
        vec![-0.375, -1.0, -1.0, 0.0],
        vec![-3.0, -1.0, -1.0, 0.5],
        vec![0.0, 0.0, 0.5, -1.0],
    ];
    let expected_solution = vec![0.25, 2.0, 0.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_cancellation_plain_4x4() {
    let matrix = vec![
        vec![0.5, 0.5, -0.75, 0.0],
        vec![0.5, -1.0, -1.0, 0.0],
        vec![-0.75, -1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let expected_solution = vec![3.0, 1.25, 4.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_fma_4x4() {
    let matrix = vec![
        vec![0.25, -0.75, 5.0, 0.0],
        vec![-0.75, -2.0, 7.0, 0.0],
        vec![5.0, 7.0, 0.0, -1.0],
        vec![0.0, 0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![-0.125, -3.0, -3.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_opposite_product_fma_4x4() {
    let matrix = vec![
        vec![0.25, -0.75, 5.0, 0.0],
        vec![-0.75, -2.0, 5.0, 0.0],
        vec![5.0, 5.0, -1.0, -3.0],
        vec![0.0, 0.0, -3.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, -1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_plain_4x4() {
    let matrix = vec![
        vec![-4.0, 4.0, 1.0, 0.0],
        vec![4.0, -1.5, -0.75, 0.0],
        vec![1.0, -0.75, 4.0, -1.0],
        vec![0.0, 0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_same_product_plain_4x4() {
    let matrix = vec![
        vec![-0.5, 0.0625, -3.0, 0.0],
        vec![0.0625, -3.0, -2.0, 0.0],
        vec![-3.0, -2.0, -0.25, -1.0],
        vec![0.0, 0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, -1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_same_product_update_plain_4x4() {
    let matrix = vec![
        vec![0.5, 0.5, -0.75, 0.0],
        vec![0.5, -1.0, -0.5, 0.0],
        vec![-0.75, -0.5, 1.0, 0.25],
        vec![0.0, 0.0, 0.25, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 4.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_same_product_fma_4x4() {
    let matrix = vec![
        vec![-3.0, 1.75, 2.0, 0.0],
        vec![1.75, 0.0, -0.75, 0.0],
        vec![2.0, -0.75, -3.0, 0.5],
        vec![0.0, 0.0, 0.5, -0.25],
    ];
    let expected_solution = vec![0.0, -1.0, 0.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_dominant_plain_solve_4x4() {
    let matrix = vec![
        vec![-3.0, 1.75, 1.0, 0.0],
        vec![1.75, 0.0, -0.75, 0.0],
        vec![1.0, -0.75, -3.0, 1.0],
        vec![0.0, 0.0, 1.0, -0.25],
    ];
    let expected_solution = vec![0.0, -1.0, 0.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_wide_trailing_plain_5x5() {
    let matrix = vec![
        vec![0.125, -5.0, 3.0, 0.0, 0.0],
        vec![-5.0, 3.5, -1.5, 0.0, 0.0],
        vec![3.0, -1.5, 0.375, 4.0, 0.0],
        vec![0.0, 0.0, 4.0, -0.5, -2.0],
        vec![0.0, 0.0, 0.0, -2.0, 0.25],
    ];
    let expected_solution = vec![0.0, 0.0, -1.0, -0.625, -5.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_second_dot_order_5x5() {
    let matrix = vec![
        vec![-0.75, -0.75, 0.0, 0.0, 0.0],
        vec![-0.75, -1.25, -2.0, -1.0, 0.0],
        vec![0.0, -2.0, 0.0625, 0.375, 0.0],
        vec![0.0, -1.0, 0.375, 0.0, -1.0],
        vec![0.0, 0.0, 0.0, -1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, 1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_wide_second_dot_order_5x5() {
    let matrix = vec![
        vec![0.75, 0.375, -1.0, 0.0, 0.0],
        vec![0.375, -2.0, 1.0, 0.0, 0.0],
        vec![-1.0, 1.0, -1.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, -0.75, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 2.0, 0.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_same_sign_beta_fma_4x4() {
    let matrix = vec![
        vec![3.0, 1.25, 0.25, 0.0],
        vec![1.25, -0.25, -0.75, 0.0],
        vec![0.25, -0.75, 0.5, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_opposite_update_plain_4x4() {
    let matrix = vec![
        vec![0.25, 0.4375, 0.3125, 0.0],
        vec![0.4375, -0.375, -0.75, 0.0],
        vec![0.3125, -0.75, -1.25, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 4.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_path_plus_chord_tpp_dominant_current_plain_4x4() {
    let matrix = vec![
        vec![0.25, -0.5, -0.0625, 0.0],
        vec![-0.5, -4.0, -1.0, 0.0],
        vec![-0.0625, -1.0, -3.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 1.0, 0.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_arrow_tpp_inertia_cancellation_5x5() {
    let matrix = vec![
        vec![-0.375, 0.125, -7.0, 3.0, 0.125],
        vec![0.125, -0.75, 0.0, 0.0, 0.0],
        vec![-7.0, 0.0, -0.25, 0.0, 0.0],
        vec![3.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.125, 0.0, 0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_arrow_5x5() {
    let matrix = vec![
        vec![0.0, 1.0, -1.0, -5.0, -1.0],
        vec![1.0, -1.0, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0, 0.0, 0.0],
        vec![-5.0, 0.0, 0.0, -3.0, 0.0],
        vec![-1.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, 0.0, 1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}

#[test]
#[ignore = "manual exact parity witness while debugging the native-vs-rust mismatch"]
fn rust_and_native_spral_minimized_arrow_triangular_solve_5x5() {
    let matrix = vec![
        vec![-0.25, 2.0, -1.0, -2.0, 6.0],
        vec![2.0, 1.0, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, -1.0, 0.0, 0.0],
        vec![-2.0, 0.0, 0.0, 0.5, 0.0],
        vec![6.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let expected_solution = vec![0.0, 0.0, 0.0, 0.0, -1.0];
    assert_exact_bitwise_parity_witness(&matrix, &expected_solution);
}
