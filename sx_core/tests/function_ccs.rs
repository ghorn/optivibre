use rstest::rstest;
use sx_core::{CCS, NamedMatrix, SX, SXFunction, SXMatrix};

#[rstest]
fn named_matrix_rejects_blank_name() {
    let err = NamedMatrix::new("   ", SXMatrix::scalar(1.0)).expect_err("blank names must fail");
    assert!(err.to_string().contains("cannot be empty"));
}

#[rstest]
fn sx_function_rejects_blank_function_name() {
    let x = SXMatrix::sym_dense("x", 1, 1).expect("symbolic input");
    let err = SXFunction::new(
        " ",
        vec![NamedMatrix::new("x", x.clone()).expect("input slot")],
        vec![NamedMatrix::new("out", x).expect("output slot")],
    )
    .expect_err("blank function names must fail");
    assert!(err.to_string().contains("function name cannot be empty"));
}

#[rstest]
fn sx_function_rejects_duplicate_input_names() {
    let x0 = SX::sym("x0");
    let x1 = SX::sym("x1");
    let err = SXFunction::new(
        "dup_inputs",
        vec![
            NamedMatrix::new("x", SXMatrix::scalar(x0)).expect("first input"),
            NamedMatrix::new("x", SXMatrix::scalar(x1)).expect("second input"),
        ],
        vec![NamedMatrix::new("out", SXMatrix::scalar(x0 + x1)).expect("output")],
    )
    .expect_err("duplicate input names must fail");
    assert!(err.to_string().contains("duplicate input name"));
}

#[rstest]
fn sx_function_rejects_duplicate_output_names() {
    let x = SX::sym("x");
    let input = NamedMatrix::new("x", SXMatrix::scalar(x)).expect("input slot");
    let err = SXFunction::new(
        "dup_outputs",
        vec![input],
        vec![
            NamedMatrix::new("out", SXMatrix::scalar(x + 1.0)).expect("first output"),
            NamedMatrix::new("out", SXMatrix::scalar(x - 1.0)).expect("second output"),
        ],
    )
    .expect_err("duplicate output names must fail");
    assert!(err.to_string().contains("duplicate output name"));
}

#[rstest]
fn sx_function_rejects_nonsymbolic_input_entries() {
    let err = SXFunction::new(
        "bad_input",
        vec![NamedMatrix::new("x", SXMatrix::scalar(SX::sym("x") + 1.0)).expect("input slot")],
        vec![NamedMatrix::new("out", SXMatrix::scalar(1.0)).expect("output slot")],
    )
    .expect_err("non-symbolic inputs must fail");
    assert!(
        err.to_string()
            .contains("must contain only symbolic primitives")
    );
}

#[rstest]
fn sx_function_rejects_reused_symbol_across_input_slots() {
    let x = SX::sym("shared");
    let err = SXFunction::new(
        "reused_symbol",
        vec![
            NamedMatrix::new("x0", SXMatrix::scalar(x)).expect("first input"),
            NamedMatrix::new("x1", SXMatrix::scalar(x)).expect("second input"),
        ],
        vec![NamedMatrix::new("out", SXMatrix::scalar(x)).expect("output slot")],
    )
    .expect_err("reused symbols must fail");
    assert!(err.to_string().contains("appears in multiple input slots"));
}

#[rstest]
fn sx_function_input_bindings_follow_slot_and_offset_order() {
    let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic input");
    let y = SXMatrix::sym_dense("y", 1, 2).expect("symbolic input");
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let y0 = y.nz(0);
    let y1 = y.nz(1);
    let function = SXFunction::new(
        "bindings",
        vec![
            NamedMatrix::new("x", x).expect("x input"),
            NamedMatrix::new("y", y).expect("y input"),
        ],
        vec![NamedMatrix::new("out", SXMatrix::scalar(x0 + x1 + y0 + y1)).expect("output")],
    )
    .expect("valid function");

    let bindings = function.input_bindings();
    assert_eq!(bindings[&x0], (0, 0));
    assert_eq!(bindings[&x1], (0, 1));
    assert_eq!(bindings[&y0], (1, 0));
    assert_eq!(bindings[&y1], (1, 1));
}

#[rstest]
fn sx_function_free_symbols_collects_only_output_dependencies() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let function = SXFunction::new(
        "free_symbols",
        vec![
            NamedMatrix::new("x", SXMatrix::scalar(x)).expect("x input"),
            NamedMatrix::new("y", SXMatrix::scalar(y)).expect("y input"),
        ],
        vec![NamedMatrix::new("out", SXMatrix::scalar(x + y)).expect("output")],
    )
    .expect("valid function");

    let free = function.free_symbols();
    assert_eq!(free.len(), 2);
    assert!(free.contains(&x));
    assert!(free.contains(&y));
}

#[rstest]
fn ccs_new_rejects_bad_column_pointer_length() {
    let err = CCS::new(2, 2, vec![0, 1], vec![0]).expect_err("bad col_ptr length");
    assert!(err.to_string().contains("expected 3 column pointers"));
}

#[rstest]
fn ccs_new_rejects_nonzero_first_column_pointer() {
    let err = CCS::new(2, 1, vec![1, 1], vec![0]).expect_err("bad first col_ptr");
    assert!(err.to_string().contains("must start at zero"));
}

#[rstest]
fn ccs_new_rejects_final_column_pointer_not_equal_to_nnz() {
    let err = CCS::new(2, 1, vec![0, 0], vec![0]).expect_err("bad final col_ptr");
    assert!(err.to_string().contains("must equal nnz"));
}

#[rstest]
fn ccs_new_rejects_non_monotone_column_pointers() {
    let err = CCS::new(3, 3, vec![0, 2, 1, 2], vec![0, 1]).expect_err("non-monotone col_ptrs");
    assert!(err.to_string().contains("must be monotone"));
}

#[rstest]
fn ccs_new_rejects_out_of_bounds_rows() {
    let err = CCS::new(2, 1, vec![0, 1], vec![2]).expect_err("row out of bounds");
    assert!(err.to_string().contains("out of bounds"));
}

#[rstest]
fn ccs_new_rejects_unsorted_rows_within_column() {
    let err = CCS::new(3, 1, vec![0, 2], vec![2, 1]).expect_err("rows must be increasing");
    assert!(
        err.to_string()
            .contains("row indices must be strictly increasing")
    );
}

#[rstest]
fn ccs_from_positions_sorts_and_deduplicates() {
    let ccs = CCS::from_positions(3, 2, &[(2, 1), (0, 0), (1, 1), (1, 1)]).expect("CCS");
    assert_eq!(ccs.col_ptrs(), &[0, 1, 3]);
    assert_eq!(ccs.row_indices(), &[0, 1, 2]);
    assert_eq!(ccs.positions(), vec![(0, 0), (1, 1), (2, 1)]);
}

#[rstest]
fn ccs_row_adjacency_and_nz_index_match_positions() {
    let ccs = CCS::from_positions(3, 3, &[(0, 0), (2, 0), (1, 1), (1, 2)]).expect("CCS");
    assert_eq!(ccs.row_adjacency(), vec![vec![0], vec![1, 2], vec![0]]);
    assert_eq!(ccs.nz_index(0, 0), Some(0));
    assert_eq!(ccs.nz_index(2, 0), Some(1));
    assert_eq!(ccs.nz_index(1, 1), Some(2));
    assert_eq!(ccs.nz_index(1, 2), Some(3));
    assert_eq!(ccs.nz_index(2, 2), None);
    assert_eq!(ccs.nz_index(9, 0), None);
}

#[rstest]
fn lower_triangular_ccs_is_canonical() {
    let ccs = CCS::lower_triangular(4);
    assert_eq!(ccs.col_ptrs(), &[0, 4, 7, 9, 10]);
    assert_eq!(
        ccs.positions(),
        vec![
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (2, 2),
            (3, 2),
            (3, 3),
        ]
    );
}

#[rstest]
fn jacobian_construction_does_not_mutate_original_output_shape_regression() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let input_ccs = CCS::from_positions(5, 1, &[(0, 0), (3, 0)]).expect("input CCS");
    let input = SXMatrix::new(input_ccs, vec![x, y]).expect("sparse input");
    let output = SXMatrix::dense_column(vec![x + y, x, y]).expect("dense output");
    let original_shape = output.shape();

    let jacobian = output.jacobian(&input).expect("jacobian");
    assert_eq!(output.shape(), original_shape);
    assert_eq!(jacobian.shape(), (3, 2));

    let function = SXFunction::new(
        "shape_regression",
        vec![NamedMatrix::new("inp", input).expect("input slot")],
        vec![NamedMatrix::new("out", output.clone()).expect("output slot")],
    )
    .expect("function");
    assert_eq!(function.outputs()[0].matrix().shape(), original_shape);
}
