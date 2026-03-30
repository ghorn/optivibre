use sx_codegen::lower_function;
use sx_codegen_llvm::{
    CompiledJitFunction, JitOptimizationLevel, LlvmTarget, emit_object_bytes_lowered,
};
use sx_core::{NamedMatrix, SX, SXFunction, SXMatrix};

#[derive(Clone, Copy, Debug)]
enum MatrixBinaryOp {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    Hypot,
    MtimesRhsTranspose,
}

fn dense_matrix_binary_adaptor(lhs: &SXMatrix, rhs: &SXMatrix, op: MatrixBinaryOp) -> SXMatrix {
    match op {
        MatrixBinaryOp::MtimesRhsTranspose => {
            let (lhs_rows, lhs_cols) = lhs.shape();
            let (rhs_rows, rhs_cols) = rhs.shape();
            assert_eq!(lhs_cols, rhs_cols);
            let mut values = Vec::with_capacity(lhs_rows * rhs_rows);
            for col in 0..rhs_rows {
                for row in 0..lhs_rows {
                    let mut sum = SX::zero();
                    for k in 0..lhs_cols {
                        sum += lhs.get(row, k) * rhs.get(col, k);
                    }
                    values.push(sum);
                }
            }
            SXMatrix::dense(lhs_rows, rhs_rows, values).unwrap()
        }
        _ => {
            let (nrow, ncol) = lhs.shape();
            assert_eq!(rhs.shape(), (nrow, ncol));
            let mut values = Vec::with_capacity(nrow * ncol);
            for col in 0..ncol {
                for row in 0..nrow {
                    let lhs_value = lhs.get(row, col);
                    let rhs_value = rhs.get(row, col);
                    values.push(match op {
                        MatrixBinaryOp::Add => lhs_value + rhs_value,
                        MatrixBinaryOp::Sub => lhs_value - rhs_value,
                        MatrixBinaryOp::Mul => lhs_value * rhs_value,
                        MatrixBinaryOp::Max => lhs_value.max(rhs_value),
                        MatrixBinaryOp::Min => lhs_value.min(rhs_value),
                        MatrixBinaryOp::Hypot => lhs_value.hypot(rhs_value),
                        MatrixBinaryOp::MtimesRhsTranspose => unreachable!(),
                    });
                }
            }
            SXMatrix::dense(nrow, ncol, values).unwrap()
        }
    }
}

fn numeric_dense_matrix_binary(
    lhs: &[Vec<f64>],
    rhs: &[Vec<f64>],
    op: MatrixBinaryOp,
) -> Vec<Vec<f64>> {
    match op {
        MatrixBinaryOp::MtimesRhsTranspose => {
            let lhs_rows = lhs.len();
            let rhs_rows = rhs.len();
            let cols = lhs[0].len();
            let mut out = vec![vec![0.0; rhs_rows]; lhs_rows];
            for row in 0..lhs_rows {
                for col in 0..rhs_rows {
                    out[row][col] = (0..cols).map(|k| lhs[row][k] * rhs[col][k]).sum();
                }
            }
            out
        }
        _ => lhs
            .iter()
            .zip(rhs)
            .map(|(lhs_row, rhs_row)| {
                lhs_row
                    .iter()
                    .zip(rhs_row)
                    .map(|(&lhs_value, &rhs_value)| match op {
                        MatrixBinaryOp::Add => lhs_value + rhs_value,
                        MatrixBinaryOp::Sub => lhs_value - rhs_value,
                        MatrixBinaryOp::Mul => lhs_value * rhs_value,
                        MatrixBinaryOp::Max => lhs_value.max(rhs_value),
                        MatrixBinaryOp::Min => lhs_value.min(rhs_value),
                        MatrixBinaryOp::Hypot => lhs_value.hypot(rhs_value),
                        MatrixBinaryOp::MtimesRhsTranspose => unreachable!(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    }
}

fn dense_matrix_column_major(values: &[Vec<f64>]) -> Vec<f64> {
    let nrow = values.len();
    let ncol = values.first().map_or(0, Vec::len);
    let mut flattened = Vec::with_capacity(nrow * ncol);
    for col in 0..ncol {
        for row_values in values.iter().take(nrow) {
            flattened.push(row_values[col]);
        }
    }
    flattened
}

#[test]
fn llvm_codegen_matches_casadi_test_sxbinary_codegen_via_backend_adaptor() {
    let lhs = SXMatrix::sym_dense("x", 4, 2).unwrap();
    let rhs = SXMatrix::sym_dense("y", 4, 2).unwrap();
    let lhs_values = vec![
        vec![0.738, 0.2],
        vec![0.1, 0.39],
        vec![0.99, 0.999_999],
        vec![1.0, 2.0],
    ];
    let rhs_values = vec![
        vec![1.738, 0.6],
        vec![0.7, 12.0],
        vec![0.0, -6.0],
        vec![1.0, 2.0],
    ];

    for op in [
        MatrixBinaryOp::Add,
        MatrixBinaryOp::Sub,
        MatrixBinaryOp::Mul,
        MatrixBinaryOp::Max,
        MatrixBinaryOp::Min,
        MatrixBinaryOp::Hypot,
        MatrixBinaryOp::MtimesRhsTranspose,
    ] {
        let output = dense_matrix_binary_adaptor(&lhs, &rhs, op);
        let function = SXFunction::new(
            "sxbinary_codegen_parity",
            vec![
                NamedMatrix::new("x", lhs.clone()).unwrap(),
                NamedMatrix::new("y", rhs.clone()).unwrap(),
            ],
            vec![NamedMatrix::new("out", output.clone()).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let object =
            emit_object_bytes_lowered(&lowered, JitOptimizationLevel::O2, &LlvmTarget::Native)
                .unwrap();
        assert!(!object.is_empty());

        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O2).unwrap();
        let mut context = compiled.create_context();
        context
            .input_mut(0)
            .copy_from_slice(&dense_matrix_column_major(&lhs_values));
        context
            .input_mut(1)
            .copy_from_slice(&dense_matrix_column_major(&rhs_values));
        compiled.eval(&mut context);

        let expected =
            dense_matrix_column_major(&numeric_dense_matrix_binary(&lhs_values, &rhs_values, op));
        assert_eq!(context.output(0).len(), expected.len());
        for (idx, (lhs_value, rhs_value)) in context.output(0).iter().zip(expected).enumerate() {
            assert!(
                (lhs_value - rhs_value).abs() <= 1e-12,
                "op {op:?} output[{idx}] mismatch: {lhs_value} vs {rhs_value}"
            );
        }
    }
}
