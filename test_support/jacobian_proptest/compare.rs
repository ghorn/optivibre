use std::fmt;

use sx_codegen_llvm::CompiledJitFunction;
use sx_core::CCS;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.values[row * self.cols + col] = value;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixMismatchEntry {
    pub row: usize,
    pub col: usize,
    pub lhs: f64,
    pub rhs: f64,
    pub abs_error: f64,
    pub rel_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JacobianMismatchSummary {
    pub rows: usize,
    pub cols: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub worst_entry: Option<MatrixMismatchEntry>,
    pub missing_nonzeros: usize,
    pub extra_nonzeros: usize,
}

impl JacobianMismatchSummary {
    pub fn within_tolerances(&self, abs_tol: f64, rel_tol: f64) -> bool {
        self.max_abs_error <= abs_tol || self.max_rel_error <= rel_tol
    }
}

impl fmt::Display for JacobianMismatchSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "rows={} cols={} max_abs_error={:.3e} max_rel_error={:.3e} missing_nonzeros={} extra_nonzeros={}",
            self.rows,
            self.cols,
            self.max_abs_error,
            self.max_rel_error,
            self.missing_nonzeros,
            self.extra_nonzeros
        )?;
        if let Some(entry) = self.worst_entry {
            writeln!(
                f,
                "worst_entry row={} col={} lhs={:.6e} rhs={:.6e} abs={:.3e} rel={:.3e}",
                entry.row, entry.col, entry.lhs, entry.rhs, entry.abs_error, entry.rel_error
            )?;
        }
        Ok(())
    }
}

pub fn dense_from_sx_ccs_values(ccs: &CCS, values: &[f64]) -> DenseMatrix {
    assert_eq!(ccs.nnz(), values.len());
    let mut dense = DenseMatrix::zeros(ccs.nrow(), ccs.ncol());
    for col in 0..ccs.ncol() {
        for (index, value) in values
            .iter()
            .enumerate()
            .take(ccs.col_ptrs()[col + 1])
            .skip(ccs.col_ptrs()[col])
        {
            let row = ccs.row_indices()[index];
            dense.set(row, col, *value);
        }
    }
    dense
}

pub fn compare_dense_matrices(
    lhs: &DenseMatrix,
    rhs: &DenseMatrix,
    zero_tol: f64,
) -> JacobianMismatchSummary {
    assert_eq!(lhs.rows, rhs.rows);
    assert_eq!(lhs.cols, rhs.cols);
    let mut worst = None;
    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;
    let mut missing_nonzeros = 0usize;
    let mut extra_nonzeros = 0usize;
    for row in 0..lhs.rows {
        for col in 0..lhs.cols {
            let lhs_value = lhs.get(row, col);
            let rhs_value = rhs.get(row, col);
            let abs_error = (lhs_value - rhs_value).abs();
            let scale = lhs_value.abs().max(rhs_value.abs()).max(1.0);
            let rel_error = abs_error / scale;
            if lhs_value.abs() <= zero_tol && rhs_value.abs() > zero_tol {
                missing_nonzeros += 1;
            }
            if lhs_value.abs() > zero_tol && rhs_value.abs() <= zero_tol {
                extra_nonzeros += 1;
            }
            if abs_error > max_abs_error || rel_error > max_rel_error {
                max_abs_error = max_abs_error.max(abs_error);
                max_rel_error = max_rel_error.max(rel_error);
                worst = Some(MatrixMismatchEntry {
                    row,
                    col,
                    lhs: lhs_value,
                    rhs: rhs_value,
                    abs_error,
                    rel_error,
                });
            }
        }
    }
    JacobianMismatchSummary {
        rows: lhs.rows,
        cols: lhs.cols,
        max_abs_error,
        max_rel_error,
        worst_entry: worst,
        missing_nonzeros,
        extra_nonzeros,
    }
}

pub fn finite_difference_jacobian(
    function: &CompiledJitFunction,
    x: &[f64],
    step: f64,
) -> DenseMatrix {
    let mut base_context = function.create_context();
    base_context.input_mut(0).copy_from_slice(x);
    function.eval(&mut base_context);
    let rows = base_context.output(0).len();
    let cols = x.len();
    let mut dense = DenseMatrix::zeros(rows, cols);
    for col in 0..cols {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[col] += step;
        xm[col] -= step;
        let mut plus_context = function.create_context();
        let mut minus_context = function.create_context();
        plus_context.input_mut(0).copy_from_slice(&xp);
        minus_context.input_mut(0).copy_from_slice(&xm);
        function.eval(&mut plus_context);
        function.eval(&mut minus_context);
        for row in 0..rows {
            let derivative =
                (plus_context.output(0)[row] - minus_context.output(0)[row]) / (2.0 * step);
            dense.set(row, col, derivative);
        }
    }
    dense
}
