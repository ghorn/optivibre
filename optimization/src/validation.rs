use anyhow::{Result, bail};
use nalgebra::{DMatrix, DVector};

use crate::{CCS, CompiledNlpProblem, ParameterMatrix};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteDifferenceValidationOptions {
    pub first_order_step: f64,
    pub second_order_step: f64,
    pub zero_tolerance: f64,
}

impl Default for FiniteDifferenceValidationOptions {
    fn default() -> Self {
        Self {
            first_order_step: 1.0e-6,
            second_order_step: 1.0e-4,
            zero_tolerance: 1.0e-8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidationWorstEntry {
    pub row: usize,
    pub col: usize,
    pub analytic: f64,
    pub finite_difference: f64,
    pub abs_error: f64,
    pub rel_error: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationSparsitySummary {
    pub analytic_nonzeros: usize,
    pub finite_difference_nonzeros: usize,
    pub missing_from_analytic: usize,
    pub extra_in_analytic: usize,
    pub worst_missing_from_analytic: Option<ValidationWorstEntry>,
    pub worst_extra_in_analytic: Option<ValidationWorstEntry>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationSummary {
    pub rows: usize,
    pub cols: usize,
    pub max_abs_error: f64,
    pub rms_abs_error: f64,
    pub max_rel_error: f64,
    pub worst_entry: Option<ValidationWorstEntry>,
    pub sparsity: ValidationSparsitySummary,
    pub pareto_frontier: Vec<ValidationWorstEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidationTolerances {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
}

impl ValidationTolerances {
    pub const fn new(max_abs_error: f64, max_rel_error: f64) -> Self {
        Self {
            max_abs_error,
            max_rel_error,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NlpDerivativeValidationReport {
    pub objective_gradient: ValidationSummary,
    pub equality_jacobian: Option<ValidationSummary>,
    pub inequality_jacobian: Option<ValidationSummary>,
    pub lagrangian_hessian: ValidationSummary,
}

impl ValidationSummary {
    pub fn is_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        !self.pareto_frontier.iter().any(|entry| {
            entry.abs_error > tolerances.max_abs_error
                && entry.rel_error > tolerances.max_rel_error
        })
    }
}

impl NlpDerivativeValidationReport {
    pub fn first_order_is_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        self.objective_gradient.is_within_tolerances(tolerances)
            && self
                .equality_jacobian
                .as_ref()
                .is_none_or(|summary| summary.is_within_tolerances(tolerances))
            && self
                .inequality_jacobian
                .as_ref()
                .is_none_or(|summary| summary.is_within_tolerances(tolerances))
    }

    pub fn second_order_is_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        self.lagrangian_hessian.is_within_tolerances(tolerances)
    }

    pub fn all_orders_are_within_tolerances(&self, tolerances: ValidationTolerances) -> bool {
        self.first_order_is_within_tolerances(tolerances)
            && self.second_order_is_within_tolerances(tolerances)
    }
}

pub fn validate_compiled_nlp_problem_derivatives(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
    options: FiniteDifferenceValidationOptions,
) -> Result<NlpDerivativeValidationReport> {
    if options.first_order_step <= 0.0 || !options.first_order_step.is_finite() {
        bail!("first_order_step must be finite and > 0");
    }
    if options.second_order_step <= 0.0 || !options.second_order_step.is_finite() {
        bail!("second_order_step must be finite and > 0");
    }
    if options.zero_tolerance < 0.0 || !options.zero_tolerance.is_finite() {
        bail!("zero_tolerance must be finite and >= 0");
    }
    if x.len() != problem.dimension() {
        bail!(
            "decision length mismatch: got {}, expected {}",
            x.len(),
            problem.dimension()
        );
    }
    if equality_multipliers.len() != problem.equality_count() {
        bail!(
            "equality multiplier length mismatch: got {}, expected {}",
            equality_multipliers.len(),
            problem.equality_count()
        );
    }
    if inequality_multipliers.len() != problem.inequality_count() {
        bail!(
            "inequality multiplier length mismatch: got {}, expected {}",
            inequality_multipliers.len(),
            problem.inequality_count()
        );
    }

    let gradient_analytic = analytic_objective_gradient(problem, x, parameters);
    ensure_finite_vector("objective_gradient analytic", &gradient_analytic)?;
    let gradient_fd = finite_difference_objective_gradient(problem, x, parameters, options);
    ensure_finite_vector("objective_gradient finite difference", &gradient_fd)?;

    let equality_jacobian = if problem.equality_count() > 0 {
        let analytic = analytic_sparse_matrix(
            problem.equality_jacobian_ccs(),
            |values| problem.equality_jacobian_values(x, parameters, values),
            false,
        );
        ensure_finite_matrix("equality_jacobian analytic", &analytic)?;
        let fd = finite_difference_constraint_jacobian(
            problem,
            x,
            parameters,
            true,
            options.first_order_step,
        );
        ensure_finite_matrix("equality_jacobian finite difference", &fd)?;
        Some(summarize_validation(&analytic, &fd, options.zero_tolerance))
    } else {
        None
    };

    let inequality_jacobian = if problem.inequality_count() > 0 {
        let analytic = analytic_sparse_matrix(
            problem.inequality_jacobian_ccs(),
            |values| problem.inequality_jacobian_values(x, parameters, values),
            false,
        );
        ensure_finite_matrix("inequality_jacobian analytic", &analytic)?;
        let fd = finite_difference_constraint_jacobian(
            problem,
            x,
            parameters,
            false,
            options.first_order_step,
        );
        ensure_finite_matrix("inequality_jacobian finite difference", &fd)?;
        Some(summarize_validation(&analytic, &fd, options.zero_tolerance))
    } else {
        None
    };

    let hessian_analytic = analytic_sparse_matrix(
        problem.lagrangian_hessian_ccs(),
        |values| {
            problem.lagrangian_hessian_values(
                x,
                parameters,
                equality_multipliers,
                inequality_multipliers,
                values,
            )
        },
        true,
    );
    ensure_finite_matrix("lagrangian_hessian analytic", &hessian_analytic)?;
    let hessian_fd = finite_difference_lagrangian_hessian(
        problem,
        x,
        parameters,
        equality_multipliers,
        inequality_multipliers,
        options.second_order_step,
    );
    ensure_finite_matrix("lagrangian_hessian finite difference", &hessian_fd)?;

    Ok(NlpDerivativeValidationReport {
        objective_gradient: summarize_validation(
            &DMatrix::from_column_slice(gradient_analytic.len(), 1, gradient_analytic.as_slice()),
            &DMatrix::from_column_slice(gradient_fd.len(), 1, gradient_fd.as_slice()),
            options.zero_tolerance,
        ),
        equality_jacobian,
        inequality_jacobian,
        lagrangian_hessian: summarize_validation(
            &hessian_analytic,
            &hessian_fd,
            options.zero_tolerance,
        ),
    })
}

fn ensure_finite_vector(label: &str, values: &DVector<f64>) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            bail!("{label} contains non-finite value at index {index}: {value}");
        }
    }
    Ok(())
}

fn ensure_finite_matrix(label: &str, values: &DMatrix<f64>) -> Result<()> {
    for row in 0..values.nrows() {
        for col in 0..values.ncols() {
            let value = values[(row, col)];
            if !value.is_finite() {
                bail!("{label} contains non-finite value at ({row}, {col}): {value}");
            }
        }
    }
    Ok(())
}

fn analytic_objective_gradient(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
) -> DVector<f64> {
    let mut gradient = vec![0.0; problem.dimension()];
    problem.objective_gradient(x, parameters, &mut gradient);
    DVector::from_vec(gradient)
}

fn analytic_sparse_matrix(
    ccs: &CCS,
    fill_values: impl FnOnce(&mut [f64]),
    symmetric: bool,
) -> DMatrix<f64> {
    let mut values = vec![0.0; ccs.nnz()];
    fill_values(&mut values);
    dense_from_ccs(ccs, &values, symmetric)
}

fn dense_from_ccs(ccs: &CCS, values: &[f64], symmetric: bool) -> DMatrix<f64> {
    assert_eq!(values.len(), ccs.nnz());
    let mut dense = DMatrix::<f64>::zeros(ccs.nrow, ccs.ncol);
    for col in 0..ccs.ncol {
        for idx in ccs.col_ptrs[col]..ccs.col_ptrs[col + 1] {
            let row = ccs.row_indices[idx];
            let value = values[idx];
            dense[(row, col)] = value;
            if symmetric && row != col {
                dense[(col, row)] = value;
            }
        }
    }
    dense
}

fn finite_difference_objective_gradient(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: FiniteDifferenceValidationOptions,
) -> DVector<f64> {
    let n = x.len();
    let mut gradient = DVector::<f64>::zeros(n);
    for index in 0..n {
        let plus = with_perturbed_component(x, index, options.first_order_step);
        let minus = with_perturbed_component(x, index, -options.first_order_step);
        let plus_value = problem.objective_value(&plus, parameters);
        let minus_value = problem.objective_value(&minus, parameters);
        gradient[index] = (plus_value - minus_value) / (2.0 * options.first_order_step);
    }
    gradient
}

fn finite_difference_constraint_jacobian(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    equality: bool,
    step: f64,
) -> DMatrix<f64> {
    let rows = if equality {
        problem.equality_count()
    } else {
        problem.inequality_count()
    };
    let cols = x.len();
    let mut dense = DMatrix::<f64>::zeros(rows, cols);
    if rows == 0 {
        return dense;
    }
    let mut plus_values = vec![0.0; rows];
    let mut minus_values = vec![0.0; rows];
    for col in 0..cols {
        let plus = with_perturbed_component(x, col, step);
        let minus = with_perturbed_component(x, col, -step);
        if equality {
            problem.equality_values(&plus, parameters, &mut plus_values);
            problem.equality_values(&minus, parameters, &mut minus_values);
        } else {
            problem.inequality_values(&plus, parameters, &mut plus_values);
            problem.inequality_values(&minus, parameters, &mut minus_values);
        }
        for row in 0..rows {
            dense[(row, col)] = (plus_values[row] - minus_values[row]) / (2.0 * step);
        }
    }
    dense
}

fn finite_difference_lagrangian_hessian(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
    step: f64,
) -> DMatrix<f64> {
    let n = x.len();
    let mut dense = DMatrix::<f64>::zeros(n, n);
    let center = lagrangian_scalar(
        problem,
        x,
        parameters,
        equality_multipliers,
        inequality_multipliers,
    );
    for row in 0..n {
        let plus = with_perturbed_component(x, row, step);
        let minus = with_perturbed_component(x, row, -step);
        let plus_value = lagrangian_scalar(
            problem,
            &plus,
            parameters,
            equality_multipliers,
            inequality_multipliers,
        );
        let minus_value = lagrangian_scalar(
            problem,
            &minus,
            parameters,
            equality_multipliers,
            inequality_multipliers,
        );
        dense[(row, row)] = (plus_value - 2.0 * center + minus_value) / (step * step);
        for col in 0..row {
            let plus_plus = with_two_perturbed_components(x, row, step, col, step);
            let plus_minus = with_two_perturbed_components(x, row, step, col, -step);
            let minus_plus = with_two_perturbed_components(x, row, -step, col, step);
            let minus_minus = with_two_perturbed_components(x, row, -step, col, -step);
            let value = (lagrangian_scalar(
                problem,
                &plus_plus,
                parameters,
                equality_multipliers,
                inequality_multipliers,
            ) - lagrangian_scalar(
                problem,
                &plus_minus,
                parameters,
                equality_multipliers,
                inequality_multipliers,
            ) - lagrangian_scalar(
                problem,
                &minus_plus,
                parameters,
                equality_multipliers,
                inequality_multipliers,
            ) + lagrangian_scalar(
                problem,
                &minus_minus,
                parameters,
                equality_multipliers,
                inequality_multipliers,
            )) / (4.0 * step * step);
            dense[(row, col)] = value;
            dense[(col, row)] = value;
        }
    }
    dense
}

fn lagrangian_scalar(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
) -> f64 {
    let mut value = problem.objective_value(x, parameters);
    if !equality_multipliers.is_empty() {
        let mut equality_values = vec![0.0; equality_multipliers.len()];
        problem.equality_values(x, parameters, &mut equality_values);
        value += equality_multipliers
            .iter()
            .zip(equality_values.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
    }
    if !inequality_multipliers.is_empty() {
        let mut inequality_values = vec![0.0; inequality_multipliers.len()];
        problem.inequality_values(x, parameters, &mut inequality_values);
        value += inequality_multipliers
            .iter()
            .zip(inequality_values.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
    }
    value
}

fn with_perturbed_component(values: &[f64], index: usize, delta: f64) -> Vec<f64> {
    let mut perturbed = values.to_vec();
    perturbed[index] += delta;
    perturbed
}

fn with_two_perturbed_components(
    values: &[f64],
    first_index: usize,
    first_delta: f64,
    second_index: usize,
    second_delta: f64,
) -> Vec<f64> {
    let mut perturbed = values.to_vec();
    perturbed[first_index] += first_delta;
    perturbed[second_index] += second_delta;
    perturbed
}

fn summarize_validation(
    analytic: &DMatrix<f64>,
    finite_difference: &DMatrix<f64>,
    zero_tolerance: f64,
) -> ValidationSummary {
    assert_eq!(analytic.shape(), finite_difference.shape());
    let mut max_abs_error = 0.0;
    let mut max_rel_error = 0.0;
    let mut squared_error_sum = 0.0;
    let mut worst_entry = None;
    let mut sparsity = ValidationSparsitySummary::default();
    let mut pareto_frontier = Vec::new();
    let count = analytic.nrows() * analytic.ncols();

    for row in 0..analytic.nrows() {
        for col in 0..analytic.ncols() {
            let analytic_value = analytic[(row, col)];
            let fd_value = finite_difference[(row, col)];
            let abs_error = (analytic_value - fd_value).abs();
            let scale = analytic_value
                .abs()
                .max(fd_value.abs())
                .max(zero_tolerance.max(f64::EPSILON));
            let rel_error = abs_error / scale;
            squared_error_sum += abs_error * abs_error;
            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_rel_error = rel_error.max(max_rel_error);
                worst_entry = Some(ValidationWorstEntry {
                    row,
                    col,
                    analytic: analytic_value,
                    finite_difference: fd_value,
                    abs_error,
                    rel_error,
                });
            } else {
                max_rel_error = max_rel_error.max(rel_error);
            }
            record_pareto_entry(
                &mut pareto_frontier,
                ValidationWorstEntry {
                    row,
                    col,
                    analytic: analytic_value,
                    finite_difference: fd_value,
                    abs_error,
                    rel_error,
                },
            );

            let analytic_nz = analytic_value.abs() > zero_tolerance;
            let fd_nz = fd_value.abs() > zero_tolerance;
            sparsity.analytic_nonzeros += usize::from(analytic_nz);
            sparsity.finite_difference_nonzeros += usize::from(fd_nz);
            match (analytic_nz, fd_nz) {
                (false, true) => {
                    sparsity.missing_from_analytic += 1;
                    if sparsity
                        .worst_missing_from_analytic
                        .is_none_or(|entry| abs_error > entry.abs_error)
                    {
                        sparsity.worst_missing_from_analytic = Some(ValidationWorstEntry {
                            row,
                            col,
                            analytic: analytic_value,
                            finite_difference: fd_value,
                            abs_error,
                            rel_error,
                        });
                    }
                }
                (true, false) => {
                    sparsity.extra_in_analytic += 1;
                    if sparsity
                        .worst_extra_in_analytic
                        .is_none_or(|entry| abs_error > entry.abs_error)
                    {
                        sparsity.worst_extra_in_analytic = Some(ValidationWorstEntry {
                            row,
                            col,
                            analytic: analytic_value,
                            finite_difference: fd_value,
                            abs_error,
                            rel_error,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    ValidationSummary {
        rows: analytic.nrows(),
        cols: analytic.ncols(),
        max_abs_error,
        rms_abs_error: (squared_error_sum / count.max(1) as f64).sqrt(),
        max_rel_error,
        worst_entry,
        sparsity,
        pareto_frontier,
    }
}

fn record_pareto_entry(frontier: &mut Vec<ValidationWorstEntry>, candidate: ValidationWorstEntry) {
    if frontier.iter().any(|entry| {
        entry.abs_error >= candidate.abs_error && entry.rel_error >= candidate.rel_error
    }) {
        return;
    }
    frontier.retain(|entry| {
        !(candidate.abs_error >= entry.abs_error && candidate.rel_error >= entry.rel_error)
    });
    frontier.push(candidate);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_check_is_entrywise_not_maxwise() {
        let analytic = DMatrix::from_row_slice(1, 2, &[10_000.0, 0.0]);
        let finite_difference = DMatrix::from_row_slice(1, 2, &[10_000.018_65, 1.137e-5]);
        let summary = summarize_validation(&analytic, &finite_difference, 1.0e-7);

        assert!((summary.max_abs_error - 1.865e-2).abs() < 1.0e-12);
        assert!((summary.max_rel_error - 1.0).abs() < 1.0e-12);
        assert!(
            summary.is_within_tolerances(ValidationTolerances::new(1.0e-4, 1.0e-3)),
            "each entry should satisfy abs-or-rel tolerances: {:?}",
            summary
        );
    }
}
