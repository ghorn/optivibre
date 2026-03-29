use anyhow::{Result, bail};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::ZeroConeT;
use clarabel::solver::implementations::default::DefaultSettingsBuilder;
use clarabel::solver::{DefaultSolver, IPSolver, SolverStatus};
use nalgebra::{DMatrix, SymmetricEigen};
use thiserror::Error;

pub type Index = usize;
pub type SignedIndex = isize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CCS {
    pub nrow: Index,
    pub ncol: Index,
    pub col_ptrs: Vec<Index>,
    pub row_indices: Vec<Index>,
}

impl CCS {
    pub fn new(nrow: Index, ncol: Index, col_ptrs: Vec<Index>, row_indices: Vec<Index>) -> Self {
        Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        }
    }

    pub fn nnz(&self) -> Index {
        self.row_indices.len()
    }

    pub fn lower_triangular_dense(size: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(size + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col in 0..size {
            row_indices.extend(col..size);
            col_ptrs.push(row_indices.len());
        }
        Self {
            nrow: size,
            ncol: size,
            col_ptrs,
            row_indices,
        }
    }
}

pub trait CompiledObjective {
    fn dimension(&self) -> Index;
    fn value(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64], out: &mut [f64]);
    fn hessian_ccs(&self) -> &CCS;
    fn hessian_values(&self, x: &[f64], out: &mut [f64]);
}

pub trait CompiledEqualityConstraints {
    fn constraint_count(&self) -> Index;
    fn values(&self, x: &[f64], out: &mut [f64]);
    fn jacobian_ccs(&self) -> &CCS;
    fn jacobian_values(&self, x: &[f64], out: &mut [f64]);
}

#[derive(Clone, Debug)]
pub struct ClarabelSqpOptions {
    pub max_iters: Index,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub merit_penalty: f64,
    pub regularization: f64,
    pub armijo_c1: f64,
    pub line_search_beta: f64,
    pub min_step: f64,
    pub verbose: bool,
}

impl Default for ClarabelSqpOptions {
    fn default() -> Self {
        Self {
            max_iters: 25,
            dual_tol: 1e-6,
            constraint_tol: 1e-6,
            merit_penalty: 10.0,
            regularization: 1e-6,
            armijo_c1: 1e-4,
            line_search_beta: 0.5,
            min_step: 1e-4,
            verbose: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClarabelSqpSummary {
    pub x: Vec<f64>,
    pub multipliers: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub constraint_inf_norm: f64,
    pub dual_inf_norm: f64,
}

#[derive(Debug, Error)]
pub enum ClarabelSqpError {
    #[error("clarabel SQP failed to converge in {iterations} iterations")]
    MaxIterations { iterations: Index },
    #[error("clarabel solver setup failed: {0}")]
    Setup(String),
    #[error("clarabel returned status {status:?}")]
    QpSolve { status: SolverStatus },
    #[error(
        "armijo line search failed to find sufficient decrease (directional derivative {directional_derivative}, step inf-norm {step_inf_norm})"
    )]
    LineSearchFailed {
        directional_derivative: f64,
        step_inf_norm: f64,
    },
}

fn ccs_to_dense(sp: &CCS, values: &[f64]) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(sp.nrow, sp.ncol);
    for col in 0..sp.ncol {
        let start = sp.col_ptrs[col];
        let end = sp.col_ptrs[col + 1];
        for (offset, &row) in sp.row_indices[start..end].iter().enumerate() {
            dense[(row, col)] = values[start + offset];
        }
    }
    dense
}

fn lower_triangle_to_symmetric_dense(sp: &CCS, values: &[f64]) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(sp.nrow, sp.ncol);
    for col in 0..sp.ncol {
        let start = sp.col_ptrs[col];
        let end = sp.col_ptrs[col + 1];
        for (offset, &row) in sp.row_indices[start..end].iter().enumerate() {
            let value = values[start + offset];
            dense[(row, col)] = value;
            dense[(col, row)] = value;
        }
    }
    dense
}

fn dense_to_csc_upper(matrix: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = matrix.ncols();
    let mut col_ptrs = Vec::with_capacity(n + 1);
    let mut row_vals = Vec::new();
    let mut nz_vals = Vec::new();
    col_ptrs.push(0);
    for col in 0..n {
        for row in 0..=col {
            let value = matrix[(row, col)];
            if value.abs() > 1e-12 || row == col {
                row_vals.push(row);
                nz_vals.push(value);
            }
        }
        col_ptrs.push(row_vals.len());
    }
    CscMatrix::new(n, n, col_ptrs, row_vals, nz_vals)
}

fn dense_to_csc(matrix: &DMatrix<f64>) -> CscMatrix<f64> {
    let ncol = matrix.ncols();
    let nrow = matrix.nrows();
    let mut col_ptrs = Vec::with_capacity(ncol + 1);
    let mut row_vals = Vec::new();
    let mut nz_vals = Vec::new();
    col_ptrs.push(0);
    for col in 0..ncol {
        for row in 0..nrow {
            let value = matrix[(row, col)];
            if value.abs() > 1e-12 {
                row_vals.push(row);
                nz_vals.push(value);
            }
        }
        col_ptrs.push(row_vals.len());
    }
    CscMatrix::new(nrow, ncol, col_ptrs, row_vals, nz_vals)
}

fn inf_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
        .sum()
}

fn mat_vec(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; matrix.nrows()];
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            product[row] += matrix[(row, col)] * vector[col];
        }
    }
    product
}

fn lagrangian_gradient(gradient: &[f64], jacobian: &DMatrix<f64>, multipliers: &[f64]) -> Vec<f64> {
    let mut residual = gradient.to_vec();
    for row in 0..jacobian.nrows() {
        let lambda = multipliers[row];
        for col in 0..jacobian.ncols() {
            residual[col] += jacobian[(row, col)] * lambda;
        }
    }
    residual
}

fn merit_value<O, C>(
    objective: &O,
    constraints: &C,
    x: &[f64],
    scratch: &mut [f64],
    penalty: f64,
) -> f64
where
    O: CompiledObjective,
    C: CompiledEqualityConstraints,
{
    constraints.values(x, scratch);
    objective.value(x) + 0.5 * penalty * scratch.iter().map(|v| v * v).sum::<f64>()
}

fn updated_merit_penalty(
    current_penalty: f64,
    gradient: &[f64],
    constraints: &[f64],
    jacobian: &DMatrix<f64>,
    step: &[f64],
) -> f64 {
    let objective_rate = dot(gradient, step);
    let linearized_constraints = mat_vec(jacobian, step);
    let constraint_rate = dot(constraints, &linearized_constraints);
    if constraint_rate >= -1e-12 {
        return current_penalty;
    }
    let required_penalty = (objective_rate / -constraint_rate).max(0.0);
    current_penalty.max(2.0 * required_penalty + 1e-8)
}

fn merit_directional_derivative(
    gradient: &[f64],
    constraints: &[f64],
    jacobian: &DMatrix<f64>,
    step: &[f64],
    penalty: f64,
) -> f64 {
    let linearized_constraints = mat_vec(jacobian, step);
    dot(gradient, step) + penalty * dot(constraints, &linearized_constraints)
}

pub fn solve_equality_constrained_sqp<O, C>(
    objective: &O,
    constraints: &C,
    x0: &[f64],
    options: &ClarabelSqpOptions,
) -> std::result::Result<ClarabelSqpSummary, ClarabelSqpError>
where
    O: CompiledObjective,
    C: CompiledEqualityConstraints,
{
    let n = objective.dimension();
    let m = constraints.constraint_count();
    assert_eq!(x0.len(), n);
    let mut x = x0.to_vec();
    let mut gradient = vec![0.0; n];
    let mut constraints_values = vec![0.0; m];
    let mut hessian_values = vec![0.0; objective.hessian_ccs().nnz()];
    let mut jacobian_values = vec![0.0; constraints.jacobian_ccs().nnz()];
    let mut merit_penalty = options.merit_penalty;

    for iteration in 0..options.max_iters {
        let objective_value = objective.value(&x);
        objective.gradient(&x, &mut gradient);
        constraints.values(&x, &mut constraints_values);
        let constraint_inf = inf_norm(&constraints_values);

        objective.hessian_values(&x, &mut hessian_values);
        constraints.jacobian_values(&x, &mut jacobian_values);
        let mut hessian =
            lower_triangle_to_symmetric_dense(objective.hessian_ccs(), &hessian_values);
        let jacobian = ccs_to_dense(constraints.jacobian_ccs(), &jacobian_values);

        let eigen = SymmetricEigen::new(hessian.clone());
        let min_eig = eigen
            .eigenvalues
            .iter()
            .fold(f64::INFINITY, |acc, value| acc.min(*value));
        let shift = if min_eig < options.regularization {
            options.regularization - min_eig
        } else {
            options.regularization
        };
        for idx in 0..n {
            hessian[(idx, idx)] += shift;
        }

        let p = dense_to_csc_upper(&hessian);
        let a = dense_to_csc(&jacobian);
        let b = constraints_values
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let cones = vec![ZeroConeT(m)];
        let settings = DefaultSettingsBuilder::default()
            .verbose(options.verbose)
            .build()
            .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
        let mut solver = DefaultSolver::new(&p, &gradient, &a, &b, &cones, settings)
            .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
        solver.solve();
        let status = solver.solution.status;
        if !matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
            return Err(ClarabelSqpError::QpSolve { status });
        }
        let step = solver.solution.x.clone();
        let multipliers = solver.solution.z.clone();
        let dual_residual = lagrangian_gradient(&gradient, &jacobian, &multipliers);
        let dual_inf = inf_norm(&dual_residual);
        if constraint_inf <= options.constraint_tol && dual_inf <= options.dual_tol {
            return Ok(ClarabelSqpSummary {
                x,
                multipliers,
                objective: objective_value,
                iterations: iteration,
                constraint_inf_norm: constraint_inf,
                dual_inf_norm: dual_inf,
            });
        }

        merit_penalty = updated_merit_penalty(
            merit_penalty,
            &gradient,
            &constraints_values,
            &jacobian,
            &step,
        );
        let directional_derivative = merit_directional_derivative(
            &gradient,
            &constraints_values,
            &jacobian,
            &step,
            merit_penalty,
        );
        let current_merit = objective_value
            + 0.5
                * merit_penalty
                * constraints_values
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>();
        let mut alpha = 1.0;
        let mut accepted = false;
        while alpha >= options.min_step {
            let trial = x
                .iter()
                .zip(step.iter())
                .map(|(xi, di)| xi + alpha * di)
                .collect::<Vec<_>>();
            let trial_merit = merit_value(
                objective,
                constraints,
                &trial,
                &mut constraints_values,
                merit_penalty,
            );
            if trial_merit <= current_merit + options.armijo_c1 * alpha * directional_derivative {
                x = trial;
                accepted = true;
                break;
            }
            alpha *= options.line_search_beta;
        }
        if !accepted {
            return Err(ClarabelSqpError::LineSearchFailed {
                directional_derivative,
                step_inf_norm: inf_norm(&step),
            });
        }
    }

    Err(ClarabelSqpError::MaxIterations {
        iterations: options.max_iters,
    })
}

pub fn validate_problem_shapes<O, C>(objective: &O, constraints: &C) -> Result<()>
where
    O: CompiledObjective,
    C: CompiledEqualityConstraints,
{
    let n = objective.dimension();
    if objective.hessian_ccs().nrow != n || objective.hessian_ccs().ncol != n {
        bail!("objective Hessian CCS must be square with dimension {n}");
    }
    if constraints.jacobian_ccs().nrow != constraints.constraint_count()
        || constraints.jacobian_ccs().ncol != n
    {
        bail!("constraint Jacobian CCS does not match declared dimensions");
    }
    Ok(())
}
