use crate::{
    CCS, CompiledNlpProblem, Index, ParameterMatrix, complementarity_inf_norm, inf_norm,
    lagrangian_gradient, positive_part_inf_norm, validate_nlp_problem_shapes,
    validate_parameter_inputs,
};
use anyhow::{Result, bail};
use ipopt::{
    BasicProblem, ConstrainedProblem, Index as IpoptIndex, IntermediateCallbackData, Ipopt,
    NewtonProblem, Number, SolveStatus,
};
use thiserror::Error;

const IPOPT_INF: f64 = 1e20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpoptMuStrategy {
    Adaptive,
    Monotone,
}

impl IpoptMuStrategy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Monotone => "monotone",
        }
    }
}

#[derive(Clone, Debug)]
pub struct IpoptOptions {
    pub max_iters: Index,
    pub tol: f64,
    pub acceptable_tol: Option<f64>,
    pub constraint_tol: Option<f64>,
    pub complementarity_tol: Option<f64>,
    pub dual_tol: Option<f64>,
    pub print_level: i32,
    pub suppress_banner: bool,
    pub mu_strategy: IpoptMuStrategy,
}

impl Default for IpoptOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tol: 1e-8,
            acceptable_tol: Some(1e-6),
            constraint_tol: Some(1e-8),
            complementarity_tol: Some(1e-8),
            dual_tol: Some(1e-8),
            print_level: 0,
            suppress_banner: true,
            mu_strategy: IpoptMuStrategy::Adaptive,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IpoptSummary {
    pub x: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub status: SolveStatus,
    pub equality_inf_norm: f64,
    pub inequality_inf_norm: f64,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: f64,
}

#[derive(Debug, Error)]
pub enum IpoptSolveError {
    #[error("invalid IPOPT input: {0}")]
    InvalidInput(String),
    #[error("ipopt setup failed: {0}")]
    Setup(String),
    #[error("ipopt rejected option `{name}`")]
    OptionRejected { name: String },
    #[error("ipopt failed with status {status:?}")]
    Solve { status: SolveStatus },
}

fn ccs_triplet_indices(
    ccs: &CCS,
    row_offset: Index,
) -> std::result::Result<(Vec<IpoptIndex>, Vec<IpoptIndex>), IpoptSolveError> {
    let mut rows = Vec::with_capacity(ccs.nnz());
    let mut cols = Vec::with_capacity(ccs.nnz());
    for col in 0..ccs.ncol {
        let start = ccs.col_ptrs[col];
        let end = ccs.col_ptrs[col + 1];
        let col_index = IpoptIndex::try_from(col).map_err(|_| {
            IpoptSolveError::InvalidInput(format!(
                "column index {col} does not fit into Ipopt index type"
            ))
        })?;
        for &row in &ccs.row_indices[start..end] {
            let row_index = row + row_offset;
            rows.push(IpoptIndex::try_from(row_index).map_err(|_| {
                IpoptSolveError::InvalidInput(format!(
                    "row index {row_index} does not fit into Ipopt index type"
                ))
            })?);
            cols.push(col_index);
        }
    }
    Ok((rows, cols))
}

fn validate_ipopt_compatibility<P>(problem: &P) -> Result<()>
where
    P: CompiledNlpProblem,
{
    validate_nlp_problem_shapes(problem)?;
    let hessian_ccs = problem.lagrangian_hessian_ccs();
    if hessian_ccs.nrow != hessian_ccs.ncol {
        bail!("Ipopt requires a square lower-triangular Hessian pattern");
    }
    for col in 0..hessian_ccs.ncol {
        let start = hessian_ccs.col_ptrs[col];
        let end = hessian_ccs.col_ptrs[col + 1];
        for &row in &hessian_ccs.row_indices[start..end] {
            if row < col {
                bail!("Ipopt Hessian CCS must contain only lower-triangular entries");
            }
        }
    }
    Ok(())
}

struct IpoptProblemAdapter<'a, P> {
    problem: &'a P,
    x0: &'a [f64],
    parameters: &'a [ParameterMatrix<'a>],
    constraint_rows: Vec<IpoptIndex>,
    constraint_cols: Vec<IpoptIndex>,
    hessian_rows: Vec<IpoptIndex>,
    hessian_cols: Vec<IpoptIndex>,
    iterations: Index,
}

impl<'a, P> IpoptProblemAdapter<'a, P>
where
    P: CompiledNlpProblem,
{
    fn new(
        problem: &'a P,
        x0: &'a [f64],
        parameters: &'a [ParameterMatrix<'a>],
    ) -> std::result::Result<Self, IpoptSolveError> {
        let (mut constraint_rows, mut constraint_cols) =
            ccs_triplet_indices(problem.equality_jacobian_ccs(), 0)?;
        let (inequality_rows, inequality_cols) =
            ccs_triplet_indices(problem.inequality_jacobian_ccs(), problem.equality_count())?;
        constraint_rows.extend(inequality_rows);
        constraint_cols.extend(inequality_cols);
        let (hessian_rows, hessian_cols) =
            ccs_triplet_indices(problem.lagrangian_hessian_ccs(), 0)?;
        Ok(Self {
            problem,
            x0,
            parameters,
            constraint_rows,
            constraint_cols,
            hessian_rows,
            hessian_cols,
            iterations: 0,
        })
    }

    fn update_iterations(&mut self, data: IntermediateCallbackData) -> bool {
        self.iterations = data.iter_count as usize;
        true
    }

    fn objective_hessian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        let mut equality_multipliers = vec![0.0; self.problem.equality_count()];
        let mut inequality_multipliers = vec![0.0; self.problem.inequality_count()];
        self.problem.lagrangian_hessian_values(
            x,
            self.parameters,
            &equality_multipliers,
            &inequality_multipliers,
            vals,
        );
        equality_multipliers.clear();
        inequality_multipliers.clear();
        true
    }
}

impl<P> BasicProblem for IpoptProblemAdapter<'_, P>
where
    P: CompiledNlpProblem,
{
    fn num_variables(&self) -> usize {
        self.problem.dimension()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        self.problem.variable_bounds(x_l, x_u)
    }

    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.copy_from_slice(self.x0);
        true
    }

    fn objective(&self, x: &[Number], _new_x: bool, obj: &mut Number) -> bool {
        *obj = self.problem.objective_value(x, self.parameters);
        true
    }

    fn objective_grad(&self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        self.problem.objective_gradient(x, self.parameters, grad_f);
        true
    }
}

impl<P> NewtonProblem for IpoptProblemAdapter<'_, P>
where
    P: CompiledNlpProblem,
{
    fn num_hessian_non_zeros(&self) -> usize {
        self.problem.lagrangian_hessian_ccs().nnz()
    }

    fn hessian_indices(&self, rows: &mut [IpoptIndex], cols: &mut [IpoptIndex]) -> bool {
        rows.copy_from_slice(&self.hessian_rows);
        cols.copy_from_slice(&self.hessian_cols);
        true
    }

    fn hessian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        self.objective_hessian_values(x, vals)
    }
}

impl<P> ConstrainedProblem for IpoptProblemAdapter<'_, P>
where
    P: CompiledNlpProblem,
{
    fn num_constraints(&self) -> usize {
        self.problem.equality_count() + self.problem.inequality_count()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        self.constraint_rows.len()
    }

    fn constraint(&self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let equality_count = self.problem.equality_count();
        let (equality_out, inequality_out) = g.split_at_mut(equality_count);
        self.problem
            .equality_values(x, self.parameters, equality_out);
        self.problem
            .inequality_values(x, self.parameters, inequality_out);
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let equality_count = self.problem.equality_count();
        let (equality_lower, inequality_lower) = g_l.split_at_mut(equality_count);
        let (equality_upper, inequality_upper) = g_u.split_at_mut(equality_count);
        equality_lower.fill(0.0);
        equality_upper.fill(0.0);
        inequality_lower.fill(-IPOPT_INF);
        inequality_upper.fill(0.0);
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [IpoptIndex],
        cols: &mut [IpoptIndex],
    ) -> bool {
        rows.copy_from_slice(&self.constraint_rows);
        cols.copy_from_slice(&self.constraint_cols);
        true
    }

    fn constraint_jacobian_values(&self, x: &[Number], _new_x: bool, vals: &mut [Number]) -> bool {
        let equality_nnz = self.problem.equality_jacobian_ccs().nnz();
        let (equality_vals, inequality_vals) = vals.split_at_mut(equality_nnz);
        self.problem
            .equality_jacobian_values(x, self.parameters, equality_vals);
        self.problem
            .inequality_jacobian_values(x, self.parameters, inequality_vals);
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        self.problem.lagrangian_hessian_ccs().nnz()
    }

    fn hessian_indices(&self, rows: &mut [IpoptIndex], cols: &mut [IpoptIndex]) -> bool {
        rows.copy_from_slice(&self.hessian_rows);
        cols.copy_from_slice(&self.hessian_cols);
        true
    }

    fn hessian_values(
        &self,
        x: &[Number],
        _new_x: bool,
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let equality_count = self.problem.equality_count();
        let (equality_multipliers, inequality_multipliers) = lambda.split_at(equality_count);
        let mut objective_hessian = vec![0.0; vals.len()];
        self.problem.lagrangian_hessian_values(
            x,
            self.parameters,
            &vec![0.0; equality_count],
            &vec![0.0; self.problem.inequality_count()],
            &mut objective_hessian,
        );
        self.problem.lagrangian_hessian_values(
            x,
            self.parameters,
            equality_multipliers,
            inequality_multipliers,
            vals,
        );
        for (value, objective_value) in vals.iter_mut().zip(objective_hessian.iter()) {
            *value += (obj_factor - 1.0) * objective_value;
        }
        true
    }
}

fn set_ipopt_option<P, O>(
    solver: &mut Ipopt<P>,
    name: &str,
    value: O,
) -> std::result::Result<(), IpoptSolveError>
where
    P: BasicProblem,
    O: Into<ipopt::IpoptOption<'static>>,
{
    if solver.set_option(name, value).is_none() {
        return Err(IpoptSolveError::OptionRejected {
            name: name.to_owned(),
        });
    }
    Ok(())
}

fn solve_status_is_success(status: SolveStatus) -> bool {
    matches!(
        status,
        SolveStatus::SolveSucceeded
            | SolveStatus::SolvedToAcceptableLevel
            | SolveStatus::FeasiblePointFound
    )
}

pub fn solve_nlp_ipopt<'a, P>(
    problem: &'a P,
    x0: &'a [f64],
    parameters: &'a [ParameterMatrix<'a>],
    options: &IpoptOptions,
) -> std::result::Result<IpoptSummary, IpoptSolveError>
where
    P: CompiledNlpProblem,
{
    validate_ipopt_compatibility(problem)
        .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
    validate_parameter_inputs(problem, parameters)
        .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
    if x0.len() != problem.dimension() {
        return Err(IpoptSolveError::InvalidInput(format!(
            "x0 has length {}, expected {}",
            x0.len(),
            problem.dimension()
        )));
    }

    let adapter = IpoptProblemAdapter::new(problem, x0, parameters)?;
    let total_constraint_count = problem.equality_count() + problem.inequality_count();
    if total_constraint_count == 0 {
        let mut solver =
            Ipopt::new_newton(adapter).map_err(|err| IpoptSolveError::Setup(format!("{err:?}")))?;
        set_ipopt_option(&mut solver, "max_iter", options.max_iters as i32)?;
        set_ipopt_option(&mut solver, "tol", options.tol)?;
        if let Some(value) = options.acceptable_tol {
            set_ipopt_option(&mut solver, "acceptable_tol", value)?;
        }
        if let Some(value) = options.dual_tol {
            set_ipopt_option(&mut solver, "dual_inf_tol", value)?;
        }
        set_ipopt_option(&mut solver, "mu_strategy", options.mu_strategy.as_str())?;
        set_ipopt_option(&mut solver, "print_level", options.print_level)?;
        if options.suppress_banner {
            set_ipopt_option(&mut solver, "sb", "yes")?;
        }
        solver.set_intermediate_callback(Some(IpoptProblemAdapter::update_iterations));
        let solve_result = solver.solve();
        let status = solve_result.status;
        let objective = solve_result.objective_value;
        let x = solve_result.solver_data.solution.primal_variables.to_vec();
        let lower_bound_multipliers = solve_result
            .solver_data
            .solution
            .lower_bound_multipliers
            .to_vec();
        let upper_bound_multipliers = solve_result
            .solver_data
            .solution
            .upper_bound_multipliers
            .to_vec();
        let iterations = solve_result.solver_data.problem.iterations;
        if !solve_status_is_success(status) {
            return Err(IpoptSolveError::Solve { status });
        }
        return Ok(IpoptSummary {
            x,
            lower_bound_multipliers,
            upper_bound_multipliers,
            equality_multipliers: Vec::new(),
            inequality_multipliers: Vec::new(),
            objective,
            iterations,
            status,
            equality_inf_norm: 0.0,
            inequality_inf_norm: 0.0,
            primal_inf_norm: 0.0,
            dual_inf_norm: 0.0,
            complementarity_inf_norm: 0.0,
        });
    }

    let mut solver =
        Ipopt::new(adapter).map_err(|err| IpoptSolveError::Setup(format!("{err:?}")))?;
    set_ipopt_option(&mut solver, "max_iter", options.max_iters as i32)?;
    set_ipopt_option(&mut solver, "tol", options.tol)?;
    if let Some(value) = options.acceptable_tol {
        set_ipopt_option(&mut solver, "acceptable_tol", value)?;
    }
    if let Some(value) = options.constraint_tol {
        set_ipopt_option(&mut solver, "constr_viol_tol", value)?;
    }
    if let Some(value) = options.complementarity_tol {
        set_ipopt_option(&mut solver, "compl_inf_tol", value)?;
    }
    if let Some(value) = options.dual_tol {
        set_ipopt_option(&mut solver, "dual_inf_tol", value)?;
    }
    set_ipopt_option(&mut solver, "mu_strategy", options.mu_strategy.as_str())?;
    set_ipopt_option(&mut solver, "print_level", options.print_level)?;
    if options.suppress_banner {
        set_ipopt_option(&mut solver, "sb", "yes")?;
    }
    solver.set_intermediate_callback(Some(IpoptProblemAdapter::update_iterations));

    let solve_result = solver.solve();
    let status = solve_result.status;
    let objective = solve_result.objective_value;
    let x = solve_result.solver_data.solution.primal_variables.to_vec();
    let lower_bound_multipliers = solve_result
        .solver_data
        .solution
        .lower_bound_multipliers
        .to_vec();
    let upper_bound_multipliers = solve_result
        .solver_data
        .solution
        .upper_bound_multipliers
        .to_vec();
    let iterations = solve_result.solver_data.problem.iterations;
    let constraint_multipliers = solve_result
        .solver_data
        .solution
        .constraint_multipliers
        .to_vec();
    if !solve_status_is_success(status) {
        return Err(IpoptSolveError::Solve { status });
    }

    let equality_count = problem.equality_count();
    let equality_multipliers = constraint_multipliers[..equality_count].to_vec();
    let inequality_multipliers = constraint_multipliers[equality_count..].to_vec();
    let mut gradient = vec![0.0; problem.dimension()];
    problem.objective_gradient(&x, parameters, &mut gradient);
    let mut equality_values = vec![0.0; equality_count];
    let mut inequality_values = vec![0.0; problem.inequality_count()];
    problem.equality_values(&x, parameters, &mut equality_values);
    problem.inequality_values(&x, parameters, &mut inequality_values);
    let mut equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    problem.equality_jacobian_values(&x, parameters, &mut equality_jacobian_values);
    problem.inequality_jacobian_values(&x, parameters, &mut inequality_jacobian_values);
    let equality_jacobian =
        super::ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
    let inequality_jacobian = super::ccs_to_dense(
        problem.inequality_jacobian_ccs(),
        &inequality_jacobian_values,
    );
    let lagrangian_residual = lagrangian_gradient(
        &gradient,
        &equality_jacobian,
        &equality_multipliers,
        &inequality_jacobian,
        &inequality_multipliers,
    );
    let equality_inf_norm = inf_norm(&equality_values);
    let inequality_inf_norm = positive_part_inf_norm(&inequality_values);
    let primal_inf_norm = equality_inf_norm.max(inequality_inf_norm);
    let dual_inf_norm = inf_norm(&lagrangian_residual);
    let complementarity_inf_norm =
        complementarity_inf_norm(&inequality_values, &inequality_multipliers);

    Ok(IpoptSummary {
        x,
        lower_bound_multipliers,
        upper_bound_multipliers,
        equality_multipliers,
        inequality_multipliers,
        objective,
        iterations,
        status,
        equality_inf_norm,
        inequality_inf_norm,
        primal_inf_norm,
        dual_inf_norm,
        complementarity_inf_norm,
    })
}
