use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LqrError {
    #[error("system matrix A must be square")]
    NonSquareA,
    #[error("matrix dimension mismatch: {0}")]
    DimensionMismatch(&'static str),
    #[error("R must be invertible")]
    SingularR,
    #[error("time horizon must be positive")]
    NonPositiveHorizon,
    #[error("step count must be at least 1")]
    InvalidStepCount,
    #[error("query times must be finite, nondecreasing, and lie in [0, tf]")]
    InvalidQueryTimes,
}

#[derive(Clone, Debug)]
pub struct FiniteHorizonLqrProblem {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub qf: DMatrix<f64>,
    pub tf: f64,
    pub steps: usize,
}

#[derive(Clone, Debug)]
pub struct FiniteHorizonLqrSolution {
    pub sample_times: Vec<f64>,
    pub riccati: Vec<DMatrix<f64>>,
    pub gains: Vec<DMatrix<f64>>,
}

#[derive(Clone, Debug)]
pub struct ClosedLoopPoint {
    pub time: f64,
    pub state: DVector<f64>,
    pub control: DVector<f64>,
}

#[derive(Clone, Debug)]
pub struct ClosedLoopTrajectory {
    pub points: Vec<ClosedLoopPoint>,
}

pub fn solve_finite_horizon(
    problem: &FiniteHorizonLqrProblem,
) -> Result<FiniteHorizonLqrSolution, LqrError> {
    validate_problem(problem)?;
    let r_inv = invert(&problem.r)?;
    let step = problem.tf / problem.steps as f64;
    let mut riccati = vec![DMatrix::zeros(problem.a.nrows(), problem.a.ncols()); problem.steps + 1];
    riccati[problem.steps] = symmetrize(problem.qf.clone());

    for index in (0..problem.steps).rev() {
        let next = &riccati[index + 1];
        let prev = rk4_matrix_step(next, -step, |p| {
            riccati_derivative(p, &problem.a, &problem.b, &problem.q, &r_inv)
        });
        riccati[index] = symmetrize(prev);
    }

    let bt = problem.b.transpose();
    let gains = riccati.iter().map(|p| &r_inv * &bt * p).collect::<Vec<_>>();
    let sample_times = (0..=problem.steps)
        .map(|index| index as f64 * step)
        .collect::<Vec<_>>();

    Ok(FiniteHorizonLqrSolution {
        sample_times,
        riccati,
        gains,
    })
}

pub fn approximate_care(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    horizon: f64,
    steps: usize,
) -> Result<DMatrix<f64>, LqrError> {
    let solution = solve_finite_horizon(&FiniteHorizonLqrProblem {
        a: a.clone(),
        b: b.clone(),
        q: q.clone(),
        r: r.clone(),
        qf: q.clone(),
        tf: horizon,
        steps,
    })?;
    Ok(solution.riccati[0].clone())
}

pub fn steady_state_gain(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    horizon: f64,
    steps: usize,
) -> Result<DMatrix<f64>, LqrError> {
    let p = approximate_care(a, b, q, r, horizon, steps)?;
    let r_inv = invert(r)?;
    Ok(r_inv * b.transpose() * p)
}

pub fn simulate_time_varying_closed_loop(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    solution: &FiniteHorizonLqrSolution,
    z0: &DVector<f64>,
    query_times: &[f64],
    substeps_per_segment: usize,
) -> Result<ClosedLoopTrajectory, LqrError> {
    validate_solution_dimensions(a, b, solution, z0)?;
    validate_query_times(solution, query_times)?;

    let mut state = z0.clone();
    let mut current_time = 0.0;
    let mut points = Vec::with_capacity(query_times.len());
    for &target_time in query_times {
        if target_time > current_time {
            let segment_substeps = substeps_per_segment.max(1);
            let dt = (target_time - current_time) / segment_substeps as f64;
            for substep in 0..segment_substeps {
                let t = current_time + substep as f64 * dt;
                state = rk4_vector_step(
                    &state,
                    dt,
                    |z, time| {
                        let gain = gain_at(solution, time);
                        let closed_loop = a - b * gain;
                        closed_loop * z
                    },
                    t,
                );
            }
            current_time = target_time;
        }
        let gain = gain_at(solution, target_time);
        let control = -&gain * &state;
        points.push(ClosedLoopPoint {
            time: target_time,
            state: state.clone(),
            control,
        });
    }

    Ok(ClosedLoopTrajectory { points })
}

fn validate_problem(problem: &FiniteHorizonLqrProblem) -> Result<(), LqrError> {
    if problem.a.nrows() != problem.a.ncols() {
        return Err(LqrError::NonSquareA);
    }
    let n = problem.a.nrows();
    if problem.b.nrows() != n {
        return Err(LqrError::DimensionMismatch("B rows must match A"));
    }
    if problem.q.nrows() != n || problem.q.ncols() != n {
        return Err(LqrError::DimensionMismatch("Q must match A"));
    }
    if problem.qf.nrows() != n || problem.qf.ncols() != n {
        return Err(LqrError::DimensionMismatch("Qf must match A"));
    }
    let m = problem.b.ncols();
    if problem.r.nrows() != m || problem.r.ncols() != m {
        return Err(LqrError::DimensionMismatch("R must match B columns"));
    }
    if !problem.tf.is_finite() || problem.tf <= 0.0 {
        return Err(LqrError::NonPositiveHorizon);
    }
    if problem.steps == 0 {
        return Err(LqrError::InvalidStepCount);
    }
    Ok(())
}

fn validate_solution_dimensions(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    solution: &FiniteHorizonLqrSolution,
    z0: &DVector<f64>,
) -> Result<(), LqrError> {
    if a.nrows() != a.ncols() {
        return Err(LqrError::NonSquareA);
    }
    if b.nrows() != a.nrows() {
        return Err(LqrError::DimensionMismatch("B rows must match A"));
    }
    if z0.len() != a.nrows() {
        return Err(LqrError::DimensionMismatch(
            "initial state length must match A",
        ));
    }
    if solution.sample_times.len() != solution.gains.len()
        || solution.sample_times.len() != solution.riccati.len()
    {
        return Err(LqrError::DimensionMismatch(
            "solution grids must have matching lengths",
        ));
    }
    Ok(())
}

fn validate_query_times(
    solution: &FiniteHorizonLqrSolution,
    query_times: &[f64],
) -> Result<(), LqrError> {
    let tf = *solution
        .sample_times
        .last()
        .ok_or(LqrError::InvalidQueryTimes)?;
    let mut previous = 0.0;
    for (index, &time) in query_times.iter().enumerate() {
        if !time.is_finite() || time < 0.0 || time > tf {
            return Err(LqrError::InvalidQueryTimes);
        }
        if index > 0 && time < previous {
            return Err(LqrError::InvalidQueryTimes);
        }
        previous = time;
    }
    Ok(())
}

fn riccati_derivative(
    p: &DMatrix<f64>,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r_inv: &DMatrix<f64>,
) -> DMatrix<f64> {
    -(a.transpose() * p + p * a - p * b * r_inv * b.transpose() * p + q)
}

fn rk4_matrix_step(
    current: &DMatrix<f64>,
    dt: f64,
    derivative: impl Fn(&DMatrix<f64>) -> DMatrix<f64>,
) -> DMatrix<f64> {
    let k1 = derivative(current);
    let k2 = derivative(&(current + 0.5 * dt * &k1));
    let k3 = derivative(&(current + 0.5 * dt * &k2));
    let k4 = derivative(&(current + dt * &k3));
    current + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

fn rk4_vector_step(
    current: &DVector<f64>,
    dt: f64,
    derivative: impl Fn(&DVector<f64>, f64) -> DVector<f64>,
    time: f64,
) -> DVector<f64> {
    let k1 = derivative(current, time);
    let k2 = derivative(&(current + 0.5 * dt * &k1), time + 0.5 * dt);
    let k3 = derivative(&(current + 0.5 * dt * &k2), time + 0.5 * dt);
    let k4 = derivative(&(current + dt * &k3), time + dt);
    current + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

fn gain_at(solution: &FiniteHorizonLqrSolution, time: f64) -> DMatrix<f64> {
    let len = solution.sample_times.len();
    if time <= solution.sample_times[0] {
        return solution.gains[0].clone();
    }
    if time >= solution.sample_times[len - 1] {
        return solution.gains[len - 1].clone();
    }
    let upper = solution
        .sample_times
        .partition_point(|sample| *sample < time);
    let lower = upper - 1;
    let t0 = solution.sample_times[lower];
    let t1 = solution.sample_times[upper];
    let alpha = (time - t0) / (t1 - t0);
    (1.0 - alpha) * &solution.gains[lower] + alpha * &solution.gains[upper]
}

fn invert(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, LqrError> {
    matrix.clone().lu().try_inverse().ok_or(LqrError::SingularR)
}

fn symmetrize(matrix: DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (&matrix + matrix.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn care_approximation_has_small_residual_for_augmented_double_integrator() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let b = DMatrix::from_column_slice(3, 1, &[0.0, 0.0, 1.0]);
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![10.0, 1.0, 0.5]));
        let r = DMatrix::from_element(1, 1, 0.1);
        let p =
            approximate_care(&a, &b, &q, &r, 20.0, 400).expect("CARE approximation should succeed");
        let r_inv = invert(&r).expect("R should invert");
        let residual = a.transpose() * &p + &p * &a - &p * &b * r_inv * b.transpose() * &p + q;
        let max_abs = residual
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        assert!(max_abs < 1e-3, "CARE residual too large: {max_abs}");
    }

    #[test]
    fn finite_horizon_solution_rolls_out_without_nan() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let b = DMatrix::from_column_slice(3, 1, &[0.0, 0.0, 1.0]);
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![10.0, 1.0, 0.5]));
        let r = DMatrix::from_element(1, 1, 0.1);
        let qf = DMatrix::from_diagonal(&DVector::from_vec(vec![5.0, 1.0, 0.5]));
        let solution = solve_finite_horizon(&FiniteHorizonLqrProblem {
            a: a.clone(),
            b: b.clone(),
            q,
            r,
            qf,
            tf: 2.0,
            steps: 120,
        })
        .expect("finite-horizon solve should succeed");
        let trajectory = simulate_time_varying_closed_loop(
            &a,
            &b,
            &solution,
            &DVector::from_vec(vec![1.0, -0.5, 0.2]),
            &[0.0, 0.5, 1.0, 1.5, 2.0],
            16,
        )
        .expect("closed-loop rollout should succeed");
        assert_eq!(trajectory.points.len(), 5);
        assert_abs_diff_eq!(trajectory.points[0].time, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(trajectory.points[4].time, 2.0, epsilon = 1e-12);
        assert!(
            trajectory
                .points
                .iter()
                .all(|point| point.state.iter().all(|value| value.is_finite()))
        );
    }
}
