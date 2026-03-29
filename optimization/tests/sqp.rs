use approx::assert_abs_diff_eq;
use optimization::{
    CCS, ClarabelSqpOptions, CompiledEqualityConstraints, CompiledObjective,
    solve_equality_constrained_sqp, validate_problem_shapes,
};

struct RosenbrockObjective {
    ccs: CCS,
}

impl RosenbrockObjective {
    fn new() -> Self {
        Self {
            ccs: CCS::lower_triangular_dense(2),
        }
    }
}

impl CompiledObjective for RosenbrockObjective {
    fn dimension(&self) -> usize {
        2
    }

    fn value(&self, x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    fn gradient(&self, x: &[f64], out: &mut [f64]) {
        out[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        out[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    fn hessian_ccs(&self) -> &CCS {
        &self.ccs
    }

    fn hessian_values(&self, x: &[f64], out: &mut [f64]) {
        out[0] = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;
        out[1] = -400.0 * x[0];
        out[2] = 200.0;
    }
}

struct SumEqualsOne {
    ccs: CCS,
}

impl SumEqualsOne {
    fn new() -> Self {
        Self {
            ccs: CCS::new(1, 2, vec![0, 1, 2], vec![0, 0]),
        }
    }
}

impl CompiledEqualityConstraints for SumEqualsOne {
    fn constraint_count(&self) -> usize {
        1
    }

    fn values(&self, x: &[f64], out: &mut [f64]) {
        out[0] = x[0] + x[1] - 1.0;
    }

    fn jacobian_ccs(&self) -> &CCS {
        &self.ccs
    }

    fn jacobian_values(&self, _x: &[f64], out: &mut [f64]) {
        out[0] = 1.0;
        out[1] = 1.0;
    }
}

#[test]
fn clarabel_sqp_solves_constrained_rosenbrock_smoke() {
    let objective = RosenbrockObjective::new();
    let constraints = SumEqualsOne::new();
    validate_problem_shapes(&objective, &constraints).unwrap();

    let summary = solve_equality_constrained_sqp(
        &objective,
        &constraints,
        &[-1.2, 1.2],
        &ClarabelSqpOptions {
            max_iters: 30,
            merit_penalty: 20.0,
            ..ClarabelSqpOptions::default()
        },
    )
    .unwrap();

    assert_abs_diff_eq!(summary.x[0] + summary.x[1], 1.0, epsilon = 1e-4);
    assert!(summary.objective < 10.0);
    assert!(summary.constraint_inf_norm <= 1e-4);
    assert!(summary.dual_inf_norm <= 1e-4);
    assert_eq!(summary.multipliers.len(), 1);
}
