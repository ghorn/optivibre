use super::helpers::*;
use super::*;

pub(super) fn tp025() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp025",
            "tp025",
            "Schittkowski TP025 bounded three-parameter exponential fit",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), (), _>(
                "schittkowski_tp025",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let mut objective = SX::zero();
                    // The source fallback for u_i - x_2 < 0 is unreachable under
                    // the documented x_2 <= 25.6 bound; min_i u_i is about 25.63.
                    for i in 1..=99 {
                        let i_float = i as f64;
                        let u = 25.0 + (-50.0 * (0.01 * i_float).ln()).powf(2.0 / 3.0);
                        let residual = (-(u - x1).pow(x2) / x0).exp() - 0.01 * i_float;
                        objective += residual.sqr();
                    }
                    SymbolicNlpOutputs {
                        objective,
                        equalities: (),
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [100.0, 12.5, 3.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.1), Some(1e-5), Some(1e-5)],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(100.0), Some(25.6), Some(5.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[50.0, 25.0, 1.5],
            5e-4,
            0.0,
            1e-8,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
