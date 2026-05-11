use super::helpers::*;
use super::*;

pub(super) fn tp031() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp031",
            "tp031",
            "Schittkowski TP031 weighted sphere objective with bilinear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp031",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 9.0 * x0.sqr() + x1.sqr() + 9.0 * x2.sqr(),
                        equalities: (),
                        inequalities: 1.0 - x0 * x1,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [1.0, 1.0, 1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(-10.0), Some(1.0), Some(-10.0)],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(10.0), Some(10.0), Some(1.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0 / 3.0_f64.sqrt(), 3.0_f64.sqrt(), 0.0],
            X_TOL,
            6.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
