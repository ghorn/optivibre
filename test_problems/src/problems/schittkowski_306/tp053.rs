use super::helpers::*;
use super::*;

pub(super) fn tp053() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 3>, (), _, _>(
        metadata(
            "schittkowski_tp053",
            "tp053",
            "Schittkowski TP053 bounded quadratic objective with three linear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 3>, (), _>(
                "schittkowski_tp053",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr()
                            + (x1 + x2 - 2.0).sqr()
                            + (x3 - 1.0).sqr()
                            + (x4 - 1.0).sqr(),
                        equalities: VecN {
                            values: [x0 + 3.0 * x1, x2 + x3 - 2.0 * x4, x1 - x4],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [2.0; 5] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(-10.0); 5],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(10.0); 5],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[
                -33.0 / 43.0,
                11.0 / 43.0,
                27.0 / 43.0,
                -5.0 / 43.0,
                11.0 / 43.0,
            ],
            X_TOL,
            176.0 / 43.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
