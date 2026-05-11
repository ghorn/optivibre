use super::helpers::*;
use super::*;

pub(super) fn tp038() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp038",
            "tp038",
            "Schittkowski TP038 extended Rosenbrock problem with box bounds",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), (), (), _>(
                "schittkowski_tp038",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: 100.0 * (x1 - x0.sqr()).sqr()
                            + (1.0 - x0).sqr()
                            + 90.0 * (x3 - x2.sqr()).sqr()
                            + (1.0 - x2).sqr()
                            + 10.1 * ((x1 - 1.0).sqr() + (x3 - 1.0).sqr())
                            + 19.8 * (x1 - 1.0) * (x3 - 1.0),
                        equalities: (),
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [-3.0, -1.0, -3.0, -1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(-10.0); 4],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(10.0); 4],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 1.0, 1.0, 1.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
