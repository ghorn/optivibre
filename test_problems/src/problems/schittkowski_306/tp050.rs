use super::helpers::*;
use super::*;

pub(super) fn tp050() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 3>, (), _, _>(
        metadata(
            "schittkowski_tp050",
            "tp050",
            "Schittkowski TP050 chain objective with three linear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 3>, (), _>(
                "schittkowski_tp050",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr()
                            + (x1 - x2).sqr()
                            + (x2 - x3).powi(4)
                            + (x3 - x4).powi(4),
                        equalities: VecN {
                            values: [
                                x0 + 2.0 * x1 + 3.0 * x2 - 6.0,
                                x1 + 2.0 * x2 + 3.0 * x3 - 6.0,
                                x2 + 2.0 * x3 + 3.0 * x4 - 6.0,
                            ],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [35.0, -31.0, 11.0, 5.0, -5.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[1.0; 5],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
