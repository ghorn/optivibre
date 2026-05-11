use super::helpers::*;
use super::*;

pub(super) fn tp055() -> ProblemCase {
    make_typed_case::<VecN<SX, 6>, (), VecN<SX, 6>, (), _, _>(
        metadata(
            "schittkowski_tp055",
            "tp055",
            "Schittkowski TP055 exponential objective with six linear equalities and simple bounds",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 6>, (), VecN<SX, 6>, (), _>(
                "schittkowski_tp055",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    let x5 = x.values[5];
                    SymbolicNlpOutputs {
                        objective: x0 + 2.0 * x1 + 4.0 * x4 + (x0 * x3).exp(),
                        equalities: VecN {
                            values: [
                                x0 + 2.0 * x1 + 5.0 * x4 - 6.0,
                                x0 + x1 + x2 - 3.0,
                                x3 + x4 + x5 - 2.0,
                                x0 + x3 - 1.0,
                                x1 + x4 - 2.0,
                                x2 + x5 - 2.0,
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
                    values: [1.0, 2.0, 0.0, 0.0, 0.0, 2.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 6],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(1.0), None, None, Some(1.0), None, None],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 4.0 / 3.0, 5.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0],
            X_TOL,
            19.0 / 3.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
