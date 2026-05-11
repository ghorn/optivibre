use super::helpers::*;
use super::*;

pub(super) fn tp044() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), (), VecN<SX, 6>, _, _>(
        metadata(
            "schittkowski_tp044",
            "tp044",
            "Schittkowski TP044 bilinear objective with six linear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), (), VecN<SX, 6>, _>(
                "schittkowski_tp044",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: x0 - x1 - x2 - x0 * x2 + x0 * x3 + x1 * x2 - x1 * x3,
                        equalities: (),
                        inequalities: VecN {
                            values: [
                                x0 + 2.0 * x1 - 8.0,
                                4.0 * x0 + x1 - 12.0,
                                3.0 * x0 + 4.0 * x1 - 12.0,
                                2.0 * x2 + x3 - 8.0,
                                x2 + 2.0 * x3 - 8.0,
                                x2 + x3 - 5.0,
                            ],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [0.0; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 4],
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<6>(),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 3.0, 0.0, 4.0],
            X_TOL,
            -15.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
