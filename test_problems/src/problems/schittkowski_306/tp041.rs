use super::helpers::*;
use super::*;

pub(super) fn tp041() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp041",
            "tp041",
            "Schittkowski TP041 cubic objective with one linear equality and box bounds",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), SX, (), _>(
                "schittkowski_tp041",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: 2.0 - x0 * x1 * x2,
                        equalities: x0 + 2.0 * x1 + 2.0 * x2 - x3,
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [1.0; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 4],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(1.0), Some(1.0), Some(1.0), Some(2.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0],
            X_TOL,
            52.0 / 27.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
