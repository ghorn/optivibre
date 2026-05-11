use super::helpers::*;
use super::*;

pub(super) fn tp045() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp045",
            "tp045",
            "Schittkowski TP045 product objective with simple bounds",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), (), (), _>(
                "schittkowski_tp045",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: 2.0 - x0 * x1 * x2 * x3 * x4 / 120.0,
                        equalities: (),
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
                        values: [Some(0.0); 5],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            X_TOL,
            1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
