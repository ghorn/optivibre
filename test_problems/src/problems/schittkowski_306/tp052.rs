use super::helpers::*;
use super::*;

pub(super) fn tp052() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 3>, (), _, _>(
        metadata(
            "schittkowski_tp052",
            "tp052",
            "Schittkowski TP052 weighted quadratic objective with three linear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 3>, (), _>(
                "schittkowski_tp052",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (4.0 * x0 - x1).sqr()
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
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[
                -33.0 / 349.0,
                11.0 / 349.0,
                180.0 / 349.0,
                -158.0 / 349.0,
                11.0 / 349.0,
            ],
            X_TOL,
            1859.0 / 349.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
