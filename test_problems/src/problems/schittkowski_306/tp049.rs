use super::helpers::*;
use super::*;

pub(super) fn tp049() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp049",
            "tp049",
            "Schittkowski TP049 polynomial objective with two linear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp049",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr()
                            + (x2 - 1.0).sqr()
                            + (x3 - 1.0).powf(4.0)
                            + (x4 - 1.0).powf(6.0),
                        equalities: VecN {
                            values: [x0 + x1 + x2 + 4.0 * x3 - 7.0, x2 + 5.0 * x4 - 6.0],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [10.0, 7.0, 2.0, -3.0, 0.8],
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
