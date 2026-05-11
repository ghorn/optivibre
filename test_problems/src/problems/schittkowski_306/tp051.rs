use super::helpers::*;
use super::*;

pub(super) fn tp051() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 3>, (), _, _>(
        metadata(
            "schittkowski_tp051",
            "tp051",
            "Schittkowski TP051 quadratic objective with three linear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 3>, (), _>(
                "schittkowski_tp051",
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
                            values: [x0 + 3.0 * x1 - 4.0, x2 + x3 - 2.0 * x4, x1 - x4],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [2.5, 0.5, 2.0, -1.0, 0.5],
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
