use super::helpers::*;
use super::*;

pub(super) fn tp048() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp048",
            "tp048",
            "Schittkowski TP048 quadratic objective with two linear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp048",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (x0 - 1.0).sqr() + (x1 - x2).sqr() + (x3 - x4).sqr(),
                        equalities: VecN {
                            values: [x0 + x1 + x2 + x3 + x4 - 5.0, x2 - 2.0 * (x3 + x4) + 3.0],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [3.0, 5.0, -3.0, 2.0, -2.0],
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
