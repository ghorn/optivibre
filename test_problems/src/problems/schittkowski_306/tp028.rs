use super::helpers::*;
use super::*;

pub(super) fn tp028() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp028",
            "tp028",
            "Schittkowski TP028 quadratic objective with one linear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), SX, (), _>(
                "schittkowski_tp028",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: (x0 + x1).sqr() + (x1 + x2).sqr(),
                        equalities: x0 + 2.0 * x1 + 3.0 * x2 - 1.0,
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [-4.0, 1.0, 1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[0.5, -0.5, 0.5],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
