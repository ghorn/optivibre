use super::helpers::*;
use super::*;

pub(super) fn tp012() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp012",
            "tp012",
            "Schittkowski TP012 quadratic objective with ellipsoidal inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp012",
                |x, ()| SymbolicNlpOutputs {
                    objective: 0.5 * x.x.sqr() + x.y.sqr() - x.x * x.y - 7.0 * x.x - 7.0 * x.y,
                    equalities: (),
                    inequalities: 4.0 * x.x.sqr() + x.y.sqr() - 25.0,
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.0, y: 0.0 },
                parameters: (),
                bounds: scalar_inequality_upper_bound(),
            })
        },
        exact_solution_validation(
            &[2.0, 3.0],
            X_TOL,
            -30.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
