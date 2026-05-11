use super::helpers::*;
use super::*;

pub(super) fn tp011() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp011",
            "tp011",
            "Schittkowski TP011 shifted circle objective with parabolic inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp011",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x - 5.0).sqr() + x.y.sqr() - 25.0,
                    equalities: (),
                    inequalities: x.x.sqr() - x.y,
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 4.9, y: 0.1 },
                parameters: (),
                bounds: scalar_inequality_upper_bound(),
            })
        },
        exact_solution_validation(
            &tp011_solution(),
            X_TOL,
            tp011_objective(),
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
