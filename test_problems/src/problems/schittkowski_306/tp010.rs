use super::helpers::*;
use super::*;

pub(super) fn tp010() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp010",
            "tp010",
            "Schittkowski TP010 linear objective with one nonlinear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp010",
                |x, ()| SymbolicNlpOutputs {
                    objective: x.x - x.y,
                    equalities: (),
                    inequalities: 3.0 * x.x.sqr() - 2.0 * x.x * x.y + x.y.sqr() - 1.0,
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -10.0, y: 10.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: None,
                    inequality_lower: Some(None),
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 1.0],
            X_TOL,
            -1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
