use super::helpers::*;
use super::*;

pub(super) fn tp013() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp013",
            "tp013",
            "Schittkowski TP013 quadratic objective with cubic inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp013",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x - 2.0).sqr() + x.y.sqr(),
                    equalities: (),
                    inequalities: x.y - (1.0 - x.x).powi(3),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.0, y: 0.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(0.0),
                        y: Some(0.0),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 0.0],
            X_TOL,
            1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
