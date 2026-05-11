use super::helpers::*;
use super::*;

pub(super) fn tp021() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp021",
            "tp021",
            "Schittkowski TP021 quadratic objective with one linear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp021",
                |x, ()| SymbolicNlpOutputs {
                    objective: 0.01 * x.x.sqr() + x.y.sqr() - 100.0,
                    equalities: (),
                    inequalities: -10.0 * x.x + x.y + 10.0,
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: -1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(2.0),
                        y: Some(-50.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(50.0),
                        y: Some(50.0),
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[2.0, 0.0],
            X_TOL,
            -99.96,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
