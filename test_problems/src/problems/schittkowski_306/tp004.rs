use super::helpers::*;
use super::*;

pub(super) fn tp004() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp004",
            "tp004",
            "Schittkowski TP004 cubic objective with lower bounds",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp004",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x + 1.0).powi(3) / 3.0 + x.y,
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 1.125, y: 0.125 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(1.0),
                        y: Some(0.0),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 0.0],
            X_TOL,
            8.0 / 3.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
