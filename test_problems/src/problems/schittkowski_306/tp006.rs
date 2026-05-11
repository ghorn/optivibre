use super::helpers::*;
use super::*;

pub(super) fn tp006() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp006",
            "tp006",
            "Schittkowski TP006 Rosenbrock equality-constrained problem",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, (), _>(
                "schittkowski_tp006",
                |x, ()| SymbolicNlpOutputs {
                    objective: (1.0 - x.x).sqr(),
                    equalities: 10.0 * (x.y - x.x.sqr()),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -1.2, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[1.0, 1.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
