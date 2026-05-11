use super::helpers::*;
use super::*;

pub(super) fn tp001() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp001",
            "tp001",
            "Schittkowski TP001 Rosenbrock problem with one lower bound",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp001",
                |x, ()| SymbolicNlpOutputs {
                    objective: rosenbrock_objective(x.x, x.y),
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -2.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: None,
                        y: Some(-1.5),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 1.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
