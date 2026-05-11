use super::helpers::*;
use super::*;

pub(super) fn tp002() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp002",
            "tp002",
            "Schittkowski TP002 Rosenbrock problem with active lower bound",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp002",
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
                        y: Some(1.5),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &tp002_solution(),
            5e-5,
            tp002_objective(),
            1e-7,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
