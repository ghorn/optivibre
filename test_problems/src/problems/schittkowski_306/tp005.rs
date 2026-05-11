use super::helpers::*;
use super::*;

pub(super) fn tp005() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp005",
            "tp005",
            "Schittkowski TP005 bounded sinusoidal quadratic objective",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp005",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x + x.y).sin() + (x.x - x.y).sqr() - 1.5 * x.x + 2.5 * x.y + 1.0,
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.0, y: 0.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(-1.5),
                        y: Some(-3.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(4.0),
                        y: Some(3.0),
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &tp005_solution(),
            X_TOL,
            -3.0_f64.sqrt() / 2.0 - std::f64::consts::PI / 3.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
