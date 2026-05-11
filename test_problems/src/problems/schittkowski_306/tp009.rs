use super::helpers::*;
use super::*;

pub(super) fn tp009() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp009",
            "tp009",
            "Schittkowski TP009 sinusoidal objective with linear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, (), _>(
                "schittkowski_tp009",
                |x, ()| SymbolicNlpOutputs {
                    objective: (std::f64::consts::PI * x.x / 12.0).sin()
                        * (std::f64::consts::PI * x.y / 16.0).cos(),
                    equalities: 4.0 * x.x - 3.0 * x.y,
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.0, y: 0.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[-3.0, -4.0],
            X_TOL,
            -0.5,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
