use super::helpers::*;
use super::*;

pub(super) fn tp026() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp026",
            "tp026",
            "Schittkowski TP026 polynomial objective with one nonlinear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), SX, (), _>(
                "schittkowski_tp026",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr() + (x1 - x2).powf(4.0),
                        equalities: x0 * (1.0 + x1.sqr()) + x2.powf(4.0) - 3.0,
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [-2.6, 2.0, 2.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[1.0, 1.0, 1.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
