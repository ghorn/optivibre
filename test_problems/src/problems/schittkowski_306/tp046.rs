use super::helpers::*;
use super::*;

pub(super) fn tp046() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp046",
            "tp046",
            "Schittkowski TP046 five-variable problem with two nonlinear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp046",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr()
                            + (x2 - 1.0).sqr()
                            + (x3 - 1.0).powi(4)
                            + (x4 - 1.0).powi(6),
                        equalities: VecN {
                            values: [
                                x0.sqr() * x3 + (x3 - x4).sin() - 1.0,
                                x1 + x2.powi(4) * x3.sqr() - 2.0,
                            ],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.5 * 2.0_f64.sqrt(), 1.75, 0.5, 2.0, 2.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[1.0; 5],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
