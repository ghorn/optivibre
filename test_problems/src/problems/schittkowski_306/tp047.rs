use super::helpers::*;
use super::*;

pub(super) fn tp047() -> ProblemCase {
    make_typed_case::<VecN<SX, 5>, (), VecN<SX, 3>, (), _, _>(
        metadata(
            "schittkowski_tp047",
            "tp047",
            "Schittkowski TP047 chain objective with three nonlinear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 5>, (), VecN<SX, 3>, (), _>(
                "schittkowski_tp047",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr()
                            + (x1 - x2).sqr()
                            + (x2 - x3).powi(4)
                            + (x3 - x4).powi(4),
                        equalities: VecN {
                            values: [
                                x0 + x1.sqr() + x2.powi(3) - 3.0,
                                x1 - x2.sqr() + x3 - 1.0,
                                x0 * x4 - 1.0,
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
                    values: [2.0, 2.0_f64.sqrt(), -1.0, 2.0 - 2.0_f64.sqrt(), 0.5],
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
