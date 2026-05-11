use super::helpers::*;
use super::*;

pub(super) fn tp039() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp039",
            "tp039",
            "Schittkowski TP039 linear objective with two nonlinear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp039",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: -x0,
                        equalities: VecN {
                            values: [x1 - x0.powi(3) - x2.sqr(), x0.sqr() - x1 - x3.sqr()],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [2.0; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[1.0, 1.0, 0.0, 0.0],
            X_TOL,
            -1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
