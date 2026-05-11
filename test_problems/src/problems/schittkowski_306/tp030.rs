use super::helpers::*;
use super::*;

pub(super) fn tp030() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp030",
            "tp030",
            "Schittkowski TP030 sphere objective with box bounds and one inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp030",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: x0.sqr() + x1.sqr() + x2.sqr(),
                        equalities: (),
                        inequalities: 1.0 - x0.sqr() - x1.sqr(),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [1.0, 1.0, 1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(1.0), Some(-10.0), Some(-10.0)],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(10.0), Some(10.0), Some(10.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 0.0, 0.0],
            X_TOL,
            1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
