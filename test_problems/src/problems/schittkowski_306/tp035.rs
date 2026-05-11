use super::helpers::*;
use super::*;

pub(super) fn tp035() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp035",
            "tp035",
            "Schittkowski TP035 quadratic objective with one linear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp035",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 9.0 - 8.0 * x0 - 6.0 * x1 - 4.0 * x2
                            + 2.0 * x0.sqr()
                            + 2.0 * x1.sqr()
                            + x2.sqr()
                            + 2.0 * x0 * x1
                            + 2.0 * x0 * x2,
                        equalities: (),
                        inequalities: x0 + x1 + 2.0 * x2 - 3.0,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.5, 0.5, 0.5],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0],
            X_TOL,
            1.0 / 9.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
