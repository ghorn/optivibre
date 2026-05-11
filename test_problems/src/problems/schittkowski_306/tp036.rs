use super::helpers::*;
use super::*;

pub(super) fn tp036() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp036",
            "tp036",
            "Schittkowski TP036 product objective with one linear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp036",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: -x0 * x1 * x2,
                        equalities: (),
                        inequalities: x0 + 2.0 * x1 + 2.0 * x2 - 72.0,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [10.0, 10.0, 10.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(20.0), Some(11.0), Some(42.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[20.0, 11.0, 15.0],
            X_TOL,
            -3300.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
