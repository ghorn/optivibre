use super::helpers::*;
use super::*;

pub(super) fn tp037() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp037",
            "tp037",
            "Schittkowski TP037 product objective with two linear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp037",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: -x0 * x1 * x2,
                        equalities: (),
                        inequalities: VecN {
                            values: [x0 + 2.0 * x1 + 2.0 * x2 - 72.0, -x0 - 2.0 * x1 - 2.0 * x2],
                        },
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
                        values: [Some(42.0); 3],
                    }),
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<2>(),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[24.0, 12.0, 12.0],
            X_TOL,
            -3456.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
