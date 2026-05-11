use super::helpers::*;
use super::*;

pub(super) fn tp033() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp033",
            "tp033",
            "Schittkowski TP033 cubic objective with two spherical inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp033",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: (x0 - 1.0) * (x0 - 2.0) * (x0 - 3.0) + x2,
                        equalities: (),
                        inequalities: VecN {
                            values: [
                                x0.sqr() + x1.sqr() - x2.sqr(),
                                4.0 - x0.sqr() - x1.sqr() - x2.sqr(),
                            ],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.0, 0.0, 3.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: Some(VecN {
                        values: [None, None, Some(5.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<2>(),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 2.0_f64.sqrt(), 2.0_f64.sqrt()],
            X_TOL,
            2.0_f64.sqrt() - 6.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
