use super::helpers::*;
use super::*;

pub(super) fn tp043() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), (), VecN<SX, 3>, _, _>(
        metadata(
            "schittkowski_tp043",
            "tp043",
            "Schittkowski TP043 quadratic objective with three nonlinear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), (), VecN<SX, 3>, _>(
                "schittkowski_tp043",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: x0.sqr() + x1.sqr() + 2.0 * x2.sqr() + x3.sqr()
                            - 5.0 * x0
                            - 5.0 * x1
                            - 21.0 * x2
                            + 7.0 * x3,
                        equalities: (),
                        inequalities: VecN {
                            values: [
                                x0.sqr() + x1.sqr() + x2.sqr() + x3.sqr() + x0 - x1 + x2 - x3 - 8.0,
                                x0.sqr() + 2.0 * x1.sqr() + x2.sqr() + 2.0 * x3.sqr()
                                    - x0
                                    - x3
                                    - 10.0,
                                2.0 * x0.sqr() + x1.sqr() + x2.sqr() + 2.0 * x0 - x1 - x3 - 5.0,
                            ],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [0.0; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<3>(),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 1.0, 2.0, -1.0],
            X_TOL,
            -44.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
