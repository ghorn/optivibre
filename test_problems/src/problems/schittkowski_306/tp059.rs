use super::helpers::*;
use super::*;

pub(super) fn tp059() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 3>, _, _>(
        metadata(
            "schittkowski_tp059",
            "tp059",
            "Schittkowski TP059 polynomial objective with three nonlinear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 3>, _>(
                "schittkowski_tp059",
                |x, ()| {
                    let x0 = x.x;
                    let x1 = x.y;
                    SymbolicNlpOutputs {
                        objective: tp059_symbolic_objective(x0, x1),
                        equalities: (),
                        inequalities: VecN {
                            values: [
                                700.0 - x0 * x1,
                                0.008 * x0.sqr() - x1,
                                5.0 * (x0 - 55.0) - (x1 - 50.0).sqr(),
                            ],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 90.0, y: 10.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(0.0),
                        y: Some(0.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(75.0),
                        y: Some(65.0),
                    }),
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<3>(),
                    scaling: None,
                },
            })
        },
        objective_validation(
            -7.80422632408,
            1e-8,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
