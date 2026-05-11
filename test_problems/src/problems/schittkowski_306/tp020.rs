use super::helpers::*;
use super::*;

pub(super) fn tp020() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 3>, _, _>(
        metadata(
            "schittkowski_tp020",
            "tp020",
            "Schittkowski TP020 Rosenbrock objective with three nonlinear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 3>, _>(
                "schittkowski_tp020",
                |x, ()| SymbolicNlpOutputs {
                    objective: rosenbrock_objective(x.x, x.y),
                    equalities: (),
                    inequalities: VecN {
                        values: [
                            -x.y.sqr() - x.x,
                            -x.x.sqr() - x.y,
                            1.0 - x.x.sqr() - x.y.sqr(),
                        ],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.1, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(-0.5),
                        y: None,
                    }),
                    variable_upper: Some(Pair {
                        x: Some(0.5),
                        y: None,
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.5, 3.0_f64.sqrt() * 0.5],
            X_TOL,
            81.5 - 25.0 * 3.0_f64.sqrt(),
            1e-7,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
