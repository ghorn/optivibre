use super::helpers::*;
use super::*;

pub(super) fn tp023() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 5>, _, _>(
        metadata(
            "schittkowski_tp023",
            "tp023",
            "Schittkowski TP023 norm objective with five inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 5>, _>(
                "schittkowski_tp023",
                |x, ()| SymbolicNlpOutputs {
                    objective: x.x.sqr() + x.y.sqr(),
                    equalities: (),
                    inequalities: VecN {
                        values: [
                            1.0 - x.x - x.y,
                            1.0 - x.x.sqr() - x.y.sqr(),
                            9.0 - 9.0 * x.x.sqr() - x.y.sqr(),
                            x.y - x.x.sqr(),
                            x.x - x.y.sqr(),
                        ],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 3.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(-50.0),
                        y: Some(-50.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(50.0),
                        y: Some(50.0),
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(VecN {
                        values: [Some(0.0); 5],
                    }),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 1.0],
            X_TOL,
            2.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
