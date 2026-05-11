use super::helpers::*;
use super::*;

pub(super) fn tp018() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp018",
            "tp018",
            "Schittkowski TP018 diagonal quadratic objective with two lower-type inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp018",
                |x, ()| SymbolicNlpOutputs {
                    objective: 0.01 * x.x.sqr() + x.y.sqr(),
                    equalities: (),
                    inequalities: VecN {
                        values: [25.0 - x.x * x.y, 25.0 - x.x.sqr() - x.y.sqr()],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: 2.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(2.0),
                        y: Some(0.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(50.0),
                        y: Some(50.0),
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(VecN {
                        values: [Some(0.0); 2],
                    }),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &tp018_solution(),
            X_TOL,
            5.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
