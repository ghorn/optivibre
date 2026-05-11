use super::helpers::*;
use super::*;

pub(super) fn tp019() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp019",
            "tp019",
            "Schittkowski TP019 cubic objective with two annular inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp019",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x - 10.0).powf(3.0) + (x.y - 20.0).powf(3.0),
                    equalities: (),
                    inequalities: VecN {
                        values: [
                            100.0 - (x.x - 5.0).sqr() - (x.y - 5.0).sqr(),
                            (x.x - 6.0).sqr() + (x.y - 5.0).sqr() - 82.81,
                        ],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 20.1, y: 5.84 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(13.0),
                        y: Some(0.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(100.0),
                        y: Some(100.0),
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
            &tp019_solution(),
            1e-3,
            tp019_objective(),
            1e-5,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
