use super::helpers::*;
use super::*;

pub(super) fn tp016() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp016",
            "tp016",
            "Schittkowski TP016 Rosenbrock objective with bounds and two inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp016",
                |x, ()| SymbolicNlpOutputs {
                    objective: rosenbrock_objective(x.x, x.y),
                    equalities: (),
                    inequalities: VecN {
                        values: [-x.y.sqr() - x.x, -x.x.sqr() - x.y],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -2.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(-0.5),
                        y: None,
                    }),
                    variable_upper: Some(Pair {
                        x: Some(0.5),
                        y: Some(1.0),
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
            &[0.5, 0.25],
            X_TOL,
            0.25,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
