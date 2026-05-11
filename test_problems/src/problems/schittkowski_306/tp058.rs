use super::helpers::*;
use super::*;

pub(super) fn tp058() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 3>, _, _>(
        metadata(
            "schittkowski_tp058",
            "tp058",
            "Schittkowski TP058 Rosenbrock problem with three nonlinear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 3>, _>(
                "schittkowski_tp058",
                |x, ()| SymbolicNlpOutputs {
                    objective: rosenbrock_objective(x.x, x.y),
                    equalities: (),
                    inequalities: VecN {
                        values: [
                            x.x - x.y.sqr(),
                            x.y - x.x.sqr(),
                            1.0 - x.x.sqr() - x.y.sqr(),
                        ],
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
                        x: Some(-2.0),
                        y: None,
                    }),
                    variable_upper: Some(Pair {
                        x: Some(0.5),
                        y: None,
                    }),
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<3>(),
                    scaling: None,
                },
            })
        },
        objective_validation(
            3.19033354957,
            1e-8,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
