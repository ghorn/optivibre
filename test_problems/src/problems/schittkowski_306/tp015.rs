use super::helpers::*;
use super::*;

pub(super) fn tp015() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp015",
            "tp015",
            "Schittkowski TP015 scaled Rosenbrock objective with two inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp015",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.y - x.x.sqr()).sqr() + 0.01 * (1.0 - x.x).sqr(),
                    equalities: (),
                    inequalities: VecN {
                        values: [1.0 - x.x * x.y, -x.y.sqr() - x.x],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -2.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: Some(Pair {
                        x: Some(0.5),
                        y: None,
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
            &[0.5, 2.0],
            X_TOL,
            3.065,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
