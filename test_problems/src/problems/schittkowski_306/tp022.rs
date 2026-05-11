use super::helpers::*;
use super::*;

pub(super) fn tp022() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp022",
            "tp022",
            "Schittkowski TP022 quadratic objective with linear and parabolic inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp022",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x - 2.0).sqr() + (x.y - 1.0).sqr(),
                    equalities: (),
                    inequalities: VecN {
                        values: [x.x + x.y - 2.0, x.x.sqr() - x.y],
                    },
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: 2.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(VecN {
                        values: [Some(0.0); 2],
                    }),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 1.0],
            X_TOL,
            1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
