use super::helpers::*;
use super::*;

pub(super) fn tp032() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), SX, SX, _, _>(
        metadata(
            "schittkowski_tp032",
            "tp032",
            "Schittkowski TP032 quadratic objective with one equality and one inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), SX, SX, _>(
                "schittkowski_tp032",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: (x0 + 3.0 * x1 + x2).sqr() + 4.0 * (x0 - x1).sqr(),
                        equalities: 1.0 - x0 - x1 - x2,
                        inequalities: x0.powi(3) - 6.0 * x1 - 4.0 * x2 + 3.0,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.1, 0.7, 0.2],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 0.0, 1.0],
            X_TOL,
            1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
