use super::helpers::*;
use super::*;

pub(super) fn tp014() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, SX, _, _>(
        metadata(
            "schittkowski_tp014",
            "tp014",
            "Schittkowski TP014 quadratic objective with one equality and one inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, SX, _>(
                "schittkowski_tp014",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x - 2.0).sqr() + (x.y - 1.0).sqr(),
                    equalities: x.x - 2.0 * x.y + 1.0,
                    inequalities: 0.25 * x.x.sqr() + x.y.sqr() - 1.0,
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: 2.0 },
                parameters: (),
                bounds: scalar_inequality_upper_bound(),
            })
        },
        exact_solution_validation(
            &tp014_solution(),
            X_TOL,
            9.0 - 23.0 * 7.0_f64.sqrt() / 8.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
