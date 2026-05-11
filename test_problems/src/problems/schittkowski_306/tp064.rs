use super::helpers::*;
use super::*;

pub(super) fn tp064() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp064",
            "tp064",
            "Schittkowski TP064 reciprocal objective with one nonlinear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp064",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 5.0 * x0
                            + 5.0e4 / x0
                            + 20.0 * x1
                            + 7.2e4 / x1
                            + 10.0 * x2
                            + 1.44e5 / x2,
                        equalities: (),
                        inequalities: -1.0 + 4.0 / x0 + 32.0 / x1 + 120.0 / x2,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [1.0; 3] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(1e-5); 3],
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        objective_validation(
            6299.84242821,
            1e-5,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
