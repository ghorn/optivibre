use super::helpers::*;
use super::*;

pub(super) fn tp061() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp061",
            "tp061",
            "Schittkowski TP061 quadratic objective with two nonlinear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp061",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 4.0 * x0.sqr() + 2.0 * x1.sqr() + 2.0 * x2.sqr() - 33.0 * x0
                            + 16.0 * x1
                            - 24.0 * x2,
                        equalities: VecN {
                            values: [3.0 * x0 - 2.0 * x1.sqr() - 7.0, 4.0 * x0 - x2.sqr() - 11.0],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [0.0; 3] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(-143.646142201, 1e-6, PRIMAL_TOL, DUAL_TOL, None),
    )
}
