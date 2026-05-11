use super::helpers::*;
use super::*;

pub(super) fn tp063() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp063",
            "tp063",
            "Schittkowski TP063 quadratic objective with one linear and one nonlinear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp063",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 1000.0
                            - x0.sqr()
                            - 2.0 * x1.sqr()
                            - x2.sqr()
                            - x0 * x1
                            - x0 * x2,
                        equalities: VecN {
                            values: [
                                8.0 * x0 + 14.0 * x1 + 7.0 * x2 - 56.0,
                                x0.sqr() + x1.sqr() + x2.sqr() - 25.0,
                            ],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [2.0; 3] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        objective_validation(961.715172127, 1e-6, PRIMAL_TOL, DUAL_TOL, None),
    )
}
