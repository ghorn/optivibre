use super::helpers::*;
use super::*;

pub(super) fn tp062() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp062",
            "tp062",
            "Schittkowski TP062 logarithmic process design objective with one linear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), SX, (), _>(
                "schittkowski_tp062",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let b3 = x2 + 0.03;
                    let c3 = 0.13 * x2 + 0.03;
                    let b2 = b3 + x1;
                    let c2 = b3 + 0.07 * x1;
                    let b1 = b2 + x0;
                    let c1 = b2 + 0.09 * x0;
                    SymbolicNlpOutputs {
                        objective: -32.174
                            * (255.0 * (b1 / c1).log()
                                + 280.0 * (b2 / c2).log()
                                + 290.0 * (b3 / c3).log()),
                        equalities: x0 + x1 + x2 - 1.0,
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.7, 0.2, 0.1],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(1.0); 3],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        objective_validation(-26272.5144873, 1e-5, PRIMAL_TOL, DUAL_TOL, None),
    )
}
