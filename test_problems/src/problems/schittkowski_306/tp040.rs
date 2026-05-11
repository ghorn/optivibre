use super::helpers::*;
use super::*;

pub(super) fn tp040() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), VecN<SX, 3>, (), _, _>(
        metadata(
            "schittkowski_tp040",
            "tp040",
            "Schittkowski TP040 product objective with three nonlinear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), VecN<SX, 3>, (), _>(
                "schittkowski_tp040",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: -x0 * x1 * x2 * x3,
                        equalities: VecN {
                            values: [
                                x0.powf(3.0) + x1.sqr() - 1.0,
                                x0.sqr() * x3 - x2,
                                x3.sqr() - x1,
                            ],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [0.8; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(-0.25, OBJECTIVE_TOL, PRIMAL_TOL, DUAL_TOL, None),
    )
}
