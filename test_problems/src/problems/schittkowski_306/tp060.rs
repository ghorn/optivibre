use super::helpers::*;
use super::*;

pub(super) fn tp060() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp060",
            "tp060",
            "Schittkowski TP060 bounded three-variable problem with one nonlinear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), SX, (), _>(
                "schittkowski_tp060",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: (x0 - 1.0).sqr() + (x0 - x1).sqr() + (x1 - x2).powi(4),
                        equalities: x0 * (1.0 + x1.sqr()) + x2.powi(4) - 4.0 - 3.0 * 2.0_f64.sqrt(),
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
                        values: [Some(-10.0); 3],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(10.0); 3],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        objective_validation(0.0325682002513, 1e-8, PRIMAL_TOL, DUAL_TOL, None),
    )
}
