use super::helpers::*;
use super::*;

pub(super) fn tp065() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp065",
            "tp065",
            "Schittkowski TP065 least-distance problem with one spherical inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp065",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: (x0 - x1).sqr()
                            + ((x0 + x1 - 10.0) / 3.0).sqr()
                            + (x2 - 5.0).sqr(),
                        equalities: (),
                        inequalities: x0.sqr() + x1.sqr() + x2.sqr() - 48.0,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [-5.0, 5.0, 0.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(-4.5), Some(-4.5), Some(-5.0)],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(4.5), Some(4.5), Some(5.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        objective_validation(
            0.953528856757,
            1e-8,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
