use super::helpers::*;
use super::*;

pub(super) fn tp066() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp066",
            "tp066",
            "Schittkowski TP066 linear objective with two exponential inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp066",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: 0.2 * x2 - 0.8 * x0,
                        equalities: (),
                        inequalities: VecN {
                            values: [x0.exp() - x1, x1.exp() - x2],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [0.0, 1.05, 2.9],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(100.0), Some(100.0), Some(10.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<2>(),
                    scaling: None,
                },
            })
        },
        objective_validation(
            0.518163274159,
            1e-8,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
