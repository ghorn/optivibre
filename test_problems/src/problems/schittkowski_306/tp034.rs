use super::helpers::*;
use super::*;

pub(super) fn tp034() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), VecN<SX, 2>, _, _>(
        metadata(
            "schittkowski_tp034",
            "tp034",
            "Schittkowski TP034 logarithmic chain objective with two exponential inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), VecN<SX, 2>, _>(
                "schittkowski_tp034",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: -x0,
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
        exact_solution_validation(
            &[10.0_f64.ln().ln(), 10.0_f64.ln(), 10.0],
            X_TOL,
            -10.0_f64.ln().ln(),
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
