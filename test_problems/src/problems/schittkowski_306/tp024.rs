use super::helpers::*;
use super::*;

pub(super) fn tp024() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), VecN<SX, 3>, _, _>(
        metadata(
            "schittkowski_tp024",
            "tp024",
            "Schittkowski TP024 cubic objective with triangular linear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), VecN<SX, 3>, _>(
                "schittkowski_tp024",
                |x, ()| {
                    let a = 3.0_f64.sqrt();
                    SymbolicNlpOutputs {
                        objective: ((x.x - 3.0).sqr() - 9.0) * x.y.powi(3) / (27.0 * a),
                        equalities: (),
                        inequalities: VecN {
                            values: [x.y - x.x / a, -x.x - a * x.y, x.x + a * x.y - 6.0],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 1.0, y: 0.5 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(0.0),
                        y: Some(0.0),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(VecN {
                        values: [Some(0.0); 3],
                    }),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[3.0, 3.0_f64.sqrt()],
            X_TOL,
            -1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
