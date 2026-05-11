use super::helpers::*;
use super::*;

pub(super) fn tp067() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), VecN<SX, 14>, _, _>(
        metadata(
            "schittkowski_tp067",
            "tp067",
            "Schittkowski TP067 alkylation process model with fourteen nonlinear inequalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), VecN<SX, 14>, _>(
                "schittkowski_tp067",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let y = tp067_symbolic_state(x0, x1, x2);
                    SymbolicNlpOutputs {
                        objective: -(0.063 * y[0] * y[3]
                            - 5.04 * x0
                            - 3.36 * y[1]
                            - 0.035 * x1
                            - 10.0 * x2),
                        equalities: (),
                        inequalities: VecN {
                            values: [
                                -y[0],
                                -y[1],
                                85.0 - y[2],
                                90.0 - y[3],
                                3.0 - y[4],
                                0.01 - y[5],
                                145.0 - y[6],
                                y[0] - 5.0e3,
                                y[1] - 2.0e3,
                                y[2] - 93.0,
                                y[3] - 95.0,
                                y[4] - 12.0,
                                y[5] - 4.0,
                                y[6] - 162.0,
                            ],
                        },
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [1.745e3, 1.2e4, 110.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(1.0e-5); 3],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(2.0e3), Some(1.6e4), Some(120.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: inequality_upper_bounds::<14>(),
                    scaling: None,
                },
            })
        },
        objective_validation(
            -1162.03650728,
            2e-2,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
