use super::helpers::*;
use super::*;

pub(super) fn tp054() -> ProblemCase {
    make_typed_case::<VecN<SX, 6>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp054",
            "tp054",
            "Schittkowski TP054 scaled exponential objective with one linear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 6>, (), SX, (), _>(
                "schittkowski_tp054",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    let x5 = x.values[5];
                    let v0 = x0 - 1.0e4;
                    let v1 = x1 - 1.0;
                    let v2 = x2 - 2.0e6;
                    let v3 = x3 - 10.0;
                    let v4 = x4 - 1.0e-3;
                    let v5 = x5 - 1.0e8;
                    let q = (1.5625e-8 * v0.sqr() + 5.0e-5 * v0 * v1 + v1.sqr()) / 0.96
                        + v2.sqr() / 4.9e13
                        + 4.0e-4 * v3.sqr()
                        + 4.0e2 * v4.sqr()
                        + 4.0e-18 * v5.sqr();
                    SymbolicNlpOutputs {
                        objective: -((-0.5 * q).exp()),
                        equalities: x0 + 4.0e3 * x1 - 1.76e4,
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [6.0e3, 1.5, 4.0e6, 2.0, 3.0e-3, 5.0e7],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [
                            Some(0.0),
                            Some(-10.0),
                            Some(0.0),
                            Some(0.0),
                            Some(0.0),
                            Some(0.0),
                        ],
                    }),
                    variable_upper: Some(VecN {
                        values: [
                            Some(2.0e4),
                            Some(10.0),
                            Some(1.0e7),
                            Some(20.0),
                            Some(1.0),
                            Some(2.0e8),
                        ],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[9.16e4 / 7.0, 79.0 / 70.0, 2.0e6, 10.0, 1.0e-3, 1.0e8],
            X_TOL,
            -(-27.0 / 280.0_f64).exp(),
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
