use super::helpers::*;
use super::*;

pub(super) fn tp056() -> ProblemCase {
    make_typed_case::<VecN<SX, 7>, (), VecN<SX, 4>, (), _, _>(
        metadata(
            "schittkowski_tp056",
            "tp056",
            "Schittkowski TP056 product objective with four sinusoidal equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 7>, (), VecN<SX, 4>, (), _>(
                "schittkowski_tp056",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let x4 = x.values[4];
                    let x5 = x.values[5];
                    let x6 = x.values[6];
                    SymbolicNlpOutputs {
                        objective: -x0 * x1 * x2,
                        equalities: VecN {
                            values: [
                                x0 - 4.2 * x3.sin().sqr(),
                                x1 - 4.2 * x4.sin().sqr(),
                                x2 - 4.2 * x5.sin().sqr(),
                                x0 + 2.0 * x1 + 2.0 * x2 - 7.2 * x6.sin().sqr(),
                            ],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            let angle = (1.0_f64 / 4.2).sqrt().asin();
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [
                        1.0,
                        1.0,
                        1.0,
                        angle,
                        angle,
                        angle,
                        (5.0_f64 / 7.2).sqrt().asin(),
                    ],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[
                2.4,
                1.2,
                1.2,
                (4.0_f64 / 7.0).sqrt().asin(),
                (2.0_f64 / 7.0).sqrt().asin(),
                (2.0_f64 / 7.0).sqrt().asin(),
                std::f64::consts::FRAC_PI_2,
            ],
            X_TOL,
            -3.456,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
