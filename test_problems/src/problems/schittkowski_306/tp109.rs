use super::helpers::*;
use super::*;

pub(super) fn tp109() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp109",
        "tp109",
        "Schittkowski TP109 nine-variable power-flow design problem",
        [0.0, 0.0, 0.0, 0.0, 250.0, 250.0, 200.0, 0.0, 0.0],
        [
            Some(0.0),
            Some(0.0),
            Some(-0.55),
            Some(-0.55),
            Some(196.0),
            Some(196.0),
            Some(196.0),
            Some(-400.0),
            Some(-400.0),
        ],
        [
            None,
            None,
            Some(0.55),
            Some(0.55),
            Some(252.0),
            Some(252.0),
            Some(252.0),
            Some(800.0),
            Some(800.0),
        ],
        5362.06927538,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9] = x.values;
            let a = 50.176;
            let ra = 1.0 / a;
            let b = 0.25_f64.sin();
            let c = 0.25_f64.cos();
            SymbolicNlpOutputs {
                objective: 3.0 * x1 + 1e-6 * x1.powf(3.0) + 0.522074e-6 * x2.powf(3.0) + 2.0 * x2,
                equalities: VecN {
                    values: [
                        x4 - x3 + 0.55,
                        x3 - x4 + 0.55,
                        2.25e6 - x1.powf(2.0) - x8.powf(2.0),
                        2.25e6 - x2.powf(2.0) - x9.powf(2.0),
                        (x5 * x6 * (-x3 - 0.25).sin()
                            + x5 * x7 * (-x4 - 0.25).sin()
                            + 2.0 * x5.powf(2.0) * b)
                            * ra
                            + 400.0
                            - x1,
                        (x5 * x6 * (x3 - 0.25).sin()
                            + x6 * x7 * (x3 - x4 - 0.25).sin()
                            + 2.0 * x6.powf(2.0) * b)
                            * ra
                            + 400.0
                            - x2,
                        (x5 * x7 * (x4 - 0.25).sin()
                            + x6 * x7 * (x4 - x3 - 0.25).sin()
                            + 2.0 * x7.powf(2.0) * b)
                            * ra
                            + 881.779,
                        x8 + (x5 * x6 * (-x3 - 0.25).cos() + x5 * x7 * (-x4 - 0.25).cos()
                            - 2.0 * x5.powf(2.0) * c)
                            * ra
                            + 0.0007533 * x5.powf(2.0)
                            - 200.0,
                        x9 + (x5 * x6 * (x3 - 0.25).cos() + x7 * x6 * (x3 - x4 - 0.25).cos()
                            - 2.0 * x6.powf(2.0) * c)
                            * ra
                            + 0.0007533 * x6.powf(2.0)
                            - 200.0,
                        (x5 * x7 * (x4 - 0.25).cos() + x6 * x7 * (x4 - x3 - 0.25).cos()
                            - 2.0 * x7.powf(2.0) * c)
                            * ra
                            + 0.0007533 * x7.powf(2.0)
                            - 22.938,
                    ],
                },
                inequalities: (),
            }
        },
    )
}
