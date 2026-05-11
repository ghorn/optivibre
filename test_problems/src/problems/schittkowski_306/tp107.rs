use super::helpers::*;
use super::*;

pub(super) fn tp107() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp107",
        "tp107",
        "Schittkowski TP107 nine-variable AC network problem",
        [0.8, 0.8, 0.2, 0.2, 1.0454, 1.0454, 1.0454, 0.0, 0.0],
        [
            Some(0.0),
            Some(0.0),
            None,
            None,
            Some(0.90909),
            Some(0.90909),
            Some(0.90909),
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            Some(1.0909),
            Some(1.0909),
            Some(1.0909),
            None,
            None,
        ],
        5055.01180339,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9] = x.values;
            let v = 48.4 / 50.176;
            let c = v * 0.25_f64.sin();
            let d = v * 0.25_f64.cos();
            let y1 = x8.sin();
            let y2 = x8.cos();
            let y3 = x9.sin();
            let y4 = x9.cos();
            let y5 = (x8 - x9).sin();
            let y6 = (x8 - x9).cos();
            SymbolicNlpOutputs {
                objective: 3000.0 * x1
                    + 1000.0 * x1.powf(3.0)
                    + 2000.0 * x2
                    + 666.66666667 * x2.powf(3.0),
                equalities: VecN {
                    values: [
                        0.4 - x1
                            + 2.0 * c * x5.powf(2.0)
                            + x5 * x6 * (-d * y1 - c * y2)
                            + x5 * x7 * (-d * y3 - c * y4),
                        0.4 - x2
                            + 2.0 * c * x6.powf(2.0)
                            + x5 * x6 * (d * y1 - c * y2)
                            + x6 * x7 * (d * y5 - c * y6),
                        0.8 + 2.0 * c * x7.powf(2.0)
                            + x5 * x7 * (d * y3 - c * y4)
                            + x6 * x7 * (-d * y5 - c * y6),
                        0.2 - x3 + 2.0 * d * x5.powf(2.0)
                            - x5 * x6 * (-c * y1 + d * y2)
                            - x5 * x7 * (-c * y3 + d * y4),
                        0.2 - x4 + 2.0 * d * x6.powf(2.0)
                            - x5 * x6 * (c * y1 + d * y2)
                            - x6 * x7 * (c * y5 + d * y6),
                        -0.337 + 2.0 * d * x7.powf(2.0)
                            - x5 * x7 * (c * y3 + d * y4)
                            - x6 * x7 * (-c * y5 + d * y6),
                    ],
                },
                inequalities: (),
            }
        },
    )
}
