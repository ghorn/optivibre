use super::{helpers::*, *};

pub(super) fn tp114() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp114",
        "tp114",
        "Schittkowski TP114",
        [
            1745.0, 12000.0, 110.0, 3048.0, 1974.0, 89.2, 92.8, 8.0, 3.6, 145.0,
        ],
        [
            Some(1.0e-5),
            Some(1.0e-5),
            Some(1.0e-5),
            Some(1.0e-5),
            Some(1.0e-5),
            Some(85.0),
            Some(90.0),
            Some(3.0),
            Some(1.2),
            Some(145.0),
        ],
        [
            Some(2000.0),
            Some(16000.0),
            Some(120.0),
            Some(5000.0),
            Some(2000.0),
            Some(93.0),
            Some(95.0),
            Some(12.0),
            Some(4.0),
            Some(162.0),
        ],
        -1768.80696344,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] = x.values;
            let objective = 5.04 * x1 + 0.035 * x2 + 10.0 * x3 + 3.36 * x5 - 0.063 * x4 * x7;
            let g = [
                35.82 - 0.222 * x10 - 0.9 * x9,
                -133.0 + 3.0 * x7 - 0.99 * x10,
                -35.82 + 0.222 * x10 + 10.0 * x9 / 9.0,
                133.0 - 3.0 * x7 + x10 / 0.99,
                1.12 * x1 + 0.13167 * x1 * x8 - 0.00667 * x1 * x8.sqr() - 0.99 * x4,
                57.425 + 1.098 * x8 - 0.038 * x8.sqr() + 0.325 * x6 - 0.99 * x7,
                -1.12 * x1 - 0.13167 * x1 * x8 + 0.00667 * x1 * x8.sqr() + x4 / 0.99,
                -57.425 - 1.098 * x8 + 0.038 * x8.sqr() - 0.325 * x6 + x7 / 0.99,
            ];
            SymbolicNlpOutputs {
                objective,
                equalities: VecN {
                    values: [
                        1.22 * x4 - x1 - x5,
                        9.8e4 * x3 / (x4 * x9 + 1.0e3 * x3) - x6,
                        (x2 + x5) / x1 - x8,
                    ],
                },
                inequalities: VecN {
                    values: g.map(|gi| -gi),
                },
            }
        },
    )
}
