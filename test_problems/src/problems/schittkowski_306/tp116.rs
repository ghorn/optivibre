use super::{helpers::*, *};

pub(super) fn tp116() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp116",
        "tp116",
        "Schittkowski TP116",
        [
            0.5, 0.8, 0.9, 0.1, 0.14, 0.5, 489.0, 80.0, 650.0, 450.0, 150.0, 150.0, 150.0,
        ],
        [
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(1.0e-4),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(500.0),
            Some(0.1),
            Some(1.0),
            Some(1.0e-4),
            Some(1.0e-4),
        ],
        [
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(0.1),
            Some(0.9),
            Some(0.9),
            Some(1000.0),
            Some(1000.0),
            Some(1000.0),
            Some(500.0),
            Some(150.0),
            Some(150.0),
            Some(150.0),
        ],
        97.5884089805,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13] = x.values;
            let g = [
                x3 - x2,
                x2 - x1,
                1.0 - 2.0e-3 * (x7 - x8),
                x11 + x12 + x13 - 50.0,
                250.0 - x11 - x12 - x13,
                x13 - 1.262626 * x10 + 1.231059 * x3 * x10,
                x5 - 0.03475 * x2 - 0.975 * x2 * x5 + 9.75e-3 * x2.sqr(),
                x6 - 0.03475 * x3 - 0.975 * x3 * x6 + 9.75e-3 * x3.sqr(),
                x5 * x7 - x1 * x8 - x4 * x7 + x4 * x8,
                -2.0e-3 * (x2 * x9 + x5 * x8 - x1 * x8 - x6 * x9) - x6 - x5 + 1.0,
                x2 * x9 - x3 * x10 - x6 * x9 - 500.0 * (x2 - x6) + x2 * x10,
                x2 - 0.9 - 2.0e-3 * (x2 * x10 - x3 * x10),
                x4 - 0.03475 * x1 - 0.975 * x1 * x4 + 9.75e-3 * x1.sqr(),
                x11 - 1.262626 * x8 + 1.231059 * x1 * x8,
                x12 - 1.262626 * x9 + 1.231059 * x2 * x9,
            ];
            SymbolicNlpOutputs {
                objective: x11 + x12 + x13,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: g.map(|gi| -gi),
                },
            }
        },
    )
}
