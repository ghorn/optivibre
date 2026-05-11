use super::{helpers::*, *};

pub(super) fn tp113() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp113",
        "tp113",
        "Schittkowski TP113",
        [2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 7.0, 3.0, 6.0, 10.0],
        [None; 10],
        [None; 10],
        24.3062090641,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] = x.values;
            let objective = x1.sqr() + x2.sqr() + x1 * x2 - 14.0 * x1 - 16.0 * x2
                + (x3 - 10.0).sqr()
                + 4.0 * (x4 - 5.0).sqr()
                + (x5 - 3.0).sqr()
                + 2.0 * (x6 - 1.0).sqr()
                + 5.0 * x7.sqr()
                + 7.0 * (x8 - 11.0).sqr()
                + 2.0 * (x9 - 10.0).sqr()
                + (x10 - 7.0).sqr()
                + 45.0;
            let g = [
                -4.0 * x1 - 5.0 * x2 + 3.0 * x7 - 9.0 * x8 + 105.0,
                -10.0 * x1 + 8.0 * x2 + 17.0 * x7 - 2.0 * x8,
                8.0 * x1 - 2.0 * x2 - 5.0 * x9 + 2.0 * x10 + 12.0,
                -3.0 * (x1 - 2.0).sqr() - 4.0 * (x2 - 3.0).sqr() - 2.0 * x3.sqr()
                    + 7.0 * x4
                    + 120.0,
                -5.0 * x1.sqr() - 8.0 * x2 - (x3 - 6.0).sqr() + 2.0 * x4 + 40.0,
                -0.5 * (x1 - 8.0).sqr() - 2.0 * (x2 - 4.0).sqr() - 3.0 * x5.sqr() + x6 + 30.0,
                -x1.sqr() - 2.0 * (x2 - 2.0).sqr() + 2.0 * x1 * x2 - 14.0 * x5 + 6.0 * x6,
                3.0 * x1 - 6.0 * x2 - 12.0 * (x9 - 8.0).sqr() + 7.0 * x10,
            ];
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: g.map(|gi| -gi),
                },
            }
        },
    )
}
