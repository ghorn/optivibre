use super::{helpers::*, *};

pub(super) fn tp230() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp230",
        "tp230",
        "Schittkowski TP230",
        [0.0, 0.0],
        [None, None],
        [None, None],
        0.375,
        |x| {
            let [x1, x2] = x.values;
            let one_minus_x1 = 1.0 - x1;
            SymbolicNlpOutputs {
                objective: x2,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        2.0 * x1.sqr() - x1.powi(3) - x2,
                        2.0 * one_minus_x1.sqr() - one_minus_x1.powi(3) - x2,
                    ],
                },
            }
        },
    )
}
