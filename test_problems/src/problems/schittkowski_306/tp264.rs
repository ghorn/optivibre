use super::{helpers::*, *};

pub(super) fn tp264() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp264",
        "tp264",
        "Schittkowski TP264",
        [0.0; 4],
        [None; 4],
        [None; 4],
        -44.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: x1.sqr() + x2.sqr() + 2.0 * x3.sqr() + x4.sqr()
                    - 5.0 * x1
                    - 5.0 * x2
                    - 21.0 * x3
                    + 7.0 * x4,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -8.0 + x1.sqr() + x2.sqr() + x3.sqr() + x4.sqr() + x1 - x2 + x3 - x4,
                        -9.0 + x1.sqr() + 2.0 * x2.sqr() + x3.sqr() + 2.0 * x4.sqr() - x1 - x4,
                        -5.0 + 2.0 * x1.sqr() + x2.sqr() + x3.sqr() + 2.0 * x1 - x2 - x4,
                    ],
                },
            }
        },
    )
}
