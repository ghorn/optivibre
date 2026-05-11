use super::{helpers::*, *};

pub(super) fn tp225() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp225",
        "tp225",
        "Schittkowski TP225",
        [3.0, 1.0],
        [None, None],
        [None, None],
        2.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: x1.sqr() + x2.sqr(),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(x1 + x2 - 1.0),
                        -(x1.sqr() + x2.sqr() - 1.0),
                        -(9.0 * x1.sqr() + x2.sqr() - 9.0),
                        -x1.sqr() + x2,
                        -x2.sqr() + x1,
                    ],
                },
            }
        },
    )
}
