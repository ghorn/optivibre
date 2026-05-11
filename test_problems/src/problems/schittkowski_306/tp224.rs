use super::{helpers::*, *};

pub(super) fn tp224() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp224",
        "tp224",
        "Schittkowski TP224",
        [0.1, 0.1],
        [Some(0.0), Some(0.0)],
        [Some(6.0), Some(6.0)],
        -304.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: 2.0 * x1.sqr() + x2.sqr() - 48.0 * x1 - 40.0 * x2,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(x1 + 3.0 * x2),
                        -18.0 + x1 + 3.0 * x2,
                        -(x1 + x2),
                        -8.0 + x1 + x2,
                    ],
                },
            }
        },
    )
}
