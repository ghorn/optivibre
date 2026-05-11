use super::{helpers::*, *};

pub(super) fn tp250() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp250",
        "tp250",
        "Schittkowski TP250",
        [10.0, 10.0, 10.0],
        [Some(0.0), Some(0.0), Some(0.0)],
        [Some(20.0), Some(11.0), Some(42.0)],
        -3300.0,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: -x1 * x2 * x3,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(x1 + 2.0 * x2 + 2.0 * x3),
                        -72.0 + x1 + 2.0 * x2 + 2.0 * x3,
                    ],
                },
            }
        },
    )
}
