use super::{helpers::*, *};

pub(super) fn tp251() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp251",
        "tp251",
        "Schittkowski TP251",
        [10.0, 10.0, 10.0],
        [Some(0.0), Some(0.0), Some(0.0)],
        [Some(42.0), Some(42.0), Some(42.0)],
        -3456.0,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: -x1 * x2 * x3,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-72.0 + x1 + 2.0 * x2 + 2.0 * x3],
                },
            }
        },
    )
}
