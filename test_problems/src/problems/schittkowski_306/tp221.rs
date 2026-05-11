use super::{helpers::*, *};

pub(super) fn tp221() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp221",
        "tp221",
        "Schittkowski TP221",
        [0.25, 0.25],
        [Some(0.0), Some(0.0)],
        [Some(1.0), Some(1.0)],
        -1.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: -x1,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [(x1 - 1.0).powi(3) + x2],
                },
            }
        },
    )
}
