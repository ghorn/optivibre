use super::{helpers::*, *};

pub(super) fn tp222() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp222",
        "tp222",
        "Schittkowski TP222",
        [1.3, 0.2],
        [Some(0.0), Some(0.0)],
        [None, None],
        -1.5,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: -x1,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-0.125 + (x1 - 1.0).powi(3) + x2],
                },
            }
        },
    )
}
