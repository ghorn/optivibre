use super::{helpers::*, *};

pub(super) fn tp215() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp215",
        "tp215",
        "Schittkowski TP215",
        [1.0, 1.0],
        [Some(0.0), None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: x2,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [x1.sqr() - x2],
                },
            }
        },
    )
}
