use super::{helpers::*, *};

pub(super) fn tp228() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp228",
        "tp228",
        "Schittkowski TP228",
        [0.0, 0.0],
        [None, None],
        [None, None],
        -3.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: x1.sqr() + x2,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [x1 + x2 - 1.0, x1.sqr() + x2.sqr() - 9.0],
                },
            }
        },
    )
}
