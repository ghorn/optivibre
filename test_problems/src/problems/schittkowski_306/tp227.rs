use super::{helpers::*, *};

pub(super) fn tp227() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp227",
        "tp227",
        "Schittkowski TP227",
        [0.5, 0.5],
        [None, None],
        [None, None],
        1.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: (x1 - 2.0).sqr() + (x2 - 1.0).sqr(),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [x1.sqr() - x2, -x1 + x2.sqr()],
                },
            }
        },
    )
}
