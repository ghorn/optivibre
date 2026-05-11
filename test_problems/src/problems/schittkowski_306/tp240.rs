use super::{helpers::*, *};

pub(super) fn tp240() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp240",
        "tp240",
        "Schittkowski TP240",
        [100.0, -1.0, 2.5],
        [None, None, None],
        [None, None, None],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: (x1 - x2 + x3).sqr() + (-x1 + x2 + x3).sqr() + (x1 + x2 - x3).sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
