use super::{helpers::*, *};

pub(super) fn tp206_value(x1: SX, x2: SX) -> SX {
    (x2 - x1.sqr()).sqr() + 100.0 * (1.0 - x1).sqr()
}

pub(super) fn tp206() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp206",
        "tp206",
        "Schittkowski TP206",
        [-1.2, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp206_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
