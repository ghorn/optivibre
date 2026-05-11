use super::{helpers::*, *};

pub(super) fn tp209_value(x1: SX, x2: SX) -> SX {
    10_000.0 * (x2 - x1.sqr()).sqr() + (1.0 - x1).sqr()
}

pub(super) fn tp209() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp209",
        "tp209",
        "Schittkowski TP209",
        [-1.2, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp209_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
