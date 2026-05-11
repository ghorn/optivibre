use super::{helpers::*, *};

pub(super) fn tp208_value(x1: SX, x2: SX) -> SX {
    100.0 * (x2 - x1.sqr()).sqr() + (1.0 - x1).sqr()
}

pub(super) fn tp208() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp208",
        "tp208",
        "Schittkowski TP208",
        [-1.2, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp208_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
