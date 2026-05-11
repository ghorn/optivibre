use super::{helpers::*, *};

pub(super) fn tp213_value(x1: SX, x2: SX) -> SX {
    (10.0 * (x1 - x2).sqr() + (x1 - 1.0).sqr()).powi(4)
}

pub(super) fn tp213() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp213",
        "tp213",
        "Schittkowski TP213",
        [3.0, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp213_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
