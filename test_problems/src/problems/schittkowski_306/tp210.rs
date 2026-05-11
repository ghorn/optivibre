use super::{helpers::*, *};

pub(super) fn tp210_value(x1: SX, x2: SX) -> SX {
    let c = 1_000_000.0;
    (c * (x2 - x1.sqr()).sqr() + (1.0 - x1).sqr()) / c
}

pub(super) fn tp210() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp210",
        "tp210",
        "Schittkowski TP210",
        [-1.2, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp210_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
