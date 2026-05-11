use super::{helpers::*, *};

pub(super) fn tp214_value(x1: SX, x2: SX) -> SX {
    (10.0 * (x1 - x2).sqr() + (x1 - 1.0).sqr()).powf(0.25)
}

pub(super) fn tp214() -> ProblemCase {
    nonsmooth_objective_only_case_no_ineq(
        "schittkowski_tp214",
        "tp214",
        "Schittkowski TP214",
        [-1.2, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp214_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
