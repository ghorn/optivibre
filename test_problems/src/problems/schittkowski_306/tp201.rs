use super::{helpers::*, *};

pub(super) fn tp201_value(x1: SX, x2: SX) -> SX {
    4.0 * (x1 - 5.0).sqr() + (x2 - 6.0).sqr()
}

pub(super) fn tp201() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp201",
        "tp201",
        "Schittkowski TP201",
        [8.0, 9.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp201_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
