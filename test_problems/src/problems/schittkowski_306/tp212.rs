use super::{helpers::*, *};

pub(super) fn tp212_value(x1: SX, x2: SX) -> SX {
    let f1 = 4.0 * (x1 + x2);
    let f2 = 4.0 * (x1 + x2) + (x1 - x2) * ((x1 - 2.0).sqr() + x2.sqr() - 1.0);
    f1.sqr() + f2.sqr()
}

pub(super) fn tp212() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp212",
        "tp212",
        "Schittkowski TP212",
        [2.0, 0.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp212_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
