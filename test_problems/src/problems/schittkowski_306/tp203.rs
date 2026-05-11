use super::{helpers::*, *};

pub(super) fn tp203_value(x1: SX, x2: SX) -> SX {
    let c = [1.5, 2.25, 2.625];
    let mut objective = SX::zero();
    for (i, ci) in c.iter().enumerate() {
        let f = *ci - x1 * (1.0 - x2.powi((i + 1) as i32));
        objective += f.sqr();
    }
    objective
}

pub(super) fn tp203() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp203",
        "tp203",
        "Schittkowski TP203",
        [2.0, 0.2],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp203_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
