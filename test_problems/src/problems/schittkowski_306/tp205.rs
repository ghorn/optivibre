use super::{helpers::*, *};

pub(super) fn tp205_value(x1: SX, x2: SX) -> SX {
    let f1 = 1.5 - x1 * (1.0 - x2);
    let f2 = 2.25 - x1 * (1.0 - x2.sqr());
    let f3 = 2.625 - x1 * (1.0 - x2.powi(3));
    f1.sqr() + f2.sqr() + f3.sqr()
}

pub(super) fn tp205() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp205",
        "tp205",
        "Schittkowski TP205",
        [0.0, 0.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp205_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
