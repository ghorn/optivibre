use super::{helpers::*, *};

pub(super) fn tp246() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp246",
        "tp246",
        "Schittkowski TP246",
        [-1.2, 2.0, 0.0],
        [None, None, None],
        [None, None, None],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            let f1 = 10.0 * (x3 - ((x1 + x2) / 2.0).sqr());
            let f2 = 1.0 - x1;
            let f3 = 1.0 - x2;
            SymbolicNlpOutputs {
                objective: f1.sqr() + f2.sqr() + f3.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
