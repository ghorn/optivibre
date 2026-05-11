use super::{helpers::*, *};

pub(super) fn tp241() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp241",
        "tp241",
        "Schittkowski TP241",
        [1.0, 2.0, 0.0],
        [None, None, None],
        [None, None, None],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            let f1 = x1.sqr() + x2.sqr() + x3.sqr() - 1.0;
            let f2 = x1.sqr() + x2.sqr() + (x3 - 2.0).sqr() - 1.0;
            let f3 = x1 + x2 + x3 - 1.0;
            let f4 = x1 + x2 - x3 + 1.0;
            let f5 = x1.powi(3) + 3.0 * x2.sqr() + (5.0 * x3 - x1 + 1.0).sqr() - 36.0;
            SymbolicNlpOutputs {
                objective: f1.sqr() + f2.sqr() + f3.sqr() + f4.sqr() + f5.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
