use super::{helpers::*, *};

pub(super) fn tp269() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp269",
        "tp269",
        "Schittkowski TP269",
        [2.0; 5],
        [None; 5],
        [None; 5],
        176.0 / 43.0,
        |x| {
            let [x1, x2, x3, x4, x5] = x.values;
            let f1 = x1 - x2;
            let f2 = x2 + x3 - 2.0;
            let f3 = x4 - 1.0;
            let f4 = x5 - 1.0;
            SymbolicNlpOutputs {
                objective: f1.sqr() + f2.sqr() + f3.sqr() + f4.sqr(),
                equalities: VecN {
                    values: [x1 + 3.0 * x2, x3 + x4 - 2.0 * x5, x2 - x5],
                },
                inequalities: (),
            }
        },
    )
}
