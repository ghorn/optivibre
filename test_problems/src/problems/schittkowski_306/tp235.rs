use super::{helpers::*, *};

pub(super) fn tp235() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp235",
        "tp235",
        "Schittkowski TP235",
        [-2.0, 3.0, 1.0],
        [None, None, None],
        [None, None, None],
        0.04,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: (x2 - x1.sqr()).sqr() + 0.01 * (x1 - 1.0).sqr(),
                equalities: VecN {
                    values: [x1 + x3.sqr() + 1.0],
                },
                inequalities: (),
            }
        },
    )
}
