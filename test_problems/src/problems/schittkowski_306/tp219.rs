use super::{helpers::*, *};

pub(super) fn tp219() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp219",
        "tp219",
        "Schittkowski TP219",
        [10.0; 4],
        [None; 4],
        [None; 4],
        -1.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: -x1,
                equalities: VecN {
                    values: [x2 - x1.powi(3) - x3.sqr(), x1.sqr() - x2 - x4.sqr()],
                },
                inequalities: (),
            }
        },
    )
}
