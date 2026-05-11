use super::{helpers::*, *};

pub(super) fn tp256() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp256",
        "tp256",
        "Schittkowski TP256",
        [3.0, -1.0, 0.0, 1.0],
        [None; 4],
        [None; 4],
        0.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: (x1 + 10.0 * x2).sqr()
                    + 5.0 * (x3 - x4).sqr()
                    + (x2 - 2.0 * x3).powi(4)
                    + 10.0 * (x1 - x4).powi(4),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
