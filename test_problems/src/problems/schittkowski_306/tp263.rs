use super::{helpers::*, *};

pub(super) fn tp263() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp263",
        "tp263",
        "Schittkowski TP263",
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
                inequalities: VecN {
                    values: [-(x2 - x1.powi(3)), -(x1.sqr() - x2)],
                },
            }
        },
    )
}
