use super::{helpers::*, *};

pub(super) fn tp265() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp265",
        "tp265",
        "Schittkowski TP265",
        [0.0; 4],
        [Some(0.0); 4],
        [None; 4],
        0.97474658,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: 2.0
                    - (-10.0 * x1 * (-x3).exp()).exp()
                    - (-10.0 * x2 * (-x4).exp()).exp(),
                equalities: VecN {
                    values: [x1 + x2 - 1.0, x3 + x4 - 1.0],
                },
                inequalities: (),
            }
        },
    )
}
