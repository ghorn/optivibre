use super::{helpers::*, *};

pub(super) fn tp252() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp252",
        "tp252",
        "Schittkowski TP252",
        [-1.0, 2.0, 2.0],
        [None, None, None],
        [Some(-1.0), None, None],
        0.04,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: 0.01 * (x1 - 1.0).sqr() + (x2 - x1.sqr()).sqr(),
                equalities: VecN {
                    values: [x1 + x3.sqr() + 1.0],
                },
                inequalities: (),
            }
        },
    )
}
