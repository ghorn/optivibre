use super::{helpers::*, *};

pub(super) fn tp216() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp216",
        "tp216",
        "Schittkowski TP216",
        [-1.2, 1.0],
        [Some(-3.0), Some(-3.0)],
        [Some(10.0), Some(10.0)],
        1.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: 100.0 * (x1.sqr() - x2).sqr() + (x1 - 1.0).sqr(),
                equalities: VecN {
                    values: [x1 * (x1 - 4.0) - 2.0 * x2 + 12.0],
                },
                inequalities: (),
            }
        },
    )
}
