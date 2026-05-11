use super::{helpers::*, *};

pub(super) fn tp229() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp229",
        "tp229",
        "Schittkowski TP229",
        [-1.2, 1.0],
        [Some(-2.0), Some(-2.0)],
        [Some(2.0), Some(2.0)],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: 100.0 * (x2 - x1.sqr()).sqr() + (1.0 - x1).sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
