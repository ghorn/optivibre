use super::{helpers::*, *};

pub(super) fn tp248() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp248",
        "tp248",
        "Schittkowski TP248",
        [-0.1, -1.0, 0.1],
        [None, None, None],
        [None, None, None],
        -0.8,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: -x2,
                equalities: VecN {
                    values: [x1.sqr() + x2.sqr() + x3.sqr() - 1.0],
                },
                inequalities: VecN {
                    values: [-(1.0 - 2.0 * x2 + x1)],
                },
            }
        },
    )
}
