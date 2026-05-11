use super::{helpers::*, *};

pub(super) fn tp217() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp217",
        "tp217",
        "Schittkowski TP217",
        [10.0, 10.0],
        [Some(0.0), None],
        [None, None],
        -0.8,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: -x2,
                equalities: VecN {
                    values: [x1.sqr() + x2.sqr() - 1.0],
                },
                inequalities: VecN {
                    values: [-(1.0 + x1 - 2.0 * x2)],
                },
            }
        },
    )
}
