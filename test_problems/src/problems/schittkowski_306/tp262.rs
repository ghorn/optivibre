use super::{helpers::*, *};

pub(super) fn tp262() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp262",
        "tp262",
        "Schittkowski TP262",
        [1.0; 4],
        [Some(0.0); 4],
        [None; 4],
        -10.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: -0.5 * x1 - x2 - 0.5 * x3 - x4,
                equalities: VecN {
                    values: [x1 + x2 + x3 - 2.0 * x4 - 6.0],
                },
                inequalities: VecN {
                    values: [
                        -10.0 + x1 + x2 + x3 + x4,
                        -10.0 + 0.2 * x1 + 0.5 * x2 + x3 + 2.0 * x4,
                        -10.0 + 2.0 * x1 + x2 + 0.5 * x3 + 0.2 * x4,
                    ],
                },
            }
        },
    )
}
