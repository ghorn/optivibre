use super::{helpers::*, *};

pub(super) fn tp234() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp234",
        "tp234",
        "Schittkowski TP234",
        [1.0, 1.0],
        [Some(0.2), Some(0.2)],
        [Some(2.0), Some(2.0)],
        -0.8,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: (x2 - x1).powi(4) - (1.0 - x1),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [x1.sqr() + x2.sqr() - 1.0],
                },
            }
        },
    )
}
