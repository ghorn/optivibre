use super::{helpers::*, *};

pub(super) fn tp220() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp220",
        "tp220",
        "Schittkowski TP220",
        [25_000.0, 25_000.0],
        [Some(1.0), Some(0.0)],
        [None, None],
        1.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: x1,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-(x1 - 1.0).powi(3) + x2],
                },
            }
        },
    )
}
