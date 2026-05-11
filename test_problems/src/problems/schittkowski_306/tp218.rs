use super::{helpers::*, *};

pub(super) fn tp218() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp218",
        "tp218",
        "Schittkowski TP218",
        [9.0, 100.0],
        [None, Some(0.0)],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: x2,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [x1.sqr() - x2],
                },
            }
        },
    )
}
