use super::{helpers::*, *};

pub(super) fn tp233() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp233",
        "tp233",
        "Schittkowski TP233",
        [1.2, 1.0],
        [None, None],
        [None, None],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: 100.0 * (x2 - x1.sqr()).sqr() + (1.0 - x1).sqr(),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-x1.sqr() - x2.sqr() + 0.25],
                },
            }
        },
    )
}
