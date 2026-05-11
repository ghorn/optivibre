use super::{helpers::*, *};

pub(super) fn tp226() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp226",
        "tp226",
        "Schittkowski TP226",
        [0.8, 0.05],
        [Some(0.0), Some(0.0)],
        [None, None],
        -0.5,
        |x| {
            let [x1, x2] = x.values;
            let radius = x1.sqr() + x2.sqr();
            SymbolicNlpOutputs {
                objective: -x1 * x2,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-radius, radius - 1.0],
                },
            }
        },
    )
}
