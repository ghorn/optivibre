use super::{helpers::*, *};

pub(super) fn tp249() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp249",
        "tp249",
        "Schittkowski TP249",
        [1.0, 1.0, 1.0],
        [Some(1.0), None, None],
        [None, None, None],
        1.0,
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: x1.sqr() + x2.sqr() + x3.sqr(),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-x1.sqr() - x2.sqr() + 1.0],
                },
            }
        },
    )
}
