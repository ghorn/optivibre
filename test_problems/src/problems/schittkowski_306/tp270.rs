use super::{helpers::*, *};

pub(super) fn tp270() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp270",
        "tp270",
        "Schittkowski TP270",
        [1.1, 2.1, 3.1, 4.1, -1.0],
        [Some(1.0), Some(2.0), Some(3.0), Some(4.0), None],
        [None; 5],
        -1.0,
        |x| {
            let [x1, x2, x3, x4, x5] = x.values;
            SymbolicNlpOutputs {
                objective: x1 * x2 * x3 * x4 - 3.0 * x1 * x2 * x4 - 4.0 * x1 * x2 * x3
                    + 12.0 * x1 * x2
                    - x2 * x3 * x4
                    + 3.0 * x2 * x4
                    + 4.0 * x2 * x3
                    - 12.0 * x2
                    - 2.0 * x1 * x3 * x4
                    + 6.0 * x1 * x4
                    + 8.0 * x1 * x3
                    - 24.0 * x1
                    + 2.0 * x3 * x4
                    - 6.0 * x4
                    - 8.0 * x3
                    + 24.0
                    + 1.5 * x5.powi(4)
                    - 5.75 * x5.powi(3)
                    + 5.25 * x5.sqr(),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [x1.sqr() + x2.sqr() + x3.sqr() + x4.sqr() + x5.sqr() - 34.0],
                },
            }
        },
    )
}
