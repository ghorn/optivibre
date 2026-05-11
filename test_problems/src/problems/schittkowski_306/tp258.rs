use super::{helpers::*, *};

pub(super) fn tp258() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp258",
        "tp258",
        "Schittkowski TP258",
        [-3.0, -1.0, -3.0, -1.0],
        [None; 4],
        [None; 4],
        0.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: 100.0 * (x2 - x1.sqr()).sqr()
                    + (1.0 - x1).sqr()
                    + 90.0 * (x4 - x3.sqr()).sqr()
                    + (1.0 - x3).sqr()
                    + 10.1 * ((x2 - 1.0).sqr() + (x4 - 1.0).sqr())
                    + 19.8 * (x2 - 1.0) * (x4 - 1.0),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
