use super::{helpers::*, *};

pub(super) fn tp257() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp257",
        "tp257",
        "Schittkowski TP257",
        [0.0; 4],
        [Some(0.0); 4],
        [None; 4],
        0.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: 100.0 * (x1.sqr() - x2).sqr()
                    + (x1 - 1.0).sqr()
                    + 90.0 * (x3.sqr() - x4).sqr()
                    + (x3 - 1.0).sqr()
                    + 10.1 * ((x2 - 1.0).sqr() + (x4 - 1.0).sqr())
                    + 19.8 * (x1 - 1.0) * (x4 - 1.0),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
