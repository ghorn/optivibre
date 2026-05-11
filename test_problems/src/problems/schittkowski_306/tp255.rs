use super::{helpers::*, *};

pub(super) fn tp255() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp255",
        "tp255",
        "Schittkowski TP255",
        [-3.0, 1.0, -3.0, 1.0],
        [Some(-10.0); 4],
        [Some(10.0); 4],
        0.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            let f = 100.0 * (x2 - x1.sqr())
                + (1.0 - x1).sqr()
                + 90.0 * (x4 - x3.sqr())
                + (1.0 - x3).sqr()
                + 10.1 * ((x2 - 1.0).sqr() + (x4 - 1.0).sqr())
                + 19.8 * (x2 - 1.0) * (x4 - 1.0);
            SymbolicNlpOutputs {
                objective: 0.5 * f.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
