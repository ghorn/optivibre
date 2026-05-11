use super::{helpers::*, *};

pub(super) fn tp260() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp260",
        "tp260",
        "Schittkowski TP260",
        [-3.0, -1.0, -3.0, -1.0],
        [None; 4],
        [None; 4],
        0.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            let f1 = 10.0 * (x2 - x1.sqr());
            let f2 = 1.0 - x1;
            let f3 = 90.0_f64.sqrt() * (x4 - x3.sqr());
            let f4 = 1.0 - x3;
            let f5 = 9.9_f64.sqrt() * ((x2 - 1.0) + (x4 - 1.0));
            let f6 = 0.2_f64.sqrt() * (x2 - 1.0);
            let f7 = 0.2_f64.sqrt() * (x4 - 1.0);
            SymbolicNlpOutputs {
                objective: f1.sqr()
                    + f2.sqr()
                    + f3.sqr()
                    + f4.sqr()
                    + f5.sqr()
                    + f6.sqr()
                    + f7.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
