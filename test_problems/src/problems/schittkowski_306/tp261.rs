use super::{helpers::*, *};

pub(super) fn tp261() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp261",
        "tp261",
        "Schittkowski TP261",
        [0.0; 4],
        [Some(0.0); 4],
        [Some(10.0); 4],
        0.0,
        |x| {
            let [x1, x2, x3, x4] = x.values;
            let f1 = (x1.exp() - x2).sqr();
            let f2 = 10.0 * (x2 - x3).powi(3);
            let f3 = (x3 - x4).tan().sqr();
            let f4 = x1.powi(4);
            let f5 = x4 - 1.0;
            SymbolicNlpOutputs {
                objective: f1.sqr() + f2.sqr() + f3.sqr() + f4.sqr() + f5.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
