use super::{helpers::*, *};

pub(super) fn tp254() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp254",
        "tp254",
        "Schittkowski TP254",
        [1.0, 1.0, 1.0],
        [None, None, Some(1.0)],
        [None, None, None],
        -3.0_f64.sqrt(),
        |x| {
            let [x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: x3.log10() - x2,
                equalities: VecN {
                    values: [x2.sqr() + x3.sqr() - 4.0, x3 - 1.0 - x1.sqr()],
                },
                inequalities: (),
            }
        },
    )
}
