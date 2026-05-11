use super::{helpers::*, *};

pub(super) fn tp202_value(x1: SX, x2: SX) -> SX {
    let f1 = -13.0 + x1 - 2.0 * x2 + 5.0 * x2.sqr() - x2.powi(3);
    let f2 = -29.0 + x1 - 14.0 * x2 + x2.sqr() + x2.powi(3);
    f1.sqr() + f2.sqr()
}

pub(super) fn tp202() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp202",
        "tp202",
        "Schittkowski TP202",
        [15.0, -2.0],
        [Some(1.0), Some(-5.0)],
        [Some(20.0), Some(5.0)],
        0.0,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp202_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
