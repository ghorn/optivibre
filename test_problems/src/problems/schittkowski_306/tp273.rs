use super::{helpers::*, *};

pub(super) fn tp273() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp273",
        "tp273",
        "Schittkowski TP273",
        [0.0; 6],
        [None; 6],
        [None; 6],
        0.0,
        |x| {
            let mut h = SX::zero();
            for i in 0..6 {
                h += (16 - (i + 1)) as f64 * (x.values[i] - 1.0).sqr();
            }
            SymbolicNlpOutputs {
                objective: 10.0 * h * (1.0 + h),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
