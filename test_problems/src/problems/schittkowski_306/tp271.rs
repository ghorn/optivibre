use super::{helpers::*, *};

pub(super) fn tp271() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp271",
        "tp271",
        "Schittkowski TP271",
        [0.0; 6],
        [None; 6],
        [None; 6],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..6 {
                objective += (10.0 * (16 - (i + 1)) as f64) * (x.values[i] - 1.0).sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
