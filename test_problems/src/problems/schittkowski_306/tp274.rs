use super::{helpers::*, *};

pub(super) fn tp274() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp274",
        "tp274",
        "Schittkowski TP274",
        [-4.0, -2.0],
        [None; 2],
        [None; 2],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..2 {
                for j in 0..2 {
                    objective += x.values[i] * x.values[j] / ((i + 1 + j + 1 - 1) as f64);
                }
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
