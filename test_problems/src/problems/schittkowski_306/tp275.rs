use super::{helpers::*, *};

pub(super) fn tp275() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp275",
        "tp275",
        "Schittkowski TP275",
        [-4.0, -2.0, -4.0 / 3.0, -1.0],
        [None; 4],
        [None; 4],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..4 {
                for j in 0..4 {
                    objective += x.values[i] * x.values[j] / ((i + j + 1) as f64);
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
