use super::{helpers::*, *};

pub(super) fn tp276() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp276",
        "tp276",
        "Schittkowski TP276",
        [-4.0, -2.0, -4.0 / 3.0, -1.0, -0.8, -4.0 / 6.0],
        [None; 6],
        [None; 6],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..6 {
                for j in 0..6 {
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
