use super::{helpers::*, *};

pub(super) fn tp283() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp283",
        "tp283",
        "Schittkowski TP283",
        [0.0; 10],
        [None; 10],
        [None; 10],
        0.0,
        |x| {
            let mut h = SX::zero();
            for i in 0..10 {
                let k = (i + 1) as f64;
                h += k.powi(3) * (x.values[i] - 1.0).sqr();
            }
            SymbolicNlpOutputs {
                objective: h.powi(3),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
