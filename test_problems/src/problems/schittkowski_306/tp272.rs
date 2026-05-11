use super::{helpers::*, *};

pub(super) fn tp272() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp272",
        "tp272",
        "Schittkowski TP272",
        [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        [Some(0.0); 6],
        [None; 6],
        0.0,
        |x| {
            let [x1, x2, x3, x4, x5, x6] = x.values;
            let mut objective = SX::zero();
            for i in 1..=13 {
                let h = 0.1 * i as f64;
                let f = x4 * (-x1 * h).exp() - x5 * (-x2 * h).exp() + x6 * (-x3 * h).exp()
                    - (-h).exp()
                    + 5.0 * (-10.0 * h).exp()
                    - 3.0 * (-4.0 * h).exp();
                objective += f.sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
