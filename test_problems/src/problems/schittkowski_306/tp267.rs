use super::{helpers::*, *};

pub(super) fn tp267() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp267",
        "tp267",
        "Schittkowski TP267",
        [2.0; 5],
        [Some(0.0), Some(0.0), None, None, Some(0.0)],
        [Some(15.0); 5],
        0.0,
        |x| {
            let [x1, x2, x3, x4, x5] = x.values;
            let mut objective = SX::zero();
            for i in 1..=11 {
                let h = 0.1 * i as f64;
                let f = x3 * (-x1 * h).exp() - x4 * (-x2 * h).exp() + 3.0 * (-x5 * h).exp()
                    - ((-h).exp() - 5.0 * (-10.0 * h).exp() + 3.0 * (-4.0 * h).exp());
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
