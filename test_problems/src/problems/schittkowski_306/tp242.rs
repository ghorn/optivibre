use super::{helpers::*, *};

pub(super) fn tp242() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp242",
        "tp242",
        "Schittkowski TP242",
        [2.5, 10.0, 10.0],
        [Some(0.0), Some(0.0), Some(0.0)],
        [Some(10.0), Some(10.0), Some(10.0)],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            let mut objective = SX::zero();
            for i in 1..=10 {
                let ti = 0.1 * i as f64;
                let f =
                    (-x1 * ti).exp() - (-x2 * ti).exp() - x3 * ((-ti).exp() - (-10.0 * ti).exp());
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
