use super::{helpers::*, *};

pub(super) fn tp245() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp245",
        "tp245",
        "Schittkowski TP245",
        [0.0, 10.0, 20.0],
        [Some(0.0), Some(0.0), Some(0.0)],
        [Some(12.0), Some(12.0), Some(20.0)],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            let mut objective = SX::zero();
            for i in 1..=10 {
                let di = i as f64;
                let f = (-di * x1 / 10.0).exp()
                    - (-di * x2 / 10.0).exp()
                    - x3 * ((-di / 10.0).exp() - (-di).exp());
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
