use super::{helpers::*, *};

pub(super) fn tp244() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp244",
        "tp244",
        "Schittkowski TP244",
        [1.0, 2.0, 1.0],
        [Some(0.0), Some(0.0), Some(0.0)],
        [Some(1.0e10), Some(1.0e10), Some(1.0e10)],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            let mut objective = SX::zero();
            for i in 1..=8 {
                let zi = 0.1 * i as f64;
                let yi = (-zi).exp() - 5.0 * (-10.0 * zi).exp();
                let f = (-x1 * zi).exp() - x3 * (-x2 * zi).exp() - yi;
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
