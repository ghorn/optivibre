use super::{helpers::*, *};

pub(super) fn tp247() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp247",
        "tp247",
        "Schittkowski TP247",
        [0.1, 0.0, 0.0],
        [Some(0.1), None, Some(-2.5)],
        [None, None, Some(7.5)],
        0.0,
        |x| {
            let [x1, x2, x3] = x.values;
            let theta = (x2 / x1).atan() / (2.0 * std::f64::consts::PI);
            let radius = (x1.sqr() + x2.sqr()).sqrt();
            SymbolicNlpOutputs {
                objective: 100.0 * ((x3 - 10.0 * theta).sqr() + (radius - 1.0).sqr()) + x3.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
