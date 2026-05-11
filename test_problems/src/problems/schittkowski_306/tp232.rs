use super::{helpers::*, *};

pub(super) fn tp232() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp232",
        "tp232",
        "Schittkowski TP232",
        [2.0, 0.5],
        [Some(0.0), Some(0.0)],
        [None, None],
        -1.0,
        |x| {
            let [x1, x2] = x.values;
            let sqrt3 = 3.0_f64.sqrt();
            SymbolicNlpOutputs {
                objective: -((9.0 - (x1 - 3.0).sqr()) * x2.powi(3) / (27.0 * sqrt3)),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-x1 / sqrt3 + x2, -x1 - sqrt3 * x2, -6.0 + x1 + sqrt3 * x2],
                },
            }
        },
    )
}
