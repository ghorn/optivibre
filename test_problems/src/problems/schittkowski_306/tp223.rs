use super::{helpers::*, *};

pub(super) fn tp223() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp223",
        "tp223",
        "Schittkowski TP223",
        [0.1, 3.3],
        [Some(0.0), Some(0.0)],
        [Some(1.0), Some(10.0)],
        -10.0_f64.ln().ln(),
        |x| {
            let [x1, x2] = x.values;
            let exp_exp = x1.exp().exp();
            SymbolicNlpOutputs {
                objective: -x1,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-exp_exp, exp_exp - x2],
                },
            }
        },
    )
}
