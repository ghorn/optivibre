use super::helpers::*;
use super::*;

pub(super) fn tp075() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp075",
        "tp075",
        "Schittkowski TP075 four-variable trigonometric network problem",
        [0.0; 4],
        [Some(0.0), Some(0.0), Some(-0.48), Some(-0.48)],
        [Some(1200.0), Some(1200.0), Some(0.48), Some(0.48)],
        5174.41288686,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            let a = 0.48;
            SymbolicNlpOutputs {
                objective: 3.0 * x0 + 1.0e-6 * x0.powi(3) + 2.0 * x1 + 2.0e-6 / 3.0 * x1.powi(3),
                equalities: VecN {
                    values: [
                        1000.0 * ((-x2 - 0.25).sin() + (-x3 - 0.25).sin()) + 894.8 - x0,
                        1000.0 * ((x2 - 0.25).sin() + (x2 - x3 - 0.25).sin()) + 894.8 - x1,
                        1000.0 * ((x3 - 0.25).sin() + (x3 - x2 - 0.25).sin()) - 1294.8,
                    ],
                },
                inequalities: VecN {
                    values: [-(x3 - x2 + a), -(x2 - x3 + a)],
                },
            }
        },
    )
}
