use super::helpers::*;
use super::*;

pub(super) fn tp074() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp074",
        "tp074",
        "Schittkowski TP074 four-variable trigonometric network problem",
        [0.0; 4],
        [Some(0.0), Some(0.0), Some(-0.55), Some(-0.55)],
        [Some(1200.0), Some(1200.0), Some(0.55), Some(0.55)],
        5126.49810934,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            let a = 0.55;
            SymbolicNlpOutputs {
                objective: 3.0 * x0
                    + 1.0e-6 * x0.powf(3.0)
                    + 2.0 * x1
                    + 2.0e-6 / 3.0 * x1.powf(3.0),
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
