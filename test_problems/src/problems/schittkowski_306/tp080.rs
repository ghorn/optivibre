use super::helpers::*;
use super::*;

pub(super) fn tp080() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp080",
        "tp080",
        "Schittkowski TP080 bounded five-variable exponential equality problem",
        [-2.0, 2.0, 2.0, -1.0, -1.0],
        [Some(-2.3), Some(-2.3), Some(-3.2), Some(-3.2), Some(-3.2)],
        [Some(2.3), Some(2.3), Some(3.2), Some(3.2), Some(3.2)],
        0.0539498477624,
        |x| {
            let [x0, x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: (x0 * x1 * x2 * x3 * x4).exp(),
                equalities: VecN {
                    values: [
                        x0.sqr() + x1.sqr() + x2.sqr() + x3.sqr() + x4.sqr() - 10.0,
                        x1 * x2 - 5.0 * x3 * x4,
                        x0.powf(3.0) + x1.powf(3.0) + 1.0,
                    ],
                },
                inequalities: (),
            }
        },
    )
}
