use super::helpers::*;
use super::*;

pub(super) fn tp078() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp078",
        "tp078",
        "Schittkowski TP078 product objective with three nonlinear equalities",
        [-2.0, 1.5, 2.0, -1.0, -1.0],
        [None; 5],
        [None; 5],
        -2.91970040911,
        |x| {
            let [x0, x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: x0 * x1 * x2 * x3 * x4,
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
