use super::helpers::*;
use super::*;

pub(super) fn tp076() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp076",
        "tp076",
        "Schittkowski TP076 convex quadratic with three linear inequalities",
        [0.5; 4],
        [Some(0.0); 4],
        [None; 4],
        -4.68181818182,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: x0.sqr() + x2.sqr() + 0.5 * (x1.sqr() + x3.sqr()) - x0 * x2 + x2 * x3
                    - x0
                    - 3.0 * x1
                    + x2
                    - x3,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(-x0 - 2.0 * x1 - x2 - x3 + 5.0),
                        -(-3.0 * x0 - x1 - 2.0 * x2 + x3 + 4.0),
                        -(x1 + 4.0 * x2 - 1.5),
                    ],
                },
            }
        },
    )
}
