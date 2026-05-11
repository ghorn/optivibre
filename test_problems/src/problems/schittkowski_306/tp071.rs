use super::helpers::*;
use super::*;

pub(super) fn tp071() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp071",
        "tp071",
        "Schittkowski TP071 HS071 product inequality problem",
        [1.0, 5.0, 5.0, 1.0],
        [Some(1.0); 4],
        [Some(5.0); 4],
        17.0140172895,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: x0 * x3 * (x0 + x1 + x2) + x2,
                equalities: VecN {
                    values: [(x0.sqr() + x1.sqr() + x2.sqr() + x3.sqr() - 40.0) / 40.0],
                },
                inequalities: VecN {
                    values: [(25.0 - x0 * x1 * x2 * x3) / 25.0],
                },
            }
        },
    )
}
