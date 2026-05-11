use super::{helpers::*, *};

pub(super) fn tp237() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp237",
        "tp237",
        "Schittkowski TP237",
        [65.0, 10.0],
        [Some(54.0), None],
        [Some(75.0), Some(65.0)],
        -58.9034360,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp236_239_symbolic_objective(x1, x2),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        700.0 - x1 * x2,
                        5.0 * (x1 / 25.0).sqr() - x2,
                        5.0 * (x1 - 55.0) - (x2 - 50.0).sqr(),
                    ],
                },
            }
        },
    )
}
