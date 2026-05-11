use super::{helpers::*, *};

pub(super) fn tp239() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp239",
        "tp239",
        "Schittkowski TP239",
        [10.0, 10.0],
        [Some(0.0), Some(0.0)],
        [Some(75.0), Some(65.0)],
        -58.903436,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp236_239_symbolic_objective(x1, x2),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [700.0 - x1 * x2],
                },
            }
        },
    )
}
