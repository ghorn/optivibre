use super::{helpers::*, *};

pub(super) fn tp306() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp306",
        "tp306",
        "Schittkowski TP306",
        [1.0, 1.0],
        [None; 2],
        [None; 2],
        -3.0 / std::f64::consts::E,
        |x| SymbolicNlpOutputs {
            objective: -(-x.values[0] - x.values[1]).exp()
                * (2.0 * x.values[0].sqr() + 3.0 * x.values[1].sqr()),
            equalities: VecN { values: [] },
            inequalities: (),
        },
    )
}
