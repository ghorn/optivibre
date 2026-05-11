use super::{helpers::*, *};

pub(super) fn tp286() -> ProblemCase {
    let mut x0 = [-1.2; 20];
    for value in x0.iter_mut().skip(10) {
        *value = 1.0;
    }

    objective_only_case_no_ineq(
        "schittkowski_tp286",
        "tp286",
        "Schittkowski TP286",
        x0,
        [None; 20],
        [None; 20],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..10 {
                objective += (x.values[i] - 1.0).sqr();
                objective += (10.0 * (x.values[i].sqr() - x.values[i + 10])).sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
