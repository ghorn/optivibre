use super::{helpers::*, *};

pub(super) fn tp282() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp282",
        "tp282",
        "Schittkowski TP282",
        [-1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [None; 10],
        [None; 10],
        0.0,
        |x| {
            let mut objective = (x.values[0] - 1.0).sqr() + (x.values[9] - 1.0).sqr();
            for i in 0..9 {
                let f = ((10 - (i + 1)) * 10) as f64;
                objective += f * (x.values[i].sqr() - x.values[i + 1]).sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
