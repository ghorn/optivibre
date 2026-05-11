use super::{helpers::*, *};

pub(super) fn tp287() -> ProblemCase {
    let mut x0 = [0.0; 20];
    for i in 0..5 {
        x0[i] = -3.0;
        x0[i + 5] = -1.0;
        x0[i + 10] = -3.0;
        x0[i + 15] = -1.0;
    }

    objective_only_case_no_ineq(
        "schittkowski_tp287",
        "tp287",
        "Schittkowski TP287",
        x0,
        [None; 20],
        [None; 20],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..5 {
                objective += 100.0 * (x.values[i].sqr() - x.values[i + 5]).sqr()
                    + (x.values[i] - 1.0).sqr()
                    + 90.0 * (x.values[i + 10].sqr() - x.values[i + 15]).sqr()
                    + (x.values[i + 10] - 1.0).sqr()
                    + 10.1 * ((x.values[i + 5] - 1.0).sqr() + (x.values[i + 15] - 1.0).sqr())
                    + 19.8 * (x.values[i + 5] - 1.0) * (x.values[i + 15] - 1.0);
            }
            SymbolicNlpOutputs {
                objective: objective * 1.0e-5,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
