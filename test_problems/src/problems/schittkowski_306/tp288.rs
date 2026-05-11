use super::{helpers::*, *};

pub(super) fn tp288() -> ProblemCase {
    let mut x0 = [0.0; 20];
    for i in 0..5 {
        x0[i] = 3.0;
        x0[i + 5] = -1.0;
        x0[i + 10] = 0.0;
        x0[i + 15] = 1.0;
    }

    objective_only_case_no_ineq(
        "schittkowski_tp288",
        "tp288",
        "Schittkowski TP288",
        x0,
        [None; 20],
        [None; 20],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..5 {
                let f0 = x.values[i] + 10.0 * x.values[i + 5];
                let f1 = 5.0_f64.sqrt() * (x.values[i + 10] - x.values[i + 15]);
                let f2 = (x.values[i + 5] - 2.0 * x.values[i + 10]).sqr();
                let f3 = 10.0_f64.sqrt() * (x.values[i] - x.values[i + 15]).sqr();
                objective += f0.sqr() + f1.sqr() + f2.sqr() + f3.sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
