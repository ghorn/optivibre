use super::{helpers::*, *};

pub(super) fn tp289() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp289",
        "tp289",
        "Schittkowski TP289",
        std::array::from_fn(|i| {
            let sign = if (i + 1) % 2 == 0 { 1.0 } else { -1.0 };
            sign * (1.0 + (i + 1) as f64 / 30.0)
        }),
        [None; 30],
        [None; 30],
        0.0,
        |x| {
            let mut norm2 = SX::zero();
            for i in 0..30 {
                norm2 += x.values[i].sqr();
            }
            SymbolicNlpOutputs {
                objective: 1.0 - (-norm2 / 60.0).exp(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
