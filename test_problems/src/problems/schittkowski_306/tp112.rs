use super::{helpers::*, *};

const C: [f64; 10] = [
    -6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179,
];

pub(super) fn tp112() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp112",
        "tp112",
        "Schittkowski TP112",
        [0.1; 10],
        [Some(1.0e-4); 10],
        [None; 10],
        -47.761086,
        |x| {
            let total = x
                .values
                .iter()
                .copied()
                .fold(SX::zero(), |sum, xi| sum + xi);
            let log_total = total.log();
            let mut objective = SX::zero();
            for (i, ci) in C.iter().enumerate() {
                objective += x.values[i] * (*ci + x.values[i].log() - log_total);
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN {
                    values: [
                        x.values[0]
                            + 2.0 * x.values[1]
                            + 2.0 * x.values[2]
                            + x.values[5]
                            + x.values[9]
                            - 2.0,
                        x.values[3] + 2.0 * x.values[4] + x.values[5] + x.values[6] - 1.0,
                        x.values[2] + x.values[6] + x.values[7] + 2.0 * x.values[8] + x.values[9]
                            - 1.0,
                    ],
                },
                inequalities: (),
            }
        },
    )
}
