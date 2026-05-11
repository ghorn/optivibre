use super::{helpers::*, *};

const C: [f64; 10] = [
    -6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179,
];

pub(super) fn tp111() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp111",
        "tp111",
        "Schittkowski TP111",
        [-2.3; 10],
        [Some(-100.0); 10],
        [Some(100.0); 10],
        -47.7610902637,
        |x| {
            let exp_x = x.values.map(|xi| xi.exp());
            let total = exp_x.iter().copied().fold(SX::zero(), |sum, xi| sum + xi);
            let log_total = total.log();
            let mut objective = SX::zero();
            for (i, ci) in C.iter().enumerate() {
                objective += exp_x[i] * (*ci + x.values[i] - log_total);
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN {
                    values: [
                        exp_x[0] + 2.0 * exp_x[1] + 2.0 * exp_x[2] + exp_x[5] + exp_x[9] - 2.0,
                        exp_x[3] + 2.0 * exp_x[4] + exp_x[5] + exp_x[6] - 1.0,
                        exp_x[2] + exp_x[6] + exp_x[7] + 2.0 * exp_x[8] + exp_x[9] - 1.0,
                    ],
                },
                inequalities: (),
            }
        },
    )
}
