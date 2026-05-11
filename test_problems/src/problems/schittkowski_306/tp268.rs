use super::{helpers::*, *};

pub(super) fn tp268() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp268",
        "tp268",
        "Schittkowski TP268",
        [1.0; 5],
        [None; 5],
        [None; 5],
        0.0,
        |x| {
            let dd = [
                [10197.0, -12454.0, -1013.0, 1948.0, 329.0],
                [-12454.0, 20909.0, -1733.0, -4914.0, -186.0],
                [-1013.0, -1733.0, 1755.0, 1089.0, -174.0],
                [1948.0, -4914.0, 1089.0, 1515.0, -22.0],
                [329.0, -186.0, -174.0, -22.0, 27.0],
            ];
            let ddvekt = [-9170.0, 17099.0, -2271.0, -4336.0, -43.0];
            let xs = x.values;
            let mut objective = SX::from(14463.0);
            for i in 0..5 {
                let mut hf = SX::zero();
                for j in 0..5 {
                    hf += dd[i][j] * xs[j];
                }
                objective += xs[i] * (hf - 2.0 * ddvekt[i]);
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        xs[0] + xs[1] + xs[2] + xs[3] + xs[4] - 5.0,
                        -10.0 * xs[0] - 10.0 * xs[1] + 3.0 * xs[2] - 5.0 * xs[3] - 4.0 * xs[4]
                            + 20.0,
                        8.0 * xs[0] - xs[1] + 2.0 * xs[2] + 5.0 * xs[3] - 3.0 * xs[4] - 40.0,
                        -8.0 * xs[0] + xs[1] - 2.0 * xs[2] - 5.0 * xs[3] + 3.0 * xs[4] + 11.0,
                        4.0 * xs[0] + 2.0 * xs[1] - 3.0 * xs[2] + 5.0 * xs[3] - xs[4] - 30.0,
                    ],
                },
            }
        },
    )
}
