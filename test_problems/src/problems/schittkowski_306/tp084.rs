use super::helpers::*;
use super::*;

pub(super) fn tp084() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp084",
        "tp084",
        "Schittkowski TP084 five-variable process design problem",
        [2.52, 2.0, 37.5, 9.25, 6.8],
        [Some(0.0), Some(1.2), Some(20.0), Some(9.0), Some(6.5)],
        [Some(1000.0), Some(2.4), Some(60.0), Some(9.3), Some(7.0)],
        -5280335.13306,
        |x| {
            let [x0, x1, x2, x3, x4] = x.values;
            let a = [
                -2.4345e4,
                -8.720288849e6,
                1.505125253e5,
                -1.566950325e2,
                4.764703222e5,
                7.294828271e5,
                -1.45421402e5,
                2.9311506e3,
                -40.427932,
                5.106192e3,
                1.571136e4,
                -1.550111084e5,
                4.36053352e3,
                12.9492344,
                1.0236884e4,
                1.3176786e4,
                -3.266695104e5,
                7.39068412e3,
                -27.8986976,
                1.6643076e4,
                3.0988146e4,
            ];
            let v1 = x0 * (a[7] + a[8] * x1 + a[9] * x2 + a[10] * x3 + a[11] * x4);
            let v2 = x0 * (a[12] + a[13] * x1 + a[14] * x2 + a[15] * x3 + a[16] * x4);
            let v3 = x0 * (a[16] + a[17] * x1 + a[18] * x2 + a[19] * x3 + a[20] * x4);
            SymbolicNlpOutputs {
                objective: -(a[0] + x0 * (a[1] + a[2] * x1 + a[3] * x2 + a[4] * x3 + a[5] * x4)),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -v1,
                        -v2,
                        -v3,
                        -(2.94e5 - v1),
                        -(2.94e5 - v2),
                        -(2.772e5 - v3),
                    ],
                },
            }
        },
    )
}
