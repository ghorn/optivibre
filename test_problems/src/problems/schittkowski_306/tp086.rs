#![allow(clippy::needless_range_loop)]
use super::helpers::*;
use super::*;

pub(super) fn tp086() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp086",
        "tp086",
        "Schittkowski TP086 five-variable cubic quadratic program",
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [Some(0.0); 5],
        [None; 5],
        -32.3486789716,
        |x| {
            let e = [-15.0, -27.0, -36.0, -18.0, -12.0];
            let d = [4.0, 8.0, 10.0, 6.0, 2.0];
            let c = [
                [30.0, -20.0, -10.0, 32.0, -10.0],
                [-20.0, 39.0, -6.0, -31.0, 32.0],
                [-10.0, -6.0, 10.0, -6.0, -10.0],
                [32.0, -31.0, -6.0, 39.0, -20.0],
                [-10.0, 32.0, -10.0, -20.0, 30.0],
            ];
            let a = [
                [-16.0, 2.0, 0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0, 0.4, 2.0],
                [-3.5, 0.0, 2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, -4.0, -1.0],
                [0.0, -9.0, -2.0, 1.0, -2.8],
                [2.0, 0.0, -4.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -2.0, -3.0, -2.0, -1.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ];
            let b = [-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0];
            let mut obj = SX::zero();
            for i in 0..5 {
                obj += e[i] * x.values[i] + d[i] * x.values[i].powf(3.0);
                for j in 0..5 {
                    obj += c[j][i] * x.values[i] * x.values[j];
                }
            }
            let vals: [SX; 10] = std::array::from_fn(|i| {
                let mut g = SX::zero();
                for j in 0..5 {
                    g += a[i][j] * x.values[j];
                }
                -(g - b[i])
            });
            SymbolicNlpOutputs {
                objective: obj,
                equalities: VecN { values: [] },
                inequalities: VecN { values: vals },
            }
        },
    )
}
