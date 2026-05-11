use super::{helpers::*, *};

const E: [f64; 5] = [-15.0, -27.0, -36.0, -18.0, -12.0];
const D: [f64; 5] = [4.0, 8.0, 10.0, 6.0, 2.0];
const B: [f64; 10] = [-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0];
const C: [[f64; 5]; 5] = [
    [30.0, -20.0, -10.0, 32.0, -10.0],
    [-20.0, 39.0, -6.0, -31.0, 32.0],
    [-10.0, -6.0, 10.0, -6.0, -10.0],
    [32.0, -31.0, -6.0, 39.0, -20.0],
    [-10.0, 32.0, -10.0, -20.0, 30.0],
];
const A: [[f64; 5]; 10] = [
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

pub(super) fn tp117() -> ProblemCase {
    let mut x0 = [1.0e-3; 15];
    x0[6] = 60.0;
    objective_only_case(
        "schittkowski_tp117",
        "tp117",
        "Schittkowski TP117",
        x0,
        [Some(0.0); 15],
        [None; 15],
        32.3486789791,
        |x| {
            let mut quadratic = SX::zero();
            let mut cubic = SX::zero();
            for j in 0..5 {
                cubic += D[j] * x.values[10 + j].powf(3.0);
                for (i, row) in C.iter().enumerate() {
                    quadratic += row[j] * x.values[10 + i] * x.values[10 + j];
                }
            }
            let mut linear = SX::zero();
            for (i, bi) in B.iter().enumerate() {
                linear += *bi * x.values[i];
            }
            let mut g = VecN {
                values: [SX::zero(), SX::zero(), SX::zero(), SX::zero(), SX::zero()],
            };
            for j in 0..5 {
                let mut t4 = SX::zero();
                for (i, row) in C.iter().enumerate() {
                    t4 += row[j] * x.values[10 + i];
                }
                let mut t5 = SX::zero();
                for (i, row) in A.iter().enumerate() {
                    t5 += row[j] * x.values[i];
                }
                g.values[j] = -(2.0 * t4 + 3.0 * D[j] * x.values[10 + j].sqr() + E[j] - t5);
            }
            SymbolicNlpOutputs {
                objective: -(linear - quadratic - 2.0 * cubic),
                equalities: VecN { values: [] },
                inequalities: g,
            }
        },
    )
}
