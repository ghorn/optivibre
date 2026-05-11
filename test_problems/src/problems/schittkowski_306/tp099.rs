use super::helpers::*;
use super::*;

pub(super) fn tp099() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp099",
        "tp099",
        "Schittkowski TP099 seven-angle trajectory problem with two equalities",
        [0.5; 7],
        [Some(0.0); 7],
        [Some(1.58); 7],
        -831079891.516,
        |x| {
            let a = [0.0, 50.0, 50.0, 75.0, 75.0, 75.0, 100.0, 100.0];
            let t = [0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 290.0, 380.0];
            let mut p: [SX; 8] = std::array::from_fn(|_| SX::zero());
            let mut q: [SX; 8] = std::array::from_fn(|_| SX::zero());
            let mut r: [SX; 8] = std::array::from_fn(|_| SX::zero());
            let mut s: [SX; 8] = std::array::from_fn(|_| SX::zero());
            for i in 1..8 {
                let im = i - 1;
                let v1 = a[i] * x.values[im].sin() - 32.0;
                let v2 = a[i] * x.values[im].cos();
                let v3 = t[i] - t[im];
                let v4 = 0.5 * v3 * v3;
                p[i] = v2 * v4 + v3 * r[im] + p[im];
                q[i] = v1 * v4 + v3 * s[im] + q[im];
                r[i] = v2 * v3 + r[im];
                s[i] = v1 * v3 + s[im];
            }
            SymbolicNlpOutputs {
                objective: -r[7].sqr(),
                equalities: VecN {
                    values: [q[7] - 1.0e5, s[7] - 1.0e3],
                },
                inequalities: (),
            }
        },
    )
}
