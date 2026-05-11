#![allow(clippy::approx_constant)]
use super::helpers::*;
use super::*;

pub(super) fn tp070() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp070",
        "tp070",
        "Schittkowski TP070 four-variable chromatography fit with one nonlinear inequality",
        [2.0, 4.0, 0.04, 2.0],
        [Some(1e-5); 4],
        [Some(100.0), Some(100.0), Some(1.0), Some(100.0)],
        0.00749846356143,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            let c = [
                0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0,
            ];
            let yo = [
                0.00189, 0.1038, 0.268, 0.506, 0.577, 0.604, 0.725, 0.898, 0.947, 0.845, 0.702,
                0.528, 0.385, 0.257, 0.159, 0.0869, 0.0453, 0.01509, 0.00189,
            ];
            let b = x2 + (1.0 - x2) * x3;
            let h3 = 1.0 / 7.658;
            let h5 = b * h3;
            let h4 = h5 / x3;
            let h6 = 12.0 * x0 / (12.0 * x0 + 1.0);
            let h7 = 12.0 * x1 / (12.0 * x1 + 1.0);
            let z1 = x2 * b.pow(x1);
            let z2 = (x1 / 6.2832).sqrt();
            let z5 = 1.0 - x2;
            let z6 = (b / x3).pow(x0);
            let z7 = (x0 / 6.2832).sqrt();
            let mut objective = SX::zero();
            for i in 0..19 {
                let ci = c[i];
                let u1 =
                    z1 * z2 * SX::from(ci * h3).pow(x1 - 1.0) * (x1 * (1.0 - ci * h5)).exp() * h7;
                let u2 = z5
                    * z6
                    * z7
                    * SX::from(ci * h3).pow(x0 - 1.0)
                    * (x0 * (1.0 - ci * h4)).exp()
                    * h6;
                let r = u1 + u2 - yo[i];
                objective += r.sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-(x2 + (1.0 - x2) * x3)],
                },
            }
        },
    )
}
