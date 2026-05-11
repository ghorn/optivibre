#![allow(clippy::approx_constant)]
use super::helpers::*;
use super::*;

pub(super) fn tp087() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp087",
        "tp087",
        "Schittkowski TP087 six-variable nonsmooth power-flow problem",
        [390.0, 1000.0, 419.5, 340.5, 198.175, 0.5],
        [
            Some(0.0),
            Some(0.0),
            Some(340.0),
            Some(340.0),
            Some(-1000.0),
            Some(0.0),
        ],
        [
            Some(400.0),
            Some(1000.0),
            Some(420.0),
            Some(420.0),
            Some(1000.0),
            Some(0.5236),
        ],
        8927.59773493,
        |x| {
            let [x0, x1, x2, x3, x4, x5] = x.values;
            let a = 131.078;
            let b = 1.48477;
            let c = 0.90798;
            let d = 1.47588_f64.cos();
            let e = 1.47588_f64.sin();
            let ind0 = (x0 - 300.0).max(0.0) / (x0 - 300.0).abs().max(1.0e-9);
            let ind1 = (x1 - 100.0).max(0.0) / (x1 - 100.0).abs().max(1.0e-9);
            let ind2 = (x1 - 200.0).max(0.0) / (x1 - 200.0).abs().max(1.0e-9);
            SymbolicNlpOutputs {
                objective: (30.0 + ind0) * x0 + (28.0 + ind1 + ind2) * x1,
                equalities: VecN {
                    values: [
                        -x0 + 300.0 - x2 * x3 / a * (b - x5).cos() + c * x2.sqr() / a * d,
                        -x1 - x2 * x3 / a * (b + x5).cos() + c * x3.sqr() / a * d,
                        -x4 - x2 * x3 / a * (b + x5).sin() + c * x3.sqr() / a * e,
                        200.0 - x2 * x3 / a * (b - x5).sin() + c * x2.sqr() / a * e,
                    ],
                },
                inequalities: (),
            }
        },
    )
}
