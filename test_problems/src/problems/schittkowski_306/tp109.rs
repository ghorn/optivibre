use super::{helpers::*, *};

pub(super) fn tp109() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp109",
        "tp109",
        "Schittkowski TP109 nine-variable power-flow design problem",
        [0.0, 0.0, 0.0, 0.0, 250.0, 250.0, 200.0, 0.0, 0.0],
        [
            Some(0.0),
            Some(0.0),
            Some(-0.55),
            Some(-0.55),
            Some(196.0),
            Some(196.0),
            Some(196.0),
            Some(-400.0),
            Some(-400.0),
        ],
        [
            None,
            None,
            Some(0.55),
            Some(0.55),
            Some(252.0),
            Some(252.0),
            Some(252.0),
            Some(800.0),
            Some(800.0),
        ],
        5362.06927538,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9] = x.values;
            let a = 50.176;
            let ra = 1.0 / a;
            let b = 0.25_f64.sin();
            let c = 0.25_f64.cos();
            let g1 = x4 - x3 + 0.55;
            let g2 = x3 - x4 + 0.55;
            let g3 = 2.25e6 - x1.powi(2) - x8.powi(2);
            let g4 = 2.25e6 - x2.powi(2) - x9.powi(2);
            let g5 = (x5 * x6 * (-x3 - 0.25).sin()
                + x5 * x7 * (-x4 - 0.25).sin()
                + 2.0 * x5.powi(2) * b)
                * ra
                + 400.0
                - x1;
            let g6 = (x5 * x6 * (x3 - 0.25).sin()
                + x6 * x7 * (x3 - x4 - 0.25).sin()
                + 2.0 * x6.powi(2) * b)
                * ra
                + 400.0
                - x2;
            let g7 = (x5 * x7 * (x4 - 0.25).sin()
                + x6 * x7 * (x4 - x3 - 0.25).sin()
                + 2.0 * x7.powi(2) * b)
                * ra
                + 881.779;
            let g8 = x8
                + (x5 * x6 * (-x3 - 0.25).cos() + x5 * x7 * (-x4 - 0.25).cos()
                    - 2.0 * x5.powi(2) * c)
                    * ra
                + 0.0007533 * x5.powi(2)
                - 200.0;
            let g9 = x9
                + (x5 * x6 * (x3 - 0.25).cos() + x7 * x6 * (x3 - x4 - 0.25).cos()
                    - 2.0 * x6.powi(2) * c)
                    * ra
                + 0.0007533 * x6.powi(2)
                - 200.0;
            let g10 = (x5 * x7 * (x4 - 0.25).cos() + x6 * x7 * (x4 - x3 - 0.25).cos()
                - 2.0 * x7.powi(2) * c)
                * ra
                + 0.0007533 * x7.powi(2)
                - 22.938;
            SymbolicNlpOutputs {
                objective: 3.0 * x1 + 1e-6 * x1.powi(3) + 0.522074e-6 * x2.powi(3) + 2.0 * x2,
                equalities: VecN {
                    values: [g5, g6, g7, g8, g9, g10],
                },
                inequalities: VecN {
                    values: [-g1, -g2, -g3, -g4],
                },
            }
        },
    )
}
