use super::{helpers::*, *};

#[allow(clippy::approx_constant)]
pub(super) fn tp104() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp104",
        "tp104",
        "Schittkowski TP104 eight-variable allocation problem",
        [6.0, 3.0, 0.4, 0.2, 6.0, 6.0, 1.0, 0.5],
        [Some(0.1); 8],
        [Some(10.0); 8],
        3.95116343955,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8] = x.values;
            let bx = 0.4 * (x1.powf(0.67) / x7.powf(0.67) + x2.powf(0.67) / x8.powf(0.67)) + 10.0
                - x1
                - x2;
            let a = 0.0588;
            let g1 = -a * x5 * x7 - 0.1 * x1 + 1.0;
            let g2 = -a * x6 * x8 - 0.1 * x1 - 0.1 * x2 + 1.0;
            let g3 =
                (-4.0 * x3 - 2.0 * x3.abs().powf(-0.71)) / x5 - a * x3.abs().powf(-1.3) * x7 + 1.0;
            let g4 =
                (-4.0 * x4 - 2.0 * x4.abs().powf(-0.71)) / x6 - a * x4.abs().powf(-1.3) * x8 + 1.0;
            SymbolicNlpOutputs {
                objective: bx,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-g1, -g2, -g3, -g4, 1.0 - bx, bx - 4.2],
                },
            }
        },
    )
}
