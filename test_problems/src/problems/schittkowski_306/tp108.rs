use super::helpers::*;
use super::*;

pub(super) fn tp108() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp108",
        "tp108",
        "Schittkowski TP108 nine-variable packing problem",
        [1.0; 9],
        [Some(0.0); 9],
        [Some(1.0); 9],
        -0.866025403841,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8, x9] = x.values;
            let objective = -0.5 * (x1 * x4 - x2 * x3 + x3 * x9 - x5 * x9 + x5 * x8 - x6 * x7);
            let g = [
                1.0 - x3.powf(2.0) - x4.powf(2.0),
                1.0 - x9.powf(2.0),
                1.0 - x5.powf(2.0) - x6.powf(2.0),
                1.0 - x1.powf(2.0) - (x2 - x9).powf(2.0),
                1.0 - (x1 - x5).powf(2.0) - (x2 - x6).powf(2.0),
                1.0 - (x1 - x7).powf(2.0) - (x2 - x8).powf(2.0),
                1.0 - (x3 - x5).powf(2.0) - (x4 - x6).powf(2.0),
                1.0 - (x3 - x7).powf(2.0) - (x4 - x8).powf(2.0),
                1.0 - x7.powf(2.0) - (x8 - x9).powf(2.0),
                x1 * x4 - x2 * x3,
                x3 * x9,
                -x5 * x9,
                x5 * x8 - x6 * x7,
            ];
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: g.map(|v| -v),
                },
            }
        },
    )
}
