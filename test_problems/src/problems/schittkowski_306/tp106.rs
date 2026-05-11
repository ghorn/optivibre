use super::helpers::*;
use super::*;

pub(super) fn tp106() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp106",
        "tp106",
        "Schittkowski TP106 eight-variable process synthesis problem",
        [5000.0, 5000.0, 5000.0, 200.0, 350.0, 150.0, 225.0, 425.0],
        [
            Some(100.0),
            Some(1000.0),
            Some(1000.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
            Some(10.0),
        ],
        [
            Some(10000.0),
            Some(10000.0),
            Some(10000.0),
            Some(1000.0),
            Some(1000.0),
            Some(1000.0),
            Some(1000.0),
            Some(1000.0),
        ],
        7049.2480,
        |x| {
            let [x1, x2, x3, x4, x5, x6, x7, x8] = x.values;
            let vals = [
                -(-0.0025 * (x4 + x6) + 1.0),
                -(-0.0025 * (x5 + x7 - x4) + 1.0),
                -(-0.01 * (x8 - x5) + 1.0),
                -(-833.33252 * x4 - 100.0 * x1 + 83333.333 + x1 * x6),
                -(-1250.0 * x5 - x2 * x4 + 1250.0 * x4 + x2 * x7),
                -(-1.25e6 - x3 * x5 + 2500.0 * x5 + x3 * x8),
            ];
            SymbolicNlpOutputs {
                objective: x1 + x2 + x3,
                equalities: VecN { values: [] },
                inequalities: VecN { values: vals },
            }
        },
    )
}
