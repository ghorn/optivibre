use super::helpers::*;
use super::*;

pub(super) fn tp072() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp072",
        "tp072",
        "Schittkowski TP072 reciprocal constraints with four positive variables",
        [1.0; 4],
        [Some(1e-3); 4],
        [Some(4.0e5), Some(3.0e5), Some(2.0e5), Some(1.0e5)],
        727.67936,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: 1.0 + x0 + x1 + x2 + x3,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        4.0 / x0 + 2.25 / x1 + 1.0 / x2 + 0.25 / x3 - 0.0401,
                        0.16 / x.values[0]
                            + 0.36 / x.values[1]
                            + 0.64 * (1.0 / x.values[2] + 1.0 / x.values[3])
                            - 0.010085,
                    ],
                },
            }
        },
    )
}
