use super::helpers::*;
use super::*;

pub(super) fn tp100() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp100",
        "tp100",
        "Schittkowski TP100 seven-variable polynomial problem with four nonlinear inequalities",
        [1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0],
        [None; 7],
        [None; 7],
        680.630057275,
        |x| {
            let [x0, x1, x2, x3, x4, x5, x6] = x.values;
            SymbolicNlpOutputs {
                objective: (x0 - 10.0).sqr()
                    + 5.0 * (x1 - 12.0).sqr()
                    + x2.powi(4)
                    + 3.0 * (x3 - 11.0).sqr()
                    + 10.0 * x4.powi(6)
                    + 7.0 * x5.sqr()
                    + x6.powi(4)
                    - 4.0 * x5 * x6
                    - 10.0 * x5
                    - 8.0 * x6,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(-2.0 * x0.sqr() - 3.0 * x1.powi(4) - x2 - 4.0 * x3.sqr() - 5.0 * x4
                            + 127.0),
                        -(-7.0 * x0 - 3.0 * x1 - 10.0 * x2.sqr() - x3 + x4 + 282.0),
                        -(-23.0 * x0 - x1.sqr() - 6.0 * x5.sqr() + 8.0 * x6 + 196.0),
                        -(-4.0 * x0.sqr() - x1.sqr() + 3.0 * x0 * x1 - 2.0 * x2.sqr() - 5.0 * x5
                            + 11.0 * x6),
                    ],
                },
            }
        },
    )
}
