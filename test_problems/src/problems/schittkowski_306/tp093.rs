use super::helpers::*;
use super::*;

pub(super) fn tp093() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp093",
        "tp093",
        "Schittkowski TP093 six-variable process design problem",
        [5.54, 4.4, 12.02, 11.82, 0.702, 0.852],
        [Some(0.0); 6],
        [None; 6],
        135.075961229,
        |x| {
            let [x0, x1, x2, x3, x4, x5] = x.values;
            let v1 = x0 + x1 + x2;
            let v2 = x0 + 1.57 * x1 + x3;
            let v3 = x0 * x3;
            let v4 = x2 * x1;
            SymbolicNlpOutputs {
                objective: 0.0204 * v3 * v1
                    + 0.0187 * v4 * v2
                    + 0.0607 * v3 * v1 * x4.sqr()
                    + 0.0437 * v4 * v2 * x5.sqr(),
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(0.001 * x0 * x1 * x2 * x3 * x4 * x5 - 2.07),
                        -(1.0
                            - 0.00062 * x0 * x3 * x4.sqr() * (x0 + x1 + x2)
                            - 0.00058 * x1 * x2 * x5.sqr() * (x0 + 1.57 * x1 + x3)),
                    ],
                },
            }
        },
    )
}
