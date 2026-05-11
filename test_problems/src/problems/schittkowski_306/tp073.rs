use super::helpers::*;
use super::*;

pub(super) fn tp073() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp073",
        "tp073",
        "Schittkowski TP073 four-variable linear cost with one nonlinear inequality and one equality",
        [1.0; 4],
        [Some(0.0); 4],
        [None; 4],
        29.8943781573,
        |x| {
            let [x0, x1, x2, x3] = x.values;
            SymbolicNlpOutputs {
                objective: 24.55 * x0 + 26.75 * x1 + 39.0 * x2 + 40.5 * x3,
                equalities: VecN {
                    values: [x0 + x1 + x2 + x3 - 1.0],
                },
                inequalities: VecN {
                    values: [
                        -(2.3 * x0 + 5.6 * x1 + 11.1 * x2 + 1.3 * x3 - 5.0),
                        -(12.0 * x0 + 11.9 * x1 + 41.8 * x2 + 52.1 * x3
                            - 1.645
                                * (0.28 * x0.sqr()
                                    + 0.19 * x1.sqr()
                                    + 20.5 * x2.sqr()
                                    + 0.62 * x3.sqr())
                                .sqrt()
                            - 21.0),
                    ],
                },
            }
        },
    )
}
