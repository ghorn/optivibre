use super::helpers::*;
use super::*;

pub(super) fn tp077() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp077",
        "tp077",
        "Schittkowski TP077 five-variable problem with two nonlinear equalities",
        [2.0; 5],
        [None; 5],
        [None; 5],
        0.241505128786,
        |x| {
            let [x0, x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: (x0 - 1.0).sqr()
                    + (x0 - x1).sqr()
                    + (x2 - 1.0).sqr()
                    + (x3 - 1.0).powi(4)
                    + (x4 - 1.0).powi(6),
                equalities: VecN {
                    values: [
                        x0.sqr() * x3 + (x3 - x4).sin() - 2.0 * 2.0_f64.sqrt(),
                        x1 + x2.powi(4) * x3.sqr() - 8.0 - 2.0_f64.sqrt(),
                    ],
                },
                inequalities: (),
            }
        },
    )
}
