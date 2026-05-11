use super::helpers::*;
use super::*;

pub(super) fn tp079() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp079",
        "tp079",
        "Schittkowski TP079 chain objective with three nonlinear equalities",
        [2.0; 5],
        [None; 5],
        [None; 5],
        0.0787768208538,
        |x| {
            let [x0, x1, x2, x3, x4] = x.values;
            SymbolicNlpOutputs {
                objective: (x0 - 1.0).sqr()
                    + (x0 - x1).sqr()
                    + (x1 - x2).sqr()
                    + (x2 - x3).powf(4.0)
                    + (x3 - x4).powf(4.0),
                equalities: VecN {
                    values: [
                        x0 + x1.sqr() + x2.powf(3.0) - 2.0 - 3.0 * 2.0_f64.sqrt(),
                        x1 - x2.sqr() + x3 + 2.0 - 2.0 * 2.0_f64.sqrt(),
                        x0 * x4 - 2.0,
                    ],
                },
                inequalities: (),
            }
        },
    )
}
