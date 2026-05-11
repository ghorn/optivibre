use super::helpers::*;
use super::*;

pub(super) fn tp083() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp083",
        "tp083",
        "Schittkowski TP083 five-variable industrial design problem",
        [78.0, 33.0, 27.0, 27.0, 27.0],
        [Some(78.0), Some(33.0), Some(27.0), Some(27.0), Some(27.0)],
        [Some(102.0), Some(45.0), Some(45.0), Some(45.0), Some(45.0)],
        -30665.5386717,
        |x| {
            let [x0, x1, x2, x3, x4] = x.values;
            let v1 = 85.334407 + 0.0056858 * x1 * x4 + 0.0006262 * x0 * x3 - 0.0022053 * x2 * x4;
            let v2 =
                80.51249 + 0.0071317 * x1 * x4 + 0.0029955 * x0 * x1 + 0.0021813 * x2.sqr() - 90.0;
            let v3 =
                9.300961 + 0.0047026 * x2 * x4 + 0.0012547 * x0 * x2 + 0.0019085 * x2 * x3 - 20.0;
            SymbolicNlpOutputs {
                objective: 5.3578547 * x2.sqr() + 0.8356891 * x0 * x4 + 37.293239 * x.values[0]
                    - 40792.141,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-v1, -v2, -v3, -(92.0 - v1), -(20.0 - v2), -(5.0 - v3)],
                },
            }
        },
    )
}
