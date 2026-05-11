use super::{helpers::*, *};

pub(super) fn tp253() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp253",
        "tp253",
        "Schittkowski TP253",
        [0.0, 2.0, 0.0],
        [Some(0.0), Some(0.0), Some(0.0)],
        [None, None, None],
        69.282032,
        |x| {
            let points = [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 10.0],
                [0.0, 10.0, 10.0],
            ];
            let xs = x.values;
            let mut objective = SX::zero();
            for p in points {
                objective +=
                    ((xs[0] - p[0]).sqr() + (xs[1] - p[1]).sqr() + (xs[2] - p[2]).sqr()).sqrt();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-30.0 + 3.0 * xs[0] + 3.0 * xs[2]],
                },
            }
        },
    )
}
