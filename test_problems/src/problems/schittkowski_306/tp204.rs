use super::{helpers::*, *};

pub(super) fn tp204_value(x1: SX, x2: SX) -> SX {
    let a = [0.13294, -0.244378, 0.325895];
    let d = [2.5074, -1.36401, 1.02282];
    let h = [
        [-0.564255, 0.392417],
        [-0.404979, 0.927589],
        [-0.0735084, 0.535493],
    ];
    let mut objective = SX::zero();
    for i in 0..3 {
        let prod = h[i][0] * x1 + h[i][1] * x2;
        let f = a[i] + prod + 0.5 * d[i] * prod.sqr();
        objective += f.sqr();
    }
    objective
}

pub(super) fn tp204() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp204",
        "tp204",
        "Schittkowski TP204",
        [0.1, 0.1],
        [None, None],
        [None, None],
        0.183601,
        |x| {
            let [x1, x2] = x.values;
            SymbolicNlpOutputs {
                objective: tp204_value(x1, x2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
