use super::{helpers::*, *};

pub(super) fn tp243() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp243",
        "tp243",
        "Schittkowski TP243",
        [0.1, 0.1, 0.1],
        [None, None, None],
        [None, None, None],
        0.79657853,
        |x| {
            let a = [0.14272, -0.184981, -0.521869, -0.685306];
            let d = [1.75168, -1.35195, -0.479048, -0.3648];
            let b = [
                [2.95137, 4.87407, -2.0506],
                [4.87407, 9.39321, -3.93185],
                [-2.0506, -3.93189, 2.64745],
            ];
            let e = [
                [-0.564255, 0.392417, -0.404979],
                [0.927589, -0.0735083, 0.535493],
                [0.658799, -0.636666, -0.681091],
                [-0.869487, 0.586387, 0.289826],
            ];
            let xs = x.values;
            let mut xbx = SX::zero();
            for i in 0..3 {
                for j in 0..3 {
                    xbx += xs[i] * b[i][j] * xs[j];
                }
            }
            let mut objective = SX::zero();
            for i in 0..4 {
                let f =
                    a[i] + e[i][0] * xs[0] + e[i][1] * xs[1] + e[i][2] * xs[2] + 0.5 * d[i] * xbx;
                objective += f.sqr();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
