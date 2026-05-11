use super::{helpers::*, *};

pub(super) fn tp277() -> ProblemCase {
    hilbert_linear_case::<4>("schittkowski_tp277", "tp277", "Schittkowski TP277")
}

pub(super) fn hilbert_linear_case<const N: usize>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
) -> ProblemCase {
    let mut fex = 0.0;
    for i in 0..N {
        for j in 0..N {
            fex += 1.0 / ((i + j + 1) as f64);
        }
    }
    objective_only_case(
        id,
        family,
        description,
        [0.0; N],
        [Some(0.0); N],
        [None; N],
        fex,
        |x| {
            let mut objective = SX::zero();
            let mut inequalities: VecN<SX, N> = VecN {
                values: std::array::from_fn(|_| SX::zero()),
            };
            for i in 0..N {
                let mut h = SX::zero();
                for j in 0..N {
                    let a = 1.0 / ((i + j + 1) as f64);
                    objective += a * x.values[i];
                    h += a * (x.values[j] - 1.0);
                }
                inequalities.values[i] = -h;
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities,
            }
        },
    )
}
