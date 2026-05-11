use super::{helpers::*, *};

pub(super) fn tp110() -> ProblemCase {
    objective_only_case_no_ineq(
        "schittkowski_tp110",
        "tp110",
        "Schittkowski TP110",
        [9.0; 10],
        [Some(2.001); 10],
        [Some(9.999); 10],
        -45.7784697153,
        |x| {
            let mut product = SX::from(1.0);
            let mut barrier = SX::zero();
            for xi in x.values {
                product *= xi;
                barrier += (xi - 2.0).log().sqr() + (10.0 - xi).log().sqr();
            }
            SymbolicNlpOutputs {
                objective: barrier - product.powf(0.2),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}
