use super::{helpers::*, *};

#[allow(clippy::needless_range_loop)]
pub(super) fn tp105() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp105",
        "tp105",
        "Schittkowski TP105 Gaussian mixture maximum likelihood problem",
        [0.1, 0.2, 100.0, 125.0, 175.0, 11.2, 13.2, 15.8],
        [
            Some(1e-3),
            Some(1e-3),
            Some(100.0),
            Some(130.0),
            Some(170.0),
            Some(5.0),
            Some(5.0),
            Some(5.0),
        ],
        [
            Some(0.499),
            Some(0.499),
            Some(180.0),
            Some(210.0),
            Some(240.0),
            Some(25.0),
            Some(25.0),
            Some(25.0),
        ],
        1138.41623960,
        |x| {
            let [p1, p2, m1, m2, m3, s1, s2, s3] = x.values;
            let mut objective = SX::zero();
            let v = 1.0 / (8.0 * std::f64::consts::FRAC_PI_4).sqrt();
            for i in 1..=235 {
                let y = if i <= 1 {
                    95.0
                } else if i <= 2 {
                    105.0
                } else if i <= 6 {
                    110.0
                } else if i <= 10 {
                    115.0
                } else if i <= 25 {
                    120.0
                } else if i <= 40 {
                    125.0
                } else if i <= 55 {
                    130.0
                } else if i <= 68 {
                    135.0
                } else if i <= 89 {
                    140.0
                } else if i <= 101 {
                    145.0
                } else if i <= 118 {
                    150.0
                } else if i <= 122 {
                    155.0
                } else if i <= 142 {
                    160.0
                } else if i <= 150 {
                    165.0
                } else if i <= 167 {
                    170.0
                } else if i <= 175 {
                    175.0
                } else if i <= 181 {
                    180.0
                } else if i <= 187 {
                    185.0
                } else if i <= 194 {
                    190.0
                } else if i <= 198 {
                    195.0
                } else if i <= 201 {
                    200.0
                } else if i <= 204 {
                    205.0
                } else if i <= 212 {
                    210.0
                } else if i <= 213 {
                    215.0
                } else if i <= 219 {
                    220.0
                } else if i <= 224 {
                    230.0
                } else if i <= 225 {
                    235.0
                } else if i <= 232 {
                    240.0
                } else if i <= 233 {
                    245.0
                } else {
                    260.0
                };
                let a = p1 / s1 * ((-(y - m1).powi(2)) / (2.0 * s1.powi(2))).max(-10.0).exp();
                let b = p2 / s2 * ((-(y - m2).powi(2)) / (2.0 * s2.powi(2))).max(-10.0).exp();
                let c = (1.0 - p1 - p2) / s3
                    * ((-(y - m3).powi(2)) / (2.0 * s3.powi(2))).max(-10.0).exp();
                objective -= ((a + b + c) * v).log();
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-(1.0 - p1 - p2)],
                },
            }
        },
    )
}
