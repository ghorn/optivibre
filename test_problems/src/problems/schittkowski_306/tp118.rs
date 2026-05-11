use super::{helpers::*, *};

pub(super) fn tp118() -> ProblemCase {
    objective_only_case(
        "schittkowski_tp118",
        "tp118",
        "Schittkowski TP118",
        [
            20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0,
            20.0,
        ],
        [
            Some(8.0),
            Some(43.0),
            Some(3.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
        ],
        [
            Some(21.0),
            Some(57.0),
            Some(16.0),
            Some(90.0),
            Some(120.0),
            Some(60.0),
            Some(90.0),
            Some(120.0),
            Some(60.0),
            Some(90.0),
            Some(120.0),
            Some(60.0),
            Some(90.0),
            Some(120.0),
            Some(60.0),
        ],
        664.820449993,
        |x| {
            let mut objective = SX::zero();
            for m in 0..5 {
                let i = 3 * m;
                objective += 2.3 * x.values[i]
                    + 1.0e-4 * x.values[i].sqr()
                    + 1.7 * x.values[i + 1]
                    + 1.0e-4 * x.values[i + 1].sqr()
                    + 2.2 * x.values[i + 2]
                    + 1.5e-4 * x.values[i + 2].sqr();
            }
            let mut g = VecN {
                values: [
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                    SX::zero(),
                ],
            };
            for i in 0..4 {
                g.values[i] = -(x.values[3 * i + 3] - x.values[3 * i] + 7.0);
                g.values[i + 4] = -(x.values[3 * i + 4] - x.values[3 * i + 1] + 7.0);
                g.values[i + 8] = -(x.values[3 * i + 5] - x.values[3 * i + 2] + 7.0);
                g.values[i + 12] = -(x.values[3 * i] - x.values[3 * i + 3] + 6.0);
                g.values[i + 16] = -(x.values[3 * i + 1] - x.values[3 * i + 4] + 7.0);
                g.values[i + 20] = -(x.values[3 * i + 2] - x.values[3 * i + 5] + 6.0);
            }
            g.values[24] = -(x.values[0] + x.values[1] + x.values[2] - 60.0);
            g.values[25] = -(x.values[3] + x.values[4] + x.values[5] - 50.0);
            g.values[26] = -(x.values[6] + x.values[7] + x.values[8] - 70.0);
            g.values[27] = -(x.values[9] + x.values[10] + x.values[11] - 85.0);
            g.values[28] = -(x.values[12] + x.values[13] + x.values[14] - 100.0);
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: g,
            }
        },
    )
}
