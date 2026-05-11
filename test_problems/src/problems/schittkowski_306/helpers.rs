use super::*;

pub(super) fn metadata(
    id: &'static str,
    variant: &'static str,
    description: &'static str,
) -> CaseMetadata {
    CaseMetadata::new(id, variant, variant, SOURCE, description, false)
        .with_test_set(ProblemTestSet::Schittkowski306)
}

pub(super) fn tp057_data() -> ([f64; 44], [f64; 44]) {
    (
        [
            8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 12.0, 14.0, 14.0, 14.0, 16.0, 16.0,
            16.0, 18.0, 18.0, 20.0, 20.0, 20.0, 22.0, 22.0, 22.0, 24.0, 24.0, 24.0, 26.0, 26.0,
            26.0, 28.0, 28.0, 30.0, 30.0, 30.0, 32.0, 32.0, 34.0, 36.0, 36.0, 38.0, 38.0, 40.0,
            42.0,
        ],
        [
            0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46, 0.45, 0.43, 0.45, 0.43, 0.43, 0.44,
            0.43, 0.43, 0.46, 0.45, 0.42, 0.42, 0.43, 0.41, 0.41, 0.4, 0.42, 0.4, 0.4, 0.41, 0.4,
            0.41, 0.41, 0.4, 0.4, 0.4, 0.38, 0.41, 0.4, 0.4, 0.41, 0.38, 0.4, 0.4, 0.39, 0.39,
        ],
    )
}

pub(super) fn tp059_symbolic_objective(x0: SX, x1: SX) -> SX {
    let x0_2 = x0.sqr();
    let x0_3 = x0_2 * x0;
    let x0_4 = x0_3 * x0;
    let x1_2 = x1.sqr();
    let x1_3 = x1_2 * x1;
    let x1_4 = x1_3 * x1;
    -75.196 + 3.8112 * x0 - 0.12694 * x0_2 + 2.0567e-3 * x0_3 - 1.0345e-5 * x0_4 + 6.8306 * x1
        - 3.0234e-2 * x0 * x1
        + 1.28134e-3 * x0_2 * x1
        - 3.5256e-5 * x0_3 * x1
        + 2.266e-7 * x0_4 * x1
        - 0.25645 * x1_2
        + 3.4604e-3 * x1_3
        - 1.3514e-5 * x1_4
        + 28.106 / (x1 + 1.0)
        + 5.2375e-6 * x0_2 * x1_2
        + 6.3e-8 * x0_3 * x1_2
        - 7.0e-10 * x0_3 * x1_3
        - 3.4054e-4 * x0 * x1_2
        + 1.6638e-6 * x0 * x1_3
        + 2.8673 * (5.0e-4 * x0 * x1).exp()
}

pub(super) fn tp236_239_symbolic_objective(x0: SX, x1: SX) -> SX {
    let b = [
        75.1963666677,
        -3.8112755343,
        0.1269366345,
        -2.0567665e-3,
        1.0345e-5,
        -6.8306567613,
        3.02344793e-2,
        -1.2813448e-3,
        3.52559e-5,
        -2.266e-7,
        0.2564581253,
        -3.460403e-3,
        1.35139e-5,
        -28.1064434908,
        -5.2375e-6,
        -6.3e-9,
        7.0e-10,
        3.405462e-4,
        -1.6638e-6,
        -2.8673112392,
    ];
    let x0_2 = x0.sqr();
    let x0_3 = x0_2 * x0;
    let x0_4 = x0_3 * x0;
    let x1_2 = x1.sqr();
    let x1_3 = x1_2 * x1;
    let x1_4 = x1_3 * x1;
    -(b[0]
        + b[1] * x0
        + b[2] * x0_2
        + b[3] * x0_3
        + b[4] * x0_4
        + b[5] * x1
        + b[6] * x0 * x1
        + b[7] * x0_2 * x1
        + b[8] * x0_3 * x1
        + b[9] * x0_4 * x1
        + b[10] * x1_2
        + b[11] * x1_3
        + b[12] * x1_4
        + b[13] / (x1 + 1.0)
        + b[14] * x0_2 * x1_2
        + b[15] * x0_3 * x1_2
        + b[16] * x0_3 * x1_3
        + b[17] * x0 * x1_2
        + b[18] * x0 * x1_3
        + b[19] * (5.0e-4 * x0 * x1).exp())
}

pub(super) fn tp067_symbolic_state(x0: SX, x1: SX, x2: SX) -> [SX; 7] {
    let rx = 1.0 / x0;
    let mut y2 = 1.6 * x0;
    for _ in 0..20 {
        let y3 = 1.22 * y2 - x0;
        let y6 = (x1 + y3) * rx;
        let v2 = (112.0 + (13.167 - 0.6667 * y6) * y6) * 0.01;
        y2 = x0 * v2;
    }

    let y3 = 1.22 * y2 - x0;
    let y6 = (x1 + y3) * rx;
    let mut y4 = SX::from(93.0);
    for _ in 0..20 {
        let y5 = 86.35 + 1.098 * y6 - 0.038 * y6.sqr() + 0.325 * (y4 - 89.0);
        let y8 = -133.0 + 3.0 * y5;
        let y7 = 35.82 - 0.222 * y8;
        y4 = 9.8e4 * x2 / (y2 * y7 + 1.0e3 * x2);
    }

    let y5 = 86.35 + 1.098 * y6 - 0.038 * y6.sqr() + 0.325 * (y4 - 89.0);
    let y8 = -133.0 + 3.0 * y5;
    let y7 = 35.82 - 0.222 * y8;
    [y2, y3, y4, y5, y6, y7, y8]
}

pub(super) fn rosenbrock_objective(x: SX, y: SX) -> SX {
    100.0 * (y - x.sqr()).sqr() + (1.0 - x).sqr()
}

pub(super) fn scalar_inequality_upper_bound() -> TypedRuntimeNlpBounds<Pair<SX>, SX> {
    TypedRuntimeNlpBounds {
        variable_lower: None,
        variable_upper: None,
        inequality_lower: None,
        inequality_upper: Some(Some(0.0)),
        scaling: None,
    }
}

pub(super) fn scalar_inequality_upper_bound_for_vec3() -> TypedRuntimeNlpBounds<VecN<SX, 3>, SX> {
    TypedRuntimeNlpBounds {
        variable_lower: None,
        variable_upper: None,
        inequality_lower: None,
        inequality_upper: Some(Some(0.0)),
        scaling: None,
    }
}

pub(super) fn inequality_upper_bounds<const N: usize>() -> Option<VecN<Option<f64>, N>> {
    Some(VecN {
        values: [Some(0.0); N],
    })
}

pub(super) fn tp002_solution() -> [f64; 2] {
    let w1 = (598.0_f64 / 1200.0).sqrt();
    let angle = (0.0025 / w1.powi(3)).acos() / 3.0;
    let x = 2.0 * w1 * angle.cos();
    [x, 1.5]
}

pub(super) fn tp002_objective() -> f64 {
    let [x, y] = tp002_solution();
    100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
}

pub(super) fn tp005_solution() -> [f64; 2] {
    let x = 0.5 - std::f64::consts::PI / 3.0;
    [x, x - 1.0]
}

pub(super) fn tp011_solution() -> [f64; 2] {
    let aex = 7.5 * 6.0_f64.sqrt();
    let aw = ((aex * aex + 1.0).sqrt() + aex).powf(1.0 / 3.0);
    let qaw = aw * aw;
    [
        (aw - 1.0 / aw) / 6.0_f64.sqrt(),
        (qaw - 2.0 + 1.0 / qaw) / 6.0,
    ]
}

pub(super) fn tp011_objective() -> f64 {
    let [x, y] = tp011_solution();
    (x - 5.0).powi(2) + y.powi(2) - 25.0
}

pub(super) fn tp014_solution() -> [f64; 2] {
    let w7 = 7.0_f64.sqrt();
    [(w7 - 1.0) * 0.5, (w7 + 1.0) * 0.25]
}

pub(super) fn tp018_solution() -> [f64; 2] {
    let x = 250.0_f64.sqrt();
    [x, 0.1 * x]
}

pub(super) fn tp019_solution() -> [f64; 2] {
    let aex = 17.280975_f64.sqrt();
    [14.095, 5.0 - aex]
}

pub(super) fn tp019_objective() -> f64 {
    let [x, y] = tp019_solution();
    (x - 10.0).powi(3) + (y - 20.0).powi(3)
}

pub(super) fn normal_cdf_approx(x: SX) -> SX {
    // Smooth symbolic Abramowitz-Stegun style approximation; used for TP068/TP069
    // because SX does not currently expose erf/erfc.
    let sign = x / (x.abs() + 1.0e-12);
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * z);
    let poly = (((1.330_274_429 * t - 1.821_255_978) * t + 1.781_477_937) * t - 0.356_563_782) * t
        + 0.319_381_530;
    let pdf = 0.398_942_280_401_432_7 * (-0.5 * z.sqr()).exp();
    let cdf_pos = 1.0 - pdf * poly * t;
    0.5 * (1.0 + sign * (2.0 * cdf_pos - 1.0))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tp068_069_case(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    k: usize,
    x0_lower: f64,
    x0_upper: f64,
    x3_lower: f64,
    fex: f64,
) -> ProblemCase {
    let a = [1.0e-4, 0.1][k];
    let b = [1.0, 1.0e3][k];
    let z: f64 = [24.0, 4.0][k];
    make_typed_case::<VecN<SX, 4>, (), VecN<SX, 2>, (), _, _>(
        metadata(id, family, description),
        move |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), VecN<SX, 2>, (), _>(
                id,
                move |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    let v1 = x0.exp() - 1.0;
                    SymbolicNlpOutputs {
                        objective: (a * z - x3 * (b * v1 - x2) / (v1 + x3)) / x0,
                        equalities: VecN {
                            values: [
                                x2 - 2.0 * normal_cdf_approx(-x1),
                                x3 - normal_cdf_approx(-x1 + z.sqrt())
                                    - normal_cdf_approx(-x1 - z.sqrt()),
                            ],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [1.0; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [Some(x0_lower), Some(0.0), Some(0.0), Some(x3_lower)],
                    }),
                    variable_upper: Some(VecN {
                        values: [Some(x0_upper), Some(100.0), Some(2.0), Some(2.0)],
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        objective_validation(fex, 1e-8, PRIMAL_TOL, DUAL_TOL, None),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn objective_only_case<const N: usize, const NE: usize, const NI: usize, F>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    x0: [f64; N],
    lower: [Option<f64>; N],
    upper: [Option<f64>; N],
    fex: f64,
    build: F,
) -> ProblemCase
where
    F: Fn(VecN<SX, N>) -> SymbolicNlpOutputs<VecN<SX, NE>, VecN<SX, NI>>
        + Copy
        + Send
        + Sync
        + 'static,
{
    make_typed_case::<VecN<SX, N>, (), VecN<SX, NE>, VecN<SX, NI>, _, _>(
        metadata(id, family, description),
        move |options| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), VecN<SX, NE>, VecN<SX, NI>, _>(
                id,
                move |x, ()| build(x.clone()),
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: x0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN { values: lower }),
                    variable_upper: Some(VecN { values: upper }),
                    inequality_lower: None,
                    inequality_upper: Some(VecN {
                        values: [Some(0.0); NI],
                    }),
                    scaling: None,
                },
            })
        },
        objective_validation(fex, 1e-8, PRIMAL_TOL, DUAL_TOL, Some(COMPLEMENTARITY_TOL)),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn objective_only_case_no_ineq<const N: usize, const NE: usize, F>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    x0: [f64; N],
    lower: [Option<f64>; N],
    upper: [Option<f64>; N],
    fex: f64,
    build: F,
) -> ProblemCase
where
    F: Fn(VecN<SX, N>) -> SymbolicNlpOutputs<VecN<SX, NE>, ()> + Copy + Send + Sync + 'static,
{
    make_typed_case::<VecN<SX, N>, (), VecN<SX, NE>, (), _, _>(
        metadata(id, family, description),
        move |options| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), VecN<SX, NE>, (), _>(
                id,
                move |x, ()| build(x.clone()),
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: x0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN { values: lower }),
                    variable_upper: Some(VecN { values: upper }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        objective_validation(fex, 1e-8, PRIMAL_TOL, DUAL_TOL, None),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn nonsmooth_objective_only_case_no_ineq<const N: usize, const NE: usize, F>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    x0: [f64; N],
    lower: [Option<f64>; N],
    upper: [Option<f64>; N],
    fex: f64,
    build: F,
) -> ProblemCase
where
    F: Fn(VecN<SX, N>) -> SymbolicNlpOutputs<VecN<SX, NE>, ()> + Copy + Send + Sync + 'static,
{
    make_typed_case::<VecN<SX, N>, (), VecN<SX, NE>, (), _, _>(
        metadata(id, family, description),
        move |options| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), VecN<SX, NE>, (), _>(
                id,
                move |x, ()| build(x.clone()),
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: x0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN { values: lower }),
                    variable_upper: Some(VecN { values: upper }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        objective_value_validation(fex, 1e-8, 1e-6, PRIMAL_TOL),
    )
}

pub(super) fn tp284_285_case(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    coefficients: [f64; 15],
) -> ProblemCase {
    const B: [f64; 10] = [
        385.0, 470.0, 560.0, 565.0, 645.0, 430.0, 485.0, 455.0, 390.0, 460.0,
    ];
    const A_FLAT: [f64; 150] = [
        100.0, 90.0, 70.0, 50.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 100.0, 100.0, 50.0, 0.0, 10.0,
        0.0, 60.0, 30.0, 70.0, 10.0, 10.0, 10.0, 0.0, 0.0, 70.0, 50.0, 30.0, 40.0, 10.0, 100.0,
        5.0, 35.0, 55.0, 65.0, 60.0, 95.0, 90.0, 25.0, 35.0, 5.0, 10.0, 20.0, 25.0, 35.0, 45.0,
        50.0, 0.0, 40.0, 25.0, 20.0, 0.0, 5.0, 100.0, 100.0, 45.0, 35.0, 30.0, 25.0, 65.0, 5.0,
        0.0, 0.0, 40.0, 35.0, 0.0, 10.0, 5.0, 15.0, 0.0, 10.0, 25.0, 35.0, 50.0, 60.0, 35.0, 60.0,
        25.0, 10.0, 30.0, 35.0, 0.0, 55.0, 0.0, 0.0, 65.0, 0.0, 0.0, 80.0, 0.0, 95.0, 10.0, 25.0,
        30.0, 15.0, 5.0, 45.0, 70.0, 20.0, 0.0, 70.0, 55.0, 20.0, 60.0, 0.0, 75.0, 15.0, 20.0,
        30.0, 25.0, 20.0, 5.0, 0.0, 10.0, 75.0, 100.0, 20.0, 25.0, 30.0, 0.0, 10.0, 45.0, 40.0,
        30.0, 35.0, 75.0, 0.0, 70.0, 5.0, 15.0, 35.0, 20.0, 25.0, 0.0, 30.0, 10.0, 5.0, 15.0, 65.0,
        50.0, 10.0, 0.0, 10.0, 40.0, 65.0, 0.0, 5.0, 15.0, 20.0, 55.0, 30.0,
    ];

    let fex = -coefficients.iter().sum::<f64>();
    objective_only_case::<15, 0, 10, _>(
        id,
        family,
        description,
        [0.0; 15],
        [None; 15],
        [None; 15],
        fex,
        move |x| {
            let mut objective = SX::zero();
            for (j, coefficient) in coefficients.iter().enumerate() {
                objective -= *coefficient * x.values[j];
            }
            let inequalities: [SX; 10] = std::array::from_fn(|i| {
                let mut used_capacity = SX::zero();
                for j in 0..15 {
                    used_capacity += A_FLAT[j * 10 + i] * x.values[j].sqr();
                }
                used_capacity - B[i]
            });
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: inequalities,
                },
            }
        },
    )
}

pub(super) fn tp290_293_case<const N: usize>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
) -> ProblemCase {
    objective_only_case_no_ineq(
        id,
        family,
        description,
        [1.0; N],
        [None; N],
        [None; N],
        0.0,
        |x| {
            let mut residual = SX::zero();
            for i in 0..N {
                residual += (i + 1) as f64 * x.values[i].sqr();
            }
            SymbolicNlpOutputs {
                objective: residual.sqr(),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}

pub(super) fn tp294_299_case<const N: usize>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
) -> ProblemCase {
    let x0 = std::array::from_fn(|i| if (i + 1) % 2 == 0 { 1.0 } else { -1.2 });
    objective_only_case_no_ineq(
        id,
        family,
        description,
        x0,
        [None; N],
        [None; N],
        0.0,
        |x| {
            let mut objective = SX::zero();
            for i in 0..(N - 1) {
                let f0 = 10.0 * (x.values[i + 1] - x.values[i].sqr());
                let f1 = 1.0 - x.values[i];
                objective += f0.sqr() + f1.sqr();
            }
            SymbolicNlpOutputs {
                objective: objective * 1.0e-4,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}

pub(super) fn tp300_302_case<const N: usize>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
) -> ProblemCase {
    objective_only_case_no_ineq(
        id,
        family,
        description,
        [0.0; N],
        [None; N],
        [None; N],
        -(N as f64),
        |x| {
            let mut objective = x.values[0].sqr() - 2.0 * x.values[0];
            for i in 1..N {
                objective += 2.0 * x.values[i].sqr() - 2.0 * x.values[i - 1] * x.values[i];
            }
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}

pub(super) fn tp303_305_case<const N: usize>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
) -> ProblemCase {
    objective_only_case_no_ineq(
        id,
        family,
        description,
        [0.1; N],
        [None; N],
        [None; N],
        0.0,
        |x| {
            let mut pom = SX::zero();
            let mut norm2 = SX::zero();
            for i in 0..N {
                pom += 0.5 * (i + 1) as f64 * x.values[i];
                norm2 += x.values[i].sqr();
            }
            SymbolicNlpOutputs {
                objective: norm2 + pom.sqr() + pom.powi(4),
                equalities: VecN { values: [] },
                inequalities: (),
            }
        },
    )
}

#[allow(
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]
pub(super) fn tp088_092_case<const N: usize>(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    fex: f64,
) -> ProblemCase {
    make_typed_case::<VecN<SX, N>, (), (), SX, _, _>(
        metadata(id, family, description),
        move |options| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), (), SX, _>(
                id,
                move |x, ()| {
                    let mut objective = SX::zero();
                    for i in 0..N {
                        objective += x.values[i].sqr();
                    }
                    let mue: [f64; 30] = [
                        0.86033358901937973,
                        3.425618459481707,
                        6.4372981791719468,
                        9.5293344053619631,
                        12.645287223856643,
                        15.771284874815882,
                        18.902409956860023,
                        22.036496727938001,
                        25.172446326646487,
                        28.309642854451948,
                        31.447714637546206,
                        34.586424215288908,
                        37.725612827776494,
                        40.86517033048807,
                        44.005017920830838,
                        47.145097736761024,
                        50.285366337773645,
                        53.425790477394671,
                        56.566344279821514,
                        59.707007305335452,
                        62.847763194454451,
                        65.988598698490392,
                        69.129502973895256,
                        72.27046706030896,
                        75.411483488848148,
                        78.552545984242926,
                        81.693649235601697,
                        84.834788718042276,
                        87.975960552493206,
                        91.117161394464745,
                    ];
                    let mut t: [SX; N] = std::array::from_fn(|_| SX::zero());
                    t[N - 1] = x.values[N - 1].sqr();
                    for ii in 1..N {
                        let i = N - 1 - ii;
                        t[i] = t[i + 1] + x.values[i].sqr();
                    }
                    let mut v = SX::zero();
                    for j in 0..30 {
                        let mu: f64 = mue[j];
                        let a = 2.0 * mu.sin() / (mu + mu.sin() * mu.cos());
                        let dcosko = (mu.sin() / mu - mu.cos()) / (mu * mu);
                        let v3 = -(mu * mu);
                        let mut rho = if N % 2 == 0 {
                            SX::from(1.0)
                        } else {
                            SX::from(-1.0)
                        };
                        for i in 1..N {
                            let sign = if (N - i) % 2 == 0 { 1.0 } else { -1.0 };
                            rho += 2.0 * sign * (v3 * t[N - i]).exp();
                        }
                        rho = (rho + (v3 * t[0]).exp()) / v3;
                        v -= v3 * a * rho * (mu * mu.sin() * rho - 2.0 * dcosko);
                    }
                    SymbolicNlpOutputs {
                        objective,
                        equalities: (),
                        inequalities: 1.0e-4 - v - 2.0 / 15.0,
                    }
                },
                options,
            )?;
            let mut start = [0.0; N];
            for i in 0..N {
                start[i] = if i % 2 == 0 { 0.5 } else { -0.5 };
            }
            let mut lo = [Some(-10.0); N];
            lo[0] = Some(0.1);
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: start },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN { values: lo }),
                    variable_upper: Some(VecN {
                        values: [Some(10.0); N],
                    }),
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        objective_validation(fex, 1e-8, PRIMAL_TOL, DUAL_TOL, Some(COMPLEMENTARITY_TOL)),
    )
}

pub(super) fn tp095_098_case(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    b: [f64; 4],
    fex: f64,
) -> ProblemCase {
    objective_only_case(
        id,
        family,
        description,
        [0.0; 6],
        [Some(0.0); 6],
        [
            Some(0.31),
            Some(0.046),
            Some(0.068),
            Some(0.042),
            Some(0.028),
            Some(0.0134),
        ],
        fex,
        move |x| {
            let [x0, x1, x2, x3, x4, x5] = x.values;
            SymbolicNlpOutputs {
                objective: 4.3 * x0 + 31.8 * x1 + 63.3 * x2 + 15.8 * x3 + 68.5 * x4 + 4.7 * x5,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [
                        -(17.1 * x0
                            + 38.2 * x1
                            + 204.2 * x2
                            + 212.3 * x3
                            + 623.4 * x4
                            + 1495.5 * x5
                            - 169.0 * x0 * x2
                            - 3580.0 * x2 * x4
                            - 3810.0 * x3 * x4
                            - 18500.0 * x3 * x5
                            - 24300.0 * x4 * x5
                            - b[0]),
                        -(17.9 * x0
                            + 36.8 * x1
                            + 113.9 * x2
                            + 169.7 * x3
                            + 337.8 * x4
                            + 1385.2 * x5
                            - 139.0 * x0 * x2
                            - 2450.0 * x3 * x4
                            - 16600.0 * x3 * x5
                            - 17200.0 * x4 * x5
                            - b[1]),
                        -(-273.0 * x1 - 70.0 * x3 - 819.0 * x4 + 26000.0 * x3 * x4 - b[2]),
                        -(159.9 * x0 - 311.0 * x1 + 587.0 * x3 + 391.0 * x4 + 2198.0 * x5
                            - 14000.0 * x.values[0] * x5
                            - b[3]),
                    ],
                },
            }
        },
    )
}

pub(super) fn tp101_103_case(
    id: &'static str,
    family: &'static str,
    description: &'static str,
    a7: f64,
    fex: f64,
) -> ProblemCase {
    objective_only_case(
        id,
        family,
        description,
        [6.0; 7],
        [
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.01),
        ],
        [Some(10.0); 7],
        fex,
        move |x| {
            let [x1, x2, x3, x4, x5, x6, x7] = x.values;
            let term1 = 10.0 * x1 * x4.powi(2) * x7.powf(a7) / (x2 * x6.powi(3));
            let term2 = 15.0 * x3 * x4 / (x1 * x2.powi(2) * x5 * x7.sqrt());
            let term3 = 20.0 * x2 * x6 / (x1.powi(2) * x4 * x5.powi(2));
            let term4 = 25.0 * x1.powi(2) * x2.powi(2) * x5.sqrt() * x7 / (x3 * x6.powi(2));
            let objective = term1 + term2 + term3 + term4;
            let g1 = 1.0
                - 0.5 * x1.abs().sqrt() * x7 / (x3 * x6.powi(2))
                - 0.7 * x1.powi(3) * x2 * x6 * x7.abs().sqrt() / x3.powi(2)
                - 0.2 * x3 * x6.abs().powf(2.0 / 3.0) * x7.abs().powf(0.25)
                    / (x2 * x4.abs().sqrt());
            let g2 = 1.0
                - 1.3 * x2 * x6 / (x1.abs().sqrt() * x3 * x5)
                - 0.8 * x3 * x6.powi(2) / (x4 * x5)
                - 3.1 * x2.abs().sqrt() * x6.abs().powf(1.0 / 3.0) / (x1 * x4.powi(2) * x5);
            let g3 = 1.0
                - 2.0 * x1 * x5 * x7.abs().powf(1.0 / 3.0) / (x3.abs().powf(1.5) * x6)
                - 0.1 * x2 * x5 / ((x3 * x7).abs().sqrt() * x6)
                - x2 * x3.abs().sqrt() * x5 / x1
                - 0.65 * x3 * x5 * x7 / (x2.powi(2) * x6);
            let g4 = 1.0
                - 0.2 * x2 * x5.abs().sqrt() * x7.abs().powf(1.0 / 3.0) / (x1.powi(2) * x4)
                - 0.3
                    * x1.abs().sqrt()
                    * x2.powi(2)
                    * x3
                    * x4.abs().powf(1.0 / 3.0)
                    * x7.abs().powf(0.25)
                    / x5.abs().powf(2.0 / 3.0)
                - 0.4 * x3 * x5 * x7.abs().powf(0.75) / (x1.powi(3) * x2.powi(2))
                - 0.5 * x4 * x7.abs().sqrt() / x3.powi(2);
            SymbolicNlpOutputs {
                objective,
                equalities: VecN { values: [] },
                inequalities: VecN {
                    values: [-g1, -g2, -g3, -g4, 100.0 - objective, objective - 3000.0],
                },
            }
        },
    )
}
