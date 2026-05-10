use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

// Hand-written ports of John Burkardt's MIT-licensed test_nonlin systems:
// https://people.sc.fsu.edu/~jburkardt/m_src/test_nonlin/test_nonlin.html
use super::{
    CaseMetadata, ProblemCase, TypedProblemData, VecN, make_typed_case, objective_validation,
    symbolic_compile,
};
use crate::manifest::ProblemTestSet;

const SOURCE: &str = "burkardt_test_nonlin";
const OBJECTIVE_TOL: f64 = 1e-8;
const PRIMAL_TOL: f64 = 1e-9;
const DUAL_TOL: f64 = 1e-6;

type ResidualBuilder<const N: usize> = fn(&[SX; N]) -> [SX; N];
type StartBuilder<const N: usize> = fn() -> [f64; N];

macro_rules! three_size_cases {
    ($prefix:literal, $family:literal, $description:literal, $residuals:path, $start:path, $min:literal, $min_suffix:literal) => {
        vec![
            case_for::<$min>(
                concat!($prefix, $min_suffix),
                $family,
                concat!("n=", stringify!($min)),
                $description,
                $residuals,
                $start,
            ),
            case_for::<10>(
                concat!($prefix, "_n10"),
                $family,
                "n=10",
                $description,
                $residuals,
                $start,
            ),
            case_for::<20>(
                concat!($prefix, "_n20"),
                $family,
                "n=20",
                $description,
                $residuals,
                $start,
            ),
        ]
    };
}

pub(crate) fn cases() -> Vec<ProblemCase> {
    let mut cases = Vec::new();
    cases.extend([
        case_for::<2>(
            "burkardt_p01_generalized_rosenbrock_n02",
            "p01_generalized_rosenbrock",
            "n=2",
            "Generalized Rosenbrock nonlinear equations",
            p01_residuals,
            p01_start,
        ),
        case_for::<10>(
            "burkardt_p01_generalized_rosenbrock_n10",
            "p01_generalized_rosenbrock",
            "n=10",
            "Generalized Rosenbrock nonlinear equations",
            p01_residuals,
            p01_start,
        ),
        case_for::<20>(
            "burkardt_p01_generalized_rosenbrock_n20",
            "p01_generalized_rosenbrock",
            "n=20",
            "Generalized Rosenbrock nonlinear equations",
            p01_residuals,
            p01_start,
        ),
        case_for::<4>(
            "burkardt_p02_powell_singular_n04",
            "p02_powell_singular",
            "n=4",
            "Powell singular nonlinear equations",
            p02_residuals,
            || [3.0, -1.0, 0.0, 1.0],
        ),
        case_for::<2>(
            "burkardt_p03_powell_badly_scaled_n02",
            "p03_powell_badly_scaled",
            "n=2",
            "Powell badly scaled nonlinear equations",
            p03_residuals,
            || [0.0, 1.0],
        ),
        case_for::<4>(
            "burkardt_p04_wood_n04",
            "p04_wood",
            "n=4",
            "Wood nonlinear equations",
            p04_residuals,
            || [-3.0, -1.0, -3.0, -1.0],
        ),
        case_for::<3>(
            "burkardt_p05_helical_valley_n03",
            "p05_helical_valley",
            "n=3",
            "Helical valley nonlinear equations",
            p05_residuals,
            || [-1.0, 0.0, 0.0],
        ),
    ]);
    cases.extend(three_size_cases!(
        "burkardt_p06_watson",
        "p06_watson",
        "Watson nonlinear equations",
        p06_residuals,
        p06_start,
        2,
        "_n02"
    ));
    cases.extend([
        case_for::<1>(
            "burkardt_p07_chebyquad_n01",
            "p07_chebyquad",
            "n=1",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<2>(
            "burkardt_p07_chebyquad_n02",
            "p07_chebyquad",
            "n=2",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<3>(
            "burkardt_p07_chebyquad_n03",
            "p07_chebyquad",
            "n=3",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<4>(
            "burkardt_p07_chebyquad_n04",
            "p07_chebyquad",
            "n=4",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<5>(
            "burkardt_p07_chebyquad_n05",
            "p07_chebyquad",
            "n=5",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<6>(
            "burkardt_p07_chebyquad_n06",
            "p07_chebyquad",
            "n=6",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<7>(
            "burkardt_p07_chebyquad_n07",
            "p07_chebyquad",
            "n=7",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
        case_for::<9>(
            "burkardt_p07_chebyquad_n09",
            "p07_chebyquad",
            "n=9",
            "Chebyquad nonlinear equations",
            p07_residuals,
            p07_start,
        ),
    ]);
    cases.extend(three_size_cases!(
        "burkardt_p08_brown_almost_linear",
        "p08_brown_almost_linear",
        "Brown almost linear nonlinear equations",
        p08_residuals,
        p08_start,
        1,
        "_n01"
    ));
    cases.extend(three_size_cases!(
        "burkardt_p09_discrete_boundary_value",
        "p09_discrete_boundary_value",
        "Discrete boundary value nonlinear equations",
        p09_residuals,
        p09_start,
        1,
        "_n01"
    ));
    cases.extend(three_size_cases!(
        "burkardt_p10_discrete_integral_equation",
        "p10_discrete_integral_equation",
        "Discrete integral equation nonlinear equations",
        p10_residuals,
        p10_start,
        1,
        "_n01"
    ));
    cases.extend(three_size_cases!(
        "burkardt_p11_trigonometric",
        "p11_trigonometric",
        "Trigonometric nonlinear equations",
        p11_residuals,
        p11_start,
        1,
        "_n01"
    ));
    cases.extend(three_size_cases!(
        "burkardt_p12_variably_dimensioned",
        "p12_variably_dimensioned",
        "Variably dimensioned nonlinear equations",
        p12_residuals,
        p12_start,
        1,
        "_n01"
    ));
    cases.extend(three_size_cases!(
        "burkardt_p13_broyden_tridiagonal",
        "p13_broyden_tridiagonal",
        "Broyden tridiagonal nonlinear equations",
        p13_residuals,
        p13_start,
        1,
        "_n01"
    ));
    cases.extend(three_size_cases!(
        "burkardt_p14_broyden_banded",
        "p14_broyden_banded",
        "Broyden banded nonlinear equations",
        p14_residuals,
        p14_start,
        1,
        "_n01"
    ));
    cases.extend([
        case_for::<4>(
            "burkardt_p15_hammarling_2x2_n04",
            "p15_hammarling_2x2",
            "n=4",
            "Hammarling 2 by 2 matrix square root nonlinear equations",
            p15_residuals,
            || [1.0, 0.0, 0.0, 1.0],
        ),
        case_for::<9>(
            "burkardt_p16_hammarling_3x3_n09",
            "p16_hammarling_3x3",
            "n=9",
            "Hammarling 3 by 3 matrix square root nonlinear equations",
            p16_residuals,
            || [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ),
        case_for::<2>(
            "burkardt_p17_dennis_schnabel_n02",
            "p17_dennis_schnabel",
            "n=2",
            "Dennis and Schnabel nonlinear equations",
            p17_residuals,
            || [1.0, 5.0],
        ),
        case_for::<2>(
            "burkardt_p18_sample18_n02",
            "p18_sample18",
            "n=2",
            "Burkardt sample problem 18 nonlinear equations",
            p18_residuals,
            || [2.0, 2.0],
        ),
        case_for::<2>(
            "burkardt_p19_sample19_n02",
            "p19_sample19",
            "n=2",
            "Burkardt sample problem 19 nonlinear equations",
            p19_residuals,
            || [3.0, 3.0],
        ),
        case_for::<1>(
            "burkardt_p20_scalar_n01",
            "p20_scalar",
            "n=1",
            "Scalar nonlinear equation x * (x - 5)^2",
            p20_residuals,
            || [1.0],
        ),
        case_for::<2>(
            "burkardt_p21_freudenstein_roth_n02",
            "p21_freudenstein_roth",
            "n=2",
            "Freudenstein-Roth nonlinear equations",
            p21_residuals,
            || [0.5, -2.0],
        ),
        case_for::<2>(
            "burkardt_p22_boggs_n02",
            "p22_boggs",
            "n=2",
            "Boggs nonlinear equations",
            p22_residuals,
            || [1.0, 0.0],
        ),
    ]);
    cases.extend(three_size_cases!(
        "burkardt_p23_chandrasekhar",
        "p23_chandrasekhar",
        "Chandrasekhar H-function nonlinear equations",
        p23_residuals,
        p23_start,
        1,
        "_n01"
    ));
    cases
}

fn case_for<const N: usize>(
    id: &'static str,
    family: &'static str,
    variant: &'static str,
    description: &'static str,
    residuals: ResidualBuilder<N>,
    start: StartBuilder<N>,
) -> ProblemCase {
    make_typed_case::<VecN<SX, N>, (), VecN<SX, N>, (), _, _>(
        CaseMetadata::new(id, family, variant, SOURCE, description, false)
            .with_test_set(ProblemTestSet::BurkardtTestNonlin),
        move |options| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), VecN<SX, N>, (), _>(
                id,
                move |x, ()| SymbolicNlpOutputs {
                    objective: SX::zero(),
                    equalities: VecN {
                        values: residuals(&x.values),
                    },
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: start() },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(0.0, OBJECTIVE_TOL, PRIMAL_TOL, DUAL_TOL, None),
    )
}

fn p01_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let mut residuals: [SX; N] = std::array::from_fn(|_| SX::zero());
    residuals[0] = 1.0 - x[0];
    for idx in 1..N {
        residuals[idx] = 10.0 * (x[idx] - x[idx - 1].sqr());
    }
    residuals
}

fn p01_start<const N: usize>() -> [f64; N] {
    std::array::from_fn(|idx| if idx == 0 { -1.2 } else { 1.0 })
}

fn p02_residuals(x: &[SX; 4]) -> [SX; 4] {
    [
        x[0] + 10.0 * x[1],
        5.0_f64.sqrt() * (x[2] - x[3]),
        (x[1] - 2.0 * x[2]).sqr(),
        10.0_f64.sqrt() * (x[0] - x[3]).sqr(),
    ]
}

fn p03_residuals(x: &[SX; 2]) -> [SX; 2] {
    [
        10_000.0 * x[0] * x[1] - 1.0,
        (-x[0]).exp() + (-x[1]).exp() - 1.0001,
    ]
}

fn p04_residuals(x: &[SX; 4]) -> [SX; 4] {
    let temp1 = x[1] - x[0].sqr();
    let temp2 = x[3] - x[2].sqr();
    [
        -200.0 * x[0] * temp1 - (1.0 - x[0]),
        200.0 * temp1 + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0),
        -180.0 * x[2] * temp2 - (1.0 - x[2]),
        180.0 * temp2 + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1.0),
    ]
}

fn p05_residuals(x: &[SX; 3]) -> [SX; 3] {
    let temp = x[1].atan2(x[0]) / (2.0 * std::f64::consts::PI);
    [
        10.0 * (x[2] - 10.0 * temp),
        10.0 * ((x[0].sqr() + x[1].sqr()).sqrt() - 1.0),
        x[2],
    ]
}

fn p06_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let mut residuals: [SX; N] = std::array::from_fn(|_| SX::zero());
    for i in 1..=29 {
        let ti = i as f64 / 29.0;
        let mut sum1 = SX::zero();
        let mut temp = 1.0;
        for j in 2..=N {
            sum1 += (j - 1) as f64 * temp * x[j - 1];
            temp *= ti;
        }

        let mut sum2 = SX::zero();
        temp = 1.0;
        for j in 1..=N {
            sum2 += temp * x[j - 1];
            temp *= ti;
        }

        temp = 1.0 / ti;
        for k in 1..=N {
            residuals[k - 1] +=
                temp * (sum1 - sum2.sqr() - 1.0) * ((k - 1) as f64 - 2.0 * ti * sum2);
            temp *= ti;
        }
    }
    residuals[0] += 3.0 * x[0] - 2.0 * x[0] * x[1] + 2.0 * x[0].powi(3);
    residuals[1] += x[1] - x[0].sqr() - 1.0;
    residuals
}

fn p06_start<const N: usize>() -> [f64; N] {
    [0.0; N]
}

fn p07_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let mut residuals: [SX; N] = std::array::from_fn(|_| SX::zero());
    for value in x {
        let mut t1 = SX::from(1.0);
        let mut t2 = *value;
        for residual in &mut residuals {
            *residual += t2;
            let t3 = 2.0 * *value * t2 - t1;
            t1 = t2;
            t2 = t3;
        }
    }
    for (idx, residual) in residuals.iter_mut().enumerate() {
        let i = idx + 1;
        *residual /= SX::from(N as f64);
        if i % 2 == 0 {
            *residual += SX::from(1.0 / (i * i - 1) as f64);
        }
    }
    residuals
}

fn p07_start<const N: usize>() -> [f64; N] {
    std::array::from_fn(|idx| (2 * (idx + 1) - 1) as f64 - N as f64)
        .map(|value| value / (N + 1) as f64)
}

fn p08_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let sum = x.iter().fold(SX::zero(), |acc, value| acc + *value);
    let product = x.iter().fold(SX::from(1.0), |acc, value| acc * *value);
    std::array::from_fn(|idx| {
        if idx + 1 == N {
            product - 1.0
        } else {
            x[idx] + sum - (N + 1) as f64
        }
    })
}

fn p08_start<const N: usize>() -> [f64; N] {
    [0.5; N]
}

fn p09_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let h = 1.0 / (N + 1) as f64;
    std::array::from_fn(|idx| {
        let k = idx + 1;
        let mut residual = 2.0 * x[idx] + 0.5 * h * h * (x[idx] + k as f64 * h + 1.0).powi(3);
        if idx > 0 {
            residual -= x[idx - 1];
        }
        if idx + 1 < N {
            residual -= x[idx + 1];
        }
        residual
    })
}

fn p09_start<const N: usize>() -> [f64; N] {
    std::array::from_fn(|idx| {
        let i = (idx + 1) as f64;
        let n1 = (N + 1) as f64;
        i * (i - n1) / (n1 * n1)
    })
}

fn p10_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let h = 1.0 / (N + 1) as f64;
    std::array::from_fn(|idx| {
        let k = idx + 1;
        let tk = k as f64 / (N + 1) as f64;
        let mut sum1 = SX::zero();
        for j in 1..=k {
            let tj = j as f64 * h;
            sum1 += tj * (x[j - 1] + tj + 1.0).powi(3);
        }
        let mut sum2 = SX::zero();
        for j in (k + 1)..=N {
            let tj = j as f64 * h;
            sum2 += (1.0 - tj) * (x[j - 1] + tj + 1.0).powi(3);
        }
        x[idx] + h * ((1.0 - tk) * sum1 + tk * sum2) / 2.0
    })
}

fn p10_start<const N: usize>() -> [f64; N] {
    p09_start::<N>()
}

fn p11_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let cos_sum = x.iter().fold(SX::zero(), |acc, value| acc + value.cos());
    std::array::from_fn(|idx| {
        let k = (idx + 1) as f64;
        N as f64 - cos_sum + k * (1.0 - x[idx].cos()) - x[idx].sin()
    })
}

fn p11_start<const N: usize>() -> [f64; N] {
    [1.0 / N as f64; N]
}

fn p12_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let sum1 = x.iter().enumerate().fold(SX::zero(), |acc, (idx, value)| {
        acc + (idx + 1) as f64 * (*value - 1.0)
    });
    std::array::from_fn(|idx| x[idx] - 1.0 + (idx + 1) as f64 * sum1 * (1.0 + 2.0 * sum1.sqr()))
}

fn p12_start<const N: usize>() -> [f64; N] {
    std::array::from_fn(|idx| 1.0 - (idx + 1) as f64 / N as f64)
}

fn p13_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    std::array::from_fn(|idx| {
        let mut residual = (3.0 - 2.0 * x[idx]) * x[idx] + 1.0;
        if idx > 0 {
            residual -= x[idx - 1];
        }
        if idx + 1 < N {
            residual -= 2.0 * x[idx + 1];
        }
        residual
    })
}

fn p13_start<const N: usize>() -> [f64; N] {
    [-1.0; N]
}

fn p14_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let ml = 5usize;
    let mu = 1usize;
    std::array::from_fn(|idx| {
        let k1 = idx.saturating_sub(ml);
        let k2 = (N - 1).min(idx + mu);
        let mut temp = SX::zero();
        for (j, xj) in x.iter().enumerate().take(k2 + 1).skip(k1) {
            if j != idx {
                temp += *xj * (1.0 + *xj);
            }
        }
        x[idx] * (2.0 + 5.0 * x[idx].sqr()) + 1.0 - temp
    })
}

fn p14_start<const N: usize>() -> [f64; N] {
    [-1.0; N]
}

fn p15_residuals(x: &[SX; 4]) -> [SX; 4] {
    [
        x[0] * x[0] + x[1] * x[2] - 0.0001,
        x[0] * x[1] + x[1] * x[3] - 1.0,
        x[2] * x[0] + x[3] * x[2],
        x[2] * x[1] + x[3] * x[3] - 0.0001,
    ]
}

fn p16_residuals(x: &[SX; 9]) -> [SX; 9] {
    [
        x[0] * x[0] + x[1] * x[3] + x[2] * x[6] - 0.0001,
        x[0] * x[1] + x[1] * x[4] + x[2] * x[7] - 1.0,
        x[0] * x[2] + x[1] * x[5] + x[2] * x[8],
        x[3] * x[0] + x[4] * x[3] + x[5] * x[6],
        x[3] * x[1] + x[4] * x[4] + x[5] * x[7] - 0.0001,
        x[3] * x[2] + x[4] * x[5] + x[5] * x[8],
        x[6] * x[0] + x[7] * x[3] + x[8] * x[6],
        x[6] * x[1] + x[7] * x[4] + x[8] * x[7],
        x[6] * x[2] + x[7] * x[5] + x[8] * x[8] - 0.0001,
    ]
}

fn p17_residuals(x: &[SX; 2]) -> [SX; 2] {
    [x[0] + x[1] - 3.0, x[0].sqr() + x[1].sqr() - 9.0]
}

fn p18_residuals(x: &[SX; 2]) -> [SX; 2] {
    let x0_den = source_zero_branch_denominator(x[0]);
    let x1_den = source_zero_branch_denominator(x[1]);
    [
        x[1].sqr() * (1.0 - (-x[0].sqr()).exp()) / x0_den,
        x[0] * (1.0 - (-x[1].sqr()).exp()) / x1_den,
    ]
}

fn p19_residuals(x: &[SX; 2]) -> [SX; 2] {
    let radius_sq = x[0].sqr() + x[1].sqr();
    [x[0] * radius_sq, x[1] * radius_sq]
}

fn p20_residuals(x: &[SX; 1]) -> [SX; 1] {
    [x[0] * (x[0] - 5.0).sqr()]
}

fn p21_residuals(x: &[SX; 2]) -> [SX; 2] {
    [
        x[0] - x[1].powi(3) + 5.0 * x[1].sqr() - 2.0 * x[1] - 13.0,
        x[0] + x[1].powi(3) + x[1].sqr() - 14.0 * x[1] - 29.0,
    ]
}

fn p22_residuals(x: &[SX; 2]) -> [SX; 2] {
    [
        x[0].sqr() - x[1] + 1.0,
        x[0] - (0.5 * std::f64::consts::PI * x[1]).cos(),
    ]
}

fn p23_residuals<const N: usize>(x: &[SX; N]) -> [SX; N] {
    let c = 0.9;
    let mu: [f64; N] = std::array::from_fn(|idx| (2 * (idx + 1) - 1) as f64 / (2 * N) as f64);
    std::array::from_fn(|idx| {
        let mut sum = SX::zero();
        for j in 0..N {
            sum += mu[idx] * x[j] / (mu[idx] + mu[j]);
        }
        let term = 1.0 - c * sum / (2 * N) as f64;
        x[idx] - 1.0 / term
    })
}

fn p23_start<const N: usize>() -> [f64; N] {
    [1.0; N]
}

fn source_zero_branch_denominator(value: SX) -> SX {
    value + (1.0 - value.abs().sign())
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimization::CompiledNlpProblem;

    #[test]
    fn known_burkardt_solutions_evaluate_to_zero() {
        assert_root("p01/n10", p01_residuals, [1.0; 10], 1e-12);
        assert_root("p02", p02_residuals, [0.0; 4], 1e-12);
        assert_root("p03", p03_residuals, [1.098_159e-5, 9.106_146], 1e-6);
        assert_root("p04", p04_residuals, [1.0; 4], 1e-12);
        assert_root("p05", p05_residuals, [1.0, 0.0, 0.0], 1e-12);
        assert_root("p07/n9", p07_residuals, chebyquad_solution_9(), 1e-8);
        assert_root("p08/n10", p08_residuals, [1.0; 10], 1e-12);
        assert_root("p12/n10", p12_residuals, [1.0; 10], 1e-12);
        assert_root("p15", p15_residuals, [0.01, 50.0, 0.0, 0.01], 1e-12);
        assert_root(
            "p16",
            p16_residuals,
            [0.01, 50.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            1e-12,
        );
        assert_root("p17", p17_residuals, [0.0, 3.0], 1e-12);
        assert_root("p18", p18_residuals, [0.0, 0.0], 1e-12);
        assert_root("p19", p19_residuals, [0.0, 0.0], 1e-12);
        assert_root("p20/zero", p20_residuals, [0.0], 1e-12);
        assert_root("p20/five", p20_residuals, [5.0], 1e-12);
        assert_root("p21", p21_residuals, [5.0, 4.0], 1e-12);
        assert_root("p22", p22_residuals, [0.0, 1.0], 1e-12);
    }

    #[test]
    fn burkardt_starting_points_evaluate_finitely() {
        assert_finite_start("p01/n10", p01_residuals, p01_start::<10>());
        assert_finite_start("p06/n10", p06_residuals, p06_start::<10>());
        assert_finite_start("p07/n9", p07_residuals, p07_start::<9>());
        assert_finite_start("p08/n10", p08_residuals, p08_start::<10>());
        assert_finite_start("p09/n10", p09_residuals, p09_start::<10>());
        assert_finite_start("p10/n10", p10_residuals, p10_start::<10>());
        assert_finite_start("p11/n10", p11_residuals, p11_start::<10>());
        assert_finite_start("p12/n10", p12_residuals, p12_start::<10>());
        assert_finite_start("p13/n10", p13_residuals, p13_start::<10>());
        assert_finite_start("p14/n10", p14_residuals, p14_start::<10>());
        assert_finite_start("p23/n10", p23_residuals, p23_start::<10>());
    }

    fn assert_root<const N: usize>(
        name: &str,
        residuals: ResidualBuilder<N>,
        x: [f64; N],
        tol: f64,
    ) {
        let value = residual_inf_norm(name, residuals, x);
        assert!(
            value.is_finite() && value <= tol,
            "{name} expected root residual <= {tol:.1e}, got {value:.6e}"
        );
    }

    fn assert_finite_start<const N: usize>(name: &str, residuals: ResidualBuilder<N>, x: [f64; N]) {
        let value = residual_inf_norm(name, residuals, x);
        assert!(
            value.is_finite(),
            "{name} starting residual should be finite, got {value:?}"
        );
    }

    fn residual_inf_norm<const N: usize>(
        name: &str,
        residuals: ResidualBuilder<N>,
        x: [f64; N],
    ) -> f64 {
        let compiled = symbolic_compile::<VecN<SX, N>, (), VecN<SX, N>, (), _>(
            &format!("burkardt_source_sanity_{name}").replace('/', "_"),
            move |vars, ()| SymbolicNlpOutputs {
                objective: SX::zero(),
                equalities: VecN {
                    values: residuals(&vars.values),
                },
                inequalities: (),
            },
            Default::default(),
        )
        .expect("sanity problem should compile");
        let bound_problem = compiled
            .bind_runtime_bounds(&TypedRuntimeNlpBounds::<VecN<SX, N>, ()>::default())
            .expect("default bounds should bind");
        let mut values = vec![0.0; bound_problem.equality_count()];
        bound_problem.equality_values(&x, &[], &mut values);
        values
            .into_iter()
            .map(f64::abs)
            .fold(0.0, |acc, value| acc.max(value))
    }

    fn chebyquad_solution_9() -> [f64; 9] {
        [
            2.0 * 4.420_534_614_999_999e-2 - 1.0,
            2.0 * 0.199_490_672_3 - 1.0,
            2.0 * 0.235_619_108_45 - 1.0,
            2.0 * 0.416_046_907_9 - 1.0,
            0.0,
            2.0 * 0.583_953_092_1 - 1.0,
            2.0 * 0.764_380_891_55 - 1.0,
            2.0 * 0.800_509_327_699_999_9 - 1.0,
            2.0 * 0.955_794_653_85 - 1.0,
        ]
    }
}
