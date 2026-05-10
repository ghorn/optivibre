use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownStatus {
    KnownPassing,
    KnownFailing,
    Skipped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemSpeed {
    Fast,
    Slow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemTestSet {
    Core,
    BurkardtTestNonlin,
    Schittkowski306,
}

impl ProblemTestSet {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::BurkardtTestNonlin => "burkardt_test_nonlin",
            Self::Schittkowski306 => "schittkowski_306",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterationLimits {
    pub sqp: usize,
    pub nlip: usize,
    pub ipopt: usize,
}

impl IterationLimits {
    pub const fn with_default(default: usize) -> Self {
        Self {
            sqp: default,
            nlip: default,
            ipopt: default,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProblemManifestEntry {
    pub id: &'static str,
    pub sqp: KnownStatus,
    pub nlip: KnownStatus,
    pub ipopt: KnownStatus,
    pub max_iters: IterationLimits,
    pub speed: ProblemSpeed,
    pub test_set: ProblemTestSet,
}

pub const DEFAULT_MAX_ITERS: usize = 300;

pub const fn manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt: nlip,
        max_iters,
        speed: ProblemSpeed::Fast,
        test_set: ProblemTestSet::Core,
    }
}

pub const fn manifest_entry_with_ipopt(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Fast,
        test_set: ProblemTestSet::Core,
    }
}

pub const fn slow_manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt: nlip,
        max_iters,
        speed: ProblemSpeed::Slow,
        test_set: ProblemTestSet::Core,
    }
}

pub const fn slow_manifest_entry_with_ipopt(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Slow,
        test_set: ProblemTestSet::Core,
    }
}

pub const fn burkardt_manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Fast,
        test_set: ProblemTestSet::BurkardtTestNonlin,
    }
}

pub const fn slow_burkardt_manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Slow,
        test_set: ProblemTestSet::BurkardtTestNonlin,
    }
}

pub const fn schittkowski_manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    nlip: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        nlip,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Fast,
        test_set: ProblemTestSet::Schittkowski306,
    }
}

const MANIFEST: &[ProblemManifestEntry] = &[
    manifest_entry_with_ipopt(
        "rosenbrock_2",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "generalized_rosenbrock_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "generalized_rosenbrock_8",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "generalized_rosenbrock_16",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 400,
            nlip: 400,
            ipopt: 400,
        },
    ),
    manifest_entry_with_ipopt(
        "disk_rosenbrock",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "powell_singular_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "wood_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "brown_almost_linear_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "brown_almost_linear_8",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "brown_almost_linear_16",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 400,
            nlip: 400,
            ipopt: 400,
        },
    ),
    manifest_entry_with_ipopt(
        "trigonometric_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "trigonometric_8",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "trigonometric_16",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 400,
            nlip: 400,
            ipopt: 400,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_eq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_eq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            nlip: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            nlip: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_eq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            nlip: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            nlip: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            nlip: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            nlip: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            nlip: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            nlip: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_eq_quadratic",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            nlip: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            nlip: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            nlip: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            nlip: 800,
            ipopt: 800,
        },
    ),
    manifest_entry_with_ipopt(
        "parameterized_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry(
        "hs021",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry(
        "hs035",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry(
        "hs071",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits {
            sqp: 400,
            nlip: 400,
            ipopt: 400,
        },
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p01_generalized_rosenbrock_n02",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p01_generalized_rosenbrock_n10",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p01_generalized_rosenbrock_n20",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p02_powell_singular_n04",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p03_powell_badly_scaled_n02",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p04_wood_n04",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p05_helical_valley_n03",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p06_watson_n02",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p06_watson_n10",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p06_watson_n20",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n02",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n03",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n04",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n05",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n06",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n07",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p07_chebyquad_n09",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p08_brown_almost_linear_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p08_brown_almost_linear_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p08_brown_almost_linear_n20",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p09_discrete_boundary_value_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p09_discrete_boundary_value_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p09_discrete_boundary_value_n20",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p10_discrete_integral_equation_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p10_discrete_integral_equation_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p10_discrete_integral_equation_n20",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p11_trigonometric_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p11_trigonometric_n10",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p11_trigonometric_n20",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p12_variably_dimensioned_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p12_variably_dimensioned_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p12_variably_dimensioned_n20",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p13_broyden_tridiagonal_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p13_broyden_tridiagonal_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p13_broyden_tridiagonal_n20",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p14_broyden_banded_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p14_broyden_banded_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p14_broyden_banded_n20",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p15_hammarling_2x2_n04",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p16_hammarling_3x3_n09",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p17_dennis_schnabel_n02",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p18_sample18_n02",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p19_sample19_n02",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p20_scalar_n01",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p21_freudenstein_roth_n02",
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p22_boggs_n02",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p23_chandrasekhar_n01",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    burkardt_manifest_entry(
        "burkardt_p23_chandrasekhar_n10",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    slow_burkardt_manifest_entry(
        "burkardt_p23_chandrasekhar_n20",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp001",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp002",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp003",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp004",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp005",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp006",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp007",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp008",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp009",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
    schittkowski_manifest_entry(
        "schittkowski_tp010",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(500),
    ),
];

pub fn manifest_entries() -> &'static [ProblemManifestEntry] {
    MANIFEST
}

pub fn manifest_entry_by_id(id: &str) -> Option<&'static ProblemManifestEntry> {
    MANIFEST.iter().find(|entry| entry.id == id)
}
