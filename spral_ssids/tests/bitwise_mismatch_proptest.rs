use proptest::array::{uniform5, uniform10};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestCaseError, TestRng, TestRunner};
use spral_ssids::{
    Inertia, NativeOrdering, NativeSpral, NumericFactorOptions, OrderingStrategy, PivotStats,
    SsidsOptions, SymmetricCscMatrix, analyse, factorize,
};

const MAX_DIM: usize = 5;
const EDGE_SLOTS: [(usize, usize); 10] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixFamily {
    Path,
    Arrow,
    PathPlusChord,
    TwoByTwoPivot,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dyadic {
    numerator: i16,
    shift: u8,
}

impl Dyadic {
    fn as_f64(self) -> f64 {
        f64::from(self.numerator) / f64::from(1_u32 << self.shift)
    }
}

#[derive(Clone, Debug)]
struct SparseBitwiseParityCase {
    active_dim: usize,
    family: MatrixFamily,
    chord_selector: u8,
    diagonals: [Dyadic; MAX_DIM],
    edge_values: [Dyadic; EDGE_SLOTS.len()],
    pivot_edge: Dyadic,
    expected_solution: [Dyadic; MAX_DIM],
}

#[derive(Clone, Debug)]
struct MaterializedBitwiseCase {
    active_dim: usize,
    family: MatrixFamily,
    chord_selector: u8,
    dense_matrix: Vec<Vec<f64>>,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<f64>,
    expected_solution: Vec<f64>,
    rhs: Vec<f64>,
}

#[derive(Clone, Debug)]
enum FactorOutcome {
    Success {
        inertia: Inertia,
        pivot_stats: PivotStats,
    },
    Error(String),
}

#[derive(Clone, Debug)]
enum SolveOutcome {
    Success(Vec<f64>),
    Error(String),
}

const DEFAULT_PARITY_SEED: u64 = 0xB17_515E;

fn deterministic_runner(cases: u32) -> TestRunner {
    seeded_runner(cases, DEFAULT_PARITY_SEED)
}

fn seeded_runner(cases: u32, seed: u64) -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases,
            max_local_rejects: cases.saturating_mul(16).max(65_536),
            failure_persistence: None,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        },
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    )
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| parse_u64(&value))
        .unwrap_or(default)
}

fn parse_u64(value: &str) -> Option<u64> {
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map_or_else(
            || value.parse::<u64>().ok(),
            |hex| u64::from_str_radix(hex, 16).ok(),
        )
}

fn dyadic_strategy() -> impl Strategy<Value = Dyadic> {
    (-8_i16..=8, 0_u8..=4).prop_map(|(numerator, shift)| Dyadic { numerator, shift })
}

fn nonzero_dyadic_strategy() -> impl Strategy<Value = Dyadic> {
    (-8_i16..=8, 0_u8..=4)
        .prop_filter("pivot edge must be nonzero", |(numerator, _)| {
            *numerator != 0
        })
        .prop_map(|(numerator, shift)| Dyadic { numerator, shift })
}

fn family_case_strategy(family: MatrixFamily) -> BoxedStrategy<SparseBitwiseParityCase> {
    (
        3_usize..=MAX_DIM,
        any::<u8>(),
        uniform5(dyadic_strategy()),
        uniform10(dyadic_strategy()),
        nonzero_dyadic_strategy(),
        uniform5(dyadic_strategy()),
    )
        .prop_map(
            move |(
                active_dim,
                chord_selector,
                diagonals,
                edge_values,
                pivot_edge,
                expected_solution,
            )| {
                SparseBitwiseParityCase {
                    active_dim,
                    family,
                    chord_selector,
                    diagonals,
                    edge_values,
                    pivot_edge,
                    expected_solution,
                }
            },
        )
        .boxed()
}

fn sparse_bitwise_case_strategy() -> BoxedStrategy<SparseBitwiseParityCase> {
    prop_oneof![
        5 => family_case_strategy(MatrixFamily::Path),
        4 => family_case_strategy(MatrixFamily::Arrow),
        3 => family_case_strategy(MatrixFamily::PathPlusChord),
        2 => family_case_strategy(MatrixFamily::TwoByTwoPivot),
        1 => family_case_strategy(MatrixFamily::Complete),
    ]
    .prop_filter(
        "case should keep its family skeleton active and avoid the all-zero exact solution",
        SparseBitwiseParityCase::is_smoke_interesting,
    )
    .boxed()
}

fn load_native_or_skip() -> Option<NativeSpral> {
    match NativeSpral::load() {
        Ok(native) => Some(native),
        Err(error) => {
            if std::env::var_os("AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY").is_some() {
                panic!("native SPRAL is required for fail-closed mismatch search: {error}");
            }
            eprintln!("skipping native SPRAL mismatch search: {error}");
            None
        }
    }
}

fn edge_slot_index(lhs: usize, rhs: usize) -> usize {
    let edge = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
    EDGE_SLOTS
        .iter()
        .position(|&candidate| candidate == edge)
        .expect("edge must be part of the candidate superset")
}

fn dense_mul(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(value, x_i)| value * x_i)
                .sum()
        })
        .collect()
}

fn dense_to_lower_csc(matrix: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let dimension = matrix.len();
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..dimension {
        for (row, dense_row) in matrix.iter().enumerate().skip(col) {
            let value = dense_row[col];
            if row == col || value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices, values)
}

fn set_symmetric_entry(matrix: &mut [Vec<f64>], lhs: usize, rhs: usize, value: f64) {
    matrix[lhs][rhs] = value;
    matrix[rhs][lhs] = value;
}

fn path_edges(active_dim: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..active_dim.saturating_sub(1)).map(|index| (index, index + 1))
}

fn path_plus_chord_edge(active_dim: usize, selector: u8) -> Option<(usize, usize)> {
    let chords = EDGE_SLOTS
        .iter()
        .copied()
        .filter(|&(lhs, rhs)| rhs < active_dim && rhs != lhs + 1)
        .collect::<Vec<_>>();
    (!chords.is_empty()).then(|| chords[usize::from(selector) % chords.len()])
}

impl SparseBitwiseParityCase {
    fn keeps_family_skeleton_active(&self) -> bool {
        match self.family {
            MatrixFamily::Path => path_edges(self.active_dim)
                .all(|(lhs, rhs)| self.edge_values[edge_slot_index(lhs, rhs)].numerator != 0),
            MatrixFamily::Arrow => (1..self.active_dim)
                .all(|rhs| self.edge_values[edge_slot_index(0, rhs)].numerator != 0),
            MatrixFamily::PathPlusChord => {
                path_edges(self.active_dim)
                    .all(|(lhs, rhs)| self.edge_values[edge_slot_index(lhs, rhs)].numerator != 0)
                    && path_plus_chord_edge(self.active_dim, self.chord_selector).is_some_and(
                        |(lhs, rhs)| self.edge_values[edge_slot_index(lhs, rhs)].numerator != 0,
                    )
            }
            MatrixFamily::TwoByTwoPivot => self.pivot_edge.numerator != 0,
            MatrixFamily::Complete => EDGE_SLOTS
                .iter()
                .copied()
                .filter(|&(_, rhs)| rhs < self.active_dim)
                .all(|(lhs, rhs)| self.edge_values[edge_slot_index(lhs, rhs)].numerator != 0),
        }
    }

    fn has_nonzero_expected_solution(&self) -> bool {
        self.expected_solution[..self.active_dim]
            .iter()
            .any(|value| value.numerator != 0)
    }

    fn is_smoke_interesting(&self) -> bool {
        self.keeps_family_skeleton_active() && self.has_nonzero_expected_solution()
    }

    fn materialize(&self) -> MaterializedBitwiseCase {
        let mut dense_matrix = vec![vec![0.0; self.active_dim]; self.active_dim];
        for (index, row) in dense_matrix.iter_mut().enumerate().take(self.active_dim) {
            row[index] = self.diagonals[index].as_f64();
        }

        match self.family {
            MatrixFamily::Path => {
                for (lhs, rhs) in path_edges(self.active_dim) {
                    let value = self.edge_values[edge_slot_index(lhs, rhs)].as_f64();
                    set_symmetric_entry(&mut dense_matrix, lhs, rhs, value);
                }
            }
            MatrixFamily::Arrow => {
                for rhs in 1..self.active_dim {
                    let value = self.edge_values[edge_slot_index(0, rhs)].as_f64();
                    set_symmetric_entry(&mut dense_matrix, 0, rhs, value);
                }
            }
            MatrixFamily::PathPlusChord => {
                for (lhs, rhs) in path_edges(self.active_dim) {
                    let value = self.edge_values[edge_slot_index(lhs, rhs)].as_f64();
                    set_symmetric_entry(&mut dense_matrix, lhs, rhs, value);
                }
                if let Some((lhs, rhs)) = path_plus_chord_edge(self.active_dim, self.chord_selector)
                {
                    let value = self.edge_values[edge_slot_index(lhs, rhs)].as_f64();
                    set_symmetric_entry(&mut dense_matrix, lhs, rhs, value);
                }
            }
            MatrixFamily::TwoByTwoPivot => {
                dense_matrix[0][0] = 0.0;
                dense_matrix[1][1] = 0.0;
                set_symmetric_entry(&mut dense_matrix, 0, 1, self.pivot_edge.as_f64());
                for (lhs, rhs) in path_edges(self.active_dim).skip(1) {
                    let value = self.edge_values[edge_slot_index(lhs, rhs)].as_f64();
                    set_symmetric_entry(&mut dense_matrix, lhs, rhs, value);
                }
            }
            MatrixFamily::Complete => {
                for (lhs, rhs) in EDGE_SLOTS
                    .iter()
                    .copied()
                    .filter(|&(_, rhs)| rhs < self.active_dim)
                {
                    let value = self.edge_values[edge_slot_index(lhs, rhs)].as_f64();
                    set_symmetric_entry(&mut dense_matrix, lhs, rhs, value);
                }
            }
        }

        let expected_solution = self.expected_solution[..self.active_dim]
            .iter()
            .map(|value| value.as_f64())
            .collect::<Vec<_>>();
        let rhs = dense_mul(&dense_matrix, &expected_solution);
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense_matrix);
        MaterializedBitwiseCase {
            active_dim: self.active_dim,
            family: self.family,
            chord_selector: self.chord_selector,
            dense_matrix,
            col_ptrs,
            row_indices,
            values,
            expected_solution,
            rhs,
        }
    }
}

fn bit_patterns(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn format_scalar_bits(value: f64) -> String {
    format!("{value:?} (0x{:016x})", value.to_bits())
}

fn format_vector_bits(values: &[f64]) -> Vec<String> {
    values
        .iter()
        .map(|&value| format_scalar_bits(value))
        .collect()
}

fn format_matrix_bits(matrix: &[Vec<f64>]) -> Vec<Vec<String>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&value| format_scalar_bits(value)).collect())
        .collect()
}

fn format_factor_outcome(outcome: &FactorOutcome) -> String {
    match outcome {
        FactorOutcome::Success {
            inertia,
            pivot_stats,
        } => format!(
            "success inertia={:?} two_by_two_pivots={} delayed_pivots={}",
            inertia, pivot_stats.two_by_two_pivots, pivot_stats.delayed_pivots
        ),
        FactorOutcome::Error(detail) => format!("error {detail}"),
    }
}

fn format_solve_outcome(outcome: &SolveOutcome) -> String {
    match outcome {
        SolveOutcome::Success(solution) => format!(
            "success values={:?} bits={:?}",
            solution,
            bit_patterns(solution)
        ),
        SolveOutcome::Error(detail) => format!("error {detail}"),
    }
}

fn mismatch_context(
    materialized: &MaterializedBitwiseCase,
    rust_factor: &FactorOutcome,
    native_factor: &FactorOutcome,
    rust_solve: Option<&SolveOutcome>,
    native_solve: Option<&SolveOutcome>,
) -> String {
    format!(
        concat!(
            "active_dim={}\n",
            "family={:?}\n",
            "chord_selector={}\n",
            "dense_matrix={:?}\n",
            "col_ptrs={:?}\n",
            "row_indices={:?}\n",
            "values={:?}\n",
            "expected_solution={:?}\n",
            "expected_solution_bits={:?}\n",
            "rhs={:?}\n",
            "rhs_bits={:?}\n",
            "rust_factor={}\n",
            "native_factor={}\n",
            "rust_solve={}\n",
            "native_solve={}\n"
        ),
        materialized.active_dim,
        materialized.family,
        materialized.chord_selector,
        format_matrix_bits(&materialized.dense_matrix),
        materialized.col_ptrs,
        materialized.row_indices,
        format_vector_bits(&materialized.values),
        materialized.expected_solution,
        bit_patterns(&materialized.expected_solution),
        materialized.rhs,
        bit_patterns(&materialized.rhs),
        format_factor_outcome(rust_factor),
        format_factor_outcome(native_factor),
        rust_solve
            .map(format_solve_outcome)
            .unwrap_or_else(|| "not attempted".into()),
        native_solve
            .map(format_solve_outcome)
            .unwrap_or_else(|| "not attempted".into()),
    )
}

fn assert_sparse_case_bitwise_parity(
    native: &NativeSpral,
    test_case: &SparseBitwiseParityCase,
) -> Result<(), String> {
    let materialized = test_case.materialize();
    let matrix = SymmetricCscMatrix::new(
        materialized.active_dim,
        &materialized.col_ptrs,
        &materialized.row_indices,
        Some(&materialized.values),
    )
    .map_err(|error| format!("generated invalid CSC matrix: {error:?}"))?;

    let options = NumericFactorOptions::default();
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .map_err(|error| format!("rust analyse failed for generated case: {error:?}"))?;

    let rust_factorization = factorize(matrix, &symbolic, &options);
    let mut native_session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
        .map_err(|error| format!("native analyse failed for generated case: {error}"))?;
    let native_factorization = native_session.factorize(matrix);

    let rust_factor_outcome = match &rust_factorization {
        Ok((factor, _)) => FactorOutcome::Success {
            inertia: factor.inertia(),
            pivot_stats: factor.pivot_stats(),
        },
        Err(error) => FactorOutcome::Error(error.to_string()),
    };
    let native_factor_outcome = match &native_factorization {
        Ok(info) => FactorOutcome::Success {
            inertia: info.inertia,
            pivot_stats: PivotStats {
                two_by_two_pivots: info.two_by_two_pivots,
                delayed_pivots: info.delayed_pivots,
            },
        },
        Err(error) => FactorOutcome::Error(error.to_string()),
    };

    match (&rust_factor_outcome, &native_factor_outcome) {
        (FactorOutcome::Success { .. }, FactorOutcome::Error(_))
        | (FactorOutcome::Error(_), FactorOutcome::Success { .. }) => {
            return Err(format!(
                "factorization success mismatch\n{}",
                mismatch_context(
                    &materialized,
                    &rust_factor_outcome,
                    &native_factor_outcome,
                    None,
                    None
                )
            ));
        }
        (FactorOutcome::Error(_), FactorOutcome::Error(_)) => {
            return Ok(());
        }
        (
            FactorOutcome::Success {
                inertia: rust_inertia,
                pivot_stats: rust_stats,
            },
            FactorOutcome::Success {
                inertia: native_inertia,
                pivot_stats: native_stats,
            },
        ) => {
            if rust_inertia != native_inertia {
                return Err(format!(
                    "inertia mismatch\n{}",
                    mismatch_context(
                        &materialized,
                        &rust_factor_outcome,
                        &native_factor_outcome,
                        None,
                        None
                    )
                ));
            }
            if rust_stats != native_stats {
                return Err(format!(
                    "pivot stats mismatch\n{}",
                    mismatch_context(
                        &materialized,
                        &rust_factor_outcome,
                        &native_factor_outcome,
                        None,
                        None
                    )
                ));
            }
        }
    }

    let (mut rust_factor, _) = rust_factorization.expect("checked success above");
    let rust_solve = rust_factor.solve(&materialized.rhs).map_or_else(
        |error| SolveOutcome::Error(error.to_string()),
        SolveOutcome::Success,
    );
    let native_solve = native_session.solve(&materialized.rhs).map_or_else(
        |error| SolveOutcome::Error(error.to_string()),
        SolveOutcome::Success,
    );

    match (&rust_solve, &native_solve) {
        (SolveOutcome::Success(_), SolveOutcome::Error(_))
        | (SolveOutcome::Error(_), SolveOutcome::Success(_)) => Err(format!(
            "solve success mismatch\n{}",
            mismatch_context(
                &materialized,
                &rust_factor_outcome,
                &native_factor_outcome,
                Some(&rust_solve),
                Some(&native_solve)
            )
        )),
        (SolveOutcome::Error(_), SolveOutcome::Error(_)) => Ok(()),
        (SolveOutcome::Success(rust_solution), SolveOutcome::Success(native_solution)) => {
            if bit_patterns(rust_solution) != bit_patterns(native_solution) {
                return Err(format!(
                    "solve bitwise mismatch\n{}",
                    mismatch_context(
                        &materialized,
                        &rust_factor_outcome,
                        &native_factor_outcome,
                        Some(&rust_solve),
                        Some(&native_solve)
                    )
                ));
            }
            Ok(())
        }
    }
}

#[test]
#[ignore = "manual mismatch search; fixed witnesses should be debugged via exact regression tests"]
fn rust_and_native_spral_match_bitwise_on_shrinkable_sparse_smoke_cases() {
    let Some(native) = load_native_or_skip() else {
        return;
    };

    let strategy = sparse_bitwise_case_strategy();
    let mut runner = deterministic_runner(96);
    runner
        .run(&strategy, |test_case| {
            assert_sparse_case_bitwise_parity(&native, &test_case).map_err(TestCaseError::fail)
        })
        .expect("deterministic smoke search should preserve native-vs-rust parity");
}

#[test]
#[ignore = "manual native-vs-rust mismatch hunt with deeper shrinking"]
fn rust_and_native_spral_match_bitwise_on_shrinkable_sparse_deep_search() {
    let Some(native) = load_native_or_skip() else {
        return;
    };

    let strategy = sparse_bitwise_case_strategy();
    let cases = env_u32("SPRAL_SSIDS_PARITY_CASES", 262_144);
    let seed = env_u64("SPRAL_SSIDS_PARITY_SEED", DEFAULT_PARITY_SEED);
    let mut runner = seeded_runner(cases, seed);
    runner
        .run(&strategy, |test_case| {
            assert_sparse_case_bitwise_parity(&native, &test_case).map_err(TestCaseError::fail)
        })
        .expect("deep mismatch search should either shrink a witness or preserve parity");
}
