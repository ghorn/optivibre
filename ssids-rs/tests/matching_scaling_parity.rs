use std::time::{Duration, Instant};

use ssids_rs::{
    NativeOrdering, NativeSpral, NativeSpralAnalyseInfo, NativeSpralFactorInfo,
    NumericFactorOptions, SymmetricCscMatrix, analyse_with_user_ordering, factorize,
};

fn dense_mul(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(value, x_i)| value * x_i)
                .sum::<f64>()
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

#[derive(Clone, Debug)]
struct DenseBoundaryRng {
    state: u64,
}

impl DenseBoundaryRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn usize_inclusive(&mut self, low: usize, high: usize) -> usize {
        debug_assert!(low <= high);
        low + (self.next_u64() as usize % (high - low + 1))
    }

    fn dyadic(&mut self, numerator_radius: i16, max_shift: u8) -> f64 {
        let span = i32::from(numerator_radius) * 2 + 1;
        let numerator = self.next_u64() as i32 % span - i32::from(numerator_radius);
        let shift = self.next_u64() as u8 % (max_shift + 1);
        f64::from(numerator) / f64::from(1_u32 << shift)
    }
}

fn random_dense_dyadic_matrix(dimension: usize, rng: &mut DenseBoundaryRng) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    let mut row = 0;
    while row < dimension {
        let mut col = 0;
        while col <= row {
            let value = if row == col {
                rng.dyadic(8, 6)
            } else {
                rng.dyadic(16, 7)
            };
            matrix[row][col] = value;
            matrix[col][row] = value;
            col += 1;
        }
        row += 1;
    }
    matrix
}

fn random_dyadic_solution(dimension: usize, rng: &mut DenseBoundaryRng) -> Vec<f64> {
    (0..dimension).map(|_| rng.dyadic(8, 4)).collect()
}

fn dense_boundary_case(seed: u64, case_index: usize) -> (usize, Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = DenseBoundaryRng::new(seed);
    let mut dimension = 0;
    let mut matrix = Vec::new();
    let mut expected_solution = Vec::new();
    for _ in 0..=case_index {
        dimension = rng.usize_inclusive(33, 160);
        matrix = random_dense_dyadic_matrix(dimension, &mut rng);
        expected_solution = random_dyadic_solution(dimension, &mut rng);
    }
    (dimension, matrix, expected_solution)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
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

fn load_native_or_skip() -> Option<NativeSpral> {
    match NativeSpral::load() {
        Ok(native) => Some(native),
        Err(error) => {
            if std::env::var_os("AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY").is_some() {
                panic!("native SPRAL is required for matching/scaling parity runs: {error}");
            }
            eprintln!("skipping native SPRAL matching/scaling test: {error}");
            None
        }
    }
}

fn bit_patterns(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn hash_usize(values: &[usize]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &value| {
        let mixed = hash ^ u64::try_from(value).unwrap_or(u64::MAX);
        mixed.wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hash_f64_bits(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        let mixed = hash ^ value.to_bits();
        mixed.wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn residual_inf(matrix: &[Vec<f64>], solution: &[f64], rhs: &[f64]) -> f64 {
    dense_mul(matrix, solution)
        .into_iter()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}

fn solution_delta_inf(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}

fn bit_mismatch_summary(lhs: &[f64], rhs: &[f64]) -> (usize, Option<usize>) {
    let mut count = 0;
    let mut first = None;
    for (index, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        if lhs.to_bits() != rhs.to_bits() {
            count += 1;
            first.get_or_insert(index);
        }
    }
    (count, first)
}

#[derive(Debug)]
struct NativeRun {
    analyse_info: NativeSpralAnalyseInfo,
    factor_info: NativeSpralFactorInfo,
    factor_time: Duration,
    solve_time: Duration,
    solution: Vec<f64>,
    residual_inf: f64,
}

fn run_native_session(
    mut session: ssids_rs::NativeSpralSession,
    matrix: SymmetricCscMatrix<'_>,
    dense_matrix: &[Vec<f64>],
    rhs: &[f64],
) -> NativeRun {
    let analyse_info = session.analyse_info();
    let factor_started = Instant::now();
    let factor_info = session.factorize(matrix).expect("native factorize");
    let factor_time = factor_started.elapsed();
    let solve_started = Instant::now();
    let solution = session.solve(rhs).expect("native solve");
    let solve_time = solve_started.elapsed();
    let residual_inf = residual_inf(dense_matrix, &solution, rhs);
    NativeRun {
        analyse_info,
        factor_info,
        factor_time,
        solve_time,
        solution,
        residual_inf,
    }
}

#[derive(Debug)]
struct RustRun {
    factor_time: Duration,
    solve_time: Duration,
    inertia: ssids_rs::Inertia,
    two_by_two_pivots: usize,
    delayed_pivots: usize,
    solution: Vec<f64>,
    residual_inf: f64,
}

fn run_rust_with_user_order(
    matrix: SymmetricCscMatrix<'_>,
    dense_matrix: &[Vec<f64>],
    rhs: &[f64],
    order: &[usize],
    options: &NumericFactorOptions,
) -> (Duration, RustRun) {
    let analyse_started = Instant::now();
    let (symbolic, _) = analyse_with_user_ordering(matrix, order).expect("rust analyse");
    let analyse_time = analyse_started.elapsed();
    let factor_started = Instant::now();
    let (mut factor, _) = factorize(matrix, &symbolic, options).expect("rust factorize");
    let factor_time = factor_started.elapsed();
    let solve_started = Instant::now();
    let solution = factor.solve(rhs).expect("rust solve");
    let solve_time = solve_started.elapsed();
    let residual_inf = residual_inf(dense_matrix, &solution, rhs);
    let pivot_stats = factor.pivot_stats();
    (
        analyse_time,
        RustRun {
            factor_time,
            solve_time,
            inertia: factor.inertia(),
            two_by_two_pivots: pivot_stats.two_by_two_pivots,
            delayed_pivots: pivot_stats.delayed_pivots,
            solution,
            residual_inf,
        },
    )
}

#[test]
#[ignore = "manual matching/scaling observation lane; native matching is the source oracle"]
fn native_matching_scaling_order_observation_dense_boundary_case() {
    let Some(native) = load_native_or_skip() else {
        return;
    };

    let seed = env_u64("SPRAL_SSIDS_MATCHING_PARITY_SEED", 0x7061_7269_7479);
    let case_index = env_usize("SPRAL_SSIDS_MATCHING_PARITY_CASE", 58);
    let (dimension, dense_matrix, expected_solution) = dense_boundary_case(seed, case_index);
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense_matrix);
    let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
        .expect("valid matching/scaling witness CSC");
    let rhs = dense_mul(&dense_matrix, &expected_solution);
    let options = NumericFactorOptions::default();

    let matching_analyse_started = Instant::now();
    let matching_session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Matching)
        .expect("native matching analyse");
    let matching_analyse_time = matching_analyse_started.elapsed();
    let captured_order = matching_session
        .analysis_order()
        .expect("native matching analyse should expose an order buffer")
        .to_vec();
    let native_matching = run_native_session(matching_session, matrix, &dense_matrix, &rhs);

    let native_user_analyse_started = Instant::now();
    let native_user_session = native
        .analyse_with_options_and_user_ordering(matrix, &options, &captured_order)
        .expect("native captured-order analyse");
    let native_user_analyse_time = native_user_analyse_started.elapsed();
    let native_user = run_native_session(native_user_session, matrix, &dense_matrix, &rhs);

    let (rust_analyse_time, rust_user) =
        run_rust_with_user_order(matrix, &dense_matrix, &rhs, &captured_order, &options);

    let (matching_vs_user_bits, matching_vs_user_first) =
        bit_mismatch_summary(&native_matching.solution, &native_user.solution);
    let (rust_vs_native_bits, rust_vs_native_first) =
        bit_mismatch_summary(&rust_user.solution, &native_user.solution);

    eprintln!("=== SPRAL matching/scaling observation ===");
    eprintln!(
        "case seed=0x{seed:016x} case={case_index} dimension={dimension} nnz={} rhs_hash=0x{:016x}",
        values.len(),
        hash_f64_bits(&rhs)
    );
    eprintln!(
        "captured_order convention=order[original_column]=pivot_position hash=0x{:016x} prefix={:?}",
        hash_usize(&captured_order),
        &captured_order[..captured_order.len().min(16)]
    );
    eprintln!(
        "analysis_scaling=opaque source=SPRAL saved akeep%scaling for options%scaling=3; C scale buffer is factor-time only"
    );
    eprintln!(
        "native_matching_scaling analyse={:?} factor={:?} solve={:?} residual={:.3e} inertia={:?} two_by_two={} delayed={} analyse_info={:?}",
        matching_analyse_time,
        native_matching.factor_time,
        native_matching.solve_time,
        native_matching.residual_inf,
        native_matching.factor_info.inertia,
        native_matching.factor_info.two_by_two_pivots,
        native_matching.factor_info.delayed_pivots,
        native_matching.analyse_info
    );
    eprintln!(
        "native_captured_order_no_scaling analyse={:?} factor={:?} solve={:?} residual={:.3e} inertia={:?} two_by_two={} delayed={} analyse_info={:?}",
        native_user_analyse_time,
        native_user.factor_time,
        native_user.solve_time,
        native_user.residual_inf,
        native_user.factor_info.inertia,
        native_user.factor_info.two_by_two_pivots,
        native_user.factor_info.delayed_pivots,
        native_user.analyse_info
    );
    eprintln!(
        "rust_captured_order_no_scaling analyse={:?} factor={:?} solve={:?} residual={:.3e} inertia={:?} two_by_two={} delayed={}",
        rust_analyse_time,
        rust_user.factor_time,
        rust_user.solve_time,
        rust_user.residual_inf,
        rust_user.inertia,
        rust_user.two_by_two_pivots,
        rust_user.delayed_pivots
    );
    eprintln!(
        "matching_scaling_vs_captured_order_no_scaling delta={:.3e} bit_mismatches={} first_mismatch={:?}",
        solution_delta_inf(&native_matching.solution, &native_user.solution),
        matching_vs_user_bits,
        matching_vs_user_first
    );
    eprintln!(
        "rust_vs_native_captured_order_no_scaling delta={:.3e} bit_mismatches={} first_mismatch={:?}",
        solution_delta_inf(&rust_user.solution, &native_user.solution),
        rust_vs_native_bits,
        rust_vs_native_first
    );

    assert_eq!(
        rust_user.inertia, native_user.factor_info.inertia,
        "captured-order/no-scaling inertia mismatch"
    );
    assert_eq!(
        rust_user.two_by_two_pivots, native_user.factor_info.two_by_two_pivots,
        "captured-order/no-scaling two-by-two pivot mismatch"
    );
    assert_eq!(
        rust_user.delayed_pivots, native_user.factor_info.delayed_pivots,
        "captured-order/no-scaling delayed pivot mismatch"
    );
    assert_eq!(
        bit_patterns(&rust_user.solution),
        bit_patterns(&native_user.solution),
        "captured-order/no-scaling solve bits diverged; shrink this as a core SSIDS parity witness before porting matching/scaling"
    );
}
