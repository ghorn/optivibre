#![allow(clippy::needless_range_loop)]

use std::process::Command;

use rayon::prelude::*;
use ssids_rs::{
    Inertia, NativeOrdering, NativeSpral, NumericFactorOptions, OrderingStrategy, PivotStats,
    SsidsOptions, SymmetricCscMatrix, analyse, factorize,
};

#[derive(Clone)]
struct DenseCase {
    dense: Vec<Vec<f64>>,
    expected_solution: Vec<f64>,
}

#[derive(Clone, Debug)]
struct RustOutcome {
    inertia: Inertia,
    pivot_stats: PivotStats,
    stored_nnz: usize,
    supernode_count: usize,
    max_supernode_width: usize,
    factor_bytes: usize,
    solution: Vec<f64>,
    solution_bits: Vec<u64>,
    residual_inf: f64,
}

#[derive(Clone, Debug)]
struct NativeOutcome {
    inertia: Inertia,
    pivot_stats: PivotStats,
    solution: Vec<f64>,
    residual_inf: f64,
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

fn residual_inf_norm(matrix: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
    dense_mul(matrix, x)
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn delta_inf(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn bit_patterns(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
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

fn dense_to_complete_lower_csc(matrix: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let dimension = matrix.len();
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::with_capacity(dimension * (dimension + 1) / 2);
    let mut values = Vec::with_capacity(dimension * (dimension + 1) / 2);
    col_ptrs.push(0);
    for col in 0..dimension {
        for (row, dense_row) in matrix.iter().enumerate().skip(col) {
            row_indices.push(row);
            values.push(dense_row[col]);
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices, values)
}

fn deterministic_complete_dyadic_matrix(dimension: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for col in 0..=row {
            let value = if row == col {
                f64::from((row % 7) as i16 - 3) / 64.0
            } else {
                let lower_triangle_index = row * (row + 1) / 2 + col;
                let magnitude = f64::from(lower_triangle_index as u16 + 1) / 512.0;
                let sign = if (row * 13 + col * 19 + 5) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                sign * magnitude
            };
            matrix[row][col] = value;
            matrix[col][row] = value;
        }
    }
    matrix
}

fn dense_spd_app_matrix(dimension: usize, shift: f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        let mut offdiag_sum = 0.0;
        for col in 0..row {
            let numerator = ((row * 17 + col * 31 + 11) % 17) as f64 - 8.0;
            let value = (numerator + shift * 0.125) / 2048.0;
            matrix[row][col] = value;
            matrix[col][row] = value;
            offdiag_sum += value.abs();
        }
        for previous_row in matrix.iter().take(row) {
            offdiag_sum += previous_row[row].abs();
        }
        matrix[row][row] = offdiag_sum + 2.0 + row as f64 / 1024.0 + shift / 4096.0;
    }
    matrix
}

fn block_diagonal_multi_root_case() -> DenseCase {
    let block_size = 40;
    let blocks = 3;
    let dimension = block_size * blocks;
    let mut dense = vec![vec![0.0; dimension]; dimension];
    for block in 0..blocks {
        let local = dense_spd_app_matrix(block_size, block as f64);
        let offset = block * block_size;
        for row in 0..block_size {
            for col in 0..block_size {
                dense[offset + row][offset + col] = local[row][col];
            }
        }
    }
    let expected_solution = (0..dimension)
        .map(|index| f64::from((index % 13) as i16 - 6) / 16.0)
        .collect();
    DenseCase {
        dense,
        expected_solution,
    }
}

fn dense_app_boundary_case() -> DenseCase {
    let dimension = 96;
    let dense = deterministic_complete_dyadic_matrix(dimension);
    let expected_solution = (0..dimension)
        .map(|index| f64::from((index % 11) as i16 - 5) / 8.0)
        .collect();
    DenseCase {
        dense,
        expected_solution,
    }
}

fn rust_outcome(case: &DenseCase) -> RustOutcome {
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&case.dense);
    let matrix = SymmetricCscMatrix::new(case.dense.len(), &col_ptrs, &row_indices, Some(&values))
        .expect("valid CSC");
    let rhs = dense_mul(&case.dense, &case.expected_solution);
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("analyse");
    let (mut factor, _) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factorize");
    let solution = factor.solve(&rhs).expect("solve");
    RustOutcome {
        inertia: factor.inertia(),
        pivot_stats: factor.pivot_stats(),
        stored_nnz: factor.stored_nnz(),
        supernode_count: factor.supernode_count(),
        max_supernode_width: factor.max_supernode_width(),
        factor_bytes: factor.factor_bytes(),
        solution_bits: bit_patterns(&solution),
        residual_inf: residual_inf_norm(&case.dense, &solution, &rhs),
        solution,
    }
}

fn rust_outcome_in_pool(threads: usize, case: &DenseCase) -> RustOutcome {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
        .install(|| rust_outcome(case))
}

fn assert_exact_rust_outcome_eq(expected: &RustOutcome, actual: &RustOutcome, label: &str) {
    assert_eq!(actual.inertia, expected.inertia, "{label} inertia");
    assert_eq!(
        actual.pivot_stats, expected.pivot_stats,
        "{label} pivot stats"
    );
    assert_eq!(actual.stored_nnz, expected.stored_nnz, "{label} stored nnz");
    assert_eq!(
        actual.supernode_count, expected.supernode_count,
        "{label} supernode count"
    );
    assert_eq!(
        actual.max_supernode_width, expected.max_supernode_width,
        "{label} max supernode width"
    );
    assert_eq!(
        actual.factor_bytes, expected.factor_bytes,
        "{label} factor bytes"
    );
    assert_eq!(
        actual.solution_bits, expected.solution_bits,
        "{label} solution bits"
    );
    assert_eq!(
        actual.residual_inf.to_bits(),
        expected.residual_inf.to_bits(),
        "{label} residual bits"
    );
}

#[test]
fn parallel_determinism_ssids_rs_rayon_matches_serial_bits() {
    for (label, case) in [
        ("dense APP boundary", dense_app_boundary_case()),
        ("multi-root front tree", block_diagonal_multi_root_case()),
    ] {
        let serial = rust_outcome_in_pool(1, &case);
        assert!(serial.residual_inf <= 1e-9, "{label} serial residual");
        for repetition in 0..8 {
            let parallel = rust_outcome_in_pool(4, &case);
            assert_exact_rust_outcome_eq(
                &serial,
                &parallel,
                &format!("{label} rayon repetition {repetition}"),
            );
        }
    }
}

fn refactorize_sequence_bits(dimension: usize, job: usize) -> Vec<Vec<u64>> {
    let base = dense_spd_app_matrix(dimension, job as f64 / 8.0);
    let (col_ptrs, row_indices, values) = dense_to_complete_lower_csc(&base);
    let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
        .expect("valid base CSC");
    let (symbolic, _) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("analyse");
    let (mut factor, _) =
        factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factorize");

    let mut output = Vec::new();
    for step in 0..5 {
        let dense = dense_spd_app_matrix(dimension, job as f64 / 8.0 + step as f64 / 16.0);
        let (_, _, values) = dense_to_complete_lower_csc(&dense);
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
            .expect("valid refactor CSC");
        let info = factor.refactorize(matrix).expect("refactorize");
        assert!(info.factorization_residual_max_abs <= 1e-9);
        let expected_solution = (0..dimension)
            .map(|index| f64::from(((index + job + step) % 17) as i16 - 8) / 32.0)
            .collect::<Vec<_>>();
        let rhs = dense_mul(&dense, &expected_solution);
        let solution = factor.solve(&rhs).expect("solve");
        assert!(residual_inf_norm(&dense, &solution, &rhs) <= 1e-9);
        output.push(bit_patterns(&solution));
    }
    output
}

#[test]
fn concurrent_ssids_rs_stress_independent_jobs_and_refactors_are_stable() {
    let jobs = 0..32;
    let expected = jobs
        .clone()
        .map(|job| {
            let case = if job % 2 == 0 {
                dense_app_boundary_case()
            } else {
                block_diagonal_multi_root_case()
            };
            let outcome = rust_outcome_in_pool(1, &case);
            let refactor_bits = refactorize_sequence_bits(48 + job % 4, job);
            (outcome, refactor_bits)
        })
        .collect::<Vec<_>>();

    let actual = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("rayon pool")
        .install(|| {
            jobs.into_par_iter()
                .map(|job| {
                    let case = if job % 2 == 0 {
                        dense_app_boundary_case()
                    } else {
                        block_diagonal_multi_root_case()
                    };
                    let outcome = rust_outcome(&case);
                    let refactor_bits = refactorize_sequence_bits(48 + job % 4, job);
                    (outcome, refactor_bits)
                })
                .collect::<Vec<_>>()
        });

    for (job, ((expected_outcome, expected_refactor), (actual_outcome, actual_refactor))) in
        expected.iter().zip(actual.iter()).enumerate()
    {
        assert_exact_rust_outcome_eq(
            expected_outcome,
            actual_outcome,
            &format!("concurrent job {job}"),
        );
        assert_eq!(
            actual_refactor, expected_refactor,
            "concurrent refactor sequence bits for job {job}"
        );
    }
}

fn native_source_feature_enabled() -> bool {
    cfg!(any(
        feature = "native-spral-src",
        feature = "native-spral-src-openmp"
    ))
}

fn native_outcome(case: &DenseCase) -> NativeOutcome {
    let native = NativeSpral::load().expect("load native SPRAL");
    let (col_ptrs, row_indices, values) = dense_to_lower_csc(&case.dense);
    let matrix = SymmetricCscMatrix::new(case.dense.len(), &col_ptrs, &row_indices, Some(&values))
        .expect("valid CSC");
    let rhs = dense_mul(&case.dense, &case.expected_solution);
    let options = NumericFactorOptions::default();
    let mut session = native
        .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
        .expect("native analyse");
    let info = session.factorize(matrix).expect("native factorize");
    let solution = session.solve(&rhs).expect("native solve");
    NativeOutcome {
        inertia: info.inertia,
        pivot_stats: PivotStats {
            two_by_two_pivots: info.two_by_two_pivots,
            delayed_pivots: info.delayed_pivots,
        },
        residual_inf: residual_inf_norm(&case.dense, &solution, &rhs),
        solution,
    }
}

fn encode_native_outcome(outcome: &NativeOutcome) -> String {
    let solution_bits = outcome
        .solution
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{};{};{};{};{};{:016x};{}",
        outcome.inertia.positive,
        outcome.inertia.negative,
        outcome.inertia.zero,
        outcome.pivot_stats.two_by_two_pivots,
        outcome.pivot_stats.delayed_pivots,
        outcome.residual_inf.to_bits(),
        solution_bits
    )
}

fn parse_native_outcome(encoded: &str) -> NativeOutcome {
    let fields = encoded.trim().split(';').collect::<Vec<_>>();
    assert_eq!(fields.len(), 7, "native outcome field count");
    let parse_usize = |index: usize| fields[index].parse::<usize>().expect("usize field");
    let solution = if fields[6].is_empty() {
        Vec::new()
    } else {
        fields[6]
            .split(',')
            .map(|bits| f64::from_bits(u64::from_str_radix(bits, 16).expect("solution bits")))
            .collect()
    };
    NativeOutcome {
        inertia: Inertia {
            positive: parse_usize(0),
            negative: parse_usize(1),
            zero: parse_usize(2),
        },
        pivot_stats: PivotStats {
            two_by_two_pivots: parse_usize(3),
            delayed_pivots: parse_usize(4),
        },
        residual_inf: f64::from_bits(u64::from_str_radix(fields[5], 16).expect("residual bits")),
        solution,
    }
}

fn run_native_child(test_name: &str, omp_threads: usize) -> NativeOutcome {
    let output = Command::new(std::env::current_exe().expect("current test executable"))
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env("SSIDS_RS_NATIVE_THREAD_CHILD", "1")
        .env("AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY", "1")
        .env("OMP_CANCELLATION", "true")
        .env("OMP_NUM_THREADS", omp_threads.to_string())
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .expect("spawn native thread child");
    assert!(
        output.status.success(),
        "native child failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let encoded = stdout
        .lines()
        .find_map(|line| line.strip_prefix("SSIDS_RS_NATIVE_THREAD_OUTCOME\t"))
        .expect("native child outcome line");
    parse_native_outcome(encoded)
}

fn native_thread_bounded_check(test_name: &str) {
    if std::env::var_os("SSIDS_RS_NATIVE_THREAD_CHILD").is_some() {
        let outcome = native_outcome(&dense_app_boundary_case());
        println!(
            "SSIDS_RS_NATIVE_THREAD_OUTCOME\t{}",
            encode_native_outcome(&outcome)
        );
        return;
    }

    if !native_source_feature_enabled() {
        if std::env::var_os("AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY").is_some() {
            panic!(
                "native SPRAL source feature is required for parallel native thread safety checks"
            );
        }
        eprintln!("skipping native thread safety check: native SPRAL source feature is disabled");
        return;
    }

    let rust_reference = rust_outcome_in_pool(1, &dense_app_boundary_case());
    let native_one = run_native_child(test_name, 1);
    let native_four = run_native_child(test_name, 4);

    assert_eq!(native_four.inertia, native_one.inertia);
    assert_eq!(native_four.pivot_stats, native_one.pivot_stats);
    const NATIVE_THREADED_RESIDUAL_TOL: f64 = 1e-8;
    const NATIVE_THREADED_SOLUTION_TOL: f64 = 1e-8;
    assert!(
        native_one.residual_inf <= NATIVE_THREADED_RESIDUAL_TOL,
        "OMP=1 native residual {} exceeds {NATIVE_THREADED_RESIDUAL_TOL}",
        native_one.residual_inf
    );
    assert!(
        native_four.residual_inf <= NATIVE_THREADED_RESIDUAL_TOL,
        "OMP=4 native residual {} exceeds {NATIVE_THREADED_RESIDUAL_TOL}",
        native_four.residual_inf
    );
    let native_thread_delta = delta_inf(&native_one.solution, &native_four.solution);
    assert!(
        native_thread_delta <= NATIVE_THREADED_SOLUTION_TOL,
        "native OMP=1 vs OMP=4 solution delta {native_thread_delta} exceeds {NATIVE_THREADED_SOLUTION_TOL}"
    );
    let rust_native_delta = delta_inf(&rust_reference.solution, &native_four.solution);
    assert!(
        rust_native_delta <= NATIVE_THREADED_SOLUTION_TOL,
        "Rust serial vs native OMP=4 solution delta {rust_native_delta} exceeds {NATIVE_THREADED_SOLUTION_TOL}"
    );
}

#[test]
fn parallel_native_threads_bounded_correctness() {
    native_thread_bounded_check("parallel_native_threads_bounded_correctness");
}

#[test]
#[ignore = "known source-built OpenBLAS OpenMP APP solve correctness failure; run before enabling this path"]
fn parallel_openblas_threads_bounded_correctness() {
    if !cfg!(feature = "native-spral-src-openmp") {
        eprintln!("skipping OpenBLAS OpenMP check: native-spral-src-openmp feature is disabled");
        return;
    }
    native_thread_bounded_check("parallel_openblas_threads_bounded_correctness");
}
