use std::time::{Duration, Instant};

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
use std::fs;
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
use std::path::{Path, PathBuf};
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
use std::process::Command;
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
};

use ssids_rs::{
    NativeOrdering, NativeSpral, NativeSpralAnalyseInfo, NativeSpralFactorInfo,
    NumericFactorOptions, SpralCscTrace, SpralMatchingTrace, SsidsOptions, SymmetricCscMatrix,
    analyse, analyse_with_user_ordering, factorize, spral_matching_trace,
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

fn hash_isize(values: &[isize]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &value| {
        let mixed = hash ^ u64::from_ne_bytes((value as i64).to_ne_bytes());
        mixed.wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hash_option_usize(values: &[Option<usize>]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        let value = value.map_or(u64::MAX, |entry| {
            u64::try_from(entry).unwrap_or(u64::MAX - 1)
        });
        (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hash_f64_bits(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        let mixed = hash ^ value.to_bits();
        mixed.wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn scaling_range(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        })
}

fn csc_hashes(matrix: &SpralCscTrace) -> (u64, u64, u64) {
    (
        hash_usize(&matrix.col_ptrs),
        hash_usize(&matrix.row_indices),
        hash_f64_bits(&matrix.values),
    )
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

#[derive(Debug)]
struct NativeMatchOrderRun {
    flag: i32,
    stat: i32,
    scale_logs: Vec<f64>,
    matching: Vec<Option<usize>>,
    split_matching: Vec<isize>,
    compressed_col_ptrs: Vec<usize>,
    compressed_row_indices: Vec<usize>,
    compressed_metis_flag: i32,
    compressed_metis_stat: i32,
    compressed_metis_perm: Vec<usize>,
    compressed_metis_invp: Vec<usize>,
    order: Vec<usize>,
    scaling: Vec<f64>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeMetisRun {
    flag: i32,
    stat: i32,
    perm: Vec<usize>,
    invp: Vec<usize>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeMetisPruneRun {
    stat: i32,
    pruning_active: bool,
    kept_vertex_count: usize,
    directed_edge_count: usize,
    piperm: Vec<usize>,
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    vertex_weights: Vec<isize>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeMmdRun {
    stat: i32,
    perm: Vec<usize>,
    invp: Vec<usize>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeSeparatorRun {
    stat: i32,
    mincut: isize,
    part_weights: [isize; 3],
    where_part: Vec<usize>,
    boundary: Vec<usize>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeNodeNdTopSeparatorRun {
    stat: i32,
    compression_active: bool,
    nseps: usize,
    compressed_vertex_count: usize,
    compressed_edge_count: usize,
    compressed_cptr: Vec<usize>,
    compressed_cind: Vec<usize>,
    separator: NativeSeparatorRun,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeMetisCcComponentsRun {
    stat: i32,
    separator: NativeSeparatorRun,
    cptr: Vec<usize>,
    cind: Vec<usize>,
    subgraph_labels: Vec<Vec<usize>>,
    subgraph_offsets: Vec<Vec<usize>>,
    subgraph_neighbors: Vec<Vec<usize>>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeMetisRandomRun {
    stat: i32,
    sequence: Vec<usize>,
    permutation: Vec<usize>,
}

#[derive(Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct NativeNodeBisectionStagesRun {
    stat: i32,
    coarse_vertex_count: usize,
    coarse_edge_count: usize,
    coarse_xadj: Vec<usize>,
    coarse_adjncy: Vec<usize>,
    coarse_weights: Vec<usize>,
    original_cmap: Vec<usize>,
    coarse_labels: Vec<usize>,
    edge_mincut: isize,
    edge_part_weights: [isize; 2],
    edge_where: Vec<usize>,
    edge_boundary: Vec<usize>,
    edge_id: Vec<isize>,
    edge_ed: Vec<isize>,
    initial: NativeSeparatorRun,
    final_separator: NativeSeparatorRun,
}

#[derive(Clone, Debug)]
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
struct SpralSrcMetadata {
    fc: String,
    module_dir: PathBuf,
    match_order_source: PathBuf,
    metis_include_dir: PathBuf,
    metis_source_lib_dir: PathBuf,
    gklib_include_dir: PathBuf,
    spral_lflags: String,
    runtime_link_dirs: Vec<PathBuf>,
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_MATCH_ORDER_SHIM_SOURCE: &str = r#"
program spral_match_order_shim
  use iso_fortran_env, only : int64
  use spral_match_order_debug, only : mo_scale, mo_split, debug_ncomp, debug_ptr3, debug_row3
  use spral_metis_wrapper, only : metis_order
  implicit none

  integer :: n, ne, flag, stat, i, j, metis_flag, metis_stat
  integer(int64), allocatable :: ptr(:), ptr2(:)
  integer, allocatable :: row(:), row2(:), order(:), cperm(:), metis_perm(:), metis_invp(:)
  real(kind(0d0)), allocatable :: val(:), val2(:), scale(:)
  integer(int64) :: bits, k
  character(len=4096) :: input_path

  call get_command_argument(1, input_path)
  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10, *) n, ne
  allocate(ptr(n+1), row(ne), val(ne))
  do i = 1, n + 1
     read(10, *) ptr(i)
  end do
  do i = 1, ne
     read(10, *) row(i)
  end do
  do i = 1, ne
     read(10, '(Z16)') bits
     val(i) = transfer(bits, val(i))
  end do
  close(10)

  allocate(ptr2(n+1), row2(ne), val2(ne), order(n), cperm(n), scale(n))
  k = 1
  do i = 1, n
     ptr2(i) = k
     do j = ptr(i), ptr(i+1)-1
        if (val(j) .eq. 0.0) cycle
        row2(k) = row(j)
        val2(k) = abs(val(j))
        k = k + 1
     end do
  end do
  ptr2(n+1) = k

  flag = 0
  stat = 0
  call mo_scale(n, ptr2, row2, val2, scale, flag, stat, perm=cperm)

  write(*, *) flag, stat
  do i = 1, n
     bits = transfer(scale(i), bits)
     write(*, '(Z16.16)') bits
  end do
  do i = 1, n
     write(*, '(I0)') cperm(i)
  end do

  if (flag .ge. 0) then
     call mo_split(n, row2, ptr2, order, cperm, flag, stat)
  end if

  do i = 1, n
     write(*, '(I0)') cperm(i)
  end do
  write(*, '(I0)') debug_ncomp
  if (debug_ncomp .gt. 0) then
     do i = 1, debug_ncomp + 1
        write(*, '(I0)') debug_ptr3(i)
     end do
     do i = 1, debug_ptr3(debug_ncomp+1) - 1
        write(*, '(I0)') debug_row3(i)
     end do
     allocate(metis_perm(debug_ncomp), metis_invp(debug_ncomp))
     call metis_order(debug_ncomp, debug_ptr3, debug_row3, metis_perm, metis_invp, metis_flag, metis_stat)
     write(*, '(I0)') metis_flag
     write(*, '(I0)') metis_stat
     do i = 1, debug_ncomp
        write(*, '(I0)') metis_perm(i)
     end do
     do i = 1, debug_ncomp
        write(*, '(I0)') metis_invp(i)
     end do
  else
     write(*, '(I0)') 0
     write(*, '(I0)') 0
  end if
  do i = 1, n
     write(*, '(I0)') order(i)
  end do

  scale(1:n) = exp(scale(1:n))
  do i = 1, n
     bits = transfer(scale(i), bits)
     write(*, '(Z16.16)') bits
  end do
end program spral_match_order_shim
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_METIS_ORDER_SHIM_SOURCE: &str = r#"
program spral_metis_order_shim
  use iso_fortran_env, only : int64
  use spral_metis_wrapper, only : metis_order
  implicit none

  integer :: n, ne, flag, stat, i
  integer(int64), allocatable :: ptr(:)
  integer, allocatable :: row(:), perm(:), invp(:)
  character(len=4096) :: input_path

  call get_command_argument(1, input_path)
  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10, *) n, ne
  allocate(ptr(n+1), row(ne), perm(n), invp(n))
  do i = 1, n + 1
     read(10, *) ptr(i)
  end do
  do i = 1, ne
     read(10, *) row(i)
  end do
  close(10)

  flag = 0
  stat = 0
  invp(:) = -1
  call metis_order(n, ptr, row, perm, invp, flag, stat)
  write(*, '(I0)') flag
  write(*, '(I0)') stat
  do i = 1, n
     write(*, '(I0)') perm(i)
  end do
  do i = 1, n
     write(*, '(I0)') invp(i)
  end do
end program spral_metis_order_shim
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_METIS_NODE_ND_OPTIONS_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metis.h"

static int lower_to_full(idx_t n, idx_t ne, idx_t *ptr, idx_t *rows,
    idx_t **r_xadj, idx_t **r_adjncy) {
  idx_t i, j, col, row, total;
  idx_t *counts, *next, *xadj, *adjncy;

  counts = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!counts || !next) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  free(counts);
  free(next);
  *r_xadj = xadj;
  *r_adjncy = adjncy;
  return 0;
}

int main(int argc, char **argv) {
  FILE *input;
  idx_t n, ne, i, compress, ccorder, pfactor, options[METIS_NOPTIONS];
  idx_t *ptr, *rows, *xadj, *adjncy, *perm, *iperm;
  int status;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX " %" SCIDX " %" SCIDX " %" SCIDX,
      &n, &ne, &compress, &ccorder, &pfactor) != 5) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }

  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  perm = calloc((size_t)n, sizeof(idx_t));
  iperm = calloc((size_t)n, sizeof(idx_t));
  if (!ptr || !rows || !perm || !iperm) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  if (lower_to_full(n, ne, ptr, rows, &xadj, &adjncy) != 0) return 3;

  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_COMPRESS] = compress;
  options[METIS_OPTION_CCORDER] = ccorder;
  options[METIS_OPTION_PFACTOR] = pfactor;
  status = METIS_NodeND(&n, xadj, adjncy, NULL, options, perm, iperm);

  printf("%d\n", status);
  printf("0\n");
  for (i = 0; i < n; i++) printf("%" PRIDX "\n", iperm[i] + 1);
  for (i = 0; i < n; i++) printf("%" PRIDX "\n", perm[i] + 1);

  free(ptr); free(rows); free(xadj); free(adjncy); free(perm); free(iperm);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_METIS_PRUNE_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metislib.h"

static int lower_to_full(idx_t n, idx_t ne, idx_t *ptr, idx_t *rows,
    idx_t **r_xadj, idx_t **r_adjncy) {
  idx_t i, j, col, row, total;
  idx_t *counts, *next, *xadj, *adjncy;

  counts = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!counts || !next) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  free(counts);
  free(next);
  *r_xadj = xadj;
  *r_adjncy = adjncy;
  return 0;
}

int main(int argc, char **argv) {
  FILE *input;
  ctrl_t *ctrl;
  graph_t *graph;
  idx_t n, ne, i, pfactor, options[METIS_NOPTIONS];
  idx_t *ptr, *rows, *xadj, *adjncy, *piperm;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX " %" SCIDX, &n, &ne, &pfactor) != 3) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }
  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  piperm = calloc((size_t)n, sizeof(idx_t));
  if (!ptr || !rows || !piperm) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  if (lower_to_full(n, ne, ptr, rows, &xadj, &adjncy) != 0) return 3;

  if (!gk_malloc_init()) return 3;
  METIS_SetDefaultOptions(options);
  ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL);
  if (ctrl == NULL) return 4;
  graph = PruneGraph(ctrl, n, xadj, adjncy, NULL, piperm, 0.1 * (real_t)pfactor);

  printf("0\n");
  printf("%d\n", graph != NULL);
  printf("%" PRIDX "\n", graph != NULL ? graph->nvtxs : n);
  printf("%" PRIDX "\n", graph != NULL ? graph->nedges : xadj[n]);
  for (i = 0; i < n; i++) printf("%" PRIDX "\n", graph != NULL ? piperm[i] : i);
  if (graph != NULL) {
    for (i = 0; i < graph->nvtxs + 1; i++) printf("%" PRIDX "\n", graph->xadj[i]);
    for (i = 0; i < graph->nedges; i++) printf("%" PRIDX "\n", graph->adjncy[i]);
    for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->vwgt[i]);
    FreeGraph(&graph);
  }

  FreeCtrl(&ctrl);
  free(ptr); free(rows); free(xadj); free(adjncy); free(piperm);
  gk_malloc_cleanup(0);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_METIS_CC_COMPONENTS_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metislib.h"

static int lower_to_full(idx_t n, idx_t ne, idx_t *ptr, idx_t *rows,
    idx_t **r_xadj, idx_t **r_adjncy) {
  idx_t i, j, col, row, total;
  idx_t *counts, *next, *xadj, *adjncy;

  counts = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!counts || !next) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  free(counts);
  free(next);
  *r_xadj = xadj;
  *r_adjncy = adjncy;
  return 0;
}

static graph_t *prepare_graph(ctrl_t *ctrl, idx_t n, idx_t *xadj, idx_t *adjncy) {
  graph_t *graph = NULL;
  idx_t nnvtxs = 0;
  idx_t *piperm = NULL, *cptr = NULL, *cind = NULL;

  if (ctrl->pfactor > 0.0) {
    piperm = imalloc(n, "cc shim piperm");
    graph = PruneGraph(ctrl, n, xadj, adjncy, NULL, piperm, ctrl->pfactor);
    if (graph == NULL) {
      gk_free((void **)&piperm, LTERM);
      ctrl->pfactor = 0.0;
    } else {
      nnvtxs = graph->nvtxs;
      ctrl->compress = 0;
    }
  }

  if (ctrl->compress) {
    cptr = imalloc(n + 1, "cc shim cptr");
    cind = imalloc(n, "cc shim cind");
    graph = CompressGraph(ctrl, n, xadj, adjncy, NULL, cptr, cind);
    if (graph == NULL) {
      gk_free((void **)&cptr, &cind, LTERM);
      ctrl->compress = 0;
    } else {
      nnvtxs = graph->nvtxs;
      ctrl->cfactor = 1.0 * n / nnvtxs;
      if (ctrl->cfactor > 1.5 && ctrl->nseps == 1)
        ctrl->nseps = 2;
    }
  }

  if (ctrl->pfactor == 0.0 && ctrl->compress == 0)
    graph = SetupGraph(ctrl, n, 1, xadj, adjncy, NULL, NULL, NULL);

  gk_free((void **)&piperm, &cptr, &cind, LTERM);
  return graph;
}

int main(int argc, char **argv) {
  FILE *input;
  ctrl_t *ctrl;
  graph_t *graph, **sgraphs;
  idx_t n, ne, i, j, k, compress, ccorder, pfactor, options[METIS_NOPTIONS], ncmps;
  idx_t *ptr, *rows, *xadj, *adjncy, *cptr, *cind;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX " %" SCIDX " %" SCIDX " %" SCIDX,
      &n, &ne, &compress, &ccorder, &pfactor) != 5) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }
  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  if (!ptr || !rows) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  if (lower_to_full(n, ne, ptr, rows, &xadj, &adjncy) != 0) return 3;

  if (!gk_malloc_init()) return 3;
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_COMPRESS] = compress;
  options[METIS_OPTION_CCORDER] = ccorder;
  options[METIS_OPTION_PFACTOR] = pfactor;
  ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL);
  if (ctrl == NULL) return 4;

  graph = prepare_graph(ctrl, n, xadj, adjncy);
  if (graph == NULL) return 4;
  AllocateWorkSpace(ctrl, graph);
  MlevelNodeBisectionMultiple(ctrl, graph);

  cptr = iwspacemalloc(ctrl, graph->nvtxs + 1);
  cind = iwspacemalloc(ctrl, graph->nvtxs);
  ncmps = FindSepInducedComponents(ctrl, graph, cptr, cind);
  sgraphs = SplitGraphOrderCC(ctrl, graph, ncmps, cptr, cind);

  printf("0\n");
  printf("%" PRIDX "\n", graph->nvtxs);
  printf("%" PRIDX "\n", graph->mincut);
  printf("%" PRIDX "\n", graph->pwgts[0]);
  printf("%" PRIDX "\n", graph->pwgts[1]);
  printf("%" PRIDX "\n", graph->pwgts[2]);
  printf("%" PRIDX "\n", graph->nbnd);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->where[i]);
  for (i = 0; i < graph->nbnd; i++) printf("%" PRIDX "\n", graph->bndind[i]);
  printf("%" PRIDX "\n", ncmps);
  for (i = 0; i < ncmps + 1; i++) printf("%" PRIDX "\n", cptr[i]);
  for (i = 0; i < cptr[ncmps]; i++) printf("%" PRIDX "\n", cind[i]);
  for (i = 0; i < ncmps; i++) {
    printf("%" PRIDX "\n", sgraphs[i]->nvtxs);
    printf("%" PRIDX "\n", sgraphs[i]->nedges);
    for (j = 0; j < sgraphs[i]->nvtxs; j++) printf("%" PRIDX "\n", sgraphs[i]->label[j]);
    for (j = 0; j < sgraphs[i]->nvtxs + 1; j++) printf("%" PRIDX "\n", sgraphs[i]->xadj[j]);
    for (k = 0; k < sgraphs[i]->nedges; k++) printf("%" PRIDX "\n", sgraphs[i]->adjncy[k]);
  }

  for (i = 0; i < ncmps; i++) FreeGraph(&sgraphs[i]);
  gk_free((void **)&sgraphs, LTERM);
  FreeGraph(&graph);
  FreeCtrl(&ctrl);
  free(ptr); free(rows); free(xadj); free(adjncy);
  gk_malloc_cleanup(0);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_MMD_ORDER_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metis.h"

void libmetis__genmmd(idx_t neqns, idx_t *xadj, idx_t *adjncy, idx_t *invp,
    idx_t *perm, idx_t delta, idx_t *head, idx_t *qsize, idx_t *list,
    idx_t *marker, idx_t maxint, idx_t *ncsub);

int main(int argc, char **argv) {
  FILE *input;
  idx_t n, ne, i, j, col, row, total, ncsub;
  idx_t *ptr, *rows, *counts, *ends, *next, *xadj, *adjncy;
  idx_t *perm, *invp, *head, *qsize, *list, *marker;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX, &n, &ne) != 2) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }

  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  counts = calloc((size_t)n, sizeof(idx_t));
  ends = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!ptr || !rows || !counts || !ends || !next) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    ends[i] = total;
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  for (i = 0; i < total; i++) adjncy[i]++;
  for (i = 0; i < n + 1; i++) xadj[i]++;

  perm = calloc((size_t)n + 5, sizeof(idx_t));
  invp = calloc((size_t)n + 5, sizeof(idx_t));
  head = calloc((size_t)n + 5, sizeof(idx_t));
  qsize = calloc((size_t)n + 5, sizeof(idx_t));
  list = calloc((size_t)n + 5, sizeof(idx_t));
  marker = calloc((size_t)n + 5, sizeof(idx_t));
  if (!perm || !invp || !head || !qsize || !list || !marker) return 3;

  libmetis__genmmd(n, xadj, adjncy, invp, perm, 1, head, qsize, list, marker, IDX_MAX, &ncsub);

  printf("0\n");
  for (i = 0; i < n; i++) printf("%" PRIDX "\n", perm[i]);
  for (i = 0; i < n; i++) printf("%" PRIDX "\n", invp[i]);

  free(ptr); free(rows); free(counts); free(ends); free(next);
  free(xadj); free(adjncy);
  free(perm); free(invp); free(head); free(qsize); free(list); free(marker);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_SEPARATOR_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metislib.h"

static int lower_to_full(idx_t n, idx_t ne, idx_t *ptr, idx_t *rows,
    idx_t **r_xadj, idx_t **r_adjncy) {
  idx_t i, j, col, row, total;
  idx_t *counts, *next, *xadj, *adjncy;

  counts = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!counts || !next) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  free(counts);
  free(next);
  *r_xadj = xadj;
  *r_adjncy = adjncy;
  return 0;
}

int main(int argc, char **argv) {
  FILE *input;
  ctrl_t *ctrl;
  graph_t *graph;
  idx_t n, ne, i, options[METIS_NOPTIONS];
  idx_t *ptr, *rows, *xadj, *adjncy;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX, &n, &ne) != 2) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }

  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  if (!ptr || !rows) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  if (lower_to_full(n, ne, ptr, rows, &xadj, &adjncy) != 0) return 3;
  free(ptr);
  free(rows);

  if (!gk_malloc_init()) return 3;
  METIS_SetDefaultOptions(options);
  ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL);
  if (ctrl == NULL) return 4;
  ctrl->compress = 0;
  graph = SetupGraph(ctrl, n, 1, xadj, adjncy, NULL, NULL, NULL);
  if (graph == NULL) return 4;
  AllocateWorkSpace(ctrl, graph);

  MlevelNodeBisectionMultiple(ctrl, graph);

  printf("0\n");
  printf("%" PRIDX "\n", graph->mincut);
  printf("%" PRIDX "\n", graph->pwgts[0]);
  printf("%" PRIDX "\n", graph->pwgts[1]);
  printf("%" PRIDX "\n", graph->pwgts[2]);
  printf("%" PRIDX "\n", graph->nbnd);
  for (i = 0; i < n; i++) printf("%" PRIDX "\n", graph->where[i]);
  for (i = 0; i < graph->nbnd; i++) printf("%" PRIDX "\n", graph->bndind[i]);

  FreeGraph(&graph);
  FreeCtrl(&ctrl);
  free(xadj);
  free(adjncy);
  gk_malloc_cleanup(0);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_NODE_ND_TOP_SEPARATOR_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metislib.h"

static int lower_to_full(idx_t n, idx_t ne, idx_t *ptr, idx_t *rows,
    idx_t **r_xadj, idx_t **r_adjncy) {
  idx_t i, j, col, row, total;
  idx_t *counts, *next, *xadj, *adjncy;

  counts = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!counts || !next) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  free(counts);
  free(next);
  *r_xadj = xadj;
  *r_adjncy = adjncy;
  return 0;
}

static void print_separator_state(graph_t *graph) {
  idx_t i;
  printf("%" PRIDX "\n", graph->mincut);
  printf("%" PRIDX "\n", graph->pwgts[0]);
  printf("%" PRIDX "\n", graph->pwgts[1]);
  printf("%" PRIDX "\n", graph->pwgts[2]);
  printf("%" PRIDX "\n", graph->nbnd);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->where[i]);
  for (i = 0; i < graph->nbnd; i++) printf("%" PRIDX "\n", graph->bndind[i]);
}

int main(int argc, char **argv) {
  FILE *input;
  ctrl_t *ctrl;
  graph_t *graph = NULL;
  idx_t n, ne, i, options[METIS_NOPTIONS], compressed = 0, nnvtxs = 0, cind_len = 0;
  idx_t *ptr, *rows, *xadj, *adjncy, *cptr = NULL, *cind = NULL;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX, &n, &ne) != 2) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }

  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  if (!ptr || !rows) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  if (lower_to_full(n, ne, ptr, rows, &xadj, &adjncy) != 0) return 3;
  free(ptr);
  free(rows);

  if (!gk_malloc_init()) return 3;
  METIS_SetDefaultOptions(options);
  ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL);
  if (ctrl == NULL) return 4;

  if (ctrl->compress) {
    cptr = calloc((size_t)n + 1, sizeof(idx_t));
    cind = calloc((size_t)n, sizeof(idx_t));
    if (!cptr || !cind) return 3;
    graph = CompressGraph(ctrl, n, xadj, adjncy, NULL, cptr, cind);
    if (graph == NULL) {
      free(cptr);
      free(cind);
      cptr = NULL;
      cind = NULL;
      ctrl->compress = 0;
    } else {
      compressed = 1;
      nnvtxs = graph->nvtxs;
      cind_len = cptr[nnvtxs];
      ctrl->cfactor = 1.0 * n / nnvtxs;
      if (ctrl->cfactor > 1.5 && ctrl->nseps == 1)
        ctrl->nseps = 2;
    }
  }

  if (!compressed)
    graph = SetupGraph(ctrl, n, 1, xadj, adjncy, NULL, NULL, NULL);
  if (graph == NULL) return 4;

  AllocateWorkSpace(ctrl, graph);
  MlevelNodeBisectionMultiple(ctrl, graph);

  printf("0\n");
  printf("%" PRIDX "\n", compressed);
  printf("%" PRIDX "\n", ctrl->nseps);
  printf("%" PRIDX "\n", graph->nvtxs);
  printf("%" PRIDX "\n", graph->nedges);
  printf("%" PRIDX "\n", cind_len);
  if (compressed) {
    for (i = 0; i <= graph->nvtxs; i++) printf("%" PRIDX "\n", cptr[i]);
    for (i = 0; i < cind_len; i++) printf("%" PRIDX "\n", cind[i]);
  } else {
    for (i = 0; i <= graph->nvtxs; i++) printf("%" PRIDX "\n", i);
  }
  print_separator_state(graph);

  FreeGraph(&graph);
  FreeCtrl(&ctrl);
  free(cptr);
  free(cind);
  free(xadj);
  free(adjncy);
  gk_malloc_cleanup(0);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_METIS_RANDOM_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metislib.h"

int main(int argc, char **argv) {
  idx_t seed, count, n, nshuffles, flag, i;
  idx_t *perm;

  if (argc != 6) {
    fprintf(stderr, "usage: %s seed count n nshuffles flag\n", argv[0]);
    return 2;
  }
  seed = (idx_t)strtol(argv[1], NULL, 10);
  count = (idx_t)strtol(argv[2], NULL, 10);
  n = (idx_t)strtol(argv[3], NULL, 10);
  nshuffles = (idx_t)strtol(argv[4], NULL, 10);
  flag = (idx_t)strtol(argv[5], NULL, 10);

  InitRandom(seed);
  printf("0\n");
  printf("%" PRIDX "\n", count);
  for (i = 0; i < count; i++) {
    printf("%" PRIDX "\n", irand());
  }

  printf("%" PRIDX "\n", n);
  perm = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
  if (perm == NULL) return 3;
  for (i = 0; i < n; i++) {
    perm[i] = n - 1 - i;
  }
  if (n > 0) {
    irandArrayPermute(n, perm, nshuffles, flag);
  }
  for (i = 0; i < n; i++) {
    printf("%" PRIDX "\n", perm[i]);
  }

  free(perm);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
const NATIVE_NODE_BISECTION_STAGE_SHIM_SOURCE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "metislib.h"

static int lower_to_full(idx_t n, idx_t ne, idx_t *ptr, idx_t *rows,
    idx_t **r_xadj, idx_t **r_adjncy) {
  idx_t i, j, col, row, total;
  idx_t *counts, *next, *xadj, *adjncy;

  counts = calloc((size_t)n, sizeof(idx_t));
  next = calloc((size_t)n, sizeof(idx_t));
  if (!counts || !next) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row != col) {
        counts[row]++;
        counts[col]++;
      }
    }
  }

  total = 0;
  for (i = 0; i < n; i++) {
    total += counts[i];
    next[i] = total;
  }

  xadj = calloc((size_t)n + 1, sizeof(idx_t));
  adjncy = calloc((size_t)total, sizeof(idx_t));
  if (!xadj || !adjncy) return 3;

  for (col = 0; col < n; col++) {
    for (j = ptr[col]; j < ptr[col + 1]; j++) {
      row = rows[j];
      if (row == col) continue;
      next[row]--;
      adjncy[next[row]] = col;
      next[col]--;
      adjncy[next[col]] = row;
    }
  }
  for (i = 0; i < n; i++) xadj[i] = next[i];
  xadj[n] = total;

  free(counts);
  free(next);
  *r_xadj = xadj;
  *r_adjncy = adjncy;
  return 0;
}

static void print_separator_state(graph_t *graph) {
  idx_t i;
  printf("%" PRIDX "\n", graph->mincut);
  printf("%" PRIDX "\n", graph->pwgts[0]);
  printf("%" PRIDX "\n", graph->pwgts[1]);
  printf("%" PRIDX "\n", graph->pwgts[2]);
  printf("%" PRIDX "\n", graph->nbnd);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->where[i]);
  for (i = 0; i < graph->nbnd; i++) printf("%" PRIDX "\n", graph->bndind[i]);
}

static void print_edge_partition_state(graph_t *graph) {
  idx_t i;
  printf("%" PRIDX "\n", graph->mincut);
  printf("%" PRIDX "\n", graph->pwgts[0]);
  printf("%" PRIDX "\n", graph->pwgts[1]);
  printf("%" PRIDX "\n", graph->nbnd);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->where[i]);
  for (i = 0; i < graph->nbnd; i++) printf("%" PRIDX "\n", graph->bndind[i]);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->id[i]);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->ed[i]);
}

int main(int argc, char **argv) {
  FILE *input;
  ctrl_t *ctrl;
  graph_t *graph, *cgraph;
  real_t ntpwgts[2] = {0.5, 0.5};
  idx_t n, ne, i, options[METIS_NOPTIONS], niparts;
  idx_t *ptr, *rows, *xadj, *adjncy;

  if (argc != 2) {
    fprintf(stderr, "usage: %s input\n", argv[0]);
    return 2;
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    perror("open input");
    return 2;
  }
  if (fscanf(input, "%" SCIDX " %" SCIDX, &n, &ne) != 2) {
    fprintf(stderr, "failed to read header\n");
    return 2;
  }

  ptr = calloc((size_t)n + 1, sizeof(idx_t));
  rows = calloc((size_t)ne, sizeof(idx_t));
  if (!ptr || !rows) return 3;

  for (i = 0; i < n + 1; i++) {
    if (fscanf(input, "%" SCIDX, &ptr[i]) != 1) return 2;
    ptr[i]--;
  }
  for (i = 0; i < ne; i++) {
    if (fscanf(input, "%" SCIDX, &rows[i]) != 1) return 2;
    rows[i]--;
  }
  fclose(input);

  if (lower_to_full(n, ne, ptr, rows, &xadj, &adjncy) != 0) return 3;
  free(ptr);
  free(rows);

  if (!gk_malloc_init()) return 3;
  METIS_SetDefaultOptions(options);
  ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL);
  if (ctrl == NULL) return 4;
  ctrl->compress = 0;
  graph = SetupGraph(ctrl, n, 1, xadj, adjncy, NULL, NULL, NULL);
  if (graph == NULL) return 4;
  AllocateWorkSpace(ctrl, graph);

  ctrl->CoarsenTo = graph->nvtxs / 8;
  if (ctrl->CoarsenTo > 100)
    ctrl->CoarsenTo = 100;
  else if (ctrl->CoarsenTo < 40)
    ctrl->CoarsenTo = 40;

  cgraph = CoarsenGraph(ctrl, graph);
  niparts = gk_max(1, (cgraph->nvtxs <= ctrl->CoarsenTo ? LARGENIPARTS / 2 : LARGENIPARTS));

  printf("0\n");
  printf("%" PRIDX "\n", cgraph->nvtxs);
  printf("%" PRIDX "\n", cgraph->nedges);
  for (i = 0; i <= cgraph->nvtxs; i++) printf("%" PRIDX "\n", cgraph->xadj[i]);
  for (i = 0; i < cgraph->nedges; i++) printf("%" PRIDX "\n", cgraph->adjncy[i]);
  for (i = 0; i < cgraph->nvtxs; i++) printf("%" PRIDX "\n", cgraph->vwgt[i]);
  for (i = 0; i < graph->nvtxs; i++) printf("%" PRIDX "\n", graph->cmap[i]);
  for (i = 0; i < cgraph->nvtxs; i++) printf("%" PRIDX "\n", cgraph->label == NULL ? i : cgraph->label[i]);

  Setup2WayBalMultipliers(ctrl, cgraph, ntpwgts);
  if (cgraph->nedges == 0)
    RandomBisection(ctrl, cgraph, ntpwgts, niparts);
  else
    GrowBisection(ctrl, cgraph, ntpwgts, niparts);
  Compute2WayPartitionParams(ctrl, cgraph);
  print_edge_partition_state(cgraph);
  ConstructSeparator(ctrl, cgraph);
  print_separator_state(cgraph);

  Refine2WayNode(ctrl, graph, cgraph);
  print_separator_state(graph);

  FreeGraph(&graph);
  FreeCtrl(&ctrl);
  free(xadj);
  free(adjncy);
  gk_malloc_cleanup(0);
  return 0;
}
"#;

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_match_order_metis(trace: &SpralMatchingTrace) -> Option<NativeMatchOrderRun> {
    match run_native_match_order_metis_impl(trace) {
        Ok(run) => Some(run),
        Err(error) => {
            eprintln!("native_phase match_order_metis_direct unavailable: {error}");
            None
        }
    }
}

#[cfg(not(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
)))]
fn run_native_match_order_metis(_trace: &SpralMatchingTrace) -> Option<NativeMatchOrderRun> {
    None
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_match_order_metis_impl(
    trace: &SpralMatchingTrace,
) -> Result<NativeMatchOrderRun, String> {
    let shim = native_match_order_shim()?;
    let input = native_match_order_input(trace);
    let input_path = unique_shim_input_path(&shim, "spral-match-order-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_match_order_output(trace.expanded_full.dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_metis_order_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<NativeMetisRun, String> {
    let shim = native_metis_order_shim()?;
    let input = native_metis_order_input(dimension, col_ptrs, row_indices);
    let input_path = unique_shim_input_path(&shim, "spral-metis-order-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_metis_order_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_metis_node_nd_options_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    options: metis_ordering::MetisNodeNdOptions,
) -> Result<NativeMetisRun, String> {
    let shim = native_metis_node_nd_options_shim()?;
    let input = native_metis_node_nd_options_input(dimension, col_ptrs, row_indices, options);
    let input_path = unique_shim_input_path(&shim, "spral-metis-node-nd-options-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_metis_order_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_metis_prune_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    pfactor: usize,
) -> Result<NativeMetisPruneRun, String> {
    let shim = native_metis_prune_shim()?;
    let input = native_metis_prune_input(dimension, col_ptrs, row_indices, pfactor);
    let input_path = unique_shim_input_path(&shim, "spral-metis-prune-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_metis_prune_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_metis_cc_components_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    options: metis_ordering::MetisNodeNdOptions,
) -> Result<NativeMetisCcComponentsRun, String> {
    let shim = native_metis_cc_components_shim()?;
    let input = native_metis_node_nd_options_input(dimension, col_ptrs, row_indices, options);
    let input_path = unique_shim_input_path(&shim, "spral-metis-cc-components-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_metis_cc_components_output(&output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_mmd_order_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<NativeMmdRun, String> {
    let shim = native_mmd_order_shim()?;
    let input = native_metis_order_input(dimension, col_ptrs, row_indices);
    let input_path = unique_shim_input_path(&shim, "spral-mmd-order-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_mmd_order_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_separator_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<NativeSeparatorRun, String> {
    let shim = native_separator_shim()?;
    let input = native_metis_order_input(dimension, col_ptrs, row_indices);
    let input_path = unique_shim_input_path(&shim, "spral-separator-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_separator_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_node_nd_top_separator_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<NativeNodeNdTopSeparatorRun, String> {
    let shim = native_node_nd_top_separator_shim()?;
    let input = native_metis_order_input(dimension, col_ptrs, row_indices);
    let input_path = unique_shim_input_path(&shim, "spral-node-nd-top-separator-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_node_nd_top_separator_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_metis_random(
    seed: i32,
    count: usize,
    n: usize,
    nshuffles: usize,
    flag: bool,
) -> Result<NativeMetisRandomRun, String> {
    let shim = native_metis_random_shim()?;
    let output = Command::new(&shim)
        .arg(seed.to_string())
        .arg(count.to_string())
        .arg(n.to_string())
        .arg(nshuffles.to_string())
        .arg(if flag { "1" } else { "0" })
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_metis_random_output(&output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn run_native_node_bisection_stages_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<NativeNodeBisectionStagesRun, String> {
    let shim = native_node_bisection_stage_shim()?;
    let input = native_metis_order_input(dimension, col_ptrs, row_indices);
    let input_path = unique_shim_input_path(&shim, "spral-node-bisection-stage-input");
    fs::write(&input_path, input)
        .map_err(|error| format!("failed to write {}: {error}", input_path.display()))?;
    let output = Command::new(&shim)
        .arg(&input_path)
        .envs(native_match_order_runtime_env()?)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", shim.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            shim.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_native_node_bisection_stages_output(dimension, &output.stdout)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn unique_shim_input_path(shim: &Path, prefix: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let serial = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    shim.with_file_name(format!("{prefix}-{}-{serial}.txt", std::process::id()))
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_match_order_input(trace: &SpralMatchingTrace) -> String {
    let mut input = String::new();
    input.push_str(&format!(
        "{} {}\n",
        trace.expanded_full.dimension,
        trace.expanded_full.values.len()
    ));
    for &entry in &trace.expanded_full.col_ptrs {
        input.push_str(&format!("{}\n", entry + 1));
    }
    for &entry in &trace.expanded_full.row_indices {
        input.push_str(&format!("{}\n", entry + 1));
    }
    for &entry in &trace.expanded_full.values {
        input.push_str(&format!("{:016x}\n", entry.to_bits()));
    }
    input
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_order_input(dimension: usize, col_ptrs: &[usize], row_indices: &[usize]) -> String {
    let mut input = String::new();
    input.push_str(&format!("{} {}\n", dimension, row_indices.len()));
    for &entry in col_ptrs {
        input.push_str(&format!("{}\n", entry + 1));
    }
    for &entry in row_indices {
        input.push_str(&format!("{}\n", entry + 1));
    }
    input
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_node_nd_options_input(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    options: metis_ordering::MetisNodeNdOptions,
) -> String {
    let mut input = String::new();
    input.push_str(&format!(
        "{} {} {} {} {}\n",
        dimension,
        row_indices.len(),
        usize::from(options.compress),
        usize::from(options.ccorder),
        options.pfactor
    ));
    for &entry in col_ptrs {
        input.push_str(&format!("{}\n", entry + 1));
    }
    for &entry in row_indices {
        input.push_str(&format!("{}\n", entry + 1));
    }
    input
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_prune_input(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    pfactor: usize,
) -> String {
    let mut input = String::new();
    input.push_str(&format!(
        "{} {} {}\n",
        dimension,
        row_indices.len(),
        pfactor
    ));
    for &entry in col_ptrs {
        input.push_str(&format!("{}\n", entry + 1));
    }
    for &entry in row_indices {
        input.push_str(&format!("{}\n", entry + 1));
    }
    input
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_metis_order_output(
    dimension: usize,
    output: &[u8],
) -> Result<NativeMetisRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native metis_order output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let flag = tokens
        .next()
        .ok_or_else(|| "native metis_order output missing flag".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native metis_order flag parse failed: {error}"))?;
    let stat = tokens
        .next()
        .ok_or_else(|| "native metis_order output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native metis_order stat parse failed: {error}"))?;
    let mut perm = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native metis_order output missing perm[{index}]"))?
            .parse::<isize>()
            .map_err(|error| format!("native metis_order perm[{index}] parse failed: {error}"))?;
        perm.push(
            usize::try_from(entry - 1)
                .map_err(|_| format!("native metis_order perm[{index}]={entry} is invalid"))?,
        );
    }
    let mut invp = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native metis_order output missing invp[{index}]"))?
            .parse::<isize>()
            .map_err(|error| format!("native metis_order invp[{index}] parse failed: {error}"))?;
        invp.push(
            usize::try_from(entry - 1)
                .map_err(|_| format!("native metis_order invp[{index}]={entry} is invalid"))?,
        );
    }
    Ok(NativeMetisRun {
        flag,
        stat,
        perm,
        invp,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_metis_prune_output(
    dimension: usize,
    output: &[u8],
) -> Result<NativeMetisPruneRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native METIS prune output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let stat = tokens
        .next()
        .ok_or_else(|| "native METIS prune output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native METIS prune stat parse failed: {error}"))?;
    let active = tokens
        .next()
        .ok_or_else(|| "native METIS prune output missing active flag".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native METIS prune active parse failed: {error}"))?
        != 0;
    let kept_vertex_count = tokens
        .next()
        .ok_or_else(|| "native METIS prune output missing kept count".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS prune kept count parse failed: {error}"))?;
    let directed_edge_count = tokens
        .next()
        .ok_or_else(|| "native METIS prune output missing edge count".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS prune edge count parse failed: {error}"))?;
    let mut piperm = Vec::with_capacity(dimension);
    for index in 0..dimension {
        piperm.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS prune output missing piperm[{index}]"))?
                .parse::<usize>()
                .map_err(|error| format!("native METIS prune piperm parse failed: {error}"))?,
        );
    }
    let mut offsets = Vec::new();
    let mut neighbors = Vec::new();
    let mut vertex_weights = Vec::new();
    if active {
        while offsets.len() < kept_vertex_count + 1 {
            let entry = tokens
                .next()
                .ok_or_else(|| {
                    format!(
                        "native METIS prune output missing offsets[{}]",
                        offsets.len()
                    )
                })?
                .parse::<usize>()
                .map_err(|error| format!("native METIS prune offset parse failed: {error}"))?;
            offsets.push(entry);
        }
        while neighbors.len() < directed_edge_count {
            let entry = tokens
                .next()
                .ok_or_else(|| {
                    format!(
                        "native METIS prune output missing neighbors[{}]",
                        neighbors.len()
                    )
                })?
                .parse::<usize>()
                .map_err(|error| format!("native METIS prune neighbor parse failed: {error}"))?;
            neighbors.push(entry);
        }
        while vertex_weights.len() < kept_vertex_count {
            let entry = tokens
                .next()
                .ok_or_else(|| {
                    format!(
                        "native METIS prune output missing vertex_weights[{}]",
                        vertex_weights.len()
                    )
                })?
                .parse::<isize>()
                .map_err(|error| {
                    format!("native METIS prune vertex weight parse failed: {error}")
                })?;
            vertex_weights.push(entry);
        }
    }
    Ok(NativeMetisPruneRun {
        stat,
        pruning_active: active,
        kept_vertex_count,
        directed_edge_count,
        piperm,
        offsets,
        neighbors,
        vertex_weights,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_metis_cc_components_output(
    output: &[u8],
) -> Result<NativeMetisCcComponentsRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native METIS CC output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let stat = tokens
        .next()
        .ok_or_else(|| "native METIS CC output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native METIS CC stat parse failed: {error}"))?;
    let vertex_count = tokens
        .next()
        .ok_or_else(|| "native METIS CC output missing vertex count".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS CC vertex count parse failed: {error}"))?;
    let mincut = tokens
        .next()
        .ok_or_else(|| "native METIS CC output missing mincut".to_string())?
        .parse::<isize>()
        .map_err(|error| format!("native METIS CC mincut parse failed: {error}"))?;
    let part_weights = [
        tokens
            .next()
            .ok_or_else(|| "native METIS CC output missing part weight 0".to_string())?
            .parse::<isize>()
            .map_err(|error| format!("native METIS CC part weight parse failed: {error}"))?,
        tokens
            .next()
            .ok_or_else(|| "native METIS CC output missing part weight 1".to_string())?
            .parse::<isize>()
            .map_err(|error| format!("native METIS CC part weight parse failed: {error}"))?,
        tokens
            .next()
            .ok_or_else(|| "native METIS CC output missing part weight 2".to_string())?
            .parse::<isize>()
            .map_err(|error| format!("native METIS CC part weight parse failed: {error}"))?,
    ];
    let boundary_len = tokens
        .next()
        .ok_or_else(|| "native METIS CC output missing boundary length".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS CC boundary length parse failed: {error}"))?;
    let where_len = vertex_count;
    let mut where_part = Vec::with_capacity(where_len);
    for index in 0..where_len {
        where_part.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS CC output missing where[{index}]"))?
                .parse::<usize>()
                .map_err(|error| format!("native METIS CC where parse failed: {error}"))?,
        );
    }
    let mut boundary = Vec::with_capacity(boundary_len);
    for index in 0..boundary_len {
        boundary.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS CC output missing boundary[{index}]"))?
                .parse::<usize>()
                .map_err(|error| format!("native METIS CC boundary parse failed: {error}"))?,
        );
    }
    let ncmps = tokens
        .next()
        .ok_or_else(|| "native METIS CC output missing component count".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS CC component count parse failed: {error}"))?;
    let mut cptr = Vec::with_capacity(ncmps + 1);
    for index in 0..=ncmps {
        cptr.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS CC output missing cptr[{index}]"))?
                .parse::<usize>()
                .map_err(|error| format!("native METIS CC cptr parse failed: {error}"))?,
        );
    }
    let mut cind = Vec::with_capacity(*cptr.last().unwrap_or(&0));
    for index in 0..*cptr.last().unwrap_or(&0) {
        cind.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS CC output missing cind[{index}]"))?
                .parse::<usize>()
                .map_err(|error| format!("native METIS CC cind parse failed: {error}"))?,
        );
    }
    let mut subgraph_labels = Vec::with_capacity(ncmps);
    let mut subgraph_offsets = Vec::with_capacity(ncmps);
    let mut subgraph_neighbors = Vec::with_capacity(ncmps);
    for component in 0..ncmps {
        let nvtxs = tokens
            .next()
            .ok_or_else(|| format!("native METIS CC output missing subgraph {component} nvtxs"))?
            .parse::<usize>()
            .map_err(|error| format!("native METIS CC subgraph nvtxs parse failed: {error}"))?;
        let nedges = tokens
            .next()
            .ok_or_else(|| format!("native METIS CC output missing subgraph {component} nedges"))?
            .parse::<usize>()
            .map_err(|error| format!("native METIS CC subgraph nedges parse failed: {error}"))?;
        let mut labels = Vec::with_capacity(nvtxs);
        for index in 0..nvtxs {
            labels.push(
                tokens
                    .next()
                    .ok_or_else(|| {
                        format!(
                            "native METIS CC output missing subgraph {component} label[{index}]"
                        )
                    })?
                    .parse::<usize>()
                    .map_err(|error| format!("native METIS CC label parse failed: {error}"))?,
            );
        }
        let mut offsets = Vec::with_capacity(nvtxs + 1);
        for index in 0..=nvtxs {
            offsets.push(
                tokens
                    .next()
                    .ok_or_else(|| {
                        format!(
                            "native METIS CC output missing subgraph {component} offset[{index}]"
                        )
                    })?
                    .parse::<usize>()
                    .map_err(|error| format!("native METIS CC offset parse failed: {error}"))?,
            );
        }
        let mut neighbors = Vec::with_capacity(nedges);
        for index in 0..nedges {
            neighbors.push(
                tokens
                    .next()
                    .ok_or_else(|| {
                        format!(
                            "native METIS CC output missing subgraph {component} neighbor[{index}]"
                        )
                    })?
                    .parse::<usize>()
                    .map_err(|error| format!("native METIS CC neighbor parse failed: {error}"))?,
            );
        }
        subgraph_labels.push(labels);
        subgraph_offsets.push(offsets);
        subgraph_neighbors.push(neighbors);
    }

    Ok(NativeMetisCcComponentsRun {
        stat,
        separator: NativeSeparatorRun {
            stat,
            mincut,
            part_weights,
            where_part,
            boundary,
        },
        cptr,
        cind,
        subgraph_labels,
        subgraph_offsets,
        subgraph_neighbors,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_mmd_order_output(dimension: usize, output: &[u8]) -> Result<NativeMmdRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native MMD output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let stat = tokens
        .next()
        .ok_or_else(|| "native MMD output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native MMD stat parse failed: {error}"))?;
    let mut perm = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native MMD output missing perm[{index}]"))?
            .parse::<isize>()
            .map_err(|error| format!("native MMD perm[{index}] parse failed: {error}"))?;
        perm.push(
            usize::try_from(entry - 1)
                .map_err(|_| format!("native MMD perm[{index}]={entry} is invalid"))?,
        );
    }
    let mut invp = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native MMD output missing invp[{index}]"))?
            .parse::<isize>()
            .map_err(|error| format!("native MMD invp[{index}] parse failed: {error}"))?;
        invp.push(
            usize::try_from(entry - 1)
                .map_err(|_| format!("native MMD invp[{index}]={entry} is invalid"))?,
        );
    }
    Ok(NativeMmdRun { stat, perm, invp })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_separator_output(
    dimension: usize,
    output: &[u8],
) -> Result<NativeSeparatorRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native separator output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let stat = tokens
        .next()
        .ok_or_else(|| "native separator output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native separator stat parse failed: {error}"))?;
    let mincut = tokens
        .next()
        .ok_or_else(|| "native separator output missing mincut".to_string())?
        .parse::<isize>()
        .map_err(|error| format!("native separator mincut parse failed: {error}"))?;
    let mut part_weights = [0isize; 3];
    for (index, weight) in part_weights.iter_mut().enumerate() {
        *weight = tokens
            .next()
            .ok_or_else(|| format!("native separator output missing part weight {index}"))?
            .parse::<isize>()
            .map_err(|error| {
                format!("native separator part weight {index} parse failed: {error}")
            })?;
    }
    let boundary_len = tokens
        .next()
        .ok_or_else(|| "native separator output missing boundary length".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native separator boundary length parse failed: {error}"))?;
    let mut where_part = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native separator output missing where[{index}]"))?
            .parse::<usize>()
            .map_err(|error| format!("native separator where[{index}] parse failed: {error}"))?;
        where_part.push(entry);
    }
    let mut boundary = Vec::with_capacity(boundary_len);
    for index in 0..boundary_len {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native separator output missing boundary[{index}]"))?
            .parse::<usize>()
            .map_err(|error| format!("native separator boundary[{index}] parse failed: {error}"))?;
        boundary.push(entry);
    }
    Ok(NativeSeparatorRun {
        stat,
        mincut,
        part_weights,
        where_part,
        boundary,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_node_nd_top_separator_output(
    original_dimension: usize,
    output: &[u8],
) -> Result<NativeNodeNdTopSeparatorRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native NodeND top separator output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let stat = tokens
        .next()
        .ok_or_else(|| "native NodeND top separator output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native NodeND top separator stat parse failed: {error}"))?;
    let compression_active = tokens
        .next()
        .ok_or_else(|| "native NodeND top separator output missing compression flag".to_string())?
        .parse::<usize>()
        .map_err(|error| {
            format!("native NodeND top separator compression flag parse failed: {error}")
        })?
        != 0;
    let nseps = tokens
        .next()
        .ok_or_else(|| "native NodeND top separator output missing nseps".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native NodeND top separator nseps parse failed: {error}"))?;
    let compressed_vertex_count = tokens
        .next()
        .ok_or_else(|| {
            "native NodeND top separator output missing compressed vertex count".to_string()
        })?
        .parse::<usize>()
        .map_err(|error| {
            format!("native NodeND top separator compressed vertex count parse failed: {error}")
        })?;
    let compressed_edge_count = tokens
        .next()
        .ok_or_else(|| {
            "native NodeND top separator output missing compressed edge count".to_string()
        })?
        .parse::<usize>()
        .map_err(|error| {
            format!("native NodeND top separator compressed edge count parse failed: {error}")
        })?;
    let cind_len = tokens
        .next()
        .ok_or_else(|| "native NodeND top separator output missing cind length".to_string())?
        .parse::<usize>()
        .map_err(|error| {
            format!("native NodeND top separator cind length parse failed: {error}")
        })?;
    let mut compressed_cptr = Vec::with_capacity(compressed_vertex_count + 1);
    for index in 0..=compressed_vertex_count {
        compressed_cptr.push(
            tokens
                .next()
                .ok_or_else(|| format!("native NodeND top separator output missing cptr[{index}]"))?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native NodeND top separator cptr[{index}] parse failed: {error}")
                })?,
        );
    }
    let mut compressed_cind = Vec::with_capacity(cind_len);
    for index in 0..cind_len {
        compressed_cind.push(
            tokens
                .next()
                .ok_or_else(|| format!("native NodeND top separator output missing cind[{index}]"))?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native NodeND top separator cind[{index}] parse failed: {error}")
                })?,
        );
    }
    if compression_active && cind_len != original_dimension {
        return Err(format!(
            "native NodeND top separator compressed cind length {cind_len} != original dimension {original_dimension}"
        ));
    }
    let mincut = tokens
        .next()
        .ok_or_else(|| "native NodeND top separator output missing mincut".to_string())?
        .parse::<isize>()
        .map_err(|error| format!("native NodeND top separator mincut parse failed: {error}"))?;
    let mut part_weights = [0isize; 3];
    for (index, weight) in part_weights.iter_mut().enumerate() {
        *weight = tokens
            .next()
            .ok_or_else(|| {
                format!("native NodeND top separator output missing part weight {index}")
            })?
            .parse::<isize>()
            .map_err(|error| {
                format!("native NodeND top separator part weight {index} parse failed: {error}")
            })?;
    }
    let boundary_len = tokens
        .next()
        .ok_or_else(|| "native NodeND top separator output missing boundary length".to_string())?
        .parse::<usize>()
        .map_err(|error| {
            format!("native NodeND top separator boundary length parse failed: {error}")
        })?;
    let mut where_part = Vec::with_capacity(compressed_vertex_count);
    for index in 0..compressed_vertex_count {
        where_part.push(
            tokens
                .next()
                .ok_or_else(|| {
                    format!("native NodeND top separator output missing where[{index}]")
                })?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native NodeND top separator where[{index}] parse failed: {error}")
                })?,
        );
    }
    let mut boundary = Vec::with_capacity(boundary_len);
    for index in 0..boundary_len {
        boundary.push(
            tokens
                .next()
                .ok_or_else(|| {
                    format!("native NodeND top separator output missing boundary[{index}]")
                })?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native NodeND top separator boundary[{index}] parse failed: {error}")
                })?,
        );
    }
    Ok(NativeNodeNdTopSeparatorRun {
        stat,
        compression_active,
        nseps,
        compressed_vertex_count,
        compressed_edge_count,
        compressed_cptr,
        compressed_cind,
        separator: NativeSeparatorRun {
            stat,
            mincut,
            part_weights,
            where_part,
            boundary,
        },
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_metis_random_output(output: &[u8]) -> Result<NativeMetisRandomRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native METIS random output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let stat = tokens
        .next()
        .ok_or_else(|| "native METIS random output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native METIS random stat parse failed: {error}"))?;
    let sequence_len = tokens
        .next()
        .ok_or_else(|| "native METIS random output missing sequence length".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS random sequence length parse failed: {error}"))?;
    let mut sequence = Vec::with_capacity(sequence_len);
    for index in 0..sequence_len {
        sequence.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS random output missing sequence[{index}]"))?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native METIS random sequence[{index}] parse failed: {error}")
                })?,
        );
    }
    let permutation_len = tokens
        .next()
        .ok_or_else(|| "native METIS random output missing permutation length".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("native METIS random permutation length parse failed: {error}"))?;
    let mut permutation = Vec::with_capacity(permutation_len);
    for index in 0..permutation_len {
        permutation.push(
            tokens
                .next()
                .ok_or_else(|| format!("native METIS random output missing permutation[{index}]"))?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native METIS random permutation[{index}] parse failed: {error}")
                })?,
        );
    }
    Ok(NativeMetisRandomRun {
        stat,
        sequence,
        permutation,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_node_bisection_stages_output(
    original_dimension: usize,
    output: &[u8],
) -> Result<NativeNodeBisectionStagesRun, String> {
    let stdout = String::from_utf8(output.to_vec()).map_err(|error| {
        format!("native METIS node-bisection stage output was not UTF-8: {error}")
    })?;
    let mut tokens = stdout.split_whitespace();
    let stat = parse_next::<i32>(&mut tokens, "node-bisection stage stat")?;
    let coarse_vertex_count =
        parse_next::<usize>(&mut tokens, "node-bisection coarse vertex count")?;
    let coarse_edge_count = parse_next::<usize>(&mut tokens, "node-bisection coarse edge count")?;

    let mut coarse_xadj = Vec::with_capacity(coarse_vertex_count + 1);
    for index in 0..=coarse_vertex_count {
        coarse_xadj.push(parse_next(
            &mut tokens,
            &format!("node-bisection coarse xadj[{index}]"),
        )?);
    }
    let mut coarse_adjncy = Vec::with_capacity(coarse_edge_count);
    for index in 0..coarse_edge_count {
        coarse_adjncy.push(parse_next(
            &mut tokens,
            &format!("node-bisection coarse adjncy[{index}]"),
        )?);
    }
    let mut coarse_weights = Vec::with_capacity(coarse_vertex_count);
    for index in 0..coarse_vertex_count {
        coarse_weights.push(parse_next(
            &mut tokens,
            &format!("node-bisection coarse weight[{index}]"),
        )?);
    }
    let mut original_cmap = Vec::with_capacity(original_dimension);
    for index in 0..original_dimension {
        original_cmap.push(parse_next(
            &mut tokens,
            &format!("node-bisection original cmap[{index}]"),
        )?);
    }
    let mut coarse_labels = Vec::with_capacity(coarse_vertex_count);
    for index in 0..coarse_vertex_count {
        coarse_labels.push(parse_next(
            &mut tokens,
            &format!("node-bisection coarse label[{index}]"),
        )?);
    }
    let edge_mincut = parse_next(&mut tokens, "node-bisection edge mincut")?;
    let mut edge_part_weights = [0isize; 2];
    for (index, weight) in edge_part_weights.iter_mut().enumerate() {
        *weight = parse_next(
            &mut tokens,
            &format!("node-bisection edge part weight {index}"),
        )?;
    }
    let edge_boundary_len =
        parse_next::<usize>(&mut tokens, "node-bisection edge boundary length")?;
    let mut edge_where = Vec::with_capacity(coarse_vertex_count);
    for index in 0..coarse_vertex_count {
        edge_where.push(parse_next(
            &mut tokens,
            &format!("node-bisection edge where[{index}]"),
        )?);
    }
    let mut edge_boundary = Vec::with_capacity(edge_boundary_len);
    for index in 0..edge_boundary_len {
        edge_boundary.push(parse_next(
            &mut tokens,
            &format!("node-bisection edge boundary[{index}]"),
        )?);
    }
    let mut edge_id = Vec::with_capacity(coarse_vertex_count);
    for index in 0..coarse_vertex_count {
        edge_id.push(parse_next(
            &mut tokens,
            &format!("node-bisection edge id[{index}]"),
        )?);
    }
    let mut edge_ed = Vec::with_capacity(coarse_vertex_count);
    for index in 0..coarse_vertex_count {
        edge_ed.push(parse_next(
            &mut tokens,
            &format!("node-bisection edge ed[{index}]"),
        )?);
    }
    let initial = parse_native_separator_tokens(&mut tokens, coarse_vertex_count, "initial")?;
    let final_separator = parse_native_separator_tokens(&mut tokens, original_dimension, "final")?;

    Ok(NativeNodeBisectionStagesRun {
        stat,
        coarse_vertex_count,
        coarse_edge_count,
        coarse_xadj,
        coarse_adjncy,
        coarse_weights,
        original_cmap,
        coarse_labels,
        edge_mincut,
        edge_part_weights,
        edge_where,
        edge_boundary,
        edge_id,
        edge_ed,
        initial,
        final_separator,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_separator_tokens<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    dimension: usize,
    label: &str,
) -> Result<NativeSeparatorRun, String> {
    let mincut = parse_next(tokens, &format!("node-bisection {label} mincut"))?;
    let mut part_weights = [0isize; 3];
    for (index, weight) in part_weights.iter_mut().enumerate() {
        *weight = parse_next(
            tokens,
            &format!("node-bisection {label} part weight {index}"),
        )?;
    }
    let boundary_len =
        parse_next::<usize>(tokens, &format!("node-bisection {label} boundary length"))?;
    let mut where_part = Vec::with_capacity(dimension);
    for index in 0..dimension {
        where_part.push(parse_next(
            tokens,
            &format!("node-bisection {label} where[{index}]"),
        )?);
    }
    let mut boundary = Vec::with_capacity(boundary_len);
    for index in 0..boundary_len {
        boundary.push(parse_next(
            tokens,
            &format!("node-bisection {label} boundary[{index}]"),
        )?);
    }
    Ok(NativeSeparatorRun {
        stat: 0,
        mincut,
        part_weights,
        where_part,
        boundary,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_next<'a, T: std::str::FromStr>(
    tokens: &mut impl Iterator<Item = &'a str>,
    label: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    tokens
        .next()
        .ok_or_else(|| format!("native output missing {label}"))?
        .parse::<T>()
        .map_err(|error| format!("native output {label} parse failed: {error}"))
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn parse_native_match_order_output(
    dimension: usize,
    output: &[u8],
) -> Result<NativeMatchOrderRun, String> {
    let stdout = String::from_utf8(output.to_vec())
        .map_err(|error| format!("native match_order output was not UTF-8: {error}"))?;
    let mut tokens = stdout.split_whitespace();
    let flag = tokens
        .next()
        .ok_or_else(|| "native match_order output missing flag".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native match_order flag parse failed: {error}"))?;
    let stat = tokens
        .next()
        .ok_or_else(|| "native match_order output missing stat".to_string())?
        .parse::<i32>()
        .map_err(|error| format!("native match_order stat parse failed: {error}"))?;
    let mut order = Vec::with_capacity(dimension);
    let mut scale_logs = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let token = tokens
            .next()
            .ok_or_else(|| format!("native match_order output missing scale_logs[{index}] bits"))?;
        let bits = u64::from_str_radix(token, 16).map_err(|error| {
            format!("native match_order scale_logs[{index}] bits parse failed: {error}")
        })?;
        scale_logs.push(f64::from_bits(bits));
    }
    let mut matching = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native match_order output missing matching[{index}]"))?
            .parse::<isize>()
            .map_err(|error| {
                format!("native match_order matching[{index}] parse failed: {error}")
            })?;
        matching.push(if entry < 0 {
            None
        } else {
            Some(
                usize::try_from(entry - 1).map_err(|_| {
                    format!("native match_order matching[{index}]={entry} is invalid")
                })?,
            )
        });
    }
    let mut split_matching = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native match_order output missing split_matching[{index}]"))?
            .parse::<isize>()
            .map_err(|error| {
                format!("native match_order split_matching[{index}] parse failed: {error}")
            })?;
        split_matching.push(if entry > 0 { entry - 1 } else { entry });
    }
    let compressed_dimension = tokens
        .next()
        .ok_or_else(|| "native match_order output missing compressed dimension".to_string())?
        .parse::<usize>()
        .map_err(|error| {
            format!("native match_order compressed dimension parse failed: {error}")
        })?;
    let mut compressed_col_ptrs = Vec::with_capacity(compressed_dimension + 1);
    if compressed_dimension == 0 {
        compressed_col_ptrs.push(0);
    } else {
        for index in 0..=compressed_dimension {
            let entry = tokens
                .next()
                .ok_or_else(|| {
                    format!("native match_order output missing compressed ptr[{index}]")
                })?
                .parse::<usize>()
                .map_err(|error| {
                    format!("native match_order compressed ptr[{index}] parse failed: {error}")
                })?;
            compressed_col_ptrs.push(entry - 1);
        }
    }
    let compressed_nnz = compressed_col_ptrs.last().copied().unwrap_or(0);
    let mut compressed_row_indices = Vec::with_capacity(compressed_nnz);
    for index in 0..compressed_nnz {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native match_order output missing compressed row[{index}]"))?
            .parse::<usize>()
            .map_err(|error| {
                format!("native match_order compressed row[{index}] parse failed: {error}")
            })?;
        compressed_row_indices.push(entry - 1);
    }
    let compressed_metis_flag = tokens
        .next()
        .ok_or_else(|| "native match_order output missing compressed METIS flag".to_string())?
        .parse::<i32>()
        .map_err(|error| {
            format!("native match_order compressed METIS flag parse failed: {error}")
        })?;
    let compressed_metis_stat = tokens
        .next()
        .ok_or_else(|| "native match_order output missing compressed METIS stat".to_string())?
        .parse::<i32>()
        .map_err(|error| {
            format!("native match_order compressed METIS stat parse failed: {error}")
        })?;
    let mut compressed_metis_perm = Vec::with_capacity(compressed_dimension);
    for index in 0..compressed_dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| {
                format!("native match_order output missing compressed METIS perm[{index}]")
            })?
            .parse::<usize>()
            .map_err(|error| {
                format!("native match_order compressed METIS perm[{index}] parse failed: {error}")
            })?;
        compressed_metis_perm.push(entry - 1);
    }
    let mut compressed_metis_invp = Vec::with_capacity(compressed_dimension);
    for index in 0..compressed_dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| {
                format!("native match_order output missing compressed METIS invp[{index}]")
            })?
            .parse::<usize>()
            .map_err(|error| {
                format!("native match_order compressed METIS invp[{index}] parse failed: {error}")
            })?;
        compressed_metis_invp.push(entry - 1);
    }
    for index in 0..dimension {
        let entry = tokens
            .next()
            .ok_or_else(|| format!("native match_order output missing order[{index}]"))?
            .parse::<i32>()
            .map_err(|error| format!("native match_order order[{index}] parse failed: {error}"))?;
        order.push(
            usize::try_from(entry - 1)
                .map_err(|_| format!("native match_order order[{index}]={entry} is invalid"))?,
        );
    }
    let mut scaling = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let token = tokens
            .next()
            .ok_or_else(|| format!("native match_order output missing scaling[{index}] bits"))?;
        let bits = u64::from_str_radix(token, 16).map_err(|error| {
            format!("native match_order scaling[{index}] bits parse failed: {error}")
        })?;
        scaling.push(f64::from_bits(bits));
    }
    Ok(NativeMatchOrderRun {
        flag,
        stat,
        scale_logs,
        matching,
        split_matching,
        compressed_col_ptrs,
        compressed_row_indices,
        compressed_metis_flag,
        compressed_metis_stat,
        compressed_metis_perm,
        compressed_metis_invp,
        order,
        scaling,
    })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_match_order_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_match_order_shim).clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_order_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_metis_order_shim).clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_node_nd_options_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_metis_node_nd_options_shim)
        .clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_prune_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_metis_prune_shim).clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_cc_components_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_metis_cc_components_shim)
        .clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_mmd_order_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_mmd_order_shim).clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_separator_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_separator_shim).clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_node_nd_top_separator_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_node_nd_top_separator_shim)
        .clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_metis_random_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_metis_random_shim).clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_node_bisection_stage_shim() -> Result<PathBuf, String> {
    static SHIM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    SHIM.get_or_init(build_native_node_bisection_stage_shim)
        .clone()
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_match_order_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_match_order_shim.f90");
    let debug_source = shim_dir.join("spral_match_order_debug.f90");
    let exe = shim_dir.join("spral_match_order_shim");
    let source_copy = fs::read_to_string(&metadata.match_order_source).map_err(|error| {
        format!(
            "failed to read {}: {error}",
            metadata.match_order_source.display()
        )
    })?;
    let source_copy = source_copy
        .replacen("module spral_match_order", "module spral_match_order_debug", 1)
        .replace("end module spral_match_order", "end module spral_match_order_debug")
        .replace(
            "  integer, parameter :: long = selected_int_kind(18)",
            "  integer, parameter :: long = selected_int_kind(18)\n  integer, public :: debug_ncomp = 0\n  integer, dimension(:), allocatable, public :: debug_ptr3, debug_row3",
        )
        .replace(
            "public :: match_order_metis ! Find a matching-based ordering using the",
            "public :: match_order_metis ! Find a matching-based ordering using the\n  public :: mo_scale, mo_split",
        )
        .replace(
            "    allocate(invp(ncomp), stat=stat)",
            "    debug_ncomp = ncomp\n    if (allocated(debug_ptr3)) deallocate(debug_ptr3)\n    if (allocated(debug_row3)) deallocate(debug_row3)\n    allocate(debug_ptr3(ncomp+1), debug_row3(jj-1), stat=stat)\n    if (stat .ne. 0) return\n    debug_ptr3(1:ncomp+1) = ptr3(1:ncomp+1)\n    debug_row3(1:jj-1) = row3(1:jj-1)\n\n    allocate(invp(ncomp), stat=stat)",
        );
    fs::write(&debug_source, source_copy)
        .map_err(|error| format!("failed to write {}: {error}", debug_source.display()))?;
    fs::write(&source, NATIVE_MATCH_ORDER_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let mut command = Command::new(&metadata.fc);
    command
        .arg("-O0")
        .arg("-I")
        .arg(&metadata.module_dir)
        .arg("-J")
        .arg(&shim_dir)
        .arg(&debug_source)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native match_order shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native match_order shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_metis_order_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_metis_order_shim.f90");
    let exe = shim_dir.join("spral_metis_order_shim");
    fs::write(&source, NATIVE_METIS_ORDER_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let mut command = Command::new(&metadata.fc);
    command
        .arg("-O0")
        .arg("-I")
        .arg(&metadata.module_dir)
        .arg("-J")
        .arg(&shim_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native metis_order shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native metis_order shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_metis_node_nd_options_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_metis_node_nd_options_shim.c");
    let exe = shim_dir.join("spral_metis_node_nd_options_shim");
    fs::write(&source, NATIVE_METIS_NODE_ND_OPTIONS_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native METIS NodeND options shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native METIS NodeND options shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_metis_prune_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_metis_prune_shim.c");
    let exe = shim_dir.join("spral_metis_prune_shim");
    fs::write(&source, NATIVE_METIS_PRUNE_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg("-I")
        .arg(&metadata.metis_source_lib_dir)
        .arg("-I")
        .arg(&metadata.gklib_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native METIS prune shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native METIS prune shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_metis_cc_components_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_metis_cc_components_shim.c");
    let exe = shim_dir.join("spral_metis_cc_components_shim");
    fs::write(&source, NATIVE_METIS_CC_COMPONENTS_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg("-I")
        .arg(&metadata.metis_source_lib_dir)
        .arg("-I")
        .arg(&metadata.gklib_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native METIS CC components shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native METIS CC components shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_mmd_order_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_mmd_order_shim.c");
    let exe = shim_dir.join("spral_mmd_order_shim");
    fs::write(&source, NATIVE_MMD_ORDER_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native MMD shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native MMD shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_separator_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_separator_shim.c");
    let exe = shim_dir.join("spral_separator_shim");
    fs::write(&source, NATIVE_SEPARATOR_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg("-I")
        .arg(&metadata.metis_source_lib_dir)
        .arg("-I")
        .arg(&metadata.gklib_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native separator shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native separator shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_node_nd_top_separator_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_node_nd_top_separator_shim.c");
    let exe = shim_dir.join("spral_node_nd_top_separator_shim");
    fs::write(&source, NATIVE_NODE_ND_TOP_SEPARATOR_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg("-I")
        .arg(&metadata.metis_source_lib_dir)
        .arg("-I")
        .arg(&metadata.gklib_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native NodeND top separator shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native NodeND top separator shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_metis_random_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_metis_random_shim.c");
    let exe = shim_dir.join("spral_metis_random_shim");
    fs::write(&source, NATIVE_METIS_RANDOM_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg("-I")
        .arg(&metadata.metis_source_lib_dir)
        .arg("-I")
        .arg(&metadata.gklib_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command
        .output()
        .map_err(|error| format!("failed to compile native METIS random shim: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native METIS random shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn build_native_node_bisection_stage_shim() -> Result<PathBuf, String> {
    let metadata = spral_src_metadata()?;
    let workspace = workspace_root();
    let shim_dir = workspace.join("target/spral-match-order-parity-shim");
    fs::create_dir_all(&shim_dir)
        .map_err(|error| format!("failed to create {}: {error}", shim_dir.display()))?;
    let source = shim_dir.join("spral_node_bisection_stage_shim.c");
    let exe = shim_dir.join("spral_node_bisection_stage_shim");
    fs::write(&source, NATIVE_NODE_BISECTION_STAGE_SHIM_SOURCE)
        .map_err(|error| format!("failed to write {}: {error}", source.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command
        .arg("-O0")
        .arg("-g")
        .arg("-DIDXTYPEWIDTH=32")
        .arg("-DREALTYPEWIDTH=32")
        .arg("-I")
        .arg(&metadata.metis_include_dir)
        .arg("-I")
        .arg(&metadata.metis_source_lib_dir)
        .arg("-I")
        .arg(&metadata.gklib_include_dir)
        .arg(&source);
    for flag in metadata.spral_lflags.split_whitespace() {
        command.arg(flag);
    }
    command.arg("-o").arg(&exe);
    command.current_dir(&shim_dir);
    let output = command.output().map_err(|error| {
        format!("failed to compile native METIS node-bisection stage shim: {error}")
    })?;
    if !output.status.success() {
        return Err(format!(
            "native METIS node-bisection stage shim compile failed with status {:?}\ncommand: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn spral_src_metadata() -> Result<SpralSrcMetadata, String> {
    let build_root = workspace_root().join("target/debug/build");
    let mut candidates = Vec::new();
    for entry in fs::read_dir(&build_root)
        .map_err(|error| format!("failed to read {}: {error}", build_root.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read build entry: {error}"))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("spral-src-") {
            continue;
        }
        let output_path = path.join("output");
        if !output_path.exists() {
            continue;
        }
        let output = fs::read_to_string(&output_path)
            .map_err(|error| format!("failed to read {}: {error}", output_path.display()))?;
        if !output.contains("cargo:OPENBLAS_THREADING=serial") {
            continue;
        }
        let Some(spral_lflags) = metadata_line(&output, "SPRAL_LFLAGS") else {
            continue;
        };
        let fc = metadata_line(&output, "FC").unwrap_or_else(|| "gfortran".to_string());
        let runtime_link_dirs = metadata_line(&output, "RUNTIME_LINK_DIRS")
            .map(|value| {
                value
                    .split(';')
                    .filter(|entry| !entry.is_empty())
                    .map(PathBuf::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let module_dir = path.join("out/build/spral/libspral.a.p");
        if !module_dir.join("spral_match_order.mod").exists() {
            continue;
        }
        let match_order_source = path.join("out/sources/spral-2025.09.18/src/match_order.f90");
        if !match_order_source.exists() {
            continue;
        }
        let metis_include_dir = path.join("out/sources/METIS-5.2.1/include");
        if !metis_include_dir.join("metis.h").exists() {
            continue;
        }
        let metis_source_lib_dir = path.join("out/sources/METIS-5.2.1/libmetis");
        if !metis_source_lib_dir.join("metislib.h").exists() {
            continue;
        }
        let gklib_include_dir = find_gklib_include_dir(&path.join("out/sources"))?;
        let modified = fs::metadata(&output_path)
            .and_then(|metadata| metadata.modified())
            .ok();
        candidates.push((
            modified,
            SpralSrcMetadata {
                fc,
                module_dir,
                match_order_source,
                metis_include_dir,
                metis_source_lib_dir,
                gklib_include_dir,
                spral_lflags,
                runtime_link_dirs,
            },
        ));
    }
    candidates.sort_by_key(|(modified, _)| *modified);
    candidates
        .pop()
        .map(|(_, metadata)| metadata)
        .ok_or_else(|| {
            format!(
                "could not find source-built SPRAL metadata under {}",
                build_root.display()
            )
        })
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn find_gklib_include_dir(sources_dir: &Path) -> Result<PathBuf, String> {
    for entry in fs::read_dir(sources_dir)
        .map_err(|error| format!("failed to read {}: {error}", sources_dir.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read source entry: {error}"))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("GKlib-") {
            continue;
        }
        let include_dir = path.join("include");
        if include_dir.join("GKlib.h").exists() {
            return Ok(include_dir);
        }
    }
    Err(format!(
        "could not find GKlib include directory under {}",
        sources_dir.display()
    ))
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn metadata_line(output: &str, key: &str) -> Option<String> {
    output
        .lines()
        .find_map(|line| line.strip_prefix(&format!("cargo:{key}=")))
        .map(str::to_string)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn native_match_order_runtime_env() -> Result<Vec<(&'static str, String)>, String> {
    let metadata = spral_src_metadata()?;
    let mut paths = metadata
        .runtime_link_dirs
        .into_iter()
        .filter_map(|path| path.into_os_string().into_string().ok())
        .collect::<Vec<_>>();
    if let Ok(existing) = std::env::var("DYLD_LIBRARY_PATH")
        && !existing.is_empty()
    {
        paths.push(existing);
    }
    Ok(vec![("DYLD_LIBRARY_PATH", paths.join(":"))])
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ssids-rs lives under workspace root")
        .to_path_buf()
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

fn run_rust_spral_matching(
    matrix: SymmetricCscMatrix<'_>,
    dense_matrix: &[Vec<f64>],
    rhs: &[f64],
) -> (Duration, RustRun) {
    let analyse_started = Instant::now();
    let (symbolic, _) = analyse(matrix, &SsidsOptions::spral_default()).expect("rust analyse");
    let analyse_time = analyse_started.elapsed();
    let factor_started = Instant::now();
    let (mut factor, _) = factorize(matrix, &symbolic, &NumericFactorOptions::spral_default())
        .expect("rust factorize");
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

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn lower_csc_pattern_from_edges(
    dimension: usize,
    edges: &[(usize, usize)],
) -> (Vec<usize>, Vec<usize>) {
    let mut cols = vec![vec![]; dimension];
    for (col, col_rows) in cols.iter_mut().enumerate() {
        col_rows.push(col);
    }
    for &(lhs, rhs) in edges {
        assert_ne!(lhs, rhs);
        let col = lhs.min(rhs);
        let row = lhs.max(rhs);
        cols[col].push(row);
    }
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col_rows in &mut cols {
        col_rows.sort_unstable();
        col_rows.dedup();
        row_indices.extend(col_rows.iter().copied());
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn twin_path_edges(groups: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(groups + 4 * groups.saturating_sub(1));
    for group in 0..groups {
        let base = 2 * group;
        edges.push((base, base + 1));
        if group + 1 < groups {
            let next = base + 2;
            edges.push((base, next));
            edges.push((base, next + 1));
            edges.push((base + 1, next));
            edges.push((base + 1, next + 1));
        }
    }
    edges
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn reconstruct_small_node_nd_from_separator(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    where_part: &[usize],
    boundary: &[usize],
) -> Vec<usize> {
    let mut order = vec![usize::MAX; dimension];
    let mut lastvtx = dimension;
    for &separator in boundary {
        lastvtx -= 1;
        order[separator] = lastvtx;
    }

    let left = (0..dimension)
        .filter(|&vertex| where_part[vertex] == 0)
        .collect::<Vec<_>>();
    let right = (0..dimension)
        .filter(|&vertex| where_part[vertex] == 1)
        .collect::<Vec<_>>();

    assign_mmd_leaf_order(
        &mut order,
        col_ptrs,
        row_indices,
        &left,
        lastvtx - right.len(),
    );
    assign_mmd_leaf_order(&mut order, col_ptrs, row_indices, &right, lastvtx);

    assert!(
        order.iter().all(|&entry| entry != usize::MAX),
        "reconstructed NodeND order must assign every vertex"
    );
    order
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn assign_mmd_leaf_order(
    order: &mut [usize],
    col_ptrs: &[usize],
    row_indices: &[usize],
    labels: &[usize],
    lastvtx: usize,
) {
    if labels.is_empty() {
        return;
    }
    let (leaf_col_ptrs, leaf_row_indices) =
        induced_lower_csc_pattern(col_ptrs, row_indices, labels);
    let permutation = metis_ordering::metis_mmd_order_from_lower_csc(
        labels.len(),
        &leaf_col_ptrs,
        &leaf_row_indices,
    )
    .expect("Rust MMD leaf ordering")
    .permutation;
    let firstvtx = lastvtx - labels.len();
    for (local, &global) in labels.iter().enumerate() {
        order[global] = firstvtx + permutation.inverse()[local];
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn induced_lower_csc_pattern(
    col_ptrs: &[usize],
    row_indices: &[usize],
    labels: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut local = vec![usize::MAX; col_ptrs.len() - 1];
    for (index, &vertex) in labels.iter().enumerate() {
        local[vertex] = index;
    }
    let mut edges = Vec::new();
    for &col in labels {
        let local_col = local[col];
        for &row in &row_indices[col_ptrs[col]..col_ptrs[col + 1]] {
            if row == col {
                continue;
            }
            let local_row = local[row];
            if local_row == usize::MAX {
                continue;
            }
            edges.push((local_col.min(local_row), local_col.max(local_row)));
        }
    }
    lower_csc_pattern_from_edges(labels.len(), &edges)
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS NodeND fixture oracle for staged source porting"]
fn native_metis_node_nd_fixture_phase_tests() {
    let mut fixtures = Vec::new();

    let complete_dimension = 69;
    let mut complete_col_ptrs = Vec::with_capacity(complete_dimension + 1);
    let mut complete_row_indices = Vec::new();
    complete_col_ptrs.push(0);
    for col in 0..complete_dimension {
        complete_row_indices.extend(col..complete_dimension);
        complete_col_ptrs.push(complete_row_indices.len());
    }
    fixtures.push((
        "complete_69_compresses_to_single_component",
        complete_dimension,
        complete_col_ptrs,
        complete_row_indices,
        true,
    ));

    let (empty_col_ptrs, empty_row_indices) = lower_csc_pattern_from_edges(3, &[]);
    fixtures.push(("empty_3", 3, empty_col_ptrs, empty_row_indices, true));

    let path_edges = (0..5)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(6, &path_edges);
    fixtures.push((
        "path_6_no_compression_candidate",
        6,
        path_col_ptrs,
        path_row_indices,
        true,
    ));

    let star_edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(7, &star_edges);
    fixtures.push((
        "star_7_partial_compression_candidate",
        7,
        star_col_ptrs,
        star_row_indices,
        true,
    ));

    let path_54_edges = (0..53)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_54_col_ptrs, path_54_row_indices) = lower_csc_pattern_from_edges(54, &path_54_edges);
    fixtures.push((
        "path_54_match_rm_projection",
        54,
        path_54_col_ptrs,
        path_54_row_indices,
        true,
    ));

    let path_121_edges = (0..120)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_121_col_ptrs, path_121_row_indices) =
        lower_csc_pattern_from_edges(121, &path_121_edges);
    fixtures.push((
        "path_121_single_level_projection",
        121,
        path_121_col_ptrs,
        path_121_row_indices,
        true,
    ));

    let path_300_edges = (0..299)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_300_col_ptrs, path_300_row_indices) =
        lower_csc_pattern_from_edges(300, &path_300_edges);
    fixtures.push((
        "path_300_recursive_projection",
        300,
        path_300_col_ptrs,
        path_300_row_indices,
        true,
    ));

    let path_1000_edges = (0..999)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_1000_col_ptrs, path_1000_row_indices) =
        lower_csc_pattern_from_edges(1000, &path_1000_edges);
    fixtures.push((
        "path_1000_multilevel_recursive_projection",
        1000,
        path_1000_col_ptrs,
        path_1000_row_indices,
        true,
    ));

    let twin_path_2400_edges = twin_path_edges(1200);
    let (twin_path_2400_col_ptrs, twin_path_2400_row_indices) =
        lower_csc_pattern_from_edges(2400, &twin_path_2400_edges);
    fixtures.push((
        "twin_path_2400_compression_nseps_retry",
        2400,
        twin_path_2400_col_ptrs,
        twin_path_2400_row_indices,
        true,
    ));

    let path_5000_edges = (0..4999)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_5000_col_ptrs, path_5000_row_indices) =
        lower_csc_pattern_from_edges(5000, &path_5000_edges);
    fixtures.push((
        "path_5000_l2_projection",
        5000,
        path_5000_col_ptrs,
        path_5000_row_indices,
        true,
    ));

    for (name, dimension, col_ptrs, row_indices, must_match) in fixtures {
        let native = run_native_metis_order_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| panic!("native metis fixture {name} failed: {error}"));
        assert_eq!(native.flag, 0, "native metis fixture {name} flag");
        assert_eq!(native.stat, 0, "native metis fixture {name} stat");
        let rust =
            metis_ordering::metis_node_nd_order_from_lower_csc(dimension, &col_ptrs, &row_indices)
                .unwrap_or_else(|error| panic!("Rust metis fixture {name} failed: {error}"))
                .permutation;
        let rust_perm = rust.inverse();
        let rust_invp = rust.perm();
        let perm_eq = native.perm == rust_perm;
        let invp_eq = native.invp == rust_invp;
        eprintln!(
            "metis_fixture name={name} dim={dimension} nnz={} native_perm_hash=0x{:016x} native_perm_prefix={:?} rust_perm_hash=0x{:016x} rust_perm_prefix={:?} native_invp_hash=0x{:016x} rust_invp_hash=0x{:016x} perm_eq={} invp_eq={}",
            row_indices.len(),
            hash_usize(&native.perm),
            &native.perm[..native.perm.len().min(16)],
            hash_usize(rust_perm),
            &rust_perm[..rust_perm.len().min(16)],
            hash_usize(&native.invp),
            hash_usize(rust_invp),
            perm_eq,
            invp_eq
        );
        if must_match {
            assert!(perm_eq, "native/Rust METIS perm mismatch for {name}");
            assert!(invp_eq, "native/Rust METIS invp mismatch for {name}");
        }
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS NodeND non-default option oracle"]
fn native_metis_node_nd_non_default_option_phase_tests() {
    let mut complete_col_ptrs = Vec::with_capacity(70);
    let mut complete_row_indices = Vec::new();
    complete_col_ptrs.push(0);
    for col in 0..69 {
        complete_row_indices.extend(col..69);
        complete_col_ptrs.push(complete_row_indices.len());
    }

    let star_edges = (1..10).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(10, &star_edges);

    let path_edges = (0..5)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(6, &path_edges);

    let twin_edges = twin_path_edges(128);
    let (twin_col_ptrs, twin_row_indices) = lower_csc_pattern_from_edges(256, &twin_edges);

    let fixtures = vec![
        (
            "complete_69_default_regression",
            69,
            complete_col_ptrs.clone(),
            complete_row_indices.clone(),
            metis_ordering::MetisNodeNdOptions::spral_default(),
        ),
        (
            "complete_69_forced_no_compression",
            69,
            complete_col_ptrs,
            complete_row_indices,
            metis_ordering::MetisNodeNdOptions {
                compress: false,
                ..metis_ordering::MetisNodeNdOptions::spral_default()
            },
        ),
        (
            "star_10_ccorder_components",
            10,
            star_col_ptrs.clone(),
            star_row_indices.clone(),
            metis_ordering::MetisNodeNdOptions {
                compress: false,
                ccorder: true,
                pfactor: 0,
            },
        ),
        (
            "star_10_pfactor_no_prune",
            10,
            star_col_ptrs.clone(),
            star_row_indices.clone(),
            metis_ordering::MetisNodeNdOptions {
                pfactor: 100,
                ..metis_ordering::MetisNodeNdOptions::spral_default()
            },
        ),
        (
            "star_10_pfactor_partial_prune_disables_compression",
            10,
            star_col_ptrs.clone(),
            star_row_indices.clone(),
            metis_ordering::MetisNodeNdOptions {
                pfactor: 20,
                ..metis_ordering::MetisNodeNdOptions::spral_default()
            },
        ),
        (
            "path_6_pfactor_all_pruned_ignored",
            6,
            path_col_ptrs,
            path_row_indices,
            metis_ordering::MetisNodeNdOptions {
                pfactor: 1,
                ..metis_ordering::MetisNodeNdOptions::spral_default()
            },
        ),
        (
            "star_10_ccorder_with_pruning",
            10,
            star_col_ptrs,
            star_row_indices,
            metis_ordering::MetisNodeNdOptions {
                compress: true,
                ccorder: true,
                pfactor: 20,
            },
        ),
        (
            "twin_path_256_ccorder_with_compression",
            256,
            twin_col_ptrs,
            twin_row_indices,
            metis_ordering::MetisNodeNdOptions {
                ccorder: true,
                ..metis_ordering::MetisNodeNdOptions::spral_default()
            },
        ),
    ];

    for (name, dimension, col_ptrs, row_indices, options) in fixtures {
        let native =
            run_native_metis_node_nd_options_lower_csc(dimension, &col_ptrs, &row_indices, options)
                .unwrap_or_else(|error| {
                    panic!("native METIS NodeND option fixture {name} failed: {error}")
                });
        assert_eq!(native.flag, 1, "native METIS status for {name}");
        assert_eq!(native.stat, 0, "native METIS stat for {name}");
        let rust = metis_ordering::metis_node_nd_order_from_lower_csc_with_options(
            dimension,
            &col_ptrs,
            &row_indices,
            options,
        )
        .unwrap_or_else(|error| panic!("Rust METIS NodeND option fixture {name} failed: {error}"))
        .permutation;
        let rust_perm = rust.inverse();
        let rust_invp = rust.perm();
        eprintln!(
            "metis_option_fixture name={name} dim={dimension} options={options:?} native_perm_hash=0x{:016x} rust_perm_hash=0x{:016x} native_invp_hash=0x{:016x} rust_invp_hash=0x{:016x}",
            hash_usize(&native.perm),
            hash_usize(rust_perm),
            hash_usize(&native.invp),
            hash_usize(rust_invp),
        );
        assert_eq!(
            native.perm, rust_perm,
            "native/Rust METIS perm mismatch for {name}"
        );
        assert_eq!(
            native.invp, rust_invp,
            "native/Rust METIS invp mismatch for {name}"
        );
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS PruneGraph oracle"]
fn native_metis_prune_phase_tests() {
    let star_edges = (1..10).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(10, &star_edges);
    for (name, pfactor) in [
        ("star_10_no_prune", 100usize),
        ("star_10_partial_prune", 20usize),
    ] {
        let native =
            run_native_metis_prune_lower_csc(10, &star_col_ptrs, &star_row_indices, pfactor)
                .unwrap_or_else(|error| {
                    panic!("native METIS prune fixture {name} failed: {error}")
                });
        let rust = metis_ordering::metis_debug_prune_from_lower_csc(
            10,
            &star_col_ptrs,
            &star_row_indices,
            pfactor,
        )
        .unwrap_or_else(|error| panic!("Rust METIS prune fixture {name} failed: {error}"));
        eprintln!(
            "metis_prune name={name} pfactor={pfactor} native_active={} rust_active={} native_kept={} rust_kept={} native_edges={} rust_edges={}",
            native.pruning_active,
            rust.pruning_active,
            native.kept_vertex_count,
            rust.kept_vertex_count,
            native.directed_edge_count,
            rust.neighbors.len()
        );
        assert_eq!(native.stat, 0);
        assert_eq!(native.pruning_active, rust.pruning_active);
        assert_eq!(native.kept_vertex_count, rust.kept_vertex_count);
        if native.pruning_active {
            assert_eq!(native.directed_edge_count, rust.neighbors.len());
            assert_eq!(native.piperm, rust.piperm);
            assert_eq!(native.offsets, rust.offsets);
            assert_eq!(native.neighbors, rust.neighbors);
            assert_eq!(native.vertex_weights, rust.vertex_weights);
        }
    }

    let path_edges = (0..5)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(6, &path_edges);
    let native = run_native_metis_prune_lower_csc(6, &path_col_ptrs, &path_row_indices, 1)
        .unwrap_or_else(|error| panic!("native METIS all-pruned fixture failed: {error}"));
    let rust =
        metis_ordering::metis_debug_prune_from_lower_csc(6, &path_col_ptrs, &path_row_indices, 1)
            .unwrap_or_else(|error| panic!("Rust METIS all-pruned fixture failed: {error}"));
    assert_eq!(native.stat, 0);
    assert!(!native.pruning_active);
    assert!(!rust.pruning_active);
    assert_eq!(native.kept_vertex_count, 6);
    assert_eq!(rust.kept_vertex_count, 6);
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS CC component split oracle"]
fn native_metis_cc_component_phase_tests() {
    let star_edges = (1..10).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(10, &star_edges);
    let options = metis_ordering::MetisNodeNdOptions {
        compress: false,
        ccorder: true,
        pfactor: 0,
    };
    let native =
        run_native_metis_cc_components_lower_csc(10, &star_col_ptrs, &star_row_indices, options)
            .unwrap_or_else(|error| panic!("native METIS CC component fixture failed: {error}"));
    let rust = metis_ordering::metis_debug_cc_components_from_lower_csc_with_options(
        10,
        &star_col_ptrs,
        &star_row_indices,
        options,
    )
    .unwrap_or_else(|error| panic!("Rust METIS CC component fixture failed: {error}"));

    eprintln!(
        "metis_cc_components native_cptr={:?} rust_cptr={:?} native_cind={:?} rust_cind={:?}",
        native.cptr, rust.cptr, native.cind, rust.cind
    );
    assert_eq!(native.stat, 0);
    assert_eq!(native.separator.mincut, rust.separator.mincut);
    assert_eq!(native.separator.part_weights, rust.separator.part_weights);
    assert_eq!(native.separator.where_part, rust.separator.where_part);
    assert_eq!(native.separator.boundary, rust.separator.boundary);
    assert_eq!(native.cptr, rust.cptr);
    assert_eq!(native.cind, rust.cind);
    assert_eq!(native.subgraph_labels, rust.subgraph_labels);
    assert_eq!(native.subgraph_offsets, rust.subgraph_offsets);
    assert_eq!(native.subgraph_neighbors, rust.subgraph_neighbors);
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS top separator oracle for staged NodeND source porting"]
fn native_metis_node_nd_top_separator_compression_retry_phase_test() {
    let twin_edges = twin_path_edges(1200);
    let (twin_col_ptrs, twin_row_indices) = lower_csc_pattern_from_edges(2400, &twin_edges);
    let path_edges = (0..4999)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(5000, &path_edges);
    let (empty_col_ptrs, empty_row_indices) = lower_csc_pattern_from_edges(3, &[]);

    let fixtures = [
        ("empty_3", 3, empty_col_ptrs, empty_row_indices),
        (
            "twin_path_2400_compression_nseps_retry",
            2400,
            twin_col_ptrs,
            twin_row_indices,
        ),
        (
            "path_5000_l2_projection",
            5000,
            path_col_ptrs,
            path_row_indices,
        ),
    ];

    for (name, dimension, col_ptrs, row_indices) in fixtures {
        let native = run_native_node_nd_top_separator_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| {
                panic!("native NodeND top separator fixture {name} failed: {error}")
            });
        assert_eq!(native.stat, 0);
        let rust = metis_ordering::metis_debug_node_nd_top_separator_from_lower_csc(
            dimension,
            &col_ptrs,
            &row_indices,
        )
        .unwrap_or_else(|error| panic!("Rust NodeND top separator fixture {name} failed: {error}"));

        let mut rust_cptr = Vec::with_capacity(rust.compressed_original_vertices.len() + 1);
        let mut rust_cind = Vec::new();
        rust_cptr.push(0);
        for component in &rust.compressed_original_vertices {
            rust_cind.extend(component.iter().copied());
            rust_cptr.push(rust_cind.len());
        }

        eprintln!(
            "node_nd_top_separator name={name} native compressed={} nseps={} cnvtxs={} nedges={} cptr_hash=0x{:016x} cind_hash=0x{:016x} mincut={} pwgts={:?} where_hash=0x{:016x} boundary_hash=0x{:016x}",
            native.compression_active,
            native.nseps,
            native.compressed_vertex_count,
            native.compressed_edge_count,
            hash_usize(&native.compressed_cptr),
            hash_usize(&native.compressed_cind),
            native.separator.mincut,
            native.separator.part_weights,
            hash_usize(&native.separator.where_part),
            hash_usize(&native.separator.boundary)
        );
        eprintln!(
            "node_nd_top_separator name={name} rust compressed={} nseps={} cnvtxs={} nedges={} cptr_hash=0x{:016x} cind_hash=0x{:016x} mincut={} pwgts={:?} where_hash=0x{:016x} boundary_hash=0x{:016x}",
            rust.compression_active,
            rust.nseps,
            rust.compressed_vertex_count,
            rust.compressed_directed_edge_count,
            hash_usize(&rust_cptr),
            hash_usize(&rust_cind),
            rust.separator.mincut,
            rust.separator.part_weights,
            hash_usize(&rust.separator.where_part),
            hash_usize(&rust.separator.boundary)
        );

        assert_eq!(rust.compression_active, native.compression_active);
        assert_eq!(rust.nseps, native.nseps);
        assert_eq!(rust.compressed_vertex_count, native.compressed_vertex_count);
        assert_eq!(
            rust.compressed_directed_edge_count,
            native.compressed_edge_count
        );
        assert_eq!(rust_cptr, native.compressed_cptr);
        if native.compression_active {
            assert_eq!(rust_cind, native.compressed_cind);
        }
        assert_eq!(rust.separator.mincut, native.separator.mincut);
        assert_eq!(rust.separator.part_weights, native.separator.part_weights);
        assert_eq!(rust.separator.where_part, native.separator.where_part);
        assert_eq!(rust.separator.boundary, native.separator.boundary);
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS random oracle for staged NodeND source porting"]
fn native_metis_random_fixture_phase_tests() {
    let sequence = run_native_metis_random(-1, 12, 0, 0, true)
        .unwrap_or_else(|error| panic!("native METIS random sequence failed: {error}"));
    assert_eq!(sequence.stat, 0, "native METIS random sequence stat");
    let rust_sequence = metis_ordering::metis_debug_irand_sequence(-1, 12);
    eprintln!(
        "metis_random_sequence seed=-1 native={:?} rust={:?}",
        sequence.sequence, rust_sequence
    );
    assert_eq!(
        sequence.sequence, rust_sequence,
        "native/Rust METIS random sequence mismatch"
    );

    let fixtures = [
        ("small_n_lt_10_flag_init", -1, 6usize, 6usize, true),
        ("wide_n_ge_10_flag_init", -1, 12, 4, true),
        ("wide_n_ge_10_existing_array", 123, 10, 5, false),
    ];
    for (name, seed, n, nshuffles, flag) in fixtures {
        let native = run_native_metis_random(seed, 0, n, nshuffles, flag)
            .unwrap_or_else(|error| panic!("native METIS random fixture {name} failed: {error}"));
        assert_eq!(native.stat, 0, "native METIS random fixture {name} stat");
        let rust = metis_ordering::metis_debug_irand_array_permute(seed, n, nshuffles, flag);
        eprintln!(
            "metis_random_permute name={name} seed={seed} n={n} nshuffles={nshuffles} flag={flag} native={:?} rust={:?}",
            native.permutation, rust
        );
        assert_eq!(
            native.permutation, rust,
            "native/Rust METIS irandArrayPermute mismatch for {name}"
        );
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS node-bisection stage oracle for staged NodeND source porting"]
fn native_metis_node_bisection_stage_phase_tests() {
    let path_edges = (0..5)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(6, &path_edges);

    let star_edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(7, &star_edges);

    let path_54_edges = (0..53)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_54_col_ptrs, path_54_row_indices) = lower_csc_pattern_from_edges(54, &path_54_edges);

    let fixtures = [
        ("path_6", 6, path_col_ptrs, path_row_indices),
        ("star_7", 7, star_col_ptrs, star_row_indices),
        (
            "path_54_match_rm",
            54,
            path_54_col_ptrs,
            path_54_row_indices,
        ),
    ];
    for (name, dimension, col_ptrs, row_indices) in fixtures {
        let native = run_native_node_bisection_stages_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| {
                panic!("native node-bisection stage fixture {name} failed: {error}")
            });
        eprintln!(
            "node_bisection_stage name={name} coarse_n={} coarse_edges={} coarse_xadj={:?} coarse_adjncy={:?} coarse_weights={:?} original_cmap={:?} coarse_labels={:?} edge_mincut={} edge_part_weights={:?} edge_where={:?} edge_boundary={:?} edge_id={:?} edge_ed={:?} initial=({:?}) final=({:?})",
            native.coarse_vertex_count,
            native.coarse_edge_count,
            native.coarse_xadj,
            native.coarse_adjncy,
            native.coarse_weights,
            native.original_cmap,
            native.coarse_labels,
            native.edge_mincut,
            native.edge_part_weights,
            native.edge_where,
            native.edge_boundary,
            native.edge_id,
            native.edge_ed,
            native.initial,
            native.final_separator
        );
        assert_eq!(
            native.stat, 0,
            "native node-bisection stage fixture {name} stat"
        );
        let (
            expected_coarse_xadj,
            expected_coarse_adjncy,
            expected_coarse_weights,
            expected_original_cmap,
            expected_coarse_labels,
            expected_edge_mincut,
            expected_edge_part_weights,
            expected_edge_where,
            expected_edge_boundary,
            expected_edge_id,
            expected_edge_ed,
            expected_mincut,
            expected_part_weights,
            expected_where,
            expected_boundary,
        ) = match name {
            "path_6" => (
                vec![0, 1, 3, 5, 7, 9, 10],
                vec![1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
                vec![1, 1, 1, 1, 1, 1],
                vec![0, 1, 2, 3, 4, 5],
                vec![0, 1, 2, 3, 4, 5],
                1isize,
                [3isize, 3],
                vec![1usize, 1, 1, 0, 0, 0],
                vec![2usize, 3],
                vec![1isize, 2, 1, 1, 2, 1],
                vec![0isize, 0, 1, 1, 0, 0],
                1isize,
                [3isize, 2, 1],
                vec![1usize, 1, 2, 0, 0, 0],
                vec![2usize],
            ),
            "star_7" => (
                vec![0, 6, 7, 8, 9, 10, 11, 12],
                vec![1, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0],
                vec![1, 1, 1, 1, 1, 1, 1],
                vec![0, 1, 2, 3, 4, 5, 6],
                vec![0, 1, 2, 3, 4, 5, 6],
                3isize,
                [3isize, 4],
                vec![1usize, 0, 1, 0, 1, 1, 0],
                vec![0usize, 1, 3, 6],
                vec![3isize, 0, 1, 0, 1, 1, 0],
                vec![3isize, 1, 0, 1, 0, 0, 1],
                1isize,
                [3isize, 3, 1],
                vec![2usize, 0, 1, 0, 1, 1, 0],
                vec![0usize],
            ),
            "path_54_match_rm" => (
                vec![
                    0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
                    41, 43, 45, 47, 49, 51, 53, 55, 57, 58,
                ],
                vec![
                    1, 2, 0, 3, 1, 4, 2, 5, 3, 4, 6, 7, 5, 6, 8, 9, 7, 10, 8, 11, 9, 12, 10, 13,
                    11, 14, 12, 13, 15, 16, 14, 17, 15, 18, 16, 17, 19, 20, 18, 21, 19, 22, 20, 21,
                    23, 24, 22, 25, 23, 26, 24, 27, 25, 28, 26, 29, 27, 28,
                ],
                vec![
                    2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2,
                    2, 2, 2, 1,
                ],
                vec![
                    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
                    13, 13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24,
                    24, 25, 25, 26, 26, 27, 27, 28, 28, 29,
                ],
                (0..30).collect::<Vec<_>>(),
                1isize,
                [27isize, 27],
                vec![
                    1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                ],
                vec![14usize, 15],
                vec![
                    1isize, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 1,
                ],
                vec![
                    0isize, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                ],
                1isize,
                [27isize, 26, 1],
                vec![
                    1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                ],
                vec![14usize],
            ),
            _ => unreachable!("unexpected fixture {name}"),
        };
        assert_eq!(
            native.coarse_vertex_count,
            expected_coarse_weights.len(),
            "node-bisection coarse vertex count mismatch for {name}"
        );
        assert_eq!(
            native.coarse_edge_count,
            expected_coarse_adjncy.len(),
            "node-bisection coarse edge count mismatch for {name}"
        );
        assert_eq!(
            native.coarse_xadj, expected_coarse_xadj,
            "node-bisection coarse xadj mismatch for {name}"
        );
        assert_eq!(
            native.coarse_adjncy, expected_coarse_adjncy,
            "node-bisection coarse adjncy mismatch for {name}"
        );
        assert_eq!(
            native.coarse_weights, expected_coarse_weights,
            "node-bisection coarse weights mismatch for {name}"
        );
        assert_eq!(
            native.original_cmap, expected_original_cmap,
            "node-bisection original cmap mismatch for {name}"
        );
        assert_eq!(
            native.coarse_labels, expected_coarse_labels,
            "node-bisection coarse labels mismatch for {name}"
        );
        let rust_coarse = metis_ordering::metis_debug_l1_coarsen_from_lower_csc(
            dimension,
            &col_ptrs,
            &row_indices,
        )
        .unwrap_or_else(|error| panic!("Rust L1 coarsen fixture {name} failed: {error}"));
        assert_eq!(
            rust_coarse.vertex_count, native.coarse_vertex_count,
            "native/Rust L1 coarse vertex count mismatch for {name}"
        );
        assert_eq!(
            rust_coarse.directed_edge_count, native.coarse_edge_count,
            "native/Rust L1 coarse edge count mismatch for {name}"
        );
        assert_eq!(
            rust_coarse.offsets, native.coarse_xadj,
            "native/Rust L1 coarse xadj mismatch for {name}"
        );
        assert_eq!(
            rust_coarse.neighbors, native.coarse_adjncy,
            "native/Rust L1 coarse adjncy mismatch for {name}"
        );
        assert_eq!(
            rust_coarse.vertex_weights,
            native
                .coarse_weights
                .iter()
                .copied()
                .map(|weight| weight as isize)
                .collect::<Vec<_>>(),
            "native/Rust L1 coarse weights mismatch for {name}"
        );
        assert_eq!(
            native.edge_mincut, expected_edge_mincut,
            "node-bisection edge mincut mismatch for {name}"
        );
        assert_eq!(
            native.edge_part_weights, expected_edge_part_weights,
            "node-bisection edge part weights mismatch for {name}"
        );
        assert_eq!(
            native.edge_where, expected_edge_where,
            "node-bisection edge where mismatch for {name}"
        );
        assert_eq!(
            native.edge_boundary, expected_edge_boundary,
            "node-bisection edge boundary mismatch for {name}"
        );
        assert_eq!(
            native.edge_id, expected_edge_id,
            "node-bisection edge id mismatch for {name}"
        );
        assert_eq!(
            native.edge_ed, expected_edge_ed,
            "node-bisection edge ed mismatch for {name}"
        );
        let rust_edge = metis_ordering::metis_debug_l1_edge_bisection_from_lower_csc(
            dimension,
            &col_ptrs,
            &row_indices,
        )
        .unwrap_or_else(|error| panic!("Rust L1 edge bisection fixture {name} failed: {error}"));
        assert_eq!(
            rust_edge.mincut, native.edge_mincut,
            "native/Rust L1 edge bisection mincut mismatch for {name}"
        );
        assert_eq!(
            rust_edge.part_weights, native.edge_part_weights,
            "native/Rust L1 edge bisection part weights mismatch for {name}"
        );
        assert_eq!(
            rust_edge.where_part, native.edge_where,
            "native/Rust L1 edge bisection where mismatch for {name}"
        );
        assert_eq!(
            rust_edge.boundary, native.edge_boundary,
            "native/Rust L1 edge bisection boundary mismatch for {name}"
        );
        assert_eq!(
            rust_edge.internal_degree, native.edge_id,
            "native/Rust L1 edge bisection id mismatch for {name}"
        );
        assert_eq!(
            rust_edge.external_degree, native.edge_ed,
            "native/Rust L1 edge bisection ed mismatch for {name}"
        );
        assert_eq!(
            native.initial.mincut, expected_mincut,
            "node-bisection initial mincut mismatch for {name}"
        );
        assert_eq!(
            native.initial.part_weights, expected_part_weights,
            "node-bisection initial part weights mismatch for {name}"
        );
        assert_eq!(
            native.initial.where_part, expected_where,
            "node-bisection initial where mismatch for {name}"
        );
        assert_eq!(
            native.initial.boundary, expected_boundary,
            "node-bisection initial boundary mismatch for {name}"
        );
        let rust_separator = metis_ordering::metis_debug_l1_construct_separator_from_lower_csc(
            dimension,
            &col_ptrs,
            &row_indices,
        )
        .unwrap_or_else(|error| {
            panic!("Rust L1 construct separator fixture {name} failed: {error}")
        });
        assert_eq!(
            rust_separator.mincut, native.initial.mincut,
            "native/Rust ConstructSeparator mincut mismatch for {name}"
        );
        assert_eq!(
            rust_separator.part_weights, native.initial.part_weights,
            "native/Rust ConstructSeparator part weights mismatch for {name}"
        );
        assert_eq!(
            rust_separator.where_part, native.initial.where_part,
            "native/Rust ConstructSeparator where mismatch for {name}"
        );
        assert_eq!(
            rust_separator.boundary, native.initial.boundary,
            "native/Rust ConstructSeparator boundary mismatch for {name}"
        );
        let separator = run_native_separator_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| panic!("native separator fixture {name} failed: {error}"));
        assert_eq!(
            native.final_separator.mincut, separator.mincut,
            "node-bisection stage final mincut must match full separator shim for {name}"
        );
        assert_eq!(
            native.final_separator.part_weights, separator.part_weights,
            "node-bisection stage final part weights must match full separator shim for {name}"
        );
        assert_eq!(
            native.final_separator.where_part, separator.where_part,
            "node-bisection stage final where must match full separator shim for {name}"
        );
        assert_eq!(
            native.final_separator.boundary, separator.boundary,
            "node-bisection stage final boundary must match full separator shim for {name}"
        );
        let rust_projected = metis_ordering::metis_debug_l1_projected_separator_from_lower_csc(
            dimension,
            &col_ptrs,
            &row_indices,
        )
        .unwrap_or_else(|error| {
            panic!("Rust L1 projected separator fixture {name} failed: {error}")
        });
        assert_eq!(
            rust_projected.mincut, native.final_separator.mincut,
            "native/Rust projected separator mincut mismatch for {name}"
        );
        assert_eq!(
            rust_projected.part_weights, native.final_separator.part_weights,
            "native/Rust projected separator part weights mismatch for {name}"
        );
        assert_eq!(
            rust_projected.where_part, native.final_separator.where_part,
            "native/Rust projected separator where mismatch for {name}"
        );
        assert_eq!(
            rust_projected.boundary, native.final_separator.boundary,
            "native/Rust projected separator boundary mismatch for {name}"
        );
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native genmmd fixture oracle for staged METIS source porting"]
fn native_metis_mmd_fixture_phase_tests() {
    let mut fixtures = Vec::new();

    let path_edges = (0..5)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(6, &path_edges);
    fixtures.push(("path_6", 6, path_col_ptrs, path_row_indices));

    let star_edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(7, &star_edges);
    fixtures.push(("star_7", 7, star_col_ptrs, star_row_indices));

    let disconnected_edges = [(0, 1), (1, 2), (3, 4)];
    let (disconnected_col_ptrs, disconnected_row_indices) =
        lower_csc_pattern_from_edges(6, &disconnected_edges);
    fixtures.push((
        "disconnected_6",
        6,
        disconnected_col_ptrs,
        disconnected_row_indices,
    ));

    let (empty_col_ptrs, empty_row_indices) = lower_csc_pattern_from_edges(5, &[]);
    fixtures.push(("empty_5", 5, empty_col_ptrs, empty_row_indices));

    for (name, dimension, col_ptrs, row_indices) in fixtures {
        let native = run_native_mmd_order_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| panic!("native MMD fixture {name} failed: {error}"));
        assert_eq!(native.stat, 0, "native MMD fixture {name} stat");
        let rust =
            metis_ordering::metis_mmd_order_from_lower_csc(dimension, &col_ptrs, &row_indices)
                .unwrap_or_else(|error| panic!("Rust MMD fixture {name} failed: {error}"))
                .permutation;
        let rust_perm = rust.perm();
        let rust_invp = rust.inverse();
        eprintln!(
            "mmd_fixture name={name} dim={dimension} nnz={} native_perm={:?} rust_perm={:?} native_invp={:?} rust_invp={:?}",
            row_indices.len(),
            native.perm,
            rust_perm,
            native.invp,
            rust_invp
        );
        assert_eq!(
            native.perm, rust_perm,
            "native/Rust MMD perm mismatch for {name}"
        );
        assert_eq!(
            native.invp, rust_invp,
            "native/Rust MMD invp mismatch for {name}"
        );
    }
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
#[test]
#[ignore = "manual native METIS separator fixture oracle for staged NodeND source porting"]
fn native_metis_separator_fixture_phase_tests() {
    let path_edges = (0..5)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_col_ptrs, path_row_indices) = lower_csc_pattern_from_edges(6, &path_edges);

    let star_edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
    let (star_col_ptrs, star_row_indices) = lower_csc_pattern_from_edges(7, &star_edges);

    let path_54_edges = (0..53)
        .map(|vertex| (vertex, vertex + 1))
        .collect::<Vec<_>>();
    let (path_54_col_ptrs, path_54_row_indices) = lower_csc_pattern_from_edges(54, &path_54_edges);

    let fixtures = [
        (
            "path_6",
            6,
            path_col_ptrs,
            path_row_indices,
            1isize,
            [3isize, 2, 1],
            vec![1usize, 1, 2, 0, 0, 0],
            vec![2usize],
        ),
        (
            "star_7",
            7,
            star_col_ptrs,
            star_row_indices,
            1isize,
            [3isize, 3, 1],
            vec![2usize, 0, 1, 0, 1, 1, 0],
            vec![0usize],
        ),
        (
            "path_54_match_rm_projection",
            54,
            path_54_col_ptrs,
            path_54_row_indices,
            1isize,
            [27isize, 26, 1],
            vec![
                1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![26usize],
        ),
    ];

    for (
        name,
        dimension,
        col_ptrs,
        row_indices,
        expected_mincut,
        expected_part_weights,
        expected_where,
        expected_boundary,
    ) in fixtures
    {
        let native = run_native_separator_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| panic!("native separator fixture {name} failed: {error}"));
        eprintln!(
            "separator_fixture name={name} dim={dimension} nnz={} mincut={} part_weights={:?} where={:?} boundary={:?}",
            row_indices.len(),
            native.mincut,
            native.part_weights,
            native.where_part,
            native.boundary
        );
        assert_eq!(native.stat, 0, "native separator fixture {name} stat");
        assert_eq!(
            native.mincut, expected_mincut,
            "native separator fixture {name} mincut"
        );
        assert_eq!(
            native.part_weights, expected_part_weights,
            "native separator fixture {name} part weights"
        );
        assert_eq!(
            native.where_part, expected_where,
            "native separator fixture {name} where labels"
        );
        assert_eq!(
            native.boundary, expected_boundary,
            "native separator fixture {name} boundary"
        );
        let rust_separator = metis_ordering::metis_debug_l1_projected_separator_from_lower_csc(
            dimension,
            &col_ptrs,
            &row_indices,
        )
        .unwrap_or_else(|error| panic!("Rust projected separator fixture {name} failed: {error}"));
        assert_eq!(
            rust_separator.mincut, native.mincut,
            "native/Rust projected separator fixture {name} mincut"
        );
        assert_eq!(
            rust_separator.part_weights, native.part_weights,
            "native/Rust projected separator fixture {name} part weights"
        );
        assert_eq!(
            rust_separator.where_part, native.where_part,
            "native/Rust projected separator fixture {name} where labels"
        );
        assert_eq!(
            rust_separator.boundary, native.boundary,
            "native/Rust projected separator fixture {name} boundary"
        );
        let native_node_nd = run_native_metis_order_lower_csc(dimension, &col_ptrs, &row_indices)
            .unwrap_or_else(|error| panic!("native NodeND fixture {name} failed: {error}"));
        let reconstructed = reconstruct_small_node_nd_from_separator(
            dimension,
            &col_ptrs,
            &row_indices,
            &native.where_part,
            &native.boundary,
        );
        assert_eq!(
            native_node_nd.perm, reconstructed,
            "native separator plus Rust MMD leaves must reconstruct NodeND order for {name}"
        );
        assert_eq!(native.where_part.len(), dimension);
        assert_eq!(
            native.boundary.len() as isize,
            native.part_weights[2],
            "native separator fixture {name} separator weight"
        );
    }
}

fn print_rust_matching_trace(trace: &SpralMatchingTrace) {
    let (expanded_ptr_hash, expanded_row_hash, expanded_value_hash) =
        csc_hashes(&trace.expanded_full);
    let (compact_ptr_hash, compact_row_hash, compact_value_hash) = csc_hashes(&trace.compact_abs);
    let (compressed_ptr_hash, compressed_row_hash, compressed_value_hash) =
        csc_hashes(&trace.compressed_lower);
    let (scaling_min, scaling_max) = scaling_range(&trace.scaling);
    eprintln!(
        "rust_phase expanded_full dim={} nnz={} ptr_hash=0x{:016x} row_hash=0x{:016x} value_hash=0x{:016x} ptr_prefix={:?} row_prefix={:?}",
        trace.expanded_full.dimension,
        trace.expanded_full.row_indices.len(),
        expanded_ptr_hash,
        expanded_row_hash,
        expanded_value_hash,
        &trace.expanded_full.col_ptrs[..trace.expanded_full.col_ptrs.len().min(10)],
        &trace.expanded_full.row_indices[..trace.expanded_full.row_indices.len().min(16)]
    );
    eprintln!(
        "rust_phase compact_abs nnz={} ptr_hash=0x{:016x} row_hash=0x{:016x} value_hash=0x{:016x} ptr_prefix={:?} row_prefix={:?}",
        trace.compact_abs.row_indices.len(),
        compact_ptr_hash,
        compact_row_hash,
        compact_value_hash,
        &trace.compact_abs.col_ptrs[..trace.compact_abs.col_ptrs.len().min(10)],
        &trace.compact_abs.row_indices[..trace.compact_abs.row_indices.len().min(16)]
    );
    eprintln!(
        "rust_phase mo_match rank={} matching_hash=0x{:016x} matching_prefix={:?} scale_log_hash=0x{:016x}",
        trace
            .matching
            .iter()
            .filter(|entry| entry.is_some())
            .count(),
        hash_option_usize(&trace.matching),
        &trace.matching[..trace.matching.len().min(16)],
        hash_f64_bits(&trace.scale_logs)
    );
    eprintln!(
        "rust_phase mo_split cperm_hash=0x{:016x} cperm_prefix={:?}",
        hash_isize(&trace.split_matching),
        &trace.split_matching[..trace.split_matching.len().min(16)]
    );
    eprintln!(
        "rust_phase compressed_lower dim={} nnz={} ptr_hash=0x{:016x} row_hash=0x{:016x} value_hash=0x{:016x} ptr_prefix={:?} row_prefix={:?}",
        trace.compressed_lower.dimension,
        trace.compressed_lower.row_indices.len(),
        compressed_ptr_hash,
        compressed_row_hash,
        compressed_value_hash,
        &trace.compressed_lower.col_ptrs[..trace.compressed_lower.col_ptrs.len().min(10)],
        &trace.compressed_lower.row_indices[..trace.compressed_lower.row_indices.len().min(16)]
    );
    eprintln!(
        "rust_phase compressed_metis component_position_hash=0x{:016x} component_position_prefix={:?} position_component_hash=0x{:016x} position_component_prefix={:?}",
        hash_usize(&trace.compressed_component_position),
        &trace.compressed_component_position[..trace.compressed_component_position.len().min(16)],
        hash_usize(&trace.compressed_position_component),
        &trace.compressed_position_component[..trace.compressed_position_component.len().min(16)]
    );
    eprintln!(
        "rust_phase final_order hash=0x{:016x} prefix={:?}",
        hash_usize(&trace.final_order),
        &trace.final_order[..trace.final_order.len().min(16)]
    );
    eprintln!(
        "rust_phase saved_scaling hash=0x{:016x} min={:.3e} max={:.3e} prefix={:?}",
        hash_f64_bits(&trace.scaling),
        scaling_min,
        scaling_max,
        &trace.scaling[..trace.scaling.len().min(8)]
    );
}

fn assert_native_singular_match_order_fixture(
    name: &str,
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    compare_ordering: bool,
) {
    let matrix = SymmetricCscMatrix::new(dimension, col_ptrs, row_indices, Some(values))
        .unwrap_or_else(|error| panic!("valid singular fixture {name}: {error}"));
    let rust = spral_matching_trace(matrix)
        .unwrap_or_else(|error| panic!("Rust singular matching trace {name}: {error}"));
    let Some(native) = run_native_match_order_metis(&rust) else {
        eprintln!("native match_order oracle unavailable for singular fixture {name}");
        return;
    };

    eprintln!(
        "singular_match_order name={name} flag={} stat={} rank={} native_scale_log_hash=0x{:016x} rust_scale_log_hash=0x{:016x} native_scaling={:?} rust_scaling={:?} native_matching={:?} rust_matching={:?}",
        native.flag,
        native.stat,
        native
            .matching
            .iter()
            .filter(|entry| entry.is_some())
            .count(),
        hash_f64_bits(&native.scale_logs),
        hash_f64_bits(&rust.scale_logs),
        native.scaling,
        rust.scaling,
        native.matching,
        rust.matching
    );

    assert_eq!(native.flag, 1, "native singular warning flag for {name}");
    assert_eq!(native.stat, 0, "native singular stat for {name}");
    assert_eq!(
        rust.matching, native.matching,
        "matching differs for {name}"
    );
    assert_eq!(
        bit_patterns(&rust.scale_logs),
        bit_patterns(&native.scale_logs),
        "scale logs differ for {name}"
    );
    assert_eq!(
        rust.split_matching, native.split_matching,
        "split matching differs for {name}"
    );
    assert_eq!(
        rust.compressed_lower.col_ptrs, native.compressed_col_ptrs,
        "compressed column pointers differ for {name}"
    );
    assert_eq!(
        rust.compressed_lower.row_indices, native.compressed_row_indices,
        "compressed rows differ for {name}"
    );
    if compare_ordering {
        assert_eq!(
            rust.compressed_component_position, native.compressed_metis_perm,
            "compressed METIS permutation differs for {name}"
        );
        assert_eq!(
            rust.compressed_position_component, native.compressed_metis_invp,
            "compressed METIS inverse permutation differs for {name}"
        );
        assert_eq!(
            rust.final_order, native.order,
            "final order differs for {name}"
        );
    }
    assert_eq!(
        bit_patterns(&rust.scaling),
        bit_patterns(&native.scaling),
        "saved scaling differs for {name}"
    );
}

#[test]
#[ignore = "manual native singular mo_match oracle for staged source porting"]
fn native_singular_mo_match_phase_tests() {
    assert_native_singular_match_order_fixture(
        "isolated_missing_column",
        3,
        &[0, 1, 2, 2],
        &[0, 1],
        &[4.0, 9.0],
        true,
    );
    assert_native_singular_match_order_fixture(
        "path3_no_diagonal",
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[2.0, 3.0],
        true,
    );
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
    let rust_trace = spral_matching_trace(matrix).expect("rust matching trace");
    let native_match_order = run_native_match_order_metis(&rust_trace);

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
    let (rust_matching_analyse_time, rust_matching) =
        run_rust_spral_matching(matrix, &dense_matrix, &rhs);

    let (matching_vs_user_bits, matching_vs_user_first) =
        bit_mismatch_summary(&native_matching.solution, &native_user.solution);
    let (rust_vs_native_bits, rust_vs_native_first) =
        bit_mismatch_summary(&rust_user.solution, &native_user.solution);
    let (rust_matching_vs_native_bits, rust_matching_vs_native_first) =
        bit_mismatch_summary(&rust_matching.solution, &native_matching.solution);

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
    print_rust_matching_trace(&rust_trace);
    if let Some(native_match_order) = &native_match_order {
        let (scaling_min, scaling_max) = scaling_range(&native_match_order.scaling);
        let scaling_delta = solution_delta_inf(&rust_trace.scaling, &native_match_order.scaling);
        let (scaling_bits, scaling_first) =
            bit_mismatch_summary(&rust_trace.scaling, &native_match_order.scaling);
        let scale_log_delta =
            solution_delta_inf(&rust_trace.scale_logs, &native_match_order.scale_logs);
        let (scale_log_bits, scale_log_first) =
            bit_mismatch_summary(&rust_trace.scale_logs, &native_match_order.scale_logs);
        let matching_eq_native = rust_trace.matching == native_match_order.matching;
        let split_matching_eq_native =
            rust_trace.split_matching == native_match_order.split_matching;
        let compressed_ptr_eq_native =
            rust_trace.compressed_lower.col_ptrs == native_match_order.compressed_col_ptrs;
        let compressed_row_eq_native =
            rust_trace.compressed_lower.row_indices == native_match_order.compressed_row_indices;
        let compressed_graph_eq_native = compressed_ptr_eq_native && compressed_row_eq_native;
        let compressed_metis_perm_eq_native =
            rust_trace.compressed_component_position == native_match_order.compressed_metis_perm;
        let compressed_metis_invp_eq_native =
            rust_trace.compressed_position_component == native_match_order.compressed_metis_invp;
        let native_direct_order_eq_analyse = native_match_order.order == captured_order;
        let rust_order_eq_native_direct = rust_trace.final_order == native_match_order.order;
        let rust_scaling_eq_native_direct =
            bit_patterns(&rust_trace.scaling) == bit_patterns(&native_match_order.scaling);
        eprintln!(
            "native_phase match_order_metis_direct flag={} stat={} order_hash=0x{:016x} order_prefix={:?} scaling_hash=0x{:016x} scaling_min={:.3e} scaling_max={:.3e} scaling_prefix={:?}",
            native_match_order.flag,
            native_match_order.stat,
            hash_usize(&native_match_order.order),
            &native_match_order.order[..native_match_order.order.len().min(16)],
            hash_f64_bits(&native_match_order.scaling),
            scaling_min,
            scaling_max,
            &native_match_order.scaling[..native_match_order.scaling.len().min(8)]
        );
        eprintln!(
            "native_phase mo_scale rank={} matching_hash=0x{:016x} matching_prefix={:?} scale_log_hash=0x{:016x} scale_log_delta={:.3e} scale_log_bit_mismatches={} first={:?}",
            native_match_order
                .matching
                .iter()
                .filter(|entry| entry.is_some())
                .count(),
            hash_option_usize(&native_match_order.matching),
            &native_match_order.matching[..native_match_order.matching.len().min(16)],
            hash_f64_bits(&native_match_order.scale_logs),
            scale_log_delta,
            scale_log_bits,
            scale_log_first
        );
        eprintln!(
            "native_phase mo_split cperm_hash=0x{:016x} cperm_prefix={:?}",
            hash_isize(&native_match_order.split_matching),
            &native_match_order.split_matching[..native_match_order.split_matching.len().min(16)]
        );
        eprintln!(
            "native_phase compressed_lower dim={} nnz={} ptr_hash=0x{:016x} row_hash=0x{:016x} ptr_prefix={:?} row_prefix={:?}",
            native_match_order
                .compressed_col_ptrs
                .len()
                .saturating_sub(1),
            native_match_order.compressed_row_indices.len(),
            hash_usize(&native_match_order.compressed_col_ptrs),
            hash_usize(&native_match_order.compressed_row_indices),
            &native_match_order.compressed_col_ptrs
                [..native_match_order.compressed_col_ptrs.len().min(10)],
            &native_match_order.compressed_row_indices
                [..native_match_order.compressed_row_indices.len().min(16)]
        );
        eprintln!(
            "native_phase compressed_metis flag={} stat={} perm_hash=0x{:016x} perm_prefix={:?} invp_hash=0x{:016x} invp_prefix={:?}",
            native_match_order.compressed_metis_flag,
            native_match_order.compressed_metis_stat,
            hash_usize(&native_match_order.compressed_metis_perm),
            &native_match_order.compressed_metis_perm
                [..native_match_order.compressed_metis_perm.len().min(16)],
            hash_usize(&native_match_order.compressed_metis_invp),
            &native_match_order.compressed_metis_invp
                [..native_match_order.compressed_metis_invp.len().min(16)]
        );
        eprintln!(
            "native_phase direct_order_eq_analyse={} rust_matching_eq_native={} rust_split_matching_eq_native={} rust_compressed_graph_eq_native={} rust_compressed_metis_perm_eq_native={} rust_compressed_metis_invp_eq_native={} rust_order_eq_native_direct={} rust_scaling_bits_eq_native_direct={} rust_scaling_delta={:.3e} rust_scaling_bit_mismatches={} first={:?}",
            native_direct_order_eq_analyse,
            matching_eq_native,
            split_matching_eq_native,
            compressed_graph_eq_native,
            compressed_metis_perm_eq_native,
            compressed_metis_invp_eq_native,
            rust_order_eq_native_direct,
            rust_scaling_eq_native_direct,
            scaling_delta,
            scaling_bits,
            scaling_first
        );
        let first_divergence = if native_match_order.flag < 0 {
            "native match_order_metis direct call failed"
        } else if !native_direct_order_eq_analyse {
            "native analyse expansion/order path differs from direct match_order_metis input; inspect expansion/ABI before Rust phases"
        } else if !matching_eq_native {
            "Rust mo_match matching differs from native SPRAL mo_scale"
        } else if !rust_scaling_eq_native_direct {
            "Rust mo_scale scaling bits differ from native SPRAL after matching agrees"
        } else if !split_matching_eq_native {
            "Rust mo_split cycle splitting differs after matching/scaling agrees"
        } else if !compressed_graph_eq_native {
            "Rust compressed graph construction differs after split matching agrees"
        } else if !compressed_metis_perm_eq_native || !compressed_metis_invp_eq_native {
            "Rust METIS NodeND compressed permutation differs after compressed graph agrees"
        } else if !rust_order_eq_native_direct {
            "Rust expands the matching/METIS compressed permutation differently after compressed METIS agrees"
        } else {
            "Rust order and saved scaling match native match_order_metis; factor/solve parity is checked below"
        };
        eprintln!("first_observable_divergence={first_divergence}");
        assert_eq!(native_match_order.flag, 0, "native match_order flag");
        assert_eq!(native_match_order.stat, 0, "native match_order stat");
        assert!(
            native_direct_order_eq_analyse,
            "native direct match_order order must match native analyse order"
        );
        assert!(matching_eq_native, "Rust mo_match differs from native");
        assert!(
            rust_scaling_eq_native_direct,
            "Rust saved scaling differs from native"
        );
        assert!(
            split_matching_eq_native,
            "Rust mo_split cperm differs from native"
        );
        assert!(
            compressed_graph_eq_native,
            "Rust compressed graph differs from native"
        );
        assert!(
            compressed_metis_perm_eq_native && compressed_metis_invp_eq_native,
            "Rust compressed METIS permutation differs from native"
        );
        assert!(
            rust_order_eq_native_direct,
            "Rust final matching order differs from native"
        );
    }
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
        "rust_spral_matching_saved_scaling analyse={:?} factor={:?} solve={:?} residual={:.3e} inertia={:?} two_by_two={} delayed={}",
        rust_matching_analyse_time,
        rust_matching.factor_time,
        rust_matching.solve_time,
        rust_matching.residual_inf,
        rust_matching.inertia,
        rust_matching.two_by_two_pivots,
        rust_matching.delayed_pivots
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
    eprintln!(
        "rust_spral_matching_vs_native_matching_scaling delta={:.3e} bit_mismatches={} first_mismatch={:?}",
        solution_delta_inf(&rust_matching.solution, &native_matching.solution),
        rust_matching_vs_native_bits,
        rust_matching_vs_native_first
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
    assert_eq!(
        rust_matching.inertia, native_matching.factor_info.inertia,
        "SPRAL matching/scaling inertia mismatch"
    );
    assert_eq!(
        rust_matching.two_by_two_pivots, native_matching.factor_info.two_by_two_pivots,
        "SPRAL matching/scaling two-by-two pivot mismatch"
    );
    assert_eq!(
        rust_matching.delayed_pivots, native_matching.factor_info.delayed_pivots,
        "SPRAL matching/scaling delayed pivot mismatch"
    );
    assert_eq!(
        bit_patterns(&rust_matching.solution),
        bit_patterns(&native_matching.solution),
        "SPRAL matching/scaling solve bits diverged"
    );
}
