use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod native;

use metis_ordering::{
    CsrGraph, NestedDissectionOptions, OrderingError, Permutation,
    approximate_minimum_degree_order, nested_dissection_order,
};
use rayon::prelude::*;
use thiserror::Error;

pub use native::{
    NativeOrdering, NativeSpral, NativeSpralAnalyseInfo, NativeSpralError, NativeSpralFactorInfo,
    NativeSpralIndefEnquiry, NativeSpralSession,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveProfile {
    pub input_permutation_time: Duration,
    pub forward_substitution_time: Duration,
    pub diagonal_solve_time: Duration,
    pub backward_substitution_time: Duration,
    pub backward_trailing_update_time: Duration,
    pub backward_triangular_solve_time: Duration,
    pub backward_trailing_update_columns: usize,
    pub backward_trailing_update_dense_entries: usize,
    pub backward_triangular_columns: usize,
    pub backward_triangular_dense_entries: usize,
    pub output_permutation_time: Duration,
}

impl SolveProfile {
    pub fn total_recorded_time(&self) -> Duration {
        self.input_permutation_time
            + self.forward_substitution_time
            + self.diagonal_solve_time
            + self.backward_substitution_time
            + self.output_permutation_time
    }

    pub fn accumulate(&mut self, other: &Self) {
        self.input_permutation_time += other.input_permutation_time;
        self.forward_substitution_time += other.forward_substitution_time;
        self.diagonal_solve_time += other.diagonal_solve_time;
        self.backward_substitution_time += other.backward_substitution_time;
        self.backward_trailing_update_time += other.backward_trailing_update_time;
        self.backward_triangular_solve_time += other.backward_triangular_solve_time;
        self.backward_trailing_update_columns += other.backward_trailing_update_columns;
        self.backward_trailing_update_dense_entries += other.backward_trailing_update_dense_entries;
        self.backward_triangular_columns += other.backward_triangular_columns;
        self.backward_triangular_dense_entries += other.backward_triangular_dense_entries;
        self.output_permutation_time += other.output_permutation_time;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SymmetricCscMatrix<'a> {
    dimension: usize,
    col_ptrs: &'a [usize],
    row_indices: &'a [usize],
    values: Option<&'a [f64]>,
}

impl<'a> SymmetricCscMatrix<'a> {
    /// Construct a symmetric lower-triangular CSC view.
    ///
    /// The structure must contain diagonal entries and only rows `>= col`
    /// within each column.
    pub fn new(
        dimension: usize,
        col_ptrs: &'a [usize],
        row_indices: &'a [usize],
        values: Option<&'a [f64]>,
    ) -> Result<Self, SsidsError> {
        if col_ptrs.len() != dimension + 1 {
            return Err(SsidsError::InvalidMatrix(format!(
                "expected {} column pointers, got {}",
                dimension + 1,
                col_ptrs.len()
            )));
        }
        if col_ptrs.first().copied().unwrap_or_default() != 0 {
            return Err(SsidsError::InvalidMatrix(
                "column pointers must start at zero".into(),
            ));
        }
        if col_ptrs.last().copied().unwrap_or_default() != row_indices.len() {
            return Err(SsidsError::InvalidMatrix(
                "final column pointer must equal row index length".into(),
            ));
        }
        if col_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(SsidsError::InvalidMatrix(
                "column pointers must be monotone".into(),
            ));
        }
        if let Some(values) = values
            && values.len() != row_indices.len()
        {
            return Err(SsidsError::InvalidMatrix(format!(
                "value length mismatch: expected {}, got {}",
                row_indices.len(),
                values.len()
            )));
        }
        for col in 0..dimension {
            let mut previous = None;
            for &row in &row_indices[col_ptrs[col]..col_ptrs[col + 1]] {
                if row >= dimension {
                    return Err(SsidsError::InvalidMatrix(format!(
                        "row index {row} out of bounds for {dimension}x{dimension} matrix"
                    )));
                }
                if row < col {
                    return Err(SsidsError::InvalidMatrix(
                        "symmetric CSC input must store only diagonal and lower-triangular entries"
                            .into(),
                    ));
                }
                if let Some(prev_row) = previous
                    && prev_row >= row
                {
                    return Err(SsidsError::InvalidMatrix(
                        "row indices within each column must be strictly increasing".into(),
                    ));
                }
                previous = Some(row);
            }
        }
        Ok(Self {
            dimension,
            col_ptrs,
            row_indices,
            values,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn col_ptrs(&self) -> &'a [usize] {
        self.col_ptrs
    }

    pub fn row_indices(&self) -> &'a [usize] {
        self.row_indices
    }

    pub fn values(&self) -> Option<&'a [f64]> {
        self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderingStrategy {
    Natural,
    ApproximateMinimumDegree,
    NestedDissection(NestedDissectionOptions),
    /// Choose the ordering with the lowest estimated symbolic fill among
    /// natural, AMD, and Rust nested dissection.
    Auto(NestedDissectionOptions),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsidsOptions {
    pub ordering: OrderingStrategy,
}

impl Default for SsidsOptions {
    fn default() -> Self {
        Self {
            ordering: OrderingStrategy::Auto(NestedDissectionOptions::default()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotMethod {
    AggressiveAposteriori,
    BlockAposteriori,
    ThresholdPartial,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericFactorOptions {
    pub action_on_zero_pivot: bool,
    pub pivot_method: PivotMethod,
    pub small_pivot_tolerance: f64,
    pub threshold_pivot_u: f64,
    pub inertia_zero_tol: f64,
}

impl Default for NumericFactorOptions {
    fn default() -> Self {
        Self {
            action_on_zero_pivot: true,
            pivot_method: PivotMethod::BlockAposteriori,
            small_pivot_tolerance: 1e-20,
            threshold_pivot_u: 1e-8,
            inertia_zero_tol: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PivotStats {
    pub two_by_two_pivots: usize,
    pub delayed_pivots: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FactorInfo {
    pub factorization_residual_max_abs: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactorizationProgressSnapshot {
    pub total_fronts: usize,
    pub completed_fronts: usize,
    pub total_pivots: usize,
    pub completed_pivots: usize,
    pub total_weight: u64,
    pub completed_weight: u64,
    pub total_roots: usize,
    pub completed_roots: usize,
    pub total_root_delayed_blocks: usize,
    pub completed_root_delayed_blocks: usize,
    pub current_root_delayed_block: usize,
    pub current_root_delayed_block_size: usize,
    pub root_delayed_stage: RootDelayedBlockStage,
}

impl FactorizationProgressSnapshot {
    pub fn weighted_percent(self) -> f64 {
        if self.total_weight == 0 {
            100.0
        } else {
            100.0 * self.completed_weight as f64 / self.total_weight as f64
        }
    }

    pub fn pivot_percent(self) -> f64 {
        if self.total_pivots == 0 {
            100.0
        } else {
            100.0 * self.completed_pivots as f64 / self.total_pivots as f64
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootDelayedBlockStage {
    Idle,
    Packing,
    Factoring,
    Emitting,
}

impl RootDelayedBlockStage {
    fn from_usize(value: usize) -> Self {
        match value {
            1 => Self::Packing,
            2 => Self::Factoring,
            3 => Self::Emitting,
            _ => Self::Idle,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Packing => "packing",
            Self::Factoring => "factoring",
            Self::Emitting => "emitting",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalyseInfo {
    pub estimated_fill_nnz: usize,
    pub supernode_count: usize,
    pub max_supernode_width: usize,
    pub ordering_kind: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Supernode {
    pub start_column: usize,
    pub end_column: usize,
    pub trailing_rows: Vec<usize>,
}

impl Supernode {
    pub fn width(&self) -> usize {
        self.end_column - self.start_column
    }
}

const RELAXED_NODE_AMALGAMATION_NEMIN: usize = 32;
const APP_INNER_BLOCK_SIZE: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicFactor {
    pub permutation: Permutation,
    pub elimination_tree: Vec<Option<usize>>,
    pub column_counts: Vec<usize>,
    pub column_pattern: Vec<Vec<usize>>,
    pub supernodes: Vec<Supernode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiagonalBlock {
    size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DenseUpdateBounds {
    size: usize,
    update_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SymbolicFront {
    columns: Vec<usize>,
    interface_rows: Vec<usize>,
    parent: Option<usize>,
    children: Vec<usize>,
}

impl SymbolicFront {
    fn width(&self) -> usize {
        self.columns.len()
    }

    fn first_column(&self) -> usize {
        self.columns.first().copied().unwrap_or(usize::MAX)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SymbolicFrontTree {
    fronts: Vec<SymbolicFront>,
    roots: Vec<usize>,
}

#[derive(Debug)]
struct FactorizationProgressShared {
    total_fronts: usize,
    total_pivots: usize,
    total_weight: u64,
    total_roots: usize,
    total_root_delayed_blocks: AtomicUsize,
    completed_fronts: AtomicUsize,
    completed_pivots: AtomicUsize,
    completed_weight: AtomicU64,
    completed_roots: AtomicUsize,
    completed_root_delayed_blocks: AtomicUsize,
    current_root_delayed_block: AtomicUsize,
    current_root_delayed_block_size: AtomicUsize,
    root_delayed_stage: AtomicUsize,
}

impl FactorizationProgressShared {
    fn snapshot(&self) -> FactorizationProgressSnapshot {
        FactorizationProgressSnapshot {
            total_fronts: self.total_fronts,
            completed_fronts: self.completed_fronts.load(Ordering::Relaxed),
            total_pivots: self.total_pivots,
            completed_pivots: self.completed_pivots.load(Ordering::Relaxed),
            total_weight: self.total_weight,
            completed_weight: self.completed_weight.load(Ordering::Relaxed),
            total_roots: self.total_roots,
            completed_roots: self.completed_roots.load(Ordering::Relaxed),
            total_root_delayed_blocks: self.total_root_delayed_blocks.load(Ordering::Relaxed),
            completed_root_delayed_blocks: self
                .completed_root_delayed_blocks
                .load(Ordering::Relaxed),
            current_root_delayed_block: self.current_root_delayed_block.load(Ordering::Relaxed),
            current_root_delayed_block_size: self
                .current_root_delayed_block_size
                .load(Ordering::Relaxed),
            root_delayed_stage: RootDelayedBlockStage::from_usize(
                self.root_delayed_stage.load(Ordering::Relaxed),
            ),
        }
    }

    fn begin_root_delayed_block(&self, block_index: usize, size: usize) {
        self.current_root_delayed_block
            .store(block_index, Ordering::Relaxed);
        self.current_root_delayed_block_size
            .store(size, Ordering::Relaxed);
        self.root_delayed_stage
            .store(RootDelayedBlockStage::Packing as usize, Ordering::Relaxed);
    }

    fn set_root_delayed_stage(&self, stage: RootDelayedBlockStage) {
        self.root_delayed_stage
            .store(stage as usize, Ordering::Relaxed);
    }

    fn finish_root_delayed_block(&self) {
        self.completed_root_delayed_blocks
            .fetch_add(1, Ordering::Relaxed);
        self.current_root_delayed_block.store(0, Ordering::Relaxed);
        self.current_root_delayed_block_size
            .store(0, Ordering::Relaxed);
        self.root_delayed_stage
            .store(RootDelayedBlockStage::Idle as usize, Ordering::Relaxed);
    }
}

static FACTORIZATION_PROGRESS: OnceLock<Mutex<Option<Arc<FactorizationProgressShared>>>> =
    OnceLock::new();

fn factorization_progress_slot() -> &'static Mutex<Option<Arc<FactorizationProgressShared>>> {
    FACTORIZATION_PROGRESS.get_or_init(|| Mutex::new(None))
}

struct FactorizationProgressGuard;

impl Drop for FactorizationProgressGuard {
    fn drop(&mut self) {
        *factorization_progress_slot()
            .lock()
            .expect("factorization progress slot poisoned") = None;
    }
}

fn install_factorization_progress(
    progress: Arc<FactorizationProgressShared>,
) -> FactorizationProgressGuard {
    *factorization_progress_slot()
        .lock()
        .expect("factorization progress slot poisoned") = Some(progress);
    FactorizationProgressGuard
}

pub fn current_factorization_progress() -> Option<FactorizationProgressSnapshot> {
    factorization_progress_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|progress| progress.snapshot()))
}

#[derive(Clone, Debug, PartialEq)]
struct FactorColumn {
    global_column: usize,
    entries: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
struct ContributionBlock {
    row_ids: Vec<usize>,
    delayed_count: usize,
    dense: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct MultifrontalFactorizationOutcome {
    pivot_stats: PivotStats,
    factorization_residual_max_abs: f64,
    stored_nnz: usize,
    factor_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct FactorBlockRecord {
    size: usize,
    values: [f64; 4],
}

#[derive(Clone, Debug, PartialEq)]
struct FactorSolvePanelRecord {
    eliminated_len: usize,
    row_ids: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct SolvePanel {
    eliminated_len: usize,
    row_positions: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct FrontFactorizationResult {
    factor_order: Vec<usize>,
    factor_columns: Vec<FactorColumn>,
    block_records: Vec<FactorBlockRecord>,
    solve_panels: Vec<FactorSolvePanelRecord>,
    contribution: ContributionBlock,
    stats: PanelFactorStats,
    max_front_size: usize,
    contribution_storage_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct DenseFrontFactorization {
    factor_order: Vec<usize>,
    factor_columns: Vec<FactorColumn>,
    block_records: Vec<FactorBlockRecord>,
    solve_panels: Vec<FactorSolvePanelRecord>,
    contribution: ContributionBlock,
    stats: PanelFactorStats,
}

struct NumericFactorBuffers<'a> {
    factor_order: &'a mut Vec<usize>,
    factor_inverse: &'a mut Vec<usize>,
    lower_col_ptrs: &'a mut Vec<usize>,
    lower_row_indices: &'a mut Vec<usize>,
    lower_values: &'a mut Vec<f64>,
    diagonal_blocks: &'a mut Vec<DiagonalBlock>,
    diagonal_values: &'a mut Vec<f64>,
    solve_panels: &'a mut Vec<SolvePanel>,
    permuted_matrix_col_ptrs: &'a mut Vec<usize>,
    permuted_matrix_row_indices: &'a mut Vec<usize>,
    permuted_matrix_source_positions: &'a mut Vec<usize>,
    permuted_matrix_values: &'a mut Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericFactor {
    dimension: usize,
    permutation: Permutation,
    pattern_col_ptrs: Vec<usize>,
    pattern_row_indices: Vec<usize>,
    diagonal_blocks: Vec<DiagonalBlock>,
    diagonal_values: Vec<f64>,
    inertia: Inertia,
    pivot_stats: PivotStats,
    options: NumericFactorOptions,
    factor_order: Vec<usize>,
    factor_inverse: Vec<usize>,
    lower_col_ptrs: Vec<usize>,
    lower_row_indices: Vec<usize>,
    lower_values: Vec<f64>,
    solve_panels: Vec<SolvePanel>,
    solve_workspace: Vec<f64>,
    symbolic_front_tree: SymbolicFrontTree,
    permuted_matrix_col_ptrs: Vec<usize>,
    permuted_matrix_row_indices: Vec<usize>,
    permuted_matrix_source_positions: Vec<usize>,
    permuted_matrix_values: Vec<f64>,
    stored_nnz_cached: usize,
    factor_bytes_cached: usize,
    symbolic_supernode_count_cached: usize,
    symbolic_max_supernode_width_cached: usize,
}

impl NumericFactor {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn permutation(&self) -> &Permutation {
        &self.permutation
    }

    pub fn inertia(&self) -> Inertia {
        self.inertia
    }

    pub fn pivot_stats(&self) -> PivotStats {
        self.pivot_stats
    }

    pub fn stored_nnz(&self) -> usize {
        self.stored_nnz_cached
    }

    pub fn supernode_count(&self) -> usize {
        self.symbolic_supernode_count_cached
    }

    pub fn max_supernode_width(&self) -> usize {
        self.symbolic_max_supernode_width_cached
    }

    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes_cached
    }

    pub fn solve(&mut self, rhs: &[f64]) -> Result<Vec<f64>, SsidsError> {
        let mut solution = rhs.to_vec();
        self.solve_in_place(&mut solution)?;
        Ok(solution)
    }

    pub fn solve_with_profile(
        &mut self,
        rhs: &[f64],
    ) -> Result<(Vec<f64>, SolveProfile), SsidsError> {
        let mut solution = rhs.to_vec();
        let profile = self.solve_in_place_with_profile(&mut solution)?;
        Ok((solution, profile))
    }

    /// Solve `Ax = rhs` in place for the factorized matrix. The slice length
    /// must match the factor dimension exactly.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), SsidsError> {
        self.solve_in_place_impl(rhs, None)
    }

    pub fn solve_in_place_with_profile(
        &mut self,
        rhs: &mut [f64],
    ) -> Result<SolveProfile, SsidsError> {
        let mut profile = SolveProfile::default();
        self.solve_in_place_impl(rhs, Some(&mut profile))?;
        Ok(profile)
    }

    fn solve_in_place_impl(
        &mut self,
        rhs: &mut [f64],
        mut profile: Option<&mut SolveProfile>,
    ) -> Result<(), SsidsError> {
        if rhs.len() != self.dimension {
            return Err(SsidsError::SolveDimensionMismatch {
                expected: self.dimension,
                actual: rhs.len(),
            });
        }
        if self.dimension == 0 {
            return Ok(());
        }

        if self.solve_workspace.len() != self.dimension {
            self.solve_workspace.resize(self.dimension, 0.0);
        }
        let factor_rhs = &mut self.solve_workspace;
        let profile_enabled = profile.is_some();

        let started = profile_enabled.then(Instant::now);
        for (factor_position, &ordered_index) in self.factor_order.iter().enumerate() {
            factor_rhs[factor_position] = rhs[self.permutation.perm()[ordered_index]];
        }
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .input_permutation_time += started.elapsed();
        }

        let started = profile_enabled.then(Instant::now);
        if self.dimension <= APP_INNER_BLOCK_SIZE {
            let dense_lower = build_dense_unit_lower_from_factor(
                self.dimension,
                &self.lower_col_ptrs,
                &self.lower_row_indices,
                &self.lower_values,
            );
            for row in 0..self.dimension {
                let mut value = factor_rhs[row];
                for (col, &column_value) in factor_rhs.iter().take(row).enumerate() {
                    let offset = row * self.dimension + col;
                    if column_value != 0.0 && dense_lower.present[offset] {
                        let coefficient = dense_lower.values[offset];
                        value = (-coefficient).mul_add(column_value, value);
                    }
                }
                factor_rhs[row] = value;
            }
        } else {
            for pivot in 0..self.dimension {
                let pivot_value = factor_rhs[pivot];
                for entry in self.lower_col_ptrs[pivot]..self.lower_col_ptrs[pivot + 1] {
                    let row = self.lower_row_indices[entry];
                    factor_rhs[row] =
                        (-self.lower_values[entry]).mul_add(pivot_value, factor_rhs[row]);
                }
            }
        }
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .forward_substitution_time += started.elapsed();
        }

        let started = profile_enabled.then(Instant::now);
        let diagonal_result = (|| {
            let mut diagonal_start = 0;
            for (block, values) in self
                .diagonal_blocks
                .iter()
                .zip(self.diagonal_values.chunks_exact(4))
            {
                let start = diagonal_start;
                let end = start + block.size;
                if block.size == 1 {
                    let inverse_diagonal =
                        one_by_one_inverse_diagonal(&values[..2]).map_err(|detail| {
                            SsidsError::NumericalBreakdown {
                                pivot: start,
                                detail,
                            }
                        })?;
                    if !inverse_diagonal.is_finite() {
                        return Err(SsidsError::NumericalBreakdown {
                            pivot: start,
                            detail: "diagonal pivot vanished during solve".into(),
                        });
                    }
                    factor_rhs[start] *= inverse_diagonal;
                } else if block.size == 2 {
                    solve_two_by_two_block_in_place(values, &mut factor_rhs[start..end]).map_err(
                        |detail| SsidsError::NumericalBreakdown {
                            pivot: start,
                            detail,
                        },
                    )?;
                } else {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: start,
                        detail: format!(
                            "unexpected dense diagonal block of size {} in solve path",
                            block.size
                        ),
                    });
                }
                diagonal_start = end;
            }
            Ok(())
        })();
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .diagonal_solve_time += started.elapsed();
        }
        diagonal_result?;

        if self.dimension > 0 && self.solve_panels.is_empty() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: self.dimension - 1,
                detail: "solve panel metadata is missing for backward substitution".into(),
            });
        }
        let started = profile_enabled.then(Instant::now);
        solve_lower_transpose_front_panels_like_native(
            &self.solve_panels,
            factor_rhs,
            profile.as_deref_mut(),
        );
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .backward_substitution_time += started.elapsed();
        }
        if !factor_rhs.iter().all(|value| value.is_finite()) {
            return Err(SsidsError::NumericalBreakdown {
                pivot: self.dimension.saturating_sub(1),
                detail: "solve produced non-finite values".into(),
            });
        }
        let started = profile_enabled.then(Instant::now);
        for (factor_position, &ordered_index) in self.factor_order.iter().enumerate() {
            rhs[self.permutation.perm()[ordered_index]] = factor_rhs[factor_position];
        }
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .output_permutation_time += started.elapsed();
        }
        Ok(())
    }

    /// Reuse the analysed symbolic front tree for a new numeric factorization.
    ///
    /// The replacement matrix must have the same dimension and identical CSC
    /// sparsity structure as the original factorization.
    pub fn refactorize(
        &mut self,
        matrix: SymmetricCscMatrix<'_>,
    ) -> Result<FactorInfo, SsidsError> {
        if matrix.dimension() != self.dimension {
            return Err(SsidsError::DimensionMismatch {
                expected: self.dimension,
                actual: matrix.dimension(),
            });
        }
        if matrix.col_ptrs() != self.pattern_col_ptrs.as_slice()
            || matrix.row_indices() != self.pattern_row_indices.as_slice()
        {
            return Err(SsidsError::PatternMismatch(
                "refactorization requires identical CSC sparsity structure".into(),
            ));
        }
        self.refactorize_with_cached_symbolic(matrix)
    }

    fn refactorize_with_cached_symbolic(
        &mut self,
        matrix: SymmetricCscMatrix<'_>,
    ) -> Result<FactorInfo, SsidsError> {
        let factorization = multifrontal_factorize_with_tree(
            matrix,
            &self.permutation,
            &self.symbolic_front_tree,
            self.options,
            NumericFactorBuffers {
                factor_order: &mut self.factor_order,
                factor_inverse: &mut self.factor_inverse,
                lower_col_ptrs: &mut self.lower_col_ptrs,
                lower_row_indices: &mut self.lower_row_indices,
                lower_values: &mut self.lower_values,
                diagonal_blocks: &mut self.diagonal_blocks,
                diagonal_values: &mut self.diagonal_values,
                solve_panels: &mut self.solve_panels,
                permuted_matrix_col_ptrs: &mut self.permuted_matrix_col_ptrs,
                permuted_matrix_row_indices: &mut self.permuted_matrix_row_indices,
                permuted_matrix_source_positions: &mut self.permuted_matrix_source_positions,
                permuted_matrix_values: &mut self.permuted_matrix_values,
            },
        )?;
        let info = FactorInfo {
            factorization_residual_max_abs: factorization.factorization_residual_max_abs,
        };
        self.inertia = inertia_from_blocks(
            &self.diagonal_blocks,
            &self.diagonal_values,
            self.options.inertia_zero_tol,
        );
        self.pivot_stats = factorization.pivot_stats;
        self.stored_nnz_cached = factorization.stored_nnz;
        self.factor_bytes_cached = factorization.factor_bytes;
        Ok(info)
    }
}

#[derive(Debug, Error)]
pub enum SsidsError {
    #[error("invalid symmetric matrix: {0}")]
    InvalidMatrix(String),
    #[error("ordering failed: {0}")]
    Ordering(#[from] OrderingError),
    #[error("numeric factorization requires explicit matrix values")]
    MissingValues,
    #[error("matrix dimension {actual} does not match expected symbolic dimension {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("matrix sparsity pattern mismatch: {0}")]
    PatternMismatch(String),
    #[error("solve rhs length mismatch: expected {expected}, got {actual}")]
    SolveDimensionMismatch { expected: usize, actual: usize },
    #[error("numeric factorization broke down at pivot {pivot}: {detail}")]
    NumericalBreakdown { pivot: usize, detail: String },
}

fn analyse_debug_enabled() -> bool {
    std::env::var_os("SPRAL_SSIDS_DEBUG_ANALYSE").is_some()
}

fn analyse_debug_log(message: impl AsRef<str>) {
    if analyse_debug_enabled() {
        eprintln!("{}", message.as_ref());
    }
}

pub fn analyse(
    matrix: SymmetricCscMatrix<'_>,
    options: &SsidsOptions,
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    let graph =
        CsrGraph::from_symmetric_csc(matrix.dimension(), matrix.col_ptrs(), matrix.row_indices())?;
    let column_has_entries = (0..matrix.dimension())
        .map(|col| matrix.col_ptrs()[col + 1] > matrix.col_ptrs()[col])
        .collect::<Vec<_>>();
    let analyse_started = Instant::now();
    match options.ordering {
        OrderingStrategy::Natural => {
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=natural dim={} nnz={}",
                matrix.dimension(),
                matrix.row_indices().len(),
            ));
            let permutation = Permutation::identity(matrix.dimension());
            let result = build_symbolic_result_with_native_order(
                matrix,
                &graph,
                permutation,
                &column_has_entries,
                "natural",
            )?;
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=natural done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::ApproximateMinimumDegree => {
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=amd dim={} nnz={} start",
                matrix.dimension(),
                matrix.row_indices().len(),
            ));
            let summary = approximate_minimum_degree_order(&graph)?;
            let result = build_symbolic_result_with_native_order(
                matrix,
                &graph,
                summary.permutation,
                &column_has_entries,
                "approximate_minimum_degree",
            )?;
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=amd done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::NestedDissection(ordering_options) => {
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=nested_dissection dim={} nnz={} start",
                matrix.dimension(),
                matrix.row_indices().len(),
            ));
            let summary = nested_dissection_order(&graph, &ordering_options)?;
            let result = build_symbolic_result_with_native_order(
                matrix,
                &graph,
                summary.permutation,
                &column_has_entries,
                "nested_dissection",
            )?;
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=nested_dissection done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::Auto(ordering_options) => {
            analyse_debug_log(format!(
                "[spral_ssids::analyse] strategy=auto dim={} nnz={} start",
                matrix.dimension(),
                matrix.row_indices().len(),
            ));
            let natural_permutation = Permutation::identity(matrix.dimension());
            analyse_debug_log("[spral_ssids::analyse] auto natural symbolic start");
            let (_, natural_counts, _) = symbolic_factor_pattern(&graph);
            let natural_fill = natural_counts.iter().sum::<usize>();
            analyse_debug_log(format!(
                "[spral_ssids::analyse] auto natural symbolic done fill={} elapsed={:.3}s",
                natural_fill,
                analyse_started.elapsed().as_secs_f64(),
            ));

            let amd_started = Instant::now();
            analyse_debug_log("[spral_ssids::analyse] auto amd start");
            let amd_summary = approximate_minimum_degree_order(&graph)?;
            let amd_graph = permute_graph(&graph, &amd_summary.permutation);
            let (_, amd_counts, _) = symbolic_factor_pattern(&amd_graph);
            let amd_fill = amd_counts.iter().sum::<usize>();
            analyse_debug_log(format!(
                "[spral_ssids::analyse] auto amd done fill={} elapsed={:.3}s total={:.3}s",
                amd_fill,
                amd_started.elapsed().as_secs_f64(),
                analyse_started.elapsed().as_secs_f64(),
            ));

            let nd_started = Instant::now();
            analyse_debug_log("[spral_ssids::analyse] auto nested dissection start");
            let summary = nested_dissection_order(&graph, &ordering_options)?;
            let permuted_graph = permute_graph(&graph, &summary.permutation);
            let (_, nd_counts, _) = symbolic_factor_pattern(&permuted_graph);
            let nd_fill = nd_counts.iter().sum::<usize>();
            analyse_debug_log(format!(
                "[spral_ssids::analyse] auto nested dissection done fill={} elapsed={:.3}s total={:.3}s",
                nd_fill,
                nd_started.elapsed().as_secs_f64(),
                analyse_started.elapsed().as_secs_f64(),
            ));

            if amd_fill <= natural_fill && amd_fill <= nd_fill {
                let result = build_symbolic_result_with_native_order(
                    matrix,
                    &graph,
                    amd_summary.permutation,
                    &column_has_entries,
                    "auto_approximate_minimum_degree",
                )?;
                analyse_debug_log(format!(
                    "[spral_ssids::analyse] auto selected=amd total={:.3}s",
                    analyse_started.elapsed().as_secs_f64(),
                ));
                Ok(result)
            } else if nd_fill <= natural_fill {
                let result = build_symbolic_result_with_native_order(
                    matrix,
                    &graph,
                    summary.permutation,
                    &column_has_entries,
                    "auto_nested_dissection",
                )?;
                analyse_debug_log(format!(
                    "[spral_ssids::analyse] auto selected=nested_dissection total={:.3}s",
                    analyse_started.elapsed().as_secs_f64(),
                ));
                Ok(result)
            } else {
                let result = build_symbolic_result_with_native_order(
                    matrix,
                    &graph,
                    natural_permutation,
                    &column_has_entries,
                    "auto_natural",
                )?;
                analyse_debug_log(format!(
                    "[spral_ssids::analyse] auto selected=natural total={:.3}s",
                    analyse_started.elapsed().as_secs_f64(),
                ));
                Ok(result)
            }
        }
    }
}

pub fn approximate_minimum_degree_permutation(
    matrix: SymmetricCscMatrix<'_>,
) -> Result<Permutation, SsidsError> {
    let graph =
        CsrGraph::from_symmetric_csc(matrix.dimension(), matrix.col_ptrs(), matrix.row_indices())?;
    Ok(approximate_minimum_degree_order(&graph)?.permutation)
}

/// Perform a numeric multifrontal LDL^T factorization for a previously
/// analyzed symmetric CSC matrix.
pub fn factorize(
    matrix: SymmetricCscMatrix<'_>,
    symbolic: &SymbolicFactor,
    options: &NumericFactorOptions,
) -> Result<(NumericFactor, FactorInfo), SsidsError> {
    if matrix.dimension() != symbolic.permutation.len() {
        return Err(SsidsError::DimensionMismatch {
            expected: symbolic.permutation.len(),
            actual: matrix.dimension(),
        });
    }
    let front_tree = build_symbolic_front_tree(symbolic);
    let mut factor = NumericFactor {
        dimension: matrix.dimension(),
        permutation: symbolic.permutation.clone(),
        pattern_col_ptrs: matrix.col_ptrs().to_vec(),
        pattern_row_indices: matrix.row_indices().to_vec(),
        diagonal_blocks: Vec::new(),
        diagonal_values: Vec::new(),
        inertia: Inertia {
            positive: 0,
            negative: 0,
            zero: 0,
        },
        pivot_stats: PivotStats {
            two_by_two_pivots: 0,
            delayed_pivots: 0,
        },
        options: *options,
        factor_order: Vec::with_capacity(matrix.dimension()),
        factor_inverse: Vec::with_capacity(matrix.dimension()),
        lower_col_ptrs: Vec::with_capacity(matrix.dimension() + 1),
        lower_row_indices: Vec::new(),
        lower_values: Vec::new(),
        solve_panels: Vec::new(),
        solve_workspace: vec![0.0; matrix.dimension()],
        symbolic_front_tree: front_tree,
        permuted_matrix_col_ptrs: Vec::with_capacity(matrix.dimension() + 1),
        permuted_matrix_row_indices: Vec::with_capacity(matrix.row_indices().len()),
        permuted_matrix_source_positions: Vec::with_capacity(matrix.row_indices().len()),
        permuted_matrix_values: Vec::with_capacity(matrix.row_indices().len()),
        stored_nnz_cached: 0,
        factor_bytes_cached: 0,
        symbolic_supernode_count_cached: symbolic.supernodes.len(),
        symbolic_max_supernode_width_cached: symbolic
            .supernodes
            .iter()
            .map(Supernode::width)
            .max()
            .unwrap_or(0),
    };
    let info = factor.refactorize_with_cached_symbolic(matrix)?;
    Ok((factor, info))
}

fn permute_graph(graph: &CsrGraph, permutation: &Permutation) -> CsrGraph {
    let edges = (0..graph.vertex_count())
        .flat_map(|vertex| {
            graph
                .neighbors(vertex)
                .iter()
                .copied()
                .filter_map(move |neighbor| {
                    (vertex < neighbor).then_some((
                        permutation.inverse()[vertex],
                        permutation.inverse()[neighbor],
                    ))
                })
        })
        .collect::<Vec<_>>();
    CsrGraph::from_edges(graph.vertex_count(), &edges).expect("permutation preserves graph shape")
}

fn build_symbolic_result_with_native_order(
    matrix: SymmetricCscMatrix<'_>,
    graph: &CsrGraph,
    mut current_permutation: Permutation,
    column_has_entries: &[bool],
    ordering_kind: &'static str,
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    let expanded_pattern = expand_symmetric_pattern(matrix);
    let mut current_graph = permute_graph(graph, &current_permutation);
    let (initial_tree, _, _) = symbolic_factor_pattern(&current_graph);
    let (postorder_permutation, realn) =
        native_postorder_permutation(&initial_tree, &current_permutation, column_has_entries);
    if !is_identity_order(&postorder_permutation) {
        current_permutation = compose_ordering_with_symbolic_permutation(
            &current_permutation,
            &postorder_permutation,
        )?;
        current_graph = permute_graph(graph, &current_permutation);
    }

    let (elimination_tree, simulated_column_counts, column_pattern) =
        symbolic_factor_pattern(&current_graph);
    let column_counts =
        native_column_counts(&expanded_pattern, &current_permutation, &elimination_tree);
    debug_assert_eq!(column_counts, simulated_column_counts);
    let supernode_layout = native_supernode_layout(&elimination_tree, &column_counts, realn);
    if is_identity_order(&supernode_layout.permutation) {
        let supernodes = build_native_row_list_supernodes(
            &expanded_pattern,
            &current_permutation,
            &supernode_layout,
            &column_pattern,
        );
        return Ok(build_symbolic_result(
            current_permutation,
            elimination_tree,
            column_counts,
            column_pattern,
            supernodes,
            ordering_kind,
        ));
    }

    let final_permutation = compose_ordering_with_symbolic_permutation(
        &current_permutation,
        &supernode_layout.permutation,
    )?;
    let final_graph = permute_graph(graph, &final_permutation);
    let (final_tree, simulated_final_counts, final_pattern) = symbolic_factor_pattern(&final_graph);
    let final_counts = native_column_counts(&expanded_pattern, &final_permutation, &final_tree);
    debug_assert_eq!(final_counts, simulated_final_counts);
    let final_supernodes = build_native_row_list_supernodes(
        &expanded_pattern,
        &final_permutation,
        &supernode_layout,
        &final_pattern,
    );
    Ok(build_symbolic_result(
        final_permutation,
        final_tree,
        final_counts,
        final_pattern,
        final_supernodes,
        ordering_kind,
    ))
}

fn compose_ordering_with_symbolic_permutation(
    base_permutation: &Permutation,
    symbolic_permutation: &[usize],
) -> Result<Permutation, OrderingError> {
    let mut perm = vec![usize::MAX; base_permutation.len()];
    for (old_position, &new_position) in symbolic_permutation.iter().enumerate() {
        perm[new_position] = base_permutation.perm()[old_position];
    }
    Permutation::new(perm)
}

fn is_identity_order(order: &[usize]) -> bool {
    order
        .iter()
        .enumerate()
        .all(|(index, &value)| index == value)
}

fn native_postorder_permutation(
    elimination_tree: &[Option<usize>],
    base_permutation: &Permutation,
    column_has_entries: &[bool],
) -> (Vec<usize>, usize) {
    // Mirror SPRAL's core_analyse.find_postorder map construction. Native
    // tests empty roots against raw CSC column length, so use the original
    // column indices from the current inverse permutation rather than graph
    // degree, which omits diagonal storage.
    let n = elimination_tree.len();
    if base_permutation.len() != n || column_has_entries.len() != n {
        return ((0..n).collect(), n);
    }

    let virtual_root = n;
    let mut children = vec![Vec::new(); n + 1];
    for (node, parent) in elimination_tree.iter().copied().enumerate() {
        children[parent.unwrap_or(virtual_root)].push(node);
    }

    let mut permutation = vec![usize::MAX; n];
    let mut realn = n;
    let mut next_id = n;
    let mut stack = vec![virtual_root];
    while let Some(node) = stack.pop() {
        if node != virtual_root {
            next_id = next_id.saturating_sub(1);
            permutation[node] = next_id;
        }

        if node == virtual_root {
            for &child in &children[node] {
                let original_column = base_permutation.perm()[child];
                if column_has_entries[original_column] {
                    stack.push(child);
                }
            }
            for &child in &children[node] {
                let original_column = base_permutation.perm()[child];
                if !column_has_entries[original_column] {
                    realn -= 1;
                    stack.push(child);
                }
            }
        } else {
            for &child in &children[node] {
                stack.push(child);
            }
        }
    }

    if permutation.contains(&usize::MAX) {
        return ((0..n).collect(), n);
    }
    (permutation, realn)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeSupernodeLayout {
    permutation: Vec<usize>,
    ranges: Vec<std::ops::Range<usize>>,
    parents: Vec<Option<usize>>,
}

fn native_supernode_layout(
    elimination_tree: &[Option<usize>],
    column_counts: &[usize],
    realn: usize,
) -> NativeSupernodeLayout {
    // Mirror the renumbering part of SPRAL's core_analyse.find_supernodes:
    // merged vertices are walked with a small stack, producing both the
    // second symbolic permutation and the final supernode ranges (`sptr`).
    let n = elimination_tree.len();
    let realn = realn.min(n);
    let identity = || NativeSupernodeLayout {
        permutation: (0..n).collect(),
        ranges: (0..n).map(|node| node..node + 1).collect(),
        parents: elimination_tree.to_vec(),
    };
    let virtual_root = n;
    let mut children = vec![Vec::new(); n + 1];
    for (node, parent) in elimination_tree.iter().copied().enumerate().take(realn) {
        children[parent.unwrap_or(virtual_root)].push(node);
    }

    let mut nelim = vec![1_usize; n + 1];
    let mut nvert = vec![1_usize; n + 1];
    let mut vhead = vec![None; n + 1];
    let mut vnext = vec![None; n + 1];
    let mut marked = vec![false; n];
    nelim[virtual_root] = n + 1 + RELAXED_NODE_AMALGAMATION_NEMIN;

    for parent in 0..=n {
        children[parent].sort_by(|&lhs, &rhs| column_counts[rhs].cmp(&column_counts[lhs]));
        for &node in &children[parent] {
            if native_should_merge_supernode(node, parent, &nelim, column_counts) {
                vnext[node] = vhead[parent];
                vhead[parent] = Some(node);
                nelim[parent] += nelim[node];
                nvert[parent] += nvert[node];
                marked[node] = false;
            } else {
                marked[node] = true;
            }
        }
    }

    let mut permutation = vec![usize::MAX; n];
    let mut vertex_to_supernode = vec![usize::MAX; n + 1];
    let mut parent_vertices = Vec::new();
    let mut ranges = Vec::new();
    let mut next_position = 0;
    let mut stack = Vec::new();
    for node in 0..realn {
        if !marked[node] {
            continue;
        }
        let start = next_position;
        next_position += nvert[node];
        let supernode_index = ranges.len();
        parent_vertices.push(elimination_tree[node].unwrap_or(virtual_root));
        ranges.push(start..next_position);
        let mut position = next_position;
        stack.push(node);
        while let Some(vertex) = stack.pop() {
            position -= 1;
            permutation[vertex] = position;
            vertex_to_supernode[vertex] = supernode_index;
            if let Some(next) = vnext[vertex] {
                stack.push(next);
            }
            if let Some(head) = vhead[vertex] {
                stack.push(head);
            }
        }
    }
    for (node, slot) in permutation.iter_mut().enumerate().skip(realn) {
        *slot = node;
    }

    if permutation.contains(&usize::MAX) {
        return identity();
    }
    vertex_to_supernode[virtual_root] = ranges.len();
    let parents = parent_vertices
        .into_iter()
        .map(|parent| {
            (parent != virtual_root)
                .then_some(vertex_to_supernode[parent])
                .filter(|&parent_supernode| parent_supernode != usize::MAX)
        })
        .collect();
    NativeSupernodeLayout {
        permutation,
        ranges,
        parents,
    }
}

fn native_should_merge_supernode(
    node: usize,
    parent: usize,
    nelim: &[usize],
    column_counts: &[usize],
) -> bool {
    if parent >= column_counts.len() {
        return false;
    }
    (column_counts[parent] == column_counts[node].saturating_sub(1) && nelim[parent] == 1)
        || (nelim[parent] < RELAXED_NODE_AMALGAMATION_NEMIN
            && nelim[node] < RELAXED_NODE_AMALGAMATION_NEMIN)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpandedSymmetricPattern {
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
}

impl ExpandedSymmetricPattern {
    fn dimension(&self) -> usize {
        self.col_ptrs.len().saturating_sub(1)
    }
}

fn expand_symmetric_pattern(matrix: SymmetricCscMatrix<'_>) -> ExpandedSymmetricPattern {
    // Mirror ssids/anal.F90 expand_pattern: expand user lower-CSC input into
    // a full symmetric CSC pattern before calling core_analyse.
    let n = matrix.dimension();
    let mut counts = vec![0_usize; n];
    for col in 0..n {
        for &row in &matrix.row_indices()[matrix.col_ptrs()[col]..matrix.col_ptrs()[col + 1]] {
            counts[row] += 1;
            if row != col {
                counts[col] += 1;
            }
        }
    }

    let mut col_ptrs = vec![0_usize; n + 1];
    for col in 0..n {
        col_ptrs[col + 1] = col_ptrs[col] + counts[col];
    }

    let mut write_ptrs = col_ptrs[1..].to_vec();
    let mut row_indices = vec![usize::MAX; col_ptrs[n]];
    for col in 0..n {
        for &row in &matrix.row_indices()[matrix.col_ptrs()[col]..matrix.col_ptrs()[col + 1]] {
            write_ptrs[row] -= 1;
            row_indices[write_ptrs[row]] = col;
            if row != col {
                write_ptrs[col] -= 1;
                row_indices[write_ptrs[col]] = row;
            }
        }
    }

    debug_assert_eq!(&write_ptrs, &col_ptrs[..n]);
    ExpandedSymmetricPattern {
        col_ptrs,
        row_indices,
    }
}

fn native_column_counts(
    pattern: &ExpandedSymmetricPattern,
    permutation: &Permutation,
    elimination_tree: &[Option<usize>],
) -> Vec<usize> {
    // Mirror SPRAL core_analyse.find_col_counts. The routine tracks the net
    // number of entries first appearing at each elimination-tree node, then
    // passes those counts up the virtual forest to form final column counts.
    let n = elimination_tree.len();
    assert_eq!(pattern.dimension(), n, "symbolic matrix dimension mismatch");
    assert_eq!(permutation.len(), n, "symbolic permutation mismatch");

    let virtual_root = n;
    let mut first = (0..=n).collect::<Vec<_>>();
    let mut column_counts = vec![0_isize; n + 1];
    for node in 0..n {
        let parent = elimination_tree[node].unwrap_or(virtual_root);
        first[parent] = first[parent].min(first[node]);
        column_counts[node] = if first[node] == node { 1 } else { 0 };
    }
    column_counts[virtual_root] = (n + 1) as isize;

    let mut virtual_forest = vec![None; n + 1];
    let mut last_pivot = vec![None; n + 1];
    let mut last_neighbor = vec![None; n + 1];
    for pivot in 0..n {
        let original_col = permutation.perm()[pivot];
        for source in pattern.col_ptrs[original_col]..pattern.col_ptrs[original_col + 1] {
            let row = permutation.inverse()[pattern.row_indices[source]];
            if row <= pivot {
                continue;
            }

            let first_seen_in_subtree = last_neighbor[row].is_none_or(|last| first[pivot] > last);
            if first_seen_in_subtree {
                column_counts[pivot] += 1;
                if let Some(previous_pivot) = last_pivot[row] {
                    let lca = native_virtual_forest_find(&mut virtual_forest, previous_pivot);
                    column_counts[lca] -= 1;
                }
                last_pivot[row] = Some(pivot);
            }
            last_neighbor[row] = Some(pivot);
        }

        let parent = elimination_tree[pivot].unwrap_or(virtual_root);
        column_counts[parent] += column_counts[pivot] - 1;
        virtual_forest[pivot] = Some(parent);
    }

    column_counts
        .into_iter()
        .take(n)
        .map(|count| {
            usize::try_from(count).expect("native column count should not be negative on exit")
        })
        .collect()
}

fn native_virtual_forest_find(virtual_forest: &mut [Option<usize>], node: usize) -> usize {
    let mut current = node;
    while let Some(parent) = virtual_forest[current] {
        if let Some(grandparent) = virtual_forest[parent] {
            virtual_forest[current] = Some(grandparent);
        }
        current = parent;
    }
    current
}

fn symbolic_factor_pattern(graph: &CsrGraph) -> (Vec<Option<usize>>, Vec<usize>, Vec<Vec<usize>>) {
    let dimension = graph.vertex_count();
    let mut adjacency = (0..dimension)
        .map(|vertex| graph.neighbors(vertex).to_vec())
        .collect::<Vec<_>>();
    let mut elimination_tree = vec![None; dimension];
    let mut column_counts = vec![1; dimension];
    let mut column_pattern = vec![Vec::new(); dimension];
    for column in 0..dimension {
        let mut active_neighbors = adjacency[column]
            .iter()
            .copied()
            .filter(|&neighbor| neighbor > column)
            .collect::<Vec<_>>();
        active_neighbors.sort_unstable();
        active_neighbors.dedup();
        elimination_tree[column] = active_neighbors.first().copied();
        column_counts[column] = active_neighbors.len() + 1;
        column_pattern[column].push(column);
        column_pattern[column].extend(active_neighbors.iter().copied());
        for idx in 0..active_neighbors.len() {
            for jdx in (idx + 1)..active_neighbors.len() {
                let lhs = active_neighbors[idx];
                let rhs = active_neighbors[jdx];
                if !adjacency[lhs].contains(&rhs) {
                    adjacency[lhs].push(rhs);
                    adjacency[lhs].sort_unstable();
                }
                if !adjacency[rhs].contains(&lhs) {
                    adjacency[rhs].push(lhs);
                    adjacency[rhs].sort_unstable();
                }
            }
        }
    }
    (elimination_tree, column_counts, column_pattern)
}

fn build_native_row_list_supernodes(
    expanded_pattern: &ExpandedSymmetricPattern,
    permutation: &Permutation,
    layout: &NativeSupernodeLayout,
    column_pattern: &[Vec<usize>],
) -> Vec<Supernode> {
    let mut supernodes = Vec::new();
    let native_row_lists = native_row_lists(
        expanded_pattern,
        permutation,
        &layout.ranges,
        &layout.parents,
    );
    let mut next_column = 0;
    for (range, row_list) in layout.ranges.iter().zip(native_row_lists.iter()) {
        for column in next_column..range.start {
            supernodes.push(unit_supernode(column, column_pattern));
        }
        let trailing_rows = row_list
            .iter()
            .copied()
            .filter(|&row| row >= range.end)
            .collect();
        supernodes.push(Supernode {
            start_column: range.start,
            end_column: range.end,
            trailing_rows,
        });
        next_column = range.end;
    }
    for column in next_column..column_pattern.len() {
        supernodes.push(unit_supernode(column, column_pattern));
    }
    supernodes
}

fn native_row_lists(
    pattern: &ExpandedSymmetricPattern,
    permutation: &Permutation,
    ranges: &[std::ops::Range<usize>],
    parents: &[Option<usize>],
) -> Vec<Vec<usize>> {
    // Mirror core_analyse.find_row_lists followed by dbl_tr_sort. Native
    // builds each row list bottom-up from eliminated pivots, child row lists,
    // and expanded CSC entries under the current pivot permutation.
    let n = permutation.len();
    assert_eq!(pattern.dimension(), n, "symbolic matrix dimension mismatch");
    assert_eq!(
        parents.len(),
        ranges.len(),
        "native supernode parent/range mismatch"
    );

    let mut children = vec![Vec::new(); ranges.len()];
    for (node, parent) in parents.iter().copied().enumerate() {
        if let Some(parent) = parent
            && parent < children.len()
        {
            children[parent].push(node);
        }
    }

    let mut row_lists = vec![Vec::new(); ranges.len()];
    let mut seen = vec![0_usize; n];
    for node in 0..ranges.len() {
        let tag = node + 1;
        let range = &ranges[node];
        let mut row_list = Vec::new();

        for pivot in range.clone() {
            seen[pivot] = tag;
            row_list.push(pivot);
        }

        for &child in &children[node] {
            for &row in &row_lists[child] {
                if row < range.start || seen[row] == tag {
                    continue;
                }
                seen[row] = tag;
                row_list.push(row);
            }
        }

        for pivot in range.clone() {
            let original_col = permutation.perm()[pivot];
            for source in pattern.col_ptrs[original_col]..pattern.col_ptrs[original_col + 1] {
                let row = permutation.inverse()[pattern.row_indices[source]];
                if row < pivot || seen[row] == tag {
                    continue;
                }
                seen[row] = tag;
                row_list.push(row);
            }
        }

        row_list.sort_unstable();
        row_lists[node] = row_list;
    }
    row_lists
}

fn unit_supernode(column: usize, column_pattern: &[Vec<usize>]) -> Supernode {
    let trailing_rows = column_pattern[column]
        .iter()
        .copied()
        .filter(|&row| row > column)
        .collect::<Vec<_>>();
    Supernode {
        start_column: column,
        end_column: column + 1,
        trailing_rows,
    }
}

fn build_symbolic_result(
    permutation: Permutation,
    elimination_tree: Vec<Option<usize>>,
    column_counts: Vec<usize>,
    column_pattern: Vec<Vec<usize>>,
    supernodes: Vec<Supernode>,
    ordering_kind: &'static str,
) -> (SymbolicFactor, AnalyseInfo) {
    let supernode_count = supernodes.len();
    let max_supernode_width = supernodes.iter().map(Supernode::width).max().unwrap_or(0);
    let info = AnalyseInfo {
        estimated_fill_nnz: column_counts.iter().sum(),
        supernode_count,
        max_supernode_width,
        ordering_kind,
    };
    let symbolic = SymbolicFactor {
        permutation,
        elimination_tree,
        column_counts,
        column_pattern,
        supernodes,
    };
    (symbolic, info)
}

fn build_symbolic_front_tree(symbolic: &SymbolicFactor) -> SymbolicFrontTree {
    let mut column_to_front = vec![usize::MAX; symbolic.permutation.len()];
    let mut fronts = symbolic
        .supernodes
        .iter()
        .enumerate()
        .map(|(index, supernode)| {
            for slot in column_to_front
                .iter_mut()
                .take(supernode.end_column)
                .skip(supernode.start_column)
            {
                *slot = index;
            }
            SymbolicFront {
                columns: (supernode.start_column..supernode.end_column).collect(),
                interface_rows: supernode.trailing_rows.clone(),
                parent: None,
                children: Vec::new(),
            }
        })
        .collect::<Vec<_>>();

    for (index, front) in fronts.clone().iter().enumerate() {
        let parent = front.interface_rows.iter().copied().find_map(|row| {
            (row < column_to_front.len())
                .then_some(column_to_front[row])
                .filter(|&candidate| candidate != usize::MAX && candidate != index)
        });
        fronts[index].parent = parent;
    }
    for index in 0..fronts.len() {
        if let Some(parent) = fronts[index].parent {
            fronts[parent].children.push(index);
        }
    }
    let start_columns = fronts
        .iter()
        .map(SymbolicFront::first_column)
        .collect::<Vec<_>>();
    for front in &mut fronts {
        front.children.sort_by_key(|&child| start_columns[child]);
    }
    let roots = collect_root_fronts(&fronts);
    SymbolicFrontTree { roots, fronts }
}

fn front_work_weight(width: usize, interface_len: usize) -> u64 {
    let front_dim = width.saturating_add(interface_len) as u64;
    let width = width as u64;
    width.saturating_mul(front_dim).saturating_mul(front_dim)
}

fn collect_root_fronts(fronts: &[SymbolicFront]) -> Vec<usize> {
    fronts
        .iter()
        .enumerate()
        .filter_map(|(index, front)| front.parent.is_none().then_some(index))
        .collect()
}

struct PermutedLowerMatrix<'a> {
    dimension: usize,
    col_ptrs: &'a [usize],
    row_indices: &'a [usize],
    values: &'a [f64],
}

fn build_permuted_lower_csc_pattern(
    matrix: SymmetricCscMatrix<'_>,
    permutation: &Permutation,
    col_ptrs: &mut Vec<usize>,
    row_indices: &mut Vec<usize>,
    source_positions: &mut Vec<usize>,
) -> Result<(), SsidsError> {
    let dimension = matrix.dimension();
    let inverse = permutation.inverse();
    let nnz = matrix.row_indices().len();
    let mut counts = vec![0usize; dimension];
    for col in 0..dimension {
        let start = matrix.col_ptrs()[col];
        let end = matrix.col_ptrs()[col + 1];
        for &row in &matrix.row_indices()[start..end] {
            let permuted_col = inverse[col];
            let permuted_row = inverse[row];
            counts[permuted_col.min(permuted_row)] += 1;
        }
    }

    col_ptrs.clear();
    col_ptrs.resize(dimension + 1, 0);
    for col in 0..dimension {
        col_ptrs[col + 1] = col_ptrs[col] + counts[col];
    }

    row_indices.clear();
    row_indices.resize(nnz, 0);
    source_positions.clear();
    source_positions.resize(nnz, 0);
    let mut next = col_ptrs[..dimension].to_vec();
    for col in 0..dimension {
        let start = matrix.col_ptrs()[col];
        let end = matrix.col_ptrs()[col + 1];
        for source_index in start..end {
            let row = matrix.row_indices()[source_index];
            let permuted_col = inverse[col];
            let permuted_row = inverse[row];
            let target_col = permuted_col.min(permuted_row);
            let target_row = permuted_col.max(permuted_row);
            let slot = next[target_col];
            row_indices[slot] = target_row;
            source_positions[slot] = source_index;
            next[target_col] += 1;
        }
    }
    Ok(())
}

fn fill_permuted_lower_csc_values(
    matrix: SymmetricCscMatrix<'_>,
    source_positions: &[usize],
    values: &mut Vec<f64>,
) -> Result<(), SsidsError> {
    let source_values = matrix.values().ok_or(SsidsError::MissingValues)?;
    values.clear();
    values.resize(source_positions.len(), 0.0);
    for (index, &source_index) in source_positions.iter().enumerate() {
        let value = source_values[source_index];
        if !value.is_finite() {
            let row = matrix.row_indices()[source_index];
            let col = matrix
                .col_ptrs()
                .partition_point(|&pointer| pointer <= source_index)
                .saturating_sub(1);
            return Err(SsidsError::InvalidMatrix(format!(
                "numeric value at ({row}, {col}) is not finite"
            )));
        }
        values[index] = value;
    }
    Ok(())
}

fn dense_symmetric_swap(matrix: &mut [f64], size: usize, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    let (lhs, rhs) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };

    for col in 0..lhs {
        let lhs_offset = dense_lower_offset(size, lhs, col);
        let rhs_offset = dense_lower_offset(size, rhs, col);
        matrix.swap(lhs_offset, rhs_offset);
    }

    for index in (lhs + 1)..rhs {
        let lhs_offset = dense_lower_offset(size, index, lhs);
        let rhs_offset = dense_lower_offset(size, rhs, index);
        matrix.swap(lhs_offset, rhs_offset);
    }

    for row in (rhs + 1)..size {
        let lhs_offset = dense_lower_offset(size, row, lhs);
        let rhs_offset = dense_lower_offset(size, row, rhs);
        matrix.swap(lhs_offset, rhs_offset);
    }

    let lhs_diag = dense_lower_offset(size, lhs, lhs);
    let rhs_diag = dense_lower_offset(size, rhs, rhs);
    matrix.swap(lhs_diag, rhs_diag);
}

fn dense_symmetric_swap_with_workspace(
    matrix: &mut [f64],
    size: usize,
    lhs: usize,
    rhs: usize,
    workspace: &mut [f64],
) {
    if lhs == rhs {
        return;
    }
    let (lhs, rhs) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
    for work_row in 0..lhs {
        workspace.swap(work_row * size + lhs, work_row * size + rhs);
    }
    dense_symmetric_swap(matrix, size, lhs, rhs);
}

#[inline]
fn dense_lower_offset(size: usize, row: usize, col: usize) -> usize {
    if row >= col {
        col * size + row
    } else {
        row * size + col
    }
}

#[inline]
fn packed_lower_len(size: usize) -> usize {
    size * (size + 1) / 2
}

#[inline]
fn packed_lower_offset(size: usize, row: usize, col: usize) -> usize {
    debug_assert!(row >= col);
    col * size - col * (col.saturating_sub(1)) / 2 + (row - col)
}

fn aggregate_panel_stats(target: &mut PanelFactorStats, source: PanelFactorStats) {
    target.two_by_two_pivots += source.two_by_two_pivots;
    target.delayed_pivots += source.delayed_pivots;
    target.max_residual = target.max_residual.max(source.max_residual);
}

fn scaled_two_by_two_inverse(a11: f64, a21: f64, a22: f64, small: f64) -> Option<(f64, f64, f64)> {
    let max_pivot = a11.abs().max(a21.abs()).max(a22.abs());
    if !a11.is_finite() || !a21.is_finite() || !a22.is_finite() || max_pivot < small {
        return None;
    }
    let detscale = 1.0 / max_pivot;
    let det0 = (a11 * detscale) * a22;
    let det1 = (a21 * detscale) * a21;
    let det = det0 - det1;
    if !det.is_finite() || det.abs() < small.max(det0.abs().max(det1.abs()) / 2.0) {
        return None;
    }
    let d11 = (a22 * detscale) / det;
    let d21 = (-a21 * detscale) / det;
    let d22 = (a11 * detscale) / det;
    if d11.is_finite() && d21.is_finite() && d22.is_finite() {
        Some((d11, d21, d22))
    } else {
        None
    }
}

fn app_two_by_two_inverse(a11: f64, a21: f64, a22: f64, small: f64) -> Option<(f64, f64, f64)> {
    if !a11.is_finite() || !a21.is_finite() || !a22.is_finite() || a21.abs() < small {
        return None;
    }
    let detscale = 1.0 / a21.abs();
    // Native SPRAL's block_ldlt contracts this multiply/subtract on the local build.
    let det = (a11 * detscale).mul_add(a22, -a21.abs());
    if !det.is_finite() || det.abs() < a21.abs() / 2.0 {
        return None;
    }
    let d11 = (a22 * detscale) / det;
    let d21 = (-a21 * detscale) / det;
    let d22 = (a11 * detscale) / det;
    if d11.is_finite() && d21.is_finite() && d22.is_finite() {
        Some((d11, d21, d22))
    } else {
        None
    }
}

fn one_by_one_inverse_diagonal(values: &[f64]) -> Result<f64, String> {
    if values.len() >= 2 {
        Ok(values[0])
    } else {
        Err("invalid one-by-one block dimensions".into())
    }
}

fn reset_ldwork_column_tail(workspace: &mut [f64], size: usize, col: usize, from: usize) {
    let column = &mut workspace[col * size..(col + 1) * size];
    column[from..].fill(0.0);
}

fn dense_find_maxloc(
    matrix: &[f64],
    size: usize,
    from: usize,
    to: usize,
) -> Option<(f64, usize, usize)> {
    if from >= to {
        return None;
    }
    debug_assert!(to >= APP_INNER_BLOCK_SIZE);
    let block_start = to - APP_INNER_BLOCK_SIZE;
    debug_assert!(from >= block_start);
    let local_from = from - block_start;

    let mut primary = (-1.0_f64, to, to);
    let mut secondary = (-1.0_f64, to, to);

    let update = |slot: &mut (f64, usize, usize), local_row: usize, local_col: usize| {
        let row = block_start + local_row;
        let col = block_start + local_col;
        let value = matrix[dense_lower_offset(size, row, col)].abs();
        if value > slot.0 {
            *slot = (value, row, col);
        }
    };

    // Native SPRAL's non-AVX SimdVec path still uses two per-lane maxima. Equal
    // values keep their existing lane, so ties are not column-major.
    for local_col in local_from..APP_INNER_BLOCK_SIZE {
        update(&mut primary, local_col, local_col);
        if local_col + 1 < 2 * (local_col / 2 + 1) {
            update(&mut primary, local_col + 1, local_col);
        }
        let mut local_row = 2 * (local_col / 2 + 1);
        while local_row < APP_INNER_BLOCK_SIZE {
            update(&mut primary, local_row, local_col);
            update(&mut secondary, local_row + 1, local_col);
            local_row += 2;
        }
    }

    let best = if secondary.0 > primary.0 {
        secondary
    } else {
        primary
    };
    (best.2 < to).then_some(best)
}

fn dense_find_rc_abs_max_exclude(
    matrix: &[f64],
    size: usize,
    col: usize,
    from: usize,
    exclude: Option<usize>,
) -> f64 {
    let mut best = 0.0_f64;
    for other_col in from..col {
        if Some(other_col) == exclude {
            continue;
        }
        best = best.max(matrix[dense_lower_offset(size, col, other_col)].abs());
    }
    for row in (col + 1)..size {
        if Some(row) == exclude {
            continue;
        }
        best = best.max(matrix[dense_lower_offset(size, row, col)].abs());
    }
    best
}

fn dense_column_small(matrix: &[f64], size: usize, col: usize, from: usize, small: f64) -> bool {
    for other_col in from..col {
        if matrix[dense_lower_offset(size, col, other_col)].abs() >= small {
            return false;
        }
    }
    for row in col..size {
        if matrix[dense_lower_offset(size, row, col)].abs() >= small {
            return false;
        }
    }
    true
}

fn dense_find_row_abs_max_in_column(
    matrix: &[f64],
    size: usize,
    col: usize,
    from: usize,
) -> Option<usize> {
    if from >= col || col >= size {
        return None;
    }
    let mut best_row = from;
    let mut best_value = matrix[dense_lower_offset(size, col, from)].abs();
    for row in (from + 1)..col {
        let value = matrix[dense_lower_offset(size, col, row)].abs();
        if value > best_value {
            best_value = value;
            best_row = row;
        }
    }
    Some(best_row)
}

fn tpp_test_two_by_two(
    a11: f64,
    a21: f64,
    a22: f64,
    maxt: f64,
    maxp: f64,
    options: NumericFactorOptions,
) -> Option<(f64, f64, f64)> {
    let (d11, d21, d22) = scaled_two_by_two_inverse(a11, a21, a22, options.small_pivot_tolerance)?;
    if maxt.max(maxp) < options.small_pivot_tolerance {
        return Some((d11, d21, d22));
    }
    let x1 = d11.abs().mul_add(maxt, d21.abs() * maxp);
    let x2 = d21.abs().mul_add(maxt, d22.abs() * maxp);
    if options.threshold_pivot_u * x1.max(x2) < 1.0 {
        Some((d11, d21, d22))
    } else {
        None
    }
}

fn app_update_one_by_one(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
) {
    let ld = &workspace[pivot * size..(pivot + 1) * size];
    for (col, &preserved) in ld.iter().enumerate().take(update_end).skip(pivot + 1) {
        for row in col..update_end {
            let update_entry = dense_lower_offset(size, row, col);
            let multiplier = matrix[dense_lower_offset(size, row, pivot)];
            // Clang contracts SPRAL's scalar SimdVec update on the local build.
            matrix[update_entry] = (-preserved).mul_add(multiplier, matrix[update_entry]);
        }
    }
}

fn app_update_two_by_two(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
) {
    let first_ld = &workspace[pivot * size..(pivot + 1) * size];
    let second_ld = &workspace[(pivot + 1) * size..(pivot + 2) * size];
    for col in (pivot + 2)..update_end {
        let first_preserved = first_ld[col];
        let second_preserved = second_ld[col];
        for row in col..update_end {
            let update_entry = dense_lower_offset(size, row, col);
            let first_multiplier = matrix[dense_lower_offset(size, row, pivot)];
            let second_multiplier = matrix[dense_lower_offset(size, row, pivot + 1)];
            // Match native block_ldlt's contracted 2x2 update RHS.
            matrix[update_entry] -=
                first_preserved.mul_add(first_multiplier, second_preserved * second_multiplier);
        }
    }
}

fn factor_one_by_one_common(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    stats: &mut PanelFactorStats,
    scratch: &mut [f64],
) -> Result<(FactorColumn, FactorBlockRecord), SsidsError> {
    let work = &mut scratch[pivot * size..(pivot + 1) * size];
    let diagonal_index = dense_lower_offset(size, pivot, pivot);
    let original_diagonal = matrix[diagonal_index];
    if !original_diagonal.is_finite() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail: "diagonal pivot became non-finite".into(),
        });
    }
    let diagonal = original_diagonal;
    let inverse_diagonal = 1.0 / diagonal;
    if !inverse_diagonal.is_finite() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail: "inverse diagonal pivot became non-finite".into(),
        });
    }
    matrix[diagonal_index] = diagonal;
    stats.max_residual = stats.max_residual.max((diagonal - original_diagonal).abs());
    matrix[diagonal_index] = 1.0;

    let mut entries = Vec::new();
    for row in (pivot + 1)..update_end {
        let entry_index = dense_lower_offset(size, row, pivot);
        let original = matrix[entry_index];
        work[row] = original;
        let value = original * inverse_diagonal;
        if !value.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: format!(
                    "subdiagonal entry ({}, {}) became non-finite",
                    rows[row], rows[pivot]
                ),
            });
        }
        matrix[entry_index] = value;
        entries.push((rows[row], value));
    }

    app_update_one_by_one(matrix, size, pivot, update_end, scratch);

    Ok((
        FactorColumn {
            global_column: rows[pivot],
            entries,
        },
        FactorBlockRecord {
            size: 1,
            values: [inverse_diagonal, 0.0, 0.0, 0.0],
        },
    ))
}

fn factor_two_by_two_common(
    rows: &[usize],
    matrix: &mut [f64],
    bounds: DenseUpdateBounds,
    pivot: usize,
    inverse: (f64, f64, f64),
    stats: &mut PanelFactorStats,
    scratch: &mut [f64],
) -> Result<([FactorColumn; 2], FactorBlockRecord), SsidsError> {
    let size = bounds.size;
    let update_end = bounds.update_end;
    let second_start = (pivot + 1) * size;
    let (first_prefix, second_suffix) = scratch.split_at_mut(second_start);
    let first_scratch = &mut first_prefix[pivot * size..second_start];
    let second_scratch = &mut second_suffix[..size];
    let (inv11, inv12, inv22) = inverse;
    stats.two_by_two_pivots += 1;
    matrix[dense_lower_offset(size, pivot, pivot)] = 1.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot)] = 0.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot + 1)] = 1.0;

    let mut first_entries = Vec::new();
    let mut second_entries = Vec::new();
    for row in (pivot + 2)..update_end {
        let b1 = matrix[dense_lower_offset(size, row, pivot)];
        let b2 = matrix[dense_lower_offset(size, row, pivot + 1)];
        first_scratch[row] = b1;
        second_scratch[row] = b2;
        let l1 = inv11.mul_add(b1, inv12 * b2);
        let l2 = inv12.mul_add(b1, inv22 * b2);
        if !l1.is_finite() || !l2.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: "two-by-two multipliers became non-finite".into(),
            });
        }
        matrix[dense_lower_offset(size, row, pivot)] = l1;
        matrix[dense_lower_offset(size, row, pivot + 1)] = l2;
        first_entries.push((rows[row], l1));
        second_entries.push((rows[row], l2));
    }

    app_update_two_by_two(matrix, size, pivot, update_end, scratch);

    Ok((
        [
            FactorColumn {
                global_column: rows[pivot],
                entries: first_entries,
            },
            FactorColumn {
                global_column: rows[pivot + 1],
                entries: second_entries,
            },
        ],
        FactorBlockRecord {
            size: 2,
            values: [inv11, inv12, f64::INFINITY, inv22],
        },
    ))
}

fn remove_zero_lower_entries_targeting_row(factor_columns: &mut [FactorColumn], row: usize) {
    let row_has_nonzero_entry = factor_columns
        .iter()
        .flat_map(|column| column.entries.iter())
        .any(|&(entry_row, value)| entry_row == row && value != 0.0);
    if row_has_nonzero_entry {
        return;
    }
    for column in factor_columns {
        column
            .entries
            .retain(|&(entry_row, value)| entry_row != row || value != 0.0);
    }
}

fn app_apply_block_pivots_to_trailing_rows(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
    block_records: &[FactorBlockRecord],
    small: f64,
) {
    if block_end >= size {
        return;
    }

    const OPENBLAS_DTRSM_UNROLL_N: usize = 4;
    for row in block_end..size {
        let mut group_start = block_start;
        while group_start < block_end {
            let group_end = (group_start + OPENBLAS_DTRSM_UNROLL_N).min(block_end);
            for col in group_start..group_end {
                let mut value = matrix[dense_lower_offset(size, row, col)];
                if group_start > block_start {
                    let mut dot = 0.0;
                    for prior in block_start..group_start {
                        let prior_value = matrix[dense_lower_offset(size, row, prior)];
                        let lower_value = matrix[dense_lower_offset(size, col, prior)];
                        dot = prior_value.mul_add(lower_value, dot);
                    }
                    value = dot.mul_add(-1.0, value);
                }
                matrix[dense_lower_offset(size, row, col)] = value;
            }

            for col in group_start..group_end {
                let value = matrix[dense_lower_offset(size, row, col)];
                for target_col in (col + 1)..group_end {
                    let target_entry = dense_lower_offset(size, row, target_col);
                    let lower_value = matrix[dense_lower_offset(size, target_col, col)];
                    matrix[target_entry] = (-value).mul_add(lower_value, matrix[target_entry]);
                }
            }
            group_start = group_end;
        }
    }

    let mut col = block_start;
    for block in block_records {
        if block.size == 1 {
            let d11 = block.values[0];
            for row in block_end..size {
                let entry = dense_lower_offset(size, row, col);
                let value = matrix[entry];
                matrix[entry] = if d11 == 0.0 {
                    if value.abs() < small {
                        0.0
                    } else {
                        f64::INFINITY * value
                    }
                } else {
                    value * d11
                };
            }
            col += 1;
        } else {
            let d11 = block.values[0];
            let d21 = block.values[1];
            let d22 = block.values[3];
            for row in block_end..size {
                let first_entry = dense_lower_offset(size, row, col);
                let second_entry = dense_lower_offset(size, row, col + 1);
                let a1 = matrix[first_entry];
                let a2 = matrix[second_entry];
                matrix[first_entry] = d11.mul_add(a1, d21 * a2);
                matrix[second_entry] = d21.mul_add(a1, d22 * a2);
            }
            col += 2;
        }
    }
}

fn app_first_failed_trailing_column(
    matrix: &[f64],
    size: usize,
    block_start: usize,
    block_end: usize,
    threshold_pivot_u: f64,
) -> usize {
    if block_end >= size || threshold_pivot_u <= 0.0 {
        return block_end;
    }
    let limit = 1.0 / threshold_pivot_u;
    for col in block_start..block_end {
        for row in block_end..size {
            if matrix[dense_lower_offset(size, row, col)].abs() > limit {
                return col;
            }
        }
    }
    block_end
}

fn app_prefix_ending_on_first_half_two_by_two(
    block_records: &[FactorBlockRecord],
    local_prefix: usize,
) -> bool {
    if local_prefix == 0 {
        return false;
    }
    let last_accepted = local_prefix - 1;
    let mut cursor = 0;
    for block in block_records {
        if block.size == 2 && cursor == last_accepted {
            return true;
        }
        cursor += block.size;
    }
    false
}

fn app_adjust_passed_prefix(block_records: &[FactorBlockRecord], local_prefix: usize) -> usize {
    if app_prefix_ending_on_first_half_two_by_two(block_records, local_prefix) {
        local_prefix - 1
    } else {
        local_prefix
    }
}

fn app_truncate_records_to_prefix(
    block_records: &[FactorBlockRecord],
    local_prefix: usize,
) -> Vec<FactorBlockRecord> {
    let mut accepted = Vec::new();
    let mut cursor = 0;
    for block in block_records {
        if cursor + block.size > local_prefix {
            break;
        }
        accepted.push(block.clone());
        cursor += block.size;
    }
    debug_assert_eq!(cursor, local_prefix);
    accepted
}

fn app_restore_trailing_from_block_backup(
    rows: &[usize],
    rows_before_block: &[usize],
    matrix: &mut [f64],
    matrix_before_block: &[f64],
    size: usize,
    trailing_start: usize,
) {
    if trailing_start >= size {
        return;
    }
    let mut old_positions = vec![
        usize::MAX;
        rows.iter()
            .chain(rows_before_block)
            .copied()
            .max()
            .unwrap_or(0)
            + 1
    ];
    for (position, &row) in rows_before_block.iter().enumerate() {
        old_positions[row] = position;
    }
    for row in trailing_start..size {
        let old_row = old_positions[rows[row]];
        for col in trailing_start..=row {
            let old_col = old_positions[rows[col]];
            matrix[dense_lower_offset(size, row, col)] =
                matrix_before_block[dense_lower_offset(size, old_row, old_col)];
        }
    }
}

fn app_original_one_by_one_diagonal(inverse_diagonal: f64) -> f64 {
    if inverse_diagonal == 0.0 {
        0.0
    } else {
        1.0 / inverse_diagonal
    }
}

fn app_apply_accepted_prefix_update(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
) {
    if accepted_end >= size {
        return;
    }
    if accepted_end + 1 == size {
        let row = accepted_end;
        let entry = dense_lower_offset(size, row, row);
        let mut pivot = block_start;
        for block in block_records {
            if block.size == 1 {
                let diagonal = app_original_one_by_one_diagonal(block.values[0]);
                let row_l = matrix[dense_lower_offset(size, row, pivot)];
                let row_ld = diagonal * row_l;
                matrix[entry] = (-row_l).mul_add(row_ld, matrix[entry]);
                pivot += 1;
            } else {
                let inv11 = block.values[0];
                let inv21 = block.values[1];
                let inv22 = block.values[3];
                let det = inv11.mul_add(inv22, -(inv21 * inv21));
                let d11 = inv11 / det;
                let d21 = inv21 / det;
                let d22 = inv22 / det;
                let row_l1 = matrix[dense_lower_offset(size, row, pivot)];
                let row_l2 = matrix[dense_lower_offset(size, row, pivot + 1)];
                let row_ld1 = d22.mul_add(row_l1, -(d21 * row_l2));
                let row_ld2 = (-d21).mul_add(row_l1, d11 * row_l2);
                matrix[entry] = (-row_l1).mul_add(row_ld1, matrix[entry]);
                matrix[entry] = (-row_l2).mul_add(row_ld2, matrix[entry]);
                pivot += 2;
            }
        }
        debug_assert_eq!(pivot, accepted_end);
        return;
    }
    for row in accepted_end..size {
        for col in accepted_end..=row {
            if app_target_block_uses_gemv_forward(size, accepted_end, col) {
                app_apply_accepted_prefix_update_entry_incremental(
                    matrix,
                    size,
                    block_start,
                    accepted_end,
                    block_records,
                    row,
                    col,
                );
                continue;
            }
            let mut update = 0.0;
            let mut pivot = block_start;
            for block in block_records {
                if block.size == 1 {
                    let diagonal = app_original_one_by_one_diagonal(block.values[0]);
                    let row_l = matrix[dense_lower_offset(size, row, pivot)];
                    let col_l = matrix[dense_lower_offset(size, col, pivot)];
                    let row_ld = diagonal * row_l;
                    update = row_ld.mul_add(col_l, update);
                    pivot += 1;
                } else {
                    let inv11 = block.values[0];
                    let inv21 = block.values[1];
                    let inv22 = block.values[3];
                    let det = inv11.mul_add(inv22, -(inv21 * inv21));
                    let d11 = inv11 / det;
                    let d21 = inv21 / det;
                    let d22 = inv22 / det;
                    let row_l1 = matrix[dense_lower_offset(size, row, pivot)];
                    let row_l2 = matrix[dense_lower_offset(size, row, pivot + 1)];
                    let col_l1 = matrix[dense_lower_offset(size, col, pivot)];
                    let col_l2 = matrix[dense_lower_offset(size, col, pivot + 1)];
                    let row_ld1 = d22.mul_add(row_l1, -(d21 * row_l2));
                    let row_ld2 = (-d21).mul_add(row_l1, d11 * row_l2);
                    update = row_ld1.mul_add(col_l1, update);
                    update = row_ld2.mul_add(col_l2, update);
                    pivot += 2;
                }
            }
            debug_assert_eq!(pivot, accepted_end);
            let entry = dense_lower_offset(size, row, col);
            matrix[entry] = update.mul_add(-1.0, matrix[entry]);
        }
    }
}

fn app_target_block_uses_gemv_forward(size: usize, accepted_end: usize, col: usize) -> bool {
    let col_block_start =
        accepted_end + ((col - accepted_end) / APP_INNER_BLOCK_SIZE) * APP_INNER_BLOCK_SIZE;
    let col_block_end = (col_block_start + APP_INNER_BLOCK_SIZE).min(size);
    col_block_end - col_block_start == 1
}

fn app_apply_accepted_prefix_update_entry_incremental(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
    row: usize,
    col: usize,
) {
    let entry = dense_lower_offset(size, row, col);
    let mut pivot = block_start;
    for block in block_records {
        if block.size == 1 {
            let diagonal = app_original_one_by_one_diagonal(block.values[0]);
            let row_l = matrix[dense_lower_offset(size, row, pivot)];
            let col_l = matrix[dense_lower_offset(size, col, pivot)];
            let row_ld = diagonal * row_l;
            matrix[entry] = (-col_l).mul_add(row_ld, matrix[entry]);
            pivot += 1;
        } else {
            let inv11 = block.values[0];
            let inv21 = block.values[1];
            let inv22 = block.values[3];
            let det = inv11.mul_add(inv22, -(inv21 * inv21));
            let d11 = inv11 / det;
            let d21 = inv21 / det;
            let d22 = inv22 / det;
            let row_l1 = matrix[dense_lower_offset(size, row, pivot)];
            let row_l2 = matrix[dense_lower_offset(size, row, pivot + 1)];
            let col_l1 = matrix[dense_lower_offset(size, col, pivot)];
            let col_l2 = matrix[dense_lower_offset(size, col, pivot + 1)];
            let row_ld1 = d22.mul_add(row_l1, -(d21 * row_l2));
            let row_ld2 = (-d21).mul_add(row_l1, d11 * row_l2);
            matrix[entry] = (-col_l1).mul_add(row_ld1, matrix[entry]);
            matrix[entry] = (-col_l2).mul_add(row_ld2, matrix[entry]);
            pivot += 2;
        }
    }
    debug_assert_eq!(pivot, accepted_end);
}

fn app_build_factor_columns_for_prefix(
    rows: &[usize],
    matrix: &[f64],
    size: usize,
    start: usize,
    end: usize,
) -> Vec<FactorColumn> {
    let mut columns = Vec::with_capacity(end.saturating_sub(start));
    for col in start..end {
        let mut entries = Vec::with_capacity(size.saturating_sub(col + 1));
        for row in (col + 1)..size {
            entries.push((rows[row], matrix[dense_lower_offset(size, row, col)]));
        }
        columns.push(FactorColumn {
            global_column: rows[col],
            entries,
        });
    }
    columns
}

fn tpp_factor_one_by_one(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    _stats: &mut PanelFactorStats,
    ld: &mut [f64],
) -> Result<(FactorColumn, FactorBlockRecord), SsidsError> {
    let work = &mut ld[..size];
    let diagonal_index = dense_lower_offset(size, pivot, pivot);
    let diagonal = matrix[diagonal_index];
    if !diagonal.is_finite() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail: "TPP diagonal pivot became non-finite".into(),
        });
    }
    let inverse_diagonal = 1.0 / diagonal;
    if !inverse_diagonal.is_finite() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail: "TPP inverse diagonal pivot became non-finite".into(),
        });
    }
    matrix[diagonal_index] = 1.0;

    let mut entries = Vec::new();
    for row in (pivot + 1)..size {
        let entry_index = dense_lower_offset(size, row, pivot);
        let original = matrix[entry_index];
        work[row] = original;
        let value = original * inverse_diagonal;
        if !value.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: format!(
                    "TPP subdiagonal entry ({}, {}) became non-finite",
                    rows[row], rows[pivot]
                ),
            });
        }
        matrix[entry_index] = value;
        entries.push((rows[row], value));
    }

    root_tpp_rank1_update(matrix, size, pivot + 1, pivot, work);

    Ok((
        FactorColumn {
            global_column: rows[pivot],
            entries,
        },
        FactorBlockRecord {
            size: 1,
            values: [inverse_diagonal, 0.0, 0.0, 0.0],
        },
    ))
}

fn tpp_factor_two_by_two(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    inverse: (f64, f64, f64),
    stats: &mut PanelFactorStats,
    ld: &mut [f64],
) -> Result<([FactorColumn; 2], FactorBlockRecord), SsidsError> {
    let (first_scratch, second_scratch) = ld.split_at_mut(size);
    let (inv11, inv12, inv22) = inverse;
    stats.two_by_two_pivots += 1;
    matrix[dense_lower_offset(size, pivot, pivot)] = 1.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot)] = 0.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot + 1)] = 1.0;

    let mut first_entries = Vec::new();
    let mut second_entries = Vec::new();
    for row in (pivot + 2)..size {
        let b1 = matrix[dense_lower_offset(size, row, pivot)];
        let b2 = matrix[dense_lower_offset(size, row, pivot + 1)];
        first_scratch[row] = b1;
        second_scratch[row] = b2;
        let l1 = inv11.mul_add(b1, inv12 * b2);
        let l2 = inv12.mul_add(b1, inv22 * b2);
        if !l1.is_finite() || !l2.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: "TPP two-by-two multipliers became non-finite".into(),
            });
        }
        matrix[dense_lower_offset(size, row, pivot)] = l1;
        matrix[dense_lower_offset(size, row, pivot + 1)] = l2;
        first_entries.push((rows[row], l1));
        second_entries.push((rows[row], l2));
    }

    root_tpp_rank2_update(
        matrix,
        size,
        pivot + 2,
        pivot,
        pivot + 1,
        first_scratch,
        second_scratch,
    );

    Ok((
        [
            FactorColumn {
                global_column: rows[pivot],
                entries: first_entries,
            },
            FactorColumn {
                global_column: rows[pivot + 1],
                entries: second_entries,
            },
        ],
        FactorBlockRecord {
            size: 2,
            values: [inv11, inv12, f64::INFINITY, inv22],
        },
    ))
}

fn root_tpp_rank1_update(
    matrix: &mut [f64],
    size: usize,
    start: usize,
    multiplier_column: usize,
    preserved_column: &[f64],
) {
    // Native SPRAL routes a 1x1 trailing update through scalar dgemm, but
    // wider trailing blocks round like OpenBLAS' block kernel.
    let use_scalar_fma = size.saturating_sub(start) == 1;
    for (col, &preserved) in preserved_column.iter().enumerate().take(size).skip(start) {
        for row in col..size {
            let l_row = matrix[dense_lower_offset(size, row, multiplier_column)];
            let update_entry = dense_lower_offset(size, row, col);
            if use_scalar_fma {
                matrix[update_entry] = (-l_row).mul_add(preserved, matrix[update_entry]);
            } else {
                matrix[update_entry] -= l_row * preserved;
            }
        }
    }
}

fn root_tpp_rank2_update(
    matrix: &mut [f64],
    size: usize,
    start: usize,
    first_multiplier_column: usize,
    second_multiplier_column: usize,
    first_preserved_column: &[f64],
    second_preserved_column: &[f64],
) {
    let use_scalar_fma = size.saturating_sub(start) == 1;
    for (col, (&first_preserved, &second_preserved)) in first_preserved_column
        .iter()
        .zip(second_preserved_column.iter())
        .enumerate()
        .take(size)
        .skip(start)
    {
        for row in col..size {
            let l1_row = matrix[dense_lower_offset(size, row, first_multiplier_column)];
            let l2_row = matrix[dense_lower_offset(size, row, second_multiplier_column)];
            let update_entry = dense_lower_offset(size, row, col);
            if use_scalar_fma {
                let updated = (-l1_row).mul_add(first_preserved, matrix[update_entry]);
                matrix[update_entry] = (-l2_row).mul_add(second_preserved, updated);
            } else {
                let current = matrix[update_entry];
                // Native SPRAL calls dgemm(..., k=2, alpha=-1, beta=1).
                // OpenBLAS forms this two-term dot with the k=1 product first,
                // then applies beta*C.
                matrix[update_entry] =
                    current - l2_row.mul_add(second_preserved, l1_row * first_preserved);
            }
        }
    }
}

fn factorize_dense_tpp_tail_in_place(
    rows: &mut Vec<usize>,
    dense: &mut [f64],
    start_pivot: usize,
    candidate_len: usize,
    options: NumericFactorOptions,
    require_full_elimination: bool,
    ld: &mut [f64],
) -> Result<DenseFrontFactorization, SsidsError> {
    let size = rows.len();
    let mut stats = PanelFactorStats::default();
    let mut factor_order = Vec::with_capacity(size);
    let mut factor_columns = Vec::with_capacity(size);
    let mut block_records = Vec::new();
    let mut pivot = start_pivot;
    let active_candidate_end = (start_pivot + candidate_len).min(size);

    while pivot < active_candidate_end {
        if dense_column_small(dense, size, pivot, pivot, options.small_pivot_tolerance) {
            if !options.action_on_zero_pivot {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: rows[pivot],
                    detail: "TPP encountered a zero pivot with action disabled".into(),
                });
            }
            remove_zero_lower_entries_targeting_row(&mut factor_columns, rows[pivot]);
            zero_dense_column(dense, size, pivot);
            let column = FactorColumn {
                global_column: rows[pivot],
                entries: Vec::new(),
            };
            let block = FactorBlockRecord {
                size: 1,
                values: [0.0, 0.0, 0.0, 0.0],
            };
            factor_order.push(rows[pivot]);
            factor_columns.push(column);
            block_records.push(block);
            pivot += 1;
            continue;
        }

        let mut advanced = false;
        for candidate in (pivot + 1)..active_candidate_end {
            if dense_column_small(dense, size, candidate, pivot, options.small_pivot_tolerance) {
                if !options.action_on_zero_pivot {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: rows[pivot],
                        detail: "TPP encountered a zero pivot with action disabled".into(),
                    });
                }
                if candidate != pivot {
                    dense_symmetric_swap(dense, size, candidate, pivot);
                    rows.swap(candidate, pivot);
                }
                remove_zero_lower_entries_targeting_row(&mut factor_columns, rows[pivot]);
                zero_dense_column(dense, size, pivot);
                let column = FactorColumn {
                    global_column: rows[pivot],
                    entries: Vec::new(),
                };
                let block = FactorBlockRecord {
                    size: 1,
                    values: [0.0, 0.0, 0.0, 0.0],
                };
                factor_order.push(rows[pivot]);
                factor_columns.push(column);
                block_records.push(block);
                pivot += 1;
                advanced = true;
                break;
            }

            let Some(first) = dense_find_row_abs_max_in_column(dense, size, candidate, pivot)
            else {
                continue;
            };
            let mut second = candidate;
            let a11 = dense[dense_lower_offset(size, first, first)];
            let a22 = dense[dense_lower_offset(size, second, second)];
            let a21 = dense[dense_lower_offset(size, second, first)];
            let maxt = dense_find_rc_abs_max_exclude(dense, size, first, pivot, Some(second));
            let mut maxp = dense_find_rc_abs_max_exclude(dense, size, second, pivot, Some(first));

            if let Some(inverse) = tpp_test_two_by_two(a11, a21, a22, maxt, maxp, options) {
                if first != pivot {
                    dense_symmetric_swap(dense, size, first, pivot);
                    rows.swap(first, pivot);
                    if second == pivot {
                        second = first;
                    }
                }
                if second != pivot + 1 {
                    dense_symmetric_swap(dense, size, second, pivot + 1);
                    rows.swap(second, pivot + 1);
                }
                let (columns, block) =
                    tpp_factor_two_by_two(rows, dense, size, pivot, inverse, &mut stats, ld)?;
                factor_order.push(rows[pivot]);
                factor_order.push(rows[pivot + 1]);
                let [first_column, second_column] = columns;
                factor_columns.push(first_column);
                factor_columns.push(second_column);
                block_records.push(block);
                pivot += 2;
                advanced = true;
                break;
            }

            maxp = maxp.max(a21.abs());
            if a22.abs() >= options.threshold_pivot_u * maxp {
                if candidate != pivot {
                    dense_symmetric_swap(dense, size, candidate, pivot);
                    rows.swap(candidate, pivot);
                }
                let (column, block) =
                    tpp_factor_one_by_one(rows, dense, size, pivot, &mut stats, ld)?;
                factor_order.push(rows[pivot]);
                factor_columns.push(column);
                block_records.push(block);
                pivot += 1;
                advanced = true;
                break;
            }
        }

        if advanced {
            continue;
        }

        let current_diag = dense[dense_lower_offset(size, pivot, pivot)];
        let current_offdiag_max = dense_find_rc_abs_max_exclude(dense, size, pivot, pivot, None);
        if current_diag.abs() >= options.threshold_pivot_u * current_offdiag_max {
            let (column, block) = tpp_factor_one_by_one(rows, dense, size, pivot, &mut stats, ld)?;
            factor_order.push(rows[pivot]);
            factor_columns.push(column);
            block_records.push(block);
            pivot += 1;
            continue;
        }

        if require_full_elimination {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: "root TPP completion could not find an acceptable pivot".into(),
            });
        }
        break;
    }

    let remaining_rows = rows.split_off(pivot);
    let remaining_size = remaining_rows.len();
    let delayed_count = active_candidate_end
        .saturating_sub(pivot)
        .min(remaining_size);
    stats.delayed_pivots += delayed_count;
    let mut contribution_dense = vec![0.0; packed_lower_len(remaining_size)];
    for row in 0..remaining_size {
        for col in 0..=row {
            let value = dense[dense_lower_offset(size, pivot + row, pivot + col)];
            contribution_dense[packed_lower_offset(remaining_size, row, col)] = value;
        }
    }

    Ok(DenseFrontFactorization {
        factor_order,
        factor_columns,
        block_records,
        solve_panels: Vec::new(),
        contribution: ContributionBlock {
            row_ids: remaining_rows,
            delayed_count,
            dense: contribution_dense,
        },
        stats,
    })
}

fn build_factor_solve_panel_record(
    factor_order: &[usize],
    factor_columns: &[FactorColumn],
    trailing_rows: &[usize],
) -> Result<Option<FactorSolvePanelRecord>, SsidsError> {
    let eliminated_len = factor_order.len();
    if eliminated_len == 0 {
        return Ok(None);
    }
    if factor_columns.len() != eliminated_len {
        return Err(SsidsError::NumericalBreakdown {
            pivot: eliminated_len,
            detail: format!(
                "solve panel has {} factor columns for {eliminated_len} eliminated rows",
                factor_columns.len()
            ),
        });
    }

    let mut row_ids = Vec::with_capacity(eliminated_len + trailing_rows.len());
    row_ids.extend_from_slice(factor_order);
    row_ids.extend_from_slice(trailing_rows);
    let max_row = row_ids
        .iter()
        .copied()
        .chain(
            factor_columns
                .iter()
                .flat_map(|column| column.entries.iter().map(|&(row, _)| row)),
        )
        .max()
        .unwrap_or(0);
    let mut local_positions = vec![usize::MAX; max_row + 1];
    for (position, &row) in row_ids.iter().enumerate() {
        if local_positions[row] != usize::MAX {
            return Err(SsidsError::NumericalBreakdown {
                pivot: row,
                detail: "solve panel contains a duplicate local row".into(),
            });
        }
        local_positions[row] = position;
    }

    let local_size = row_ids.len();
    let mut values = vec![0.0; local_size * eliminated_len];
    for (local_col, column) in factor_columns.iter().enumerate() {
        if column.global_column != factor_order[local_col] {
            return Err(SsidsError::NumericalBreakdown {
                pivot: local_col,
                detail: "solve panel column order drifted away from factor order".into(),
            });
        }
        values[local_col * local_size + local_col] = 1.0;
        for &(row, value) in &column.entries {
            let local_row = local_positions.get(row).copied().unwrap_or(usize::MAX);
            if local_row == usize::MAX || local_row <= local_col {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: local_col,
                    detail: "solve panel factor column referenced an invalid local row".into(),
                });
            }
            values[local_col * local_size + local_row] = value;
        }
    }

    Ok(Some(FactorSolvePanelRecord {
        eliminated_len,
        row_ids,
        values,
    }))
}

fn zero_dense_column(matrix: &mut [f64], size: usize, pivot: usize) {
    for row in pivot..size {
        matrix[dense_lower_offset(size, row, pivot)] = 0.0;
    }
}

fn zero_dense_column_until(matrix: &mut [f64], size: usize, pivot: usize, end: usize) {
    for row in pivot..end {
        matrix[dense_lower_offset(size, row, pivot)] = 0.0;
    }
}

fn factorize_root_dense_tpp(
    mut rows: Vec<usize>,
    mut dense: Vec<f64>,
    options: NumericFactorOptions,
) -> Result<DenseFrontFactorization, SsidsError> {
    let size = rows.len();
    let mut ld = vec![0.0; size.saturating_mul(2).max(1)];
    let mut factorization =
        factorize_dense_tpp_tail_in_place(&mut rows, &mut dense, 0, size, options, true, &mut ld)?;
    if let Some(&pivot) = factorization.contribution.row_ids.first() {
        return Err(SsidsError::NumericalBreakdown {
            pivot,
            detail: format!(
                "root TPP completion retained {} delayed pivots",
                factorization.contribution.delayed_count
            ),
        });
    }
    if let Some(panel) = build_factor_solve_panel_record(
        &factorization.factor_order,
        &factorization.factor_columns,
        &factorization.contribution.row_ids,
    )? {
        factorization.solve_panels.push(panel);
    }
    Ok(factorization)
}

fn factorize_dense_front(
    mut rows: Vec<usize>,
    candidate_len: usize,
    mut dense: Vec<f64>,
    options: NumericFactorOptions,
) -> Result<DenseFrontFactorization, SsidsError> {
    let size = rows.len();
    let active_candidate_end = candidate_len.min(size);
    if matches!(options.pivot_method, PivotMethod::ThresholdPartial)
        || active_candidate_end < APP_INNER_BLOCK_SIZE
    {
        let mut tpp_ld = vec![0.0; size.saturating_mul(2).max(1)];
        let mut factorization = factorize_dense_tpp_tail_in_place(
            &mut rows,
            &mut dense,
            0,
            active_candidate_end,
            options,
            false,
            &mut tpp_ld,
        )?;
        if let Some(panel) = build_factor_solve_panel_record(
            &factorization.factor_order,
            &factorization.factor_columns,
            &factorization.contribution.row_ids,
        )? {
            factorization.solve_panels.push(panel);
        }
        return Ok(factorization);
    }

    let mut stats = PanelFactorStats::default();
    let mut factor_order = Vec::new();
    let mut factor_columns = Vec::new();
    let mut block_records = Vec::new();
    let mut scratch = vec![0.0; size.saturating_mul(size).max(1)];
    let mut pivot = 0;

    while active_candidate_end - pivot >= APP_INNER_BLOCK_SIZE {
        let block_start = pivot;
        let block_end = pivot + APP_INNER_BLOCK_SIZE;
        let rows_before_block = rows.clone();
        let dense_before_block = dense.clone();
        let mut local_stats = PanelFactorStats::default();
        let mut local_blocks = Vec::new();
        let mut block_pivot = block_start;

        while block_pivot < block_end {
            let Some((best_abs, best_row, best_col)) =
                dense_find_maxloc(&dense, size, block_pivot, block_end)
            else {
                break;
            };
            if best_abs < options.small_pivot_tolerance {
                if !options.action_on_zero_pivot {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: rows[block_pivot],
                        detail: "APP encountered a zero pivot with action disabled".into(),
                    });
                }
                while block_pivot < block_end {
                    zero_dense_column_until(&mut dense, size, block_pivot, block_end);
                    reset_ldwork_column_tail(&mut scratch, size, block_pivot, block_pivot);
                    local_blocks.push(FactorBlockRecord {
                        size: 1,
                        values: [0.0, 0.0, 0.0, 0.0],
                    });
                    block_pivot += 1;
                }
                break;
            }

            if best_row == best_col {
                if best_col != block_pivot {
                    dense_symmetric_swap_with_workspace(
                        &mut dense,
                        size,
                        best_col,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(best_col, block_pivot);
                }
                let (_, block) = factor_one_by_one_common(
                    &rows,
                    &mut dense,
                    size,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    &mut scratch,
                )?;
                local_blocks.push(block);
                block_pivot += 1;
                continue;
            }

            let first = best_col;
            let mut second = best_row;
            let a11 = dense[dense_lower_offset(size, first, first)];
            let a22 = dense[dense_lower_offset(size, second, second)];
            let a21 = dense[dense_lower_offset(size, second, first)];

            let mut two_by_two_inverse = None;
            let mut one_by_one_index = None;
            if let Some(inverse) =
                app_two_by_two_inverse(a11, a21, a22, options.small_pivot_tolerance)
            {
                two_by_two_inverse = Some(inverse);
            } else if a11.abs() > a22.abs() {
                if (a11 / a21).abs() >= options.threshold_pivot_u {
                    one_by_one_index = Some(first);
                }
            } else if (a22 / a21).abs() >= options.threshold_pivot_u {
                one_by_one_index = Some(second);
            }

            if let Some(index) = one_by_one_index {
                if index != block_pivot {
                    dense_symmetric_swap_with_workspace(
                        &mut dense,
                        size,
                        index,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(index, block_pivot);
                }
                let (_, block) = factor_one_by_one_common(
                    &rows,
                    &mut dense,
                    size,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    &mut scratch,
                )?;
                local_blocks.push(block);
                block_pivot += 1;
                continue;
            }

            if let Some(inverse) = two_by_two_inverse {
                if first != block_pivot {
                    dense_symmetric_swap_with_workspace(
                        &mut dense,
                        size,
                        first,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(first, block_pivot);
                    if second == block_pivot {
                        second = first;
                    }
                }
                if second != block_pivot + 1 {
                    dense_symmetric_swap_with_workspace(
                        &mut dense,
                        size,
                        second,
                        block_pivot + 1,
                        &mut scratch,
                    );
                    rows.swap(second, block_pivot + 1);
                }
                let (_, block) = factor_two_by_two_common(
                    &rows,
                    &mut dense,
                    DenseUpdateBounds {
                        size,
                        update_end: block_end,
                    },
                    block_pivot,
                    inverse,
                    &mut local_stats,
                    &mut scratch,
                )?;
                local_blocks.push(block);
                block_pivot += 2;
                continue;
            }

            break;
        }

        app_apply_block_pivots_to_trailing_rows(
            &mut dense,
            size,
            block_start,
            block_end,
            &local_blocks,
            options.small_pivot_tolerance,
        );
        let first_failed = app_first_failed_trailing_column(
            &dense,
            size,
            block_start,
            block_end,
            options.threshold_pivot_u,
        );
        let local_passed = app_adjust_passed_prefix(&local_blocks, first_failed - block_start);
        let accepted_end = block_start + local_passed;
        let accepted_blocks = app_truncate_records_to_prefix(&local_blocks, local_passed);

        app_restore_trailing_from_block_backup(
            &rows,
            &rows_before_block,
            &mut dense,
            &dense_before_block,
            size,
            accepted_end,
        );
        app_apply_accepted_prefix_update(
            &mut dense,
            size,
            block_start,
            accepted_end,
            &accepted_blocks,
        );

        factor_order.extend(rows[block_start..accepted_end].iter().copied());
        factor_columns.extend(app_build_factor_columns_for_prefix(
            &rows,
            &dense,
            size,
            block_start,
            accepted_end,
        ));
        stats.two_by_two_pivots += accepted_blocks
            .iter()
            .filter(|block| block.size == 2)
            .count();
        stats.max_residual = stats.max_residual.max(local_stats.max_residual);
        block_records.extend(accepted_blocks);
        pivot = accepted_end;

        if pivot < block_end {
            break;
        }
    }

    let remaining_rows = rows.split_off(pivot);
    let remaining_size = remaining_rows.len();
    let delayed_count = candidate_len.saturating_sub(pivot).min(remaining_size);
    if delayed_count > 0 {
        rows.extend(remaining_rows);
        let mut tpp_ld = vec![0.0; rows.len().saturating_mul(2).max(1)];
        let tpp_tail = factorize_dense_tpp_tail_in_place(
            &mut rows,
            &mut dense,
            pivot,
            delayed_count,
            options,
            false,
            &mut tpp_ld,
        )?;
        factor_order.extend(tpp_tail.factor_order);
        factor_columns.extend(tpp_tail.factor_columns);
        block_records.extend(tpp_tail.block_records);
        aggregate_panel_stats(&mut stats, tpp_tail.stats);
        let contribution = tpp_tail.contribution;
        let mut solve_panels = Vec::new();
        if let Some(panel) =
            build_factor_solve_panel_record(&factor_order, &factor_columns, &contribution.row_ids)?
        {
            solve_panels.push(panel);
        }
        return Ok(DenseFrontFactorization {
            factor_order,
            factor_columns,
            block_records,
            solve_panels,
            contribution,
            stats,
        });
    }

    let mut contribution_dense = vec![0.0; packed_lower_len(remaining_size)];
    stats.delayed_pivots += delayed_count;
    for row in 0..remaining_size {
        for col in 0..=row {
            let value = dense[dense_lower_offset(size, pivot + row, pivot + col)];
            contribution_dense[packed_lower_offset(remaining_size, row, col)] = value;
        }
    }

    let contribution = ContributionBlock {
        row_ids: remaining_rows,
        delayed_count,
        dense: contribution_dense,
    };
    let mut solve_panels = Vec::new();
    if let Some(panel) =
        build_factor_solve_panel_record(&factor_order, &factor_columns, &contribution.row_ids)?
    {
        solve_panels.push(panel);
    }

    Ok(DenseFrontFactorization {
        factor_order,
        factor_columns,
        block_records,
        solve_panels,
        contribution,
        stats,
    })
}

fn factor_front_recursive(
    front_id: usize,
    tree: &SymbolicFrontTree,
    matrix: &PermutedLowerMatrix<'_>,
    options: NumericFactorOptions,
    progress: Option<&FactorizationProgressShared>,
) -> Result<FrontFactorizationResult, SsidsError> {
    let front = &tree.fronts[front_id];
    let child_results =
        if front.children.len() >= 2 && front.width() + front.interface_rows.len() >= 32 {
            let raw = front
                .children
                .par_iter()
                .map(|&child| factor_front_recursive(child, tree, matrix, options, progress))
                .collect::<Vec<_>>();
            let mut collected = Vec::with_capacity(raw.len());
            for result in raw {
                collected.push(result?);
            }
            collected
        } else {
            let mut collected = Vec::with_capacity(front.children.len());
            for &child in &front.children {
                collected.push(factor_front_recursive(
                    child, tree, matrix, options, progress,
                )?);
            }
            collected
        };

    let mut factor_order = Vec::new();
    let mut factor_columns = Vec::new();
    let mut block_records = Vec::new();
    let mut solve_panels = Vec::new();
    let mut child_contributions = Vec::with_capacity(child_results.len());
    let mut stats = PanelFactorStats::default();
    let mut max_front_size = 0;
    let mut contribution_storage_bytes = 0;

    for child in child_results {
        factor_order.extend(child.factor_order);
        factor_columns.extend(child.factor_columns);
        block_records.extend(child.block_records);
        solve_panels.extend(child.solve_panels);
        child_contributions.push(child.contribution);
        aggregate_panel_stats(&mut stats, child.stats);
        max_front_size = max_front_size.max(child.max_front_size);
        contribution_storage_bytes += child.contribution_storage_bytes;
    }

    let mut row_state = vec![0_u8; matrix.dimension];
    let mut candidate_rows = Vec::with_capacity(front.columns.len());
    for &row in &front.columns {
        if row_state[row] == 0 {
            row_state[row] = 1;
            candidate_rows.push(row);
        }
    }
    for contribution in &child_contributions {
        for &row in contribution.row_ids.iter().take(contribution.delayed_count) {
            if row_state[row] == 0 {
                row_state[row] = 1;
                candidate_rows.push(row);
            }
        }
    }
    let mut interface_rows = Vec::with_capacity(front.interface_rows.len());
    for &row in &front.interface_rows {
        if row_state[row] == 0 {
            row_state[row] = 2;
            interface_rows.push(row);
        }
    }
    for contribution in &child_contributions {
        for &row in &contribution.row_ids {
            if row_state[row] == 0 {
                row_state[row] = 2;
                interface_rows.push(row);
            }
        }
    }
    interface_rows.sort_unstable();
    let mut local_rows = candidate_rows;
    local_rows.extend(interface_rows);
    let local_size = local_rows.len();
    max_front_size = max_front_size.max(local_size);
    let mut local_dense = vec![0.0; local_size * local_size];
    let mut local_positions = vec![usize::MAX; matrix.dimension];
    for (position, &row) in local_rows.iter().enumerate() {
        local_positions[row] = position;
    }
    for &column in &front.columns {
        let local_column = local_positions[column];
        for entry in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
            let row = matrix.row_indices[entry];
            let local_row = local_positions[row];
            if local_row == usize::MAX {
                continue;
            }
            let value = matrix.values[entry];
            let offset = dense_lower_offset(local_size, local_row, local_column);
            local_dense[offset] += value;
        }
    }
    for contribution in &child_contributions {
        let size = contribution.row_ids.len();
        for row in 0..size {
            let local_row = local_positions[contribution.row_ids[row]];
            for col in 0..=row {
                let local_col = local_positions[contribution.row_ids[col]];
                let value = contribution.dense[packed_lower_offset(size, row, col)];
                let offset = dense_lower_offset(local_size, local_row, local_col);
                local_dense[offset] += value;
            }
        }
    }

    let local = factorize_dense_front(local_rows, front.width(), local_dense, options)?;
    if let Some(progress) = progress {
        progress.completed_fronts.fetch_add(1, Ordering::Relaxed);
        progress
            .completed_pivots
            .fetch_add(local.factor_order.len(), Ordering::Relaxed);
        progress.completed_weight.fetch_add(
            front_work_weight(front.width(), front.interface_rows.len()),
            Ordering::Relaxed,
        );
        if front.parent.is_none() {
            progress.completed_roots.fetch_add(1, Ordering::Relaxed);
        }
    }
    factor_order.extend(local.factor_order);
    factor_columns.extend(local.factor_columns);
    block_records.extend(local.block_records);
    solve_panels.extend(local.solve_panels);
    aggregate_panel_stats(&mut stats, local.stats);
    contribution_storage_bytes += local.contribution.dense.len() * std::mem::size_of::<f64>();

    Ok(FrontFactorizationResult {
        factor_order,
        factor_columns,
        block_records,
        solve_panels,
        contribution: local.contribution,
        stats,
        max_front_size,
        contribution_storage_bytes,
    })
}

fn multifrontal_factorize_with_tree(
    matrix: SymmetricCscMatrix<'_>,
    permutation: &Permutation,
    tree: &SymbolicFrontTree,
    options: NumericFactorOptions,
    buffers: NumericFactorBuffers<'_>,
) -> Result<MultifrontalFactorizationOutcome, SsidsError> {
    let dimension = matrix.dimension();
    if buffers.permuted_matrix_source_positions.len() != matrix.row_indices().len()
        || buffers.permuted_matrix_col_ptrs.len() != dimension + 1
    {
        build_permuted_lower_csc_pattern(
            matrix,
            permutation,
            buffers.permuted_matrix_col_ptrs,
            buffers.permuted_matrix_row_indices,
            buffers.permuted_matrix_source_positions,
        )?;
    }
    fill_permuted_lower_csc_values(
        matrix,
        buffers.permuted_matrix_source_positions,
        buffers.permuted_matrix_values,
    )?;
    let permuted_matrix = PermutedLowerMatrix {
        dimension,
        col_ptrs: buffers.permuted_matrix_col_ptrs,
        row_indices: buffers.permuted_matrix_row_indices,
        values: buffers.permuted_matrix_values,
    };
    let progress = Arc::new(FactorizationProgressShared {
        total_fronts: tree.fronts.len(),
        total_pivots: dimension,
        total_weight: tree
            .fronts
            .iter()
            .map(|front| front_work_weight(front.width(), front.interface_rows.len()))
            .sum(),
        total_roots: tree.roots.len(),
        total_root_delayed_blocks: AtomicUsize::new(0),
        completed_fronts: AtomicUsize::new(0),
        completed_pivots: AtomicUsize::new(0),
        completed_weight: AtomicU64::new(0),
        completed_roots: AtomicUsize::new(0),
        completed_root_delayed_blocks: AtomicUsize::new(0),
        current_root_delayed_block: AtomicUsize::new(0),
        current_root_delayed_block_size: AtomicUsize::new(0),
        root_delayed_stage: AtomicUsize::new(RootDelayedBlockStage::Idle as usize),
    });
    let _progress_guard = install_factorization_progress(Arc::clone(&progress));
    let root_results = if tree.roots.len() >= 2 && dimension >= 64 {
        let raw = tree
            .roots
            .par_iter()
            .map(|&root| {
                factor_front_recursive(root, tree, &permuted_matrix, options, Some(&progress))
            })
            .collect::<Vec<_>>();
        let mut collected = Vec::with_capacity(raw.len());
        for result in raw {
            collected.push(result?);
        }
        collected
    } else {
        let mut collected = Vec::with_capacity(tree.roots.len());
        for &root in &tree.roots {
            collected.push(factor_front_recursive(
                root,
                tree,
                &permuted_matrix,
                options,
                Some(&progress),
            )?);
        }
        collected
    };

    let mut factor_order = Vec::with_capacity(dimension);
    let mut factor_columns = Vec::with_capacity(dimension);
    let mut block_records = Vec::new();
    let mut solve_panel_records = Vec::new();
    let mut stats = PanelFactorStats::default();
    let mut pending_root_contributions = Vec::new();
    for result in root_results {
        factor_order.extend(result.factor_order);
        factor_columns.extend(result.factor_columns);
        block_records.extend(result.block_records);
        solve_panel_records.extend(result.solve_panels);
        pending_root_contributions.push(result.contribution);
        aggregate_panel_stats(&mut stats, result.stats);
    }

    let delayed_block_total = pending_root_contributions
        .iter()
        .filter(|contribution| !contribution.row_ids.is_empty())
        .count();
    progress
        .total_root_delayed_blocks
        .store(delayed_block_total, Ordering::Relaxed);
    let mut delayed_block_index = 0;
    for contribution in pending_root_contributions {
        let ContributionBlock {
            row_ids,
            delayed_count: _,
            dense,
        } = contribution;
        if row_ids.is_empty() {
            continue;
        }
        delayed_block_index += 1;
        let size = row_ids.len();
        progress.begin_root_delayed_block(delayed_block_index, size);
        progress.set_root_delayed_stage(RootDelayedBlockStage::Factoring);
        let delayed_local = factorize_root_dense_tpp(
            row_ids,
            unpack_packed_lower_to_dense_square(size, &dense),
            options,
        )?;
        let fully_eliminated = size - delayed_local.contribution.row_ids.len();
        factor_order.extend(delayed_local.factor_order);
        factor_columns.extend(delayed_local.factor_columns);
        block_records.extend(delayed_local.block_records);
        solve_panel_records.extend(delayed_local.solve_panels);
        aggregate_panel_stats(&mut stats, delayed_local.stats);
        progress
            .completed_pivots
            .fetch_add(fully_eliminated, Ordering::Relaxed);
        progress.finish_root_delayed_block();
    }

    if factor_order.len() != dimension {
        return Err(SsidsError::NumericalBreakdown {
            pivot: factor_order.len(),
            detail: format!(
                "multifrontal factorization eliminated {} columns for a {dimension}x{dimension} matrix",
                factor_order.len()
            ),
        });
    }
    if factor_columns.len() != dimension {
        return Err(SsidsError::NumericalBreakdown {
            pivot: factor_columns.len(),
            detail: format!(
                "multifrontal factorization emitted {} factor columns for a {dimension}x{dimension} matrix",
                factor_columns.len()
            ),
        });
    }

    buffers.factor_order.clear();
    buffers.factor_order.extend(factor_order);

    buffers.factor_inverse.clear();
    buffers.factor_inverse.resize(dimension, usize::MAX);
    for (position, &ordered_index) in buffers.factor_order.iter().enumerate() {
        buffers.factor_inverse[ordered_index] = position;
    }
    buffers.lower_col_ptrs.clear();
    buffers.lower_col_ptrs.reserve(dimension + 1);
    buffers.lower_col_ptrs.push(0);
    buffers.lower_row_indices.clear();
    buffers.lower_values.clear();
    for (column_position, column) in factor_columns.iter().enumerate() {
        if column.global_column != buffers.factor_order[column_position] {
            return Err(SsidsError::NumericalBreakdown {
                pivot: column_position,
                detail: "factor column order drifted away from factor elimination order".into(),
            });
        }
        for &(row, value) in &column.entries {
            let row_position = buffers.factor_inverse[row];
            if row_position == usize::MAX || row_position <= column_position {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: column_position,
                    detail: "factor column referenced an invalid trailing row".into(),
                });
            }
            buffers.lower_row_indices.push(row_position);
            buffers.lower_values.push(value);
        }
        buffers.lower_col_ptrs.push(buffers.lower_row_indices.len());
    }

    buffers.solve_panels.clear();
    for record in solve_panel_records {
        let mut row_positions = Vec::with_capacity(record.row_ids.len());
        for row_id in record.row_ids {
            let row_position = buffers.factor_inverse[row_id];
            if row_position == usize::MAX {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: row_id,
                    detail: "solve panel referenced a row outside the factor order".into(),
                });
            }
            row_positions.push(row_position);
        }
        buffers.solve_panels.push(SolvePanel {
            eliminated_len: record.eliminated_len,
            row_positions,
            values: record.values,
        });
    }

    buffers.diagonal_blocks.clear();
    buffers.diagonal_values.clear();
    for block in block_records {
        buffers
            .diagonal_blocks
            .push(DiagonalBlock { size: block.size });
        buffers.diagonal_values.extend(block.values);
    }

    let stored_nnz = dimension
        + factor_columns
            .iter()
            .map(|column| column.entries.len())
            .sum::<usize>()
        + buffers
            .diagonal_blocks
            .iter()
            .map(|block| block.size)
            .sum::<usize>();
    let factor_bytes = std::mem::size_of::<f64>()
        * (buffers.lower_values.len()
            + buffers.diagonal_values.len()
            + buffers
                .solve_panels
                .iter()
                .map(|panel| panel.values.len())
                .sum::<usize>())
        + std::mem::size_of::<usize>()
            * (buffers.factor_order.len()
                + buffers.factor_inverse.len()
                + buffers.lower_col_ptrs.len()
                + buffers.lower_row_indices.len()
                + buffers
                    .solve_panels
                    .iter()
                    .map(|panel| panel.row_positions.len())
                    .sum::<usize>()
                + tree
                    .fronts
                    .iter()
                    .map(|front| 4 + front.interface_rows.len() + front.children.len())
                    .sum::<usize>());

    Ok(MultifrontalFactorizationOutcome {
        pivot_stats: PivotStats {
            two_by_two_pivots: stats.two_by_two_pivots,
            delayed_pivots: stats.delayed_pivots,
        },
        factorization_residual_max_abs: stats.max_residual,
        stored_nnz,
        factor_bytes,
    })
}

fn solve_two_by_two_block_in_place(values: &[f64], rhs: &mut [f64]) -> Result<(), String> {
    if values.len() != 4 || rhs.len() != 2 {
        return Err("invalid two-by-two block dimensions".into());
    }
    let d11 = values[0];
    let d21 = values[1];
    let d22 = values[3];
    let x0 = d11.mul_add(rhs[0], d21 * rhs[1]);
    let x1 = d21.mul_add(rhs[0], d22 * rhs[1]);
    if !x0.is_finite() || !x1.is_finite() {
        return Err("two-by-two diagonal block solve produced non-finite values".into());
    }
    rhs[0] = x0;
    rhs[1] = x1;
    Ok(())
}

// Mirrors SPRAL SSIDS CPU solve order in NumericSubtree.hxx and
// kernels/ldlt_app.cxx: gather each front-local RHS, apply the dense APP
// trailing GEMV update, run the unit-lower transposed triangular solve, then
// scatter only the eliminated rows.
fn solve_lower_transpose_front_panels_like_native(
    panels: &[SolvePanel],
    factor_rhs: &mut [f64],
    mut profile: Option<&mut SolveProfile>,
) {
    let profile_enabled = profile.is_some();
    let mut local_rhs = Vec::new();
    for panel in panels.iter().rev() {
        let eliminated_len = panel.eliminated_len;
        let local_size = panel.row_positions.len();
        debug_assert!(eliminated_len <= local_size);
        debug_assert_eq!(panel.values.len(), local_size * eliminated_len);

        local_rhs.resize(local_size, 0.0);
        for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
            local_rhs[local_row] = factor_rhs[factor_position];
        }

        if local_size > eliminated_len {
            let started = profile_enabled.then(Instant::now);
            for local_col in 0..eliminated_len {
                let column_start = local_col * local_size;
                let dot = openblas_gemv_t_dot_like_contiguous(
                    &panel.values[column_start + eliminated_len..column_start + local_size],
                    &local_rhs[eliminated_len..local_size],
                );
                local_rhs[local_col] = (-1.0f64).mul_add(dot, local_rhs[local_col]);
            }
            if let Some(profile) = profile.as_mut() {
                if let Some(started) = started {
                    profile.backward_trailing_update_time += started.elapsed();
                }
                profile.backward_trailing_update_columns += eliminated_len;
                profile.backward_trailing_update_dense_entries +=
                    eliminated_len * (local_size - eliminated_len);
            }
        }

        let started = profile_enabled.then(Instant::now);
        for local_col in (0..eliminated_len).rev() {
            let column_start = local_col * local_size;
            let dot = openblas_dotu_like_contiguous(
                &panel.values[column_start + local_col + 1..column_start + eliminated_len],
                &local_rhs[local_col + 1..eliminated_len],
            );
            local_rhs[local_col] -= dot;
        }
        if let Some(profile) = profile.as_mut() {
            if let Some(started) = started {
                profile.backward_triangular_solve_time += started.elapsed();
            }
            profile.backward_triangular_columns += eliminated_len;
            profile.backward_triangular_dense_entries += eliminated_len * (eliminated_len - 1) / 2;
        }

        for local_row in 0..eliminated_len {
            factor_rhs[panel.row_positions[local_row]] = local_rhs[local_row];
        }
    }
}

fn openblas_dotu_like_contiguous(lhs: &[f64], rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());
    openblas_dotu_like_contiguous_impl(lhs, rhs)
}

fn openblas_gemv_t_dot_like_contiguous(lhs: &[f64], rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());
    openblas_gemv_t_dot_like_contiguous_impl(lhs, rhs)
}

#[cfg(target_arch = "aarch64")]
fn openblas_dotu_like_contiguous_impl(lhs: &[f64], rhs: &[f64]) -> f64 {
    // Homebrew's ARM64 OpenBLAS dot kernel reduces 32 contiguous f64 products
    // through eight two-lane accumulators before handling the scalar remainder.
    const UNROLL: usize = 32;
    const ACCUMULATORS: usize = 8;
    let chunks = lhs.len() / UNROLL;
    let mut accumulators = [[0.0; 2]; ACCUMULATORS];
    for chunk in 0..chunks {
        let base = chunk * UNROLL;
        for (accumulator, lanes) in accumulators.iter_mut().enumerate() {
            let index = base + 2 * accumulator;
            lanes[0] = lhs[index].mul_add(rhs[index], lanes[0]);
            lanes[1] = lhs[index + 1].mul_add(rhs[index + 1], lanes[1]);
        }
        for (accumulator, lanes) in accumulators.iter_mut().enumerate() {
            let index = base + 16 + 2 * accumulator;
            lanes[0] = lhs[index].mul_add(rhs[index], lanes[0]);
            lanes[1] = lhs[index + 1].mul_add(rhs[index + 1], lanes[1]);
        }
    }

    let mut dot = if chunks == 0 {
        0.0
    } else {
        let v0 = [
            accumulators[0][0] + accumulators[1][0],
            accumulators[0][1] + accumulators[1][1],
        ];
        let v2 = [
            accumulators[2][0] + accumulators[3][0],
            accumulators[2][1] + accumulators[3][1],
        ];
        let v4 = [
            accumulators[4][0] + accumulators[5][0],
            accumulators[4][1] + accumulators[5][1],
        ];
        let v6 = [
            accumulators[6][0] + accumulators[7][0],
            accumulators[6][1] + accumulators[7][1],
        ];
        let v0 = [v0[0] + v2[0], v0[1] + v2[1]];
        let v4 = [v4[0] + v6[0], v4[1] + v6[1]];
        let v0 = [v0[0] + v4[0], v0[1] + v4[1]];
        v0[0] + v0[1]
    };

    for index in (chunks * UNROLL)..lhs.len() {
        dot = lhs[index].mul_add(rhs[index], dot);
    }
    dot
}

#[cfg(not(target_arch = "aarch64"))]
fn openblas_dotu_like_contiguous_impl(lhs: &[f64], rhs: &[f64]) -> f64 {
    let mut dot = 0.0;
    for (&left, &right) in lhs.iter().zip(rhs) {
        dot = left.mul_add(right, dot);
    }
    dot
}

#[cfg(target_arch = "aarch64")]
fn openblas_gemv_t_dot_like_contiguous_impl(lhs: &[f64], rhs: &[f64]) -> f64 {
    // ARM64 OpenBLAS dgemv_t reduces each output column through four two-lane
    // accumulators over 32- and 8-element chunks, then folds scalar remainder.
    const CHUNK_32: usize = 32;
    const CHUNK_8: usize = 8;
    const ACCUMULATORS: usize = 4;

    let mut accumulators = [[0.0; 2]; ACCUMULATORS];
    let mut offset = 0;
    while offset + CHUNK_32 <= lhs.len() {
        for group in 0..4 {
            let group_base = offset + group * CHUNK_8;
            for (accumulator, lanes) in accumulators.iter_mut().enumerate() {
                let index = group_base + 2 * accumulator;
                lanes[0] = lhs[index].mul_add(rhs[index], lanes[0]);
                lanes[1] = lhs[index + 1].mul_add(rhs[index + 1], lanes[1]);
            }
        }
        offset += CHUNK_32;
    }
    while offset + CHUNK_8 <= lhs.len() {
        for (accumulator, lanes) in accumulators.iter_mut().enumerate() {
            let index = offset + 2 * accumulator;
            lanes[0] = lhs[index].mul_add(rhs[index], lanes[0]);
            lanes[1] = lhs[index + 1].mul_add(rhs[index + 1], lanes[1]);
        }
        offset += CHUNK_8;
    }

    let pair0 = accumulators[0][0] + accumulators[0][1];
    let pair1 = accumulators[1][0] + accumulators[1][1];
    let pair2 = accumulators[2][0] + accumulators[2][1];
    let pair3 = accumulators[3][0] + accumulators[3][1];
    let mut dot = (pair0 + pair1) + (pair2 + pair3);
    for index in offset..lhs.len() {
        dot = lhs[index].mul_add(rhs[index], dot);
    }
    dot
}

#[cfg(not(target_arch = "aarch64"))]
fn openblas_gemv_t_dot_like_contiguous_impl(lhs: &[f64], rhs: &[f64]) -> f64 {
    let mut dot = 0.0;
    for (&left, &right) in lhs.iter().zip(rhs) {
        dot = left.mul_add(right, dot);
    }
    dot
}

struct DenseUnitLower {
    values: Vec<f64>,
    present: Vec<bool>,
}

fn build_dense_unit_lower_from_factor(
    dimension: usize,
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
    lower_values: &[f64],
) -> DenseUnitLower {
    let mut lower = DenseUnitLower {
        values: vec![0.0; dimension * dimension],
        present: vec![false; dimension * dimension],
    };
    for diagonal in 0..dimension {
        let offset = diagonal * dimension + diagonal;
        lower.values[offset] = 1.0;
        lower.present[offset] = true;
    }
    for col in 0..dimension {
        for entry in lower_col_ptrs[col]..lower_col_ptrs[col + 1] {
            let row = lower_row_indices[entry];
            let offset = row * dimension + col;
            lower.values[offset] = lower_values[entry];
            lower.present[offset] = true;
        }
    }
    lower
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PanelFactorStats {
    two_by_two_pivots: usize,
    delayed_pivots: usize,
    max_residual: f64,
}

fn accumulate_native_two_by_two_inertia(inertia: &mut Inertia, a11: f64, a21: f64, a22: f64) {
    // Native SPRAL's CPU stats count 2x2 pivots from the stored D block with
    // det/trace signs, not by forming eigenvalues. This avoids cancellation in
    // trace +/- sqrt(discriminant) for ill-conditioned inverse pivot blocks.
    let det = a11 * a22 - a21 * a21;
    let trace = a11 + a22;
    if det < 0.0 {
        inertia.positive += 1;
        inertia.negative += 1;
    } else if trace < 0.0 {
        inertia.negative += 2;
    } else {
        inertia.positive += 2;
    }
}

fn inertia_from_blocks(
    diagonal_blocks: &[DiagonalBlock],
    diagonal_values: &[f64],
    zero_tol: f64,
) -> Inertia {
    let mut inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    debug_assert_eq!(diagonal_values.len(), diagonal_blocks.len() * 4);
    for (block, values) in diagonal_blocks.iter().zip(diagonal_values.chunks_exact(4)) {
        if block.size == 1 {
            let value = one_by_one_inverse_diagonal(&values[..2]).unwrap_or(0.0);
            if value > zero_tol {
                inertia.positive += 1;
            } else if value < -zero_tol {
                inertia.negative += 1;
            } else {
                inertia.zero += 1;
            }
            continue;
        }
        debug_assert_eq!(block.size, 2);
        accumulate_native_two_by_two_inertia(&mut inertia, values[0], values[1], values[3]);
    }
    inertia
}
fn unpack_packed_lower_to_dense_square(size: usize, packed: &[f64]) -> Vec<f64> {
    let mut dense = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..=row {
            dense[dense_lower_offset(size, row, col)] = packed[packed_lower_offset(size, row, col)];
        }
    }
    dense
}

#[cfg(test)]
mod tests {
    use metis_ordering::{CsrGraph, Permutation};

    use super::{
        NativeOrdering, NativeSpral, NumericFactorOptions, OrderingStrategy, SsidsOptions,
        SymmetricCscMatrix, analyse, build_symbolic_front_tree, dense_lower_offset,
        expand_symmetric_pattern, factorize, factorize_dense_tpp_tail_in_place,
        native_column_counts, native_postorder_permutation, permute_graph, symbolic_factor_pattern,
    };

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
            low + (self.next_u64() as usize % (high - low + 1))
        }

        fn dyadic(&mut self, numerator_radius: i16, max_shift: u8) -> f64 {
            let span = i32::from(numerator_radius) * 2 + 1;
            let numerator = self.next_u64() as i32 % span - i32::from(numerator_radius);
            let shift = self.next_u64() as u8 % (max_shift + 1);
            f64::from(numerator) / f64::from(1_u32 << shift)
        }
    }

    #[test]
    fn native_postorder_matches_depth_first_mapping_for_branching_tree() {
        let elimination_tree = vec![Some(3), Some(3), Some(4), Some(4), None];
        let base_permutation = Permutation::identity(elimination_tree.len());
        let column_has_entries = vec![true; elimination_tree.len()];

        let (postorder, realn) =
            native_postorder_permutation(&elimination_tree, &base_permutation, &column_has_entries);

        assert_eq!(postorder, vec![1, 2, 0, 3, 4]);
        assert_eq!(realn, elimination_tree.len());
    }

    #[test]
    fn native_postorder_moves_empty_roots_after_real_columns() {
        let elimination_tree = vec![None, None, None];
        let base_permutation = Permutation::identity(elimination_tree.len());
        let column_has_entries = vec![true, false, true];

        let (postorder, realn) =
            native_postorder_permutation(&elimination_tree, &base_permutation, &column_has_entries);

        assert_eq!(postorder, vec![0, 2, 1]);
        assert_eq!(realn, 2);
    }

    #[test]
    fn native_column_counts_match_symbolic_patterns_for_permuted_sparse_case() {
        let dense = vec![
            vec![4.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 4.0, 0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 4.0, 1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 4.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0, 4.0, 1.0],
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 4.0],
        ];
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let matrix =
            SymmetricCscMatrix::new(6, &col_ptrs, &row_indices, Some(&values)).expect("valid CSC");
        let graph = CsrGraph::from_symmetric_csc(6, &col_ptrs, &row_indices).expect("valid graph");
        let permutation = Permutation::new(vec![2, 0, 5, 1, 4, 3]).expect("valid permutation");
        let permuted_graph = permute_graph(&graph, &permutation);
        let (elimination_tree, simulated_counts, column_pattern) =
            symbolic_factor_pattern(&permuted_graph);
        let expanded_pattern = expand_symmetric_pattern(matrix);

        let native_counts =
            native_column_counts(&expanded_pattern, &permutation, &elimination_tree);

        assert_eq!(native_counts, simulated_counts);
        assert_eq!(
            native_counts,
            column_pattern.iter().map(Vec::len).collect::<Vec<usize>>()
        );
    }

    #[test]
    fn solve_profile_preserves_solution() {
        let col_ptrs = vec![0, 2, 3];
        let row_indices = vec![0, 1, 1];
        let values = vec![2.0, 1.0, 2.0];
        let matrix =
            SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, Some(&values)).expect("valid CSC");
        let (symbolic, _) = analyse(
            matrix,
            &SsidsOptions {
                ordering: OrderingStrategy::Natural,
            },
        )
        .expect("analyse");
        let (mut factor, _) =
            factorize(matrix, &symbolic, &NumericFactorOptions::default()).expect("factorize");
        let mut rhs = vec![1.0, 0.0];

        let profile = factor
            .solve_in_place_with_profile(&mut rhs)
            .expect("profiled solve");

        assert!((rhs[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((rhs[1] + 1.0 / 3.0).abs() < 1e-12);
        assert!(profile.total_recorded_time() >= profile.diagonal_solve_time);
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

    fn square_to_dense_lower(matrix: &[Vec<f64>]) -> Vec<f64> {
        let size = matrix.len();
        let mut dense = vec![0.0; size * size];
        for row in 0..size {
            for col in 0..=row {
                dense[dense_lower_offset(size, row, col)] = matrix[row][col];
            }
        }
        dense
    }

    #[test]
    fn dense_tpp_tail_records_outgoing_delays() {
        let mut rows = vec![0, 1, 2];
        let mut dense = square_to_dense_lower(&[
            vec![-422.9265249204227, 0.0, 0.0],
            vec![0.0, -2.580878217593716e-06, 0.0],
            vec![-2221.525473541999, -7.3870375201836405, 0.0],
        ]);
        let mut ld = vec![0.0; rows.len() * 2];

        let factorization = factorize_dense_tpp_tail_in_place(
            &mut rows,
            &mut dense,
            0,
            2,
            NumericFactorOptions {
                threshold_pivot_u: 0.01,
                ..NumericFactorOptions::default()
            },
            false,
            &mut ld,
        )
        .expect("tpp tail factorization");

        assert_eq!(factorization.stats.delayed_pivots, 1);
        assert_eq!(factorization.contribution.delayed_count, 1);
        assert_eq!(factorization.factor_order.len(), 1);
        assert_eq!(factorization.contribution.row_ids, vec![1, 2]);
    }

    #[test]
    fn relaxed_symbolic_fronts_match_native_dense_gap_pivot_order() {
        let Ok(native) = NativeSpral::load() else {
            return;
        };
        let mut rng = DenseBoundaryRng::new(0x1001);
        let dimension = rng.usize_inclusive(33, 160);
        assert_eq!(dimension, 54);
        let dense = random_dense_dyadic_matrix(dimension, &mut rng);
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
            .expect("valid CSC");
        let options = NumericFactorOptions::default();

        let (symbolic, _) = analyse(
            matrix,
            &SsidsOptions {
                ordering: OrderingStrategy::Natural,
            },
        )
        .expect("rust analyse");
        let tree = build_symbolic_front_tree(&symbolic);
        assert_eq!(tree.fronts.len(), 1);
        let (rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");
        let mut native_session = native
            .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
            .expect("native analyse");
        let native_info = native_session.factorize(matrix).expect("native factorize");
        let native_enquiry = native_session
            .enquire_indef()
            .expect("native enquire indef");

        let mut native_factor_order = vec![usize::MAX; native_enquiry.pivot_order.len()];
        for (column, &pivot_position) in native_enquiry.pivot_order.iter().enumerate() {
            native_factor_order[pivot_position] = column;
        }

        assert_eq!(rust_factor.inertia(), native_info.inertia);
        assert_eq!(
            rust_factor.pivot_stats().two_by_two_pivots,
            native_info.two_by_two_pivots
        );
        assert_eq!(
            rust_factor.pivot_stats().delayed_pivots,
            native_info.delayed_pivots
        );
        assert_eq!(rust_factor.factor_order, native_factor_order);
    }

    #[test]
    fn app_find_maxloc_tie_order_matches_native_dense_seed6() {
        let Ok(native) = NativeSpral::load() else {
            return;
        };
        let mut rng = DenseBoundaryRng::new(6);
        let dimension = rng.usize_inclusive(33, 33);
        assert_eq!(dimension, 33);
        let dense = random_dense_dyadic_matrix(dimension, &mut rng);
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
            .expect("valid CSC");
        let options = NumericFactorOptions::default();
        let (symbolic, _) = analyse(
            matrix,
            &SsidsOptions {
                ordering: OrderingStrategy::Natural,
            },
        )
        .expect("rust analyse");
        let (rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");
        let mut native_session = native
            .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
            .expect("native analyse");
        native_session.factorize(matrix).expect("native factorize");
        let native_enquiry = native_session.enquire_indef().expect("native enquire");

        let mut native_factor_order = vec![usize::MAX; native_enquiry.pivot_order.len()];
        for (column, &pivot_position) in native_enquiry.pivot_order.iter().enumerate() {
            native_factor_order[pivot_position] = column;
        }

        assert_eq!(&rust_factor.factor_order[..4], &[25, 28, 2, 27]);
        assert_eq!(rust_factor.factor_order, native_factor_order);
    }
}
