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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FactorProfile {
    pub symbolic_front_tree_time: Duration,
    pub permuted_pattern_time: Duration,
    pub permuted_values_time: Duration,
    pub front_factorization_time: Duration,
    pub front_assembly_time: Duration,
    pub dense_front_factorization_time: Duration,
    pub tpp_factorization_time: Duration,
    pub app_pivot_factor_time: Duration,
    pub app_block_pivot_apply_time: Duration,
    pub app_block_triangular_solve_time: Duration,
    pub app_block_diagonal_apply_time: Duration,
    pub app_failed_pivot_scan_time: Duration,
    pub app_restore_time: Duration,
    pub app_accepted_update_time: Duration,
    pub app_column_storage_time: Duration,
    pub solve_panel_build_time: Duration,
    pub root_delayed_factorization_time: Duration,
    pub factor_inverse_time: Duration,
    pub lower_storage_time: Duration,
    pub solve_panel_storage_time: Duration,
    pub diagonal_storage_time: Duration,
    pub factor_bytes_time: Duration,
    pub front_count: usize,
    pub local_dense_entries: usize,
    pub root_delayed_blocks: usize,
}

impl FactorProfile {
    pub fn total_recorded_time(&self) -> Duration {
        self.symbolic_front_tree_time
            + self.permuted_pattern_time
            + self.permuted_values_time
            + self.front_factorization_time
            + self.root_delayed_factorization_time
            + self.factor_inverse_time
            + self.lower_storage_time
            + self.solve_panel_storage_time
            + self.diagonal_storage_time
            + self.factor_bytes_time
    }

    pub fn accumulate(&mut self, other: &Self) {
        self.symbolic_front_tree_time += other.symbolic_front_tree_time;
        self.permuted_pattern_time += other.permuted_pattern_time;
        self.permuted_values_time += other.permuted_values_time;
        self.front_factorization_time += other.front_factorization_time;
        self.front_assembly_time += other.front_assembly_time;
        self.dense_front_factorization_time += other.dense_front_factorization_time;
        self.tpp_factorization_time += other.tpp_factorization_time;
        self.app_pivot_factor_time += other.app_pivot_factor_time;
        self.app_block_pivot_apply_time += other.app_block_pivot_apply_time;
        self.app_block_triangular_solve_time += other.app_block_triangular_solve_time;
        self.app_block_diagonal_apply_time += other.app_block_diagonal_apply_time;
        self.app_failed_pivot_scan_time += other.app_failed_pivot_scan_time;
        self.app_restore_time += other.app_restore_time;
        self.app_accepted_update_time += other.app_accepted_update_time;
        self.app_column_storage_time += other.app_column_storage_time;
        self.solve_panel_build_time += other.solve_panel_build_time;
        self.root_delayed_factorization_time += other.root_delayed_factorization_time;
        self.factor_inverse_time += other.factor_inverse_time;
        self.lower_storage_time += other.lower_storage_time;
        self.solve_panel_storage_time += other.solve_panel_storage_time;
        self.diagonal_storage_time += other.diagonal_storage_time;
        self.factor_bytes_time += other.factor_bytes_time;
        self.front_count += other.front_count;
        self.local_dense_entries += other.local_dense_entries;
        self.root_delayed_blocks += other.root_delayed_blocks;
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
    profile: FactorProfile,
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
    profile: FactorProfile,
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
    profile: FactorProfile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DenseTppTailRequest {
    start_pivot: usize,
    candidate_len: usize,
    options: NumericFactorOptions,
    require_full_elimination: bool,
    profile_enabled: bool,
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
        self.refactorize_with_cached_symbolic_profile(matrix, None)
    }

    fn refactorize_with_cached_symbolic_profile(
        &mut self,
        matrix: SymmetricCscMatrix<'_>,
        profile: Option<&mut FactorProfile>,
    ) -> Result<FactorInfo, SsidsError> {
        let profile_enabled = profile.is_some();
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
            profile_enabled,
        )?;
        if let Some(profile) = profile {
            profile.accumulate(&factorization.profile);
        }
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
    factorize_impl(matrix, symbolic, options, None)
}

pub fn factorize_with_profile(
    matrix: SymmetricCscMatrix<'_>,
    symbolic: &SymbolicFactor,
    options: &NumericFactorOptions,
) -> Result<(NumericFactor, FactorInfo, FactorProfile), SsidsError> {
    let mut profile = FactorProfile::default();
    let (factor, info) = factorize_impl(matrix, symbolic, options, Some(&mut profile))?;
    Ok((factor, info, profile))
}

fn factorize_impl(
    matrix: SymmetricCscMatrix<'_>,
    symbolic: &SymbolicFactor,
    options: &NumericFactorOptions,
    mut profile: Option<&mut FactorProfile>,
) -> Result<(NumericFactor, FactorInfo), SsidsError> {
    if matrix.dimension() != symbolic.permutation.len() {
        return Err(SsidsError::DimensionMismatch {
            expected: symbolic.permutation.len(),
            actual: matrix.dimension(),
        });
    }
    let profile_enabled = profile.is_some();
    let started = profile_enabled.then(Instant::now);
    let front_tree = build_symbolic_front_tree(symbolic);
    if let (Some(profile), Some(started)) = (profile.as_mut(), started) {
        profile.symbolic_front_tree_time += started.elapsed();
    }
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
    let info = factor.refactorize_with_cached_symbolic_profile(matrix, profile)?;
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
    let det = (a11 * detscale) * a22 - a21.abs();
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
            // block_ldlt.hxx::update_2x2 forms the two-product update under
            // `#pragma omp simd`; the local optimized native path contracts the
            // first product into the second product before subtracting it.
            let combined =
                first_preserved.mul_add(first_multiplier, second_preserved * second_multiplier);
            matrix[update_entry] -= combined;
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
) -> Result<FactorBlockRecord, SsidsError> {
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
    }

    app_update_one_by_one(matrix, size, pivot, update_end, scratch);

    Ok(FactorBlockRecord {
        size: 1,
        values: [inverse_diagonal, 0.0, 0.0, 0.0],
    })
}

fn factor_two_by_two_common(
    rows: &[usize],
    matrix: &mut [f64],
    bounds: DenseUpdateBounds,
    pivot: usize,
    inverse: (f64, f64, f64),
    stats: &mut PanelFactorStats,
    scratch: &mut [f64],
) -> Result<FactorBlockRecord, SsidsError> {
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

    for row in (pivot + 2)..update_end {
        let b1 = matrix[dense_lower_offset(size, row, pivot)];
        let b2 = matrix[dense_lower_offset(size, row, pivot + 1)];
        first_scratch[row] = b1;
        second_scratch[row] = b2;
        // The local optimized block_ldlt<32> build contracts both 2x2
        // multiplier rows with the first addend folded into the second.
        let l1 = inv12.mul_add(b2, inv11 * b1);
        let l2 = inv12.mul_add(b1, inv22 * b2);
        if !l1.is_finite() || !l2.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: "two-by-two multipliers became non-finite".into(),
            });
        }
        matrix[dense_lower_offset(size, row, pivot)] = l1;
        matrix[dense_lower_offset(size, row, pivot + 1)] = l2;
    }

    app_update_two_by_two(matrix, size, pivot, update_end, scratch);

    Ok(FactorBlockRecord {
        size: 2,
        values: [inv11, inv12, f64::INFINITY, inv22],
    })
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
    profile_enabled: bool,
) -> (Duration, Duration) {
    if block_end >= size {
        return (Duration::default(), Duration::default());
    }

    let triangular_solve_time = app_solve_block_triangular_to_trailing_rows(
        matrix,
        size,
        block_start,
        block_end,
        profile_enabled,
    );
    let diagonal_apply_time = app_apply_block_diagonal_to_trailing_rows(
        matrix,
        size,
        block_start,
        block_end,
        block_records,
        small,
        profile_enabled,
    );
    (triangular_solve_time, diagonal_apply_time)
}

fn app_solve_block_triangular_to_trailing_rows(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
    profile_enabled: bool,
) -> Duration {
    // OP_N applies a diagonal block to rows below the eliminated columns, where
    // SPRAL's column-major `aval[col * lda + row]` matches our dense storage.
    let triangular_started = profile_enabled.then(Instant::now);
    const OPENBLAS_DTRSM_UNROLL_N: usize = 4;
    for row in block_end..size {
        let mut group_start = block_start;
        while group_start < block_end {
            let group_end = (group_start + OPENBLAS_DTRSM_UNROLL_N).min(block_end);
            for col in group_start..group_end {
                let entry = col * size + row;
                let mut value = matrix[entry];
                if group_start > block_start {
                    let mut dot = 0.0;
                    for prior in block_start..group_start {
                        let prior_value = matrix[prior * size + row];
                        let lower_value = matrix[prior * size + col];
                        dot = prior_value.mul_add(lower_value, dot);
                    }
                    value = dot.mul_add(-1.0, value);
                }
                matrix[entry] = value;
            }

            for col in group_start..group_end {
                let value = matrix[col * size + row];
                for target_col in (col + 1)..group_end {
                    let target_entry = target_col * size + row;
                    let lower_value = matrix[col * size + target_col];
                    matrix[target_entry] = (-value).mul_add(lower_value, matrix[target_entry]);
                }
            }
            group_start = group_end;
        }
    }
    triangular_started.map_or(Duration::default(), |started| started.elapsed())
}

fn app_apply_block_diagonal_to_trailing_rows(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
    block_records: &[FactorBlockRecord],
    small: f64,
    profile_enabled: bool,
) -> Duration {
    let diagonal_started = profile_enabled.then(Instant::now);
    let mut col = block_start;
    for block in block_records {
        if block.size == 1 {
            let d11 = block.values[0];
            for row in block_end..size {
                let entry = col * size + row;
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
                let first_entry = col * size + row;
                let second_entry = (col + 1) * size + row;
                let a1 = matrix[first_entry];
                let a2 = matrix[second_entry];
                matrix[first_entry] = d11.mul_add(a1, d21 * a2);
                matrix[second_entry] = d21.mul_add(a1, d22 * a2);
            }
            col += 2;
        }
    }
    diagonal_started.map_or(Duration::default(), |started| started.elapsed())
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

fn app_build_ld_workspace(
    matrix: &[f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
) -> Vec<f64> {
    // Accepted-update operands are below the eliminated prefix, so SPRAL's
    // column-major `aval[col * lda + row]` matches the dense lower storage.
    let accepted_width = accepted_end - block_start;
    let mut ld_values = vec![0.0; accepted_width * size];
    let mut pivot = block_start;
    for block in block_records {
        let relative_pivot = pivot - block_start;
        if block.size == 1 {
            let diagonal = app_original_one_by_one_diagonal(block.values[0]);
            for row in accepted_end..size {
                let row_l = matrix[pivot * size + row];
                ld_values[relative_pivot * size + row] = diagonal * row_l;
            }
            pivot += 1;
        } else {
            let inv11 = block.values[0];
            let inv21 = block.values[1];
            let inv22 = block.values[3];
            let det = inv11.mul_add(inv22, -(inv21 * inv21));
            let d11 = inv11 / det;
            let d21 = inv21 / det;
            let d22 = inv22 / det;
            let trailing_rows = size - accepted_end;
            let vector_tail_start = if trailing_rows > 4 {
                accepted_end + (trailing_rows & !1)
            } else {
                accepted_end
            };
            for row in accepted_end..size {
                let row_l1 = matrix[pivot * size + row];
                let row_l2 = matrix[(pivot + 1) * size + row];
                // SPRAL cpu/kernels/calc_ld.hxx calcLD<OP_N> is compiled
                // locally into a two-lane vector body plus scalar tail.
                ld_values[relative_pivot * size + row] = if row < vector_tail_start {
                    (-d21).mul_add(row_l2, d22 * row_l1)
                } else {
                    d22.mul_add(row_l1, -(d21 * row_l2))
                };
                ld_values[(relative_pivot + 1) * size + row] = (-d21).mul_add(row_l1, d11 * row_l2);
            }
            pivot += 2;
        }
    }
    debug_assert_eq!(pivot, accepted_end);
    ld_values
}

struct AppAcceptedUpdateContext<'a> {
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &'a [FactorBlockRecord],
    ld_values: &'a [f64],
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
    let ld_values = app_build_ld_workspace(matrix, size, block_start, accepted_end, block_records);
    if accepted_end + 1 == size {
        let row = accepted_end;
        let entry = row * size + row;
        let mut pivot = block_start;
        for block in block_records {
            let relative_pivot = pivot - block_start;
            if block.size == 1 {
                let row_l = matrix[pivot * size + row];
                let row_ld = ld_values[relative_pivot * size + row];
                matrix[entry] = (-row_l).mul_add(row_ld, matrix[entry]);
                pivot += 1;
            } else {
                let row_l1 = matrix[pivot * size + row];
                let row_l2 = matrix[(pivot + 1) * size + row];
                let row_ld1 = ld_values[relative_pivot * size + row];
                let row_ld2 = ld_values[(relative_pivot + 1) * size + row];
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
                    AppAcceptedUpdateContext {
                        size,
                        block_start,
                        accepted_end,
                        block_records,
                        ld_values: &ld_values,
                    },
                    row,
                    col,
                );
                continue;
            }
            let mut update = 0.0;
            let mut pivot = block_start;
            for block in block_records {
                let relative_pivot = pivot - block_start;
                if block.size == 1 {
                    let col_l = matrix[pivot * size + col];
                    let row_ld = ld_values[relative_pivot * size + row];
                    update = row_ld.mul_add(col_l, update);
                    pivot += 1;
                } else {
                    let col_l1 = matrix[pivot * size + col];
                    let col_l2 = matrix[(pivot + 1) * size + col];
                    let row_ld1 = ld_values[relative_pivot * size + row];
                    let row_ld2 = ld_values[(relative_pivot + 1) * size + row];
                    update = row_ld1.mul_add(col_l1, update);
                    update = row_ld2.mul_add(col_l2, update);
                    pivot += 2;
                }
            }
            debug_assert_eq!(pivot, accepted_end);
            let entry = col * size + row;
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
    context: AppAcceptedUpdateContext<'_>,
    row: usize,
    col: usize,
) {
    let size = context.size;
    let entry = col * size + row;
    let mut pivot = context.block_start;
    for block in context.block_records {
        let relative_pivot = pivot - context.block_start;
        if block.size == 1 {
            let col_l = matrix[pivot * size + col];
            let row_ld = context.ld_values[relative_pivot * size + row];
            matrix[entry] = (-col_l).mul_add(row_ld, matrix[entry]);
            pivot += 1;
        } else {
            let col_l1 = matrix[pivot * size + col];
            let col_l2 = matrix[(pivot + 1) * size + col];
            let row_ld1 = context.ld_values[relative_pivot * size + row];
            let row_ld2 = context.ld_values[(relative_pivot + 1) * size + row];
            matrix[entry] = (-col_l1).mul_add(row_ld1, matrix[entry]);
            matrix[entry] = (-col_l2).mul_add(row_ld2, matrix[entry]);
            pivot += 2;
        }
    }
    debug_assert_eq!(pivot, context.accepted_end);
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
            entries.push((rows[row], matrix[col * size + row]));
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
    bounds: DenseUpdateBounds,
    pivot: usize,
    _stats: &mut PanelFactorStats,
    ld: &mut [f64],
) -> Result<(FactorColumn, FactorBlockRecord), SsidsError> {
    let size = bounds.size;
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

    root_tpp_rank1_update(matrix, bounds, pivot + 1, pivot, work);

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
    bounds: DenseUpdateBounds,
    pivot: usize,
    inverse: (f64, f64, f64),
    stats: &mut PanelFactorStats,
    ld: &mut [f64],
) -> Result<([FactorColumn; 2], FactorBlockRecord), SsidsError> {
    let size = bounds.size;
    let (first_scratch, second_scratch) = ld.split_at_mut(size);
    let (inv11, inv12, inv22) = inverse;
    stats.two_by_two_pivots += 1;
    matrix[dense_lower_offset(size, pivot, pivot)] = 1.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot)] = 0.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot + 1)] = 1.0;

    let first_multiplier_row = pivot + 2;
    let trailing_rows = size.saturating_sub(first_multiplier_row);
    // The local native SPRAL build vectorizes ldlt_tpp.cxx::apply_2x2 once
    // there are at least four trailing rows. Its vector loop accumulates the
    // first multiplier as d21*b2 + d11*b1; the scalar path accumulates it as
    // the source-spelled d11*b1 + d21*b2.
    let vectorized_multiplier_rows = if trailing_rows >= 4 {
        trailing_rows / 2 * 2
    } else {
        0
    };
    let mut first_entries = Vec::new();
    let mut second_entries = Vec::new();
    for row in first_multiplier_row..size {
        let b1 = matrix[dense_lower_offset(size, row, pivot)];
        let b2 = matrix[dense_lower_offset(size, row, pivot + 1)];
        first_scratch[row] = b1;
        second_scratch[row] = b2;
        let local_row = row - first_multiplier_row;
        let l1 = if local_row < vectorized_multiplier_rows {
            inv12.mul_add(b2, inv11 * b1)
        } else {
            inv11.mul_add(b1, inv12 * b2)
        };
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
        bounds,
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
    bounds: DenseUpdateBounds,
    start: usize,
    multiplier_column: usize,
    preserved_column: &[f64],
) {
    let size = bounds.size;
    // Native SPRAL routes one-row and one-column TPP updates through dgemm
    // shapes that round like sequential scalar updates. Wider trailing blocks
    // round like OpenBLAS' block kernel.
    let use_scalar_fma =
        size.saturating_sub(start) == 1 || bounds.update_end.saturating_sub(start) == 1;
    // ldlt_tpp_factor distinguishes m rows from n candidate columns; the
    // trailing update spans all rows but only columns before n.
    for (col, &preserved) in preserved_column
        .iter()
        .enumerate()
        .take(bounds.update_end)
        .skip(start)
    {
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
    bounds: DenseUpdateBounds,
    start: usize,
    first_multiplier_column: usize,
    second_multiplier_column: usize,
    first_preserved_column: &[f64],
    second_preserved_column: &[f64],
) {
    let size = bounds.size;
    let use_scalar_fma =
        size.saturating_sub(start) == 1 || bounds.update_end.saturating_sub(start) == 1;
    // ldlt_tpp_factor distinguishes m rows from n candidate columns; the
    // trailing update spans all rows but only columns before n.
    for (col, (&first_preserved, &second_preserved)) in first_preserved_column
        .iter()
        .zip(second_preserved_column.iter())
        .enumerate()
        .take(bounds.update_end)
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
    request: DenseTppTailRequest,
    ld: &mut [f64],
) -> Result<DenseFrontFactorization, SsidsError> {
    let started = request.profile_enabled.then(Instant::now);
    let size = rows.len();
    let mut stats = PanelFactorStats::default();
    let mut factor_order = Vec::with_capacity(size);
    let mut factor_columns = Vec::with_capacity(size);
    let mut block_records = Vec::new();
    let mut pivot = request.start_pivot;
    let active_candidate_end = (request.start_pivot + request.candidate_len).min(size);

    while pivot < active_candidate_end {
        if dense_column_small(
            dense,
            size,
            pivot,
            pivot,
            request.options.small_pivot_tolerance,
        ) {
            if !request.options.action_on_zero_pivot {
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
            if dense_column_small(
                dense,
                size,
                candidate,
                pivot,
                request.options.small_pivot_tolerance,
            ) {
                if !request.options.action_on_zero_pivot {
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

            if let Some(inverse) = tpp_test_two_by_two(a11, a21, a22, maxt, maxp, request.options) {
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
                let (columns, block) = tpp_factor_two_by_two(
                    rows,
                    dense,
                    DenseUpdateBounds {
                        size,
                        update_end: active_candidate_end,
                    },
                    pivot,
                    inverse,
                    &mut stats,
                    ld,
                )?;
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
            if a22.abs() >= request.options.threshold_pivot_u * maxp {
                if candidate != pivot {
                    dense_symmetric_swap(dense, size, candidate, pivot);
                    rows.swap(candidate, pivot);
                }
                let (column, block) = tpp_factor_one_by_one(
                    rows,
                    dense,
                    DenseUpdateBounds {
                        size,
                        update_end: active_candidate_end,
                    },
                    pivot,
                    &mut stats,
                    ld,
                )?;
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
        if current_diag.abs() >= request.options.threshold_pivot_u * current_offdiag_max {
            let (column, block) = tpp_factor_one_by_one(
                rows,
                dense,
                DenseUpdateBounds {
                    size,
                    update_end: active_candidate_end,
                },
                pivot,
                &mut stats,
                ld,
            )?;
            factor_order.push(rows[pivot]);
            factor_columns.push(column);
            block_records.push(block);
            pivot += 1;
            continue;
        }

        if request.require_full_elimination {
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

    let mut profile = FactorProfile::default();
    if let Some(started) = started {
        profile.tpp_factorization_time += started.elapsed();
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
        profile,
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
    profile_enabled: bool,
) -> Result<DenseFrontFactorization, SsidsError> {
    let size = rows.len();
    let mut ld = vec![0.0; size.saturating_mul(2).max(1)];
    let mut factorization = factorize_dense_tpp_tail_in_place(
        &mut rows,
        &mut dense,
        DenseTppTailRequest {
            start_pivot: 0,
            candidate_len: size,
            options,
            require_full_elimination: true,
            profile_enabled,
        },
        &mut ld,
    )?;
    if let Some(&pivot) = factorization.contribution.row_ids.first() {
        return Err(SsidsError::NumericalBreakdown {
            pivot,
            detail: format!(
                "root TPP completion retained {} delayed pivots",
                factorization.contribution.delayed_count
            ),
        });
    }
    let started = profile_enabled.then(Instant::now);
    if let Some(panel) = build_factor_solve_panel_record(
        &factorization.factor_order,
        &factorization.factor_columns,
        &factorization.contribution.row_ids,
    )? {
        factorization.solve_panels.push(panel);
    }
    if let Some(started) = started {
        factorization.profile.solve_panel_build_time += started.elapsed();
    }
    Ok(factorization)
}

fn factorize_dense_front(
    mut rows: Vec<usize>,
    candidate_len: usize,
    mut dense: Vec<f64>,
    options: NumericFactorOptions,
    profile_enabled: bool,
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
            DenseTppTailRequest {
                start_pivot: 0,
                candidate_len: active_candidate_end,
                options,
                require_full_elimination: false,
                profile_enabled,
            },
            &mut tpp_ld,
        )?;
        let started = profile_enabled.then(Instant::now);
        if let Some(panel) = build_factor_solve_panel_record(
            &factorization.factor_order,
            &factorization.factor_columns,
            &factorization.contribution.row_ids,
        )? {
            factorization.solve_panels.push(panel);
        }
        if let Some(started) = started {
            factorization.profile.solve_panel_build_time += started.elapsed();
        }
        return Ok(factorization);
    }

    let mut stats = PanelFactorStats::default();
    let mut profile = FactorProfile::default();
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

        let started = profile_enabled.then(Instant::now);
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
                let block = factor_one_by_one_common(
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
                let block = factor_one_by_one_common(
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
                let block = factor_two_by_two_common(
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
        if let Some(started) = started {
            profile.app_pivot_factor_time += started.elapsed();
        }

        let started = profile_enabled.then(Instant::now);
        let (triangular_solve_time, diagonal_apply_time) = app_apply_block_pivots_to_trailing_rows(
            &mut dense,
            size,
            block_start,
            block_end,
            &local_blocks,
            options.small_pivot_tolerance,
            profile_enabled,
        );
        if let Some(started) = started {
            profile.app_block_pivot_apply_time += started.elapsed();
            profile.app_block_triangular_solve_time += triangular_solve_time;
            profile.app_block_diagonal_apply_time += diagonal_apply_time;
        }
        let started = profile_enabled.then(Instant::now);
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
        if let Some(started) = started {
            profile.app_failed_pivot_scan_time += started.elapsed();
        }

        let started = profile_enabled.then(Instant::now);
        app_restore_trailing_from_block_backup(
            &rows,
            &rows_before_block,
            &mut dense,
            &dense_before_block,
            size,
            accepted_end,
        );
        if let Some(started) = started {
            profile.app_restore_time += started.elapsed();
        }
        let started = profile_enabled.then(Instant::now);
        app_apply_accepted_prefix_update(
            &mut dense,
            size,
            block_start,
            accepted_end,
            &accepted_blocks,
        );
        if let Some(started) = started {
            profile.app_accepted_update_time += started.elapsed();
        }

        factor_order.extend(rows[block_start..accepted_end].iter().copied());
        let started = profile_enabled.then(Instant::now);
        factor_columns.extend(app_build_factor_columns_for_prefix(
            &rows,
            &dense,
            size,
            block_start,
            accepted_end,
        ));
        if let Some(started) = started {
            profile.app_column_storage_time += started.elapsed();
        }
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
            DenseTppTailRequest {
                start_pivot: pivot,
                candidate_len: delayed_count,
                options,
                require_full_elimination: false,
                profile_enabled,
            },
            &mut tpp_ld,
        )?;
        factor_order.extend(tpp_tail.factor_order);
        factor_columns.extend(tpp_tail.factor_columns);
        block_records.extend(tpp_tail.block_records);
        aggregate_panel_stats(&mut stats, tpp_tail.stats);
        profile.accumulate(&tpp_tail.profile);
        let contribution = tpp_tail.contribution;
        let mut solve_panels = Vec::new();
        let started = profile_enabled.then(Instant::now);
        if let Some(panel) =
            build_factor_solve_panel_record(&factor_order, &factor_columns, &contribution.row_ids)?
        {
            solve_panels.push(panel);
        }
        if let Some(started) = started {
            profile.solve_panel_build_time += started.elapsed();
        }
        return Ok(DenseFrontFactorization {
            factor_order,
            factor_columns,
            block_records,
            solve_panels,
            contribution,
            stats,
            profile,
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
    let started = profile_enabled.then(Instant::now);
    if let Some(panel) =
        build_factor_solve_panel_record(&factor_order, &factor_columns, &contribution.row_ids)?
    {
        solve_panels.push(panel);
    }
    if let Some(started) = started {
        profile.solve_panel_build_time += started.elapsed();
    }

    Ok(DenseFrontFactorization {
        factor_order,
        factor_columns,
        block_records,
        solve_panels,
        contribution,
        stats,
        profile,
    })
}

fn factor_front_recursive(
    front_id: usize,
    tree: &SymbolicFrontTree,
    matrix: &PermutedLowerMatrix<'_>,
    options: NumericFactorOptions,
    progress: Option<&FactorizationProgressShared>,
    profile_enabled: bool,
) -> Result<FrontFactorizationResult, SsidsError> {
    let front = &tree.fronts[front_id];
    let child_results =
        if front.children.len() >= 2 && front.width() + front.interface_rows.len() >= 32 {
            let raw = front
                .children
                .par_iter()
                .map(|&child| {
                    factor_front_recursive(child, tree, matrix, options, progress, profile_enabled)
                })
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
                    child,
                    tree,
                    matrix,
                    options,
                    progress,
                    profile_enabled,
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
    let mut profile = FactorProfile::default();
    let mut max_front_size = 0;
    let mut contribution_storage_bytes = 0;

    for child in child_results {
        factor_order.extend(child.factor_order);
        factor_columns.extend(child.factor_columns);
        block_records.extend(child.block_records);
        solve_panels.extend(child.solve_panels);
        child_contributions.push(child.contribution);
        aggregate_panel_stats(&mut stats, child.stats);
        profile.accumulate(&child.profile);
        max_front_size = max_front_size.max(child.max_front_size);
        contribution_storage_bytes += child.contribution_storage_bytes;
    }

    let assembly_started = profile_enabled.then(Instant::now);
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
    if let Some(started) = assembly_started {
        profile.front_assembly_time += started.elapsed();
        profile.front_count += 1;
        profile.local_dense_entries += local_size * local_size;
    }

    let factor_started = profile_enabled.then(Instant::now);
    let local = factorize_dense_front(
        local_rows,
        front.width(),
        local_dense,
        options,
        profile_enabled,
    )?;
    if let Some(started) = factor_started {
        profile.dense_front_factorization_time += started.elapsed();
    }
    profile.accumulate(&local.profile);
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
        profile,
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
    profile_enabled: bool,
) -> Result<MultifrontalFactorizationOutcome, SsidsError> {
    let dimension = matrix.dimension();
    let mut profile = FactorProfile::default();
    if buffers.permuted_matrix_source_positions.len() != matrix.row_indices().len()
        || buffers.permuted_matrix_col_ptrs.len() != dimension + 1
    {
        let started = profile_enabled.then(Instant::now);
        build_permuted_lower_csc_pattern(
            matrix,
            permutation,
            buffers.permuted_matrix_col_ptrs,
            buffers.permuted_matrix_row_indices,
            buffers.permuted_matrix_source_positions,
        )?;
        if let Some(started) = started {
            profile.permuted_pattern_time += started.elapsed();
        }
    }
    let started = profile_enabled.then(Instant::now);
    fill_permuted_lower_csc_values(
        matrix,
        buffers.permuted_matrix_source_positions,
        buffers.permuted_matrix_values,
    )?;
    if let Some(started) = started {
        profile.permuted_values_time += started.elapsed();
    }
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
        let started = profile_enabled.then(Instant::now);
        let raw = tree
            .roots
            .par_iter()
            .map(|&root| {
                factor_front_recursive(
                    root,
                    tree,
                    &permuted_matrix,
                    options,
                    Some(&progress),
                    profile_enabled,
                )
            })
            .collect::<Vec<_>>();
        if let Some(started) = started {
            profile.front_factorization_time += started.elapsed();
        }
        let mut collected = Vec::with_capacity(raw.len());
        for result in raw {
            collected.push(result?);
        }
        collected
    } else {
        let started = profile_enabled.then(Instant::now);
        let mut collected = Vec::with_capacity(tree.roots.len());
        for &root in &tree.roots {
            collected.push(factor_front_recursive(
                root,
                tree,
                &permuted_matrix,
                options,
                Some(&progress),
                profile_enabled,
            )?);
        }
        if let Some(started) = started {
            profile.front_factorization_time += started.elapsed();
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
        profile.accumulate(&result.profile);
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
        let started = profile_enabled.then(Instant::now);
        let delayed_local = factorize_root_dense_tpp(
            row_ids,
            unpack_packed_lower_to_dense_square(size, &dense),
            options,
            profile_enabled,
        )?;
        if let Some(started) = started {
            profile.root_delayed_factorization_time += started.elapsed();
            profile.root_delayed_blocks += 1;
        }
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

    let started = profile_enabled.then(Instant::now);
    buffers.factor_order.clear();
    buffers.factor_order.extend(factor_order);

    buffers.factor_inverse.clear();
    buffers.factor_inverse.resize(dimension, usize::MAX);
    for (position, &ordered_index) in buffers.factor_order.iter().enumerate() {
        buffers.factor_inverse[ordered_index] = position;
    }
    if let Some(started) = started {
        profile.factor_inverse_time += started.elapsed();
    }
    let started = profile_enabled.then(Instant::now);
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
    if let Some(started) = started {
        profile.lower_storage_time += started.elapsed();
    }

    let started = profile_enabled.then(Instant::now);
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
    if let Some(started) = started {
        profile.solve_panel_storage_time += started.elapsed();
    }

    let started = profile_enabled.then(Instant::now);
    buffers.diagonal_blocks.clear();
    buffers.diagonal_values.clear();
    for block in block_records {
        buffers
            .diagonal_blocks
            .push(DiagonalBlock { size: block.size });
        buffers.diagonal_values.extend(block.values);
    }
    if let Some(started) = started {
        profile.diagonal_storage_time += started.elapsed();
    }

    let started = profile_enabled.then(Instant::now);
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
    if let Some(started) = started {
        profile.factor_bytes_time += started.elapsed();
    }

    Ok(MultifrontalFactorizationOutcome {
        pivot_stats: PivotStats {
            two_by_two_pivots: stats.two_by_two_pivots,
            delayed_pivots: stats.delayed_pivots,
        },
        factorization_residual_max_abs: stats.max_residual,
        stored_nnz,
        factor_bytes,
        profile,
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
// kernels/ldlt_app.cxx: gather each front-local RHS, run the unit-lower
// triangular solve, apply the dense APP trailing GEMV update, then scatter the
// full front-local RHS.
#[cfg(test)]
fn solve_forward_front_panels_like_native(panels: &[SolvePanel], factor_rhs: &mut [f64]) {
    let mut local_rhs = Vec::new();
    for panel in panels {
        let eliminated_len = panel.eliminated_len;
        let local_size = panel.row_positions.len();
        debug_assert!(eliminated_len <= local_size);
        debug_assert_eq!(panel.values.len(), local_size * eliminated_len);

        local_rhs.resize(local_size, 0.0);
        for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
            local_rhs[local_row] = factor_rhs[factor_position];
        }

        openblas_trsv_lower_unit_op_n_like_native(
            &panel.values,
            local_size,
            &mut local_rhs[..eliminated_len],
        );

        if local_size > eliminated_len {
            let (solved, trailing) = local_rhs.split_at_mut(eliminated_len);
            openblas_gemv_n_update_like_native(
                local_size - eliminated_len,
                eliminated_len,
                &panel.values[eliminated_len..],
                local_size,
                solved,
                trailing,
            );
        }

        for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
            factor_rhs[factor_position] = local_rhs[local_row];
        }
    }
}

// Mirrors OpenBLAS driver/level2/trsv_L.c for lower/unit/no-transpose on the
// local panel solve path used by SPRAL's ldlt_app_solve_fwd.
#[cfg(test)]
fn openblas_trsv_lower_unit_op_n_like_native(a: &[f64], lda: usize, x: &mut [f64]) {
    const DTB_ENTRIES: usize = 64;
    let n = x.len();
    for block_start in (0..n).step_by(DTB_ENTRIES) {
        let block_len = (n - block_start).min(DTB_ENTRIES);
        let block_end = block_start + block_len;

        for local_col in 0..block_len {
            let global_col = block_start + local_col;
            let pivot_value = x[global_col];
            let alpha = -pivot_value;
            if alpha == 0.0 {
                continue;
            }
            for row in (global_col + 1)..block_end {
                let coefficient = a[global_col * lda + row];
                x[row] = coefficient.mul_add(alpha, x[row]);
            }
        }

        if block_end < n {
            let (solved, trailing) = x.split_at_mut(block_end);
            openblas_gemv_n_update_like_native(
                n - block_end,
                block_len,
                &a[block_start * lda + block_end..],
                lda,
                &solved[block_start..block_end],
                trailing,
            );
        }
    }
}

// Mirrors OpenBLAS kernel/arm64/gemv_n.S for the OP_N solve update shape used
// by dtrsv_NLU and ldlt_app_solve_fwd: each source column updates all trailing
// rows before the next column is loaded.
#[cfg(test)]
#[allow(clippy::neg_multiply)]
fn openblas_gemv_n_update_like_native(
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &[f64],
    y: &mut [f64],
) {
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    for (col, &x_col) in x.iter().enumerate() {
        let alpha_x = -1.0f64 * x_col;
        let column_start = col * lda;
        for row in 0..m {
            y[row] = a[column_start + row].mul_add(alpha_x, y[row]);
        }
    }
}

// Mirrors OpenBLAS driver/level2/trsv_U.c for lower/unit/transpose on the
// local panel solve path used by SPRAL's ldlt_app_solve_bwd.
#[cfg(test)]
fn openblas_trsv_lower_unit_op_t_like_native(a: &[f64], lda: usize, x: &mut [f64]) {
    const DTB_ENTRIES: usize = 64;
    let n = x.len();
    let mut block_end = n;
    while block_end > 0 {
        let block_len = block_end.min(DTB_ENTRIES);
        let block_start = block_end - block_len;

        if block_end < n {
            for global_col in block_start..block_end {
                let column_start = global_col * lda;
                let dot = openblas_gemv_t_dot_like_contiguous(
                    &a[column_start + block_end..column_start + n],
                    &x[block_end..n],
                );
                x[global_col] = (-1.0f64).mul_add(dot, x[global_col]);
            }
        }

        for local_index in 0..block_len {
            let global_col = block_end - local_index - 1;
            let local_len = block_end - global_col - 1;
            if local_len > 0 {
                let column_start = global_col * lda + global_col + 1;
                let dot = openblas_dotu_like_contiguous(
                    &a[column_start..column_start + local_len],
                    &x[global_col + 1..block_end],
                );
                x[global_col] -= dot;
            }
        }

        block_end = block_start;
    }
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
    use std::ffi::OsString;
    use std::fs;
    use std::os::raw::c_int;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::ptr;
    use std::sync::OnceLock;

    use libloading::Library;
    #[cfg(unix)]
    use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
    use metis_ordering::{CsrGraph, Permutation};
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRng, TestRunner};

    use super::{
        APP_INNER_BLOCK_SIZE, DenseTppTailRequest, DenseUpdateBounds, FactorBlockRecord,
        NativeOrdering, NativeSpral, NumericFactorOptions, OrderingStrategy, PanelFactorStats,
        SolvePanel, SsidsError, SsidsOptions, SymmetricCscMatrix, analyse,
        app_adjust_passed_prefix, app_apply_accepted_prefix_update,
        app_apply_block_pivots_to_trailing_rows, app_build_ld_workspace,
        app_first_failed_trailing_column, app_restore_trailing_from_block_backup,
        app_solve_block_triangular_to_trailing_rows, app_truncate_records_to_prefix,
        app_two_by_two_inverse, app_update_one_by_one, app_update_two_by_two,
        build_symbolic_front_tree, dense_find_maxloc, dense_lower_offset,
        dense_symmetric_swap_with_workspace, expand_symmetric_pattern, factor_one_by_one_common,
        factor_two_by_two_common, factorize, factorize_dense_front,
        factorize_dense_tpp_tail_in_place, native_column_counts, native_postorder_permutation,
        openblas_gemv_n_update_like_native, openblas_gemv_t_dot_like_contiguous,
        openblas_trsv_lower_unit_op_n_like_native, openblas_trsv_lower_unit_op_t_like_native,
        permute_graph, reset_ldwork_column_tail, solve_forward_front_panels_like_native,
        solve_two_by_two_block_in_place, symbolic_factor_pattern, zero_dense_column_until,
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

        fn dyadic_kernel_value(
            &mut self,
            numerator_radius: i16,
            max_shift: u8,
            allow_signed_zero: bool,
        ) -> f64 {
            if allow_signed_zero && self.next_u64().is_multiple_of(32) {
                if self.next_u64() & 1 == 0 { 0.0 } else { -0.0 }
            } else {
                self.dyadic(numerator_radius, max_shift)
            }
        }

        fn nonzero_dyadic(&mut self, numerator_radius: i16, max_shift: u8) -> f64 {
            loop {
                let value = self.dyadic(numerator_radius, max_shift);
                if value != 0.0 {
                    return value;
                }
            }
        }
    }

    type ApplyPivotOpNFn =
        unsafe extern "C" fn(c_int, c_int, *const f64, *const f64, f64, *mut f64, c_int);
    type CheckThresholdOpNFn =
        unsafe extern "C" fn(c_int, c_int, c_int, c_int, f64, *mut f64, c_int) -> c_int;
    type CalcLdOpNFn =
        unsafe extern "C" fn(c_int, c_int, *const f64, c_int, *const f64, *mut f64, c_int);
    type HostTrsmRightLowerTransUnitFn =
        unsafe extern "C" fn(c_int, c_int, *const f64, c_int, *mut f64, c_int);
    type HostTrsvLowerUnitFn = unsafe extern "C" fn(c_int, *const f64, c_int, *mut f64);
    type GemvSolveUpdateFn =
        unsafe extern "C" fn(c_int, c_int, *const f64, c_int, *const f64, *mut f64);
    type HostGemmOpNOpTUpdateFn = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        *const f64,
        c_int,
        *const f64,
        c_int,
        *mut f64,
        c_int,
    );
    type LdltAppSolveFn =
        unsafe extern "C" fn(c_int, c_int, *const f64, c_int, c_int, *mut f64, c_int);
    type LdltAppSolveDiagFn = unsafe extern "C" fn(c_int, *const f64, c_int, *mut f64, c_int);
    type AlignLdaFn = unsafe extern "C" fn(c_int) -> c_int;
    type BlockUpdate1x1Fn = unsafe extern "C" fn(c_int, *mut f64, c_int, *const f64);
    type BlockUpdate2x2Fn = unsafe extern "C" fn(c_int, *mut f64, c_int, *const f64);
    type BlockSwapColsFn =
        unsafe extern "C" fn(c_int, c_int, c_int, *mut f64, c_int, *mut f64, *mut c_int);
    type BlockFindMaxlocFn =
        unsafe extern "C" fn(c_int, *const f64, c_int, *mut f64, *mut c_int, *mut c_int);
    type BlockTest2x2Fn = unsafe extern "C" fn(f64, f64, f64, *mut f64, *mut f64) -> c_int;
    type BlockTwoByTwoMultipliersFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, *mut f64);
    type BlockFirstStep32Fn = unsafe extern "C" fn(
        c_int,
        *mut c_int,
        *mut f64,
        c_int,
        *mut f64,
        *mut f64,
        c_int,
        f64,
        f64,
        *mut c_int,
        *mut c_int,
    ) -> c_int;
    type BlockPrefixTrace32Fn = unsafe extern "C" fn(
        c_int,
        *mut c_int,
        *mut f64,
        c_int,
        *mut f64,
        *mut f64,
        c_int,
        f64,
        f64,
        *mut c_int,
        c_int,
        *mut c_int,
        *mut c_int,
        *mut c_int,
        *mut c_int,
        *mut c_int,
        *mut f64,
        *mut f64,
        *mut f64,
    ) -> c_int;
    type BlockLdlt32Fn = unsafe extern "C" fn(
        c_int,
        *mut c_int,
        *mut f64,
        c_int,
        *mut f64,
        *mut f64,
        c_int,
        f64,
        f64,
        *mut c_int,
    );
    type LdltTppFactorFn = unsafe extern "C" fn(
        c_int,
        c_int,
        *mut c_int,
        *mut f64,
        c_int,
        *mut f64,
        *mut f64,
        c_int,
        c_int,
        f64,
        f64,
        c_int,
        *mut f64,
        c_int,
    ) -> c_int;

    struct NativeKernelShim {
        _libspral: Library,
        _library: Library,
        apply_pivot_op_n: ApplyPivotOpNFn,
        check_threshold_op_n: CheckThresholdOpNFn,
        calc_ld_op_n: CalcLdOpNFn,
        host_trsm_right_lower_trans_unit: HostTrsmRightLowerTransUnitFn,
        host_trsv_lower_unit_op_n: HostTrsvLowerUnitFn,
        host_trsv_lower_unit_op_t: HostTrsvLowerUnitFn,
        gemv_op_n_solve_update: GemvSolveUpdateFn,
        gemv_op_t_solve_update: GemvSolveUpdateFn,
        host_gemm_op_n_op_t_update: HostGemmOpNOpTUpdateFn,
        ldlt_app_solve_fwd: LdltAppSolveFn,
        ldlt_app_solve_diag: LdltAppSolveDiagFn,
        ldlt_app_solve_bwd: LdltAppSolveFn,
        align_lda_double: AlignLdaFn,
        block_update_1x1_32: BlockUpdate1x1Fn,
        block_update_2x2_32: BlockUpdate2x2Fn,
        block_swap_cols_32: BlockSwapColsFn,
        block_find_maxloc_32: BlockFindMaxlocFn,
        block_test_2x2: BlockTest2x2Fn,
        block_two_by_two_multipliers: BlockTwoByTwoMultipliersFn,
        block_first_step_32: BlockFirstStep32Fn,
        block_prefix_trace_32: BlockPrefixTrace32Fn,
        block_prefix_trace_32_source: BlockPrefixTrace32Fn,
        block_ldlt_32: BlockLdlt32Fn,
        ldlt_tpp_factor: LdltTppFactorFn,
    }

    static NATIVE_KERNEL_SHIM: OnceLock<Result<NativeKernelShim, String>> = OnceLock::new();

    fn native_kernel_shim_or_skip() -> Option<&'static NativeKernelShim> {
        match NATIVE_KERNEL_SHIM.get_or_init(build_native_kernel_shim) {
            Ok(shim) => Some(shim),
            Err(error) => {
                if std::env::var_os("AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY").is_some()
                    || std::env::var_os("AD_CODEGEN_REQUIRE_SPRAL_UPSTREAM_SOURCE").is_some()
                {
                    panic!("native SPRAL kernel parity shim is required: {error}");
                }
                eprintln!("skipping native SPRAL kernel parity tests: {error}");
                None
            }
        }
    }

    fn native_spral_or_skip() -> Option<NativeSpral> {
        match NativeSpral::load() {
            Ok(native) => Some(native),
            Err(error) => {
                if std::env::var_os("AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY").is_some() {
                    panic!("native SPRAL is required for fail-closed parity runs: {error}");
                }
                eprintln!("skipping native SPRAL parity test: {error}");
                None
            }
        }
    }

    fn build_native_kernel_shim() -> Result<NativeKernelShim, String> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| "spral_ssids manifest has no parent".to_string())?
            .to_path_buf();
        let ssids_source = std::env::var_os("SPRAL_UPSTREAM_SSIDS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo_root.join("target/native/spral-upstream/src/ssids"));
        if !ssids_source.is_dir() {
            return Err(format!(
                "SPRAL source anchor missing: {}",
                ssids_source.display()
            ));
        }
        let upstream_src = ssids_source
            .parent()
            .ok_or_else(|| format!("invalid SPRAL source anchor: {}", ssids_source.display()))?;

        let libspral = std::env::var_os("SPRAL_SSIDS_NATIVE_LIB")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/Users/greg/local/ipopt-spral/lib/libspral.dylib"));
        if !libspral.is_file() {
            return Err(format!(
                "native SPRAL library missing: {}",
                libspral.display()
            ));
        }
        let libspral_dir = libspral
            .parent()
            .ok_or_else(|| format!("invalid native SPRAL library path: {}", libspral.display()))?;

        let out_dir = repo_root.join("target/spral-kernel-parity-shim");
        fs::create_dir_all(&out_dir)
            .map_err(|error| format!("failed to create {}: {error}", out_dir.display()))?;
        let source_path = out_dir.join("spral_kernel_shim.cpp");
        let library_path = out_dir.join(dynamic_library_name("spral_kernel_shim"));
        fs::write(&source_path, native_kernel_shim_source()).map_err(|error| {
            format!(
                "failed to write native kernel shim source {}: {error}",
                source_path.display()
            )
        })?;

        let cxx = std::env::var_os("CXX").unwrap_or_else(|| OsString::from("c++"));
        let mut command = Command::new(&cxx);
        command
            .arg("-std=c++17")
            .arg("-ffp-contract=off")
            .arg(dynamic_library_flag())
            .arg("-fPIC")
            .arg("-I")
            .arg(upstream_src)
            .arg(&source_path)
            .arg("-L")
            .arg(libspral_dir)
            .arg("-lspral")
            .arg(format!("-Wl,-rpath,{}", libspral_dir.display()))
            .arg("-o")
            .arg(&library_path);
        let output = command.output().map_err(|error| {
            format!(
                "failed to run C++ compiler `{}`: {error}",
                Path::new(&cxx).display()
            )
        })?;
        if !output.status.success() {
            return Err(format!(
                "native kernel shim compile failed with status {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let libspral_library = load_library_global(&libspral, "native SPRAL library")?;
        let library = unsafe {
            Library::new(&library_path).map_err(|error| {
                format!(
                    "failed to load native kernel shim {}: {error}",
                    library_path.display()
                )
            })?
        };
        let apply_pivot_op_n = unsafe {
            *library
                .get::<ApplyPivotOpNFn>(b"spral_kernel_apply_pivot_op_n\0")
                .map_err(|error| format!("failed to load apply_pivot OP_N shim: {error}"))?
        };
        let check_threshold_op_n = unsafe {
            *library
                .get::<CheckThresholdOpNFn>(b"spral_kernel_check_threshold_op_n\0")
                .map_err(|error| format!("failed to load check_threshold OP_N shim: {error}"))?
        };
        let calc_ld_op_n = unsafe {
            *library
                .get::<CalcLdOpNFn>(b"spral_kernel_calc_ld_op_n\0")
                .map_err(|error| format!("failed to load calcLD OP_N shim: {error}"))?
        };
        let host_trsm_right_lower_trans_unit = unsafe {
            *library
                .get::<HostTrsmRightLowerTransUnitFn>(
                    b"spral_kernel_host_trsm_right_lower_trans_unit\0",
                )
                .map_err(|error| {
                    format!("failed to load host_trsm right/lower/trans/unit shim: {error}")
                })?
        };
        let host_trsv_lower_unit_op_n = unsafe {
            *library
                .get::<HostTrsvLowerUnitFn>(b"spral_kernel_host_trsv_lower_unit_op_n\0")
                .map_err(|error| {
                    format!("failed to load host_trsv lower/unit OP_N shim: {error}")
                })?
        };
        let host_trsv_lower_unit_op_t = unsafe {
            *library
                .get::<HostTrsvLowerUnitFn>(b"spral_kernel_host_trsv_lower_unit_op_t\0")
                .map_err(|error| {
                    format!("failed to load host_trsv lower/unit OP_T shim: {error}")
                })?
        };
        let gemv_op_n_solve_update = unsafe {
            *library
                .get::<GemvSolveUpdateFn>(b"spral_kernel_gemv_op_n_solve_update\0")
                .map_err(|error| format!("failed to load gemv OP_N solve update shim: {error}"))?
        };
        let gemv_op_t_solve_update = unsafe {
            *library
                .get::<GemvSolveUpdateFn>(b"spral_kernel_gemv_op_t_solve_update\0")
                .map_err(|error| format!("failed to load gemv OP_T solve update shim: {error}"))?
        };
        let host_gemm_op_n_op_t_update = unsafe {
            *library
                .get::<HostGemmOpNOpTUpdateFn>(b"spral_kernel_host_gemm_op_n_op_t_update\0")
                .map_err(|error| {
                    format!("failed to load host_gemm OP_N/OP_T update shim: {error}")
                })?
        };
        let ldlt_app_solve_fwd = unsafe {
            *library
                .get::<LdltAppSolveFn>(b"spral_kernel_ldlt_app_solve_fwd\0")
                .map_err(|error| format!("failed to load ldlt_app_solve_fwd shim: {error}"))?
        };
        let ldlt_app_solve_diag = unsafe {
            *library
                .get::<LdltAppSolveDiagFn>(b"spral_kernel_ldlt_app_solve_diag\0")
                .map_err(|error| format!("failed to load ldlt_app_solve_diag shim: {error}"))?
        };
        let ldlt_app_solve_bwd = unsafe {
            *library
                .get::<LdltAppSolveFn>(b"spral_kernel_ldlt_app_solve_bwd\0")
                .map_err(|error| format!("failed to load ldlt_app_solve_bwd shim: {error}"))?
        };
        let align_lda_double = unsafe {
            *library
                .get::<AlignLdaFn>(b"spral_kernel_align_lda_double\0")
                .map_err(|error| format!("failed to load align_lda<double> shim: {error}"))?
        };
        let block_update_1x1_32 = unsafe {
            *library
                .get::<BlockUpdate1x1Fn>(b"spral_kernel_block_update_1x1_32\0")
                .map_err(|error| format!("failed to load block_update_1x1_32 shim: {error}"))?
        };
        let block_update_2x2_32 = unsafe {
            *library
                .get::<BlockUpdate2x2Fn>(b"spral_kernel_block_update_2x2_32\0")
                .map_err(|error| format!("failed to load block_update_2x2_32 shim: {error}"))?
        };
        let block_swap_cols_32 = unsafe {
            *library
                .get::<BlockSwapColsFn>(b"spral_kernel_block_swap_cols_32\0")
                .map_err(|error| format!("failed to load block_swap_cols_32 shim: {error}"))?
        };
        let block_find_maxloc_32 = unsafe {
            *library
                .get::<BlockFindMaxlocFn>(b"spral_kernel_block_find_maxloc_32\0")
                .map_err(|error| format!("failed to load block_find_maxloc_32 shim: {error}"))?
        };
        let block_test_2x2 = unsafe {
            *library
                .get::<BlockTest2x2Fn>(b"spral_kernel_block_test_2x2\0")
                .map_err(|error| format!("failed to load block_test_2x2 shim: {error}"))?
        };
        let block_two_by_two_multipliers = unsafe {
            *library
                .get::<BlockTwoByTwoMultipliersFn>(b"spral_kernel_block_two_by_two_multipliers\0")
                .map_err(|error| {
                    format!("failed to load block_two_by_two_multipliers shim: {error}")
                })?
        };
        let block_first_step_32 = unsafe {
            *library
                .get::<BlockFirstStep32Fn>(b"spral_kernel_block_first_step_32\0")
                .map_err(|error| format!("failed to load block_first_step_32 shim: {error}"))?
        };
        let block_prefix_trace_32 = unsafe {
            *library
                .get::<BlockPrefixTrace32Fn>(b"spral_kernel_block_prefix_trace_32\0")
                .map_err(|error| format!("failed to load block_prefix_trace_32 shim: {error}"))?
        };
        let block_prefix_trace_32_source = unsafe {
            *library
                .get::<BlockPrefixTrace32Fn>(b"spral_kernel_block_prefix_trace_32_source\0")
                .map_err(|error| {
                    format!("failed to load source block_prefix_trace_32 shim: {error}")
                })?
        };
        let block_ldlt_32 = unsafe {
            *library
                .get::<BlockLdlt32Fn>(b"spral_kernel_block_ldlt_32\0")
                .map_err(|error| format!("failed to load block_ldlt_32 shim: {error}"))?
        };
        let ldlt_tpp_factor = unsafe {
            *library
                .get::<LdltTppFactorFn>(b"spral_kernel_ldlt_tpp_factor\0")
                .map_err(|error| format!("failed to load ldlt_tpp_factor shim: {error}"))?
        };

        Ok(NativeKernelShim {
            _libspral: libspral_library,
            _library: library,
            apply_pivot_op_n,
            check_threshold_op_n,
            calc_ld_op_n,
            host_trsm_right_lower_trans_unit,
            host_trsv_lower_unit_op_n,
            host_trsv_lower_unit_op_t,
            gemv_op_n_solve_update,
            gemv_op_t_solve_update,
            host_gemm_op_n_op_t_update,
            ldlt_app_solve_fwd,
            ldlt_app_solve_diag,
            ldlt_app_solve_bwd,
            align_lda_double,
            block_update_1x1_32,
            block_update_2x2_32,
            block_swap_cols_32,
            block_find_maxloc_32,
            block_test_2x2,
            block_two_by_two_multipliers,
            block_first_step_32,
            block_prefix_trace_32,
            block_prefix_trace_32_source,
            block_ldlt_32,
            ldlt_tpp_factor,
        })
    }

    #[cfg(unix)]
    fn load_library_global(path: &Path, label: &str) -> Result<Library, String> {
        let library = unsafe {
            UnixLibrary::open(Some(path.as_os_str()), RTLD_NOW | RTLD_GLOBAL).map_err(|error| {
                format!(
                    "failed to load {label} {} with global symbols: {error}",
                    path.display()
                )
            })?
        };
        Ok(Library::from(library))
    }

    #[cfg(not(unix))]
    fn load_library_global(path: &Path, label: &str) -> Result<Library, String> {
        let library = unsafe {
            Library::new(path)
                .map_err(|error| format!("failed to load {label} {}: {error}", path.display()))?
        };
        Ok(library)
    }

    #[cfg(target_os = "macos")]
    fn dynamic_library_name(stem: &str) -> String {
        format!("lib{stem}.dylib")
    }

    #[cfg(not(target_os = "macos"))]
    fn dynamic_library_name(stem: &str) -> String {
        format!("lib{stem}.so")
    }

    #[cfg(target_os = "macos")]
    fn dynamic_library_flag() -> &'static str {
        "-dynamiclib"
    }

    #[cfg(not(target_os = "macos"))]
    fn dynamic_library_flag() -> &'static str {
        "-shared"
    }

    fn native_kernel_shim_source() -> &'static str {
        r#"
#include "ssids/cpu/kernels/common.hxx"
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/block_ldlt.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"
#include <cmath>

namespace spral { namespace ssids { namespace cpu {
template <enum operation op, typename T>
void calcLD(int m, int n, T const* l, int ldl, T const* d, T* ld, int ldld);

namespace ldlt_app_internal {
template <enum operation op, typename T>
void apply_pivot(int m, int n, int from, const T *diag, const T *d, const T small, T* aval, int lda);
}
}}}

extern "C" void spral_kernel_apply_pivot_op_n(
      int m, int n, const double* diag, const double* d,
      double small, double* aval, int lda) {
   spral::ssids::cpu::ldlt_app_internal::apply_pivot<spral::ssids::cpu::OP_N, double>(
         m, n, 0, diag, d, small, aval, lda);
}

extern "C" int spral_kernel_check_threshold_op_n(
      int rfrom, int rto, int cfrom, int cto, double u, double* aval, int lda) {
   // Mirrors ldlt_app.cxx::check_threshold<OP_N>.
   for(int j=cfrom; j<cto; j++)
   for(int i=rfrom; i<rto; i++)
      if(fabs(aval[j*lda+i]) > 1.0/u)
         return j;
   return cto;
}

extern "C" void spral_kernel_calc_ld_op_n(
      int m, int n, const double* l, int ldl, const double* d,
      double* ld, int ldld) {
   spral::ssids::cpu::calcLD<spral::ssids::cpu::OP_N, double>(
         m, n, l, ldl, d, ld, ldld);
}

extern "C" void spral_kernel_host_trsm_right_lower_trans_unit(
      int m, int n, const double* a, int lda, double* b, int ldb) {
   spral::ssids::cpu::host_trsm<double>(
         spral::ssids::cpu::SIDE_RIGHT,
         spral::ssids::cpu::FILL_MODE_LWR,
         spral::ssids::cpu::OP_T,
         spral::ssids::cpu::DIAG_UNIT,
         m, n, 1.0, a, lda, b, ldb);
}

extern "C" void spral_kernel_host_trsv_lower_unit_op_n(
      int n, const double* a, int lda, double* x) {
   spral::ssids::cpu::host_trsv<double>(
         spral::ssids::cpu::FILL_MODE_LWR,
         spral::ssids::cpu::OP_N,
         spral::ssids::cpu::DIAG_UNIT,
         n, a, lda, x, 1);
}

extern "C" void spral_kernel_host_trsv_lower_unit_op_t(
      int n, const double* a, int lda, double* x) {
   spral::ssids::cpu::host_trsv<double>(
         spral::ssids::cpu::FILL_MODE_LWR,
         spral::ssids::cpu::OP_T,
         spral::ssids::cpu::DIAG_UNIT,
         n, a, lda, x, 1);
}

extern "C" void spral_kernel_gemv_op_n_solve_update(
      int m, int n, const double* a, int lda, const double* x, double* y) {
   spral::ssids::cpu::gemv<double>(
         spral::ssids::cpu::OP_N,
         m, n, -1.0, a, lda, x, 1, 1.0, y, 1);
}

extern "C" void spral_kernel_gemv_op_t_solve_update(
      int m, int n, const double* a, int lda, const double* x, double* y) {
   spral::ssids::cpu::gemv<double>(
         spral::ssids::cpu::OP_T,
         m, n, -1.0, a, lda, x, 1, 1.0, y, 1);
}

extern "C" void spral_kernel_host_gemm_op_n_op_t_update(
      int m, int n, int k, const double* a, int lda,
      const double* b, int ldb, double* c, int ldc) {
   spral::ssids::cpu::host_gemm<double>(
         spral::ssids::cpu::OP_N,
         spral::ssids::cpu::OP_T,
         m, n, k, -1.0, a, lda, b, ldb, 1.0, c, ldc);
}

extern "C" void spral_kernel_ldlt_app_solve_fwd(
      int m, int n, const double* l, int ldl, int nrhs, double* x, int ldx) {
   spral::ssids::cpu::ldlt_app_solve_fwd<double>(
         m, n, l, ldl, nrhs, x, ldx);
}

extern "C" void spral_kernel_ldlt_app_solve_diag(
      int n, const double* d, int nrhs, double* x, int ldx) {
   spral::ssids::cpu::ldlt_app_solve_diag<double>(
         n, d, nrhs, x, ldx);
}

extern "C" void spral_kernel_ldlt_app_solve_bwd(
      int m, int n, const double* l, int ldl, int nrhs, double* x, int ldx) {
   spral::ssids::cpu::ldlt_app_solve_bwd<double>(
         m, n, l, ldl, nrhs, x, ldx);
}

extern "C" int spral_kernel_align_lda_double(int lda) {
   return static_cast<int>(spral::ssids::cpu::align_lda<double>(lda));
}

extern "C" void spral_kernel_block_ldlt_32(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm) {
   spral::ssids::cpu::block_ldlt<double, 32>(
         from, perm, a, lda, d, ldwork, action != 0, u, small, lperm);
}

extern "C" int spral_kernel_ldlt_tpp_factor(
      int m, int n, int* perm, double* a, int lda, double* d,
      double* ld, int ldld, int action, double u, double small,
      int nleft, double* aleft, int ldleft) {
   return spral::ssids::cpu::ldlt_tpp_factor(
         m, n, perm, a, lda, d, ld, ldld, action != 0, u, small,
         nleft, aleft, ldleft);
}

extern "C" void spral_kernel_block_update_1x1_32(
      int p, double* a, int lda, const double* ld) {
   spral::ssids::cpu::block_ldlt_internal::update_1x1<double, 32>(
         p, a, lda, ld);
}

extern "C" void spral_kernel_block_update_2x2_32(
      int p, double* a, int lda, const double* ld) {
   for(int c=p+2; c<32; c++) {
      #pragma omp simd
      for(int r=c; r<32; r++) {
         double combined = std::fma(ld[c], a[p*lda+r], ld[32+c]*a[(p+1)*lda+r]);
         a[c*lda+r] -= combined;
      }
   }
}

extern "C" void spral_kernel_block_swap_cols_32(
      int idx1, int idx2, int n, double* a, int lda, double* ldwork, int* perm) {
   spral::ssids::cpu::block_ldlt_internal::swap_cols<double, 32>(
         idx1, idx2, n, a, lda, ldwork, perm);
}

extern "C" void spral_kernel_block_find_maxloc_32(
      int from, const double* a, int lda, double* bestv, int* rloc, int* cloc) {
   spral::ssids::cpu::block_ldlt_internal::find_maxloc<double, 32>(
         from, a, lda, *bestv, *rloc, *cloc);
}

extern "C" int spral_kernel_block_test_2x2(
      double a11, double a21, double a22, double* detpiv, double* detscale) {
   double local_detpiv = 0.0;
   double local_detscale = 0.0;
   bool accepted = spral::ssids::cpu::block_ldlt_internal::test_2x2<double>(
         a11, a21, a22, local_detpiv, local_detscale);
   *detpiv = local_detpiv;
   *detscale = local_detscale;
   return accepted ? 1 : 0;
}

extern "C" void spral_kernel_block_two_by_two_multipliers(
      double d11, double d21, double d22, double work0, double work1,
      double* out) {
   out[0] = std::fma(d21, work1, d11*work0);
   out[1] = std::fma(d21, work0, d22*work1);
}

extern "C" int spral_kernel_block_first_step_32(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int* next_p) {
   using namespace spral::ssids::cpu::block_ldlt_internal;

   int p = from;
   double bestv;
   int t, m;
   find_maxloc<double, 32>(p, a, lda, bestv, t, m);

   if(fabs(bestv) < small) {
      if(!action) return -1;
      for(; p<32; ) {
         d[2*p] = 0.0; d[2*p+1] = 0.0;
         for(int r=p; r<32; r++)
            a[p*lda+r] = 0.0;
         for(int r=p; r<32; r++)
            ldwork[p*32+r] = 0.0;
         p++;
      }
      *next_p = p;
      return 0;
   }

   int pivsiz = 0;
   double a11, a21 = 0.0, a22 = 0.0, detscale = 0.0, detpiv = 0.0;
   if(t==m) {
      a11 = a[t*lda+t];
      pivsiz = 1;
   } else {
      a11 = a[m*lda+m];
      a22 = a[t*lda+t];
      a21 = a[m*lda+t];
      if(test_2x2<double>(a11, a21, a22, detpiv, detscale)) {
         pivsiz = 2;
      } else {
         if(fabs(a11) > fabs(a22)) {
            pivsiz = 1;
            t = m;
            if(fabs(a11 / a21) < u) pivsiz = 0;
         } else {
            pivsiz = 1;
            a11 = a22;
            m = t;
            if(fabs(a22 / a21) < u) pivsiz = 0;
         }
      }
   }

   if(pivsiz == 0) return -2;
   if(pivsiz == 1) {
      double d11 = 1.0/a11;
      swap_cols<double, 32>(p, t, 32, a, lda, ldwork, perm);
      if(lperm) { int temp=lperm[p]; lperm[p]=lperm[t]; lperm[t]=temp; }
      double *work = &ldwork[p*32];
      for(int r=p+1; r<32; r++) {
         work[r] = a[p*lda+r];
         a[p*lda+r] *= d11;
      }
      update_1x1<double, 32>(p, a, lda, work);
      d[2*p] = d11;
      d[2*p+1] = 0.0;
      a[p*lda+p] = 1.0;
   } else {
      swap_cols<double, 32>(p, m, 32, a, lda, ldwork, perm);
      if(lperm) { int temp=lperm[p]; lperm[p]=lperm[m]; lperm[m]=temp; }
      swap_cols<double, 32>(p+1, t, 32, a, lda, ldwork, perm);
      if(lperm) { int temp=lperm[p+1]; lperm[p+1]=lperm[t]; lperm[t]=temp; }
      double d11 = (a22*detscale)/detpiv;
      double d22 = (a11*detscale)/detpiv;
      double d21 = (-a21*detscale)/detpiv;
      double *work = &ldwork[p*32];
      for(int r=p+2; r<32; r++) {
         work[r] = a[p*lda+r];
         work[32+r] = a[(p+1)*lda+r];
         a[p*lda+r] = std::fma(d21, work[32+r], d11*work[r]);
         a[(p+1)*lda+r] = std::fma(d21, work[r], d22*work[32+r]);
      }
      spral_kernel_block_update_2x2_32(p, a, lda, work);
      d[2*p] = d11;
      d[2*p+1] = d21;
      d[2*p+2] = std::numeric_limits<double>::infinity();
      d[2*p+3] = d22;
      a[p*(lda+1)] = 1.0;
      a[p*(lda+1)+1] = 0.0;
      a[(p+1)*(lda+1)] = 1.0;
   }

   p += pivsiz;
   *next_p = p;
   return pivsiz;
}

static void spral_kernel_record_block_prefix_trace_32(
      int step, int from, int status, int next, const int* perm,
      const int* lperm, const double* a, int lda, const double* d,
      const double* ldwork, int* trace_from, int* trace_status,
      int* trace_next, int* trace_perm, int* trace_lperm,
      double* trace_matrix, double* trace_ldwork, double* trace_d) {
   trace_from[step] = from;
   trace_status[step] = status;
   trace_next[step] = next;
   for(int i=0; i<32; i++) {
      trace_perm[step*32+i] = perm[i];
      trace_lperm[step*32+i] = lperm ? lperm[i] : i;
   }
   for(int c=0; c<32; c++) {
      for(int r=0; r<32; r++) {
         trace_matrix[step*32*32 + c*32+r] = (r >= c) ? a[c*lda+r] : 0.0;
         trace_ldwork[step*32*32 + c*32+r] = ldwork[c*32+r];
      }
   }
   for(int i=0; i<64; i++)
      trace_d[step*64+i] = d[i];
}

static int spral_kernel_block_prefix_trace_32_impl(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int max_steps,
      int* trace_from, int* trace_status, int* trace_next, int* trace_perm,
      int* trace_lperm, double* trace_matrix, double* trace_ldwork,
      double* trace_d, bool source_plain) {
   using namespace spral::ssids::cpu::block_ldlt_internal;

   int trace_len = 0;
   for(int p=from; p<32; ) {
      if(trace_len >= max_steps) return trace_len;
      int step_from = p;

      double bestv;
      int t, m;
      find_maxloc<double, 32>(p, a, lda, bestv, t, m);

      if(fabs(bestv) < small) {
         if(!action) {
            spral_kernel_record_block_prefix_trace_32(
                  trace_len, step_from, -1, p, perm, lperm, a, lda, d,
                  ldwork, trace_from, trace_status, trace_next, trace_perm,
                  trace_lperm, trace_matrix, trace_ldwork, trace_d);
            return trace_len + 1;
         }
         for(; p<32; ) {
            d[2*p] = 0.0; d[2*p+1] = 0.0;
            for(int r=p; r<32; r++)
               a[p*lda+r] = 0.0;
            for(int r=p; r<32; r++)
               ldwork[p*32+r] = 0.0;
            p++;
         }
         spral_kernel_record_block_prefix_trace_32(
               trace_len, step_from, 0, p, perm, lperm, a, lda, d,
               ldwork, trace_from, trace_status, trace_next, trace_perm,
               trace_lperm, trace_matrix, trace_ldwork, trace_d);
         return trace_len + 1;
      }

      int pivsiz = 0;
      double a11, a21 = 0.0, a22 = 0.0, detscale = 0.0, detpiv = 0.0;
      if(t==m) {
         a11 = a[t*lda+t];
         pivsiz = 1;
      } else {
         a11 = a[m*lda+m];
         a22 = a[t*lda+t];
         a21 = a[m*lda+t];
         if(test_2x2<double>(a11, a21, a22, detpiv, detscale)) {
            pivsiz = 2;
         } else {
            if(fabs(a11) > fabs(a22)) {
               pivsiz = 1;
               t = m;
               if(fabs(a11 / a21) < u) pivsiz = 0;
            } else {
               pivsiz = 1;
               a11 = a22;
               m = t;
               if(fabs(a22 / a21) < u) pivsiz = 0;
            }
         }
      }

      if(pivsiz == 0) {
         spral_kernel_record_block_prefix_trace_32(
               trace_len, step_from, -2, p, perm, lperm, a, lda, d,
               ldwork, trace_from, trace_status, trace_next, trace_perm,
               trace_lperm, trace_matrix, trace_ldwork, trace_d);
         return trace_len + 1;
      }
      if(pivsiz == 1) {
         double d11 = 1.0/a11;
         swap_cols<double, 32>(p, t, 32, a, lda, ldwork, perm);
         if(lperm) { int temp=lperm[p]; lperm[p]=lperm[t]; lperm[t]=temp; }
         double *work = &ldwork[p*32];
         for(int r=p+1; r<32; r++) {
            work[r] = a[p*lda+r];
            a[p*lda+r] *= d11;
         }
         update_1x1<double, 32>(p, a, lda, work);
         d[2*p] = d11;
         d[2*p+1] = 0.0;
         a[p*lda+p] = 1.0;
      } else {
         swap_cols<double, 32>(p, m, 32, a, lda, ldwork, perm);
         if(lperm) { int temp=lperm[p]; lperm[p]=lperm[m]; lperm[m]=temp; }
         swap_cols<double, 32>(p+1, t, 32, a, lda, ldwork, perm);
         if(lperm) { int temp=lperm[p+1]; lperm[p+1]=lperm[t]; lperm[t]=temp; }
         double d11 = (a22*detscale)/detpiv;
         double d22 = (a11*detscale)/detpiv;
         double d21 = (-a21*detscale)/detpiv;
         double *work = &ldwork[p*32];
         for(int r=p+2; r<32; r++) {
            work[r] = a[p*lda+r];
            work[32+r] = a[(p+1)*lda+r];
            if(source_plain) {
               a[p*lda+r] = d11*work[r] + d21*work[32+r];
               a[(p+1)*lda+r] = d21*work[r] + d22*work[32+r];
            } else {
               a[p*lda+r] = std::fma(d21, work[32+r], d11*work[r]);
               a[(p+1)*lda+r] = std::fma(d21, work[r], d22*work[32+r]);
            }
         }
         if(source_plain) {
            update_2x2<double, 32>(p, a, lda, work);
         } else {
            spral_kernel_block_update_2x2_32(p, a, lda, work);
         }
         d[2*p] = d11;
         d[2*p+1] = d21;
         d[2*p+2] = std::numeric_limits<double>::infinity();
         d[2*p+3] = d22;
         a[p*(lda+1)] = 1.0;
         a[p*(lda+1)+1] = 0.0;
         a[(p+1)*(lda+1)] = 1.0;
      }

      p += pivsiz;
      spral_kernel_record_block_prefix_trace_32(
            trace_len, step_from, pivsiz, p, perm, lperm, a, lda, d,
            ldwork, trace_from, trace_status, trace_next, trace_perm,
            trace_lperm, trace_matrix, trace_ldwork, trace_d);
      trace_len++;
   }
   return trace_len;
}

extern "C" int spral_kernel_block_prefix_trace_32(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int max_steps,
      int* trace_from, int* trace_status, int* trace_next, int* trace_perm,
      int* trace_lperm, double* trace_matrix, double* trace_ldwork,
      double* trace_d) {
   return spral_kernel_block_prefix_trace_32_impl(
         from, perm, a, lda, d, ldwork, action, u, small, lperm, max_steps,
         trace_from, trace_status, trace_next, trace_perm, trace_lperm,
         trace_matrix, trace_ldwork, trace_d, false);
}

extern "C" int spral_kernel_block_prefix_trace_32_source(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int max_steps,
      int* trace_from, int* trace_status, int* trace_next, int* trace_perm,
      int* trace_lperm, double* trace_matrix, double* trace_ldwork,
      double* trace_d) {
   return spral_kernel_block_prefix_trace_32_impl(
         from, perm, a, lda, d, ldwork, action, u, small, lperm, max_steps,
         trace_from, trace_status, trace_next, trace_perm, trace_lperm,
         trace_matrix, trace_ldwork, trace_d, true);
}
"#
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

    fn deterministic_kernel_runner(cases: usize, seed: u64) -> TestRunner {
        TestRunner::new_with_rng(
            Config {
                cases: cases as u32,
                failure_persistence: None,
                rng_seed: RngSeed::Fixed(seed),
                ..Config::default()
            },
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        )
    }

    #[derive(Clone, Debug)]
    struct AppKernelCase {
        seed: u64,
        size: usize,
        block_start: usize,
        block_end: usize,
        matrix: Vec<f64>,
        block_records: Vec<FactorBlockRecord>,
        d_values: Vec<f64>,
        small: f64,
    }

    #[derive(Clone, Debug)]
    struct AppSolveKernelCase {
        seed: u64,
        rows: usize,
        eliminated_len: usize,
        lower: Vec<f64>,
        diagonal: Vec<f64>,
        rhs: Vec<f64>,
    }

    #[derive(Clone, Debug)]
    struct BlockLdltKernelResult {
        perm: Vec<usize>,
        local_perm: Vec<usize>,
        matrix: Vec<f64>,
        diagonal: Vec<f64>,
    }

    #[derive(Clone, Debug)]
    struct DenseTppKernelResult {
        eliminated: usize,
        perm: Vec<usize>,
        matrix: Vec<f64>,
        diagonal: Vec<f64>,
    }

    #[derive(Clone, Debug)]
    struct BlockUpdateCase {
        seed: u64,
        pivot: usize,
        matrix: Vec<f64>,
        workspace: Vec<f64>,
    }

    #[derive(Clone, Debug)]
    struct BlockSwapCase {
        seed: u64,
        lhs: usize,
        rhs: usize,
        n: usize,
        matrix: Vec<f64>,
        workspace: Vec<f64>,
        perm: Vec<usize>,
    }

    #[derive(Clone, Debug)]
    struct BlockFindMaxlocCase {
        seed: u64,
        from: usize,
        matrix: Vec<f64>,
    }

    #[derive(Clone, Debug)]
    struct BlockFirstStepCase {
        seed: u64,
        from: usize,
        matrix: Vec<f64>,
        perm: Vec<usize>,
        local_perm: Vec<usize>,
        options: NumericFactorOptions,
    }

    struct BlockFirstStepState<'a> {
        matrix: &'a [f64],
        workspace: &'a [f64],
        diagonal: &'a [f64],
    }

    #[derive(Clone, Debug)]
    struct BlockPrefixSnapshot {
        step: usize,
        from: usize,
        status: i32,
        next: usize,
        perm: Vec<usize>,
        local_perm: Vec<usize>,
        matrix: Vec<f64>,
        workspace: Vec<f64>,
        diagonal: Vec<f64>,
    }

    struct RustAppBlockStepState<'a> {
        rows: &'a mut [usize],
        matrix: &'a mut [f64],
        size: usize,
        workspace: &'a mut [f64],
        diagonal: &'a mut [f64],
        local_perm: &'a mut [usize],
    }

    struct BlockPrefixSnapshotInput<'a> {
        step: usize,
        from: usize,
        status: i32,
        next: usize,
        perm: &'a [usize],
        local_perm: &'a [usize],
        matrix: (&'a [f64], usize),
        workspace: (&'a [f64], usize),
        diagonal: &'a [f64],
    }

    #[derive(Clone, Copy, Debug)]
    struct AppKernelCaseOptions {
        allow_two_by_two: bool,
        allow_signed_zero: bool,
    }

    fn app_kernel_case_from_seed(seed: u64, options: AppKernelCaseOptions) -> AppKernelCase {
        app_kernel_case_from_seed_with_limits(seed, options, 8, 18)
    }

    fn app_kernel_case_from_seed_with_limits(
        seed: u64,
        options: AppKernelCaseOptions,
        max_block_width: usize,
        max_trailing_rows: usize,
    ) -> AppKernelCase {
        let mut rng = DenseBoundaryRng::new(seed);
        let block_start = rng.usize_inclusive(0, 4);
        let block_width = rng.usize_inclusive(1, max_block_width);
        let trailing_rows = rng.usize_inclusive(1, max_trailing_rows);
        let block_end = block_start + block_width;
        let size = block_end + trailing_rows;
        let mut matrix = vec![0.0; size * size];

        for col in block_start..block_end {
            matrix[col * size + col] = 1.0;
            for row in (col + 1)..block_end {
                matrix[col * size + row] =
                    rng.dyadic_kernel_value(12, 5, options.allow_signed_zero);
            }
            for row in block_end..size {
                matrix[col * size + row] =
                    rng.dyadic_kernel_value(18, 6, options.allow_signed_zero);
            }
        }
        for col in block_end..size {
            for row in col..size {
                matrix[col * size + row] =
                    rng.dyadic_kernel_value(12, 6, options.allow_signed_zero);
            }
        }

        let mut block_records = Vec::new();
        let mut d_values = vec![0.0; 2 * block_width];
        let mut local_col = 0;
        while local_col < block_width {
            let use_two_by_two = options.allow_two_by_two
                && local_col + 1 < block_width
                && rng.next_u64().is_multiple_of(3);
            if use_two_by_two {
                let d11 = rng.nonzero_dyadic(12, 4);
                let d21 = rng.nonzero_dyadic(12, 4);
                let d22 = rng.nonzero_dyadic(12, 4);
                block_records.push(FactorBlockRecord {
                    size: 2,
                    values: [d11, d21, f64::INFINITY, d22],
                });
                d_values[2 * local_col] = d11;
                d_values[2 * local_col + 1] = d21;
                d_values[2 * local_col + 2] = f64::INFINITY;
                d_values[2 * local_col + 3] = d22;
                local_col += 2;
            } else {
                let d11 = if rng.next_u64().is_multiple_of(7) {
                    0.0
                } else {
                    rng.nonzero_dyadic(12, 4)
                };
                block_records.push(FactorBlockRecord {
                    size: 1,
                    values: [d11, 0.0, 0.0, 0.0],
                });
                d_values[2 * local_col] = d11;
                d_values[2 * local_col + 1] = 0.0;
                local_col += 1;
            }
        }

        AppKernelCase {
            seed,
            size,
            block_start,
            block_end,
            matrix,
            block_records,
            d_values,
            small: 1e-20,
        }
    }

    fn app_solve_kernel_case_from_seed(seed: u64) -> AppSolveKernelCase {
        let mut rng = DenseBoundaryRng::new(seed);
        let eliminated_len = rng.usize_inclusive(1, 80);
        let trailing_rows = rng.usize_inclusive(0, 96);
        let rows = eliminated_len + trailing_rows;
        let mut lower = vec![0.0; rows * eliminated_len];
        for col in 0..eliminated_len {
            lower[col * rows + col] = 1.0;
            for row in (col + 1)..rows {
                lower[col * rows + row] = rng.dyadic_kernel_value(12, 6, true);
            }
        }

        let mut diagonal = vec![0.0; 2 * eliminated_len];
        let mut col = 0;
        while col < eliminated_len {
            let use_two_by_two = col + 1 < eliminated_len && rng.next_u64().is_multiple_of(3);
            if use_two_by_two {
                diagonal[2 * col] = rng.nonzero_dyadic(12, 4);
                diagonal[2 * col + 1] = rng.nonzero_dyadic(12, 4);
                diagonal[2 * col + 2] = f64::INFINITY;
                diagonal[2 * col + 3] = rng.nonzero_dyadic(12, 4);
                col += 2;
            } else {
                diagonal[2 * col] = rng.nonzero_dyadic(12, 4);
                diagonal[2 * col + 1] = 0.0;
                col += 1;
            }
        }

        let rhs = (0..rows)
            .map(|_| rng.dyadic_kernel_value(18, 6, true))
            .collect();

        AppSolveKernelCase {
            seed,
            rows,
            eliminated_len,
            lower,
            diagonal,
            rhs,
        }
    }

    fn block_update_case_from_seed(seed: u64, pivot_width: usize) -> BlockUpdateCase {
        let mut rng = DenseBoundaryRng::new(seed);
        let pivot = rng.usize_inclusive(0, APP_INNER_BLOCK_SIZE - pivot_width);
        let mut matrix = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        let mut workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * APP_INNER_BLOCK_SIZE + row] = rng.dyadic_kernel_value(24, 9, true);
            }
        }
        for row in (pivot + 1)..APP_INNER_BLOCK_SIZE {
            matrix[pivot * APP_INNER_BLOCK_SIZE + row] = rng.dyadic_kernel_value(24, 9, true);
            workspace[pivot * APP_INNER_BLOCK_SIZE + row] = rng.dyadic_kernel_value(24, 9, true);
            if row > pivot + 1 {
                matrix[(pivot + 1) * APP_INNER_BLOCK_SIZE + row] =
                    rng.dyadic_kernel_value(24, 9, true);
                workspace[(pivot + 1) * APP_INNER_BLOCK_SIZE + row] =
                    rng.dyadic_kernel_value(24, 9, true);
            }
        }
        BlockUpdateCase {
            seed,
            pivot,
            matrix,
            workspace,
        }
    }

    fn block_swap_case_from_seed(seed: u64) -> BlockSwapCase {
        let mut rng = DenseBoundaryRng::new(seed);
        let lhs = rng.usize_inclusive(0, APP_INNER_BLOCK_SIZE - 1);
        let mut rhs = rng.usize_inclusive(0, APP_INNER_BLOCK_SIZE - 1);
        if rhs == lhs {
            rhs = (rhs + 1) % APP_INNER_BLOCK_SIZE;
        }
        let n = APP_INNER_BLOCK_SIZE;
        let mut matrix = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        let mut workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * APP_INNER_BLOCK_SIZE + row] = rng.dyadic_kernel_value(24, 9, true);
            }
        }
        for value in &mut workspace {
            *value = rng.dyadic_kernel_value(24, 9, true);
        }
        let mut perm = (0..APP_INNER_BLOCK_SIZE).collect::<Vec<_>>();
        for index in 0..APP_INNER_BLOCK_SIZE {
            let other = rng.usize_inclusive(index, APP_INNER_BLOCK_SIZE - 1);
            perm.swap(index, other);
        }
        BlockSwapCase {
            seed,
            lhs,
            rhs,
            n,
            matrix,
            workspace,
            perm,
        }
    }

    fn block_find_maxloc_case_from_seed(seed: u64) -> BlockFindMaxlocCase {
        let mut rng = DenseBoundaryRng::new(seed);
        let from = rng.usize_inclusive(0, APP_INNER_BLOCK_SIZE - 1);
        let mut matrix = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * APP_INNER_BLOCK_SIZE + row] = rng.dyadic_kernel_value(24, 9, true);
            }
        }
        BlockFindMaxlocCase { seed, from, matrix }
    }

    fn block_first_step_case_from_seed(seed: u64, two_by_two: bool) -> BlockFirstStepCase {
        let mut rng = DenseBoundaryRng::new(seed);
        let from = if two_by_two {
            rng.usize_inclusive(0, APP_INNER_BLOCK_SIZE - 2)
        } else {
            rng.usize_inclusive(0, APP_INNER_BLOCK_SIZE - 1)
        };
        let mut matrix = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * APP_INNER_BLOCK_SIZE + row] = rng.dyadic_kernel_value(8, 8, false);
            }
        }

        if two_by_two {
            let first = rng.usize_inclusive(from, APP_INNER_BLOCK_SIZE - 2);
            let second = rng.usize_inclusive(first + 1, APP_INNER_BLOCK_SIZE - 1);
            let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, first, first)] = 1.0;
            matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, second, second)] = -1.0;
            matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, second, first)] = sign * 32.0;
        } else {
            let pivot = rng.usize_inclusive(from, APP_INNER_BLOCK_SIZE - 1);
            let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, pivot, pivot)] = sign * 32.0;
        }

        BlockFirstStepCase {
            seed,
            from,
            matrix,
            perm: (0..APP_INNER_BLOCK_SIZE).collect(),
            local_perm: (0..APP_INNER_BLOCK_SIZE).collect(),
            options: NumericFactorOptions::default(),
        }
    }

    fn native_host_trsm_op_n(shim: &NativeKernelShim, case: &AppKernelCase, matrix: &mut [f64]) {
        let trailing_rows = case.size - case.block_end;
        let block_width = case.block_end - case.block_start;
        let diag_offset = case.block_start * case.size + case.block_start;
        let aval_offset = case.block_start * case.size + case.block_end;

        unsafe {
            (shim.host_trsm_right_lower_trans_unit)(
                trailing_rows as c_int,
                block_width as c_int,
                matrix.as_ptr().add(diag_offset),
                case.size as c_int,
                matrix.as_mut_ptr().add(aval_offset),
                case.size as c_int,
            );
        }
    }

    fn native_host_trsv_lower_op_n(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        unsafe {
            (shim.host_trsv_lower_unit_op_n)(
                case.eliminated_len as c_int,
                case.lower.as_ptr(),
                case.rows as c_int,
                rhs.as_mut_ptr(),
            );
        }
    }

    fn native_host_trsv_lower_op_t(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        unsafe {
            (shim.host_trsv_lower_unit_op_t)(
                case.eliminated_len as c_int,
                case.lower.as_ptr(),
                case.rows as c_int,
                rhs.as_mut_ptr(),
            );
        }
    }

    fn native_gemv_op_n_solve_update(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        if case.rows == case.eliminated_len {
            return;
        }
        unsafe {
            (shim.gemv_op_n_solve_update)(
                (case.rows - case.eliminated_len) as c_int,
                case.eliminated_len as c_int,
                case.lower.as_ptr().add(case.eliminated_len),
                case.rows as c_int,
                rhs.as_ptr(),
                rhs.as_mut_ptr().add(case.eliminated_len),
            );
        }
    }

    fn native_gemv_op_t_solve_update(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        if case.rows == case.eliminated_len {
            return;
        }
        unsafe {
            (shim.gemv_op_t_solve_update)(
                (case.rows - case.eliminated_len) as c_int,
                case.eliminated_len as c_int,
                case.lower.as_ptr().add(case.eliminated_len),
                case.rows as c_int,
                rhs.as_ptr().add(case.eliminated_len),
                rhs.as_mut_ptr(),
            );
        }
    }

    fn native_apply_pivot_op_n(shim: &NativeKernelShim, case: &AppKernelCase, matrix: &mut [f64]) {
        let trailing_rows = case.size - case.block_end;
        let block_width = case.block_end - case.block_start;
        let diag_offset = case.block_start * case.size + case.block_start;
        let aval_offset = case.block_start * case.size + case.block_end;

        unsafe {
            (shim.apply_pivot_op_n)(
                trailing_rows as c_int,
                block_width as c_int,
                matrix.as_ptr().add(diag_offset),
                case.d_values.as_ptr(),
                case.small,
                matrix.as_mut_ptr().add(aval_offset),
                case.size as c_int,
            );
        }
    }

    fn native_check_threshold_op_n(
        shim: &NativeKernelShim,
        rows: usize,
        cols: usize,
        threshold_pivot_u: f64,
        matrix: &mut [f64],
        offset: usize,
        stride: usize,
    ) -> usize {
        unsafe {
            (shim.check_threshold_op_n)(
                0,
                rows as c_int,
                0,
                cols as c_int,
                threshold_pivot_u,
                matrix.as_mut_ptr().add(offset),
                stride as c_int,
            ) as usize
        }
    }

    fn native_ldlt_app_solve_fwd(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        unsafe {
            (shim.ldlt_app_solve_fwd)(
                case.rows as c_int,
                case.eliminated_len as c_int,
                case.lower.as_ptr(),
                case.rows as c_int,
                1,
                rhs.as_mut_ptr(),
                case.rows as c_int,
            );
        }
    }

    fn native_ldlt_app_solve_diag(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        unsafe {
            (shim.ldlt_app_solve_diag)(
                case.eliminated_len as c_int,
                case.diagonal.as_ptr(),
                1,
                rhs.as_mut_ptr(),
                case.rows as c_int,
            );
        }
    }

    fn native_ldlt_app_solve_bwd(
        shim: &NativeKernelShim,
        case: &AppSolveKernelCase,
        rhs: &mut [f64],
    ) {
        unsafe {
            (shim.ldlt_app_solve_bwd)(
                case.rows as c_int,
                case.eliminated_len as c_int,
                case.lower.as_ptr(),
                case.rows as c_int,
                1,
                rhs.as_mut_ptr(),
                case.rows as c_int,
            );
        }
    }

    fn native_aligned_double_stride(shim: &NativeKernelShim, rows: usize) -> usize {
        let stride = unsafe { (shim.align_lda_double)(rows as c_int) };
        assert!(
            stride >= 0,
            "native align_lda<double> returned negative stride {stride}"
        );
        stride as usize
    }

    fn native_block_update_1x1_32(
        shim: &NativeKernelShim,
        case: &BlockUpdateCase,
        matrix: &mut [f64],
    ) {
        unsafe {
            (shim.block_update_1x1_32)(
                case.pivot as c_int,
                matrix.as_mut_ptr(),
                APP_INNER_BLOCK_SIZE as c_int,
                case.workspace
                    .as_ptr()
                    .add(case.pivot * APP_INNER_BLOCK_SIZE),
            );
        }
    }

    fn native_block_update_2x2_32(
        shim: &NativeKernelShim,
        case: &BlockUpdateCase,
        matrix: &mut [f64],
    ) {
        unsafe {
            (shim.block_update_2x2_32)(
                case.pivot as c_int,
                matrix.as_mut_ptr(),
                APP_INNER_BLOCK_SIZE as c_int,
                case.workspace
                    .as_ptr()
                    .add(case.pivot * APP_INNER_BLOCK_SIZE),
            );
        }
    }

    fn native_block_swap_cols_32(
        shim: &NativeKernelShim,
        case: &BlockSwapCase,
        matrix: &mut [f64],
        workspace: &mut [f64],
        perm: &mut [usize],
    ) {
        let mut native_perm = perm.iter().map(|&entry| entry as c_int).collect::<Vec<_>>();
        unsafe {
            (shim.block_swap_cols_32)(
                case.lhs as c_int,
                case.rhs as c_int,
                case.n as c_int,
                matrix.as_mut_ptr(),
                APP_INNER_BLOCK_SIZE as c_int,
                workspace.as_mut_ptr(),
                native_perm.as_mut_ptr(),
            );
        }
        for (target, native) in perm.iter_mut().zip(native_perm) {
            *target = native as usize;
        }
    }

    fn rust_block_swap_cols_32(
        case: &BlockSwapCase,
        matrix: &mut [f64],
        workspace: &mut [f64],
        perm: &mut [usize],
    ) {
        let (lhs, rhs) = if case.lhs < case.rhs {
            (case.lhs, case.rhs)
        } else {
            (case.rhs, case.lhs)
        };
        dense_symmetric_swap_with_workspace(matrix, APP_INNER_BLOCK_SIZE, lhs, rhs, workspace);
        perm.swap(lhs, rhs);
    }

    fn native_block_find_maxloc_32(
        shim: &NativeKernelShim,
        case: &BlockFindMaxlocCase,
    ) -> (f64, usize, usize) {
        let mut bestv = 0.0;
        let mut rloc = 0;
        let mut cloc = 0;
        unsafe {
            (shim.block_find_maxloc_32)(
                case.from as c_int,
                case.matrix.as_ptr(),
                APP_INNER_BLOCK_SIZE as c_int,
                &mut bestv as *mut f64,
                &mut rloc as *mut c_int,
                &mut cloc as *mut c_int,
            );
        }
        (bestv, rloc as usize, cloc as usize)
    }

    fn native_block_two_by_two_multipliers(
        shim: &NativeKernelShim,
        inverse: (f64, f64, f64),
        values: (f64, f64),
    ) -> (f64, f64) {
        let mut out = [0.0; 2];
        unsafe {
            (shim.block_two_by_two_multipliers)(
                inverse.0,
                inverse.1,
                inverse.2,
                values.0,
                values.1,
                out.as_mut_ptr(),
            );
        }
        (out[0], out[1])
    }

    fn native_block_test_2x2(
        shim: &NativeKernelShim,
        a11: f64,
        a21: f64,
        a22: f64,
    ) -> (bool, f64, f64) {
        let mut detpiv = 0.0;
        let mut detscale = 0.0;
        let accepted = unsafe {
            (shim.block_test_2x2)(
                a11,
                a21,
                a22,
                &mut detpiv as *mut f64,
                &mut detscale as *mut f64,
            )
        } != 0;
        (accepted, detpiv, detscale)
    }

    fn native_block_first_step_32(
        shim: &NativeKernelShim,
        case: &BlockFirstStepCase,
        matrix: &mut [f64],
        workspace: &mut [f64],
        diagonal: &mut [f64],
        perm: &mut [usize],
        local_perm: &mut [usize],
    ) -> (i32, usize) {
        let mut native_perm = perm.iter().map(|&entry| entry as c_int).collect::<Vec<_>>();
        let mut native_local_perm = local_perm
            .iter()
            .map(|&entry| entry as c_int)
            .collect::<Vec<_>>();
        let mut next = 0;
        let status = unsafe {
            (shim.block_first_step_32)(
                case.from as c_int,
                native_perm.as_mut_ptr(),
                matrix.as_mut_ptr(),
                APP_INNER_BLOCK_SIZE as c_int,
                diagonal.as_mut_ptr(),
                workspace.as_mut_ptr(),
                i32::from(case.options.action_on_zero_pivot),
                case.options.threshold_pivot_u,
                case.options.small_pivot_tolerance,
                native_local_perm.as_mut_ptr(),
                &mut next as *mut c_int,
            )
        };
        for (target, native) in perm.iter_mut().zip(native_perm) {
            *target = native as usize;
        }
        for (target, native) in local_perm.iter_mut().zip(native_local_perm) {
            *target = native as usize;
        }
        (status, next as usize)
    }

    fn rust_block_first_step_32(
        case: &BlockFirstStepCase,
        matrix: &mut [f64],
        workspace: &mut [f64],
        diagonal: &mut [f64],
        perm: &mut [usize],
        local_perm: &mut [usize],
    ) -> Result<(i32, usize), SsidsError> {
        let pivot = case.from;
        let Some((best_abs, best_row, best_col)) =
            dense_find_maxloc(matrix, APP_INNER_BLOCK_SIZE, pivot, APP_INNER_BLOCK_SIZE)
        else {
            return Ok((0, pivot));
        };

        if best_abs < case.options.small_pivot_tolerance {
            if !case.options.action_on_zero_pivot {
                return Ok((-1, pivot));
            }
            for col in pivot..APP_INNER_BLOCK_SIZE {
                diagonal[2 * col] = 0.0;
                diagonal[2 * col + 1] = 0.0;
                for row in col..APP_INNER_BLOCK_SIZE {
                    matrix[col * APP_INNER_BLOCK_SIZE + row] = 0.0;
                    workspace[col * APP_INNER_BLOCK_SIZE + row] = 0.0;
                }
            }
            return Ok((0, APP_INNER_BLOCK_SIZE));
        }

        let mut rows = perm.to_vec();
        let mut stats = PanelFactorStats::default();
        if best_row == best_col {
            if best_col != pivot {
                dense_symmetric_swap_with_workspace(
                    matrix,
                    APP_INNER_BLOCK_SIZE,
                    best_col,
                    pivot,
                    workspace,
                );
                rows.swap(best_col, pivot);
                perm.swap(best_col, pivot);
                local_perm.swap(best_col, pivot);
            }
            let block = factor_one_by_one_common(
                &rows,
                matrix,
                APP_INNER_BLOCK_SIZE,
                pivot,
                APP_INNER_BLOCK_SIZE,
                &mut stats,
                workspace,
            )?;
            diagonal[2 * pivot] = block.values[0];
            diagonal[2 * pivot + 1] = 0.0;
            return Ok((1, pivot + 1));
        }

        let first = best_col;
        let mut second = best_row;
        let a11 = matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, first, first)];
        let a22 = matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, second, second)];
        let a21 = matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, second, first)];
        let mut two_by_two_inverse = None;
        let mut one_by_one_index = None;
        if let Some(inverse) =
            app_two_by_two_inverse(a11, a21, a22, case.options.small_pivot_tolerance)
        {
            two_by_two_inverse = Some(inverse);
        } else if a11.abs() > a22.abs() {
            if (a11 / a21).abs() >= case.options.threshold_pivot_u {
                one_by_one_index = Some(first);
            }
        } else if (a22 / a21).abs() >= case.options.threshold_pivot_u {
            one_by_one_index = Some(second);
        }

        if let Some(index) = one_by_one_index {
            if index != pivot {
                dense_symmetric_swap_with_workspace(
                    matrix,
                    APP_INNER_BLOCK_SIZE,
                    index,
                    pivot,
                    workspace,
                );
                rows.swap(index, pivot);
                perm.swap(index, pivot);
                local_perm.swap(index, pivot);
            }
            let block = factor_one_by_one_common(
                &rows,
                matrix,
                APP_INNER_BLOCK_SIZE,
                pivot,
                APP_INNER_BLOCK_SIZE,
                &mut stats,
                workspace,
            )?;
            diagonal[2 * pivot] = block.values[0];
            diagonal[2 * pivot + 1] = 0.0;
            return Ok((1, pivot + 1));
        }

        if let Some(inverse) = two_by_two_inverse {
            if first != pivot {
                dense_symmetric_swap_with_workspace(
                    matrix,
                    APP_INNER_BLOCK_SIZE,
                    first,
                    pivot,
                    workspace,
                );
                rows.swap(first, pivot);
                perm.swap(first, pivot);
                local_perm.swap(first, pivot);
                if second == pivot {
                    second = first;
                }
            }
            if second != pivot + 1 {
                dense_symmetric_swap_with_workspace(
                    matrix,
                    APP_INNER_BLOCK_SIZE,
                    second,
                    pivot + 1,
                    workspace,
                );
                rows.swap(second, pivot + 1);
                perm.swap(second, pivot + 1);
                local_perm.swap(second, pivot + 1);
            }
            let block = factor_two_by_two_common(
                &rows,
                matrix,
                DenseUpdateBounds {
                    size: APP_INNER_BLOCK_SIZE,
                    update_end: APP_INNER_BLOCK_SIZE,
                },
                pivot,
                inverse,
                &mut stats,
                workspace,
            )?;
            diagonal[2 * pivot] = block.values[0];
            diagonal[2 * pivot + 1] = block.values[1];
            diagonal[2 * pivot + 2] = f64::INFINITY;
            diagonal[2 * pivot + 3] = block.values[3];
            return Ok((2, pivot + 2));
        }

        Ok((-2, pivot))
    }

    fn native_block_prefix_trace_32(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
    ) -> Vec<BlockPrefixSnapshot> {
        native_block_prefix_trace_32_impl(shim, dense, size, lda, options, false)
    }

    fn native_block_prefix_trace_32_source(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
    ) -> Vec<BlockPrefixSnapshot> {
        native_block_prefix_trace_32_impl(shim, dense, size, lda, options, true)
    }

    fn native_block_prefix_trace_32_impl(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
        source_plain: bool,
    ) -> Vec<BlockPrefixSnapshot> {
        debug_assert_eq!(dense.len(), size * size);
        debug_assert!(size >= APP_INNER_BLOCK_SIZE);
        debug_assert!(lda >= APP_INNER_BLOCK_SIZE);
        let mut matrix = vec![0.0; lda * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * lda + row] = dense[dense_lower_offset(size, row, col)];
            }
        }
        let mut workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        let mut diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
        let mut perm = (0..APP_INNER_BLOCK_SIZE as c_int).collect::<Vec<_>>();
        let mut local_perm = (0..APP_INNER_BLOCK_SIZE as c_int).collect::<Vec<_>>();
        let max_steps = APP_INNER_BLOCK_SIZE;
        let mut trace_from = vec![0; max_steps];
        let mut trace_status = vec![0; max_steps];
        let mut trace_next = vec![0; max_steps];
        let mut trace_perm = vec![0; max_steps * APP_INNER_BLOCK_SIZE];
        let mut trace_local_perm = vec![0; max_steps * APP_INNER_BLOCK_SIZE];
        let mut trace_matrix = vec![0.0; max_steps * APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        let mut trace_workspace =
            vec![0.0; max_steps * APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        let mut trace_diagonal = vec![0.0; max_steps * 2 * APP_INNER_BLOCK_SIZE];
        let trace_fn = if source_plain {
            shim.block_prefix_trace_32_source
        } else {
            shim.block_prefix_trace_32
        };
        let steps = unsafe {
            (trace_fn)(
                0,
                perm.as_mut_ptr(),
                matrix.as_mut_ptr(),
                lda as c_int,
                diagonal.as_mut_ptr(),
                workspace.as_mut_ptr(),
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                local_perm.as_mut_ptr(),
                max_steps as c_int,
                trace_from.as_mut_ptr(),
                trace_status.as_mut_ptr(),
                trace_next.as_mut_ptr(),
                trace_perm.as_mut_ptr(),
                trace_local_perm.as_mut_ptr(),
                trace_matrix.as_mut_ptr(),
                trace_workspace.as_mut_ptr(),
                trace_diagonal.as_mut_ptr(),
            )
        } as usize;

        let mut trace = Vec::new();
        for step in 0..steps {
            let perm_offset = step * APP_INNER_BLOCK_SIZE;
            let matrix_offset = step * APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE;
            let diagonal_offset = step * 2 * APP_INNER_BLOCK_SIZE;
            trace.push(BlockPrefixSnapshot {
                step,
                from: trace_from[step] as usize,
                status: trace_status[step],
                next: trace_next[step] as usize,
                perm: trace_perm[perm_offset..perm_offset + APP_INNER_BLOCK_SIZE]
                    .iter()
                    .map(|&entry| entry as usize)
                    .collect(),
                local_perm: trace_local_perm[perm_offset..perm_offset + APP_INNER_BLOCK_SIZE]
                    .iter()
                    .map(|&entry| entry as usize)
                    .collect(),
                matrix: trace_matrix
                    [matrix_offset..matrix_offset + APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE]
                    .to_vec(),
                workspace: trace_workspace
                    [matrix_offset..matrix_offset + APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE]
                    .to_vec(),
                diagonal: trace_diagonal
                    [diagonal_offset..diagonal_offset + 2 * APP_INNER_BLOCK_SIZE]
                    .to_vec(),
            });
        }
        trace
    }

    fn rust_app_block_prefix_trace_32(
        dense: &[f64],
        size: usize,
        options: NumericFactorOptions,
    ) -> Vec<BlockPrefixSnapshot> {
        debug_assert_eq!(dense.len(), size * size);
        debug_assert!(size >= APP_INNER_BLOCK_SIZE);
        let mut rows = (0..size).collect::<Vec<_>>();
        let mut matrix = dense.to_vec();
        let mut workspace = vec![0.0; size * size];
        let mut diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
        let mut local_perm = (0..APP_INNER_BLOCK_SIZE).collect::<Vec<_>>();
        let mut trace = Vec::new();
        let mut from = 0;
        while from < APP_INNER_BLOCK_SIZE {
            let mut state = RustAppBlockStepState {
                rows: &mut rows,
                matrix: &mut matrix,
                size,
                workspace: &mut workspace,
                diagonal: &mut diagonal,
                local_perm: &mut local_perm,
            };
            let (status, next) = rust_app_block_first_step_32(&mut state, from, options)
                .expect("rust APP block prefix step");
            trace.push(block_prefix_snapshot(BlockPrefixSnapshotInput {
                step: trace.len(),
                from,
                status,
                next,
                perm: &rows[..APP_INNER_BLOCK_SIZE],
                local_perm: &local_perm,
                matrix: (&matrix, size),
                workspace: (&workspace, size),
                diagonal: &diagonal,
            }));
            if status <= 0 || next <= from {
                break;
            }
            from = next;
        }
        trace
    }

    fn rust_app_block_first_step_32(
        state: &mut RustAppBlockStepState<'_>,
        pivot: usize,
        options: NumericFactorOptions,
    ) -> Result<(i32, usize), SsidsError> {
        let rows = &mut *state.rows;
        let matrix = &mut *state.matrix;
        let size = state.size;
        let workspace = &mut *state.workspace;
        let diagonal = &mut *state.diagonal;
        let local_perm = &mut *state.local_perm;
        let block_end = APP_INNER_BLOCK_SIZE;
        let Some((best_abs, best_row, best_col)) =
            dense_find_maxloc(matrix, size, pivot, block_end)
        else {
            return Ok((0, pivot));
        };

        if best_abs < options.small_pivot_tolerance {
            if !options.action_on_zero_pivot {
                return Ok((-1, pivot));
            }
            let mut local_pivot = pivot;
            while local_pivot < block_end {
                diagonal[2 * local_pivot] = 0.0;
                diagonal[2 * local_pivot + 1] = 0.0;
                zero_dense_column_until(matrix, size, local_pivot, block_end);
                reset_ldwork_column_tail(workspace, size, local_pivot, local_pivot);
                local_pivot += 1;
            }
            return Ok((0, block_end));
        }

        let mut stats = PanelFactorStats::default();
        if best_row == best_col {
            if best_col != pivot {
                dense_symmetric_swap_with_workspace(matrix, size, best_col, pivot, workspace);
                rows.swap(best_col, pivot);
                local_perm.swap(best_col, pivot);
            }
            let block = factor_one_by_one_common(
                rows, matrix, size, pivot, block_end, &mut stats, workspace,
            )?;
            diagonal[2 * pivot] = block.values[0];
            diagonal[2 * pivot + 1] = 0.0;
            return Ok((1, pivot + 1));
        }

        let first = best_col;
        let mut second = best_row;
        let a11 = matrix[dense_lower_offset(size, first, first)];
        let a22 = matrix[dense_lower_offset(size, second, second)];
        let a21 = matrix[dense_lower_offset(size, second, first)];
        let mut two_by_two_inverse = None;
        let mut one_by_one_index = None;
        if let Some(inverse) = app_two_by_two_inverse(a11, a21, a22, options.small_pivot_tolerance)
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
            if index != pivot {
                dense_symmetric_swap_with_workspace(matrix, size, index, pivot, workspace);
                rows.swap(index, pivot);
                local_perm.swap(index, pivot);
            }
            let block = factor_one_by_one_common(
                rows, matrix, size, pivot, block_end, &mut stats, workspace,
            )?;
            diagonal[2 * pivot] = block.values[0];
            diagonal[2 * pivot + 1] = 0.0;
            return Ok((1, pivot + 1));
        }

        if let Some(inverse) = two_by_two_inverse {
            if first != pivot {
                dense_symmetric_swap_with_workspace(matrix, size, first, pivot, workspace);
                rows.swap(first, pivot);
                local_perm.swap(first, pivot);
                if second == pivot {
                    second = first;
                }
            }
            if second != pivot + 1 {
                dense_symmetric_swap_with_workspace(matrix, size, second, pivot + 1, workspace);
                rows.swap(second, pivot + 1);
                local_perm.swap(second, pivot + 1);
            }
            let block = factor_two_by_two_common(
                rows,
                matrix,
                DenseUpdateBounds {
                    size,
                    update_end: block_end,
                },
                pivot,
                inverse,
                &mut stats,
                workspace,
            )?;
            diagonal[2 * pivot] = block.values[0];
            diagonal[2 * pivot + 1] = block.values[1];
            diagonal[2 * pivot + 2] = f64::INFINITY;
            diagonal[2 * pivot + 3] = block.values[3];
            return Ok((2, pivot + 2));
        }

        Ok((-2, pivot))
    }

    fn block_prefix_snapshot(input: BlockPrefixSnapshotInput<'_>) -> BlockPrefixSnapshot {
        BlockPrefixSnapshot {
            step: input.step,
            from: input.from,
            status: input.status,
            next: input.next,
            perm: input.perm.to_vec(),
            local_perm: input.local_perm.to_vec(),
            matrix: extract_lower_block(input.matrix.0, input.matrix.1),
            workspace: extract_strided_block(input.workspace.0, input.workspace.1),
            diagonal: input.diagonal.to_vec(),
        }
    }

    fn extract_lower_block(matrix: &[f64], stride: usize) -> Vec<f64> {
        let mut block = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                block[col * APP_INNER_BLOCK_SIZE + row] = matrix[col * stride + row];
            }
        }
        block
    }

    fn extract_strided_block(matrix: &[f64], stride: usize) -> Vec<f64> {
        let mut block = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in 0..APP_INNER_BLOCK_SIZE {
                block[col * APP_INNER_BLOCK_SIZE + row] = matrix[col * stride + row];
            }
        }
        block
    }

    fn native_block_ldlt_32_from_lower_dense(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
    ) -> BlockLdltKernelResult {
        debug_assert_eq!(dense.len(), size * size);
        debug_assert!(size >= APP_INNER_BLOCK_SIZE);
        debug_assert!(lda >= APP_INNER_BLOCK_SIZE);
        let mut matrix = vec![0.0; lda * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * lda + row] = dense[dense_lower_offset(size, row, col)];
            }
        }
        let mut perm = (0..APP_INNER_BLOCK_SIZE as c_int).collect::<Vec<_>>();
        let mut local_perm = (0..APP_INNER_BLOCK_SIZE as c_int).collect::<Vec<_>>();
        let mut diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
        let mut ldwork = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
        unsafe {
            (shim.block_ldlt_32)(
                0,
                perm.as_mut_ptr(),
                matrix.as_mut_ptr(),
                lda as c_int,
                diagonal.as_mut_ptr(),
                ldwork.as_mut_ptr(),
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                local_perm.as_mut_ptr(),
            );
        }
        BlockLdltKernelResult {
            perm: perm.into_iter().map(|entry| entry as usize).collect(),
            local_perm: local_perm.into_iter().map(|entry| entry as usize).collect(),
            matrix,
            diagonal,
        }
    }

    fn rust_block_ldlt_32_from_lower_dense(
        dense: &[f64],
        size: usize,
        options: NumericFactorOptions,
    ) -> BlockLdltKernelResult {
        let rows = (0..size).collect::<Vec<_>>();
        let factorization = factorize_dense_front(rows, size, dense.to_vec(), options, false)
            .expect("rust block factorization");
        let mut factor_inverse = vec![usize::MAX; size];
        for (position, &row) in factorization.factor_order.iter().enumerate() {
            factor_inverse[row] = position;
        }
        let mut matrix = vec![0.0; size * APP_INNER_BLOCK_SIZE];
        for (local_col, column) in factorization
            .factor_columns
            .iter()
            .take(APP_INNER_BLOCK_SIZE)
            .enumerate()
        {
            matrix[local_col * size + local_col] = 1.0;
            for &(row, value) in &column.entries {
                let local_row = factor_inverse[row];
                if local_row < APP_INNER_BLOCK_SIZE {
                    matrix[local_col * size + local_row] = value;
                }
            }
        }
        let mut diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
        let mut cursor = 0;
        for block in &factorization.block_records {
            if cursor >= APP_INNER_BLOCK_SIZE {
                break;
            }
            if block.size == 1 {
                diagonal[2 * cursor] = block.values[0];
                diagonal[2 * cursor + 1] = 0.0;
                cursor += 1;
            } else {
                diagonal[2 * cursor] = block.values[0];
                diagonal[2 * cursor + 1] = block.values[1];
                diagonal[2 * cursor + 2] = f64::INFINITY;
                diagonal[2 * cursor + 3] = block.values[3];
                cursor += 2;
            }
        }
        BlockLdltKernelResult {
            perm: factorization.factor_order[..APP_INNER_BLOCK_SIZE].to_vec(),
            local_perm: factorization.factor_order[..APP_INNER_BLOCK_SIZE].to_vec(),
            matrix,
            diagonal,
        }
    }

    fn native_ldlt_tpp_factor_from_lower_dense(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        candidate_len: usize,
        options: NumericFactorOptions,
    ) -> DenseTppKernelResult {
        debug_assert_eq!(dense.len(), size * size);
        debug_assert!(candidate_len <= size);
        let mut matrix = copy_lower_dense_to_stride(dense, size, size);
        let mut perm = (0..size as c_int).collect::<Vec<_>>();
        let mut diagonal = vec![0.0; 2 * candidate_len.max(1)];
        let mut ld = vec![0.0; 2 * size.max(1)];
        let eliminated = unsafe {
            (shim.ldlt_tpp_factor)(
                size as c_int,
                candidate_len as c_int,
                perm.as_mut_ptr(),
                matrix.as_mut_ptr(),
                size as c_int,
                diagonal.as_mut_ptr(),
                ld.as_mut_ptr(),
                size as c_int,
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                0,
                ptr::null_mut(),
                0,
            )
        };
        assert!(
            eliminated >= 0,
            "native ldlt_tpp_factor returned {eliminated}"
        );
        DenseTppKernelResult {
            eliminated: eliminated as usize,
            perm: perm.into_iter().map(|entry| entry as usize).collect(),
            matrix,
            diagonal,
        }
    }

    fn dense_tpp_diagonal_from_blocks(
        blocks: &[FactorBlockRecord],
        candidate_len: usize,
    ) -> Vec<f64> {
        let mut diagonal = vec![0.0; 2 * candidate_len.max(1)];
        let mut pivot = 0;
        for block in blocks {
            if pivot >= candidate_len {
                break;
            }
            if block.size == 1 {
                diagonal[2 * pivot] = block.values[0];
                diagonal[2 * pivot + 1] = block.values[1];
                pivot += 1;
            } else {
                diagonal[2 * pivot] = block.values[0];
                diagonal[2 * pivot + 1] = block.values[1];
                diagonal[2 * pivot + 2] = block.values[2];
                diagonal[2 * pivot + 3] = block.values[3];
                pivot += 2;
            }
        }
        diagonal
    }

    fn rust_ldlt_tpp_factor_from_lower_dense(
        dense: &[f64],
        size: usize,
        candidate_len: usize,
        options: NumericFactorOptions,
    ) -> DenseTppKernelResult {
        debug_assert_eq!(dense.len(), size * size);
        debug_assert!(candidate_len <= size);
        let mut rows = (0..size).collect::<Vec<_>>();
        let mut matrix = dense.to_vec();
        let mut ld = vec![0.0; 2 * size.max(1)];
        let factorization = factorize_dense_tpp_tail_in_place(
            &mut rows,
            &mut matrix,
            DenseTppTailRequest {
                start_pivot: 0,
                candidate_len,
                options,
                require_full_elimination: true,
                profile_enabled: false,
            },
            &mut ld,
        )
        .expect("rust TPP factorization");
        DenseTppKernelResult {
            eliminated: factorization.factor_order.len(),
            perm: factorization.factor_order,
            matrix,
            diagonal: dense_tpp_diagonal_from_blocks(&factorization.block_records, candidate_len),
        }
    }

    fn assert_dense_tpp_kernel_results_equal(
        label: &str,
        rust: &DenseTppKernelResult,
        native: &DenseTppKernelResult,
        active_columns: usize,
    ) {
        assert_eq!(rust.eliminated, native.eliminated, "{label}");
        assert_eq!(
            &rust.perm[..active_columns],
            &native.perm[..active_columns],
            "{label}"
        );
        for (index, (&rust_value, &native_value)) in
            rust.diagonal.iter().zip(&native.diagonal).enumerate()
        {
            assert_eq!(
                rust_value.to_bits(),
                native_value.to_bits(),
                "{label}: ldlt_tpp d mismatch index={index} rust={rust_value:?} native={native_value:?}"
            );
        }
        let size = active_columns;
        for col in 0..active_columns {
            for row in col..size {
                let rust_value = rust.matrix[col * size + row];
                let native_value = native.matrix[col * size + row];
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "{label}: ldlt_tpp matrix mismatch row={row} col={col} rust={rust_value:?} native={native_value:?}"
                );
            }
        }
    }

    fn assert_dense_tpp_full_lower_matrix_equal(
        label: &str,
        rust: &DenseTppKernelResult,
        native: &DenseTppKernelResult,
        size: usize,
    ) {
        for col in 0..size {
            for row in col..size {
                let rust_value = rust.matrix[col * size + row];
                let native_value = native.matrix[col * size + row];
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "{label}: ldlt_tpp full matrix mismatch row={row} col={col} rust={rust_value:?} native={native_value:?}"
                );
            }
        }
    }

    fn rust_host_trsv_lower_op_n_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        openblas_trsv_lower_unit_op_n_like_native(
            &case.lower,
            case.rows,
            &mut rhs[..case.eliminated_len],
        );
    }

    fn rust_host_trsv_lower_op_t_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        openblas_trsv_lower_unit_op_t_like_native(
            &case.lower,
            case.rows,
            &mut rhs[..case.eliminated_len],
        );
    }

    fn rust_gemv_op_n_solve_update_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        let (solved, trailing) = rhs.split_at_mut(case.eliminated_len);
        openblas_gemv_n_update_like_native(
            case.rows - case.eliminated_len,
            case.eliminated_len,
            &case.lower[case.eliminated_len..],
            case.rows,
            solved,
            trailing,
        );
    }

    fn rust_gemv_op_t_solve_update_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        for local_col in 0..case.eliminated_len {
            let column_start = local_col * case.rows;
            let dot = openblas_gemv_t_dot_like_contiguous(
                &case.lower[column_start + case.eliminated_len..column_start + case.rows],
                &rhs[case.eliminated_len..case.rows],
            );
            rhs[local_col] = (-1.0f64).mul_add(dot, rhs[local_col]);
        }
    }

    fn rust_ldlt_app_solve_fwd_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        let panel = SolvePanel {
            eliminated_len: case.eliminated_len,
            row_positions: (0..case.rows).collect(),
            values: case.lower.clone(),
        };
        solve_forward_front_panels_like_native(&[panel], rhs);
    }

    fn rust_ldlt_app_solve_diag_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        let mut index = 0;
        while index < case.eliminated_len {
            if index + 1 == case.eliminated_len || case.diagonal[2 * index + 2].is_finite() {
                rhs[index] *= case.diagonal[2 * index];
                index += 1;
            } else {
                solve_two_by_two_block_in_place(
                    &case.diagonal[2 * index..2 * index + 4],
                    &mut rhs[index..index + 2],
                )
                .expect("generated finite two-by-two solve");
                index += 2;
            }
        }
    }

    fn rust_ldlt_app_solve_bwd_like_native(case: &AppSolveKernelCase, rhs: &mut [f64]) {
        rust_gemv_op_t_solve_update_like_native(case, rhs);
        rust_host_trsv_lower_op_t_like_native(case, rhs);
    }

    fn rust_block_update_1x1_32(case: &BlockUpdateCase, matrix: &mut [f64]) {
        app_update_one_by_one(
            matrix,
            APP_INNER_BLOCK_SIZE,
            case.pivot,
            APP_INNER_BLOCK_SIZE,
            &case.workspace,
        );
    }

    fn rust_block_update_2x2_32(case: &BlockUpdateCase, matrix: &mut [f64]) {
        app_update_two_by_two(
            matrix,
            APP_INNER_BLOCK_SIZE,
            case.pivot,
            APP_INNER_BLOCK_SIZE,
            &case.workspace,
        );
    }

    fn rust_block_two_by_two_multipliers(
        inverse: (f64, f64, f64),
        values: (f64, f64),
    ) -> (f64, f64) {
        let (inv11, inv12, inv22) = inverse;
        let (b1, b2) = values;
        (inv12.mul_add(b2, inv11 * b1), inv12.mul_add(b1, inv22 * b2))
    }

    fn assert_app_kernel_matrices_bitwise_equal(
        label: &str,
        case: &AppKernelCase,
        rust_matrix: &[f64],
        native_matrix: &[f64],
    ) {
        for (index, (&rust, &native)) in rust_matrix.iter().zip(native_matrix.iter()).enumerate() {
            assert_eq!(
                rust.to_bits(),
                native.to_bits(),
                "{label} mismatch seed={:#x} index={} rust={:?} ({:#018x}) native={:?} ({:#018x}) case={:?}",
                case.seed,
                index,
                rust,
                rust.to_bits(),
                native,
                native.to_bits(),
                case
            );
        }
    }

    fn assert_block_lower_matrix_bitwise_equal(
        label: &str,
        case: &BlockUpdateCase,
        rust_matrix: &[f64],
        native_matrix: &[f64],
    ) -> Result<(), TestCaseError> {
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                let index = col * APP_INNER_BLOCK_SIZE + row;
                prop_assert_eq!(
                    rust_matrix[index].to_bits(),
                    native_matrix[index].to_bits(),
                    "{} mismatch seed={:#x} index={} row={} col={} pivot={} rust={:?} native={:?} case={:?}",
                    label,
                    case.seed,
                    index,
                    row,
                    col,
                    case.pivot,
                    rust_matrix[index],
                    native_matrix[index],
                    case
                );
            }
        }
        Ok(())
    }

    fn assert_block_swap_matrices_bitwise_equal(
        label: &str,
        case: &BlockSwapCase,
        rust_matrix: &[f64],
        native_matrix: &[f64],
    ) -> Result<(), TestCaseError> {
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..case.n {
                let index = col * APP_INNER_BLOCK_SIZE + row;
                prop_assert_eq!(
                    rust_matrix[index].to_bits(),
                    native_matrix[index].to_bits(),
                    "{} mismatch seed={:#x} index={} row={} col={} lhs={} rhs={} n={} rust={:?} native={:?} case={:?}",
                    label,
                    case.seed,
                    index,
                    row,
                    col,
                    case.lhs,
                    case.rhs,
                    case.n,
                    rust_matrix[index],
                    native_matrix[index],
                    case
                );
            }
        }
        Ok(())
    }

    fn assert_block_first_step_state_equal(
        label: &str,
        case: &BlockFirstStepCase,
        rust: BlockFirstStepState<'_>,
        native: BlockFirstStepState<'_>,
    ) -> Result<(), TestCaseError> {
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                let index = col * APP_INNER_BLOCK_SIZE + row;
                prop_assert_eq!(
                    rust.matrix[index].to_bits(),
                    native.matrix[index].to_bits(),
                    "{} matrix mismatch seed={:#x} index={} row={} col={} from={} rust={:?} native={:?} case={:?}",
                    label,
                    case.seed,
                    index,
                    row,
                    col,
                    case.from,
                    rust.matrix[index],
                    native.matrix[index],
                    case
                );
            }
        }
        for (index, (&rust, &native)) in rust.workspace.iter().zip(native.workspace).enumerate() {
            prop_assert_eq!(
                rust.to_bits(),
                native.to_bits(),
                "{} ldwork mismatch seed={:#x} index={} from={} rust={:?} native={:?} case={:?}",
                label,
                case.seed,
                index,
                case.from,
                rust,
                native,
                case
            );
        }
        for (index, (&rust, &native)) in rust.diagonal.iter().zip(native.diagonal).enumerate() {
            prop_assert_eq!(
                rust.to_bits(),
                native.to_bits(),
                "{} diagonal mismatch seed={:#x} index={} from={} rust={:?} native={:?} case={:?}",
                label,
                case.seed,
                index,
                case.from,
                rust,
                native,
                case
            );
        }
        Ok(())
    }

    fn assert_block_prefix_traces_equal(
        rust: &[BlockPrefixSnapshot],
        native: &[BlockPrefixSnapshot],
    ) {
        for (rust_step, native_step) in rust.iter().zip(native) {
            assert_eq!(
                (
                    rust_step.step,
                    rust_step.from,
                    rust_step.status,
                    rust_step.next
                ),
                (
                    native_step.step,
                    native_step.from,
                    native_step.status,
                    native_step.next
                ),
                "block_ldlt prefix status mismatch rust={rust_step:?} native={native_step:?}"
            );
            assert_eq!(
                rust_step.perm, native_step.perm,
                "block_ldlt prefix perm mismatch step={} from={} rust={:?} native={:?}",
                rust_step.step, rust_step.from, rust_step.perm, native_step.perm
            );
            assert_eq!(
                rust_step.local_perm, native_step.local_perm,
                "block_ldlt prefix local_perm mismatch step={} from={} rust={:?} native={:?}",
                rust_step.step, rust_step.from, rust_step.local_perm, native_step.local_perm
            );
            for (index, (&rust_value, &native_value)) in rust_step
                .diagonal
                .iter()
                .zip(&native_step.diagonal)
                .enumerate()
            {
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "block_ldlt prefix d mismatch step={} from={} index={} rust={:?} native={:?}",
                    rust_step.step,
                    rust_step.from,
                    index,
                    rust_value,
                    native_value
                );
            }
            for col in 0..APP_INNER_BLOCK_SIZE {
                for row in col..APP_INNER_BLOCK_SIZE {
                    let index = col * APP_INNER_BLOCK_SIZE + row;
                    let rust_value = rust_step.matrix[index];
                    let native_value = native_step.matrix[index];
                    assert_eq!(
                        rust_value.to_bits(),
                        native_value.to_bits(),
                        "block_ldlt prefix matrix mismatch step={} from={} row={} col={} rust={:?} native={:?}",
                        rust_step.step,
                        rust_step.from,
                        row,
                        col,
                        rust_value,
                        native_value
                    );
                }
            }
            for (index, (&rust_value, &native_value)) in rust_step
                .workspace
                .iter()
                .zip(&native_step.workspace)
                .enumerate()
            {
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "block_ldlt prefix ldwork mismatch step={} from={} index={} rust={:?} native={:?}",
                    rust_step.step,
                    rust_step.from,
                    index,
                    rust_value,
                    native_value
                );
            }
        }
        if rust.len() != native.len() {
            let next_rust = rust.get(native.len()).or_else(|| rust.last());
            let next_native = native.get(rust.len()).or_else(|| native.last());
            panic!(
                "block_ldlt prefix step-count mismatch rust_len={} native_len={} last_common={} next_rust={} next_native={}",
                rust.len(),
                native.len(),
                rust.len().min(native.len()).saturating_sub(1),
                block_prefix_snapshot_summary(next_rust),
                block_prefix_snapshot_summary(next_native)
            );
        }
    }

    fn block_prefix_snapshot_summary(snapshot: Option<&BlockPrefixSnapshot>) -> String {
        snapshot.map_or_else(
            || "none".to_string(),
            |snapshot| {
                format!(
                    "step={} from={} status={} next={} perm={:?} local_perm={:?}",
                    snapshot.step,
                    snapshot.from,
                    snapshot.status,
                    snapshot.next,
                    snapshot.perm,
                    snapshot.local_perm
                )
            },
        )
    }

    #[test]
    fn app_block_triangular_solve_op_n_matches_native_host_trsm_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xd751_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_kernel_case_from_seed(
                    seed ^ case_seed,
                    AppKernelCaseOptions {
                        allow_two_by_two: true,
                        allow_signed_zero: false,
                    },
                );
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();

                native_host_trsm_op_n(shim, &case, &mut native_matrix);
                app_solve_block_triangular_to_trailing_rows(
                    &mut rust_matrix,
                    case.size,
                    case.block_start,
                    case.block_end,
                    false,
                );

                for (index, (&rust, &native)) in
                    rust_matrix.iter().zip(native_matrix.iter()).enumerate()
                {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "host_trsm OP_N mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("host_trsm OP_N kernel parity property failed");
    }

    #[test]
    #[ignore = "manual native-vs-rust host_trsm signed-zero witness"]
    fn app_block_triangular_solve_op_n_signed_zero_witness() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let case = app_kernel_case_from_seed(
            0xbffe_dbb3_2ab8_66e0,
            AppKernelCaseOptions {
                allow_two_by_two: true,
                allow_signed_zero: true,
            },
        );
        let mut rust_matrix = case.matrix.clone();
        let mut native_matrix = case.matrix.clone();

        native_host_trsm_op_n(shim, &case, &mut native_matrix);
        app_solve_block_triangular_to_trailing_rows(
            &mut rust_matrix,
            case.size,
            case.block_start,
            case.block_end,
            false,
        );

        assert_app_kernel_matrices_bitwise_equal(
            "host_trsm OP_N signed-zero witness",
            &case,
            &rust_matrix,
            &native_matrix,
        );
    }

    #[test]
    #[ignore = "manual native-vs-rust apply_pivot signed-zero witness"]
    fn app_apply_pivot_op_n_signed_zero_witness() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let case = app_kernel_case_from_seed(
            0xbffe_dbb3_2ab8_66e0,
            AppKernelCaseOptions {
                allow_two_by_two: true,
                allow_signed_zero: true,
            },
        );
        let mut rust_matrix = case.matrix.clone();
        let mut native_matrix = case.matrix.clone();

        native_apply_pivot_op_n(shim, &case, &mut native_matrix);
        app_apply_block_pivots_to_trailing_rows(
            &mut rust_matrix,
            case.size,
            case.block_start,
            case.block_end,
            &case.block_records,
            case.small,
            false,
        );

        assert_app_kernel_matrices_bitwise_equal(
            "apply_pivot OP_N signed-zero witness",
            &case,
            &rust_matrix,
            &native_matrix,
        );
    }

    #[test]
    fn app_block_triangular_solve_op_n_wide_openblas_update_regression() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let case = app_kernel_case_from_seed_with_limits(
            0x0d3b_9ddd_c779_97d6,
            AppKernelCaseOptions {
                allow_two_by_two: true,
                allow_signed_zero: true,
            },
            64,
            128,
        );
        let mut rust_matrix = case.matrix.clone();
        let mut native_matrix = case.matrix.clone();

        native_host_trsm_op_n(shim, &case, &mut native_matrix);
        app_solve_block_triangular_to_trailing_rows(
            &mut rust_matrix,
            case.size,
            case.block_start,
            case.block_end,
            false,
        );

        assert_app_kernel_matrices_bitwise_equal(
            "host_trsm OP_N wide OpenBLAS update regression",
            &case,
            &rust_matrix,
            &native_matrix,
        );
    }

    #[test]
    fn app_apply_pivot_op_n_wide_openblas_update_regression() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let case = app_kernel_case_from_seed_with_limits(
            0x00aa_b8cb_5a34_9e52,
            AppKernelCaseOptions {
                allow_two_by_two: true,
                allow_signed_zero: true,
            },
            64,
            128,
        );
        let mut rust_matrix = case.matrix.clone();
        let mut native_matrix = case.matrix.clone();

        native_apply_pivot_op_n(shim, &case, &mut native_matrix);
        app_apply_block_pivots_to_trailing_rows(
            &mut rust_matrix,
            case.size,
            case.block_start,
            case.block_end,
            &case.block_records,
            case.small,
            false,
        );

        assert_app_kernel_matrices_bitwise_equal(
            "apply_pivot OP_N wide OpenBLAS update regression",
            &case,
            &rust_matrix,
            &native_matrix,
        );
    }

    #[test]
    fn app_host_trsv_lower_op_n_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x7150_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_host_trsv_lower_op_n_like_native(&case, &mut rust_rhs);
                native_host_trsv_lower_op_n(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "host_trsv lower/unit OP_N mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("host_trsv lower/unit OP_N kernel parity property failed");
    }

    #[test]
    fn app_host_trsv_lower_op_t_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x7151_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_host_trsv_lower_op_t_like_native(&case, &mut rust_rhs);
                native_host_trsv_lower_op_t(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "host_trsv lower/unit OP_T mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("host_trsv lower/unit OP_T kernel parity property failed");
    }

    #[test]
    fn app_gemv_op_n_solve_update_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x6e50_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_gemv_op_n_solve_update_like_native(&case, &mut rust_rhs);
                native_gemv_op_n_solve_update(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "gemv OP_N solve update mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("gemv OP_N solve update kernel parity property failed");
    }

    #[test]
    fn app_gemv_op_t_solve_update_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x6e51_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_gemv_op_t_solve_update_like_native(&case, &mut rust_rhs);
                native_gemv_op_t_solve_update(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "gemv OP_T solve update mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("gemv OP_T solve update kernel parity property failed");
    }

    #[test]
    fn app_ldlt_solve_fwd_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xf0ed_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_ldlt_app_solve_fwd_like_native(&case, &mut rust_rhs);
                native_ldlt_app_solve_fwd(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "ldlt_app_solve_fwd mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("ldlt_app_solve_fwd kernel parity property failed");
    }

    #[test]
    fn app_ldlt_solve_diag_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xd1a6_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_ldlt_app_solve_diag_like_native(&case, &mut rust_rhs);
                native_ldlt_app_solve_diag(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "ldlt_app_solve_diag mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("ldlt_app_solve_diag kernel parity property failed");
    }

    #[test]
    fn app_ldlt_solve_bwd_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xbad0_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_solve_kernel_case_from_seed(seed ^ case_seed);
                let mut rust_rhs = case.rhs.clone();
                let mut native_rhs = case.rhs.clone();

                rust_ldlt_app_solve_bwd_like_native(&case, &mut rust_rhs);
                native_ldlt_app_solve_bwd(shim, &case, &mut native_rhs);

                for (index, (&rust, &native)) in rust_rhs.iter().zip(&native_rhs).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "ldlt_app_solve_bwd mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("ldlt_app_solve_bwd kernel parity property failed");
    }

    #[test]
    fn app_block_update_1x1_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xb101_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_update_case_from_seed(seed ^ case_seed, 1);
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();

                rust_block_update_1x1_32(&case, &mut rust_matrix);
                native_block_update_1x1_32(shim, &case, &mut native_matrix);

                assert_block_lower_matrix_bitwise_equal(
                    "block_ldlt update_1x1",
                    &case,
                    &rust_matrix,
                    &native_matrix,
                )?;
                Ok(())
            })
            .expect("block_ldlt update_1x1 kernel parity property failed");
    }

    #[test]
    fn app_block_update_2x2_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xb102_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_update_case_from_seed(seed ^ case_seed, 2);
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();

                rust_block_update_2x2_32(&case, &mut rust_matrix);
                native_block_update_2x2_32(shim, &case, &mut native_matrix);

                assert_block_lower_matrix_bitwise_equal(
                    "block_ldlt update_2x2",
                    &case,
                    &rust_matrix,
                    &native_matrix,
                )?;
                Ok(())
            })
            .expect("block_ldlt update_2x2 kernel parity property failed");
    }

    #[test]
    fn app_block_swap_cols_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xb5c0_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_swap_case_from_seed(seed ^ case_seed);
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();
                let mut rust_workspace = case.workspace.clone();
                let mut native_workspace = case.workspace.clone();
                let mut rust_perm = case.perm.clone();
                let mut native_perm = case.perm.clone();

                rust_block_swap_cols_32(
                    &case,
                    &mut rust_matrix,
                    &mut rust_workspace,
                    &mut rust_perm,
                );
                native_block_swap_cols_32(
                    shim,
                    &case,
                    &mut native_matrix,
                    &mut native_workspace,
                    &mut native_perm,
                );

                assert_block_swap_matrices_bitwise_equal(
                    "block_ldlt swap_cols matrix",
                    &case,
                    &rust_matrix,
                    &native_matrix,
                )?;
                for (index, (&rust, &native)) in rust_workspace.iter().zip(&native_workspace).enumerate()
                {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "block_ldlt swap_cols ldwork mismatch seed={:#x} index={} lhs={} rhs={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        case.lhs,
                        case.rhs,
                        rust,
                        native,
                        case
                    );
                }
                prop_assert_eq!(
                    rust_perm,
                    native_perm,
                    "block_ldlt swap_cols perm mismatch case={:?}",
                    case
                );
                Ok(())
            })
            .expect("block_ldlt swap_cols kernel parity property failed");
    }

    #[test]
    fn app_block_find_maxloc_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xf1d0_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_find_maxloc_case_from_seed(seed ^ case_seed);
                let rust = dense_find_maxloc(
                    &case.matrix,
                    APP_INNER_BLOCK_SIZE,
                    case.from,
                    APP_INNER_BLOCK_SIZE,
                )
                .expect("generated nonempty block");
                let native = native_block_find_maxloc_32(shim, &case);
                let rust_signed_value =
                    case.matrix[dense_lower_offset(APP_INNER_BLOCK_SIZE, rust.1, rust.2)];

                prop_assert_eq!(
                    rust_signed_value.to_bits(),
                    native.0.to_bits(),
                    "block_ldlt find_maxloc value mismatch seed={:#x} from={} rust={:?} native={:?} case={:?}",
                    case.seed,
                    case.from,
                    rust,
                    native,
                    case
                );
                prop_assert_eq!(
                    (rust.1, rust.2),
                    (native.1, native.2),
                    "block_ldlt find_maxloc location mismatch seed={:#x} from={} rust={:?} native={:?} case={:?}",
                    case.seed,
                    case.from,
                    rust,
                    native,
                    case
                );
                Ok(())
            })
            .expect("block_ldlt find_maxloc kernel parity property failed");
    }

    #[test]
    fn app_block_first_step_one_by_one_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x1511_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_first_step_case_from_seed(seed ^ case_seed, false);
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();
                let mut rust_workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
                let mut native_workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
                let mut rust_diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
                let mut native_diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
                let mut rust_perm = case.perm.clone();
                let mut native_perm = case.perm.clone();
                let mut rust_local_perm = case.local_perm.clone();
                let mut native_local_perm = case.local_perm.clone();

                let rust_status = rust_block_first_step_32(
                    &case,
                    &mut rust_matrix,
                    &mut rust_workspace,
                    &mut rust_diagonal,
                    &mut rust_perm,
                    &mut rust_local_perm,
                )
                .expect("generated one-by-one first step should factor");
                let native_status = native_block_first_step_32(
                    shim,
                    &case,
                    &mut native_matrix,
                    &mut native_workspace,
                    &mut native_diagonal,
                    &mut native_perm,
                    &mut native_local_perm,
                );

                prop_assert_eq!(
                    rust_status,
                    native_status,
                    "block_ldlt first-step 1x1 status mismatch case={:?}",
                    case
                );
                prop_assert_eq!(
                    rust_perm,
                    native_perm,
                    "block_ldlt first-step 1x1 perm mismatch case={:?}",
                    case
                );
                prop_assert_eq!(
                    rust_local_perm,
                    native_local_perm,
                    "block_ldlt first-step 1x1 local perm mismatch case={:?}",
                    case
                );
                assert_block_first_step_state_equal(
                    "block_ldlt first-step 1x1",
                    &case,
                    BlockFirstStepState {
                        matrix: &rust_matrix,
                        workspace: &rust_workspace,
                        diagonal: &rust_diagonal,
                    },
                    BlockFirstStepState {
                        matrix: &native_matrix,
                        workspace: &native_workspace,
                        diagonal: &native_diagonal,
                    },
                )?;
                Ok(())
            })
            .expect("block_ldlt first-step 1x1 kernel parity property failed");
    }

    #[test]
    fn app_block_first_step_two_by_two_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x2522_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_first_step_case_from_seed(seed ^ case_seed, true);
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();
                let mut rust_workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
                let mut native_workspace = vec![0.0; APP_INNER_BLOCK_SIZE * APP_INNER_BLOCK_SIZE];
                let mut rust_diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
                let mut native_diagonal = vec![0.0; 2 * APP_INNER_BLOCK_SIZE];
                let mut rust_perm = case.perm.clone();
                let mut native_perm = case.perm.clone();
                let mut rust_local_perm = case.local_perm.clone();
                let mut native_local_perm = case.local_perm.clone();

                let rust_status = rust_block_first_step_32(
                    &case,
                    &mut rust_matrix,
                    &mut rust_workspace,
                    &mut rust_diagonal,
                    &mut rust_perm,
                    &mut rust_local_perm,
                )
                .expect("generated two-by-two first step should factor");
                let native_status = native_block_first_step_32(
                    shim,
                    &case,
                    &mut native_matrix,
                    &mut native_workspace,
                    &mut native_diagonal,
                    &mut native_perm,
                    &mut native_local_perm,
                );

                prop_assert_eq!(
                    rust_status,
                    native_status,
                    "block_ldlt first-step 2x2 status mismatch case={:?}",
                    case
                );
                prop_assert_eq!(
                    rust_perm,
                    native_perm,
                    "block_ldlt first-step 2x2 perm mismatch case={:?}",
                    case
                );
                prop_assert_eq!(
                    rust_local_perm,
                    native_local_perm,
                    "block_ldlt first-step 2x2 local perm mismatch case={:?}",
                    case
                );
                assert_block_first_step_state_equal(
                    "block_ldlt first-step 2x2",
                    &case,
                    BlockFirstStepState {
                        matrix: &rust_matrix,
                        workspace: &rust_workspace,
                        diagonal: &rust_diagonal,
                    },
                    BlockFirstStepState {
                        matrix: &native_matrix,
                        workspace: &native_workspace,
                        diagonal: &native_diagonal,
                    },
                )?;
                Ok(())
            })
            .expect("block_ldlt first-step 2x2 kernel parity property failed");
    }

    #[test]
    fn app_block_test_2x2_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xb221_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let mut rng = DenseBoundaryRng::new(seed ^ case_seed);
                let exponent = rng.usize_inclusive(0, 16) as i32 - 8;
                let magnitude = 2.0_f64.powi(exponent);
                let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                let signed_fraction = |rng: &mut DenseBoundaryRng| {
                    let numerator = rng.usize_inclusive(0, 1024) as i32 - 512;
                    magnitude * f64::from(numerator) / 512.0
                };
                let a11 = signed_fraction(&mut rng);
                let a21 = sign * magnitude;
                let a22 = signed_fraction(&mut rng);

                let (native_accepted, native_detpiv, native_detscale) =
                    native_block_test_2x2(shim, a11, a21, a22);
                let rust_inverse = app_two_by_two_inverse(a11, a21, a22, 0.0);

                prop_assert_eq!(
                    rust_inverse.is_some(),
                    native_accepted,
                    "block_ldlt test_2x2 acceptance mismatch seed={:#x} a11={:?} a21={:?} a22={:?} native_detpiv={:?} native_detscale={:?}",
                    seed ^ case_seed,
                    a11,
                    a21,
                    a22,
                    native_detpiv,
                    native_detscale
                );
                if let Some(rust_inverse) = rust_inverse {
                    let native_inverse = (
                        (a22 * native_detscale) / native_detpiv,
                        (-a21 * native_detscale) / native_detpiv,
                        (a11 * native_detscale) / native_detpiv,
                    );
                    prop_assert_eq!(
                        rust_inverse.0.to_bits(),
                        native_inverse.0.to_bits(),
                        "block_ldlt 2x2 d11 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
                        seed ^ case_seed,
                        a11,
                        a21,
                        a22,
                        native_detpiv,
                        native_detscale
                    );
                    prop_assert_eq!(
                        rust_inverse.1.to_bits(),
                        native_inverse.1.to_bits(),
                        "block_ldlt 2x2 d21 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
                        seed ^ case_seed,
                        a11,
                        a21,
                        a22,
                        native_detpiv,
                        native_detscale
                    );
                    prop_assert_eq!(
                        rust_inverse.2.to_bits(),
                        native_inverse.2.to_bits(),
                        "block_ldlt 2x2 d22 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
                        seed ^ case_seed,
                        a11,
                        a21,
                        a22,
                        native_detpiv,
                        native_detscale
                    );
                }
                Ok(())
            })
            .expect("block_ldlt test_2x2 kernel parity property failed");
    }

    #[test]
    fn app_block_two_by_two_multipliers_match_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xb22b_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let mut rng = DenseBoundaryRng::new(seed ^ case_seed);
                let inverse = (
                    rng.dyadic_kernel_value(24, 12, false),
                    rng.dyadic_kernel_value(24, 12, false),
                    rng.dyadic_kernel_value(24, 12, false),
                );
                let values = (
                    rng.dyadic_kernel_value(24, 12, true),
                    rng.dyadic_kernel_value(24, 12, true),
                );

                let rust = rust_block_two_by_two_multipliers(inverse, values);
                let native = native_block_two_by_two_multipliers(shim, inverse, values);

                prop_assert_eq!(
                    rust.0.to_bits(),
                    native.0.to_bits(),
                    "block_ldlt 2x2 first multiplier mismatch seed={:#x} inverse={:?} values={:?} rust={:?} native={:?}",
                    seed ^ case_seed,
                    inverse,
                    values,
                    rust,
                    native
                );
                prop_assert_eq!(
                    rust.1.to_bits(),
                    native.1.to_bits(),
                    "block_ldlt 2x2 second multiplier mismatch seed={:#x} inverse={:?} values={:?} rust={:?} native={:?}",
                    seed ^ case_seed,
                    inverse,
                    values,
                    rust,
                    native
                );
                Ok(())
            })
            .expect("block_ldlt 2x2 multiplier kernel parity property failed");
    }

    #[test]
    fn app_apply_pivot_op_n_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xa991_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_kernel_case_from_seed(
                    seed ^ case_seed,
                    AppKernelCaseOptions {
                        allow_two_by_two: true,
                        allow_signed_zero: false,
                    },
                );
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();

                native_apply_pivot_op_n(shim, &case, &mut native_matrix);
                app_apply_block_pivots_to_trailing_rows(
                    &mut rust_matrix,
                    case.size,
                    case.block_start,
                    case.block_end,
                    &case.block_records,
                    case.small,
                    false,
                );

                for (index, (&rust, &native)) in
                    rust_matrix.iter().zip(native_matrix.iter()).enumerate()
                {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "apply_pivot<OP_N> mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("apply_pivot<OP_N> kernel parity property failed");
    }

    #[test]
    fn app_calc_ld_op_n_one_by_one_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xc41c_1d00_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_kernel_case_from_seed(
                    seed ^ case_seed,
                    AppKernelCaseOptions {
                        allow_two_by_two: false,
                        allow_signed_zero: false,
                    },
                );
                let trailing_rows = case.size - case.block_end;
                let block_width = case.block_end - case.block_start;
                let l_offset = case.block_start * case.size + case.block_end;
                let mut native_ld = vec![0.0; block_width * case.size];

                unsafe {
                    (shim.calc_ld_op_n)(
                        trailing_rows as c_int,
                        block_width as c_int,
                        case.matrix.as_ptr().add(l_offset),
                        case.size as c_int,
                        case.d_values.as_ptr(),
                        native_ld.as_mut_ptr().add(case.block_end),
                        case.size as c_int,
                    );
                }
                let rust_ld = app_build_ld_workspace(
                    &case.matrix,
                    case.size,
                    case.block_start,
                    case.block_end,
                    &case.block_records,
                );

                for (index, (&rust, &native)) in rust_ld.iter().zip(native_ld.iter()).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "calcLD<OP_N> mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("calcLD<OP_N> kernel parity property failed");
    }

    #[test]
    #[ignore = "manual native-vs-rust APP signed-zero/2x2 kernel mismatch hunt"]
    fn app_apply_pivot_op_n_matches_native_kernel_full_property_hunt() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 4096);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xa991_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_kernel_case_from_seed(
                    seed ^ case_seed,
                    AppKernelCaseOptions {
                        allow_two_by_two: true,
                        allow_signed_zero: true,
                    },
                );
                let mut rust_matrix = case.matrix.clone();
                let mut native_matrix = case.matrix.clone();

                native_apply_pivot_op_n(shim, &case, &mut native_matrix);
                app_apply_block_pivots_to_trailing_rows(
                    &mut rust_matrix,
                    case.size,
                    case.block_start,
                    case.block_end,
                    &case.block_records,
                    case.small,
                    false,
                );

                for (index, (&rust, &native)) in
                    rust_matrix.iter().zip(native_matrix.iter()).enumerate()
                {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "apply_pivot<OP_N> full mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("apply_pivot<OP_N> full kernel parity hunt failed");
    }

    #[test]
    fn app_calc_ld_op_n_matches_native_kernel_full_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 4096);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xc41c_1d00_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = app_kernel_case_from_seed(
                    seed ^ case_seed,
                    AppKernelCaseOptions {
                        allow_two_by_two: true,
                        allow_signed_zero: true,
                    },
                );
                let trailing_rows = case.size - case.block_end;
                let block_width = case.block_end - case.block_start;
                let l_offset = case.block_start * case.size + case.block_end;
                let mut native_ld = vec![0.0; block_width * case.size];

                unsafe {
                    (shim.calc_ld_op_n)(
                        trailing_rows as c_int,
                        block_width as c_int,
                        case.matrix.as_ptr().add(l_offset),
                        case.size as c_int,
                        case.d_values.as_ptr(),
                        native_ld.as_mut_ptr().add(case.block_end),
                        case.size as c_int,
                    );
                }
                let rust_ld = app_build_ld_workspace(
                    &case.matrix,
                    case.size,
                    case.block_start,
                    case.block_end,
                    &case.block_records,
                );

                for (index, (&rust, &native)) in rust_ld.iter().zip(native_ld.iter()).enumerate() {
                    prop_assert_eq!(
                        rust.to_bits(),
                        native.to_bits(),
                        "calcLD<OP_N> full mismatch seed={:#x} index={} rust={:?} native={:?} case={:?}",
                        case.seed,
                        index,
                        rust,
                        native,
                        case
                    );
                }
                Ok(())
            })
            .expect("calcLD<OP_N> full kernel parity property failed");
    }

    #[test]
    fn app_calc_ld_op_n_two_by_two_vector_row_regression() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let case = app_kernel_case_from_seed(
            0x3ef5_99a6_74c4_6051,
            AppKernelCaseOptions {
                allow_two_by_two: true,
                allow_signed_zero: true,
            },
        );
        let trailing_rows = case.size - case.block_end;
        let block_width = case.block_end - case.block_start;
        let l_offset = case.block_start * case.size + case.block_end;
        let mut native_ld = vec![0.0; block_width * case.size];

        unsafe {
            (shim.calc_ld_op_n)(
                trailing_rows as c_int,
                block_width as c_int,
                case.matrix.as_ptr().add(l_offset),
                case.size as c_int,
                case.d_values.as_ptr(),
                native_ld.as_mut_ptr().add(case.block_end),
                case.size as c_int,
            );
        }
        let rust_ld = app_build_ld_workspace(
            &case.matrix,
            case.size,
            case.block_start,
            case.block_end,
            &case.block_records,
        );

        for (index, (&rust, &native)) in rust_ld.iter().zip(native_ld.iter()).enumerate() {
            assert_eq!(
                rust.to_bits(),
                native.to_bits(),
                "calcLD<OP_N> two-by-two vector row mismatch index={index} rust={rust:?} native={native:?}"
            );
        }
    }

    #[test]
    fn app_calc_ld_op_n_two_by_two_scalar_rows_regression() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };

        for seed in [0x311b_d3c6_e0de_a5dc, 0x709b_7377_12ec_ad2a] {
            let case = app_kernel_case_from_seed(
                seed,
                AppKernelCaseOptions {
                    allow_two_by_two: true,
                    allow_signed_zero: true,
                },
            );
            let trailing_rows = case.size - case.block_end;
            let block_width = case.block_end - case.block_start;
            let l_offset = case.block_start * case.size + case.block_end;
            let mut native_ld = vec![0.0; block_width * case.size];

            unsafe {
                (shim.calc_ld_op_n)(
                    trailing_rows as c_int,
                    block_width as c_int,
                    case.matrix.as_ptr().add(l_offset),
                    case.size as c_int,
                    case.d_values.as_ptr(),
                    native_ld.as_mut_ptr().add(case.block_end),
                    case.size as c_int,
                );
            }
            let rust_ld = app_build_ld_workspace(
                &case.matrix,
                case.size,
                case.block_start,
                case.block_end,
                &case.block_records,
            );

            for (index, (&rust, &native)) in rust_ld.iter().zip(native_ld.iter()).enumerate() {
                assert_eq!(
                    rust.to_bits(),
                    native.to_bits(),
                    "calcLD<OP_N> scalar row mismatch seed={seed:#x} index={index} rust={rust:?} native={native:?}"
                );
            }
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
    fn dense_tpp_factor_4x4_matches_native_kernel() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let dense = square_to_dense_lower(&[
            vec![4.0, 1.0, 0.25, -0.5],
            vec![1.0, -3.0, 0.75, 0.125],
            vec![0.25, 0.75, 2.0, -1.0],
            vec![-0.5, 0.125, -1.0, 1.5],
        ]);
        let options = NumericFactorOptions::default();

        let rust = rust_ldlt_tpp_factor_from_lower_dense(&dense, 4, 4, options);
        let native = native_ldlt_tpp_factor_from_lower_dense(shim, &dense, 4, 4, options);

        assert_dense_tpp_kernel_results_equal("4x4 hand witness", &rust, &native, 4);
    }

    fn dense_tpp_dyadic_case_lower(case: usize) -> (usize, Vec<f64>) {
        let dimension = 3 + case % 4;
        let mut rng = DenseBoundaryRng::new(0x7d00_0000_0000_0000_u64 ^ case as u64);
        let mut matrix = random_dense_dyadic_matrix(dimension, &mut rng);
        for (row, values) in matrix.iter_mut().enumerate() {
            let offset = if row.is_multiple_of(2) { 4.0 } else { -4.0 };
            values[row] += offset;
        }
        (dimension, square_to_dense_lower(&matrix))
    }

    #[test]
    fn dense_tpp_dyadic_case0_two_pivot_prefix_matches_native_kernel() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (_, dense) = dense_tpp_dyadic_case_lower(0);
        let options = NumericFactorOptions::default();

        let rust = rust_ldlt_tpp_factor_from_lower_dense(&dense, 3, 2, options);
        let native = native_ldlt_tpp_factor_from_lower_dense(shim, &dense, 3, 2, options);

        assert_dense_tpp_kernel_results_equal("dyadic case=0 first two pivots", &rust, &native, 2);
        assert_dense_tpp_full_lower_matrix_equal(
            "dyadic case=0 first two pivots",
            &rust,
            &native,
            3,
        );
    }

    #[test]
    fn dense_tpp_factor_dyadic_cases_0_to_15_match_native_kernel() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let options = NumericFactorOptions::default();

        for case in 0..16 {
            let (dimension, dense) = dense_tpp_dyadic_case_lower(case);
            let rust = rust_ldlt_tpp_factor_from_lower_dense(&dense, dimension, dimension, options);
            let native = native_ldlt_tpp_factor_from_lower_dense(
                shim, &dense, dimension, dimension, options,
            );

            let label = format!("dyadic case={case} dimension={dimension}");
            assert_dense_tpp_kernel_results_equal(&label, &rust, &native, dimension);
        }
    }

    #[test]
    fn dense_tpp_dyadic_case7_two_candidate_prefix_matches_native_kernel() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let options = NumericFactorOptions::default();
        let (dimension, dense) = dense_tpp_dyadic_case_lower(7);

        let rust = rust_ldlt_tpp_factor_from_lower_dense(&dense, dimension, 2, options);
        let native = native_ldlt_tpp_factor_from_lower_dense(shim, &dense, dimension, 2, options);

        assert_dense_tpp_kernel_results_equal("dyadic case=7 candidate_len=2", &rust, &native, 2);
        assert_dense_tpp_full_lower_matrix_equal(
            "dyadic case=7 candidate_len=2",
            &rust,
            &native,
            dimension,
        );
    }

    #[test]
    fn dense_tpp_dyadic_case7_three_candidate_prefix_matches_native_kernel() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let options = NumericFactorOptions::default();
        let (dimension, dense) = dense_tpp_dyadic_case_lower(7);

        let rust = rust_ldlt_tpp_factor_from_lower_dense(&dense, dimension, 3, options);
        let native = native_ldlt_tpp_factor_from_lower_dense(shim, &dense, dimension, 3, options);

        assert_dense_tpp_kernel_results_equal("dyadic case=7 candidate_len=3", &rust, &native, 3);
        assert_dense_tpp_full_lower_matrix_equal(
            "dyadic case=7 candidate_len=3",
            &rust,
            &native,
            dimension,
        );
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
            DenseTppTailRequest {
                start_pivot: 0,
                candidate_len: 2,
                options: NumericFactorOptions {
                    threshold_pivot_u: 0.01,
                    ..NumericFactorOptions::default()
                },
                require_full_elimination: false,
                profile_enabled: false,
            },
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
        let Some(native) = native_spral_or_skip() else {
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
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed6_33_matrix();
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

    fn dense_seed6_33_matrix() -> (usize, Vec<Vec<f64>>) {
        let mut rng = DenseBoundaryRng::new(6);
        let dimension = rng.usize_inclusive(33, 33);
        assert_eq!(dimension, 33);
        (dimension, random_dense_dyadic_matrix(dimension, &mut rng))
    }

    fn dense_seed09_case0_matrix() -> (usize, Vec<Vec<f64>>) {
        let mut rng = DenseBoundaryRng::new(0x09c9_134e_4eff_0004);
        let dimension = rng.usize_inclusive(33, 160);
        assert_eq!(dimension, 55);
        (dimension, random_dense_dyadic_matrix(dimension, &mut rng))
    }

    fn lower_dense_seed6_33() -> (usize, Vec<f64>) {
        let (dimension, dense) = dense_seed6_33_matrix();
        let mut lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for row in col..dimension {
                lower_dense[dense_lower_offset(dimension, row, col)] = dense[row][col];
            }
        }
        (dimension, lower_dense)
    }

    fn lower_dense_seed09_case0() -> (usize, Vec<f64>) {
        let (dimension, dense) = dense_seed09_case0_matrix();
        let mut lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for row in col..dimension {
                lower_dense[dense_lower_offset(dimension, row, col)] = dense[row][col];
            }
        }
        (dimension, lower_dense)
    }

    fn copy_lower_dense_to_stride(dense: &[f64], size: usize, stride: usize) -> Vec<f64> {
        assert!(stride >= size);
        let mut strided = vec![0.0; stride * stride];
        for col in 0..size {
            for row in col..size {
                strided[col * stride + row] = dense[dense_lower_offset(size, row, col)];
            }
        }
        strided
    }

    fn rust_inverse_diagonal_entries(factor: &super::NumericFactor) -> Vec<[f64; 2]> {
        let mut entries = Vec::with_capacity(factor.dimension);
        for (block, values) in factor
            .diagonal_blocks
            .iter()
            .zip(factor.diagonal_values.chunks_exact(4))
        {
            if block.size == 1 {
                entries.push([values[0], 0.0]);
            } else {
                debug_assert_eq!(block.size, 2);
                entries.push([values[0], values[1]]);
                entries.push([values[3], 0.0]);
            }
        }
        entries
    }

    fn inverse_diagonal_bits(entries: &[[f64; 2]]) -> Vec<u64> {
        entries
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect()
    }

    fn inverse_diagonal_entries_from_internal_diagonal(
        diagonal: &[f64],
        candidate_len: usize,
    ) -> Vec<[f64; 2]> {
        let mut entries = Vec::with_capacity(candidate_len);
        let mut cursor = 0;
        while cursor < candidate_len {
            if cursor + 1 == candidate_len || diagonal[2 * cursor + 2].is_finite() {
                entries.push([diagonal[2 * cursor], 0.0]);
                cursor += 1;
            } else {
                entries.push([diagonal[2 * cursor], diagonal[2 * cursor + 1]]);
                entries.push([diagonal[2 * cursor + 3], 0.0]);
                cursor += 2;
            }
        }
        entries
    }

    #[test]
    fn app_block_ldlt_32_prefix_trace_matches_native_dense_seed6() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed6_33();
        let options = NumericFactorOptions::default();
        let rust_trace = rust_app_block_prefix_trace_32(&lower_dense, dimension, options);
        let native_trace = native_block_prefix_trace_32(shim, &lower_dense, dimension, 34, options);
        assert_block_prefix_traces_equal(&rust_trace, &native_trace);
    }

    #[test]
    fn app_block_ldlt_32_aligned_prefix_trace_matches_native_dense_seed6() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed6_33();
        let aligned_stride = native_aligned_double_stride(shim, dimension);
        let aligned_lower_dense =
            copy_lower_dense_to_stride(&lower_dense, dimension, aligned_stride);
        let options = NumericFactorOptions::default();
        let rust_trace =
            rust_app_block_prefix_trace_32(&aligned_lower_dense, aligned_stride, options);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, aligned_stride, options);
        assert_block_prefix_traces_equal(&rust_trace, &native_trace);
    }

    #[test]
    fn app_block_ldlt_32_aligned_prefix_trace_matches_native_dense_seed09_case0() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed09_case0();
        let aligned_stride = native_aligned_double_stride(shim, dimension);
        let aligned_lower_dense =
            copy_lower_dense_to_stride(&lower_dense, dimension, aligned_stride);
        let options = NumericFactorOptions::default();
        let rust_trace =
            rust_app_block_prefix_trace_32(&aligned_lower_dense, aligned_stride, options);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, aligned_stride, options);
        assert_block_prefix_traces_equal(&rust_trace, &native_trace);
    }

    #[test]
    #[ignore = "manual native-vs-rust production APP acceptance/storage witness after matching block_ldlt<32> prefix"]
    fn app_block_ldlt_32_matches_native_dense_seed6_prefix() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed6_33();
        let options = NumericFactorOptions::default();

        let rust = rust_block_ldlt_32_from_lower_dense(&lower_dense, dimension, options);
        let native =
            native_block_ldlt_32_from_lower_dense(shim, &lower_dense, dimension, 34, options);
        assert_eq!(rust.perm, native.perm);
        assert_eq!(rust.local_perm, native.local_perm);
        for (index, (&rust_value, &native_value)) in
            rust.diagonal.iter().zip(&native.diagonal).enumerate()
        {
            assert_eq!(
                rust_value.to_bits(),
                native_value.to_bits(),
                "block_ldlt final d mismatch after matching traced APP prefix index={index} rust={rust_value:?} native={native_value:?}"
            );
        }
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                let rust_value = rust.matrix[col * dimension + row];
                let native_value = native.matrix[col * 34 + row];
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "block_ldlt final matrix mismatch after matching traced APP prefix row={row} col={col} rust={rust_value:?} native={native_value:?} local_perm={:?}",
                    rust.local_perm
                );
            }
        }
    }

    #[test]
    fn dense_seed6_production_factor_metadata_matches_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed6_33_matrix();
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
        let native_info = native_session.factorize(matrix).expect("native factorize");
        let native_enquiry = native_session.enquire_indef().expect("native enquire");
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
    fn dense_seed6_production_inverse_d_matches_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed6_33_matrix();
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

        let rust_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);

        assert_eq!(
            rust_bits, native_bits,
            "seed6 production inverse-D bit patterns differ"
        );
    }

    #[test]
    fn dense_seed09_case0_production_app_prefix_inverse_d_matches_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed09_case0_matrix();
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
        let native_info = native_session.factorize(matrix).expect("native factorize");
        let native_enquiry = native_session.enquire_indef().expect("native enquire");

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

        let rust_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);

        // This dense APP boundary case now pins the production parity ladder
        // before the first optimized block_ldlt<32> inverse-D drift.
        assert_eq!(&rust_bits[..75], &native_bits[..75]);
        assert_ne!(
            rust_bits[75], native_bits[75],
            "dense seed09 case0 no longer differs at the current APP inverse-D boundary; promote the full witness"
        );
    }

    #[test]
    fn dense_seed09_case0_production_inverse_d_entries_match_through_pivot38_except_known_gap() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed09_case0_matrix();
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

        let rust_entries = rust_inverse_diagonal_entries(&rust_factor);
        let native_entries = native_enquiry.inverse_diagonal_entries;
        assert_eq!(rust_entries.len(), native_entries.len());

        // Native enquiry writes d(1,:) as the inverse-D diagonal component and
        // d(2,:) as the off-diagonal component in pivot order. The current
        // production guard first diverges at pivot 37, component 1, but the
        // following pivot still matches bitwise; the next diagonal drift starts
        // at pivot 39.
        for pivot in 0..=38 {
            for component in 0..2 {
                if pivot == 37 && component == 1 {
                    continue;
                }
                assert_eq!(
                    rust_entries[pivot][component].to_bits(),
                    native_entries[pivot][component].to_bits(),
                    "dense seed09 production inverse-D mismatch pivot={pivot} component={component}"
                );
            }
        }
        assert_ne!(
            rust_entries[37][1].to_bits(),
            native_entries[37][1].to_bits(),
            "dense seed09 production inverse-D pivot 37 component 1 now matches; promote the full prefix"
        );
        assert_ne!(
            rust_entries[39][0].to_bits(),
            native_entries[39][0].to_bits(),
            "dense seed09 production inverse-D pivot 39 diagonal now matches; promote this guard"
        );
    }

    #[test]
    fn dense_seed09_case0_production_inverse_d_structural_zero_components_match_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed09_case0_matrix();
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

        let rust_entries = rust_inverse_diagonal_entries(&rust_factor);
        let native_entries = native_enquiry.inverse_diagonal_entries;
        assert_eq!(rust_entries.len(), native_entries.len());

        let positive_zero_bits = 0.0f64.to_bits();
        for (pivot, (&rust_entry, &native_entry)) in
            rust_entries.iter().zip(&native_entries).enumerate()
        {
            let rust_structural_zero = rust_entry[1].to_bits() == positive_zero_bits;
            let native_structural_zero = native_entry[1].to_bits() == positive_zero_bits;
            assert_eq!(
                rust_structural_zero, native_structural_zero,
                "dense seed09 inverse-D structural zero layout mismatch pivot={pivot} rust={rust_entry:?} native={native_entry:?}"
            );
            if rust_structural_zero {
                assert_eq!(
                    rust_entry[1].to_bits(),
                    native_entry[1].to_bits(),
                    "dense seed09 inverse-D structural zero bit mismatch pivot={pivot}"
                );
            }
        }
    }

    #[test]
    fn dense_seed09_case0_production_inverse_d_mismatches_are_nonzero_numeric_components() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed09_case0_matrix();
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

        let rust_entries = rust_inverse_diagonal_entries(&rust_factor);
        let native_entries = native_enquiry.inverse_diagonal_entries;
        assert_eq!(rust_entries.len(), native_entries.len());

        let mismatches = rust_entries
            .iter()
            .zip(&native_entries)
            .enumerate()
            .flat_map(|(pivot, (rust_entry, native_entry))| {
                (0..2).filter_map(move |component| {
                    let rust_bits = rust_entry[component].to_bits();
                    let native_bits = native_entry[component].to_bits();
                    (rust_bits != native_bits).then_some((pivot, component, rust_bits, native_bits))
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(
            mismatches.first().copied(),
            Some((37, 1, 0xbf54_f658_1dd6_05fe, 0xbf54_f658_1dd6_05f2)),
        );
        assert!(
            mismatches
                .iter()
                .all(
                    |&(_, _, rust_bits, native_bits)| f64::from_bits(rust_bits) != 0.0
                        && f64::from_bits(native_bits) != 0.0
                ),
            "dense seed09 inverse-D mismatch escaped nonzero numeric components: {mismatches:?}"
        );
    }

    #[test]
    #[ignore = "manual exact production inverse-D bit mismatch witness for dense seed09 case0"]
    fn dense_seed09_case0_production_inverse_d_matches_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed09_case0_matrix();
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

        let rust_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);
        if rust_bits != native_bits {
            let index = rust_bits
                .iter()
                .zip(&native_bits)
                .position(|(rust, native)| rust != native)
                .unwrap_or(usize::MAX);
            panic!(
                "production inverse-D bit patterns differ on dense seed09 case0 APP boundary: first mismatch index={index} rust={:#018x} native={:#018x}",
                rust_bits[index], native_bits[index]
            );
        }
    }

    #[test]
    fn dense_seed09_first_app_update_and_tail_tpp_match_native_kernels() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, dense) = dense_seed09_case0_matrix();
        let mut rows = (0..dimension).collect::<Vec<_>>();
        let mut lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for row in col..dimension {
                lower_dense[dense_lower_offset(dimension, row, col)] = dense[row][col];
            }
        }
        let options = NumericFactorOptions::default();
        let mut scratch = vec![0.0; dimension * dimension];
        let block_start = 0;
        let block_end = APP_INNER_BLOCK_SIZE;
        let rows_before_block = rows.clone();
        let dense_before_block = lower_dense.clone();
        let mut local_stats = PanelFactorStats::default();
        let mut local_blocks = Vec::new();
        let mut block_pivot = block_start;

        while block_pivot < block_end {
            let Some((best_abs, best_row, best_col)) =
                dense_find_maxloc(&lower_dense, dimension, block_pivot, block_end)
            else {
                break;
            };
            assert!(best_abs >= options.small_pivot_tolerance);
            if best_row == best_col {
                if best_col != block_pivot {
                    dense_symmetric_swap_with_workspace(
                        &mut lower_dense,
                        dimension,
                        best_col,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(best_col, block_pivot);
                }
                let block = factor_one_by_one_common(
                    &rows,
                    &mut lower_dense,
                    dimension,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    &mut scratch,
                )
                .expect("1x1 APP pivot");
                local_blocks.push(block);
                block_pivot += 1;
                continue;
            }

            let first = best_col;
            let mut second = best_row;
            let a11 = lower_dense[dense_lower_offset(dimension, first, first)];
            let a22 = lower_dense[dense_lower_offset(dimension, second, second)];
            let a21 = lower_dense[dense_lower_offset(dimension, second, first)];
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
                        &mut lower_dense,
                        dimension,
                        index,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(index, block_pivot);
                }
                let block = factor_one_by_one_common(
                    &rows,
                    &mut lower_dense,
                    dimension,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    &mut scratch,
                )
                .expect("1x1 APP pivot");
                local_blocks.push(block);
                block_pivot += 1;
                continue;
            }

            let Some(inverse) = two_by_two_inverse else {
                break;
            };
            if first != block_pivot {
                dense_symmetric_swap_with_workspace(
                    &mut lower_dense,
                    dimension,
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
                    &mut lower_dense,
                    dimension,
                    second,
                    block_pivot + 1,
                    &mut scratch,
                );
                rows.swap(second, block_pivot + 1);
            }
            let block = factor_two_by_two_common(
                &rows,
                &mut lower_dense,
                DenseUpdateBounds {
                    size: dimension,
                    update_end: block_end,
                },
                block_pivot,
                inverse,
                &mut local_stats,
                &mut scratch,
            )
            .expect("2x2 APP pivot");
            local_blocks.push(block);
            block_pivot += 2;
        }

        let dense_before_apply = lower_dense.clone();
        app_apply_block_pivots_to_trailing_rows(
            &mut lower_dense,
            dimension,
            block_start,
            block_end,
            &local_blocks,
            options.small_pivot_tolerance,
            false,
        );
        let first_failed = app_first_failed_trailing_column(
            &lower_dense,
            dimension,
            block_start,
            block_end,
            options.threshold_pivot_u,
        );
        let local_passed = app_adjust_passed_prefix(&local_blocks, first_failed - block_start);
        let accepted_end = block_start + local_passed;
        let accepted_blocks = app_truncate_records_to_prefix(&local_blocks, local_passed);
        assert_eq!(accepted_end, APP_INNER_BLOCK_SIZE);

        let native_lda = native_aligned_double_stride(shim, dimension);
        let mut native_apply_matrix = vec![0.0; block_end * native_lda];
        for col in block_start..block_end {
            for row in col..dimension {
                native_apply_matrix[col * native_lda + row] =
                    dense_before_apply[col * dimension + row];
            }
        }
        let mut rust_apply_matrix = dense_before_apply.clone();
        app_apply_block_pivots_to_trailing_rows(
            &mut rust_apply_matrix,
            dimension,
            block_start,
            block_end,
            &local_blocks,
            options.small_pivot_tolerance,
            false,
        );
        let diagonal_before_apply = dense_tpp_diagonal_from_blocks(&local_blocks, block_end);
        unsafe {
            (shim.apply_pivot_op_n)(
                (dimension - block_end) as c_int,
                block_end as c_int,
                native_apply_matrix.as_ptr(),
                diagonal_before_apply.as_ptr(),
                options.small_pivot_tolerance,
                native_apply_matrix.as_mut_ptr().add(block_end),
                native_lda as c_int,
            );
        }
        for col in block_start..block_end {
            for row in block_end..dimension {
                let rust_value = rust_apply_matrix[col * dimension + row];
                let native_value = native_apply_matrix[col * native_lda + row];
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "dense seed09 APP-stride apply_pivot<OP_N> mismatch row={row} col={col}"
                );
            }
        }
        let native_passed = native_check_threshold_op_n(
            shim,
            dimension - block_end,
            block_end - block_start,
            options.threshold_pivot_u,
            &mut native_apply_matrix,
            block_end,
            native_lda,
        );
        assert_eq!(
            native_passed,
            first_failed - block_start,
            "dense seed09 APP a-posteriori threshold boundary mismatch"
        );
        let native_block = native_block_ldlt_32_from_lower_dense(
            shim,
            &dense_before_block,
            dimension,
            native_lda,
            options,
        );
        let native_trace =
            native_block_prefix_trace_32(shim, &dense_before_block, dimension, native_lda, options);
        let native_trace_block = native_trace.last().expect("native APP block prefix trace");
        let native_source_trace = native_block_prefix_trace_32_source(
            shim,
            &dense_before_block,
            dimension,
            native_lda,
            options,
        );
        let native_source_trace_block = native_source_trace
            .last()
            .expect("source-shaped native APP block prefix trace");
        assert_eq!(native_block.perm, rows[..block_end]);
        assert_eq!(
            native_trace_block.perm, native_block.perm,
            "dense seed09 native APP trace/block_ldlt permutation mismatch"
        );
        assert_eq!(
            native_trace_block.local_perm, native_block.local_perm,
            "dense seed09 native APP trace/block_ldlt local permutation mismatch"
        );
        for (index, (&trace_value, &block_value)) in native_trace_block
            .diagonal
            .iter()
            .zip(&native_block.diagonal)
            .enumerate()
        {
            assert_eq!(
                trace_value.to_bits(),
                block_value.to_bits(),
                "dense seed09 native APP trace/block_ldlt D mismatch index={index}"
            );
        }
        let mut first_native_trace_block_mismatch = None;
        'native_trace_block_compare: for col in block_start..block_end {
            for row in col..block_end {
                let trace_bits =
                    native_trace_block.matrix[col * APP_INNER_BLOCK_SIZE + row].to_bits();
                let block_bits = native_block.matrix[col * native_lda + row].to_bits();
                if trace_bits != block_bits {
                    first_native_trace_block_mismatch = Some((row, col, trace_bits, block_bits));
                    break 'native_trace_block_compare;
                }
            }
        }
        assert_eq!(
            first_native_trace_block_mismatch,
            Some((30, 19, 0xbf8c_bfa8_da67_4b6b, 0xbf8c_bfa8_da67_4b6c)),
            "dense seed09 native APP trace/block_ldlt matrix boundary moved"
        );
        assert_eq!(
            native_source_trace_block.perm, native_block.perm,
            "dense seed09 source native APP trace/block_ldlt permutation mismatch"
        );
        assert_eq!(
            native_source_trace_block.local_perm, native_block.local_perm,
            "dense seed09 source native APP trace/block_ldlt local permutation mismatch"
        );
        let mut first_source_trace_d_mismatch = None;
        for (index, (&trace_value, &block_value)) in native_source_trace_block
            .diagonal
            .iter()
            .zip(&native_block.diagonal)
            .enumerate()
        {
            if trace_value.to_bits() != block_value.to_bits() {
                first_source_trace_d_mismatch =
                    Some((index, trace_value.to_bits(), block_value.to_bits()));
                break;
            }
        }
        assert_eq!(
            first_source_trace_d_mismatch,
            Some((7, 0xbf2b_4429_642a_1ee2, 0xbf2b_4429_642a_1ee4)),
            "dense seed09 source native APP trace/block_ldlt D boundary moved"
        );
        let mut first_source_trace_block_mismatch = None;
        'source_trace_block_compare: for col in block_start..block_end {
            for row in col..block_end {
                let trace_bits =
                    native_source_trace_block.matrix[col * APP_INNER_BLOCK_SIZE + row].to_bits();
                let block_bits = native_block.matrix[col * native_lda + row].to_bits();
                if trace_bits != block_bits {
                    first_source_trace_block_mismatch = Some((row, col, trace_bits, block_bits));
                    break 'source_trace_block_compare;
                }
            }
        }
        assert_eq!(
            first_source_trace_block_mismatch,
            Some((2, 0, 0x3f79_e327_dcf6_7cce, 0x3f79_e327_dcf6_7ccf)),
            "dense seed09 source native APP trace/block_ldlt matrix boundary moved"
        );
        let mut native_source_apply_matrix = native_block.matrix.clone();
        for col in block_start..block_end {
            let source_col = native_block.local_perm[col];
            for row in block_end..dimension {
                native_source_apply_matrix[col * native_lda + row] =
                    dense_before_block[source_col * dimension + row];
            }
        }
        let mut first_source_pre_apply_mismatch = None;
        'source_pre_apply_compare: for col in block_start..block_end {
            for row in block_end..dimension {
                let rust_bits = dense_before_apply[col * dimension + row].to_bits();
                let native_bits = native_source_apply_matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_source_pre_apply_mismatch = Some((row, col, rust_bits, native_bits));
                    break 'source_pre_apply_compare;
                }
            }
        }
        assert_eq!(
            first_source_pre_apply_mismatch, None,
            "dense seed09 source-shaped first APP pre-apply operand mismatch"
        );
        let aligned_lower_dense =
            copy_lower_dense_to_stride(&dense_before_block, dimension, native_lda);
        let aligned_trace =
            rust_app_block_prefix_trace_32(&aligned_lower_dense, native_lda, options);
        let aligned_block = aligned_trace.last().expect("aligned Rust APP block trace");
        let mut first_aligned_rust_diagonal_mismatch = None;
        'aligned_rust_diagonal_compare: for col in block_start..block_end {
            for row in col..block_end {
                let rust_bits = dense_before_apply[col * dimension + row].to_bits();
                let aligned_bits = aligned_block.matrix[col * APP_INNER_BLOCK_SIZE + row].to_bits();
                if rust_bits != aligned_bits {
                    first_aligned_rust_diagonal_mismatch =
                        Some((row, col, rust_bits, aligned_bits));
                    break 'aligned_rust_diagonal_compare;
                }
            }
        }
        assert_eq!(
            first_aligned_rust_diagonal_mismatch, None,
            "dense seed09 production-stride and aligned Rust APP diagonal operands diverged"
        );
        let mut first_aligned_source_diagonal_mismatch = None;
        'aligned_source_diagonal_compare: for col in block_start..block_end {
            for row in col..block_end {
                let rust_bits = aligned_block.matrix[col * APP_INNER_BLOCK_SIZE + row].to_bits();
                let native_bits = native_source_apply_matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_aligned_source_diagonal_mismatch =
                        Some((row, col, rust_bits, native_bits));
                    break 'aligned_source_diagonal_compare;
                }
            }
        }
        assert_eq!(
            first_aligned_source_diagonal_mismatch,
            Some((30, 19, 0xbf8c_bfa8_da67_4b6b, 0xbf8c_bfa8_da67_4b6c)),
            "dense seed09 source-shaped aligned first APP diagonal operand boundary moved"
        );
        let mut first_source_diagonal_mismatch = None;
        'source_diagonal_compare: for col in block_start..block_end {
            for row in col..block_end {
                let rust_bits = dense_before_apply[col * dimension + row].to_bits();
                let native_bits = native_source_apply_matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_source_diagonal_mismatch = Some((row, col, rust_bits, native_bits));
                    break 'source_diagonal_compare;
                }
            }
        }
        assert_eq!(
            first_source_diagonal_mismatch,
            Some((30, 19, 0xbf8c_bfa8_da67_4b6b, 0xbf8c_bfa8_da67_4b6c)),
            "dense seed09 source-shaped first APP diagonal operand boundary moved"
        );
        let mut rust_source_trsm_matrix = dense_before_apply.clone();
        app_solve_block_triangular_to_trailing_rows(
            &mut rust_source_trsm_matrix,
            dimension,
            block_start,
            block_end,
            false,
        );
        let mut native_source_trsm_matrix = native_source_apply_matrix.clone();
        unsafe {
            (shim.host_trsm_right_lower_trans_unit)(
                (dimension - block_end) as c_int,
                block_end as c_int,
                native_source_trsm_matrix.as_ptr(),
                native_lda as c_int,
                native_source_trsm_matrix.as_mut_ptr().add(block_end),
                native_lda as c_int,
            );
        }
        let mut first_source_trsm_mismatch = None;
        'source_trsm_compare: for col in block_start..block_end {
            for row in block_end..dimension {
                let rust_bits = rust_source_trsm_matrix[col * dimension + row].to_bits();
                let native_bits = native_source_trsm_matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_source_trsm_mismatch = Some((row, col, rust_bits, native_bits));
                    break 'source_trsm_compare;
                }
            }
        }
        assert_eq!(
            first_source_trsm_mismatch,
            Some((47, 30, 0xc009_1687_167b_6783, 0xc009_1687_167b_6782)),
            "dense seed09 source-shaped first APP host_trsm operand boundary moved"
        );
        unsafe {
            (shim.apply_pivot_op_n)(
                (dimension - block_end) as c_int,
                block_end as c_int,
                native_source_apply_matrix.as_ptr(),
                native_block.diagonal.as_ptr(),
                options.small_pivot_tolerance,
                native_source_apply_matrix.as_mut_ptr().add(block_end),
                native_lda as c_int,
            );
        }
        let mut first_source_apply_mismatch = None;
        'source_apply_compare: for col in block_start..block_end {
            for row in block_end..dimension {
                let rust_bits = rust_apply_matrix[col * dimension + row].to_bits();
                let native_bits = native_source_apply_matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_source_apply_mismatch = Some((row, col, rust_bits, native_bits));
                    break 'source_apply_compare;
                }
            }
        }
        assert_eq!(
            first_source_apply_mismatch,
            Some((47, 30, 0xbfd7_6be5_86da_b26a, 0xbfd7_6be5_86da_b269)),
            "dense seed09 source-shaped first APP apply operand boundary moved"
        );
        let native_source_passed = native_check_threshold_op_n(
            shim,
            dimension - block_end,
            block_end - block_start,
            options.threshold_pivot_u,
            &mut native_source_apply_matrix,
            block_end,
            native_lda,
        );
        assert_eq!(
            native_source_passed,
            first_failed - block_start,
            "dense seed09 source-shaped APP threshold boundary mismatch"
        );

        app_restore_trailing_from_block_backup(
            &rows,
            &rows_before_block,
            &mut lower_dense,
            &dense_before_block,
            dimension,
            accepted_end,
        );
        let restored_lower_dense = lower_dense.clone();
        let tail_size = dimension - accepted_end;
        app_apply_accepted_prefix_update(
            &mut lower_dense,
            dimension,
            block_start,
            accepted_end,
            &accepted_blocks,
        );

        let native_ldld = native_aligned_double_stride(shim, APP_INNER_BLOCK_SIZE);
        let mut l_block = vec![0.0; accepted_end * native_lda];
        for col in 0..accepted_end {
            for row in 0..tail_size {
                l_block[col * native_lda + row] = lower_dense[col * dimension + accepted_end + row];
            }
        }
        let diagonal = dense_tpp_diagonal_from_blocks(&accepted_blocks, accepted_end);
        let mut ld_block = vec![0.0; accepted_end * native_ldld];
        unsafe {
            (shim.calc_ld_op_n)(
                tail_size as c_int,
                accepted_end as c_int,
                l_block.as_ptr(),
                native_lda as c_int,
                diagonal.as_ptr(),
                ld_block.as_mut_ptr(),
                native_ldld as c_int,
            );
        }
        let mut native_tail_update = vec![0.0; tail_size * native_lda];
        for col in 0..tail_size {
            for row in col..tail_size {
                native_tail_update[col * native_lda + row] =
                    restored_lower_dense[(accepted_end + col) * dimension + accepted_end + row];
            }
        }
        unsafe {
            (shim.host_gemm_op_n_op_t_update)(
                tail_size as c_int,
                tail_size as c_int,
                accepted_end as c_int,
                ld_block.as_ptr(),
                native_ldld as c_int,
                l_block.as_ptr(),
                native_lda as c_int,
                native_tail_update.as_mut_ptr(),
                native_lda as c_int,
            );
        }
        for col in 0..tail_size {
            for row in col..tail_size {
                let rust_value = lower_dense[(accepted_end + col) * dimension + accepted_end + row];
                let native_value = native_tail_update[col * native_lda + row];
                assert_eq!(
                    rust_value.to_bits(),
                    native_value.to_bits(),
                    "dense seed09 accepted APP update mismatch row={row} col={col}"
                );
            }
        }

        let mut tail = vec![0.0; tail_size * tail_size];
        for col in 0..tail_size {
            for row in col..tail_size {
                tail[col * tail_size + row] =
                    lower_dense[(accepted_end + col) * dimension + accepted_end + row];
            }
        }

        let rust = rust_ldlt_tpp_factor_from_lower_dense(&tail, tail_size, tail_size, options);
        let native =
            native_ldlt_tpp_factor_from_lower_dense(shim, &tail, tail_size, tail_size, options);
        assert_dense_tpp_kernel_results_equal(
            "dense seed09 tail after first APP block",
            &rust,
            &native,
            tail_size,
        );

        let native_tail_lda = native_aligned_double_stride(shim, dimension);
        let native_tail_ldld = native_aligned_double_stride(shim, APP_INNER_BLOCK_SIZE);
        let mut native_tail_matrix = vec![0.0; native_tail_lda * tail_size];
        for col in 0..tail_size {
            for row in col..tail_size {
                native_tail_matrix[col * native_tail_lda + row] = tail[col * tail_size + row];
            }
        }
        let mut native_tail_perm = (0..tail_size as c_int).collect::<Vec<_>>();
        let mut native_tail_diagonal = vec![0.0; 2 * tail_size];
        let mut native_tail_ld = vec![0.0; 2 * native_tail_ldld];
        let eliminated = unsafe {
            (shim.ldlt_tpp_factor)(
                tail_size as c_int,
                tail_size as c_int,
                native_tail_perm.as_mut_ptr(),
                native_tail_matrix.as_mut_ptr(),
                native_tail_lda as c_int,
                native_tail_diagonal.as_mut_ptr(),
                native_tail_ld.as_mut_ptr(),
                native_tail_ldld as c_int,
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                0,
                ptr::null_mut(),
                0,
            )
        };
        assert_eq!(eliminated, tail_size as c_int);
        for (index, (&rust_value, &native_value)) in
            rust.diagonal.iter().zip(&native_tail_diagonal).enumerate()
        {
            assert_eq!(
                rust_value.to_bits(),
                native_value.to_bits(),
                "dense seed09 APP-stride TPP tail D mismatch index={index} rust={rust_value:?} native={native_value:?}"
            );
        }

        // Mirror ldlt_app.cxx's recursive Block::factor tail call: the second
        // 32-wide block starts at an offset inside the original front but keeps
        // the parent front leading dimension.
        let mut native_embedded_tail_matrix = vec![0.0; native_tail_lda * dimension];
        let native_embedded_tail_offset = accepted_end * native_tail_lda + accepted_end;
        for col in 0..tail_size {
            for row in col..tail_size {
                native_embedded_tail_matrix
                    [native_embedded_tail_offset + col * native_tail_lda + row] =
                    tail[col * tail_size + row];
            }
        }
        let mut native_embedded_tail_perm = (0..tail_size as c_int).collect::<Vec<_>>();
        let mut native_embedded_tail_diagonal = vec![0.0; 2 * tail_size];
        let mut native_embedded_tail_ld = vec![0.0; 2 * native_tail_ldld];
        let embedded_eliminated = unsafe {
            (shim.ldlt_tpp_factor)(
                tail_size as c_int,
                tail_size as c_int,
                native_embedded_tail_perm.as_mut_ptr(),
                native_embedded_tail_matrix
                    .as_mut_ptr()
                    .add(native_embedded_tail_offset),
                native_tail_lda as c_int,
                native_embedded_tail_diagonal.as_mut_ptr(),
                native_embedded_tail_ld.as_mut_ptr(),
                native_tail_ldld as c_int,
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                0,
                ptr::null_mut(),
                0,
            )
        };
        assert_eq!(embedded_eliminated, tail_size as c_int);
        for (index, (&plain_value, &embedded_value)) in native_tail_diagonal
            .iter()
            .zip(&native_embedded_tail_diagonal)
            .enumerate()
        {
            assert_eq!(
                plain_value.to_bits(),
                embedded_value.to_bits(),
                "dense seed09 embedded native TPP tail D mismatch index={index} plain={plain_value:?} embedded={embedded_value:?}"
            );
        }

        // Mirror factor_node_indef's second-pass TPP call after APP accepts
        // the first block: the tail submatrix is addressed inside the parent
        // node with parent ldl, ldld is the tail height, and nleft/aleft allow
        // TPP row swaps to update the previously eliminated APP columns.
        let mut native_factor_node_matrix = vec![0.0; native_tail_lda * dimension];
        for col in 0..dimension {
            for row in col..dimension {
                native_factor_node_matrix[col * native_tail_lda + row] =
                    lower_dense[col * dimension + row];
            }
        }
        let mut native_factor_node_perm = rows.iter().map(|&row| row as c_int).collect::<Vec<_>>();
        let mut native_factor_node_tail_diagonal = vec![0.0; 2 * tail_size];
        let mut native_factor_node_tail_ld = vec![0.0; 2 * tail_size];
        let native_factor_node_tail_offset = accepted_end * native_tail_lda + accepted_end;
        let factor_node_eliminated = unsafe {
            (shim.ldlt_tpp_factor)(
                tail_size as c_int,
                tail_size as c_int,
                native_factor_node_perm.as_mut_ptr().add(accepted_end),
                native_factor_node_matrix
                    .as_mut_ptr()
                    .add(native_factor_node_tail_offset),
                native_tail_lda as c_int,
                native_factor_node_tail_diagonal.as_mut_ptr(),
                native_factor_node_tail_ld.as_mut_ptr(),
                tail_size as c_int,
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                accepted_end as c_int,
                native_factor_node_matrix.as_mut_ptr().add(accepted_end),
                native_tail_lda as c_int,
            )
        };
        assert_eq!(factor_node_eliminated, tail_size as c_int);

        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
            .expect("valid CSC");
        let (symbolic, _) = analyse(
            matrix,
            &SsidsOptions {
                ordering: OrderingStrategy::Natural,
            },
        )
        .expect("rust analyse");
        let (production, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");
        let production_entries = rust_inverse_diagonal_entries(&production);
        let isolated_entries =
            inverse_diagonal_entries_from_internal_diagonal(&rust.diagonal, tail_size);
        let factor_node_entries = inverse_diagonal_entries_from_internal_diagonal(
            &native_factor_node_tail_diagonal,
            tail_size,
        );
        for (index, (&production_value, &isolated_value)) in production_entries[accepted_end..]
            .iter()
            .flat_map(|entry| entry.iter())
            .zip(isolated_entries.iter().flat_map(|entry| entry.iter()))
            .enumerate()
        {
            assert_eq!(
                production_value.to_bits(),
                isolated_value.to_bits(),
                "dense seed09 production-vs-isolated TPP tail D mismatch index={index} production={production_value:?} isolated={isolated_value:?}"
            );
        }
        for (index, (&production_value, &factor_node_value)) in production_entries[accepted_end..]
            .iter()
            .flat_map(|entry| entry.iter())
            .zip(factor_node_entries.iter().flat_map(|entry| entry.iter()))
            .enumerate()
        {
            assert_eq!(
                production_value.to_bits(),
                factor_node_value.to_bits(),
                "dense seed09 production-vs-factor-node TPP tail D mismatch index={index} production={production_value:?} factor_node={factor_node_value:?}"
            );
        }
    }
}
