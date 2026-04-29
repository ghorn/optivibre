use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod match_order;
mod native;

use metis_ordering::{
    CsrGraph, NestedDissectionOptions, OrderingError, Permutation,
    approximate_minimum_degree_order, nested_dissection_order,
};
use rayon::prelude::*;
use thiserror::Error;

#[doc(hidden)]
pub use match_order::{SpralCscTrace, SpralMatchingTrace, spral_matching_trace};
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
    #[doc(hidden)]
    pub fn debug_branch_hits(&self) -> Vec<&'static str> {
        let mut hits = Vec::new();
        if self.forward_substitution_time > Duration::default()
            || self.diagonal_solve_time > Duration::default()
            || self.backward_substitution_time > Duration::default()
            || self.backward_trailing_update_columns > 0
            || self.backward_triangular_columns > 0
        {
            hits.push("ssids.solve.app_dense.forward_diag_backward");
        }
        hits
    }

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
    pub app_maxloc_time: Duration,
    pub app_symmetric_swap_time: Duration,
    pub app_pivot_update_time: Duration,
    pub app_front_count: usize,
    pub app_panel_count: usize,
    pub app_maxloc_calls: usize,
    pub app_symmetric_swaps: usize,
    pub app_one_by_one_pivots: usize,
    pub app_two_by_two_pivots: usize,
    pub app_zero_pivots: usize,
    pub app_front_size_histogram: [usize; 8],
    pub app_block_pivot_apply_time: Duration,
    pub app_block_triangular_solve_time: Duration,
    pub app_block_diagonal_apply_time: Duration,
    pub app_failed_pivot_scan_time: Duration,
    pub app_backup_time: Duration,
    pub app_restore_time: Duration,
    pub app_accepted_update_time: Duration,
    pub app_accepted_ld_time: Duration,
    pub app_accepted_gemm_time: Duration,
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
    #[doc(hidden)]
    pub fn debug_branch_hits(&self) -> Vec<&'static str> {
        let mut hits = Vec::new();
        if self.app_pivot_factor_time > Duration::default()
            || self.app_block_pivot_apply_time > Duration::default()
            || self.app_block_triangular_solve_time > Duration::default()
            || self.app_block_diagonal_apply_time > Duration::default()
        {
            hits.push("ssids.factor.app_dense.block_ldlt");
        }
        if self.app_maxloc_time > Duration::default()
            || self.app_failed_pivot_scan_time > Duration::default()
        {
            hits.push("ssids.factor.app_dense.maxloc");
        }
        if self.app_accepted_update_time > Duration::default()
            || self.app_accepted_gemm_time > Duration::default()
        {
            hits.push("ssids.factor.app_dense.accepted_update");
        }
        hits
    }

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
        self.app_maxloc_time += other.app_maxloc_time;
        self.app_symmetric_swap_time += other.app_symmetric_swap_time;
        self.app_pivot_update_time += other.app_pivot_update_time;
        self.app_front_count += other.app_front_count;
        self.app_panel_count += other.app_panel_count;
        self.app_maxloc_calls += other.app_maxloc_calls;
        self.app_symmetric_swaps += other.app_symmetric_swaps;
        self.app_one_by_one_pivots += other.app_one_by_one_pivots;
        self.app_two_by_two_pivots += other.app_two_by_two_pivots;
        self.app_zero_pivots += other.app_zero_pivots;
        for (lhs, rhs) in self
            .app_front_size_histogram
            .iter_mut()
            .zip(other.app_front_size_histogram)
        {
            *lhs += rhs;
        }
        self.app_block_pivot_apply_time += other.app_block_pivot_apply_time;
        self.app_block_triangular_solve_time += other.app_block_triangular_solve_time;
        self.app_block_diagonal_apply_time += other.app_block_diagonal_apply_time;
        self.app_failed_pivot_scan_time += other.app_failed_pivot_scan_time;
        self.app_backup_time += other.app_backup_time;
        self.app_restore_time += other.app_restore_time;
        self.app_accepted_update_time += other.app_accepted_update_time;
        self.app_accepted_ld_time += other.app_accepted_ld_time;
        self.app_accepted_gemm_time += other.app_accepted_gemm_time;
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
    /// Mirror SPRAL SSIDS `options%ordering = 2`: matching-based scaling,
    /// cycle splitting, compressed METIS-style ordering, then native symbolic
    /// analyse. The saved analyse-time scaling is only used when numeric
    /// factorization opts into `NumericScalingStrategy::SavedMatching`.
    SpralMatching,
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

impl SsidsOptions {
    pub fn spral_default() -> Self {
        Self {
            ordering: OrderingStrategy::SpralMatching,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotMethod {
    AggressiveAposteriori,
    BlockAposteriori,
    ThresholdPartial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericScalingStrategy {
    None,
    SavedMatching,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericFactorOptions {
    pub action_on_zero_pivot: bool,
    pub pivot_method: PivotMethod,
    pub small_pivot_tolerance: f64,
    pub threshold_pivot_u: f64,
    pub inertia_zero_tol: f64,
    pub scaling: NumericScalingStrategy,
}

impl Default for NumericFactorOptions {
    fn default() -> Self {
        Self {
            action_on_zero_pivot: true,
            pivot_method: PivotMethod::BlockAposteriori,
            small_pivot_tolerance: 1e-20,
            threshold_pivot_u: 1e-8,
            inertia_zero_tol: 0.0,
            scaling: NumericScalingStrategy::None,
        }
    }
}

impl NumericFactorOptions {
    pub fn spral_default() -> Self {
        Self {
            scaling: NumericScalingStrategy::SavedMatching,
            ..Self::default()
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

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicFactor {
    pub permutation: Permutation,
    pub elimination_tree: Vec<Option<usize>>,
    pub column_counts: Vec<usize>,
    pub column_pattern: Vec<Vec<usize>>,
    pub supernodes: Vec<Supernode>,
    saved_matching_scaling: Option<Vec<f64>>,
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
    scaling: Option<Vec<f64>>,
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
            let mut value = rhs[self.permutation.perm()[ordered_index]];
            if let Some(scaling) = &self.scaling {
                value *= scaling[ordered_index];
            }
            factor_rhs[factor_position] = value;
        }
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .input_permutation_time += started.elapsed();
        }

        if self.dimension > 0 && self.solve_panels.is_empty() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: self.dimension - 1,
                detail: "solve panel metadata is missing for forward substitution".into(),
            });
        }
        let started = profile_enabled.then(Instant::now);
        solve_forward_front_panels_like_native(&self.solve_panels, factor_rhs);
        if let Some(started) = started {
            profile
                .as_mut()
                .expect("profile exists when timing is enabled")
                .forward_substitution_time += started.elapsed();
        }

        let diagonal_time_before = profile
            .as_ref()
            .map(|profile| profile.diagonal_solve_time)
            .unwrap_or_default();
        let started = profile_enabled.then(Instant::now);
        solve_diagonal_and_lower_transpose_front_panels_like_native(
            &self.solve_panels,
            &self.diagonal_blocks,
            &self.diagonal_values,
            factor_rhs,
            profile.as_deref_mut(),
        )?;
        if let Some(started) = started {
            let profile = profile
                .as_mut()
                .expect("profile exists when timing is enabled");
            let diagonal_elapsed = profile
                .diagonal_solve_time
                .saturating_sub(diagonal_time_before);
            profile.backward_substitution_time +=
                started.elapsed().saturating_sub(diagonal_elapsed);
        }
        if !factor_rhs.iter().all(|value| value.is_finite()) {
            return Err(SsidsError::NumericalBreakdown {
                pivot: self.dimension.saturating_sub(1),
                detail: "solve produced non-finite values".into(),
            });
        }
        let started = profile_enabled.then(Instant::now);
        for (factor_position, &ordered_index) in self.factor_order.iter().enumerate() {
            let mut value = factor_rhs[factor_position];
            if let Some(scaling) = &self.scaling {
                value *= scaling[ordered_index];
            }
            rhs[self.permutation.perm()[ordered_index]] = value;
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
        let scaling = self.scaling.clone();
        let factorization = multifrontal_factorize_with_tree(
            matrix,
            &self.permutation,
            &self.symbolic_front_tree,
            self.options,
            scaling.as_deref(),
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
    #[error("saved matching scaling is required but symbolic analyse did not produce it")]
    MissingSavedMatchingScaling,
}

fn analyse_debug_enabled() -> bool {
    std::env::var_os("SPRAL_SSIDS_DEBUG_ANALYSE").is_some()
}

fn analyse_debug_log(message: impl AsRef<str>) {
    if analyse_debug_enabled() {
        eprintln!("{}", message.as_ref());
    }
}

fn factor_debug_enabled() -> bool {
    std::env::var_os("SPRAL_SSIDS_DEBUG_FACTOR").is_some()
}

fn factor_app_subphase_debug_enabled() -> bool {
    matches!(
        std::env::var("SPRAL_SSIDS_DEBUG_FACTOR_APP_SUBPHASES")
            .as_deref()
            .map(str::to_ascii_lowercase),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

fn factor_debug_log(message: impl AsRef<str>) {
    if factor_debug_enabled() {
        eprintln!("{}", message.as_ref());
    }
}

fn app_front_size_histogram_bucket(size: usize) -> usize {
    match size {
        0..=32 => 0,
        33..=64 => 1,
        65..=96 => 2,
        97..=128 => 3,
        129..=160 => 4,
        161..=256 => 5,
        257..=512 => 6,
        _ => 7,
    }
}

fn graph_from_lower_csc(matrix: SymmetricCscMatrix<'_>) -> Result<CsrGraph, SsidsError> {
    let dimension = matrix.dimension();
    let mut degree = vec![0usize; dimension];
    for col in 0..dimension {
        for &row in &matrix.row_indices()[matrix.col_ptrs()[col]..matrix.col_ptrs()[col + 1]] {
            if row == col {
                continue;
            }
            degree[col] += 1;
            degree[row] += 1;
        }
    }

    let mut offsets = Vec::with_capacity(dimension + 1);
    offsets.push(0);
    for &degree in &degree {
        offsets.push(offsets[offsets.len() - 1] + degree);
    }
    let mut write = offsets[..dimension].to_vec();
    let mut neighbors = vec![0usize; offsets[dimension]];
    for col in 0..dimension {
        for &row in &matrix.row_indices()[matrix.col_ptrs()[col]..matrix.col_ptrs()[col + 1]] {
            if row == col {
                continue;
            }
            neighbors[write[row]] = col;
            write[row] += 1;
            neighbors[write[col]] = row;
            write[col] += 1;
        }
    }
    debug_assert_eq!(&write, &offsets[1..]);
    Ok(CsrGraph::from_trusted_sorted_adjacency(offsets, neighbors))
}

pub fn analyse(
    matrix: SymmetricCscMatrix<'_>,
    options: &SsidsOptions,
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    let graph = graph_from_lower_csc(matrix)?;
    let column_has_entries = (0..matrix.dimension())
        .map(|col| matrix.col_ptrs()[col + 1] > matrix.col_ptrs()[col])
        .collect::<Vec<_>>();
    let analyse_started = Instant::now();
    match options.ordering {
        OrderingStrategy::Natural => {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] strategy=natural dim={} nnz={}",
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
                "[ssids_rs::analyse] strategy=natural done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::ApproximateMinimumDegree => {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] strategy=amd dim={} nnz={} start",
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
                "[ssids_rs::analyse] strategy=amd done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::NestedDissection(ordering_options) => {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] strategy=nested_dissection dim={} nnz={} start",
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
                "[ssids_rs::analyse] strategy=nested_dissection done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::SpralMatching => {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] strategy=spral_matching dim={} nnz={} start",
                matrix.dimension(),
                matrix.row_indices().len(),
            ));
            let matching = match_order::spral_matching_order(matrix)?;
            let permutation = permutation_from_native_order(matrix.dimension(), &matching.order)?;
            let result = build_symbolic_result_with_native_order_and_scaling(
                matrix,
                &graph,
                permutation,
                &column_has_entries,
                "spral_matching",
                Some(matching.scaling),
            )?;
            analyse_debug_log(format!(
                "[ssids_rs::analyse] strategy=spral_matching done in {:.3}s fill={} supernodes={}",
                analyse_started.elapsed().as_secs_f64(),
                result.1.estimated_fill_nnz,
                result.1.supernode_count,
            ));
            Ok(result)
        }
        OrderingStrategy::Auto(ordering_options) => {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] strategy=auto dim={} nnz={} start",
                matrix.dimension(),
                matrix.row_indices().len(),
            ));
            let natural_permutation = Permutation::identity(matrix.dimension());
            analyse_debug_log("[ssids_rs::analyse] auto natural symbolic start");
            let (_, natural_counts, _) = symbolic_factor_pattern(&graph);
            let natural_fill = natural_counts.iter().sum::<usize>();
            analyse_debug_log(format!(
                "[ssids_rs::analyse] auto natural symbolic done fill={} elapsed={:.3}s",
                natural_fill,
                analyse_started.elapsed().as_secs_f64(),
            ));

            let amd_started = Instant::now();
            analyse_debug_log("[ssids_rs::analyse] auto amd start");
            let amd_summary = approximate_minimum_degree_order(&graph)?;
            let amd_graph = permute_graph(&graph, &amd_summary.permutation);
            let (_, amd_counts, _) = symbolic_factor_pattern(&amd_graph);
            let amd_fill = amd_counts.iter().sum::<usize>();
            analyse_debug_log(format!(
                "[ssids_rs::analyse] auto amd done fill={} elapsed={:.3}s total={:.3}s",
                amd_fill,
                amd_started.elapsed().as_secs_f64(),
                analyse_started.elapsed().as_secs_f64(),
            ));

            let nd_started = Instant::now();
            analyse_debug_log("[ssids_rs::analyse] auto nested dissection start");
            let summary = nested_dissection_order(&graph, &ordering_options)?;
            let permuted_graph = permute_graph(&graph, &summary.permutation);
            let (_, nd_counts, _) = symbolic_factor_pattern(&permuted_graph);
            let nd_fill = nd_counts.iter().sum::<usize>();
            analyse_debug_log(format!(
                "[ssids_rs::analyse] auto nested dissection done fill={} elapsed={:.3}s total={:.3}s",
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
                    "[ssids_rs::analyse] auto selected=amd total={:.3}s",
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
                    "[ssids_rs::analyse] auto selected=nested_dissection total={:.3}s",
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
                    "[ssids_rs::analyse] auto selected=natural total={:.3}s",
                    analyse_started.elapsed().as_secs_f64(),
                ));
                Ok(result)
            }
        }
    }
}

/// Analyse with a user-supplied pivot order in SPRAL's C-facing convention:
/// `order[original_column]` is the zero-based position of that column in the
/// pivot sequence.
pub fn analyse_with_user_ordering(
    matrix: SymmetricCscMatrix<'_>,
    order: &[usize],
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    let graph = graph_from_lower_csc(matrix)?;
    let column_has_entries = (0..matrix.dimension())
        .map(|col| matrix.col_ptrs()[col + 1] > matrix.col_ptrs()[col])
        .collect::<Vec<_>>();
    let permutation = permutation_from_native_order(matrix.dimension(), order)?;
    build_symbolic_result_with_native_order(
        matrix,
        &graph,
        permutation,
        &column_has_entries,
        "user_supplied",
    )
}

pub fn approximate_minimum_degree_permutation(
    matrix: SymmetricCscMatrix<'_>,
) -> Result<Permutation, SsidsError> {
    let graph = graph_from_lower_csc(matrix)?;
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
    profile: Option<&mut FactorProfile>,
) -> Result<(NumericFactor, FactorInfo), SsidsError> {
    if matrix.dimension() != symbolic.permutation.len() {
        return Err(SsidsError::DimensionMismatch {
            expected: symbolic.permutation.len(),
            actual: matrix.dimension(),
        });
    }
    let factor_started = factor_debug_enabled().then(Instant::now);
    let numeric_scaling = numeric_scaling_for_symbolic(symbolic, options)?;
    let debug_profile_enabled = factor_started.is_some();
    let mut debug_profile = FactorProfile::default();
    let profile_enabled = profile.is_some() || debug_profile_enabled;
    let profile_ref = match profile {
        Some(profile) => profile,
        None => &mut debug_profile,
    };
    let started = profile_enabled.then(Instant::now);
    let front_tree = build_symbolic_front_tree(symbolic);
    if let Some(started) = started {
        profile_ref.symbolic_front_tree_time += started.elapsed();
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
        scaling: numeric_scaling,
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
    let info = factor.refactorize_with_cached_symbolic_profile(
        matrix,
        profile_enabled.then_some(&mut *profile_ref),
    )?;
    if let Some(started) = factor_started {
        factor_debug_log(format!(
            "[ssids_rs::factorize] dim={} supernodes={} scaling={:?} total={:.6}s symbolic_front_tree={:.6}s permuted_pattern={:.6}s permuted_values={:.6}s front_factorization={:.6}s front_assembly={:.6}s dense_front={:.6}s tpp={:.6}s app_pivot={:.6}s app_maxloc={:.6}s app_swap={:.6}s app_pivot_update={:.6}s app_apply={:.6}s app_triangular={:.6}s app_diagonal={:.6}s app_failed_scan={:.6}s app_backup={:.6}s app_restore={:.6}s app_accepted_update={:.6}s app_accepted_ld={:.6}s app_accepted_gemm={:.6}s app_column_storage={:.6}s solve_panel_build={:.6}s root_delayed={:.6}s factor_inverse={:.6}s lower_storage={:.6}s solve_panel_storage={:.6}s diagonal_storage={:.6}s factor_bytes={:.6}s fronts={} local_dense_entries={} app_fronts={} app_panels={} app_maxloc_calls={} app_swaps={} app_1x1={} app_2x2={} app_zero={} app_front_le32={} app_front_33_64={} app_front_65_96={} app_front_97_128={} app_front_129_160={} app_front_161_256={} app_front_257_512={} app_front_gt512={}",
            matrix.dimension(),
            symbolic.supernodes.len(),
            options.scaling,
            started.elapsed().as_secs_f64(),
            profile_ref.symbolic_front_tree_time.as_secs_f64(),
            profile_ref.permuted_pattern_time.as_secs_f64(),
            profile_ref.permuted_values_time.as_secs_f64(),
            profile_ref.front_factorization_time.as_secs_f64(),
            profile_ref.front_assembly_time.as_secs_f64(),
            profile_ref.dense_front_factorization_time.as_secs_f64(),
            profile_ref.tpp_factorization_time.as_secs_f64(),
            profile_ref.app_pivot_factor_time.as_secs_f64(),
            profile_ref.app_maxloc_time.as_secs_f64(),
            profile_ref.app_symmetric_swap_time.as_secs_f64(),
            profile_ref.app_pivot_update_time.as_secs_f64(),
            profile_ref.app_block_pivot_apply_time.as_secs_f64(),
            profile_ref.app_block_triangular_solve_time.as_secs_f64(),
            profile_ref.app_block_diagonal_apply_time.as_secs_f64(),
            profile_ref.app_failed_pivot_scan_time.as_secs_f64(),
            profile_ref.app_backup_time.as_secs_f64(),
            profile_ref.app_restore_time.as_secs_f64(),
            profile_ref.app_accepted_update_time.as_secs_f64(),
            profile_ref.app_accepted_ld_time.as_secs_f64(),
            profile_ref.app_accepted_gemm_time.as_secs_f64(),
            profile_ref.app_column_storage_time.as_secs_f64(),
            profile_ref.solve_panel_build_time.as_secs_f64(),
            profile_ref.root_delayed_factorization_time.as_secs_f64(),
            profile_ref.factor_inverse_time.as_secs_f64(),
            profile_ref.lower_storage_time.as_secs_f64(),
            profile_ref.solve_panel_storage_time.as_secs_f64(),
            profile_ref.diagonal_storage_time.as_secs_f64(),
            profile_ref.factor_bytes_time.as_secs_f64(),
            profile_ref.front_count,
            profile_ref.local_dense_entries,
            profile_ref.app_front_count,
            profile_ref.app_panel_count,
            profile_ref.app_maxloc_calls,
            profile_ref.app_symmetric_swaps,
            profile_ref.app_one_by_one_pivots,
            profile_ref.app_two_by_two_pivots,
            profile_ref.app_zero_pivots,
            profile_ref.app_front_size_histogram[0],
            profile_ref.app_front_size_histogram[1],
            profile_ref.app_front_size_histogram[2],
            profile_ref.app_front_size_histogram[3],
            profile_ref.app_front_size_histogram[4],
            profile_ref.app_front_size_histogram[5],
            profile_ref.app_front_size_histogram[6],
            profile_ref.app_front_size_histogram[7],
        ));
    }
    Ok((factor, info))
}

fn numeric_scaling_for_symbolic(
    symbolic: &SymbolicFactor,
    options: &NumericFactorOptions,
) -> Result<Option<Vec<f64>>, SsidsError> {
    match options.scaling {
        NumericScalingStrategy::None => Ok(None),
        NumericScalingStrategy::SavedMatching => {
            let saved = symbolic
                .saved_matching_scaling
                .as_ref()
                .ok_or(SsidsError::MissingSavedMatchingScaling)?;
            if saved.len() != symbolic.permutation.len() {
                return Err(SsidsError::DimensionMismatch {
                    expected: symbolic.permutation.len(),
                    actual: saved.len(),
                });
            }
            Ok(Some(
                symbolic
                    .permutation
                    .perm()
                    .iter()
                    .map(|&original| saved[original])
                    .collect(),
            ))
        }
    }
}

fn permute_graph(graph: &CsrGraph, permutation: &Permutation) -> CsrGraph {
    if is_identity_order(permutation.perm()) {
        return graph.clone();
    }
    if graph.vertex_count() <= 2048 {
        return permute_graph_with_bitsets(graph, permutation);
    }

    permute_graph_with_sorted_edges(graph, permutation)
}

fn permute_graph_with_sorted_edges(graph: &CsrGraph, permutation: &Permutation) -> CsrGraph {
    let dimension = graph.vertex_count();
    let inverse = permutation.inverse();
    let mut degrees = vec![0_usize; dimension];
    for vertex in 0..dimension {
        for &neighbor in graph.neighbors(vertex) {
            if vertex < neighbor {
                degrees[inverse[vertex]] += 1;
                degrees[inverse[neighbor]] += 1;
            }
        }
    }

    let mut offsets = vec![0_usize; dimension + 1];
    for vertex in 0..dimension {
        offsets[vertex + 1] = offsets[vertex] + degrees[vertex];
    }

    let mut cursors = offsets[..dimension].to_vec();
    let mut neighbors = vec![0_usize; offsets[dimension]];
    for vertex in 0..dimension {
        for &neighbor in graph.neighbors(vertex) {
            if vertex < neighbor {
                let lhs = inverse[vertex];
                let rhs = inverse[neighbor];
                neighbors[cursors[lhs]] = rhs;
                cursors[lhs] += 1;
                neighbors[cursors[rhs]] = lhs;
                cursors[rhs] += 1;
            }
        }
    }
    for vertex in 0..dimension {
        neighbors[offsets[vertex]..offsets[vertex + 1]].sort_unstable();
    }
    CsrGraph::from_trusted_sorted_adjacency(offsets, neighbors)
}

fn permute_graph_with_bitsets(graph: &CsrGraph, permutation: &Permutation) -> CsrGraph {
    let dimension = graph.vertex_count();
    let words_per_row = dimension.div_ceil(64);
    let inverse = permutation.inverse();
    let mut edge_bits = vec![0_u64; dimension * words_per_row];
    for vertex in 0..dimension {
        for &neighbor in graph.neighbors(vertex) {
            if vertex < neighbor {
                let lhs = inverse[vertex];
                let rhs = inverse[neighbor];
                symbolic_edge_set(&mut edge_bits, words_per_row, lhs, rhs);
                symbolic_edge_set(&mut edge_bits, words_per_row, rhs, lhs);
            }
        }
    }

    let mut offsets = Vec::with_capacity(dimension + 1);
    offsets.push(0);
    for vertex in 0..dimension {
        let row_start = vertex * words_per_row;
        let degree = edge_bits[row_start..row_start + words_per_row]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        offsets.push(offsets[vertex] + degree);
    }

    let mut neighbors = Vec::with_capacity(offsets[dimension]);
    for vertex in 0..dimension {
        let row_start = vertex * words_per_row;
        for (word_index, &word) in edge_bits[row_start..row_start + words_per_row]
            .iter()
            .enumerate()
        {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let neighbor = word_index * 64 + bit;
                if neighbor < dimension {
                    neighbors.push(neighbor);
                }
                bits &= bits - 1;
            }
        }
    }
    CsrGraph::from_trusted_sorted_adjacency(offsets, neighbors)
}

fn build_symbolic_result_with_native_order(
    matrix: SymmetricCscMatrix<'_>,
    graph: &CsrGraph,
    current_permutation: Permutation,
    column_has_entries: &[bool],
    ordering_kind: &'static str,
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    build_symbolic_result_with_native_order_and_scaling(
        matrix,
        graph,
        current_permutation,
        column_has_entries,
        ordering_kind,
        None,
    )
}

fn build_symbolic_result_with_native_order_and_scaling(
    matrix: SymmetricCscMatrix<'_>,
    graph: &CsrGraph,
    mut current_permutation: Permutation,
    column_has_entries: &[bool],
    ordering_kind: &'static str,
    saved_matching_scaling: Option<Vec<f64>>,
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    let trace = analyse_debug_enabled();
    let symbolic_started = Instant::now();
    let phase_started = Instant::now();
    let mut current_graph = permute_graph(graph, &current_permutation);
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic initial_permute_graph elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let phase_started = Instant::now();
    let (initial_tree, initial_column_counts, initial_column_pattern) =
        symbolic_factor_pattern(&current_graph);
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic initial_factor_pattern elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let phase_started = Instant::now();
    let (postorder_permutation, realn) =
        native_postorder_permutation(&initial_tree, &current_permutation, column_has_entries);
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic postorder elapsed={:.6}s realn={} total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            realn,
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let postorder_is_identity = is_identity_order(&postorder_permutation);
    if !postorder_is_identity {
        let phase_started = Instant::now();
        current_permutation = compose_ordering_with_symbolic_permutation(
            &current_permutation,
            &postorder_permutation,
        )?;
        current_graph = permute_graph(graph, &current_permutation);
        if trace {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] symbolic postorder_apply elapsed={:.6}s total={:.6}s",
                phase_started.elapsed().as_secs_f64(),
                symbolic_started.elapsed().as_secs_f64(),
            ));
        }
    }

    let (elimination_tree, simulated_column_counts, column_pattern) = if postorder_is_identity {
        if trace {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] symbolic final_factor_pattern reused_initial=true total={:.6}s",
                symbolic_started.elapsed().as_secs_f64(),
            ));
        }
        (initial_tree, initial_column_counts, initial_column_pattern)
    } else {
        let phase_started = Instant::now();
        let factor_pattern = symbolic_factor_pattern(&current_graph);
        if trace {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] symbolic final_factor_pattern elapsed={:.6}s total={:.6}s",
                phase_started.elapsed().as_secs_f64(),
                symbolic_started.elapsed().as_secs_f64(),
            ));
        }
        factor_pattern
    };
    let phase_started = Instant::now();
    #[cfg(debug_assertions)]
    {
        let expanded_pattern = expand_symmetric_pattern(matrix);
        let native_counts =
            native_column_counts(&expanded_pattern, &current_permutation, &elimination_tree);
        debug_assert_eq!(native_counts, simulated_column_counts);
    }
    let column_counts = simulated_column_counts;
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic column_counts elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let phase_started = Instant::now();
    let supernode_layout = native_supernode_layout(&elimination_tree, &column_counts, realn);
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic supernode_layout elapsed={:.6}s ranges={} total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            supernode_layout.ranges.len(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    if is_identity_order(&supernode_layout.permutation) {
        let phase_started = Instant::now();
        let supernodes = build_native_row_list_supernodes_fast(&supernode_layout, &column_pattern)
            .unwrap_or_else(|| {
                let expanded_pattern = expand_symmetric_pattern(matrix);
                build_native_row_list_supernodes(
                    &expanded_pattern,
                    &current_permutation,
                    &supernode_layout,
                    &column_pattern,
                )
            });
        if trace {
            analyse_debug_log(format!(
                "[ssids_rs::analyse] symbolic row_lists elapsed={:.6}s total={:.6}s",
                phase_started.elapsed().as_secs_f64(),
                symbolic_started.elapsed().as_secs_f64(),
            ));
        }
        return Ok(build_symbolic_result(
            current_permutation,
            elimination_tree,
            column_counts,
            column_pattern,
            supernodes,
            ordering_kind,
            saved_matching_scaling,
        ));
    }

    let phase_started = Instant::now();
    let final_permutation = compose_ordering_with_symbolic_permutation(
        &current_permutation,
        &supernode_layout.permutation,
    )?;
    let final_graph = permute_graph(graph, &final_permutation);
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic supernode_apply elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let phase_started = Instant::now();
    let (final_tree, simulated_final_counts, final_pattern) = symbolic_factor_pattern(&final_graph);
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic supernode_factor_pattern elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let phase_started = Instant::now();
    #[cfg(debug_assertions)]
    {
        let expanded_pattern = expand_symmetric_pattern(matrix);
        let native_counts =
            native_column_counts(&expanded_pattern, &final_permutation, &final_tree);
        debug_assert_eq!(native_counts, simulated_final_counts);
    }
    let final_counts = simulated_final_counts;
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic final_column_counts elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    let phase_started = Instant::now();
    let final_supernodes = build_native_row_list_supernodes_fast(&supernode_layout, &final_pattern)
        .unwrap_or_else(|| {
            let expanded_pattern = expand_symmetric_pattern(matrix);
            build_native_row_list_supernodes(
                &expanded_pattern,
                &final_permutation,
                &supernode_layout,
                &final_pattern,
            )
        });
    if trace {
        analyse_debug_log(format!(
            "[ssids_rs::analyse] symbolic final_row_lists elapsed={:.6}s total={:.6}s",
            phase_started.elapsed().as_secs_f64(),
            symbolic_started.elapsed().as_secs_f64(),
        ));
    }
    Ok(build_symbolic_result(
        final_permutation,
        final_tree,
        final_counts,
        final_pattern,
        final_supernodes,
        ordering_kind,
        saved_matching_scaling,
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

fn permutation_from_native_order(
    dimension: usize,
    order: &[usize],
) -> Result<Permutation, OrderingError> {
    if order.len() != dimension {
        return Err(OrderingError::InvalidPermutation(format!(
            "expected {dimension} user-order entries, got {}",
            order.len()
        )));
    }
    let mut perm = vec![usize::MAX; dimension];
    for (original, &position) in order.iter().enumerate() {
        if position >= dimension {
            return Err(OrderingError::InvalidPermutation(format!(
                "order[{original}]={position} is out of bounds for {dimension} columns"
            )));
        }
        if perm[position] != usize::MAX {
            return Err(OrderingError::InvalidPermutation(format!(
                "duplicate pivot position {position}"
            )));
        }
        perm[position] = original;
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

#[cfg(any(debug_assertions, test))]
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

#[cfg(any(debug_assertions, test))]
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
    let words_per_row = dimension.div_ceil(64);
    let mut edge_bits = vec![0_u64; dimension * words_per_row];
    for vertex in 0..dimension {
        for &neighbor in graph.neighbors(vertex) {
            symbolic_edge_set(&mut edge_bits, words_per_row, vertex, neighbor);
        }
    }
    let mut active_words = vec![0_u64; words_per_row];
    let mut elimination_tree = vec![None; dimension];
    let mut column_counts = vec![1; dimension];
    let mut column_pattern = Vec::with_capacity(dimension);
    for column in 0..dimension {
        symbolic_trailing_row_bits(&edge_bits, words_per_row, column, &mut active_words);
        let active_count = symbolic_bit_count(&active_words);
        let mut pattern = Vec::with_capacity(active_count + 1);
        pattern.push(column);
        symbolic_push_vertices(&active_words, dimension, &mut pattern);
        elimination_tree[column] = pattern.get(1).copied();
        column_counts[column] = active_count + 1;
        for &neighbor in &pattern[1..] {
            let row_start = neighbor * words_per_row;
            for word in 0..words_per_row {
                edge_bits[row_start + word] |= active_words[word];
            }
        }
        column_pattern.push(pattern);
    }
    (elimination_tree, column_counts, column_pattern)
}

fn symbolic_trailing_row_bits(
    edge_bits: &[u64],
    words_per_row: usize,
    row: usize,
    active_words: &mut [u64],
) {
    let row_start = row * words_per_row;
    active_words.copy_from_slice(&edge_bits[row_start..row_start + words_per_row]);
    let word = row / 64;
    active_words[..word].fill(0);
    if let Some(current_word) = active_words.get_mut(word) {
        let bit = row % 64;
        *current_word &= if bit == 63 {
            0
        } else {
            !((1_u64 << (bit + 1)) - 1)
        };
    }
}

fn symbolic_bit_count(words: &[u64]) -> usize {
    words.iter().map(|word| word.count_ones() as usize).sum()
}

fn symbolic_push_vertices(words: &[u64], dimension: usize, vertices: &mut Vec<usize>) {
    for (word_index, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let vertex = word_index * 64 + bit;
            if vertex < dimension {
                vertices.push(vertex);
            }
            bits &= bits - 1;
        }
    }
}

fn symbolic_edge_set(edge_bits: &mut [u64], words_per_row: usize, row: usize, col: usize) {
    if words_per_row == 0 {
        return;
    }
    let offset = row * words_per_row + col / 64;
    edge_bits[offset] |= 1_u64 << (col % 64);
}

fn build_native_row_list_supernodes_fast(
    layout: &NativeSupernodeLayout,
    column_pattern: &[Vec<usize>],
) -> Option<Vec<Supernode>> {
    let range = layout.ranges.first()?;
    if layout.ranges.len() != 1 || range.start != 0 || range.end != column_pattern.len() {
        return None;
    }
    if layout.parents.as_slice() != [None] {
        return None;
    }

    // For a single full-rank supernode, SPRAL core_analyse.find_row_lists first
    // inserts every eliminated pivot into the row list. The later child and
    // matrix-entry passes cannot append anything new, so dbl_tr_sort leaves the
    // node with no trailing rows.
    Some(vec![Supernode {
        start_column: 0,
        end_column: column_pattern.len(),
        trailing_rows: Vec::new(),
    }])
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
    saved_matching_scaling: Option<Vec<f64>>,
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
        saved_matching_scaling,
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
        let permuted_col = inverse[col];
        let start = matrix.col_ptrs()[col];
        let end = matrix.col_ptrs()[col + 1];
        for &row in &matrix.row_indices()[start..end] {
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
        let permuted_col = inverse[col];
        let start = matrix.col_ptrs()[col];
        let end = matrix.col_ptrs()[col + 1];
        for source_index in start..end {
            let row = matrix.row_indices()[source_index];
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

fn fill_scaled_permuted_lower_csc_values(
    matrix: SymmetricCscMatrix<'_>,
    col_ptrs: &[usize],
    row_indices: &[usize],
    source_positions: &[usize],
    scaling: &[f64],
    values: &mut Vec<f64>,
) -> Result<(), SsidsError> {
    let source_values = matrix.values().ok_or(SsidsError::MissingValues)?;
    if col_ptrs.len() != scaling.len() + 1 {
        return Err(SsidsError::DimensionMismatch {
            expected: col_ptrs.len().saturating_sub(1),
            actual: scaling.len(),
        });
    }
    if row_indices.len() != source_positions.len() {
        return Err(SsidsError::PatternMismatch(
            "permuted matrix row/source length mismatch".into(),
        ));
    }
    values.clear();
    values.resize(source_positions.len(), 0.0);
    for (col, window) in col_ptrs.windows(2).enumerate() {
        let col_scale = scaling[col];
        if !col_scale.is_finite() {
            return Err(SsidsError::InvalidMatrix(format!(
                "scaling value for permuted column {col} is not finite"
            )));
        }
        for entry in window[0]..window[1] {
            let source_index = source_positions[entry];
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
            let row = row_indices[entry];
            let row_scale = scaling[row];
            if !row_scale.is_finite() {
                return Err(SsidsError::InvalidMatrix(format!(
                    "scaling value for permuted row {row} is not finite"
                )));
            }
            values[entry] = row_scale * value * col_scale;
        }
    }
    Ok(())
}

#[cfg(test)]
fn apply_permuted_symmetric_scaling(
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &mut [f64],
    scaling: &[f64],
) -> Result<(), SsidsError> {
    if col_ptrs.len() != scaling.len() + 1 {
        return Err(SsidsError::DimensionMismatch {
            expected: col_ptrs.len().saturating_sub(1),
            actual: scaling.len(),
        });
    }
    if row_indices.len() != values.len() {
        return Err(SsidsError::PatternMismatch(
            "permuted matrix row/value length mismatch".into(),
        ));
    }
    for (col, window) in col_ptrs.windows(2).enumerate() {
        let col_scale = scaling[col];
        if !col_scale.is_finite() {
            return Err(SsidsError::InvalidMatrix(format!(
                "scaling value for permuted column {col} is not finite"
            )));
        }
        for entry in window[0]..window[1] {
            let row = row_indices[entry];
            let row_scale = scaling[row];
            if !row_scale.is_finite() {
                return Err(SsidsError::InvalidMatrix(format!(
                    "scaling value for permuted row {row} is not finite"
                )));
            }
            values[entry] = row_scale * values[entry] * col_scale;
        }
    }
    Ok(())
}

fn dense_symmetric_swap(matrix: &mut [f64], size: usize, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    let (lhs, rhs) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };

    for col in 0..lhs {
        let column_offset = col * size;
        let lhs_offset = column_offset + lhs;
        let rhs_offset = column_offset + rhs;
        matrix.swap(lhs_offset, rhs_offset);
    }

    for index in (lhs + 1)..rhs {
        let lhs_offset = lhs * size + index;
        let rhs_offset = index * size + rhs;
        matrix.swap(lhs_offset, rhs_offset);
    }

    for row in (rhs + 1)..size {
        let lhs_offset = lhs * size + row;
        let rhs_offset = rhs * size + row;
        matrix.swap(lhs_offset, rhs_offset);
    }

    let lhs_diag = lhs * size + lhs;
    let rhs_diag = rhs * size + rhs;
    matrix.swap(lhs_diag, rhs_diag);
}

#[cfg(test)]
fn dense_symmetric_swap_with_workspace(
    matrix: &mut [f64],
    size: usize,
    lhs: usize,
    rhs: usize,
    workspace: &mut [f64],
) {
    dense_symmetric_swap_with_workspace_row_offset(matrix, size, lhs, rhs, workspace, 0);
}

fn dense_symmetric_swap_with_workspace_row_offset(
    matrix: &mut [f64],
    size: usize,
    lhs: usize,
    rhs: usize,
    workspace: &mut [f64],
    workspace_row_offset: usize,
) {
    if lhs == rhs {
        return;
    }
    let (lhs, rhs) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
    debug_assert!(lhs >= workspace_row_offset);
    for work_row in workspace_row_offset..lhs {
        let work_start = (work_row - workspace_row_offset) * size;
        workspace.swap(work_start + lhs, work_start + rhs);
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

fn pack_dense_lower_suffix(
    matrix: &[f64],
    size: usize,
    start: usize,
    suffix_size: usize,
) -> Vec<f64> {
    let packed_len = packed_lower_len(suffix_size);
    let mut packed = Vec::<f64>::with_capacity(packed_len);
    let packed_ptr = packed.as_mut_ptr();
    for local_col in 0..suffix_size {
        let source_col = start + local_col;
        let len = suffix_size - local_col;
        let source_start = dense_lower_offset(size, source_col, source_col);
        let packed_offset = packed_lower_offset(suffix_size, local_col, local_col);
        // SAFETY: the packed lower-triangle column ranges are disjoint and
        // cover exactly `packed_len` entries. Each source range is a contiguous
        // lower-column suffix inside the dense front.
        unsafe {
            std::ptr::copy_nonoverlapping(
                matrix.as_ptr().add(source_start),
                packed_ptr.add(packed_offset),
                len,
            );
        }
    }
    // SAFETY: every packed entry is initialized exactly once by the column
    // copies above. For an empty suffix, both capacity and length are zero.
    unsafe {
        packed.set_len(packed_len);
    }
    packed
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
    // The local SPRAL block_ldlt<32> APP kernel contracts test_2x2's
    // determinant expression inside the optimized full-block path.
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

#[cfg(test)]
fn reset_ldwork_column_tail(workspace: &mut [f64], size: usize, col: usize, from: usize) {
    reset_ldwork_column_tail_with_row_offset(workspace, size, col, from, 0);
}

fn reset_ldwork_column_tail_with_row_offset(
    workspace: &mut [f64],
    size: usize,
    col: usize,
    from: usize,
    workspace_row_offset: usize,
) {
    debug_assert!(col >= workspace_row_offset);
    let column_start = (col - workspace_row_offset) * size;
    let column = &mut workspace[column_start..column_start + size];
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

    // Native SPRAL's non-AVX SimdVec path still uses two per-lane maxima. Equal
    // values keep their existing lane, so ties are not column-major.
    for local_col in local_from..APP_INNER_BLOCK_SIZE {
        let col = block_start + local_col;
        let column_offset = col * size;
        let diag_value = matrix[column_offset + col].abs();
        if diag_value > primary.0 {
            primary = (diag_value, col, col);
        }
        if local_col + 1 < 2 * (local_col / 2 + 1) {
            let row = col + 1;
            let value = matrix[column_offset + row].abs();
            if value > primary.0 {
                primary = (value, row, col);
            }
        }
        let mut local_row = 2 * (local_col / 2 + 1);
        while local_row < APP_INNER_BLOCK_SIZE {
            let row = block_start + local_row;
            let value = matrix[column_offset + row].abs();
            if value > primary.0 {
                primary = (value, row, col);
            }
            let next_row = row + 1;
            let next_value = matrix[column_offset + next_row].abs();
            if next_value > secondary.0 {
                secondary = (next_value, next_row, col);
            }
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

#[cfg(test)]
fn app_update_one_by_one(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
) {
    app_update_one_by_one_with_row_offset(matrix, size, pivot, update_end, workspace, 0);
}

fn app_update_one_by_one_with_row_offset(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
    workspace_row_offset: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the NEON helper only performs two-lane row loads/stores when
        // `row + 1 < update_end`; scalar tails handle the remaining row.
        unsafe {
            app_update_one_by_one_neon(
                matrix,
                size,
                pivot,
                update_end,
                workspace,
                workspace_row_offset,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        app_update_one_by_one_scalar(
            matrix,
            size,
            pivot,
            update_end,
            workspace,
            workspace_row_offset,
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn app_update_one_by_one_scalar(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
    workspace_row_offset: usize,
) {
    debug_assert!(pivot >= workspace_row_offset);
    let workspace_pivot = pivot - workspace_row_offset;
    let ld = &workspace[workspace_pivot * size..(workspace_pivot + 1) * size];
    for (col, &preserved) in ld.iter().enumerate().take(update_end).skip(pivot + 1) {
        for row in col..update_end {
            app_update_one_by_one_scalar_entry(matrix, size, pivot, col, row, preserved);
        }
    }
}

fn app_update_one_by_one_scalar_entry(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    col: usize,
    row: usize,
    preserved: f64,
) {
    let update_entry = dense_lower_offset(size, row, col);
    let multiplier = matrix[dense_lower_offset(size, row, pivot)];
    // Clang contracts SPRAL's scalar SimdVec update on the local build.
    matrix[update_entry] = (-preserved).mul_add(multiplier, matrix[update_entry]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn app_update_one_by_one_neon(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
    workspace_row_offset: usize,
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

    const NEON_F64_LANES: usize = 2;
    const SOURCE_UNROLL: usize = 4;

    let matrix_ptr = matrix.as_mut_ptr();
    debug_assert!(pivot >= workspace_row_offset);
    let workspace_pivot = pivot - workspace_row_offset;
    let ld = &workspace[workspace_pivot * size..(workspace_pivot + 1) * size];
    let block_start = update_end.saturating_sub(APP_INNER_BLOCK_SIZE);
    let local_pivot = pivot - block_start;
    let source_unroll_start = block_start + SOURCE_UNROLL * (local_pivot / SOURCE_UNROLL + 1);

    for (col, &preserved) in ld
        .iter()
        .enumerate()
        .take(source_unroll_start.min(update_end))
        .skip(pivot + 1)
    {
        let mut row = col;
        let neg_preserved = vdupq_n_f64(-preserved);
        while row + 1 < update_end {
            // SAFETY: `row + 1 < update_end <= size`; dense columns are
            // contiguous by row.
            let multiplier = unsafe { vld1q_f64(matrix_ptr.add(pivot * size + row)) };
            let current = unsafe { vld1q_f64(matrix_ptr.add(col * size + row)) };
            let updated = vfmaq_f64(current, multiplier, neg_preserved);
            // SAFETY: same bounds as the load above.
            unsafe {
                vst1q_f64(matrix_ptr.add(col * size + row), updated);
            }
            row += 2;
        }
        while row < update_end {
            app_update_one_by_one_scalar_entry(matrix, size, pivot, col, row, preserved);
            row += 1;
        }
    }

    let mut col = source_unroll_start;
    while col + SOURCE_UNROLL <= update_end {
        let neg0 = vdupq_n_f64(-ld[col]);
        let neg1 = vdupq_n_f64(-ld[col + 1]);
        let neg2 = vdupq_n_f64(-ld[col + 2]);
        let neg3 = vdupq_n_f64(-ld[col + 3]);
        let local_col = col - block_start;
        let mut row = block_start + NEON_F64_LANES * (local_col / NEON_F64_LANES);
        while row + 1 < update_end {
            // Mirrors block_ldlt.hxx::update_1x1: the same pivot-column
            // vector feeds four target columns. Lanes above a target
            // column's diagonal land in unused upper storage.
            let multiplier = unsafe { vld1q_f64(matrix_ptr.add(pivot * size + row)) };
            let current0 = unsafe { vld1q_f64(matrix_ptr.add(col * size + row)) };
            let current1 = unsafe { vld1q_f64(matrix_ptr.add((col + 1) * size + row)) };
            let current2 = unsafe { vld1q_f64(matrix_ptr.add((col + 2) * size + row)) };
            let current3 = unsafe { vld1q_f64(matrix_ptr.add((col + 3) * size + row)) };
            let updated0 = vfmaq_f64(current0, multiplier, neg0);
            let updated1 = vfmaq_f64(current1, multiplier, neg1);
            let updated2 = vfmaq_f64(current2, multiplier, neg2);
            let updated3 = vfmaq_f64(current3, multiplier, neg3);
            unsafe {
                vst1q_f64(matrix_ptr.add(col * size + row), updated0);
                vst1q_f64(matrix_ptr.add((col + 1) * size + row), updated1);
                vst1q_f64(matrix_ptr.add((col + 2) * size + row), updated2);
                vst1q_f64(matrix_ptr.add((col + 3) * size + row), updated3);
            }
            row += NEON_F64_LANES;
        }
        for (target_col, &preserved) in ld.iter().enumerate().skip(col).take(SOURCE_UNROLL) {
            let mut scalar_row = row.max(target_col);
            while scalar_row < update_end {
                app_update_one_by_one_scalar_entry(
                    matrix, size, pivot, target_col, scalar_row, preserved,
                );
                scalar_row += 1;
            }
        }
        col += SOURCE_UNROLL;
    }

    for (col, &preserved) in ld.iter().enumerate().take(update_end).skip(col) {
        let mut row = col;
        let neg_preserved = vdupq_n_f64(-preserved);
        while row + 1 < update_end {
            let multiplier = unsafe { vld1q_f64(matrix_ptr.add(pivot * size + row)) };
            let current = unsafe { vld1q_f64(matrix_ptr.add(col * size + row)) };
            let updated = vfmaq_f64(current, multiplier, neg_preserved);
            unsafe {
                vst1q_f64(matrix_ptr.add(col * size + row), updated);
            }
            row += 2;
        }
        while row < update_end {
            app_update_one_by_one_scalar_entry(matrix, size, pivot, col, row, preserved);
            row += 1;
        }
    }
}

#[cfg(test)]
fn app_update_two_by_two(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
) {
    app_update_two_by_two_with_row_offset(matrix, size, pivot, update_end, workspace, 0);
}

fn app_update_two_by_two_with_row_offset(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
    workspace_row_offset: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the NEON helper only performs two-lane row loads/stores when
        // `row + 1 < update_end`; scalar tails handle the remaining row.
        unsafe {
            app_update_two_by_two_neon(
                matrix,
                size,
                pivot,
                update_end,
                workspace,
                workspace_row_offset,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        app_update_two_by_two_scalar(
            matrix,
            size,
            pivot,
            update_end,
            workspace,
            workspace_row_offset,
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn app_update_two_by_two_scalar(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
    workspace_row_offset: usize,
) {
    debug_assert!(pivot >= workspace_row_offset);
    let workspace_pivot = pivot - workspace_row_offset;
    let first_ld = &workspace[workspace_pivot * size..(workspace_pivot + 1) * size];
    let second_ld = &workspace[(workspace_pivot + 1) * size..(workspace_pivot + 2) * size];
    for col in (pivot + 2)..update_end {
        let first_preserved = first_ld[col];
        let second_preserved = second_ld[col];
        for row in col..update_end {
            app_update_two_by_two_scalar_entry(
                matrix,
                size,
                pivot,
                col,
                row,
                first_preserved,
                second_preserved,
            );
        }
    }
}

fn app_update_two_by_two_scalar_entry(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    col: usize,
    row: usize,
    first_preserved: f64,
    second_preserved: f64,
) {
    let update_entry = dense_lower_offset(size, row, col);
    let first_multiplier = matrix[dense_lower_offset(size, row, pivot)];
    let second_multiplier = matrix[dense_lower_offset(size, row, pivot + 1)];
    // block_ldlt.hxx::update_2x2 forms the two-product update under
    // `#pragma omp simd`; the local optimized native path contracts the
    // first product into the second product before subtracting it.
    let combined = first_preserved.mul_add(first_multiplier, second_preserved * second_multiplier);
    matrix[update_entry] -= combined;
}

struct AppWorkspaceMut<'a> {
    values: &'a mut [f64],
    row_offset: usize,
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn app_update_two_by_two_neon(
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    workspace: &[f64],
    workspace_row_offset: usize,
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};

    let matrix_ptr = matrix.as_mut_ptr();
    debug_assert!(pivot >= workspace_row_offset);
    let workspace_pivot = pivot - workspace_row_offset;
    let first_ld = &workspace[workspace_pivot * size..(workspace_pivot + 1) * size];
    let second_ld = &workspace[(workspace_pivot + 1) * size..(workspace_pivot + 2) * size];
    for col in (pivot + 2)..update_end {
        let first_preserved = first_ld[col];
        let second_preserved = second_ld[col];
        let first_preserved_vec = vdupq_n_f64(first_preserved);
        let second_preserved_vec = vdupq_n_f64(second_preserved);
        let mut row = col;
        while row + 1 < update_end {
            // SAFETY: `row + 1 < update_end <= size`; dense columns are
            // contiguous by row.
            let first_multiplier = unsafe { vld1q_f64(matrix_ptr.add(pivot * size + row)) };
            let second_multiplier = unsafe { vld1q_f64(matrix_ptr.add((pivot + 1) * size + row)) };
            let second_product = vmulq_f64(second_multiplier, second_preserved_vec);
            let combined = vfmaq_f64(second_product, first_multiplier, first_preserved_vec);
            let current = unsafe { vld1q_f64(matrix_ptr.add(col * size + row)) };
            let updated = vsubq_f64(current, combined);
            // SAFETY: same bounds as the load above.
            unsafe {
                vst1q_f64(matrix_ptr.add(col * size + row), updated);
            }
            row += 2;
        }
        while row < update_end {
            app_update_two_by_two_scalar_entry(
                matrix,
                size,
                pivot,
                col,
                row,
                first_preserved,
                second_preserved,
            );
            row += 1;
        }
    }
}

#[cfg(test)]
fn factor_one_by_one_common(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    stats: &mut PanelFactorStats,
    scratch: &mut [f64],
) -> Result<FactorBlockRecord, SsidsError> {
    factor_one_by_one_common_with_workspace_offset(
        rows,
        matrix,
        size,
        pivot,
        update_end,
        stats,
        AppWorkspaceMut {
            values: scratch,
            row_offset: 0,
        },
    )
}

fn factor_one_by_one_common_with_workspace_offset(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    update_end: usize,
    stats: &mut PanelFactorStats,
    workspace: AppWorkspaceMut<'_>,
) -> Result<FactorBlockRecord, SsidsError> {
    let workspace_row_offset = workspace.row_offset;
    debug_assert!(pivot >= workspace_row_offset);
    let workspace_pivot = pivot - workspace_row_offset;
    let work = &mut workspace.values[workspace_pivot * size..(workspace_pivot + 1) * size];
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

    app_update_one_by_one_with_row_offset(
        matrix,
        size,
        pivot,
        update_end,
        workspace.values,
        workspace_row_offset,
    );

    Ok(FactorBlockRecord {
        size: 1,
        values: [inverse_diagonal, 0.0, 0.0, 0.0],
    })
}

#[cfg(test)]
fn factor_two_by_two_common(
    rows: &[usize],
    matrix: &mut [f64],
    bounds: DenseUpdateBounds,
    pivot: usize,
    inverse: (f64, f64, f64),
    stats: &mut PanelFactorStats,
    scratch: &mut [f64],
) -> Result<FactorBlockRecord, SsidsError> {
    factor_two_by_two_common_with_workspace_offset(
        rows,
        matrix,
        bounds,
        pivot,
        inverse,
        stats,
        AppWorkspaceMut {
            values: scratch,
            row_offset: 0,
        },
    )
}

fn factor_two_by_two_common_with_workspace_offset(
    rows: &[usize],
    matrix: &mut [f64],
    bounds: DenseUpdateBounds,
    pivot: usize,
    inverse: (f64, f64, f64),
    stats: &mut PanelFactorStats,
    workspace: AppWorkspaceMut<'_>,
) -> Result<FactorBlockRecord, SsidsError> {
    let size = bounds.size;
    let update_end = bounds.update_end;
    let workspace_row_offset = workspace.row_offset;
    debug_assert!(pivot >= workspace_row_offset);
    let workspace_pivot = pivot - workspace_row_offset;
    let first_start = workspace_pivot * size;
    let second_start = (workspace_pivot + 1) * size;
    let (first_prefix, second_suffix) = workspace.values.split_at_mut(second_start);
    let first_scratch = &mut first_prefix[first_start..second_start];
    let second_scratch = &mut second_suffix[..size];
    let (inv11, inv12, inv22) = inverse;
    stats.two_by_two_pivots += 1;
    matrix[dense_lower_offset(size, pivot, pivot)] = 1.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot)] = 0.0;
    matrix[dense_lower_offset(size, pivot + 1, pivot + 1)] = 1.0;

    let first_multiplier_row = pivot + 2;
    let trailing_rows = update_end.saturating_sub(first_multiplier_row);
    // Mirrors SPRAL ssids/cpu/kernels/block_ldlt.hxx::block_ldlt on the local
    // optimized native build: the vectorized body contracts the second source
    // product into the first rounded product, while the scalar tail contracts
    // the first source product into the second rounded product.
    let vectorized_multiplier_rows = if trailing_rows >= 4 {
        trailing_rows / 2 * 2
    } else if trailing_rows == 3 {
        2
    } else {
        0
    };
    for row in first_multiplier_row..update_end {
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
                detail: "two-by-two multipliers became non-finite".into(),
            });
        }
        matrix[dense_lower_offset(size, row, pivot)] = l1;
        matrix[dense_lower_offset(size, row, pivot + 1)] = l2;
    }

    app_update_two_by_two_with_row_offset(
        matrix,
        size,
        pivot,
        update_end,
        workspace.values,
        workspace_row_offset,
    );

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
    app_solve_block_triangular_to_trailing_rows_impl(matrix, size, block_start, block_end);
    triangular_started.map_or(Duration::default(), |started| started.elapsed())
}

#[cfg(target_arch = "aarch64")]
fn app_solve_block_triangular_to_trailing_rows_impl(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
) {
    // SAFETY: the helper bounds every two-lane load/store by `row + 1 < size`;
    // scalar tails handle all remaining shapes.
    unsafe {
        app_solve_block_triangular_to_trailing_rows_neon(matrix, size, block_start, block_end);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn app_solve_block_triangular_to_trailing_rows_impl(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
) {
    app_solve_block_triangular_to_trailing_rows_scalar(matrix, size, block_start, block_end);
}

#[cfg(not(target_arch = "aarch64"))]
fn app_solve_block_triangular_to_trailing_rows_scalar(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
) {
    const OPENBLAS_DTRSM_UNROLL_N: usize = 4;
    for row in block_end..size {
        let mut group_start = block_start;
        while group_start < block_end {
            let group_end = (group_start + OPENBLAS_DTRSM_UNROLL_N).min(block_end);
            app_solve_block_triangular_row_group_scalar(
                matrix,
                size,
                block_start,
                group_start,
                group_end,
                row,
            );
            group_start = group_end;
        }
    }
}

fn app_solve_block_triangular_row_group_scalar(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    group_start: usize,
    group_end: usize,
    row: usize,
) {
    if group_end - group_start == 4 {
        let col0 = group_start;
        let col1 = group_start + 1;
        let col2 = group_start + 2;
        let col3 = group_start + 3;

        let mut value0 = matrix[col0 * size + row];
        let mut value1 = matrix[col1 * size + row];
        let mut value2 = matrix[col2 * size + row];
        let mut value3 = matrix[col3 * size + row];
        if group_start > block_start {
            let mut dot0 = 0.0;
            let mut dot1 = 0.0;
            let mut dot2 = 0.0;
            let mut dot3 = 0.0;
            for prior in block_start..group_start {
                let prior_value = matrix[prior * size + row];
                dot0 = prior_value.mul_add(matrix[prior * size + col0], dot0);
                dot1 = prior_value.mul_add(matrix[prior * size + col1], dot1);
                dot2 = prior_value.mul_add(matrix[prior * size + col2], dot2);
                dot3 = prior_value.mul_add(matrix[prior * size + col3], dot3);
            }
            value0 = dot0.mul_add(-1.0, value0);
            value1 = dot1.mul_add(-1.0, value1);
            value2 = dot2.mul_add(-1.0, value2);
            value3 = dot3.mul_add(-1.0, value3);
        }

        value1 = (-value0).mul_add(matrix[col0 * size + col1], value1);
        value2 = (-value0).mul_add(matrix[col0 * size + col2], value2);
        value3 = (-value0).mul_add(matrix[col0 * size + col3], value3);
        value2 = (-value1).mul_add(matrix[col1 * size + col2], value2);
        value3 = (-value1).mul_add(matrix[col1 * size + col3], value3);
        value3 = (-value2).mul_add(matrix[col2 * size + col3], value3);

        matrix[col0 * size + row] = value0;
        matrix[col1 * size + row] = value1;
        matrix[col2 * size + row] = value2;
        matrix[col3 * size + row] = value3;
        return;
    }

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
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn app_solve_block_triangular_to_trailing_rows_neon(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    block_end: usize,
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

    const OPENBLAS_DTRSM_UNROLL_N: usize = 4;
    let matrix_ptr = matrix.as_mut_ptr();
    let mut row = block_end;
    while row + 1 < size {
        let mut group_start = block_start;
        while group_start < block_end {
            let group_end = (group_start + OPENBLAS_DTRSM_UNROLL_N).min(block_end);
            if group_end - group_start != 4 {
                app_solve_block_triangular_row_group_scalar(
                    matrix,
                    size,
                    block_start,
                    group_start,
                    group_end,
                    row,
                );
                app_solve_block_triangular_row_group_scalar(
                    matrix,
                    size,
                    block_start,
                    group_start,
                    group_end,
                    row + 1,
                );
                group_start = group_end;
                continue;
            }

            let col0 = group_start;
            let col1 = group_start + 1;
            let col2 = group_start + 2;
            let col3 = group_start + 3;

            // SAFETY: `row + 1 < size`; each column is stored contiguously by
            // row in the dense lower-column buffer.
            let mut value0 = unsafe { vld1q_f64(matrix_ptr.add(col0 * size + row)) };
            let mut value1 = unsafe { vld1q_f64(matrix_ptr.add(col1 * size + row)) };
            let mut value2 = unsafe { vld1q_f64(matrix_ptr.add(col2 * size + row)) };
            let mut value3 = unsafe { vld1q_f64(matrix_ptr.add(col3 * size + row)) };

            if group_start > block_start {
                let mut dot0 = vdupq_n_f64(0.0);
                let mut dot1 = vdupq_n_f64(0.0);
                let mut dot2 = vdupq_n_f64(0.0);
                let mut dot3 = vdupq_n_f64(0.0);
                for prior in block_start..group_start {
                    // SAFETY: same `row + 1 < size` bound as above.
                    let prior_value = unsafe { vld1q_f64(matrix_ptr.add(prior * size + row)) };
                    dot0 = vfmaq_f64(
                        dot0,
                        prior_value,
                        vdupq_n_f64(unsafe { *matrix_ptr.add(prior * size + col0) }),
                    );
                    dot1 = vfmaq_f64(
                        dot1,
                        prior_value,
                        vdupq_n_f64(unsafe { *matrix_ptr.add(prior * size + col1) }),
                    );
                    dot2 = vfmaq_f64(
                        dot2,
                        prior_value,
                        vdupq_n_f64(unsafe { *matrix_ptr.add(prior * size + col2) }),
                    );
                    dot3 = vfmaq_f64(
                        dot3,
                        prior_value,
                        vdupq_n_f64(unsafe { *matrix_ptr.add(prior * size + col3) }),
                    );
                }
                let minus_one = vdupq_n_f64(-1.0);
                value0 = vfmaq_f64(value0, dot0, minus_one);
                value1 = vfmaq_f64(value1, dot1, minus_one);
                value2 = vfmaq_f64(value2, dot2, minus_one);
                value3 = vfmaq_f64(value3, dot3, minus_one);
            }

            value1 = vfmaq_f64(
                value1,
                value0,
                vdupq_n_f64(-unsafe { *matrix_ptr.add(col0 * size + col1) }),
            );
            value2 = vfmaq_f64(
                value2,
                value0,
                vdupq_n_f64(-unsafe { *matrix_ptr.add(col0 * size + col2) }),
            );
            value3 = vfmaq_f64(
                value3,
                value0,
                vdupq_n_f64(-unsafe { *matrix_ptr.add(col0 * size + col3) }),
            );
            value2 = vfmaq_f64(
                value2,
                value1,
                vdupq_n_f64(-unsafe { *matrix_ptr.add(col1 * size + col2) }),
            );
            value3 = vfmaq_f64(
                value3,
                value1,
                vdupq_n_f64(-unsafe { *matrix_ptr.add(col1 * size + col3) }),
            );
            value3 = vfmaq_f64(
                value3,
                value2,
                vdupq_n_f64(-unsafe { *matrix_ptr.add(col2 * size + col3) }),
            );

            // SAFETY: same `row + 1 < size` bound as the loads.
            unsafe {
                vst1q_f64(matrix_ptr.add(col0 * size + row), value0);
                vst1q_f64(matrix_ptr.add(col1 * size + row), value1);
                vst1q_f64(matrix_ptr.add(col2 * size + row), value2);
                vst1q_f64(matrix_ptr.add(col3 * size + row), value3);
            }
            group_start = group_end;
        }
        row += 2;
    }

    while row < size {
        let mut group_start = block_start;
        while group_start < block_end {
            let group_end = (group_start + OPENBLAS_DTRSM_UNROLL_N).min(block_end);
            app_solve_block_triangular_row_group_scalar(
                matrix,
                size,
                block_start,
                group_start,
                group_end,
                row,
            );
            group_start = group_end;
        }
        row += 1;
    }
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

fn app_record_count_for_prefix(block_records: &[FactorBlockRecord], local_prefix: usize) -> usize {
    let mut count = 0;
    let mut cursor = 0;
    for block in block_records {
        if cursor + block.size > local_prefix {
            break;
        }
        count += 1;
        cursor += block.size;
    }
    debug_assert_eq!(cursor, local_prefix);
    count
}

#[cfg(test)]
fn app_truncate_records_to_prefix(
    block_records: &[FactorBlockRecord],
    local_prefix: usize,
) -> Vec<FactorBlockRecord> {
    block_records[..app_record_count_for_prefix(block_records, local_prefix)].to_vec()
}

#[cfg(test)]
fn app_backup_trailing_lower(matrix: &[f64], size: usize, backup_start: usize) -> Vec<f64> {
    let mut backup = Vec::new();
    app_backup_trailing_lower_into(matrix, size, backup_start, &mut backup);
    backup
}

fn app_backup_trailing_lower_into(
    matrix: &[f64],
    size: usize,
    backup_start: usize,
    backup: &mut Vec<f64>,
) {
    let backup_size = size - backup_start;
    let backup_len = packed_lower_len(backup_size);
    backup.clear();
    if backup.capacity() < backup_len {
        backup.reserve(backup_len);
    }
    let backup_ptr = backup.as_mut_ptr();
    for local_col in 0..backup_size {
        let source_col = backup_start + local_col;
        let len = backup_size - local_col;
        let source_start = dense_lower_offset(size, source_col, source_col);
        let backup_offset = packed_lower_offset(backup_size, local_col, local_col);
        // SAFETY: the packed lower-triangle column ranges are disjoint and
        // cover exactly `backup_len` entries. Each source range is a contiguous
        // lower-column suffix inside the dense front.
        unsafe {
            std::ptr::copy_nonoverlapping(
                matrix.as_ptr().add(source_start),
                backup_ptr.add(backup_offset),
                len,
            );
        }
    }
    // SAFETY: every packed entry is initialized exactly once by the column
    // copies above. For an empty suffix, both capacity and length are zero.
    unsafe {
        backup.set_len(backup_len);
    }
}

#[derive(Clone, Copy)]
struct AppRestoreRange {
    backup_start: usize,
    block_end: usize,
    trailing_start: usize,
}

fn app_restore_trailing_from_block_backup(
    rows: &[usize],
    rows_before_block: &[usize],
    matrix: &mut [f64],
    matrix_before_block: &[f64],
    size: usize,
    range: AppRestoreRange,
) {
    let AppRestoreRange {
        backup_start,
        block_end,
        trailing_start,
    } = range;
    if trailing_start >= size || trailing_start >= block_end {
        return;
    }
    let backup_size = size - backup_start;
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
            debug_assert!(old_row >= backup_start);
            debug_assert!(old_col >= backup_start);
            let local_old_row = old_row - backup_start;
            let local_old_col = old_col - backup_start;
            let (backup_row, backup_col) = if local_old_row >= local_old_col {
                (local_old_row, local_old_col)
            } else {
                (local_old_col, local_old_row)
            };
            matrix[dense_lower_offset(size, row, col)] =
                matrix_before_block[packed_lower_offset(backup_size, backup_row, backup_col)];
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

#[cfg(test)]
fn app_build_ld_workspace(
    matrix: &[f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
) -> Vec<f64> {
    let accepted_width = accepted_end - block_start;
    let mut ld_values = vec![0.0; accepted_width * size];
    app_build_ld_workspace_into(
        matrix,
        size,
        block_start,
        accepted_end,
        block_records,
        &mut ld_values,
    );
    ld_values
}

fn app_build_ld_workspace_into(
    matrix: &[f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
    ld_values: &mut [f64],
) {
    // Accepted-update operands are below the eliminated prefix, so SPRAL's
    // column-major `aval[col * lda + row]` matches the dense lower storage.
    let accepted_width = accepted_end - block_start;
    debug_assert!(ld_values.len() >= accepted_width * size);
    let ld_values = &mut ld_values[..accepted_width * size];
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
            // Mirrors ldlt_app.cxx::Block::update: calcLD<OP_N> is called
            // separately for each target block, so the vector/scalar split
            // resets at every APP_INNER_BLOCK_SIZE row tile.
            let mut tile_start = accepted_end;
            while tile_start < size {
                let tile_end = (tile_start + APP_INNER_BLOCK_SIZE).min(size);
                let tile_rows = tile_end - tile_start;
                let vector_rows = if tile_rows > 4 { tile_rows & !1 } else { 0 };
                for row in tile_start..tile_end {
                    let local_row = row - tile_start;
                    let row_l1 = matrix[pivot * size + row];
                    let row_l2 = matrix[(pivot + 1) * size + row];
                    ld_values[relative_pivot * size + row] = if local_row < vector_rows {
                        (-d21).mul_add(row_l2, d22 * row_l1)
                    } else {
                        d22.mul_add(row_l1, -(d21 * row_l2))
                    };
                    ld_values[(relative_pivot + 1) * size + row] =
                        (-d21).mul_add(row_l1, d11 * row_l2);
                }
                tile_start = tile_end;
            }
            pivot += 2;
        }
    }
    debug_assert_eq!(pivot, accepted_end);
}

#[cfg(test)]
fn app_build_ld_tile_workspace(
    matrix: &[f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
    row_start: usize,
    row_end: usize,
) -> Vec<f64> {
    let accepted_width = accepted_end - block_start;
    let tile_rows = row_end - row_start;
    let mut ld_values = vec![0.0; accepted_width * tile_rows.max(1)];
    let mut pivot = block_start;
    for block in block_records {
        let relative_pivot = pivot - block_start;
        if block.size == 1 {
            let diagonal = app_original_one_by_one_diagonal(block.values[0]);
            for row in row_start..row_end {
                let row_l = matrix[pivot * size + row];
                ld_values[relative_pivot * tile_rows + (row - row_start)] = diagonal * row_l;
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
            let vector_rows = if tile_rows > 4 { tile_rows & !1 } else { 0 };
            for row in row_start..row_end {
                let local_row = row - row_start;
                let row_l1 = matrix[pivot * size + row];
                let row_l2 = matrix[(pivot + 1) * size + row];
                // Mirrors SPRAL ssids/cpu/kernels/calc_ld.hxx
                // calcLD<OP_N> as called from ldlt_app.cxx::Block::update.
                ld_values[relative_pivot * tile_rows + local_row] = if local_row < vector_rows {
                    (-d21).mul_add(row_l2, d22 * row_l1)
                } else {
                    d22.mul_add(row_l1, -(d21 * row_l2))
                };
                ld_values[(relative_pivot + 1) * tile_rows + local_row] =
                    (-d21).mul_add(row_l1, d11 * row_l2);
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

#[cfg(test)]
fn app_apply_accepted_prefix_update(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
) {
    let accepted_width = accepted_end - block_start;
    let mut ld_values = vec![0.0; accepted_width * size];
    app_apply_accepted_prefix_update_with_workspace(
        matrix,
        size,
        block_start,
        accepted_end,
        block_records,
        &mut ld_values,
        false,
    );
}

fn app_apply_accepted_prefix_update_with_workspace(
    matrix: &mut [f64],
    size: usize,
    block_start: usize,
    accepted_end: usize,
    block_records: &[FactorBlockRecord],
    ld_values: &mut [f64],
    profile_enabled: bool,
) -> (Duration, Duration) {
    if accepted_end >= size {
        return (Duration::default(), Duration::default());
    }
    let ld_started = profile_enabled.then(Instant::now);
    app_build_ld_workspace_into(
        matrix,
        size,
        block_start,
        accepted_end,
        block_records,
        ld_values,
    );
    let ld_time = ld_started.map_or(Duration::default(), |started| started.elapsed());
    let update_started = profile_enabled.then(Instant::now);
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
        let update_time = update_started.map_or(Duration::default(), |started| started.elapsed());
        return (ld_time, update_time);
    }
    let accepted_width = accepted_end - block_start;
    debug_assert!(accepted_width <= APP_INNER_BLOCK_SIZE);
    let incremental_column = app_gemv_forward_singleton_column(size, accepted_end);
    let mut column_l_values = [0.0; APP_INNER_BLOCK_SIZE];
    let mut next_column_l_values = [0.0; APP_INNER_BLOCK_SIZE];
    let mut third_column_l_values = [0.0; APP_INNER_BLOCK_SIZE];
    let mut fourth_column_l_values = [0.0; APP_INNER_BLOCK_SIZE];

    let mut col = accepted_end;
    while col < size {
        if incremental_column == Some(col) {
            for row in col..size {
                app_apply_accepted_prefix_update_entry_incremental(
                    matrix,
                    AppAcceptedUpdateContext {
                        size,
                        block_start,
                        accepted_end,
                        block_records,
                        ld_values,
                    },
                    row,
                    col,
                );
            }
            col += 1;
            continue;
        }

        for relative_pivot in 0..accepted_width {
            column_l_values[relative_pivot] = matrix[(block_start + relative_pivot) * size + col];
        }
        #[cfg(target_arch = "aarch64")]
        if col + 3 < size
            && incremental_column != Some(col + 1)
            && incremental_column != Some(col + 2)
            && incremental_column != Some(col + 3)
        {
            for relative_pivot in 0..accepted_width {
                next_column_l_values[relative_pivot] =
                    matrix[(block_start + relative_pivot) * size + col + 1];
                third_column_l_values[relative_pivot] =
                    matrix[(block_start + relative_pivot) * size + col + 2];
                fourth_column_l_values[relative_pivot] =
                    matrix[(block_start + relative_pivot) * size + col + 3];
            }
            // SAFETY: the helper applies the lower-triangle diagonal boundary
            // with scalar entry updates before using vector stores in rows
            // valid for all four target columns.
            unsafe {
                app_apply_accepted_prefix_update_four_columns_neon(
                    matrix,
                    size,
                    col,
                    accepted_width,
                    ld_values,
                    AppAcceptedFourColumnValues {
                        first: &column_l_values,
                        second: &next_column_l_values,
                        third: &third_column_l_values,
                        fourth: &fourth_column_l_values,
                    },
                );
            }
            col += 4;
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        if col + 1 < size && incremental_column != Some(col + 1) {
            for relative_pivot in 0..accepted_width {
                next_column_l_values[relative_pivot] =
                    matrix[(block_start + relative_pivot) * size + col + 1];
            }
            // SAFETY: the helper handles the first column's diagonal entry
            // separately and only uses vector loads/stores for rows inside both
            // target columns.
            unsafe {
                app_apply_accepted_prefix_update_two_columns_neon(
                    matrix,
                    size,
                    col,
                    accepted_width,
                    ld_values,
                    &column_l_values,
                    &next_column_l_values,
                );
            }
            col += 2;
            continue;
        }
        app_apply_accepted_prefix_update_column(
            matrix,
            size,
            col,
            accepted_width,
            ld_values,
            &column_l_values,
        );
        col += 1;
    }
    let update_time = update_started.map_or(Duration::default(), |started| started.elapsed());
    (ld_time, update_time)
}

#[cfg(target_arch = "aarch64")]
struct AppAcceptedFourColumnValues<'a> {
    first: &'a [f64; APP_INNER_BLOCK_SIZE],
    second: &'a [f64; APP_INNER_BLOCK_SIZE],
    third: &'a [f64; APP_INNER_BLOCK_SIZE],
    fourth: &'a [f64; APP_INNER_BLOCK_SIZE],
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn app_apply_accepted_prefix_update_four_columns_neon(
    matrix: &mut [f64],
    size: usize,
    col: usize,
    accepted_width: usize,
    ld_values: &[f64],
    column_values: AppAcceptedFourColumnValues<'_>,
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

    let column_l_values = column_values.first;
    let next_column_l_values = column_values.second;
    let third_column_l_values = column_values.third;
    let fourth_column_l_values = column_values.fourth;

    let col1 = col + 1;
    let col2 = col + 2;
    let col3 = col + 3;

    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col,
        col,
        accepted_width,
        ld_values,
        column_l_values,
    );
    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col,
        col1,
        accepted_width,
        ld_values,
        column_l_values,
    );
    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col1,
        col1,
        accepted_width,
        ld_values,
        next_column_l_values,
    );
    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col,
        col2,
        accepted_width,
        ld_values,
        column_l_values,
    );
    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col1,
        col2,
        accepted_width,
        ld_values,
        next_column_l_values,
    );
    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col2,
        col2,
        accepted_width,
        ld_values,
        third_column_l_values,
    );

    let matrix_ptr = matrix.as_mut_ptr();
    let ld_ptr = ld_values.as_ptr();
    let col0_l_ptr = column_l_values.as_ptr();
    let col1_l_ptr = next_column_l_values.as_ptr();
    let col2_l_ptr = third_column_l_values.as_ptr();
    let col3_l_ptr = fourth_column_l_values.as_ptr();
    let mut row = col3;
    while row + 7 < size {
        let mut update00 = vdupq_n_f64(0.0);
        let mut update01 = vdupq_n_f64(0.0);
        let mut update02 = vdupq_n_f64(0.0);
        let mut update03 = vdupq_n_f64(0.0);
        let mut update10 = vdupq_n_f64(0.0);
        let mut update11 = vdupq_n_f64(0.0);
        let mut update12 = vdupq_n_f64(0.0);
        let mut update13 = vdupq_n_f64(0.0);
        let mut update20 = vdupq_n_f64(0.0);
        let mut update21 = vdupq_n_f64(0.0);
        let mut update22 = vdupq_n_f64(0.0);
        let mut update23 = vdupq_n_f64(0.0);
        let mut update30 = vdupq_n_f64(0.0);
        let mut update31 = vdupq_n_f64(0.0);
        let mut update32 = vdupq_n_f64(0.0);
        let mut update33 = vdupq_n_f64(0.0);

        macro_rules! accumulate_pivot {
            ($pivot:expr) => {{
                let pivot = $pivot;
                // SAFETY: caller passes `row + 7 < size`, `pivot < accepted_width`,
                // and the LD workspace is full-width.
                let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot * size + row)) };
                let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot * size + row + 2)) };
                let row_ld2 = unsafe { vld1q_f64(ld_ptr.add(pivot * size + row + 4)) };
                let row_ld3 = unsafe { vld1q_f64(ld_ptr.add(pivot * size + row + 6)) };
                let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot) });
                let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot) });
                let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot) });
                let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot) });

                update00 = vfmaq_f64(update00, row_ld0, col0_l);
                update01 = vfmaq_f64(update01, row_ld1, col0_l);
                update02 = vfmaq_f64(update02, row_ld2, col0_l);
                update03 = vfmaq_f64(update03, row_ld3, col0_l);
                update10 = vfmaq_f64(update10, row_ld0, col1_l);
                update11 = vfmaq_f64(update11, row_ld1, col1_l);
                update12 = vfmaq_f64(update12, row_ld2, col1_l);
                update13 = vfmaq_f64(update13, row_ld3, col1_l);
                update20 = vfmaq_f64(update20, row_ld0, col2_l);
                update21 = vfmaq_f64(update21, row_ld1, col2_l);
                update22 = vfmaq_f64(update22, row_ld2, col2_l);
                update23 = vfmaq_f64(update23, row_ld3, col2_l);
                update30 = vfmaq_f64(update30, row_ld0, col3_l);
                update31 = vfmaq_f64(update31, row_ld1, col3_l);
                update32 = vfmaq_f64(update32, row_ld2, col3_l);
                update33 = vfmaq_f64(update33, row_ld3, col3_l);
            }};
        }

        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            accumulate_pivot!(relative_pivot);
            accumulate_pivot!(relative_pivot + 1);
            accumulate_pivot!(relative_pivot + 2);
            accumulate_pivot!(relative_pivot + 3);
            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            accumulate_pivot!(relative_pivot);
            relative_pivot += 1;
        }

        let entry0 = col * size + row;
        let entry1 = col1 * size + row;
        let entry2 = col2 * size + row;
        let entry3 = col3 * size + row;
        let minus_one = vdupq_n_f64(-1.0);
        let current00 = unsafe { vld1q_f64(matrix_ptr.add(entry0)) };
        let current01 = unsafe { vld1q_f64(matrix_ptr.add(entry0 + 2)) };
        let current02 = unsafe { vld1q_f64(matrix_ptr.add(entry0 + 4)) };
        let current03 = unsafe { vld1q_f64(matrix_ptr.add(entry0 + 6)) };
        let current10 = unsafe { vld1q_f64(matrix_ptr.add(entry1)) };
        let current11 = unsafe { vld1q_f64(matrix_ptr.add(entry1 + 2)) };
        let current12 = unsafe { vld1q_f64(matrix_ptr.add(entry1 + 4)) };
        let current13 = unsafe { vld1q_f64(matrix_ptr.add(entry1 + 6)) };
        let current20 = unsafe { vld1q_f64(matrix_ptr.add(entry2)) };
        let current21 = unsafe { vld1q_f64(matrix_ptr.add(entry2 + 2)) };
        let current22 = unsafe { vld1q_f64(matrix_ptr.add(entry2 + 4)) };
        let current23 = unsafe { vld1q_f64(matrix_ptr.add(entry2 + 6)) };
        let current30 = unsafe { vld1q_f64(matrix_ptr.add(entry3)) };
        let current31 = unsafe { vld1q_f64(matrix_ptr.add(entry3 + 2)) };
        let current32 = unsafe { vld1q_f64(matrix_ptr.add(entry3 + 4)) };
        let current33 = unsafe { vld1q_f64(matrix_ptr.add(entry3 + 6)) };
        let updated00 = vfmaq_f64(current00, update00, minus_one);
        let updated01 = vfmaq_f64(current01, update01, minus_one);
        let updated02 = vfmaq_f64(current02, update02, minus_one);
        let updated03 = vfmaq_f64(current03, update03, minus_one);
        let updated10 = vfmaq_f64(current10, update10, minus_one);
        let updated11 = vfmaq_f64(current11, update11, minus_one);
        let updated12 = vfmaq_f64(current12, update12, minus_one);
        let updated13 = vfmaq_f64(current13, update13, minus_one);
        let updated20 = vfmaq_f64(current20, update20, minus_one);
        let updated21 = vfmaq_f64(current21, update21, minus_one);
        let updated22 = vfmaq_f64(current22, update22, minus_one);
        let updated23 = vfmaq_f64(current23, update23, minus_one);
        let updated30 = vfmaq_f64(current30, update30, minus_one);
        let updated31 = vfmaq_f64(current31, update31, minus_one);
        let updated32 = vfmaq_f64(current32, update32, minus_one);
        let updated33 = vfmaq_f64(current33, update33, minus_one);
        // SAFETY: `row + 7 < size`, so all two-lane stores stay inside the
        // four lower-triangular target columns.
        unsafe {
            vst1q_f64(matrix_ptr.add(entry0), updated00);
            vst1q_f64(matrix_ptr.add(entry0 + 2), updated01);
            vst1q_f64(matrix_ptr.add(entry0 + 4), updated02);
            vst1q_f64(matrix_ptr.add(entry0 + 6), updated03);
            vst1q_f64(matrix_ptr.add(entry1), updated10);
            vst1q_f64(matrix_ptr.add(entry1 + 2), updated11);
            vst1q_f64(matrix_ptr.add(entry1 + 4), updated12);
            vst1q_f64(matrix_ptr.add(entry1 + 6), updated13);
            vst1q_f64(matrix_ptr.add(entry2), updated20);
            vst1q_f64(matrix_ptr.add(entry2 + 2), updated21);
            vst1q_f64(matrix_ptr.add(entry2 + 4), updated22);
            vst1q_f64(matrix_ptr.add(entry2 + 6), updated23);
            vst1q_f64(matrix_ptr.add(entry3), updated30);
            vst1q_f64(matrix_ptr.add(entry3 + 2), updated31);
            vst1q_f64(matrix_ptr.add(entry3 + 4), updated32);
            vst1q_f64(matrix_ptr.add(entry3 + 6), updated33);
        }
        row += 8;
    }
    while row + 3 < size {
        let mut update00 = vdupq_n_f64(0.0);
        let mut update01 = vdupq_n_f64(0.0);
        let mut update10 = vdupq_n_f64(0.0);
        let mut update11 = vdupq_n_f64(0.0);
        let mut update20 = vdupq_n_f64(0.0);
        let mut update21 = vdupq_n_f64(0.0);
        let mut update30 = vdupq_n_f64(0.0);
        let mut update31 = vdupq_n_f64(0.0);
        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            // SAFETY: caller passes `row + 3 < size`, the four unrolled pivots
            // are below `accepted_width`, and the LD workspace is full-width.
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row + 2)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(relative_pivot) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(relative_pivot) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(relative_pivot) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(relative_pivot) });
            update00 = vfmaq_f64(update00, row_ld0, col0_l);
            update01 = vfmaq_f64(update01, row_ld1, col0_l);
            update10 = vfmaq_f64(update10, row_ld0, col1_l);
            update11 = vfmaq_f64(update11, row_ld1, col1_l);
            update20 = vfmaq_f64(update20, row_ld0, col2_l);
            update21 = vfmaq_f64(update21, row_ld1, col2_l);
            update30 = vfmaq_f64(update30, row_ld0, col3_l);
            update31 = vfmaq_f64(update31, row_ld1, col3_l);

            let pivot1 = relative_pivot + 1;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row + 2)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot1) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot1) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot1) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot1) });
            update00 = vfmaq_f64(update00, row_ld0, col0_l);
            update01 = vfmaq_f64(update01, row_ld1, col0_l);
            update10 = vfmaq_f64(update10, row_ld0, col1_l);
            update11 = vfmaq_f64(update11, row_ld1, col1_l);
            update20 = vfmaq_f64(update20, row_ld0, col2_l);
            update21 = vfmaq_f64(update21, row_ld1, col2_l);
            update30 = vfmaq_f64(update30, row_ld0, col3_l);
            update31 = vfmaq_f64(update31, row_ld1, col3_l);

            let pivot2 = relative_pivot + 2;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row + 2)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot2) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot2) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot2) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot2) });
            update00 = vfmaq_f64(update00, row_ld0, col0_l);
            update01 = vfmaq_f64(update01, row_ld1, col0_l);
            update10 = vfmaq_f64(update10, row_ld0, col1_l);
            update11 = vfmaq_f64(update11, row_ld1, col1_l);
            update20 = vfmaq_f64(update20, row_ld0, col2_l);
            update21 = vfmaq_f64(update21, row_ld1, col2_l);
            update30 = vfmaq_f64(update30, row_ld0, col3_l);
            update31 = vfmaq_f64(update31, row_ld1, col3_l);

            let pivot3 = relative_pivot + 3;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row + 2)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot3) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot3) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot3) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot3) });
            update00 = vfmaq_f64(update00, row_ld0, col0_l);
            update01 = vfmaq_f64(update01, row_ld1, col0_l);
            update10 = vfmaq_f64(update10, row_ld0, col1_l);
            update11 = vfmaq_f64(update11, row_ld1, col1_l);
            update20 = vfmaq_f64(update20, row_ld0, col2_l);
            update21 = vfmaq_f64(update21, row_ld1, col2_l);
            update30 = vfmaq_f64(update30, row_ld0, col3_l);
            update31 = vfmaq_f64(update31, row_ld1, col3_l);

            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            // SAFETY: caller passes `row + 3 < size`, `relative_pivot < accepted_width`,
            // and `ld_values` contains at least `accepted_width * size` entries.
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row + 2)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(relative_pivot) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(relative_pivot) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(relative_pivot) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(relative_pivot) });
            update00 = vfmaq_f64(update00, row_ld0, col0_l);
            update01 = vfmaq_f64(update01, row_ld1, col0_l);
            update10 = vfmaq_f64(update10, row_ld0, col1_l);
            update11 = vfmaq_f64(update11, row_ld1, col1_l);
            update20 = vfmaq_f64(update20, row_ld0, col2_l);
            update21 = vfmaq_f64(update21, row_ld1, col2_l);
            update30 = vfmaq_f64(update30, row_ld0, col3_l);
            update31 = vfmaq_f64(update31, row_ld1, col3_l);
            relative_pivot += 1;
        }

        let entry0 = col * size + row;
        let entry1 = col1 * size + row;
        let entry2 = col2 * size + row;
        let entry3 = col3 * size + row;
        let minus_one = vdupq_n_f64(-1.0);
        let current00 = unsafe { vld1q_f64(matrix_ptr.add(entry0)) };
        let current01 = unsafe { vld1q_f64(matrix_ptr.add(entry0 + 2)) };
        let current10 = unsafe { vld1q_f64(matrix_ptr.add(entry1)) };
        let current11 = unsafe { vld1q_f64(matrix_ptr.add(entry1 + 2)) };
        let current20 = unsafe { vld1q_f64(matrix_ptr.add(entry2)) };
        let current21 = unsafe { vld1q_f64(matrix_ptr.add(entry2 + 2)) };
        let current30 = unsafe { vld1q_f64(matrix_ptr.add(entry3)) };
        let current31 = unsafe { vld1q_f64(matrix_ptr.add(entry3 + 2)) };
        let updated00 = vfmaq_f64(current00, update00, minus_one);
        let updated01 = vfmaq_f64(current01, update01, minus_one);
        let updated10 = vfmaq_f64(current10, update10, minus_one);
        let updated11 = vfmaq_f64(current11, update11, minus_one);
        let updated20 = vfmaq_f64(current20, update20, minus_one);
        let updated21 = vfmaq_f64(current21, update21, minus_one);
        let updated30 = vfmaq_f64(current30, update30, minus_one);
        let updated31 = vfmaq_f64(current31, update31, minus_one);
        // SAFETY: `row + 3 < size`, so all two-lane stores stay inside the
        // four lower-triangular target columns.
        unsafe {
            vst1q_f64(matrix_ptr.add(entry0), updated00);
            vst1q_f64(matrix_ptr.add(entry0 + 2), updated01);
            vst1q_f64(matrix_ptr.add(entry1), updated10);
            vst1q_f64(matrix_ptr.add(entry1 + 2), updated11);
            vst1q_f64(matrix_ptr.add(entry2), updated20);
            vst1q_f64(matrix_ptr.add(entry2 + 2), updated21);
            vst1q_f64(matrix_ptr.add(entry3), updated30);
            vst1q_f64(matrix_ptr.add(entry3 + 2), updated31);
        }
        row += 4;
    }
    while row + 1 < size {
        let mut update0 = vdupq_n_f64(0.0);
        let mut update1 = vdupq_n_f64(0.0);
        let mut update2 = vdupq_n_f64(0.0);
        let mut update3 = vdupq_n_f64(0.0);
        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            // SAFETY: caller passes `row + 1 < size`, the four unrolled pivots
            // are below `accepted_width`, and the LD workspace is full-width.
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(relative_pivot) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(relative_pivot) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(relative_pivot) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(relative_pivot) });
            update0 = vfmaq_f64(update0, row_ld, col0_l);
            update1 = vfmaq_f64(update1, row_ld, col1_l);
            update2 = vfmaq_f64(update2, row_ld, col2_l);
            update3 = vfmaq_f64(update3, row_ld, col3_l);

            let pivot1 = relative_pivot + 1;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot1) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot1) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot1) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot1) });
            update0 = vfmaq_f64(update0, row_ld, col0_l);
            update1 = vfmaq_f64(update1, row_ld, col1_l);
            update2 = vfmaq_f64(update2, row_ld, col2_l);
            update3 = vfmaq_f64(update3, row_ld, col3_l);

            let pivot2 = relative_pivot + 2;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot2) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot2) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot2) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot2) });
            update0 = vfmaq_f64(update0, row_ld, col0_l);
            update1 = vfmaq_f64(update1, row_ld, col1_l);
            update2 = vfmaq_f64(update2, row_ld, col2_l);
            update3 = vfmaq_f64(update3, row_ld, col3_l);

            let pivot3 = relative_pivot + 3;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(pivot3) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(pivot3) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(pivot3) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(pivot3) });
            update0 = vfmaq_f64(update0, row_ld, col0_l);
            update1 = vfmaq_f64(update1, row_ld, col1_l);
            update2 = vfmaq_f64(update2, row_ld, col2_l);
            update3 = vfmaq_f64(update3, row_ld, col3_l);

            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            // SAFETY: caller passes `row + 1 < size`, `relative_pivot < accepted_width`,
            // and `ld_values` contains at least `accepted_width * size` entries.
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let col0_l = vdupq_n_f64(unsafe { *col0_l_ptr.add(relative_pivot) });
            let col1_l = vdupq_n_f64(unsafe { *col1_l_ptr.add(relative_pivot) });
            let col2_l = vdupq_n_f64(unsafe { *col2_l_ptr.add(relative_pivot) });
            let col3_l = vdupq_n_f64(unsafe { *col3_l_ptr.add(relative_pivot) });
            update0 = vfmaq_f64(update0, row_ld, col0_l);
            update1 = vfmaq_f64(update1, row_ld, col1_l);
            update2 = vfmaq_f64(update2, row_ld, col2_l);
            update3 = vfmaq_f64(update3, row_ld, col3_l);
            relative_pivot += 1;
        }

        let entry0 = col * size + row;
        let entry1 = col1 * size + row;
        let entry2 = col2 * size + row;
        let entry3 = col3 * size + row;
        let minus_one = vdupq_n_f64(-1.0);
        let current0 = unsafe { vld1q_f64(matrix_ptr.add(entry0)) };
        let current1 = unsafe { vld1q_f64(matrix_ptr.add(entry1)) };
        let current2 = unsafe { vld1q_f64(matrix_ptr.add(entry2)) };
        let current3 = unsafe { vld1q_f64(matrix_ptr.add(entry3)) };
        let updated0 = vfmaq_f64(current0, update0, minus_one);
        let updated1 = vfmaq_f64(current1, update1, minus_one);
        let updated2 = vfmaq_f64(current2, update2, minus_one);
        let updated3 = vfmaq_f64(current3, update3, minus_one);
        // SAFETY: `row + 1 < size`, so all two-lane stores stay inside the
        // four lower-triangular target columns.
        unsafe {
            vst1q_f64(matrix_ptr.add(entry0), updated0);
            vst1q_f64(matrix_ptr.add(entry1), updated1);
            vst1q_f64(matrix_ptr.add(entry2), updated2);
            vst1q_f64(matrix_ptr.add(entry3), updated3);
        }
        row += 2;
    }
    while row < size {
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col,
            row,
            accepted_width,
            ld_values,
            column_l_values,
        );
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col1,
            row,
            accepted_width,
            ld_values,
            next_column_l_values,
        );
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col2,
            row,
            accepted_width,
            ld_values,
            third_column_l_values,
        );
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col3,
            row,
            accepted_width,
            ld_values,
            fourth_column_l_values,
        );
        row += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn app_apply_accepted_prefix_update_two_columns_neon(
    matrix: &mut [f64],
    size: usize,
    col: usize,
    accepted_width: usize,
    ld_values: &[f64],
    column_l_values: &[f64; APP_INNER_BLOCK_SIZE],
    next_column_l_values: &[f64; APP_INNER_BLOCK_SIZE],
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

    app_apply_accepted_prefix_update_scalar_row(
        matrix,
        size,
        col,
        col,
        accepted_width,
        ld_values,
        column_l_values,
    );

    let matrix_ptr = matrix.as_mut_ptr();
    let ld_ptr = ld_values.as_ptr();
    let col_l_ptr = column_l_values.as_ptr();
    let next_col_l_ptr = next_column_l_values.as_ptr();
    let next_col = col + 1;
    let mut row = next_col;
    while row + 3 < size {
        let mut update0 = vdupq_n_f64(0.0);
        let mut update1 = vdupq_n_f64(0.0);
        let mut next_update0 = vdupq_n_f64(0.0);
        let mut next_update1 = vdupq_n_f64(0.0);
        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            // SAFETY: caller passes `row + 3 < size`, the four unrolled pivots
            // are below `accepted_width`, and the LD workspace is full-width.
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(relative_pivot) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);
            next_update0 = vfmaq_f64(next_update0, row_ld0, next_col_l);
            next_update1 = vfmaq_f64(next_update1, row_ld1, next_col_l);

            let pivot1 = relative_pivot + 1;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot1) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(pivot1) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);
            next_update0 = vfmaq_f64(next_update0, row_ld0, next_col_l);
            next_update1 = vfmaq_f64(next_update1, row_ld1, next_col_l);

            let pivot2 = relative_pivot + 2;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot2) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(pivot2) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);
            next_update0 = vfmaq_f64(next_update0, row_ld0, next_col_l);
            next_update1 = vfmaq_f64(next_update1, row_ld1, next_col_l);

            let pivot3 = relative_pivot + 3;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot3) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(pivot3) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);
            next_update0 = vfmaq_f64(next_update0, row_ld0, next_col_l);
            next_update1 = vfmaq_f64(next_update1, row_ld1, next_col_l);

            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            // SAFETY: caller passes `row + 3 < size`, `relative_pivot < accepted_width`,
            // and `ld_values` contains at least `accepted_width * size` entries.
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row + 2)) };
            // SAFETY: `relative_pivot < accepted_width <= APP_INNER_BLOCK_SIZE`.
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(relative_pivot) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);
            next_update0 = vfmaq_f64(next_update0, row_ld0, next_col_l);
            next_update1 = vfmaq_f64(next_update1, row_ld1, next_col_l);
            relative_pivot += 1;
        }
        let entry = col * size + row;
        let next_entry = next_col * size + row;
        // SAFETY: `row + 3 < size`, so both two-lane load/stores stay inside
        // both dense target columns.
        let current0 = unsafe { vld1q_f64(matrix_ptr.add(entry)) };
        let current1 = unsafe { vld1q_f64(matrix_ptr.add(entry + 2)) };
        let next_current0 = unsafe { vld1q_f64(matrix_ptr.add(next_entry)) };
        let next_current1 = unsafe { vld1q_f64(matrix_ptr.add(next_entry + 2)) };
        let minus_one = vdupq_n_f64(-1.0);
        let updated0 = vfmaq_f64(current0, update0, minus_one);
        let updated1 = vfmaq_f64(current1, update1, minus_one);
        let next_updated0 = vfmaq_f64(next_current0, next_update0, minus_one);
        let next_updated1 = vfmaq_f64(next_current1, next_update1, minus_one);
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_f64(matrix_ptr.add(entry), updated0);
            vst1q_f64(matrix_ptr.add(entry + 2), updated1);
            vst1q_f64(matrix_ptr.add(next_entry), next_updated0);
            vst1q_f64(matrix_ptr.add(next_entry + 2), next_updated1);
        }
        row += 4;
    }
    while row + 1 < size {
        let mut update = vdupq_n_f64(0.0);
        let mut next_update = vdupq_n_f64(0.0);
        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            // SAFETY: caller passes `row + 1 < size`, the four unrolled pivots
            // are below `accepted_width`, and the LD workspace is full-width.
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(relative_pivot) });
            update = vfmaq_f64(update, row_ld, col_l);
            next_update = vfmaq_f64(next_update, row_ld, next_col_l);

            let pivot1 = relative_pivot + 1;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot1) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(pivot1) });
            update = vfmaq_f64(update, row_ld, col_l);
            next_update = vfmaq_f64(next_update, row_ld, next_col_l);

            let pivot2 = relative_pivot + 2;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot2) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(pivot2) });
            update = vfmaq_f64(update, row_ld, col_l);
            next_update = vfmaq_f64(next_update, row_ld, next_col_l);

            let pivot3 = relative_pivot + 3;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot3) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(pivot3) });
            update = vfmaq_f64(update, row_ld, col_l);
            next_update = vfmaq_f64(next_update, row_ld, next_col_l);

            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            // SAFETY: caller passes `row + 1 < size`, `relative_pivot < accepted_width`,
            // and `ld_values` contains at least `accepted_width * size` entries.
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            // SAFETY: `relative_pivot < accepted_width <= APP_INNER_BLOCK_SIZE`.
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            let next_col_l = vdupq_n_f64(unsafe { *next_col_l_ptr.add(relative_pivot) });
            update = vfmaq_f64(update, row_ld, col_l);
            next_update = vfmaq_f64(next_update, row_ld, next_col_l);
            relative_pivot += 1;
        }
        let entry = col * size + row;
        let next_entry = next_col * size + row;
        // SAFETY: `row + 1 < size`, so the two-lane load/store stays inside
        // both dense target columns.
        let current = unsafe { vld1q_f64(matrix_ptr.add(entry)) };
        let next_current = unsafe { vld1q_f64(matrix_ptr.add(next_entry)) };
        let updated = vfmaq_f64(current, update, vdupq_n_f64(-1.0));
        let next_updated = vfmaq_f64(next_current, next_update, vdupq_n_f64(-1.0));
        // SAFETY: same bounds as the load above.
        unsafe {
            vst1q_f64(matrix_ptr.add(entry), updated);
            vst1q_f64(matrix_ptr.add(next_entry), next_updated);
        }
        row += 2;
    }
    if row < size {
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col,
            row,
            accepted_width,
            ld_values,
            column_l_values,
        );
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            next_col,
            row,
            accepted_width,
            ld_values,
            next_column_l_values,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn app_apply_accepted_prefix_update_column(
    matrix: &mut [f64],
    size: usize,
    col: usize,
    accepted_width: usize,
    ld_values: &[f64],
    column_l_values: &[f64; APP_INNER_BLOCK_SIZE],
) {
    unsafe {
        app_apply_accepted_prefix_update_column_neon(
            matrix,
            size,
            col,
            accepted_width,
            ld_values,
            column_l_values,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn app_apply_accepted_prefix_update_column_neon(
    matrix: &mut [f64],
    size: usize,
    col: usize,
    accepted_width: usize,
    ld_values: &[f64],
    column_l_values: &[f64; APP_INNER_BLOCK_SIZE],
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

    let matrix_ptr = matrix.as_mut_ptr();
    let ld_ptr = ld_values.as_ptr();
    let col_l_ptr = column_l_values.as_ptr();
    let mut row = col;
    while row + 3 < size {
        let mut update0 = vdupq_n_f64(0.0);
        let mut update1 = vdupq_n_f64(0.0);
        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            // SAFETY: caller passes `row + 3 < size`, `relative_pivot < accepted_width`,
            // and `ld_values` contains at least `accepted_width * size` entries.
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row + 2)) };
            // SAFETY: `relative_pivot < accepted_width <= APP_INNER_BLOCK_SIZE`.
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);

            let pivot1 = relative_pivot + 1;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot1) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);

            let pivot2 = relative_pivot + 2;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot2) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);

            let pivot3 = relative_pivot + 3;
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot3) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);

            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            let row_ld0 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let row_ld1 = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row + 2)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            update0 = vfmaq_f64(update0, row_ld0, col_l);
            update1 = vfmaq_f64(update1, row_ld1, col_l);
            relative_pivot += 1;
        }
        let entry = col * size + row;
        // SAFETY: `row + 3 < size`, so both two-lane load/stores stay inside
        // the dense column `col`.
        let current0 = unsafe { vld1q_f64(matrix_ptr.add(entry)) };
        let current1 = unsafe { vld1q_f64(matrix_ptr.add(entry + 2)) };
        let minus_one = vdupq_n_f64(-1.0);
        let updated0 = vfmaq_f64(current0, update0, minus_one);
        let updated1 = vfmaq_f64(current1, update1, minus_one);
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_f64(matrix_ptr.add(entry), updated0);
            vst1q_f64(matrix_ptr.add(entry + 2), updated1);
        }
        row += 4;
    }
    while row + 1 < size {
        let mut update = vdupq_n_f64(0.0);
        let mut relative_pivot = 0;
        while relative_pivot + 4 <= accepted_width {
            // SAFETY: caller passes `row + 1 < size`, `relative_pivot < accepted_width`,
            // and `ld_values` contains at least `accepted_width * size` entries.
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            // SAFETY: `relative_pivot < accepted_width <= APP_INNER_BLOCK_SIZE`.
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            update = vfmaq_f64(update, row_ld, col_l);

            let pivot1 = relative_pivot + 1;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot1 * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot1) });
            update = vfmaq_f64(update, row_ld, col_l);

            let pivot2 = relative_pivot + 2;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot2 * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot2) });
            update = vfmaq_f64(update, row_ld, col_l);

            let pivot3 = relative_pivot + 3;
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(pivot3 * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(pivot3) });
            update = vfmaq_f64(update, row_ld, col_l);

            relative_pivot += 4;
        }
        while relative_pivot < accepted_width {
            let row_ld = unsafe { vld1q_f64(ld_ptr.add(relative_pivot * size + row)) };
            let col_l = vdupq_n_f64(unsafe { *col_l_ptr.add(relative_pivot) });
            update = vfmaq_f64(update, row_ld, col_l);
            relative_pivot += 1;
        }
        let entry = col * size + row;
        // SAFETY: `row + 1 < size`, so the two-lane load/store stays inside the
        // dense column `col`.
        let current = unsafe { vld1q_f64(matrix_ptr.add(entry)) };
        let updated = vfmaq_f64(current, update, vdupq_n_f64(-1.0));
        // SAFETY: same bounds as the load above.
        unsafe { vst1q_f64(matrix_ptr.add(entry), updated) };
        row += 2;
    }
    if row < size {
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col,
            row,
            accepted_width,
            ld_values,
            column_l_values,
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn app_apply_accepted_prefix_update_column(
    matrix: &mut [f64],
    size: usize,
    col: usize,
    accepted_width: usize,
    ld_values: &[f64],
    column_l_values: &[f64; APP_INNER_BLOCK_SIZE],
) {
    for row in col..size {
        app_apply_accepted_prefix_update_scalar_row(
            matrix,
            size,
            col,
            row,
            accepted_width,
            ld_values,
            column_l_values,
        );
    }
}

#[inline(always)]
fn app_apply_accepted_prefix_update_scalar_row(
    matrix: &mut [f64],
    size: usize,
    col: usize,
    row: usize,
    accepted_width: usize,
    ld_values: &[f64],
    column_l_values: &[f64; APP_INNER_BLOCK_SIZE],
) {
    let mut update = 0.0;
    let mut relative_pivot = 0;
    while relative_pivot + 4 <= accepted_width {
        // SAFETY: `accepted_width <= APP_INNER_BLOCK_SIZE`, `row < size`,
        // and `ld_values` was sliced to `accepted_width * size` above.
        let row_ld0 = unsafe { *ld_values.get_unchecked(relative_pivot * size + row) };
        let col_l0 = unsafe { *column_l_values.get_unchecked(relative_pivot) };
        update = row_ld0.mul_add(col_l0, update);

        let pivot1 = relative_pivot + 1;
        let row_ld1 = unsafe { *ld_values.get_unchecked(pivot1 * size + row) };
        let col_l1 = unsafe { *column_l_values.get_unchecked(pivot1) };
        update = row_ld1.mul_add(col_l1, update);

        let pivot2 = relative_pivot + 2;
        let row_ld2 = unsafe { *ld_values.get_unchecked(pivot2 * size + row) };
        let col_l2 = unsafe { *column_l_values.get_unchecked(pivot2) };
        update = row_ld2.mul_add(col_l2, update);

        let pivot3 = relative_pivot + 3;
        let row_ld3 = unsafe { *ld_values.get_unchecked(pivot3 * size + row) };
        let col_l3 = unsafe { *column_l_values.get_unchecked(pivot3) };
        update = row_ld3.mul_add(col_l3, update);

        relative_pivot += 4;
    }
    while relative_pivot < accepted_width {
        // SAFETY: `accepted_width <= APP_INNER_BLOCK_SIZE`, `row < size`,
        // and `ld_values` was sliced to `accepted_width * size` above.
        let row_ld = unsafe { *ld_values.get_unchecked(relative_pivot * size + row) };
        // SAFETY: same `accepted_width <= APP_INNER_BLOCK_SIZE` bound.
        let col_l = unsafe { *column_l_values.get_unchecked(relative_pivot) };
        update = row_ld.mul_add(col_l, update);
        relative_pivot += 1;
    }
    let entry = col * size + row;
    matrix[entry] = update.mul_add(-1.0, matrix[entry]);
}

fn app_gemv_forward_singleton_column(size: usize, accepted_end: usize) -> Option<usize> {
    let trailing = size.checked_sub(accepted_end)?;
    (trailing % APP_INNER_BLOCK_SIZE == 1).then_some(size - 1)
}

#[cfg(test)]
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

#[cfg(test)]
fn app_build_factor_columns_for_prefix(
    rows: &[usize],
    matrix: &[f64],
    size: usize,
    start: usize,
    end: usize,
) -> Vec<FactorColumn> {
    let mut columns = Vec::with_capacity(end.saturating_sub(start));
    app_extend_factor_columns_for_prefix(&mut columns, rows, matrix, size, start, end);
    columns
}

fn app_extend_factor_columns_for_prefix(
    columns: &mut Vec<FactorColumn>,
    rows: &[usize],
    matrix: &[f64],
    size: usize,
    start: usize,
    end: usize,
) {
    debug_assert!(rows.len() >= size);
    debug_assert_eq!(matrix.len(), size * size);
    columns.reserve(end.saturating_sub(start));
    for col in start..end {
        let entry_count = size.saturating_sub(col + 1);
        let mut entries: Vec<(usize, f64)> = Vec::with_capacity(entry_count);
        let entry_ptr = entries.as_mut_ptr();
        let column_offset = col * size;
        for (offset, row) in ((col + 1)..size).enumerate() {
            // SAFETY: debug assertions and loop bounds ensure `row < size`,
            // `offset < entry_count`, and both source slices cover the APP
            // dense-front storage. `(usize, f64)` is fully initialized here
            // before the vector length is exposed.
            unsafe {
                entry_ptr.add(offset).write((
                    *rows.get_unchecked(row),
                    *matrix.get_unchecked(column_offset + row),
                ));
            }
        }
        // SAFETY: the loop above initializes exactly `entry_count` entries.
        unsafe {
            entries.set_len(entry_count);
        }
        columns.push(FactorColumn {
            global_column: rows[col],
            entries,
        });
    }
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
    let mut block_records = Vec::with_capacity(size);
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
    let contribution_dense = pack_dense_lower_suffix(dense, size, pivot, remaining_size);

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

fn build_dense_front_solve_panel_record(
    factor_order: &[usize],
    trailing_rows: &[usize],
    dense: &[f64],
    size: usize,
    eliminated_len: usize,
) -> Result<Option<FactorSolvePanelRecord>, SsidsError> {
    if eliminated_len == 0 {
        return Ok(None);
    }
    if eliminated_len > size || factor_order.len() != eliminated_len {
        return Err(SsidsError::NumericalBreakdown {
            pivot: eliminated_len,
            detail: "dense solve panel request does not match eliminated front width".into(),
        });
    }
    let local_size = factor_order.len() + trailing_rows.len();
    if local_size != size {
        return Err(SsidsError::NumericalBreakdown {
            pivot: eliminated_len,
            detail: format!("dense solve panel has {local_size} row ids for a {size} row front"),
        });
    }

    let mut row_ids = Vec::with_capacity(size);
    row_ids.extend_from_slice(factor_order);
    row_ids.extend_from_slice(trailing_rows);

    let mut values = Vec::<f64>::with_capacity(size * eliminated_len);
    let values_ptr = values.as_mut_ptr();
    for local_col in 0..eliminated_len {
        let column_start = local_col * size;
        for row in 0..local_col {
            // SAFETY: `values` has capacity for every panel entry and each
            // column range is written exactly once before length is exposed.
            unsafe {
                values_ptr.add(column_start + row).write(0.0);
            }
        }
        // SAFETY: same one-pass initialization invariant as above.
        unsafe {
            values_ptr.add(column_start + local_col).write(1.0);
        }
        let below_diagonal_start = local_col + 1;
        if below_diagonal_start < size {
            let entry_count = size - below_diagonal_start;
            // SAFETY: source and destination are in-bounds for the dense
            // column and the uninitialized panel column tail respectively.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dense.as_ptr().add(column_start + below_diagonal_start),
                    values_ptr.add(column_start + below_diagonal_start),
                    entry_count,
                );
            }
        }
    }
    // SAFETY: every column wrote `size` entries above.
    unsafe {
        values.set_len(size * eliminated_len);
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
    if profile_enabled {
        profile.app_front_count += 1;
        profile.app_front_size_histogram[app_front_size_histogram_bucket(size)] += 1;
    }
    let mut factor_order = Vec::with_capacity(active_candidate_end);
    let mut factor_columns = Vec::with_capacity(active_candidate_end);
    let mut block_records = Vec::with_capacity(active_candidate_end);
    // SPRAL's block_ldlt<32> uses a local 32-row ldwork block; columns still
    // use the dense front lda, so workspace helpers take the current block's
    // row offset.
    let mut scratch = vec![0.0; APP_INNER_BLOCK_SIZE.saturating_mul(size).max(1)];
    let app_subphase_profile_enabled = profile_enabled && factor_app_subphase_debug_enabled();
    let mut rows_before_block = Vec::with_capacity(size);
    let mut dense_before_block = Vec::new();
    let mut pivot = 0;

    while active_candidate_end - pivot >= APP_INNER_BLOCK_SIZE {
        let block_start = pivot;
        let block_end = pivot + APP_INNER_BLOCK_SIZE;
        if profile_enabled {
            profile.app_panel_count += 1;
        }
        let started = profile_enabled.then(Instant::now);
        rows_before_block.clear();
        rows_before_block.extend_from_slice(&rows);
        app_backup_trailing_lower_into(&dense, size, block_start, &mut dense_before_block);
        if let Some(started) = started {
            profile.app_backup_time += started.elapsed();
        }
        let mut local_stats = PanelFactorStats::default();
        let mut local_blocks = Vec::new();
        let mut block_pivot = block_start;

        let started = profile_enabled.then(Instant::now);
        while block_pivot < block_end {
            let maxloc_started = app_subphase_profile_enabled.then(Instant::now);
            let maxloc = dense_find_maxloc(&dense, size, block_pivot, block_end);
            if profile_enabled {
                profile.app_maxloc_calls += 1;
            }
            if let Some(started) = maxloc_started {
                profile.app_maxloc_time += started.elapsed();
            }
            let Some((best_abs, best_row, best_col)) = maxloc else {
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
                    reset_ldwork_column_tail_with_row_offset(
                        &mut scratch,
                        size,
                        block_pivot,
                        block_pivot,
                        block_start,
                    );
                    local_blocks.push(FactorBlockRecord {
                        size: 1,
                        values: [0.0, 0.0, 0.0, 0.0],
                    });
                    if profile_enabled {
                        profile.app_zero_pivots += 1;
                    }
                    block_pivot += 1;
                }
                break;
            }

            if best_row == best_col {
                if best_col != block_pivot {
                    let swap_started = app_subphase_profile_enabled.then(Instant::now);
                    dense_symmetric_swap_with_workspace_row_offset(
                        &mut dense,
                        size,
                        best_col,
                        block_pivot,
                        &mut scratch,
                        block_start,
                    );
                    rows.swap(best_col, block_pivot);
                    if profile_enabled {
                        profile.app_symmetric_swaps += 1;
                    }
                    if let Some(started) = swap_started {
                        profile.app_symmetric_swap_time += started.elapsed();
                    }
                }
                let update_started = app_subphase_profile_enabled.then(Instant::now);
                let block = factor_one_by_one_common_with_workspace_offset(
                    &rows,
                    &mut dense,
                    size,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    AppWorkspaceMut {
                        values: &mut scratch,
                        row_offset: block_start,
                    },
                );
                if let Some(started) = update_started {
                    profile.app_pivot_update_time += started.elapsed();
                }
                let block = block?;
                if profile_enabled {
                    profile.app_one_by_one_pivots += 1;
                }
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
                    let swap_started = app_subphase_profile_enabled.then(Instant::now);
                    dense_symmetric_swap_with_workspace_row_offset(
                        &mut dense,
                        size,
                        index,
                        block_pivot,
                        &mut scratch,
                        block_start,
                    );
                    rows.swap(index, block_pivot);
                    if profile_enabled {
                        profile.app_symmetric_swaps += 1;
                    }
                    if let Some(started) = swap_started {
                        profile.app_symmetric_swap_time += started.elapsed();
                    }
                }
                let update_started = app_subphase_profile_enabled.then(Instant::now);
                let block = factor_one_by_one_common_with_workspace_offset(
                    &rows,
                    &mut dense,
                    size,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    AppWorkspaceMut {
                        values: &mut scratch,
                        row_offset: block_start,
                    },
                );
                if let Some(started) = update_started {
                    profile.app_pivot_update_time += started.elapsed();
                }
                let block = block?;
                if profile_enabled {
                    profile.app_one_by_one_pivots += 1;
                }
                local_blocks.push(block);
                block_pivot += 1;
                continue;
            }

            if let Some(inverse) = two_by_two_inverse {
                if first != block_pivot {
                    let swap_started = app_subphase_profile_enabled.then(Instant::now);
                    dense_symmetric_swap_with_workspace_row_offset(
                        &mut dense,
                        size,
                        first,
                        block_pivot,
                        &mut scratch,
                        block_start,
                    );
                    rows.swap(first, block_pivot);
                    if profile_enabled {
                        profile.app_symmetric_swaps += 1;
                    }
                    if second == block_pivot {
                        second = first;
                    }
                    if let Some(started) = swap_started {
                        profile.app_symmetric_swap_time += started.elapsed();
                    }
                }
                if second != block_pivot + 1 {
                    let swap_started = app_subphase_profile_enabled.then(Instant::now);
                    dense_symmetric_swap_with_workspace_row_offset(
                        &mut dense,
                        size,
                        second,
                        block_pivot + 1,
                        &mut scratch,
                        block_start,
                    );
                    rows.swap(second, block_pivot + 1);
                    if profile_enabled {
                        profile.app_symmetric_swaps += 1;
                    }
                    if let Some(started) = swap_started {
                        profile.app_symmetric_swap_time += started.elapsed();
                    }
                }
                let update_started = app_subphase_profile_enabled.then(Instant::now);
                let block = factor_two_by_two_common_with_workspace_offset(
                    &rows,
                    &mut dense,
                    DenseUpdateBounds {
                        size,
                        update_end: block_end,
                    },
                    block_pivot,
                    inverse,
                    &mut local_stats,
                    AppWorkspaceMut {
                        values: &mut scratch,
                        row_offset: block_start,
                    },
                );
                if let Some(started) = update_started {
                    profile.app_pivot_update_time += started.elapsed();
                }
                let block = block?;
                if profile_enabled {
                    profile.app_two_by_two_pivots += 1;
                }
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
        let accepted_block_count = app_record_count_for_prefix(&local_blocks, local_passed);
        let accepted_blocks = &local_blocks[..accepted_block_count];
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
            AppRestoreRange {
                backup_start: block_start,
                block_end,
                trailing_start: accepted_end,
            },
        );
        if let Some(started) = started {
            profile.app_restore_time += started.elapsed();
        }
        let started = profile_enabled.then(Instant::now);
        let (accepted_ld_time, accepted_gemm_time) =
            app_apply_accepted_prefix_update_with_workspace(
                &mut dense,
                size,
                block_start,
                accepted_end,
                accepted_blocks,
                &mut scratch,
                profile_enabled,
            );
        if let Some(started) = started {
            profile.app_accepted_update_time += started.elapsed();
            profile.app_accepted_ld_time += accepted_ld_time;
            profile.app_accepted_gemm_time += accepted_gemm_time;
        }

        factor_order.extend(rows[block_start..accepted_end].iter().copied());
        let started = profile_enabled.then(Instant::now);
        app_extend_factor_columns_for_prefix(
            &mut factor_columns,
            &rows,
            &dense,
            size,
            block_start,
            accepted_end,
        );
        if let Some(started) = started {
            profile.app_column_storage_time += started.elapsed();
        }
        stats.two_by_two_pivots += accepted_blocks
            .iter()
            .filter(|block| block.size == 2)
            .count();
        stats.max_residual = stats.max_residual.max(local_stats.max_residual);
        block_records.extend_from_slice(accepted_blocks);
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

    let contribution_dense = pack_dense_lower_suffix(&dense, size, pivot, remaining_size);
    stats.delayed_pivots += delayed_count;

    let contribution = ContributionBlock {
        row_ids: remaining_rows,
        delayed_count,
        dense: contribution_dense,
    };
    let mut solve_panels = Vec::new();
    let started = profile_enabled.then(Instant::now);
    if let Some(panel) = build_dense_front_solve_panel_record(
        &factor_order,
        &contribution.row_ids,
        &dense,
        size,
        pivot,
    )? {
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

    let front_elimination_capacity = front.columns.len() + front.interface_rows.len();
    let mut factor_order = Vec::with_capacity(front_elimination_capacity);
    let mut factor_columns = Vec::with_capacity(front_elimination_capacity);
    let mut block_records = Vec::with_capacity(front_elimination_capacity);
    let mut solve_panels = Vec::with_capacity(child_results.len() + 1);
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
    let contiguous_leaf_front = child_contributions.is_empty()
        && front.interface_rows.is_empty()
        && front
            .columns
            .windows(2)
            .all(|window| window[1] == window[0] + 1);
    let (local_rows, local_dense) = if contiguous_leaf_front {
        let local_rows = front.columns.clone();
        let local_size = local_rows.len();
        let mut local_dense = vec![0.0; local_size * local_size];
        let first_row = local_rows.first().copied().unwrap_or(0);
        let row_limit = first_row + local_size;
        for &column in &front.columns {
            let local_column = column - first_row;
            for entry in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
                let row = matrix.row_indices[entry];
                if row >= first_row && row < row_limit {
                    // In this guarded path local numbering is a contiguous
                    // offset of the permuted lower CSC numbering.
                    let local_row = row - first_row;
                    local_dense[dense_lower_offset(local_size, local_row, local_column)] =
                        matrix.values[entry];
                }
            }
        }
        (local_rows, local_dense)
    } else {
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
                // Mirrors SPRAL ssids/cpu/kernels/assemble.hxx::add_a_block:
                // original A entries are assigned into the node. Contributions
                // from children are accumulated separately below.
                local_dense[offset] = value;
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
        (local_rows, local_dense)
    };
    let local_size = local_rows.len();
    max_front_size = max_front_size.max(local_size);
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
    scaling: Option<&[f64]>,
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
    if let Some(scaling) = scaling {
        fill_scaled_permuted_lower_csc_values(
            matrix,
            buffers.permuted_matrix_col_ptrs,
            buffers.permuted_matrix_row_indices,
            buffers.permuted_matrix_source_positions,
            scaling,
            buffers.permuted_matrix_values,
        )?;
    } else {
        fill_permuted_lower_csc_values(
            matrix,
            buffers.permuted_matrix_source_positions,
            buffers.permuted_matrix_values,
        )?;
    }
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
    let mut block_records = Vec::with_capacity(dimension);
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
    let lower_entry_count = factor_columns
        .iter()
        .map(|column| column.entries.len())
        .sum::<usize>();
    buffers.lower_col_ptrs.clear();
    buffers.lower_col_ptrs.reserve(dimension + 1);
    buffers.lower_col_ptrs.push(0);
    buffers.lower_row_indices.clear();
    buffers.lower_row_indices.reserve(lower_entry_count);
    buffers.lower_values.clear();
    buffers.lower_values.reserve(lower_entry_count);
    let row_position_ptr = buffers.lower_row_indices.as_mut_ptr();
    let value_ptr = buffers.lower_values.as_mut_ptr();
    let mut lower_entry_cursor = 0;
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
            debug_assert!(lower_entry_cursor < lower_entry_count);
            // SAFETY: `reserve(lower_entry_count)` above gives both output
            // vectors enough capacity for every validated factor entry. The
            // cursor is advanced exactly once per entry, and lengths are
            // exposed only after all writes complete.
            unsafe {
                row_position_ptr.add(lower_entry_cursor).write(row_position);
                value_ptr.add(lower_entry_cursor).write(value);
            }
            lower_entry_cursor += 1;
        }
        buffers.lower_col_ptrs.push(lower_entry_cursor);
    }
    debug_assert_eq!(lower_entry_cursor, lower_entry_count);
    // SAFETY: the loop above initialized exactly `lower_entry_cursor` entries
    // in both vectors.
    unsafe {
        buffers.lower_row_indices.set_len(lower_entry_cursor);
        buffers.lower_values.set_len(lower_entry_cursor);
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
        + lower_entry_count
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

// Mirrors SPRAL SSIDS CPU solve order in NumericSubtree.hxx:
// solve_diag_bwd_inner<true, true>() gathers a front-local RHS, applies the
// front-local inverse-D blocks, runs ldlt_app_solve_bwd(), then scatters only
// the eliminated rows. Full native solves call this combined path from
// fkeep.F90::inner_solve_cpu for SSIDS_SOLVE_JOB_ALL.
fn solve_diagonal_and_lower_transpose_front_panels_like_native(
    panels: &[SolvePanel],
    diagonal_blocks: &[DiagonalBlock],
    diagonal_values: &[f64],
    factor_rhs: &mut [f64],
    mut profile: Option<&mut SolveProfile>,
) -> Result<(), SsidsError> {
    if diagonal_values.len() != diagonal_blocks.len() * 4 {
        return Err(SsidsError::NumericalBreakdown {
            pivot: 0,
            detail: "diagonal solve metadata has inconsistent block/value counts".into(),
        });
    }

    let mut panel_offsets: Vec<usize> = Vec::with_capacity(panels.len() + 1);
    panel_offsets.push(0);
    for panel in panels {
        let next = panel_offsets
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(panel.eliminated_len)
            .ok_or_else(|| SsidsError::NumericalBreakdown {
                pivot: factor_rhs.len().saturating_sub(1),
                detail: "solve panel eliminated-length sum overflowed".into(),
            })?;
        panel_offsets.push(next);
    }
    let total_eliminated = panel_offsets.last().copied().unwrap_or(0);
    if total_eliminated != factor_rhs.len() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: total_eliminated.min(factor_rhs.len().saturating_sub(1)),
            detail: format!(
                "solve panels eliminate {total_eliminated} rows for a {}-row factor",
                factor_rhs.len()
            ),
        });
    }

    let mut panel_block_ranges = Vec::with_capacity(panels.len());
    let mut block_index = 0;
    let mut block_start = 0;
    for window in panel_offsets.windows(2) {
        let panel_start = window[0];
        let panel_end = window[1];
        let first_block = block_index;
        while block_start < panel_end {
            let block =
                diagonal_blocks
                    .get(block_index)
                    .ok_or_else(|| SsidsError::NumericalBreakdown {
                        pivot: block_start,
                        detail: "solve panel diagonal metadata ended early".into(),
                    })?;
            if block.size != 1 && block.size != 2 {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: block_start,
                    detail: format!(
                        "unexpected dense diagonal block of size {} in solve path",
                        block.size
                    ),
                });
            }
            let block_end = block_start + block.size;
            if block_end > panel_end {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: block_start,
                    detail: "diagonal block crosses a native solve panel boundary".into(),
                });
            }
            block_start = block_end;
            block_index += 1;
        }
        if block_start != panel_end || panel_start > panel_end {
            return Err(SsidsError::NumericalBreakdown {
                pivot: panel_start,
                detail: "solve panel diagonal block range is inconsistent".into(),
            });
        }
        panel_block_ranges.push((first_block, block_index));
    }
    if block_index != diagonal_blocks.len() || block_start != factor_rhs.len() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: block_start.min(factor_rhs.len().saturating_sub(1)),
            detail: "diagonal solve metadata does not cover exactly the factor dimension".into(),
        });
    }

    let profile_enabled = profile.is_some();
    let mut local_rhs = Vec::new();
    for (panel_index, panel) in panels.iter().enumerate().rev() {
        let eliminated_len = panel.eliminated_len;
        let local_size = panel.row_positions.len();
        debug_assert!(eliminated_len <= local_size);
        debug_assert_eq!(panel.values.len(), local_size * eliminated_len);

        local_rhs.resize(local_size, 0.0);
        for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
            local_rhs[local_row] = factor_rhs[factor_position];
        }

        let started = profile_enabled.then(Instant::now);
        let (first_block, end_block) = panel_block_ranges[panel_index];
        let mut local_start = 0;
        for (block_index, &block) in diagonal_blocks
            .iter()
            .enumerate()
            .take(end_block)
            .skip(first_block)
        {
            let values_start = 4 * block_index;
            let values = &diagonal_values[values_start..values_start + 4];
            if block.size == 1 {
                let inverse_diagonal =
                    one_by_one_inverse_diagonal(&values[..2]).map_err(|detail| {
                        SsidsError::NumericalBreakdown {
                            pivot: panel_offsets[panel_index] + local_start,
                            detail,
                        }
                    })?;
                if !inverse_diagonal.is_finite() {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: panel_offsets[panel_index] + local_start,
                        detail: "diagonal pivot vanished during solve".into(),
                    });
                }
                local_rhs[local_start] *= inverse_diagonal;
                local_start += 1;
            } else {
                let local_end = local_start + 2;
                solve_two_by_two_block_in_place(values, &mut local_rhs[local_start..local_end])
                    .map_err(|detail| SsidsError::NumericalBreakdown {
                        pivot: panel_offsets[panel_index] + local_start,
                        detail,
                    })?;
                local_start = local_end;
            }
        }
        debug_assert_eq!(local_start, eliminated_len);
        if let Some(profile) = profile.as_mut()
            && let Some(started) = started
        {
            profile.diagonal_solve_time += started.elapsed();
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
        openblas_trsv_lower_unit_op_t_like_native(
            &panel.values,
            local_size,
            &mut local_rhs[..eliminated_len],
        );
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

    Ok(())
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
        APP_INNER_BLOCK_SIZE, AppRestoreRange, DenseTppTailRequest, DenseUpdateBounds,
        DiagonalBlock, FactorBlockRecord, NativeOrdering, NativeSpral, NumericFactorOptions,
        OrderingStrategy, PanelFactorStats, PivotMethod, SolvePanel, SsidsError, SsidsOptions,
        SymmetricCscMatrix, analyse, app_adjust_passed_prefix, app_apply_accepted_prefix_update,
        app_apply_block_pivots_to_trailing_rows, app_backup_trailing_lower,
        app_build_factor_columns_for_prefix, app_build_ld_tile_workspace, app_build_ld_workspace,
        app_first_failed_trailing_column, app_gemv_forward_singleton_column,
        app_restore_trailing_from_block_backup, app_solve_block_triangular_to_trailing_rows,
        app_target_block_uses_gemv_forward, app_truncate_records_to_prefix, app_two_by_two_inverse,
        app_update_one_by_one, app_update_two_by_two, apply_permuted_symmetric_scaling,
        build_dense_front_solve_panel_record, build_factor_solve_panel_record,
        build_native_row_list_supernodes, build_native_row_list_supernodes_fast,
        build_permuted_lower_csc_pattern, build_symbolic_front_tree, dense_find_maxloc,
        dense_lower_offset, dense_symmetric_swap_with_workspace, expand_symmetric_pattern,
        factor_one_by_one_common, factor_two_by_two_common, factorize, factorize_dense_front,
        factorize_dense_tpp_tail_in_place, fill_permuted_lower_csc_values,
        fill_scaled_permuted_lower_csc_values, native_column_counts, native_postorder_permutation,
        native_supernode_layout, openblas_gemv_n_update_like_native,
        openblas_gemv_t_dot_like_contiguous, openblas_trsv_lower_unit_op_n_like_native,
        openblas_trsv_lower_unit_op_t_like_native, permute_graph, permute_graph_with_bitsets,
        permute_graph_with_sorted_edges, reset_ldwork_column_tail,
        solve_diagonal_and_lower_transpose_front_panels_like_native,
        solve_forward_front_panels_like_native, solve_two_by_two_block_in_place,
        symbolic_factor_pattern, zero_dense_column_until,
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
    type LdltAppFactorFn = unsafe extern "C" fn(
        c_int,
        c_int,
        *mut c_int,
        *mut f64,
        c_int,
        *mut f64,
        c_int,
        f64,
        f64,
        c_int,
        c_int,
    ) -> c_int;
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
        ldlt_app_factor: LdltAppFactorFn,
        ldlt_app_solve_fwd: LdltAppSolveFn,
        ldlt_app_solve_diag: LdltAppSolveDiagFn,
        ldlt_app_solve_bwd: LdltAppSolveFn,
        align_lda_double: AlignLdaFn,
        block_update_1x1_32: BlockUpdate1x1Fn,
        block_update_2x2_32: BlockUpdate2x2Fn,
        block_swap_cols_32: BlockSwapColsFn,
        block_find_maxloc_32: BlockFindMaxlocFn,
        block_test_2x2: BlockTest2x2Fn,
        block_test_2x2_full_block_codegen: BlockTest2x2Fn,
        block_two_by_two_multipliers: BlockTwoByTwoMultipliersFn,
        block_first_step_32: BlockFirstStep32Fn,
        block_prefix_trace_32: BlockPrefixTrace32Fn,
        block_prefix_trace_32_source_multiplier: BlockPrefixTrace32Fn,
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
            .ok_or_else(|| "ssids_rs manifest has no parent".to_string())?
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
        let ldlt_app_factor = unsafe {
            *library
                .get::<LdltAppFactorFn>(b"spral_kernel_ldlt_app_factor\0")
                .map_err(|error| format!("failed to load ldlt_app_factor shim: {error}"))?
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
        let block_test_2x2_full_block_codegen = unsafe {
            *library
                .get::<BlockTest2x2Fn>(b"spral_kernel_block_test_2x2_full_block_codegen\0")
                .map_err(|error| {
                    format!("failed to load full-block-codegen block_test_2x2 shim: {error}")
                })?
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
        let block_prefix_trace_32_source_multiplier = unsafe {
            *library
                .get::<BlockPrefixTrace32Fn>(
                    b"spral_kernel_block_prefix_trace_32_source_multiplier\0",
                )
                .map_err(|error| {
                    format!("failed to load source-multiplier block_prefix_trace_32 shim: {error}")
                })?
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
            ldlt_app_factor,
            ldlt_app_solve_fwd,
            ldlt_app_solve_diag,
            ldlt_app_solve_bwd,
            align_lda_double,
            block_update_1x1_32,
            block_update_2x2_32,
            block_swap_cols_32,
            block_find_maxloc_32,
            block_test_2x2,
            block_test_2x2_full_block_codegen,
            block_two_by_two_multipliers,
            block_first_step_32,
            block_prefix_trace_32,
            block_prefix_trace_32_source_multiplier,
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
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ssids/cpu/kernels/common.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/block_ldlt.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

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

static bool spral_kernel_block_test_2x2_full_block_codegen_impl(
      double a11, double a21, double a22, double* detpiv, double* detscale) {
   *detscale = 1.0/fabs(a21);
   *detpiv = std::fma(a11*(*detscale), a22, -fabs(a21));
   return fabs(*detpiv) >= fabs(a21)/2;
}

extern "C" int spral_kernel_block_test_2x2_full_block_codegen(
      double a11, double a21, double a22, double* detpiv, double* detscale) {
   return spral_kernel_block_test_2x2_full_block_codegen_impl(
         a11, a21, a22, detpiv, detscale) ? 1 : 0;
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

extern "C" int spral_kernel_ldlt_app_factor(
      int m, int n, int* perm, double* a, int lda, double* d, int action,
      double u, double small, int cpu_block_size, int pivot_method) {
   spral::ssids::cpu::cpu_factor_options options;
   options.print_level = -1;
   options.action = action != 0;
   options.small = small;
   options.u = u;
   options.multiplier = 1.1;
   options.small_subtree_threshold = 4000000;
   options.cpu_block_size = cpu_block_size;
   options.pivot_method =
      static_cast<spral::ssids::cpu::PivotMethod>(pivot_method);
   options.failed_pivot_method = spral::ssids::cpu::FailedPivotMethod::tpp;
   std::vector<spral::ssids::cpu::Workspace> work;
   work.reserve(256);
   for(int i = 0; i < 256; ++i)
      work.emplace_back(1);
   size_t const backup_bytes =
      spral::ssids::cpu::align_lda<double>(m) * n * sizeof(double)
#if defined(__AVX512F__)
      + 64;
#elif defined(__AVX__)
      + 32;
#else
      + 16;
#endif
   spral::ssids::cpu::BuddyAllocator<double, std::allocator<double>> alloc(
         std::max<size_t>(backup_bytes, 1024));
   return spral::ssids::cpu::ldlt_app_factor<double>(
         m, n, perm, a, lda, d, 0.0, nullptr, 0, options, work, alloc);
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

static double spral_kernel_block_first_multiplier(
      int p, int r, double d11, double d21, const double* work) {
   int const first_row = p + 2;
   int const trailing_rows = 32 - first_row;
   int const vector_rows = (trailing_rows >= 4) ? (trailing_rows / 2) * 2
                         : (trailing_rows == 3) ? 2
                         : 0;
   if(r - first_row < vector_rows)
      return std::fma(d21, work[32+r], d11*work[r]);
   else
      return std::fma(d11, work[r], d21*work[32+r]);
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
      if(spral_kernel_block_test_2x2_full_block_codegen_impl(a11, a21, a22, &detpiv, &detscale)) {
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
         a[p*lda+r] = spral_kernel_block_first_multiplier(p, r, d11, d21, work);
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

template <bool source_multiplier, bool source_update>
static int spral_kernel_block_prefix_trace_32_impl(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int max_steps,
      int* trace_from, int* trace_status, int* trace_next, int* trace_perm,
      int* trace_lperm, double* trace_matrix, double* trace_ldwork,
      double* trace_d) {
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
         if(spral_kernel_block_test_2x2_full_block_codegen_impl(a11, a21, a22, &detpiv, &detscale)) {
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
            if(source_multiplier) {
               a[p*lda+r] = d11*work[r] + d21*work[32+r];
               a[(p+1)*lda+r] = d21*work[r] + d22*work[32+r];
            } else {
               a[p*lda+r] = spral_kernel_block_first_multiplier(p, r, d11, d21, work);
               a[(p+1)*lda+r] = std::fma(d21, work[r], d22*work[32+r]);
            }
         }
         if(source_update) {
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
   return spral_kernel_block_prefix_trace_32_impl<false, false>(
         from, perm, a, lda, d, ldwork, action, u, small, lperm, max_steps,
         trace_from, trace_status, trace_next, trace_perm, trace_lperm,
         trace_matrix, trace_ldwork, trace_d);
}

extern "C" int spral_kernel_block_prefix_trace_32_source_multiplier(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int max_steps,
      int* trace_from, int* trace_status, int* trace_next, int* trace_perm,
      int* trace_lperm, double* trace_matrix, double* trace_ldwork,
      double* trace_d) {
   return spral_kernel_block_prefix_trace_32_impl<true, false>(
         from, perm, a, lda, d, ldwork, action, u, small, lperm, max_steps,
         trace_from, trace_status, trace_next, trace_perm, trace_lperm,
         trace_matrix, trace_ldwork, trace_d);
}

extern "C" int spral_kernel_block_prefix_trace_32_source(
      int from, int* perm, double* a, int lda, double* d, double* ldwork,
      int action, double u, double small, int* lperm, int max_steps,
      int* trace_from, int* trace_status, int* trace_next, int* trace_perm,
      int* trace_lperm, double* trace_matrix, double* trace_ldwork,
      double* trace_d) {
   return spral_kernel_block_prefix_trace_32_impl<true, true>(
         from, perm, a, lda, d, ldwork, action, u, small, lperm, max_steps,
         trace_from, trace_status, trace_next, trace_perm, trace_lperm,
         trace_matrix, trace_ldwork, trace_d);
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

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BlockContinuationMismatch {
        step: usize,
        from: usize,
        status: i32,
        next: usize,
        component: &'static str,
        index: usize,
        row: usize,
        col: usize,
        continued_bits: u64,
        block_bits: u64,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BlockPrefixTraceMismatch {
        step: usize,
        from: usize,
        status: i32,
        next: usize,
        component: &'static str,
        index: usize,
        row: usize,
        col: usize,
        rust_bits: u64,
        native_bits: u64,
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

    fn solve_panel_block_ranges(
        panels: &[SolvePanel],
        diagonal_blocks: &[DiagonalBlock],
    ) -> Vec<(usize, usize)> {
        let mut ranges = Vec::with_capacity(panels.len());
        let mut block_index = 0;
        for panel in panels {
            let first_block = block_index;
            let mut covered = 0;
            while covered < panel.eliminated_len {
                let block = diagonal_blocks
                    .get(block_index)
                    .expect("diagonal blocks cover solve panels");
                covered += block.size;
                block_index += 1;
            }
            assert_eq!(covered, panel.eliminated_len);
            ranges.push((first_block, block_index));
        }
        assert_eq!(block_index, diagonal_blocks.len());
        ranges
    }

    fn native_app_diagonal_for_block_range(
        diagonal_blocks: &[DiagonalBlock],
        diagonal_values: &[f64],
        block_range: (usize, usize),
        eliminated_len: usize,
    ) -> Vec<f64> {
        let mut diagonal = vec![0.0; 2 * eliminated_len.max(1)];
        let mut local_col = 0;
        for block_index in block_range.0..block_range.1 {
            let block = diagonal_blocks[block_index];
            let values = &diagonal_values[4 * block_index..4 * block_index + 4];
            if block.size == 1 {
                diagonal[2 * local_col] = values[0];
                diagonal[2 * local_col + 1] = values[1];
                local_col += 1;
            } else {
                assert_eq!(block.size, 2);
                diagonal[2 * local_col] = values[0];
                diagonal[2 * local_col + 1] = values[1];
                diagonal[2 * local_col + 2] = values[2];
                diagonal[2 * local_col + 3] = values[3];
                local_col += 2;
            }
        }
        assert_eq!(local_col, eliminated_len);
        diagonal
    }

    fn native_solve_forward_front_panels(
        shim: &NativeKernelShim,
        panels: &[SolvePanel],
        factor_rhs: &mut [f64],
    ) {
        let mut local_rhs = Vec::new();
        for panel in panels {
            local_rhs.resize(panel.row_positions.len(), 0.0);
            for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
                local_rhs[local_row] = factor_rhs[factor_position];
            }
            let case = AppSolveKernelCase {
                seed: 0,
                rows: panel.row_positions.len(),
                eliminated_len: panel.eliminated_len,
                lower: panel.values.clone(),
                diagonal: Vec::new(),
                rhs: Vec::new(),
            };
            native_ldlt_app_solve_fwd(shim, &case, &mut local_rhs);
            for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
                factor_rhs[factor_position] = local_rhs[local_row];
            }
        }
    }

    fn native_solve_diagonal_and_bwd_front_panels(
        shim: &NativeKernelShim,
        panels: &[SolvePanel],
        diagonal_blocks: &[DiagonalBlock],
        diagonal_values: &[f64],
        factor_rhs: &mut [f64],
    ) {
        let ranges = solve_panel_block_ranges(panels, diagonal_blocks);
        let mut local_rhs = Vec::new();
        for (panel, block_range) in panels.iter().zip(ranges).rev() {
            local_rhs.resize(panel.row_positions.len(), 0.0);
            for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
                local_rhs[local_row] = factor_rhs[factor_position];
            }
            let case = AppSolveKernelCase {
                seed: 0,
                rows: panel.row_positions.len(),
                eliminated_len: panel.eliminated_len,
                lower: panel.values.clone(),
                diagonal: native_app_diagonal_for_block_range(
                    diagonal_blocks,
                    diagonal_values,
                    block_range,
                    panel.eliminated_len,
                ),
                rhs: Vec::new(),
            };
            native_ldlt_app_solve_diag(shim, &case, &mut local_rhs);
            native_ldlt_app_solve_bwd(shim, &case, &mut local_rhs);
            for local_row in 0..panel.eliminated_len {
                factor_rhs[panel.row_positions[local_row]] = local_rhs[local_row];
            }
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

    fn native_block_test_2x2_full_block_codegen(
        shim: &NativeKernelShim,
        a11: f64,
        a21: f64,
        a22: f64,
    ) -> (bool, f64, f64) {
        let mut detpiv = 0.0;
        let mut detscale = 0.0;
        let accepted = unsafe {
            (shim.block_test_2x2_full_block_codegen)(
                a11,
                a21,
                a22,
                &mut detpiv as *mut f64,
                &mut detscale as *mut f64,
            )
        } != 0;
        (accepted, detpiv, detscale)
    }

    fn app_two_by_two_inverse_source_test_2x2(
        a11: f64,
        a21: f64,
        a22: f64,
        small: f64,
    ) -> Option<(f64, f64, f64)> {
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
        native_block_prefix_trace_32_impl(dense, size, lda, options, shim.block_prefix_trace_32)
    }

    fn native_block_prefix_trace_32_source(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
    ) -> Vec<BlockPrefixSnapshot> {
        native_block_prefix_trace_32_impl(
            dense,
            size,
            lda,
            options,
            shim.block_prefix_trace_32_source,
        )
    }

    fn native_block_prefix_trace_32_source_multiplier(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
    ) -> Vec<BlockPrefixSnapshot> {
        native_block_prefix_trace_32_impl(
            dense,
            size,
            lda,
            options,
            shim.block_prefix_trace_32_source_multiplier,
        )
    }

    fn native_block_prefix_trace_32_impl(
        dense: &[f64],
        size: usize,
        lda: usize,
        options: NumericFactorOptions,
        trace_fn: BlockPrefixTrace32Fn,
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

    fn native_ldlt_app_factor_from_lower_dense(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        options: NumericFactorOptions,
    ) -> DenseTppKernelResult {
        native_ldlt_app_factor_from_lower_dense_with_block_size(shim, dense, size, options, 256)
    }

    fn native_ldlt_app_factor_from_lower_dense_with_block_size(
        shim: &NativeKernelShim,
        dense: &[f64],
        size: usize,
        options: NumericFactorOptions,
        cpu_block_size: i32,
    ) -> DenseTppKernelResult {
        debug_assert_eq!(dense.len(), size * size);
        let lda = native_aligned_double_stride(shim, size);
        let mut matrix = vec![0.0; lda * size];
        for col in 0..size {
            for row in col..size {
                matrix[col * lda + row] = dense[dense_lower_offset(size, row, col)];
            }
        }
        let mut perm = (0..size as c_int).collect::<Vec<_>>();
        let mut diagonal = vec![0.0; 2 * size.max(1)];
        let pivot_method = match options.pivot_method {
            PivotMethod::AggressiveAposteriori => 1,
            PivotMethod::BlockAposteriori => 2,
            PivotMethod::ThresholdPartial => 3,
        };
        let eliminated = unsafe {
            (shim.ldlt_app_factor)(
                size as c_int,
                size as c_int,
                perm.as_mut_ptr(),
                matrix.as_mut_ptr(),
                lda as c_int,
                diagonal.as_mut_ptr(),
                i32::from(options.action_on_zero_pivot),
                options.threshold_pivot_u,
                options.small_pivot_tolerance,
                cpu_block_size,
                pivot_method,
            )
        };
        assert!(
            eliminated >= 0,
            "native ldlt_app_factor returned {eliminated}"
        );
        DenseTppKernelResult {
            eliminated: eliminated as usize,
            perm: perm.into_iter().map(|entry| entry as usize).collect(),
            matrix,
            diagonal,
        }
    }

    fn native_block_ldlt_32_continue_from_snapshot(
        shim: &NativeKernelShim,
        snapshot: &BlockPrefixSnapshot,
        lda: usize,
        options: NumericFactorOptions,
    ) -> BlockLdltKernelResult {
        debug_assert!(lda >= APP_INNER_BLOCK_SIZE);
        let mut matrix = vec![0.0; lda * APP_INNER_BLOCK_SIZE];
        for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                matrix[col * lda + row] = snapshot.matrix[col * APP_INNER_BLOCK_SIZE + row];
            }
        }
        let mut perm = snapshot
            .perm
            .iter()
            .map(|&entry| entry as c_int)
            .collect::<Vec<_>>();
        let mut local_perm = snapshot
            .local_perm
            .iter()
            .map(|&entry| entry as c_int)
            .collect::<Vec<_>>();
        let mut diagonal = snapshot.diagonal.clone();
        let mut ldwork = snapshot.workspace.clone();
        unsafe {
            (shim.block_ldlt_32)(
                snapshot.next as c_int,
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

    fn first_native_block_continuation_mismatch(
        shim: &NativeKernelShim,
        snapshots: &[BlockPrefixSnapshot],
        expected: &BlockLdltKernelResult,
        lda: usize,
        options: NumericFactorOptions,
    ) -> Option<BlockContinuationMismatch> {
        for snapshot in snapshots {
            let continued =
                native_block_ldlt_32_continue_from_snapshot(shim, snapshot, lda, options);
            if continued.perm != expected.perm {
                return Some(BlockContinuationMismatch {
                    step: snapshot.step,
                    from: snapshot.from,
                    status: snapshot.status,
                    next: snapshot.next,
                    component: "perm",
                    index: 0,
                    row: 0,
                    col: 0,
                    continued_bits: 0,
                    block_bits: 0,
                });
            }
            if continued.local_perm != expected.local_perm {
                return Some(BlockContinuationMismatch {
                    step: snapshot.step,
                    from: snapshot.from,
                    status: snapshot.status,
                    next: snapshot.next,
                    component: "local_perm",
                    index: 0,
                    row: 0,
                    col: 0,
                    continued_bits: 0,
                    block_bits: 0,
                });
            }
            for (index, (&continued_value, &block_value)) in continued
                .diagonal
                .iter()
                .zip(&expected.diagonal)
                .enumerate()
            {
                if continued_value.to_bits() != block_value.to_bits() {
                    return Some(BlockContinuationMismatch {
                        step: snapshot.step,
                        from: snapshot.from,
                        status: snapshot.status,
                        next: snapshot.next,
                        component: "diagonal",
                        index,
                        row: 0,
                        col: 0,
                        continued_bits: continued_value.to_bits(),
                        block_bits: block_value.to_bits(),
                    });
                }
            }
            for col in 0..APP_INNER_BLOCK_SIZE {
                for row in col..APP_INNER_BLOCK_SIZE {
                    let continued_bits = continued.matrix[col * lda + row].to_bits();
                    let block_bits = expected.matrix[col * lda + row].to_bits();
                    if continued_bits != block_bits {
                        return Some(BlockContinuationMismatch {
                            step: snapshot.step,
                            from: snapshot.from,
                            status: snapshot.status,
                            next: snapshot.next,
                            component: "matrix",
                            index: row * APP_INNER_BLOCK_SIZE + col,
                            row,
                            col,
                            continued_bits,
                            block_bits,
                        });
                    }
                }
            }
        }
        None
    }

    #[derive(Debug, PartialEq, Eq)]
    struct FrozenPrefixMismatch {
        step: usize,
        next: usize,
        row: usize,
        col: usize,
        source_row: usize,
        snapshot_bits: u64,
        block_bits: u64,
    }

    fn first_frozen_prefix_mismatch_against_block(
        snapshot: &BlockPrefixSnapshot,
        expected: &BlockLdltKernelResult,
        lda: usize,
    ) -> Option<FrozenPrefixMismatch> {
        for col in 0..snapshot.next {
            debug_assert_eq!(snapshot.local_perm[col], expected.local_perm[col]);
            for row in col..APP_INNER_BLOCK_SIZE {
                let source_row = snapshot.local_perm[row];
                let final_row = expected
                    .local_perm
                    .iter()
                    .position(|&entry| entry == source_row)
                    .expect("snapshot source row exists in final block permutation");
                let snapshot_value = snapshot.matrix[col * APP_INNER_BLOCK_SIZE + row];
                let block_value = expected.matrix[col * lda + final_row];
                if snapshot_value.to_bits() != block_value.to_bits() {
                    return Some(FrozenPrefixMismatch {
                        step: snapshot.step,
                        next: snapshot.next,
                        row,
                        col,
                        source_row,
                        snapshot_bits: snapshot_value.to_bits(),
                        block_bits: block_value.to_bits(),
                    });
                }
            }
        }
        None
    }

    fn two_by_two_first_multiplier_expression_bits(
        snapshot: &BlockPrefixSnapshot,
        row: usize,
    ) -> (u64, u64, u64, u64) {
        let pivot = snapshot.from;
        let d11 = snapshot.diagonal[2 * pivot];
        let d21 = snapshot.diagonal[2 * pivot + 1];
        let first_work = snapshot.workspace[pivot * APP_INNER_BLOCK_SIZE + row];
        let second_work = snapshot.workspace[(pivot + 1) * APP_INNER_BLOCK_SIZE + row];
        let source = d11 * first_work + d21 * second_work;
        let fma_second = d21.mul_add(second_work, d11 * first_work);
        let fma_first = d11.mul_add(first_work, d21 * second_work);
        let stored = snapshot.matrix[pivot * APP_INNER_BLOCK_SIZE + row];
        (
            source.to_bits(),
            fma_second.to_bits(),
            fma_first.to_bits(),
            stored.to_bits(),
        )
    }

    fn two_by_two_first_multiplier_block_expr_bits(
        snapshot: &BlockPrefixSnapshot,
        expected: &BlockLdltKernelResult,
        lda: usize,
    ) -> Vec<(usize, usize, u64, u64, u64, u64, u64)> {
        let pivot = snapshot.from;
        ((pivot + 2)..APP_INNER_BLOCK_SIZE)
            .map(|row| {
                let source_row = snapshot.local_perm[row];
                let final_row = expected
                    .local_perm
                    .iter()
                    .position(|&entry| entry == source_row)
                    .expect("snapshot source row exists in final block permutation");
                let (source, fma_second, fma_first, stored) =
                    two_by_two_first_multiplier_expression_bits(snapshot, row);
                let block = expected.matrix[pivot * lda + final_row].to_bits();
                (
                    row, source_row, source, fma_second, fma_first, stored, block,
                )
            })
            .collect()
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

    fn first_app_kernel_bit_mismatch(
        rust_matrix: &[f64],
        native_matrix: &[f64],
    ) -> Option<(usize, u64, u64)> {
        rust_matrix
            .iter()
            .zip(native_matrix)
            .enumerate()
            .find_map(|(index, (&rust, &native))| {
                let rust_bits = rust.to_bits();
                let native_bits = native.to_bits();
                (rust_bits != native_bits).then_some((index, rust_bits, native_bits))
            })
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

    #[test]
    fn app_packed_block_backup_restores_permuted_trailing_lower() {
        let size = 8;
        let backup_start = 2;
        let trailing_start = 4;
        let mut original = vec![0.0; size * size];
        for col in 0..size {
            for row in col..size {
                original[dense_lower_offset(size, row, col)] = (100 * col + row) as f64 + 0.25;
            }
        }

        let rows_before_block = (0..size).collect::<Vec<_>>();
        let rows = vec![0, 1, 5, 3, 2, 4, 6, 7];
        let backup = app_backup_trailing_lower(&original, size, backup_start);
        let mut matrix = original.clone();
        for row in trailing_start..size {
            for col in trailing_start..=row {
                matrix[dense_lower_offset(size, row, col)] = -1.0;
            }
        }

        app_restore_trailing_from_block_backup(
            &rows,
            &rows_before_block,
            &mut matrix,
            &backup,
            size,
            AppRestoreRange {
                backup_start,
                block_end: 6,
                trailing_start,
            },
        );

        for row in trailing_start..size {
            for col in trailing_start..=row {
                let old_row = rows[row];
                let old_col = rows[col];
                let expected = original[dense_lower_offset(size, old_row, old_col)];
                let actual = matrix[dense_lower_offset(size, row, col)];
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "restored value mismatch row={row} col={col} old_row={old_row} old_col={old_col}"
                );
            }
        }
    }

    #[test]
    fn app_block_backup_skips_restore_after_full_pass() {
        let size = 8;
        let backup_start = 2;
        let block_end = 4;
        let trailing_start = block_end;
        let mut original = vec![0.0; size * size];
        for col in 0..size {
            for row in col..size {
                original[dense_lower_offset(size, row, col)] = (10 * col + row) as f64;
            }
        }
        let backup = app_backup_trailing_lower(&original, size, backup_start);
        let rows = (0..size).collect::<Vec<_>>();
        let mut matrix = original.clone();
        matrix[dense_lower_offset(size, trailing_start, trailing_start)] = -99.0;

        app_restore_trailing_from_block_backup(
            &rows,
            &rows,
            &mut matrix,
            &backup,
            size,
            AppRestoreRange {
                backup_start,
                block_end,
                trailing_start,
            },
        );

        assert_eq!(
            matrix[dense_lower_offset(size, trailing_start, trailing_start)].to_bits(),
            (-99.0f64).to_bits(),
            "full-pass APP restore should be a no-op"
        );
    }

    #[test]
    fn dense_front_solve_panel_record_matches_generic_factor_columns() {
        let size = 5;
        let eliminated_len = 3;
        let rows = vec![3, 1, 4, 0, 2];
        let mut dense = vec![0.0; size * size];
        for col in 0..size {
            for row in col..size {
                dense[dense_lower_offset(size, row, col)] =
                    ((17 * (col + 1) + 5 * (row + 3)) as f64) / 13.0;
            }
        }

        let factor_columns =
            app_build_factor_columns_for_prefix(&rows, &dense, size, 0, eliminated_len);
        let generic = build_factor_solve_panel_record(
            &rows[..eliminated_len],
            &factor_columns,
            &rows[eliminated_len..],
        )
        .unwrap()
        .unwrap();
        let direct = build_dense_front_solve_panel_record(
            &rows[..eliminated_len],
            &rows[eliminated_len..],
            &dense,
            size,
            eliminated_len,
        )
        .unwrap()
        .unwrap();

        assert_eq!(direct, generic);
    }

    fn first_block_prefix_trace_mismatch(
        rust: &[BlockPrefixSnapshot],
        native: &[BlockPrefixSnapshot],
    ) -> Option<BlockPrefixTraceMismatch> {
        for (rust_step, native_step) in rust.iter().zip(native) {
            if (
                rust_step.step,
                rust_step.from,
                rust_step.status,
                rust_step.next,
            ) != (
                native_step.step,
                native_step.from,
                native_step.status,
                native_step.next,
            ) {
                return Some(BlockPrefixTraceMismatch {
                    step: rust_step.step,
                    from: rust_step.from,
                    status: rust_step.status,
                    next: rust_step.next,
                    component: "status",
                    index: 0,
                    row: 0,
                    col: 0,
                    rust_bits: 0,
                    native_bits: 0,
                });
            }
            if rust_step.perm != native_step.perm {
                return Some(BlockPrefixTraceMismatch {
                    step: rust_step.step,
                    from: rust_step.from,
                    status: rust_step.status,
                    next: rust_step.next,
                    component: "perm",
                    index: 0,
                    row: 0,
                    col: 0,
                    rust_bits: 0,
                    native_bits: 0,
                });
            }
            if rust_step.local_perm != native_step.local_perm {
                return Some(BlockPrefixTraceMismatch {
                    step: rust_step.step,
                    from: rust_step.from,
                    status: rust_step.status,
                    next: rust_step.next,
                    component: "local_perm",
                    index: 0,
                    row: 0,
                    col: 0,
                    rust_bits: 0,
                    native_bits: 0,
                });
            }
            for (index, (&rust_value, &native_value)) in rust_step
                .diagonal
                .iter()
                .zip(&native_step.diagonal)
                .enumerate()
            {
                if rust_value.to_bits() != native_value.to_bits() {
                    return Some(BlockPrefixTraceMismatch {
                        step: rust_step.step,
                        from: rust_step.from,
                        status: rust_step.status,
                        next: rust_step.next,
                        component: "diagonal",
                        index,
                        row: 0,
                        col: 0,
                        rust_bits: rust_value.to_bits(),
                        native_bits: native_value.to_bits(),
                    });
                }
            }
            for col in 0..APP_INNER_BLOCK_SIZE {
                for row in col..APP_INNER_BLOCK_SIZE {
                    let index = row * APP_INNER_BLOCK_SIZE + col;
                    let rust_value = rust_step.matrix[col * APP_INNER_BLOCK_SIZE + row];
                    let native_value = native_step.matrix[col * APP_INNER_BLOCK_SIZE + row];
                    if rust_value.to_bits() != native_value.to_bits() {
                        return Some(BlockPrefixTraceMismatch {
                            step: rust_step.step,
                            from: rust_step.from,
                            status: rust_step.status,
                            next: rust_step.next,
                            component: "matrix",
                            index,
                            row,
                            col,
                            rust_bits: rust_value.to_bits(),
                            native_bits: native_value.to_bits(),
                        });
                    }
                }
            }
            for (index, (&rust_value, &native_value)) in rust_step
                .workspace
                .iter()
                .zip(&native_step.workspace)
                .enumerate()
            {
                if rust_value.to_bits() != native_value.to_bits() {
                    return Some(BlockPrefixTraceMismatch {
                        step: rust_step.step,
                        from: rust_step.from,
                        status: rust_step.status,
                        next: rust_step.next,
                        component: "workspace",
                        index,
                        row: index % APP_INNER_BLOCK_SIZE,
                        col: index / APP_INNER_BLOCK_SIZE,
                        rust_bits: rust_value.to_bits(),
                        native_bits: native_value.to_bits(),
                    });
                }
            }
        }
        (rust.len() != native.len()).then_some(BlockPrefixTraceMismatch {
            step: rust.len().min(native.len()),
            from: 0,
            status: 0,
            next: 0,
            component: "len",
            index: 0,
            row: 0,
            col: 0,
            rust_bits: rust.len() as u64,
            native_bits: native.len() as u64,
        })
    }

    #[test]
    fn permuted_symmetric_scaling_uses_spral_multiplication_order() {
        let col_ptrs = [0, 1, 1];
        let row_indices = [1];
        let original_value = f64::from_bits(0x3fd8_b332_7756_0eaf);
        let col_scale = f64::from_bits(0x3fc9_5124_5767_cd7a);
        let row_scale = f64::from_bits(0x3f52_62eb_bdd2_832b);
        let mut values = [original_value];
        let product_first_order = original_value * (row_scale * col_scale);

        apply_permuted_symmetric_scaling(
            &col_ptrs,
            &row_indices,
            &mut values,
            &[col_scale, row_scale],
        )
        .unwrap();

        let spral_order = row_scale * original_value * col_scale;
        assert_eq!(values[0].to_bits(), spral_order.to_bits());
        assert_ne!(values[0].to_bits(), product_first_order.to_bits());
    }

    #[test]
    fn fused_scaled_permuted_values_match_fill_then_scale_bits() {
        let dense = vec![
            vec![f64::from_bits(0x3ff0_0000_0000_0001), 0.0, 0.0, 0.0],
            vec![
                f64::from_bits(0xbfd8_0000_0000_0003),
                f64::from_bits(0x3fe8_0000_0000_0005),
                0.0,
                0.0,
            ],
            vec![
                f64::from_bits(0x3fb9_9999_9999_999a),
                f64::from_bits(0xbfc4_0000_0000_0007),
                f64::from_bits(0x3ff8_0000_0000_000b),
                0.0,
            ],
            vec![
                f64::from_bits(0xbfa0_0000_0000_000d),
                f64::from_bits(0x3fd0_0000_0000_000f),
                f64::from_bits(0xbfe0_0000_0000_0011),
                f64::from_bits(0x3fc8_0000_0000_0013),
            ],
        ];
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let matrix = SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values))
            .expect("valid matrix");
        let permutation = Permutation::new(vec![2, 0, 3, 1]).expect("valid permutation");
        let scaling = [
            f64::from_bits(0x3fb9_5124_5767_cd7a),
            f64::from_bits(0x3fc2_62eb_bdd2_832b),
            f64::from_bits(0x3fd1_f3b6_45a1_c935),
            f64::from_bits(0x3fc8_b332_7756_0eaf),
        ];

        let mut permuted_col_ptrs = Vec::new();
        let mut permuted_row_indices = Vec::new();
        let mut source_positions = Vec::new();
        build_permuted_lower_csc_pattern(
            matrix,
            &permutation,
            &mut permuted_col_ptrs,
            &mut permuted_row_indices,
            &mut source_positions,
        )
        .expect("permuted pattern");

        let mut separate = Vec::new();
        fill_permuted_lower_csc_values(matrix, &source_positions, &mut separate)
            .expect("fill values");
        apply_permuted_symmetric_scaling(
            &permuted_col_ptrs,
            &permuted_row_indices,
            &mut separate,
            &scaling,
        )
        .expect("apply scaling");

        let mut fused = Vec::new();
        fill_scaled_permuted_lower_csc_values(
            matrix,
            &permuted_col_ptrs,
            &permuted_row_indices,
            &source_positions,
            &scaling,
            &mut fused,
        )
        .expect("fused fill");

        assert_eq!(
            fused
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            separate
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
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
    fn app_apply_pivot_and_host_trsm_signed_zero_boundaries_are_complementary() {
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
        let mut rust_trsm_matrix = case.matrix.clone();
        let mut native_trsm_matrix = case.matrix.clone();
        let mut rust_apply_matrix = case.matrix.clone();
        let mut native_apply_matrix = case.matrix.clone();

        native_host_trsm_op_n(shim, &case, &mut native_trsm_matrix);
        app_solve_block_triangular_to_trailing_rows(
            &mut rust_trsm_matrix,
            case.size,
            case.block_start,
            case.block_end,
            false,
        );

        native_apply_pivot_op_n(shim, &case, &mut native_apply_matrix);
        app_apply_block_pivots_to_trailing_rows(
            &mut rust_apply_matrix,
            case.size,
            case.block_start,
            case.block_end,
            &case.block_records,
            case.small,
            false,
        );

        let host_trsm_mismatch =
            first_app_kernel_bit_mismatch(&rust_trsm_matrix, &native_trsm_matrix);
        let apply_pivot_mismatch =
            first_app_kernel_bit_mismatch(&rust_apply_matrix, &native_apply_matrix);

        assert_eq!(
            host_trsm_mismatch,
            Some((82, 0x0000_0000_0000_0000, 0x8000_0000_0000_0000)),
            "host_trsm OP_N signed-zero boundary moved for case={case:?}"
        );
        assert_eq!(
            apply_pivot_mismatch,
            Some((82, 0x8000_0000_0000_0000, 0x0000_0000_0000_0000)),
            "apply_pivot OP_N signed-zero boundary moved for case={case:?}"
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
    fn app_block_first_step_two_by_two_wide_rows_match_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0x2522_900d_0001);
        let mut runner = deterministic_kernel_runner(cases, seed);

        runner
            .run(&any::<u64>(), |case_seed| {
                let case = block_first_step_case_from_seed(seed ^ case_seed, true);
                // The local native first-step shim models the vectorized
                // all-FMA path. Narrow and odd scalar-tail coverage is pinned
                // by the seed6/seed09 exact APP witnesses below.
                let trailing_rows = APP_INNER_BLOCK_SIZE.saturating_sub(case.from + 2);
                if trailing_rows < 4 || !trailing_rows.is_multiple_of(2) {
                    return Ok(());
                }
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
    fn app_block_source_test_2x2_matches_native_kernel_property_cases() {
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
                let rust_inverse = app_two_by_two_inverse_source_test_2x2(a11, a21, a22, 0.0);

                prop_assert_eq!(
                    rust_inverse.is_some(),
                    native_accepted,
                    "block_ldlt source test_2x2 acceptance mismatch seed={:#x} a11={:?} a21={:?} a22={:?} native_detpiv={:?} native_detscale={:?}",
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
                        "block_ldlt source 2x2 d11 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
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
                        "block_ldlt source 2x2 d21 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
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
                        "block_ldlt source 2x2 d22 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
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
            .expect("block_ldlt source test_2x2 kernel parity property failed");
    }

    #[test]
    fn app_block_full_codegen_test_2x2_matches_native_kernel_property_cases() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let cases = env_usize("SPRAL_SSIDS_KERNEL_PARITY_CASES", 512);
        let seed = env_u64("SPRAL_SSIDS_KERNEL_PARITY_SEED", 0xb221_900d_0002);
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
                    native_block_test_2x2_full_block_codegen(shim, a11, a21, a22);
                let rust_inverse = app_two_by_two_inverse(a11, a21, a22, 0.0);

                prop_assert_eq!(
                    rust_inverse.is_some(),
                    native_accepted,
                    "block_ldlt full-codegen test_2x2 acceptance mismatch seed={:#x} a11={:?} a21={:?} a22={:?} native_detpiv={:?} native_detscale={:?}",
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
                        "block_ldlt full-codegen 2x2 d11 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
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
                        "block_ldlt full-codegen 2x2 d21 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
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
                        "block_ldlt full-codegen 2x2 d22 mismatch seed={:#x} values=({:?}, {:?}, {:?}) native_detpiv={:?} native_detscale={:?}",
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
            .expect("block_ldlt full-codegen test_2x2 kernel parity property failed");
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
    fn bitset_permute_graph_matches_sorted_edge_path() {
        let edges = vec![
            (0, 1),
            (0, 4),
            (1, 2),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (5, 6),
            (6, 7),
            (0, 7),
        ];
        let graph = CsrGraph::from_edges(8, &edges).expect("valid graph");
        let permutation =
            Permutation::new(vec![6, 2, 7, 1, 4, 0, 5, 3]).expect("valid permutation");

        assert_eq!(
            permute_graph_with_bitsets(&graph, &permutation),
            permute_graph_with_sorted_edges(&graph, &permutation)
        );
        assert_eq!(
            permute_graph(&graph, &permutation),
            permute_graph_with_sorted_edges(&graph, &permutation)
        );
    }

    #[test]
    fn fast_single_supernode_row_list_matches_generic_native_row_list_path() {
        let dense = vec![
            vec![4.0, 1.0, 2.0, 3.0],
            vec![1.0, 5.0, 4.0, 6.0],
            vec![2.0, 4.0, 6.0, 7.0],
            vec![3.0, 6.0, 7.0, 8.0],
        ];
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let matrix =
            SymmetricCscMatrix::new(4, &col_ptrs, &row_indices, Some(&values)).expect("valid CSC");
        let graph = CsrGraph::from_symmetric_csc(4, &col_ptrs, &row_indices).expect("valid graph");
        let permutation = Permutation::identity(4);
        let permuted_graph = permute_graph(&graph, &permutation);
        let (elimination_tree, column_counts, column_pattern) =
            symbolic_factor_pattern(&permuted_graph);
        let layout = native_supernode_layout(&elimination_tree, &column_counts, 4);
        assert_eq!(layout.ranges, vec![0..4]);

        let expanded_pattern = expand_symmetric_pattern(matrix);
        let generic = build_native_row_list_supernodes(
            &expanded_pattern,
            &permutation,
            &layout,
            &column_pattern,
        );
        let fast = build_native_row_list_supernodes_fast(&layout, &column_pattern)
            .expect("single full supernode fast path");

        assert_eq!(fast, generic);
    }

    #[test]
    fn app_gemv_forward_singleton_column_matches_source_tile_predicate() {
        for size in 1..96 {
            for accepted_end in 0..size {
                let singleton = app_gemv_forward_singleton_column(size, accepted_end);
                for col in accepted_end..size {
                    assert_eq!(
                        singleton == Some(col),
                        app_target_block_uses_gemv_forward(size, accepted_end, col),
                        "size={size} accepted_end={accepted_end} col={col}"
                    );
                }
            }
        }
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

    fn random_dyadic_solution(dimension: usize, rng: &mut DenseBoundaryRng) -> Vec<f64> {
        (0..dimension).map(|_| rng.dyadic(8, 4)).collect()
    }

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

    fn dense_seed6_33_matrix_and_solution() -> (usize, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = DenseBoundaryRng::new(6);
        let dimension = rng.usize_inclusive(33, 33);
        assert_eq!(dimension, 33);
        let matrix = random_dense_dyadic_matrix(dimension, &mut rng);
        let solution = random_dyadic_solution(dimension, &mut rng);
        (dimension, matrix, solution)
    }

    fn dense_seed1001_33_matrix() -> (usize, Vec<Vec<f64>>) {
        let mut rng = DenseBoundaryRng::new(0x1001);
        let dimension = rng.usize_inclusive(33, 33);
        assert_eq!(dimension, 33);
        (dimension, random_dense_dyadic_matrix(dimension, &mut rng))
    }

    fn dense_seed1001_33_matrix_and_solution() -> (usize, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = DenseBoundaryRng::new(0x1001);
        let dimension = rng.usize_inclusive(33, 33);
        assert_eq!(dimension, 33);
        let matrix = random_dense_dyadic_matrix(dimension, &mut rng);
        let solution = random_dyadic_solution(dimension, &mut rng);
        (dimension, matrix, solution)
    }

    fn dense_seed09_case0_matrix() -> (usize, Vec<Vec<f64>>) {
        let mut rng = DenseBoundaryRng::new(0x09c9_134e_4eff_0004);
        let dimension = rng.usize_inclusive(33, 160);
        assert_eq!(dimension, 55);
        (dimension, random_dense_dyadic_matrix(dimension, &mut rng))
    }

    fn dense_boundary_case_matrix_and_solution(
        seed: u64,
        case_index: usize,
    ) -> (usize, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = DenseBoundaryRng::new(seed);
        let mut dimension = 0;
        let mut matrix = Vec::new();
        let mut solution = Vec::new();
        for _ in 0..=case_index {
            dimension = rng.usize_inclusive(33, 160);
            matrix = random_dense_dyadic_matrix(dimension, &mut rng);
            solution = random_dyadic_solution(dimension, &mut rng);
        }
        (dimension, matrix, solution)
    }

    fn dense_seed706172697479_case58_matrix_and_solution() -> (usize, Vec<Vec<f64>>, Vec<f64>) {
        let (dimension, matrix, solution) =
            dense_boundary_case_matrix_and_solution(0x7061_7269_7479, 58);
        assert_eq!(dimension, 137);
        (dimension, matrix, solution)
    }

    fn first_bit_mismatch(left: &[f64], right: &[f64]) -> Option<(usize, u64, u64)> {
        left.iter()
            .zip(right)
            .enumerate()
            .find_map(|(index, (&left, &right))| {
                let left_bits = left.to_bits();
                let right_bits = right.to_bits();
                (left_bits != right_bits).then_some((index, left_bits, right_bits))
            })
    }

    #[derive(Debug)]
    struct AppBlockReplay {
        rows: Vec<usize>,
        after_factor: Vec<f64>,
        after_apply: Vec<f64>,
        after_update: Vec<f64>,
        restored: Vec<f64>,
        accepted_end: usize,
        first_failed: usize,
        local_blocks: Vec<FactorBlockRecord>,
        accepted_blocks: Vec<FactorBlockRecord>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct AppAcceptedUpdateMismatch {
        phase: &'static str,
        row: usize,
        col: usize,
        rust_bits: u64,
        native_bits: u64,
    }

    fn replay_app_block_for_debug(
        mut rows: Vec<usize>,
        mut lower_dense: Vec<f64>,
        block_start: usize,
        options: NumericFactorOptions,
    ) -> AppBlockReplay {
        let size = rows.len();
        let block_end = (block_start + APP_INNER_BLOCK_SIZE).min(size);
        assert_eq!(
            block_end - block_start,
            APP_INNER_BLOCK_SIZE,
            "debug APP replay expects a full APP block"
        );
        let before_block = app_backup_trailing_lower(&lower_dense, size, block_start);
        let rows_before_block = rows.clone();
        let mut scratch = vec![0.0; size.saturating_mul(size).max(1)];
        let mut local_stats = PanelFactorStats::default();
        let mut local_blocks = Vec::new();
        let mut block_pivot = block_start;

        while block_pivot < block_end {
            let Some((best_abs, best_row, best_col)) =
                dense_find_maxloc(&lower_dense, size, block_pivot, block_end)
            else {
                break;
            };
            assert!(
                best_abs >= options.small_pivot_tolerance,
                "debug APP replay hit a zero-pivot branch"
            );
            if best_row == best_col {
                if best_col != block_pivot {
                    dense_symmetric_swap_with_workspace(
                        &mut lower_dense,
                        size,
                        best_col,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(best_col, block_pivot);
                }
                let block = factor_one_by_one_common(
                    &rows,
                    &mut lower_dense,
                    size,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    &mut scratch,
                )
                .expect("debug APP 1x1 pivot");
                local_blocks.push(block);
                block_pivot += 1;
                continue;
            }

            let first = best_col;
            let mut second = best_row;
            let a11 = lower_dense[dense_lower_offset(size, first, first)];
            let a22 = lower_dense[dense_lower_offset(size, second, second)];
            let a21 = lower_dense[dense_lower_offset(size, second, first)];
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
                        size,
                        index,
                        block_pivot,
                        &mut scratch,
                    );
                    rows.swap(index, block_pivot);
                }
                let block = factor_one_by_one_common(
                    &rows,
                    &mut lower_dense,
                    size,
                    block_pivot,
                    block_end,
                    &mut local_stats,
                    &mut scratch,
                )
                .expect("debug APP fallback 1x1 pivot");
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
                    &mut lower_dense,
                    size,
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
                    size,
                    update_end: block_end,
                },
                block_pivot,
                inverse,
                &mut local_stats,
                &mut scratch,
            )
            .expect("debug APP 2x2 pivot");
            local_blocks.push(block);
            block_pivot += 2;
        }

        let after_factor = lower_dense.clone();
        app_apply_block_pivots_to_trailing_rows(
            &mut lower_dense,
            size,
            block_start,
            block_end,
            &local_blocks,
            options.small_pivot_tolerance,
            false,
        );
        let after_apply = lower_dense.clone();
        let first_failed = app_first_failed_trailing_column(
            &lower_dense,
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
            &mut lower_dense,
            &before_block,
            size,
            AppRestoreRange {
                backup_start: block_start,
                block_end,
                trailing_start: accepted_end,
            },
        );
        let restored = lower_dense.clone();
        app_apply_accepted_prefix_update(
            &mut lower_dense,
            size,
            block_start,
            accepted_end,
            &accepted_blocks,
        );

        AppBlockReplay {
            rows,
            after_factor,
            after_apply,
            after_update: lower_dense,
            restored,
            accepted_end,
            first_failed,
            local_blocks,
            accepted_blocks,
        }
    }

    fn first_native_apply_pivot_tile_mismatch(
        shim: &NativeKernelShim,
        replay: &AppBlockReplay,
        block_start: usize,
    ) -> Option<AppAcceptedUpdateMismatch> {
        let size = replay.rows.len();
        let block_end = block_start + APP_INNER_BLOCK_SIZE;
        let block_width = block_end - block_start;
        let native_lda = native_aligned_double_stride(shim, size);
        let diagonal = dense_tpp_diagonal_from_blocks(&replay.local_blocks, block_width);
        let mut row_tile_start = block_end;
        while row_tile_start < size {
            let row_tile_end = (row_tile_start + APP_INNER_BLOCK_SIZE).min(size);
            let tile_rows = row_tile_end - row_tile_start;
            let mut native_matrix = vec![0.0; native_lda * size];
            for col in block_start..block_end {
                for row in col..block_end {
                    native_matrix[col * native_lda + row] = replay.after_factor[col * size + row];
                }
                for row in row_tile_start..row_tile_end {
                    native_matrix[col * native_lda + row] = replay.after_factor[col * size + row];
                }
            }
            unsafe {
                (shim.apply_pivot_op_n)(
                    tile_rows as c_int,
                    block_width as c_int,
                    native_matrix
                        .as_ptr()
                        .add(block_start * native_lda + block_start),
                    diagonal.as_ptr(),
                    NumericFactorOptions::default().small_pivot_tolerance,
                    native_matrix
                        .as_mut_ptr()
                        .add(block_start * native_lda + row_tile_start),
                    native_lda as c_int,
                );
            }
            for col in block_start..block_end {
                for row in row_tile_start..row_tile_end {
                    let rust = replay.after_apply[col * size + row];
                    let native = native_matrix[col * native_lda + row];
                    if rust.to_bits() != native.to_bits() {
                        return Some(AppAcceptedUpdateMismatch {
                            phase: "apply_pivot_op_n_tile",
                            row,
                            col,
                            rust_bits: rust.to_bits(),
                            native_bits: native.to_bits(),
                        });
                    }
                }
            }
            row_tile_start += APP_INNER_BLOCK_SIZE;
        }
        None
    }

    fn first_native_accepted_update_mismatch(
        shim: &NativeKernelShim,
        replay: &AppBlockReplay,
        block_start: usize,
    ) -> Option<AppAcceptedUpdateMismatch> {
        let size = replay.rows.len();
        let accepted_width = replay.accepted_end - block_start;
        let tail_size = size - replay.accepted_end;
        if accepted_width == 0 || tail_size == 0 {
            return None;
        }
        let diagonal = dense_tpp_diagonal_from_blocks(&replay.accepted_blocks, accepted_width);
        let native_lda = native_aligned_double_stride(shim, size);
        let native_ldld = native_aligned_double_stride(shim, APP_INNER_BLOCK_SIZE);
        let mut col_tile_start = replay.accepted_end;
        while col_tile_start < size {
            let col_tile_end = (col_tile_start + APP_INNER_BLOCK_SIZE).min(size);
            let target_width = col_tile_end - col_tile_start;
            let mut row_tile_start = col_tile_start;
            while row_tile_start < size {
                let row_tile_end = (row_tile_start + APP_INNER_BLOCK_SIZE).min(size);
                let tile_rows = row_tile_end - row_tile_start;
                let mut l_block = vec![0.0; accepted_width * native_lda];
                for local_col in 0..accepted_width {
                    let source_col = block_start + local_col;
                    for local_row in 0..tile_rows {
                        l_block[local_col * native_lda + local_row] =
                            replay.after_update[source_col * size + row_tile_start + local_row];
                    }
                }
                let mut native_ld = vec![0.0; accepted_width * native_ldld];
                unsafe {
                    (shim.calc_ld_op_n)(
                        tile_rows as c_int,
                        accepted_width as c_int,
                        l_block.as_ptr(),
                        native_lda as c_int,
                        diagonal.as_ptr(),
                        native_ld.as_mut_ptr(),
                        native_ldld as c_int,
                    );
                }
                let rust_ld = app_build_ld_tile_workspace(
                    &replay.after_update,
                    size,
                    block_start,
                    replay.accepted_end,
                    &replay.accepted_blocks,
                    row_tile_start,
                    row_tile_end,
                );
                for local_col in 0..accepted_width {
                    for local_row in 0..tile_rows {
                        let rust = rust_ld[local_col * tile_rows + local_row];
                        let native = native_ld[local_col * native_ldld + local_row];
                        if rust.to_bits() != native.to_bits() {
                            return Some(AppAcceptedUpdateMismatch {
                                phase: "calc_ld_op_n_tile",
                                row: row_tile_start + local_row,
                                col: block_start + local_col,
                                rust_bits: rust.to_bits(),
                                native_bits: native.to_bits(),
                            });
                        }
                    }
                }

                let mut col_l_block = vec![0.0; accepted_width * native_lda];
                for local_pivot in 0..accepted_width {
                    let source_col = block_start + local_pivot;
                    for local_col in 0..target_width {
                        col_l_block[local_pivot * native_lda + local_col] =
                            replay.after_update[source_col * size + col_tile_start + local_col];
                    }
                }
                let mut native_target = vec![0.0; target_width * native_lda];
                for local_col in 0..target_width {
                    let col = col_tile_start + local_col;
                    for local_row in 0..tile_rows {
                        let row = row_tile_start + local_row;
                        if row >= col {
                            native_target[local_col * native_lda + local_row] =
                                replay.restored[col * size + row];
                        }
                    }
                }
                unsafe {
                    (shim.host_gemm_op_n_op_t_update)(
                        tile_rows as c_int,
                        target_width as c_int,
                        accepted_width as c_int,
                        native_ld.as_ptr(),
                        native_ldld as c_int,
                        col_l_block.as_ptr(),
                        native_lda as c_int,
                        native_target.as_mut_ptr(),
                        native_lda as c_int,
                    );
                }
                for local_col in 0..target_width {
                    let col = col_tile_start + local_col;
                    let first_row = if row_tile_start == col_tile_start {
                        col
                    } else {
                        row_tile_start
                    };
                    for row in first_row..row_tile_end {
                        let local_row = row - row_tile_start;
                        let rust = replay.after_update[col * size + row];
                        let native = native_target[local_col * native_lda + local_row];
                        if rust.to_bits() != native.to_bits() {
                            return Some(AppAcceptedUpdateMismatch {
                                phase: "host_gemm_op_n_op_t_tile",
                                row,
                                col,
                                rust_bits: rust.to_bits(),
                                native_bits: native.to_bits(),
                            });
                        }
                    }
                }
                row_tile_start += APP_INNER_BLOCK_SIZE;
            }
            col_tile_start += APP_INNER_BLOCK_SIZE;
        }
        None
    }

    fn assert_first_app_accepted_update_matches_native_host_gemm_tiles(
        shim: &NativeKernelShim,
        seed: u64,
        case_index: usize,
        expected_dimension: usize,
    ) {
        let (dimension, dense, _) = dense_boundary_case_matrix_and_solution(seed, case_index);
        assert_eq!(dimension, expected_dimension);
        let mut lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for row in col..dimension {
                lower_dense[dense_lower_offset(dimension, row, col)] = dense[row][col];
            }
        }

        let options = NumericFactorOptions::default();
        let replay = replay_app_block_for_debug((0..dimension).collect(), lower_dense, 0, options);
        assert!(
            replay.accepted_end > 0 && replay.accepted_end < dimension,
            "fixture must exercise accepted-prefix trailing update"
        );

        // This pins the same source boundary as
        // ldlt_app.cxx::Block::update: calcLD<OP_N> followed by
        // host_gemm(OP_N, OP_T) over APP row/column tiles.
        assert_eq!(
            first_native_accepted_update_mismatch(shim, &replay, 0),
            None,
            "dense seed={seed:#x} case={case_index} accepted APP update diverged from native tiled calcLD/host_gemm"
        );
    }

    #[test]
    fn app_accepted_update_dense_witnesses_match_native_host_gemm_tiles() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };

        assert_first_app_accepted_update_matches_native_host_gemm_tiles(
            shim,
            0x7061_7269_7479,
            58,
            137,
        );
        assert_first_app_accepted_update_matches_native_host_gemm_tiles(
            shim,
            0x7061_7269_7479_2026,
            59,
            160,
        );
    }

    fn assert_no_bit_mismatch(left: &[f64], right: &[f64], label: &str) {
        assert_eq!(
            first_bit_mismatch(left, right),
            None,
            "{label} first bit mismatch"
        );
    }

    fn deterministic_complete_dyadic_matrix(dimension: usize) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; dimension]; dimension];
        let mut row = 0;
        while row < dimension {
            let mut col = 0;
            while col <= row {
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
                col += 1;
            }
            row += 1;
        }
        matrix
    }

    fn deterministic_complete_dyadic_solution(dimension: usize) -> Vec<f64> {
        (0..dimension)
            .map(|index| f64::from((index % 11) as i16 - 5) / 8.0)
            .collect()
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

    fn lower_dense_seed1001_33() -> (usize, Vec<f64>) {
        let (dimension, dense) = dense_seed1001_33_matrix();
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

    fn bit_patterns(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
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
        assert_eq!(
            first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
            None,
            "dense seed6 APP prefix trace mismatch"
        );
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
        assert_eq!(
            first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
            None,
            "dense seed6 aligned APP prefix trace mismatch"
        );
    }

    #[test]
    fn app_block_ldlt_32_prefix_trace_matches_native_dense_seed1001() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed1001_33();
        let native_lda = native_aligned_double_stride(shim, dimension);
        let options = NumericFactorOptions::default();
        let rust_trace = rust_app_block_prefix_trace_32(&lower_dense, dimension, options);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, native_lda, options);
        assert_eq!(
            first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
            None,
            "dense seed1001 APP prefix trace mismatch"
        );
    }

    #[test]
    fn dense_seed1001_app_block_storage_matches_native() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed1001_33();
        let options = NumericFactorOptions::default();
        let native_lda = native_aligned_double_stride(shim, dimension);

        let rust = rust_block_ldlt_32_from_lower_dense(&lower_dense, dimension, options);
        let native = native_block_ldlt_32_from_lower_dense(
            shim,
            &lower_dense,
            dimension,
            native_lda,
            options,
        );
        assert_eq!(rust.perm, native.perm);
        assert_eq!(rust.local_perm, native.local_perm);
        let first_diagonal_mismatch = rust
            .diagonal
            .iter()
            .zip(&native.diagonal)
            .enumerate()
            .find_map(|(index, (&rust_value, &native_value))| {
                (rust_value.to_bits() != native_value.to_bits()).then_some((
                    index,
                    rust_value.to_bits(),
                    native_value.to_bits(),
                ))
            });
        assert_eq!(
            first_diagonal_mismatch, None,
            "dense seed1001 APP block D mismatch"
        );

        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, native_lda, options);
        assert_eq!(
            first_native_block_continuation_mismatch(
                shim,
                &native_trace,
                &native,
                native_lda,
                options,
            ),
            None,
            "dense seed1001 native APP continuation mismatch"
        );

        let mut first_matrix_mismatch = None;
        'matrix_compare: for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                let rust_bits = rust.matrix[col * dimension + row].to_bits();
                let native_bits = native.matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_matrix_mismatch = Some((row, col, rust_bits, native_bits));
                    break 'matrix_compare;
                }
            }
        }
        assert_eq!(
            first_matrix_mismatch, None,
            "dense seed1001 APP block L-storage mismatch"
        );
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
        assert_eq!(
            first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
            None,
            "dense seed09 aligned APP prefix trace mismatch"
        );
    }

    #[test]
    fn dense_seed6_app_block_storage_matches_native() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed6_33();
        let options = NumericFactorOptions::default();
        let native_lda = native_aligned_double_stride(shim, dimension);
        assert_eq!(native_lda, 34);

        let rust_trace = rust_app_block_prefix_trace_32(&lower_dense, dimension, options);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, native_lda, options);
        assert_eq!(
            first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
            None,
            "dense seed6 APP prefix trace mismatch"
        );

        let rust = rust_block_ldlt_32_from_lower_dense(&lower_dense, dimension, options);
        let native = native_block_ldlt_32_from_lower_dense(
            shim,
            &lower_dense,
            dimension,
            native_lda,
            options,
        );
        assert_eq!(rust.perm, native.perm);
        assert_eq!(rust.local_perm, native.local_perm);
        for (index, (&rust_value, &native_value)) in
            rust.diagonal.iter().zip(&native.diagonal).enumerate()
        {
            assert_eq!(
                rust_value.to_bits(),
                native_value.to_bits(),
                "dense seed6 APP block D mismatch after matching prefix trace index={index}"
            );
        }

        let mut first_mismatch = None;
        'matrix_compare: for col in 0..APP_INNER_BLOCK_SIZE {
            for row in col..APP_INNER_BLOCK_SIZE {
                let rust_bits = rust.matrix[col * dimension + row].to_bits();
                let native_bits = native.matrix[col * native_lda + row].to_bits();
                if rust_bits != native_bits {
                    first_mismatch = Some((row, col, rust_bits, native_bits));
                    break 'matrix_compare;
                }
            }
        }
        assert_eq!(
            first_mismatch, None,
            "dense seed6 APP block L-storage mismatch"
        );
    }

    #[test]
    fn dense_seed6_native_block_continuation_matches_full_block() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed6_33();
        let options = NumericFactorOptions::default();
        let native_lda = native_aligned_double_stride(shim, dimension);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, native_lda, options);
        let native_block = native_block_ldlt_32_from_lower_dense(
            shim,
            &lower_dense,
            dimension,
            native_lda,
            options,
        );

        assert_eq!(
            first_native_block_continuation_mismatch(
                shim,
                &native_trace,
                &native_block,
                native_lda,
                options,
            ),
            None,
            "dense seed6 native APP continuation mismatch"
        );
    }

    #[test]
    fn dense_seed6_final_app_pivot_source_expression_matches_native_block() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, lower_dense) = lower_dense_seed6_33();
        let options = NumericFactorOptions::default();
        let native_lda = native_aligned_double_stride(shim, dimension);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, native_lda, options);
        let native_block = native_block_ldlt_32_from_lower_dense(
            shim,
            &lower_dense,
            dimension,
            native_lda,
            options,
        );

        let fma_pivot_snapshot = native_trace
            .iter()
            .find(|snapshot| snapshot.step == 14 && snapshot.from == 28 && snapshot.status == 2)
            .expect("dense seed6 FMA pivot-28 snapshot");
        let multiplier_col = fma_pivot_snapshot.from;
        let final_multiplier_row = 30;
        let final_multiplier_source = native_block.local_perm[final_multiplier_row];
        let multiplier_row = fma_pivot_snapshot
            .local_perm
            .iter()
            .position(|&entry| entry == final_multiplier_source)
            .expect("dense seed6 FMA pivot-28 final multiplier row source");
        assert_eq!(
            multiplier_row, 30,
            "dense seed6 FMA pivot-28 source row moved"
        );
        assert_eq!(multiplier_col + 2, fma_pivot_snapshot.next);
        assert!(multiplier_row >= fma_pivot_snapshot.next);
        let d11 = fma_pivot_snapshot.diagonal[2 * multiplier_col];
        let d21 = fma_pivot_snapshot.diagonal[2 * multiplier_col + 1];
        let first_work =
            fma_pivot_snapshot.workspace[multiplier_col * APP_INNER_BLOCK_SIZE + multiplier_row];
        let second_work = fma_pivot_snapshot.workspace
            [(multiplier_col + 1) * APP_INNER_BLOCK_SIZE + multiplier_row];
        let reconstructed_first_multiplier = d11.mul_add(first_work, d21 * second_work);
        let source_first_multiplier = d11 * first_work + d21 * second_work;
        let trace_first_multiplier =
            fma_pivot_snapshot.matrix[multiplier_col * APP_INNER_BLOCK_SIZE + multiplier_row];
        let native_block_first_multiplier =
            native_block.matrix[multiplier_col * native_lda + final_multiplier_row];
        assert_eq!(
            reconstructed_first_multiplier.to_bits(),
            trace_first_multiplier.to_bits(),
            "dense seed6 FMA pivot-28 first-row multiplier reconstruction moved"
        );
        assert_eq!(
            (
                trace_first_multiplier.to_bits(),
                source_first_multiplier.to_bits(),
                native_block_first_multiplier.to_bits()
            ),
            (
                0xbf81_6117_c4f8_2730,
                0xbf81_6117_c4f8_2730,
                0xbf81_6117_c4f8_2730,
            ),
            "dense seed6 FMA pivot-28 first-row multiplier boundary moved"
        );

        let final_second_row_source = native_block.local_perm[31];
        let second_multiplier_row = fma_pivot_snapshot
            .local_perm
            .iter()
            .position(|&entry| entry == final_second_row_source)
            .expect("dense seed6 FMA pivot-28 final second row source");
        assert_eq!(
            second_multiplier_row, 31,
            "dense seed6 FMA pivot-28 second source row moved"
        );
        let second_row_first_work = fma_pivot_snapshot.workspace
            [multiplier_col * APP_INNER_BLOCK_SIZE + second_multiplier_row];
        let second_row_second_work = fma_pivot_snapshot.workspace
            [(multiplier_col + 1) * APP_INNER_BLOCK_SIZE + second_multiplier_row];
        let second_row_fma = d11.mul_add(second_row_first_work, d21 * second_row_second_work);
        let second_row_source = d11 * second_row_first_work + d21 * second_row_second_work;
        let second_row_native =
            native_block.matrix[multiplier_col * native_lda + final_multiplier_row + 1];
        assert_eq!(
            (
                second_row_fma.to_bits(),
                second_row_source.to_bits(),
                second_row_native.to_bits()
            ),
            (
                0x3f66_e35e_c782_734a,
                0x3f66_e35e_c782_7340,
                0x3f66_e35e_c782_734a,
            ),
            "dense seed6 FMA pivot-28 second-row multiplier boundary moved"
        );
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
    fn dense_seed6_solution_bits_match_after_scalar_tail_multiplier() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense, expected_solution) = dense_seed6_33_matrix_and_solution();
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
        let (mut rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");

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

        let rust_d_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_d_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);
        assert_eq!(
            rust_d_bits, native_d_bits,
            "seed6 production inverse-D bit patterns differ before solve"
        );

        let rhs = dense_mul(&dense, &expected_solution);
        let rust_solution = rust_factor.solve(&rhs).expect("rust solve");
        let native_solution = native_session.solve(&rhs).expect("native solve");
        let rust_bits = bit_patterns(&rust_solution);
        let native_bits = bit_patterns(&native_solution);

        assert_eq!(
            rust_bits, native_bits,
            "seed6 solution bits should match after scalar-tail APP multiplier"
        );
    }

    #[test]
    fn deterministic_65_metadata_inverse_d_and_panel_solve_match_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let dimension = 65;
        let dense = deterministic_complete_dyadic_matrix(dimension);
        let expected_solution = deterministic_complete_dyadic_solution(dimension);
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
        let (mut rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");

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

        let rust_d_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_d_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);
        assert_eq!(
            rust_d_bits, native_d_bits,
            "deterministic 65x65 inverse-D bit patterns differ"
        );

        let rhs = dense_mul(&dense, &expected_solution);
        let mut rust_panel_rhs = vec![0.0; dimension];
        for (factor_position, &ordered_index) in rust_factor.factor_order.iter().enumerate() {
            rust_panel_rhs[factor_position] = rhs[rust_factor.permutation.perm()[ordered_index]];
        }
        let mut native_panel_rhs = rust_panel_rhs.clone();

        // Mirrors NumericSubtree.hxx::solve_fwd and
        // ldlt_app.cxx::ldlt_app_solve_fwd on Rust's stored panels.
        solve_forward_front_panels_like_native(&rust_factor.solve_panels, &mut rust_panel_rhs);
        native_solve_forward_front_panels(shim, &rust_factor.solve_panels, &mut native_panel_rhs);
        assert_eq!(
            bit_patterns(&rust_panel_rhs),
            bit_patterns(&native_panel_rhs),
            "deterministic 65x65 forward panel replay differs from native kernels"
        );

        let block_ranges =
            solve_panel_block_ranges(&rust_factor.solve_panels, &rust_factor.diagonal_blocks);
        assert_eq!(
            rust_factor.solve_panels.len(),
            1,
            "deterministic 65x65 solve-panel shape moved"
        );
        let panel = &rust_factor.solve_panels[0];
        let replay_case = AppSolveKernelCase {
            seed: 0,
            rows: panel.row_positions.len(),
            eliminated_len: panel.eliminated_len,
            lower: panel.values.clone(),
            diagonal: native_app_diagonal_for_block_range(
                &rust_factor.diagonal_blocks,
                &rust_factor.diagonal_values,
                block_ranges[0],
                panel.eliminated_len,
            ),
            rhs: Vec::new(),
        };
        let mut rust_local = vec![0.0; panel.row_positions.len()];
        let mut native_local = vec![0.0; panel.row_positions.len()];
        for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
            rust_local[local_row] = rust_panel_rhs[factor_position];
            native_local[local_row] = native_panel_rhs[factor_position];
        }
        rust_ldlt_app_solve_diag_like_native(&replay_case, &mut rust_local);
        native_ldlt_app_solve_diag(shim, &replay_case, &mut native_local);
        assert_eq!(
            bit_patterns(&rust_local),
            bit_patterns(&native_local),
            "deterministic 65x65 diagonal replay differs from native kernel"
        );
        rust_ldlt_app_solve_bwd_like_native(&replay_case, &mut rust_local);
        native_ldlt_app_solve_bwd(shim, &replay_case, &mut native_local);
        assert_eq!(
            bit_patterns(&rust_local),
            bit_patterns(&native_local),
            "deterministic 65x65 backward replay differs from native kernel"
        );

        // Mirrors NumericSubtree.hxx::solve_diag_bwd_inner<true,true> and
        // ldlt_app.cxx::ldlt_app_solve_diag / ldlt_app_solve_bwd on Rust data.
        solve_diagonal_and_lower_transpose_front_panels_like_native(
            &rust_factor.solve_panels,
            &rust_factor.diagonal_blocks,
            &rust_factor.diagonal_values,
            &mut rust_panel_rhs,
            None,
        )
        .expect("rust panel diagonal/backward replay");
        native_solve_diagonal_and_bwd_front_panels(
            shim,
            &rust_factor.solve_panels,
            &rust_factor.diagonal_blocks,
            &rust_factor.diagonal_values,
            &mut native_panel_rhs,
        );
        assert_eq!(
            bit_patterns(&rust_panel_rhs),
            bit_patterns(&native_panel_rhs),
            "deterministic 65x65 diag+bwd panel replay differs from native kernels"
        );

        let rust_solution = rust_factor.solve(&rhs).expect("rust solve");
        let native_solution = native_session.solve(&rhs).expect("native solve");
        let first_solution_mismatch = bit_patterns(&rust_solution)
            .iter()
            .zip(bit_patterns(&native_solution))
            .position(|(rust, native)| rust != &native);
        assert_eq!(
            first_solution_mismatch, None,
            "deterministic 65x65 full native solution mismatch"
        );
    }

    #[test]
    #[ignore = "manual case58 APP/full-factor classification harness"]
    fn dense_seed706172697479_case58_first_app_prefix_trace_matches_native() {
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, dense, _) = dense_seed706172697479_case58_matrix_and_solution();
        let mut lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for row in col..dimension {
                lower_dense[dense_lower_offset(dimension, row, col)] = dense[row][col];
            }
        }
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let mut csc_lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for entry in col_ptrs[col]..col_ptrs[col + 1] {
                let row = row_indices[entry];
                csc_lower_dense[dense_lower_offset(dimension, row, col)] = values[entry];
            }
        }
        let options = NumericFactorOptions::default();
        let native_lda = native_aligned_double_stride(shim, dimension);
        let aligned_lower_dense = copy_lower_dense_to_stride(&lower_dense, dimension, native_lda);

        let rust_unaligned_trace = rust_app_block_prefix_trace_32(&lower_dense, dimension, options);
        let rust_trace = rust_app_block_prefix_trace_32(&aligned_lower_dense, native_lda, options);
        let native_trace =
            native_block_prefix_trace_32(shim, &lower_dense, dimension, native_lda, options);
        let csc_aligned_lower_dense =
            copy_lower_dense_to_stride(&csc_lower_dense, dimension, native_lda);
        let rust_csc_trace =
            rust_app_block_prefix_trace_32(&csc_aligned_lower_dense, native_lda, options);
        let native_csc_trace =
            native_block_prefix_trace_32(shim, &csc_lower_dense, dimension, native_lda, options);
        eprintln!(
            "case58 csc first APP prefix mismatch={:?}",
            first_block_prefix_trace_mismatch(&rust_csc_trace, &native_csc_trace)
        );
        let rust_block = rust_block_ldlt_32_from_lower_dense(&lower_dense, dimension, options);
        let native_block = native_block_ldlt_32_from_lower_dense(
            shim,
            &lower_dense,
            dimension,
            native_lda,
            options,
        );
        let rust_full = factorize_dense_front(
            (0..dimension).collect(),
            dimension,
            lower_dense.clone(),
            options,
            false,
        )
        .expect("rust full APP factor");
        let rust_csc_full = factorize_dense_front(
            (0..dimension).collect(),
            dimension,
            csc_lower_dense.clone(),
            options,
            false,
        )
        .expect("rust full APP factor from CSC-shaped lower dense");
        eprintln!(
            "case58 rust dense perm22={} perm27={} rust csc perm22={} perm27={}",
            rust_full.factor_order[22],
            rust_full.factor_order[27],
            rust_csc_full.factor_order[22],
            rust_csc_full.factor_order[27],
        );
        let rust_csc_diagonal =
            dense_tpp_diagonal_from_blocks(&rust_csc_full.block_records, dimension);
        let rust_csc_entries =
            inverse_diagonal_entries_from_internal_diagonal(&rust_csc_diagonal, dimension);
        let native_csc_full =
            native_ldlt_app_factor_from_lower_dense(shim, &csc_lower_dense, dimension, options);
        let native_csc_entries =
            inverse_diagonal_entries_from_internal_diagonal(&native_csc_full.diagonal, dimension);
        let first_csc_d_mismatch = rust_csc_entries
            .iter()
            .flat_map(|entry| entry.iter())
            .zip(native_csc_entries.iter().flat_map(|entry| entry.iter()))
            .position(|(rust, native)| rust.to_bits() != native.to_bits());
        eprintln!(
            "case58 direct csc rust/native D mismatch={:?}",
            first_csc_d_mismatch.map(|index| (
                index,
                rust_csc_entries[index / 2][index % 2],
                native_csc_entries[index / 2][index % 2],
            ))
        );
        for cpu_block_size in [16, 32, 64, 128, 256] {
            let direct = native_ldlt_app_factor_from_lower_dense_with_block_size(
                shim,
                &lower_dense,
                dimension,
                options,
                cpu_block_size,
            );
            let direct_entries =
                inverse_diagonal_entries_from_internal_diagonal(&direct.diagonal, dimension);
            let first_direct_d_mismatch = rust_csc_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(direct_entries.iter().flat_map(|entry| entry.iter()))
                .position(|(rust, native)| rust.to_bits() != native.to_bits());
            eprintln!(
                "case58 direct ldlt_app cpu_block_size={cpu_block_size} eliminated={} perm22={} perm27={} rust_csc_d_mismatch={:?}",
                direct.eliminated,
                direct.perm[22],
                direct.perm[27],
                first_direct_d_mismatch.map(|index| (
                    index,
                    rust_csc_entries[index / 2][index % 2],
                    direct_entries[index / 2][index % 2],
                )),
            );
        }
        let native_full =
            native_ldlt_app_factor_from_lower_dense(shim, &lower_dense, dimension, options);
        eprintln!(
            "case58 direct ldlt_app_factor eliminated={} tail_size={}",
            native_full.eliminated,
            dimension - native_full.eliminated
        );
        assert_eq!(
            (
                first_block_prefix_trace_mismatch(&rust_unaligned_trace, &native_trace),
                first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
                (rust_block.perm == native_block.perm).then_some(()),
                (rust_block.local_perm == native_block.local_perm).then_some(()),
                (rust_full.factor_order == native_full.perm).then_some(()),
            ),
            (None, None, Some(()), Some(()), Some(())),
            "case58 first APP block prefix trace differs from native block_ldlt.hxx::block_ldlt"
        );
    }

    #[test]
    #[ignore = "manual case58 production Rust/native classification harness"]
    fn dense_seed706172697479_case58_classifies_remaining_solution_gap() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, dense, expected_solution) =
            dense_seed706172697479_case58_matrix_and_solution();
        let (col_ptrs, row_indices, values) = dense_to_lower_csc(&dense);
        let mut csc_lower_dense = vec![0.0; dimension * dimension];
        for col in 0..dimension {
            for entry in col_ptrs[col]..col_ptrs[col + 1] {
                let row = row_indices[entry];
                csc_lower_dense[dense_lower_offset(dimension, row, col)] = values[entry];
            }
        }
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
            .expect("valid CSC");
        let options = NumericFactorOptions::default();

        let first_app_replay =
            replay_app_block_for_debug((0..dimension).collect(), csc_lower_dense, 0, options);
        let first_apply_mismatch =
            first_native_apply_pivot_tile_mismatch(shim, &first_app_replay, 0);
        let first_update_mismatch =
            first_native_accepted_update_mismatch(shim, &first_app_replay, 0);
        eprintln!(
            "case58 direct APP block0 accepted_end={} first_failed={} apply_mismatch={:?} accepted_update_mismatch={:?}",
            first_app_replay.accepted_end,
            first_app_replay.first_failed,
            first_apply_mismatch,
            first_update_mismatch
        );
        if first_update_mismatch.is_none()
            && first_app_replay.accepted_end + APP_INNER_BLOCK_SIZE <= dimension
        {
            let tail_start = first_app_replay.accepted_end;
            let tail_size = dimension - tail_start;
            let mut tail_dense = vec![0.0; tail_size * tail_size];
            for local_col in 0..tail_size {
                for local_row in local_col..tail_size {
                    tail_dense[local_col * tail_size + local_row] = first_app_replay.after_update
                        [(tail_start + local_col) * dimension + tail_start + local_row];
                }
            }
            let tail_lda = native_aligned_double_stride(shim, tail_size);
            let parent_lda = native_aligned_double_stride(shim, dimension);
            let rust_tail_trace = rust_app_block_prefix_trace_32(&tail_dense, tail_size, options);
            let native_tail_trace =
                native_block_prefix_trace_32(shim, &tail_dense, tail_size, tail_lda, options);
            let native_tail_source_multiplier_trace =
                native_block_prefix_trace_32_source_multiplier(
                    shim,
                    &tail_dense,
                    tail_size,
                    tail_lda,
                    options,
                );
            let native_tail_source_trace = native_block_prefix_trace_32_source(
                shim,
                &tail_dense,
                tail_size,
                tail_lda,
                options,
            );
            let native_parent_stride_tail_trace =
                native_block_prefix_trace_32(shim, &tail_dense, tail_size, parent_lda, options);
            let native_tail_block = native_block_ldlt_32_from_lower_dense(
                shim,
                &tail_dense,
                tail_size,
                tail_lda,
                options,
            );
            let native_tail_continuation_mismatch = first_native_block_continuation_mismatch(
                shim,
                &native_tail_trace,
                &native_tail_block,
                tail_lda,
                options,
            );
            let native_tail_source_multiplier_continuation_mismatch =
                first_native_block_continuation_mismatch(
                    shim,
                    &native_tail_source_multiplier_trace,
                    &native_tail_block,
                    tail_lda,
                    options,
                );
            let native_tail_source_continuation_mismatch = first_native_block_continuation_mismatch(
                shim,
                &native_tail_source_trace,
                &native_tail_block,
                tail_lda,
                options,
            );
            let native_tail_frozen_mismatch = native_tail_trace
                .iter()
                .find(|snapshot| snapshot.step == 7)
                .and_then(|snapshot| {
                    first_frozen_prefix_mismatch_against_block(
                        snapshot,
                        &native_tail_block,
                        tail_lda,
                    )
                });
            let native_tail_step7_expr_bits = native_tail_trace
                .iter()
                .find(|snapshot| snapshot.step == 7)
                .map(|snapshot| two_by_two_first_multiplier_expression_bits(snapshot, 31));
            let native_tail_block_entries = inverse_diagonal_entries_from_internal_diagonal(
                &native_tail_block.diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let rust_tail_trace_entries = inverse_diagonal_entries_from_internal_diagonal(
                &rust_tail_trace
                    .last()
                    .expect("case58 direct compact tail APP trace")
                    .diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let first_actual_block_d_mismatch = rust_tail_trace_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(
                    native_tail_block_entries
                        .iter()
                        .flat_map(|entry| entry.iter()),
                )
                .position(|(rust, native)| rust.to_bits() != native.to_bits());
            eprintln!(
                "case58 direct APP block1 prefix_mismatch tail_stride={:?} parent_stride={:?} continuation_mismatch={:?} source_multiplier_continuation={:?} source_continuation={:?} frozen_step7_mismatch={:?} step7_row31_expr_bits={:?} actual_block_d_mismatch={:?}",
                first_block_prefix_trace_mismatch(&rust_tail_trace, &native_tail_trace),
                first_block_prefix_trace_mismatch(
                    &rust_tail_trace,
                    &native_parent_stride_tail_trace
                ),
                native_tail_continuation_mismatch,
                native_tail_source_multiplier_continuation_mismatch,
                native_tail_source_continuation_mismatch,
                native_tail_frozen_mismatch,
                native_tail_step7_expr_bits,
                first_actual_block_d_mismatch.map(|index| (
                    index,
                    rust_tail_trace_entries[index / 2][index % 2],
                    native_tail_block_entries[index / 2][index % 2],
                )),
            );

            let second_app_replay = replay_app_block_for_debug(
                first_app_replay.rows.clone(),
                first_app_replay.after_update.clone(),
                first_app_replay.accepted_end,
                options,
            );
            eprintln!(
                "case58 direct APP block1 accepted_end={} first_failed={} apply_mismatch={:?} accepted_update_mismatch={:?}",
                second_app_replay.accepted_end,
                second_app_replay.first_failed,
                first_native_apply_pivot_tile_mismatch(
                    shim,
                    &second_app_replay,
                    first_app_replay.accepted_end,
                ),
                first_native_accepted_update_mismatch(
                    shim,
                    &second_app_replay,
                    first_app_replay.accepted_end,
                )
            );
        }

        let (symbolic, _) = analyse(
            matrix,
            &SsidsOptions {
                ordering: OrderingStrategy::Natural,
            },
        )
        .expect("rust analyse");
        let (mut rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");

        let mut native_session = native
            .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
            .expect("native analyse");
        eprintln!(
            "case58 classifier symbolic rust_supernodes={} rust_max_supernode={} native_analyse={:?}",
            symbolic.supernodes.len(),
            symbolic
                .supernodes
                .iter()
                .map(|supernode| supernode.width())
                .max()
                .unwrap_or(0),
            native_session.analyse_info(),
        );
        let tree = build_symbolic_front_tree(&symbolic);
        eprintln!(
            "case58 rust permutation first32={:?} front_columns first32={:?}",
            &symbolic.permutation.perm()[..32.min(symbolic.permutation.perm().len())],
            &tree.fronts[0].columns[..32.min(tree.fronts[0].columns.len())],
        );
        let mut permuted_col_ptrs = Vec::new();
        let mut permuted_row_indices = Vec::new();
        let mut permuted_source_positions = Vec::new();
        super::build_permuted_lower_csc_pattern(
            matrix,
            &symbolic.permutation,
            &mut permuted_col_ptrs,
            &mut permuted_row_indices,
            &mut permuted_source_positions,
        )
        .expect("case58 permuted pattern");
        let mut permuted_values = Vec::new();
        super::fill_permuted_lower_csc_values(
            matrix,
            &permuted_source_positions,
            &mut permuted_values,
        )
        .expect("case58 permuted values");
        let front = &tree.fronts[0];
        let mut production_local_rows = front.columns.clone();
        let mut production_interface_rows = front.interface_rows.clone();
        production_interface_rows.sort_unstable();
        production_local_rows.extend(production_interface_rows);
        let production_local_size = production_local_rows.len();
        let mut production_local_positions = vec![usize::MAX; dimension];
        for (position, &row) in production_local_rows.iter().enumerate() {
            production_local_positions[row] = position;
        }
        let mut production_local_dense = vec![0.0; production_local_size * production_local_size];
        for &column in &front.columns {
            let local_column = production_local_positions[column];
            for entry in permuted_col_ptrs[column]..permuted_col_ptrs[column + 1] {
                let row = permuted_row_indices[entry];
                let local_row = production_local_positions[row];
                if local_row == usize::MAX {
                    continue;
                }
                production_local_dense
                    [dense_lower_offset(production_local_size, local_row, local_column)] =
                    permuted_values[entry];
            }
        }
        let production_first_app_replay =
            replay_app_block_for_debug(production_local_rows, production_local_dense, 0, options);
        let production_first_apply_mismatch =
            first_native_apply_pivot_tile_mismatch(shim, &production_first_app_replay, 0);
        let production_first_update_mismatch =
            first_native_accepted_update_mismatch(shim, &production_first_app_replay, 0);
        eprintln!(
            "case58 production APP block0 accepted_end={} first_failed={} apply_mismatch={:?} accepted_update_mismatch={:?}",
            production_first_app_replay.accepted_end,
            production_first_app_replay.first_failed,
            production_first_apply_mismatch,
            production_first_update_mismatch
        );
        if production_first_update_mismatch.is_none()
            && production_first_app_replay.accepted_end + APP_INNER_BLOCK_SIZE <= dimension
        {
            let tail_start = production_first_app_replay.accepted_end;
            let tail_size = dimension - tail_start;
            let mut tail_dense = vec![0.0; tail_size * tail_size];
            for local_col in 0..tail_size {
                for local_row in local_col..tail_size {
                    tail_dense[local_col * tail_size + local_row] = production_first_app_replay
                        .after_update
                        [(tail_start + local_col) * dimension + tail_start + local_row];
                }
            }
            let tail_lda = native_aligned_double_stride(shim, tail_size);
            let parent_lda = native_aligned_double_stride(shim, dimension);
            let rust_tail_trace = rust_app_block_prefix_trace_32(&tail_dense, tail_size, options);
            let native_tail_trace =
                native_block_prefix_trace_32(shim, &tail_dense, tail_size, tail_lda, options);
            let native_tail_source_multiplier_trace =
                native_block_prefix_trace_32_source_multiplier(
                    shim,
                    &tail_dense,
                    tail_size,
                    tail_lda,
                    options,
                );
            let native_tail_source_trace = native_block_prefix_trace_32_source(
                shim,
                &tail_dense,
                tail_size,
                tail_lda,
                options,
            );
            let native_parent_stride_tail_trace =
                native_block_prefix_trace_32(shim, &tail_dense, tail_size, parent_lda, options);
            let native_tail_block = native_block_ldlt_32_from_lower_dense(
                shim,
                &tail_dense,
                tail_size,
                tail_lda,
                options,
            );
            let native_tail_continuation_mismatch = first_native_block_continuation_mismatch(
                shim,
                &native_tail_trace,
                &native_tail_block,
                tail_lda,
                options,
            );
            let native_tail_source_multiplier_continuation_mismatch =
                first_native_block_continuation_mismatch(
                    shim,
                    &native_tail_source_multiplier_trace,
                    &native_tail_block,
                    tail_lda,
                    options,
                );
            let native_tail_source_continuation_mismatch = first_native_block_continuation_mismatch(
                shim,
                &native_tail_source_trace,
                &native_tail_block,
                tail_lda,
                options,
            );
            let native_tail_frozen_mismatch = native_tail_trace
                .iter()
                .find(|snapshot| snapshot.step == 7)
                .and_then(|snapshot| {
                    first_frozen_prefix_mismatch_against_block(
                        snapshot,
                        &native_tail_block,
                        tail_lda,
                    )
                });
            let native_tail_step7_expr_bits = native_tail_trace
                .iter()
                .find(|snapshot| snapshot.step == 7)
                .map(|snapshot| two_by_two_first_multiplier_expression_bits(snapshot, 31));
            let native_tail_block_entries = inverse_diagonal_entries_from_internal_diagonal(
                &native_tail_block.diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let compact_tail_entries = inverse_diagonal_entries_from_internal_diagonal(
                &rust_tail_trace
                    .last()
                    .expect("case58 compact tail APP trace")
                    .diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let first_actual_block_d_mismatch = compact_tail_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(
                    native_tail_block_entries
                        .iter()
                        .flat_map(|entry| entry.iter()),
                )
                .position(|(rust, native)| rust.to_bits() != native.to_bits());
            eprintln!(
                "case58 production APP block1 prefix_mismatch tail_stride={:?} parent_stride={:?} continuation_mismatch={:?} source_multiplier_continuation={:?} source_continuation={:?} frozen_step7_mismatch={:?} step7_row31_expr_bits={:?} actual_block_d_mismatch={:?}",
                first_block_prefix_trace_mismatch(&rust_tail_trace, &native_tail_trace),
                first_block_prefix_trace_mismatch(
                    &rust_tail_trace,
                    &native_parent_stride_tail_trace
                ),
                native_tail_continuation_mismatch,
                native_tail_source_multiplier_continuation_mismatch,
                native_tail_source_continuation_mismatch,
                native_tail_frozen_mismatch,
                native_tail_step7_expr_bits,
                first_actual_block_d_mismatch.map(|index| (
                    index,
                    compact_tail_entries[index / 2][index % 2],
                    native_tail_block_entries[index / 2][index % 2],
                )),
            );

            let production_second_app_replay = replay_app_block_for_debug(
                production_first_app_replay.rows.clone(),
                production_first_app_replay.after_update.clone(),
                production_first_app_replay.accepted_end,
                options,
            );
            let second_replay_diagonal = dense_tpp_diagonal_from_blocks(
                &production_second_app_replay.accepted_blocks,
                production_second_app_replay.accepted_end
                    - production_first_app_replay.accepted_end,
            );
            let second_replay_entries = inverse_diagonal_entries_from_internal_diagonal(
                &second_replay_diagonal,
                production_second_app_replay.accepted_end
                    - production_first_app_replay.accepted_end,
            );
            let native_block1_tpp = native_ldlt_tpp_factor_from_lower_dense(
                shim,
                &tail_dense,
                tail_size,
                APP_INNER_BLOCK_SIZE,
                options,
            );
            let native_block1_tpp_entries = inverse_diagonal_entries_from_internal_diagonal(
                &native_block1_tpp.diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let first_block1_tpp_d_mismatch = second_replay_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(
                    native_block1_tpp_entries
                        .iter()
                        .flat_map(|entry| entry.iter()),
                )
                .position(|(app, tpp)| app.to_bits() != tpp.to_bits());
            let first_full_vs_compact_d_mismatch = second_replay_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(compact_tail_entries.iter().flat_map(|entry| entry.iter()))
                .position(|(full, compact)| full.to_bits() != compact.to_bits());
            let mut replay_blocks = production_first_app_replay.accepted_blocks.clone();
            replay_blocks.extend(production_second_app_replay.accepted_blocks.clone());
            let replay_diagonal = dense_tpp_diagonal_from_blocks(
                &replay_blocks,
                production_second_app_replay.accepted_end,
            );
            let replay_entries = inverse_diagonal_entries_from_internal_diagonal(
                &replay_diagonal,
                production_second_app_replay.accepted_end,
            );
            let rust_entries = rust_inverse_diagonal_entries(&rust_factor);
            let first_replay_vs_factor_d_mismatch = replay_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(rust_entries.iter().flat_map(|entry| entry.iter()))
                .position(|(replay, factor)| replay.to_bits() != factor.to_bits());
            eprintln!(
                "case58 production APP block1 accepted_end={} first_failed={} apply_mismatch={:?} accepted_update_mismatch={:?} app_vs_tpp_d_mismatch={:?} full_vs_compact_d_mismatch={:?} replay_vs_factor_d_mismatch={:?}",
                production_second_app_replay.accepted_end,
                production_second_app_replay.first_failed,
                first_native_apply_pivot_tile_mismatch(
                    shim,
                    &production_second_app_replay,
                    production_first_app_replay.accepted_end,
                ),
                first_native_accepted_update_mismatch(
                    shim,
                    &production_second_app_replay,
                    production_first_app_replay.accepted_end,
                ),
                first_block1_tpp_d_mismatch.map(|index| (
                    index,
                    second_replay_entries[index / 2][index % 2],
                    native_block1_tpp_entries[index / 2][index % 2],
                )),
                first_full_vs_compact_d_mismatch.map(|index| (
                    index,
                    second_replay_entries[index / 2][index % 2],
                    compact_tail_entries[index / 2][index % 2],
                )),
                first_replay_vs_factor_d_mismatch.map(|index| (
                    index,
                    replay_entries[index / 2][index % 2],
                    rust_entries[index / 2][index % 2],
                )),
            );
            if production_second_app_replay.accepted_end + APP_INNER_BLOCK_SIZE <= dimension {
                let third_tail_start = production_second_app_replay.accepted_end;
                let third_tail_size = dimension - third_tail_start;
                let mut third_tail_dense = vec![0.0; third_tail_size * third_tail_size];
                for local_col in 0..third_tail_size {
                    for local_row in local_col..third_tail_size {
                        third_tail_dense[local_col * third_tail_size + local_row] =
                            production_second_app_replay.after_update[(third_tail_start
                                + local_col)
                                * dimension
                                + third_tail_start
                                + local_row];
                    }
                }
                let third_tail_lda = native_aligned_double_stride(shim, third_tail_size);
                let third_rust_trace =
                    rust_app_block_prefix_trace_32(&third_tail_dense, third_tail_size, options);
                let third_native_trace = native_block_prefix_trace_32(
                    shim,
                    &third_tail_dense,
                    third_tail_size,
                    third_tail_lda,
                    options,
                );
                let third_native_block = native_block_ldlt_32_from_lower_dense(
                    shim,
                    &third_tail_dense,
                    third_tail_size,
                    third_tail_lda,
                    options,
                );
                let third_continuation_mismatch = first_native_block_continuation_mismatch(
                    shim,
                    &third_native_trace,
                    &third_native_block,
                    third_tail_lda,
                    options,
                );
                let third_frozen_mismatch = third_continuation_mismatch
                    .as_ref()
                    .and_then(|mismatch| {
                        third_native_trace
                            .iter()
                            .find(|snapshot| snapshot.step == mismatch.step)
                    })
                    .and_then(|snapshot| {
                        first_frozen_prefix_mismatch_against_block(
                            snapshot,
                            &third_native_block,
                            third_tail_lda,
                        )
                    });
                let third_expr_bits = third_frozen_mismatch.as_ref().and_then(|mismatch| {
                    third_native_trace
                        .iter()
                        .find(|snapshot| snapshot.step == mismatch.step)
                        .map(|snapshot| {
                            two_by_two_first_multiplier_expression_bits(snapshot, mismatch.row)
                        })
                });
                let third_expr_rows = third_frozen_mismatch.as_ref().and_then(|mismatch| {
                    third_native_trace
                        .iter()
                        .find(|snapshot| snapshot.step == mismatch.step)
                        .map(|snapshot| {
                            two_by_two_first_multiplier_block_expr_bits(
                                snapshot,
                                &third_native_block,
                                third_tail_lda,
                            )
                        })
                });
                let third_native_entries = inverse_diagonal_entries_from_internal_diagonal(
                    &third_native_block.diagonal,
                    APP_INNER_BLOCK_SIZE,
                );
                let third_compact_entries = inverse_diagonal_entries_from_internal_diagonal(
                    &third_rust_trace
                        .last()
                        .expect("case58 third compact APP trace")
                        .diagonal,
                    APP_INNER_BLOCK_SIZE,
                );
                let third_first_d_mismatch = third_compact_entries
                    .iter()
                    .flat_map(|entry| entry.iter())
                    .zip(third_native_entries.iter().flat_map(|entry| entry.iter()))
                    .position(|(rust, native)| rust.to_bits() != native.to_bits());
                let production_third_app_replay = replay_app_block_for_debug(
                    production_second_app_replay.rows.clone(),
                    production_second_app_replay.after_update.clone(),
                    third_tail_start,
                    options,
                );
                eprintln!(
                    "case58 production APP block2 prefix_mismatch={:?} continuation_mismatch={:?} frozen_mismatch={:?} expr_bits={:?} expr_rows={:?} actual_block_d_mismatch={:?} accepted_end={} first_failed={} apply_mismatch={:?} accepted_update_mismatch={:?}",
                    first_block_prefix_trace_mismatch(&third_rust_trace, &third_native_trace),
                    third_continuation_mismatch,
                    third_frozen_mismatch,
                    third_expr_bits,
                    third_expr_rows,
                    third_first_d_mismatch.map(|index| (
                        index,
                        third_compact_entries[index / 2][index % 2],
                        third_native_entries[index / 2][index % 2],
                    )),
                    production_third_app_replay.accepted_end,
                    production_third_app_replay.first_failed,
                    first_native_apply_pivot_tile_mismatch(
                        shim,
                        &production_third_app_replay,
                        third_tail_start,
                    ),
                    first_native_accepted_update_mismatch(
                        shim,
                        &production_third_app_replay,
                        third_tail_start,
                    ),
                );
            }
        }
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
        let rust_factor_order_original = rust_factor
            .factor_order
            .iter()
            .map(|&ordered_index| rust_factor.permutation.perm()[ordered_index])
            .collect::<Vec<_>>();
        let first_order_mismatch = rust_factor_order_original
            .iter()
            .zip(&native_factor_order)
            .position(|(rust, native)| rust != native);
        assert_eq!(
            first_order_mismatch.map(|index| {
                (
                    index,
                    rust_factor_order_original[index],
                    native_factor_order[index],
                    native_enquiry.pivot_order[rust_factor_order_original[index]],
                    native_enquiry.pivot_order[native_factor_order[index]],
                )
            }),
            None,
            "case58 factor order differs from native pivot order"
        );

        let rust_d_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_d_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);
        let first_d_mismatch = rust_d_bits
            .iter()
            .zip(&native_d_bits)
            .position(|(rust, native)| rust != native);
        assert_eq!(
            first_d_mismatch.map(|index| {
                (
                    index,
                    rust_d_bits[index],
                    native_d_bits[index],
                    rust_inverse_diagonal_entries(&rust_factor)[index / 2][index % 2],
                    native_enquiry.inverse_diagonal_entries[index / 2][index % 2],
                )
            }),
            None,
            "case58 inverse-D bit patterns differ from native"
        );

        let rhs = dense_mul(&dense, &expected_solution);
        let mut rust_panel_rhs = vec![0.0; dimension];
        for (factor_position, &ordered_index) in rust_factor.factor_order.iter().enumerate() {
            rust_panel_rhs[factor_position] = rhs[rust_factor.permutation.perm()[ordered_index]];
        }
        let mut native_panel_rhs = rust_panel_rhs.clone();

        // Mirrors NumericSubtree.hxx::solve_fwd and
        // ldlt_app.cxx::ldlt_app_solve_fwd on Rust's stored panels.
        solve_forward_front_panels_like_native(&rust_factor.solve_panels, &mut rust_panel_rhs);
        native_solve_forward_front_panels(shim, &rust_factor.solve_panels, &mut native_panel_rhs);
        assert_no_bit_mismatch(
            &rust_panel_rhs,
            &native_panel_rhs,
            "case58 forward panel replay",
        );

        let panel_block_ranges =
            solve_panel_block_ranges(&rust_factor.solve_panels, &rust_factor.diagonal_blocks);
        assert_eq!(
            rust_factor.solve_panels.len(),
            1,
            "case58 solve-panel shape moved"
        );
        let panel = &rust_factor.solve_panels[0];
        let replay_case = AppSolveKernelCase {
            seed: 0,
            rows: panel.row_positions.len(),
            eliminated_len: panel.eliminated_len,
            lower: panel.values.clone(),
            diagonal: native_app_diagonal_for_block_range(
                &rust_factor.diagonal_blocks,
                &rust_factor.diagonal_values,
                panel_block_ranges[0],
                panel.eliminated_len,
            ),
            rhs: Vec::new(),
        };
        let mut rust_local = vec![0.0; panel.row_positions.len()];
        let mut native_local = vec![0.0; panel.row_positions.len()];
        for (local_row, &factor_position) in panel.row_positions.iter().enumerate() {
            rust_local[local_row] = rust_panel_rhs[factor_position];
            native_local[local_row] = native_panel_rhs[factor_position];
        }
        rust_ldlt_app_solve_diag_like_native(&replay_case, &mut rust_local);
        native_ldlt_app_solve_diag(shim, &replay_case, &mut native_local);
        assert_no_bit_mismatch(&rust_local, &native_local, "case58 diagonal replay");

        rust_ldlt_app_solve_bwd_like_native(&replay_case, &mut rust_local);
        native_ldlt_app_solve_bwd(shim, &replay_case, &mut native_local);
        assert_no_bit_mismatch(&rust_local, &native_local, "case58 backward replay");

        // Mirrors NumericSubtree.hxx::solve_diag_bwd_inner<true,true> and
        // ldlt_app.cxx::ldlt_app_solve_diag / ldlt_app_solve_bwd on Rust data.
        solve_diagonal_and_lower_transpose_front_panels_like_native(
            &rust_factor.solve_panels,
            &rust_factor.diagonal_blocks,
            &rust_factor.diagonal_values,
            &mut rust_panel_rhs,
            None,
        )
        .expect("rust panel diagonal/backward replay");
        native_solve_diagonal_and_bwd_front_panels(
            shim,
            &rust_factor.solve_panels,
            &rust_factor.diagonal_blocks,
            &rust_factor.diagonal_values,
            &mut native_panel_rhs,
        );
        assert_no_bit_mismatch(
            &rust_panel_rhs,
            &native_panel_rhs,
            "case58 diag+bwd panel replay",
        );

        let rust_solution = rust_factor.solve(&rhs).expect("rust solve");
        let native_solution = native_session.solve(&rhs).expect("native solve");
        assert_eq!(
            first_bit_mismatch(&rust_solution, &native_solution),
            None,
            "case58 full solution differs after matching metadata, inverse-D, and Rust-data panel replays"
        );
    }

    fn classify_dense_seed09_case_remaining_solution_gap(case_index: usize) {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let Some(shim) = native_kernel_shim_or_skip() else {
            return;
        };
        let (dimension, dense, expected_solution) =
            dense_boundary_case_matrix_and_solution(0x09c9_134e_4eff_0004, case_index);
        eprintln!("dense seed09 case{case_index} dimension={dimension}");
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
        let (mut rust_factor, _) = factorize(matrix, &symbolic, &options).expect("rust factorize");
        let mut native_session = native
            .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
            .expect("native analyse");
        let native_info = native_session.factorize(matrix).expect("native factorize");
        let native_enquiry = native_session.enquire_indef().expect("native enquire");
        eprintln!(
            "dense seed09 case{case_index} metadata rust_inertia={:?} native_inertia={:?} rust_2x2={} native_2x2={} rust_delayed={} native_delayed={}",
            rust_factor.inertia(),
            native_info.inertia,
            rust_factor.pivot_stats().two_by_two_pivots,
            native_info.two_by_two_pivots,
            rust_factor.pivot_stats().delayed_pivots,
            native_info.delayed_pivots,
        );
        let rust_entries = rust_inverse_diagonal_entries(&rust_factor);
        let rust_d_bits = inverse_diagonal_bits(&rust_entries);
        let native_d_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);
        let first_d_mismatch = rust_d_bits
            .iter()
            .zip(&native_d_bits)
            .position(|(rust, native)| rust != native);
        eprintln!(
            "dense seed09 case{case_index} first_d_mismatch={:?}",
            first_d_mismatch.map(|index| (
                index,
                rust_entries[index / 2][index % 2],
                native_enquiry.inverse_diagonal_entries[index / 2][index % 2],
            )),
        );

        let tree = build_symbolic_front_tree(&symbolic);
        let mut permuted_col_ptrs = Vec::new();
        let mut permuted_row_indices = Vec::new();
        let mut permuted_source_positions = Vec::new();
        super::build_permuted_lower_csc_pattern(
            matrix,
            &symbolic.permutation,
            &mut permuted_col_ptrs,
            &mut permuted_row_indices,
            &mut permuted_source_positions,
        )
        .expect("permuted pattern");
        let mut permuted_values = Vec::new();
        super::fill_permuted_lower_csc_values(
            matrix,
            &permuted_source_positions,
            &mut permuted_values,
        )
        .expect("permuted values");
        let front = &tree.fronts[0];
        let mut local_rows = front.columns.clone();
        let mut interface_rows = front.interface_rows.clone();
        interface_rows.sort_unstable();
        local_rows.extend(interface_rows);
        let local_size = local_rows.len();
        let mut local_positions = vec![usize::MAX; dimension];
        for (position, &row) in local_rows.iter().enumerate() {
            local_positions[row] = position;
        }
        let mut local_dense = vec![0.0; local_size * local_size];
        for &column in &front.columns {
            let local_column = local_positions[column];
            for entry in permuted_col_ptrs[column]..permuted_col_ptrs[column + 1] {
                let row = permuted_row_indices[entry];
                let local_row = local_positions[row];
                if local_row != usize::MAX {
                    local_dense[dense_lower_offset(local_size, local_row, local_column)] =
                        permuted_values[entry];
                }
            }
        }

        let mut rows = local_rows;
        let mut dense_state = local_dense;
        let mut block_start = 0;
        while block_start + APP_INNER_BLOCK_SIZE <= local_size {
            let tail_size = local_size - block_start;
            let mut tail_dense = vec![0.0; tail_size * tail_size];
            for local_col in 0..tail_size {
                for local_row in local_col..tail_size {
                    tail_dense[local_col * tail_size + local_row] = dense_state
                        [(block_start + local_col) * local_size + block_start + local_row];
                }
            }
            let tail_lda = native_aligned_double_stride(shim, tail_size);
            let rust_trace = rust_app_block_prefix_trace_32(&tail_dense, tail_size, options);
            let native_trace =
                native_block_prefix_trace_32(shim, &tail_dense, tail_size, tail_lda, options);
            let native_block = native_block_ldlt_32_from_lower_dense(
                shim,
                &tail_dense,
                tail_size,
                tail_lda,
                options,
            );
            let native_entries = inverse_diagonal_entries_from_internal_diagonal(
                &native_block.diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let rust_entries = inverse_diagonal_entries_from_internal_diagonal(
                &rust_trace.last().expect("dense seed09 APP trace").diagonal,
                APP_INNER_BLOCK_SIZE,
            );
            let first_block_d_mismatch = rust_entries
                .iter()
                .flat_map(|entry| entry.iter())
                .zip(native_entries.iter().flat_map(|entry| entry.iter()))
                .position(|(rust, native)| rust.to_bits() != native.to_bits());
            let continuation_mismatch = first_native_block_continuation_mismatch(
                shim,
                &native_trace,
                &native_block,
                tail_lda,
                options,
            );
            let frozen_mismatch = continuation_mismatch
                .as_ref()
                .and_then(|mismatch| {
                    native_trace
                        .iter()
                        .find(|snapshot| snapshot.step == mismatch.step)
                })
                .and_then(|snapshot| {
                    first_frozen_prefix_mismatch_against_block(snapshot, &native_block, tail_lda)
                });
            let expr_rows = frozen_mismatch.as_ref().and_then(|mismatch| {
                native_trace
                    .iter()
                    .find(|snapshot| snapshot.step == mismatch.step)
                    .map(|snapshot| {
                        two_by_two_first_multiplier_block_expr_bits(
                            snapshot,
                            &native_block,
                            tail_lda,
                        )
                    })
            });
            let replay =
                replay_app_block_for_debug(rows.clone(), dense_state.clone(), block_start, options);
            eprintln!(
                "dense seed09 case{case_index} APP block_start={block_start} accepted_end={} first_failed={} prefix_mismatch={:?} continuation_mismatch={:?} frozen_mismatch={:?} expr_rows={:?} block_d_mismatch={:?} apply_mismatch={:?} accepted_update_mismatch={:?}",
                replay.accepted_end,
                replay.first_failed,
                first_block_prefix_trace_mismatch(&rust_trace, &native_trace),
                continuation_mismatch,
                frozen_mismatch,
                expr_rows,
                first_block_d_mismatch.map(|index| (
                    index,
                    rust_entries[index / 2][index % 2],
                    native_entries[index / 2][index % 2],
                )),
                first_native_apply_pivot_tile_mismatch(shim, &replay, block_start),
                first_native_accepted_update_mismatch(shim, &replay, block_start),
            );
            rows = replay.rows;
            dense_state = replay.after_update;
            block_start = replay.accepted_end;
        }

        let rhs = dense_mul(&dense, &expected_solution);
        let rust_solution = rust_factor.solve(&rhs).expect("rust solve");
        let native_solution = native_session.solve(&rhs).expect("native solve");
        assert_eq!(
            first_bit_mismatch(&rust_solution, &native_solution),
            None,
            "dense seed09 case{case_index} solution differs"
        );
    }

    #[test]
    #[ignore = "manual dense random APP block classifier"]
    fn dense_seed09_case5_classifies_remaining_solution_gap() {
        classify_dense_seed09_case_remaining_solution_gap(5);
    }

    #[test]
    #[ignore = "manual dense random APP block classifier"]
    fn dense_seed09_case15_classifies_remaining_solution_gap() {
        classify_dense_seed09_case_remaining_solution_gap(15);
    }

    #[test]
    fn dense_seed1001_production_inverse_d_matches_native() {
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let (dimension, dense, _expected_solution) = dense_seed1001_33_matrix_and_solution();
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

        let rust_d_bits = inverse_diagonal_bits(&rust_inverse_diagonal_entries(&rust_factor));
        let native_d_bits = inverse_diagonal_bits(&native_enquiry.inverse_diagonal_entries);
        assert_eq!(
            rust_d_bits
                .iter()
                .zip(&native_d_bits)
                .position(|(rust, native)| rust != native),
            None,
            "dense seed1001 production inverse-D mismatch"
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

        assert_eq!(
            rust_bits, native_bits,
            "dense seed09 case0 production inverse-D bit patterns differ"
        );
    }

    #[test]
    fn dense_seed09_case0_production_inverse_d_entries_match_native() {
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

        for pivot in 0..rust_entries.len() {
            for component in 0..2 {
                assert_eq!(
                    rust_entries[pivot][component].to_bits(),
                    native_entries[pivot][component].to_bits(),
                    "dense seed09 production inverse-D mismatch pivot={pivot} component={component}"
                );
            }
        }
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
            mismatches,
            Vec::<(usize, usize, u64, u64)>::new(),
            "dense seed09 inverse-D mismatches should be empty"
        );
    }

    #[test]
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
        let dense_restore_backup = app_backup_trailing_lower(&lower_dense, dimension, block_start);
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
        let native_source_multiplier_trace = native_block_prefix_trace_32_source_multiplier(
            shim,
            &dense_before_block,
            dimension,
            native_lda,
            options,
        );
        let native_source_multiplier_trace_block = native_source_multiplier_trace
            .last()
            .expect("source-multiplier native APP block prefix trace");
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
            first_native_trace_block_mismatch, None,
            "dense seed09 native APP trace/block_ldlt matrix mismatch"
        );
        let first_fma_continuation_mismatch = first_native_block_continuation_mismatch(
            shim,
            &native_trace,
            &native_block,
            native_lda,
            options,
        );
        assert_eq!(
            first_fma_continuation_mismatch, None,
            "dense seed09 native APP trace continuation mismatch"
        );
        let fma_pivot_snapshot = native_trace
            .iter()
            .find(|snapshot| snapshot.step == 10 && snapshot.from == 19 && snapshot.status == 2)
            .expect("dense seed09 FMA pivot-19 snapshot");
        let multiplier_col = fma_pivot_snapshot.from;
        let final_multiplier_row = 30;
        let final_multiplier_source = native_block.local_perm[final_multiplier_row];
        let multiplier_row = fma_pivot_snapshot
            .local_perm
            .iter()
            .position(|&entry| entry == final_multiplier_source)
            .expect("dense seed09 FMA pivot-19 final multiplier row source");
        assert_eq!(
            multiplier_row, 31,
            "dense seed09 FMA pivot-19 source row moved"
        );
        assert_eq!(multiplier_col + 2, fma_pivot_snapshot.next);
        assert!(multiplier_row >= fma_pivot_snapshot.next);
        let d11 = fma_pivot_snapshot.diagonal[2 * multiplier_col];
        let d21 = fma_pivot_snapshot.diagonal[2 * multiplier_col + 1];
        let first_work =
            fma_pivot_snapshot.workspace[multiplier_col * APP_INNER_BLOCK_SIZE + multiplier_row];
        let second_work = fma_pivot_snapshot.workspace
            [(multiplier_col + 1) * APP_INNER_BLOCK_SIZE + multiplier_row];
        let reconstructed_first_multiplier = d11.mul_add(first_work, d21 * second_work);
        let source_first_multiplier = d11 * first_work + d21 * second_work;
        let trace_first_multiplier =
            fma_pivot_snapshot.matrix[multiplier_col * APP_INNER_BLOCK_SIZE + multiplier_row];
        let native_block_first_multiplier =
            native_block.matrix[multiplier_col * native_lda + final_multiplier_row];
        assert_eq!(
            reconstructed_first_multiplier.to_bits(),
            trace_first_multiplier.to_bits(),
            "dense seed09 FMA pivot-19 first-row multiplier reconstruction moved"
        );
        assert_eq!(
            (
                trace_first_multiplier.to_bits(),
                source_first_multiplier.to_bits(),
                native_block_first_multiplier.to_bits()
            ),
            (
                0xbf8c_bfa8_da67_4b6c,
                0xbf8c_bfa8_da67_4b6c,
                0xbf8c_bfa8_da67_4b6c,
            ),
            "dense seed09 FMA pivot-19 first-row multiplier boundary moved"
        );
        let first_source_multiplier_continuation_mismatch =
            first_native_block_continuation_mismatch(
                shim,
                &native_source_multiplier_trace,
                &native_block,
                native_lda,
                options,
            );
        assert_eq!(
            first_source_multiplier_continuation_mismatch,
            Some(BlockContinuationMismatch {
                step: 0,
                from: 0,
                status: 2,
                next: 2,
                component: "diagonal",
                index: 7,
                row: 0,
                col: 0,
                continued_bits: 0xbf2b_4429_642a_1ee2,
                block_bits: 0xbf2b_4429_642a_1ee4,
            }),
            "dense seed09 source-multiplier native APP trace continuation boundary moved"
        );
        assert_eq!(
            native_source_multiplier_trace_block.perm, native_block.perm,
            "dense seed09 source-multiplier native APP trace/block_ldlt permutation mismatch"
        );
        assert_eq!(
            native_source_multiplier_trace_block.local_perm, native_block.local_perm,
            "dense seed09 source-multiplier native APP trace/block_ldlt local permutation mismatch"
        );
        let mut first_source_multiplier_trace_d_mismatch = None;
        for (index, (&trace_value, &block_value)) in native_source_multiplier_trace_block
            .diagonal
            .iter()
            .zip(&native_block.diagonal)
            .enumerate()
        {
            if trace_value.to_bits() != block_value.to_bits() {
                first_source_multiplier_trace_d_mismatch =
                    Some((index, trace_value.to_bits(), block_value.to_bits()));
                break;
            }
        }
        assert_eq!(
            first_source_multiplier_trace_d_mismatch,
            Some((7, 0xbf2b_4429_642a_1ee2, 0xbf2b_4429_642a_1ee4)),
            "dense seed09 source-multiplier native APP trace/block_ldlt D boundary moved"
        );
        let mut first_source_multiplier_trace_block_mismatch = None;
        'source_multiplier_trace_block_compare: for col in block_start..block_end {
            for row in col..block_end {
                let trace_bits = native_source_multiplier_trace_block.matrix
                    [col * APP_INNER_BLOCK_SIZE + row]
                    .to_bits();
                let block_bits = native_block.matrix[col * native_lda + row].to_bits();
                if trace_bits != block_bits {
                    first_source_multiplier_trace_block_mismatch =
                        Some((row, col, trace_bits, block_bits));
                    break 'source_multiplier_trace_block_compare;
                }
            }
        }
        assert_eq!(
            first_source_multiplier_trace_block_mismatch,
            Some((2, 0, 0x3f79_e327_dcf6_7cce, 0x3f79_e327_dcf6_7ccf)),
            "dense seed09 source-multiplier native APP trace/block_ldlt matrix boundary moved"
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
        let first_source_continuation_mismatch = first_native_block_continuation_mismatch(
            shim,
            &native_source_trace,
            &native_block,
            native_lda,
            options,
        );
        assert_eq!(
            first_source_continuation_mismatch,
            Some(BlockContinuationMismatch {
                step: 0,
                from: 0,
                status: 2,
                next: 2,
                component: "diagonal",
                index: 7,
                row: 0,
                col: 0,
                continued_bits: 0xbf2b_4429_642a_1ee2,
                block_bits: 0xbf2b_4429_642a_1ee4,
            }),
            "dense seed09 source native APP trace continuation boundary moved"
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
            first_aligned_source_diagonal_mismatch, None,
            "dense seed09 source-shaped aligned first APP diagonal operand mismatch"
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
            first_source_diagonal_mismatch, None,
            "dense seed09 source-shaped first APP diagonal operand mismatch"
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
            first_source_trsm_mismatch, None,
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
            first_source_apply_mismatch, None,
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
            &dense_restore_backup,
            dimension,
            AppRestoreRange {
                backup_start: block_start,
                block_end,
                trailing_start: accepted_end,
            },
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
        let Some(native) = native_spral_or_skip() else {
            return;
        };
        let mut native_session = native
            .analyse_with_options_and_ordering(matrix, &options, NativeOrdering::Natural)
            .expect("native analyse");
        native_session.factorize(matrix).expect("native factorize");
        let native_enquiry = native_session.enquire_indef().expect("native enquire");
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
        let native_tail_entries = &native_enquiry.inverse_diagonal_entries[accepted_end..];
        assert_eq!(
            native_tail_entries.len(),
            factor_node_entries.len(),
            "dense seed09 native production and factor-node replay tail lengths differ"
        );
        let mut first_native_production_factor_node_mismatch = None;
        'native_production_factor_node_compare: for (
            local_pivot,
            (native_entry, factor_node_entry),
        ) in native_tail_entries
            .iter()
            .zip(&factor_node_entries)
            .enumerate()
        {
            for component in 0..2 {
                let native_bits = native_entry[component].to_bits();
                let factor_node_bits = factor_node_entry[component].to_bits();
                if native_bits != factor_node_bits {
                    first_native_production_factor_node_mismatch = Some((
                        accepted_end + local_pivot,
                        component,
                        native_bits,
                        factor_node_bits,
                    ));
                    break 'native_production_factor_node_compare;
                }
            }
        }
        assert_eq!(
            first_native_production_factor_node_mismatch, None,
            "dense seed09 native production-vs-factor-node TPP replay boundary moved"
        );
    }
}
