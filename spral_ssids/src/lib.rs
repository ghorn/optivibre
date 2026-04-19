use std::collections::BTreeSet;

use metis_ordering::{
    CsrGraph, NestedDissectionOptions, OrderingError, Permutation,
    approximate_minimum_degree_order, nested_dissection_order,
};
use rayon::prelude::*;
use thiserror::Error;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericFactorOptions {
    pub pivot_regularization: f64,
    pub inertia_zero_tol: f64,
    pub two_by_two_pivot_threshold: f64,
}

impl Default for NumericFactorOptions {
    fn default() -> Self {
        Self {
            pivot_regularization: 1e-9,
            inertia_zero_tol: 1e-10,
            two_by_two_pivot_threshold: (1.0 + 17.0_f64.sqrt()) / 8.0,
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
    pub regularized_pivots: usize,
    pub two_by_two_pivots: usize,
    pub delayed_pivots: usize,
    pub min_abs_pivot: f64,
    pub max_abs_pivot: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FactorInfo {
    pub factorization_residual_max_abs: f64,
    pub regularized_pivots: usize,
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
    start: usize,
    size: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct DiagonalBlockValue {
    block: DiagonalBlock,
    values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SymbolicFront {
    start_column: usize,
    end_column: usize,
    interface_rows: Vec<usize>,
    parent: Option<usize>,
    children: Vec<usize>,
}

impl SymbolicFront {
    fn width(&self) -> usize {
        self.end_column - self.start_column
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SymbolicFrontTree {
    fronts: Vec<SymbolicFront>,
    roots: Vec<usize>,
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
    front_count: usize,
    max_front_size: usize,
    contribution_storage_bytes: usize,
    delayed_front_propagations: usize,
    stored_nnz: usize,
    factor_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct FactorBlockRecord {
    size: usize,
    values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct FrontFactorizationResult {
    factor_order: Vec<usize>,
    factor_columns: Vec<FactorColumn>,
    block_records: Vec<FactorBlockRecord>,
    contribution: ContributionBlock,
    stats: PanelFactorStats,
    max_front_size: usize,
    contribution_storage_bytes: usize,
    delayed_front_propagations: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct DenseFrontFactorization {
    factor_order: Vec<usize>,
    factor_columns: Vec<FactorColumn>,
    block_records: Vec<FactorBlockRecord>,
    contribution: ContributionBlock,
    stats: PanelFactorStats,
    delayed_front_propagations: usize,
}

struct NumericFactorBuffers<'a> {
    factor_order: &'a mut Vec<usize>,
    factor_inverse: &'a mut Vec<usize>,
    dense_lower: &'a mut Vec<f64>,
    diagonal_blocks: &'a mut Vec<DiagonalBlockValue>,
    dense_matrix_scratch: &'a mut Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericFactor {
    dimension: usize,
    permutation: Permutation,
    pattern_col_ptrs: Vec<usize>,
    pattern_row_indices: Vec<usize>,
    diagonal_blocks: Vec<DiagonalBlockValue>,
    inertia: Inertia,
    pivot_stats: PivotStats,
    options: NumericFactorOptions,
    factor_order: Vec<usize>,
    factor_inverse: Vec<usize>,
    dense_lower: Vec<f64>,
    symbolic_front_tree: SymbolicFrontTree,
    dense_matrix_scratch: Vec<f64>,
    front_count_cached: usize,
    max_front_size_cached: usize,
    contribution_storage_bytes_cached: usize,
    delayed_front_propagations_cached: usize,
    symbolic_reuse_cached: bool,
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

    pub fn front_count(&self) -> usize {
        self.front_count_cached
    }

    pub fn max_front_size(&self) -> usize {
        self.max_front_size_cached
    }

    pub fn contribution_storage_bytes(&self) -> usize {
        self.contribution_storage_bytes_cached
    }

    pub fn delayed_front_propagations(&self) -> usize {
        self.delayed_front_propagations_cached
    }

    pub fn reused_symbolic_structure(&self) -> bool {
        self.symbolic_reuse_cached
    }

    pub fn uses_multifrontal_backend(&self) -> bool {
        true
    }

    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes_cached
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SsidsError> {
        let mut solution = rhs.to_vec();
        self.solve_in_place(&mut solution)?;
        Ok(solution)
    }

    /// Solve `Ax = rhs` in place for the factorized matrix. The slice length
    /// must match the factor dimension exactly.
    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), SsidsError> {
        if rhs.len() != self.dimension {
            return Err(SsidsError::SolveDimensionMismatch {
                expected: self.dimension,
                actual: rhs.len(),
            });
        }
        if self.dimension == 0 {
            return Ok(());
        }

        let mut permuted_rhs = vec![0.0; self.dimension];
        for (ordered, &original) in self.permutation.perm().iter().enumerate() {
            permuted_rhs[ordered] = rhs[original];
        }

        let mut factor_rhs = vec![0.0; self.dimension];
        for (factor_position, &ordered_index) in self.factor_order.iter().enumerate() {
            factor_rhs[factor_position] = permuted_rhs[ordered_index];
        }

        for pivot in 0..self.dimension {
            let pivot_value = factor_rhs[pivot];
            for (row, rhs_value) in factor_rhs.iter_mut().enumerate().skip(pivot + 1) {
                *rhs_value -= self.dense_lower[row * self.dimension + pivot] * pivot_value;
            }
        }

        let mut z = factor_rhs;
        for block in &self.diagonal_blocks {
            let start = block.block.start;
            let end = start + block.block.size;
            if block.block.size == 1 {
                let diagonal = block.values[0];
                if !diagonal.is_finite() || diagonal.abs() < f64::EPSILON {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: start,
                        detail: "diagonal pivot vanished during solve".into(),
                    });
                }
                z[start] /= diagonal;
            } else {
                solve_dense_block_in_place(&block.values, block.block.size, &mut z[start..end])
                    .map_err(|detail| SsidsError::NumericalBreakdown {
                        pivot: start,
                        detail,
                    })?;
            }
        }

        let mut factor_solution = z;
        for pivot in (0..self.dimension).rev() {
            let mut value = factor_solution[pivot];
            for (row, &row_value) in factor_solution.iter().enumerate().skip(pivot + 1) {
                value -= self.dense_lower[row * self.dimension + pivot] * row_value;
            }
            factor_solution[pivot] = value;
        }

        let mut ordered_solution = vec![0.0; self.dimension];
        for (factor_position, &ordered_index) in self.factor_order.iter().enumerate() {
            ordered_solution[ordered_index] = factor_solution[factor_position];
        }
        if !ordered_solution.iter().all(|value| value.is_finite()) {
            return Err(SsidsError::NumericalBreakdown {
                pivot: self.dimension.saturating_sub(1),
                detail: "solve produced non-finite values".into(),
            });
        }
        for (ordered, &original) in self.permutation.perm().iter().enumerate() {
            rhs[original] = ordered_solution[ordered];
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
        self.refactorize_with_cached_symbolic(matrix, true)
    }

    fn refactorize_with_cached_symbolic(
        &mut self,
        matrix: SymmetricCscMatrix<'_>,
        reused_symbolic: bool,
    ) -> Result<FactorInfo, SsidsError> {
        let factorization = multifrontal_factorize_with_tree(
            matrix,
            &self.permutation,
            &self.symbolic_front_tree,
            self.options,
            NumericFactorBuffers {
                factor_order: &mut self.factor_order,
                factor_inverse: &mut self.factor_inverse,
                dense_lower: &mut self.dense_lower,
                diagonal_blocks: &mut self.diagonal_blocks,
                dense_matrix_scratch: &mut self.dense_matrix_scratch,
            },
        )?;
        let info = FactorInfo {
            factorization_residual_max_abs: factorization.factorization_residual_max_abs,
            regularized_pivots: factorization.pivot_stats.regularized_pivots,
        };
        self.inertia =
            inertia_from_blocks(&[], &self.diagonal_blocks, self.options.inertia_zero_tol);
        self.pivot_stats = factorization.pivot_stats;
        self.front_count_cached = factorization.front_count;
        self.max_front_size_cached = factorization.max_front_size;
        self.contribution_storage_bytes_cached = factorization.contribution_storage_bytes;
        self.delayed_front_propagations_cached = factorization.delayed_front_propagations;
        self.symbolic_reuse_cached = reused_symbolic;
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

pub fn analyse(
    matrix: SymmetricCscMatrix<'_>,
    options: &SsidsOptions,
) -> Result<(SymbolicFactor, AnalyseInfo), SsidsError> {
    let graph =
        CsrGraph::from_symmetric_csc(matrix.dimension(), matrix.col_ptrs(), matrix.row_indices())?;
    match options.ordering {
        OrderingStrategy::Natural => {
            let permutation = Permutation::identity(matrix.dimension());
            let (elimination_tree, column_counts, column_pattern) = symbolic_factor_pattern(&graph);
            Ok(build_symbolic_result(
                permutation,
                elimination_tree,
                column_counts,
                column_pattern,
                "natural",
            ))
        }
        OrderingStrategy::ApproximateMinimumDegree => {
            let summary = approximate_minimum_degree_order(&graph)?;
            let permuted_graph = permute_graph(&graph, &summary.permutation);
            let (elimination_tree, column_counts, column_pattern) =
                symbolic_factor_pattern(&permuted_graph);
            Ok(build_symbolic_result(
                summary.permutation,
                elimination_tree,
                column_counts,
                column_pattern,
                "approximate_minimum_degree",
            ))
        }
        OrderingStrategy::NestedDissection(ordering_options) => {
            let summary = nested_dissection_order(&graph, &ordering_options)?;
            let permuted_graph = permute_graph(&graph, &summary.permutation);
            let (elimination_tree, column_counts, column_pattern) =
                symbolic_factor_pattern(&permuted_graph);
            Ok(build_symbolic_result(
                summary.permutation,
                elimination_tree,
                column_counts,
                column_pattern,
                "nested_dissection",
            ))
        }
        OrderingStrategy::Auto(ordering_options) => {
            let natural_permutation = Permutation::identity(matrix.dimension());
            let (natural_tree, natural_counts, natural_pattern) = symbolic_factor_pattern(&graph);
            let natural_fill = natural_counts.iter().sum::<usize>();

            let amd_summary = approximate_minimum_degree_order(&graph)?;
            let amd_graph = permute_graph(&graph, &amd_summary.permutation);
            let (amd_tree, amd_counts, amd_pattern) = symbolic_factor_pattern(&amd_graph);
            let amd_fill = amd_counts.iter().sum::<usize>();

            let summary = nested_dissection_order(&graph, &ordering_options)?;
            let permuted_graph = permute_graph(&graph, &summary.permutation);
            let (nd_tree, nd_counts, nd_pattern) = symbolic_factor_pattern(&permuted_graph);
            let nd_fill = nd_counts.iter().sum::<usize>();

            if amd_fill <= natural_fill && amd_fill <= nd_fill {
                Ok(build_symbolic_result(
                    amd_summary.permutation,
                    amd_tree,
                    amd_counts,
                    amd_pattern,
                    "auto_approximate_minimum_degree",
                ))
            } else if nd_fill <= natural_fill {
                Ok(build_symbolic_result(
                    summary.permutation,
                    nd_tree,
                    nd_counts,
                    nd_pattern,
                    "auto_nested_dissection",
                ))
            } else {
                Ok(build_symbolic_result(
                    natural_permutation,
                    natural_tree,
                    natural_counts,
                    natural_pattern,
                    "auto_natural",
                ))
            }
        }
    }
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
        inertia: Inertia {
            positive: 0,
            negative: 0,
            zero: 0,
        },
        pivot_stats: PivotStats {
            regularized_pivots: 0,
            two_by_two_pivots: 0,
            delayed_pivots: 0,
            min_abs_pivot: 0.0,
            max_abs_pivot: 0.0,
        },
        options: *options,
        factor_order: Vec::with_capacity(matrix.dimension()),
        factor_inverse: Vec::with_capacity(matrix.dimension()),
        dense_lower: Vec::with_capacity(matrix.dimension() * matrix.dimension()),
        symbolic_front_tree: front_tree,
        dense_matrix_scratch: Vec::with_capacity(matrix.dimension() * matrix.dimension()),
        front_count_cached: 0,
        max_front_size_cached: 0,
        contribution_storage_bytes_cached: 0,
        delayed_front_propagations_cached: 0,
        symbolic_reuse_cached: false,
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
    let info = factor.refactorize_with_cached_symbolic(matrix, false)?;
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

fn build_supernodes(
    elimination_tree: &[Option<usize>],
    column_pattern: &[Vec<usize>],
) -> Vec<Supernode> {
    let mut supernodes = Vec::new();
    let mut column = 0;
    while column < column_pattern.len() {
        let mut end = column + 1;
        while end < column_pattern.len()
            && elimination_tree[end - 1] == Some(end)
            && column_pattern[end - 1][1..] == column_pattern[end][..]
        {
            end += 1;
        }
        let trailing_rows = column_pattern[end - 1]
            .iter()
            .copied()
            .skip(1)
            .filter(|&row| row >= end)
            .collect::<Vec<_>>();
        supernodes.push(Supernode {
            start_column: column,
            end_column: end,
            trailing_rows,
        });
        column = end;
    }
    supernodes
}

fn build_symbolic_result(
    permutation: Permutation,
    elimination_tree: Vec<Option<usize>>,
    column_counts: Vec<usize>,
    column_pattern: Vec<Vec<usize>>,
    ordering_kind: &'static str,
) -> (SymbolicFactor, AnalyseInfo) {
    let supernodes = build_supernodes(&elimination_tree, &column_pattern);
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
                start_column: supernode.start_column,
                end_column: supernode.end_column,
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
        .map(|front| front.start_column)
        .collect::<Vec<_>>();
    for front in &mut fronts {
        front.children.sort_by_key(|&child| start_columns[child]);
    }
    let roots = collect_root_fronts(&fronts);
    SymbolicFrontTree { fronts, roots }
}

fn collect_root_fronts(fronts: &[SymbolicFront]) -> Vec<usize> {
    fronts
        .iter()
        .enumerate()
        .filter_map(|(index, front)| front.parent.is_none().then_some(index))
        .collect()
}

fn fill_permuted_dense_matrix_from_csc(
    matrix: SymmetricCscMatrix<'_>,
    permutation: &Permutation,
    dense: &mut Vec<f64>,
) -> Result<(), SsidsError> {
    let values = matrix.values().ok_or(SsidsError::MissingValues)?;
    let dimension = matrix.dimension();
    let inverse = permutation.inverse();
    dense.clear();
    dense.resize(dimension * dimension, 0.0);
    for col in 0..dimension {
        let start = matrix.col_ptrs()[col];
        let end = matrix.col_ptrs()[col + 1];
        for (&row, &value) in matrix.row_indices()[start..end]
            .iter()
            .zip(values[start..end].iter())
        {
            if !value.is_finite() {
                return Err(SsidsError::InvalidMatrix(format!(
                    "numeric value at ({row}, {col}) is not finite"
                )));
            }
            let permuted_col = inverse[col];
            let permuted_row = inverse[row];
            dense[permuted_row * dimension + permuted_col] += value;
            if permuted_row != permuted_col {
                dense[permuted_col * dimension + permuted_row] += value;
            }
        }
    }
    Ok(())
}

fn dense_symmetric_swap(matrix: &mut [f64], size: usize, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    for column in 0..size {
        matrix.swap(lhs * size + column, rhs * size + column);
    }
    for row in 0..size {
        matrix.swap(row * size + lhs, row * size + rhs);
    }
}

fn dense_block_to_packed_lower(values: &[f64], size: usize) -> Vec<f64> {
    let mut packed = vec![0.0; size * (size + 1) / 2];
    for row in 0..size {
        for col in 0..=row {
            diagonal_block_set(&mut packed, size, row, col, values[row * size + col]);
        }
    }
    packed
}

fn aggregate_panel_stats(target: &mut PanelFactorStats, source: PanelFactorStats) {
    target.regularized_pivots += source.regularized_pivots;
    target.two_by_two_pivots += source.two_by_two_pivots;
    target.delayed_pivots += source.delayed_pivots;
    target.min_abs_pivot = target.min_abs_pivot.min(source.min_abs_pivot);
    target.max_abs_pivot = target.max_abs_pivot.max(source.max_abs_pivot);
    target.max_residual = target.max_residual.max(source.max_residual);
}

fn dense_factor_one_by_one(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    options: NumericFactorOptions,
    stats: &mut PanelFactorStats,
) -> Result<(FactorColumn, FactorBlockRecord), SsidsError> {
    let diagonal_index = pivot * size + pivot;
    let original_diagonal = matrix[diagonal_index];
    if !original_diagonal.is_finite() {
        return Err(SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail: "diagonal pivot became non-finite".into(),
        });
    }
    let mut diagonal = original_diagonal;
    if diagonal.abs() < options.pivot_regularization {
        stats.regularized_pivots += 1;
        diagonal = if diagonal.is_sign_negative() {
            -options.pivot_regularization
        } else {
            options.pivot_regularization
        };
    }
    if diagonal.abs() < f64::EPSILON {
        return Err(SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail: "diagonal pivot is numerically zero".into(),
        });
    }
    matrix[diagonal_index] = diagonal;
    stats.max_residual = stats.max_residual.max((diagonal - original_diagonal).abs());
    let abs_pivot = diagonal.abs();
    stats.min_abs_pivot = stats.min_abs_pivot.min(abs_pivot);
    stats.max_abs_pivot = stats.max_abs_pivot.max(abs_pivot);

    let mut entries = Vec::new();
    for row in (pivot + 1)..size {
        let entry_index = row * size + pivot;
        let value = matrix[entry_index] / diagonal;
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
        matrix[pivot * size + row] = value;
        if value != 0.0 {
            entries.push((rows[row], value));
        }
    }

    for row in (pivot + 1)..size {
        let l_row = matrix[row * size + pivot];
        if l_row == 0.0 {
            continue;
        }
        for col in (pivot + 1)..=row {
            let updated = matrix[row * size + col] - l_row * diagonal * matrix[col * size + pivot];
            matrix[row * size + col] = updated;
            matrix[col * size + row] = updated;
        }
    }

    Ok((
        FactorColumn {
            global_column: rows[pivot],
            entries,
        },
        FactorBlockRecord {
            size: 1,
            values: vec![diagonal],
        },
    ))
}

fn choose_two_by_two_partner(
    matrix: &[f64],
    size: usize,
    pivot: usize,
    active_candidate_end: usize,
    options: NumericFactorOptions,
) -> Option<usize> {
    let diagonal = matrix[pivot * size + pivot].abs();
    let mut max_offdiag = 0.0_f64;
    let mut partner = None;
    for candidate in (pivot + 1)..active_candidate_end {
        let coupling = matrix[candidate * size + pivot].abs();
        if coupling > max_offdiag {
            max_offdiag = coupling;
            partner = Some(candidate);
        }
    }
    if max_offdiag <= options.pivot_regularization {
        return None;
    }
    if diagonal >= options.two_by_two_pivot_threshold * max_offdiag {
        return None;
    }
    partner
}

fn dense_factor_two_by_two(
    rows: &[usize],
    matrix: &mut [f64],
    size: usize,
    pivot: usize,
    options: NumericFactorOptions,
    stats: &mut PanelFactorStats,
) -> Result<([FactorColumn; 2], FactorBlockRecord), SsidsError> {
    let mut values = vec![
        matrix[pivot * size + pivot],
        matrix[(pivot + 1) * size + pivot],
        matrix[(pivot + 1) * size + pivot + 1],
    ];
    let (inverse, regularized, max_shift) = stabilized_dense_block_inverse(&mut values, 2, options)
        .map_err(|detail| SsidsError::NumericalBreakdown {
            pivot: rows[pivot],
            detail,
        })?;
    stats.regularized_pivots += regularized;
    stats.two_by_two_pivots += 1;
    stats.max_residual = stats.max_residual.max(max_shift);
    for eigenvalue in jacobi_eigenvalues(&values, 2) {
        let abs_pivot = eigenvalue.abs();
        stats.min_abs_pivot = stats.min_abs_pivot.min(abs_pivot);
        stats.max_abs_pivot = stats.max_abs_pivot.max(abs_pivot);
    }

    let d11 = values[0];
    let d21 = values[1];
    let d22 = values[2];
    matrix[pivot * size + pivot] = d11;
    matrix[(pivot + 1) * size + pivot] = d21;
    matrix[pivot * size + pivot + 1] = d21;
    matrix[(pivot + 1) * size + pivot + 1] = d22;

    let inv11 = inverse[0];
    let inv12 = inverse[1];
    let inv22 = inverse[3];
    let mut first_entries = Vec::new();
    let mut second_entries = Vec::new();
    for row in (pivot + 2)..size {
        let b1 = matrix[row * size + pivot];
        let b2 = matrix[row * size + pivot + 1];
        let l1 = b1 * inv11 + b2 * inv12;
        let l2 = b1 * inv12 + b2 * inv22;
        if !l1.is_finite() || !l2.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: rows[pivot],
                detail: "two-by-two multipliers became non-finite".into(),
            });
        }
        matrix[row * size + pivot] = l1;
        matrix[pivot * size + row] = l1;
        matrix[row * size + pivot + 1] = l2;
        matrix[(pivot + 1) * size + row] = l2;
        if l1 != 0.0 {
            first_entries.push((rows[row], l1));
        }
        if l2 != 0.0 {
            second_entries.push((rows[row], l2));
        }
    }

    for row in (pivot + 2)..size {
        let l1_row = matrix[row * size + pivot];
        let l2_row = matrix[row * size + pivot + 1];
        for col in (pivot + 2)..=row {
            let l1_col = matrix[col * size + pivot];
            let l2_col = matrix[col * size + pivot + 1];
            let update = d11 * l1_row * l1_col
                + d21 * (l1_row * l2_col + l2_row * l1_col)
                + d22 * l2_row * l2_col;
            let updated = matrix[row * size + col] - update;
            matrix[row * size + col] = updated;
            matrix[col * size + row] = updated;
        }
    }

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
        FactorBlockRecord { size: 2, values },
    ))
}

fn factorize_dense_front(
    mut rows: Vec<usize>,
    candidate_len: usize,
    mut dense: Vec<f64>,
    options: NumericFactorOptions,
) -> Result<DenseFrontFactorization, SsidsError> {
    let size = rows.len();
    let mut stats = PanelFactorStats {
        min_abs_pivot: f64::INFINITY,
        ..PanelFactorStats::default()
    };
    let mut factor_order = Vec::new();
    let mut factor_columns = Vec::new();
    let mut block_records = Vec::new();
    let mut pivot = 0;
    let mut active_candidate_end = candidate_len.min(size);
    let mut delayed_front_propagations = 0;

    while pivot < active_candidate_end {
        let diagonal = dense[pivot * size + pivot].abs();
        let mut max_offdiag = 0.0_f64;
        for row in (pivot + 1)..size {
            max_offdiag = max_offdiag.max(dense[row * size + pivot].abs());
        }
        let one_by_one = max_offdiag <= options.pivot_regularization
            || diagonal >= options.two_by_two_pivot_threshold * max_offdiag;
        if one_by_one {
            let (column, block) =
                dense_factor_one_by_one(&rows, &mut dense, size, pivot, options, &mut stats)?;
            factor_order.push(rows[pivot]);
            factor_columns.push(column);
            block_records.push(block);
            pivot += 1;
            continue;
        }

        if let Some(partner) =
            choose_two_by_two_partner(&dense, size, pivot, active_candidate_end, options)
        {
            if partner != pivot + 1 {
                dense_symmetric_swap(&mut dense, size, partner, pivot + 1);
                rows.swap(partner, pivot + 1);
            }
            let (columns, block) =
                dense_factor_two_by_two(&rows, &mut dense, size, pivot, options, &mut stats)?;
            factor_order.push(rows[pivot]);
            factor_order.push(rows[pivot + 1]);
            factor_columns.push(columns[0].clone());
            factor_columns.push(columns[1].clone());
            block_records.push(block);
            pivot += 2;
            continue;
        }

        if active_candidate_end == 0 {
            break;
        }
        dense_symmetric_swap(&mut dense, size, pivot, active_candidate_end - 1);
        rows.swap(pivot, active_candidate_end - 1);
        active_candidate_end -= 1;
        stats.delayed_pivots += 1;
        delayed_front_propagations += 1;
    }

    if size == 0 || stats.min_abs_pivot == f64::INFINITY {
        stats.min_abs_pivot = 0.0;
    }

    let remaining_rows = rows[pivot..].to_vec();
    let remaining_size = remaining_rows.len();
    let delayed_count = candidate_len.saturating_sub(pivot).min(remaining_size);
    let mut contribution_dense = vec![0.0; remaining_size * remaining_size];
    for row in 0..remaining_size {
        for col in 0..remaining_size {
            contribution_dense[row * remaining_size + col] =
                dense[(pivot + row) * size + (pivot + col)];
        }
    }

    Ok(DenseFrontFactorization {
        factor_order,
        factor_columns,
        block_records,
        contribution: ContributionBlock {
            row_ids: remaining_rows,
            delayed_count,
            dense: contribution_dense,
        },
        stats,
        delayed_front_propagations,
    })
}

fn factor_front_recursive(
    front_id: usize,
    tree: &SymbolicFrontTree,
    dense_matrix: &[f64],
    dimension: usize,
    options: NumericFactorOptions,
) -> Result<FrontFactorizationResult, SsidsError> {
    let front = &tree.fronts[front_id];
    let child_results =
        if front.children.len() >= 2 && front.width() + front.interface_rows.len() >= 32 {
            let raw = front
                .children
                .par_iter()
                .map(|&child| factor_front_recursive(child, tree, dense_matrix, dimension, options))
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
                    dense_matrix,
                    dimension,
                    options,
                )?);
            }
            collected
        };

    let mut factor_order = Vec::new();
    let mut factor_columns = Vec::new();
    let mut block_records = Vec::new();
    let mut child_contributions = Vec::with_capacity(child_results.len());
    let mut stats = PanelFactorStats {
        min_abs_pivot: f64::INFINITY,
        ..PanelFactorStats::default()
    };
    let mut max_front_size = 0;
    let mut contribution_storage_bytes = 0;
    let mut delayed_front_propagations = 0;

    for child in child_results {
        factor_order.extend(child.factor_order);
        factor_columns.extend(child.factor_columns);
        block_records.extend(child.block_records);
        child_contributions.push(child.contribution);
        aggregate_panel_stats(&mut stats, child.stats);
        max_front_size = max_front_size.max(child.max_front_size);
        contribution_storage_bytes += child.contribution_storage_bytes;
        delayed_front_propagations += child.delayed_front_propagations;
    }

    let mut candidate_rows = (front.start_column..front.end_column).collect::<Vec<_>>();
    let mut all_rows = candidate_rows.iter().copied().collect::<BTreeSet<_>>();
    for &row in &front.interface_rows {
        all_rows.insert(row);
    }
    for contribution in &child_contributions {
        for &row in &contribution.row_ids {
            all_rows.insert(row);
        }
        for &row in contribution.row_ids.iter().take(contribution.delayed_count) {
            if !candidate_rows.contains(&row) {
                candidate_rows.push(row);
            }
        }
    }
    let mut interface_rows = all_rows
        .into_iter()
        .filter(|row| !candidate_rows.contains(row))
        .collect::<Vec<_>>();
    interface_rows.sort_unstable();
    let mut local_rows = candidate_rows;
    local_rows.extend(interface_rows);
    let local_size = local_rows.len();
    max_front_size = max_front_size.max(local_size);
    let mut local_dense = vec![0.0; local_size * local_size];
    let mut local_positions = vec![usize::MAX; dimension];
    for (position, &row) in local_rows.iter().enumerate() {
        local_positions[row] = position;
    }
    for column in front.start_column..front.end_column {
        let local_column = local_positions[column];
        for &row in &local_rows {
            if row < column {
                continue;
            }
            let value = dense_matrix[row * dimension + column];
            if value == 0.0 {
                continue;
            }
            let local_row = local_positions[row];
            local_dense[local_row * local_size + local_column] += value;
            if local_row != local_column {
                local_dense[local_column * local_size + local_row] += value;
            }
        }
    }
    for contribution in &child_contributions {
        let size = contribution.row_ids.len();
        for row in 0..size {
            let local_row = local_positions[contribution.row_ids[row]];
            for col in 0..size {
                let local_col = local_positions[contribution.row_ids[col]];
                local_dense[local_row * local_size + local_col] +=
                    contribution.dense[row * size + col];
            }
        }
    }

    let local = factorize_dense_front(local_rows, front.width(), local_dense, options)?;
    factor_order.extend(local.factor_order);
    factor_columns.extend(local.factor_columns);
    block_records.extend(local.block_records);
    aggregate_panel_stats(&mut stats, local.stats);
    contribution_storage_bytes += local.contribution.dense.len() * std::mem::size_of::<f64>();
    delayed_front_propagations += local.delayed_front_propagations;

    Ok(FrontFactorizationResult {
        factor_order,
        factor_columns,
        block_records,
        contribution: local.contribution,
        stats,
        max_front_size,
        contribution_storage_bytes,
        delayed_front_propagations,
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
    fill_permuted_dense_matrix_from_csc(matrix, permutation, buffers.dense_matrix_scratch)?;
    let dense_matrix = buffers.dense_matrix_scratch.as_slice();
    let root_results = if tree.roots.len() >= 2 && dimension >= 64 {
        let raw = tree
            .roots
            .par_iter()
            .map(|&root| factor_front_recursive(root, tree, dense_matrix, dimension, options))
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
                dense_matrix,
                dimension,
                options,
            )?);
        }
        collected
    };

    let mut factor_order = Vec::with_capacity(dimension);
    let mut factor_columns = Vec::with_capacity(dimension);
    let mut block_records = Vec::new();
    let mut stats = PanelFactorStats {
        min_abs_pivot: f64::INFINITY,
        ..PanelFactorStats::default()
    };
    let mut max_front_size = 0;
    let mut contribution_storage_bytes = 0;
    let mut delayed_front_propagations = 0;
    let mut pending_root_contributions = Vec::new();
    for result in root_results {
        factor_order.extend(result.factor_order);
        factor_columns.extend(result.factor_columns);
        block_records.extend(result.block_records);
        pending_root_contributions.push(result.contribution);
        aggregate_panel_stats(&mut stats, result.stats);
        max_front_size = max_front_size.max(result.max_front_size);
        contribution_storage_bytes += result.contribution_storage_bytes;
        delayed_front_propagations += result.delayed_front_propagations;
    }

    for contribution in pending_root_contributions {
        if contribution.row_ids.is_empty() {
            continue;
        }
        let size = contribution.row_ids.len();
        let mut values = dense_block_to_packed_lower(&contribution.dense, size);
        let (_, regularized, max_shift) =
            stabilized_dense_block_inverse(&mut values, size, options).map_err(|detail| {
                SsidsError::NumericalBreakdown {
                    pivot: contribution.row_ids[0],
                    detail,
                }
            })?;
        stats.regularized_pivots += regularized;
        stats.max_residual = stats.max_residual.max(max_shift);
        if size == 1 {
            let abs_pivot = values[0].abs();
            stats.min_abs_pivot = stats.min_abs_pivot.min(abs_pivot);
            stats.max_abs_pivot = stats.max_abs_pivot.max(abs_pivot);
        } else {
            stats.delayed_pivots += size;
            for eigenvalue in jacobi_eigenvalues(&values, size) {
                let abs_pivot = eigenvalue.abs();
                stats.min_abs_pivot = stats.min_abs_pivot.min(abs_pivot);
                stats.max_abs_pivot = stats.max_abs_pivot.max(abs_pivot);
            }
        }
        for &row in &contribution.row_ids {
            factor_order.push(row);
            factor_columns.push(FactorColumn {
                global_column: row,
                entries: Vec::new(),
            });
        }
        block_records.push(FactorBlockRecord { size, values });
    }

    if stats.min_abs_pivot == f64::INFINITY {
        stats.min_abs_pivot = 0.0;
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
    buffers.dense_lower.clear();
    buffers.dense_lower.resize(dimension * dimension, 0.0);
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
            buffers.dense_lower[row_position * dimension + column_position] = value;
        }
    }

    buffers.diagonal_blocks.clear();
    let mut start = 0;
    for block in block_records {
        buffers.diagonal_blocks.push(DiagonalBlockValue {
            block: DiagonalBlock {
                start,
                size: block.size,
            },
            values: block.values,
        });
        start += buffers
            .diagonal_blocks
            .last()
            .map(|block| block.block.size)
            .unwrap_or(0);
    }

    let stored_nnz = dimension
        + factor_columns
            .iter()
            .map(|column| column.entries.len())
            .sum::<usize>()
        + buffers
            .diagonal_blocks
            .iter()
            .map(|block| block.values.len().saturating_sub(block.block.size))
            .sum::<usize>();
    let factor_bytes = std::mem::size_of::<f64>()
        * (buffers.dense_lower.len()
            + buffers
                .diagonal_blocks
                .iter()
                .map(|block| block.values.len())
                .sum::<usize>())
        + std::mem::size_of::<usize>()
            * (buffers.factor_order.len()
                + buffers.factor_inverse.len()
                + tree
                    .fronts
                    .iter()
                    .map(|front| 4 + front.interface_rows.len() + front.children.len())
                    .sum::<usize>());

    Ok(MultifrontalFactorizationOutcome {
        pivot_stats: PivotStats {
            regularized_pivots: stats.regularized_pivots,
            two_by_two_pivots: stats.two_by_two_pivots,
            delayed_pivots: stats.delayed_pivots,
            min_abs_pivot: stats.min_abs_pivot,
            max_abs_pivot: stats.max_abs_pivot,
        },
        factorization_residual_max_abs: stats.max_residual,
        front_count: tree.fronts.len(),
        max_front_size,
        contribution_storage_bytes,
        delayed_front_propagations,
        stored_nnz,
        factor_bytes,
    })
}

fn diagonal_block_index(row: usize, col: usize) -> usize {
    row * (row + 1) / 2 + col
}

fn diagonal_block_get(values: &[f64], size: usize, row: usize, col: usize) -> f64 {
    debug_assert!(row < size && col < size);
    let (row, col) = if row >= col { (row, col) } else { (col, row) };
    values[diagonal_block_index(row, col)]
}

fn diagonal_block_set(values: &mut [f64], size: usize, row: usize, col: usize, value: f64) {
    debug_assert!(row < size && col < size);
    let (row, col) = if row >= col { (row, col) } else { (col, row) };
    values[diagonal_block_index(row, col)] = value;
}

fn dense_matrix_from_diagonal_block(values: &[f64], size: usize) -> Vec<f64> {
    let mut dense = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..=row {
            let value = values[diagonal_block_index(row, col)];
            dense[row * size + col] = value;
            dense[col * size + row] = value;
        }
    }
    dense
}

fn solve_dense_matrix_in_place(dense: &[f64], size: usize, rhs: &mut [f64]) -> Result<(), String> {
    let mut matrix = dense.to_vec();
    for pivot in 0..size {
        let mut pivot_row = pivot;
        let mut pivot_abs = matrix[pivot * size + pivot].abs();
        for row in (pivot + 1)..size {
            let candidate = matrix[row * size + pivot].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs < f64::EPSILON {
            return Err("dense diagonal block became singular during solve".into());
        }
        if pivot_row != pivot {
            for col in pivot..size {
                matrix.swap(pivot * size + col, pivot_row * size + col);
            }
            rhs.swap(pivot, pivot_row);
        }
        let pivot_value = matrix[pivot * size + pivot];
        for row in (pivot + 1)..size {
            let factor = matrix[row * size + pivot] / pivot_value;
            matrix[row * size + pivot] = 0.0;
            for col in (pivot + 1)..size {
                matrix[row * size + col] -= factor * matrix[pivot * size + col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    for row in (0..size).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..size {
            value -= matrix[row * size + col] * rhs[col];
        }
        let diagonal = matrix[row * size + row];
        if !diagonal.is_finite() || diagonal.abs() < f64::EPSILON {
            return Err("dense diagonal block became singular during back solve".into());
        }
        rhs[row] = value / diagonal;
    }
    Ok(())
}

fn solve_dense_block_in_place(values: &[f64], size: usize, rhs: &mut [f64]) -> Result<(), String> {
    solve_dense_matrix_in_place(&dense_matrix_from_diagonal_block(values, size), size, rhs)
}

fn invert_dense_block(values: &[f64], size: usize) -> Result<Vec<f64>, String> {
    let dense = dense_matrix_from_diagonal_block(values, size);
    let mut inverse = vec![0.0; size * size];
    let mut rhs = vec![0.0; size];
    for col in 0..size {
        rhs.fill(0.0);
        rhs[col] = 1.0;
        solve_dense_matrix_in_place(&dense, size, &mut rhs)?;
        for row in 0..size {
            inverse[row * size + col] = rhs[row];
        }
    }
    Ok(inverse)
}

fn stabilized_dense_block_inverse(
    values: &mut [f64],
    size: usize,
    options: NumericFactorOptions,
) -> Result<(Vec<f64>, usize, f64), String> {
    if size == 0 {
        return Ok((Vec::new(), 0, 0.0));
    }
    let mut applied_regularization = false;
    let mut max_shift = 0.0_f64;
    for attempt in 0..6 {
        if let Ok(inverse) = invert_dense_block(values, size) {
            return Ok((
                inverse,
                if applied_regularization { size } else { 0 },
                max_shift,
            ));
        }
        let shift = options.pivot_regularization * 10.0_f64.powi(attempt);
        for diagonal in 0..size {
            let current = diagonal_block_get(values, size, diagonal, diagonal);
            let adjusted = current
                + if current.is_sign_negative() {
                    -shift
                } else {
                    shift
                };
            diagonal_block_set(values, size, diagonal, diagonal, adjusted);
        }
        applied_regularization = true;
        max_shift = max_shift.max(shift);
    }
    Err("dense diagonal block remained singular after regularization".into())
}

fn jacobi_eigenvalues(values: &[f64], size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![values[0]];
    }
    let mut dense = dense_matrix_from_diagonal_block(values, size);
    let sweep_limit = 32 * size * size;
    for _ in 0..sweep_limit {
        let mut max_value = 0.0_f64;
        let mut pivot = (0, 1);
        for row in 0..size {
            for col in 0..row {
                let value = dense[row * size + col].abs();
                if value > max_value {
                    max_value = value;
                    pivot = (row, col);
                }
            }
        }
        if max_value <= 1e-12 {
            break;
        }
        let (p, q) = pivot;
        let app = dense[p * size + p];
        let aqq = dense[q * size + q];
        let apq = dense[p * size + q];
        if apq.abs() <= 1e-15 {
            continue;
        }
        let tau = (aqq - app) / (2.0 * apq);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        for k in 0..size {
            if k == p || k == q {
                continue;
            }
            let aik = dense[p * size + k];
            let akq = dense[q * size + k];
            let new_pk = cosine * aik - sine * akq;
            let new_qk = sine * aik + cosine * akq;
            dense[p * size + k] = new_pk;
            dense[k * size + p] = new_pk;
            dense[q * size + k] = new_qk;
            dense[k * size + q] = new_qk;
        }
        let new_pp = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        let new_qq = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        dense[p * size + p] = new_pp;
        dense[q * size + q] = new_qq;
        dense[p * size + q] = 0.0;
        dense[q * size + p] = 0.0;
    }
    (0..size).map(|index| dense[index * size + index]).collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PanelFactorStats {
    regularized_pivots: usize,
    two_by_two_pivots: usize,
    delayed_pivots: usize,
    min_abs_pivot: f64,
    max_abs_pivot: f64,
    max_residual: f64,
}

fn inertia_from_blocks(
    diag: &[f64],
    diagonal_blocks: &[DiagonalBlockValue],
    zero_tol: f64,
) -> Inertia {
    let mut inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    for block in diagonal_blocks {
        if block.block.size == 1 {
            let value = block.values[0];
            if value > zero_tol {
                inertia.positive += 1;
            } else if value < -zero_tol {
                inertia.negative += 1;
            } else {
                inertia.zero += 1;
            }
            continue;
        }
        for value in jacobi_eigenvalues(&block.values, block.block.size) {
            if value > zero_tol {
                inertia.positive += 1;
            } else if value < -zero_tol {
                inertia.negative += 1;
            } else {
                inertia.zero += 1;
            }
        }
    }
    if diagonal_blocks.is_empty() {
        for &value in diag {
            if value > zero_tol {
                inertia.positive += 1;
            } else if value < -zero_tol {
                inertia.negative += 1;
            } else {
                inertia.zero += 1;
            }
        }
    }
    inertia
}
