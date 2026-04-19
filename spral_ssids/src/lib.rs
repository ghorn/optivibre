use std::sync::OnceLock;

use libloading::{Library, Symbol};
use metis_ordering::{
    CsrGraph, NestedDissectionOptions, OrderingError, Permutation,
    approximate_minimum_degree_order, nested_dissection_order,
};
use rayon::prelude::*;
use thiserror::Error;

const DENSE_PANEL_MIN_SUPERNODE_WIDTH: usize = 3;
const PARALLEL_PANEL_UPDATE_MIN_COLUMNS: usize = 8;
const PARALLEL_PANEL_UPDATE_MIN_WORK: usize = 512;
const DEFERRED_DENSE_BLOCK_MIN_SIZE: usize = 3;
const BLAS_TRIANGULAR_SOLVE_MIN_WIDTH: usize = 16;
const BLAS_RANK1_UPDATE_MIN_WORK: usize = 4096;
const BLAS_SINGLE_COLUMN_UPDATE_MIN_WORK: usize = 2048;
const BLAS_BLOCK_UPDATE_MIN_WORK: usize = 4096;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;
const CBLAS_LOWER: i32 = 122;
const CBLAS_UNIT: i32 = 132;

type CblasDgerFn =
    unsafe extern "C" fn(i32, i32, i32, f64, *const f64, i32, *const f64, i32, *mut f64, i32);
type CblasDsyrFn = unsafe extern "C" fn(i32, i32, i32, f64, *const f64, i32, *mut f64, i32);
type CblasDgemmFn = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f64,
    *const f64,
    i32,
    *const f64,
    i32,
    f64,
    *mut f64,
    i32,
);
type CblasDtrsvFn = unsafe extern "C" fn(i32, i32, i32, i32, i32, *const f64, i32, *mut f64, i32);
type OpenBlasSetThreadsFn = unsafe extern "C" fn(i32);

struct BlasKernel {
    _library: Library,
    dger: CblasDgerFn,
    dsyr: CblasDsyrFn,
    dgemm: CblasDgemmFn,
    dtrsv: CblasDtrsvFn,
}

fn try_load_blas_kernel() -> Option<BlasKernel> {
    let candidates = [
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        "/opt/homebrew/lib/libopenblas.dylib",
        "libopenblas.dylib",
        "libopenblas.so",
    ];
    for candidate in candidates {
        let Ok(library) = (unsafe { Library::new(candidate) }) else {
            continue;
        };
        let dger = unsafe {
            let symbol: Symbol<'_, CblasDgerFn> = library.get(b"cblas_dger\0").ok()?;
            *symbol
        };
        let dsyr = unsafe {
            let symbol: Symbol<'_, CblasDsyrFn> = library.get(b"cblas_dsyr\0").ok()?;
            *symbol
        };
        let dgemm = unsafe {
            let symbol: Symbol<'_, CblasDgemmFn> = library.get(b"cblas_dgemm\0").ok()?;
            *symbol
        };
        let dtrsv = unsafe {
            let symbol: Symbol<'_, CblasDtrsvFn> = library.get(b"cblas_dtrsv\0").ok()?;
            *symbol
        };
        if let Ok(symbol) =
            unsafe { library.get::<OpenBlasSetThreadsFn>(b"openblas_set_num_threads\0") }
        {
            unsafe {
                symbol(1);
            }
        } else if let Ok(symbol) =
            unsafe { library.get::<OpenBlasSetThreadsFn>(b"goto_set_num_threads\0") }
        {
            unsafe {
                symbol(1);
            }
        }
        return Some(BlasKernel {
            _library: library,
            dger,
            dsyr,
            dgemm,
            dtrsv,
        });
    }
    None
}

fn blas_kernel() -> Option<&'static BlasKernel> {
    static BLAS: OnceLock<Option<BlasKernel>> = OnceLock::new();
    BLAS.get_or_init(try_load_blas_kernel).as_ref()
}

fn blas_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

fn try_blas_dtrsv_lower_unit(
    transposed: bool,
    matrix: &[f64],
    width: usize,
    rhs: &mut [f64],
) -> bool {
    if width == 0 {
        return true;
    }
    let Some(blas) = blas_kernel() else {
        return false;
    };
    let (Some(width_i32), Some(lda_i32)) = (blas_i32(width), blas_i32(width)) else {
        return false;
    };
    unsafe {
        (blas.dtrsv)(
            CBLAS_ROW_MAJOR,
            CBLAS_LOWER,
            if transposed {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            CBLAS_UNIT,
            width_i32,
            matrix.as_ptr(),
            lda_i32,
            rhs.as_mut_ptr(),
            1,
        );
    }
    true
}

fn try_blas_dsyr_lower_strided(
    n: usize,
    alpha: f64,
    x_ptr: *const f64,
    incx: usize,
    a_ptr: *mut f64,
    lda: usize,
) -> bool {
    if n == 0 {
        return true;
    }
    let Some(blas) = blas_kernel() else {
        return false;
    };
    let (Some(n_i32), Some(incx_i32), Some(lda_i32)) = (blas_i32(n), blas_i32(incx), blas_i32(lda))
    else {
        return false;
    };
    unsafe {
        (blas.dsyr)(
            CBLAS_ROW_MAJOR,
            CBLAS_LOWER,
            n_i32,
            alpha,
            x_ptr,
            incx_i32,
            a_ptr,
            lda_i32,
        );
    }
    true
}

struct DgerSpec {
    m: usize,
    n: usize,
    alpha: f64,
    x_ptr: *const f64,
    incx: usize,
    y_ptr: *const f64,
    incy: usize,
    a_ptr: *mut f64,
    lda: usize,
}

fn try_blas_dger_strided(spec: DgerSpec) -> bool {
    if spec.m == 0 || spec.n == 0 {
        return true;
    }
    let Some(blas) = blas_kernel() else {
        return false;
    };
    let (Some(m_i32), Some(n_i32), Some(incx_i32), Some(incy_i32), Some(lda_i32)) = (
        blas_i32(spec.m),
        blas_i32(spec.n),
        blas_i32(spec.incx),
        blas_i32(spec.incy),
        blas_i32(spec.lda),
    ) else {
        return false;
    };
    unsafe {
        (blas.dger)(
            CBLAS_ROW_MAJOR,
            m_i32,
            n_i32,
            spec.alpha,
            spec.x_ptr,
            incx_i32,
            spec.y_ptr,
            incy_i32,
            spec.a_ptr,
            lda_i32,
        );
    }
    true
}

struct GemmSpec<'a> {
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &'a [f64],
    lda: usize,
    b: &'a [f64],
    ldb: usize,
    beta: f64,
    c: &'a mut [f64],
    ldc: usize,
}

fn try_blas_dgemm_row_major(spec: GemmSpec<'_>) -> bool {
    if spec.m == 0 || spec.n == 0 || spec.k == 0 {
        return true;
    }
    let Some(blas) = blas_kernel() else {
        return false;
    };
    let (Some(m_i32), Some(n_i32), Some(k_i32), Some(lda_i32), Some(ldb_i32), Some(ldc_i32)) = (
        blas_i32(spec.m),
        blas_i32(spec.n),
        blas_i32(spec.k),
        blas_i32(spec.lda),
        blas_i32(spec.ldb),
        blas_i32(spec.ldc),
    ) else {
        return false;
    };
    unsafe {
        (blas.dgemm)(
            CBLAS_ROW_MAJOR,
            if spec.trans_a {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            if spec.trans_b {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            m_i32,
            n_i32,
            k_i32,
            spec.alpha,
            spec.a.as_ptr(),
            lda_i32,
            spec.b.as_ptr(),
            ldb_i32,
            spec.beta,
            spec.c.as_mut_ptr(),
            ldc_i32,
        );
    }
    true
}

#[derive(Clone, Copy, Debug)]
pub struct SymmetricCscMatrix<'a> {
    dimension: usize,
    col_ptrs: &'a [usize],
    row_indices: &'a [usize],
    values: Option<&'a [f64]>,
}

impl<'a> SymmetricCscMatrix<'a> {
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

#[derive(Clone, Debug, PartialEq)]
struct NumericSupernode {
    start_column: usize,
    end_column: usize,
    trailing_rows: Vec<usize>,
    diagonal_block: Vec<f64>,
    trailing_block: Vec<f64>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct NumericFactor {
    dimension: usize,
    permutation: Permutation,
    lower_col_ptrs: Vec<usize>,
    lower_row_indices: Vec<usize>,
    lower_values: Vec<f64>,
    row_to_columns: Vec<Vec<usize>>,
    supernodes: Vec<NumericSupernode>,
    diag: Vec<f64>,
    diagonal_blocks: Vec<DiagonalBlockValue>,
    inertia: Inertia,
    pivot_stats: PivotStats,
    options: NumericFactorOptions,
    workspace_buffer: Vec<f64>,
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
        self.dimension
            + self.lower_values.len()
            + self
                .diagonal_blocks
                .iter()
                .map(|block| block.values.len().saturating_sub(block.block.size))
                .sum::<usize>()
    }

    pub fn supernode_count(&self) -> usize {
        self.supernodes.len()
    }

    pub fn max_supernode_width(&self) -> usize {
        self.supernodes
            .iter()
            .map(|supernode| supernode.end_column - supernode.start_column)
            .max()
            .unwrap_or(0)
    }

    pub fn factor_bytes(&self) -> usize {
        let numeric_value_bytes = std::mem::size_of::<f64>()
            * (self.diag.len()
                + self.lower_values.len()
                + self
                    .diagonal_blocks
                    .iter()
                    .map(|block| block.values.len())
                    .sum::<usize>()
                + self
                    .supernodes
                    .iter()
                    .map(|supernode| {
                        supernode.diagonal_block.len() + supernode.trailing_block.len()
                    })
                    .sum::<usize>());
        let index_bytes = std::mem::size_of::<usize>()
            * (self.lower_col_ptrs.len()
                + self.lower_row_indices.len()
                + self
                    .supernodes
                    .iter()
                    .map(|supernode| 2 + supernode.trailing_rows.len())
                    .sum::<usize>());
        numeric_value_bytes + index_bytes
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SsidsError> {
        let mut solution = rhs.to_vec();
        self.solve_in_place(&mut solution)?;
        Ok(solution)
    }

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

        let mut y = vec![0.0; self.dimension];
        y.copy_from_slice(&permuted_rhs);
        for supernode in &self.supernodes {
            let width = supernode.end_column - supernode.start_column;
            let block_start = supernode.start_column;
            let block_end = block_start + width;
            {
                let y_block = &mut y[block_start..block_end];
                let used_blas = width >= BLAS_TRIANGULAR_SOLVE_MIN_WIDTH
                    && try_blas_dtrsv_lower_unit(false, &supernode.diagonal_block, width, y_block);
                if !used_blas {
                    for local_col in 0..width {
                        let y_col = y_block[local_col];
                        for (local_row, y_value) in
                            y_block.iter_mut().enumerate().skip(local_col + 1)
                        {
                            *y_value -= supernode.diagonal_block
                                [supernode_block_index(local_row, local_col, width)]
                                * y_col;
                        }
                    }
                }
            }
            for (trailing_offset, &row) in supernode.trailing_rows.iter().enumerate() {
                let mut update = 0.0;
                for local_col in 0..width {
                    let y_col = y[block_start + local_col];
                    update += supernode.trailing_block
                        [supernode_block_index(trailing_offset, local_col, width)]
                        * y_col;
                }
                y[row] -= update;
            }
        }

        let mut z = vec![0.0; self.dimension];
        for block in &self.diagonal_blocks {
            if block.block.size == 1 {
                let row = block.block.start;
                let diag = block.values[0];
                if !diag.is_finite() || diag.abs() < f64::EPSILON {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: row,
                        detail: "diagonal pivot vanished during solve".into(),
                    });
                }
                z[row] = y[row] / diag;
                continue;
            }
            let start = block.block.start;
            let end = start + block.block.size;
            z[start..end].copy_from_slice(&y[start..end]);
            solve_dense_block_in_place(&block.values, block.block.size, &mut z[start..end])
                .map_err(|detail| SsidsError::NumericalBreakdown {
                    pivot: start,
                    detail,
                })?;
        }

        let mut permuted_solution = z;
        for supernode in self.supernodes.iter().rev() {
            let width = supernode.end_column - supernode.start_column;
            let block_start = supernode.start_column;
            let block_end = block_start + width;
            let trailing_solution = supernode
                .trailing_rows
                .iter()
                .map(|&row| permuted_solution[row])
                .collect::<Vec<_>>();
            {
                let solution_block = &mut permuted_solution[block_start..block_end];
                for (local_col, value_slot) in solution_block.iter_mut().enumerate() {
                    let mut value = *value_slot;
                    for (trailing_offset, &trailing_value) in trailing_solution.iter().enumerate() {
                        value -= supernode.trailing_block
                            [supernode_block_index(trailing_offset, local_col, width)]
                            * trailing_value;
                    }
                    *value_slot = value;
                }
                let used_blas = width >= BLAS_TRIANGULAR_SOLVE_MIN_WIDTH
                    && try_blas_dtrsv_lower_unit(
                        true,
                        &supernode.diagonal_block,
                        width,
                        solution_block,
                    );
                if !used_blas {
                    for local_col in (0..width).rev() {
                        let mut value = solution_block[local_col];
                        for (offset, &row_value) in
                            solution_block[(local_col + 1)..].iter().enumerate()
                        {
                            let local_row = local_col + 1 + offset;
                            value -= supernode.diagonal_block
                                [supernode_block_index(local_row, local_col, width)]
                                * row_value;
                        }
                        solution_block[local_col] = value;
                    }
                }
            }
        }

        if !permuted_solution.iter().all(|value| value.is_finite()) {
            return Err(SsidsError::NumericalBreakdown {
                pivot: self.dimension.saturating_sub(1),
                detail: "solve produced non-finite values".into(),
            });
        }

        for (ordered, &original) in self.permutation.perm().iter().enumerate() {
            rhs[original] = permuted_solution[ordered];
        }
        Ok(())
    }

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
        let symbolic_supernodes = self
            .supernodes
            .iter()
            .map(|supernode| Supernode {
                start_column: supernode.start_column,
                end_column: supernode.end_column,
                trailing_rows: supernode.trailing_rows.clone(),
            })
            .collect::<Vec<_>>();
        let layout = NumericFactorLayout {
            lower_col_ptrs: &self.lower_col_ptrs,
            lower_row_indices: &self.lower_row_indices,
            row_to_columns: &self.row_to_columns,
            supernodes: &symbolic_supernodes,
        };
        let buffers = NumericFactorBuffers {
            lower_values: &mut self.lower_values,
            diag: &mut self.diag,
            workspace_buffer: &mut self.workspace_buffer,
        };
        let factorization = sparse_ldlt_factorize_in_place(
            matrix,
            &self.permutation,
            layout,
            buffers,
            self.options,
        )?;
        let info = FactorInfo {
            factorization_residual_max_abs: factorization.factorization_residual_max_abs,
            regularized_pivots: factorization.pivot_stats.regularized_pivots,
        };
        rebuild_numeric_supernodes(
            &mut self.supernodes,
            &self.lower_col_ptrs,
            &self.lower_row_indices,
            &self.lower_values,
            &self.diag,
        )?;
        self.diagonal_blocks = factorization.diagonal_blocks;
        self.inertia = inertia_from_blocks(
            &self.diag,
            &self.diagonal_blocks,
            self.options.inertia_zero_tol,
        );
        self.pivot_stats = factorization.pivot_stats;
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
    let numeric_pattern = numeric_pattern_from_symbolic(symbolic)?;
    let mut lower_values = vec![0.0; numeric_pattern.lower_row_indices.len()];
    let mut diag = vec![0.0; matrix.dimension()];
    let mut workspace_buffer = vec![0.0; max_supernode_workspace_len(symbolic)];
    let layout = NumericFactorLayout {
        lower_col_ptrs: &numeric_pattern.lower_col_ptrs,
        lower_row_indices: &numeric_pattern.lower_row_indices,
        row_to_columns: &numeric_pattern.row_to_columns,
        supernodes: &symbolic.supernodes,
    };
    let buffers = NumericFactorBuffers {
        lower_values: &mut lower_values,
        diag: &mut diag,
        workspace_buffer: &mut workspace_buffer,
    };
    let factorization =
        sparse_ldlt_factorize_in_place(matrix, &symbolic.permutation, layout, buffers, *options)?;
    let info = FactorInfo {
        factorization_residual_max_abs: factorization.factorization_residual_max_abs,
        regularized_pivots: factorization.pivot_stats.regularized_pivots,
    };
    let inertia = inertia_from_blocks(
        &diag,
        &factorization.diagonal_blocks,
        options.inertia_zero_tol,
    );
    let supernodes = pack_numeric_supernodes(
        &symbolic.supernodes,
        &numeric_pattern.lower_col_ptrs,
        &numeric_pattern.lower_row_indices,
        &lower_values,
        &diag,
    )?;
    let factor = NumericFactor {
        dimension: matrix.dimension(),
        permutation: symbolic.permutation.clone(),
        lower_col_ptrs: numeric_pattern.lower_col_ptrs,
        lower_row_indices: numeric_pattern.lower_row_indices,
        lower_values,
        row_to_columns: numeric_pattern.row_to_columns,
        supernodes,
        diag,
        diagonal_blocks: factorization.diagonal_blocks,
        inertia,
        pivot_stats: factorization.pivot_stats,
        options: *options,
        workspace_buffer,
    };
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct NumericPattern {
    lower_col_ptrs: Vec<usize>,
    lower_row_indices: Vec<usize>,
    row_to_columns: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
struct PermutedLowerMatrix {
    diagonal: Vec<f64>,
    lower_values: Vec<f64>,
}

struct NumericFactorLayout<'a> {
    lower_col_ptrs: &'a [usize],
    lower_row_indices: &'a [usize],
    row_to_columns: &'a [Vec<usize>],
    supernodes: &'a [Supernode],
}

struct NumericFactorBuffers<'a> {
    lower_values: &'a mut [f64],
    diag: &'a mut [f64],
    workspace_buffer: &'a mut Vec<f64>,
}

struct PanelUpdateInputs<'a> {
    lower_col_ptrs: &'a [usize],
    lower_row_indices: &'a [usize],
    lower_values: &'a [f64],
    diag: &'a [f64],
    block_size_at_start: &'a [usize],
    block_values_at_start: &'a [Vec<f64>],
    panel_row_positions: &'a [usize],
    candidate_columns: &'a [usize],
}

struct ScalarFactorInputs<'a> {
    lower_col_ptrs: &'a [usize],
    lower_row_indices: &'a [usize],
    row_to_columns: &'a [Vec<usize>],
    column_block_start: &'a [usize],
    block_size_at_start: &'a [usize],
    block_values_at_start: &'a [Vec<f64>],
    lower_values: &'a mut [f64],
    diag: &'a mut [f64],
}

struct PanelUpdateScratch {
    supernode_positions: Vec<usize>,
    supernode_values: Vec<f64>,
    trailing_positions: Vec<usize>,
    trailing_values: Vec<f64>,
    dense_panel_vector: Vec<f64>,
    dense_trailing_vector: Vec<f64>,
    block_panel: Vec<f64>,
    block_trailing: Vec<f64>,
    block_product_panel: Vec<f64>,
    block_product_trailing: Vec<f64>,
    block_update: Vec<f64>,
    trailing_update: Vec<f64>,
}

impl PanelUpdateScratch {
    fn new(width: usize, trailing_len: usize) -> Self {
        Self {
            supernode_positions: Vec::with_capacity(width),
            supernode_values: Vec::with_capacity(width),
            trailing_positions: Vec::with_capacity(trailing_len),
            trailing_values: Vec::with_capacity(trailing_len),
            dense_panel_vector: Vec::new(),
            dense_trailing_vector: Vec::new(),
            block_panel: Vec::new(),
            block_trailing: Vec::new(),
            block_product_panel: Vec::new(),
            block_product_trailing: Vec::new(),
            block_update: Vec::new(),
            trailing_update: Vec::new(),
        }
    }
}

fn max_supernode_workspace_len(symbolic: &SymbolicFactor) -> usize {
    symbolic
        .supernodes
        .iter()
        .map(|supernode| {
            let width = supernode.width();
            width * width + supernode.trailing_rows.len() * width
        })
        .max()
        .unwrap_or(0)
}

fn supernode_block_index(row: usize, col: usize, width: usize) -> usize {
    row * width + col
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

fn numeric_pattern_from_symbolic(symbolic: &SymbolicFactor) -> Result<NumericPattern, SsidsError> {
    let dimension = symbolic.permutation.len();
    if symbolic.column_pattern.len() != dimension {
        return Err(SsidsError::InvalidMatrix(format!(
            "symbolic column pattern length {} does not match permutation size {dimension}",
            symbolic.column_pattern.len()
        )));
    }

    let mut lower_col_ptrs = Vec::with_capacity(dimension + 1);
    let mut lower_row_indices = Vec::new();
    let mut row_to_columns = vec![Vec::new(); dimension];
    lower_col_ptrs.push(0);

    for (column, pattern) in symbolic.column_pattern.iter().enumerate() {
        let Some((&diagonal, lower_rows)) = pattern.split_first() else {
            return Err(SsidsError::InvalidMatrix(format!(
                "symbolic column pattern for column {column} is empty"
            )));
        };
        if diagonal != column {
            return Err(SsidsError::InvalidMatrix(format!(
                "symbolic column pattern for column {column} must start with diagonal entry {column}"
            )));
        }
        let mut previous = column;
        for &row in lower_rows {
            if row >= dimension {
                return Err(SsidsError::InvalidMatrix(format!(
                    "symbolic pattern row {row} out of bounds for {dimension}x{dimension} factor"
                )));
            }
            if row <= column {
                return Err(SsidsError::InvalidMatrix(format!(
                    "symbolic column pattern for column {column} must contain only rows greater than the column after the diagonal"
                )));
            }
            if row <= previous {
                return Err(SsidsError::InvalidMatrix(format!(
                    "symbolic column pattern for column {column} must be strictly increasing"
                )));
            }
            lower_row_indices.push(row);
            row_to_columns[row].push(column);
            previous = row;
        }
        lower_col_ptrs.push(lower_row_indices.len());
    }

    Ok(NumericPattern {
        lower_col_ptrs,
        lower_row_indices,
        row_to_columns,
    })
}

fn permuted_lower_matrix_from_csc(
    matrix: SymmetricCscMatrix<'_>,
    permutation: &Permutation,
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
) -> Result<PermutedLowerMatrix, SsidsError> {
    let values = matrix.values().ok_or(SsidsError::MissingValues)?;
    let dimension = matrix.dimension();
    let inverse = permutation.inverse();
    let mut diagonal = vec![0.0; dimension];
    let mut lower_values = vec![0.0; lower_row_indices.len()];

    for col in 0..dimension {
        let permuted_col = inverse[col];
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
            let permuted_row = inverse[row];
            let (target_col, target_row) = if permuted_row >= permuted_col {
                (permuted_col, permuted_row)
            } else {
                (permuted_row, permuted_col)
            };
            if target_row == target_col {
                diagonal[target_col] += value;
                continue;
            }
            let target_index =
                find_row_index(lower_col_ptrs, lower_row_indices, target_col, target_row)?;
            lower_values[target_index] += value;
        }
    }

    Ok(PermutedLowerMatrix {
        diagonal,
        lower_values,
    })
}

fn sparse_ldlt_factorize_in_place(
    matrix: SymmetricCscMatrix<'_>,
    permutation: &Permutation,
    layout: NumericFactorLayout<'_>,
    buffers: NumericFactorBuffers<'_>,
    options: NumericFactorOptions,
) -> Result<NumericFactorizationOutcome, SsidsError> {
    let numeric_matrix = permuted_lower_matrix_from_csc(
        matrix,
        permutation,
        layout.lower_col_ptrs,
        layout.lower_row_indices,
    )?;
    buffers.lower_values.fill(0.0);
    buffers.diag.fill(0.0);
    let mut regularized_pivots = 0;
    let mut two_by_two_pivots = 0;
    let mut delayed_pivots = 0;
    let mut min_abs_pivot = f64::INFINITY;
    let mut max_abs_pivot = 0.0_f64;
    let mut max_residual = 0.0_f64;
    let mut diagonal_blocks = Vec::with_capacity(matrix.dimension());
    let dimension = matrix.dimension();
    let mut column_block_start = (0..dimension).collect::<Vec<_>>();
    let mut block_size_at_start = vec![1; dimension];
    let mut block_values_at_start = vec![Vec::new(); dimension];
    let mut panel_row_positions = vec![usize::MAX; dimension];
    let mut marked_candidate_columns = vec![false; dimension];
    let mut candidate_columns = Vec::new();

    for supernode in layout.supernodes {
        let width = supernode.width();
        if width < DENSE_PANEL_MIN_SUPERNODE_WIDTH {
            let factorization = factorize_supernode_scalar(
                supernode,
                &numeric_matrix,
                ScalarFactorInputs {
                    lower_col_ptrs: layout.lower_col_ptrs,
                    lower_row_indices: layout.lower_row_indices,
                    row_to_columns: layout.row_to_columns,
                    column_block_start: &column_block_start,
                    block_size_at_start: &block_size_at_start,
                    block_values_at_start: &block_values_at_start,
                    lower_values: buffers.lower_values,
                    diag: buffers.diag,
                },
                options,
            )?;
            regularized_pivots += factorization.stats.regularized_pivots;
            two_by_two_pivots += factorization.stats.two_by_two_pivots;
            delayed_pivots += factorization.stats.delayed_pivots;
            min_abs_pivot = min_abs_pivot.min(factorization.stats.min_abs_pivot);
            max_abs_pivot = max_abs_pivot.max(factorization.stats.max_abs_pivot);
            max_residual = max_residual.max(factorization.stats.max_residual);
            for block in &factorization.diagonal_blocks {
                update_diagonal_block_metadata(
                    block,
                    &mut column_block_start,
                    &mut block_size_at_start,
                    &mut block_values_at_start,
                );
            }
            diagonal_blocks.extend(factorization.diagonal_blocks);
            continue;
        }
        let trailing_len = supernode.trailing_rows.len();
        let workspace_len = width * width + trailing_len * width;
        if buffers.workspace_buffer.len() < workspace_len {
            buffers.workspace_buffer.resize(workspace_len, 0.0);
        }
        let (block_workspace, trailing_workspace) =
            buffers.workspace_buffer[..workspace_len].split_at_mut(width * width);

        extract_supernode_panel(
            supernode,
            layout.lower_col_ptrs,
            layout.lower_row_indices,
            &numeric_matrix,
            block_workspace,
            trailing_workspace,
        )?;
        collect_panel_row_positions(supernode, &mut panel_row_positions);
        collect_candidate_columns(
            supernode,
            layout.row_to_columns,
            &column_block_start,
            &mut marked_candidate_columns,
            &mut candidate_columns,
        );
        apply_panel_updates_from_previous_columns(
            supernode,
            PanelUpdateInputs {
                lower_col_ptrs: layout.lower_col_ptrs,
                lower_row_indices: layout.lower_row_indices,
                lower_values: buffers.lower_values,
                diag: buffers.diag,
                block_size_at_start: &block_size_at_start,
                block_values_at_start: &block_values_at_start,
                panel_row_positions: &panel_row_positions,
                candidate_columns: &candidate_columns,
            },
            block_workspace,
            trailing_workspace,
        );
        clear_panel_row_positions(supernode, &mut panel_row_positions);
        clear_candidate_columns(&candidate_columns, &mut marked_candidate_columns);

        let factorization =
            factorize_supernode_panel(supernode, block_workspace, trailing_workspace, options)?;
        regularized_pivots += factorization.stats.regularized_pivots;
        two_by_two_pivots += factorization.stats.two_by_two_pivots;
        delayed_pivots += factorization.stats.delayed_pivots;
        min_abs_pivot = min_abs_pivot.min(factorization.stats.min_abs_pivot);
        max_abs_pivot = max_abs_pivot.max(factorization.stats.max_abs_pivot);
        max_residual = max_residual.max(factorization.stats.max_residual);
        for block in &factorization.diagonal_blocks {
            update_diagonal_block_metadata(
                block,
                &mut column_block_start,
                &mut block_size_at_start,
                &mut block_values_at_start,
            );
        }
        diagonal_blocks.extend(factorization.diagonal_blocks.iter().cloned());

        scatter_supernode_panel(
            supernode,
            layout.lower_col_ptrs,
            layout.lower_row_indices,
            buffers.lower_values,
            buffers.diag,
            block_workspace,
            trailing_workspace,
        )?;
    }

    if buffers.diag.is_empty() {
        min_abs_pivot = 0.0;
    }

    Ok(NumericFactorizationOutcome {
        pivot_stats: PivotStats {
            regularized_pivots,
            two_by_two_pivots,
            delayed_pivots,
            min_abs_pivot,
            max_abs_pivot,
        },
        factorization_residual_max_abs: max_residual,
        diagonal_blocks,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct PanelFactorStats {
    regularized_pivots: usize,
    two_by_two_pivots: usize,
    delayed_pivots: usize,
    min_abs_pivot: f64,
    max_abs_pivot: f64,
    max_residual: f64,
}

struct SupernodeFactorization {
    stats: PanelFactorStats,
    diagonal_blocks: Vec<DiagonalBlockValue>,
}

struct NumericFactorizationOutcome {
    pivot_stats: PivotStats,
    factorization_residual_max_abs: f64,
    diagonal_blocks: Vec<DiagonalBlockValue>,
}

fn extract_supernode_panel(
    supernode: &Supernode,
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
    numeric_matrix: &PermutedLowerMatrix,
    block_workspace: &mut [f64],
    trailing_workspace: &mut [f64],
) -> Result<(), SsidsError> {
    let width = supernode.width();
    block_workspace.fill(0.0);
    trailing_workspace.fill(0.0);
    for local_col in 0..width {
        let column = supernode.start_column + local_col;
        block_workspace[supernode_block_index(local_col, local_col, width)] =
            numeric_matrix.diagonal[column];
        let start = lower_col_ptrs[column];
        let end = lower_col_ptrs[column + 1];
        let column_rows = &lower_row_indices[start..end];
        let column_values = &numeric_matrix.lower_values[start..end];
        let mut offset = 0;
        for local_row in (local_col + 1)..width {
            let expected_row = supernode.start_column + local_row;
            let Some((&row, &value)) = column_rows.get(offset).zip(column_values.get(offset))
            else {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} is missing dense in-panel row {expected_row}"
                )));
            };
            if row != expected_row {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} expected dense in-panel row {expected_row}, found {row}"
                )));
            }
            block_workspace[supernode_block_index(local_row, local_col, width)] = value;
            offset += 1;
        }
        for (trailing_offset, &row) in supernode.trailing_rows.iter().enumerate() {
            let Some((&actual_row, &value)) =
                column_rows.get(offset).zip(column_values.get(offset))
            else {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} is missing trailing row {row}"
                )));
            };
            if actual_row != row {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} expected trailing row {row}, found {actual_row}"
                )));
            }
            trailing_workspace[supernode_block_index(trailing_offset, local_col, width)] = value;
            offset += 1;
        }
        if offset != column_rows.len() {
            return Err(SsidsError::InvalidMatrix(format!(
                "supernode column {column} has unexpected extra rows beyond its packed panel"
            )));
        }
    }
    Ok(())
}

fn collect_panel_row_positions(supernode: &Supernode, panel_row_positions: &mut [usize]) {
    for local_row in 0..supernode.width() {
        panel_row_positions[supernode.start_column + local_row] = local_row;
    }
    for (trailing_offset, &row) in supernode.trailing_rows.iter().enumerate() {
        panel_row_positions[row] = supernode.width() + trailing_offset;
    }
}

fn clear_panel_row_positions(supernode: &Supernode, panel_row_positions: &mut [usize]) {
    for local_row in 0..supernode.width() {
        panel_row_positions[supernode.start_column + local_row] = usize::MAX;
    }
    for &row in &supernode.trailing_rows {
        panel_row_positions[row] = usize::MAX;
    }
}

fn collect_candidate_columns(
    supernode: &Supernode,
    row_to_columns: &[Vec<usize>],
    column_block_start: &[usize],
    marked_candidate_columns: &mut [bool],
    candidate_columns: &mut Vec<usize>,
) {
    candidate_columns.clear();
    for local_row in 0..supernode.width() {
        let row = supernode.start_column + local_row;
        for &column in &row_to_columns[row] {
            let block_start = column_block_start[column];
            if block_start >= supernode.start_column || marked_candidate_columns[block_start] {
                continue;
            }
            marked_candidate_columns[block_start] = true;
            candidate_columns.push(block_start);
        }
    }
    candidate_columns.sort_unstable();
}

fn clear_candidate_columns(candidate_columns: &[usize], marked_candidate_columns: &mut [bool]) {
    for &column in candidate_columns {
        marked_candidate_columns[column] = false;
    }
}

fn apply_panel_updates_from_previous_columns(
    supernode: &Supernode,
    inputs: PanelUpdateInputs<'_>,
    block_workspace: &mut [f64],
    trailing_workspace: &mut [f64],
) {
    let width = supernode.width();
    let trailing_len = supernode.trailing_rows.len();
    let panel_work = width * (width + trailing_len);
    if inputs.candidate_columns.len() >= PARALLEL_PANEL_UPDATE_MIN_COLUMNS
        && panel_work >= PARALLEL_PANEL_UPDATE_MIN_WORK
    {
        let block_len = width * width;
        let trailing_block_len = trailing_len * width;
        let (block_delta, trailing_delta, _) = inputs
            .candidate_columns
            .par_iter()
            .fold(
                || {
                    (
                        vec![0.0; block_len],
                        vec![0.0; trailing_block_len],
                        PanelUpdateScratch::new(width, trailing_len),
                    )
                },
                |(mut block_delta, mut trailing_delta, mut scratch), &block_start| {
                    apply_single_previous_block_update(
                        supernode,
                        &inputs,
                        block_start,
                        &mut block_delta,
                        &mut trailing_delta,
                        &mut scratch,
                    );
                    (block_delta, trailing_delta, scratch)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0; block_len],
                        vec![0.0; trailing_block_len],
                        PanelUpdateScratch::new(width, trailing_len),
                    )
                },
                |(mut left_block, mut left_trailing, left_scratch),
                 (right_block, right_trailing, _)| {
                    for (lhs, rhs) in left_block.iter_mut().zip(right_block.into_iter()) {
                        *lhs += rhs;
                    }
                    for (lhs, rhs) in left_trailing.iter_mut().zip(right_trailing.into_iter()) {
                        *lhs += rhs;
                    }
                    (left_block, left_trailing, left_scratch)
                },
            );
        for (target, delta) in block_workspace.iter_mut().zip(block_delta.into_iter()) {
            *target += delta;
        }
        for (target, delta) in trailing_workspace
            .iter_mut()
            .zip(trailing_delta.into_iter())
        {
            *target += delta;
        }
        return;
    }

    let mut scratch = PanelUpdateScratch::new(width, trailing_len);
    for &block_start in inputs.candidate_columns {
        apply_single_previous_block_update(
            supernode,
            &inputs,
            block_start,
            block_workspace,
            trailing_workspace,
            &mut scratch,
        );
    }
}

fn factorize_supernode_panel(
    supernode: &Supernode,
    block_workspace: &mut [f64],
    trailing_workspace: &mut [f64],
    options: NumericFactorOptions,
) -> Result<SupernodeFactorization, SsidsError> {
    let width = supernode.width();
    let trailing_len = supernode.trailing_rows.len();
    let mut stats = PanelFactorStats {
        min_abs_pivot: f64::INFINITY,
        ..PanelFactorStats::default()
    };
    let mut diagonal_blocks = Vec::with_capacity(width);
    let mut local_col = 0;

    while local_col < width {
        match select_pivot_action(
            block_workspace,
            trailing_workspace,
            width,
            trailing_len,
            local_col,
            options,
        ) {
            PivotAction::DeferredDenseBlock => {
                diagonal_blocks.push(defer_dense_diagonal_block(
                    supernode,
                    block_workspace,
                    trailing_workspace,
                    local_col,
                    options,
                    &mut stats,
                )?);
                break;
            }
            PivotAction::TwoByTwo => {
                let next = local_col + 1;
                let global_col = supernode.start_column + local_col;
                let d11 = symmetric_block_get(block_workspace, width, local_col, local_col);
                let d21 = symmetric_block_get(block_workspace, width, next, local_col);
                let d22 = symmetric_block_get(block_workspace, width, next, next);
                if !d11.is_finite() || !d21.is_finite() || !d22.is_finite() {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: global_col,
                        detail: "two-by-two pivot became non-finite".into(),
                    });
                }
                let determinant = d11 * d22 - d21 * d21;
                if !determinant.is_finite() || determinant.abs() < f64::EPSILON {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: global_col,
                        detail: "two-by-two pivot is numerically singular".into(),
                    });
                }
                let inv11 = d22 / determinant;
                let inv12 = -d21 / determinant;
                let inv22 = d11 / determinant;
                let (lambda_max, lambda_min) = two_by_two_eigenvalues(d11, d21, d22);
                stats.two_by_two_pivots += 1;
                stats.min_abs_pivot = stats
                    .min_abs_pivot
                    .min(lambda_max.abs().min(lambda_min.abs()));
                stats.max_abs_pivot = stats
                    .max_abs_pivot
                    .max(lambda_max.abs().max(lambda_min.abs()));
                diagonal_blocks.push(DiagonalBlockValue {
                    block: DiagonalBlock {
                        start: global_col,
                        size: 2,
                    },
                    values: vec![d11, d21, d22],
                });

                for future_row in (next + 1)..width {
                    let b1 = symmetric_block_get(block_workspace, width, future_row, local_col);
                    let b2 = symmetric_block_get(block_workspace, width, future_row, next);
                    let l1 = b1 * inv11 + b2 * inv12;
                    let l2 = b1 * inv12 + b2 * inv22;
                    if !l1.is_finite() || !l2.is_finite() {
                        return Err(SsidsError::NumericalBreakdown {
                            pivot: global_col,
                            detail: format!(
                                "two-by-two subdiagonal entry ({}, {}) became non-finite",
                                supernode.start_column + future_row,
                                global_col
                            ),
                        });
                    }
                    symmetric_block_set(block_workspace, width, future_row, local_col, l1);
                    symmetric_block_set(block_workspace, width, future_row, next, l2);
                }
                for trailing_offset in 0..trailing_len {
                    let b1 = trailing_workspace
                        [supernode_block_index(trailing_offset, local_col, width)];
                    let b2 =
                        trailing_workspace[supernode_block_index(trailing_offset, next, width)];
                    let l1 = b1 * inv11 + b2 * inv12;
                    let l2 = b1 * inv12 + b2 * inv22;
                    if !l1.is_finite() || !l2.is_finite() {
                        return Err(SsidsError::NumericalBreakdown {
                            pivot: global_col,
                            detail: format!(
                                "two-by-two trailing entry ({}, {}) became non-finite",
                                supernode.trailing_rows[trailing_offset], global_col
                            ),
                        });
                    }
                    trailing_workspace[supernode_block_index(trailing_offset, local_col, width)] =
                        l1;
                    trailing_workspace[supernode_block_index(trailing_offset, next, width)] = l2;
                }

                for future_col in (next + 1)..width {
                    let l1_col = symmetric_block_get(block_workspace, width, future_col, local_col);
                    let l2_col = symmetric_block_get(block_workspace, width, future_col, next);
                    for future_row in future_col..width {
                        let l1_row =
                            symmetric_block_get(block_workspace, width, future_row, local_col);
                        let l2_row = symmetric_block_get(block_workspace, width, future_row, next);
                        let update = d11 * l1_row * l1_col
                            + d21 * (l1_row * l2_col + l2_row * l1_col)
                            + d22 * l2_row * l2_col;
                        let updated =
                            symmetric_block_get(block_workspace, width, future_row, future_col)
                                - update;
                        symmetric_block_set(
                            block_workspace,
                            width,
                            future_row,
                            future_col,
                            updated,
                        );
                    }
                    for trailing_offset in 0..trailing_len {
                        let l1_trailing = trailing_workspace
                            [supernode_block_index(trailing_offset, local_col, width)];
                        let l2_trailing =
                            trailing_workspace[supernode_block_index(trailing_offset, next, width)];
                        let update = d11 * l1_trailing * l1_col
                            + d21 * (l1_trailing * l2_col + l2_trailing * l1_col)
                            + d22 * l2_trailing * l2_col;
                        trailing_workspace
                            [supernode_block_index(trailing_offset, future_col, width)] -= update;
                    }
                }

                symmetric_block_set(block_workspace, width, local_col, local_col, d11);
                symmetric_block_set(block_workspace, width, next, next, d22);
                symmetric_block_set(block_workspace, width, next, local_col, 0.0);
                local_col += 2;
            }
            PivotAction::OneByOne => {
                let diagonal_index = supernode_block_index(local_col, local_col, width);
                let mut diagonal = block_workspace[diagonal_index];
                if !diagonal.is_finite() {
                    return Err(SsidsError::NumericalBreakdown {
                        pivot: supernode.start_column + local_col,
                        detail: "diagonal pivot became non-finite".into(),
                    });
                }

                let original_diagonal = diagonal;
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
                        pivot: supernode.start_column + local_col,
                        detail: "diagonal pivot is numerically zero".into(),
                    });
                }

                block_workspace[diagonal_index] = diagonal;
                stats.max_residual = stats.max_residual.max((diagonal - original_diagonal).abs());
                let abs_pivot = diagonal.abs();
                stats.min_abs_pivot = stats.min_abs_pivot.min(abs_pivot);
                stats.max_abs_pivot = stats.max_abs_pivot.max(abs_pivot);
                diagonal_blocks.push(DiagonalBlockValue {
                    block: DiagonalBlock {
                        start: supernode.start_column + local_col,
                        size: 1,
                    },
                    values: vec![diagonal],
                });

                for local_row in (local_col + 1)..width {
                    let entry_index = supernode_block_index(local_row, local_col, width);
                    let value = block_workspace[entry_index] / diagonal;
                    if !value.is_finite() {
                        return Err(SsidsError::NumericalBreakdown {
                            pivot: supernode.start_column + local_col,
                            detail: format!(
                                "subdiagonal entry ({}, {}) became non-finite",
                                supernode.start_column + local_row,
                                supernode.start_column + local_col
                            ),
                        });
                    }
                    block_workspace[entry_index] = value;
                }
                for trailing_offset in 0..trailing_len {
                    let entry_index = supernode_block_index(trailing_offset, local_col, width);
                    let value = trailing_workspace[entry_index] / diagonal;
                    if !value.is_finite() {
                        return Err(SsidsError::NumericalBreakdown {
                            pivot: supernode.start_column + local_col,
                            detail: format!(
                                "trailing entry ({}, {}) became non-finite",
                                supernode.trailing_rows[trailing_offset],
                                supernode.start_column + local_col
                            ),
                        });
                    }
                    trailing_workspace[entry_index] = value;
                }

                let remaining = width.saturating_sub(local_col + 1);
                if remaining > 0 {
                    let panel_column_ptr = block_workspace
                        .as_ptr()
                        .wrapping_add(supernode_block_index(local_col + 1, local_col, width));
                    let trailing_column_ptr = trailing_workspace
                        .as_ptr()
                        .wrapping_add(supernode_block_index(0, local_col, width));
                    let block_submatrix_ptr = block_workspace
                        .as_mut_ptr()
                        .wrapping_add(supernode_block_index(local_col + 1, local_col + 1, width));
                    let trailing_submatrix_ptr = trailing_workspace
                        .as_mut_ptr()
                        .wrapping_add(supernode_block_index(0, local_col + 1, width));
                    let blas_work = remaining * (remaining + trailing_len);
                    let used_blas = blas_work >= BLAS_RANK1_UPDATE_MIN_WORK
                        && try_blas_dsyr_lower_strided(
                            remaining,
                            -diagonal,
                            panel_column_ptr,
                            width,
                            block_submatrix_ptr,
                            width,
                        )
                        && try_blas_dger_strided(DgerSpec {
                            m: trailing_len,
                            n: remaining,
                            alpha: -diagonal,
                            x_ptr: trailing_column_ptr,
                            incx: width,
                            y_ptr: panel_column_ptr,
                            incy: width,
                            a_ptr: trailing_submatrix_ptr,
                            lda: width,
                        });
                    if !used_blas {
                        for future_col in (local_col + 1)..width {
                            let scale = diagonal
                                * block_workspace
                                    [supernode_block_index(future_col, local_col, width)];
                            for future_row in future_col..width {
                                block_workspace
                                    [supernode_block_index(future_row, future_col, width)] -=
                                    block_workspace
                                        [supernode_block_index(future_row, local_col, width)]
                                        * scale;
                            }
                            for trailing_offset in 0..trailing_len {
                                trailing_workspace
                                    [supernode_block_index(trailing_offset, future_col, width)] -=
                                    trailing_workspace
                                        [supernode_block_index(trailing_offset, local_col, width)]
                                        * scale;
                            }
                        }
                    }
                }
                local_col += 1;
            }
        }
    }

    if width == 0 {
        stats.min_abs_pivot = 0.0;
    }
    Ok(SupernodeFactorization {
        stats,
        diagonal_blocks,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PivotAction {
    OneByOne,
    TwoByTwo,
    DeferredDenseBlock,
}

fn select_pivot_action(
    block_workspace: &[f64],
    trailing_workspace: &[f64],
    width: usize,
    trailing_len: usize,
    local_col: usize,
    options: NumericFactorOptions,
) -> PivotAction {
    let diagonal = symmetric_block_get(block_workspace, width, local_col, local_col).abs();
    let mut max_offdiag = 0.0_f64;
    for future_row in (local_col + 1)..width {
        max_offdiag = max_offdiag
            .max(symmetric_block_get(block_workspace, width, future_row, local_col).abs());
    }
    for trailing_offset in 0..trailing_len {
        max_offdiag = max_offdiag.max(
            trailing_workspace[supernode_block_index(trailing_offset, local_col, width)].abs(),
        );
    }
    if max_offdiag <= options.pivot_regularization
        || diagonal >= options.two_by_two_pivot_threshold * max_offdiag
    {
        return PivotAction::OneByOne;
    }
    if should_use_two_by_two_pivot(
        block_workspace,
        trailing_workspace,
        width,
        trailing_len,
        local_col,
        options,
    ) {
        return PivotAction::TwoByTwo;
    }
    if width.saturating_sub(local_col) >= DEFERRED_DENSE_BLOCK_MIN_SIZE {
        PivotAction::DeferredDenseBlock
    } else {
        PivotAction::OneByOne
    }
}

fn defer_dense_diagonal_block(
    supernode: &Supernode,
    block_workspace: &mut [f64],
    trailing_workspace: &mut [f64],
    local_col: usize,
    options: NumericFactorOptions,
    stats: &mut PanelFactorStats,
) -> Result<DiagonalBlockValue, SsidsError> {
    let width = supernode.width();
    let trailing_len = supernode.trailing_rows.len();
    let block_size = width - local_col;
    let mut values = vec![0.0; block_size * (block_size + 1) / 2];
    for row in 0..block_size {
        for col in 0..=row {
            diagonal_block_set(
                &mut values,
                block_size,
                row,
                col,
                symmetric_block_get(block_workspace, width, local_col + row, local_col + col),
            );
        }
    }
    let (inverse, regularized_pivots, max_shift) =
        stabilized_dense_block_inverse(&mut values, block_size, options).map_err(|detail| {
            SsidsError::NumericalBreakdown {
                pivot: supernode.start_column + local_col,
                detail,
            }
        })?;
    stats.regularized_pivots += regularized_pivots;
    stats.delayed_pivots += block_size;
    stats.max_residual = stats.max_residual.max(max_shift);
    for eigenvalue in jacobi_eigenvalues(&values, block_size) {
        let abs_value = eigenvalue.abs();
        stats.min_abs_pivot = stats.min_abs_pivot.min(abs_value);
        stats.max_abs_pivot = stats.max_abs_pivot.max(abs_value);
    }

    let mut projected = vec![0.0; block_size];
    for trailing_offset in 0..trailing_len {
        projected.fill(0.0);
        for col in 0..block_size {
            for rhs in 0..block_size {
                projected[col] += trailing_workspace
                    [supernode_block_index(trailing_offset, local_col + rhs, width)]
                    * inverse[rhs * block_size + col];
            }
        }
        for col in 0..block_size {
            trailing_workspace[supernode_block_index(trailing_offset, local_col + col, width)] =
                projected[col];
        }
    }

    for col in local_col..width {
        let local_offset = col - local_col;
        block_workspace[supernode_block_index(col, col, width)] =
            diagonal_block_get(&values, block_size, local_offset, local_offset);
        for row in (col + 1)..width {
            block_workspace[supernode_block_index(row, col, width)] = 0.0;
        }
    }

    Ok(DiagonalBlockValue {
        block: DiagonalBlock {
            start: supernode.start_column + local_col,
            size: block_size,
        },
        values,
    })
}

fn apply_single_previous_block_update(
    supernode: &Supernode,
    inputs: &PanelUpdateInputs<'_>,
    block_start: usize,
    block_workspace: &mut [f64],
    trailing_workspace: &mut [f64],
    scratch: &mut PanelUpdateScratch,
) {
    let width = supernode.width();
    let trailing_len = supernode.trailing_rows.len();
    let block_size = inputs.block_size_at_start[block_start];
    if block_size == 1 {
        let previous_start = inputs.lower_col_ptrs[block_start];
        let previous_end = inputs.lower_col_ptrs[block_start + 1];
        let previous_rows = &inputs.lower_row_indices[previous_start..previous_end];
        let previous_values = &inputs.lower_values[previous_start..previous_end];
        scratch.supernode_positions.clear();
        scratch.supernode_values.clear();
        scratch.trailing_positions.clear();
        scratch.trailing_values.clear();
        for (&row, &value) in previous_rows.iter().zip(previous_values.iter()) {
            let position = inputs.panel_row_positions[row];
            if position == usize::MAX {
                continue;
            }
            if position < width {
                scratch.supernode_positions.push(position);
                scratch.supernode_values.push(value);
            } else {
                scratch.trailing_positions.push(position - width);
                scratch.trailing_values.push(value);
            }
        }
        if scratch.supernode_positions.is_empty() {
            return;
        }
        let pivot = inputs.diag[block_start];
        let dense_work =
            scratch.supernode_positions.len() * (scratch.supernode_positions.len() + trailing_len);
        if dense_work >= BLAS_SINGLE_COLUMN_UPDATE_MIN_WORK && blas_kernel().is_some() {
            scratch.dense_panel_vector.resize(width, 0.0);
            scratch.dense_panel_vector.fill(0.0);
            for (&position, &value) in scratch
                .supernode_positions
                .iter()
                .zip(scratch.supernode_values.iter())
            {
                scratch.dense_panel_vector[position] = value;
            }
            scratch.dense_trailing_vector.resize(trailing_len, 0.0);
            scratch.dense_trailing_vector.fill(0.0);
            for (&position, &value) in scratch
                .trailing_positions
                .iter()
                .zip(scratch.trailing_values.iter())
            {
                scratch.dense_trailing_vector[position] = value;
            }
            let used_blas = try_blas_dsyr_lower_strided(
                width,
                -pivot,
                scratch.dense_panel_vector.as_ptr(),
                1,
                block_workspace.as_mut_ptr(),
                width,
            ) && try_blas_dger_strided(DgerSpec {
                m: trailing_len,
                n: width,
                alpha: -pivot,
                x_ptr: scratch.dense_trailing_vector.as_ptr(),
                incx: 1,
                y_ptr: scratch.dense_panel_vector.as_ptr(),
                incy: 1,
                a_ptr: trailing_workspace.as_mut_ptr(),
                lda: width,
            });
            if used_blas {
                return;
            }
        }
        for lhs in 0..scratch.supernode_positions.len() {
            let row = scratch.supernode_positions[lhs];
            let row_value = scratch.supernode_values[lhs];
            let scale = pivot * row_value;
            for rhs in 0..=lhs {
                let col = scratch.supernode_positions[rhs];
                let (target_row, target_col, target_value) = if row >= col {
                    (row, col, scratch.supernode_values[rhs])
                } else {
                    (col, row, row_value)
                };
                block_workspace[supernode_block_index(target_row, target_col, width)] -=
                    scale * target_value;
            }
            for (trailing_position, trailing_value) in scratch
                .trailing_positions
                .iter()
                .copied()
                .zip(scratch.trailing_values.iter().copied())
            {
                trailing_workspace[supernode_block_index(trailing_position, row, width)] -=
                    scale * trailing_value;
            }
        }
        return;
    }

    let block_values = &inputs.block_values_at_start[block_start];
    scratch.block_panel.resize(width * block_size, 0.0);
    scratch.block_panel.fill(0.0);
    scratch
        .block_trailing
        .resize(trailing_len * block_size, 0.0);
    scratch.block_trailing.fill(0.0);

    for local_slot in 0..block_size {
        let column = block_start + local_slot;
        let start = inputs.lower_col_ptrs[column];
        let end = inputs.lower_col_ptrs[column + 1];
        for (&row, &value) in inputs.lower_row_indices[start..end]
            .iter()
            .zip(inputs.lower_values[start..end].iter())
        {
            let position = inputs.panel_row_positions[row];
            if position == usize::MAX {
                continue;
            }
            if position < width {
                scratch.block_panel[position * block_size + local_slot] = value;
            } else {
                scratch.block_trailing[(position - width) * block_size + local_slot] = value;
            }
        }
    }

    let dense_work = width * block_size * (block_size + trailing_len.max(width));
    if dense_work >= BLAS_BLOCK_UPDATE_MIN_WORK && block_size >= 4 {
        let dense_block = dense_matrix_from_diagonal_block(block_values, block_size);
        scratch.block_product_panel.resize(width * block_size, 0.0);
        scratch.block_product_panel.fill(0.0);
        let mut used_blas = try_blas_dgemm_row_major(GemmSpec {
            trans_a: false,
            trans_b: false,
            m: width,
            n: block_size,
            k: block_size,
            alpha: 1.0,
            a: &scratch.block_panel,
            lda: block_size,
            b: &dense_block,
            ldb: block_size,
            beta: 0.0,
            c: &mut scratch.block_product_panel,
            ldc: block_size,
        });
        if used_blas {
            scratch.block_update.resize(width * width, 0.0);
            scratch.block_update.fill(0.0);
            used_blas = try_blas_dgemm_row_major(GemmSpec {
                trans_a: false,
                trans_b: true,
                m: width,
                n: width,
                k: block_size,
                alpha: 1.0,
                a: &scratch.block_product_panel,
                lda: block_size,
                b: &scratch.block_panel,
                ldb: block_size,
                beta: 0.0,
                c: &mut scratch.block_update,
                ldc: width,
            });
        }
        if used_blas && trailing_len > 0 {
            scratch
                .block_product_trailing
                .resize(trailing_len * block_size, 0.0);
            scratch.block_product_trailing.fill(0.0);
            used_blas = try_blas_dgemm_row_major(GemmSpec {
                trans_a: false,
                trans_b: false,
                m: trailing_len,
                n: block_size,
                k: block_size,
                alpha: 1.0,
                a: &scratch.block_trailing,
                lda: block_size,
                b: &dense_block,
                ldb: block_size,
                beta: 0.0,
                c: &mut scratch.block_product_trailing,
                ldc: block_size,
            });
            if used_blas {
                scratch.trailing_update.resize(trailing_len * width, 0.0);
                scratch.trailing_update.fill(0.0);
                used_blas = try_blas_dgemm_row_major(GemmSpec {
                    trans_a: false,
                    trans_b: true,
                    m: trailing_len,
                    n: width,
                    k: block_size,
                    alpha: 1.0,
                    a: &scratch.block_product_trailing,
                    lda: block_size,
                    b: &scratch.block_panel,
                    ldb: block_size,
                    beta: 0.0,
                    c: &mut scratch.trailing_update,
                    ldc: width,
                });
            }
        }
        if used_blas {
            for row in 0..width {
                for col in 0..=row {
                    block_workspace[supernode_block_index(row, col, width)] -=
                        scratch.block_update[row * width + col];
                }
            }
            for trailing_position in 0..trailing_len {
                for row in 0..width {
                    trailing_workspace[supernode_block_index(trailing_position, row, width)] -=
                        scratch.trailing_update[trailing_position * width + row];
                }
            }
            return;
        }
    }

    for row in 0..width {
        let row_slice = &scratch.block_panel[row * block_size..(row + 1) * block_size];
        if row_slice.iter().all(|value| *value == 0.0) {
            continue;
        }
        for col in 0..=row {
            let col_slice = &scratch.block_panel[col * block_size..(col + 1) * block_size];
            if col_slice.iter().all(|value| *value == 0.0) {
                continue;
            }
            let mut update = 0.0;
            for (lhs, &row_value) in row_slice.iter().enumerate() {
                for (rhs, &col_value) in col_slice.iter().enumerate() {
                    update += row_value
                        * diagonal_block_get(block_values, block_size, lhs, rhs)
                        * col_value;
                }
            }
            block_workspace[supernode_block_index(row, col, width)] -= update;
        }
        for trailing_position in 0..trailing_len {
            let trailing_slice = &scratch.block_trailing
                [trailing_position * block_size..(trailing_position + 1) * block_size];
            if trailing_slice.iter().all(|value| *value == 0.0) {
                continue;
            }
            let mut update = 0.0;
            for (lhs, &row_value) in row_slice.iter().enumerate() {
                for (rhs, &trailing_value) in trailing_slice.iter().enumerate() {
                    update += row_value
                        * diagonal_block_get(block_values, block_size, lhs, rhs)
                        * trailing_value;
                }
            }
            trailing_workspace[supernode_block_index(trailing_position, row, width)] -= update;
        }
    }
}

fn factorize_supernode_scalar(
    supernode: &Supernode,
    numeric_matrix: &PermutedLowerMatrix,
    inputs: ScalarFactorInputs<'_>,
    options: NumericFactorOptions,
) -> Result<SupernodeFactorization, SsidsError> {
    if supernode.width() == 2
        && width_two_may_need_block_pivot(supernode, numeric_matrix, &inputs, options)
    {
        let width = supernode.width();
        let trailing_len = supernode.trailing_rows.len();
        let mut block_workspace = vec![0.0; width * width];
        let mut trailing_workspace = vec![0.0; trailing_len * width];
        extract_supernode_panel(
            supernode,
            inputs.lower_col_ptrs,
            inputs.lower_row_indices,
            numeric_matrix,
            &mut block_workspace,
            &mut trailing_workspace,
        )?;
        let mut panel_row_positions = vec![usize::MAX; inputs.diag.len()];
        let mut marked_candidate_columns = vec![false; inputs.diag.len()];
        let mut candidate_columns = Vec::new();
        collect_panel_row_positions(supernode, &mut panel_row_positions);
        collect_candidate_columns(
            supernode,
            inputs.row_to_columns,
            inputs.column_block_start,
            &mut marked_candidate_columns,
            &mut candidate_columns,
        );
        apply_panel_updates_from_previous_columns(
            supernode,
            PanelUpdateInputs {
                lower_col_ptrs: inputs.lower_col_ptrs,
                lower_row_indices: inputs.lower_row_indices,
                lower_values: inputs.lower_values,
                diag: inputs.diag,
                block_size_at_start: inputs.block_size_at_start,
                block_values_at_start: inputs.block_values_at_start,
                panel_row_positions: &panel_row_positions,
                candidate_columns: &candidate_columns,
            },
            &mut block_workspace,
            &mut trailing_workspace,
        );
        clear_panel_row_positions(supernode, &mut panel_row_positions);
        clear_candidate_columns(&candidate_columns, &mut marked_candidate_columns);
        if should_use_two_by_two_pivot(
            &block_workspace,
            &trailing_workspace,
            width,
            trailing_len,
            0,
            options,
        ) {
            let factorization = factorize_supernode_panel(
                supernode,
                &mut block_workspace,
                &mut trailing_workspace,
                options,
            )?;
            scatter_supernode_panel(
                supernode,
                inputs.lower_col_ptrs,
                inputs.lower_row_indices,
                inputs.lower_values,
                inputs.diag,
                &block_workspace,
                &trailing_workspace,
            )?;
            return Ok(factorization);
        }
    }

    let mut stats = PanelFactorStats {
        min_abs_pivot: f64::INFINITY,
        ..PanelFactorStats::default()
    };
    let mut diagonal_blocks = Vec::with_capacity(supernode.width());

    for column in supernode.start_column..supernode.end_column {
        let start = inputs.lower_col_ptrs[column];
        let end = inputs.lower_col_ptrs[column + 1];
        let column_rows = &inputs.lower_row_indices[start..end];
        let mut workspace = numeric_matrix.lower_values[start..end].to_vec();
        let mut diagonal = numeric_matrix.diagonal[column];
        let mut previous_blocks = Vec::new();

        for &previous in &inputs.row_to_columns[column] {
            let block_start = inputs.column_block_start[previous];
            if previous_blocks.contains(&block_start) {
                continue;
            }
            previous_blocks.push(block_start);
            if inputs.block_size_at_start[block_start] == 1 {
                let previous_start = inputs.lower_col_ptrs[block_start];
                let previous_end = inputs.lower_col_ptrs[block_start + 1];
                let previous_rows = &inputs.lower_row_indices[previous_start..previous_end];
                let position_in_previous =
                    previous_rows
                        .binary_search(&column)
                        .map_err(|_| SsidsError::InvalidMatrix(format!(
                            "symbolic pattern is missing coupling ({column}, {block_start}) required by previous factor column"
                        )))?;
                let coupling = inputs.lower_values[previous_start + position_in_previous];
                diagonal -= coupling * coupling * inputs.diag[block_start];

                for (local_offset, &row) in previous_rows
                    .iter()
                    .enumerate()
                    .skip(position_in_previous + 1)
                {
                    let current_offset = column_rows.binary_search(&row).map_err(|_| {
                        SsidsError::InvalidMatrix(format!(
                            "symbolic pattern is missing fill entry ({row}, {column}) required by column {block_start}"
                        ))
                    })?;
                    workspace[current_offset] -= inputs.lower_values[previous_start + local_offset]
                        * inputs.diag[block_start]
                        * coupling;
                }
                continue;
            }

            let block_size = inputs.block_size_at_start[block_start];
            let block_values = &inputs.block_values_at_start[block_start];
            let mut coupling = vec![0.0; block_size];
            for (local_slot, coupling_slot) in coupling.iter_mut().enumerate() {
                let slot_start = inputs.lower_col_ptrs[block_start + local_slot];
                let slot_end = inputs.lower_col_ptrs[block_start + local_slot + 1];
                let slot_rows = &inputs.lower_row_indices[slot_start..slot_end];
                *coupling_slot = slot_rows
                    .binary_search(&column)
                    .ok()
                    .map(|offset| inputs.lower_values[slot_start + offset])
                    .unwrap_or(0.0);
            }
            for lhs in 0..block_size {
                for rhs in 0..block_size {
                    diagonal -= coupling[lhs]
                        * diagonal_block_get(block_values, block_size, lhs, rhs)
                        * coupling[rhs];
                }
            }

            let mut scale = vec![0.0; block_size];
            for (lhs, scale_value) in scale.iter_mut().enumerate() {
                for (rhs, &coupling_value) in coupling.iter().enumerate() {
                    *scale_value +=
                        diagonal_block_get(block_values, block_size, lhs, rhs) * coupling_value;
                }
            }
            for (local_slot, &scale_value) in scale.iter().enumerate() {
                let slot_start = inputs.lower_col_ptrs[block_start + local_slot];
                let slot_end = inputs.lower_col_ptrs[block_start + local_slot + 1];
                for (local_offset, &row) in inputs.lower_row_indices[slot_start..slot_end]
                    .iter()
                    .enumerate()
                {
                    if row <= column {
                        continue;
                    }
                    let current_offset = column_rows.binary_search(&row).map_err(|_| {
                        SsidsError::InvalidMatrix(format!(
                            "symbolic pattern is missing fill entry ({row}, {column}) required by delayed block starting at column {block_start}"
                        ))
                    })?;
                    workspace[current_offset] -=
                        inputs.lower_values[slot_start + local_offset] * scale_value;
                }
            }
        }

        if !diagonal.is_finite() {
            return Err(SsidsError::NumericalBreakdown {
                pivot: column,
                detail: "diagonal pivot became non-finite".into(),
            });
        }

        let original_diagonal = diagonal;
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
                pivot: column,
                detail: "diagonal pivot is numerically zero".into(),
            });
        }

        inputs.diag[column] = diagonal;
        stats.max_residual = stats.max_residual.max((diagonal - original_diagonal).abs());
        let abs_pivot = diagonal.abs();
        stats.min_abs_pivot = stats.min_abs_pivot.min(abs_pivot);
        stats.max_abs_pivot = stats.max_abs_pivot.max(abs_pivot);
        diagonal_blocks.push(DiagonalBlockValue {
            block: DiagonalBlock {
                start: column,
                size: 1,
            },
            values: vec![diagonal],
        });

        for (offset, &entry) in workspace.iter().enumerate() {
            let lower_value = entry / diagonal;
            if !lower_value.is_finite() {
                return Err(SsidsError::NumericalBreakdown {
                    pivot: column,
                    detail: format!(
                        "subdiagonal entry ({}, {column}) became non-finite",
                        column_rows[offset]
                    ),
                });
            }
            inputs.lower_values[start + offset] = lower_value;
        }
    }

    if supernode.width() == 0 {
        stats.min_abs_pivot = 0.0;
    }
    Ok(SupernodeFactorization {
        stats,
        diagonal_blocks,
    })
}

fn width_two_may_need_block_pivot(
    supernode: &Supernode,
    numeric_matrix: &PermutedLowerMatrix,
    inputs: &ScalarFactorInputs<'_>,
    options: NumericFactorOptions,
) -> bool {
    let column = supernode.start_column;
    let start = inputs.lower_col_ptrs[column];
    let end = inputs.lower_col_ptrs[column + 1];
    let coupling = inputs.lower_row_indices[start..end]
        .iter()
        .zip(numeric_matrix.lower_values[start..end].iter())
        .find_map(|(&row, &value)| (row == column + 1).then_some(value.abs()))
        .unwrap_or(0.0);
    if coupling <= options.pivot_regularization {
        return false;
    }
    if numeric_matrix.diagonal[column].abs() < options.two_by_two_pivot_threshold * coupling {
        return true;
    }
    inputs.row_to_columns[column].iter().any(|&previous| {
        let block_start = inputs.column_block_start[previous];
        inputs.block_size_at_start[block_start] == 2
    })
}

fn symmetric_block_get(block_workspace: &[f64], width: usize, row: usize, col: usize) -> f64 {
    let (row, col) = if row >= col { (row, col) } else { (col, row) };
    block_workspace[supernode_block_index(row, col, width)]
}

fn symmetric_block_set(
    block_workspace: &mut [f64],
    width: usize,
    row: usize,
    col: usize,
    value: f64,
) {
    let (row, col) = if row >= col { (row, col) } else { (col, row) };
    block_workspace[supernode_block_index(row, col, width)] = value;
}

fn should_use_two_by_two_pivot(
    block_workspace: &[f64],
    trailing_workspace: &[f64],
    width: usize,
    trailing_len: usize,
    local_col: usize,
    options: NumericFactorOptions,
) -> bool {
    if local_col + 1 >= width {
        return false;
    }
    let diagonal = symmetric_block_get(block_workspace, width, local_col, local_col).abs();
    let coupling = symmetric_block_get(block_workspace, width, local_col + 1, local_col).abs();
    if coupling <= options.pivot_regularization {
        return false;
    }
    let mut max_offdiag = coupling;
    for future_row in (local_col + 2)..width {
        max_offdiag = max_offdiag
            .max(symmetric_block_get(block_workspace, width, future_row, local_col).abs());
    }
    for trailing_offset in 0..trailing_len {
        max_offdiag = max_offdiag.max(
            trailing_workspace[supernode_block_index(trailing_offset, local_col, width)].abs(),
        );
    }
    if max_offdiag <= options.pivot_regularization
        || diagonal >= options.two_by_two_pivot_threshold * max_offdiag
    {
        return false;
    }
    let d11 = symmetric_block_get(block_workspace, width, local_col, local_col);
    let d21 = symmetric_block_get(block_workspace, width, local_col + 1, local_col);
    let d22 = symmetric_block_get(block_workspace, width, local_col + 1, local_col + 1);
    let determinant = d11 * d22 - d21 * d21;
    determinant.is_finite()
        && determinant.abs() >= options.pivot_regularization * options.pivot_regularization
}

fn update_diagonal_block_metadata(
    block_value: &DiagonalBlockValue,
    column_block_start: &mut [usize],
    block_size_at_start: &mut [usize],
    block_values_at_start: &mut [Vec<f64>],
) {
    let block = block_value.block;
    column_block_start[block.start] = block.start;
    block_size_at_start[block.start] = block.size;
    block_values_at_start[block.start] = block_value.values.clone();
    for offset in 1..block.size {
        column_block_start[block.start + offset] = block.start;
        block_size_at_start[block.start + offset] = 0;
        block_values_at_start[block.start + offset].clear();
    }
}

fn two_by_two_eigenvalues(d11: f64, d21: f64, d22: f64) -> (f64, f64) {
    let trace_half = 0.5 * (d11 + d22);
    let radius = (0.5 * (d11 - d22)).hypot(d21);
    (trace_half + radius, trace_half - radius)
}

fn scatter_supernode_panel(
    supernode: &Supernode,
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
    lower_values: &mut [f64],
    diag: &mut [f64],
    block_workspace: &[f64],
    trailing_workspace: &[f64],
) -> Result<(), SsidsError> {
    let width = supernode.width();
    for local_col in 0..width {
        let column = supernode.start_column + local_col;
        diag[column] = block_workspace[supernode_block_index(local_col, local_col, width)];
        let start = lower_col_ptrs[column];
        let end = lower_col_ptrs[column + 1];
        let column_rows = &lower_row_indices[start..end];
        let column_values = &mut lower_values[start..end];
        let mut offset = 0;
        for local_row in (local_col + 1)..width {
            let expected_row = supernode.start_column + local_row;
            let Some(&row) = column_rows.get(offset) else {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} is missing dense in-panel row {expected_row}"
                )));
            };
            if row != expected_row {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} expected dense in-panel row {expected_row}, found {row}"
                )));
            }
            column_values[offset] =
                block_workspace[supernode_block_index(local_row, local_col, width)];
            offset += 1;
        }
        for (trailing_offset, &row) in supernode.trailing_rows.iter().enumerate() {
            let Some(&actual_row) = column_rows.get(offset) else {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} is missing trailing row {row}"
                )));
            };
            if actual_row != row {
                return Err(SsidsError::InvalidMatrix(format!(
                    "supernode column {column} expected trailing row {row}, found {actual_row}"
                )));
            }
            column_values[offset] =
                trailing_workspace[supernode_block_index(trailing_offset, local_col, width)];
            offset += 1;
        }
        if offset != column_rows.len() {
            return Err(SsidsError::InvalidMatrix(format!(
                "supernode column {column} has unexpected extra rows beyond its packed panel"
            )));
        }
    }
    Ok(())
}

fn pack_numeric_supernodes(
    supernodes: &[Supernode],
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
    lower_values: &[f64],
    diag: &[f64],
) -> Result<Vec<NumericSupernode>, SsidsError> {
    supernodes
        .iter()
        .map(|supernode| {
            let width = supernode.width();
            let mut diagonal_block = vec![0.0; width * width];
            let mut trailing_block = vec![0.0; supernode.trailing_rows.len() * width];
            for local_col in 0..width {
                let column = supernode.start_column + local_col;
                diagonal_block[supernode_block_index(local_col, local_col, width)] = diag[column];
                let start = lower_col_ptrs[column];
                let end = lower_col_ptrs[column + 1];
                let column_rows = &lower_row_indices[start..end];
                let column_values = &lower_values[start..end];
                for local_row in (local_col + 1)..width {
                    let row = supernode.start_column + local_row;
                    let offset = column_rows.binary_search(&row).map_err(|_| {
                        SsidsError::InvalidMatrix(format!(
                            "supernode column {column} is missing dense in-panel row {row}"
                        ))
                    })?;
                    diagonal_block[supernode_block_index(local_row, local_col, width)] =
                        column_values[offset];
                }
                for (trailing_offset, &row) in supernode.trailing_rows.iter().enumerate() {
                    let offset = column_rows.binary_search(&row).map_err(|_| {
                        SsidsError::InvalidMatrix(format!(
                            "supernode column {column} is missing trailing row {row}"
                        ))
                    })?;
                    trailing_block[supernode_block_index(trailing_offset, local_col, width)] =
                        column_values[offset];
                }
            }
            Ok(NumericSupernode {
                start_column: supernode.start_column,
                end_column: supernode.end_column,
                trailing_rows: supernode.trailing_rows.clone(),
                diagonal_block,
                trailing_block,
            })
        })
        .collect()
}

fn rebuild_numeric_supernodes(
    supernodes: &mut [NumericSupernode],
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
    lower_values: &[f64],
    diag: &[f64],
) -> Result<(), SsidsError> {
    let symbolic_supernodes = supernodes
        .iter()
        .map(|supernode| Supernode {
            start_column: supernode.start_column,
            end_column: supernode.end_column,
            trailing_rows: supernode.trailing_rows.clone(),
        })
        .collect::<Vec<_>>();
    let rebuilt = pack_numeric_supernodes(
        &symbolic_supernodes,
        lower_col_ptrs,
        lower_row_indices,
        lower_values,
        diag,
    )?;
    for (target, source) in supernodes.iter_mut().zip(rebuilt.into_iter()) {
        target.diagonal_block = source.diagonal_block;
        target.trailing_block = source.trailing_block;
    }
    Ok(())
}

fn find_row_index(
    lower_col_ptrs: &[usize],
    lower_row_indices: &[usize],
    column: usize,
    row: usize,
) -> Result<usize, SsidsError> {
    let start = lower_col_ptrs[column];
    let end = lower_col_ptrs[column + 1];
    lower_row_indices[start..end]
        .binary_search(&row)
        .map(|offset| start + offset)
        .map_err(|_| {
            SsidsError::InvalidMatrix(format!(
                "symbolic pattern for column {column} is missing numeric row {row}"
            ))
        })
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
