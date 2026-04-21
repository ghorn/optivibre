use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use thiserror::Error;

use crate::{Inertia, NumericFactorOptions, PivotMethod, SymmetricCscMatrix};

type SpralDefaultOptionsFn = unsafe extern "C" fn(*mut SpralSsidsOptions);
type SpralAnalyseFn = unsafe extern "C" fn(
    bool,
    i32,
    *mut i32,
    *const i64,
    *const i32,
    *const f64,
    *mut *mut c_void,
    *const SpralSsidsOptions,
    *mut SpralSsidsInform,
);
type SpralFactorFn = unsafe extern "C" fn(
    bool,
    *const i64,
    *const i32,
    *const f64,
    *mut f64,
    *mut c_void,
    *mut *mut c_void,
    *const SpralSsidsOptions,
    *mut SpralSsidsInform,
);
type SpralSolve1Fn = unsafe extern "C" fn(
    i32,
    *mut f64,
    *mut c_void,
    *mut c_void,
    *const SpralSsidsOptions,
    *mut SpralSsidsInform,
);
type SpralFreeFn = unsafe extern "C" fn(*mut *mut c_void, *mut *mut c_void) -> i32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpralSsidsOptions {
    array_base: i32,
    print_level: i32,
    unit_diagnostics: i32,
    unit_error: i32,
    unit_warning: i32,
    ordering: i32,
    nemin: i32,
    ignore_numa: bool,
    use_gpu: bool,
    min_gpu_work: i64,
    max_load_inbalance: f32,
    gpu_perf_coeff: f32,
    scaling: i32,
    small_subtree_threshold: i64,
    cpu_block_size: i32,
    action: bool,
    pivot_method: i32,
    small: f64,
    u: f64,
    unused: [u8; 80],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpralSsidsInform {
    flag: i32,
    matrix_dup: i32,
    matrix_missing_diag: i32,
    matrix_outrange: i32,
    matrix_rank: i32,
    maxdepth: i32,
    maxfront: i32,
    num_delay: i32,
    num_factor: i64,
    num_flops: i64,
    num_neg: i32,
    num_sup: i32,
    num_two: i32,
    stat: i32,
    cuda_error: i32,
    cublas_error: i32,
    maxsupernode: i32,
    unused: [u8; 76],
}

impl Default for SpralSsidsInform {
    fn default() -> Self {
        Self {
            flag: 0,
            matrix_dup: 0,
            matrix_missing_diag: 0,
            matrix_outrange: 0,
            matrix_rank: 0,
            maxdepth: 0,
            maxfront: 0,
            num_delay: 0,
            num_factor: 0,
            num_flops: 0,
            num_neg: 0,
            num_sup: 0,
            num_two: 0,
            stat: 0,
            cuda_error: 0,
            cublas_error: 0,
            maxsupernode: 0,
            unused: [0; 76],
        }
    }
}

#[derive(Debug)]
struct NativeSpralLibrary {
    _library: Library,
    default_options: SpralDefaultOptionsFn,
    analyse: SpralAnalyseFn,
    factor: SpralFactorFn,
    solve1: SpralSolve1Fn,
    free: SpralFreeFn,
}

#[derive(Clone, Debug)]
pub struct NativeSpral {
    inner: Arc<NativeSpralLibrary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NativeOrdering {
    #[default]
    LibraryDefault,
    Natural,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeSpralAnalyseInfo {
    pub supernode_count: usize,
    pub max_front_size: usize,
    pub max_supernode_width: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeSpralFactorInfo {
    pub inertia: Inertia,
    pub two_by_two_pivots: usize,
    pub delayed_pivots: usize,
    pub supernode_count: usize,
    pub max_supernode_width: usize,
}

#[derive(Debug)]
pub struct NativeSpralSession {
    inner: Arc<NativeSpralLibrary>,
    dimension: usize,
    pattern_col_ptrs: Vec<usize>,
    pattern_row_indices: Vec<usize>,
    options: SpralSsidsOptions,
    analyse_info: NativeSpralAnalyseInfo,
    factor_info: Option<NativeSpralFactorInfo>,
    analysed: *mut c_void,
    factorized: *mut c_void,
}

#[derive(Debug, Error)]
pub enum NativeSpralError {
    #[error("unable to load native SPRAL library: {0}")]
    LoadLibrary(String),
    #[error("failed to load `{symbol}` from native SPRAL library `{library}`")]
    MissingSymbol {
        symbol: &'static str,
        library: String,
    },
    #[error("native SPRAL supports at most 32-bit dimensions, got {dimension}")]
    DimensionTooLarge { dimension: usize },
    #[error("native SPRAL requires explicit numeric values")]
    MissingValues,
    #[error("native SPRAL dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("native SPRAL pattern mismatch: {0}")]
    PatternMismatch(String),
    #[error("native SPRAL analyse failed with flag {flag}")]
    AnalyseFailed { flag: i32 },
    #[error("native SPRAL factorization failed with flag {flag}")]
    FactorizationFailed { flag: i32 },
    #[error("native SPRAL solve failed with flag {flag}")]
    SolveFailed { flag: i32 },
    #[error("native SPRAL session has not been factorized yet")]
    NotFactorized,
}

impl NativeSpral {
    pub fn load() -> Result<Self, NativeSpralError> {
        let mut candidates = native_spral_library_candidates();
        if let Ok(cwd) = std::env::current_dir() {
            for ancestor in cwd.ancestors() {
                candidates.insert(
                    0,
                    ancestor.join("target/native/spral-upstream/builddir/libspral.dylib"),
                );
            }
        }
        let mut last_error = None;
        for candidate in candidates {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => {
                    let default_options = load_symbol::<SpralDefaultOptionsFn>(
                        &library,
                        &candidate,
                        b"spral_ssids_default_options\0",
                    )?;
                    let analyse = load_symbol::<SpralAnalyseFn>(
                        &library,
                        &candidate,
                        b"spral_ssids_analyse\0",
                    )?;
                    let factor = load_symbol::<SpralFactorFn>(
                        &library,
                        &candidate,
                        b"spral_ssids_factor\0",
                    )?;
                    let solve1 = load_symbol::<SpralSolve1Fn>(
                        &library,
                        &candidate,
                        b"spral_ssids_solve1\0",
                    )?;
                    let free =
                        load_symbol::<SpralFreeFn>(&library, &candidate, b"spral_ssids_free\0")?;
                    return Ok(Self {
                        inner: Arc::new(NativeSpralLibrary {
                            _library: library,
                            default_options,
                            analyse,
                            factor,
                            solve1,
                            free,
                        }),
                    });
                }
                Err(error) => last_error = Some(error.to_string()),
            }
        }
        Err(NativeSpralError::LoadLibrary(
            last_error.unwrap_or_else(|| "no candidates tried".into()),
        ))
    }

    pub fn analyse(
        &self,
        matrix: SymmetricCscMatrix<'_>,
    ) -> Result<NativeSpralSession, NativeSpralError> {
        self.analyse_with_options_and_ordering(
            matrix,
            &NumericFactorOptions::default(),
            NativeOrdering::LibraryDefault,
        )
    }

    pub fn analyse_with_options(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        numeric_options: &NumericFactorOptions,
    ) -> Result<NativeSpralSession, NativeSpralError> {
        self.analyse_with_options_and_ordering(
            matrix,
            numeric_options,
            NativeOrdering::LibraryDefault,
        )
    }

    pub fn analyse_with_options_and_ordering(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        numeric_options: &NumericFactorOptions,
        ordering: NativeOrdering,
    ) -> Result<NativeSpralSession, NativeSpralError> {
        if matrix.dimension() > i32::MAX as usize {
            return Err(NativeSpralError::DimensionTooLarge {
                dimension: matrix.dimension(),
            });
        }
        let col_ptrs64 = matrix
            .col_ptrs()
            .iter()
            .map(|&entry| i64::try_from(entry).unwrap_or(i64::MAX))
            .collect::<Vec<_>>();
        let row_indices32 = matrix
            .row_indices()
            .iter()
            .map(|&entry| i32::try_from(entry).unwrap_or(i32::MAX))
            .collect::<Vec<_>>();
        let mut options = unsafe { std::mem::zeroed::<SpralSsidsOptions>() };
        unsafe { (self.inner.default_options)(&mut options) };
        options.array_base = 0;
        options.use_gpu = false;
        options.ignore_numa = true;
        options.print_level = 0;
        apply_numeric_factor_options(&mut options, numeric_options);
        let mut order_storage = None;
        apply_native_ordering(
            &mut options,
            matrix.dimension(),
            ordering,
            &mut order_storage,
        );

        let mut inform = SpralSsidsInform::default();
        let mut analysed = ptr::null_mut();
        unsafe {
            (self.inner.analyse)(
                true,
                i32::try_from(matrix.dimension()).unwrap_or(i32::MAX),
                order_storage
                    .as_mut()
                    .map_or(ptr::null_mut(), |order| order.as_mut_ptr()),
                col_ptrs64.as_ptr(),
                row_indices32.as_ptr(),
                ptr::null(),
                &mut analysed,
                &options,
                &mut inform,
            );
        }
        if inform.flag < 0 {
            let mut factorized = ptr::null_mut();
            unsafe {
                let _ = (self.inner.free)(&mut analysed, &mut factorized);
            }
            return Err(NativeSpralError::AnalyseFailed { flag: inform.flag });
        }
        Ok(NativeSpralSession {
            inner: Arc::clone(&self.inner),
            dimension: matrix.dimension(),
            pattern_col_ptrs: matrix.col_ptrs().to_vec(),
            pattern_row_indices: matrix.row_indices().to_vec(),
            options,
            analyse_info: NativeSpralAnalyseInfo {
                supernode_count: inform.num_sup.max(0) as usize,
                max_front_size: inform.maxfront.max(0) as usize,
                max_supernode_width: inform.maxsupernode.max(0) as usize,
            },
            factor_info: None,
            analysed,
            factorized: ptr::null_mut(),
        })
    }
}

fn native_spral_library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(override_path) = std::env::var_os("SPRAL_SSIDS_NATIVE_LIB") {
        candidates.push(PathBuf::from(override_path));
    }
    candidates.extend([
        PathBuf::from("/Users/greg/local/ipopt-spral/lib/libspral.dylib"),
        PathBuf::from("/Users/greg/local/ipopt-spral/lib/libspral.so"),
        PathBuf::from("target/native/spral-upstream/builddir/libspral.dylib"),
        PathBuf::from("libspral.dylib"),
        PathBuf::from("/usr/local/lib/libspral.dylib"),
        PathBuf::from("/opt/homebrew/lib/libspral.dylib"),
        PathBuf::from("libspral.so"),
        PathBuf::from("/usr/local/lib/libspral.so"),
    ]);
    candidates
}

fn apply_numeric_factor_options(native: &mut SpralSsidsOptions, numeric: &NumericFactorOptions) {
    native.action = numeric.action_on_zero_pivot;
    native.pivot_method = match numeric.pivot_method {
        PivotMethod::AggressiveAposteriori => 1,
        PivotMethod::BlockAposteriori => 2,
        PivotMethod::ThresholdPartial => 3,
    };
    native.small = numeric.small_pivot_tolerance;
    native.u = numeric.threshold_pivot_u;
}

fn apply_native_ordering(
    native: &mut SpralSsidsOptions,
    dimension: usize,
    ordering: NativeOrdering,
    order_storage: &mut Option<Vec<i32>>,
) {
    match ordering {
        NativeOrdering::LibraryDefault => {}
        NativeOrdering::Natural => {
            native.ordering = 0;
            *order_storage = Some(
                (0..dimension)
                    .map(|index| i32::try_from(index).unwrap_or(i32::MAX))
                    .collect(),
            );
        }
    }
}

impl NativeSpralSession {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn analyse_info(&self) -> NativeSpralAnalyseInfo {
        self.analyse_info
    }

    pub fn factor_info(&self) -> Option<NativeSpralFactorInfo> {
        self.factor_info
    }

    pub fn factorize(
        &mut self,
        matrix: SymmetricCscMatrix<'_>,
    ) -> Result<NativeSpralFactorInfo, NativeSpralError> {
        self.ensure_pattern_matches(matrix)?;
        self.factorize_values(matrix.values().ok_or(NativeSpralError::MissingValues)?)
    }

    pub fn refactorize(
        &mut self,
        matrix: SymmetricCscMatrix<'_>,
    ) -> Result<NativeSpralFactorInfo, NativeSpralError> {
        self.ensure_pattern_matches(matrix)?;
        self.factorize_values(matrix.values().ok_or(NativeSpralError::MissingValues)?)
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, NativeSpralError> {
        let mut solution = rhs.to_vec();
        self.solve_in_place(&mut solution)?;
        Ok(solution)
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), NativeSpralError> {
        if rhs.len() != self.dimension {
            return Err(NativeSpralError::DimensionMismatch {
                expected: self.dimension,
                actual: rhs.len(),
            });
        }
        if self.factorized.is_null() {
            return Err(NativeSpralError::NotFactorized);
        }
        let mut inform = SpralSsidsInform::default();
        unsafe {
            (self.inner.solve1)(
                0,
                rhs.as_mut_ptr(),
                self.analysed,
                self.factorized,
                &self.options,
                &mut inform,
            );
        }
        if inform.flag < 0 {
            return Err(NativeSpralError::SolveFailed { flag: inform.flag });
        }
        Ok(())
    }

    fn factorize_values(
        &mut self,
        values: &[f64],
    ) -> Result<NativeSpralFactorInfo, NativeSpralError> {
        if values.len() != self.pattern_row_indices.len() {
            return Err(NativeSpralError::PatternMismatch(format!(
                "value length mismatch: expected {}, got {}",
                self.pattern_row_indices.len(),
                values.len()
            )));
        }
        let mut inform = SpralSsidsInform::default();
        unsafe {
            (self.inner.factor)(
                false,
                ptr::null(),
                ptr::null(),
                values.as_ptr(),
                ptr::null_mut(),
                self.analysed,
                &mut self.factorized,
                &self.options,
                &mut inform,
            );
        }
        if inform.flag < 0 {
            return Err(NativeSpralError::FactorizationFailed { flag: inform.flag });
        }
        let factor_info = NativeSpralFactorInfo {
            inertia: inertia_from_inform(self.dimension, &inform),
            two_by_two_pivots: inform.num_two.max(0) as usize,
            delayed_pivots: inform.num_delay.max(0) as usize,
            supernode_count: inform.num_sup.max(0) as usize,
            max_supernode_width: inform.maxsupernode.max(0) as usize,
        };
        self.factor_info = Some(factor_info);
        Ok(factor_info)
    }

    fn ensure_pattern_matches(
        &self,
        matrix: SymmetricCscMatrix<'_>,
    ) -> Result<(), NativeSpralError> {
        if matrix.dimension() != self.dimension {
            return Err(NativeSpralError::DimensionMismatch {
                expected: self.dimension,
                actual: matrix.dimension(),
            });
        }
        if matrix.col_ptrs() != self.pattern_col_ptrs.as_slice()
            || matrix.row_indices() != self.pattern_row_indices.as_slice()
        {
            return Err(NativeSpralError::PatternMismatch(
                "refactorization requires identical CSC sparsity structure".into(),
            ));
        }
        Ok(())
    }
}

impl Drop for NativeSpralSession {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.inner.free)(&mut self.analysed, &mut self.factorized);
        }
    }
}

fn load_symbol<T: Copy>(
    library: &Library,
    candidate: &Path,
    symbol_name: &'static [u8],
) -> Result<T, NativeSpralError> {
    let symbol = unsafe {
        library
            .get::<Symbol<'_, T>>(symbol_name)
            .map_err(|_| NativeSpralError::MissingSymbol {
                symbol: std::str::from_utf8(symbol_name)
                    .unwrap_or("unknown")
                    .trim_end_matches('\0'),
                library: candidate.display().to_string(),
            })?
    };
    Ok(**symbol)
}

fn inertia_from_inform(dimension: usize, inform: &SpralSsidsInform) -> Inertia {
    let zero = if inform.matrix_rank > 0 && (inform.matrix_rank as usize) < dimension {
        dimension - inform.matrix_rank as usize
    } else {
        0
    };
    let negative = inform.num_neg.max(0) as usize;
    let positive = dimension.saturating_sub(negative + zero);
    Inertia {
        positive,
        negative,
        zero,
    }
}
