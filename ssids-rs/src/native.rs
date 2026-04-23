use std::ffi::c_void;
#[cfg(feature = "dynamic-spral-parity")]
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

#[cfg(feature = "dynamic-spral-parity")]
use libloading::{Library, Symbol};
use thiserror::Error;

use crate::{Inertia, NumericFactorOptions, PivotMethod, SymmetricCscMatrix};
#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
use spral_src as _;

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
type SpralAnalysePtr32Fn = unsafe extern "C" fn(
    bool,
    i32,
    *mut i32,
    *const i32,
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
type SpralFactorPtr32Fn = unsafe extern "C" fn(
    bool,
    *const i32,
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
type SpralSolveFn = unsafe extern "C" fn(
    i32,
    i32,
    *mut f64,
    i32,
    *mut c_void,
    *mut c_void,
    *const SpralSsidsOptions,
    *mut SpralSsidsInform,
);
type SpralEnquireIndefFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const SpralSsidsOptions,
    *mut SpralSsidsInform,
    *mut i32,
    *mut f64,
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
    #[cfg(all(
        feature = "dynamic-spral-parity",
        not(any(
            feature = "native-spral-src",
            feature = "native-spral-src-pthreads",
            feature = "native-spral-src-openmp"
        ))
    ))]
    _library: Library,
    #[cfg(any(
        feature = "native-spral-src",
        feature = "native-spral-src-pthreads",
        feature = "native-spral-src-openmp"
    ))]
    _linked: (),
    default_options: SpralDefaultOptionsFn,
    analyse: SpralAnalyseFn,
    analyse_ptr32: SpralAnalysePtr32Fn,
    factor: SpralFactorFn,
    factor_ptr32: SpralFactorPtr32Fn,
    solve1: SpralSolve1Fn,
    solve: SpralSolveFn,
    enquire_indef: SpralEnquireIndefFn,
    free: SpralFreeFn,
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
unsafe extern "C" {
    fn spral_ssids_default_options(options: *mut SpralSsidsOptions);
    fn spral_ssids_analyse(
        check: bool,
        n: i32,
        order: *mut i32,
        ptr: *const i64,
        row: *const i32,
        val: *const f64,
        akeep: *mut *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_analyse_ptr32(
        check: bool,
        n: i32,
        order: *mut i32,
        ptr: *const i32,
        row: *const i32,
        val: *const f64,
        akeep: *mut *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_factor(
        posdef: bool,
        ptr: *const i64,
        row: *const i32,
        val: *const f64,
        scaling: *mut f64,
        akeep: *mut c_void,
        fkeep: *mut *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_factor_ptr32(
        posdef: bool,
        ptr: *const i32,
        row: *const i32,
        val: *const f64,
        scaling: *mut f64,
        akeep: *mut c_void,
        fkeep: *mut *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_solve1(
        job: i32,
        rhs: *mut f64,
        akeep: *mut c_void,
        fkeep: *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_solve(
        job: i32,
        nrhs: i32,
        rhs: *mut f64,
        ldx: i32,
        akeep: *mut c_void,
        fkeep: *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_enquire_indef(
        akeep: *const c_void,
        fkeep: *const c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
        order: *mut i32,
        dinv: *mut f64,
    );
    fn spral_ssids_free(akeep: *mut *mut c_void, fkeep: *mut *mut c_void) -> i32;
}

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp"
))]
fn linked_spral_library() -> NativeSpralLibrary {
    NativeSpralLibrary {
        _linked: (),
        default_options: spral_ssids_default_options,
        analyse: spral_ssids_analyse,
        analyse_ptr32: spral_ssids_analyse_ptr32,
        factor: spral_ssids_factor,
        factor_ptr32: spral_ssids_factor_ptr32,
        solve1: spral_ssids_solve1,
        solve: spral_ssids_solve,
        enquire_indef: spral_ssids_enquire_indef,
        free: spral_ssids_free,
    }
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
    Matching,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeOrderingSpec<'a> {
    LibraryDefault,
    Natural,
    Matching,
    /// User order in SPRAL's C-facing convention: `order[original_column]`
    /// is the zero-based position of that column in the pivot sequence.
    UserSupplied(&'a [usize]),
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

#[derive(Clone, Debug, PartialEq)]
pub struct NativeSpralIndefEnquiry {
    /// Position of each original column in the pivot sequence.
    pub pivot_order: Vec<usize>,
    /// Inverse-D entries in pivot order. For a 1x1 pivot this stores
    /// `[d11, 0.0]`; for a 2x2 pivot the two consecutive entries store
    /// `[d11, d21]` and `[d22, 0.0]`.
    pub inverse_diagonal_entries: Vec<[f64; 2]>,
}

#[derive(Debug)]
pub struct NativeSpralSession {
    inner: Arc<NativeSpralLibrary>,
    dimension: usize,
    pattern_col_ptrs: Vec<usize>,
    pattern_row_indices: Vec<usize>,
    pattern_col_ptrs64: Vec<i64>,
    pattern_col_ptrs32: Vec<i32>,
    pattern_row_indices32: Vec<i32>,
    use_ipopt_ptr32_indexing: bool,
    options: SpralSsidsOptions,
    scaling: Option<Vec<f64>>,
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
    #[error("native SPRAL invalid user ordering: {0}")]
    InvalidOrdering(String),
    #[error("native SPRAL analyse failed with flag {flag}")]
    AnalyseFailed { flag: i32 },
    #[error("native SPRAL factorization failed with flag {flag}")]
    FactorizationFailed { flag: i32 },
    #[error("native SPRAL solve failed with flag {flag}")]
    SolveFailed { flag: i32 },
    #[error("native SPRAL indefinite enquiry failed with flag {flag}")]
    EnquireFailed { flag: i32 },
    #[error("native SPRAL session has not been factorized yet")]
    NotFactorized,
}

impl NativeSpral {
    pub fn load() -> Result<Self, NativeSpralError> {
        #[cfg(any(
            feature = "native-spral-src",
            feature = "native-spral-src-pthreads",
            feature = "native-spral-src-openmp"
        ))]
        {
            let native = Self {
                inner: Arc::new(linked_spral_library()),
            };
            native_spral_smoke_test(&native)?;
            Ok(native)
        }

        #[cfg(all(
            not(any(
                feature = "native-spral-src",
                feature = "native-spral-src-pthreads",
                feature = "native-spral-src-openmp"
            )),
            feature = "dynamic-spral-parity"
        ))]
        {
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
                        let analyse_ptr32 = load_symbol::<SpralAnalysePtr32Fn>(
                            &library,
                            &candidate,
                            b"spral_ssids_analyse_ptr32\0",
                        )?;
                        let factor = load_symbol::<SpralFactorFn>(
                            &library,
                            &candidate,
                            b"spral_ssids_factor\0",
                        )?;
                        let factor_ptr32 = load_symbol::<SpralFactorPtr32Fn>(
                            &library,
                            &candidate,
                            b"spral_ssids_factor_ptr32\0",
                        )?;
                        let solve1 = load_symbol::<SpralSolve1Fn>(
                            &library,
                            &candidate,
                            b"spral_ssids_solve1\0",
                        )?;
                        let solve = load_symbol::<SpralSolveFn>(
                            &library,
                            &candidate,
                            b"spral_ssids_solve\0",
                        )?;
                        let enquire_indef = load_symbol::<SpralEnquireIndefFn>(
                            &library,
                            &candidate,
                            b"spral_ssids_enquire_indef\0",
                        )?;
                        let free = load_symbol::<SpralFreeFn>(
                            &library,
                            &candidate,
                            b"spral_ssids_free\0",
                        )?;
                        let native = Self {
                            inner: Arc::new(NativeSpralLibrary {
                                _library: library,
                                default_options,
                                analyse,
                                analyse_ptr32,
                                factor,
                                factor_ptr32,
                                solve1,
                                solve,
                                enquire_indef,
                                free,
                            }),
                        };
                        match native_spral_smoke_test(&native) {
                            Ok(()) => return Ok(native),
                            Err(error) => {
                                last_error = Some(format!(
                                    "{} loaded but failed a factorization smoke test: {error}",
                                    candidate.display()
                                ));
                            }
                        }
                    }
                    Err(error) => last_error = Some(error.to_string()),
                }
            }
            Err(NativeSpralError::LoadLibrary(
                last_error.unwrap_or_else(|| "no candidates tried".into()),
            ))
        }

        #[cfg(all(
            not(any(
                feature = "native-spral-src",
                feature = "native-spral-src-pthreads",
                feature = "native-spral-src-openmp"
            )),
            not(feature = "dynamic-spral-parity")
        ))]
        {
            Err(NativeSpralError::LoadLibrary(
                "native SPRAL is disabled; enable `native-spral-src`, `native-spral-src-pthreads`, or `native-spral-src-openmp` for the source-built distribution path, or `dynamic-spral-parity` for parity-only dynamic loading"
                    .into(),
            ))
        }
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
        let ordering = match ordering {
            NativeOrdering::LibraryDefault => NativeOrderingSpec::LibraryDefault,
            NativeOrdering::Natural => NativeOrderingSpec::Natural,
            NativeOrdering::Matching => NativeOrderingSpec::Matching,
        };
        self.analyse_with_ordering_spec(matrix, numeric_options, ordering, false)
    }

    /// Analyse through the same SPRAL C entrypoints and index convention used by
    /// IPOPT's `IpSpralSolverInterface.cpp`: `spral_ssids_analyse_ptr32` with
    /// one-based compressed sparse indices, followed by `factor_ptr32`.
    pub fn analyse_ipopt_compatible_with_options_and_ordering(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        numeric_options: &NumericFactorOptions,
        ordering: NativeOrdering,
    ) -> Result<NativeSpralSession, NativeSpralError> {
        let ordering = match ordering {
            NativeOrdering::LibraryDefault => NativeOrderingSpec::LibraryDefault,
            NativeOrdering::Natural => NativeOrderingSpec::Natural,
            NativeOrdering::Matching => NativeOrderingSpec::Matching,
        };
        self.analyse_with_ordering_spec(matrix, numeric_options, ordering, true)
    }

    /// Analyse with an explicit user ordering.
    ///
    /// `order[original_column]` is the zero-based position of that original
    /// column in the pivot sequence, matching SPRAL's C interface convention.
    pub fn analyse_with_options_and_user_ordering(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        numeric_options: &NumericFactorOptions,
        order: &[usize],
    ) -> Result<NativeSpralSession, NativeSpralError> {
        self.analyse_with_ordering_spec(
            matrix,
            numeric_options,
            NativeOrderingSpec::UserSupplied(order),
            false,
        )
    }

    fn analyse_with_ordering_spec(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        numeric_options: &NumericFactorOptions,
        ordering: NativeOrderingSpec<'_>,
        use_ipopt_ptr32_indexing: bool,
    ) -> Result<NativeSpralSession, NativeSpralError> {
        if matrix.dimension() > i32::MAX as usize {
            return Err(NativeSpralError::DimensionTooLarge {
                dimension: matrix.dimension(),
            });
        }
        if use_ipopt_ptr32_indexing
            && matrix.col_ptrs().last().copied().unwrap_or(0) >= i32::MAX as usize
        {
            return Err(NativeSpralError::DimensionTooLarge {
                dimension: matrix.col_ptrs().last().copied().unwrap_or(0),
            });
        }
        let col_ptrs64 = matrix
            .col_ptrs()
            .iter()
            .map(|&entry| i64::try_from(entry).unwrap_or(i64::MAX))
            .collect::<Vec<_>>();
        let pattern_col_ptrs32 = if use_ipopt_ptr32_indexing {
            matrix
                .col_ptrs()
                .iter()
                .map(|&entry| i32::try_from(entry + 1).unwrap_or(i32::MAX))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let row_indices32 = if use_ipopt_ptr32_indexing {
            matrix
                .row_indices()
                .iter()
                .map(|&entry| i32::try_from(entry + 1).unwrap_or(i32::MAX))
                .collect::<Vec<_>>()
        } else {
            matrix
                .row_indices()
                .iter()
                .map(|&entry| i32::try_from(entry).unwrap_or(i32::MAX))
                .collect::<Vec<_>>()
        };
        let mut options = unsafe { std::mem::zeroed::<SpralSsidsOptions>() };
        unsafe { (self.inner.default_options)(&mut options) };
        options.array_base = if use_ipopt_ptr32_indexing { 1 } else { 0 };
        options.use_gpu = false;
        options.ignore_numa = true;
        options.print_level = -1;
        apply_numeric_factor_options(&mut options, numeric_options);
        let mut order_storage = None;
        let order_array_base = options.array_base;
        apply_native_ordering(
            &mut options,
            matrix.dimension(),
            ordering,
            order_array_base,
            &mut order_storage,
        )?;

        let mut inform = SpralSsidsInform::default();
        let mut analysed = ptr::null_mut();
        let values_ptr = if ordering == NativeOrderingSpec::Matching {
            matrix
                .values()
                .ok_or(NativeSpralError::MissingValues)?
                .as_ptr()
        } else {
            matrix
                .values()
                .map_or(ptr::null(), |values| values.as_ptr())
        };

        unsafe {
            if use_ipopt_ptr32_indexing {
                (self.inner.analyse_ptr32)(
                    false,
                    i32::try_from(matrix.dimension()).unwrap_or(i32::MAX),
                    order_storage
                        .as_mut()
                        .map_or(ptr::null_mut(), |order| order.as_mut_ptr()),
                    pattern_col_ptrs32.as_ptr(),
                    row_indices32.as_ptr(),
                    values_ptr,
                    &mut analysed,
                    &options,
                    &mut inform,
                );
            } else {
                (self.inner.analyse)(
                    false,
                    i32::try_from(matrix.dimension()).unwrap_or(i32::MAX),
                    order_storage
                        .as_mut()
                        .map_or(ptr::null_mut(), |order| order.as_mut_ptr()),
                    col_ptrs64.as_ptr(),
                    row_indices32.as_ptr(),
                    values_ptr,
                    &mut analysed,
                    &options,
                    &mut inform,
                );
            }
        }
        if inform.flag < 0 {
            let mut factorized = ptr::null_mut();
            unsafe {
                let _ = (self.inner.free)(&mut analysed, &mut factorized);
            }
            return Err(NativeSpralError::AnalyseFailed { flag: inform.flag });
        }
        let scaling = (options.scaling != 0).then(|| vec![0.0; matrix.dimension()]);
        Ok(NativeSpralSession {
            inner: Arc::clone(&self.inner),
            dimension: matrix.dimension(),
            pattern_col_ptrs: matrix.col_ptrs().to_vec(),
            pattern_row_indices: matrix.row_indices().to_vec(),
            pattern_col_ptrs64: col_ptrs64,
            pattern_col_ptrs32,
            pattern_row_indices32: row_indices32,
            use_ipopt_ptr32_indexing,
            options,
            scaling,
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

#[cfg(any(
    feature = "native-spral-src",
    feature = "native-spral-src-pthreads",
    feature = "native-spral-src-openmp",
    feature = "dynamic-spral-parity"
))]
fn native_spral_smoke_test(native: &NativeSpral) -> Result<(), NativeSpralError> {
    let col_ptrs = [0, 2, 3];
    let row_indices = [0, 1, 1];
    let values = [2.0, -1.0, 2.0];
    let matrix = SymmetricCscMatrix::new(2, &col_ptrs, &row_indices, Some(&values))
        .expect("hard-coded native SPRAL smoke matrix is valid");
    let mut session = native.analyse(matrix)?;
    session.factorize(matrix)?;
    let mut rhs = [1.0, 0.0];
    session.solve_in_place(&mut rhs)?;
    Ok(())
}

#[cfg(feature = "dynamic-spral-parity")]
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
        PathBuf::from("libspral.so"),
    ]);
    candidates
}

fn apply_numeric_factor_options(native: &mut SpralSsidsOptions, numeric: &NumericFactorOptions) {
    native.action = numeric.action_on_zero_pivot;
    // SPRAL's C API copies this value directly into the Fortran
    // `ssids_options%pivot_method`, whose constants are 1-based:
    // APP aggressive=1, APP block=2, TPP=3.
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
    ordering: NativeOrderingSpec<'_>,
    array_base: i32,
    order_storage: &mut Option<Vec<i32>>,
) -> Result<(), NativeSpralError> {
    match ordering {
        NativeOrderingSpec::LibraryDefault => {}
        NativeOrderingSpec::Natural => {
            native.ordering = 0;
            *order_storage = Some(
                (0..dimension)
                    .map(|index| i32::try_from(index).unwrap_or(i32::MAX) + array_base)
                    .collect(),
            );
        }
        NativeOrderingSpec::Matching => {
            native.ordering = 2;
            native.scaling = 3;
        }
        NativeOrderingSpec::UserSupplied(order) => {
            if order.len() != dimension {
                return Err(NativeSpralError::InvalidOrdering(format!(
                    "expected {dimension} entries, got {}",
                    order.len()
                )));
            }
            let mut seen = vec![false; dimension];
            for (original, &position) in order.iter().enumerate() {
                if position >= dimension {
                    return Err(NativeSpralError::InvalidOrdering(format!(
                        "order[{original}]={position} is out of bounds for {dimension} columns"
                    )));
                }
                if seen[position] {
                    return Err(NativeSpralError::InvalidOrdering(format!(
                        "duplicate pivot position {position}"
                    )));
                }
                seen[position] = true;
            }
            native.ordering = 0;
            *order_storage = Some(
                order
                    .iter()
                    .map(|&position| i32::try_from(position).unwrap_or(i32::MAX) + array_base)
                    .collect(),
            );
        }
    }
    Ok(())
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

    pub fn solve_ipopt_single_rhs(&self, rhs: &[f64]) -> Result<Vec<f64>, NativeSpralError> {
        let mut solution = rhs.to_vec();
        self.solve_ipopt_single_rhs_in_place(&mut solution)?;
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

    pub fn solve_ipopt_single_rhs_in_place(&self, rhs: &mut [f64]) -> Result<(), NativeSpralError> {
        if rhs.len() != self.dimension {
            return Err(NativeSpralError::DimensionMismatch {
                expected: self.dimension,
                actual: rhs.len(),
            });
        }
        if self.factorized.is_null() {
            return Err(NativeSpralError::NotFactorized);
        }
        let ldx =
            i32::try_from(self.dimension).map_err(|_| NativeSpralError::DimensionTooLarge {
                dimension: self.dimension,
            })?;
        let mut inform = SpralSsidsInform::default();
        unsafe {
            (self.inner.solve)(
                0,
                1,
                rhs.as_mut_ptr(),
                ldx,
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

    pub fn enquire_indef(&self) -> Result<NativeSpralIndefEnquiry, NativeSpralError> {
        if self.factorized.is_null() {
            return Err(NativeSpralError::NotFactorized);
        }

        let mut pivot_order = vec![0; self.dimension];
        let mut inverse_diagonal_raw = vec![0.0; 2 * self.dimension];
        let mut inform = SpralSsidsInform::default();
        unsafe {
            (self.inner.enquire_indef)(
                self.analysed,
                self.factorized,
                &self.options,
                &mut inform,
                pivot_order.as_mut_ptr(),
                inverse_diagonal_raw.as_mut_ptr(),
            );
        }
        if inform.flag < 0 {
            return Err(NativeSpralError::EnquireFailed { flag: inform.flag });
        }

        let pivot_order = pivot_order
            .into_iter()
            .map(|entry| usize::try_from(entry).unwrap_or(usize::MAX))
            .collect();
        let inverse_diagonal_entries = inverse_diagonal_raw
            .chunks_exact(2)
            .map(|entry| [entry[0], entry[1]])
            .collect();
        Ok(NativeSpralIndefEnquiry {
            pivot_order,
            inverse_diagonal_entries,
        })
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
        let scaling_ptr = self
            .scaling
            .as_mut()
            .map_or(ptr::null_mut(), |scaling| scaling.as_mut_ptr());
        unsafe {
            if self.use_ipopt_ptr32_indexing {
                (self.inner.factor_ptr32)(
                    false,
                    self.pattern_col_ptrs32.as_ptr(),
                    self.pattern_row_indices32.as_ptr(),
                    values.as_ptr(),
                    scaling_ptr,
                    self.analysed,
                    &mut self.factorized,
                    &self.options,
                    &mut inform,
                );
            } else {
                (self.inner.factor)(
                    false,
                    self.pattern_col_ptrs64.as_ptr(),
                    self.pattern_row_indices32.as_ptr(),
                    values.as_ptr(),
                    scaling_ptr,
                    self.analysed,
                    &mut self.factorized,
                    &self.options,
                    &mut inform,
                );
            }
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

#[cfg(feature = "dynamic-spral-parity")]
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
