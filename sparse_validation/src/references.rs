use std::{ffi::c_void, path::PathBuf, ptr, time::Instant};

use amd::Control;
use anyhow::{Context, Result, bail};
use libloading::{Library, Symbol};
use metis_ordering::{CsrGraph, Permutation};
use spral_ssids::SymmetricCscMatrix;

use crate::corpus::SymmetricPatternMatrix;

type MetisNodeNdFn = unsafe extern "C" fn(
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
) -> i32;

const METIS_OK: i32 = 1;

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
pub struct NativeMetisReference {
    _library: Library,
    node_nd: MetisNodeNdFn,
}

#[derive(Debug)]
pub struct NativeSpralReference {
    _library: Library,
    default_options: SpralDefaultOptionsFn,
    analyse: SpralAnalyseFn,
    factor: SpralFactorFn,
    solve1: SpralSolve1Fn,
    free: SpralFreeFn,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeSpralNumericResult {
    pub factor_elapsed_ms: f64,
    pub solve_elapsed_ms: f64,
    pub refactor_elapsed_ms: Option<f64>,
    pub solution: Vec<f64>,
    pub refactor_solution: Option<Vec<f64>>,
    pub num_neg: i32,
    pub num_two: i32,
    pub num_delay: i32,
    pub num_sup: i32,
    pub max_supernode: i32,
}

impl NativeMetisReference {
    pub fn load() -> Result<Self> {
        let candidates = [
            "libmetis.dylib",
            "/opt/homebrew/lib/libmetis.dylib",
            "/opt/homebrew/opt/metis/lib/libmetis.dylib",
            "/usr/local/lib/libmetis.dylib",
            "libmetis.so",
            "/usr/lib/libmetis.so",
            "/usr/local/lib/libmetis.so",
            "metis.dll",
        ];
        let mut last_error = None;
        for candidate in candidates {
            match unsafe { Library::new(candidate) } {
                Ok(library) => {
                    let node_nd = unsafe {
                        let symbol: Symbol<'_, MetisNodeNdFn> =
                            library.get(b"METIS_NodeND\0").with_context(|| {
                                format!("failed to load METIS_NodeND from {candidate}")
                            })?;
                        *symbol
                    };
                    return Ok(Self {
                        _library: library,
                        node_nd,
                    });
                }
                Err(error) => last_error = Some(error),
            }
        }
        bail!(
            "unable to load libmetis from common locations: {}",
            last_error
                .map(|error| error.to_string())
                .unwrap_or_else(|| "no candidates tried".into())
        )
    }

    pub fn order(&self, graph: &CsrGraph) -> Result<Permutation> {
        if graph.vertex_count() > i32::MAX as usize || graph.edge_count() * 2 > i32::MAX as usize {
            bail!("graph is too large for 32-bit native METIS adapter");
        }

        let mut nvtxs = graph.vertex_count() as i32;
        let mut xadj = Vec::with_capacity(graph.vertex_count() + 1);
        xadj.push(0);
        let mut adjncy = Vec::with_capacity(graph.edge_count() * 2);
        for vertex in 0..graph.vertex_count() {
            adjncy.extend(
                graph
                    .neighbors(vertex)
                    .iter()
                    .map(|&neighbor| neighbor as i32),
            );
            xadj.push(adjncy.len() as i32);
        }
        let mut perm = vec![0_i32; graph.vertex_count()];
        let mut iperm = vec![0_i32; graph.vertex_count()];
        let status = unsafe {
            (self.node_nd)(
                &mut nvtxs,
                xadj.as_mut_ptr(),
                adjncy.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                perm.as_mut_ptr(),
                iperm.as_mut_ptr(),
            )
        };
        if status != METIS_OK {
            bail!("METIS_NodeND returned status {status}");
        }
        let permutation = perm
            .into_iter()
            .map(|entry| usize::try_from(entry).unwrap_or(usize::MAX))
            .collect::<Vec<_>>();
        Ok(Permutation::new(permutation)?)
    }
}

impl NativeSpralReference {
    pub fn load() -> Result<Self> {
        let mut candidates = vec![
            PathBuf::from("target/native/spral-upstream/builddir/libspral.dylib"),
            PathBuf::from("libspral.dylib"),
            PathBuf::from("/usr/local/lib/libspral.dylib"),
            PathBuf::from("/opt/homebrew/lib/libspral.dylib"),
            PathBuf::from("libspral.so"),
            PathBuf::from("/usr/local/lib/libspral.so"),
        ];
        if let Ok(cwd) = std::env::current_dir() {
            candidates.insert(
                0,
                cwd.join("target/native/spral-upstream/builddir/libspral.dylib"),
            );
        }
        let mut last_error = None;
        for candidate in candidates {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => {
                    let default_options = unsafe {
                        let symbol: Symbol<'_, SpralDefaultOptionsFn> = library
                            .get(b"spral_ssids_default_options\0")
                            .with_context(|| {
                                format!(
                                    "failed to load spral_ssids_default_options from {}",
                                    candidate.display()
                                )
                            })?;
                        *symbol
                    };
                    let analyse = unsafe {
                        let symbol: Symbol<'_, SpralAnalyseFn> =
                            library.get(b"spral_ssids_analyse\0").with_context(|| {
                                format!(
                                    "failed to load spral_ssids_analyse from {}",
                                    candidate.display()
                                )
                            })?;
                        *symbol
                    };
                    let factor = unsafe {
                        let symbol: Symbol<'_, SpralFactorFn> =
                            library.get(b"spral_ssids_factor\0").with_context(|| {
                                format!(
                                    "failed to load spral_ssids_factor from {}",
                                    candidate.display()
                                )
                            })?;
                        *symbol
                    };
                    let solve1 = unsafe {
                        let symbol: Symbol<'_, SpralSolve1Fn> =
                            library.get(b"spral_ssids_solve1\0").with_context(|| {
                                format!(
                                    "failed to load spral_ssids_solve1 from {}",
                                    candidate.display()
                                )
                            })?;
                        *symbol
                    };
                    let free = unsafe {
                        let symbol: Symbol<'_, SpralFreeFn> =
                            library.get(b"spral_ssids_free\0").with_context(|| {
                                format!(
                                    "failed to load spral_ssids_free from {}",
                                    candidate.display()
                                )
                            })?;
                        *symbol
                    };
                    return Ok(Self {
                        _library: library,
                        default_options,
                        analyse,
                        factor,
                        solve1,
                        free,
                    });
                }
                Err(error) => last_error = Some(error),
            }
        }
        bail!(
            "unable to load libspral from common locations: {}",
            last_error
                .map(|error| error.to_string())
                .unwrap_or_else(|| "no candidates tried".into())
        )
    }

    pub fn factor_solve(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        rhs: &[f64],
        updated_values: Option<&[f64]>,
        updated_rhs: Option<&[f64]>,
    ) -> Result<NativeSpralNumericResult> {
        let values = matrix
            .values()
            .context("native SPRAL factorization requires numeric values")?;
        if rhs.len() != matrix.dimension() {
            bail!(
                "native SPRAL solve rhs has dimension {}, expected {}",
                rhs.len(),
                matrix.dimension()
            );
        }
        let col_ptrs64 = matrix
            .col_ptrs()
            .iter()
            .map(|&entry| i64::try_from(entry).unwrap_or(i64::MAX))
            .collect::<Vec<_>>();
        let row = matrix
            .row_indices()
            .iter()
            .map(|&entry| i32::try_from(entry).unwrap_or(i32::MAX))
            .collect::<Vec<_>>();

        let mut options = unsafe { std::mem::zeroed::<SpralSsidsOptions>() };
        unsafe { (self.default_options)(&mut options) };
        options.array_base = 0;
        options.use_gpu = false;
        options.ignore_numa = true;
        options.print_level = 0;

        let mut analyse_inform = SpralSsidsInform::default();
        let mut factor_inform = SpralSsidsInform::default();
        let mut solve_inform = SpralSsidsInform::default();
        let mut akeep: *mut c_void = ptr::null_mut();
        let mut fkeep: *mut c_void = ptr::null_mut();

        let cleanup = |akeep: &mut *mut c_void, fkeep: &mut *mut c_void| unsafe {
            let _ = (self.free)(akeep, fkeep);
        };

        unsafe {
            (self.analyse)(
                true,
                i32::try_from(matrix.dimension()).unwrap_or(i32::MAX),
                ptr::null_mut(),
                col_ptrs64.as_ptr(),
                row.as_ptr(),
                ptr::null(),
                &mut akeep,
                &options,
                &mut analyse_inform,
            );
        }
        if analyse_inform.flag < 0 {
            cleanup(&mut akeep, &mut fkeep);
            bail!(
                "native SPRAL analyse failed with flag {}",
                analyse_inform.flag
            );
        }

        let factor_started = Instant::now();
        unsafe {
            (self.factor)(
                false,
                ptr::null(),
                ptr::null(),
                values.as_ptr(),
                ptr::null_mut(),
                akeep,
                &mut fkeep,
                &options,
                &mut factor_inform,
            );
        }
        let factor_elapsed_ms = factor_started.elapsed().as_secs_f64() * 1000.0;
        if factor_inform.flag < 0 {
            cleanup(&mut akeep, &mut fkeep);
            bail!(
                "native SPRAL factor failed with flag {}",
                factor_inform.flag
            );
        }

        let mut solution = rhs.to_vec();
        let solve_started = Instant::now();
        unsafe {
            (self.solve1)(
                0,
                solution.as_mut_ptr(),
                akeep,
                fkeep,
                &options,
                &mut solve_inform,
            );
        }
        let solve_elapsed_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
        if solve_inform.flag < 0 {
            cleanup(&mut akeep, &mut fkeep);
            bail!("native SPRAL solve failed with flag {}", solve_inform.flag);
        }

        let mut refactor_solution = None;
        let mut refactor_elapsed_ms = None;
        if let Some(updated_values) = updated_values {
            let updated_rhs =
                updated_rhs.context("updated rhs must be provided with updated values")?;
            let mut refactor_inform = SpralSsidsInform::default();
            let factor_started = Instant::now();
            unsafe {
                (self.factor)(
                    false,
                    ptr::null(),
                    ptr::null(),
                    updated_values.as_ptr(),
                    ptr::null_mut(),
                    akeep,
                    &mut fkeep,
                    &options,
                    &mut refactor_inform,
                );
            }
            refactor_elapsed_ms = Some(factor_started.elapsed().as_secs_f64() * 1000.0);
            if refactor_inform.flag < 0 {
                cleanup(&mut akeep, &mut fkeep);
                bail!(
                    "native SPRAL refactor failed with flag {}",
                    refactor_inform.flag
                );
            }
            let mut updated_solution = updated_rhs.to_vec();
            let mut updated_solve_inform = SpralSsidsInform::default();
            unsafe {
                (self.solve1)(
                    0,
                    updated_solution.as_mut_ptr(),
                    akeep,
                    fkeep,
                    &options,
                    &mut updated_solve_inform,
                );
            }
            if updated_solve_inform.flag < 0 {
                cleanup(&mut akeep, &mut fkeep);
                bail!(
                    "native SPRAL refactor solve failed with flag {}",
                    updated_solve_inform.flag
                );
            }
            refactor_solution = Some(updated_solution);
            factor_inform = refactor_inform;
        }

        cleanup(&mut akeep, &mut fkeep);

        Ok(NativeSpralNumericResult {
            factor_elapsed_ms,
            solve_elapsed_ms,
            refactor_elapsed_ms,
            solution,
            refactor_solution,
            num_neg: factor_inform.num_neg,
            num_two: factor_inform.num_two,
            num_delay: factor_inform.num_delay,
            num_sup: factor_inform.num_sup,
            max_supernode: factor_inform.maxsupernode,
        })
    }
}

pub fn amd_permutation(matrix: &SymmetricPatternMatrix) -> Result<Permutation> {
    let n = matrix.dimension() as isize;
    let col_ptrs = matrix
        .col_ptrs()
        .iter()
        .map(|&index| index as isize)
        .collect::<Vec<_>>();
    let row_indices = matrix
        .row_indices()
        .iter()
        .map(|&index| index as isize)
        .collect::<Vec<_>>();
    let (perm, _, _) = amd::order(n, &col_ptrs, &row_indices, &Control::default())
        .map_err(|status| anyhow::anyhow!("amd::order failed with status {status:?}"))?;
    let permutation = perm
        .into_iter()
        .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    Ok(Permutation::new(permutation)?)
}
