use std::time::Instant;

use amd::Control;
use anyhow::{Context, Result, bail};
use libloading::{Library, Symbol};
use metis_ordering::{CsrGraph, Permutation};
use ssids_rs::{NativeSpral, SymmetricCscMatrix};

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

#[derive(Debug)]
pub struct NativeMetisReference {
    _library: Library,
    node_nd: MetisNodeNdFn,
}

#[derive(Debug)]
pub struct NativeSpralReference {
    inner: NativeSpral,
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
        let inner = NativeSpral::load().map_err(anyhow::Error::from)?;
        Ok(Self { inner })
    }

    pub fn factor_solve(
        &self,
        matrix: SymmetricCscMatrix<'_>,
        rhs: &[f64],
        updated_values: Option<&[f64]>,
        updated_rhs: Option<&[f64]>,
    ) -> Result<NativeSpralNumericResult> {
        matrix
            .values()
            .context("native SPRAL factorization requires numeric values")?;
        if rhs.len() != matrix.dimension() {
            bail!(
                "native SPRAL solve rhs has dimension {}, expected {}",
                rhs.len(),
                matrix.dimension()
            );
        }
        let mut session = self.inner.analyse(matrix)?;
        let factor_started = Instant::now();
        let mut factor_info = session.factorize(matrix)?;
        let factor_elapsed_ms = factor_started.elapsed().as_secs_f64() * 1000.0;
        let solve_started = Instant::now();
        let solution = session.solve(rhs)?;
        let solve_elapsed_ms = solve_started.elapsed().as_secs_f64() * 1000.0;

        let mut refactor_solution = None;
        let mut refactor_elapsed_ms = None;
        if let Some(updated_values) = updated_values {
            let updated_rhs =
                updated_rhs.context("updated rhs must be provided with updated values")?;
            let updated_matrix = SymmetricCscMatrix::new(
                matrix.dimension(),
                matrix.col_ptrs(),
                matrix.row_indices(),
                Some(updated_values),
            )?;
            let factor_started = Instant::now();
            factor_info = session.refactorize(updated_matrix)?;
            refactor_elapsed_ms = Some(factor_started.elapsed().as_secs_f64() * 1000.0);
            refactor_solution = Some(session.solve(updated_rhs)?);
        }

        Ok(NativeSpralNumericResult {
            factor_elapsed_ms,
            solve_elapsed_ms,
            refactor_elapsed_ms,
            solution,
            refactor_solution,
            num_neg: i32::try_from(factor_info.inertia.negative).unwrap_or(i32::MAX),
            num_two: i32::try_from(factor_info.two_by_two_pivots).unwrap_or(i32::MAX),
            num_delay: i32::try_from(factor_info.delayed_pivots).unwrap_or(i32::MAX),
            num_sup: i32::try_from(factor_info.supernode_count).unwrap_or(i32::MAX),
            max_supernode: i32::try_from(factor_info.max_supernode_width).unwrap_or(i32::MAX),
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
