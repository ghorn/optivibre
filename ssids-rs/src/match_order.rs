use metis_ordering::{OrderingError, metis_node_nd_order_from_lower_csc};
use std::time::Instant;

use crate::{SsidsError, SymmetricCscMatrix};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpralMatchingOrder {
    pub order: Vec<usize>,
    pub scaling: Vec<f64>,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct SpralCscTrace {
    pub dimension: usize,
    pub col_ptrs: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct SpralMatchingTrace {
    pub expanded_full: SpralCscTrace,
    pub compact_abs: SpralCscTrace,
    pub scale_logs: Vec<f64>,
    pub matching: Vec<Option<usize>>,
    pub split_matching: Vec<isize>,
    pub compressed_lower: SpralCscTrace,
    pub compressed_component_position: Vec<usize>,
    pub compressed_position_component: Vec<usize>,
    pub final_order: Vec<usize>,
    pub scaling: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct FullCscMatrix {
    dimension: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<f64>,
}

pub(crate) fn spral_matching_order(
    matrix: SymmetricCscMatrix<'_>,
) -> Result<SpralMatchingOrder, SsidsError> {
    let trace = matching_debug_enabled();
    let total_started = trace.then(Instant::now);
    if trace {
        matching_debug_log(format!(
            "[ssids_rs::matching] start dim={} nnz={}",
            matrix.dimension(),
            matrix.row_indices().len()
        ));
    }
    let started = trace.then(Instant::now);
    let expanded = expand_lower_to_full_spral(matrix)?;
    matching_debug_log_elapsed("expand_lower_to_full_spral", started);
    let started = trace.then(Instant::now);
    let compact_abs = remove_explicit_zeroes_and_abs(&expanded);
    matching_debug_log_elapsed("remove_explicit_zeroes_and_abs", started);
    let started = trace.then(Instant::now);
    let (scale_logs, matching) = mo_match(&compact_abs)?;
    matching_debug_log_elapsed("mo_match", started);
    let started = trace.then(Instant::now);
    let order = mo_split(&compact_abs, &matching)?;
    matching_debug_log_elapsed("mo_split", started);
    let started = trace.then(Instant::now);
    let scaling = scale_logs.into_iter().map(f64::exp).collect();
    matching_debug_log_elapsed("scaling_exp", started);
    if let Some(started) = total_started {
        matching_debug_log(format!(
            "[ssids_rs::matching] done elapsed={:.6}s",
            started.elapsed().as_secs_f64()
        ));
    }
    Ok(SpralMatchingOrder { order, scaling })
}

#[doc(hidden)]
pub fn spral_matching_trace(
    matrix: SymmetricCscMatrix<'_>,
) -> Result<SpralMatchingTrace, SsidsError> {
    let expanded = expand_lower_to_full_spral(matrix)?;
    let compact_abs = remove_explicit_zeroes_and_abs(&expanded);
    let (scale_logs, matching) = mo_match(&compact_abs)?;
    let split = mo_split_trace(&compact_abs, &matching)?;
    let scaling = scale_logs.iter().copied().map(f64::exp).collect();

    Ok(SpralMatchingTrace {
        expanded_full: SpralCscTrace::from_full(&expanded),
        compact_abs: SpralCscTrace::from_full(&compact_abs),
        scale_logs,
        matching,
        split_matching: split.split_matching,
        compressed_lower: SpralCscTrace::from_full(&split.compressed_lower),
        compressed_component_position: split.compressed_component_position,
        compressed_position_component: split.compressed_position_component,
        final_order: split.order,
        scaling,
    })
}

impl SpralCscTrace {
    fn from_full(matrix: &FullCscMatrix) -> Self {
        Self {
            dimension: matrix.dimension,
            col_ptrs: matrix.col_ptrs.clone(),
            row_indices: matrix.row_indices.clone(),
            values: matrix.values.clone(),
        }
    }
}

fn matching_debug_enabled() -> bool {
    std::env::var_os("SPRAL_SSIDS_DEBUG_MATCHING").is_some()
}

fn matching_debug_log(message: impl AsRef<str>) {
    if matching_debug_enabled() {
        eprintln!("{}", message.as_ref());
    }
}

fn matching_debug_log_elapsed(label: &str, started: Option<Instant>) {
    if let Some(started) = started {
        matching_debug_log(format!(
            "[ssids_rs::matching] {label} elapsed={:.6}s",
            started.elapsed().as_secs_f64()
        ));
    }
}

fn expand_lower_to_full_spral(matrix: SymmetricCscMatrix<'_>) -> Result<FullCscMatrix, SsidsError> {
    let dimension = matrix.dimension();
    let values = matrix.values().ok_or(SsidsError::MissingValues)?;
    let mut counts = vec![0usize; dimension];
    for col in 0..dimension {
        for source in matrix.col_ptrs()[col]..matrix.col_ptrs()[col + 1] {
            let row = matrix.row_indices()[source];
            counts[row] += 1;
            if row != col {
                counts[col] += 1;
            }
        }
    }

    let mut ends = vec![0usize; dimension];
    let mut total = 0usize;
    for (index, count) in counts.into_iter().enumerate() {
        total += count;
        ends[index] = total;
    }
    let mut next = ends;
    let mut row_indices = vec![0usize; total];
    let mut full_values = vec![0.0; total];
    for col in 0..dimension {
        for (source, &value) in values
            .iter()
            .enumerate()
            .take(matrix.col_ptrs()[col + 1])
            .skip(matrix.col_ptrs()[col])
        {
            let row = matrix.row_indices()[source];

            let target = row;
            next[target] -= 1;
            row_indices[next[target]] = col;
            full_values[next[target]] = value;

            if row != col {
                let target = col;
                next[target] -= 1;
                row_indices[next[target]] = row;
                full_values[next[target]] = value;
            }
        }
    }

    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    col_ptrs.extend(next);
    col_ptrs.push(total);
    Ok(FullCscMatrix {
        dimension,
        col_ptrs,
        row_indices,
        values: full_values,
    })
}

fn remove_explicit_zeroes_and_abs(matrix: &FullCscMatrix) -> FullCscMatrix {
    let mut col_ptrs = Vec::with_capacity(matrix.dimension + 1);
    let mut row_indices = Vec::with_capacity(matrix.row_indices.len());
    let mut values = Vec::with_capacity(matrix.values.len());
    col_ptrs.push(0);
    for col in 0..matrix.dimension {
        for entry in matrix.col_ptrs[col]..matrix.col_ptrs[col + 1] {
            let value = matrix.values[entry];
            if value == 0.0 {
                continue;
            }
            row_indices.push(matrix.row_indices[entry]);
            values.push(value.abs());
        }
        col_ptrs.push(row_indices.len());
    }
    FullCscMatrix {
        dimension: matrix.dimension,
        col_ptrs,
        row_indices,
        values,
    }
}

#[cfg(test)]
fn mo_scale(matrix: &FullCscMatrix) -> Result<(Vec<f64>, Vec<Option<usize>>), SsidsError> {
    let compact = remove_explicit_zeroes_and_abs(matrix);
    mo_match(&compact)
}

fn mo_match(matrix: &FullCscMatrix) -> Result<(Vec<f64>, Vec<Option<usize>>), SsidsError> {
    let n = matrix.dimension;
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let trace = matching_debug_enabled();
    let started = trace.then(Instant::now);
    let mut cmax = vec![0.0_f64; n];
    for (col, cmax_col) in cmax.iter_mut().enumerate() {
        let max_entry = matrix.values[matrix.col_ptrs[col]..matrix.col_ptrs[col + 1]]
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        *cmax_col = if max_entry != 0.0 {
            max_entry.ln()
        } else {
            0.0
        };
    }
    matching_debug_log_elapsed("mo_match_cmax", started);

    let started = trace.then(Instant::now);
    let costs = CostCsc::from_matrix(matrix, &cmax);
    matching_debug_log_elapsed("mo_match_costs", started);
    let started = trace.then(Instant::now);
    let hungarian = spral_hungarian_match(n, n, &costs);
    matching_debug_log_elapsed("mo_match_hungarian", started);
    let rank = hungarian.matched;

    if rank == n {
        let started = trace.then(Instant::now);
        let scaling = (0..n)
            .map(|index| {
                (hungarian.row_dual[index] + hungarian.col_dual[index] - cmax[index]) / 2.0
            })
            .collect();
        matching_debug_log_elapsed("mo_match_scaling_full_rank", started);
        return Ok((scaling, hungarian.assignment));
    }

    let started = trace.then(Instant::now);
    let mut old_to_new = vec![usize::MAX; n];
    let mut new_to_old = Vec::with_capacity(rank);
    for (row, matched_col) in hungarian.assignment.iter().enumerate() {
        if matched_col.is_some() {
            old_to_new[row] = new_to_old.len();
            new_to_old.push(row);
        }
    }
    let reduced_costs = costs.reduce_to_matched_submatrix(&old_to_new, rank);
    matching_debug_log_elapsed("mo_match_reduce_singular", started);
    let started = trace.then(Instant::now);
    let reduced_hungarian = spral_hungarian_match(rank, rank, &reduced_costs);
    matching_debug_log_elapsed("mo_match_hungarian_singular", started);

    let started = trace.then(Instant::now);
    let mut scaling = vec![-f64::MAX; n];
    let mut matching = vec![None; n];
    for reduced_row in 0..rank {
        let old_row = new_to_old[reduced_row];
        let reduced_col = reduced_hungarian.assignment[reduced_row].ok_or_else(|| {
            SsidsError::InvalidMatrix("reduced matching unexpectedly failed".into())
        })?;
        let old_col = new_to_old[reduced_col];
        scaling[old_row] = (reduced_hungarian.row_dual[reduced_row]
            + reduced_hungarian.col_dual[reduced_row]
            - cmax[old_row])
            / 2.0;
        matching[old_row] = Some(old_col);
    }
    matching_debug_log_elapsed("mo_match_scaling_singular", started);

    Ok((scaling, matching))
}

struct CostCsc {
    ptr: Vec<usize>,
    row: Vec<usize>,
    val: Vec<f64>,
}

impl CostCsc {
    fn from_matrix(matrix: &FullCscMatrix, cmax: &[f64]) -> Self {
        let n = matrix.dimension;
        let mut ptr = vec![0usize; n + 2];
        let capacity = matrix.row_indices.len() + 1;
        let mut row = Vec::with_capacity(capacity);
        let mut val = Vec::with_capacity(capacity);
        row.push(0);
        val.push(0.0);
        for col in 0..n {
            ptr[col + 1] = row.len();
            for entry in matrix.col_ptrs[col]..matrix.col_ptrs[col + 1] {
                let value = matrix.values[entry];
                debug_assert!(value != 0.0, "SPRAL mo_match expects compact nonzero costs");
                row.push(matrix.row_indices[entry] + 1);
                val.push(cmax[col] - value.ln());
            }
        }
        ptr[n + 1] = row.len();
        Self { ptr, row, val }
    }

    fn reduce_to_matched_submatrix(&self, old_to_new: &[usize], reduced_dimension: usize) -> Self {
        let mut ptr = vec![0usize; reduced_dimension + 2];
        let mut row = Vec::with_capacity(self.row.len());
        let mut val = Vec::with_capacity(self.val.len());
        row.push(0);
        val.push(0.0);
        for old_col in 0..old_to_new.len() {
            let new_col = old_to_new[old_col];
            if new_col == usize::MAX {
                continue;
            }
            ptr[new_col + 1] = row.len();
            for entry in self.ptr[old_col + 1]..self.ptr[old_col + 2] {
                let old_row = self.row[entry] - 1;
                let new_row = old_to_new[old_row];
                if new_row == usize::MAX {
                    continue;
                }
                row.push(new_row + 1);
                val.push(self.val[entry]);
            }
        }
        ptr[reduced_dimension + 1] = row.len();
        Self { ptr, row, val }
    }
}

struct SpralHungarianResult {
    assignment: Vec<Option<usize>>,
    matched: usize,
    row_dual: Vec<f64>,
    col_dual: Vec<f64>,
}

fn spral_hungarian_match(m: usize, n: usize, cost: &CostCsc) -> SpralHungarianResult {
    const RINF: f64 = f64::MAX;

    if m == 0 || n == 0 {
        return SpralHungarianResult {
            assignment: Vec::new(),
            matched: 0,
            row_dual: Vec::new(),
            col_dual: Vec::new(),
        };
    }

    let mut jperm = vec![0usize; n + 1];
    let mut out = vec![0usize; n + 1];
    let mut pr = vec![0isize; n + 1];
    let mut q = vec![0usize; m + 1];
    let mut longwork = vec![0usize; m + 1];
    let mut l = vec![0usize; m + 1];
    let mut d = vec![0.0f64; m.max(n) + 1];
    let mut dualu = vec![0.0f64; m + 1];
    let mut dualv = vec![0.0f64; n + 1];
    let mut iperm = vec![0usize; m + 1];
    let mut num = 0usize;

    spral_hungarian_init_heuristic(
        m,
        n,
        cost,
        &mut num,
        &mut iperm,
        &mut jperm,
        &mut dualu,
        &mut d,
        &mut longwork,
        &mut out,
    );

    if num != m.min(n) {
        d[1..=m].fill(RINF);
        l[1..=m].fill(0);
        for jord in 1..=n {
            if jperm[jord] != 0 {
                continue;
            }

            let mut dmin = RINF;
            let mut qlen = 0usize;
            let mut low = m + 1;
            let mut up = m + 1;
            let mut csp = RINF;
            let mut isp = 0usize;
            let mut jsp = 0usize;
            let mut j = jord;
            pr[j] = -1;

            for klong in cost.ptr[j]..cost.ptr[j + 1] {
                let i = cost.row[klong];
                let dnew = cost.val[klong] - dualu[i];
                if dnew >= csp {
                    continue;
                }
                if iperm[i] == 0 {
                    csp = dnew;
                    isp = klong;
                    jsp = j;
                } else {
                    if dnew < dmin {
                        dmin = dnew;
                    }
                    d[i] = dnew;
                    qlen += 1;
                    longwork[qlen] = klong;
                }
            }

            let q0 = qlen;
            qlen = 0;
            for &klong in longwork.iter().take(q0 + 1).skip(1) {
                let i = cost.row[klong];
                if csp <= d[i] {
                    d[i] = RINF;
                    continue;
                }
                if d[i] <= dmin {
                    low -= 1;
                    q[low] = i;
                    l[i] = low;
                } else {
                    qlen += 1;
                    l[i] = qlen;
                    heap_update(i, q.len() - 1, &mut q, &d, &mut l);
                }
                let jj = iperm[i];
                out[jj] = klong;
                pr[jj] = j as isize;
            }

            for _ in 1..=num {
                if low == up {
                    if qlen == 0 {
                        break;
                    }
                    let i = q[1];
                    if d[i] >= csp {
                        break;
                    }
                    dmin = d[i];
                    while qlen > 0 {
                        let i = q[1];
                        if d[i] > dmin {
                            break;
                        }
                        let i = heap_pop(&mut qlen, q.len() - 1, &mut q, &d, &mut l);
                        low -= 1;
                        q[low] = i;
                        l[i] = low;
                    }
                }

                let q0 = q[up - 1];
                let dq0 = d[q0];
                if dq0 >= csp {
                    break;
                }
                up -= 1;

                j = iperm[q0];
                let vj = dq0 - cost.val[jperm[j]] + dualu[q0];
                for klong in cost.ptr[j]..cost.ptr[j + 1] {
                    let i = cost.row[klong];
                    if l[i] >= up {
                        continue;
                    }
                    let dnew = vj + cost.val[klong] - dualu[i];
                    if dnew >= csp {
                        continue;
                    }
                    if iperm[i] == 0 {
                        csp = dnew;
                        isp = klong;
                        jsp = j;
                    } else {
                        let di = d[i];
                        if di <= dnew {
                            continue;
                        }
                        if l[i] >= low {
                            continue;
                        }
                        d[i] = dnew;
                        if dnew <= dmin {
                            let lpos = l[i];
                            if lpos != 0 {
                                heap_delete(lpos, &mut qlen, q.len() - 1, &mut q, &d, &mut l);
                            }
                            low -= 1;
                            q[low] = i;
                            l[i] = low;
                        } else {
                            if l[i] == 0 {
                                qlen += 1;
                                l[i] = qlen;
                            }
                            heap_update(i, q.len() - 1, &mut q, &d, &mut l);
                        }
                        let jj = iperm[i];
                        out[jj] = klong;
                        pr[jj] = j as isize;
                    }
                }
            }

            if csp != RINF {
                num += 1;
                let i = cost.row[isp];
                iperm[i] = jsp;
                jperm[jsp] = isp;
                j = jsp;
                for _ in 1..=num {
                    let jj = pr[j];
                    if jj == -1 {
                        break;
                    }
                    let jj = jj as usize;
                    let klong = out[j];
                    let i = cost.row[klong];
                    iperm[i] = jj;
                    jperm[jj] = klong;
                    j = jj;
                }
            }

            if up <= m {
                for &i in q.iter().take(m + 1).skip(up) {
                    dualu[i] += d[i] - csp;
                }
            }
            if low <= m {
                for &i in q.iter().take(m + 1).skip(low) {
                    d[i] = RINF;
                    l[i] = 0;
                }
            }
            for &i in q.iter().take(qlen + 1).skip(1) {
                d[i] = RINF;
                l[i] = 0;
            }
        }
    }

    for (j, dualv_j) in dualv.iter_mut().enumerate().take(n + 1).skip(1) {
        let klong = jperm[j];
        *dualv_j = if klong != 0 {
            cost.val[klong] - dualu[cost.row[klong]]
        } else {
            0.0
        };
    }
    for i in 1..=m {
        if iperm[i] == 0 {
            dualu[i] = 0.0;
        }
    }

    SpralHungarianResult {
        assignment: iperm[1..=m]
            .iter()
            .map(|&col| (col != 0).then(|| col - 1))
            .collect(),
        matched: num,
        row_dual: dualu[1..=m].to_vec(),
        col_dual: dualv[1..=n].to_vec(),
    }
}

#[allow(clippy::too_many_arguments)]
fn spral_hungarian_init_heuristic(
    m: usize,
    n: usize,
    cost: &CostCsc,
    num: &mut usize,
    iperm: &mut [usize],
    jperm: &mut [usize],
    dualu: &mut [f64],
    d: &mut [f64],
    l: &mut [usize],
    search_from: &mut [usize],
) {
    const RINF: f64 = f64::MAX;

    dualu[1..=m].fill(RINF);
    l[1..=m].fill(0);
    for j in 1..=n {
        for k in cost.ptr[j]..cost.ptr[j + 1] {
            let i = cost.row[k];
            if cost.val[k] > dualu[i] {
                continue;
            }
            dualu[i] = cost.val[k];
            iperm[i] = j;
            l[i] = k;
        }
    }

    for i in 1..=m {
        let j = iperm[i];
        if j == 0 {
            continue;
        }
        iperm[i] = 0;
        if jperm[j] != 0 {
            continue;
        }
        if cost.ptr[j + 1] - cost.ptr[j] > m / 10 && m > 50 {
            continue;
        }
        *num += 1;
        iperm[i] = j;
        jperm[j] = l[i];
    }

    if *num == m.min(n) {
        return;
    }

    d[1..=n].fill(0.0);
    search_from[1..=n].copy_from_slice(&cost.ptr[1..=n]);
    'improve_assign: for j in 1..=n {
        if jperm[j] != 0 || cost.ptr[j] > cost.ptr[j + 1] - 1 {
            continue;
        }
        let mut i0 = cost.row[cost.ptr[j]];
        let mut vj = cost.val[cost.ptr[j]] - dualu[i0];
        let mut k0 = cost.ptr[j];
        for k in cost.ptr[j] + 1..cost.ptr[j + 1] {
            let i = cost.row[k];
            let di = cost.val[k] - dualu[i];
            if di > vj {
                continue;
            }
            if di == vj && di != RINF && (iperm[i] != 0 || iperm[i0] == 0) {
                continue;
            }
            vj = di;
            i0 = i;
            k0 = k;
        }
        d[j] = vj;
        if iperm[i0] == 0 {
            *num += 1;
            jperm[j] = k0;
            iperm[i0] = j;
            search_from[j] = k0 + 1;
            continue;
        }
        for k in k0..cost.ptr[j + 1] {
            let i = cost.row[k];
            if cost.val[k] - dualu[i] > vj {
                continue;
            }
            let jj = iperm[i];
            for kk in search_from[jj]..cost.ptr[jj + 1] {
                let ii = cost.row[kk];
                if iperm[ii] > 0 {
                    continue;
                }
                if cost.val[kk] - dualu[ii] <= d[jj] {
                    jperm[jj] = kk;
                    iperm[ii] = jj;
                    search_from[jj] = kk + 1;
                    *num += 1;
                    jperm[j] = k;
                    iperm[i] = j;
                    search_from[j] = k + 1;
                    continue 'improve_assign;
                }
            }
            search_from[jj] = cost.ptr[jj + 1];
        }
    }
}

fn heap_update(idx: usize, _n: usize, q: &mut [usize], val: &[f64], l: &mut [usize]) {
    let mut pos = l[idx];
    if pos <= 1 {
        q[pos] = idx;
        return;
    }

    let v = val[idx];
    while pos > 1 {
        let parent_pos = pos / 2;
        let parent_idx = q[parent_pos];
        if v >= val[parent_idx] {
            break;
        }
        q[pos] = parent_idx;
        l[parent_idx] = pos;
        pos = parent_pos;
    }
    q[pos] = idx;
    l[idx] = pos;
}

fn heap_pop(qlen: &mut usize, n: usize, q: &mut [usize], val: &[f64], l: &mut [usize]) -> usize {
    let popped = q[1];
    heap_delete(1, qlen, n, q, val, l);
    popped
}

fn heap_delete(
    pos0: usize,
    qlen: &mut usize,
    _n: usize,
    q: &mut [usize],
    d: &[f64],
    l: &mut [usize],
) {
    if *qlen == pos0 {
        *qlen -= 1;
        return;
    }

    let idx = q[*qlen];
    let v = d[idx];
    *qlen -= 1;
    let mut pos = pos0;

    if pos > 1 {
        loop {
            let parent = pos / 2;
            let qk = q[parent];
            if v >= d[qk] {
                break;
            }
            q[pos] = qk;
            l[qk] = pos;
            pos = parent;
            if pos <= 1 {
                break;
            }
        }
    }
    q[pos] = idx;
    l[idx] = pos;
    if pos != pos0 {
        return;
    }

    loop {
        let mut child = 2 * pos;
        if child > *qlen {
            break;
        }
        let mut dk = d[q[child]];
        if child < *qlen {
            let dr = d[q[child + 1]];
            if dk > dr {
                child += 1;
                dk = dr;
            }
        }
        if v <= dk {
            break;
        }
        let qk = q[child];
        q[pos] = qk;
        l[qk] = pos;
        pos = child;
    }
    q[pos] = idx;
    l[idx] = pos;
}

fn mo_split(matrix: &FullCscMatrix, matching: &[Option<usize>]) -> Result<Vec<usize>, SsidsError> {
    Ok(mo_split_trace(matrix, matching)?.order)
}

struct MoSplitTrace {
    order: Vec<usize>,
    split_matching: Vec<isize>,
    compressed_lower: FullCscMatrix,
    compressed_component_position: Vec<usize>,
    compressed_position_component: Vec<usize>,
}

fn mo_split_trace(
    matrix: &FullCscMatrix,
    matching: &[Option<usize>],
) -> Result<MoSplitTrace, SsidsError> {
    let n = matrix.dimension;
    if matching.len() != n {
        return Err(SsidsError::InvalidMatrix(
            "matching length does not match matrix dimension".into(),
        ));
    }
    const UNMATCHED: isize = -2;
    const SINGLETON: isize = -1;
    const UNSEEN: isize = isize::MIN;

    let mut cperm = matching
        .iter()
        .map(|entry| entry.map(|value| value as isize).unwrap_or(UNMATCHED))
        .collect::<Vec<_>>();
    let mut iwork = vec![UNSEEN; n];
    for i in 0..n {
        if iwork[i] != UNSEEN {
            continue;
        }
        let mut j = i;
        loop {
            let matched = cperm[j];
            if matched == UNMATCHED {
                iwork[j] = UNMATCHED;
                break;
            }
            if matched == i as isize {
                iwork[j] = SINGLETON;
                break;
            }
            let jj = matched as usize;
            iwork[j] = jj as isize;
            iwork[jj] = j as isize;
            let next = cperm[jj];
            if next < 0 {
                break;
            }
            j = next as usize;
            if j == i {
                break;
            }
        }
    }
    cperm.clone_from(&iwork);

    let mut old_to_new = vec![usize::MAX; n];
    let mut new_to_old = Vec::new();
    for i in 0..n {
        let j = cperm[i];
        if j >= 0 && (j as usize) < i {
            continue;
        }
        let component = new_to_old.len();
        old_to_new[i] = component;
        new_to_old.push(i);
        if j >= 0 {
            old_to_new[j as usize] = component;
        }
    }
    let ncomp = new_to_old.len();

    let compressed = compressed_lower_pattern(matrix, &cperm, &old_to_new, ncomp);
    let permutation = metis_node_nd_order_from_lower_csc(
        compressed.dimension,
        &compressed.col_ptrs,
        &compressed.row_indices,
    )
    .map_err(map_ordering_error)?
    .permutation;
    let component_order = permutation.inverse();
    let mut position_to_component = vec![0usize; ncomp];
    for (component, &position) in component_order.iter().enumerate() {
        position_to_component[position] = component;
    }

    let mut order = vec![usize::MAX; n];
    let mut next_position = 0usize;
    for &component in &position_to_component {
        let first = new_to_old[component];
        order[first] = next_position;
        next_position += 1;
        if cperm[first] >= 0 {
            let second = cperm[first] as usize;
            order[second] = next_position;
            next_position += 1;
        }
    }
    if order.contains(&usize::MAX) {
        return Err(SsidsError::InvalidMatrix(
            "matching split did not assign every variable".into(),
        ));
    }

    Ok(MoSplitTrace {
        order,
        split_matching: cperm,
        compressed_lower: compressed,
        compressed_component_position: component_order.to_vec(),
        compressed_position_component: position_to_component,
    })
}

fn compressed_lower_pattern(
    matrix: &FullCscMatrix,
    cperm: &[isize],
    old_to_new: &[usize],
    ncomp: usize,
) -> FullCscMatrix {
    let mut raw_cols = vec![Vec::<(usize, f64)>::new(); ncomp];
    let mut seen = vec![usize::MAX; ncomp];
    for i in 0..matrix.dimension {
        let j = cperm[i];
        if j >= 0 && (j as usize) < i {
            continue;
        }
        let col = old_to_new[i];
        let marker = i;
        for entry in matrix.col_ptrs[i]..matrix.col_ptrs[i + 1] {
            let row = old_to_new[matrix.row_indices[entry]];
            if seen[row] == marker {
                continue;
            }
            raw_cols[col].push((row, 1.0));
            seen[row] = marker;
        }
        if j >= 0 {
            let paired = j as usize;
            for entry in matrix.col_ptrs[paired]..matrix.col_ptrs[paired + 1] {
                let row = old_to_new[matrix.row_indices[entry]];
                if seen[row] == marker {
                    continue;
                }
                raw_cols[col].push((row, 1.0));
                seen[row] = marker;
            }
        }
    }

    let lower_cols = raw_cols
        .into_iter()
        .enumerate()
        .map(|(col, rows)| {
            rows.into_iter()
                .filter(|&(row, _)| row >= col)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    build_full_csc_from_columns(ncomp, lower_cols)
}

fn build_full_csc_from_columns(dimension: usize, cols: Vec<Vec<(usize, f64)>>) -> FullCscMatrix {
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let nnz = cols.iter().map(Vec::len).sum();
    let mut row_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    col_ptrs.push(0);
    for col in cols {
        for (row, value) in col {
            row_indices.push(row);
            values.push(value);
        }
        col_ptrs.push(row_indices.len());
    }
    FullCscMatrix {
        dimension,
        col_ptrs,
        row_indices,
        values,
    }
}

fn map_ordering_error(error: OrderingError) -> SsidsError {
    SsidsError::Ordering(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix<'a>(
        dimension: usize,
        col_ptrs: &'a [usize],
        row_indices: &'a [usize],
        values: &'a [f64],
    ) -> SymmetricCscMatrix<'a> {
        SymmetricCscMatrix::new(dimension, col_ptrs, row_indices, Some(values)).unwrap()
    }

    #[test]
    fn expand_lower_to_full_matches_spral_order() {
        let col_ptrs = [0, 3, 5, 6];
        let row_indices = [0, 1, 2, 1, 2, 2];
        let values = [1.0, 2.0, 3.0, 4.0, 0.0, 6.0];
        let expanded =
            expand_lower_to_full_spral(matrix(3, &col_ptrs, &row_indices, &values)).unwrap();
        assert_eq!(expanded.col_ptrs, vec![0, 3, 6, 9]);
        assert_eq!(expanded.row_indices, vec![2, 1, 0, 2, 1, 0, 2, 1, 0]);
        assert_eq!(
            expanded.values,
            vec![3.0, 2.0, 1.0, 0.0, 4.0, 2.0, 6.0, 0.0, 3.0]
        );
    }

    #[test]
    fn zero_removal_and_abs_preserves_column_order() {
        let full = FullCscMatrix {
            dimension: 2,
            col_ptrs: vec![0, 3, 5],
            row_indices: vec![0, 1, 1, 0, 1],
            values: vec![0.0, -2.0, -0.0, 3.0, -4.0],
        };
        let compact = remove_explicit_zeroes_and_abs(&full);
        assert_eq!(compact.col_ptrs, vec![0, 1, 3]);
        assert_eq!(compact.row_indices, vec![1, 0, 1]);
        assert_eq!(compact.values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mo_scale_diagonal_full_rank_uses_symmetric_inverse_sqrt_scale() {
        let col_ptrs = [0, 1, 2];
        let row_indices = [0, 1];
        let values = [4.0, 9.0];
        let expanded =
            expand_lower_to_full_spral(matrix(2, &col_ptrs, &row_indices, &values)).unwrap();
        let (scale_logs, matching) = mo_scale(&expanded).unwrap();
        let scaling = scale_logs
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        assert_eq!(matching, vec![Some(0), Some(1)]);
        assert!((scaling[0] - 0.5).abs() < 1e-12, "{scaling:?}");
        assert!((scaling[1] - (1.0 / 3.0)).abs() < 1e-12, "{scaling:?}");
    }

    #[test]
    fn mo_scale_singular_unmatched_uses_spral_huge_sentinel() {
        let col_ptrs = [0, 1, 2, 2];
        let row_indices = [0, 1];
        let values = [4.0, 9.0];
        let expanded =
            expand_lower_to_full_spral(matrix(3, &col_ptrs, &row_indices, &values)).unwrap();
        let (scale_logs, matching) = mo_scale(&expanded).unwrap();
        let scaling = scale_logs
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        assert_eq!(matching, vec![Some(0), Some(1), None]);
        assert_eq!(scale_logs[2].to_bits(), (-f64::MAX).to_bits());
        assert_eq!(scaling, vec![0.5, 1.0 / 3.0, 0.0]);
    }

    #[test]
    fn mo_split_keeps_singletons_and_pairs_adjacent() {
        let full = FullCscMatrix {
            dimension: 4,
            col_ptrs: vec![0, 2, 4, 6, 8],
            row_indices: vec![0, 1, 0, 1, 2, 3, 2, 3],
            values: vec![1.0; 8],
        };
        let order = mo_split(&full, &[Some(1), Some(0), Some(2), None]).unwrap();
        assert_eq!(order.len(), 4);
        assert!(order[0].abs_diff(order[1]) == 1, "{order:?}");
        assert!(order[2] < 4);
        assert!(order[3] < 4);
    }

    #[test]
    fn compressed_lower_pattern_preserves_spral_row_order() {
        let full = FullCscMatrix {
            dimension: 3,
            col_ptrs: vec![0, 3, 3, 3],
            row_indices: vec![2, 0, 1],
            values: vec![1.0; 3],
        };
        let compressed = compressed_lower_pattern(&full, &[-1, -1, -1], &[0, 1, 2], 3);
        assert_eq!(compressed.col_ptrs, vec![0, 3, 3, 3]);
        assert_eq!(compressed.row_indices, vec![2, 0, 1]);
    }

    #[test]
    fn spral_matching_order_returns_saved_original_scaling() {
        let col_ptrs = [0, 1, 2];
        let row_indices = [0, 1];
        let values = [4.0, 9.0];
        let result = spral_matching_order(matrix(2, &col_ptrs, &row_indices, &values)).unwrap();
        assert_eq!(result.order, vec![0, 1]);
        assert!((result.scaling[0] - 0.5).abs() < 1e-12, "{result:?}");
        assert!(
            (result.scaling[1] - (1.0 / 3.0)).abs() < 1e-12,
            "{result:?}"
        );
    }
}
