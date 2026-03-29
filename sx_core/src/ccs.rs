use std::collections::HashSet;

use crate::Index;
use crate::error::{Result, SxError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CCS {
    nrow: Index,
    ncol: Index,
    col_ptrs: Vec<Index>,
    row_indices: Vec<Index>,
}

impl CCS {
    pub fn new(
        nrow: Index,
        ncol: Index,
        col_ptrs: Vec<Index>,
        row_indices: Vec<Index>,
    ) -> Result<Self> {
        if col_ptrs.len() != ncol + 1 {
            return Err(SxError::Ccs(format!(
                "expected {} column pointers, got {}",
                ncol + 1,
                col_ptrs.len()
            )));
        }
        if col_ptrs.first().copied().unwrap_or_default() != 0 {
            return Err(SxError::Ccs("column pointers must start at zero".into()));
        }
        if col_ptrs.last().copied().unwrap_or_default() != row_indices.len() {
            return Err(SxError::Ccs("final column pointer must equal nnz".into()));
        }
        if col_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(SxError::Ccs("column pointers must be monotone".into()));
        }
        for col in 0..ncol {
            let start = col_ptrs[col];
            let end = col_ptrs[col + 1];
            let slice = &row_indices[start..end];
            let mut prev = None;
            let mut seen = HashSet::new();
            for &row in slice {
                if row >= nrow {
                    return Err(SxError::Ccs(format!(
                        "row index {row} out of bounds for {nrow} rows"
                    )));
                }
                if let Some(prev_row) = prev
                    && prev_row >= row
                {
                    return Err(SxError::Ccs(
                        "row indices must be strictly increasing within a column".into(),
                    ));
                }
                if !seen.insert(row) {
                    return Err(SxError::Ccs("duplicate row index".into()));
                }
                prev = Some(row);
            }
        }
        Ok(Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        })
    }

    pub fn empty(nrow: Index, ncol: Index) -> Self {
        Self {
            nrow,
            ncol,
            col_ptrs: vec![0; ncol + 1],
            row_indices: Vec::new(),
        }
    }

    pub fn scalar() -> Self {
        Self {
            nrow: 1,
            ncol: 1,
            col_ptrs: vec![0, 1],
            row_indices: vec![0],
        }
    }

    pub fn dense(nrow: Index, ncol: Index) -> Result<Self> {
        let mut col_ptrs = Vec::with_capacity(ncol + 1);
        let Some(capacity) = nrow.checked_mul(ncol) else {
            return Err(SxError::Ccs("dense CCS size overflow".into()));
        };
        let mut row_indices = Vec::with_capacity(capacity);
        col_ptrs.push(0);
        for _ in 0..ncol {
            row_indices.extend(0..nrow);
            col_ptrs.push(row_indices.len());
        }
        Ok(Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        })
    }

    pub fn column_vector(nrow: Index) -> Result<Self> {
        Self::dense(nrow, 1)
    }

    pub fn lower_triangular(size: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(size + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col in 0..size {
            row_indices.extend(col..size);
            col_ptrs.push(row_indices.len());
        }
        Self {
            nrow: size,
            ncol: size,
            col_ptrs,
            row_indices,
        }
    }

    pub fn from_positions(nrow: Index, ncol: Index, positions: &[(Index, Index)]) -> Result<Self> {
        let mut cols = vec![Vec::new(); ncol];
        for &(row, col) in positions {
            if row >= nrow || col >= ncol {
                return Err(SxError::Ccs(format!(
                    "position ({row}, {col}) out of bounds for {nrow}x{ncol}"
                )));
            }
            cols[col].push(row);
        }
        let mut col_ptrs = Vec::with_capacity(ncol + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for mut col_rows in cols {
            col_rows.sort_unstable();
            col_rows.dedup();
            row_indices.extend(col_rows);
            col_ptrs.push(row_indices.len());
        }
        Self::new(nrow, ncol, col_ptrs, row_indices)
    }

    pub fn nrow(&self) -> Index {
        self.nrow
    }

    pub fn ncol(&self) -> Index {
        self.ncol
    }

    pub fn nnz(&self) -> Index {
        self.row_indices.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.nrow == 1 && self.ncol == 1 && self.nnz() == 1
    }

    pub fn col_ptrs(&self) -> &[Index] {
        &self.col_ptrs
    }

    pub fn row_indices(&self) -> &[Index] {
        &self.row_indices
    }

    pub fn positions(&self) -> Vec<(Index, Index)> {
        let mut positions = Vec::with_capacity(self.nnz());
        for col in 0..self.ncol {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];
            for idx in start..end {
                positions.push((self.row_indices[idx], col));
            }
        }
        positions
    }

    pub fn transpose(&self) -> Self {
        let mut counts = vec![0; self.nrow];
        for &row in &self.row_indices {
            counts[row] += 1;
        }

        let mut col_ptrs = vec![0; self.nrow + 1];
        for i in 0..self.nrow {
            col_ptrs[i + 1] = col_ptrs[i] + counts[i];
        }

        let mut row_indices = vec![0; self.nnz()];
        let mut offsets = col_ptrs[..self.nrow].to_vec();
        for col in 0..self.ncol {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];
            for idx in start..end {
                let row = self.row_indices[idx];
                let out = offsets[row];
                row_indices[out] = col;
                offsets[row] += 1;
            }
        }

        Self {
            nrow: self.ncol,
            ncol: self.nrow,
            col_ptrs,
            row_indices,
        }
    }

    pub fn row_adjacency(&self) -> Vec<Vec<Index>> {
        let mut rows = vec![Vec::new(); self.nrow];
        for col in 0..self.ncol {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];
            for idx in start..end {
                rows[self.row_indices[idx]].push(col);
            }
        }
        rows
    }

    pub fn nz_index(&self, row: Index, col: Index) -> Option<Index> {
        if row >= self.nrow || col >= self.ncol {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| start + offset)
    }
}
