use std::collections::HashSet;

use crate::error::{Result, SxError};
use crate::{Index, SignedIndex, checked_len_product};

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

    pub fn diag(size: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(size + 1);
        let mut row_indices = Vec::with_capacity(size);
        col_ptrs.push(0);
        for idx in 0..size {
            row_indices.push(idx);
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

    pub fn triplet(
        nrow: Index,
        ncol: Index,
        row_indices: &[Index],
        col_indices: &[Index],
    ) -> Result<Self> {
        if row_indices.len() != col_indices.len() {
            return Err(SxError::Ccs(format!(
                "triplet row/col index length mismatch: {} rows, {} cols",
                row_indices.len(),
                col_indices.len()
            )));
        }
        let positions = row_indices
            .iter()
            .copied()
            .zip(col_indices.iter().copied())
            .collect::<Vec<_>>();
        Self::from_positions(nrow, ncol, &positions)
    }

    pub fn rowcol(rows: &[Index], cols: &[Index], nrow: Index, ncol: Index) -> Result<Self> {
        let mut positions = Vec::with_capacity(rows.len().saturating_mul(cols.len()));
        for &col in cols {
            for &row in rows {
                positions.push((row, col));
            }
        }
        Self::from_positions(nrow, ncol, &positions)
    }

    pub fn nonzeros(nrow: Index, ncol: Index, linear_indices: &[Index]) -> Result<Self> {
        let Some(numel) = checked_len_product(nrow, ncol) else {
            return Err(SxError::Ccs("CCS nonzero constructor size overflow".into()));
        };
        let mut positions = Vec::with_capacity(linear_indices.len());
        for &linear in linear_indices {
            if linear >= numel {
                return Err(SxError::Ccs(format!(
                    "linear index {linear} out of bounds for {nrow}x{ncol}"
                )));
            }
            let row = linear % nrow;
            let col = linear / nrow;
            positions.push((row, col));
        }
        Self::from_positions(nrow, ncol, &positions)
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

    pub fn get_ccs(&self) -> (Vec<Index>, Vec<Index>) {
        (self.col_ptrs.clone(), self.row_indices.clone())
    }

    pub fn find(&self) -> Vec<Index> {
        self.positions()
            .into_iter()
            .map(|(row, col)| row + col * self.nrow)
            .collect()
    }

    pub fn serialize(&self) -> String {
        let col_ptrs = self
            .col_ptrs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let row_indices = self
            .row_indices
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "ccs-v1|{}|{}|{}|{}",
            self.nrow, self.ncol, col_ptrs, row_indices
        )
    }

    pub fn deserialize(serialized: &str) -> Result<Self> {
        let mut parts = serialized.split('|');
        let Some(version) = parts.next() else {
            return Err(SxError::Ccs("missing CCS serialization version".into()));
        };
        if version != "ccs-v1" {
            return Err(SxError::Ccs(format!(
                "unsupported CCS serialization version `{version}`"
            )));
        }
        let Some(nrow) = parts.next() else {
            return Err(SxError::Ccs("missing serialized row count".into()));
        };
        let Some(ncol) = parts.next() else {
            return Err(SxError::Ccs("missing serialized column count".into()));
        };
        let Some(col_ptrs) = parts.next() else {
            return Err(SxError::Ccs("missing serialized column pointers".into()));
        };
        let Some(row_indices) = parts.next() else {
            return Err(SxError::Ccs("missing serialized row indices".into()));
        };
        if parts.next().is_some() {
            return Err(SxError::Ccs("trailing CCS serialization fields".into()));
        }

        let nrow = nrow
            .parse::<Index>()
            .map_err(|_| SxError::Ccs(format!("invalid serialized row count `{nrow}`")))?;
        let ncol = ncol
            .parse::<Index>()
            .map_err(|_| SxError::Ccs(format!("invalid serialized column count `{ncol}`")))?;

        let parse_list = |label: &str, payload: &str| -> Result<Vec<Index>> {
            if payload.is_empty() {
                return Ok(Vec::new());
            }
            payload
                .split(',')
                .map(|item| {
                    item.parse::<Index>().map_err(|_| {
                        SxError::Ccs(format!("invalid serialized {label} entry `{item}`"))
                    })
                })
                .collect()
        };

        Self::new(
            nrow,
            ncol,
            parse_list("column pointer", col_ptrs)?,
            parse_list("row index", row_indices)?,
        )
    }

    pub fn get_diag(&self) -> Result<(Self, Vec<Index>)> {
        if self.ncol == 1 {
            let diag_positions = self
                .positions()
                .into_iter()
                .map(|(row, _)| (row, row))
                .collect::<Vec<_>>();
            return Ok((
                Self::from_positions(self.nrow, self.nrow, &diag_positions)?,
                (0..self.nnz()).collect(),
            ));
        }
        if self.nrow == 1 {
            let diag_positions = self
                .positions()
                .into_iter()
                .map(|(_, col)| (col, col))
                .collect::<Vec<_>>();
            return Ok((
                Self::from_positions(self.ncol, self.ncol, &diag_positions)?,
                (0..self.nnz()).collect(),
            ));
        }

        let mut diag_rows = Vec::new();
        let mut mapping = Vec::new();
        for (idx, (row, col)) in self.positions().into_iter().enumerate() {
            if row == col {
                diag_rows.push((row, 0));
                mapping.push(idx);
            }
        }
        let diag = Self::from_positions(self.nrow.min(self.ncol), 1, &diag_rows)?;
        Ok((diag, mapping))
    }

    pub fn get_lower(&self) -> Result<Self> {
        let positions = self
            .positions()
            .into_iter()
            .filter(|&(row, col)| row >= col)
            .collect::<Vec<_>>();
        Self::from_positions(self.nrow, self.ncol, &positions)
    }

    pub fn pattern_inverse(&self) -> Result<Self> {
        let positions = (0..self.ncol)
            .flat_map(|col| (0..self.nrow).map(move |row| (row, col)))
            .filter(|&(row, col)| self.nz_index(row, col).is_none())
            .collect::<Vec<_>>();
        Self::from_positions(self.nrow, self.ncol, &positions)
    }

    pub fn kron(&self, other: &Self) -> Result<Self> {
        let Some(nrow) = self.nrow.checked_mul(other.nrow) else {
            return Err(SxError::Ccs("kron row count overflow".into()));
        };
        let Some(ncol) = self.ncol.checked_mul(other.ncol) else {
            return Err(SxError::Ccs("kron column count overflow".into()));
        };
        let mut positions = Vec::with_capacity(self.nnz().saturating_mul(other.nnz()));
        for (lhs_row, lhs_col) in self.positions() {
            for (rhs_row, rhs_col) in other.positions() {
                positions.push((
                    lhs_row * other.nrow + rhs_row,
                    lhs_col * other.ncol + rhs_col,
                ));
            }
        }
        Self::from_positions(nrow, ncol, &positions)
    }

    pub fn get_nz(&self, linear_indices: &[Index]) -> Result<Vec<SignedIndex>> {
        let Some(numel) = checked_len_product(self.nrow, self.ncol) else {
            return Err(SxError::Ccs("CCS get_nz size overflow".into()));
        };
        linear_indices
            .iter()
            .copied()
            .map(|linear| {
                if linear >= numel {
                    return Err(SxError::Ccs(format!(
                        "linear index {linear} out of bounds for {}x{}",
                        self.nrow, self.ncol
                    )));
                }
                let row = linear % self.nrow;
                let col = linear / self.nrow;
                Ok(match self.nz_index(row, col) {
                    Some(idx) => SignedIndex::try_from(idx).map_err(|_| {
                        SxError::Ccs("nonzero index does not fit into isize".into())
                    })?,
                    None => -1,
                })
            })
            .collect()
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

    pub fn get_crs(&self) -> (Vec<Index>, Vec<Index>) {
        self.transpose().get_ccs()
    }

    pub fn reshape(&self, nrow: Index, ncol: Index) -> Result<Self> {
        let Some(current_numel) = checked_len_product(self.nrow, self.ncol) else {
            return Err(SxError::Ccs("current CCS shape overflow".into()));
        };
        let Some(target_numel) = checked_len_product(nrow, ncol) else {
            return Err(SxError::Ccs("target CCS shape overflow".into()));
        };
        if current_numel != target_numel {
            return Err(SxError::Ccs(format!(
                "reshape requires identical element count, got {} and {}",
                current_numel, target_numel
            )));
        }

        let positions = self
            .positions()
            .into_iter()
            .map(|(row, col)| {
                let linear = row + col * self.nrow;
                (linear % nrow, linear / nrow)
            })
            .collect::<Vec<_>>();
        Self::from_positions(nrow, ncol, &positions)
    }

    pub fn enlarge(
        &self,
        nrow: Index,
        ncol: Index,
        row_map: &[Index],
        col_map: &[Index],
    ) -> Result<Self> {
        if row_map.len() != self.nrow {
            return Err(SxError::Ccs(format!(
                "row map length {} does not match current row count {}",
                row_map.len(),
                self.nrow
            )));
        }
        if col_map.len() != self.ncol {
            return Err(SxError::Ccs(format!(
                "column map length {} does not match current column count {}",
                col_map.len(),
                self.ncol
            )));
        }
        if row_map.iter().any(|&row| row >= nrow) {
            return Err(SxError::Ccs(
                "row map contains out-of-bounds target row".into(),
            ));
        }
        if col_map.iter().any(|&col| col >= ncol) {
            return Err(SxError::Ccs(
                "column map contains out-of-bounds target column".into(),
            ));
        }

        let positions = self
            .positions()
            .into_iter()
            .map(|(row, col)| (row_map[row], col_map[col]))
            .collect::<Vec<_>>();
        Self::from_positions(nrow, ncol, &positions)
    }

    pub fn unite(&self, other: &Self) -> Result<Self> {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            return Err(SxError::Ccs(format!(
                "cannot unite CCS patterns with shapes {}x{} and {}x{}",
                self.nrow, self.ncol, other.nrow, other.ncol
            )));
        }
        let mut positions = self.positions();
        positions.extend(other.positions());
        Self::from_positions(self.nrow, self.ncol, &positions)
    }

    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            return Err(SxError::Ccs(format!(
                "cannot intersect CCS patterns with shapes {}x{} and {}x{}",
                self.nrow, self.ncol, other.nrow, other.ncol
            )));
        }
        let positions = self
            .positions()
            .into_iter()
            .filter(|&(row, col)| other.nz_index(row, col).is_some())
            .collect::<Vec<_>>();
        Self::from_positions(self.nrow, self.ncol, &positions)
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            return false;
        }
        self.positions()
            .into_iter()
            .all(|(row, col)| other.nz_index(row, col).is_some())
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
