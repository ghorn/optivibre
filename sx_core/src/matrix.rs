use std::collections::HashMap;

use crate::Index;
use crate::ccs::CCS;
use crate::error::{Result, SxError};
use crate::sx::{
    JacobianStructure, SX, forward_directional, forward_directional_basis_batch,
    greedy_color_disjoint, jacobian_structure, reverse_directional, reverse_directional_batch,
    scalar_hessian_basis_batch,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SXMatrix {
    ccs: CCS,
    nonzeros: Vec<SX>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HessianStrategy {
    #[default]
    LowerTriangleByColumn,
    LowerTriangleSelectedOutputs,
    LowerTriangleColored,
}

impl HessianStrategy {
    pub const ALL: [Self; 3] = [
        Self::LowerTriangleByColumn,
        Self::LowerTriangleSelectedOutputs,
        Self::LowerTriangleColored,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Self::LowerTriangleByColumn => "by_column",
            Self::LowerTriangleSelectedOutputs => "selected_outputs",
            Self::LowerTriangleColored => "colored",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LowerTriangleByColumn => "1. By column",
            Self::LowerTriangleSelectedOutputs => "2. Selected outputs",
            Self::LowerTriangleColored => "3. Colored",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::LowerTriangleByColumn => {
                "Reverse gradient, then one forward sweep per Hessian column."
            }
            Self::LowerTriangleSelectedOutputs => {
                "One forward sweep per column, but only over the needed gradient suffix."
            }
            Self::LowerTriangleColored => {
                "Compressed lower-triangle sweeps using structural coloring."
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HessianOptions {
    pub strategy: HessianStrategy,
}

impl HessianOptions {
    pub fn with_strategy(strategy: HessianStrategy) -> Self {
        Self { strategy }
    }
}

const JACOBIAN_BATCH_WIDTH: usize = 8;

impl SXMatrix {
    fn build_lower_triangle(size: Index, columns: Vec<Vec<(Index, SX)>>) -> Result<SXMatrix> {
        let mut positions = Vec::new();
        let mut values = Vec::new();
        for (col, entries) in columns.into_iter().enumerate() {
            for (row, value) in entries {
                if !value.is_zero() {
                    positions.push((row, col));
                    values.push(value);
                }
            }
        }
        let ccs = CCS::from_positions(size, size, &positions)?;
        SXMatrix::new(ccs, values)
    }

    fn hessian_row_supports(
        &self,
        wrt: &SXMatrix,
    ) -> Result<(SX, Vec<Vec<Index>>, Vec<Vec<Index>>)> {
        let expr = self.scalar_expr()?;
        let grad = self.gradient(wrt)?;
        let n = wrt.nnz();
        let full_ccs = grad.jacobian_ccs(wrt)?;
        let mut full_row_sets = vec![Vec::new(); n];
        let mut lower_row_sets = vec![Vec::new(); n];
        for (row, col) in full_ccs.positions() {
            full_row_sets[col].push(row);
            if row >= col {
                lower_row_sets[col].push(row);
            }
        }
        Ok((expr, full_row_sets, lower_row_sets))
    }

    fn hessian_lower_triangle_program(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let (expr, full_row_sets, lower_row_sets) = self.hessian_row_supports(wrt)?;
        let n = wrt.nnz();
        let active_columns = lower_row_sets
            .iter()
            .enumerate()
            .filter_map(|(col, rows)| (!rows.is_empty()).then_some(col))
            .collect::<Vec<_>>();
        // Color against the full column support, not just the stored lower triangle.
        // Otherwise an upper-triangular dependency from another active column can alias
        // into the compressed Hessian-vector product for this column.
        let active_full_row_sets = active_columns
            .iter()
            .map(|&col| full_row_sets[col].clone())
            .collect::<Vec<_>>();
        let color_columns = greedy_color_disjoint(&active_full_row_sets)
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|active_idx| active_columns[active_idx])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut columns = vec![Vec::new(); n];
        for group_batch in color_columns.chunks(JACOBIAN_BATCH_WIDTH) {
            let sensitivities = scalar_hessian_basis_batch(expr, &wrt.nonzeros, group_batch)?;
            for (direction, group) in group_batch.iter().enumerate() {
                let sens = &sensitivities[direction];
                for &col in group {
                    for &row in &lower_row_sets[col] {
                        columns[col].push((row, sens[row]));
                    }
                }
            }
        }
        Self::build_lower_triangle(n, columns)
    }

    fn hessian_lower_triangle_by_column(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let (expr, _, lower_row_sets) = self.hessian_row_supports(wrt)?;
        let n = wrt.nnz();

        let mut columns = vec![Vec::new(); n];
        let active_columns = lower_row_sets
            .iter()
            .enumerate()
            .filter_map(|(col, rows)| (!rows.is_empty()).then_some(col))
            .collect::<Vec<_>>();
        for column_batch in active_columns.chunks(JACOBIAN_BATCH_WIDTH) {
            let active_groups = column_batch
                .iter()
                .map(|&col| vec![col])
                .collect::<Vec<_>>();
            let sensitivities = scalar_hessian_basis_batch(expr, &wrt.nonzeros, &active_groups)?;
            for (direction, &col) in column_batch.iter().enumerate() {
                let sens = &sensitivities[direction];
                for &row in &lower_row_sets[col] {
                    columns[col].push((row, sens[row]));
                }
            }
        }
        Self::build_lower_triangle(n, columns)
    }

    fn hessian_lower_triangle_selected_outputs(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        self.hessian_lower_triangle_program(wrt)
    }

    fn hessian_lower_triangle_colored(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        self.hessian_lower_triangle_program(wrt)
    }

    pub fn new(ccs: CCS, nonzeros: Vec<SX>) -> Result<Self> {
        if ccs.nnz() != nonzeros.len() {
            return Err(SxError::Shape(format!(
                "CCS nnz {} does not match value nnz {}",
                ccs.nnz(),
                nonzeros.len()
            )));
        }
        Ok(Self { ccs, nonzeros })
    }

    pub fn scalar(value: impl Into<SX>) -> Self {
        Self {
            ccs: CCS::scalar(),
            nonzeros: vec![value.into()],
        }
    }

    pub fn dense(nrow: Index, ncol: Index, values: Vec<SX>) -> Result<Self> {
        Self::new(CCS::dense(nrow, ncol)?, values)
    }

    pub fn dense_column(values: Vec<SX>) -> Result<Self> {
        Self::dense(values.len(), 1, values)
    }

    pub fn map_nonzeros(&self, mut f: impl FnMut(SX) -> SX) -> Self {
        Self {
            ccs: self.ccs.clone(),
            nonzeros: self.nonzeros.iter().copied().map(&mut f).collect(),
        }
    }

    pub fn sym(name: impl AsRef<str>, ccs: CCS) -> Result<Self> {
        let values = ccs
            .positions()
            .into_iter()
            .enumerate()
            .map(|(idx, (row, col))| {
                let base = name.as_ref();
                if ccs.is_scalar() {
                    SX::sym(base)
                } else {
                    SX::sym(format!("{base}_{row}_{col}_{idx}"))
                }
            })
            .collect();
        Self::new(ccs, values)
    }

    pub fn sym_dense(name: impl AsRef<str>, nrow: Index, ncol: Index) -> Result<Self> {
        Self::sym(name, CCS::dense(nrow, ncol)?)
    }

    pub fn ccs(&self) -> &CCS {
        &self.ccs
    }

    pub fn nonzeros(&self) -> &[SX] {
        &self.nonzeros
    }

    pub fn nnz(&self) -> Index {
        self.nonzeros.len()
    }

    pub fn shape(&self) -> (Index, Index) {
        (self.ccs.nrow(), self.ccs.ncol())
    }

    pub fn transpose(&self) -> Self {
        let ccs = self.ccs.transpose();
        let nonzeros = ccs
            .positions()
            .into_iter()
            .map(|(row, col)| self.get(col, row))
            .collect();
        Self { ccs, nonzeros }
    }

    pub fn reshape(&self, nrow: Index, ncol: Index) -> Result<Self> {
        let ccs = self.ccs.reshape(nrow, ncol)?;
        let old_nrow = self.ccs.nrow();
        let value_by_linear = self
            .ccs
            .positions()
            .into_iter()
            .zip(self.nonzeros.iter().copied())
            .map(|((row, col), value)| (row + col * old_nrow, value))
            .collect::<HashMap<_, _>>();
        let nonzeros = ccs
            .positions()
            .into_iter()
            .map(|(row, col)| {
                let linear = row + col * nrow;
                value_by_linear.get(&linear).copied().ok_or_else(|| {
                    SxError::Shape(format!(
                        "missing reshaped nonzero value at linear index {linear}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(ccs, nonzeros)
    }

    pub fn scalar_expr(&self) -> Result<SX> {
        if self.ccs.is_scalar() {
            Ok(self.nonzeros[0])
        } else {
            Err(SxError::Shape("expected scalar SXMatrix".into()))
        }
    }

    pub fn nz(&self, idx: Index) -> SX {
        self.nonzeros[idx]
    }

    pub fn get(&self, row: Index, col: Index) -> SX {
        self.ccs
            .nz_index(row, col)
            .map_or_else(SX::zero, |idx| self.nonzeros[idx])
    }

    pub fn gradient(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let expr = self.scalar_expr()?;
        let grads = reverse_directional(&[expr], &wrt.nonzeros, &[SX::one()])?;
        SXMatrix::new(wrt.ccs.clone(), grads)
    }

    pub fn forward(&self, wrt: &SXMatrix, seed: &SXMatrix) -> Result<SXMatrix> {
        if wrt.ccs != seed.ccs {
            return Err(SxError::Shape(
                "forward seed CCS must match differentiation variables".into(),
            ));
        }
        let outputs = forward_directional(&self.nonzeros, &wrt.nonzeros, &seed.nonzeros)?;
        SXMatrix::new(self.ccs.clone(), outputs)
    }

    pub fn reverse(&self, wrt: &SXMatrix, seed: &SXMatrix) -> Result<SXMatrix> {
        if self.ccs != seed.ccs {
            return Err(SxError::Shape(
                "reverse seed CCS must match matrix output".into(),
            ));
        }
        let outputs = reverse_directional(&self.nonzeros, &wrt.nonzeros, &seed.nonzeros)?;
        SXMatrix::new(wrt.ccs.clone(), outputs)
    }

    fn jacobian_forward_colored(
        &self,
        wrt: &SXMatrix,
        structure: &JacobianStructure,
    ) -> Result<Vec<SX>> {
        let mut values = vec![SX::zero(); structure.ccs.nnz()];
        for color_batch in structure.forward_color_groups.chunks(JACOBIAN_BATCH_WIDTH) {
            let sensitivities =
                forward_directional_basis_batch(&self.nonzeros, &wrt.nonzeros, color_batch)?;
            for (direction, columns) in color_batch.iter().enumerate() {
                let direction_sens = &sensitivities[direction];
                for &col in columns {
                    let start = structure.ccs.col_ptrs()[col];
                    let end = structure.ccs.col_ptrs()[col + 1];
                    for nz_index in start..end {
                        values[nz_index] = direction_sens[structure.ccs.row_indices()[nz_index]];
                    }
                }
            }
        }
        Ok(values)
    }

    fn jacobian_reverse_colored(
        &self,
        wrt: &SXMatrix,
        structure: &JacobianStructure,
    ) -> Result<Vec<SX>> {
        let nout = self.nnz();
        let mut values = vec![SX::zero(); structure.ccs.nnz()];
        for color_batch in structure.reverse_color_groups.chunks(JACOBIAN_BATCH_WIDTH) {
            let seeds_by_direction = color_batch
                .iter()
                .map(|rows| {
                    let mut seed = vec![SX::zero(); nout];
                    for &row in rows {
                        seed[row] = SX::one();
                    }
                    seed
                })
                .collect::<Vec<_>>();
            let sensitivities =
                reverse_directional_batch(&self.nonzeros, &wrt.nonzeros, &seeds_by_direction)?;
            for (direction, rows) in color_batch.iter().enumerate() {
                let direction_sens = &sensitivities[direction];
                for &row in rows {
                    for &(col, nz_index) in &structure.positions_by_row[row] {
                        values[nz_index] = direction_sens[col];
                    }
                }
            }
        }
        Ok(values)
    }

    pub fn jacobian(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let nout = self.nnz();
        let nvar = wrt.nnz();
        let structure = jacobian_structure(&self.nonzeros, &wrt.nonzeros)?;
        let forward_batches = structure
            .forward_color_groups
            .len()
            .div_ceil(JACOBIAN_BATCH_WIDTH);
        let reverse_batches = structure
            .reverse_color_groups
            .len()
            .div_ceil(JACOBIAN_BATCH_WIDTH);
        let values = if forward_batches <= reverse_batches {
            self.jacobian_forward_colored(wrt, &structure)?
        } else {
            self.jacobian_reverse_colored(wrt, &structure)?
        };

        let mut positions = Vec::with_capacity(structure.ccs.nnz());
        let mut filtered_values = Vec::with_capacity(structure.ccs.nnz());
        for col in 0..nvar {
            for nz_index in structure.ccs.col_ptrs()[col]..structure.ccs.col_ptrs()[col + 1] {
                let value = values[nz_index];
                if !value.is_zero() {
                    positions.push((structure.ccs.row_indices()[nz_index], col));
                    filtered_values.push(value);
                }
            }
        }
        let filtered_ccs = CCS::from_positions(nout, nvar, &positions)?;
        SXMatrix::new(filtered_ccs, filtered_values)
    }

    pub fn jacobian_ccs(&self, wrt: &SXMatrix) -> Result<CCS> {
        Ok(jacobian_structure(&self.nonzeros, &wrt.nonzeros)?.ccs)
    }

    pub fn hessian(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        self.hessian_with_options(wrt, HessianOptions::default())
    }

    pub fn hessian_with_strategy(
        &self,
        wrt: &SXMatrix,
        strategy: HessianStrategy,
    ) -> Result<SXMatrix> {
        self.hessian_with_options(wrt, HessianOptions::with_strategy(strategy))
    }

    pub fn hessian_with_options(
        &self,
        wrt: &SXMatrix,
        options: HessianOptions,
    ) -> Result<SXMatrix> {
        match options.strategy {
            HessianStrategy::LowerTriangleByColumn => self.hessian_lower_triangle_by_column(wrt),
            HessianStrategy::LowerTriangleSelectedOutputs => {
                self.hessian_lower_triangle_selected_outputs(wrt)
            }
            HessianStrategy::LowerTriangleColored => self.hessian_lower_triangle_colored(wrt),
        }
    }
}
