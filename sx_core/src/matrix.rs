use std::collections::{BTreeSet, HashMap};

use crate::ccs::CCS;
use crate::error::{Result, SxError};
use crate::sx::{SX, depends_on, forward_directional, reverse_directional};
use crate::{Index, checked_len_product};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SXMatrix {
    ccs: CCS,
    nonzeros: Vec<SX>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HessianStrategy {
    LowerTriangleByColumn,
    #[default]
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

    fn hessian_lower_triangle_by_column(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let grad = self.gradient(wrt)?;
        let n = wrt.nnz();
        let mut columns = vec![Vec::new(); n];
        for col in 0..n {
            let mut seed = vec![SX::zero(); n];
            seed[col] = SX::one();
            let sens = forward_directional(&grad.nonzeros, &wrt.nonzeros, &seed)?;
            for (row, value) in sens.iter().copied().enumerate().skip(col) {
                columns[col].push((row, value));
            }
        }
        Self::build_lower_triangle(n, columns)
    }

    fn hessian_lower_triangle_selected_outputs(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let grad = self.gradient(wrt)?;
        let n = wrt.nnz();
        let mut columns = vec![Vec::new(); n];
        for col in 0..n {
            let mut seed = vec![SX::zero(); n];
            seed[col] = SX::one();
            let outputs = &grad.nonzeros[col..];
            let sens = forward_directional(outputs, &wrt.nonzeros, &seed)?;
            for (offset, value) in sens.into_iter().enumerate() {
                columns[col].push((col + offset, value));
            }
        }
        Self::build_lower_triangle(n, columns)
    }

    fn hessian_lower_triangle_colored(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let grad = self.gradient(wrt)?;
        let n = wrt.nnz();
        let full_ccs = grad.jacobian_ccs(wrt)?;
        let mut row_sets = vec![Vec::new(); n];
        for (row, col) in full_ccs.positions() {
            row_sets[col].push(row);
        }

        let mut color_unions = Vec::<BTreeSet<Index>>::new();
        let mut color_columns = Vec::<Vec<Index>>::new();
        for (col, rows) in row_sets.iter().enumerate() {
            let color_idx = color_unions
                .iter()
                .position(|union| rows.iter().all(|row| !union.contains(row)))
                .unwrap_or_else(|| {
                    color_unions.push(BTreeSet::new());
                    color_columns.push(Vec::new());
                    color_unions.len() - 1
                });
            for &row in rows {
                color_unions[color_idx].insert(row);
            }
            color_columns[color_idx].push(col);
        }

        let mut columns = vec![Vec::new(); n];
        for group in color_columns {
            let mut seed = vec![SX::zero(); n];
            for &col in &group {
                seed[col] = SX::one();
            }
            let sens = forward_directional(&grad.nonzeros, &wrt.nonzeros, &seed)?;
            for col in group {
                for &row in &row_sets[col] {
                    if row >= col {
                        columns[col].push((row, sens[row]));
                    }
                }
            }
        }

        Self::build_lower_triangle(n, columns)
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

    pub fn jacobian(&self, wrt: &SXMatrix) -> Result<SXMatrix> {
        let nout = self.nnz();
        let nvar = wrt.nnz();
        let Some(len) = checked_len_product(nout, nvar) else {
            return Err(SxError::Shape("jacobian storage size overflow".into()));
        };
        let mut dense = vec![SX::zero(); len];

        if nvar <= nout {
            for var_idx in 0..nvar {
                let mut seed = vec![SX::zero(); nvar];
                seed[var_idx] = SX::one();
                let sens = forward_directional(&self.nonzeros, &wrt.nonzeros, &seed)?;
                for (row, value) in sens.into_iter().enumerate() {
                    dense[row + var_idx * nout] = value;
                }
            }
        } else {
            for out_idx in 0..nout {
                let mut seed = vec![SX::zero(); nout];
                seed[out_idx] = SX::one();
                let sens = reverse_directional(&self.nonzeros, &wrt.nonzeros, &seed)?;
                for (col, value) in sens.into_iter().enumerate() {
                    dense[out_idx + col * nout] = value;
                }
            }
        }

        let mut positions = Vec::new();
        let mut values = Vec::new();
        for col in 0..nvar {
            for row in 0..nout {
                let value = dense[row + col * nout];
                if !value.is_zero() {
                    positions.push((row, col));
                    values.push(value);
                }
            }
        }
        let ccs = CCS::from_positions(nout, nvar, &positions)?;
        SXMatrix::new(ccs, values)
    }

    pub fn jacobian_ccs(&self, wrt: &SXMatrix) -> Result<CCS> {
        let mut positions = Vec::new();
        for (col, &var) in wrt.nonzeros.iter().enumerate() {
            for (row, &out) in self.nonzeros.iter().enumerate() {
                if depends_on(out, var, &mut HashMap::new()) {
                    positions.push((row, col));
                }
            }
        }
        CCS::from_positions(self.nnz(), wrt.nnz(), &positions)
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
