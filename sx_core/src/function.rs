use std::collections::{BTreeSet, HashMap, HashSet};

use crate::Index;
use crate::error::{Result, SxError};
use crate::{SX, SXMatrix};

#[derive(Clone, Debug, PartialEq)]
pub struct NamedMatrix {
    name: String,
    matrix: SXMatrix,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SXFunction {
    name: String,
    inputs: Vec<NamedMatrix>,
    outputs: Vec<NamedMatrix>,
}

impl NamedMatrix {
    pub fn new(name: impl Into<String>, matrix: SXMatrix) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(SxError::Graph(
                "matrix argument name cannot be empty".into(),
            ));
        }
        Ok(Self { name, matrix })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn matrix(&self) -> &SXMatrix {
        &self.matrix
    }
}

impl SXFunction {
    pub fn new(
        name: impl Into<String>,
        inputs: Vec<NamedMatrix>,
        outputs: Vec<NamedMatrix>,
    ) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(SxError::Graph("function name cannot be empty".into()));
        }
        let mut input_names = HashSet::new();
        let mut output_names = HashSet::new();
        let mut input_symbols = HashSet::new();
        for input in &inputs {
            if !input_names.insert(input.name.clone()) {
                return Err(SxError::Graph(format!(
                    "duplicate input name {}",
                    input.name
                )));
            }
            for &symbol in input.matrix.nonzeros() {
                if !symbol.is_symbolic() {
                    return Err(SxError::Graph(format!(
                        "input {} must contain only symbolic primitives",
                        input.name
                    )));
                }
                if !input_symbols.insert(symbol) {
                    return Err(SxError::Graph(format!(
                        "symbol {symbol} appears in multiple input slots"
                    )));
                }
            }
        }
        for output in &outputs {
            if !output_names.insert(output.name.clone()) {
                return Err(SxError::Graph(format!(
                    "duplicate output name {}",
                    output.name
                )));
            }
            for &expr in output.matrix.nonzeros() {
                for symbol in expr.free_symbols() {
                    if !input_symbols.contains(&symbol) {
                        return Err(SxError::Graph(format!(
                            "output {} references free symbol {symbol} not present in inputs",
                            output.name
                        )));
                    }
                }
            }
        }
        Ok(Self {
            name,
            inputs,
            outputs,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn n_in(&self) -> Index {
        self.inputs.len()
    }

    pub fn n_out(&self) -> Index {
        self.outputs.len()
    }

    pub fn inputs(&self) -> &[NamedMatrix] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[NamedMatrix] {
        &self.outputs
    }

    pub fn size_in(&self, slot: Index) -> (Index, Index) {
        self.inputs[slot].matrix.shape()
    }

    pub fn size_out(&self, slot: Index) -> (Index, Index) {
        self.outputs[slot].matrix.shape()
    }

    pub fn input_bindings(&self) -> HashMap<SX, (Index, Index)> {
        self.inputs
            .iter()
            .enumerate()
            .flat_map(|(slot, input)| {
                input
                    .matrix
                    .nonzeros()
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(offset, symbol)| (symbol, (slot, offset)))
            })
            .collect()
    }

    pub fn free_symbols(&self) -> BTreeSet<SX> {
        let mut free = BTreeSet::new();
        for output in &self.outputs {
            for &expr in output.matrix.nonzeros() {
                free.extend(expr.free_symbols());
            }
        }
        free
    }
}
