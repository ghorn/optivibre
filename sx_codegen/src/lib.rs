use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::process::{Command, Stdio};

use anyhow::{Result, anyhow, bail};
use sx_core::{BinaryOp, CCS, Index, NodeView, SX, SXFunction, UnaryOp};

#[derive(Clone, Debug, PartialEq)]
pub struct LoweredFunction {
    pub name: String,
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Slot>,
    pub instructions: Vec<Instruction>,
    pub output_values: Vec<Vec<ValueRef>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub name: String,
    pub ccs: CCS,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValueRef {
    Input { slot: Index, offset: Index },
    Temp(Index),
    Const(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstructionKind {
    Unary {
        op: UnaryOp,
        input: ValueRef,
    },
    Binary {
        op: BinaryOp,
        lhs: ValueRef,
        rhs: ValueRef,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    pub temp: Index,
    pub kind: InstructionKind,
}

pub fn sanitize_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 2);
    for (idx, ch) in name.chars().enumerate() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_';
        if idx == 0 {
            if ch.is_ascii_alphabetic() || ch == '_' {
                out.push(ch);
            } else if ok {
                out.push('_');
                out.push(ch);
            } else {
                out.push('_');
            }
        } else if ok {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "_generated".into()
    } else {
        out
    }
}

pub fn to_pascal_case(name: &str) -> String {
    let mut out = String::new();
    for piece in sanitize_ident(name)
        .split('_')
        .filter(|part| !part.is_empty())
    {
        let mut chars = piece.chars();
        if let Some(first) = chars.next() {
            out.push(first.to_ascii_uppercase());
            out.extend(chars.map(|ch| ch.to_ascii_lowercase()));
        }
    }
    if out.is_empty() {
        "Generated".into()
    } else {
        out
    }
}

fn topo_visit(
    expr: SX,
    input_bindings: &HashMap<SX, (Index, Index)>,
    seen: &mut HashSet<SX>,
    order: &mut Vec<SX>,
) -> Result<()> {
    if !seen.insert(expr) {
        return Ok(());
    }
    match expr.inspect() {
        NodeView::Constant(_) => Ok(()),
        NodeView::Symbol { .. } => {
            if input_bindings.contains_key(&expr) {
                Ok(())
            } else {
                bail!("symbol {expr} is not bound as a function input")
            }
        }
        NodeView::Unary { arg, .. } => {
            topo_visit(arg, input_bindings, seen, order)?;
            order.push(expr);
            Ok(())
        }
        NodeView::Binary { lhs, rhs, .. } => {
            topo_visit(lhs, input_bindings, seen, order)?;
            topo_visit(rhs, input_bindings, seen, order)?;
            order.push(expr);
            Ok(())
        }
    }
}

fn value_ref(
    expr: SX,
    inputs: &HashMap<SX, (Index, Index)>,
    temps: &HashMap<SX, Index>,
) -> Result<ValueRef> {
    Ok(match expr.inspect() {
        NodeView::Constant(v) => ValueRef::Const(v),
        NodeView::Symbol { .. } => {
            let (slot, offset) = inputs
                .get(&expr)
                .copied()
                .ok_or_else(|| anyhow!("symbol {expr} is not a declared function input"))?;
            ValueRef::Input { slot, offset }
        }
        NodeView::Unary { .. } | NodeView::Binary { .. } => ValueRef::Temp(
            *temps
                .get(&expr)
                .ok_or_else(|| anyhow!("internal node {expr} was not topologically lowered"))?,
        ),
    })
}

pub fn lower_function(function: &SXFunction) -> Result<LoweredFunction> {
    let input_bindings = function.input_bindings();
    let inputs = function
        .inputs()
        .iter()
        .map(|slot| Slot {
            name: sanitize_ident(slot.name()),
            ccs: slot.matrix().ccs().clone(),
        })
        .collect::<Vec<_>>();
    let outputs = function
        .outputs()
        .iter()
        .map(|slot| Slot {
            name: sanitize_ident(slot.name()),
            ccs: slot.matrix().ccs().clone(),
        })
        .collect::<Vec<_>>();

    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for output in function.outputs() {
        for &expr in output.matrix().nonzeros() {
            topo_visit(expr, &input_bindings, &mut seen, &mut order)?;
        }
    }

    let mut temps = HashMap::new();
    let mut instructions = Vec::with_capacity(order.len());
    for (temp, node) in order.iter().copied().enumerate() {
        temps.insert(node, temp);
        let kind = match node.inspect() {
            NodeView::Unary { op, arg } => InstructionKind::Unary {
                op,
                input: value_ref(arg, &input_bindings, &temps)?,
            },
            NodeView::Binary { op, lhs, rhs } => InstructionKind::Binary {
                op,
                lhs: value_ref(lhs, &input_bindings, &temps)?,
                rhs: value_ref(rhs, &input_bindings, &temps)?,
            },
            NodeView::Constant(_) | NodeView::Symbol { .. } => {
                bail!("topological order should only contain unary or binary operations")
            }
        };
        instructions.push(Instruction { temp, kind });
    }

    let output_values = function
        .outputs()
        .iter()
        .map(|slot| {
            slot.matrix()
                .nonzeros()
                .iter()
                .copied()
                .map(|expr| value_ref(expr, &input_bindings, &temps))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LoweredFunction {
        name: sanitize_ident(function.name()),
        inputs,
        outputs,
        instructions,
        output_values,
    })
}

pub fn format_rust_source(source: &str) -> Result<String> {
    let rustfmt = std::env::var("RUSTFMT").unwrap_or_else(|_| "rustfmt".into());
    let mut child = Command::new(rustfmt)
        .arg("--edition")
        .arg("2024")
        .arg("--emit")
        .arg("stdout")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let Some(stdin) = child.stdin.as_mut() else {
            bail!("failed to open rustfmt stdin");
        };
        stdin.write_all(source.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).map_err(Into::into)
}
