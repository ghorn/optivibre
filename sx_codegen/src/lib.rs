use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::process::{Command, Stdio};

use anyhow::{Result, anyhow, bail};
use sx_core::{
    BinaryOp, CCS, CallPolicy, CallPolicyConfig, CompileStats, CompileWarning, Index, InlineStage,
    NodeView, SX, SXFunction, SXMatrix, UnaryOp, lookup_function_ref, rewrite_function_for_stage,
};

#[derive(Clone, Debug, PartialEq)]
pub struct LoweredFunction {
    pub name: String,
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Slot>,
    pub instructions: Vec<Instruction>,
    pub output_values: Vec<Vec<ValueRef>>,
    pub subfunctions: Vec<LoweredSubfunction>,
    pub stats: CompileStats,
    pub warnings: Vec<CompileWarning>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoweredSubfunction {
    pub name: String,
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Slot>,
    pub instructions: Vec<Instruction>,
    pub output_values: Vec<Vec<ValueRef>>,
    pub call_policy: CallPolicy,
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
pub enum Instruction {
    Unary {
        temp: Index,
        op: UnaryOp,
        input: ValueRef,
    },
    Binary {
        temp: Index,
        op: BinaryOp,
        lhs: ValueRef,
        rhs: ValueRef,
    },
    Call {
        temps: Vec<Index>,
        callee: Index,
        inputs: Vec<ValueRef>,
    },
}

impl Instruction {
    pub fn output_temps(&self) -> &[Index] {
        match self {
            Self::Unary { temp, .. } | Self::Binary { temp, .. } => std::slice::from_ref(temp),
            Self::Call { temps, .. } => temps,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CallSiteKey {
    function_id: usize,
    inputs: Vec<SXMatrix>,
}

#[derive(Default)]
struct SharedLoweringState {
    subfunctions: Vec<LoweredSubfunction>,
    lowered_functions: HashMap<usize, Index>,
    counted_call_sites: HashSet<CallSiteKey>,
    counted_functions: HashSet<usize>,
}

struct CallableLoweringState<'a> {
    input_bindings: HashMap<SX, (Index, Index)>,
    instructions: Vec<Instruction>,
    expr_values: HashMap<SX, ValueRef>,
    call_outputs: HashMap<CallSiteKey, Vec<Index>>,
    next_temp: Index,
    shared: &'a mut SharedLoweringState,
    config: CallPolicyConfig,
    stats: &'a mut CompileStats,
    warnings: &'a mut Vec<CompileWarning>,
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

fn effective_call_policy(function: &SXFunction, config: CallPolicyConfig) -> CallPolicy {
    if config.respect_function_overrides {
        function
            .call_policy_override()
            .unwrap_or(config.default_policy)
    } else {
        config.default_policy
    }
}

fn prepare_function_for_lowering(
    function: &SXFunction,
    config: CallPolicyConfig,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
) -> Result<SXFunction> {
    let after_inline_at_call =
        rewrite_function_for_stage(function, config, InlineStage::Call, stats, warnings)?;
    Ok(rewrite_function_for_stage(
        &after_inline_at_call,
        config,
        InlineStage::Lowering,
        stats,
        warnings,
    )?)
}

fn make_slots(matrices: &[sx_core::NamedMatrix]) -> Vec<Slot> {
    matrices
        .iter()
        .map(|slot| Slot {
            name: sanitize_ident(slot.name()),
            ccs: slot.matrix().ccs().clone(),
        })
        .collect()
}

fn alloc_temp_range(next_temp: &mut Index, count: Index) -> Vec<Index> {
    let start = *next_temp;
    *next_temp += count;
    (start..start + count).collect()
}

fn call_output_linear_index(function: &SXFunction, slot: Index, offset: Index) -> Index {
    function
        .outputs()
        .iter()
        .take(slot)
        .map(|output| output.matrix().nnz())
        .sum::<Index>()
        + offset
}

impl CallableLoweringState<'_> {
    fn lower_expr(&mut self, expr: SX) -> Result<ValueRef> {
        if let Some(existing) = self.expr_values.get(&expr).copied() {
            return Ok(existing);
        }

        let value = match expr.inspect() {
            NodeView::Constant(value) => ValueRef::Const(value),
            NodeView::Symbol { .. } => {
                let (slot, offset) = self
                    .input_bindings
                    .get(&expr)
                    .copied()
                    .ok_or_else(|| anyhow!("symbol {expr} is not a declared function input"))?;
                ValueRef::Input { slot, offset }
            }
            NodeView::Unary { op, arg } => {
                let input = self.lower_expr(arg)?;
                let temp = self.next_temp;
                self.next_temp += 1;
                self.instructions
                    .push(Instruction::Unary { temp, op, input });
                ValueRef::Temp(temp)
            }
            NodeView::Binary { op, lhs, rhs } => {
                let lhs = self.lower_expr(lhs)?;
                let rhs = self.lower_expr(rhs)?;
                let temp = self.next_temp;
                self.next_temp += 1;
                self.instructions
                    .push(Instruction::Binary { temp, op, lhs, rhs });
                ValueRef::Temp(temp)
            }
            NodeView::Call {
                function_id,
                inputs,
                output_slot,
                output_offset,
                ..
            } => {
                let key = CallSiteKey {
                    function_id,
                    inputs: inputs.clone(),
                };
                if let Some(temps) = self.call_outputs.get(&key) {
                    ValueRef::Temp(
                        temps[call_output_linear_index(
                            &lookup_function_ref(function_id)
                                .expect("lowering should only reference known functions"),
                            output_slot,
                            output_offset,
                        )],
                    )
                } else {
                    if self.shared.counted_call_sites.insert(key.clone()) {
                        self.stats.call_site_count += 1;
                    }
                    let callee = lookup_function_ref(function_id)
                        .expect("lowering should only reference known functions");
                    let callee_policy = effective_call_policy(&callee, self.config);
                    let callee_index = if let Some(existing) =
                        self.shared.lowered_functions.get(&function_id).copied()
                    {
                        existing
                    } else {
                        let prepared = prepare_function_for_lowering(
                            &callee,
                            self.config,
                            self.stats,
                            self.warnings,
                        )?;
                        if self.shared.counted_functions.insert(function_id) {
                            self.stats.symbolic_function_count += 1;
                        }
                        let lowered = lower_callable(
                            &prepared,
                            callee_policy,
                            self.config,
                            self.shared,
                            self.stats,
                            self.warnings,
                        )?;
                        let index = self.shared.subfunctions.len();
                        self.shared.subfunctions.push(lowered);
                        self.shared.lowered_functions.insert(function_id, index);
                        self.stats.llvm_subfunctions_emitted = self.shared.subfunctions.len();
                        index
                    };

                    let output_count = self.shared.subfunctions[callee_index]
                        .output_values
                        .iter()
                        .map(Vec::len)
                        .sum::<Index>();
                    let lowered_inputs = inputs
                        .iter()
                        .flat_map(|input| input.nonzeros().iter().copied())
                        .map(|value| self.lower_expr(value))
                        .collect::<Result<Vec<_>>>()?;
                    let temps = alloc_temp_range(&mut self.next_temp, output_count);
                    self.instructions.push(Instruction::Call {
                        temps: temps.clone(),
                        callee: callee_index,
                        inputs: lowered_inputs,
                    });
                    self.call_outputs.insert(key, temps.clone());
                    self.stats.llvm_call_instructions_emitted += 1;
                    ValueRef::Temp(
                        temps[call_output_linear_index(&callee, output_slot, output_offset)],
                    )
                }
            }
        };
        self.expr_values.insert(expr, value);
        Ok(value)
    }
}

fn lower_callable(
    function: &SXFunction,
    call_policy: CallPolicy,
    config: CallPolicyConfig,
    shared: &mut SharedLoweringState,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
) -> Result<LoweredSubfunction> {
    let input_bindings = function.input_bindings();
    let inputs = make_slots(function.inputs());
    let outputs = make_slots(function.outputs());
    let mut state = CallableLoweringState {
        input_bindings,
        instructions: Vec::new(),
        expr_values: HashMap::new(),
        call_outputs: HashMap::new(),
        next_temp: 0,
        shared,
        config,
        stats,
        warnings,
    };
    let output_values = function
        .outputs()
        .iter()
        .map(|slot| {
            slot.matrix()
                .nonzeros()
                .iter()
                .copied()
                .map(|expr| state.lower_expr(expr))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LoweredSubfunction {
        name: sanitize_ident(function.name()),
        inputs,
        outputs,
        instructions: state.instructions,
        output_values,
        call_policy,
    })
}

pub fn lower_function(function: &SXFunction) -> Result<LoweredFunction> {
    lower_function_with_policies(function, CallPolicyConfig::default())
}

pub fn lower_function_with_policies(
    function: &SXFunction,
    config: CallPolicyConfig,
) -> Result<LoweredFunction> {
    let mut stats = CompileStats::default();
    let mut warnings = Vec::new();
    let prepared = prepare_function_for_lowering(function, config, &mut stats, &mut warnings)?;
    let mut shared = SharedLoweringState::default();
    if shared.counted_functions.insert(function.id()) {
        stats.symbolic_function_count += 1;
    }
    let root = lower_callable(
        &prepared,
        effective_call_policy(function, config),
        config,
        &mut shared,
        &mut stats,
        &mut warnings,
    )?;
    stats.llvm_root_instructions_emitted = root.instructions.len();
    stats.llvm_total_instructions_emitted = root.instructions.len()
        + shared
            .subfunctions
            .iter()
            .map(|subfunction| subfunction.instructions.len())
            .sum::<usize>();
    Ok(LoweredFunction {
        name: root.name,
        inputs: root.inputs,
        outputs: root.outputs,
        instructions: root.instructions,
        output_values: root.output_values,
        subfunctions: shared.subfunctions,
        stats,
        warnings,
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
