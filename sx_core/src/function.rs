use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::Index;
use crate::error::{Result, SxError};
use crate::sx::{call_output, depends_on, forward_directional, reverse_directional};
use crate::{BinaryOp, NodeView, SX, SXMatrix, UnaryOp};

pub type FunctionId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallPolicy {
    InlineAtCall,
    InlineAtLowering,
    InlineInLLVM,
    NoInlineLLVM,
}

impl Default for CallPolicy {
    fn default() -> Self {
        Self::InlineAtLowering
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CallPolicyConfig {
    pub default_policy: CallPolicy,
    pub respect_function_overrides: bool,
}

impl Default for CallPolicyConfig {
    fn default() -> Self {
        Self {
            default_policy: CallPolicy::InlineAtLowering,
            respect_function_overrides: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    pub symbolic_function_count: usize,
    pub call_site_count: usize,
    pub max_call_depth: usize,
    pub inline_at_call_policy_count: usize,
    pub inline_at_lowering_policy_count: usize,
    pub inline_in_llvm_policy_count: usize,
    pub no_inline_llvm_policy_count: usize,
    pub overrides_applied: usize,
    pub overrides_ignored: usize,
    pub inlines_at_call: usize,
    pub inlines_at_lowering: usize,
    pub llvm_subfunctions_emitted: usize,
    pub llvm_call_instructions_emitted: usize,
}

impl CompileStats {
    pub fn absorb(&mut self, other: &Self) {
        self.symbolic_function_count += other.symbolic_function_count;
        self.call_site_count += other.call_site_count;
        self.max_call_depth = self.max_call_depth.max(other.max_call_depth);
        self.inline_at_call_policy_count += other.inline_at_call_policy_count;
        self.inline_at_lowering_policy_count += other.inline_at_lowering_policy_count;
        self.inline_in_llvm_policy_count += other.inline_in_llvm_policy_count;
        self.no_inline_llvm_policy_count += other.no_inline_llvm_policy_count;
        self.overrides_applied += other.overrides_applied;
        self.overrides_ignored += other.overrides_ignored;
        self.inlines_at_call += other.inlines_at_call;
        self.inlines_at_lowering += other.inlines_at_lowering;
        self.llvm_subfunctions_emitted += other.llvm_subfunctions_emitted;
        self.llvm_call_instructions_emitted += other.llvm_call_instructions_emitted;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileWarning {
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedMatrix {
    name: String,
    matrix: SXMatrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SXFunction {
    id: FunctionId,
    name: String,
    inputs: Vec<NamedMatrix>,
    outputs: Vec<NamedMatrix>,
    call_policy_override: Option<CallPolicy>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DependencyProfile {
    input_offsets: Vec<Index>,
    output_offsets: Vec<Index>,
    deps: Vec<Vec<bool>>,
}

#[derive(Default)]
struct FunctionRegistry {
    definitions: HashMap<FunctionId, SXFunction>,
}

#[derive(Default)]
struct FunctionCaches {
    dependency_profiles: HashMap<FunctionId, DependencyProfile>,
    forward_helpers: HashMap<FunctionId, SXFunction>,
    reverse_helpers: HashMap<FunctionId, SXFunction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineStage {
    Call,
    Lowering,
}

impl InlineStage {
    fn should_inline(self, policy: CallPolicy) -> bool {
        match self {
            Self::Call => matches!(policy, CallPolicy::InlineAtCall),
            Self::Lowering => matches!(
                policy,
                CallPolicy::InlineAtCall | CallPolicy::InlineAtLowering
            ),
        }
    }
}

#[derive(Default)]
struct RewriteState {
    warned_ignored_overrides: HashSet<FunctionId>,
}

static NEXT_FUNCTION_ID: AtomicUsize = AtomicUsize::new(1);
static FUNCTION_REGISTRY: OnceLock<Mutex<FunctionRegistry>> = OnceLock::new();
static FUNCTION_CACHES: OnceLock<Mutex<FunctionCaches>> = OnceLock::new();

fn lock_registry() -> MutexGuard<'static, FunctionRegistry> {
    match FUNCTION_REGISTRY
        .get_or_init(|| Mutex::new(FunctionRegistry::default()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lock_caches() -> MutexGuard<'static, FunctionCaches> {
    match FUNCTION_CACHES
        .get_or_init(|| Mutex::new(FunctionCaches::default()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn register_function(function: SXFunction) {
    lock_registry().definitions.insert(function.id, function);
}

fn next_function_id() -> FunctionId {
    NEXT_FUNCTION_ID.fetch_add(1, Ordering::Relaxed)
}

fn validate_function(
    name: &str,
    inputs: &[NamedMatrix],
    outputs: &[NamedMatrix],
) -> Result<()> {
    if name.trim().is_empty() {
        return Err(SxError::Graph("function name cannot be empty".into()));
    }
    let mut input_names = HashSet::new();
    let mut output_names = HashSet::new();
    let mut input_symbols = HashSet::new();
    for input in inputs {
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
    for output in outputs {
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
    Ok(())
}

pub(crate) fn function_by_id(id: FunctionId) -> Option<SXFunction> {
    lock_registry().definitions.get(&id).cloned()
}

pub fn lookup_function(id: FunctionId) -> Option<SXFunction> {
    function_by_id(id)
}

pub(crate) fn function_name(id: FunctionId) -> Option<String> {
    function_by_id(id).map(|function| function.name)
}

impl DependencyProfile {
    fn input_index(&self, slot: Index, offset: Index) -> Index {
        self.input_offsets[slot] + offset
    }

    fn output_index(&self, slot: Index, offset: Index) -> Index {
        self.output_offsets[slot] + offset
    }

    pub(crate) fn output_depends_on(
        &self,
        output_slot: Index,
        output_offset: Index,
        input_slot: Index,
        input_offset: Index,
    ) -> bool {
        self.deps[self.output_index(output_slot, output_offset)]
            [self.input_index(input_slot, input_offset)]
    }
}

pub(crate) fn dependency_profile(function_id: FunctionId) -> DependencyProfile {
    if let Some(profile) = lock_caches().dependency_profiles.get(&function_id).cloned() {
        return profile;
    }
    let function = function_by_id(function_id).expect("function id must be registered");

    let mut input_offsets = Vec::with_capacity(function.inputs.len());
    let mut flat_inputs = Vec::new();
    for input in function.inputs() {
        input_offsets.push(flat_inputs.len());
        flat_inputs.extend(input.matrix().nonzeros().iter().copied());
    }

    let mut output_offsets = Vec::with_capacity(function.outputs.len());
    let mut deps = Vec::new();
    for output in function.outputs() {
        output_offsets.push(deps.len());
        for &expr in output.matrix().nonzeros() {
            deps.push(
                flat_inputs
                    .iter()
                    .copied()
                    .map(|symbol| depends_on(expr, symbol, &mut HashMap::new()))
                    .collect(),
            );
        }
    }

    let profile = DependencyProfile {
        input_offsets,
        output_offsets,
        deps,
    };
    lock_caches()
        .dependency_profiles
        .insert(function_id, profile.clone());
    profile
}

pub(crate) fn forward_helper(function_id: FunctionId) -> Result<SXFunction> {
    if let Some(helper) = lock_caches().forward_helpers.get(&function_id).cloned() {
        return Ok(helper);
    }
    let function = function_by_id(function_id).expect("function id must be registered");
    let mut helper_inputs = function.inputs.clone();
    let mut seed_inputs = Vec::with_capacity(function.inputs.len());
    for input in function.inputs() {
        let seed_matrix = SXMatrix::sym(
            format!("{}_seed_{}", function.name(), input.name()),
            input.matrix().ccs().clone(),
        )?;
        seed_inputs.push(seed_matrix.clone());
        helper_inputs.push(NamedMatrix::new(
            format!("seed_{}", input.name()),
            seed_matrix,
        )?);
    }

    let vars = function
        .inputs()
        .iter()
        .flat_map(|input| input.matrix().nonzeros().iter().copied())
        .collect::<Vec<_>>();
    let seeds = seed_inputs
        .iter()
        .flat_map(|seed| seed.nonzeros().iter().copied())
        .collect::<Vec<_>>();
    let helper_outputs = function
        .outputs()
        .iter()
        .map(|output| {
            let directional =
                forward_directional(output.matrix().nonzeros(), &vars, &seeds)?;
            NamedMatrix::new(
                format!("{}_directional", output.name()),
                SXMatrix::new(output.matrix().ccs().clone(), directional)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let helper = SXFunction::new(
        format!("{}_forward_helper", function.name()),
        helper_inputs,
        helper_outputs,
    )?;
    lock_caches()
        .forward_helpers
        .insert(function_id, helper.clone());
    Ok(helper)
}

pub(crate) fn reverse_helper(function_id: FunctionId) -> Result<SXFunction> {
    if let Some(helper) = lock_caches().reverse_helpers.get(&function_id).cloned() {
        return Ok(helper);
    }
    let function = function_by_id(function_id).expect("function id must be registered");
    let mut helper_inputs = function.inputs.clone();
    let mut output_seeds = Vec::with_capacity(function.outputs.len());
    for output in function.outputs() {
        let seed_matrix = SXMatrix::sym(
            format!("{}_seed_{}", function.name(), output.name()),
            output.matrix().ccs().clone(),
        )?;
        output_seeds.push(seed_matrix.clone());
        helper_inputs.push(NamedMatrix::new(
            format!("seed_{}", output.name()),
            seed_matrix,
        )?);
    }

    let vars = function
        .inputs()
        .iter()
        .flat_map(|input| input.matrix().nonzeros().iter().copied())
        .collect::<Vec<_>>();
    let outputs = function
        .outputs()
        .iter()
        .flat_map(|output| output.matrix().nonzeros().iter().copied())
        .collect::<Vec<_>>();
    let seeds = output_seeds
        .iter()
        .flat_map(|seed| seed.nonzeros().iter().copied())
        .collect::<Vec<_>>();
    let adjoints = reverse_directional(&outputs, &vars, &seeds)?;

    let mut helper_outputs = Vec::with_capacity(function.inputs.len());
    let mut offset = 0;
    for input in function.inputs() {
        let next_offset = offset + input.matrix().nnz();
        helper_outputs.push(NamedMatrix::new(
            format!("{}_adjoint", input.name()),
            SXMatrix::new(
                input.matrix().ccs().clone(),
                adjoints[offset..next_offset].to_vec(),
            )?,
        )?);
        offset = next_offset;
    }

    let helper = SXFunction::new(
        format!("{}_reverse_helper", function.name()),
        helper_inputs,
        helper_outputs,
    )?;
    lock_caches()
        .reverse_helpers
        .insert(function_id, helper.clone());
    Ok(helper)
}

fn apply_unary(op: UnaryOp, arg: SX) -> SX {
    match op {
        UnaryOp::Abs => arg.abs(),
        UnaryOp::Sign => arg.sign(),
        UnaryOp::Floor => arg.floor(),
        UnaryOp::Ceil => arg.ceil(),
        UnaryOp::Round => arg.round(),
        UnaryOp::Trunc => arg.trunc(),
        UnaryOp::Sqrt => arg.sqrt(),
        UnaryOp::Exp => arg.exp(),
        UnaryOp::Log => arg.log(),
        UnaryOp::Sin => arg.sin(),
        UnaryOp::Cos => arg.cos(),
        UnaryOp::Tan => arg.tan(),
        UnaryOp::Asin => arg.asin(),
        UnaryOp::Acos => arg.acos(),
        UnaryOp::Atan => arg.atan(),
        UnaryOp::Sinh => arg.sinh(),
        UnaryOp::Cosh => arg.cosh(),
        UnaryOp::Tanh => arg.tanh(),
        UnaryOp::Asinh => arg.asinh(),
        UnaryOp::Acosh => arg.acosh(),
        UnaryOp::Atanh => arg.atanh(),
    }
}

fn apply_binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Pow => lhs.pow(rhs),
        BinaryOp::Atan2 => lhs.atan2(rhs),
        BinaryOp::Hypot => lhs.hypot(rhs),
        BinaryOp::Mod => lhs.modulo(rhs),
        BinaryOp::Copysign => lhs.copysign(rhs),
    }
}

fn resolve_call_policy(
    function: &SXFunction,
    config: CallPolicyConfig,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
    state: &mut RewriteState,
) -> CallPolicy {
    match function.call_policy_override() {
        Some(policy) if config.respect_function_overrides => {
            stats.overrides_applied += 1;
            policy
        }
        Some(policy) => {
            stats.overrides_ignored += 1;
            if state.warned_ignored_overrides.insert(function.id()) {
                warnings.push(CompileWarning {
                    message: format!(
                        "ignored call policy override {:?} on {}; enforced global policy {:?}",
                        policy,
                        function.name(),
                        config.default_policy
                    ),
                });
            }
            config.default_policy
        }
        None => config.default_policy,
    }
}

fn record_policy(stats: &mut CompileStats, policy: CallPolicy) {
    match policy {
        CallPolicy::InlineAtCall => stats.inline_at_call_policy_count += 1,
        CallPolicy::InlineAtLowering => stats.inline_at_lowering_policy_count += 1,
        CallPolicy::InlineInLLVM => stats.inline_in_llvm_policy_count += 1,
        CallPolicy::NoInlineLLVM => stats.no_inline_llvm_policy_count += 1,
    }
}

fn rewrite_expr(
    expr: SX,
    bindings: &HashMap<SX, SX>,
    config: CallPolicyConfig,
    stage: InlineStage,
    memo: &mut HashMap<SX, SX>,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
    state: &mut RewriteState,
    depth: usize,
) -> Result<SX> {
    if let Some(existing) = memo.get(&expr) {
        return Ok(*existing);
    }
    let rewritten = match expr.inspect() {
        NodeView::Constant(value) => SX::from(value),
        NodeView::Symbol { .. } => bindings.get(&expr).copied().unwrap_or(expr),
        NodeView::Unary { op, arg } => apply_unary(
            op,
            rewrite_expr(
                arg, bindings, config, stage, memo, stats, warnings, state, depth,
            )?,
        ),
        NodeView::Binary { op, lhs, rhs } => apply_binary(
            op,
            rewrite_expr(
                lhs, bindings, config, stage, memo, stats, warnings, state, depth,
            )?,
            rewrite_expr(
                rhs, bindings, config, stage, memo, stats, warnings, state, depth,
            )?,
        ),
        NodeView::Call {
            function_id,
            inputs,
            output_slot,
            output_offset,
            ..
        } => {
            let callee = function_by_id(function_id)
                .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
            let policy = resolve_call_policy(&callee, config, stats, warnings, state);
            record_policy(stats, policy);
            let rewritten_inputs = inputs
                .iter()
                .map(|input| {
                    rewrite_matrix(
                        input, bindings, config, stage, stats, warnings, state, depth + 1,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            stats.max_call_depth = stats.max_call_depth.max(depth + 1);
            if stage.should_inline(policy) {
                match stage {
                    InlineStage::Call => stats.inlines_at_call += 1,
                    InlineStage::Lowering => stats.inlines_at_lowering += 1,
                }
                rewrite_function_outputs(
                    &callee,
                    &rewritten_inputs,
                    config,
                    stage,
                    stats,
                    warnings,
                    state,
                    depth + 1,
                )?[output_slot]
                    .nz(output_offset)
            } else {
                call_output(function_id, rewritten_inputs, output_slot, output_offset)
            }
        }
    };
    memo.insert(expr, rewritten);
    Ok(rewritten)
}

fn rewrite_matrix(
    matrix: &SXMatrix,
    bindings: &HashMap<SX, SX>,
    config: CallPolicyConfig,
    stage: InlineStage,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
    state: &mut RewriteState,
    depth: usize,
) -> Result<SXMatrix> {
    let mut memo = HashMap::new();
    SXMatrix::new(
        matrix.ccs().clone(),
        matrix
            .nonzeros()
            .iter()
            .copied()
            .map(|expr| {
                rewrite_expr(expr, bindings, config, stage, &mut memo, stats, warnings, state, depth)
            })
            .collect::<Result<Vec<_>>>()?,
    )
}

fn rewrite_function_outputs(
    function: &SXFunction,
    inputs: &[SXMatrix],
    config: CallPolicyConfig,
    stage: InlineStage,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
    state: &mut RewriteState,
    depth: usize,
) -> Result<Vec<SXMatrix>> {
    let bindings = function
        .inputs()
        .iter()
        .zip(inputs)
        .flat_map(|(formal, actual)| {
            formal
                .matrix()
                .nonzeros()
                .iter()
                .copied()
                .zip(actual.nonzeros().iter().copied())
        })
        .collect::<HashMap<_, _>>();
    function
        .outputs()
        .iter()
        .map(|output| {
            rewrite_matrix(
                output.matrix(),
                &bindings,
                config,
                stage,
                stats,
                warnings,
                state,
                depth,
            )
        })
        .collect()
}

pub fn rewrite_function_for_stage(
    function: &SXFunction,
    config: CallPolicyConfig,
    stage: InlineStage,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
) -> Result<SXFunction> {
    let mut state = RewriteState::default();
    let inputs = function.inputs.clone();
    let outputs = function
        .outputs()
        .iter()
        .map(|output| {
            Ok(NamedMatrix::new(
                output.name(),
                rewrite_matrix(
                    output.matrix(),
                    &HashMap::new(),
                    config,
                    stage,
                    stats,
                    warnings,
                    &mut state,
                    0,
                )?,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    SXFunction::from_parts(
        function.name.clone(),
        inputs,
        outputs,
        function.call_policy_override,
    )
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
        Self::from_parts(name.into(), inputs, outputs, None)
    }

    fn from_parts(
        name: String,
        inputs: Vec<NamedMatrix>,
        outputs: Vec<NamedMatrix>,
        call_policy_override: Option<CallPolicy>,
    ) -> Result<Self> {
        validate_function(&name, &inputs, &outputs)?;
        let function = Self {
            id: next_function_id(),
            name,
            inputs,
            outputs,
            call_policy_override,
        };
        register_function(function.clone());
        Ok(function)
    }

    pub fn id(&self) -> FunctionId {
        self.id
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

    pub fn call_policy_override(&self) -> Option<CallPolicy> {
        self.call_policy_override
    }

    pub fn with_call_policy_override(&self, policy: CallPolicy) -> Self {
        Self::from_parts(
            self.name.clone(),
            self.inputs.clone(),
            self.outputs.clone(),
            Some(policy),
        )
        .expect("existing function definition should remain valid when adding a policy override")
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

    pub fn call(&self, inputs: &[SXMatrix]) -> Result<Vec<SXMatrix>> {
        if inputs.len() != self.inputs.len() {
            return Err(SxError::Shape(format!(
                "function {} expected {} input slots, got {}",
                self.name,
                self.inputs.len(),
                inputs.len()
            )));
        }
        for (slot, (actual, formal)) in inputs.iter().zip(&self.inputs).enumerate() {
            if actual.ccs() != formal.matrix().ccs() {
                return Err(SxError::Shape(format!(
                    "call input slot {slot} for {} must match declared CCS",
                    self.name
                )));
            }
        }
        self.outputs
            .iter()
            .enumerate()
            .map(|(slot, output)| {
                let nonzeros = (0..output.matrix().nnz())
                    .map(|offset| call_output(self.id, inputs.to_vec(), slot, offset))
                    .collect();
                SXMatrix::new(output.matrix().ccs().clone(), nonzeros)
            })
            .collect()
    }

    pub fn call_output(&self, inputs: &[SXMatrix]) -> Result<SXMatrix> {
        let mut outputs = self.call(inputs)?;
        if outputs.len() != 1 {
            return Err(SxError::Shape(format!(
                "function {} has {} outputs, expected exactly one",
                self.name,
                outputs.len()
            )));
        }
        Ok(outputs.remove(0))
    }

    pub fn call_scalar(&self, inputs: &[SXMatrix]) -> Result<SX> {
        self.call_output(inputs)?.scalar_expr()
    }
}
