use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::Index;
use crate::error::{Result, SxError};
use crate::sx::{
    CallInputs, NodeKind, call_output, call_output_with_inputs, forward_directional,
    forward_directional_batch, node_kind, reverse_directional, reverse_directional_batch,
    with_sx_context_id,
};
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
    pub llvm_root_instructions_emitted: usize,
    pub llvm_total_instructions_emitted: usize,
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
        self.llvm_root_instructions_emitted += other.llvm_root_instructions_emitted;
        self.llvm_total_instructions_emitted += other.llvm_total_instructions_emitted;
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
    context_id: u32,
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
    definitions: HashMap<FunctionId, Arc<SXFunction>>,
}

#[derive(Default)]
struct FunctionCaches {
    dependency_profiles: HashMap<FunctionId, Arc<DependencyProfile>>,
    forward_helpers: HashMap<FunctionId, Arc<SXFunction>>,
    forward_batch_helpers: HashMap<(FunctionId, usize), Arc<SXFunction>>,
    reverse_scalar_helpers: HashMap<(FunctionId, Index, Index), Arc<SXFunction>>,
    reverse_output_batch_helpers: HashMap<(FunctionId, Index, Index, usize), Arc<SXFunction>>,
    reverse_batch_helpers: HashMap<(FunctionId, usize), Arc<SXFunction>>,
    rewrite_stage_results: HashMap<RewriteStageCacheKey, Arc<RewriteStageCacheEntry>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    resolved_call_policies: HashMap<RewrittenCallSiteKey, CallPolicy>,
    expanded_calls: HashMap<RewrittenCallSiteKey, Vec<SXMatrix>>,
    expanding_calls: HashSet<RewrittenCallSiteKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RewrittenCallSiteKey {
    function_id: FunctionId,
    inputs: Vec<SXMatrix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RewriteStageCacheKey {
    function_id: FunctionId,
    config: CallPolicyConfig,
    stage: InlineStage,
}

#[derive(Clone, Debug)]
struct RewriteStageCacheEntry {
    function: SXFunction,
    stats: CompileStats,
    warnings: Vec<CompileWarning>,
}

const MAX_STAGE_REWRITE_RECURSION_DEPTH: usize = 16_384;

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
    lock_registry()
        .definitions
        .insert(function.id, Arc::new(function));
}

fn next_function_id() -> FunctionId {
    NEXT_FUNCTION_ID.fetch_add(1, Ordering::Relaxed)
}

fn validate_function(name: &str, inputs: &[NamedMatrix], outputs: &[NamedMatrix]) -> Result<()> {
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
    }

    let mut missing_symbol_memo = HashMap::new();
    for output in outputs {
        for &expr in output.matrix.nonzeros() {
            if let Some(symbol) =
                first_non_input_symbol(expr, &input_symbols, &mut missing_symbol_memo)
            {
                return Err(SxError::Graph(format!(
                    "output {} references free symbol {symbol} not present in inputs",
                    output.name
                )));
            }
        }
    }
    Ok(())
}

fn infer_function_context_id(inputs: &[NamedMatrix], outputs: &[NamedMatrix]) -> Result<u32> {
    let mut context_id = None;
    for matrix in inputs.iter().chain(outputs.iter()) {
        for &expr in matrix.matrix().nonzeros() {
            let expr_context = expr.context_id();
            if let Some(existing) = context_id {
                if existing != expr_context {
                    return Err(SxError::Graph(format!(
                        "function mixes SX contexts: {} vs {}",
                        existing, expr_context
                    )));
                }
            } else {
                context_id = Some(expr_context);
            }
        }
    }
    Ok(context_id.unwrap_or(1))
}

fn first_non_input_symbol(
    expr: SX,
    input_symbols: &HashSet<SX>,
    memo: &mut HashMap<SX, Option<SX>>,
) -> Option<SX> {
    if let Some(existing) = memo.get(&expr) {
        return *existing;
    }

    enum SymbolScanFrame {
        Visit(SX),
        Finish(SX),
    }

    let mut stack = vec![SymbolScanFrame::Visit(expr)];
    while let Some(frame) = stack.pop() {
        match frame {
            SymbolScanFrame::Visit(current) => {
                if memo.contains_key(&current) {
                    continue;
                }
                stack.push(SymbolScanFrame::Finish(current));
                match node_kind(current) {
                    NodeKind::Constant(_) | NodeKind::Symbol { .. } => {}
                    NodeKind::Unary { arg, .. } => {
                        if !memo.contains_key(&arg) {
                            stack.push(SymbolScanFrame::Visit(arg));
                        }
                    }
                    NodeKind::Binary { lhs, rhs, .. } => {
                        if !memo.contains_key(&rhs) {
                            stack.push(SymbolScanFrame::Visit(rhs));
                        }
                        if !memo.contains_key(&lhs) {
                            stack.push(SymbolScanFrame::Visit(lhs));
                        }
                    }
                    NodeKind::Call {
                        function,
                        inputs,
                        output_slot,
                        output_offset,
                    } => {
                        let profile = dependency_profile(function);
                        for (slot, input) in inputs.iter().enumerate().rev() {
                            for (offset, &value) in input.nonzeros().iter().enumerate().rev() {
                                if profile.output_depends_on(
                                    output_slot,
                                    output_offset,
                                    slot,
                                    offset,
                                ) && !memo.contains_key(&value)
                                {
                                    stack.push(SymbolScanFrame::Visit(value));
                                }
                            }
                        }
                    }
                }
            }
            SymbolScanFrame::Finish(current) => {
                if memo.contains_key(&current) {
                    continue;
                }
                let missing = match node_kind(current) {
                    NodeKind::Constant(_) => None,
                    NodeKind::Symbol { .. } => {
                        (!input_symbols.contains(&current)).then_some(current)
                    }
                    NodeKind::Unary { arg, .. } => memo.get(&arg).copied().flatten(),
                    NodeKind::Binary { lhs, rhs, .. } => memo
                        .get(&lhs)
                        .copied()
                        .flatten()
                        .or_else(|| memo.get(&rhs).copied().flatten()),
                    NodeKind::Call {
                        function,
                        inputs,
                        output_slot,
                        output_offset,
                    } => {
                        let profile = dependency_profile(function);
                        let mut missing = None;
                        'slots: for (slot, input) in inputs.iter().enumerate() {
                            for (offset, &value) in input.nonzeros().iter().enumerate() {
                                if profile.output_depends_on(
                                    output_slot,
                                    output_offset,
                                    slot,
                                    offset,
                                ) {
                                    if let Some(symbol) = memo.get(&value).copied().flatten() {
                                        missing = Some(symbol);
                                        break 'slots;
                                    }
                                }
                            }
                        }
                        missing
                    }
                };
                memo.insert(current, missing);
            }
        }
    }

    memo.get(&expr).copied().flatten()
}

pub(crate) fn function_by_id(id: FunctionId) -> Option<Arc<SXFunction>> {
    lock_registry().definitions.get(&id).cloned()
}

pub fn lookup_function(id: FunctionId) -> Option<SXFunction> {
    function_by_id(id).map(|function| (*function).clone())
}

#[allow(dead_code)]
pub fn lookup_function_ref(id: FunctionId) -> Option<Arc<SXFunction>> {
    function_by_id(id)
}

pub(crate) fn function_name(id: FunctionId) -> Option<String> {
    lock_registry()
        .definitions
        .get(&id)
        .map(|function| function.name.clone())
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

pub(crate) fn dependency_profile(function_id: FunctionId) -> Arc<DependencyProfile> {
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
    let input_index_by_symbol = flat_inputs
        .iter()
        .copied()
        .enumerate()
        .map(|(index, symbol)| (symbol, index))
        .collect::<HashMap<_, _>>();
    let mut memo = HashMap::<SX, Vec<bool>>::new();

    let mut output_offsets = Vec::with_capacity(function.outputs.len());
    let mut deps = Vec::new();
    for output in function.outputs() {
        output_offsets.push(deps.len());
        for &expr in output.matrix().nonzeros() {
            deps.push(expr_dependency_mask(
                expr,
                &input_index_by_symbol,
                &mut memo,
            ));
        }
    }

    let profile = Arc::new(DependencyProfile {
        input_offsets,
        output_offsets,
        deps,
    });
    lock_caches()
        .dependency_profiles
        .insert(function_id, profile.clone());
    profile
}

fn expr_dependency_mask(
    expr: SX,
    input_index_by_symbol: &HashMap<SX, usize>,
    memo: &mut HashMap<SX, Vec<bool>>,
) -> Vec<bool> {
    if let Some(existing) = memo.get(&expr) {
        return existing.clone();
    }

    let input_count = input_index_by_symbol.len();
    let deps = match node_kind(expr) {
        NodeKind::Constant(_) => vec![false; input_count],
        NodeKind::Symbol { .. } => {
            let mut deps = vec![false; input_count];
            if let Some(&index) = input_index_by_symbol.get(&expr) {
                deps[index] = true;
            }
            deps
        }
        NodeKind::Unary { arg, .. } => expr_dependency_mask(arg, input_index_by_symbol, memo),
        NodeKind::Binary { lhs, rhs, .. } => {
            let mut deps = expr_dependency_mask(lhs, input_index_by_symbol, memo);
            let rhs_deps = expr_dependency_mask(rhs, input_index_by_symbol, memo);
            merge_dependency_masks(&mut deps, &rhs_deps);
            deps
        }
        NodeKind::Call {
            function,
            inputs,
            output_slot,
            output_offset,
        } => {
            let profile = dependency_profile(function);
            let mut deps = vec![false; input_count];
            for (slot, input) in inputs.iter().enumerate() {
                for (offset, &value) in input.nonzeros().iter().enumerate() {
                    if profile.output_depends_on(output_slot, output_offset, slot, offset) {
                        let value_deps = expr_dependency_mask(value, input_index_by_symbol, memo);
                        merge_dependency_masks(&mut deps, &value_deps);
                    }
                }
            }
            deps
        }
    };

    memo.insert(expr, deps.clone());
    deps
}

fn merge_dependency_masks(dst: &mut [bool], src: &[bool]) {
    for (dst_value, &src_value) in dst.iter_mut().zip(src.iter()) {
        *dst_value |= src_value;
    }
}

pub(crate) fn forward_helper(function_id: FunctionId) -> Result<Arc<SXFunction>> {
    if let Some(helper) = lock_caches().forward_helpers.get(&function_id).cloned() {
        return Ok(helper);
    }
    let function = function_by_id(function_id).expect("function id must be registered");
    let helper = with_sx_context_id(function.context_id(), || -> Result<Arc<SXFunction>> {
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
                let directional = forward_directional(output.matrix().nonzeros(), &vars, &seeds)?;
                NamedMatrix::new(
                    format!("{}_directional", output.name()),
                    SXMatrix::new(output.matrix().ccs().clone(), directional)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(SXFunction::from_parts(
            format!("{}_forward_helper", function.name()),
            helper_inputs,
            helper_outputs,
            function.call_policy_override(),
        )?))
    })?;
    lock_caches()
        .forward_helpers
        .insert(function_id, helper.clone());
    Ok(helper)
}

pub(crate) fn forward_batch_helper(
    function_id: FunctionId,
    directions: usize,
) -> Result<Arc<SXFunction>> {
    if directions <= 1 {
        return forward_helper(function_id);
    }
    if let Some(helper) = lock_caches()
        .forward_batch_helpers
        .get(&(function_id, directions))
        .cloned()
    {
        return Ok(helper);
    }

    let function = function_by_id(function_id).expect("function id must be registered");
    let helper = with_sx_context_id(function.context_id(), || -> Result<Arc<SXFunction>> {
        let mut helper_inputs = function.inputs.clone();
        let mut seed_inputs_by_direction = Vec::with_capacity(directions);
        for direction in 0..directions {
            let mut direction_seed_inputs = Vec::with_capacity(function.inputs.len());
            for input in function.inputs() {
                let seed_matrix = SXMatrix::sym(
                    format!("{}_seed_{}_{}", function.name(), direction, input.name()),
                    input.matrix().ccs().clone(),
                )?;
                direction_seed_inputs.push(seed_matrix.clone());
                helper_inputs.push(NamedMatrix::new(
                    format!("seed_{}_{}", direction, input.name()),
                    seed_matrix,
                )?);
            }
            seed_inputs_by_direction.push(direction_seed_inputs);
        }

        let vars = function
            .inputs()
            .iter()
            .flat_map(|input| input.matrix().nonzeros().iter().copied())
            .collect::<Vec<_>>();
        let seeds_by_direction = seed_inputs_by_direction
            .iter()
            .map(|direction_inputs| {
                direction_inputs
                    .iter()
                    .flat_map(|seed| seed.nonzeros().iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut helper_outputs = Vec::with_capacity(function.outputs.len() * directions);
        for output in function.outputs() {
            let directional_by_direction =
                forward_directional_batch(output.matrix().nonzeros(), &vars, &seeds_by_direction)?;
            for (direction, directional) in directional_by_direction.into_iter().enumerate() {
                helper_outputs.push(NamedMatrix::new(
                    format!("{}_directional_{}", output.name(), direction),
                    SXMatrix::new(output.matrix().ccs().clone(), directional)?,
                )?);
            }
        }

        Ok(Arc::new(SXFunction::from_parts(
            format!("{}_forward_batch_helper_{}", function.name(), directions),
            helper_inputs,
            helper_outputs,
            function.call_policy_override(),
        )?))
    })?;
    lock_caches()
        .forward_batch_helpers
        .insert((function_id, directions), helper.clone());
    Ok(helper)
}

pub(crate) fn reverse_scalar_helper(
    function_id: FunctionId,
    output_slot: Index,
    output_offset: Index,
) -> Result<Arc<SXFunction>> {
    if let Some(helper) = lock_caches()
        .reverse_scalar_helpers
        .get(&(function_id, output_slot, output_offset))
        .cloned()
    {
        return Ok(helper);
    }

    let function = function_by_id(function_id).expect("function id must be registered");
    let helper = with_sx_context_id(function.context_id(), || -> Result<Arc<SXFunction>> {
        let vars = function
            .inputs()
            .iter()
            .flat_map(|input| input.matrix().nonzeros().iter().copied())
            .collect::<Vec<_>>();
        let selected_output = function.outputs()[output_slot].matrix().nz(output_offset);
        let adjoints = reverse_directional(&[selected_output], &vars, &[SX::one()])?;

        let mut helper_outputs = Vec::with_capacity(function.inputs.len());
        let mut offset = 0;
        for input in function.inputs() {
            let next_offset = offset + input.matrix().nnz();
            helper_outputs.push(NamedMatrix::new(
                format!("{}_grad", input.name()),
                SXMatrix::new(
                    input.matrix().ccs().clone(),
                    adjoints[offset..next_offset].to_vec(),
                )?,
            )?);
            offset = next_offset;
        }

        Ok(Arc::new(SXFunction::from_parts(
            format!(
                "{}_reverse_output_{}_{}_helper",
                function.name(),
                output_slot,
                output_offset
            ),
            function.inputs.clone(),
            helper_outputs,
            function.call_policy_override(),
        )?))
    })?;
    lock_caches()
        .reverse_scalar_helpers
        .insert((function_id, output_slot, output_offset), helper.clone());
    Ok(helper)
}

pub(crate) fn reverse_batch_helper(
    function_id: FunctionId,
    directions: usize,
) -> Result<Arc<SXFunction>> {
    if directions == 0 {
        return Err(SxError::Graph(
            "reverse helper requires at least one adjoint direction".into(),
        ));
    }
    if let Some(helper) = lock_caches()
        .reverse_batch_helpers
        .get(&(function_id, directions))
        .cloned()
    {
        return Ok(helper);
    }

    let function = function_by_id(function_id).expect("function id must be registered");
    let helper = with_sx_context_id(function.context_id(), || -> Result<Arc<SXFunction>> {
        let mut helper_inputs = function.inputs.clone();
        let mut seed_inputs_by_direction = Vec::with_capacity(directions);
        for direction in 0..directions {
            let mut direction_seed_inputs = Vec::with_capacity(function.outputs.len());
            for output in function.outputs() {
                let seed_matrix = SXMatrix::sym(
                    format!(
                        "{}_adj_seed_{}_{}",
                        function.name(),
                        direction,
                        output.name()
                    ),
                    output.matrix().ccs().clone(),
                )?;
                direction_seed_inputs.push(seed_matrix.clone());
                helper_inputs.push(NamedMatrix::new(
                    format!("adj_seed_{}_{}", direction, output.name()),
                    seed_matrix,
                )?);
            }
            seed_inputs_by_direction.push(direction_seed_inputs);
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
        let seeds_by_direction = seed_inputs_by_direction
            .iter()
            .map(|direction_inputs| {
                direction_inputs
                    .iter()
                    .flat_map(|seed| seed.nonzeros().iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let adjoints_by_direction =
            reverse_directional_batch(&outputs, &vars, &seeds_by_direction)?;

        let mut helper_outputs = Vec::with_capacity(function.inputs.len() * directions);
        for (direction, adjoints) in adjoints_by_direction.into_iter().enumerate() {
            let mut offset = 0;
            for input in function.inputs() {
                let next_offset = offset + input.matrix().nnz();
                helper_outputs.push(NamedMatrix::new(
                    format!("{}_reverse_{}", input.name(), direction),
                    SXMatrix::new(
                        input.matrix().ccs().clone(),
                        adjoints[offset..next_offset].to_vec(),
                    )?,
                )?);
                offset = next_offset;
            }
        }

        Ok(Arc::new(SXFunction::from_parts(
            format!("{}_reverse_batch_helper_{}", function.name(), directions),
            helper_inputs,
            helper_outputs,
            function.call_policy_override(),
        )?))
    })?;
    lock_caches()
        .reverse_batch_helpers
        .insert((function_id, directions), helper.clone());
    Ok(helper)
}

pub(crate) fn reverse_output_batch_helper(
    function_id: FunctionId,
    output_slot: Index,
    output_offset: Index,
    directions: usize,
) -> Result<Arc<SXFunction>> {
    if directions == 0 {
        return Err(SxError::Graph(
            "reverse helper requires at least one adjoint direction".into(),
        ));
    }
    if let Some(helper) = lock_caches()
        .reverse_output_batch_helpers
        .get(&(function_id, output_slot, output_offset, directions))
        .cloned()
    {
        return Ok(helper);
    }

    let function = function_by_id(function_id).expect("function id must be registered");
    let helper = with_sx_context_id(function.context_id(), || -> Result<Arc<SXFunction>> {
        let mut helper_inputs = function.inputs.clone();
        let mut seed_inputs = Vec::with_capacity(directions);
        for direction in 0..directions {
            let seed_matrix = SXMatrix::scalar(SX::sym(format!(
                "{}_adj_seed_{}_{}_{}",
                function.name(),
                output_slot,
                output_offset,
                direction
            )));
            seed_inputs.push(seed_matrix.clone());
            helper_inputs.push(NamedMatrix::new(
                format!("adj_seed_{}", direction),
                seed_matrix,
            )?);
        }

        let vars = function
            .inputs()
            .iter()
            .flat_map(|input| input.matrix().nonzeros().iter().copied())
            .collect::<Vec<_>>();
        let selected_output = function.outputs()[output_slot].matrix().nz(output_offset);
        let seeds_by_direction = seed_inputs
            .iter()
            .map(|seed| vec![seed.nz(0)])
            .collect::<Vec<_>>();
        let adjoints_by_direction =
            reverse_directional_batch(&[selected_output], &vars, &seeds_by_direction)?;

        let mut helper_outputs = Vec::with_capacity(function.inputs.len() * directions);
        for (direction, adjoints) in adjoints_by_direction.into_iter().enumerate() {
            let mut offset = 0;
            for input in function.inputs() {
                let next_offset = offset + input.matrix().nnz();
                helper_outputs.push(NamedMatrix::new(
                    format!("{}_reverse_{}", input.name(), direction),
                    SXMatrix::new(
                        input.matrix().ccs().clone(),
                        adjoints[offset..next_offset].to_vec(),
                    )?,
                )?);
                offset = next_offset;
            }
        }

        Ok(Arc::new(SXFunction::from_parts(
            format!(
                "{}_reverse_output_batch_{}_{}_helper_{}",
                function.name(),
                output_slot,
                output_offset,
                directions
            ),
            helper_inputs,
            helper_outputs,
            function.call_policy_override(),
        )?))
    })?;
    lock_caches().reverse_output_batch_helpers.insert(
        (function_id, output_slot, output_offset, directions),
        helper.clone(),
    );
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

fn scan_call_policy(function: &SXFunction, config: CallPolicyConfig) -> CallPolicy {
    match function.call_policy_override() {
        Some(policy) if config.respect_function_overrides => policy,
        Some(_) => config.default_policy,
        None => config.default_policy,
    }
}

#[derive(Default)]
struct ScanStageResult {
    requires_inlining: bool,
    stats: CompileStats,
    warnings: Vec<CompileWarning>,
}

fn scan_expr_for_stage(
    expr: SX,
    config: CallPolicyConfig,
    stage: InlineStage,
    seen_exprs: &mut HashSet<SX>,
    seen_calls: &mut HashSet<RewrittenCallSiteKey>,
    scan: &mut ScanStageResult,
    warned_ignored_overrides: &mut HashSet<FunctionId>,
    call_depth: usize,
    recursion_depth: usize,
) -> Result<()> {
    let mut stack = vec![(expr, call_depth, recursion_depth)];
    while let Some((current, current_call_depth, current_recursion_depth)) = stack.pop() {
        if current_recursion_depth > MAX_STAGE_REWRITE_RECURSION_DEPTH {
            return Err(SxError::Graph(format!(
                "stage scan exceeded recursion limit ({MAX_STAGE_REWRITE_RECURSION_DEPTH}); expression graph is too deep to rewrite recursively"
            )));
        }
        if !seen_exprs.insert(current) {
            continue;
        }
        match current.inspect() {
            NodeView::Constant(_) | NodeView::Symbol { .. } => {}
            NodeView::Unary { arg, .. } => {
                stack.push((arg, current_call_depth, current_recursion_depth + 1));
            }
            NodeView::Binary { lhs, rhs, .. } => {
                stack.push((rhs, current_call_depth, current_recursion_depth + 1));
                stack.push((lhs, current_call_depth, current_recursion_depth + 1));
            }
            NodeView::Call {
                function_id,
                inputs,
                ..
            } => {
                let callee = function_by_id(function_id)
                    .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
                scan.stats.max_call_depth = scan.stats.max_call_depth.max(current_call_depth + 1);
                let key = RewrittenCallSiteKey {
                    function_id,
                    inputs: inputs.clone(),
                };
                if seen_calls.insert(key) {
                    let policy = scan_call_policy(&callee, config);
                    record_policy(&mut scan.stats, policy);
                    match callee.call_policy_override() {
                        Some(_) if config.respect_function_overrides => {
                            scan.stats.overrides_applied += 1;
                        }
                        Some(policy) => {
                            scan.stats.overrides_ignored += 1;
                            if warned_ignored_overrides.insert(callee.id()) {
                                scan.warnings.push(CompileWarning {
                                    message: format!(
                                        "ignored call policy override {:?} on {}; enforced global policy {:?}",
                                        policy,
                                        callee.name(),
                                        config.default_policy
                                    ),
                                });
                            }
                        }
                        None => {}
                    }
                    if stage.should_inline(policy) {
                        scan.requires_inlining = true;
                    }
                }
                for input in inputs.iter().rev() {
                    for &value in input.nonzeros().iter().rev() {
                        stack.push((value, current_call_depth + 1, current_recursion_depth + 1));
                    }
                }
            }
        }
    }
    Ok(())
}

fn scan_function_for_stage(
    function: &SXFunction,
    config: CallPolicyConfig,
    stage: InlineStage,
) -> Result<ScanStageResult> {
    let mut seen_exprs = HashSet::new();
    let mut seen_calls = HashSet::new();
    let mut warned_ignored_overrides = HashSet::new();
    let mut scan = ScanStageResult::default();
    for output in function.outputs() {
        for &expr in output.matrix().nonzeros() {
            scan_expr_for_stage(
                expr,
                config,
                stage,
                &mut seen_exprs,
                &mut seen_calls,
                &mut scan,
                &mut warned_ignored_overrides,
                0,
                0,
            )?;
        }
    }
    Ok(scan)
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
    call_depth: usize,
    recursion_depth: usize,
) -> Result<SX> {
    if let Some(existing) = memo.get(&expr) {
        return Ok(*existing);
    }
    enum RewriteFrame {
        Visit {
            expr: SX,
            call_depth: usize,
            recursion_depth: usize,
        },
        Finish {
            expr: SX,
            call_depth: usize,
            recursion_depth: usize,
        },
    }
    let mut stack = vec![RewriteFrame::Visit {
        expr,
        call_depth,
        recursion_depth,
    }];
    while let Some(frame) = stack.pop() {
        match frame {
            RewriteFrame::Visit {
                expr: current,
                call_depth: current_call_depth,
                recursion_depth: current_recursion_depth,
            } => {
                if current_recursion_depth > MAX_STAGE_REWRITE_RECURSION_DEPTH {
                    return Err(SxError::Graph(format!(
                        "stage rewrite exceeded recursion limit ({MAX_STAGE_REWRITE_RECURSION_DEPTH}); expression graph is too deep to rewrite recursively"
                    )));
                }
                if memo.contains_key(&current) {
                    continue;
                }
                stack.push(RewriteFrame::Finish {
                    expr: current,
                    call_depth: current_call_depth,
                    recursion_depth: current_recursion_depth,
                });
                match current.inspect() {
                    NodeView::Constant(_) | NodeView::Symbol { .. } => {}
                    NodeView::Unary { arg, .. } => {
                        if !memo.contains_key(&arg) {
                            stack.push(RewriteFrame::Visit {
                                expr: arg,
                                call_depth: current_call_depth,
                                recursion_depth: current_recursion_depth + 1,
                            });
                        }
                    }
                    NodeView::Binary { lhs, rhs, .. } => {
                        if !memo.contains_key(&rhs) {
                            stack.push(RewriteFrame::Visit {
                                expr: rhs,
                                call_depth: current_call_depth,
                                recursion_depth: current_recursion_depth + 1,
                            });
                        }
                        if !memo.contains_key(&lhs) {
                            stack.push(RewriteFrame::Visit {
                                expr: lhs,
                                call_depth: current_call_depth,
                                recursion_depth: current_recursion_depth + 1,
                            });
                        }
                    }
                    NodeView::Call { inputs, .. } => {
                        for input in inputs.iter().rev() {
                            for &value in input.nonzeros().iter().rev() {
                                if !memo.contains_key(&value) {
                                    stack.push(RewriteFrame::Visit {
                                        expr: value,
                                        call_depth: current_call_depth + 1,
                                        recursion_depth: current_recursion_depth + 1,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            RewriteFrame::Finish {
                expr: current,
                call_depth: current_call_depth,
                recursion_depth: current_recursion_depth,
            } => {
                if memo.contains_key(&current) {
                    continue;
                }
                let rewritten = match current.inspect() {
                    NodeView::Constant(value) => SX::from(value),
                    NodeView::Symbol { .. } => bindings.get(&current).copied().unwrap_or(current),
                    NodeView::Unary { op, arg } => apply_unary(
                        op,
                        *memo.get(&arg).ok_or_else(|| {
                            SxError::Graph(
                                "missing rewritten unary argument during stage rewrite".into(),
                            )
                        })?,
                    ),
                    NodeView::Binary { op, lhs, rhs } => apply_binary(
                        op,
                        *memo.get(&lhs).ok_or_else(|| {
                            SxError::Graph(
                                "missing rewritten binary lhs during stage rewrite".into(),
                            )
                        })?,
                        *memo.get(&rhs).ok_or_else(|| {
                            SxError::Graph(
                                "missing rewritten binary rhs during stage rewrite".into(),
                            )
                        })?,
                    ),
                    NodeView::Call {
                        function_id,
                        inputs,
                        output_slot,
                        output_offset,
                        ..
                    } => {
                        let callee = function_by_id(function_id).ok_or_else(|| {
                            SxError::Graph(format!("unknown function id {function_id}"))
                        })?;
                        let rewritten_inputs = inputs
                            .iter()
                            .map(|input| {
                                rewrite_matrix_with_memo(
                                    input,
                                    bindings,
                                    config,
                                    stage,
                                    memo,
                                    stats,
                                    warnings,
                                    state,
                                    current_call_depth + 1,
                                    current_recursion_depth + 1,
                                )
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let key = RewrittenCallSiteKey {
                            function_id,
                            inputs: rewritten_inputs.clone(),
                        };
                        let policy = if let Some(existing) =
                            state.resolved_call_policies.get(&key).copied()
                        {
                            existing
                        } else {
                            let policy =
                                resolve_call_policy(&callee, config, stats, warnings, state);
                            record_policy(stats, policy);
                            state.resolved_call_policies.insert(key.clone(), policy);
                            policy
                        };
                        stats.max_call_depth = stats.max_call_depth.max(current_call_depth + 1);
                        if stage.should_inline(policy) {
                            if let Some(outputs) = state.expanded_calls.get(&key) {
                                outputs[output_slot].nz(output_offset)
                            } else if !state.expanding_calls.insert(key.clone()) {
                                return Err(SxError::Graph(format!(
                                    "recursive call expansion detected while rewriting {} for {:?}",
                                    callee.name(),
                                    stage
                                )));
                            } else {
                                let outputs_result = {
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
                                        current_call_depth + 1,
                                        current_recursion_depth + 1,
                                    )
                                };
                                state.expanding_calls.remove(&key);
                                let outputs = outputs_result?;
                                let value = outputs[output_slot].nz(output_offset);
                                state.expanded_calls.insert(key, outputs);
                                value
                            }
                        } else {
                            call_output(function_id, rewritten_inputs, output_slot, output_offset)
                        }
                    }
                };
                memo.insert(current, rewritten);
            }
        }
    }
    memo.get(&expr)
        .copied()
        .ok_or_else(|| SxError::Graph("failed to rewrite expression".into()))
}

fn rewrite_matrix_with_memo(
    matrix: &SXMatrix,
    bindings: &HashMap<SX, SX>,
    config: CallPolicyConfig,
    stage: InlineStage,
    mut memo: &mut HashMap<SX, SX>,
    stats: &mut CompileStats,
    warnings: &mut Vec<CompileWarning>,
    state: &mut RewriteState,
    call_depth: usize,
    recursion_depth: usize,
) -> Result<SXMatrix> {
    SXMatrix::new(
        matrix.ccs().clone(),
        matrix
            .nonzeros()
            .iter()
            .copied()
            .map(|expr| {
                rewrite_expr(
                    expr,
                    bindings,
                    config,
                    stage,
                    &mut memo,
                    stats,
                    warnings,
                    state,
                    call_depth,
                    recursion_depth + 1,
                )
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
    call_depth: usize,
    recursion_depth: usize,
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
    let mut memo = HashMap::new();
    function
        .outputs()
        .iter()
        .map(|output| {
            rewrite_matrix_with_memo(
                output.matrix(),
                &bindings,
                config,
                stage,
                &mut memo,
                stats,
                warnings,
                state,
                call_depth,
                recursion_depth + 1,
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
    let cache_key = RewriteStageCacheKey {
        function_id: function.id(),
        config,
        stage,
    };
    if let Some(entry) = lock_caches().rewrite_stage_results.get(&cache_key).cloned() {
        stats.absorb(&entry.stats);
        warnings.extend(entry.warnings.iter().cloned());
        return Ok(entry.function.clone());
    }

    let mut local_stats = CompileStats::default();
    let mut local_warnings = Vec::new();
    let mut state = RewriteState::default();
    let mut memo = HashMap::new();
    let inputs = function.inputs.clone();
    let scan = scan_function_for_stage(function, config, stage)?;
    let rewritten = if !scan.requires_inlining {
        local_stats.absorb(&scan.stats);
        local_warnings.extend(scan.warnings);
        function.clone()
    } else {
        let outputs = function
            .outputs()
            .iter()
            .map(|output| {
                Ok(NamedMatrix::new(
                    output.name(),
                    rewrite_matrix_with_memo(
                        output.matrix(),
                        &HashMap::new(),
                        config,
                        stage,
                        &mut memo,
                        &mut local_stats,
                        &mut local_warnings,
                        &mut state,
                        0,
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
        )?
    };

    stats.absorb(&local_stats);
    warnings.extend(local_warnings.iter().cloned());
    lock_caches().rewrite_stage_results.insert(
        cache_key,
        Arc::new(RewriteStageCacheEntry {
            function: rewritten.clone(),
            stats: local_stats,
            warnings: local_warnings,
        }),
    );
    Ok(rewritten)
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
        let context_id = infer_function_context_id(&inputs, &outputs)?;
        let function = Self {
            id: next_function_id(),
            context_id,
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

    pub fn context_id(&self) -> u32 {
        self.context_id
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
        let function = Self {
            id: next_function_id(),
            context_id: self.context_id,
            name: self.name.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            call_policy_override: Some(policy),
        };
        register_function(function.clone());
        function
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
            if actual
                .nonzeros()
                .iter()
                .any(|expr| expr.context_id() != self.context_id)
            {
                return Err(SxError::Graph(format!(
                    "call input slot {slot} for {} uses a different SX context",
                    self.name
                )));
            }
        }
        let call_inputs = CallInputs::from_slice(inputs);
        self.outputs
            .iter()
            .enumerate()
            .map(|(slot, output)| {
                let nonzeros = (0..output.matrix().nnz())
                    .map(|offset| {
                        call_output_with_inputs(self.id, call_inputs.clone(), slot, offset)
                    })
                    .collect();
                SXMatrix::new(output.matrix().ccs().clone(), nonzeros)
            })
            .collect()
    }

    pub fn forward(&self, directions: usize) -> Result<Self> {
        if directions == 0 {
            return Err(SxError::Graph(
                "forward helper requires at least one tangent direction".into(),
            ));
        }
        let helper = if directions == 1 {
            forward_helper(self.id())?
        } else {
            forward_batch_helper(self.id(), directions)?
        };
        Ok((*helper).clone())
    }

    pub fn reverse(&self, directions: usize) -> Result<Self> {
        if directions == 0 {
            return Err(SxError::Graph(
                "reverse helper requires at least one adjoint direction".into(),
            ));
        }
        let helper = reverse_batch_helper(self.id(), directions)?;
        Ok((*helper).clone())
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
